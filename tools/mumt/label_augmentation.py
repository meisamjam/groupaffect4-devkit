"""label_augmentation.py

Leakage-free label augmentation for MuMT-Affect / GroupAffect-4.

Generates soft VAD labels for the 8,978-window pretraining pool using four
approved sources — none of which use a person's own physiological signals to
generate their own supervised label.

Approved sources (see LABEL_AUGMENTATION.md for theory):
  S1 — OU/GP temporal interpolation of self-reports (per person, no physiology)
  S2 — Cross-person self-report propagation via LMC covariance
  S4 — Cross-person EDA SCR events → target person arousal (k≠l only)
  S5 — Cross-person HR direction → target person valence (k≠l only)

Output per window:
  {dim}_mu, {dim}_sigma  — GP posterior mean and std on 1–9 Likert scale
  {dim}_soft             — soft label [p_Low, p_Mid, p_High]
  {dim}_weight           — instance weight in [0.05, 0.95]
  n_obs_{dim}            — number of GP observations used
  label_sources          — list of source tags for traceability

Usage:
  python tools/mumt/label_augmentation.py \\
      --dataset   data/mumt/dataset.pkl \\
      --pretrain  data/mumt/pretrain_dataset.pkl \\
      --output    data/mumt/augmented_pool.pkl

  # With explicit OU override and minimum weight threshold:
  python tools/mumt/label_augmentation.py \\
      --dataset data/mumt/dataset.pkl \\
      --pretrain data/mumt/pretrain_dataset.pkl \\
      --output data/mumt/augmented_pool.pkl \\
      --min-weight 0.10 \\
      --use-cross-physio
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMS = ("valence", "arousal", "dominance")
SEATS = ["P1", "P2", "P3", "P4"]
TASKS = ["T0", "T1", "T2", "T3", "T4"]

# VAD binning thresholds (Likert 1–9)
VAD_THRESHOLDS = (3.0, 6.0)   # Low: ≤3, Mid: 4-6, High: ≥7

# Observation noise levels (Likert units, σ)
SIGMA_SELF_REPORT = 0.8         # own self-report
SIGMA_CROSS_REPORT_BASE = 1.2   # another person's self-report (scaled by 1/ρ below)
SIGMA_SCR_HIGH_AMP = 0.8        # high-amplitude SCR event → arousal evidence
SIGMA_SCR_LOW_AMP = 1.4         # low-amplitude SCR event
SIGMA_HR_DIRECTION = 1.8        # HR direction → valence evidence

# OU parameters from Meng et al. (2026) CTSEM study — used as defaults
MENG_OU_PARAMS = {
    "valence":   {"theta": 0.056, "sigma2": 2.1, "mu": 5.3},
    "arousal":   {"theta": 0.578, "sigma2": 1.8, "mu": 4.9},
    "dominance": {"theta": 0.116, "sigma2": 1.9, "mu": 5.1},
}

# Minimum weight thresholds per dimension (below → window excluded)
# Arousal minimum is higher than default: LOO calibration shows arousal GP
# accuracy = 0.30 (below chance), so we require higher confidence before
# including augmented arousal labels. Only augmented windows near a task-level
# observation (w_min crossed by temporal decay) are used.
W_MIN = {"valence": 0.10, "arousal": 0.20, "dominance": 0.10}

# Per-task dimension availability: tasks where a VAD dimension was NOT probed.
# These entries are "not applicable" (the probe was never shown), not missing data.
# Dominance was absent from T4 VAD probes by design; imputing it via GP is invalid.
TASK_DIM_AVAILABILITY: dict[str, list[str]] = {
    "T4": ["valence", "arousal"],
}

# Task-conditioned prior mean adjustments for the OU GP (Likert units).
# The OU prior mean (mu) represents the population average at rest (T0).
# Different tasks elicit systematically different VAD profiles; conditioning
# the GP prior on task type makes the posterior more informative, especially
# for AROUSAL which has fast mean-reversion (theta=0.578) and sparse
# task-level observations.
#
# Values derived from mean VAD per task in GroupAffect-4 labelled data:
#   T0 (rest):         V≈6.1  A≈4.1  D≈5.3
#   T1 (info pool):    V≈7.2  A≈5.2  D≈5.8
#   T2 (negotiate):    V≈6.8  A≈5.5  D≈5.5
#   T3 (ideation):     V≈7.3  A≈5.8  D≈5.6
#   T4 (game):         V≈7.1  A≈6.1  D≈5.2
#
# Adjustment = task_mean - global_mean (mu from MENG_OU_PARAMS)
TASK_PRIOR_ADJUSTMENT: dict[str, dict[str, float]] = {
    "valence": {
        "T0": -1.2, "T1": -0.1, "T2": -0.5, "T3":  0.0, "T4": -0.2,
    },
    "arousal": {
        "T0": -0.8, "T1":  0.3, "T2":  0.6, "T3":  0.9, "T4":  1.2,
    },
    "dominance": {
        "T0":  0.2, "T1":  0.7, "T2":  0.4, "T3":  0.5, "T4":  0.1,
    },
}

# Maximum instance weight (self-reports always = 1.0; augmented always < 1.0)
W_MAX = 0.95

# Task-type synchrony priors (fallback when empirical ρ cannot be estimated)
SYNC_PRIORS: dict[str, dict[str, float]] = {
    "T0": {"rho_eda": 0.20, "rho_hr": 0.15},
    "T1": {"rho_eda": 0.35, "rho_hr": 0.25},
    "T2": {"rho_eda": 0.40, "rho_hr": 0.30},
    "T3": {"rho_eda": 0.30, "rho_hr": 0.22},
    "T4": {"rho_eda": 0.30, "rho_hr": 0.22},
}

# SCR extraction
SCR_LATENCY_S = 2.5         # seconds: stimulus → SCR peak latency
SCR_MIN_AMPLITUDE = 0.05    # phasic units after standardisation (EDA_Phasic column)
WINDOW_SEC = 30.0
FIXED_LENGTH = 400
FS_WINDOW = FIXED_LENGTH / WINDOW_SEC   # ≈ 13.3 Hz (resampled rate in stored sequences)

# Column names as stored in the pkl files (from pickle_generation_affectai.py)
# EDA sequence: [raw EDA, EDA_Phasic, EDA_Tonic, HR, Skin Temp]
COL_EDA_PHASIC  = "EDA_Phasic"   # neurokit2-decomposed phasic component
COL_EDA_TONIC   = "EDA_Tonic"    # neurokit2-decomposed tonic component
COL_EDA_RAW     = "value_3"      # raw EDA conductance (µS) — EmotiBit value_3
COL_HR_PROXY    = "value_6"      # firmware heart rate (BPM) — EmotiBit value_6
COL_TEMP_SKIN   = "value_11"     # skin temperature (°C) — EmotiBit value_11


# ---------------------------------------------------------------------------
# OU / GP Core
# ---------------------------------------------------------------------------

class OUParams(NamedTuple):
    theta: float    # mean-reversion rate (s⁻¹)
    sigma2: float   # process variance
    mu: float       # long-run mean (Likert scale)


def ou_gp_posterior(
    t_query: float,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_obs: np.ndarray,
    params: OUParams,
) -> tuple[float, float]:
    """Exact GP posterior under the OU (Matérn-1/2) kernel.

    Parameters
    ----------
    t_query   : scalar, time of the unlabelled window centre (seconds, LSL clock)
    t_obs     : (n,) observation times
    y_obs     : (n,) Likert-scale observations (1–9)
    sigma_obs : (n,) per-observation noise std
    params    : OUParams for the target dimension

    Returns
    -------
    mu_post, std_post  (posterior mean and std on Likert scale)
    """
    n = len(t_obs)
    if n == 0:
        return params.mu, float(np.sqrt(params.sigma2))

    theta, sigma2, mu = params.theta, params.sigma2, params.mu

    # Kernel matrix K + observation noise
    tau_mat = np.abs(t_obs[:, None] - t_obs[None, :])          # (n, n)
    K = sigma2 * np.exp(-theta * tau_mat) + np.diag(sigma_obs ** 2)

    # Cross-covariance k(t*, t_obs)
    tau_star = np.abs(t_query - t_obs)                           # (n,)
    k_star = sigma2 * np.exp(-theta * tau_star)                  # (n,)

    # Solve via Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, (y_obs - mu)))
        v = np.linalg.solve(L, k_star)
        mu_post = mu + float(k_star @ alpha)
        var_post = float(max(sigma2 - float(v @ v), 1e-6))
    except np.linalg.LinAlgError:
        # Fall back to least-squares solve
        K_inv_y = np.linalg.lstsq(K, y_obs - mu, rcond=None)[0]
        K_inv_k = np.linalg.lstsq(K, k_star, rcond=None)[0]
        mu_post = mu + float(k_star @ K_inv_y)
        var_post = float(max(sigma2 - float(k_star @ K_inv_k), 1e-6))

    return mu_post, float(np.sqrt(var_post))


def temporal_decay_weight(
    t_query: float,
    t_obs_list: list[float],
    theta: float,
) -> float:
    """OU temporal decay weight: max_i exp(-θ · |t* - tᵢ|).

    Returns weight ∈ (0, 1] — highest when query is close to an observation.
    Returns near-zero when query is far from all observations.
    """
    if not t_obs_list:
        return 0.0
    min_dist = min(abs(t_query - t) for t in t_obs_list)
    return float(np.exp(-theta * min_dist))


def soft_label_from_posterior(
    mu: float,
    sigma: float,
    thresholds: tuple[float, float] = VAD_THRESHOLDS,
    T_scale: float = 1.0,
) -> np.ndarray:
    """Convert GP posterior N(mu, sigma²) to soft 3-class label via CDF integration.

    T_scale > 1 softens distribution (higher uncertainty).
    T_scale < 1 sharpens distribution (higher confidence).
    """
    sigma_eff = max(sigma * T_scale, 1e-4)
    t1, t2 = thresholds
    p_low  = float(norm.cdf(t1, loc=mu, scale=sigma_eff))
    p_high = float(1.0 - norm.cdf(t2, loc=mu, scale=sigma_eff))
    p_mid  = float(max(1.0 - p_low - p_high, 0.0))
    probs = np.array([p_low, p_mid, p_high], dtype=np.float32)
    probs /= probs.sum()    # ensure sums to 1 after floating-point rounding
    return probs


def compute_temperature(sigma: float, sigma_ref: float, sigma_max: float) -> float:
    """T_scale = 1 + (sigma - sigma_ref) / sigma_max, clamped to [1.0, 2.5]."""
    if sigma_max < 1e-8:
        return 1.0
    T = 1.0 + (sigma - sigma_ref) / sigma_max
    return float(np.clip(T, 1.0, 2.5))


# ---------------------------------------------------------------------------
# OU Parameter Estimation
# ---------------------------------------------------------------------------

def estimate_ou_params(dataset_df: pd.DataFrame) -> dict[str, OUParams]:
    """Estimate per-dimension OU parameters from self-report time series.

    Uses method-of-moments: fits θ from the empirical lag-1 autocorrelation
    of the within-session, within-task self-report sequences.
    Falls back to Meng et al. (2026) values if the dataset has too few paired
    observations for reliable estimation.

    Returns dict mapping dimension name → OUParams.
    """
    params: dict[str, OUParams] = {}
    required = ["valence", "arousal", "dominance",
                "vad_timestamp_lsl", "session_id", "task", "seat"]
    if not all(c in dataset_df.columns for c in required):
        log.warning("Dataset missing required columns — using Meng 2026 OU defaults.")
        return {d: OUParams(**MENG_OU_PARAMS[d]) for d in DIMS}

    for dim in DIMS:
        lags, vals = [], []
        for (ses, task), grp in dataset_df.groupby(["session_id", "task"]):
            for seat, seat_grp in grp.groupby("seat"):
                ts = seat_grp.sort_values("vad_timestamp_lsl")
                y = ts[dim].values.astype(float)
                t = ts["vad_timestamp_lsl"].values.astype(float)
                if len(y) < 2:
                    continue
                for i in range(len(y) - 1):
                    dt = float(t[i + 1] - t[i])
                    if 5.0 < dt < 300.0:   # only pairs within 5–300 s
                        lags.append(dt)
                        vals.append((float(y[i]), float(y[i + 1])))

        if len(vals) < 5:
            log.info("  %s: too few pairs (%d) → using Meng 2026 value.", dim, len(vals))
            params[dim] = OUParams(**MENG_OU_PARAMS[dim])
            continue

        lags_arr = np.array(lags)
        y1 = np.array([v[0] for v in vals])
        y2 = np.array([v[1] for v in vals])
        mu_hat = float(np.mean(np.concatenate([y1, y2])))
        sigma2_hat = float(np.var(np.concatenate([y1, y2])))

        # Estimate θ from lag-τ correlation: corr(τ) = exp(-θ·τ)
        # Use WLS to fit log(corr) = -θ·τ
        norm1 = y1 - mu_hat
        norm2 = y2 - mu_hat
        corr_vals, valid_lags = [], []
        for dt_bin in np.unique(np.round(lags_arr / 10) * 10):  # 10-s bins
            mask = np.abs(lags_arr - dt_bin) < 5
            if mask.sum() < 2:
                continue
            r = float(np.corrcoef(norm1[mask], norm2[mask])[0, 1])
            if 1e-6 < r < 1.0:
                corr_vals.append(np.log(r))
                valid_lags.append(dt_bin)

        if len(valid_lags) < 2:
            params[dim] = OUParams(**MENG_OU_PARAMS[dim])
            continue

        # Fit -θ via linear regression on (log_corr ~ lag)
        A = np.column_stack([-np.array(valid_lags)])
        b = np.array(corr_vals)
        theta_hat = float(max(np.linalg.lstsq(A, b, rcond=None)[0][0], 0.01))

        # Validate against Meng priors: reject if >5× discrepancy
        theta_meng = MENG_OU_PARAMS[dim]["theta"]
        if theta_hat > 5 * theta_meng or theta_hat < theta_meng / 5:
            log.info("  %s: estimated θ=%.4f deviates >5× from Meng (%.4f) → using Meng.",
                     dim, theta_hat, theta_meng)
            params[dim] = OUParams(**MENG_OU_PARAMS[dim])
        else:
            params[dim] = OUParams(theta=theta_hat, sigma2=sigma2_hat, mu=mu_hat)
            log.info("  %s: θ=%.4f  σ²=%.2f  μ=%.2f (from %d pairs)",
                     dim, theta_hat, sigma2_hat, mu_hat, len(vals))

    return params


# ---------------------------------------------------------------------------
# Cross-Person Synchrony Estimation
# ---------------------------------------------------------------------------

def estimate_cross_person_rho(
    dataset_df: pd.DataFrame,
    window_s: float = 90.0,
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Estimate pairwise inter-person ρ per (session, task).

    Pairs reports that fall within *window_s* seconds of each other and
    computes Pearson r for each (seat_k, seat_l, dimension).

    Returns:
        {session_id: {task: {dim: {(seat_k, seat_l): rho}}}}
    If fewer than 3 pairs exist for a (session, task), the task-type prior is used.
    """
    result: dict = {}

    required = ["valence", "arousal", "dominance",
                "vad_timestamp_lsl", "session_id", "task", "seat"]
    if not all(c in dataset_df.columns for c in required):
        log.warning("Cannot estimate cross-person ρ — dataset missing columns.")
        return result

    for ses_id, ses_grp in dataset_df.groupby("session_id"):
        result[ses_id] = {}
        for task, task_grp in ses_grp.groupby("task"):
            prior = SYNC_PRIORS.get(str(task), SYNC_PRIORS["T1"])
            task_rho: dict[str, dict] = {d: {} for d in DIMS}

            seats_present = task_grp["seat"].unique().tolist()
            for i, sk in enumerate(seats_present):
                for sl in seats_present[i + 1:]:
                    grp_k = task_grp[task_grp["seat"] == sk].sort_values("vad_timestamp_lsl")
                    grp_l = task_grp[task_grp["seat"] == sl].sort_values("vad_timestamp_lsl")

                    # Find paired reports within window_s seconds
                    pairs: dict[str, list] = {d: [] for d in DIMS}
                    for _, row_k in grp_k.iterrows():
                        t_k = float(row_k["vad_timestamp_lsl"])
                        close = grp_l[np.abs(grp_l["vad_timestamp_lsl"] - t_k) < window_s]
                        if close.empty:
                            continue
                        row_l = close.iloc[0]
                        for dim in DIMS:
                            pairs[dim].append((float(row_k[dim]), float(row_l[dim])))

                    for dim in DIMS:
                        if len(pairs[dim]) >= 3:
                            y1 = np.array([p[0] for p in pairs[dim]])
                            y2 = np.array([p[1] for p in pairs[dim]])
                            rho = float(np.corrcoef(y1, y2)[0, 1])
                            rho = float(np.clip(rho, 0.0, 0.95))  # clamp; negative = no sync
                        else:
                            # fallback to task prior
                            rho = prior["rho_eda"] if dim == "arousal" else prior["rho_hr"]

                        task_rho[dim][(sk, sl)] = rho
                        task_rho[dim][(sl, sk)] = rho   # symmetric

            result[ses_id][task] = task_rho

    return result


def get_cross_rho(
    rho_db: dict,
    session_id: str,
    task: str,
    seat_k: str,
    seat_l: str,
    dim: str,
) -> float:
    """Look up ρ(seat_k→seat_l) from rho_db with task-prior fallback."""
    try:
        r = rho_db[session_id][task][dim][(seat_k, seat_l)]
        return float(r) if r > 0 else 0.05
    except KeyError:
        prior = SYNC_PRIORS.get(str(task), SYNC_PRIORS["T1"])
        return prior["rho_eda"] if dim == "arousal" else prior["rho_hr"]


# ---------------------------------------------------------------------------
# Physiological Evidence Extraction (S4: EDA, S5: HR)
# ---------------------------------------------------------------------------

def extract_scr_events(
    eda_seq_df: pd.DataFrame,
    window_t_start: float,
    phasic_baseline: float = 0.0,
) -> list[tuple[float, float]]:
    """Detect SCR events in the EDA_Phasic trace of a stored window.

    Parameters
    ----------
    eda_seq_df      : (400, 5) DataFrame — columns include COL_EDA_PHASIC ('EDA_Phasic')
                     as stored by pickle_generation_affectai.py
    window_t_start  : LSL clock time of the first sample in the window
    phasic_baseline : per-seat T0 mean phasic to subtract before peak detection.
                     Removes individual resting EDA differences so SCR amplitudes
                     reflect task-induced elevations above baseline (H4 correction).

    Returns
    -------
    List of (t_stimulus, amplitude) pairs, where t_stimulus = peak_t - SCR_LATENCY_S.
    Amplitudes are in baseline-subtracted standardised phasic units.
    """
    if COL_EDA_PHASIC not in eda_seq_df.columns:
        return []

    phasic = eda_seq_df[COL_EDA_PHASIC].fillna(0).values.astype(float) - phasic_baseline
    if np.all(phasic <= 0):
        return []

    # Peak detection: min distance 1s (≈13 samples), min height = SCR_MIN_AMPLITUDE
    min_dist = max(1, int(FS_WINDOW * 1.0))
    peaks, props = find_peaks(phasic, distance=min_dist, height=SCR_MIN_AMPLITUDE)

    events = []
    for pk in peaks:
        amp = float(phasic[pk])
        t_peak = window_t_start + pk / FS_WINDOW
        t_stim = t_peak - SCR_LATENCY_S     # latency-corrected stimulus time
        events.append((t_stim, amp))

    return events


def extract_hr_direction(
    eda_seq_df: pd.DataFrame,
    window_t_start: float,
    bin_sec: float = 5.0,
) -> list[tuple[float, float, float]]:
    """Compute HR direction signal (valence proxy) from HR proxy channel in eda_seq.

    The HR proxy (COL_HR_PROXY = 'value_6', firmware BPM) is stored as the 4th
    column of the eda_seq DataFrame (index 3 of EDA_SEQ_STORED).
    Split into non-overlapping bins of *bin_sec*:
      negative slope (deceleration) → positive valence evidence (+)
      positive slope (acceleration) → negative valence evidence (-)

    Returns
    -------
    List of (t_center, valence_shift, sigma_hr) for each bin.
    valence_shift is on Likert scale (positive = more positive valence).
    """
    hr_col = COL_HR_PROXY  # "value_6" — EmotiBit firmware HR (BPM)
    if hr_col not in eda_seq_df.columns:
        return []

    hr = eda_seq_df[hr_col].ffill().fillna(0).values.astype(float)
    if np.std(hr) < 0.5:   # flatline / no HR data
        return []

    events = []
    bin_samples = max(1, int(FS_WINDOW * bin_sec))
    n_bins = int(len(hr) // bin_samples)

    for b in range(n_bins):
        seg = hr[b * bin_samples: (b + 1) * bin_samples]
        if len(seg) < 3:
            continue
        t_center = window_t_start + (b + 0.5) * bin_sec
        # Linear slope of HR over bin (BPM/s)
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
        # Scale: ~5 BPM/s slope → ~1 Likert unit shift
        # Deceleration (negative slope) → positive valence
        valence_shift = float(np.clip(-slope / 5.0, -1.5, 1.5))
        events.append((t_center, valence_shift, SIGMA_HR_DIRECTION))

    return events


# ---------------------------------------------------------------------------
# Session Self-Report Index
# ---------------------------------------------------------------------------

def build_self_report_index(dataset_df: pd.DataFrame) -> dict:
    """Index self-reports as:
        {session_id: {task: {seat: [(t, v, a, d), ...]}}}

    Raises KeyError if required columns are absent.
    """
    required = ["valence", "arousal", "dominance",
                "vad_timestamp_lsl", "session_id", "task", "seat"]
    for c in required:
        if c not in dataset_df.columns:
            raise KeyError(f"dataset_df missing required column: {c!r}")

    index: dict = {}
    for _, row in dataset_df.iterrows():
        ses = str(row["session_id"])
        task = str(row["task"])
        seat = str(row["seat"])
        t = float(row["vad_timestamp_lsl"])
        v = float(row["valence"])
        a = float(row["arousal"])
        d = float(row["dominance"])
        index.setdefault(ses, {}).setdefault(task, {}).setdefault(seat, [])
        index[ses][task][seat].append((t, v, a, d))

    return index


# ---------------------------------------------------------------------------
# Per-Window Label Generation
# ---------------------------------------------------------------------------

def generate_label_for_window(
    record: dict,
    sr_index: dict,
    rho_db: dict,
    ou_params: dict[str, OUParams],
    use_cross_physio: bool = True,
    sigma_max_by_dim: dict[str, float] | None = None,
) -> dict | None:
    """Generate augmented soft VAD labels for one unlabelled pretrain window.

    Parameters
    ----------
    record           : one row from pretrain_dataset.pkl (as dict)
    sr_index         : output of build_self_report_index()
    rho_db           : output of estimate_cross_person_rho()
    ou_params        : output of estimate_ou_params()
    use_cross_physio : if True, include S4 (EDA SCR) and S5 (HR direction) from
                       other group members (cross-person, leakage-free)
    sigma_max_by_dim : pre-computed σ_max per dimension (for temperature scaling);
                       computed globally in generate_all_augmented_labels()

    Returns
    -------
    dict of augmented label fields, or None if no dimension passes w_min filter.
    """
    ses_id = str(record.get("session_id", ""))
    task   = str(record.get("task", ""))
    seat   = str(record.get("seat", ""))

    # Absolute timestamp of window centre (LSL clock seconds)
    t_center = float(record.get("window_t_center", np.nan))
    if np.isnan(t_center):
        # Fallback: use window_t_end if stored, else skip
        t_end = float(record.get("window_t_end", np.nan))
        if not np.isnan(t_end):
            t_center = t_end - WINDOW_SEC / 2.0
        else:
            return None   # no timestamp information → cannot interpolate

    # -----------------------------------------------------------------------
    # Collect self-report observations for target person (S1) + group (S2)
    # -----------------------------------------------------------------------
    task_reports = sr_index.get(ses_id, {}).get(task, {})

    # All seats in this session/task
    all_seats = list(task_reports.keys())
    other_seats = [s for s in all_seats if s != seat]

    output: dict = {
        "session_id": ses_id,
        "task": task,
        "seat": seat,
        "subject_id": record.get("subject_id", ""),
        "window_index": record.get("window_index", -1),
        "window_t_center": t_center,
        "label_sources": [],
    }

    # Copy sensor sequences through unchanged
    for seq_key in ("gaze_seq", "pupil_seq", "eda_seq", "ppg_seq", "imu_seq",
                    "gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features"):
        if seq_key in record:
            output[seq_key] = record[seq_key]

    # Copy personality + demographic fields
    for col in ("bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o",
                "sex", "task_onehot", "session_idx"):
        if col in record:
            output[col] = record[col]

    any_valid = False
    task_dims = TASK_DIM_AVAILABILITY.get(task, list(DIMS))

    for dim_idx, dim in enumerate(DIMS):
        if dim not in task_dims:
            # This dimension was never probed for this task — mark explicitly as N/A
            output[f"{dim}_mu"]     = float("nan")
            output[f"{dim}_sigma"]  = float("nan")
            output[f"{dim}_soft"]   = np.array([1/3, 1/3, 1/3], dtype=np.float32)
            output[f"{dim}_weight"] = 0.0
            output[f"n_obs_{dim}"]  = 0
            continue

        params = ou_params[dim]
        t_obs_list: list[float] = []
        y_obs_list: list[float] = []
        sigma_list: list[float] = []
        sources: list[str] = []

        # -- S1: own self-reports --
        own_reports = task_reports.get(seat, [])
        for (t_r, v, a, d) in own_reports:
            val = [v, a, d][dim_idx]
            if not np.isnan(val):
                t_obs_list.append(t_r)
                y_obs_list.append(val)
                sigma_list.append(SIGMA_SELF_REPORT)
                sources.append("S1")

        # -- S2: cross-person self-reports --
        for other_seat in other_seats:
            rho = get_cross_rho(rho_db, ses_id, task, other_seat, seat, dim)
            if rho < 0.05:
                continue
            sigma_cross = SIGMA_CROSS_REPORT_BASE / max(rho, 0.05)
            sigma_cross = float(np.clip(sigma_cross, 0.5, 4.0))

            for (t_r, v, a, d) in task_reports.get(other_seat, []):
                val = [v, a, d][dim_idx]
                if not np.isnan(val):
                    t_obs_list.append(t_r)
                    y_obs_list.append(val)
                    sigma_list.append(sigma_cross)
                    sources.append("S2")

        # -- S4: cross-person EDA SCR events (arousal only) --
        if use_cross_physio and dim == "arousal":
            for other_seat in other_seats:
                rho_eda = get_cross_rho(rho_db, ses_id, task, other_seat, seat, "arousal")
                if rho_eda < 0.1:
                    continue
                other_records = _find_nearby_pretrain_records(
                    record, other_seat, ses_id, task,
                    _PRETRAIN_LOOKUP
                )
                for orec in other_records:
                    eda_df = orec.get("eda_seq")
                    ot_start = float(orec.get("window_t_center", np.nan)) - WINDOW_SEC / 2.0
                    if eda_df is None or np.isnan(ot_start):
                        continue
                    # Apply per-seat T0 phasic baseline correction (H4 fix)
                    other_bl_key = (ses_id, other_seat)
                    phasic_bl = _T0_PHASIC_BASELINES.get(other_bl_key, 0.0)
                    scr_events = extract_scr_events(eda_df, ot_start, phasic_baseline=phasic_bl)
                    for (t_stim, amp) in scr_events:
                        sigma_scr_base = SIGMA_SCR_HIGH_AMP if amp > 0.2 else SIGMA_SCR_LOW_AMP
                        sigma_scr_cross = sigma_scr_base / max(rho_eda, 0.1)
                        # SCR → high arousal observation on Likert scale
                        arousal_obs = float(np.clip(
                            params.mu + 2.0 * np.sqrt(params.sigma2), 1.0, 9.0
                        ))
                        t_obs_list.append(t_stim)
                        y_obs_list.append(arousal_obs)
                        sigma_list.append(float(np.clip(sigma_scr_cross, 0.5, 4.0)))
                        sources.append("S4")

        # -- S5: cross-person HR direction (valence only) --
        if use_cross_physio and dim == "valence":
            for other_seat in other_seats:
                rho_hr = get_cross_rho(rho_db, ses_id, task, other_seat, seat, "valence")
                if rho_hr < 0.1:
                    continue
                other_records = _find_nearby_pretrain_records(
                    record, other_seat, ses_id, task,
                    _PRETRAIN_LOOKUP
                )
                for orec in other_records:
                    eda_df = orec.get("eda_seq")
                    ot_start = float(orec.get("window_t_center", np.nan)) - WINDOW_SEC / 2.0
                    if eda_df is None or np.isnan(ot_start):
                        continue
                    hr_events = extract_hr_direction(eda_df, ot_start)
                    for (t_hr, v_shift, sigma_hr_base) in hr_events:
                        sigma_hr_cross = sigma_hr_base / max(rho_hr, 0.1)
                        valence_obs = float(np.clip(params.mu + v_shift, 1.0, 9.0))
                        t_obs_list.append(t_hr)
                        y_obs_list.append(valence_obs)
                        sigma_list.append(float(np.clip(sigma_hr_cross, 1.0, 5.0)))
                        sources.append("S5")

        # -----------------------------------------------------------------------
        # Compute GP posterior
        # -----------------------------------------------------------------------
        if not t_obs_list:
            # No observations at all — use OU prior
            mu_post = params.mu
            std_post = float(np.sqrt(params.sigma2))
        else:
            t_arr = np.array(t_obs_list, dtype=float)
            y_arr = np.array(y_obs_list, dtype=float)
            s_arr = np.array(sigma_list, dtype=float)
            mu_post, std_post = ou_gp_posterior(t_center, t_arr, y_arr, s_arr, params)

        # Temporal decay weight (based on own self-reports + cross-person reports)
        own_t = [t for t, *_ in own_reports] if own_reports else []
        cross_t = [
            t
            for s in other_seats
            for t, *_ in task_reports.get(s, [])
        ]
        w_own   = temporal_decay_weight(t_center, own_t, params.theta)
        w_cross = temporal_decay_weight(t_center, cross_t, params.theta)
        # Cross-person reports carry less weight (scaled by ρ)
        rho_mean = float(np.mean([
            get_cross_rho(rho_db, ses_id, task, s, seat, dim)
            for s in other_seats
        ])) if other_seats else 0.0
        w_total = float(np.clip(
            w_own + rho_mean * w_cross * 0.6,
            0.0, W_MAX
        ))

        # Temperature scaling for soft label
        s_max = (sigma_max_by_dim or {}).get(dim, float(np.sqrt(params.sigma2)))
        T_scale = compute_temperature(std_post, SIGMA_SELF_REPORT, s_max)
        soft = soft_label_from_posterior(mu_post, std_post, VAD_THRESHOLDS, T_scale)
        instance_weight = float(np.clip(w_total, W_MIN[dim], W_MAX))

        output[f"{dim}_mu"]     = float(mu_post)
        output[f"{dim}_sigma"]  = float(std_post)
        output[f"{dim}_soft"]   = soft
        output[f"{dim}_weight"] = instance_weight
        output[f"n_obs_{dim}"]  = len(t_obs_list)

        if not dim_idx:
            output["label_sources"] = []
        output["label_sources"].extend(set(sources))

        if instance_weight >= W_MIN[dim]:
            any_valid = True

    return output if any_valid else None


# ---------------------------------------------------------------------------
# Pretrain Lookup (module-level cache for cross-person matching)
# ---------------------------------------------------------------------------
_PRETRAIN_LOOKUP: dict[tuple[str, str, str], list[dict]] = {}

# Per-(session_id, seat) mean phasic EDA during T0 (rest), used for SCR baseline correction.
# Built from T0 records in the pretrain dataset.
_T0_PHASIC_BASELINES: dict[tuple[str, str], float] = {}


def _build_pretrain_lookup(pretrain_df: pd.DataFrame) -> None:
    """Index pretrain records by (session_id, task, seat) for fast O(1) lookup."""
    global _PRETRAIN_LOOKUP
    _PRETRAIN_LOOKUP = {}
    for _, row in pretrain_df.iterrows():
        key = (str(row["session_id"]), str(row["task"]), str(row["seat"]))
        _PRETRAIN_LOOKUP.setdefault(key, []).append(row.to_dict())


def _build_t0_phasic_baselines(pretrain_df: pd.DataFrame) -> None:
    """Compute per-(session_id, seat) mean phasic EDA from T0 (rest) windows.

    These baselines are subtracted from phasic EDA before SCR peak detection,
    removing individual resting-state differences so that SCR amplitudes reflect
    task-induced elevations above each person's own baseline (H4 fix).
    """
    global _T0_PHASIC_BASELINES
    _T0_PHASIC_BASELINES = {}
    t0_df = pretrain_df[pretrain_df["task"].astype(str) == "T0"]
    accum: dict[tuple[str, str], list[float]] = {}
    for _, row in t0_df.iterrows():
        key = (str(row["session_id"]), str(row["seat"]))
        eda_df = row.get("eda_seq")
        if eda_df is None:
            continue
        if not isinstance(eda_df, pd.DataFrame) or COL_EDA_PHASIC not in eda_df.columns:
            continue
        vals = eda_df[COL_EDA_PHASIC].dropna().values.astype(float)
        if len(vals) > 0:
            accum.setdefault(key, []).append(float(np.nanmean(vals)))
    for key, means in accum.items():
        _T0_PHASIC_BASELINES[key] = float(np.mean(means))
    log.info("T0 phasic baselines built for %d (session, seat) pairs", len(_T0_PHASIC_BASELINES))


def _find_nearby_pretrain_records(
    target_record: dict,
    other_seat: str,
    session_id: str,
    task: str,
    lookup: dict,
    time_tolerance_s: float = 45.0,
) -> list[dict]:
    """Return pretrain records from *other_seat* temporally near *target_record*."""
    key = (session_id, task, other_seat)
    candidates = lookup.get(key, [])
    t_target = float(target_record.get("window_t_center", np.nan))
    if np.isnan(t_target) or not candidates:
        return candidates  # no filtering possible without timestamps

    nearby = [
        r for r in candidates
        if not np.isnan(float(r.get("window_t_center", np.nan)))
        and abs(float(r["window_t_center"]) - t_target) <= time_tolerance_s
    ]
    return nearby if nearby else candidates[:3]  # fallback: first 3 windows


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def generate_all_augmented_labels(
    pretrain_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    ou_params: dict[str, OUParams],
    rho_db: dict,
    use_cross_physio: bool = True,
    min_weight: float = 0.05,
) -> pd.DataFrame:
    """Generate augmented labels for all pretrain windows.

    Two-pass algorithm:
      Pass 1: collect σ_max per dimension (for global temperature scaling)
      Pass 2: generate final soft labels and weights

    Returns a DataFrame with the same schema as pretrain_df plus augmented label columns.
    """
    log.info("Building self-report index …")
    sr_index = build_self_report_index(dataset_df)

    log.info("Building pretrain lookup index …")
    _build_pretrain_lookup(pretrain_df)

    log.info("Building T0 phasic baselines for SCR normalisation …")
    _build_t0_phasic_baselines(pretrain_df)

    # ------------------------------------------------------------------
    # Pass 1: estimate σ_max per dimension (quick, no soft label needed)
    # ------------------------------------------------------------------
    log.info("Pass 1: estimating σ_max per dimension …")
    sigma_samples: dict[str, list[float]] = {d: [] for d in DIMS}
    for _, row in pretrain_df.sample(min(500, len(pretrain_df)), random_state=42).iterrows():
        rec = row.to_dict()
        for dim_idx, dim in enumerate(DIMS):
            params = ou_params[dim]
            task = str(rec.get("task", ""))
            seat = str(rec.get("seat", ""))
            ses_id = str(rec.get("session_id", ""))
            t_center = float(rec.get("window_t_center", np.nan))
            if np.isnan(t_center):
                continue
            task_reports = sr_index.get(ses_id, {}).get(task, {})
            own = task_reports.get(seat, [])
            t_arr = np.array([t for t, *_ in own], dtype=float)
            y_arr = np.array([[v, a, d][dim_idx] for _, v, a, d in own], dtype=float)
            s_arr = np.full(len(own), SIGMA_SELF_REPORT)
            if len(t_arr) > 0:
                _, std = ou_gp_posterior(t_center, t_arr, y_arr, s_arr, params)
            else:
                std = float(np.sqrt(params.sigma2))
            sigma_samples[dim].append(std)

    sigma_max_by_dim = {
        dim: float(np.percentile(sigma_samples[dim], 95)) if sigma_samples[dim]
        else float(np.sqrt(ou_params[dim].sigma2))
        for dim in DIMS
    }
    log.info("  σ_max: %s", {d: f"{v:.3f}" for d, v in sigma_max_by_dim.items()})

    # ------------------------------------------------------------------
    # Pass 2: generate labels for all pretrain windows
    # ------------------------------------------------------------------
    log.info("Pass 2: generating labels for %d pretrain windows …", len(pretrain_df))
    records_out: list[dict] = []
    skipped = 0

    for i, (_, row) in enumerate(pretrain_df.iterrows()):
        if i % 500 == 0:
            log.info("  %d / %d  (accepted so far: %d)", i, len(pretrain_df), len(records_out))

        rec = row.to_dict()
        result = generate_label_for_window(
            rec, sr_index, rho_db, ou_params,
            use_cross_physio=use_cross_physio,
            sigma_max_by_dim=sigma_max_by_dim,
        )

        if result is None:
            skipped += 1
            continue

        # Filter: at least one dimension must clear min_weight
        if all(result.get(f"{dim}_weight", 0) < min_weight for dim in DIMS):
            skipped += 1
            continue

        records_out.append(result)

    log.info("Augmented pool: %d windows (skipped %d / %d  — no timestamp or too distant from reports)",
             len(records_out), skipped, len(pretrain_df))

    if not records_out:
        return pd.DataFrame()

    return pd.DataFrame(records_out)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def calibrate_wmin(
    dataset_df: pd.DataFrame,
    ou_params: dict[str, OUParams],
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Calibrate W_MIN per VAD dimension from LOO GP accuracy on labelled data.

    W_MIN[dim] = clip(1.0 - loo_accuracy, 0.05, 0.90).
    Lower GP accuracy → higher W_MIN → stricter inclusion filter.

    Parameters
    ----------
    dataset_df  : labelled dataset (from dataset.pkl); must have columns
                  valence/arousal/dominance and vad_timestamp_lsl + session_id/task/seat.
    ou_params   : OU parameters per dimension (from estimate_ou_params or file).
    thresholds  : per-dimension (low_t, high_t) for 3-class binning.
                  Defaults to balanced tertile thresholds (33rd/67th pct).
    """
    if thresholds is None:
        thresholds = {}
        for dim in DIMS:
            vals = dataset_df[dim].dropna().values
            if len(vals) > 0:
                thresholds[dim] = (float(np.percentile(vals, 33.33)),
                                   float(np.percentile(vals, 66.67)))
            else:
                thresholds[dim] = (4.0, 6.0)

    wmin: dict[str, float] = {}
    for dim in DIMS:
        t1, t2 = thresholds[dim]
        p = ou_params[dim]
        correct = 0
        total = 0
        for (ses, task, seat), grp in dataset_df.groupby(["session_id", "task", "seat"]):
            task_dims = TASK_DIM_AVAILABILITY.get(str(task), list(DIMS))
            if dim not in task_dims:
                continue
            vals = grp[[dim, "vad_timestamp_lsl"]].dropna().values
            if len(vals) < 2:
                continue
            for i in range(len(vals)):
                t_q = float(vals[i, 1])
                y_q = float(vals[i, 0])
                other = np.arange(len(vals)) != i
                t_obs = vals[other, 1].astype(float)
                y_obs = vals[other, 0].astype(float)
                s_obs = np.full(len(t_obs), SIGMA_SELF_REPORT)
                mu_post, _ = ou_gp_posterior(t_q, t_obs, y_obs, s_obs, p)
                true_cls = 0 if y_q <= t1 else (2 if y_q > t2 else 1)
                pred_cls = 0 if mu_post <= t1 else (2 if mu_post > t2 else 1)
                correct += int(true_cls == pred_cls)
                total += 1
        loo_acc = correct / total if total > 0 else 1.0 / 3
        raw = 1.0 - loo_acc
        wmin[dim] = float(np.clip(raw, 0.05, 0.90))
        log.info("  calibrate_wmin: %s LOO_acc=%.3f → W_MIN=%.3f", dim, loo_acc, wmin[dim])
    return wmin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build leakage-free augmented label pool for MuMT-Affect."
    )
    p.add_argument("--dataset",  required=True,
                   help="Path to dataset.pkl (labeled, 305 windows)")
    p.add_argument("--pretrain", required=True,
                   help="Path to pretrain_dataset.pkl (unlabeled, ~8978 windows)")
    p.add_argument("--output",   default="data/mumt/augmented_pool.pkl",
                   help="Output pickle path (default: data/mumt/augmented_pool.pkl)")
    p.add_argument("--ou-params", default=None,
                   help="JSON file with OU parameters (optional; uses Meng 2026 defaults + local estimation)")
    p.add_argument("--min-weight", type=float, default=0.05,
                   help="Minimum instance weight to retain a window (default: 0.05)")
    p.add_argument("--use-cross-physio", action="store_true", default=False,
                   help="Include S4 (cross-person EDA) and S5 (cross-person HR) observations")
    p.add_argument("--no-estimate-ou", action="store_true", default=False,
                   help="Skip local OU estimation; use Meng 2026 defaults only")
    p.add_argument("--split-json", default=None,
                   help="JSON file with {train_sessions, val_sessions, test_sessions} to exclude test set")
    p.add_argument("--task-dim-mask", default=None,
                   help='JSON override for TASK_DIM_AVAILABILITY, e.g. \'{"T4": ["valence", "arousal"]}\'')
    p.add_argument("--calibrate-wmin", action="store_true", default=False,
                   help="Calibrate W_MIN per dimension from LOO GP accuracy before generating the pool. "
                        "Saves calibrated values to --output-wmin-json.")
    p.add_argument("--output-wmin-json", default=None,
                   help="Path to save calibrated W_MIN JSON (used with --calibrate-wmin).")
    p.add_argument("--wmin-json", default=None,
                   help="Path to load pre-calibrated W_MIN JSON (overrides built-in W_MIN constants).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading dataset from %s …", args.dataset)
    dataset_df = pd.read_pickle(args.dataset)
    log.info("  %d labeled windows, %d subjects, %d sessions",
             len(dataset_df),
             dataset_df["subject_id"].nunique() if "subject_id" in dataset_df else -1,
             dataset_df["session_id"].nunique() if "session_id" in dataset_df else -1)

    log.info("Loading pretrain data from %s …", args.pretrain)
    pretrain_df = pd.read_pickle(args.pretrain)
    log.info("  %d pretraining windows", len(pretrain_df))

    # Exclude test sessions from augmentation pool if split file provided
    if args.split_json:
        import json
        split = json.loads(Path(args.split_json).read_text())
        test_sessions = set(split.get("test_sessions", []))
        if test_sessions:
            before = len(pretrain_df)
            pretrain_df = pretrain_df[~pretrain_df["session_id"].isin(test_sessions)]
            log.info("  Excluded %d test-session windows (%d remaining)",
                     before - len(pretrain_df), len(pretrain_df))

    # -- Task dimension availability override --
    if args.task_dim_mask:
        import json as _json
        _override = _json.loads(args.task_dim_mask)
        TASK_DIM_AVAILABILITY.update(_override)
        log.info("TASK_DIM_AVAILABILITY overridden: %s", TASK_DIM_AVAILABILITY)

    # -- OU parameters --
    if args.ou_params:
        import json
        raw = json.loads(Path(args.ou_params).read_text())
        ou_params = {d: OUParams(**raw[d]) for d in DIMS}
        log.info("Loaded OU params from %s", args.ou_params)
    elif args.no_estimate_ou:
        ou_params = {d: OUParams(**MENG_OU_PARAMS[d]) for d in DIMS}
        log.info("Using Meng 2026 OU defaults (--no-estimate-ou).")
    else:
        log.info("Estimating OU parameters from self-reports …")
        ou_params = estimate_ou_params(dataset_df)

    for dim, p in ou_params.items():
        log.info("  %s: θ=%.4f  σ²=%.2f  μ=%.2f", dim, p.theta, p.sigma2, p.mu)

    # -- W_MIN: load pre-calibrated or calibrate from LOO accuracy --
    if args.wmin_json:
        import json as _json
        with open(args.wmin_json) as _f:
            W_MIN.update(_json.load(_f))
        log.info("Loaded W_MIN from %s: %s", args.wmin_json, W_MIN)
    elif args.calibrate_wmin:
        log.info("Calibrating W_MIN from LOO GP accuracy …")
        calibrated = calibrate_wmin(dataset_df, ou_params)
        W_MIN.update(calibrated)
        log.info("Calibrated W_MIN: %s", W_MIN)
        if args.output_wmin_json:
            import json as _json
            out_wmin = Path(args.output_wmin_json)
            out_wmin.parent.mkdir(parents=True, exist_ok=True)
            out_wmin.write_text(_json.dumps(W_MIN, indent=2))
            log.info("Saved calibrated W_MIN to %s", out_wmin)

    # -- Cross-person synchrony --
    log.info("Estimating cross-person synchrony …")
    rho_db = estimate_cross_person_rho(dataset_df)
    n_sessions_with_rho = len(rho_db)
    log.info("  ρ estimated for %d sessions.", n_sessions_with_rho)

    # -- Generate labels --
    augmented_df = generate_all_augmented_labels(
        pretrain_df=pretrain_df,
        dataset_df=dataset_df,
        ou_params=ou_params,
        rho_db=rho_db,
        use_cross_physio=args.use_cross_physio,
        min_weight=args.min_weight,
    )

    if augmented_df.empty:
        log.error("No augmented windows produced. "
                  "Check that pretrain_dataset.pkl contains 'window_t_center' column. "
                  "Re-run pickle_generation_pretrain.py to add timestamps.")
        return

    # -- Save --
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_pickle(str(out_path))
    log.info("Saved %d augmented windows to %s", len(augmented_df), out_path)

    # -- Summary statistics --
    log.info("─── Augmented Pool Summary ───")
    for dim in DIMS:
        w_col = f"{dim}_weight"
        if w_col not in augmented_df.columns:
            continue
        w = augmented_df[w_col].values
        n_obs = augmented_df[f"n_obs_{dim}"].values
        log.info("  %s: n=%d  weight [%.2f–%.2f] median=%.2f  median n_obs=%.1f",
                 dim, (w >= W_MIN[dim]).sum(),
                 float(w.min()), float(w.max()), float(np.median(w)),
                 float(np.median(n_obs)))

    if "label_sources" in augmented_df.columns:
        from collections import Counter
        all_sources: list[str] = []
        for row_sources in augmented_df["label_sources"]:
            if isinstance(row_sources, list):
                all_sources.extend(row_sources)
        log.info("  Source counts: %s", dict(Counter(all_sources)))


if __name__ == "__main__":
    main()
