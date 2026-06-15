"""improved_preprocessing.py

Enhanced feature extraction for gaze, pupil, and speech modalities.
Addresses three weaknesses in the original pickle_generation_affectai.py:

1. **Gaze features** (was 6, now 22):
   - Fixation detection (I-VT algorithm): fixation count, mean duration, dispersion
   - Saccade metrics: count, mean amplitude, peak velocity
   - Scan-path: total length, convex hull area, entropy of spatial distribution
   - Temporal: gaze transition frequency, dwell time in quadrants

2. **Pupil features** (was 7, now 18):
   - LHIPA (Low/High Index of Pupillary Activity) — cognitive load proxy
   - Pupil dilation derivative (rate of change)
   - TEPR (task-evoked pupillary response) — slope of initial pupil response
   - RMSSD of pupil size (variability)
   - Blink duration and inter-blink interval

3. **Speech integration** (best 6 features selected):
   - speech_energy_t0_delta (r=0.22 with arousal)
   - group_speech_total (r=0.16 with arousal)
   - speech_share (r=0.13 with valence)
   - speech_energy_mean (r=0.22 with valence)
   - group_n_speakers (r=0.11 with valence)
   - speech_t0_delta (r=0.17 with arousal)

Usage
-----
  python tools/mumt/improved_preprocessing.py
  python tools/mumt/improved_preprocessing.py \\
      --input  data/mumt/dataset_15s_speech_enriched.pkl \\
      --output data/mumt/dataset_15s_v2.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Gaze feature extraction ──────────────────────────────────────────────────

def _velocity(x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    """Point-to-point angular velocity (deg/s-like in normalised coords)."""
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    return dist / dt


def detect_fixations_ivt(x: np.ndarray, y: np.ndarray, sr: float,
                         velocity_threshold: float = 0.1) -> list[dict]:
    """I-VT (velocity threshold) fixation detection.

    Args:
        x, y: gaze coordinates [0,1] normalised screen space
        sr: sampling rate (Hz)
        velocity_threshold: threshold in normalised units/sample

    Returns:
        List of fixation dicts: {start, end, duration_ms, cx, cy, dispersion}
    """
    if len(x) < 3:
        return []
    dt = 1.0 / max(sr, 1)
    vel = _velocity(x, y, dt)
    # Classify each sample: vel < threshold → fixation
    is_fix = np.concatenate([[True], vel < velocity_threshold])

    fixations: list[dict] = []
    start_i = 0
    in_fix = is_fix[0]
    for i in range(1, len(is_fix)):
        if is_fix[i] != in_fix:
            if in_fix and (i - start_i) >= 3:  # min 3 samples
                fx = x[start_i:i]
                fy = y[start_i:i]
                dur_ms = (i - start_i) / max(sr, 1) * 1000
                cx, cy = float(np.mean(fx)), float(np.mean(fy))
                disp = float(np.max(fx) - np.min(fx) + np.max(fy) - np.min(fy))
                fixations.append({
                    "start": start_i, "end": i,
                    "duration_ms": dur_ms, "cx": cx, "cy": cy,
                    "dispersion": disp,
                })
            start_i = i
            in_fix = is_fix[i]
    # Handle last segment
    if in_fix and (len(is_fix) - start_i) >= 3:
        fx = x[start_i:]
        fy = y[start_i:]
        dur_ms = (len(is_fix) - start_i) / max(sr, 1) * 1000
        fixations.append({
            "start": start_i, "end": len(is_fix),
            "duration_ms": dur_ms, "cx": float(np.mean(fx)), "cy": float(np.mean(fy)),
            "dispersion": float(np.max(fx) - np.min(fx) + np.max(fy) - np.min(fy)),
        })
    return fixations


def compute_enhanced_gaze_features(gaze_seq: np.ndarray, sr: float = 50.0) -> dict:
    """Extract 22 gaze features from a (T, 9) gaze sequence.

    Columns: [gaze_x, gaze_y, dir_lx, dir_ly, dir_lz, dir_rx, dir_ry, dir_rz, validity]
    """
    feats: dict = {}
    if gaze_seq is None or len(gaze_seq) == 0:
        # Return NaN dict
        return {k: float("nan") for k in [
            "gaze_x_mean", "gaze_x_std", "gaze_y_mean", "gaze_y_std",
            "validity_rate", "saccade_count", "saccade_amp_mean", "saccade_vel_mean",
            "fixation_count", "fixation_dur_mean", "fixation_dur_std",
            "fixation_dispersion_mean", "scanpath_length", "scanpath_hull_area",
            "gaze_entropy_x", "gaze_entropy_y", "gaze_entropy_2d",
            "gaze_transition_rate", "dwell_q1", "dwell_q2", "dwell_q3", "dwell_q4",
        ]}

    x = gaze_seq[:, 0].astype(float)
    y = gaze_seq[:, 1].astype(float)
    validity = gaze_seq[:, -1].astype(float) if gaze_seq.shape[1] >= 9 else np.ones(len(x))

    # Filter to valid samples
    valid_mask = (validity > 0.5) & np.isfinite(x) & np.isfinite(y)
    xv = x[valid_mask]
    yv = y[valid_mask]

    feats["validity_rate"] = float(valid_mask.mean())

    if len(xv) < 5:
        feats.update({k: float("nan") for k in [
            "gaze_x_mean", "gaze_x_std", "gaze_y_mean", "gaze_y_std",
            "saccade_count", "saccade_amp_mean", "saccade_vel_mean",
            "fixation_count", "fixation_dur_mean", "fixation_dur_std",
            "fixation_dispersion_mean", "scanpath_length", "scanpath_hull_area",
            "gaze_entropy_x", "gaze_entropy_y", "gaze_entropy_2d",
            "gaze_transition_rate", "dwell_q1", "dwell_q2", "dwell_q3", "dwell_q4",
        ]})
        return feats

    # ── Basic position stats ──────────────────────────────────────────────
    feats["gaze_x_mean"] = float(np.mean(xv))
    feats["gaze_x_std"]  = float(np.std(xv))
    feats["gaze_y_mean"] = float(np.mean(yv))
    feats["gaze_y_std"]  = float(np.std(yv))

    # ── Saccade metrics ───────────────────────────────────────────────────
    dt = 1.0 / max(sr, 1)
    vel = _velocity(xv, yv, dt)
    # Threshold adapted for resampled data (~13 Hz): lower than raw 50 Hz
    # At 13 Hz, dt=0.075s; a 2°/s saccade = 2*0.075=0.15 normalised units
    saccade_threshold = 0.03 if sr < 20 else 0.1
    sac_mask = vel > saccade_threshold
    sac_amps = vel[sac_mask] * dt  # amplitude = velocity * dt
    feats["saccade_count"]    = float(sac_mask.sum())
    feats["saccade_amp_mean"] = float(np.mean(sac_amps)) if sac_amps.size > 0 else 0.0
    feats["saccade_vel_mean"] = float(np.mean(vel[sac_mask])) if sac_mask.sum() > 0 else 0.0

    # ── Fixation metrics (I-VT) ──────────────────────────────────────────
    fixations = detect_fixations_ivt(xv, yv, sr, velocity_threshold=saccade_threshold)
    feats["fixation_count"] = float(len(fixations))
    if fixations:
        durs = [f["duration_ms"] for f in fixations]
        disps = [f["dispersion"] for f in fixations]
        feats["fixation_dur_mean"]        = float(np.mean(durs))
        feats["fixation_dur_std"]         = float(np.std(durs))
        feats["fixation_dispersion_mean"] = float(np.mean(disps))
    else:
        feats["fixation_dur_mean"]        = float("nan")
        feats["fixation_dur_std"]         = float("nan")
        feats["fixation_dispersion_mean"] = float("nan")

    # ── Scan-path ─────────────────────────────────────────────────────────
    dx = np.diff(xv)
    dy = np.diff(yv)
    scanpath = float(np.sum(np.sqrt(dx**2 + dy**2)))
    feats["scanpath_length"] = scanpath

    # Convex hull area of gaze points (spatial spread)
    try:
        if len(xv) >= 4:
            points = np.column_stack([xv, yv])
            hull = ConvexHull(points)
            feats["scanpath_hull_area"] = float(hull.volume)  # 2D: volume = area
        else:
            feats["scanpath_hull_area"] = 0.0
    except Exception:
        feats["scanpath_hull_area"] = 0.0

    # ── Entropy (spatial distribution) ────────────────────────────────────
    # Discretise into 10 bins per axis
    nbins = 10
    hx, _ = np.histogram(xv, bins=nbins, range=(0, 1), density=True)
    hy, _ = np.histogram(yv, bins=nbins, range=(0, 1), density=True)
    hx = hx / (hx.sum() + 1e-8)
    hy = hy / (hy.sum() + 1e-8)
    feats["gaze_entropy_x"] = float(scipy_entropy(hx + 1e-10))
    feats["gaze_entropy_y"] = float(scipy_entropy(hy + 1e-10))

    # 2D entropy (10x10 grid)
    h2d, _, _ = np.histogram2d(xv, yv, bins=nbins, range=[[0, 1], [0, 1]])
    h2d_flat = h2d.ravel() / (h2d.sum() + 1e-8)
    feats["gaze_entropy_2d"] = float(scipy_entropy(h2d_flat + 1e-10))

    # ── Gaze transitions (quadrant changes / sec) ─────────────────────────
    quads = (xv > 0.5).astype(int) * 2 + (yv > 0.5).astype(int)  # 0-3
    transitions = float(np.sum(np.diff(quads) != 0))
    window_sec = len(x) / max(sr, 1)
    feats["gaze_transition_rate"] = transitions / max(window_sec, 0.1)

    # ── Quadrant dwell proportions ────────────────────────────────────────
    for qi in range(4):
        feats[f"dwell_q{qi+1}"] = float(np.mean(quads == qi))

    return feats


# ── Pupil feature extraction ──────────────────────────────────────────────────

def compute_enhanced_pupil_features(pupil_seq: np.ndarray, sr: float = 50.0) -> dict:
    """Extract 18 pupil features from a (T, 3) pupil sequence.

    Columns: [pupil_left_mm, pupil_right_mm, validity]
    """
    feats: dict = {}
    nan_feats = {k: float("nan") for k in [
        "pupil_left_mean", "pupil_left_std", "pupil_right_mean", "pupil_right_std",
        "pupil_avg_mean", "pupil_avg_std", "pupil_avg_median",
        "blink_rate", "blink_dur_mean", "inter_blink_interval_mean",
        "pupil_rmssd", "pupil_slope", "pupil_range",
        "pupil_derivative_abs_mean", "pupil_derivative_std",
        "lhipa", "pupil_skewness", "pupil_kurtosis",
    ]}

    if pupil_seq is None or len(pupil_seq) == 0:
        return nan_feats

    left  = pupil_seq[:, 0].astype(float)
    right = pupil_seq[:, 1].astype(float)
    validity = pupil_seq[:, 2].astype(float) if pupil_seq.shape[1] >= 3 else np.ones(len(left))

    valid_mask = (validity > 0.5) & np.isfinite(left) & (left > 0.5) & (left < 10)
    lv = left[valid_mask]
    rv = right[valid_mask & np.isfinite(right) & (right > 0.5) & (right < 10)]

    if len(lv) < 10:
        return nan_feats

    avg = (lv + rv[:len(lv)]) / 2 if len(rv) >= len(lv) else lv

    # ── Basic stats ───────────────────────────────────────────────────────
    feats["pupil_left_mean"]  = float(np.mean(lv))
    feats["pupil_left_std"]   = float(np.std(lv))
    feats["pupil_right_mean"] = float(np.mean(rv)) if len(rv) > 0 else float("nan")
    feats["pupil_right_std"]  = float(np.std(rv))  if len(rv) > 0 else float("nan")
    feats["pupil_avg_mean"]   = float(np.mean(avg))
    feats["pupil_avg_std"]    = float(np.std(avg))
    feats["pupil_avg_median"] = float(np.median(avg))

    # ── Blink detection (validity drops) ──────────────────────────────────
    blink_mask = ~valid_mask
    # Find blink episodes (consecutive invalid samples)
    blink_starts: list[int] = []
    blink_ends: list[int] = []
    in_blink = False
    for i, is_blink in enumerate(blink_mask):
        if is_blink and not in_blink:
            blink_starts.append(i)
            in_blink = True
        elif not is_blink and in_blink:
            blink_ends.append(i)
            in_blink = False
    if in_blink:
        blink_ends.append(len(blink_mask))

    n_blinks = len(blink_starts)
    window_sec = len(left) / max(sr, 1)
    feats["blink_rate"] = float(n_blinks) / max(window_sec, 0.1)

    if n_blinks > 0:
        blink_durs = [(blink_ends[i] - blink_starts[i]) / max(sr, 1) * 1000
                      for i in range(n_blinks)]
        feats["blink_dur_mean"] = float(np.mean(blink_durs))
        if n_blinks > 1:
            ibis = [(blink_starts[i+1] - blink_ends[i]) / max(sr, 1)
                    for i in range(n_blinks - 1)]
            feats["inter_blink_interval_mean"] = float(np.mean(ibis))
        else:
            feats["inter_blink_interval_mean"] = float("nan")
    else:
        feats["blink_dur_mean"] = 0.0
        feats["inter_blink_interval_mean"] = float("nan")

    # ── Pupil dynamics ────────────────────────────────────────────────────
    # RMSSD (successive differences — variability measure)
    diffs = np.diff(avg)
    feats["pupil_rmssd"] = float(np.sqrt(np.mean(diffs**2)))

    # Slope (linear trend over window)
    t = np.arange(len(avg), dtype=float)
    if len(avg) > 2:
        slope = float(np.polyfit(t, avg, 1)[0])
    else:
        slope = 0.0
    feats["pupil_slope"] = slope

    # Range
    feats["pupil_range"] = float(np.max(avg) - np.min(avg))

    # Derivative stats
    deriv = np.diff(avg) * sr
    feats["pupil_derivative_abs_mean"] = float(np.mean(np.abs(deriv)))
    feats["pupil_derivative_std"]      = float(np.std(deriv))

    # ── LHIPA (Low/High Index of Pupillary Activity) ──────────────────────
    # Based on Duchowski et al. 2018 — ratio of wavelet energy in low vs high bands
    # Simplified: ratio of power in slow (< 0.5 Hz) vs fast (> 2 Hz) fluctuations
    try:
        if len(avg) > 20:
            # Smooth with Savitzky-Golay for slow component
            win_len = min(21, len(avg) - (1 if len(avg) % 2 == 0 else 0))
            if win_len >= 5 and win_len % 2 == 1:
                slow = savgol_filter(avg, win_len, 3)
            else:
                slow = avg
            fast = avg - slow
            power_slow = float(np.mean(slow**2))
            power_fast = float(np.mean(fast**2))
            feats["lhipa"] = power_fast / (power_slow + 1e-8)
        else:
            feats["lhipa"] = float("nan")
    except Exception:
        feats["lhipa"] = float("nan")

    # ── Distribution shape ────────────────────────────────────────────────
    from scipy.stats import skew, kurtosis
    feats["pupil_skewness"] = float(skew(avg))
    feats["pupil_kurtosis"] = float(kurtosis(avg))

    return feats


# ── Speech feature selection ──────────────────────────────────────────────────

# Top speech features ranked by univariate correlation with VAD
SELECTED_SPEECH_KEYS = [
    "speech_energy_t0_delta",    # r=0.22 arousal, 0.21 dom, 0.21 val
    "speech_energy_mean",        # r=0.22 valence, 0.13 arousal
    "speech_t0_delta",           # r=0.17 arousal, 0.13 dominance
    "group_speech_total",        # r=0.16 arousal
    "speech_share",              # r=0.13 valence
    "group_n_speakers",          # r=0.11 valence, 0.10 dominance
]


def extract_selected_speech(speech_dict: dict | None) -> dict:
    """Return only the top-6 most informative speech features."""
    if not isinstance(speech_dict, dict):
        return {k: 0.0 for k in SELECTED_SPEECH_KEYS}
    return {k: float(speech_dict.get(k, 0.0)) if not (
        isinstance(speech_dict.get(k), float) and np.isnan(speech_dict.get(k, 0.0))
    ) else 0.0 for k in SELECTED_SPEECH_KEYS}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def rebuild_features(df: pd.DataFrame, gaze_sr: float = 50.0) -> pd.DataFrame:
    """Recompute gaze and pupil features from stored sequences, add selected speech."""
    df = df.copy()
    new_gaze: list[dict]  = []
    new_pupil: list[dict] = []
    new_speech: list[dict] = []

    # Effective SR: sequences are resampled to 200 points over WINDOW_SEC
    # Original Tobii = 50 Hz, 15s window = 750 pts → resampled to 200 → ~13.3 Hz
    effective_sr = 200.0 / 15.0  # ~13.3 Hz
    log.info("Using effective SR=%.1f Hz for resampled sequences", effective_sr)

    for idx, row in df.iterrows():
        # Gaze — stored as DataFrame or ndarray
        gaze_seq = row.get("gaze_seq")
        if gaze_seq is not None and hasattr(gaze_seq, "values"):
            gaze_arr = gaze_seq.values.astype(float)
        elif isinstance(gaze_seq, np.ndarray) and gaze_seq.ndim == 2:
            gaze_arr = gaze_seq.astype(float)
        else:
            gaze_arr = None
        gf = compute_enhanced_gaze_features(gaze_arr, sr=effective_sr)
        new_gaze.append(gf)

        # Pupil — stored as DataFrame or ndarray
        pupil_seq = row.get("pupil_seq")
        if pupil_seq is not None and hasattr(pupil_seq, "values"):
            pupil_arr = pupil_seq.values.astype(float)
        elif isinstance(pupil_seq, np.ndarray) and pupil_seq.ndim == 2:
            pupil_arr = pupil_seq.astype(float)
        else:
            pupil_arr = None
        pf = compute_enhanced_pupil_features(pupil_arr, sr=effective_sr)
        new_pupil.append(pf)

        # Speech (selected features only)
        speech_orig = row.get("speech_features")
        sf = extract_selected_speech(speech_orig)
        new_speech.append(sf)

    df["gaze_features"]   = new_gaze
    df["pupil_features"]  = new_pupil
    df["speech_features"] = new_speech
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuild gaze/pupil features + select speech features",
    )
    p.add_argument("--input",  default="data/mumt/dataset_15s_speech_enriched.pkl",
                   help="Input pickle (with speech_features enriched)")
    p.add_argument("--output", default="data/mumt/dataset_15s_v2.pkl",
                   help="Output pickle with improved features")
    p.add_argument("--gaze-sr", type=float, default=50.0,
                   help="Tobii sampling rate (Hz)")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading: %s", in_path)
    with open(in_path, "rb") as f:
        df = pickle.load(f)
    log.info("Input: %d rows", len(df))

    # Check available sequences
    has_gaze  = "gaze_seq" in df.columns
    has_pupil = "pupil_seq" in df.columns
    log.info("Has gaze_seq: %s  Has pupil_seq: %s", has_gaze, has_pupil)
    if has_gaze:
        sample = df["gaze_seq"].iloc[0]
        log.info("  gaze_seq shape: %s", getattr(sample, "shape", "N/A"))
    if has_pupil:
        sample = df["pupil_seq"].iloc[0]
        log.info("  pupil_seq shape: %s", getattr(sample, "shape", "N/A"))

    # Report original feature counts
    orig_gaze = df["gaze_features"].iloc[0] if "gaze_features" in df.columns else {}
    orig_pupil = df["pupil_features"].iloc[0] if "pupil_features" in df.columns else {}
    log.info("Original gaze features: %d keys", len(orig_gaze) if isinstance(orig_gaze, dict) else 0)
    log.info("Original pupil features: %d keys", len(orig_pupil) if isinstance(orig_pupil, dict) else 0)

    log.info("Recomputing features...")
    df_out = rebuild_features(df, gaze_sr=args.gaze_sr)

    # Report new feature counts
    new_gaze = df_out["gaze_features"].iloc[0]
    new_pupil = df_out["pupil_features"].iloc[0]
    new_speech = df_out["speech_features"].iloc[0]
    log.info("New gaze features: %d keys — %s", len(new_gaze), sorted(new_gaze.keys()))
    log.info("New pupil features: %d keys — %s", len(new_pupil), sorted(new_pupil.keys()))
    log.info("New speech features: %d keys — %s", len(new_speech), sorted(new_speech.keys()))

    with open(out_path, "wb") as f:
        pickle.dump(df_out, f, protocol=4)
    log.info("Saved: %s  (%d rows)", out_path, len(df_out))

    # Quick quality check: correlation of new features with VAD
    print("\nTop correlations: new features vs VAD labels")
    all_keys = list(new_gaze.keys()) + list(new_pupil.keys()) + list(new_speech.keys())
    for feat_col, key_list in [("gaze_features", list(new_gaze.keys())),
                               ("pupil_features", list(new_pupil.keys())),
                               ("speech_features", list(new_speech.keys()))]:
        for k in key_list:
            vals = []
            for r in df_out[feat_col]:
                v = r.get(k, np.nan) if isinstance(r, dict) else np.nan
                vals.append(float(v) if v is not None else np.nan)
            df_out[f"__{k}"] = vals
            for dim in ["valence", "arousal", "dominance"]:
                sub = df_out[[f"__{k}", dim]].dropna()
                if len(sub) > 10:
                    r = sub.corr().iloc[0, 1]
                    if abs(r) > 0.10:
                        print(f"  {k:35s} vs {dim:10s}: r={r:.3f}")


if __name__ == "__main__":
    main()
