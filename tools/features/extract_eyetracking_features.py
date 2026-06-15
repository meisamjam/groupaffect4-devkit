"""Extract participant-level and rolling-window Tobii eye-tracking features with QC.

Expects task-split BIDS-like files under ``et/`` named with
``_task-T*_run-01_acq-P*_tobii.tsv.gz``.  Column layout (value_* indices):

  value_0  gaze_x        (normalised 0–1, scene camera horizontal)
  value_1  gaze_y        (normalised 0–1, scene camera vertical)
  value_2  pupil_left    (mm diameter)
  value_3  pupil_right   (mm diameter)
  value_4  gaze_valid    (1.0 = valid, 0.0 = invalid)

Outputs (written to --out-dir):
  et_participant_task.tsv    — one row per session × task × participant
  et_window_30s.tsv          — rolling 30-second windows (15 s step)
  et_qc_summary.tsv          — QC flags per participant-task
  et_feature_definitions.tsv — feature documentation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tools.features.common import (
        add_common_io_args,
        discover_session_dirs,
        linear_slope,
        numeric_columns,
        parse_participant_from_name,
        parse_session_from_path,
        parse_task_from_name,
        read_tsv,
        rolling_windows,
        sample_rate_hz,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.features.common import (  # type: ignore[no-redef]
        add_common_io_args,
        discover_session_dirs,
        linear_slope,
        numeric_columns,
        parse_participant_from_name,
        parse_session_from_path,
        parse_task_from_name,
        read_tsv,
        rolling_windows,
        sample_rate_hz,
    )

LOG = logging.getLogger("extract_eyetracking_features")

# ── QC flag catalogue ─────────────────────────────────────────────────────────

FLAG_DESCRIPTIONS: dict[str, str] = {
    "missing_et": (
        "No Tobii eye-tracking file found for this participant-task."
    ),
    "short_duration": (
        "Recording duration is below the minimum usable threshold (30 s)."
    ),
    "sample_rate_unavailable": (
        "Could not estimate the device sample rate from timestamps."
    ),
    "sample_rate_unusual": (
        "Estimated sample rate is outside the expected Tobii Pro Glasses 3 range (40–60 Hz)."
    ),
    "gaze_low_validity": (
        "Fewer than 70 % of gaze samples are marked valid; "
        "gaze-position and velocity features may be unreliable."
    ),
    "pupil_low_coverage": (
        "Combined (mean) pupil diameter is missing for more than 30 % of samples; "
        "pupil features should be treated with caution."
    ),
    "pupil_left_low_coverage": (
        "Left-eye pupil diameter is missing for more than 30 % of samples."
    ),
    "pupil_right_low_coverage": (
        "Right-eye pupil diameter is missing for more than 30 % of samples."
    ),
    "pupil_implausible": (
        "Mean pupil diameter is outside the physiologically plausible range (1.5–9.0 mm); "
        "check device calibration or unit scaling."
    ),
    "pupil_asymmetry_large": (
        "Mean absolute left–right pupil difference exceeds 1.5 mm; "
        "possible anisocoria or per-eye tracking failure."
    ),
    "blink_rate_unusual": (
        "Estimated blink rate is outside the expected range (3–45 blinks/min); "
        "check for systematic gaze-validity dropouts unrelated to blinking."
    ),
    "gaze_out_of_bounds": (
        "More than 10 % of valid gaze samples fall outside the scene-camera frame "
        "[0, 1] in either axis; possible calibration drift or head rotation artefacts."
    ),
    "pupil_velocity_spike": (
        "Mean pupil dilation velocity exceeds 1.0 mm/s, suggesting rapid "
        "artefactual jumps rather than smooth physiological dilation."
    ),
}

# ── Baseline-normalisable features ───────────────────────────────────────────

BASELINE_FEATURES: dict[str, str | None] = {
    "pupil_mean": "pupil_std",
    "pupil_left_mean": "pupil_left_std",
    "pupil_right_mean": "pupil_right_std",
    "gaze_x_mean": "gaze_x_std",
    "gaze_y_mean": "gaze_y_std",
}

# ── Feature catalogue (for documentation output) ─────────────────────────────

FEATURE_DEFINITIONS = [
    ("et_available", "Whether a Tobii eye-tracking file was found and read for this participant-task."),
    ("sample_rate_hz", "Estimated device sample rate in Hz from LSL timestamps."),
    ("duration_s", "Task/window duration covered by the eye-tracking samples."),
    # Gaze validity
    ("gaze_valid_frac", "Fraction of samples with gaze_valid == 1.0."),
    ("gaze_out_of_bounds_frac", "Fraction of valid samples where gaze_x or gaze_y falls outside [0, 1]."),
    # Gaze position
    ("gaze_x_mean", "Mean horizontal gaze position (normalised scene camera, 0=left, 1=right)."),
    ("gaze_y_mean", "Mean vertical gaze position (normalised scene camera, 0=top, 1=bottom)."),
    ("gaze_x_std", "Standard deviation of horizontal gaze position on valid samples."),
    ("gaze_y_std", "Standard deviation of vertical gaze position on valid samples."),
    ("gaze_dispersion", "Mean Euclidean distance of each valid gaze sample from the per-window centroid."),
    # Gaze dynamics
    ("gaze_velocity_mean", "Mean frame-to-frame gaze displacement per second (normalised units/s) on valid samples."),
    ("gaze_velocity_p95", "95th-percentile frame-to-frame gaze velocity; proxy for peak saccade rate."),
    # Pupil — binocular
    ("pupil_mean", "Mean binocular pupil diameter (mm); pairwise mean of available left/right values."),
    ("pupil_std", "Standard deviation of binocular pupil diameter (mm)."),
    ("pupil_slope_per_s", "Linear trend of binocular pupil diameter per second (mm/s)."),
    ("pupil_missing_frac", "Fraction of samples where binocular pupil diameter is NaN."),
    ("pupil_range", "Within-window range of binocular pupil diameter (max − min, mm)."),
    ("pupil_velocity_mean", "Mean absolute first-difference of pupil diameter per second (mm/s); proxy for dilation rate."),
    # Pupil — monocular
    ("pupil_left_mean", "Mean left-eye pupil diameter (mm)."),
    ("pupil_right_mean", "Mean right-eye pupil diameter (mm)."),
    ("pupil_left_std", "Standard deviation of left-eye pupil diameter (mm)."),
    ("pupil_right_std", "Standard deviation of right-eye pupil diameter (mm)."),
    ("pupil_left_missing_frac", "Fraction of samples where left-eye pupil diameter is NaN."),
    ("pupil_right_missing_frac", "Fraction of samples where right-eye pupil diameter is NaN."),
    ("pupil_lr_diff_mean", "Mean absolute left–right pupil difference (mm); proxy for anisocoria or asymmetric tracking failure."),
    # Blink estimates
    ("blink_count_est", "Estimated blink count: runs of consecutive invalid samples lasting 0.05–0.40 s."),
    ("blink_rate_per_min", "Estimated blink rate in blinks per minute."),
    ("blink_duration_mean_s", "Mean duration of detected blink-like invalidity gaps (s)."),
    # T0 baselines (added by add_t0_baselines)
    ("pupil_mean_delta_t0", "Within-participant change in mean pupil diameter relative to T0 baseline (mm)."),
    ("pupil_mean_z_t0", "Z-score of pupil mean relative to T0 baseline, normalised by T0 standard deviation."),
    # QC
    ("qc_flag", "Semicolon-separated machine-readable QC flags; 'ok' means no flag fired."),
    ("qc_notes", "Human-readable plain-language description of each QC flag."),
]

# ── Internal helpers ──────────────────────────────────────────────────────────


def _safe_nanmean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask]))


def _safe_nanstd(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.std(x[mask], ddof=0))


def _safe_nanpercentile(x: np.ndarray, pct: float) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.nanpercentile(x[mask], pct))


def _finite_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(x)))


def _value(df: pd.DataFrame, idx: int) -> np.ndarray:
    col = f"value_{idx}"
    if col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float, copy=False)


def _pairwise_nanmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise mean of two arrays, ignoring NaN in either."""
    if a.size == 0 and b.size == 0:
        return np.array([], dtype=float)
    if a.size == 0:
        return b.astype(float, copy=True)
    if b.size == 0:
        return a.astype(float, copy=True)
    n = min(len(a), len(b))
    stack = np.vstack([a[:n], b[:n]]).astype(float, copy=False)
    mask = np.isfinite(stack)
    counts = mask.sum(axis=0)
    sums = np.where(mask, stack, 0.0).sum(axis=0)
    out = np.full(n, np.nan, dtype=float)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _estimate_blinks(
    gaze_valid: np.ndarray,
    t: np.ndarray,
    min_blink_s: float = 0.05,
    max_blink_s: float = 0.40,
) -> tuple[int, float]:
    """Detect blink-like gaps: runs of invalid samples in a plausible blink-duration window.

    Returns (count, mean_duration_s). Returns (0, nan) if the signal is empty.
    """
    if gaze_valid.size < 3 or t.size < 3:
        return 0, float("nan")
    invalid = (~(gaze_valid > 0.5)).astype(int)
    # Find run-length encoding of invalidity
    diffs = np.diff(np.concatenate([[0], invalid, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0, float("nan")
    durations = []
    for s, e in zip(starts, ends):
        if s >= len(t) or e - 1 >= len(t):
            continue
        dur = float(t[min(e - 1, len(t) - 1)] - t[s])
        if min_blink_s <= dur <= max_blink_s:
            durations.append(dur)
    count = len(durations)
    mean_dur = float(np.mean(durations)) if durations else float("nan")
    return count, mean_dur


def _gaze_velocity(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    gaze_valid: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float]:
    """Compute mean and 95th-percentile frame-to-frame gaze velocity (normalised units/s)."""
    if gaze_x.size < 2:
        return float("nan"), float("nan")
    valid_mask = np.isfinite(gaze_x) & np.isfinite(gaze_y) & (gaze_valid > 0.5)
    if valid_mask.sum() < 2:
        return float("nan"), float("nan")
    vx = gaze_x.copy()
    vy = gaze_y.copy()
    vx[~valid_mask] = np.nan
    vy[~valid_mask] = np.nan
    dx = np.diff(vx)
    dy = np.diff(vy)
    dt = np.diff(t)
    dt_safe = np.where(dt > 0, dt, np.nan)
    speed = np.sqrt(dx**2 + dy**2) / dt_safe
    # Exclude transitions across invalid segments (NaN propagates naturally)
    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite_speed)), float(np.nanpercentile(finite_speed, 95))


def _qc_flag_text(flags: list[str]) -> str:
    return ";".join(flags) if flags else "ok"


def _qc_notes_text(flags: list[str]) -> str:
    if not flags:
        return "ok"
    return " | ".join(FLAG_DESCRIPTIONS.get(f, f) for f in flags)


# ── Core feature computation ──────────────────────────────────────────────────


def compute_et_features(
    df: pd.DataFrame,
    gaze_x_idx: int = 0,
    gaze_y_idx: int = 1,
    pupil_left_idx: int = 2,
    pupil_right_idx: int = 3,
    gaze_valid_idx: int = 4,
    min_duration_s: float = 30.0,
    sample_rate_min_hz: float = 40.0,
    sample_rate_max_hz: float = 60.0,
    min_gaze_valid_frac: float = 0.70,
    min_pupil_coverage_frac: float = 0.70,
    pupil_plausible_min_mm: float = 1.5,
    pupil_plausible_max_mm: float = 9.0,
    pupil_asymmetry_threshold_mm: float = 1.5,
    blink_rate_min_per_min: float = 3.0,
    blink_rate_max_per_min: float = 45.0,
    gaze_oob_threshold: float = 0.10,
    pupil_velocity_spike_threshold: float = 1.0,
) -> dict[str, object]:
    """Compute eye-tracking features and QC flags from a single participant-task TSV."""
    if df.empty or "lsl_time" not in df.columns:
        flags = ["missing_et"]
        return {
            "et_available": False,
            "sample_rate_hz": float("nan"),
            "duration_s": float("nan"),
            "gaze_valid_frac": float("nan"),
            "gaze_out_of_bounds_frac": float("nan"),
            "gaze_x_mean": float("nan"),
            "gaze_y_mean": float("nan"),
            "gaze_x_std": float("nan"),
            "gaze_y_std": float("nan"),
            "gaze_dispersion": float("nan"),
            "gaze_velocity_mean": float("nan"),
            "gaze_velocity_p95": float("nan"),
            "pupil_mean": float("nan"),
            "pupil_std": float("nan"),
            "pupil_slope_per_s": float("nan"),
            "pupil_missing_frac": 1.0,
            "pupil_range": float("nan"),
            "pupil_velocity_mean": float("nan"),
            "pupil_left_mean": float("nan"),
            "pupil_right_mean": float("nan"),
            "pupil_left_std": float("nan"),
            "pupil_right_std": float("nan"),
            "pupil_left_missing_frac": 1.0,
            "pupil_right_missing_frac": 1.0,
            "pupil_lr_diff_mean": float("nan"),
            "blink_count_est": 0,
            "blink_rate_per_min": float("nan"),
            "blink_duration_mean_s": float("nan"),
            "qc_flag": _qc_flag_text(flags),
            "qc_notes": _qc_notes_text(flags),
        }

    t = df["lsl_time"].to_numpy(dtype=float, copy=False)
    fs = sample_rate_hz(df)
    duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) > 1 else float("nan")

    gaze_x = _value(df, gaze_x_idx)
    gaze_y = _value(df, gaze_y_idx)
    left = _value(df, pupil_left_idx)
    right = _value(df, pupil_right_idx)
    gaze_valid_raw = _value(df, gaze_valid_idx)

    # Treat positive value as valid; also propagate nan
    gaze_valid = np.where(np.isfinite(gaze_valid_raw), gaze_valid_raw, 0.0)
    valid_mask = gaze_valid > 0.5

    # ── Gaze validity ──────────────────────────────────────────────────────
    gaze_valid_frac = float(np.mean(valid_mask)) if valid_mask.size > 0 else 0.0

    # ── Gaze position (valid samples only) ────────────────────────────────
    gx_valid = gaze_x[valid_mask] if valid_mask.any() else np.array([], dtype=float)
    gy_valid = gaze_y[valid_mask] if valid_mask.any() else np.array([], dtype=float)

    gaze_x_mean = _safe_nanmean(gx_valid)
    gaze_y_mean = _safe_nanmean(gy_valid)
    gaze_x_std = _safe_nanstd(gx_valid)
    gaze_y_std = _safe_nanstd(gy_valid)

    if gx_valid.size > 1:
        dist_from_centroid = np.sqrt(
            (gx_valid - gaze_x_mean) ** 2 + (gy_valid - gaze_y_mean) ** 2
        )
        gaze_dispersion = float(np.mean(dist_from_centroid[np.isfinite(dist_from_centroid)]))
    else:
        gaze_dispersion = float("nan")

    # Fraction of valid samples outside scene camera frame [0, 1]
    if gx_valid.size > 0:
        oob_mask = (
            np.isfinite(gx_valid) & np.isfinite(gy_valid)
            & ((gx_valid < 0.0) | (gx_valid > 1.0) | (gy_valid < 0.0) | (gy_valid > 1.0))
        )
        gaze_out_of_bounds_frac = float(np.mean(oob_mask))
    else:
        gaze_out_of_bounds_frac = float("nan")

    # ── Gaze velocity ─────────────────────────────────────────────────────
    gaze_velocity_mean, gaze_velocity_p95 = _gaze_velocity(gaze_x, gaze_y, gaze_valid, t)

    # ── Pupil diameter ────────────────────────────────────────────────────
    pupil = _pairwise_nanmean(left, right)

    pupil_mean = _safe_nanmean(pupil)
    pupil_std = _safe_nanstd(pupil)
    pupil_missing_frac = 1.0 - _finite_fraction(pupil)
    pupil_slope_per_s = linear_slope(t, pupil) if np.any(np.isfinite(pupil)) else float("nan")

    finite_pupil = pupil[np.isfinite(pupil)]
    pupil_range = float(np.max(finite_pupil) - np.min(finite_pupil)) if finite_pupil.size > 1 else float("nan")

    # Pupil velocity (absolute first-difference / dt)
    if len(pupil) > 1:
        dpupil = np.abs(np.diff(pupil))
        dt_arr = np.diff(t)
        dt_safe = np.where(dt_arr > 0, dt_arr, np.nan)
        pv = dpupil / dt_safe
        pv_finite = pv[np.isfinite(pv)]
        pupil_velocity_mean = float(np.mean(pv_finite)) if pv_finite.size > 0 else float("nan")
    else:
        pupil_velocity_mean = float("nan")

    pupil_left_mean = _safe_nanmean(left)
    pupil_right_mean = _safe_nanmean(right)
    pupil_left_std = _safe_nanstd(left)
    pupil_right_std = _safe_nanstd(right)
    pupil_left_missing_frac = 1.0 - _finite_fraction(left)
    pupil_right_missing_frac = 1.0 - _finite_fraction(right)

    # Per-sample absolute L-R difference on rows where both are finite
    if left.size > 0 and right.size > 0:
        n = min(len(left), len(right))
        lr_diff = np.abs(left[:n] - right[:n])
        lr_diff[~(np.isfinite(left[:n]) & np.isfinite(right[:n]))] = np.nan
        pupil_lr_diff_mean = _safe_nanmean(lr_diff)
    else:
        pupil_lr_diff_mean = float("nan")

    # ── Blink detection ───────────────────────────────────────────────────
    blink_count, blink_duration_mean_s = _estimate_blinks(gaze_valid, t)
    blink_rate_per_min = (
        blink_count / (duration_s / 60.0)
        if np.isfinite(duration_s) and duration_s > 0
        else float("nan")
    )

    # ── QC flags ──────────────────────────────────────────────────────────
    flags: list[str] = []

    if not len(t):
        flags.append("missing_et")
    if np.isfinite(duration_s) and duration_s < min_duration_s:
        flags.append("short_duration")
    if fs is None or not np.isfinite(fs):
        flags.append("sample_rate_unavailable")
    elif not (sample_rate_min_hz <= fs <= sample_rate_max_hz):
        flags.append("sample_rate_unusual")
    if gaze_valid_frac < min_gaze_valid_frac:
        flags.append("gaze_low_validity")
    if pupil_missing_frac > (1.0 - min_pupil_coverage_frac):
        flags.append("pupil_low_coverage")
    if pupil_left_missing_frac > (1.0 - min_pupil_coverage_frac):
        flags.append("pupil_left_low_coverage")
    if pupil_right_missing_frac > (1.0 - min_pupil_coverage_frac):
        flags.append("pupil_right_low_coverage")
    if np.isfinite(pupil_mean) and not (pupil_plausible_min_mm <= pupil_mean <= pupil_plausible_max_mm):
        flags.append("pupil_implausible")
    if np.isfinite(pupil_lr_diff_mean) and pupil_lr_diff_mean > pupil_asymmetry_threshold_mm:
        flags.append("pupil_asymmetry_large")
    if np.isfinite(blink_rate_per_min) and not (blink_rate_min_per_min <= blink_rate_per_min <= blink_rate_max_per_min):
        flags.append("blink_rate_unusual")
    if np.isfinite(gaze_out_of_bounds_frac) and gaze_out_of_bounds_frac > gaze_oob_threshold:
        flags.append("gaze_out_of_bounds")
    if np.isfinite(pupil_velocity_mean) and pupil_velocity_mean > pupil_velocity_spike_threshold:
        flags.append("pupil_velocity_spike")

    return {
        "et_available": True,
        "sample_rate_hz": fs if fs is not None else float("nan"),
        "duration_s": duration_s,
        "gaze_valid_frac": gaze_valid_frac,
        "gaze_out_of_bounds_frac": gaze_out_of_bounds_frac,
        "gaze_x_mean": gaze_x_mean,
        "gaze_y_mean": gaze_y_mean,
        "gaze_x_std": gaze_x_std,
        "gaze_y_std": gaze_y_std,
        "gaze_dispersion": gaze_dispersion,
        "gaze_velocity_mean": gaze_velocity_mean,
        "gaze_velocity_p95": gaze_velocity_p95,
        "pupil_mean": pupil_mean,
        "pupil_std": pupil_std,
        "pupil_slope_per_s": pupil_slope_per_s,
        "pupil_missing_frac": pupil_missing_frac,
        "pupil_range": pupil_range,
        "pupil_velocity_mean": pupil_velocity_mean,
        "pupil_left_mean": pupil_left_mean,
        "pupil_right_mean": pupil_right_mean,
        "pupil_left_std": pupil_left_std,
        "pupil_right_std": pupil_right_std,
        "pupil_left_missing_frac": pupil_left_missing_frac,
        "pupil_right_missing_frac": pupil_right_missing_frac,
        "pupil_lr_diff_mean": pupil_lr_diff_mean,
        "blink_count_est": blink_count,
        "blink_rate_per_min": blink_rate_per_min,
        "blink_duration_mean_s": blink_duration_mean_s,
        "qc_flag": _qc_flag_text(flags),
        "qc_notes": _qc_notes_text(flags),
    }


# ── T0 baseline normalisation ─────────────────────────────────────────────────


def add_t0_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-participant T0 baseline delta and z-score columns."""
    if df.empty:
        return df
    out = df.copy()
    task_col = "task_id" if "task_id" in out.columns else "task"
    for feature in BASELINE_FEATURES:
        out[f"{feature}_delta_t0"] = np.nan
        out[f"{feature}_z_t0"] = np.nan

    for (_session_id, _participant_id), group in out.groupby(["session_id", "participant_id"]):
        baseline = group[group[task_col] == "T0"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for feature, sd_feature in BASELINE_FEATURES.items():
            if feature not in out.columns:
                continue
            base_value = pd.to_numeric(pd.Series([base.get(feature)]), errors="coerce").iloc[0]
            values = pd.to_numeric(out.loc[group.index, feature], errors="coerce")
            out.loc[group.index, f"{feature}_delta_t0"] = values - base_value
            if sd_feature and sd_feature in out.columns:
                denom = pd.to_numeric(pd.Series([base.get(sd_feature)]), errors="coerce").iloc[0]
                if np.isfinite(denom) and abs(float(denom)) > 1e-9:
                    out.loc[group.index, f"{feature}_z_t0"] = (values - base_value) / float(denom)
    return out


# ── QC summary table ──────────────────────────────────────────────────────────


def build_qc_summary(
    task_df: pd.DataFrame,
    expected_sessions: list[str] | None = None,
    expected_tasks: list[str] | None = None,
    expected_participants: list[str] | None = None,
) -> pd.DataFrame:
    """Build one QC row per participant-task for paper tables."""
    task_col = "task_id" if "task_id" in task_df.columns else "task"
    columns = [
        "session_id",
        "participant_id",
        task_col,
        "et_available",
        "gaze_usable",
        "pupil_usable",
        "duration_s",
        "gaze_valid_frac",
        "pupil_missing_frac",
        "qc_flag",
        "qc_notes",
    ]
    if task_df.empty:
        return pd.DataFrame(columns=columns)

    gaze_usable = (
        (task_df["gaze_valid_frac"] >= 0.70)
        & ~task_df["qc_flag"].str.contains("gaze_out_of_bounds", na=False)
    )
    pupil_usable = (
        (task_df["pupil_missing_frac"] <= 0.30)
        & ~task_df["qc_flag"].str.contains("pupil_implausible", na=False)
    )

    out = pd.DataFrame(
        {
            "session_id": task_df["session_id"],
            "participant_id": task_df["participant_id"],
            task_col: task_df[task_col],
            "et_available": task_df["et_available"],
            "gaze_usable": gaze_usable,
            "pupil_usable": pupil_usable,
            "duration_s": task_df["duration_s"],
            "gaze_valid_frac": task_df["gaze_valid_frac"],
            "pupil_missing_frac": task_df["pupil_missing_frac"],
            "qc_flag": task_df["qc_flag"],
            "qc_notes": task_df["qc_notes"],
        }
    )

    if not expected_sessions:
        return out[columns].sort_values(["session_id", task_col, "participant_id"])

    expected_tasks = expected_tasks or ["T0", "T1", "T2", "T3", "T4"]
    expected_participants = expected_participants or ["P1", "P2", "P3", "P4"]
    grid = pd.DataFrame(
        [
            {"session_id": session, "participant_id": participant, task_col: task}
            for session in expected_sessions
            for task in expected_tasks
            for participant in expected_participants
        ]
    )
    merged = grid.merge(out, on=["session_id", "participant_id", task_col], how="left")
    merged["et_available"] = merged["et_available"].fillna(False).astype(bool)
    for col in ["gaze_usable", "pupil_usable"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["qc_flag"] = merged["qc_flag"].fillna("missing_et")
    merged["qc_notes"] = merged["qc_notes"].fillna(FLAG_DESCRIPTIONS["missing_et"])
    return merged[columns].sort_values(["session_id", task_col, "participant_id"])


# ── Feature definitions output ────────────────────────────────────────────────


def write_feature_definitions(out_dir: Path) -> Path:
    path = out_dir / "et_feature_definitions.tsv"
    pd.DataFrame(FEATURE_DEFINITIONS, columns=["feature", "definition"]).to_csv(
        path, sep="\t", index=False
    )
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Tobii eye-tracking features with QC from task-split TSV files."
    )
    add_common_io_args(p)
    p.add_argument("--window-s", type=float, default=30.0, help="Rolling window length (seconds).")
    p.add_argument("--step-s", type=float, default=15.0, help="Rolling window step (seconds).")
    p.add_argument("--gaze-x-idx", type=int, default=0, help="Gaze X channel index in value_* columns.")
    p.add_argument("--gaze-y-idx", type=int, default=1, help="Gaze Y channel index in value_* columns.")
    p.add_argument("--pupil-left-idx", type=int, default=2, help="Left pupil index in value_* columns.")
    p.add_argument("--pupil-right-idx", type=int, default=3, help="Right pupil index in value_* columns.")
    p.add_argument("--gaze-valid-idx", type=int, default=4, help="Gaze-valid flag index in value_* columns.")
    p.add_argument(
        "--min-duration-s", type=float, default=30.0,
        help="Minimum task duration (s) to flag short_duration (default: 30).",
    )
    p.add_argument(
        "--min-gaze-valid-frac", type=float, default=0.70,
        help="Minimum gaze validity fraction below which gaze_low_validity is raised (default: 0.70).",
    )
    p.add_argument(
        "--min-pupil-coverage-frac", type=float, default=0.70,
        help="Minimum pupil coverage fraction below which pupil_low_coverage is raised (default: 0.70).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    session_dirs = discover_session_dirs(args.data_root, args.sessions)
    LOG.info("Found %d session directories.", len(session_dirs))

    task_rows: list[dict] = []
    window_rows: list[dict] = []

    for session_dir in session_dirs:
        et_dir = session_dir / "et"
        if not et_dir.exists():
            continue
        files = sorted(et_dir.glob("*_acq-P*_tobii.tsv.gz"))
        for path in files:
            task = parse_task_from_name(path.name)
            participant = parse_participant_from_name(path.name)
            if task is None or participant is None:
                continue
            df = read_tsv(path)
            numeric_columns(df)
            if "lsl_time" not in df.columns:
                LOG.warning("No lsl_time column in %s — skipping.", path.name)
                continue
            df = df.dropna(subset=["lsl_time"]).sort_values("lsl_time").reset_index(drop=True)

            meta = {
                "session_id": parse_session_from_path(path),
                "task_id": task,
                "participant_id": participant,
                "source_file": str(path),
            }

            feat = compute_et_features(
                df,
                gaze_x_idx=args.gaze_x_idx,
                gaze_y_idx=args.gaze_y_idx,
                pupil_left_idx=args.pupil_left_idx,
                pupil_right_idx=args.pupil_right_idx,
                gaze_valid_idx=args.gaze_valid_idx,
                min_duration_s=args.min_duration_s,
                min_gaze_valid_frac=args.min_gaze_valid_frac,
                min_pupil_coverage_frac=args.min_pupil_coverage_frac,
            )
            task_rows.append({**meta, **feat})

            for w in rolling_windows(df, args.window_s, args.step_s):
                w_meta = {
                    **meta,
                    "window_index": int(w["window_index"].iloc[0]),
                    "window_start_lsl": float(w["window_start_lsl"].iloc[0]),
                    "window_end_lsl": float(w["window_end_lsl"].iloc[0]),
                }
                w_feat = compute_et_features(
                    w,
                    gaze_x_idx=args.gaze_x_idx,
                    gaze_y_idx=args.gaze_y_idx,
                    pupil_left_idx=args.pupil_left_idx,
                    pupil_right_idx=args.pupil_right_idx,
                    gaze_valid_idx=args.gaze_valid_idx,
                    min_duration_s=0.0,  # windows are short by design; don't re-flag duration
                    min_gaze_valid_frac=args.min_gaze_valid_frac,
                    min_pupil_coverage_frac=args.min_pupil_coverage_frac,
                )
                window_rows.append({**w_meta, **w_feat})

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_df = pd.DataFrame(task_rows)
    window_df = pd.DataFrame(window_rows)

    if not task_df.empty:
        task_df = add_t0_baselines(task_df)
        task_df = task_df.sort_values(["session_id", "task_id", "participant_id"])

    if not window_df.empty:
        window_df = window_df.sort_values(
            ["session_id", "task_id", "participant_id", "window_index"]
        )

    task_path = out_dir / "et_participant_task.tsv"
    window_path = out_dir / "et_window_30s.tsv"
    task_df.to_csv(task_path, sep="\t", index=False)
    window_df.to_csv(window_path, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", task_path, len(task_rows))
    LOG.info("Wrote %s (%d rows)", window_path, len(window_rows))

    if not task_df.empty:
        qc_df = build_qc_summary(task_df)
        qc_path = out_dir / "et_qc_summary.tsv"
        qc_df.to_csv(qc_path, sep="\t", index=False)
        LOG.info("Wrote %s (%d rows)", qc_path, len(qc_df))

    defs_path = write_feature_definitions(out_dir)
    LOG.info("Wrote %s", defs_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
