"""Extract paper-ready participant-level and rolling-window EmotiBit features.

The tool expects task-split BIDS-like files under ``physio/`` named with
``_task-T*_run-01_acq-P*_emotibit.tsv.gz``. Outputs use anonymised P1--P4
participant IDs only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, find_peaks, sosfiltfilt
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments.
    butter = None
    find_peaks = None
    sosfiltfilt = None

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

LOG = logging.getLogger("extract_physio_features")

FLAG_DESCRIPTIONS: dict[str, str] = {
    "missing_physio": "No EmotiBit file found for this participant-task.",
    "short_duration": "Data duration is below the minimum threshold.",
    "sample_rate_unavailable": "Could not determine the signal sample rate.",
    "sample_rate_unusual": "Sample rate is outside the expected EmotiBit device range.",
    "ppg_low_coverage": "PPG channel has insufficient usable samples (< 80%).",
    "eda_low_coverage": "EDA channel has insufficient usable samples (< 80%).",
    "temp_low_coverage": "Temperature channel has insufficient usable samples (< 80%).",
    "accel_low_coverage": "Accelerometer channels have insufficient usable samples (< 80%).",
    "gyro_low_coverage": "Gyroscope channels have insufficient usable samples (< 80%).",
    "mag_low_coverage": "Magnetometer channels have insufficient usable samples (< 80%).",
    "hr_low_coverage": "Device HR channel has insufficient usable samples (< 50%).",
    "hr_implausible": "Mean HR is outside the plausible adult range (35–220 bpm).",
    "ppg_hr_mismatch": "PPG-derived HR disagrees with device HR by more than 20 bpm.",
    "hrv_unreliable": "Beat-detection quality is insufficient for HRV computation.",
    "temp_implausible": "Mean skin temperature is outside the plausible range (20–45 °C).",
    "temp_aux_implausible": "Auxiliary temperature is outside the plausible range (20–45 °C).",
    "motion_contaminated": "High-acceleration fraction exceeds 20%, indicating likely motion artefacts.",
    "eda_phasic_detection_limited": (
        "EDA phasic peak detection is limited: scipy not available or EDA coverage is low; "
        "eda_phasic_rate_hz should be treated as unreliable."
    ),
    "channel_map_unconfirmed": (
        "EmotiBit value_* channel mapping has not been explicitly confirmed for this dataset; "
        "verify PPG/EDA/temperature/IMU column indices before interpreting features."
    ),
}

BASELINE_FEATURES = {
    "ppg_green_mean": "ppg_green_std",
    "ppg_red_mean": "ppg_red_std",
    "ppg_ir_mean": "ppg_ir_std",
    "hr_mean_bpm": "hr_sd_bpm",
    "hrv_rmssd_ms": None,
    "eda_tonic_mean": "eda_std",
    "eda_phasic_rate_hz": None,
    "eda_scr_amplitude_mean": None,
    "temp_mean": "temp_std",
    "thermopile_mean": "thermopile_std",
    "temp_aux_mean": "temp_aux_std",
    "accel_motion_mean": "accel_motion_std",
    "accel_dynamic_mean": None,
    "gyro_motion_mean": "gyro_motion_std",
    "mag_motion_mean": "mag_motion_std",
}

FEATURE_DEFINITIONS = [
    ("physio_available", "Whether a participant-task EmotiBit file was found and read."),
    ("duration_s", "Task/window duration covered by the physio samples."),
    ("coverage_pct", "Minimum finite-sample coverage across PPG, EDA, and temperature."),
    ("hr_mean_bpm", "Mean heart rate in beats per minute, preferring device HR when present."),
    ("hr_sd_bpm", "Standard deviation of device heart rate, when present."),
    ("hrv_rmssd_ms", "Quality-gated PPG-derived RMSSD; blank when beat detection is unreliable."),
    ("hrv_quality_score", "0-1 PPG beat-quality score combining clean intervals and HR agreement."),
    ("ppg_hr_agreement_bpm", "Absolute difference between PPG-derived and device-derived HR."),
    ("ppg_channel_idx", "Selected PPG value_* channel used for PPG-derived HR/HRV diagnostics."),
    ("hrv_rmssd_ms_delta_t0", "Task-level RMSSD change relative to same participant's T0 baseline."),
    ("ppg_green_mean", "Mean green PPG channel value."),
    ("ppg_red_mean", "Mean red PPG channel value."),
    ("ppg_ir_mean", "Mean infrared PPG channel value."),
    ("eda_tonic_mean", "Mean electrodermal activity level from the configured EDA channel."),
    ("eda_tonic_slope", "Linear EDA slope per second within the task/window."),
    ("eda_phasic_rate_hz", "Quality-gated phasic EDA peak count divided by duration."),
    ("eda_scr_amplitude_mean", "Mean amplitude of detected phasic EDA responses."),
    ("temp_mean", "Mean skin temperature from the configured temperature channel."),
    ("temp_slope", "Linear temperature slope per second within the task/window."),
    ("thermopile_mean", "Mean thermopile/infrared temperature-like channel when available."),
    ("temp_aux_mean", "Mean auxiliary temperature-like channel when available."),
    ("accel_motion_mean", "Mean accelerometer vector magnitude from configured IMU channels."),
    ("accel_dynamic_mean", "Mean absolute acceleration deviation from median posture."),
    ("accel_jerk_mean", "Mean absolute first difference of acceleration magnitude per second."),
    ("gyro_motion_mean", "Mean gyroscope vector magnitude."),
    ("mag_motion_mean", "Mean magnetometer vector magnitude."),
    ("qc_flag", "Semicolon-separated machine-readable quality flags; ok means no rule fired."),
    ("qc_notes", "Human-readable plain-language description of each quality flag."),
    ("eda_phasic_detection_limited", "EDA phasic peak detection is limited; eda_phasic_rate_hz may be unreliable."),
    ("channel_map_unconfirmed", "EmotiBit value_* channel mapping has not been confirmed; pass --channel-map-confirmed once verified."),
]

PPG_LABELS = {0: "green", 1: "red", 2: "ir"}


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


def _safe_nanpercentile(x: np.ndarray, percentile: float) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.nanpercentile(x[mask], percentile))


def _finite_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(x)))


def _clean_signal(
    x: np.ndarray,
    low: float | None = None,
    high: float | None = None,
    sentinel_abs: float | None = None,
) -> np.ndarray:
    out = x.astype(float, copy=True)
    mask = np.isfinite(out)
    if low is not None:
        mask &= out >= low
    if high is not None:
        mask &= out <= high
    if sentinel_abs is not None:
        mask &= np.abs(out) < sentinel_abs
    out[~mask] = np.nan
    return out


def _peak_count(signal: np.ndarray, min_distance: int = 8) -> int:
    if len(signal) < 3:
        return 0
    x = signal.copy()
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return 0
    x = x[mask]
    thr = np.nanmean(x) + 0.5 * np.nanstd(x)
    peaks = []
    last = -10_000
    for i in range(1, len(x) - 1):
        if x[i] > thr and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            if i - last >= min_distance:
                peaks.append(i)
                last = i
    return len(peaks)


def _lowpass_signal(x: np.ndarray, fs: float | None, cutoff_hz: float) -> np.ndarray:
    x = _interpolate_finite(x)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return np.array([], dtype=float)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return x
    cutoff_hz = min(cutoff_hz, fs * 0.45)
    if cutoff_hz <= 0:
        return x
    if butter is not None and sosfiltfilt is not None and len(x) > int(fs * 6):
        sos = butter(2, cutoff_hz, btype="lowpass", fs=fs, output="sos")
        return sosfiltfilt(sos, x)
    window = max(3, int(fs / max(cutoff_hz, 1e-6)))
    if window % 2 == 0:
        window += 1
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _basic_signal_features(prefix: str, t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_available": bool(_finite_fraction(x) > 0.0),
        f"{prefix}_coverage_pct": _finite_fraction(x) * 100.0,
        f"{prefix}_mean": _safe_nanmean(x),
        f"{prefix}_median": _safe_nanpercentile(x, 50.0),
        f"{prefix}_std": _safe_nanstd(x),
        f"{prefix}_min": _safe_nanpercentile(x, 0.0),
        f"{prefix}_p95": _safe_nanpercentile(x, 95.0),
        f"{prefix}_max": _safe_nanpercentile(x, 100.0),
        f"{prefix}_slope": linear_slope(t, x) if len(x) else float("nan"),
    }


def _eda_deep_features(t: np.ndarray, eda: np.ndarray, fs: float | None) -> dict[str, float]:
    base = _basic_signal_features("eda", t, eda)
    if eda.size == 0 or _finite_fraction(eda) == 0.0:
        return {
            **base,
            "eda_tonic_mean": float("nan"),
            "eda_tonic_median": float("nan"),
            "eda_tonic_std": float("nan"),
            "eda_tonic_slope": float("nan"),
            "eda_phasic_mean": float("nan"),
            "eda_phasic_std": float("nan"),
            "eda_scr_count": 0.0,
            "eda_scr_rate_hz": float("nan"),
            "eda_scr_amplitude_mean": float("nan"),
            "eda_scr_amplitude_p95": float("nan"),
            "eda_scr_peak_rate_per_min": float("nan"),
        }

    eda_interp = _interpolate_finite(eda)
    tonic = _lowpass_signal(eda_interp, fs, cutoff_hz=0.05)
    phasic = eda_interp - tonic if tonic.size else np.array([], dtype=float)
    duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) else float("nan")
    if phasic.size and find_peaks is not None and fs is not None and np.isfinite(fs) and fs > 0:
        prominence = max(float(np.nanstd(phasic) * 0.5), 0.002)
        peaks, props = find_peaks(phasic, distance=max(1, int(fs * 0.8)), prominence=prominence)
        amplitudes = props.get("prominences", np.array([], dtype=float))
    else:
        peaks = np.array([], dtype=int)
        amplitudes = np.array([], dtype=float)
    scr_count = float(len(peaks))
    scr_rate = float(scr_count / duration_s) if np.isfinite(duration_s) and duration_s > 0 else float("nan")
    return {
        **base,
        "eda_tonic_mean": _safe_nanmean(tonic),
        "eda_tonic_median": _safe_nanpercentile(tonic, 50.0),
        "eda_tonic_std": _safe_nanstd(tonic),
        "eda_tonic_slope": linear_slope(t, tonic) if len(tonic) else float("nan"),
        "eda_phasic_mean": _safe_nanmean(phasic),
        "eda_phasic_std": _safe_nanstd(phasic),
        "eda_scr_count": scr_count,
        "eda_scr_rate_hz": scr_rate,
        "eda_scr_amplitude_mean": _safe_nanmean(amplitudes),
        "eda_scr_amplitude_p95": _safe_nanpercentile(amplitudes, 95.0),
        "eda_scr_peak_rate_per_min": scr_rate * 60.0 if np.isfinite(scr_rate) else float("nan"),
    }


def _vector_features(
    prefix: str,
    matrix: np.ndarray,
    fs: float | None,
    high_threshold: float | None = None,
) -> dict[str, float | bool]:
    if matrix.size == 0:
        return {
            f"{prefix}_available": False,
            f"{prefix}_coverage_pct": 0.0,
            f"{prefix}_motion_mean": float("nan"),
            f"{prefix}_motion_std": float("nan"),
            f"{prefix}_motion_p95": float("nan"),
            f"{prefix}_motion_max": float("nan"),
            f"{prefix}_dynamic_mean": float("nan"),
            f"{prefix}_dynamic_p95": float("nan"),
            f"{prefix}_jerk_mean": float("nan"),
            f"{prefix}_jerk_p95": float("nan"),
            f"{prefix}_high_fraction": float("nan"),
        }
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    mag = np.sqrt(np.nansum(matrix**2, axis=1))
    dynamic = np.abs(mag - _safe_nanpercentile(mag, 50.0))
    if len(mag) > 1:
        jerk = np.abs(np.diff(mag)) * (fs if fs is not None and np.isfinite(fs) else 1.0)
    else:
        jerk = np.array([], dtype=float)
    high_fraction = (
        float(np.mean(mag[np.isfinite(mag)] > high_threshold))
        if high_threshold is not None and np.any(np.isfinite(mag))
        else float("nan")
    )
    return {
        f"{prefix}_available": bool(np.any(finite_rows)),
        f"{prefix}_coverage_pct": float(np.mean(finite_rows) * 100.0),
        f"{prefix}_motion_mean": _safe_nanmean(mag),
        f"{prefix}_motion_std": _safe_nanstd(mag),
        f"{prefix}_motion_p95": _safe_nanpercentile(mag, 95.0),
        f"{prefix}_motion_max": _safe_nanpercentile(mag, 100.0),
        f"{prefix}_dynamic_mean": _safe_nanmean(dynamic),
        f"{prefix}_dynamic_p95": _safe_nanpercentile(dynamic, 95.0),
        f"{prefix}_jerk_mean": _safe_nanmean(jerk),
        f"{prefix}_jerk_p95": _safe_nanpercentile(jerk, 95.0),
        f"{prefix}_high_fraction": high_fraction,
    }


def _ppg_bpm(ppg: np.ndarray, fs: float | None) -> float:
    if fs is None or not np.isfinite(fs) or fs <= 0 or len(ppg) < 64:
        return float("nan")
    x = ppg.copy()
    if np.isfinite(x).sum() < 64:
        return float("nan")
    x = np.nan_to_num(x - _safe_nanmean(x), nan=0.0)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    band = (freqs >= 0.6) & (freqs <= 3.5)
    if not np.any(band):
        return float("nan")
    f = freqs[band][int(np.argmax(mag[band]))]
    return float(f * 60.0)


def _hrv_from_ppg(ppg: np.ndarray, fs: float | None) -> tuple[float, float, float]:
    if fs is None or fs <= 0 or len(ppg) < 128:
        return (float("nan"), float("nan"), float("nan"))
    x = ppg.copy()
    if np.isfinite(x).sum() < 128:
        return (float("nan"), float("nan"), float("nan"))
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        return (float("nan"), float("nan"), float("nan"))
    thr = med + 0.8 * mad
    min_distance = int(max(1, fs * 0.4))
    peaks = []
    last = -10_000
    for i in range(1, len(x) - 1):
        if x[i] > thr and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            if i - last >= min_distance:
                peaks.append(i)
                last = i
    if len(peaks) < 3:
        return (float("nan"), float("nan"), float("nan"))
    ibi_ms = np.diff(np.array(peaks) / fs) * 1000.0
    if len(ibi_ms) < 2:
        return (float("nan"), float("nan"), float("nan"))
    diff_ms = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(diff_ms**2))) if len(diff_ms) else float("nan")
    sdnn = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else float("nan")
    pnn50 = float(np.mean(np.abs(diff_ms) > 50.0) * 100.0) if len(diff_ms) else float("nan")
    return rmssd, sdnn, pnn50


def _interpolate_finite(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.full_like(x, np.nan, dtype=float)
    if np.all(mask):
        return x.astype(float, copy=True)
    idx = np.arange(len(x))
    return np.interp(idx, idx[mask], x[mask]).astype(float, copy=False)


def _bandpass_ppg(x: np.ndarray, fs: float) -> np.ndarray:
    x = _interpolate_finite(x)
    if x.size < int(fs * 10) or not np.any(np.isfinite(x)):
        return np.array([], dtype=float)
    x = x - np.nanmedian(x)
    if butter is not None and sosfiltfilt is not None:
        high = min(3.0, fs * 0.45)
        if high <= 0.7:
            return np.array([], dtype=float)
        sos = butter(3, [0.7, high], btype="bandpass", fs=fs, output="sos")
        y = sosfiltfilt(sos, x)
    else:
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        fft = np.fft.rfft(x)
        fft[(freqs < 0.7) | (freqs > 3.0)] = 0.0
        y = np.fft.irfft(fft, n=len(x))
    scale = np.nanstd(y)
    if not np.isfinite(scale) or scale <= 0:
        return np.array([], dtype=float)
    return (y - np.nanmean(y)) / scale


def _detect_ppg_peaks(filtered: np.ndarray, fs: float, expected_hr_bpm: float) -> np.ndarray:
    if filtered.size < int(fs * 10):
        return np.array([], dtype=int)
    if np.isfinite(expected_hr_bpm) and expected_hr_bpm > 0:
        expected_ibi_s = 60.0 / expected_hr_bpm
        min_distance_s = float(np.clip(expected_ibi_s * 0.65, 0.35, 0.85))
    else:
        min_distance_s = 0.45
    distance = max(1, int(min_distance_s * fs))
    if find_peaks is not None:
        peaks, _ = find_peaks(filtered, distance=distance, prominence=0.35)
        return peaks.astype(int, copy=False)
    candidates = []
    last = -10_000
    threshold = float(np.nanmedian(filtered) + 0.25 * np.nanstd(filtered))
    for i in range(1, len(filtered) - 1):
        if filtered[i] > threshold and filtered[i] >= filtered[i - 1] and filtered[i] > filtered[i + 1]:
            if i - last >= distance:
                candidates.append(i)
                last = i
    return np.array(candidates, dtype=int)


def _ppg_candidate_features(
    ppg: np.ndarray,
    fs: float | None,
    expected_hr_bpm: float,
) -> dict[str, float]:
    empty = {
        "ppg_rate_proxy_bpm": float("nan"),
        "hrv_rmssd_ms": float("nan"),
        "hrv_sdnn_ms": float("nan"),
        "hrv_pnn50_pct": float("nan"),
        "ppg_peak_count": 0.0,
        "ppg_clean_ibi_count": 0.0,
        "ppg_clean_ibi_fraction": 0.0,
        "ppg_hr_agreement_bpm": float("nan"),
        "hrv_quality_score": 0.0,
    }
    if fs is None or not np.isfinite(fs) or fs <= 0 or ppg.size < int((fs or 1.0) * 10):
        return empty

    best = empty
    for polarity in (1.0, -1.0):
        filtered = _bandpass_ppg(ppg * polarity, fs)
        peaks = _detect_ppg_peaks(filtered, fs, expected_hr_bpm)
        if len(peaks) < 3:
            continue
        ibi_ms = np.diff(peaks) / fs * 1000.0
        plausible = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 2000.0)]
        if len(plausible) < 2:
            continue
        median_ibi = float(np.nanmedian(plausible))
        max_deviation = max(250.0, 0.25 * median_ibi)
        clean = plausible[np.abs(plausible - median_ibi) <= max_deviation]
        if len(clean) < 2:
            continue
        clean_fraction = float(len(clean) / max(len(ibi_ms), 1))
        bpm = float(60000.0 / np.nanmedian(clean))
        diff_ms = np.diff(clean)
        rmssd = float(np.sqrt(np.nanmean(diff_ms**2))) if len(diff_ms) else float("nan")
        sdnn = float(np.nanstd(clean, ddof=1)) if len(clean) > 1 else float("nan")
        pnn50 = float(np.nanmean(np.abs(diff_ms) > 50.0) * 100.0) if len(diff_ms) else float("nan")
        agreement = (
            float(abs(bpm - expected_hr_bpm)) if np.isfinite(expected_hr_bpm) else float("nan")
        )
        agreement_score = (
            float(np.clip(1.0 - agreement / 30.0, 0.0, 1.0))
            if np.isfinite(agreement)
            else 0.5
        )
        count_score = float(np.clip(len(clean) / 30.0, 0.0, 1.0))
        regularity_score = float(np.clip(1.0 - (rmssd / 300.0), 0.0, 1.0)) if np.isfinite(rmssd) else 0.0
        quality = 0.4 * clean_fraction + 0.3 * agreement_score + 0.2 * regularity_score + 0.1 * count_score
        candidate = {
            "ppg_rate_proxy_bpm": bpm,
            "hrv_rmssd_ms": rmssd,
            "hrv_sdnn_ms": sdnn,
            "hrv_pnn50_pct": pnn50,
            "ppg_peak_count": float(len(peaks)),
            "ppg_clean_ibi_count": float(len(clean)),
            "ppg_clean_ibi_fraction": clean_fraction,
            "ppg_hr_agreement_bpm": agreement,
            "hrv_quality_score": quality,
        }
        if quality > best["hrv_quality_score"]:
            best = candidate
    return best


def _select_ppg_features(
    df: pd.DataFrame,
    indices: tuple[int, ...],
    fs: float | None,
    expected_hr_bpm: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for idx in indices:
        candidate = _ppg_candidate_features(
            _clean_signal(_value(df, idx), low=0.0, high=300_000.0, sentinel_abs=999_999.0),
            fs,
            expected_hr_bpm,
        )
        candidate["ppg_channel_idx"] = float(idx)
        if best is None or candidate["hrv_quality_score"] > best["hrv_quality_score"]:
            best = candidate
    if best is None:
        best = _ppg_candidate_features(np.array([], dtype=float), fs, expected_hr_bpm)
        best["ppg_channel_idx"] = float("nan")
    return best


def _value(df: pd.DataFrame, idx: int) -> np.ndarray:
    col = f"value_{idx}"
    if col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float, copy=False)


def _value_matrix(df: pd.DataFrame, indices: tuple[int, ...]) -> np.ndarray:
    arrays = [_value(df, idx) for idx in indices]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return np.empty((0, 0), dtype=float)
    n = min(len(a) for a in arrays)
    return np.vstack([a[:n] for a in arrays]).T


def _clean_matrix(
    matrix: np.ndarray,
    low: float | None = None,
    high: float | None = None,
    sentinel_abs: float | None = None,
) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    out = matrix.astype(float, copy=True)
    mask = np.isfinite(out)
    if low is not None:
        mask &= out >= low
    if high is not None:
        mask &= out <= high
    if sentinel_abs is not None:
        mask &= np.abs(out) < sentinel_abs
    out[~mask] = np.nan
    return out


def _range_fraction(x: np.ndarray, low: float, high: float) -> float:
    if x.size == 0:
        return 0.0
    mask = np.isfinite(x)
    if not np.any(mask):
        return 0.0
    xv = x[mask]
    return float(np.mean((xv >= low) & (xv <= high)))


def _qc_flag_text(flags: list[str]) -> str:
    return "ok" if not flags else ";".join(dict.fromkeys(flags))


def _qc_notes_text(flags: list[str]) -> str:
    if not flags:
        return "All quality checks passed."
    return " ".join(FLAG_DESCRIPTIONS.get(f, f) for f in dict.fromkeys(flags))


def compute_physio_features(
    df: pd.DataFrame,
    ppg_idx: int,
    eda_idx: int,
    temp_idx: int,
    hr_idx: int | None = 6,
    ppg_candidate_indices: tuple[int, ...] | None = None,
    thermopile_idx: int | None = 9,
    temp_aux_idx: int | None = 11,
    accel_indices: tuple[int, int, int] = (13, 14, 15),
    gyro_indices: tuple[int, int, int] = (16, 17, 18),
    mag_indices: tuple[int, int, int] = (19, 20, 21),
    motion_threshold_g: float = 1.5,
    min_duration_s: float = 60.0,
    min_coverage_pct: float = 80.0,
    sample_rate_min_hz: float = 10.0,
    sample_rate_max_hz: float = 60.0,
    channel_map_confirmed: bool = False,
) -> dict[str, float | str | bool]:
    fs = sample_rate_hz(df)
    t = df["lsl_time"].to_numpy(dtype=float, copy=False)
    ppg = _clean_signal(_value(df, ppg_idx), low=0.0, high=300_000.0, sentinel_abs=999_999.0)
    eda = _clean_signal(_value(df, eda_idx), low=0.0, high=100.0, sentinel_abs=999.0)
    temp = _clean_signal(_value(df, temp_idx), low=0.0, high=60.0, sentinel_abs=999.0)
    hr = _value(df, hr_idx) if hr_idx is not None and hr_idx >= 0 else np.array([], dtype=float)
    thermopile = (
        _clean_signal(_value(df, thermopile_idx), low=-50.0, high=100.0, sentinel_abs=999.0)
        if thermopile_idx is not None and thermopile_idx >= 0
        else np.array([], dtype=float)
    )
    temp_aux = (
        _clean_signal(_value(df, temp_aux_idx), low=0.0, high=60.0, sentinel_abs=999.0)
        if temp_aux_idx is not None and temp_aux_idx >= 0
        else np.array([], dtype=float)
    )
    accel = _clean_matrix(_value_matrix(df, accel_indices), low=-16.0, high=16.0, sentinel_abs=999.0)
    gyro = _clean_matrix(_value_matrix(df, gyro_indices), low=-800.0, high=800.0, sentinel_abs=999.0)
    mag = _clean_matrix(_value_matrix(df, mag_indices), low=-1000.0, high=1000.0, sentinel_abs=999.0)

    ppg_coverage = _finite_fraction(ppg)
    eda_coverage = _finite_fraction(eda)
    temp_coverage = _finite_fraction(temp)
    coverage = min(ppg_coverage, eda_coverage, temp_coverage)
    duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) else float("nan")
    eda_features = _eda_deep_features(t, eda, fs)
    clean_hr = hr[(hr >= 35.0) & (hr <= 220.0) & np.isfinite(hr)]
    hr_valid_fraction = float(len(clean_hr) / len(hr)) if len(hr) else 0.0
    hr_available = len(clean_hr) > 0 and hr_valid_fraction >= 0.5
    hr_mean = _safe_nanmean(clean_hr) if hr_available else _ppg_bpm(ppg, fs)
    hr_sd = _safe_nanstd(clean_hr) if hr_available else float("nan")
    ppg_candidates = ppg_candidate_indices or (ppg_idx,)
    ppg_features = _select_ppg_features(df, tuple(ppg_candidates), fs, hr_mean)
    rmssd = ppg_features["hrv_rmssd_ms"]
    sdnn = ppg_features["hrv_sdnn_ms"]
    pnn50 = ppg_features["hrv_pnn50_pct"]
    ppg_bpm = ppg_features["ppg_rate_proxy_bpm"]
    if ppg_features["hrv_quality_score"] < 0.65 or not np.isfinite(rmssd) or rmssd > 300.0:
        rmssd = float("nan")
        sdnn = float("nan")
        pnn50 = float("nan")

    ppg_signal_features: dict[str, float | bool] = {}
    for idx in ppg_candidates:
        label = PPG_LABELS.get(idx, f"idx{idx}")
        ppg_signal_features.update(
            _basic_signal_features(
                f"ppg_{label}",
                t,
                _clean_signal(_value(df, idx), low=0.0, high=300_000.0, sentinel_abs=999_999.0),
            )
        )
    temp_features = _basic_signal_features("temp_skin", t, temp)
    thermopile_features = _basic_signal_features("thermopile", t, thermopile)
    temp_aux_features = _basic_signal_features("temp_aux", t, temp_aux)
    accel_features = _vector_features("accel", accel, fs, high_threshold=motion_threshold_g)
    gyro_features = _vector_features("gyro", gyro, fs)
    mag_features = _vector_features("mag", mag, fs)
    motion_mean = float(accel_features["accel_motion_mean"])
    motion_high_fraction = float(accel_features["accel_high_fraction"])

    flags: list[str] = []
    if not len(t):
        flags.append("missing_physio")
    if np.isfinite(duration_s) and duration_s < min_duration_s:
        flags.append("short_duration")
    if fs is None or not np.isfinite(fs):
        flags.append("sample_rate_unavailable")
    elif fs < sample_rate_min_hz or fs > sample_rate_max_hz:
        flags.append("sample_rate_unusual")
    if ppg_coverage * 100.0 < min_coverage_pct:
        flags.append("ppg_low_coverage")
    if eda_coverage * 100.0 < min_coverage_pct:
        flags.append("eda_low_coverage")
    if temp_coverage * 100.0 < min_coverage_pct:
        flags.append("temp_low_coverage")
    if accel_features["accel_coverage_pct"] < min_coverage_pct:
        flags.append("accel_low_coverage")
    if gyro_features["gyro_coverage_pct"] < min_coverage_pct:
        flags.append("gyro_low_coverage")
    if mag_features["mag_coverage_pct"] < min_coverage_pct:
        flags.append("mag_low_coverage")
    if hr.size and hr_valid_fraction < 0.5:
        flags.append("hr_low_coverage")
    if np.isfinite(hr_mean) and not 35.0 <= hr_mean <= 220.0:
        flags.append("hr_implausible")
    if np.isfinite(ppg_features["ppg_hr_agreement_bpm"]) and ppg_features["ppg_hr_agreement_bpm"] > 20.0:
        flags.append("ppg_hr_mismatch")
    if not np.isfinite(rmssd) or rmssd <= 0.0:
        flags.append("hrv_unreliable")
    temp_mean = _safe_nanmean(temp)
    if np.isfinite(temp_mean) and not 20.0 <= temp_mean <= 45.0:
        flags.append("temp_implausible")
    temp_aux_mean = _safe_nanmean(temp_aux)
    if np.isfinite(temp_aux_mean) and not 20.0 <= temp_aux_mean <= 45.0:
        flags.append("temp_aux_implausible")
    if np.isfinite(motion_high_fraction) and motion_high_fraction > 0.20:
        flags.append("motion_contaminated")
    eda_phasic_limited = find_peaks is None or fs is None or eda_coverage * 100.0 < min_coverage_pct
    if eda_phasic_limited:
        flags.append("eda_phasic_detection_limited")
    if not channel_map_confirmed:
        flags.append("channel_map_unconfirmed")

    return {
        "physio_available": bool(len(t)),
        "sample_rate_hz": fs if fs is not None else float("nan"),
        "duration_s": duration_s,
        "coverage_pct": coverage * 100.0,
        "ppg_available": bool(ppg_coverage > 0.0),
        "eda_available": bool(eda_coverage > 0.0),
        "temp_available": bool(temp_coverage > 0.0),
        "imu_available": bool(
            (accel.size and np.any(np.isfinite(accel)))
            or (gyro.size and np.any(np.isfinite(gyro)))
            or (mag.size and np.any(np.isfinite(mag)))
        ),
        "ppg_coverage_pct": ppg_coverage * 100.0,
        "eda_coverage_pct": eda_coverage * 100.0,
        "temp_coverage_pct": temp_coverage * 100.0,
        **ppg_signal_features,
        "ppg_mean": _safe_nanmean(ppg),
        "ppg_std": _safe_nanstd(ppg),
        "ppg_rate_proxy_bpm": ppg_bpm,
        "ppg_channel_idx": ppg_features["ppg_channel_idx"],
        "ppg_peak_count": ppg_features["ppg_peak_count"],
        "ppg_clean_ibi_count": ppg_features["ppg_clean_ibi_count"],
        "ppg_clean_ibi_fraction": ppg_features["ppg_clean_ibi_fraction"],
        "ppg_hr_agreement_bpm": ppg_features["ppg_hr_agreement_bpm"],
        "hrv_quality_score": ppg_features["hrv_quality_score"],
        "hr_mean_bpm": hr_mean,
        "hr_sd_bpm": hr_sd,
        "hr_valid_fraction": hr_valid_fraction,
        "hr_source": "device_hr" if hr_available else "ppg_fft_proxy",
        "hrv_rmssd_ms": rmssd,
        "hrv_sdnn_ms": sdnn,
        "hrv_pnn50_pct": pnn50,
        "ppg_rmssd_ms": rmssd,
        "ppg_sdnn_ms": sdnn,
        "ppg_pnn50_pct": pnn50,
        **eda_features,
        "eda_mean": eda_features["eda_mean"],
        "eda_std": eda_features["eda_std"],
        "eda_slope_per_s": eda_features["eda_slope"],
        "scr_count": eda_features["eda_scr_count"],
        "eda_phasic_rate_hz": eda_features["eda_scr_rate_hz"],
        "scr_rate_hz": eda_features["eda_scr_rate_hz"],
        **temp_features,
        **thermopile_features,
        **temp_aux_features,
        "temp_mean": temp_mean,
        "temp_std": _safe_nanstd(temp),
        "temp_slope": linear_slope(t, temp) if len(temp) else float("nan"),
        "temp_slope_per_s": linear_slope(t, temp) if len(temp) else float("nan"),
        **accel_features,
        **gyro_features,
        **mag_features,
        "motion_mean": motion_mean,
        "motion_high_fraction": motion_high_fraction,
        "motion_flag": bool(np.isfinite(motion_high_fraction) and motion_high_fraction > 0.20),
        "qc_flag": _qc_flag_text(flags),
        "qc_notes": _qc_notes_text(flags),
    }


def add_t0_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-participant T0 baseline deltas and simple baseline z scores."""
    if df.empty:
        return df
    out = df.copy()
    for feature in BASELINE_FEATURES:
        out[f"{feature}_delta_t0"] = np.nan
        out[f"{feature}_z_t0"] = np.nan

    for (_session_id, _participant_id), group in out.groupby(["session_id", "participant_id"]):
        baseline = group[group["task_id"] == "T0"]
        if baseline.empty:
            mask = group.index
            for feature in BASELINE_FEATURES:
                out.loc[mask, f"{feature}_delta_t0"] = np.nan
                out.loc[mask, f"{feature}_z_t0"] = np.nan
            continue
        base = baseline.iloc[0]
        for feature, sd_feature in BASELINE_FEATURES.items():
            if feature not in out.columns:
                continue
            delta_col = f"{feature}_delta_t0"
            z_col = f"{feature}_z_t0"
            base_value = pd.to_numeric(pd.Series([base.get(feature)]), errors="coerce").iloc[0]
            values = pd.to_numeric(out.loc[group.index, feature], errors="coerce")
            out.loc[group.index, delta_col] = values - base_value
            if sd_feature and sd_feature in out.columns:
                denom = pd.to_numeric(pd.Series([base.get(sd_feature)]), errors="coerce").iloc[0]
                if np.isfinite(denom) and abs(float(denom)) > 1e-9:
                    out.loc[group.index, z_col] = (values - base_value) / float(denom)
    return out


def build_qc_summary(
    task_df: pd.DataFrame,
    expected_sessions: list[str] | None = None,
    expected_tasks: list[str] | None = None,
    expected_participants: list[str] | None = None,
) -> pd.DataFrame:
    """Build one QC row per participant-task for paper tables and handoff."""
    columns = [
        "session_id",
        "participant_id",
        "task_id",
        "physio_available",
        "ppg_usable",
        "eda_usable",
        "temp_usable",
        "imu_usable",
        "duration_s",
        "coverage_pct",
        "qc_flag",
        "qc_notes",
    ]
    if task_df.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = pd.DataFrame(
            {
                "session_id": task_df["session_id"],
                "participant_id": task_df["participant_id"],
                "task_id": task_df["task_id"],
                "physio_available": task_df["physio_available"],
                "ppg_usable": (task_df["ppg_coverage_pct"] >= 80.0)
                & ~task_df["qc_flag"].str.contains("hr_implausible", na=False),
                "eda_usable": task_df["eda_coverage_pct"] >= 80.0,
                "temp_usable": (task_df["temp_coverage_pct"] >= 80.0)
                & ~task_df["qc_flag"].str.contains("temp_implausible", na=False),
                "imu_usable": task_df["imu_available"],
                "duration_s": task_df["duration_s"],
                "coverage_pct": task_df["coverage_pct"],
                "qc_flag": task_df["qc_flag"],
                "qc_notes": task_df["qc_notes"],
            }
        )
    if not expected_sessions:
        return out[columns].sort_values(["session_id", "task_id", "participant_id"])

    expected_tasks = expected_tasks or ["T0", "T1", "T2", "T3", "T4"]
    expected_participants = expected_participants or ["P1", "P2", "P3", "P4"]
    grid = pd.DataFrame(
        [
            {"session_id": session, "participant_id": participant, "task_id": task}
            for session in expected_sessions
            for task in expected_tasks
            for participant in expected_participants
        ]
    )
    merged = grid.merge(out, on=["session_id", "participant_id", "task_id"], how="left")
    merged["physio_available"] = merged["physio_available"].fillna(False).astype(bool)
    for col in ["ppg_usable", "eda_usable", "temp_usable", "imu_usable"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["qc_flag"] = merged["qc_flag"].fillna("missing_physio")
    merged["qc_notes"] = merged["qc_notes"].fillna("missing_physio")
    return merged[columns].sort_values(["session_id", "task_id", "participant_id"])


def write_feature_definitions(out_dir: Path) -> Path:
    path = out_dir / "physio_feature_definitions.tsv"
    pd.DataFrame(FEATURE_DEFINITIONS, columns=["feature", "definition"]).to_csv(
        path,
        sep="\t",
        index=False,
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract EmotiBit physio features from task-split TSV files.")
    add_common_io_args(p)
    p.add_argument("--window-s", type=float, default=30.0, help="Rolling window length (seconds).")
    p.add_argument("--step-s", type=float, default=15.0, help="Rolling window step (seconds).")
    p.add_argument("--ppg-idx", type=int, default=0, help="PPG channel index in value_* columns.")
    p.add_argument(
        "--ppg-candidate-idx",
        type=int,
        nargs="*",
        default=(0, 1, 2),
        help="Candidate PPG channels for quality-gated PPG HR/HRV diagnostics.",
    )
    p.add_argument("--eda-idx", type=int, default=3, help="EDA/GSR channel index in value_* columns.")
    p.add_argument("--temp-idx", type=int, default=10, help="Temperature channel index in value_* columns.")
    p.add_argument(
        "--thermopile-idx",
        type=int,
        default=9,
        help="Thermopile/infrared temperature-like channel index; use -1 to disable.",
    )
    p.add_argument(
        "--temp-aux-idx",
        type=int,
        default=11,
        help="Auxiliary temperature-like channel index; use -1 to disable.",
    )
    p.add_argument("--hr-idx", type=int, default=6, help="Device heart-rate channel index; use -1 to disable.")
    p.add_argument(
        "--accel-idx",
        type=int,
        nargs=3,
        default=(13, 14, 15),
        metavar=("X", "Y", "Z"),
        help="Accelerometer channel indices in value_* columns.",
    )
    p.add_argument(
        "--gyro-idx",
        type=int,
        nargs=3,
        default=(16, 17, 18),
        metavar=("X", "Y", "Z"),
        help="Gyroscope channel indices in value_* columns.",
    )
    p.add_argument(
        "--mag-idx",
        type=int,
        nargs=3,
        default=(19, 20, 21),
        metavar=("X", "Y", "Z"),
        help="Magnetometer channel indices in value_* columns.",
    )
    p.add_argument("--motion-threshold-g", type=float, default=1.5, help="Motion flag threshold.")
    p.add_argument("--min-duration-s", type=float, default=60.0, help="Short-duration QC threshold.")
    p.add_argument("--min-coverage-pct", type=float, default=80.0, help="Per-signal coverage QC threshold.")
    p.add_argument("--sample-rate-min-hz", type=float, default=10.0, help="Minimum plausible sample rate.")
    p.add_argument("--sample-rate-max-hz", type=float, default=60.0, help="Maximum plausible sample rate.")
    p.add_argument(
        "--no-legacy-aliases",
        action="store_true",
        help="Only write canonical paper filenames, not old features_physio_* aliases.",
    )
    p.add_argument(
        "--channel-map-confirmed",
        action="store_true",
        help=(
            "Assert that the EmotiBit value_* channel mapping has been verified for this dataset. "
            "Without this flag every row is flagged channel_map_unconfirmed."
        ),
    )
    p.add_argument(
        "--include-missing-qc",
        action="store_true",
        help="Add missing participant-task rows to physio_qc_summary.tsv for paper coverage.",
    )
    p.add_argument(
        "--expected-tasks",
        nargs="*",
        default=["T0", "T1", "T2", "T3", "T4"],
        help="Expected task IDs used with --include-missing-qc.",
    )
    p.add_argument(
        "--expected-participants",
        nargs="*",
        default=["P1", "P2", "P3", "P4"],
        help="Expected participant IDs used with --include-missing-qc.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def extract_physio_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_dirs = discover_session_dirs(args.data_root, args.sessions)
    LOG.info("Found %d session directories.", len(session_dirs))

    task_rows: list[dict] = []
    window_rows: list[dict] = []

    for session_dir in session_dirs:
        physio_dir = session_dir / "physio"
        if not physio_dir.exists():
            continue
        files = sorted(physio_dir.glob("*_acq-P*_emotibit.tsv.gz"))
        for path in files:
            task = parse_task_from_name(path.name)
            participant = parse_participant_from_name(path.name)
            if task is None or participant is None:
                continue
            df = read_tsv(path)
            numeric_columns(df)
            if "lsl_time" not in df.columns:
                continue
            df = df.dropna(subset=["lsl_time"]).sort_values("lsl_time").reset_index(drop=True)
            if df.empty:
                continue

            meta = {
                "session_id": parse_session_from_path(path),
                "task_id": task,
                "task": task,
                "participant_id": participant,
                "source_file": str(path),
            }
            feat = compute_physio_features(
                df,
                args.ppg_idx,
                args.eda_idx,
                args.temp_idx,
                hr_idx=args.hr_idx if args.hr_idx >= 0 else None,
                ppg_candidate_indices=tuple(args.ppg_candidate_idx),
                thermopile_idx=args.thermopile_idx if args.thermopile_idx >= 0 else None,
                temp_aux_idx=args.temp_aux_idx if args.temp_aux_idx >= 0 else None,
                accel_indices=tuple(args.accel_idx),
                gyro_indices=tuple(args.gyro_idx),
                mag_indices=tuple(args.mag_idx),
                motion_threshold_g=args.motion_threshold_g,
                min_duration_s=args.min_duration_s,
                min_coverage_pct=args.min_coverage_pct,
                sample_rate_min_hz=args.sample_rate_min_hz,
                sample_rate_max_hz=args.sample_rate_max_hz,
                channel_map_confirmed=args.channel_map_confirmed,
            )
            task_rows.append({**meta, **feat})

            for w in rolling_windows(df, args.window_s, args.step_s):
                w_meta = {
                    **meta,
                    "window_index": int(w["window_index"].iloc[0]),
                    "window_start_lsl": float(w["window_start_lsl"].iloc[0]),
                    "window_end_lsl": float(w["window_end_lsl"].iloc[0]),
                }
                w_feat = compute_physio_features(
                    w,
                    args.ppg_idx,
                    args.eda_idx,
                    args.temp_idx,
                    hr_idx=args.hr_idx if args.hr_idx >= 0 else None,
                    ppg_candidate_indices=tuple(args.ppg_candidate_idx),
                    thermopile_idx=args.thermopile_idx if args.thermopile_idx >= 0 else None,
                    temp_aux_idx=args.temp_aux_idx if args.temp_aux_idx >= 0 else None,
                    accel_indices=tuple(args.accel_idx),
                    gyro_indices=tuple(args.gyro_idx),
                    mag_indices=tuple(args.mag_idx),
                    motion_threshold_g=args.motion_threshold_g,
                    min_duration_s=args.min_duration_s,
                    min_coverage_pct=args.min_coverage_pct,
                    sample_rate_min_hz=args.sample_rate_min_hz,
                    sample_rate_max_hz=args.sample_rate_max_hz,
                    channel_map_confirmed=args.channel_map_confirmed,
                )
                window_rows.append({**w_meta, **w_feat})

    task_df = pd.DataFrame(task_rows)
    window_df = pd.DataFrame(window_rows)
    if not task_df.empty:
        task_df = add_t0_baselines(task_df)
        task_df = task_df.sort_values(["session_id", "task_id", "participant_id"])
    if not window_df.empty:
        window_df = window_df.sort_values(["session_id", "task_id", "participant_id", "window_index"])
    expected_sessions = [p.name for p in session_dirs] if args.include_missing_qc else None
    qc_df = build_qc_summary(
        task_df,
        expected_sessions=expected_sessions,
        expected_tasks=args.expected_tasks,
        expected_participants=args.expected_participants,
    )
    return task_df, window_df, qc_df


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    task_df, window_df, qc_df = extract_physio_tables(args)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / "physio_participant_task.tsv"
    window_path = out_dir / "physio_window_30s.tsv"
    qc_path = out_dir / "physio_qc_summary.tsv"
    task_df.to_csv(task_path, sep="\t", index=False)
    window_df.to_csv(window_path, sep="\t", index=False)
    qc_df.to_csv(qc_path, sep="\t", index=False)
    defs_path = write_feature_definitions(out_dir)
    LOG.info("Wrote %s (%d rows)", task_path, len(task_df))
    LOG.info("Wrote %s (%d rows)", window_path, len(window_df))
    LOG.info("Wrote %s (%d rows)", qc_path, len(qc_df))
    LOG.info("Wrote %s", defs_path)

    if not args.no_legacy_aliases:
        legacy_task_path = out_dir / "features_physio_participant_task.tsv"
        legacy_window_path = out_dir / "features_physio_window_30s.tsv"
        task_df.to_csv(legacy_task_path, sep="\t", index=False)
        window_df.to_csv(legacy_window_path, sep="\t", index=False)
        LOG.info("Wrote legacy alias %s", legacy_task_path)
        LOG.info("Wrote legacy alias %s", legacy_window_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
