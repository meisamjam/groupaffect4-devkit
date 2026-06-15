"""pickle_generation_affectai.py

Preprocessing script for the GroupAffect-4 BIDS dataset (zenodo release).
Produces dataset.pkl — one row per VAD self-report, each containing the
30-second sensor window ending at the report timestamp.

Verified against the actual zenodo file format (data/zenodo/sub-01/...).

Directory structure expected:
  <bids_root>/
    participants.tsv                      ← global participant registry (optional)
    sub-<N>/ses-<session_id>/
      annot/
        *_group_participants.tsv          ← seat → participant_id, BFI-44, demographics
        *_task-T0T1T2T3T4_task_run_windows.tsv  ← LSL time range per task
      beh/
        *_task-T0T1T2T3T4_stimuli_answers.tsv   ← all VAD responses for the session
      et/
        *_task-{T}_run-01_acq-{seat}_tobii.tsv.gz   ← Tobii per participant per task
      physio/
        *_task-{T}_run-01_acq-{seat}_emotibit.tsv.gz ← EmotiBit per participant per task

Output: dataset.pkl
  One row per VAD response, columns:
    session_id, subject_id, seat, task, vad_timestamp_lsl
    gaze_seq   (DataFrame, 400×9)
    pupil_seq  (DataFrame, 400×3)
    eda_seq    (DataFrame, 400×5)   [raw EDA, EDA_Phasic, EDA_Tonic, HR, Temp]
    ppg_seq    (DataFrame, 400×3)   [IR, Red, Green raw ADC]
    imu_seq    (DataFrame, 400×6)   [ACC xyz, GYR xyz]
    gaze_features, pupil_features, eda_features, ppg_features, imu_features  (dict)
    valence, arousal, dominance    (float, 1–9 Likert scale)
    bfi44_e/a/c/n/o                (float)
    age, sex                       (int)
    task_start_lsl, task_end_lsl   (float, for temporal context)

Usage:
  python tools/mumt/pickle_generation_affectai.py --dataset-path data/zenodo
  python tools/mumt/pickle_generation_affectai.py --dataset-path data/zenodo \
      --output data/mumt/dataset.pkl
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from neurokit2.misc import NeuroKitWarning
    warnings.filterwarnings("ignore", category=NeuroKitWarning)
except ImportError:
    pass

# Optional audio libraries — gracefully disabled if not installed
try:
    import soundfile as _sf
    import librosa as _librosa
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False
    log_audio = logging.getLogger(__name__)
    log_audio.warning("soundfile/librosa not installed — audio features will be NaN.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — verified against actual zenodo data columns
# ---------------------------------------------------------------------------

FIXED_LENGTH = 400      # resampled timepoints per window  (overridden by --fixed-length)
WINDOW_SEC   = 30.0     # seconds of history before each VAD response  (overridden by --window-sec)
TASKS        = ["T0", "T1", "T2", "T3", "T4"]
SEATS        = ["P1", "P2", "P3", "P4"]

# Features to include in per-user T0 baseline normalisation (delta and z-score).
# Only features with a meaningful "resting state" baseline are listed here.
T0_BASELINE_FEATURES = [
    "pupil_left_mean", "pupil_right_mean", "pupil_avg_mean",
    "pupil_left_std",  "pupil_right_std",  "pupil_avg_std",
    "blink_rate_proxy",
    "eda_phasic_mean", "eda_phasic_std", "eda_tonic_mean", "eda_tonic_std",
    "scr_peak_count",  "scr_amplitude_mean",
    "hr_mean_mean",    "hrv_rmssd_mean",   "temp_skin_mean",
    "ppg_hr_mean",     "ppg_hr_std",       "ppg_rmssd",     "ppg_sdnn",
]

# ----- Tobii Glasses 3 (ET) column indices -----
# Confirmed from: sub-01_ses-..._task-T0_run-01_acq-P1_tobii.tsv.gz
TOBII_GAZE_X      = "value_0"   # screen-normalised gaze x [0-1]
TOBII_GAZE_Y      = "value_1"   # screen-normalised gaze y [0-1]
TOBII_PUPIL_L     = "value_2"   # left pupil diameter (mm)
TOBII_PUPIL_R     = "value_3"   # right pupil diameter (mm, may be NaN)
TOBII_VALIDITY    = "value_4"   # gaze validity (1=valid, 0=invalid)
# value_5–12: 3-D gaze positions in mm (left + right eye in camera space; not used)
TOBII_GAZE_DIR_LX = "value_14"  # gaze unit-vector left eye x
TOBII_GAZE_DIR_LY = "value_15"  # gaze unit-vector left eye y
TOBII_GAZE_DIR_LZ = "value_16"  # gaze unit-vector left eye z
TOBII_GAZE_DIR_RX = "value_17"  # gaze unit-vector right eye x (NaN if monocular)
TOBII_GAZE_DIR_RY = "value_18"  # gaze unit-vector right eye y
TOBII_GAZE_DIR_RZ = "value_19"  # gaze unit-vector right eye z
# value_20–25: head position vectors; value_26–28: NaN

GAZE_COLS  = [TOBII_GAZE_X, TOBII_GAZE_Y,
              TOBII_GAZE_DIR_LX, TOBII_GAZE_DIR_LY, TOBII_GAZE_DIR_LZ,
              TOBII_GAZE_DIR_RX, TOBII_GAZE_DIR_RY, TOBII_GAZE_DIR_RZ,
              TOBII_VALIDITY]
PUPIL_COLS = [TOBII_PUPIL_L, TOBII_PUPIL_R, TOBII_VALIDITY]

# ----- EmotiBit (physio) column indices -----
# Confirmed from: sub-01_ses-..._task-T0_run-01_acq-P1_emotibit.tsv.gz
EMOTIBIT_PPG_IR  = "value_0"    # PPG infrared channel (ADC counts, ~3000–5000)
EMOTIBIT_PPG_RED = "value_1"    # PPG red channel (ADC counts, ~70000–80000)
EMOTIBIT_PPG_GRN = "value_2"    # PPG green channel (ADC counts, highest SNR)
EMOTIBIT_EDA     = "value_3"    # EDA skin conductance (µS, ~0.05–5.0)
# value_4–5: NaN in current firmware
EMOTIBIT_HR      = "value_6"    # Heart rate from firmware algorithm (BPM)
EMOTIBIT_HRV_SDNN = "value_7"   # HRV SDNN (ms) — firmware computed
EMOTIBIT_HRV_RMSSD = "value_8"  # HRV RMSSD fraction — firmware computed
EMOTIBIT_TEMP_THERM = "value_9"  # Thermopile object temperature (°C, ~36–38)
EMOTIBIT_TEMP_AMB   = "value_10" # Barom/thermistor temperature (°C)
EMOTIBIT_TEMP_SKIN  = "value_11" # Skin temperature (°C) — closest to body temp
# value_12: NaN
EMOTIBIT_ACC_X  = "value_13"    # Accelerometer x (g)
EMOTIBIT_ACC_Y  = "value_14"    # Accelerometer y (g)
EMOTIBIT_ACC_Z  = "value_15"    # Accelerometer z (g)
EMOTIBIT_GYR_X  = "value_16"    # Gyroscope x (deg/s)
EMOTIBIT_GYR_Y  = "value_17"    # Gyroscope y (deg/s)
EMOTIBIT_GYR_Z  = "value_18"    # Gyroscope z (deg/s)
EMOTIBIT_MAG_X  = "value_19"    # Magnetometer x (µT)
EMOTIBIT_MAG_Y  = "value_20"    # Magnetometer y (µT)
EMOTIBIT_MAG_Z  = "value_21"    # Magnetometer z (µT)

PPG_COLS = [EMOTIBIT_PPG_IR, EMOTIBIT_PPG_RED, EMOTIBIT_PPG_GRN]
EDA_COLS_RAW = [EMOTIBIT_EDA, EMOTIBIT_HR, EMOTIBIT_TEMP_SKIN]
IMU_COLS = [EMOTIBIT_ACC_X, EMOTIBIT_ACC_Y, EMOTIBIT_ACC_Z,
            EMOTIBIT_GYR_X, EMOTIBIT_GYR_Y, EMOTIBIT_GYR_Z]
# EDA sequence stored in pkl = [raw EDA, EDA_Phasic, EDA_Tonic, HR, Skin Temp]
EDA_SEQ_STORED = [EMOTIBIT_EDA, "EDA_Phasic", "EDA_Tonic", EMOTIBIT_HR, EMOTIBIT_TEMP_SKIN]

ALL_EMOTIBIT_COLS = PPG_COLS + [EMOTIBIT_EDA, EMOTIBIT_HR,
                                 EMOTIBIT_HRV_SDNN, EMOTIBIT_HRV_RMSSD,
                                 EMOTIBIT_TEMP_THERM, EMOTIBIT_TEMP_AMB, EMOTIBIT_TEMP_SKIN] + \
                   IMU_COLS + [EMOTIBIT_MAG_X, EMOTIBIT_MAG_Y, EMOTIBIT_MAG_Z]

SEX_MAP = {"male": 1, "female": 0, "other": -1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_stat(func, arr: np.ndarray) -> float:
    valid = arr[np.isfinite(arr)]
    return float(func(valid)) if valid.size > 0 else float("nan")


def downsample_to_fixed(df: pd.DataFrame, cols: list[str],
                        target: int | None = None) -> pd.DataFrame:
    """Resample a DataFrame to *target* rows (linear interp for numeric cols).

    *target* defaults to the module-level FIXED_LENGTH, which can be changed at
    runtime via --fixed-length before processing begins.
    """
    if target is None:
        target = FIXED_LENGTH
    n = len(df)
    if n == 0:
        return pd.DataFrame(np.zeros((target, len(cols))), columns=cols)
    old_idx = np.arange(n, dtype=float)
    new_idx = np.linspace(0, n - 1, target)
    data = {}
    for col in cols:
        if col in df.columns:
            data[col] = np.interp(new_idx, old_idx,
                                  df[col].ffill().fillna(0).values.astype(float))
        else:
            data[col] = np.zeros(target, dtype=float)
    return pd.DataFrame(data)


def load_tsv_gz(path: Path, cols: list[str] | None = None) -> pd.DataFrame | None:
    """Load a .tsv.gz file keeping lsl_time + requested value columns.

    Deduplicates column list so duplicate entries (e.g. value_4 appearing
    in both GAZE_COLS and PUPIL_COLS) don't produce duplicate DataFrame columns.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", compression="gzip")
    except Exception as exc:
        log.warning("Cannot read %s: %s", path.name, exc)
        return None
    if "lsl_time" not in df.columns:
        log.warning("No lsl_time column in %s", path.name)
        return None
    # Deduplicate while preserving order
    seen: set[str] = {"lsl_time"}
    keep: list[str] = ["lsl_time"]
    for c in (cols or []):
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    return df[keep].sort_values("lsl_time").reset_index(drop=True)


def extract_window(df: pd.DataFrame, t_end: float,
                   duration: float | None = None) -> pd.DataFrame | None:
    """Return rows in [t_end - duration, t_end]; None if fewer than 20 samples.

    *duration* defaults to the module-level WINDOW_SEC.
    """
    if duration is None:
        duration = WINDOW_SEC
    t_start = t_end - duration
    win = df[(df["lsl_time"] >= t_start) & (df["lsl_time"] <= t_end)]
    return win.reset_index(drop=True) if len(win) >= 20 else None


# ---------------------------------------------------------------------------
# Feature extraction (per window)
# ---------------------------------------------------------------------------

def compute_gaze_features(win: pd.DataFrame) -> dict:
    mask = (win[TOBII_VALIDITY].values == 1) if TOBII_VALIDITY in win.columns else np.ones(len(win), bool)
    gx = win[TOBII_GAZE_X].values[mask] if TOBII_GAZE_X in win.columns else np.array([])
    gy = win[TOBII_GAZE_Y].values[mask] if TOBII_GAZE_Y in win.columns else np.array([])
    feats = {
        "gaze_x_mean":  safe_stat(np.mean, gx),
        "gaze_x_std":   safe_stat(np.std,  gx),
        "gaze_y_mean":  safe_stat(np.mean, gy),
        "gaze_y_std":   safe_stat(np.std,  gy),
        "validity_rate": float(np.mean(mask)) if len(mask) > 0 else 0.0,
        "saccade_proxy": float(np.mean(np.abs(np.diff(gx)) + np.abs(np.diff(gy)))) if len(gx) > 1 else float("nan"),
    }
    return feats


def compute_pupil_features(win: pd.DataFrame) -> dict:
    mask = (win[TOBII_VALIDITY].values == 1) if TOBII_VALIDITY in win.columns else np.ones(len(win), bool)
    feats = {}
    for side, col in [("left", TOBII_PUPIL_L), ("right", TOBII_PUPIL_R)]:
        vals = win[col].values[mask] if col in win.columns else np.array([])
        vals = vals[np.isfinite(vals)]
        feats[f"pupil_{side}_mean"] = safe_stat(np.mean, vals)
        feats[f"pupil_{side}_std"]  = safe_stat(np.std,  vals)
    avg = np.where(mask,
                   (win.get(TOBII_PUPIL_L, pd.Series(np.nan, index=win.index)).values +
                    win.get(TOBII_PUPIL_R, pd.Series(np.nan, index=win.index)).values) / 2,
                   np.nan)
    feats["pupil_avg_mean"]    = safe_stat(np.nanmean, avg)
    feats["pupil_avg_std"]     = safe_stat(np.nanstd,  avg)
    feats["blink_rate_proxy"]  = float(1.0 - np.mean(mask))
    return feats


def add_eda_decomposition(win: pd.DataFrame, sr: int) -> pd.DataFrame:
    """Append EDA_Phasic and EDA_Tonic columns to a copy of *win*."""
    win = win.copy()
    if EMOTIBIT_EDA not in win.columns:
        win["EDA_Phasic"] = 0.0
        win["EDA_Tonic"]  = 0.0
        return win
    eda = win[EMOTIBIT_EDA].ffill().fillna(0).values.astype(float)
    try:
        eda_z = nk.standardize(eda)
        data, _ = nk.eda_process(eda_z, sampling_rate=max(sr, 1), method="neurokit")
        win["EDA_Phasic"] = data["EDA_Phasic"].values
        win["EDA_Tonic"]  = data["EDA_Tonic"].values
    except Exception:
        win["EDA_Phasic"] = 0.0
        win["EDA_Tonic"]  = 0.0
    return win


def compute_eda_features(win: pd.DataFrame, sr: int) -> dict:
    feats: dict = {}
    if EMOTIBIT_EDA not in win.columns:
        return feats
    eda = win[EMOTIBIT_EDA].ffill().fillna(0).values.astype(float)
    try:
        eda_z = nk.standardize(eda)
        data, result = nk.eda_process(eda_z, sampling_rate=max(sr, 1), method="neurokit")
        phasic = data["EDA_Phasic"].values
        tonic  = data["EDA_Tonic"].values
        feats["eda_phasic_mean"] = safe_stat(np.mean, phasic)
        feats["eda_phasic_std"]  = safe_stat(np.std,  phasic)
        feats["eda_tonic_mean"]  = safe_stat(np.mean, tonic)
        feats["eda_tonic_std"]   = safe_stat(np.std,  tonic)
        scr_amp = result.get("SCR_Amplitude", np.array([]))
        if hasattr(scr_amp, "values"):
            scr_amp = scr_amp.dropna().values
        scr_amp = np.array(scr_amp, dtype=float)
        feats["scr_peak_count"]      = len(scr_amp)
        feats["scr_amplitude_mean"]  = safe_stat(np.mean, scr_amp)
        feats["scr_amplitude_std"]   = safe_stat(np.std,  scr_amp)
    except Exception:
        for k in ["eda_phasic_mean", "eda_phasic_std", "eda_tonic_mean",
                  "eda_tonic_std", "scr_peak_count", "scr_amplitude_mean", "scr_amplitude_std"]:
            feats[k] = float("nan")
    for col, name in [(EMOTIBIT_TEMP_SKIN, "temp_skin"), (EMOTIBIT_HR, "hr_mean"),
                      (EMOTIBIT_HRV_RMSSD, "hrv_rmssd")]:
        if col in win.columns:
            v = win[col].dropna().values.astype(float)
            feats[f"{name}_mean"] = safe_stat(np.mean, v)
            feats[f"{name}_std"]  = safe_stat(np.std,  v)
    return feats


def compute_ppg_features(win: pd.DataFrame, sr: int) -> dict:
    feats: dict = {}
    # Use green channel (highest SNR on EmotiBit)
    col = EMOTIBIT_PPG_GRN
    if col not in win.columns:
        return feats
    raw = win[col].ffill().fillna(0).values.astype(float)
    try:
        ppg_z = nk.standardize(raw)
        signals, info = nk.ppg_process(ppg_z, sampling_rate=max(sr, 1))
        rate = signals["PPG_Rate"].values
        feats["ppg_hr_mean"] = safe_stat(np.mean, rate)
        feats["ppg_hr_std"]  = safe_stat(np.std,  rate)
        feats["ppg_hr_min"]  = safe_stat(np.min,  rate)
        feats["ppg_hr_max"]  = safe_stat(np.max,  rate)
        peaks = info.get("PPG_Peaks", np.array([]))
        peaks = np.array(peaks, dtype=int)
        if len(peaks) > 1:
            ibi = np.diff(peaks) / max(sr, 1)
            feats["ppg_rmssd"]      = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 1 else float("nan")
            feats["ppg_sdnn"]       = safe_stat(np.std, ibi)
            feats["ppg_peak_count"] = float(len(peaks))
        else:
            feats["ppg_rmssd"]      = float("nan")
            feats["ppg_sdnn"]       = float("nan")
            feats["ppg_peak_count"] = 0.0
    except Exception:
        for k in ["ppg_hr_mean", "ppg_hr_std", "ppg_hr_min", "ppg_hr_max",
                  "ppg_rmssd", "ppg_sdnn", "ppg_peak_count"]:
            feats[k] = float("nan")
    return feats


def compute_imu_features(win: pd.DataFrame) -> dict:
    feats: dict = {}
    acc_cols = [EMOTIBIT_ACC_X, EMOTIBIT_ACC_Y, EMOTIBIT_ACC_Z]
    gyr_cols = [EMOTIBIT_GYR_X, EMOTIBIT_GYR_Y, EMOTIBIT_GYR_Z]
    for prefix, cols in [("acc", acc_cols), ("gyr", gyr_cols)]:
        present = [c for c in cols if c in win.columns]
        if not present:
            continue
        vals = win[present].fillna(0).values
        mag  = np.linalg.norm(vals, axis=1)
        feats[f"{prefix}_mag_mean"] = safe_stat(np.mean, mag)
        feats[f"{prefix}_mag_std"]  = safe_stat(np.std,  mag)
        for ax, col in zip(["x", "y", "z"], cols):
            if col in win.columns:
                v = win[col].fillna(0).values.astype(float)
                feats[f"{prefix}_{ax}_mean"] = safe_stat(np.mean, v)
                feats[f"{prefix}_{ax}_std"]  = safe_stat(np.std,  v)
    return feats


# ---------------------------------------------------------------------------
# Session data loading
# ---------------------------------------------------------------------------

def load_task_windows(annot_dir: Path) -> dict[str, tuple[float, float]]:
    """Load task LSL time boundaries from task_run_windows.tsv.

    Returns {task: (start_lsl, end_lsl)}.
    """
    files = list(annot_dir.glob("*task_run_windows.tsv"))
    if not files:
        return {}
    df = pd.read_csv(files[0], sep="\t")
    result: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        task = str(row.get("task", "")).strip()
        if task in TASKS:
            result[task] = (float(row["start_lsl"]), float(row["end_lsl"]))
    return result


def load_group_participants(annot_dir: Path) -> pd.DataFrame:
    """Load seat → participant metadata from group_participants.tsv."""
    files = list(annot_dir.glob("*group_participants.tsv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0], sep="\t")


def load_vad_responses(beh_dir: Path) -> pd.DataFrame:
    """Load all VAD self-reports from stimuli_answers.tsv.

    Returns DataFrame with columns:
        lsl_clock, task, participant (seat), valence, arousal, dominance
    """
    files = list(beh_dir.glob("*stimuli_answers.tsv"))
    if not files:
        return pd.DataFrame()

    df = pd.read_csv(files[0], sep="\t")
    vad = df[
        (df["response_type"] == "vad") &
        (df["item_key"].isin(["valence", "arousal", "dominance"]))
    ].copy()

    if vad.empty:
        return pd.DataFrame()

    vad["item_value"] = pd.to_numeric(vad["item_value"], errors="coerce")

    wide = (
        vad.pivot_table(
            index=["lsl_clock", "task", "participant"],
            columns="item_key",
            values="item_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    # Ensure all three dimensions are present
    for dim in ["valence", "arousal", "dominance"]:
        if dim not in wide.columns:
            wide[dim] = float("nan")

    return wide.dropna(subset=["valence", "arousal"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

# DPA mic names used in audio file paths (seats P1-P4 may map to any of these)
_AUDIO_MIC_NAMES = ["mic9", "mic10", "mic11", "mic12"]
# Per-seat mic assignment (verified from transcript mic column)
_SEAT_TO_MIC: dict[str, str] = {"P1": "mic9", "P2": "mic10", "P3": "mic11", "P4": "mic12"}
# Fallback order when seat is unknown
_AUDIO_GROUP_MICS = ["mic9", "mic10", "mic11", "mic12"]


def load_audio_lsl_anchors(annot_dir: Path) -> dict[str, tuple[float, float]]:
    """Read lsl_sync.tsv to build {mic_name: (first_lsl_time, first_audio_pos_sec)}.

    The ffmpeg_progress_dpa_mic{N}_aud streams each emit (lsl_time, audio_pos_sec)
    pairs at ~2 Hz.  The first row gives us the linear anchor:
        audio_pos_sec ≈ first_audio_pos + (lsl_time - first_lsl_time)
    Clock drift over a 2-hour session is < 50 ms, so linear is sufficient.
    """
    sync_file = next(annot_dir.glob("*task-T0T1T2T3T4_acq-lsl_sync.tsv"), None)
    if sync_file is None:
        return {}
    try:
        sync = pd.read_csv(sync_file, sep="\t")
    except Exception:
        return {}
    anchors: dict[str, tuple[float, float]] = {}
    for mic in _AUDIO_MIC_NAMES:
        stream_key = f"dpa_{mic}_aud"
        rows = sync[sync["stream_name"].str.contains(stream_key, na=False)].sort_values("lsl_time")
        if not rows.empty:
            first = rows.iloc[0]
            anchors[mic] = (float(first["lsl_time"]), float(first["value_0"]))
    return anchors


def find_audio_file_for_task(audio_dir: Path, ses_id: str, task: str,
                             mic: str) -> tuple[Path | None, bool]:
    """Return (audio_file, is_full_session) for the best matching audio file.

    is_full_session=True means the file covers the whole session and the LSL
    anchor should be used to compute the offset.
    is_full_session=False means the file starts at task_start_lsl (offset 0 s).

    Tries (in order):
      1. Per-task split: *task-{T}_run-01_acq-*{mic}*_audio.wav  → not full session
      2. Full-session:   *task-T0T1T2T3T4_acq-*{mic}*_audio.wav → full session
    """
    # Per-task split files start at task start (offset from task_start_lsl)
    task_hits = list(audio_dir.glob(f"*task-{task}_run-01_acq-*{mic}*_audio.wav"))
    if task_hits:
        return task_hits[0], False

    # Full-session file: use LSL anchor
    full_hits = list(audio_dir.glob(f"*task-T0T1T2T3T4_acq-*{mic}*_audio.wav"))
    if full_hits:
        return full_hits[0], True

    return None, False


def _lsl_to_audio_sec(lsl_time: float, anchor: tuple[float, float]) -> float:
    """Convert a LSL timestamp to an audio-file position in seconds."""
    anchor_lsl, anchor_audio = anchor
    return anchor_audio + (lsl_time - anchor_lsl)


def compute_audio_features(audio_path: Path, audio_start_sec: float,
                            duration_sec: float) -> dict:
    """Extract handcrafted audio features for a time window.

    Features (33 values):
      rms_mean, rms_std, rms_log          — energy
      zcr_mean                            — voice activity proxy
      sc_mean, sc_std                     — spectral centroid (brightness)
      sr_mean                             — spectral rolloff
      speech_activity                     — fraction of frames above RMS threshold
      mfcc{1-13}_mean, mfcc{1-13}_std    — timbre / spectral shape
    All NaN on failure or if audio unavailable.
    """
    nan_dict: dict = {
        "audio_rms_mean": float("nan"), "audio_rms_std": float("nan"),
        "audio_rms_log": float("nan"), "audio_zcr_mean": float("nan"),
        "audio_sc_mean": float("nan"), "audio_sc_std": float("nan"),
        "audio_sr_mean": float("nan"), "audio_speech_activity": float("nan"),
    }
    for i in range(1, 14):
        nan_dict[f"audio_mfcc{i}_mean"] = float("nan")
        nan_dict[f"audio_mfcc{i}_std"]  = float("nan")

    if not _AUDIO_AVAILABLE or audio_path is None or not audio_path.exists():
        return nan_dict
    if audio_start_sec < 0:
        audio_start_sec = 0.0

    try:
        info = _sf.info(str(audio_path))
        sr: int = info.samplerate
        start_frame = int(audio_start_sec * sr)
        n_frames = int(duration_sec * sr)
        if start_frame >= info.frames:
            return nan_dict
        n_frames = min(n_frames, info.frames - start_frame)
        if n_frames < sr * 0.1:   # < 100 ms — too short
            return nan_dict
        chunk, _ = _sf.read(str(audio_path), start=start_frame,
                             frames=n_frames, dtype="float32", always_2d=False)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)   # mix to mono
        if len(chunk) < 256:
            return nan_dict

        # ── Energy ──────────────────────────────────────────────────────────
        rms = _librosa.feature.rms(y=chunk, frame_length=1024, hop_length=256)[0]
        feats: dict = {
            "audio_rms_mean":        float(np.mean(rms)),
            "audio_rms_std":         float(np.std(rms)),
            "audio_rms_log":         float(np.log1p(np.mean(rms))),
        }

        # ── Voice activity ───────────────────────────────────────────────────
        zcr = _librosa.feature.zero_crossing_rate(chunk, frame_length=1024, hop_length=256)[0]
        feats["audio_zcr_mean"] = float(np.mean(zcr))

        # ── Spectral shape ───────────────────────────────────────────────────
        sc = _librosa.feature.spectral_centroid(y=chunk, sr=sr, hop_length=256)[0]
        feats["audio_sc_mean"] = float(np.mean(sc))
        feats["audio_sc_std"]  = float(np.std(sc))

        ro = _librosa.feature.spectral_rolloff(y=chunk, sr=sr, hop_length=256)[0]
        feats["audio_sr_mean"] = float(np.mean(ro))

        # ── Speech activity ──────────────────────────────────────────────────
        rms_threshold = 5e-4   # empirical: ~-66 dBFS for near-silence
        feats["audio_speech_activity"] = float(np.mean(rms > rms_threshold))

        # ── MFCCs ────────────────────────────────────────────────────────────
        mfccs = _librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, hop_length=256)
        for i in range(13):
            feats[f"audio_mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            feats[f"audio_mfcc{i+1}_std"]  = float(np.std(mfccs[i]))

        return feats

    except Exception as exc:
        logging.getLogger(__name__).debug("Audio feature extraction failed: %s", exc)
        return nan_dict


def get_audio_features_for_window(
    audio_dir: Path | None,
    ses_id: str,
    task: str,
    t_start: float,
    t_end: float,
    lsl_anchors: dict[str, tuple[float, float]],
    task_start_lsl: float,
    seat: str = "",
) -> dict:
    """High-level wrapper: pick a mic, find the audio file, extract features.

    Uses the participant's seat-specific mic (P1→mic9 … P4→mic12) when seat is
    provided; otherwise falls back to the first available mic in _AUDIO_GROUP_MICS.
    Falls back to NaN features if no audio or no sync anchor is found.
    """
    if audio_dir is None or not _AUDIO_AVAILABLE:
        return compute_audio_features(None, 0, t_end - t_start)

    duration_sec = t_end - t_start

    # Seat-specific mic first, then all mics as fallback
    mics_to_try: list[str] = []
    if seat and seat in _SEAT_TO_MIC:
        mics_to_try.append(_SEAT_TO_MIC[seat])
    for m in _AUDIO_GROUP_MICS:
        if m not in mics_to_try:
            mics_to_try.append(m)

    for mic in mics_to_try:
        audio_path, is_full_session = find_audio_file_for_task(audio_dir, ses_id, task, mic)
        if audio_path is None:
            continue

        if is_full_session:
            # Full-session file: use LSL → audio-time anchor
            if mic not in lsl_anchors:
                continue
            anchor = lsl_anchors[mic]
            audio_start_sec = _lsl_to_audio_sec(t_start, anchor)
        else:
            # Per-task split: file starts exactly at task_start_lsl
            audio_start_sec = t_start - task_start_lsl

        return compute_audio_features(audio_path, audio_start_sec, duration_sec)

    # No mic found — return NaN features
    return compute_audio_features(None, 0, duration_sec)


def get_speech_features_for_window(
    audio_dir: Path | None,
    ses_id: str,
    task: str,
    seat: str,
    subject_id: str,
    t_start: float,
    t_end: float,
    task_start_lsl: float,
) -> dict:
    """Extract per-participant speech features from the transcript for a window.

    The transcript TSV contains utterances from all speakers; we filter to the
    target participant (matched by seat prefix, e.g. 'P1_sub-017') and count
    how much they spoke in the window [t_start, t_end].

    Onset times in the transcript are in seconds from task start
    (i.e. onset = lsl_time - task_start_lsl).

    Features (7 values):
      speech_time_sec        — total speaking time in window [0, window_sec]
      speech_n_utterances    — number of utterances in window
      speech_words           — total word count (spaces + 1 per utterance)
      speech_energy_mean     — mean RMS energy of utterances (0 if none)
      speech_energy_ratio    — mean energy_ratio (SNR proxy, 0 if none)
      speech_backchannel_frac — fraction of utterances that are backchannels
      speech_fraction        — speech_time_sec / window_sec
    """
    nan_dict = {
        "speech_time_sec": float("nan"),
        "speech_n_utterances": float("nan"),
        "speech_words": float("nan"),
        "speech_energy_mean": float("nan"),
        "speech_energy_ratio": float("nan"),
        "speech_backchannel_frac": float("nan"),
        "speech_fraction": float("nan"),
    }

    if audio_dir is None:
        return nan_dict

    # Find transcript file for this task
    pattern = f"*ses-*_task-{task}_run-01_desc-withBackchannels_transcript.tsv"
    hits = list(audio_dir.glob(pattern))
    if not hits:
        return nan_dict

    try:
        df = pd.read_csv(hits[0], sep="\t")
    except Exception:
        return nan_dict

    # Determine expected speaker prefix (e.g. "P1" for seat P1)
    # Speaker labels are like "P1_sub-017" — match by seat prefix
    mask_speaker = df["speaker"].str.startswith(seat + "_", na=False)
    if not mask_speaker.any():
        # Fallback: try matching by subject_id suffix
        mask_speaker = df["speaker"].str.contains(subject_id, na=False)
    if not mask_speaker.any():
        # This participant may not have spoken — return zeros (not NaN)
        window_sec = t_end - t_start
        return {
            "speech_time_sec": 0.0,
            "speech_n_utterances": 0.0,
            "speech_words": 0.0,
            "speech_energy_mean": 0.0,
            "speech_energy_ratio": 0.0,
            "speech_backchannel_frac": 0.0,
            "speech_fraction": 0.0,
        }

    part_df = df[mask_speaker].copy()

    # Convert task-relative onset to LSL-relative: onset_lsl = onset + task_start_lsl
    # Window is [t_start, t_end] in LSL time
    # Utterance overlaps window if onset_lsl < t_end AND onset_lsl + duration > t_start
    window_sec = t_end - t_start
    onset_lsl = part_df["onset"] + task_start_lsl
    utt_end_lsl = onset_lsl + part_df["duration"]

    in_window = (onset_lsl < t_end) & (utt_end_lsl > t_start)
    win_utts = part_df[in_window]

    if win_utts.empty:
        return {
            "speech_time_sec": 0.0,
            "speech_n_utterances": 0.0,
            "speech_words": 0.0,
            "speech_energy_mean": 0.0,
            "speech_energy_ratio": 0.0,
            "speech_backchannel_frac": 0.0,
            "speech_fraction": 0.0,
        }

    # Clip utterance durations to window boundaries
    clipped_durations = (
        win_utts["duration"]
        .clip(upper=(t_end - (win_utts["onset"] + task_start_lsl)).clip(lower=0))
    )

    speech_time = float(clipped_durations.sum())
    n_utt = len(win_utts)
    word_count = float(win_utts["text"].apply(
        lambda t: len(str(t).split()) if pd.notna(t) else 0
    ).sum())

    energy_col = "energy" if "energy" in win_utts.columns else None
    ratio_col  = "energy_ratio" if "energy_ratio" in win_utts.columns else None
    conf_col   = "confidence" if "confidence" in win_utts.columns else None

    energy_mean = float(win_utts[energy_col].mean()) if energy_col else 0.0
    energy_ratio = float(win_utts[ratio_col].mean()) if ratio_col else 0.0
    backchannel_frac = float(
        (win_utts[conf_col] == "backchannel").mean()
    ) if conf_col else 0.0

    return {
        "speech_time_sec": speech_time,
        "speech_n_utterances": float(n_utt),
        "speech_words": word_count,
        "speech_energy_mean": energy_mean,
        "speech_energy_ratio": energy_ratio,
        "speech_backchannel_frac": backchannel_frac,
        "speech_fraction": min(speech_time / max(window_sec, 1e-6), 1.0),
    }


# ---------------------------------------------------------------------------
# Per-window record builder
# ---------------------------------------------------------------------------

def _add_delta_features(raw_feats: dict, t0_stats: dict) -> dict:
    """Return {feat}_delta_t0 and {feat}_z_t0 for each feature in T0_BASELINE_FEATURES.

    t0_stats: {feat: (mean, std)} computed from T0 windows for this (session, seat).
    """
    out: dict = {}
    for feat in T0_BASELINE_FEATURES:
        val = raw_feats.get(feat, float("nan"))
        mean, std = t0_stats.get(feat, (float("nan"), float("nan")))
        if np.isfinite(val) and np.isfinite(mean):
            out[f"{feat}_delta_t0"] = val - mean
            if np.isfinite(std) and std > 1e-6:
                out[f"{feat}_z_t0"] = (val - mean) / std
            else:
                out[f"{feat}_z_t0"] = float("nan")
        else:
            out[f"{feat}_delta_t0"] = float("nan")
            out[f"{feat}_z_t0"] = float("nan")
    return out


def build_window_record(
    tobii_df: pd.DataFrame | None,
    emotibit_df: pd.DataFrame | None,
    t_response: float,
    metadata: dict,
    audio_dir: Path | None = None,
    lsl_anchors: dict | None = None,
    task_start_lsl: float = 0.0,
    t0_stats: dict | None = None,
) -> dict | None:
    """Extract and featurise one 30-second window ending at *t_response*.

    Returns None if no sensor data is available for this window.
    Optionally extracts audio features when *audio_dir* and *lsl_anchors* are provided.
    """
    gaze_win_raw     = extract_window(tobii_df,    t_response) if tobii_df    is not None else None
    emotibit_win_raw = extract_window(emotibit_df, t_response) if emotibit_df is not None else None

    if gaze_win_raw is None and emotibit_win_raw is None:
        return None

    record: dict = {**metadata, "vad_timestamp_lsl": t_response}

    # -- Gaze --
    if gaze_win_raw is not None:
        sr_et = max(1, int(len(gaze_win_raw) / WINDOW_SEC))
        gaze_seq  = downsample_to_fixed(gaze_win_raw, GAZE_COLS)
        pupil_seq = downsample_to_fixed(gaze_win_raw, PUPIL_COLS)
        record["gaze_seq"]      = gaze_seq
        record["pupil_seq"]     = pupil_seq
        record["gaze_features"] = compute_gaze_features(gaze_win_raw)
        record["pupil_features"]= compute_pupil_features(gaze_win_raw)
    else:
        record["gaze_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(GAZE_COLS))),  columns=GAZE_COLS)
        record["pupil_seq"]     = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PUPIL_COLS))), columns=PUPIL_COLS)
        record["gaze_features"] = {}
        record["pupil_features"]= {}

    # -- EmotiBit (EDA, PPG, IMU) --
    if emotibit_win_raw is not None:
        sr_em = max(1, int(len(emotibit_win_raw) / WINDOW_SEC))

        # EDA + decomposition
        eda_decomp    = add_eda_decomposition(emotibit_win_raw, sr_em)
        eda_seq       = downsample_to_fixed(eda_decomp, EDA_SEQ_STORED)
        record["eda_seq"]      = eda_seq
        record["eda_features"] = compute_eda_features(emotibit_win_raw, sr_em)

        # PPG
        ppg_seq = downsample_to_fixed(emotibit_win_raw, PPG_COLS)
        record["ppg_seq"]      = ppg_seq
        record["ppg_features"] = compute_ppg_features(emotibit_win_raw, sr_em)

        # IMU
        imu_seq = downsample_to_fixed(emotibit_win_raw, IMU_COLS)
        record["imu_seq"]      = imu_seq
        record["imu_features"] = compute_imu_features(emotibit_win_raw)
    else:
        record["eda_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(EDA_SEQ_STORED))), columns=EDA_SEQ_STORED)
        record["ppg_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PPG_COLS))),        columns=PPG_COLS)
        record["imu_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(IMU_COLS))),        columns=IMU_COLS)
        record["eda_features"] = {}
        record["ppg_features"] = {}
        record["imu_features"] = {}

    # -- Audio (per-seat DPA mic) and speech features (from transcript) --
    t_start    = t_response - WINDOW_SEC
    seat       = metadata.get("seat", "")
    subject_id = metadata.get("subject_id", "")
    ses_id     = metadata.get("session_id", "")
    task       = metadata.get("task", "")

    record["audio_features"] = get_audio_features_for_window(
        audio_dir=audio_dir,
        ses_id=ses_id,
        task=task,
        t_start=t_start,
        t_end=t_response,
        lsl_anchors=lsl_anchors or {},
        task_start_lsl=task_start_lsl,
        seat=seat,
    )
    record["speech_features"] = get_speech_features_for_window(
        audio_dir=audio_dir,
        ses_id=ses_id,
        task=task,
        seat=seat,
        subject_id=subject_id,
        t_start=t_start,
        t_end=t_response,
        task_start_lsl=task_start_lsl,
    )

    # T0 baseline-normalised features (delta and z-score relative to resting state).
    # Only populated for non-T0 tasks when t0_stats are provided.
    if t0_stats and metadata.get("task") != "T0":
        all_raw: dict = {}
        for fk in ("gaze_features", "pupil_features", "eda_features",
                   "ppg_features", "imu_features"):
            fd = record.get(fk, {})
            if isinstance(fd, dict):
                all_raw.update(fd)
        record["baseline_features"] = _add_delta_features(all_raw, t0_stats)
    else:
        record["baseline_features"] = {}

    return record


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def process_session(session_dir: Path,
                    with_t0_baseline_features: bool = False,
                    dry_run: bool = False) -> list[dict]:
    ses_id = session_dir.name   # e.g. ses-20260312_grp-07_run01

    annot_dir  = session_dir / "annot"
    beh_dir    = session_dir / "beh"
    et_dir     = session_dir / "et"
    physio_dir = session_dir / "physio"
    audio_dir  = session_dir / "audio"
    audio_dir  = audio_dir if audio_dir.is_dir() else None

    # ── Metadata ──────────────────────────────────────────────────────────
    task_windows   = load_task_windows(annot_dir)
    group_parts    = load_group_participants(annot_dir)
    vad_responses  = load_vad_responses(beh_dir)

    # ── Audio LSL sync anchors (one per session) ────────────────────────
    lsl_anchors = load_audio_lsl_anchors(annot_dir) if audio_dir is not None else {}
    if lsl_anchors:
        log.debug("  Audio anchors for %s: %s", ses_id, list(lsl_anchors.keys()))

    if group_parts.empty:
        log.warning("No group_participants.tsv in %s — skipping.", ses_id)
        return []
    if vad_responses.empty:
        log.warning("No VAD responses in %s — skipping.", ses_id)
        return []

    if dry_run:
        # Coverage-only mode: count available files without full feature extraction
        coverage: dict = {"session_id": ses_id, "n_participants": len(group_parts)}
        for seat in SEATS:
            for task in TASKS:
                et_files = list(et_dir.glob(f"*task-{task}_run-01_acq-{seat}_tobii.tsv.gz"))
                em_files = list(physio_dir.glob(f"*task-{task}_run-01_acq-{seat}_emotibit.tsv.gz"))
                coverage[f"{seat}_{task}_et"] = len(et_files) > 0
                coverage[f"{seat}_{task}_physio"] = len(em_files) > 0
        return [coverage]

    # ── Pass 1: compute T0 per-seat baselines (only when requested) ──
    t0_baselines: dict[str, dict] = {}   # {seat: {feat: (mean, std)}}
    if with_t0_baseline_features:
        for seat in SEATS:
            t0_vad = vad_responses[(vad_responses["participant"] == seat) &
                                   (vad_responses["task"] == "T0")]
            if t0_vad.empty:
                continue
            tobii_t0 = list(et_dir.glob(f"*task-T0_run-01_acq-{seat}_tobii.tsv.gz"))
            emoti_t0 = list(physio_dir.glob(f"*task-T0_run-01_acq-{seat}_emotibit.tsv.gz"))
            tobii_df_t0 = load_tsv_gz(tobii_t0[0], GAZE_COLS + PUPIL_COLS) if tobii_t0 else None
            emoti_df_t0 = load_tsv_gz(emoti_t0[0], ALL_EMOTIBIT_COLS) if emoti_t0 else None
            t_start_t0, t_end_t0 = task_windows.get("T0", (float("-inf"), float("inf")))
            feat_lists: dict[str, list] = {f: [] for f in T0_BASELINE_FEATURES}
            for _, vad_row in t0_vad.iterrows():
                t_resp = float(vad_row["lsl_clock"])
                if not (t_start_t0 <= t_resp <= t_end_t0):
                    continue
                if tobii_df_t0 is not None:
                    gaze_win = extract_window(tobii_df_t0, t_resp)
                    if gaze_win is not None:
                        for k, v in compute_pupil_features(gaze_win).items():
                            if k in feat_lists:
                                feat_lists[k].append(float(v) if np.isfinite(float(v)) else float("nan"))
                if emoti_df_t0 is not None:
                    sr_em = max(1, int(25 / WINDOW_SEC))
                    emoti_win = extract_window(emoti_df_t0, t_resp)
                    if emoti_win is not None:
                        sr_em = max(1, int(len(emoti_win) / WINDOW_SEC))
                        for k, v in compute_eda_features(emoti_win, sr_em).items():
                            if k in feat_lists:
                                feat_lists[k].append(float(v) if np.isfinite(float(v)) else float("nan"))
                        for k, v in compute_ppg_features(emoti_win, sr_em).items():
                            if k in feat_lists:
                                feat_lists[k].append(float(v) if np.isfinite(float(v)) else float("nan"))
            seat_stats: dict[str, tuple] = {}
            for feat, vals in feat_lists.items():
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    seat_stats[feat] = (float(np.mean(finite)), float(np.std(finite)))
            if seat_stats:
                t0_baselines[seat] = seat_stats
        if t0_baselines:
            log.debug("  T0 baselines computed for %d seats.", len(t0_baselines))

    records: list[dict] = []

    for seat in SEATS:
        # ── Participant info ────────────────────────────────────────────
        row = group_parts[group_parts["seat"] == seat]
        if row.empty:
            continue
        row = row.iloc[0]

        participant_id = str(row["participant_id"])
        personality = {
            "bfi44_e": float(row.get("bfi44_e", float("nan"))),
            "bfi44_a": float(row.get("bfi44_a", float("nan"))),
            "bfi44_c": float(row.get("bfi44_c", float("nan"))),
            "bfi44_n": float(row.get("bfi44_n", float("nan"))),
            "bfi44_o": float(row.get("bfi44_o", float("nan"))),
        }
        sex_str = str(row.get("sex", "")).lower()
        sex_int = SEX_MAP.get(sex_str, -1)
        age_val = int(row.get("age", -1)) if pd.notna(row.get("age")) else -1

        # ── VAD responses for this seat ──────────────────────────────
        seat_vad = vad_responses[vad_responses["participant"] == seat]
        if seat_vad.empty:
            continue

        # ── Load sensor files per task ───────────────────────────────
        # Process task by task to save memory and keep task boundaries clean
        for task in TASKS:
            task_vad = seat_vad[seat_vad["task"] == task]
            if task_vad.empty:
                continue

            # Load Tobii for this seat+task
            tobii_files = list(et_dir.glob(f"*task-{task}_run-01_acq-{seat}_tobii.tsv.gz"))
            tobii_df    = load_tsv_gz(tobii_files[0], GAZE_COLS + PUPIL_COLS) if tobii_files else None

            # Load EmotiBit for this seat+task
            emoti_files = list(physio_dir.glob(f"*task-{task}_run-01_acq-{seat}_emotibit.tsv.gz"))
            emoti_df    = load_tsv_gz(emoti_files[0], ALL_EMOTIBIT_COLS) if emoti_files else None

            if tobii_df is None and emoti_df is None:
                log.debug("  No sensor files for %s / %s / %s", ses_id, seat, task)
                continue

            t_start, t_end = task_windows.get(task, (float("-inf"), float("inf")))

            for _, vad_row in task_vad.iterrows():
                t_response = float(vad_row["lsl_clock"])
                valence    = float(vad_row.get("valence",   float("nan")))
                arousal    = float(vad_row.get("arousal",   float("nan")))
                dominance  = float(vad_row.get("dominance", float("nan")))

                # Response must fall within this task's LSL window
                if not (t_start <= t_response <= t_end):
                    log.debug("  VAD response at %.1f outside task %s [%.1f–%.1f]",
                              t_response, task, t_start, t_end)
                    continue

                metadata = {
                    "session_id":    ses_id,
                    "subject_id":    participant_id,
                    "seat":          seat,
                    "task":          task,
                    "task_start_lsl": t_start,
                    "task_end_lsl":   t_end,
                    "valence":       valence,
                    "arousal":       arousal,
                    "dominance":     dominance,
                    "sex":           sex_int,
                    "age":           age_val,
                    **personality,
                }

                rec = build_window_record(
                    tobii_df, emoti_df, t_response, metadata,
                    audio_dir=audio_dir,
                    lsl_anchors=lsl_anchors,
                    task_start_lsl=t_start,
                    t0_stats=t0_baselines.get(seat) if with_t0_baseline_features else None,
                )
                if rec is not None:
                    records.append(rec)

    n_speech_valid = sum(
        1 for r in records
        if isinstance(r.get("speech_features"), dict)
        and np.isfinite(float(r["speech_features"].get("speech_n_utterances", float("nan"))))
    )
    if records:
        log.info("  %s: %d/%d records with valid speech features",
                 ses_id, n_speech_valid, len(records))

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(dataset_path: str, output_path: str,
         window_sec: float = 30.0, fixed_length: int = 400,
         with_t0_baseline_features: bool = False,
         validate_coverage: bool = False) -> None:
    global WINDOW_SEC, FIXED_LENGTH
    WINDOW_SEC   = window_sec
    FIXED_LENGTH = fixed_length
    log.info("Window: %.1f s  |  Fixed length: %d timepoints  (%.2f Hz effective)",
             WINDOW_SEC, FIXED_LENGTH, FIXED_LENGTH / WINDOW_SEC)
    if with_t0_baseline_features:
        log.info("T0 baseline features ENABLED: delta_t0 and z_t0 features will be added.")

    root = Path(dataset_path)

    # Discover session directories (sub-*/ses-*)
    session_dirs = sorted(root.glob("sub-*/ses-*"))
    session_dirs = [s for s in session_dirs if s.is_dir() and (s / "annot").exists()]
    log.info("Found %d session directories.", len(session_dirs))

    all_records: list[dict] = []
    for i, ses_dir in enumerate(session_dirs, 1):
        log.info("[%d/%d] %s …", i, len(session_dirs), ses_dir.name)
        recs = process_session(ses_dir,
                               with_t0_baseline_features=with_t0_baseline_features,
                               dry_run=validate_coverage)
        log.info("  → %d labeled windows", len(recs))
        all_records.extend(recs)

    if validate_coverage:
        # Print coverage matrix and exit without writing pickle
        import json as _json
        print("\n=== Coverage Report ===")
        for cov in all_records:
            print(_json.dumps(cov, indent=2))
        return

    if not all_records:
        log.error("No records extracted. Check dataset path and file patterns.")
        return

    df = pd.DataFrame(all_records)
    log.info("Total: %d windows  |  %d participants  |  %d sessions",
             len(df),
             df["subject_id"].nunique() if "subject_id" in df else -1,
             df["session_id"].nunique() if "session_id" in df else -1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(str(out))
    log.info("Saved to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract labeled VAD windows from GroupAffect-4 BIDS (zenodo format)."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Root of the zenodo BIDS dataset (contains sub-*/ses-*).")
    parser.add_argument("--output", default="",
                        help="Output pickle path. Defaults to data/mumt/dataset_{W}s.pkl "
                             "where W = --window-sec value.")
    parser.add_argument("--window-sec", type=float, default=30.0,
                        help="Length of sensor history window in seconds (default: 30). "
                             "Try 15 for ~2× more labeled windows, 60 for more context.")
    parser.add_argument("--fixed-length", type=int, default=0,
                        help="Number of timepoints after resampling (default: auto = "
                             "window_sec / 30 * 400, preserving ~13 Hz effective rate).")
    parser.add_argument("--with-t0-baseline-features", action="store_true", default=False,
                        help="Add per-user T0 baseline-normalised features (delta_t0, z_t0) "
                             "to every non-T0 window. Uses a two-pass algorithm: T0 windows are "
                             "processed first to compute per-(session,seat) baselines.")
    parser.add_argument("--validate-coverage", action="store_true", default=False,
                        help="Dry-run mode: print a coverage matrix (files available per "
                             "seat × task × modality) for every session and exit without "
                             "writing a pickle.")
    args = parser.parse_args()

    # Auto fixed-length: preserve 13.3 Hz effective rate regardless of window size
    fl = args.fixed_length if args.fixed_length > 0 else int(args.window_sec / 30.0 * 400)

    # Auto output path encodes window size so different runs don't overwrite each other
    out = args.output
    if not out:
        w_tag = f"{int(args.window_sec)}s" if args.window_sec == int(args.window_sec) else f"{args.window_sec}s"
        out = f"data/mumt/dataset_{w_tag}.pkl"

    main(args.dataset_path, out, window_sec=args.window_sec, fixed_length=fl,
         with_t0_baseline_features=args.with_t0_baseline_features,
         validate_coverage=args.validate_coverage)
