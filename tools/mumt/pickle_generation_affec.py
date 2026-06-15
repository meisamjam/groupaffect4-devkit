"""pickle_generation_affec.py

Feature extraction for the AFFEC dataset (Zenodo DOI 10.5281/zenodo.14794876).

Produces data/mumt/dataset_affec.pkl — one row per labelled trial,
with the same schema as dataset_15s.pkl so the existing baselines.py and
train_simple.py can consume it without modification.

Mapping to GroupAffect-4 schema
--------------------------------
session_id    ← participant ID   (e.g. 'sub-acl')
subject_id    ← participant ID   (same)
seat          ← 'P1'             (individual, no group seating)
task          ← 'T0' .. 'T3'    (run-0 → T0, run-1 → T1, run-2 → T2, run-3 → T3)
valence       ← f_emotion_v      (felt valence, 1–9 Likert)
arousal       ← f_emotion_a      (felt arousal, 1–9 Likert)
dominance     ← NaN              (not administered in AFFEC)
bfi44_e/a/c/n/o ← E, A, C, N, O from participants.tsv (raw BFI scores)
gaze_features ← Gazepoint FPOG*/LPOG*/RPOG* summary stats
pupil_features← Gazepoint LPUPILD/RPUPILD summary stats
eda_features  ← GSR_Conductance_cal via cvxEDA (phasic/tonic/SCR/temp)
ppg_features  ← {} (no PPG in AFFEC — SVM fills with 0)
imu_features  ← Low_Noise_Accelerometer_X/Y/Z from GSR file (partial)
gaze_seq      ← 200-point resampled gaze stream  (T×6 DataFrame)
pupil_seq     ← 200-point resampled pupil stream (T×3 DataFrame)
eda_seq       ← 200-point resampled EDA stream   (T×3 DataFrame)
ppg_seq       ← empty DataFrame
imu_seq       ← 200-point resampled accel stream (T×3 DataFrame)

AFFEC data lives in three zip archives extracted under data/affec/raw/:
  core.zip  → sub-*/beh/*_beh.tsv, sub-*/*_events.tsv, participants.tsv
  gsr.zip   → gsr/sub-*/beh/*_recording-gsr_physio.tsv.gz + .json
  gaze.zip  → gaze/sub-*/beh/*_recording-gaze_physio.tsv.gz + .json
  pupil.zip → pupil/sub-*/beh/*_recording-pupil_physio.tsv.gz + .json

NOTE: The pipeline treats each participant independently — no group-level
      features are used, matching the individual-level GroupAffect-4 baseline.

Usage:
  python tools/mumt/pickle_generation_affec.py \\
      --data-root data/affec/raw --output data/mumt/dataset_affec.pkl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import neurokit2 as nk
    _NK_OK = True
except ImportError:
    _NK_OK = False
    warnings.warn("neurokit2 not found — EDA decomposition disabled; using raw EDA stats only.")

try:
    from scipy.signal import resample as scipy_resample
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_LENGTH = 200   # resampled timepoints per window (matches GroupAffect-4)
GSR_RATE_HZ  = 50.0
GAZE_RATE_HZ = 62.0
PUPIL_RATE_HZ = 149.0
GSR_EXTRA_SEC = 10.0   # extra GSR window after video end (capture delayed SCR)

RUN_TO_TASK = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}  # run-0 → T0 (train) … run-3 → T3 (test)

# BFI column mapping in participants.tsv
BFI_MAP = {"bfi44_e": "E", "bfi44_a": "A", "bfi44_c": "C", "bfi44_n": "N", "bfi44_o": "O"}

# ---------------------------------------------------------------------------
# Zip helpers
# ---------------------------------------------------------------------------

class _ZipBundle:
    """Lazy access to one or more zip archives (core, gsr, gaze, pupil)."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root
        self._zips: dict[str, zipfile.ZipFile | None] = {}
        for name in ("core", "gsr", "gaze", "pupil"):
            p = data_root / f"{name}.zip"
            self._zips[name] = zipfile.ZipFile(p) if p.exists() else None

    def _namelist(self, zname: str) -> list[str]:
        z = self._zips.get(zname)
        return z.namelist() if z else []

    def open(self, zname: str, path: str):
        """Open a file path inside zip *zname*; returns a binary stream."""
        z = self._zips.get(zname)
        if z is None:
            raise FileNotFoundError(f"{zname}.zip not found in {self.root}")
        return z.open(path)

    def has(self, zname: str, path: str) -> bool:
        z = self._zips.get(zname)
        if z is None:
            return False
        try:
            z.getinfo(path)
            return True
        except KeyError:
            return False

    def subjects(self) -> list[str]:
        names = self._namelist("core")
        return sorted({n.split("/")[0] for n in names if n.startswith("sub-")})

    def runs_for(self, subject: str) -> list[int]:
        names = self._namelist("core")
        runs = set()
        for n in names:
            if n.startswith(f"{subject}/") and n.endswith("_events.tsv"):
                try:
                    run = int(n.split("run-")[1].split("_")[0])
                    runs.add(run)
                except (IndexError, ValueError):
                    pass
        return sorted(runs)


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_tsv_gz(zb: _ZipBundle, zname: str, path: str,
                 needed: set[str] | None = None) -> pd.DataFrame | None:
    """Read a gzipped TSV from a zip archive.

    Header column names come from the paired JSON sidecar (same path, .json).
    Returns None if the file is absent or unreadable.
    """
    if not zb.has(zname, path):
        return None
    json_path = path.replace(".tsv.gz", ".json")
    cols: list[str] = []
    if zb.has(zname, json_path):
        with zb.open(zname, json_path) as jf:
            try:
                meta = json.load(jf)
                cols = meta.get("Columns", [])
            except Exception:
                cols = []

    try:
        with zb.open(zname, path) as raw:
            with gzip.open(raw, "rt", encoding="utf-8", errors="ignore") as gf:
                df = pd.read_csv(
                    gf, sep="\t", header=None,
                    names=cols if cols else None,
                    on_bad_lines="skip",
                )
    except Exception as e:
        log.debug("Failed reading %s/%s: %s", zname, path, e)
        return None

    if df.empty:
        return None

    # Normalise time column
    time_col = next((c for c in df.columns if str(c).lower() in ("onset", "time", "timestamp")), None)
    if time_col is None:
        return None
    df = df.rename(columns={time_col: "onset"})
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    if df.empty:
        return None
    df["onset"] -= df["onset"].iloc[0]   # zero-start

    if needed:
        keep = [c for c in df.columns if c in needed or c == "onset"]
        df = df[keep]

    for c in df.columns:
        if c != "onset":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _slice(df: pd.DataFrame | None, start: float, end: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[(df["onset"] >= start) & (df["onset"] <= end)].copy().reset_index(drop=True)


def _resample_seq(df: pd.DataFrame, cols: list[str], n: int = FIXED_LENGTH) -> pd.DataFrame:
    """Resample a windowed DataFrame to *n* timepoints.

    Missing values are forward-filled then zero-filled before resampling.
    Returns an (n, len(cols)) DataFrame.
    """
    out = {}
    for c in cols:
        if c not in df.columns:
            out[c] = np.zeros(n, dtype=np.float32)
            continue
        arr = pd.to_numeric(df[c], errors="coerce").ffill().fillna(0.0).to_numpy(np.float32)
        if len(arr) == 0:
            out[c] = np.zeros(n, dtype=np.float32)
        elif len(arr) == n:
            out[c] = arr
        elif _SCIPY_OK:
            out[c] = scipy_resample(arr, n).astype(np.float32)
        else:
            idx = np.linspace(0, len(arr) - 1, n)
            out[c] = np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Feature extractors — mirror GroupAffect-4 feature keys exactly
# ---------------------------------------------------------------------------

def _gaze_features(gaze: pd.DataFrame) -> dict[str, float]:
    """6 gaze summary features matching GroupAffect-4 gaze_features keys."""
    feats: dict[str, float] = {}
    if gaze.empty:
        for k in ("gaze_x_mean", "gaze_x_std", "gaze_y_mean", "gaze_y_std",
                  "validity_rate", "saccade_proxy"):
            feats[k] = 0.0
        return feats

    x = pd.to_numeric(gaze.get("FPOGX", gaze.get("BPOGX", pd.Series())), errors="coerce").ffill().fillna(0.5)
    y = pd.to_numeric(gaze.get("FPOGY", gaze.get("BPOGY", pd.Series())), errors="coerce").ffill().fillna(0.5)
    v = pd.to_numeric(gaze.get("FPOGV", gaze.get("BPOGV", pd.Series())), errors="coerce").fillna(0.0)
    feats["gaze_x_mean"] = float(x.mean())
    feats["gaze_x_std"]  = float(x.std(ddof=0))
    feats["gaze_y_mean"] = float(y.mean())
    feats["gaze_y_std"]  = float(y.std(ddof=0))
    feats["validity_rate"] = float(v.mean())
    if len(x) > 1:
        dx = x.diff().abs()
        dy = y.diff().abs()
        feats["saccade_proxy"] = float(np.sqrt(dx**2 + dy**2).mean())
    else:
        feats["saccade_proxy"] = 0.0
    return feats


def _pupil_features(pupil: pd.DataFrame) -> dict[str, float]:
    """7 pupil summary features matching GroupAffect-4 pupil_features keys.

    Gazepoint uses LPD/RPD (pixel diameter) not LPUPILD/RPUPILD (always zero).
    Blink rate approximated from RPV validity flag (0 = invalid / blink).
    """
    feats: dict[str, float] = {}
    keys = ("pupil_left_mean", "pupil_left_std", "pupil_right_mean",
            "pupil_right_std", "pupil_avg_mean", "pupil_avg_std", "blink_rate")
    for k in keys:
        feats[k] = 0.0
    if pupil.empty:
        return feats

    lp = pd.to_numeric(pupil.get("LPD", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan)
    rp = pd.to_numeric(pupil.get("RPD", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan)
    rv = pd.to_numeric(pupil.get("RPV", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    if lp.notna().any():
        lp_v = lp.dropna()
        feats["pupil_left_mean"] = float(lp_v.mean())
        feats["pupil_left_std"]  = float(lp_v.std(ddof=0))
    if rp.notna().any():
        rp_v = rp.dropna()
        feats["pupil_right_mean"] = float(rp_v.mean())
        feats["pupil_right_std"]  = float(rp_v.std(ddof=0))

    avg = pd.concat([lp, rp], axis=1).mean(axis=1).dropna()
    if not avg.empty:
        feats["pupil_avg_mean"] = float(avg.mean())
        feats["pupil_avg_std"]  = float(avg.std(ddof=0))

    feats["blink_rate"] = float(1.0 - rv.clip(0, 1).mean())
    return feats


def _eda_features(gsr: pd.DataFrame, sr: float = GSR_RATE_HZ) -> dict[str, float]:
    """13 EDA summary features matching GroupAffect-4 eda_features keys."""
    feats: dict[str, float] = {
        "eda_phasic_mean": 0.0, "eda_phasic_std": 0.0,
        "eda_tonic_mean":  0.0, "eda_tonic_std":  0.0,
        "scr_peak_count":  0.0, "scr_amplitude_mean": 0.0,
        "eda_raw_mean":    0.0, "eda_raw_std":    0.0,
        "eda_range":       0.0,
        "temp_mean":       0.0, "temp_std":        0.0,
        "acc_mag_mean":    0.0, "acc_mag_std":     0.0,
    }
    if gsr.empty:
        return feats

    # Raw EDA conductance
    raw_col = next((c for c in ("GSR_Conductance_cal", "GSR_cal", "GSR_raw")
                    if c in gsr.columns), None)
    if raw_col is None:
        return feats

    eda_raw = pd.to_numeric(gsr[raw_col], errors="coerce").ffill().fillna(0.0).to_numpy(np.float64)
    feats["eda_raw_mean"] = float(np.mean(eda_raw))
    feats["eda_raw_std"]  = float(np.std(eda_raw))
    feats["eda_range"]    = float(np.max(eda_raw) - np.min(eda_raw))

    if _NK_OK and len(eda_raw) >= int(sr * 2):
        try:
            signals, _ = nk.eda_process(eda_raw, sampling_rate=int(sr))
            phasic = signals["EDA_Phasic"].to_numpy()
            tonic  = signals["EDA_Tonic"].to_numpy()
            feats["eda_phasic_mean"] = float(np.nanmean(phasic))
            feats["eda_phasic_std"]  = float(np.nanstd(phasic))
            feats["eda_tonic_mean"]  = float(np.nanmean(tonic))
            feats["eda_tonic_std"]   = float(np.nanstd(tonic))
            peaks = signals.get("SCR_Peaks")
            if peaks is not None:
                peak_idx = np.where(peaks.to_numpy() == 1)[0]
                feats["scr_peak_count"] = float(len(peak_idx))
                if len(peak_idx) > 0 and "SCR_Amplitude" in signals.columns:
                    feats["scr_amplitude_mean"] = float(np.nanmean(signals["SCR_Amplitude"].iloc[peak_idx]))
        except Exception:
            pass
    else:
        # Fallback: approximate phasic via high-pass
        if len(eda_raw) >= 4:
            smooth = np.convolve(eda_raw, np.ones(max(1, int(sr))) / max(1, int(sr)), "same")
            phasic = eda_raw - smooth
            feats["eda_phasic_mean"] = float(np.mean(phasic))
            feats["eda_phasic_std"]  = float(np.std(phasic))
            feats["eda_tonic_mean"]  = float(np.mean(smooth))
            feats["eda_tonic_std"]   = float(np.std(smooth))

    # Skin temperature
    if "Temperature_cal" in gsr.columns:
        temp = pd.to_numeric(gsr["Temperature_cal"], errors="coerce").dropna()
        if not temp.empty:
            feats["temp_mean"] = float(temp.mean())
            feats["temp_std"]  = float(temp.std(ddof=0))

    # Accelerometer magnitude (partial IMU)
    axc = [c for c in ("Low_Noise_Accelerometer_X_cal",
                        "Low_Noise_Accelerometer_Y_cal",
                        "Low_Noise_Accelerometer_Z_cal") if c in gsr.columns]
    if axc:
        acc = np.stack([pd.to_numeric(gsr[c], errors="coerce").ffill().fillna(0.0).to_numpy()
                        for c in axc], axis=1)
        mag = np.linalg.norm(acc, axis=1)
        feats["acc_mag_mean"] = float(np.mean(mag))
        feats["acc_mag_std"]  = float(np.std(mag))

    return feats


def _imu_features(gsr: pd.DataFrame) -> dict[str, float]:
    """16 IMU features: 3-axis accelerometer + 3-axis gyroscope (both in Shimmer GSR file)."""
    keys = ("acc_x_mean", "acc_x_std", "acc_y_mean", "acc_y_std",
            "acc_z_mean", "acc_z_std", "acc_mag_mean", "acc_mag_std",
            "gyr_x_mean", "gyr_x_std", "gyr_y_mean", "gyr_y_std",
            "gyr_z_mean", "gyr_z_std", "gyr_mag_mean", "gyr_mag_std")
    feats = {k: 0.0 for k in keys}
    if gsr.empty:
        return feats
    sensor_map = [
        ("acc_x", "Low_Noise_Accelerometer_X_cal"),
        ("acc_y", "Low_Noise_Accelerometer_Y_cal"),
        ("acc_z", "Low_Noise_Accelerometer_Z_cal"),
        ("gyr_x", "Gyroscope_X_cal"),
        ("gyr_y", "Gyroscope_Y_cal"),
        ("gyr_z", "Gyroscope_Z_cal"),
    ]
    acc_arrs: list[np.ndarray] = []
    gyr_arrs: list[np.ndarray] = []
    for prefix, col in sensor_map:
        if col not in gsr.columns:
            continue
        arr = pd.to_numeric(gsr[col], errors="coerce").ffill().fillna(0.0).to_numpy(np.float32)
        feats[f"{prefix}_mean"] = float(arr.mean())
        feats[f"{prefix}_std"]  = float(arr.std())
        if prefix.startswith("acc"):
            acc_arrs.append(arr)
        else:
            gyr_arrs.append(arr)
    if len(acc_arrs) == 3:
        mag = np.sqrt(sum(a**2 for a in acc_arrs))
        feats["acc_mag_mean"] = float(mag.mean())
        feats["acc_mag_std"]  = float(mag.std())
    if len(gyr_arrs) == 3:
        mag = np.sqrt(sum(a**2 for a in gyr_arrs))
        feats["gyr_mag_mean"] = float(mag.mean())
        feats["gyr_mag_std"]  = float(mag.std())
    return feats


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------

def _extract_trial(
    subj: str,
    run: int,
    label_row: pd.Series,
    events_df: pd.DataFrame,
    gsr_df: pd.DataFrame | None,
    gaze_df: pd.DataFrame | None,
    pupil_df: pd.DataFrame | None,
    bfi: dict[str, float],
    age: float,
    sex: int,
) -> dict | None:
    """Extract one trial row.  Returns None if video timing cannot be found."""
    stim = label_row.get("stim_file", "")
    video_ev = events_df[
        (events_df.get("flag", pd.Series()).astype(str).str.lower() == "video") &
        (events_df.get("stim_file", pd.Series()).astype(str) == str(stim))
    ]
    if video_ev.empty:
        # Fallback: any event matching the stimulus
        video_ev = events_df[events_df.get("stim_file", pd.Series()).astype(str) == str(stim)]
    if video_ev.empty:
        return None

    onset    = float(pd.to_numeric(video_ev["onset"], errors="coerce").dropna().iloc[0])
    dur_col  = video_ev["duration"] if "duration" in video_ev.columns else pd.Series([5.0])
    duration = float(pd.to_numeric(dur_col, errors="coerce").dropna().iloc[0]) if len(dur_col) > 0 else 5.0
    offset   = onset + max(duration, 1.0)

    win_gaze   = _slice(gaze_df,  onset, offset)
    win_pupil  = _slice(pupil_df, onset, offset)
    win_gsr    = _slice(gsr_df,   onset, offset + GSR_EXTRA_SEC)

    # ── Feature extraction ──────────────────────────────────────────────────
    merged_gaze  = win_gaze  if not win_gaze.empty  else pd.DataFrame()
    merged_pupil = win_pupil if not win_pupil.empty else pd.DataFrame()

    gaze_feats  = _gaze_features(merged_gaze)
    pupil_feats = _pupil_features(merged_pupil)
    eda_feats   = _eda_features(win_gsr)
    imu_feats   = _imu_features(win_gsr)

    # ── Sequences (200-point resampled) ─────────────────────────────────────
    gaze_seq  = _resample_seq(merged_gaze,
                              ["FPOGX", "FPOGY", "FPOGV",
                               "BPOGX", "BPOGY", "BPOGV"],
                              FIXED_LENGTH)
    pupil_seq = _resample_seq(merged_pupil,
                              ["LPD", "RPD", "RPV"],
                              FIXED_LENGTH)

    eda_seq_cols = [c for c in ("GSR_Conductance_cal", "GSR_cal", "GSR_raw",
                                "Temperature_cal") if c in (win_gsr.columns if win_gsr is not None and not win_gsr.empty else [])]
    eda_seq = _resample_seq(win_gsr if win_gsr is not None else pd.DataFrame(),
                            eda_seq_cols, FIXED_LENGTH) if eda_seq_cols else pd.DataFrame()

    imu_seq_cols = [c for c in ("Low_Noise_Accelerometer_X_cal",
                                "Low_Noise_Accelerometer_Y_cal",
                                "Low_Noise_Accelerometer_Z_cal",
                                "Gyroscope_X_cal", "Gyroscope_Y_cal", "Gyroscope_Z_cal")
                    if c in (win_gsr.columns if win_gsr is not None and not win_gsr.empty else [])]
    imu_seq = _resample_seq(win_gsr if win_gsr is not None else pd.DataFrame(),
                            imu_seq_cols, FIXED_LENGTH) if imu_seq_cols else pd.DataFrame()

    return {
        "session_id":    subj,
        "subject_id":    subj,
        "seat":          "P1",
        "task":          RUN_TO_TASK.get(run, f"run{run}"),
        "task_start_lsl": float(onset),
        "task_end_lsl":  float(offset),
        "vad_timestamp_lsl": float(offset),
        "valence":   float(label_row.get("f_emotion_v", float("nan"))),
        "arousal":   float(label_row.get("f_emotion_a", float("nan"))),
        "dominance": float("nan"),   # not collected in AFFEC
        # Optional perceived labels stored as extra columns (not used by baseline)
        "p_valence": float(label_row.get("p_emotion_v", float("nan"))),
        "p_arousal": float(label_row.get("p_emotion_a", float("nan"))),
        "trial_type": str(label_row.get("trial_type", "")),
        "sex":  sex,
        "age":  int(age) if not np.isnan(age) else -1,
        **bfi,
        "gaze_features":  gaze_feats,
        "pupil_features": pupil_feats,
        "eda_features":   eda_feats,
        "ppg_features":   {},   # not available in AFFEC
        "imu_features":   imu_feats,
        "gaze_seq":  gaze_seq,
        "pupil_seq": pupil_seq,
        "eda_seq":   eda_seq,
        "ppg_seq":   pd.DataFrame(),
        "imu_seq":   imu_seq,
    }


# ---------------------------------------------------------------------------
# Modality path resolver (handles gsr/ and gaze/ sub-folder structure)
# ---------------------------------------------------------------------------

def _find_path(zb: _ZipBundle, zname: str, subj: str, run: int, recording: str,
               ext: str = ".tsv.gz") -> str | None:
    """Try both beh/-in-subject and separate-zip-root layouts."""
    for base in (
        f"{subj}/beh/{subj}_task-fer_run-{run}_recording-{recording}_physio",
        f"{zname}/{subj}/beh/{subj}_task-fer_run-{run}_recording-{recording}_physio",
        f"data/{subj}/beh/{subj}_task-fer_run-{run}_recording-{recording}_physio",
    ):
        p = base + ext
        if zb.has(zname, p):
            return p
    return None


def _load_modality(zb: _ZipBundle, zname: str, subj: str, run: int,
                   recording: str, needed: set[str]) -> pd.DataFrame | None:
    path = _find_path(zb, zname, subj, run, recording)
    if path is None:
        return None
    return _read_tsv_gz(zb, zname, path, needed=needed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(data_root: Path, max_subjects: int | None = None) -> pd.DataFrame:
    """Extract all AFFEC trials into a DataFrame compatible with GroupAffect-4."""
    zb = _ZipBundle(data_root)
    subjects = zb.subjects()
    if not subjects:
        raise RuntimeError(f"No sub-* folders found in core.zip under {data_root}")
    if max_subjects:
        subjects = subjects[:max_subjects]

    # Participants metadata
    with zb.open("core", "participants.tsv") as f:
        participants = pd.read_csv(f, sep="\t")
    participants.columns = [c.strip() for c in participants.columns]
    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()

    NEEDED_GSR   = {"onset", "GSR_raw", "GSR_cal", "GSR_Conductance_cal",
                    "Temperature_cal",
                    "Low_Noise_Accelerometer_X_cal",
                    "Low_Noise_Accelerometer_Y_cal",
                    "Low_Noise_Accelerometer_Z_cal",
                    "Gyroscope_X_cal", "Gyroscope_Y_cal", "Gyroscope_Z_cal"}
    NEEDED_GAZE  = {"onset", "FPOGX", "FPOGY", "FPOGV", "FPOGS", "FPOGD",
                    "BPOGX", "BPOGY", "BPOGV",
                    "LPOGX", "LPOGY", "LPOGV", "RPOGX", "RPOGY", "RPOGV",
                    "LPUPILD", "RPUPILD", "LPV", "LPUPILV", "RPV", "RPUPILV"}
    NEEDED_PUPIL = {"onset", "LPCX", "LPCY", "LPD", "LPV", "LPS",
                    "RPCX", "RPCY", "RPD", "RPV", "RPS",
                    "LPUPILD", "RPUPILD", "LPUPILV", "RPUPILV"}

    rows: list[dict] = []
    for subj in subjects:
        pdata = participants[participants["participant_id"] == subj]
        if pdata.empty:
            log.warning("No participant record for %s — skipping.", subj)
            continue
        pr = pdata.iloc[0]
        age = float(pr.get("age", float("nan"))) if pd.notna(pr.get("age", float("nan"))) else float("nan")
        gender_raw = str(pr.get("gender", "")).strip().lower()
        sex = 1 if gender_raw.startswith("m") else (0 if gender_raw.startswith("f") else -1)
        bfi = {
            "bfi44_e": float(pr.get("E", float("nan"))),
            "bfi44_a": float(pr.get("A", float("nan"))),
            "bfi44_c": float(pr.get("C", float("nan"))),
            "bfi44_n": float(pr.get("N", float("nan"))),
            "bfi44_o": float(pr.get("O", float("nan"))),
        }

        for run in zb.runs_for(subj):
            task = RUN_TO_TASK.get(run, f"run{run}")

            # Labels
            beh_path = f"{subj}/beh/{subj}_task-fer_run-{run}_beh.tsv"
            if not zb.has("core", beh_path):
                continue
            with zb.open("core", beh_path) as f:
                labels_df = pd.read_csv(f, sep="\t")

            # Events
            ev_path = f"{subj}/{subj}_task-fer_run-{run}_events.tsv"
            if not zb.has("core", ev_path):
                continue
            with zb.open("core", ev_path) as f:
                events_df = pd.read_csv(f, sep="\t")
            events_df["onset"] = pd.to_numeric(events_df.get("onset", pd.Series()), errors="coerce")
            events_df = events_df.dropna(subset=["onset"])

            # Physio (lazy — only load if zip available)
            gsr_df  = _load_modality(zb, "gsr",   subj, run, "gsr",   NEEDED_GSR)
            gaze_df = _load_modality(zb, "gaze",  subj, run, "gaze",  NEEDED_GAZE)
            pupil_df = _load_modality(zb, "pupil", subj, run, "pupil", NEEDED_PUPIL)

            # If gaze file has pupil columns (merged format), no separate pupil needed
            if gaze_df is not None and "LPUPILD" in gaze_df.columns:
                pupil_df = gaze_df  # gaze file already contains pupil data

            for _, label_row in labels_df.iterrows():
                rec = _extract_trial(
                    subj, run, label_row, events_df,
                    gsr_df, gaze_df, pupil_df, bfi, age, sex,
                )
                if rec is not None:
                    rows.append(rec)

        log.info("%s: %d trials so far (total %d)", subj, len(rows), len(rows))

    if not rows:
        raise RuntimeError("No trials extracted — check that gsr.zip/gaze.zip/pupil.zip are present.")

    df = pd.DataFrame(rows)
    log.info("Total: %d trials, %d subjects", len(df), df["subject_id"].nunique())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate AFFEC pickle for GroupAffect-4 SVM pipeline.")
    ap.add_argument("--data-root", default="data/affec/raw", type=Path,
                    help="Directory containing core.zip, gsr.zip, gaze.zip, pupil.zip")
    ap.add_argument("--output", default="data/mumt/dataset_affec.pkl", type=Path)
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="Limit to N subjects (for quick testing)")
    args = ap.parse_args()

    log.info("Building AFFEC dataset from %s", args.data_root)
    df = build_dataset(args.data_root, args.max_subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output)
    log.info("Saved %d rows → %s", len(df), args.output)

    # Quick stats
    for dim in ("valence", "arousal"):
        vals = df[dim].dropna()
        log.info("%s: N=%d, mean=%.2f, std=%.2f, range=[%.0f,%.0f]",
                 dim, len(vals), vals.mean(), vals.std(), vals.min(), vals.max())
    splits = df.groupby("task").size()
    log.info("Task split:\n%s", splits.to_string())


if __name__ == "__main__":
    main()
