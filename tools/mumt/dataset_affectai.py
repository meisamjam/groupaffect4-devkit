"""dataset_affectai.py

PyTorch Dataset for the GroupAffect-4 adapted MuMTAffect model.

Loads the pickle produced by pickle_generation_affectai.py.
Each sample returns:
  gaze_seq   (T, 9)    – Tobii gaze + direction (resampled to FIXED_LENGTH)
  pupil_seq  (T, 3)    – Tobii pupil L/R + validity
  eda_seq    (T, 5)    – EmotiBit EDA raw + phasic + tonic + HR + temp
  ppg_seq    (T, 3)    – EmotiBit 3-channel raw PPG (IR / Red / Green)
  imu_seq    (T, 6)    – EmotiBit accelerometer XYZ + gyroscope XYZ
  personality (5,)     – Big Five BFI-44 scores
  emotion_binned (3,)  – VAD binned to 3 classes (Low=0, Moderate=1, High=2)
  user_id    int        – integer index for the subject
  summary    (K,)      – trial-level summary features (all modalities)
  sex        int        – 0=female, 1=male (-1 if unknown)
  task_onehot (N_TASKS,) – one-hot encoding of the task label (T0–T4)

VAD binning (Likert 1–9 → 3 classes):
  Fixed:        Low (0): ≤3,  Mid (1): 4–6,  High (2): ≥7
  Data-driven:  thresholds computed as 33rd/67th percentile of training data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

# Sequence columns produced by pickle_generation_affectai.py
# Column names verified against actual zenodo EmotiBit and Tobii data.
# See pickle_generation_affectai.py for full column documentation.
GAZE_SEQ_COLS = [
    "value_0",  "value_1",                                 # gaze x, y (normalised screen)
    "value_14", "value_15", "value_16",                    # gaze dir left  (unit vector xyz)
    "value_17", "value_18", "value_19",                    # gaze dir right (unit vector xyz, NaN if monocular)
    "value_4",                                             # validity flag (1=valid)
]
PUPIL_SEQ_COLS = [
    "value_2",   # left pupil diameter (mm)
    "value_3",   # right pupil diameter (mm, may be NaN)
    "value_4",   # validity flag
]

EDA_SEQ_COLS = [
    "value_3",      # raw EDA conductance (µS) — EmotiBit value_3
    "EDA_Phasic",   # neurokit2 phasic component (SCR)
    "EDA_Tonic",    # neurokit2 tonic component (SCL)
    "value_6",      # firmware heart rate (BPM) — EmotiBit value_6
    "value_11",     # skin temperature (°C) — EmotiBit value_11
]
PPG_SEQ_COLS = [
    "value_0",   # PPG infrared channel (ADC counts)
    "value_1",   # PPG red channel (ADC counts)
    "value_2",   # PPG green channel (ADC counts, highest SNR)
]
IMU_SEQ_COLS = [
    "value_13", "value_14", "value_15",                    # accelerometer XYZ (g)
    "value_16", "value_17", "value_18",                    # gyroscope XYZ (deg/s)
]

BIG_FIVE_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]

# Task labels in study order (T0=baseline, T1–T4=tasks)
TASK_LABELS = ["T0", "T1", "T2", "T3", "T4"]
N_TASKS = len(TASK_LABELS)
_TASK_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TASK_LABELS)}

VAD_BINS = [1, 3, 6, 9]   # boundaries: ≤3→0, 4-6→1, ≥7→2


def bin_vad(value: float) -> int:
    """Map a 1–9 Likert score to a 3-class bin using fixed thresholds (0=Low, 1=Moderate, 2=High).
    
    Uses standard Likert-based thresholds:
      - Low (0):      1–4
      - Moderate (1): 5–6
      - High (2):     7–9
    """
    if np.isnan(value):
        return 1  # default to Moderate
    v = float(value)
    if v <= 4:
        return 0
    if v <= 6:
        return 1
    return 2


def bin_vad_adaptive(value: float, thresholds: tuple[float, float]) -> int:
    """Map a Likert score to a 3-class bin using data-driven tertile thresholds.

    Args:
        value: Raw Likert score.
        thresholds: (t1, t2) where t1=33rd percentile, t2=67th percentile of training data.
                    If either is NaN, falls back to fixed binning.

    Returns:
        0 (Low), 1 (Moderate), or 2 (High).
    """
    if np.isnan(value):
        return 1
    v = float(value)
    t1, t2 = thresholds
    
    # Fallback to fixed binning if thresholds are NaN (can happen with small folds)
    if np.isnan(t1) or np.isnan(t2):
        return bin_vad(value)
    
    if v <= t1:
        return 0
    if v <= t2:
        return 1
    return 2


def task_onehot(task_label: str) -> np.ndarray:
    """Return a float32 one-hot vector of length N_TASKS for *task_label*."""
    vec = np.zeros(N_TASKS, dtype=np.float32)
    idx = _TASK_TO_IDX.get(str(task_label).upper(), 0)
    vec[idx] = 1.0
    return vec


def seq_to_array(modal_df: pd.DataFrame, desired_cols: list[str]) -> np.ndarray:
    """Extract *desired_cols* from *modal_df*, fill NaN/inf with 0, return float32 array (T, F)."""
    available = [c for c in desired_cols if c in modal_df.columns]
    df = modal_df[available].copy().fillna(0)
    # Pad missing columns with zeros
    for c in desired_cols:
        if c not in df.columns:
            df[c] = 0.0
    arr = df[desired_cols].values.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_features(feat_dict: dict, key_order: list[str] | None = None) -> np.ndarray:
    """Flatten a feature dictionary to a 1-D float32 array.

    If *key_order* is provided the output is padded/filtered to exactly those
    keys (in that order), guaranteeing a constant-length vector across all
    samples.  Missing keys are filled with 0.0.
    """
    if key_order is not None:
        vals = []
        for k in key_order:
            v = feat_dict.get(k, 0.0)
            if isinstance(v, (list, np.ndarray)):
                vals.append(float(np.nanmean(v)))
            else:
                vals.append(float(v) if v is not None else 0.0)
        arr = np.array(vals, dtype=np.float32)
    else:
        vals = []
        for key in sorted(feat_dict.keys()):
            v = feat_dict[key]
            if isinstance(v, (list, np.ndarray)):
                vals.append(float(np.nanmean(v)))
            else:
                vals.append(float(v) if not (v is None) else float("nan"))
        arr = np.array(vals, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0)


def build_summary_key_order(df: pd.DataFrame) -> list[str]:
    """Collect the union of all feature keys across all rows, sorted.

    Returns a stable key list that *flatten_features* can use to produce
    equal-length vectors from every sample.  Includes baseline_features
    when present (added by pickle_generation with --with-t0-baseline-features).
    """
    key_sets: set[str] = set()
    for col in ["gaze_features", "pupil_features", "eda_features",
                "ppg_features", "imu_features", "audio_features", "speech_features",
                "sync_features", "baseline_features"]:
        if col not in df.columns:
            continue
        for feat in df[col]:
            if isinstance(feat, dict):
                key_sets.update(feat.keys())
    return sorted(key_sets)


# ---------------------------------------------------------------------------
# T0 baseline normalization utilities (shared by train_simple.py and train_ordinal.py)
# ---------------------------------------------------------------------------

def compute_t0_baselines(
    df: pd.DataFrame,
    summary_key_order: list[str],
) -> dict[tuple, np.ndarray]:
    """Return per-(session_id, seat) mean summary feature vector computed from T0 windows.

    The caller passes the full training DataFrame (which may still contain T0 rows) or
    a pre-filtered T0-only DataFrame — the function filters to task=="T0" internally.
    """
    baselines: dict[tuple, np.ndarray] = {}
    t0_df = df[df["task"] == "T0"]
    for (ses, seat), grp in t0_df.groupby(["session_id", "seat"]):
        vecs = []
        for _, r in grp.iterrows():
            feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features", "audio_features", "speech_features",
                        "baseline_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    feats.update(fd)
            vecs.append(flatten_features(feats, key_order=summary_key_order))
        if vecs:
            baselines[(str(ses), str(seat))] = np.stack(vecs).mean(axis=0)
    return baselines


def apply_t0_baseline(
    summary_list: list[np.ndarray],
    df: pd.DataFrame,
    baselines: dict[tuple, np.ndarray],
) -> list[np.ndarray]:
    """Subtract per-seat T0 mean from each window's summary feature vector."""
    out = []
    df_reset = df.reset_index(drop=True)
    for i, (_, r) in enumerate(df_reset.iterrows()):
        key = (str(r.get("session_id", "")), str(r.get("seat", "")))
        if key in baselines:
            out.append((summary_list[i] - baselines[key]).astype(np.float32))
        else:
            out.append(summary_list[i])
    return out


# ---------------------------------------------------------------------------
# Augmentation helpers (same as original MuMTAffect)
# ---------------------------------------------------------------------------

def time_warp(signal: np.ndarray, warp_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Speed-up / slow-down augmentation."""
    factor = np.random.uniform(*warp_range)
    new_len = max(1, int(signal.shape[0] * factor))
    return resample(signal, new_len, axis=0)


def random_crop(signal: np.ndarray, crop_size: int) -> np.ndarray:
    """Random temporal crop."""
    if signal.shape[0] <= crop_size:
        return signal
    start = np.random.randint(0, signal.shape[0] - crop_size)
    return signal[start : start + crop_size]


def noise_injection(signal: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise."""
    return signal + np.random.normal(0.0, std, signal.shape)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GroupAffectDataset(Dataset):
    """PyTorch Dataset wrapping the GroupAffect-4 MuMTAffect pickle.

    Args:
        df: DataFrame loaded from dataset.pkl.
        user2idx: Mapping from subject_id string to integer index.
        modality_scalers: Optional dict with keys 'gaze', 'pupil', 'eda'
            and fitted sklearn-style scalers.
        augment: If True, apply random time-warp + noise during __getitem__.
        device: Torch device for tensor allocation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        modality_scalers: dict | None = None,
        augment: bool = False,
        device: torch.device | None = None,
        summary_key_order: list[str] | None = None,
        vad_thresholds: dict[str, tuple[float, float]] | None = None,
        session2idx: dict[str, int] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.modality_scalers = modality_scalers or {}
        self.augment = augment
        self.device = device if device is not None else torch.device("cpu")
        # Build once from the full df so every split shares the same key order
        self.summary_key_order = summary_key_order if summary_key_order is not None \
            else build_summary_key_order(df)
        # Data-driven VAD thresholds: {"valence": (t1, t2), ...}; None → fixed bins
        self.vad_thresholds = vad_thresholds
        self.session2idx = session2idx or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- Sequence modalities ---
        gaze_seq = seq_to_array(row["gaze_seq"], GAZE_SEQ_COLS)
        pupil_seq = seq_to_array(row["pupil_seq"], PUPIL_SEQ_COLS)
        eda_seq = seq_to_array(row["eda_seq"], EDA_SEQ_COLS)
        ppg_seq = seq_to_array(row["ppg_seq"], PPG_SEQ_COLS)
        imu_seq = seq_to_array(row["imu_seq"], IMU_SEQ_COLS)

        # --- Optional normalisation ---
        if "gaze" in self.modality_scalers:
            gaze_seq = self.modality_scalers["gaze"].transform(gaze_seq)
        if "pupil" in self.modality_scalers:
            pupil_seq = self.modality_scalers["pupil"].transform(pupil_seq)
        if "eda" in self.modality_scalers:
            eda_seq = self.modality_scalers["eda"].transform(eda_seq)
        if "ppg" in self.modality_scalers:
            ppg_seq = self.modality_scalers["ppg"].transform(ppg_seq)
        if "imu" in self.modality_scalers:
            imu_seq = self.modality_scalers["imu"].transform(imu_seq)

        # --- Augmentation (time-warp + noise, then resample back to original length) ---
        if self.augment:
            orig_len = gaze_seq.shape[0]
            gaze_seq  = resample(noise_injection(time_warp(gaze_seq)),  orig_len, axis=0).astype(np.float32)
            pupil_seq = resample(noise_injection(time_warp(pupil_seq)), orig_len, axis=0).astype(np.float32)
            eda_seq   = resample(noise_injection(time_warp(eda_seq)),   orig_len, axis=0).astype(np.float32)
            ppg_seq   = resample(noise_injection(time_warp(ppg_seq)),   orig_len, axis=0).astype(np.float32)
            imu_seq   = resample(noise_injection(time_warp(imu_seq)),   orig_len, axis=0).astype(np.float32)

        # --- Personality (regression target AND auxiliary input) ---
        personality = np.array(
            [row[c] for c in BIG_FIVE_COLS], dtype=np.float32
        )
        personality = np.nan_to_num(personality, nan=0.0)

        # --- Emotion labels (VAD): fixed or data-driven binning ---
        if self.vad_thresholds is not None:
            valence_bin   = bin_vad_adaptive(row.get("valence",   float("nan")), self.vad_thresholds["valence"])
            arousal_bin   = bin_vad_adaptive(row.get("arousal",   float("nan")), self.vad_thresholds["arousal"])
            dominance_bin = bin_vad_adaptive(row.get("dominance", float("nan")), self.vad_thresholds["dominance"])
        else:
            valence_bin   = bin_vad(row.get("valence",   float("nan")))
            arousal_bin   = bin_vad(row.get("arousal",   float("nan")))
            dominance_bin = bin_vad(row.get("dominance", float("nan")))
        emotion_binned = np.array([valence_bin, arousal_bin, dominance_bin], dtype=np.int64)

        # --- User ID ---
        uid = self.user2idx.get(str(row["subject_id"]), 0)

        # --- Summary features — fixed-length via canonical key order ---
        all_feats: dict = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features", "sync_features", "baseline_features"]:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                all_feats.update(fd)
        summary = flatten_features(all_feats, key_order=self.summary_key_order)

        # --- Sex ---
        sex_raw = str(row.get("sex", "unknown")).lower()
        sex = 0 if sex_raw == "female" else (1 if sex_raw == "male" else -1)

        # --- Task one-hot encoding ---
        task_vec = task_onehot(str(row.get("task", "T0")))

        # --- Session index (for pretraining session classification) ---
        sid = self.session2idx.get(str(row.get("session_id", "")), 0)

        dev = self.device
        return (
            torch.tensor(gaze_seq, device=dev),      # 0
            torch.tensor(pupil_seq, device=dev),     # 1
            torch.tensor(eda_seq, device=dev),       # 2
            torch.tensor(ppg_seq, device=dev),       # 3
            torch.tensor(imu_seq, device=dev),       # 4
            torch.tensor(personality, device=dev),   # 5
            torch.tensor(emotion_binned, dtype=torch.long, device=dev),  # 6
            torch.tensor(uid, dtype=torch.long, device=dev),             # 7
            torch.tensor(summary, device=dev),       # 8
            torch.tensor(sex, dtype=torch.long, device=dev),             # 9
            torch.tensor(task_vec, device=dev),      # 10 — task one-hot (N_TASKS,)
            torch.tensor(sid, dtype=torch.long, device=dev),             # 11 — session idx
        )


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_samples(s1: tuple, s2: tuple, lam: float) -> tuple:
    """Mix two samples; categorical outputs (emotion, sex, task) stay as-is from s1."""
    mixed = []
    for i, (a, b) in enumerate(zip(s1, s2)):
        if i in (6, 9, 10):  # emotion_binned (idx 6), sex (idx 9), task_onehot (idx 10)
            mixed.append(a)
        elif i == 7:  # user_id — mark mixed
            mixed.append(torch.tensor(-1, dtype=a.dtype))
        else:
            if isinstance(a, torch.Tensor):
                mixed.append(lam * a + (1 - lam) * b)
            else:
                mixed.append(lam * a + (1 - lam) * b)
    return tuple(mixed)


# ---------------------------------------------------------------------------
# Utilities for train/test split
# ---------------------------------------------------------------------------

def make_user2idx(df: pd.DataFrame) -> dict[str, int]:
    """Build a {subject_id: int} mapping from the DataFrame."""
    users = sorted(df["subject_id"].unique())
    return {u: i for i, u in enumerate(users)}


def make_session2idx(df: pd.DataFrame) -> dict[str, int]:
    """Build a {session_id: int} mapping from the DataFrame."""
    sessions = sorted(df["session_id"].unique())
    return {s: i for i, s in enumerate(sessions)}


# ---------------------------------------------------------------------------
# Dataset for self-supervised pre-training (pretrain_dataset.pkl)
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Dataset wrapping pretrain_dataset.pkl for Phase 0 self-supervised pretraining.

     Each sample returns a 14-tuple:
      (gaze_seq, pupil_seq, eda_seq, ppg_seq, imu_seq,
       summary, subject_idx, session_idx, task_idx, personality,
         sex_label, age, next_summary, has_next)

    Labels (subject/session/task/sex) are classification targets;
    personality and age are regression targets. ``next_summary`` is a temporal
    self-supervision target (predict the next window summary within the same
    session/seat/task stream). ``has_next`` is 1.0 when target exists, else 0.0.

    Age and sex are joined from participants.tsv at dataset construction time
    so no pickle re-generation is needed.

    Args:
        participants_tsv: Path to BIDS-root participants.tsv.  If None, sex=−1 and age=0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        device: torch.device | None = None,
        augment: bool = False,
        participants_tsv: str | None = None,
        scalers: dict | None = None,
        normalize_targets: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.session2idx = session2idx
        self.summary_key_order = summary_key_order
        self.device = device if device is not None else torch.device("cpu")
        self.augment = augment
        self.scalers = scalers or {}
        self.normalize_targets = normalize_targets

        # Eagerly materialise all sequences to numpy arrays — avoids per-call pandas overhead
        print(f"PretrainDataset: materialising {len(self.df)} sequences …", flush=True)
        self._gaze  = [seq_to_array(r["gaze_seq"],  GAZE_SEQ_COLS)  for _, r in self.df.iterrows()]
        self._pupil = [seq_to_array(r["pupil_seq"], PUPIL_SEQ_COLS) for _, r in self.df.iterrows()]
        self._eda   = [seq_to_array(r["eda_seq"],   EDA_SEQ_COLS)   for _, r in self.df.iterrows()]
        self._ppg   = [seq_to_array(r["ppg_seq"],   PPG_SEQ_COLS)   for _, r in self.df.iterrows()]
        self._imu   = [seq_to_array(r["imu_seq"],   IMU_SEQ_COLS)   for _, r in self.df.iterrows()]
        # Apply scalers in-place after materialisation
        if self.scalers:
            for i in range(len(self.df)):
                if "gaze"  in self.scalers: self._gaze[i]  = self.scalers["gaze"].transform(self._gaze[i])
                if "pupil" in self.scalers: self._pupil[i] = self.scalers["pupil"].transform(self._pupil[i])
                if "eda"   in self.scalers: self._eda[i]   = self.scalers["eda"].transform(self._eda[i])
                if "ppg"   in self.scalers: self._ppg[i]   = self.scalers["ppg"].transform(self._ppg[i])
                if "imu"   in self.scalers: self._imu[i]   = self.scalers["imu"].transform(self._imu[i])
            # Clip to [-10, 10] after scaling to kill any residual outliers
            self._gaze  = [np.clip(a, -10, 10) for a in self._gaze]
            self._pupil = [np.clip(a, -10, 10) for a in self._pupil]
            self._eda   = [np.clip(a, -10, 10) for a in self._eda]
            self._ppg   = [np.clip(a, -10, 10) for a in self._ppg]
            self._imu   = [np.clip(a, -10, 10) for a in self._imu]
        # Summary features
        self._summary = []
        for _, r in self.df.iterrows():
            all_feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features", "baseline_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    all_feats.update(fd)
            self._summary.append(flatten_features(all_feats, key_order=self.summary_key_order))

        # Build per-row index for next-window lookup: same (session, seat, task), window+1
        self._next_idx = np.full(len(self.df), -1, dtype=np.int64)
        key_to_idx: dict[tuple[str, str, str, int], int] = {}
        for i, r in self.df.iterrows():
            key = (
                str(r.get("session_id", "")),
                str(r.get("seat", "")),
                str(r.get("task", "T0")),
                int(r.get("window_index", -1)),
            )
            key_to_idx[key] = int(i)
        for i, r in self.df.iterrows():
            next_key = (
                str(r.get("session_id", "")),
                str(r.get("seat", "")),
                str(r.get("task", "T0")),
                int(r.get("window_index", -1)) + 1,
            )
            self._next_idx[i] = key_to_idx.get(next_key, -1)

        print("PretrainDataset: materialisation complete.", flush=True)

        # Build (session_id_normalised, seat) → {sex: int, age: float} lookup
        # participants.tsv session_id is "20260312_grp-07_run01" (no "ses-" prefix)
        self._demo: dict[tuple[str, str], dict] = {}
        if participants_tsv is not None:
            try:
                pts = pd.read_csv(participants_tsv, sep="\t")
                for _, row in pts.iterrows():
                    raw_ses = str(row.get("session_id", "")).strip()
                    seat    = str(row.get("seat", "")).strip()
                    # normalise: strip leading "ses-" so both formats match
                    ses_key = raw_ses.lstrip("ses-")
                    sex_raw = str(row.get("sex", "unknown")).lower()
                    sex_int = 0 if sex_raw == "female" else (1 if sex_raw == "male" else -1)
                    age_val = float(row.get("age", float("nan")))
                    self._demo[(ses_key, seat)] = {"sex": sex_int, "age": age_val}
            except (FileNotFoundError, ValueError):
                pass

        # Regression target normalisation stats (Phase-0 only)
        self._personality_mean = np.zeros(len(BIG_FIVE_COLS), dtype=np.float32)
        self._personality_std = np.ones(len(BIG_FIVE_COLS), dtype=np.float32)
        self._age_mean = 0.0
        self._age_std = 1.0
        self._summary_mean = np.zeros(len(self.summary_key_order), dtype=np.float32)
        self._summary_std = np.ones(len(self.summary_key_order), dtype=np.float32)

        if self.normalize_targets:
            # Personality statistics from pretrain dataframe
            per_vals = self.df[BIG_FIVE_COLS].to_numpy(dtype=np.float32)
            per_vals = np.nan_to_num(per_vals, nan=0.0)
            self._personality_mean = per_vals.mean(axis=0).astype(np.float32)
            self._personality_std = per_vals.std(axis=0).astype(np.float32)
            self._personality_std = np.clip(self._personality_std, 1e-6, None)

            # Age statistics from joined demographics matched to each row
            age_vals: list[float] = []
            for _, row in self.df.iterrows():
                ses_key = str(row.get("session_id", "")).lstrip("ses-")
                seat = str(row.get("seat", ""))
                demo = self._demo.get((ses_key, seat), {})
                age_val = float(demo.get("age", float("nan")))
                if not np.isnan(age_val):
                    age_vals.append(age_val)
            if age_vals:
                age_arr = np.array(age_vals, dtype=np.float32)
                self._age_mean = float(age_arr.mean())
                self._age_std = float(max(age_arr.std(), 1e-6))

            # Next-summary target statistics
            if self._summary:
                summary_stack = np.vstack(self._summary).astype(np.float32)
                self._summary_mean = summary_stack.mean(axis=0).astype(np.float32)
                self._summary_std = summary_stack.std(axis=0).astype(np.float32)
                self._summary_std = np.clip(self._summary_std, 1e-6, None)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dev = self.device

        gaze_seq  = self._gaze[idx].copy()
        pupil_seq = self._pupil[idx].copy()
        eda_seq   = self._eda[idx].copy()
        ppg_seq   = self._ppg[idx].copy()
        imu_seq   = self._imu[idx].copy()
        summary   = self._summary[idx]

        if self.augment:
            orig_len = gaze_seq.shape[0]
            from scipy.signal import resample as _resample
            gaze_seq  = _resample(noise_injection(time_warp(gaze_seq)),  orig_len, axis=0).astype(np.float32)
            pupil_seq = _resample(noise_injection(time_warp(pupil_seq)), orig_len, axis=0).astype(np.float32)
            eda_seq   = _resample(noise_injection(time_warp(eda_seq)),   orig_len, axis=0).astype(np.float32)
            ppg_seq   = _resample(noise_injection(time_warp(ppg_seq)),   orig_len, axis=0).astype(np.float32)
            imu_seq   = _resample(noise_injection(time_warp(imu_seq)),   orig_len, axis=0).astype(np.float32)

        # Labels
        subject_idx = self.user2idx.get(str(row["subject_id"]), 0)
        session_idx = self.session2idx.get(str(row["session_id"]), 0)
        task_idx    = _TASK_TO_IDX.get(str(row.get("task", "T0")).upper(), 0)

        personality = np.array(
            [row.get(c, 0.0) for c in BIG_FIVE_COLS], dtype=np.float32
        )
        personality = np.nan_to_num(personality, nan=0.0)
        if self.normalize_targets:
            personality = (personality - self._personality_mean) / self._personality_std

        # Demographic labels from participants.tsv lookup
        ses_key = str(row.get("session_id", "")).lstrip("ses-")
        seat    = str(row.get("seat", ""))
        demo    = self._demo.get((ses_key, seat), {})
        sex_int = int(demo.get("sex", -1))
        age_val = float(demo.get("age", 0.0))
        if np.isnan(age_val):
            age_val = 0.0
        if self.normalize_targets:
            age_val = (age_val - self._age_mean) / self._age_std

        next_i = int(self._next_idx[idx])
        if next_i >= 0:
            next_summary = self._summary[next_i]
            has_next = 1.0
        else:
            next_summary = np.zeros_like(summary, dtype=np.float32)
            has_next = 0.0

        if self.normalize_targets:
            next_summary = (next_summary - self._summary_mean) / self._summary_std

        return (
            torch.tensor(gaze_seq,   device=dev),                              # 0
            torch.tensor(pupil_seq,  device=dev),                              # 1
            torch.tensor(eda_seq,    device=dev),                              # 2
            torch.tensor(ppg_seq,    device=dev),                              # 3
            torch.tensor(imu_seq,    device=dev),                              # 4
            torch.tensor(summary,    device=dev),                              # 5
            torch.tensor(subject_idx, dtype=torch.long, device=dev),          # 6
            torch.tensor(session_idx, dtype=torch.long, device=dev),          # 7
            torch.tensor(task_idx,    dtype=torch.long, device=dev),          # 8
            torch.tensor(personality, device=dev),                             # 9
            torch.tensor(sex_int,     dtype=torch.long, device=dev),          # 10 sex label
            torch.tensor(age_val,     dtype=torch.float32, device=dev),       # 11 age (regression)
            torch.tensor(next_summary, dtype=torch.float32, device=dev),      # 12 next-window summary
            torch.tensor(has_next, dtype=torch.float32, device=dev),          # 13 next target valid mask
        )


def split_by_subject(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by subject (no data leakage across splits).

    Each subject appears in exactly one of train/val/test.

    Returns (train_df, val_df, test_df).
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)
    shuffled = list(rng.permutation(subjects))

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_subs = shuffled[:n_test]
    val_subs = shuffled[n_test : n_test + n_val]
    train_subs = shuffled[n_test + n_val :]

    train_df = df[df["subject_id"].isin(train_subs)].copy()
    val_df = df[df["subject_id"].isin(val_subs)].copy()
    test_df = df[df["subject_id"].isin(test_subs)].copy()

    return train_df, val_df, test_df


def split_by_subject_stratified(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset per-subject, so each subject has data in all three splits.

    For each subject, randomly splits their rows into train/val/test.
    Then concatenates across all subjects.

    This ensures:
    - Each subject is represented in train, val, and test
    - No data leakage (per-subject rows are not duplicated)
    - More balanced class distributions across splits

    Args:
        df: DataFrame with 'subject_id' column.
        test_frac: Fraction of each subject's data for test (default 0.15).
        val_frac: Fraction of each subject's data for val (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) with overlapping subjects.
    """
    rng = np.random.default_rng(seed)
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for subject in sorted(df["subject_id"].unique()):
        subj_df = df[df["subject_id"] == subject].copy()
        n = len(subj_df)

        if n < 3:
            # Not enough data for all three splits; put all in train
            train_dfs.append(subj_df)
            continue

        # Shuffle subject's rows
        indices = np.arange(n)
        rng.shuffle(indices)
        shuffled_df = subj_df.iloc[indices].reset_index(drop=True)

        # Compute split boundaries
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_test - n_val

        # Ensure at least 1 sample in each split (if possible)
        if n_train < 1:
            n_train = 1
            n_val = max(1, n - n_test - n_train)
            n_test = n - n_train - n_val

        # Split rows
        test_dfs.append(shuffled_df.iloc[:n_test])
        val_dfs.append(shuffled_df.iloc[n_test : n_test + n_val])
        train_dfs.append(shuffled_df.iloc[n_test + n_val :])

    # Concatenate across all subjects
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    return train_df, val_df, test_df
