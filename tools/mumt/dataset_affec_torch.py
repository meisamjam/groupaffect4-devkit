"""dataset_affec_torch.py

PyTorch Dataset adapter for the AFFEC dataset (individual FER study).

Mirrors the GroupAffectDataset API so that the same models from
model_simple.py (MLPNet / PoolNet / ConvNet) can be trained on AFFEC
with zero changes to the model code.

Key differences vs GroupAffectDataset
--------------------------------------
Dimension      | GroupAffect-4        | AFFEC
---------------|----------------------|-------------------------------
Scale          | 292 windows, 10 grps | 5807 trials, 72 subjects
Trial duration | 7–20 min per window  | ~3.5 s per image stimulus
Seq length     | 400 timesteps        | 200 timesteps (AFFEC_SEQ_LEN)
Gaze channels  | 9 (Tobii Pro G3)     | 6 (Gazepoint GP3 HD)
EDA channels   | 5 (raw+ph+ton+HR+T)  | 4 (raw+ph+ton+T — no HR)
PPG channels   | 3 (IR/Red/Green)     | 0 — ABSENT; zero-padded to 3
IMU channels   | 6 (acc+gyr XYZ)      | 6 (acc+gyr XYZ — same)
VAD dims       | V, A, D              | V, A only  (D always NaN)
BFI scores     | per-item 1–5 scale   | 44-item raw sum (~8–40 range)
Group struct.  | 4 seats per session  | individual (seat always P1)

PPG handling: AFFEC has no PPG.  The dataset returns a (T, 3) all-zero
array for ppg_seq so that PoolNet / ConvNet can use the same architecture
without modification.  The SequenceScaler is NOT fitted for PPG (the
all-zero signal has zero variance).

BFI normalisation: AFFEC BFI-44 scores are stored as summed item scores
(approx range 8–40 per trait for the 8-item Extraversion scale, etc.).
We divide by 40 so that all values fall in ~[0, 1], matching the per-item
1–5 scale convention used for GroupAffect-4 (mean ≈ 3.3 → 3.3/5 = 0.66).

Usage (standalone smoke test)
------------------------------
  python tools/mumt/dataset_affec_torch.py data/mumt/dataset_affec.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import (
    BIG_FIVE_COLS,
    bin_vad_adaptive,
    flatten_features,
    build_summary_key_order,
    make_user2idx,
    task_onehot,
)

# ---------------------------------------------------------------------------
# AFFEC-specific sequence channel definitions
# ---------------------------------------------------------------------------

AFFEC_SEQ_LEN = 200   # resampled timesteps per trial

# Gazepoint GP3 HD: fixation POG (x, y, validity) + best POG (x, y, validity)
AFFEC_GAZE_SEQ_COLS: list[str] = [
    "FPOGX", "FPOGY", "FPOGV",   # fixation POG
    "BPOGX", "BPOGY", "BPOGV",   # best POG (blended estimate)
]

# Gazepoint pupil: left diameter, right diameter, right validity
AFFEC_PUPIL_SEQ_COLS: list[str] = ["LPD", "RPD", "RPV"]

# Shimmer3 GSR+: conductance (calibrated), raw conductance, raw GSR, skin temperature
AFFEC_EDA_SEQ_COLS: list[str] = [
    "GSR_Conductance_cal",   # calibrated conductance (µS)
    "GSR_cal",               # neurokit2 phasic (EDA_Phasic)
    "GSR_raw",               # neurokit2 tonic  (EDA_Tonic)
    "Temperature_cal",       # skin temperature (°C)
]

# PPG: ABSENT in AFFEC.  Zero-padded to 3 channels for architecture compatibility.
AFFEC_PPG_SEQ_COLS: list[str] = []    # no PPG; handled as zero array

# Shimmer3 IMU: accelerometer XYZ + gyroscope XYZ (calibrated)
AFFEC_IMU_SEQ_COLS: list[str] = [
    "Low_Noise_Accelerometer_X_cal",
    "Low_Noise_Accelerometer_Y_cal",
    "Low_Noise_Accelerometer_Z_cal",
    "Gyroscope_X_cal",
    "Gyroscope_Y_cal",
    "Gyroscope_Z_cal",
]

# Modality dims for model_simple.build_simple_model()
AFFEC_MODALITY_DIMS: dict[str, int] = {
    "gaze":  len(AFFEC_GAZE_SEQ_COLS),    # 6
    "pupil": len(AFFEC_PUPIL_SEQ_COLS),   # 3
    "eda":   len(AFFEC_EDA_SEQ_COLS),     # 4
    "ppg":   3,                            # 3 — zero-padded for model compat
    "imu":   len(AFFEC_IMU_SEQ_COLS),     # 6
}

# AFFEC has T0–T3 (4 tasks)
AFFEC_TASK_LABELS: list[str] = ["T0", "T1", "T2", "T3"]
N_AFFEC_TASKS = len(AFFEC_TASK_LABELS)
_AFFEC_TASK_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(AFFEC_TASK_LABELS)}

# BFI normalisation factor (see module docstring)
_BFI_NORM = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_to_array(modal_df: pd.DataFrame, desired_cols: list[str]) -> np.ndarray:
    """Extract *desired_cols* from modal_df, zero-fill missing, return (T, F) float32."""
    if modal_df is None or (hasattr(modal_df, "__len__") and len(modal_df) == 0):
        return np.zeros((AFFEC_SEQ_LEN, len(desired_cols)), dtype=np.float32)
    available = [c for c in desired_cols if c in modal_df.columns]
    df = modal_df[available].copy().fillna(0.0)
    for c in desired_cols:
        if c not in df.columns:
            df[c] = 0.0
    arr = df[desired_cols].values.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _affec_task_onehot(task_label: str) -> np.ndarray:
    vec = np.zeros(N_AFFEC_TASKS, dtype=np.float32)
    idx = _AFFEC_TASK_TO_IDX.get(str(task_label), -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class AFFECDataset(Dataset):
    """PyTorch Dataset for AFFEC — mirrors GroupAffectDataset interface.

    Parameters
    ----------
    df : DataFrame
        Subset of dataset_affec.pkl (train, val, or test).
    vad_thresholds : dict[str, tuple[float,float]] | None
        Per-dimension (low_thr, high_thr) from compute_tertile_thresholds.
        If None, uses fixed ≤3 / 4–6 / ≥7 binning.
    modality_scalers : dict[str, SequenceScaler] | None
        Fitted scalers keyed by modality name.  If None, no scaling.
    augment : bool
        Enable time-warp + Gaussian noise augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vad_thresholds: dict[str, tuple[float, float]] | None = None,
        modality_scalers: dict | None = None,
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vad_thresholds = vad_thresholds
        self.modality_scalers = modality_scalers or {}
        self.augment = augment
        self.user2idx = make_user2idx(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Sequences ──────────────────────────────────────────────────────
        gaze_seq  = _seq_to_array(row["gaze_seq"],  AFFEC_GAZE_SEQ_COLS)
        pupil_seq = _seq_to_array(row["pupil_seq"], AFFEC_PUPIL_SEQ_COLS)
        eda_seq   = _seq_to_array(row["eda_seq"],   AFFEC_EDA_SEQ_COLS)
        # PPG absent — zero array, shape (T, 3) for architecture compatibility
        ppg_seq   = np.zeros((gaze_seq.shape[0], 3), dtype=np.float32)
        imu_seq   = _seq_to_array(row["imu_seq"],   AFFEC_IMU_SEQ_COLS)

        # ── Scale ──────────────────────────────────────────────────────────
        sc = self.modality_scalers
        if "gaze"  in sc: gaze_seq  = sc["gaze"].transform(gaze_seq)
        if "pupil" in sc: pupil_seq = sc["pupil"].transform(pupil_seq)
        if "eda"   in sc: eda_seq   = sc["eda"].transform(eda_seq)
        if "imu"   in sc: imu_seq   = sc["imu"].transform(imu_seq)
        # PPG stays zero; never scaled

        # ── Augmentation (time-warp + noise) ───────────────────────────────
        if self.augment:
            from dataset_affectai import noise_injection, time_warp
            T = gaze_seq.shape[0]
            gaze_seq  = resample(noise_injection(time_warp(gaze_seq)),  T, axis=0).astype(np.float32)
            pupil_seq = resample(noise_injection(time_warp(pupil_seq)), T, axis=0).astype(np.float32)
            eda_seq   = resample(noise_injection(time_warp(eda_seq)),   T, axis=0).astype(np.float32)
            imu_seq   = resample(noise_injection(time_warp(imu_seq)),   T, axis=0).astype(np.float32)

        # ── Personality (BFI normalised to ~[0,1]) ─────────────────────────
        personality = np.array(
            [row[c] for c in BIG_FIVE_COLS], dtype=np.float32
        ) / _BFI_NORM
        personality = np.nan_to_num(personality, nan=0.0)

        # ── Emotion labels (V, A only; D masked to -1) ─────────────────────
        if self.vad_thresholds is not None:
            v_bin = bin_vad_adaptive(row.get("valence",   float("nan")), self.vad_thresholds["valence"])
            a_bin = bin_vad_adaptive(row.get("arousal",   float("nan")), self.vad_thresholds["arousal"])
        else:
            from dataset_affectai import bin_vad
            v_bin = bin_vad(row.get("valence", float("nan")))
            a_bin = bin_vad(row.get("arousal", float("nan")))
        emotion_binned = np.array([v_bin, a_bin, -1], dtype=np.int64)  # D always -1

        # ── User ID ────────────────────────────────────────────────────────
        uid = self.user2idx.get(str(row["subject_id"]), 0)

        # ── Summary features ───────────────────────────────────────────────
        all_feats: dict = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features"]:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                all_feats.update(fd)
        # Build from pre-computed key order (passed in via dataset construction)
        summary = flatten_features(all_feats, key_order=self.summary_key_order)

        # ── Sex (AFFEC: 0=female, 1=male, stored as int 0/1) ───────────────
        sex_val = row.get("sex", -1)
        try:
            sex = int(sex_val)
        except (TypeError, ValueError):
            sex = -1

        # ── Task one-hot ───────────────────────────────────────────────────
        task_vec = _affec_task_onehot(str(row.get("task", "T0")))

        return {
            "gaze_seq":      torch.from_numpy(gaze_seq),
            "pupil_seq":     torch.from_numpy(pupil_seq),
            "eda_seq":       torch.from_numpy(eda_seq),
            "ppg_seq":       torch.from_numpy(ppg_seq),
            "imu_seq":       torch.from_numpy(imu_seq),
            "personality":   torch.from_numpy(personality),
            "emotion_binned":torch.from_numpy(emotion_binned),
            "user_id":       torch.tensor(uid, dtype=torch.long),
            "summary":       torch.from_numpy(summary),
            "sex":           torch.tensor(sex, dtype=torch.long),
            "task_onehot":   torch.from_numpy(task_vec),
        }

    # summary_key_order must be set after construction (avoids double-pass)
    @property
    def summary_key_order(self) -> list[str]:
        return self._summary_key_order

    @summary_key_order.setter
    def summary_key_order(self, v: list[str]) -> None:
        self._summary_key_order = v


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_affec_summary_key_order(df: pd.DataFrame) -> list[str]:
    """Collect all feature keys present in df and return a stable sorted list."""
    keys: set[str] = set()
    for col in ["gaze_features", "pupil_features", "eda_features",
                "ppg_features", "imu_features"]:
        for fd in df[col].dropna():
            if isinstance(fd, dict):
                keys.update(fd.keys())
    return sorted(keys)


def make_affec_datasets(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    vad_thresholds: dict,
    augment: bool = False,
) -> tuple["AFFECDataset", "AFFECDataset", "AFFECDataset"]:
    """Construct train/val/test AFFECDatasets with shared scalers and key order.

    Fits SequenceScalers on train_df only (excluding PPG — all zeros).
    """
    from train_simple import SequenceScaler  # reuse GA4 scaler class

    # Build shared key order from full training set
    key_order = build_affec_summary_key_order(
        pd.concat([train_df, val_df, test_df], ignore_index=True)
    )

    # Fit scalers on train
    MODALITY_COLS_MAP = {
        "gaze":  AFFEC_GAZE_SEQ_COLS,
        "pupil": AFFEC_PUPIL_SEQ_COLS,
        "eda":   AFFEC_EDA_SEQ_COLS,
        "imu":   AFFEC_IMU_SEQ_COLS,
    }
    scalers: dict = {}
    for mod, cols in MODALITY_COLS_MAP.items():
        arrs = [_seq_to_array(row[f"{mod}_seq"], cols) for _, row in train_df.iterrows()]
        scalers[mod] = SequenceScaler().fit(arrs)

    datasets = []
    for df, aug in [(train_df, augment), (val_df, False), (test_df, False)]:
        ds = AFFECDataset(df, vad_thresholds=vad_thresholds,
                          modality_scalers=scalers, augment=aug)
        ds.summary_key_order = key_order
        datasets.append(ds)

    return tuple(datasets)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle

    path = sys.argv[1] if len(sys.argv) > 1 else "data/mumt/dataset_affec.pkl"
    df   = pickle.load(open(path, "rb"))

    key_order = build_affec_summary_key_order(df)
    print(f"Summary key order ({len(key_order)} features): {key_order[:5]} ...")

    from dataset_affectai import make_user2idx
    from baselines_affec import within_subject_split, compute_thresholds

    tr, va, te = within_subject_split(df)
    thresh      = compute_thresholds(tr)
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")

    tr_ds, va_ds, te_ds = make_affec_datasets(tr, va, te, thresh, augment=False)
    sample = tr_ds[0]
    print("\nSample tensor shapes:")
    for k, v in sample.items():
        print(f"  {k:20s}: {tuple(v.shape) if hasattr(v, 'shape') else v}")
    print("\nSmoke test passed.")
