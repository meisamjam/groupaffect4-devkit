"""dataset_augmented.py

PyTorch Dataset that combines the original labeled windows (dataset.pkl) with
soft-labeled augmented windows (augmented_pool.pkl) produced by label_augmentation.py.

Key design:
  - Original self-report windows: hard one-hot labels, instance weight = 1.0
  - Augmented windows: soft [p_Low, p_Mid, p_High] labels, instance weight < 1.0
  - Both types share the same 5-modality input format as GroupAffectDataset
  - Leakage guarantee: augmented labels were generated without using the target
    person's own physiology (see LABEL_AUGMENTATION.md §1.3)

Usage in train_affectai.py (Phase 2 / Phase 3):

    from dataset_augmented import build_combined_loader

    train_loader = build_combined_loader(
        labeled_df=train_df,
        augmented_df=augmented_df,
        scalers=scalers,
        batch_size=64,
        augmented_fraction=0.5,   # 50% of each batch = augmented
    )

The loss function must accept soft labels:

    def affectai_loss_soft(logits, soft_targets, weights):
        # soft_targets: (B, 3)  weights: (B,)
        log_p = F.log_softmax(logits, dim=-1)
        ce    = -(soft_targets * log_p).sum(dim=-1)   # (B,)
        return (weights * ce).mean()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from dataset_affectai import (
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    N_TASKS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    TASK_LABELS,
    _TASK_TO_IDX,
    flatten_features,
    seq_to_array,
    task_onehot,
)

DIMS = ("valence", "arousal", "dominance")
_SUMMARY_KEY_ORDER: list[str] | None = None   # populated lazily on first use


def _get_summary_key_order(df: pd.DataFrame) -> list[str]:
    """Return a stable sorted key list from feature dicts in *df*."""
    global _SUMMARY_KEY_ORDER
    if _SUMMARY_KEY_ORDER is None:
        keys: set[str] = set()
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features"]:
            if col in df.columns:
                for d in df[col]:
                    if isinstance(d, dict):
                        keys.update(d.keys())
        _SUMMARY_KEY_ORDER = sorted(keys)
    return _SUMMARY_KEY_ORDER


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Hard-label sample (original self-report)
# ---------------------------------------------------------------------------

def _labeled_sample(row: pd.Series, summary_key_order: list[str]) -> dict:
    """Convert one labeled row (from dataset.pkl) to a sample dict.

    Returns:
        gaze, pupil, eda, ppg, imu : (T, F) float32 tensors
        personality                : (5,) float32 tensor
        vad_hard                   : (3,) int64 tensor [bin_v, bin_a, bin_d]
        vad_soft                   : (3, 3) float32 tensor (one-hot rows)
        vad_weight                 : (3,) float32 tensor (all 1.0)
        summary                    : (K,) float32 tensor
        uid                        : int
        sex                        : int
        task_onehot_vec            : (N_TASKS,) float32 tensor
        is_hard                    : True
    """
    from dataset_affectai import bin_vad

    gaze  = _to_tensor(seq_to_array(row["gaze_seq"],  GAZE_SEQ_COLS))
    pupil = _to_tensor(seq_to_array(row["pupil_seq"], PUPIL_SEQ_COLS))
    eda   = _to_tensor(seq_to_array(row["eda_seq"],   EDA_SEQ_COLS))
    ppg   = _to_tensor(seq_to_array(row["ppg_seq"],   PPG_SEQ_COLS))
    imu   = _to_tensor(seq_to_array(row["imu_seq"],   IMU_SEQ_COLS))

    # Personality
    pers = np.array([float(row.get(c, 0.0)) for c in BIG_FIVE_COLS], dtype=np.float32)

    # VAD labels
    vad_hard = torch.tensor(
        [bin_vad(row.get("valence", 5)),
         bin_vad(row.get("arousal", 5)),
         bin_vad(row.get("dominance", 5))],
        dtype=torch.long,
    )
    # Soft label = one-hot of the hard label
    vad_soft = torch.zeros(3, 3, dtype=torch.float32)
    for i in range(3):
        vad_soft[i, int(vad_hard[i])] = 1.0

    vad_weight = torch.ones(3, dtype=torch.float32)

    # Summary features
    feat_dicts = [row.get(c, {}) or {} for c in
                  ["gaze_features", "pupil_features", "eda_features",
                   "ppg_features", "imu_features"]]
    merged: dict = {}
    for d in feat_dicts:
        merged.update(d)
    summary = _to_tensor(flatten_features(merged, summary_key_order))

    uid = int(row.get("user_id", 0))
    sex = int(row.get("sex", -1))
    task_vec = task_onehot(row.get("task", "T0"))

    return dict(
        gaze=gaze, pupil=pupil, eda=eda, ppg=ppg, imu=imu,
        personality=torch.from_numpy(pers),
        vad_hard=vad_hard,
        vad_soft=vad_soft,
        vad_weight=vad_weight,
        summary=summary,
        uid=uid, sex=sex,
        task_onehot_vec=_to_tensor(task_vec),
        is_hard=True,
    )


# ---------------------------------------------------------------------------
# Soft-label sample (augmented window)
# ---------------------------------------------------------------------------

def _augmented_sample(row: pd.Series, summary_key_order: list[str]) -> dict:
    """Convert one augmented row (from augmented_pool.pkl) to a sample dict."""
    gaze  = _to_tensor(seq_to_array(row["gaze_seq"],  GAZE_SEQ_COLS))
    pupil = _to_tensor(seq_to_array(row["pupil_seq"], PUPIL_SEQ_COLS))
    eda   = _to_tensor(seq_to_array(row["eda_seq"],   EDA_SEQ_COLS))
    ppg   = _to_tensor(seq_to_array(row["ppg_seq"],   PPG_SEQ_COLS))
    imu   = _to_tensor(seq_to_array(row["imu_seq"],   IMU_SEQ_COLS))

    pers = np.array([float(row.get(c, 0.0)) for c in BIG_FIVE_COLS], dtype=np.float32)

    # Soft VAD labels — shape (3, 3): one soft distribution per VAD dimension
    vad_soft = torch.zeros(3, 3, dtype=torch.float32)
    vad_weight = torch.zeros(3, dtype=torch.float32)
    vad_hard = torch.tensor([1, 1, 1], dtype=torch.long)  # placeholder Mid class

    for dim_idx, dim in enumerate(DIMS):
        soft_key = f"{dim}_soft"
        w_key    = f"{dim}_weight"
        mu_key   = f"{dim}_mu"

        soft = row.get(soft_key)
        if soft is not None and len(soft) == 3:
            vad_soft[dim_idx] = torch.from_numpy(np.array(soft, dtype=np.float32))
            vad_hard[dim_idx] = int(np.argmax(soft))
        else:
            vad_soft[dim_idx, 1] = 1.0   # default to Mid

        weight = float(row.get(w_key, 0.1))
        vad_weight[dim_idx] = float(np.clip(weight, 0.0, 1.0))

    feat_dicts = [row.get(c, {}) or {} for c in
                  ["gaze_features", "pupil_features", "eda_features",
                   "ppg_features", "imu_features"]]
    merged: dict = {}
    for d in feat_dicts:
        merged.update(d)
    summary = _to_tensor(flatten_features(merged, summary_key_order))

    uid = int(row.get("user_id", 0))
    sex = int(row.get("sex", -1))
    task_vec = task_onehot(row.get("task", "T0"))

    return dict(
        gaze=gaze, pupil=pupil, eda=eda, ppg=ppg, imu=imu,
        personality=torch.from_numpy(pers),
        vad_hard=vad_hard,
        vad_soft=vad_soft,
        vad_weight=vad_weight,
        summary=summary,
        uid=uid, sex=sex,
        task_onehot_vec=_to_tensor(task_vec),
        is_hard=False,
    )


# ---------------------------------------------------------------------------
# Combined Dataset
# ---------------------------------------------------------------------------

class CombinedAffectDataset(Dataset):
    """Dataset combining hard-labeled and soft-labeled augmented windows.

    Parameters
    ----------
    labeled_df    : DataFrame from dataset.pkl (original 219/53 train/val windows)
    augmented_df  : DataFrame from augmented_pool.pkl (soft-labeled pool)
    scalers       : dict of {modality: StandardScaler} from fit_scalers() in train_affectai.py
    apply_augment : if True, apply time_warp + noise_injection to training windows
    """

    def __init__(
        self,
        labeled_df: pd.DataFrame,
        augmented_df: pd.DataFrame | None = None,
        scalers: dict | None = None,
        apply_augment: bool = False,
    ) -> None:
        self.scalers = scalers or {}
        self.apply_augment = apply_augment

        # Build stable summary key order from both DataFrames
        all_df = labeled_df if augmented_df is None else pd.concat(
            [labeled_df.head(50), augmented_df.head(50)], ignore_index=True
        )
        self.summary_key_order = _get_summary_key_order(all_df)

        # Build item list: (row_dict, is_hard)
        self._items: list[tuple[dict, bool]] = []

        for _, row in labeled_df.iterrows():
            self._items.append((row.to_dict(), True))

        if augmented_df is not None and not augmented_df.empty:
            for _, row in augmented_df.iterrows():
                self._items.append((row.to_dict(), False))

        self._n_labeled = len(labeled_df)
        self._n_augmented = len(augmented_df) if augmented_df is not None else 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        row_dict, is_hard = self._items[idx]
        row = pd.Series(row_dict)

        if is_hard:
            sample = _labeled_sample(row, self.summary_key_order)
        else:
            sample = _augmented_sample(row, self.summary_key_order)

        # Apply scalers (StandardScaler per modality)
        for modality, seq_key, cols in [
            ("gaze",  "gaze",  GAZE_SEQ_COLS),
            ("pupil", "pupil", PUPIL_SEQ_COLS),
            ("eda",   "eda",   EDA_SEQ_COLS),
            ("ppg",   "ppg",   PPG_SEQ_COLS),
            ("imu",   "imu",   IMU_SEQ_COLS),
        ]:
            scaler = self.scalers.get(modality)
            if scaler is not None:
                arr = sample[seq_key].numpy()   # (T, F)
                arr = scaler.transform(arr)
                sample[seq_key] = torch.from_numpy(arr.astype(np.float32))

        # Data augmentation (training only, hard-label windows preferred but also apply to soft)
        if self.apply_augment:
            from dataset_affectai import time_warp, noise_injection
            for key in ("gaze", "pupil", "eda", "ppg", "imu"):
                arr = sample[key].numpy()
                arr = time_warp(arr)
                # Pad / crop back to FIXED_LENGTH
                from scipy.signal import resample as sp_resample
                import importlib
                import sys
                from dataset_affectai import GAZE_SEQ_COLS as _
                target = 400
                if arr.shape[0] != target:
                    arr = sp_resample(arr, target, axis=0)
                arr = noise_injection(arr)
                sample[key] = torch.from_numpy(arr.astype(np.float32))

        return sample

    def instance_weights(self) -> np.ndarray:
        """Return per-sample weights for WeightedRandomSampler.

        Hard-label samples: weight = 1.0
        Augmented samples:  weight = mean(vad_weight over dims that pass W_MIN)
        """
        weights = []
        for row_dict, is_hard in self._items:
            if is_hard:
                weights.append(1.0)
            else:
                w_vals = []
                for dim in DIMS:
                    w = float(row_dict.get(f"{dim}_weight", 0.1))
                    w_vals.append(w)
                weights.append(float(np.mean(w_vals)))
        return np.array(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def soft_ce_loss(
    logits: torch.Tensor,
    vad_soft: torch.Tensor,
    vad_weight: torch.Tensor,
) -> torch.Tensor:
    """Soft cross-entropy loss for one VAD dimension.

    Parameters
    ----------
    logits     : (B, 3) — model output for one VAD dimension
    vad_soft   : (B, 3) — soft label distributions (rows sum to 1)
    vad_weight : (B,)   — per-sample instance weights

    Returns
    -------
    Scalar loss (weighted mean soft-CE).
    """
    import torch.nn.functional as F
    log_p = F.log_softmax(logits, dim=-1)           # (B, 3)
    ce    = -(vad_soft * log_p).sum(dim=-1)         # (B,)
    return (vad_weight * ce).mean()


def combined_vad_loss(
    val_logits: torch.Tensor,
    aro_logits: torch.Tensor,
    dom_logits: torch.Tensor,
    batch: dict,
    dim_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Compute combined VAD soft-CE loss from a batch dict.

    Works for both hard-label batches (vad_weight all 1.0, vad_soft one-hot)
    and soft-label augmented batches.

    Parameters
    ----------
    val_logits, aro_logits, dom_logits : (B, 3) model outputs
    batch      : dict from CombinedAffectDataset.__getitem__() collated to batch
    dim_weights: relative weighting of (valence, arousal, dominance) losses
    """
    vad_soft   = batch["vad_soft"]    # (B, 3, 3)
    vad_weight = batch["vad_weight"]  # (B, 3)

    loss_v = soft_ce_loss(val_logits, vad_soft[:, 0, :], vad_weight[:, 0])
    loss_a = soft_ce_loss(aro_logits, vad_soft[:, 1, :], vad_weight[:, 1])
    loss_d = soft_ce_loss(dom_logits, vad_soft[:, 2, :], vad_weight[:, 2])

    wv, wa, wd = dim_weights
    return (wv * loss_v + wa * loss_a + wd * loss_d) / (wv + wa + wd)


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_combined_loader(
    labeled_df: pd.DataFrame,
    augmented_df: pd.DataFrame | None,
    scalers: dict | None = None,
    batch_size: int = 64,
    augmented_fraction: float = 0.5,
    num_workers: int = 0,
    apply_augment: bool = True,
) -> DataLoader:
    """Build a DataLoader that ensures *augmented_fraction* of each batch
    is drawn from the augmented pool, using WeightedRandomSampler.

    Parameters
    ----------
    labeled_df          : original labeled training windows
    augmented_df        : soft-labeled augmented pool (can be None)
    scalers             : per-modality StandardScaler dict
    batch_size          : total samples per batch
    augmented_fraction  : target fraction of augmented windows per batch [0, 1]
    num_workers         : DataLoader num_workers
    apply_augment       : apply time-warp + noise to input signals
    """
    dataset = CombinedAffectDataset(
        labeled_df=labeled_df,
        augmented_df=augmented_df,
        scalers=scalers,
        apply_augment=apply_augment,
    )

    if augmented_df is None or augmented_df.empty:
        # No augmented data — plain random sampler
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)

    # Build sampling weights to achieve target augmented_fraction
    n_hard = dataset._n_labeled
    n_aug  = dataset._n_augmented
    n_total = len(dataset)

    # Target: augmented fraction of batch = augmented_fraction
    # weight_hard * n_hard / (weight_hard * n_hard + weight_aug * n_aug) = 1 - augmented_fraction
    # => weight_aug / weight_hard = (n_hard * augmented_fraction) / (n_aug * (1-augmented_fraction))
    if n_aug > 0 and 0 < augmented_fraction < 1:
        ratio = (n_hard * augmented_fraction) / (n_aug * (1.0 - augmented_fraction))
    else:
        ratio = 1.0

    base_weights = dataset.instance_weights()  # confidence-based weights
    hard_mask = np.zeros(n_total, dtype=np.float32)
    hard_mask[:n_hard] = 1.0
    aug_mask = 1.0 - hard_mask

    sampling_weights = hard_mask * 1.0 + aug_mask * float(ratio) * base_weights[n_hard:]
    # Rescale so max = 1
    sampling_weights = sampling_weights / (sampling_weights.max() + 1e-8)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sampling_weights),
        num_samples=n_total,
        replacement=True,
    )

    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers)
