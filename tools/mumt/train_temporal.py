"""train_temporal.py

Two-track training for per-modality temporal models on GroupAffect-4.

Architecture: TemporalFusionNet (Conv1D or GRU per modality) from model_temporal.py.
Also runs an MLP-only baseline (encoder_type='mlp') that uses summary features only.

Two-track training strategy
---------------------------
  Track A — labeled windows (T0+T1, ~103 windows):
    Raw sequences (200×d_m) + summary features + BFI → full mode of TemporalFusionNet
    Hard cross-entropy targets, sample weight = 1.0

  Track B — augmented pool windows (train tasks only):
    Summary features only + BFI → summary-only mode of TemporalFusionNet
    Soft pseudo-label targets from AP1 (BFI cosine similarity trust weight)

Test sets always use ground-truth SAM labels only (task-CV: train=T0+T1, val=T2, test=T3).

Augmentation variants (--aug)
------------------------------
  none    Track A only (no pool augmentation)
  a2      GP all dims (conf ≥ 0.5)  — best pure-GP variant
  ap1     BFI similarity only (best overall SVM variant)
  ap2     BFI_sim × GP confidence

Encoder types (--encoder)
--------------------------
  mlp     Summary features only (no temporal encoder) — neural baseline
  conv1d  Conv1D per modality + summary features (two-track)
  gru     Bidirectional GRU per modality + summary features (two-track)

Usage
-----
  python tools/mumt/train_temporal.py
  python tools/mumt/train_temporal.py --encoder gru --aug ap1 --epochs 200
  python tools/mumt/train_temporal.py --encoder conv1d --aug none --out results/temporal.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import (  # noqa: E402
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    build_summary_key_order,
    flatten_features,
    seq_to_array,
)
from model_temporal import (  # noqa: E402
    MODALITIES,
    MODALITY_DIMS,
    SUMMARY_DIM,
    SoftVADLoss,
    TemporalFusionNet,
)
from train_simple import (  # noqa: E402
    bin_vad_from_thresholds,
    compute_tertile_thresholds,
    task_split,
)

VAD_DIMS = ["valence", "arousal", "dominance"]
FEAT_COLS = ["gaze_features", "pupil_features", "eda_features",
             "ppg_features", "imu_features"]
MODALITY_COLS = {
    "gaze":  GAZE_SEQ_COLS,
    "pupil": PUPIL_SEQ_COLS,
    "eda":   EDA_SEQ_COLS,
    "ppg":   PPG_SEQ_COLS,
    "imu":   IMU_SEQ_COLS,
}
BFI_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]


# ── Sequence augmentation ──────────────────────────────────────────────────────

def augment_sequence(
    arr: np.ndarray,
    noise_sigma: float = 0.05,
    scale_lo: float = 0.85,
    scale_hi: float = 1.15,
    max_shift: int = 20,
) -> np.ndarray:
    """Apply stochastic time-domain augmentation to a (T, D) sequence.

    Three independent transforms applied in sequence:
      1. Additive Gaussian noise scaled to per-feature std
      2. Amplitude jitter (global scale)
      3. Circular time shift (± max_shift steps)

    Returns a new (T, D) float32 array.
    """
    T, D = arr.shape
    # 1. Gaussian noise proportional to per-feature std
    feat_std = arr.std(axis=0, keepdims=True).clip(1e-6)
    arr = arr + (np.random.randn(T, D) * noise_sigma * feat_std).astype(np.float32)
    # 2. Amplitude jitter
    scale = np.random.uniform(scale_lo, scale_hi)
    arr = (arr * scale).astype(np.float32)
    # 3. Circular shift
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        arr = np.roll(arr, shift, axis=0).astype(np.float32)
    return arr


# ── Feature helpers ────────────────────────────────────────────────────────────

class SequenceScaler:
    def __init__(self) -> None:
        self._sk = StandardScaler()

    def fit(self, arrays: list[np.ndarray]) -> "SequenceScaler":
        self._sk.fit(np.concatenate(arrays, axis=0))
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return self._sk.transform(arr).astype(np.float32)


def fit_seq_scalers(train_df: pd.DataFrame) -> dict[str, SequenceScaler]:
    scalers: dict[str, SequenceScaler] = {}
    for mod, cols in MODALITY_COLS.items():
        arrs = [seq_to_array(row[f"{mod}_seq"], cols)
                for _, row in train_df.iterrows()]
        scalers[mod] = SequenceScaler().fit(arrs)
    return scalers


def extract_summary(df: pd.DataFrame, key_order: list[str]) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in FEAT_COLS:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    X = np.stack(rows, axis=0).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def get_hard_labels(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = np.full((len(df), 3), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if not np.isnan(v):
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


def compute_class_weight_tensors(
    train_labels: np.ndarray,
    device: torch.device,
) -> list[torch.Tensor]:
    weights = []
    for d in range(3):
        vals = train_labels[:, d]
        valid = vals[vals >= 0]
        counts = np.bincount(valid, minlength=3).astype(float)
        counts = np.clip(counts, 1, None)
        w = 1.0 / counts
        w = w / w.sum() * 3.0
        weights.append(torch.tensor(w, dtype=torch.float32, device=device))
    return weights


# ── Augmented pool helpers ─────────────────────────────────────────────────────

def recompute_soft_labels(
    pool: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    dim: str,
) -> np.ndarray:
    t1, t2 = thresholds[dim]
    mu = pool[f"{dim}_mu"].fillna(5.0).values.astype(float)
    sig = np.clip(pool[f"{dim}_sigma"].fillna(1.5).values.astype(float), 1e-4, None)
    p_low  = norm.cdf(t1, loc=mu, scale=sig)
    p_high = 1.0 - norm.cdf(t2, loc=mu, scale=sig)
    p_mid  = np.clip(1.0 - p_low - p_high, 0.0, 1.0)
    soft = np.stack([p_low, p_mid, p_high], axis=1).astype(np.float32)
    row_sums = soft.sum(axis=1, keepdims=True)
    return soft / np.where(row_sums < 1e-8, 1.0, row_sums)


def compute_bfi_similarity_map(
    dataset_df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    avail = [c for c in BFI_COLS if c in dataset_df.columns]
    if not avail:
        return {}
    person_bfi = (
        dataset_df[["session_id", "seat"] + avail]
        .dropna(subset=avail)
        .groupby(["session_id", "seat"])
        .first()
        .reset_index()
    )
    bfi_map: dict[tuple[str, str], float] = {}
    for session_id, grp in person_bfi.groupby("session_id"):
        seats = grp["seat"].values
        vecs  = grp[avail].values.astype(float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        sim_mat = (vecs / norms) @ (vecs / norms).T
        for i, seat in enumerate(seats):
            others = [j for j in range(len(seats)) if j != i]
            sim = float(np.mean(sim_mat[i, others])) if others else 0.5
            bfi_map[(session_id, str(seat))] = float(np.clip(sim, 0.0, 1.0))
    return bfi_map


def build_pool_pseudo_labels(
    pool: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    bfi_sim_map: dict,
    aug_mode: str,
    conf_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (soft_labels (N,3,3), weights (N,3), mask (N,3) bool)."""
    N = len(pool)
    soft_all = np.zeros((N, 3, 3), dtype=np.float32)
    weights  = np.zeros((N, 3), dtype=np.float32)
    mask     = np.zeros((N, 3), dtype=bool)

    bfi_sims = np.array([
        bfi_sim_map.get((str(r.session_id), str(r.seat)), 0.5)
        for r in pool.itertuples()
    ], dtype=np.float32)

    for col_i, dim in enumerate(VAD_DIMS):
        weight_col = f"{dim}_weight"
        if weight_col not in pool.columns:
            continue

        gp_conf = pool[weight_col].fillna(0.0).values.astype(float)
        if dim == "dominance":
            gp_conf = np.where(pool["task"].values == "T4", 0.0, gp_conf)

        if aug_mode == "ap1":
            conf = bfi_sims.astype(float)
        elif aug_mode == "ap2":
            conf = (bfi_sims * gp_conf).astype(float)
        elif aug_mode == "a2":
            conf = gp_conf
        else:
            continue

        soft = recompute_soft_labels(pool, thresholds, dim)
        pseudo = np.argmax(soft, axis=1)
        max_p  = soft[np.arange(N), pseudo]

        accepted = (conf >= conf_threshold) & (max_p >= 0.5)
        soft_all[accepted, col_i, :] = soft[accepted]
        weights[accepted, col_i]     = conf[accepted].astype(np.float32)
        mask[accepted, col_i]        = True

    return soft_all, weights, mask


# ── Dataset classes ────────────────────────────────────────────────────────────

class LabeledDataset(Dataset):
    """Labeled windows with raw sequences + summary features + BFI."""

    def __init__(
        self,
        df: pd.DataFrame,
        key_order: list[str],
        thresholds: dict[str, tuple[float, float]],
        seq_scalers: dict[str, SequenceScaler] | None = None,
        use_sequences: bool = True,
        augment: bool = False,
        seq_aug_fn: callable | None = None,
    ) -> None:
        self.df            = df.reset_index(drop=True)
        self.key_order     = key_order
        self.thresholds    = thresholds
        self.seq_scalers   = seq_scalers
        self.use_sequences = use_sequences
        self.augment       = augment  # time-domain seq aug at __getitem__ time
        self.seq_aug_fn    = seq_aug_fn  # custom augmentation function (T,D)->ndarray

        self.summary = extract_summary(df, key_order)
        self.labels  = get_hard_labels(df, thresholds)

        avail = [c for c in BFI_COLS if c in df.columns]
        if avail:
            bfi_raw = df[avail].fillna(0.0).values.astype(np.float32)
        else:
            bfi_raw = np.zeros((len(df), 5), dtype=np.float32)
        self.bfi = bfi_raw

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        summary_t = torch.from_numpy(self.summary[idx])
        bfi_t     = torch.from_numpy(self.bfi[idx])
        labels_t  = torch.from_numpy(self.labels[idx])

        seqs: dict[str, torch.Tensor] | None = None
        if self.use_sequences:
            seqs = {}
            for mod, cols in MODALITY_COLS.items():
                arr = seq_to_array(row[f"{mod}_seq"], cols)
                if self.seq_scalers and mod in self.seq_scalers:
                    arr = self.seq_scalers[mod].transform(arr)
                if self.augment:
                    if self.seq_aug_fn is not None:
                        arr = self.seq_aug_fn(arr)
                    else:
                        arr = augment_sequence(arr)
                seqs[mod] = torch.from_numpy(arr)

        return {
            "summary":     summary_t,
            "bfi":         bfi_t,
            "labels":      labels_t,
            "has_seq":     self.use_sequences,
            "sequences":   seqs,
            "soft_labels": None,
            "weight":      torch.ones(3, dtype=torch.float32),
        }


class AugPoolDataset(Dataset):
    """Augmented pool windows — summary features + BFI + optional raw sequences.

    When ``use_sequences=True`` (and pool contains *_seq columns), each window
    uses the full encoder path — eliminating the zero-padding signal conflict
    that hurt Conv1D/GRU performance in the two-track design.
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        key_order: list[str],
        thresholds: dict[str, tuple[float, float]],
        bfi_sim_map: dict,
        aug_mode: str,
        seq_scalers: dict[str, SequenceScaler] | None = None,
        use_sequences: bool = True,
    ) -> None:
        self.seq_scalers  = seq_scalers
        # Detect whether pool actually has sequence columns
        has_seq_cols = all(f"{m}_seq" in pool.columns for m in MODALITIES)
        self.use_sequences = use_sequences and has_seq_cols and seq_scalers is not None

        summary_arr = extract_summary(pool, key_order)
        soft_all, weights, mask = build_pool_pseudo_labels(
            pool, thresholds, bfi_sim_map, aug_mode
        )
        valid_rows = mask.any(axis=1)

        self.summary_arr = summary_arr[valid_rows]
        self.soft_labels = soft_all[valid_rows]    # (N', 3, 3)
        self.weights     = weights[valid_rows]     # (N', 3)

        pool_valid = pool.reset_index(drop=True).loc[valid_rows].reset_index(drop=True)
        avail = [c for c in BFI_COLS if c in pool.columns]
        if avail:
            self.bfi = pool_valid[avail].fillna(0.0).values.astype(np.float32)
        else:
            self.bfi = np.zeros((len(pool_valid), 5), dtype=np.float32)

        # Pre-compute sequences at init time to avoid per-sample DataFrame access
        # in __getitem__ (which causes severe CPU bottleneck during DataLoader iteration).
        self.seq_arrays: dict[str, np.ndarray] | None = None
        if self.use_sequences:
            self.seq_arrays = {}
            for mod, cols in MODALITY_COLS.items():
                arrays = []
                for _, row in pool_valid.iterrows():
                    arr = seq_to_array(row[f"{mod}_seq"], cols)
                    if seq_scalers and mod in seq_scalers:
                        arr = seq_scalers[mod].transform(arr)
                    arrays.append(arr)
                self.seq_arrays[mod] = np.stack(arrays, axis=0)  # (N', T, D)

        log.info("  Pool → %d/%d windows accepted (aug=%s, use_seq=%s)",
                 valid_rows.sum(), len(pool), aug_mode, self.use_sequences)

    def __len__(self) -> int:
        return len(self.summary_arr)

    def __getitem__(self, idx: int) -> dict:
        seqs: dict[str, torch.Tensor] | None = None
        if self.use_sequences and self.seq_arrays is not None:
            seqs = {mod: torch.from_numpy(self.seq_arrays[mod][idx])
                    for mod in MODALITIES}

        return {
            "summary":     torch.from_numpy(self.summary_arr[idx]),
            "bfi":         torch.from_numpy(self.bfi[idx]),
            "labels":      torch.full((3,), -1, dtype=torch.int64),
            "has_seq":     self.use_sequences,
            "sequences":   seqs,
            "soft_labels": torch.from_numpy(self.soft_labels[idx]),  # (3, 3)
            "weight":      torch.from_numpy(self.weights[idx]),       # (3,)
        }


def collate_labeled(batch: list[dict]) -> dict:
    """Custom collate that stacks sequences into per-modality tensors."""
    keys_scalar = ["summary", "bfi", "labels", "weight"]
    result = {k: torch.stack([b[k] for b in batch]) for k in keys_scalar}
    result["has_seq"] = batch[0]["has_seq"]
    result["soft_labels"] = None

    if batch[0]["has_seq"] and batch[0]["sequences"] is not None:
        seqs: dict[str, torch.Tensor] = {}
        for mod in MODALITIES:
            seqs[mod] = torch.stack([b["sequences"][mod] for b in batch])
        result["sequences"] = seqs
    else:
        result["sequences"] = None

    return result


def collate_pool(batch: list[dict]) -> dict:
    keys_scalar = ["summary", "bfi", "labels", "weight", "soft_labels"]
    result = {k: torch.stack([b[k] for b in batch]) for k in keys_scalar}
    result["has_seq"] = batch[0]["has_seq"]

    if batch[0]["has_seq"] and batch[0]["sequences"] is not None:
        seqs: dict[str, torch.Tensor] = {}
        for mod in MODALITIES:
            seqs[mod] = torch.stack([b["sequences"][mod] for b in batch])
        result["sequences"] = seqs
    else:
        result["sequences"] = None
    return result


# ── Summary feature scaler ─────────────────────────────────────────────────────

class SummaryScaler:
    def __init__(self) -> None:
        self._sk = StandardScaler()

    def fit(self, X: np.ndarray) -> "SummaryScaler":
        self._sk.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._sk.transform(X).astype(np.float32)


# ── Training helpers ───────────────────────────────────────────────────────────

def predict_hard(model: TemporalFusionNet, batch: dict, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        summary = batch["summary"].to(device)
        bfi     = batch["bfi"].to(device)
        seqs    = {k: v.to(device) for k, v in batch["sequences"].items()} \
                  if batch["sequences"] is not None else None
        logits = model(summary, seqs, bfi)  # (B, 3, 3)
        preds  = logits.argmax(dim=-1).cpu().numpy()  # (B, 3)
    return preds


def eval_macro_f1(
    model: TemporalFusionNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float]]:
    all_preds  = []
    all_labels = []
    for batch in loader:
        preds = predict_hard(model, batch, device)
        all_preds.append(preds)
        all_labels.append(batch["labels"].numpy())

    preds_arr  = np.concatenate(all_preds,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    f1s = []
    for d, dim in enumerate(VAD_DIMS):
        valid = labels_arr[:, d] >= 0
        if valid.sum() == 0:
            f1s.append(0.0)
        else:
            f1s.append(float(f1_score(
                labels_arr[valid, d], preds_arr[valid, d],
                average="macro", zero_division=0,
            )))
    mean_f1 = float(np.mean(f1s))
    return mean_f1, f1s


def make_one_hot_soft(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(B, 3) int64 → (B, 3, 3) one-hot float, -1 mapped to uniform 1/3."""
    B = labels.shape[0]
    soft = torch.zeros(B, 3, 3, device=device)
    for d in range(3):
        col = labels[:, d]
        valid = col >= 0
        if valid.any():
            soft[valid, d, :] = F.one_hot(col[valid], num_classes=3).float()
        soft[~valid, d, :] = 1.0 / 3.0
    return soft


# ── One training fold ─────────────────────────────────────────────────────────

def run_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pool: pd.DataFrame | None,
    bfi_sim_map: dict,
    key_order: list[str],
    encoder_type: str,
    aug_mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    enc_dim: int,
    dropout: float,
    patience: int,
    device: torch.device,
    seq_aug: bool = False,
    pool_seqs: bool = True,
    pretrained_encoders: dict | None = None,
    freeze_epochs: int = 0,
    seq_aug_fn: callable | None = None,
) -> dict:
    """Train one fold.

    Parameters
    ----------
    seq_aug   : apply time-domain sequence augmentation (noise/scale/shift) on
                labeled windows during training.
    pool_seqs : use raw sequences from pool windows (when available) so the pool
                track uses the same encoder path as the labeled track.
    pretrained_encoders : optional dict of pre-trained encoder state_dicts to
                load before fine-tuning.
    freeze_epochs : number of initial epochs with frozen encoder weights
                (only used when pretrained_encoders is provided).
    seq_aug_fn : optional custom augmentation function (arr: ndarray) -> ndarray.
                If provided and seq_aug=True, this replaces the default
                augment_sequence (noise/jitter/shift).
    """

    thresholds = compute_tertile_thresholds(train_df)
    seq_scalers = fit_seq_scalers(train_df) if encoder_type != "mlp" else None
    use_seq = encoder_type != "mlp"

    # Summary scaler fit on train
    train_summary = extract_summary(train_df, key_order)
    summary_sc = SummaryScaler().fit(train_summary)

    def make_labeled_loader(
        df: pd.DataFrame, shuffle: bool = False, augment: bool = False
    ) -> DataLoader:
        ds = LabeledDataset(df, key_order, thresholds, seq_scalers, use_seq,
                            augment=augment, seq_aug_fn=seq_aug_fn)
        ds.summary = summary_sc.transform(ds.summary)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_labeled, drop_last=False)

    train_loader = make_labeled_loader(train_df, shuffle=True,
                                       augment=(seq_aug and use_seq))
    val_loader   = make_labeled_loader(val_df,   shuffle=False)
    test_loader  = make_labeled_loader(test_df,  shuffle=False)

    # Pool loader (only if augmenting)
    pool_loader: DataLoader | None = None
    if aug_mode != "none" and pool is not None:
        pool_ds = AugPoolDataset(
            pool, key_order, thresholds, bfi_sim_map, aug_mode,
            seq_scalers=seq_scalers if pool_seqs else None,
            use_sequences=pool_seqs and use_seq,
        )
        pool_ds.summary_arr = summary_sc.transform(pool_ds.summary_arr)
        if len(pool_ds) > 0:
            pool_loader = DataLoader(pool_ds, batch_size=batch_size * 2, shuffle=True,
                                     collate_fn=collate_pool, drop_last=False)

    # Model
    train_labels = get_hard_labels(train_df, thresholds)
    class_weights = compute_class_weight_tensors(train_labels, device)

    bfi_dim = len([c for c in BFI_COLS if c in train_df.columns])
    model = TemporalFusionNet(
        encoder_type=encoder_type if encoder_type != "mlp" else "conv1d",
        enc_dim=enc_dim,
        dropout=dropout,
        bfi_dim=bfi_dim,
    ).to(device)

    # For MLP mode, always use summary-only (no sequences fed)
    if encoder_type == "mlp":
        model.encoders = nn.ModuleDict()  # empty — not used

    # Load pre-trained encoder weights if provided
    if pretrained_encoders is not None and encoder_type != "mlp":
        for name in MODALITIES:
            if name in pretrained_encoders:
                model.encoders[name].load_state_dict(
                    pretrained_encoders[name].state_dict()
                )
        log.info("  Loaded pre-trained encoder weights (freeze_epochs=%d).", freeze_epochs)

    criterion = SoftVADLoss(class_weights=class_weights, label_smooth=0.1)

    # Optimizer with optional encoder freezing
    def _make_optimizer(freeze: bool) -> torch.optim.Optimizer:
        if freeze and encoder_type != "mlp":
            for name in MODALITIES:
                for p in model.encoders[name].parameters():
                    p.requires_grad = False
        elif encoder_type != "mlp":
            for name in MODALITIES:
                for p in model.encoders[name].parameters():
                    p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    use_freeze = (pretrained_encoders is not None and freeze_epochs > 0)
    optimizer = _make_optimizer(freeze=use_freeze)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for epoch in range(epochs):
        # Unfreeze encoders after freeze_epochs
        if use_freeze and epoch == freeze_epochs:
            optimizer = _make_optimizer(freeze=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch
            )

        model.train()

        # Track A — labeled windows
        for batch in train_loader:
            optimizer.zero_grad()
            summary = batch["summary"].to(device)
            bfi     = batch["bfi"].to(device)
            labels  = batch["labels"].to(device)
            seqs    = ({k: v.to(device) for k, v in batch["sequences"].items()}
                       if batch["sequences"] is not None else None)

            logits = model(summary, seqs, bfi)  # (B, 3, 3)
            soft   = make_one_hot_soft(labels, device)
            sw     = batch["weight"].to(device)
            # dim_mask: valid where label != -1
            dim_mask = (labels >= 0)
            loss = criterion(logits, labels, soft_targets=soft,
                             sample_weights=sw[:, 0], dim_mask=dim_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Track B — augmented pool (with sequences if pool_seqs=True, else summary-only)
        if pool_loader is not None:
            model.train()
            pool_iter = iter(pool_loader)
            for _ in range(max(1, len(pool_loader) // 2)):
                try:
                    pbatch = next(pool_iter)
                except StopIteration:
                    break
                optimizer.zero_grad()
                summary     = pbatch["summary"].to(device)
                bfi         = pbatch["bfi"].to(device)
                labels      = pbatch["labels"].to(device)
                soft_labels = pbatch["soft_labels"].to(device)
                sw          = pbatch["weight"].to(device)
                # Use pool sequences when available (eliminates zero-padding conflict)
                p_seqs = ({k: v.to(device) for k, v in pbatch["sequences"].items()}
                          if pbatch["sequences"] is not None else None)

                logits = model(summary, p_seqs, bfi)
                pseudo_hard = soft_labels.argmax(dim=-1).clone()  # (B, 3)
                pseudo_hard[sw == 0] = -1
                loss = criterion(logits, pseudo_hard, soft_targets=soft_labels,
                                 sample_weights=sw.mean(dim=1))
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        scheduler.step()

        # Validation
        val_f1, _ = eval_macro_f1(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stop at epoch %d (val F1=%.4f)", epoch + 1, best_val_f1)
                break

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    test_f1, test_f1s = eval_macro_f1(model, test_loader, device)

    return {
        "val_f1":   round(best_val_f1, 4),
        "test_f1":  round(test_f1, 4),
        "v_f1":     round(test_f1s[0], 4),
        "a_f1":     round(test_f1s[1], 4),
        "d_f1":     round(test_f1s[2], 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool",     default="data/mumt/augmented_pool_slow.pkl")
    parser.add_argument("--out",      default="results/temporal_comparison.csv")
    parser.add_argument("--encoder",  default="all",
                        choices=["all", "mlp", "conv1d", "gru"])
    parser.add_argument("--aug",      default="all",
                        choices=["all", "none", "a2", "ap1", "ap2"])
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--enc-dim",  type=int,   default=32)
    parser.add_argument("--dropout",  type=float, default=0.3)
    parser.add_argument("--patience", type=int,   default=30)
    parser.add_argument("--test-task", default="T3")
    # Improvement flags
    parser.add_argument("--seq-aug",    action="store_true",
                        help="Apply noise/scale/shift augmentation on labeled sequences")
    parser.add_argument("--no-pool-seqs", action="store_true",
                        help="Disable pool raw sequences (revert to zero-padding two-track)")
    parser.add_argument("--tag", default="",
                        help="Optional tag appended to encoder name in output CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d windows", len(df))

    pool: pd.DataFrame | None = None
    if Path(args.pool).exists():
        pool = pd.read_pickle(args.pool)
        log.info("Pool: %d windows", len(pool))
    else:
        log.warning("Pool not found: %s — augmentation disabled", args.pool)

    key_order   = build_summary_key_order(df)
    bfi_sim_map = compute_bfi_similarity_map(df)
    log.info("Summary dim: %d  |  BFI map entries: %d", len(key_order), len(bfi_sim_map))

    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Pool restricted to training tasks to prevent leakage
    if pool is not None:
        train_tasks = train_df["task"].unique().tolist()
        pool_train = pool[pool["task"].isin(train_tasks)].reset_index(drop=True)
        log.info("Pool (train tasks only): %d", len(pool_train))
    else:
        pool_train = None

    encoders = ["mlp", "conv1d", "gru"] if args.encoder == "all" else [args.encoder]
    augs     = ["none", "a2", "ap1", "ap2"] if args.aug == "all" else [args.aug]
    pool_seqs = not args.no_pool_seqs
    tag = ("+" + args.tag) if args.tag else ""
    if args.seq_aug:
        tag = "+seqaug" + tag
    if not pool_seqs:
        tag = "+nopool" + tag

    records = []
    for enc in encoders:
        for aug in augs:
            if aug != "none" and pool_train is None:
                log.warning("Skip %s/%s — pool not available", enc, aug)
                continue
            log.info("=== encoder=%s%s  aug=%s  seq_aug=%s  pool_seqs=%s ===",
                     enc, tag, aug, args.seq_aug, pool_seqs)
            result = run_fold(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                pool=pool_train if aug != "none" else None,
                bfi_sim_map=bfi_sim_map,
                key_order=key_order,
                encoder_type=enc,
                aug_mode=aug,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                enc_dim=args.enc_dim,
                dropout=args.dropout,
                patience=args.patience,
                device=device,
                seq_aug=args.seq_aug,
                pool_seqs=pool_seqs,
            )
            enc_label = enc + tag
            records.append({
                "encoder": enc_label,
                "aug":     aug,
                **result,
            })
            log.info("  V=%.4f  A=%.4f  D=%.4f  Mean=%.4f  (val=%.4f)",
                     result["v_f1"], result["a_f1"], result["d_f1"],
                     result["test_f1"], result["val_f1"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    print("\n=== Temporal model comparison ===")
    print(out_df[["encoder", "aug", "v_f1", "a_f1", "d_f1", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
