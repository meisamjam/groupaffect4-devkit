"""Standalone ordinal training for GroupAffect-4 original 1-9 SAM labels.

This script does not modify or depend on the 3-class training code path.  It
uses cumulative ordinal targets for valence/arousal/dominance:

    target_k = 1[y > k],  k = 1..8

Dominance NaNs are masked.  Optional GP augmentation reads stored posterior
moments (mu, sigma) and converts them to ordinal soft cumulative targets.
Stored 3-class GP labels are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from scipy.stats import norm, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import (  # noqa: E402
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    flatten_features,
    make_session2idx,
    make_user2idx,
    seq_to_array,
    task_onehot,
)
from model_ordinal import build_ordinal_model  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


VAD_DIMS = ("valence", "arousal", "dominance")
MODALITY_COLS = {
    "gaze": GAZE_SEQ_COLS,
    "pupil": PUPIL_SEQ_COLS,
    "eda": EDA_SEQ_COLS,
    "ppg": PPG_SEQ_COLS,
    "imu": IMU_SEQ_COLS,
}
SUMMARY_FEATURE_COLS = [
    "gaze_features",
    "pupil_features",
    "eda_features",
    "ppg_features",
    "imu_features",
]
ORDINAL_CUTS = np.arange(1, 9, dtype=np.float32)


class SequenceScaler:
    """Per-feature scaler for a list of (T, F) sequence arrays."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, arrays: list[np.ndarray]) -> "SequenceScaler":
        self.scaler.fit(np.concatenate(arrays, axis=0))
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        return self.scaler.transform(array).astype(np.float32)


def fit_scalers(train_df: pd.DataFrame) -> dict[str, SequenceScaler]:
    scalers: dict[str, SequenceScaler] = {}
    for mod, cols in MODALITY_COLS.items():
        arrays = [seq_to_array(row[f"{mod}_seq"], cols) for _, row in train_df.iterrows()]
        scalers[mod] = SequenceScaler().fit(arrays)
    return scalers


def build_physio_summary_key_order(df: pd.DataFrame) -> list[str]:
    keys: set[str] = set()
    for col in SUMMARY_FEATURE_COLS:
        if col not in df.columns:
            continue
        for feat in df[col]:
            if isinstance(feat, dict):
                keys.update(feat.keys())
    return sorted(keys)


def get_eda_summary_keys(df: pd.DataFrame) -> set[str]:
    """Return the set of summary-vector keys that originate from the eda_features column."""
    keys: set[str] = set()
    if "eda_features" in df.columns:
        for feat in df["eda_features"]:
            if isinstance(feat, dict):
                keys.update(feat.keys())
    return keys


def flatten_physio_summary(row: pd.Series, key_order: list[str]) -> np.ndarray:
    feats: dict = {}
    for col in SUMMARY_FEATURE_COLS:
        fd = row.get(col, {})
        if isinstance(fd, dict):
            feats.update(fd)
    return flatten_features(feats, key_order=key_order)


def make_unimodal_ordinal_targets(labels: np.ndarray, sigma: float) -> np.ndarray:
    """Convert integer labels (N,) to soft cumulative ordinal targets (N, 8).

    Places a Gaussian of width ``sigma`` over the 1-9 scale centered on each
    label value, then returns the cumulative CDF at thresholds 1..8.
    When sigma→0 this recovers hard step-function targets.
    Recommended: sigma=1.0 (one Likert-step smoothing).
    """
    k = np.arange(1, 10, dtype=np.float32)            # classes 1..9
    diff2 = (k[None, :] - labels[:, None]) ** 2       # (N, 9)
    p = np.exp(-0.5 * diff2 / (sigma ** 2))
    p = p / p.sum(axis=1, keepdims=True)                  # (N, 9) normalize
    cdf      = np.cumsum(p, axis=1)[:, :8]               # P(Y<=k) for k=1..8
    # Model target format is P(Y>k) = 1 - P(Y<=k)  (matches hard step-function convention)
    return (1.0 - cdf).astype(np.float32)


def hard_ordinal_targets(
    labels: np.ndarray,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative targets (N, 3, 8) and per-dimension weights (N, 3).

    When sigma > 0 uses unimodal Gaussian smoothing over the ordinal scale
    (Wen et al., ICCV 2023) instead of hard step-function targets.
    """
    n = labels.shape[0]
    targets = np.zeros((n, 3, 8), dtype=np.float32)
    weights = np.ones((n, 3), dtype=np.float32)
    for d_idx in range(3):
        values = labels[:, d_idx]
        valid = np.isfinite(values)
        weights[~valid, d_idx] = 0.0
        safe_values = np.where(valid, values, 1.0)
        if sigma > 0:
            targets[:, d_idx, :] = make_unimodal_ordinal_targets(safe_values, sigma)
        else:
            targets[:, d_idx, :] = (safe_values[:, None] > ORDINAL_CUTS[None, :]).astype(
                np.float32
            )
    return targets, weights


def gp_ordinal_targets(
    row: pd.Series,
    dim: str,
) -> np.ndarray:
    """Convert stored GP posterior moments to cumulative ordinal soft targets."""
    mu = float(row.get(f"{dim}_mu", 5.0))
    sigma = max(float(row.get(f"{dim}_sigma", 1.5)), 1e-4)
    # Boundary between integer categories k and k+1 is k+0.5.
    boundaries = ORDINAL_CUTS + 0.5
    probs = 1.0 - norm.cdf(boundaries, loc=mu, scale=sigma)
    return np.clip(probs, 0.0, 1.0).astype(np.float32)


def compute_pos_weight(train_targets: np.ndarray, train_weights: np.ndarray) -> torch.Tensor:
    """Positive class weights for cumulative BCE, shape (3, 8)."""
    weights = np.zeros((3, 8), dtype=np.float32)
    for d in range(3):
        valid = train_weights[:, d] > 0
        t = train_targets[valid, d, :]
        pos = t.sum(axis=0)
        neg = t.shape[0] - pos
        weights[d, :] = np.clip(neg / np.clip(pos, 1.0, None), 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


class OrdinalLabelDataset(Dataset):
    """Hard-labelled ordinal dataset using original 1-9 SAM labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        scalers: dict[str, SequenceScaler],
        augment: bool = False,
        sigma: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.user2idx = user2idx
        self.session2idx = session2idx
        self.summary_key_order = summary_key_order

        self.gaze = self._scaled_sequences("gaze", GAZE_SEQ_COLS, scalers)
        self.pupil = self._scaled_sequences("pupil", PUPIL_SEQ_COLS, scalers)
        self.eda = self._scaled_sequences("eda", EDA_SEQ_COLS, scalers)
        self.ppg = self._scaled_sequences("ppg", PPG_SEQ_COLS, scalers)
        self.imu = self._scaled_sequences("imu", IMU_SEQ_COLS, scalers)

        self.summary = np.stack(
            [flatten_physio_summary(row, summary_key_order) for _, row in self.df.iterrows()]
        ).astype(np.float32)
        labels = self.df[list(VAD_DIMS)].to_numpy(dtype=np.float32)
        self.labels = labels
        self.targets, self.weights = hard_ordinal_targets(labels, sigma=sigma)
        self.personality = self.df[BIG_FIVE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        self.uid = np.array(
            [user2idx.get(str(v), 0) for v in self.df["subject_id"]], dtype=np.int64
        )
        self.sid = np.array(
            [session2idx.get(str(v), 0) for v in self.df["session_id"]], dtype=np.int64
        )
        self.task = np.stack(
            [task_onehot(str(v)) for v in self.df.get("task", pd.Series(["T0"] * len(self.df)))]
        ).astype(np.float32)
        self.sex = np.array(
            [
                int(str(v).strip()) if str(v).lstrip("-").isdigit() else -1
                for v in self.df.get("sex", pd.Series(["-1"] * len(self.df)))
            ],
            dtype=np.int64,
        )

    def _scaled_sequences(
        self,
        mod: str,
        cols: list[str],
        scalers: dict[str, SequenceScaler],
    ) -> list[np.ndarray]:
        out = []
        for _, row in self.df.iterrows():
            arr = seq_to_array(row[f"{mod}_seq"], cols)
            out.append(np.clip(scalers[mod].transform(arr), -10.0, 10.0))
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gaze = self.gaze[idx].copy()
        pupil = self.pupil[idx].copy()
        eda = self.eda[idx].copy()
        ppg = self.ppg[idx].copy()
        imu = self.imu[idx].copy()
        if self.augment:
            gaze = self._augment_seq(gaze)
            pupil = self._augment_seq(pupil)
            eda = self._augment_seq(eda)
            ppg = self._augment_seq(ppg)
            imu = self._augment_seq(imu)
        return {
            "gaze": torch.tensor(gaze, dtype=torch.float32),
            "pupil": torch.tensor(pupil, dtype=torch.float32),
            "eda": torch.tensor(eda, dtype=torch.float32),
            "ppg": torch.tensor(ppg, dtype=torch.float32),
            "imu": torch.tensor(imu, dtype=torch.float32),
            "summary": torch.tensor(self.summary[idx], dtype=torch.float32),
            "personality": torch.tensor(self.personality[idx], dtype=torch.float32),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "weights": torch.tensor(self.weights[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "uid": torch.tensor(self.uid[idx], dtype=torch.long),
            "sid": torch.tensor(self.sid[idx], dtype=torch.long),
            "sex": torch.tensor(self.sex[idx], dtype=torch.long),
            "task": torch.tensor(self.task[idx], dtype=torch.float32),
        }

    @staticmethod
    def _augment_seq(arr: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, 0.01, size=arr.shape).astype(np.float32)
        factor = np.random.uniform(0.9, 1.1)
        warped = resample(arr + noise, max(1, int(arr.shape[0] * factor)), axis=0)
        return resample(warped, arr.shape[0], axis=0).astype(np.float32)


class OrdinalGPDataset(Dataset):
    """Augmented GP dataset with ordinal soft cumulative targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        scalers: dict[str, SequenceScaler],
        target_t: int,
        dim_weight_scale: dict[str, float] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.session2idx = session2idx
        self.summary_key_order = summary_key_order
        self.target_t = target_t
        self.dim_weight_scale = dim_weight_scale or {}

        self.gaze = self._scaled_sequences("gaze", GAZE_SEQ_COLS, scalers)
        self.pupil = self._scaled_sequences("pupil", PUPIL_SEQ_COLS, scalers)
        self.eda = self._scaled_sequences("eda", EDA_SEQ_COLS, scalers)
        self.ppg = self._scaled_sequences("ppg", PPG_SEQ_COLS, scalers)
        self.imu = self._scaled_sequences("imu", IMU_SEQ_COLS, scalers)

        self.summary = np.stack(
            [flatten_physio_summary(row, summary_key_order) for _, row in self.df.iterrows()]
        ).astype(np.float32)
        self.targets = np.zeros((len(self.df), 3, 8), dtype=np.float32)
        self.weights = np.zeros((len(self.df), 3), dtype=np.float32)
        for row_idx, row in self.df.iterrows():
            for d_idx, dim in enumerate(VAD_DIMS):
                self.targets[row_idx, d_idx, :] = gp_ordinal_targets(row, dim)
                raw_weight = float(row.get(f"{dim}_weight", 0.0))
                scale = float(self.dim_weight_scale.get(dim, 1.0))
                self.weights[row_idx, d_idx] = float(np.clip(raw_weight * scale, 0.0, 1.0))
        # T4 Dominance: probe never administered — zero weight regardless of pool values
        t4_rows = (self.df["task"].astype(str) == "T4").values
        if t4_rows.any():
            self.weights[t4_rows, 2] = 0.0  # dominance is dim index 2
            log.debug("  OrdinalGPDataset: zeroed dominance weights for %d T4 windows", t4_rows.sum())
        self.labels = np.full((len(self.df), 3), np.nan, dtype=np.float32)
        self.personality = self.df.reindex(columns=BIG_FIVE_COLS).fillna(0.0).to_numpy(
            dtype=np.float32
        )
        self.uid = np.array(
            [user2idx.get(str(v), 0) for v in self.df.get("subject_id", "")],
            dtype=np.int64,
        )
        self.sid = np.array(
            [session2idx.get(str(v), 0) for v in self.df.get("session_id", "")],
            dtype=np.int64,
        )
        self.task = np.stack(
            [task_onehot(str(v)) for v in self.df.get("task", pd.Series(["T0"] * len(self.df)))]
        ).astype(np.float32)
        self.sex = np.full(len(self.df), -1, dtype=np.int64)

    def _scaled_sequences(
        self,
        mod: str,
        cols: list[str],
        scalers: dict[str, SequenceScaler],
    ) -> list[np.ndarray]:
        out = []
        for _, row in self.df.iterrows():
            arr = seq_to_array(row[f"{mod}_seq"], cols)
            if arr.shape[0] != self.target_t:
                arr = resample(arr, self.target_t, axis=0).astype(np.float32)
            out.append(np.clip(scalers[mod].transform(arr), -10.0, 10.0))
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "gaze": torch.tensor(self.gaze[idx], dtype=torch.float32),
            "pupil": torch.tensor(self.pupil[idx], dtype=torch.float32),
            "eda": torch.tensor(self.eda[idx], dtype=torch.float32),
            "ppg": torch.tensor(self.ppg[idx], dtype=torch.float32),
            "imu": torch.tensor(self.imu[idx], dtype=torch.float32),
            "summary": torch.tensor(self.summary[idx], dtype=torch.float32),
            "personality": torch.tensor(self.personality[idx], dtype=torch.float32),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "weights": torch.tensor(self.weights[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "uid": torch.tensor(self.uid[idx], dtype=torch.long),
            "sid": torch.tensor(self.sid[idx], dtype=torch.long),
            "sex": torch.tensor(self.sex[idx], dtype=torch.long),
            "task": torch.tensor(self.task[idx], dtype=torch.float32),
        }


class OrdinalLoss(nn.Module):
    """Masked cumulative BCE plus optional personality auxiliary loss."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        label_smooth: float = 0.0,
        alpha: float = 0.05,
        dim_only: str = "all",
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.label_smooth = label_smooth
        self.alpha = alpha
        self.dim_only = dim_only
        self.personality_loss = nn.SmoothL1Loss()

    def _dim_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        dim_idx: int,
    ) -> torch.Tensor:
        if self.label_smooth > 0:
            targets = targets * (1.0 - self.label_smooth) + 0.5 * self.label_smooth
        pos_w = None if self.pos_weight is None else self.pos_weight[dim_idx].to(logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), pos_weight=pos_w, reduction="none"
        )
        loss = loss.mean(dim=-1)
        total_weight = weights.sum()
        if total_weight < 1e-8:
            return torch.zeros((), device=logits.device)
        return (loss * weights).sum() / total_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        weights: torch.Tensor,
        personality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _zero = torch.zeros((), device=targets.device)
        _active = {"all": {0, 1, 2}, "valence": {0}, "arousal": {1}, "dominance": {2}}
        active = _active.get(self.dim_only, {0, 1, 2})
        loss_v = self._dim_loss(outputs["valence_logits"],   targets[:, 0], weights[:, 0], 0) if 0 in active else _zero
        loss_a = self._dim_loss(outputs["arousal_logits"],   targets[:, 1], weights[:, 1], 1) if 1 in active else _zero
        loss_d = self._dim_loss(outputs["dominance_logits"], targets[:, 2], weights[:, 2], 2) if 2 in active else _zero
        n_active = max(1, len(active))
        emotion = (loss_v + loss_a + loss_d) / n_active
        personality_loss = self.personality_loss(
            outputs["personality_pred"].float(), personality.float()
        )
        total = (1.0 - self.alpha) * emotion + self.alpha * personality_loss
        return {
            "total": total,
            "emotion": emotion,
            "personality": personality_loss,
            "valence": loss_v,
            "arousal": loss_a,
            "dominance": loss_d,
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def expected_scores(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    probs = torch.stack(
        [
            torch.sigmoid(outputs["valence_logits"]),
            torch.sigmoid(outputs["arousal_logits"]),
            torch.sigmoid(outputs["dominance_logits"]),
        ],
        dim=1,
    )
    return 1.0 + probs.sum(dim=-1)


def rounded_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(scores), 1, 9).astype(int)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: OrdinalLoss,
    device: torch.device,
    df: pd.DataFrame | None = None,
    return_preds: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            gaze_seq=batch["gaze"],
            pupil_seq=batch["pupil"],
            eda_seq=batch["eda"],
            ppg_seq=batch["ppg"],
            imu_seq=batch["imu"],
            summary=batch["summary"],
            personality=batch["personality"],
            task_onehot=batch["task"],
        )
        loss = loss_fn(out, batch["targets"], batch["weights"], batch["personality"])
        losses.append(float(loss["total"].detach().cpu()))
        preds.append(expected_scores(out).detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    metrics = compute_metrics(pred_arr, label_arr)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    if df is not None:
        tasks = df["task"].reset_index(drop=True)
        for task_name in sorted(tasks.unique()):
            mask = (tasks == task_name).values
            if mask.sum() < 5:
                continue
            task_m = compute_metrics(pred_arr[mask], label_arr[mask])
            for k, v in task_m.items():
                if k != "cm3_flat" and not k.endswith("_cm3_flat"):
                    metrics[f"{task_name}_{k}"] = v
    if return_preds:
        return metrics, pred_arr, label_arr
    return metrics


_TERTILE_BINS = np.array([3.5, 6.5])  # Low: 1-3, Mid: 4-6, High: 7-9 on 1-9 Likert scale


def compute_metrics(pred_scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    maes: list[float] = []
    spears: list[float] = []
    qwks: list[float] = []
    f1s: list[float] = []
    accs_w1: list[float] = []
    accs_3: list[float] = []
    for d_idx, dim in enumerate(VAD_DIMS):
        y = labels[:, d_idx]
        valid = np.isfinite(y)
        if not np.any(valid):
            for key in ("mae", "spearman", "qwk", "macro_f1", "acc_within_1", "acc_3class"):
                metrics[f"{dim}_{key}"] = float("nan")
            metrics[f"{dim}_cm3_flat"] = [0] * 9
            continue
        y_true = y[valid].astype(int)
        score = pred_scores[valid, d_idx]
        y_pred = rounded_scores(score)
        mae = float(np.mean(np.abs(score - y_true)))
        rho = spearmanr(y_true, score).correlation
        if not np.isfinite(rho):
            rho = 0.0
        qwk = cohen_kappa_score(y_true, y_pred, labels=list(range(1, 10)), weights="quadratic")
        if not np.isfinite(qwk):
            qwk = 0.0
        f1 = f1_score(y_true, y_pred, labels=list(range(1, 10)), average="macro", zero_division=0)
        # Accuracy within 1 Likert step (lenient, appropriate for self-report noise)
        acc_w1 = float(np.mean(np.abs(y_pred - y_true) <= 1))
        # 3-class accuracy using balanced tertile bins on 1-9 scale
        y_true_3 = np.digitize(y_true, _TERTILE_BINS)
        y_pred_3 = np.digitize(y_pred, _TERTILE_BINS)
        acc3 = float(np.mean(y_true_3 == y_pred_3))
        cm3 = confusion_matrix(y_true_3, y_pred_3, labels=[0, 1, 2])
        metrics[f"{dim}_mae"] = mae
        metrics[f"{dim}_spearman"] = float(rho)
        metrics[f"{dim}_qwk"] = float(qwk)
        metrics[f"{dim}_macro_f1"] = float(f1)
        metrics[f"{dim}_acc_within_1"] = acc_w1
        metrics[f"{dim}_acc_3class"] = acc3
        metrics[f"{dim}_cm3_flat"] = cm3.flatten().tolist()
        maes.append(mae)
        spears.append(float(rho))
        qwks.append(float(qwk))
        f1s.append(float(f1))
        accs_w1.append(acc_w1)
        accs_3.append(acc3)
    metrics["mean_mae"] = float(np.mean(maes)) if maes else float("nan")
    metrics["mean_spearman"] = float(np.mean(spears)) if spears else float("nan")
    metrics["mean_qwk"] = float(np.mean(qwks)) if qwks else float("nan")
    metrics["mean_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    metrics["mean_acc_within_1"] = float(np.mean(accs_w1)) if accs_w1 else float("nan")
    metrics["mean_acc_3class"] = float(np.mean(accs_3)) if accs_3 else float("nan")
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: OrdinalLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            gaze_seq=batch["gaze"],
            pupil_seq=batch["pupil"],
            eda_seq=batch["eda"],
            ppg_seq=batch["ppg"],
            imu_seq=batch["imu"],
            summary=batch["summary"],
            personality=batch["personality"],
            task_onehot=batch["task"],
        )
        loss = loss_fn(out, batch["targets"], batch["weights"], batch["personality"])
        loss["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_n = batch["summary"].shape[0]
        for key, value in loss.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_n
        n += batch_n
    return {key: value / max(n, 1) for key, value in totals.items()}


# ─── FixMatch Arousal consistency regularisation ────────────────────────────

def _ordinal_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B,8) cumulative logits → (B,9) class probabilities P(Y=k), k=1..9."""
    cum = torch.sigmoid(logits)
    zeros = torch.zeros(cum.shape[0], 1, device=cum.device)
    ones  = torch.ones(cum.shape[0],  1, device=cum.device)
    cdf   = torch.cat([zeros, cum, ones], dim=1)   # (B, 10)
    return (cdf[:, 1:] - cdf[:, :-1]).clamp(min=1e-8)  # (B, 9)


def _ordinal_3class_conf(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate ordinal predictions into 3-class (Low/Mid/High) at thresholds 3.5/6.5.

    Model logits encode the survival function: sigmoid(logit_k) = P(Y > k).
    Returns (confidence, class_idx) where confidence = max(P_Low, P_Mid, P_High) and
    class_idx ∈ {0=Low, 1=Mid, 2=High}.
    3-class max probability is typically 0.4–0.9, making threshold=0.45 effective.
    """
    surv   = torch.sigmoid(logits)                # P(Y>k) for k=1..8, shape (B,8)
    # P(Y<=3) = 1 - P(Y>3) = 1 - surv[:, 2]  (index 2 = threshold at k=3)
    # P(Y>6)  = surv[:, 5]                    (index 5 = threshold at k=6)
    p_low  = 1.0 - surv[:, 2]                     # P(Y<=3)
    p_high = surv[:, 5]                            # P(Y>6)
    p_mid  = (1.0 - p_low - p_high).clamp(min=0)
    p3     = torch.stack([p_low, p_mid, p_high], dim=1)
    p3     = p3 / p3.sum(dim=1, keepdim=True).clamp(min=1e-8)
    conf    = p3.max(dim=1).values
    cls_idx = p3.argmax(dim=1)
    return conf, cls_idx


# MAP label for each 3-class bin centre (1-3→2, 4-6→5, 7-9→8)
_3CLASS_CENTER = torch.tensor([2.0, 5.0, 8.0])


def _ordinal_confidence(logits: torch.Tensor) -> torch.Tensor:
    """3-class max probability as confidence proxy, shape (B,)."""
    conf, _ = _ordinal_3class_conf(logits)
    return conf


def _make_fixmatch_cumulative_targets(logits: torch.Tensor) -> torch.Tensor:
    """Pseudo-label from 3-class argmax (centres 2/5/8) → hard cumulative (B,8).

    Using 3-class bins instead of 9 individual classes gives meaningful confidence
    scores (typically 0.5–0.9) that are compatible with threshold=0.50.
    """
    _, cls_idx = _ordinal_3class_conf(logits)                  # (B,) in {0,1,2}
    centers    = _3CLASS_CENTER.to(logits.device)
    pseudo     = centers[cls_idx]                              # (B,) values 2/5/8
    cuts       = torch.arange(1, 9, device=logits.device).float()   # [1..8]
    return (pseudo.unsqueeze(1) > cuts.unsqueeze(0)).float()   # (B, 8)


def _augment_batch_strong(
    batch: dict[str, torch.Tensor],
    noise_std: float = 0.10,
    dropout_p: float = 0.20,
) -> dict[str, torch.Tensor]:
    """Strong augmentation: additive Gaussian noise + random feature dropout."""
    aug: dict[str, torch.Tensor] = {}
    for key, val in batch.items():
        if key in ("gaze", "pupil", "eda", "ppg", "imu", "summary") and val.is_floating_point():
            noise = torch.randn_like(val) * noise_std
            mask  = (torch.rand_like(val) > dropout_p).float()
            aug[key] = (val + noise) * mask
        else:
            aug[key] = val
    return aug


def fixmatch_arousal_loss(
    model: nn.Module,
    unlabeled_batch: dict[str, torch.Tensor],
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """FixMatch consistency loss for the Arousal dimension.

    Pass 1 (no_grad): clean features → pseudo arousal label + confidence.
    Pass 2 (grad):    strongly-augmented features → BCE against pseudo label.
    Samples where confidence < threshold are masked out.
    """
    def _forward(b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return model(
            gaze_seq=b["gaze"],
            pupil_seq=b["pupil"],
            eda_seq=b["eda"],
            ppg_seq=b["ppg"],
            imu_seq=b["imu"],
            summary=b["summary"],
            personality=b["personality"],
            task_onehot=b["task"],
        )

    with torch.no_grad():
        out_clean   = _forward(unlabeled_batch)
        a_logits    = out_clean["arousal_logits"]           # (B, 8)
        conf        = _ordinal_confidence(a_logits)         # (B,)
        pseudo_tgt  = _make_fixmatch_cumulative_targets(a_logits)  # (B, 8)

    mask = (conf > threshold).float()
    if mask.sum() < 1:
        return torch.zeros((), device=device)

    aug_batch     = _augment_batch_strong(unlabeled_batch)
    out_strong    = _forward(aug_batch)
    a_logits_str  = out_strong["arousal_logits"]            # (B, 8)
    bce           = F.binary_cross_entropy_with_logits(
        a_logits_str.float(), pseudo_tgt.float(), reduction="none"
    ).mean(dim=-1)                                          # (B,)
    return (bce * mask).sum() / mask.sum()


def train_epoch_with_fixmatch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: "OrdinalLoss",
    device: torch.device,
    fixmatch_lambda: float = 1.0,
    fixmatch_threshold: float = 0.80,
) -> dict[str, float]:
    """Training epoch that adds a FixMatch Arousal consistency term."""
    model.train()
    totals: dict[str, float] = {}
    n = 0
    unlabeled_iter = iter(unlabeled_loader)
    for batch in labeled_loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out = model(
            gaze_seq=batch["gaze"],
            pupil_seq=batch["pupil"],
            eda_seq=batch["eda"],
            ppg_seq=batch["ppg"],
            imu_seq=batch["imu"],
            summary=batch["summary"],
            personality=batch["personality"],
            task_onehot=batch["task"],
        )
        sup_losses = loss_fn(out, batch["targets"], batch["weights"], batch["personality"])
        total_loss = sup_losses["total"]

        # FixMatch Arousal consistency
        try:
            ub = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            ub = next(unlabeled_iter)
        ub = move_batch(ub, device)
        fm_loss = fixmatch_arousal_loss(model, ub, fixmatch_threshold, device)
        total_loss = total_loss + fixmatch_lambda * fm_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_n = batch["summary"].shape[0]
        for key, val in sup_losses.items():
            totals[key] = totals.get(key, 0.0) + float(val.detach().cpu()) * batch_n
        totals["fixmatch"] = totals.get("fixmatch", 0.0) + float(fm_loss.detach().cpu()) * batch_n
        n += batch_n
    return {key: val / max(n, 1) for key, val in totals.items()}

# ─────────────────────────────────────────────────────────────────────────────


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if epochs <= warmup:
            return 1.0
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def task_split(
    df: pd.DataFrame,
    test_task: str = "T3",
    t0_mode: str = "train",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by task.  t0_mode controls T0 treatment: 'train' | 'baseline-only' | 'exclude'."""
    task_order = ["T0", "T1", "T2", "T3", "T4"]
    if test_task not in task_order[2:]:
        raise ValueError("--test-task for task split must be one of T2/T3/T4")
    test_idx = task_order.index(test_task)
    val_task = task_order[test_idx - 1]
    train_tasks = task_order[: test_idx - 1]
    train_df = df[df["task"].isin(train_tasks)].copy()
    val_df = df[df["task"] == val_task].copy()
    test_df = df[df["task"] == test_task].copy()
    if t0_mode in ("baseline-only", "exclude"):
        train_df = train_df[train_df["task"] != "T0"].copy()
    return train_df, val_df, test_df


def per_subject_window_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for _, grp in df.groupby("subject_id"):
        idx = list(grp.index)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])
    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


def subject_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique())
    order = list(rng.permutation(len(subjects)))
    n_test = max(1, int(len(subjects) * test_frac))
    n_val = max(1, int(len(subjects) * val_frac))
    test_subjects = [subjects[i] for i in order[:n_test]]
    val_subjects = [subjects[i] for i in order[n_test : n_test + n_val]]
    train_subjects = [subjects[i] for i in order[n_test + n_val :]]
    return (
        df[df["subject_id"].isin(train_subjects)].copy(),
        df[df["subject_id"].isin(val_subjects)].copy(),
        df[df["subject_id"].isin(test_subjects)].copy(),
    )


def parse_dim_aug_scale(text: str) -> dict[str, float]:
    if not text:
        return {}
    aliases = {"v": "valence", "a": "arousal", "d": "dominance"}
    out: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        dim = aliases.get(key.strip().lower(), key.strip().lower())
        if dim not in VAD_DIMS:
            raise ValueError(f"Unknown dim in --dim-aug-scale: {key}")
        out[dim] = float(value)
    return out


def normalize_personality(
    train_ds: OrdinalLabelDataset,
    datasets: list[Dataset],
) -> None:
    mean = train_ds.personality.mean(axis=0, keepdims=True)
    std = train_ds.personality.std(axis=0, keepdims=True).clip(min=1e-6)
    for ds in datasets:
        if hasattr(ds, "personality"):
            ds.personality = ((ds.personality - mean) / std).astype(np.float32)


def apply_summary_standardization(
    train_ds: OrdinalLabelDataset,
    datasets: list[Dataset],
) -> None:
    mean = train_ds.summary.mean(axis=0, keepdims=True)
    std = train_ds.summary.std(axis=0, keepdims=True).clip(min=1e-6)
    for ds in datasets:
        if hasattr(ds, "summary"):
            ds.summary = ((ds.summary - mean) / std).astype(np.float32)


def build_train_loader(
    train_ds: OrdinalLabelDataset,
    aug_ds: OrdinalGPDataset | None,
    batch_size: int,
    aug_frac: float,
) -> DataLoader:
    if aug_ds is None or len(aug_ds) == 0 or aug_frac <= 0:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    combined = ConcatDataset([train_ds, aug_ds])
    hard_w = np.ones(len(train_ds), dtype=np.float32)
    aug_conf = aug_ds.weights.mean(axis=1).astype(np.float32)
    conf_sum = float(aug_conf.sum()) + 1e-8
    scale = (len(train_ds) * aug_frac) / (conf_sum * max(1.0 - aug_frac, 1e-6))
    soft_w = scale * aug_conf
    all_w = np.concatenate([hard_w, soft_w]).astype(np.float32)
    all_w = all_w / (all_w.max() + 1e-8)
    eff = float(soft_w.sum() / (hard_w.sum() + soft_w.sum() + 1e-8))
    log.info(
        "  GP ordinal augmentation: hard=%d soft=%d target_frac=%.2f effective_frac=%.3f",
        len(train_ds),
        len(aug_ds),
        aug_frac,
        eff,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(all_w),
        num_samples=len(train_ds) + len(aug_ds),
        replacement=True,
    )
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True)


def run_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    fold_tag: str = "",
    aug_pool: pd.DataFrame | None = None,
    return_preds: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
    log.info(
        "%s Train=%d Val=%d Test=%d",
        fold_tag or "split",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    exclude_eda = getattr(args, "exclude_eda", False)

    # Replace SAM arousal with GSR-derived arousal labels when requested
    gsr_labels_path = getattr(args, "gsr_arousal_labels", "")
    if gsr_labels_path:
        gsr_df = pd.read_csv(gsr_labels_path)
        # Support both column names (gsr_arousal from Ridge regression, gsr_physio_arousal from physio script)
        label_col = "gsr_arousal" if "gsr_arousal" in gsr_df.columns else "gsr_physio_arousal"
        gsr_map: dict[int, float] = dict(zip(gsr_df["window_index"].astype(int),
                                              gsr_df[label_col].astype(float)))
        for part in (train_df, val_df, test_df):
            if "_orig_idx" in part.columns:
                part["arousal"] = part["_orig_idx"].map(lambda i: gsr_map.get(int(i), np.nan))
        n_valid = sum(np.isfinite(v) for v in train_df["arousal"])
        log.info("  GSR arousal labels loaded: %d/%d train rows have valid GSR arousal",
                 n_valid, len(train_df))

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2idx = make_user2idx(full_df)
    session2idx = make_session2idx(full_df)
    summary_key_order = build_physio_summary_key_order(full_df)
    if exclude_eda:
        eda_keys = get_eda_summary_keys(full_df)
        summary_key_order = [k for k in summary_key_order if k not in eda_keys]
        log.info("  EDA excluded: summary shrunk to %d features", len(summary_key_order))
    else:
        log.info("  Summary features: %d (physio/eye-tracking only)", len(summary_key_order))

    scalers = fit_scalers(train_df)
    label_sigma = getattr(args, "ordinal_label_sigma", 0.0)
    train_ds = OrdinalLabelDataset(
        train_df, user2idx, session2idx, summary_key_order, scalers,
        augment=args.augment, sigma=label_sigma,
    )
    val_ds = OrdinalLabelDataset(val_df, user2idx, session2idx, summary_key_order, scalers)
    test_ds = OrdinalLabelDataset(test_df, user2idx, session2idx, summary_key_order, scalers)

    aug_ds = None
    if aug_pool is not None and len(aug_pool) > 0:
        # Verify T4 dominance weights are explicitly zero (not just NaN)
        if "T4" in aug_pool["task"].values and "dominance_weight" in aug_pool.columns:
            t4_dom_w = aug_pool.loc[aug_pool["task"] == "T4", "dominance_weight"]
            if not (t4_dom_w == 0.0).all():
                log.warning(
                    "T4 dominance_weight has non-zero values — regenerate pool with updated "
                    "label_augmentation.py to apply TASK_DIM_AVAILABILITY masking."
                )
        if args.task_cv or args.split_mode == "task":
            train_tasks = set(train_df["task"].astype(str).unique())
            fold_aug = aug_pool[aug_pool["task"].astype(str).isin(train_tasks)].copy()
            log.info(
                "  GP pool filtered to train tasks %s: %d / %d",
                sorted(train_tasks),
                len(fold_aug),
                len(aug_pool),
            )
        else:
            test_sessions = set(test_df["session_id"].astype(str).unique())
            fold_aug = aug_pool[~aug_pool["session_id"].astype(str).isin(test_sessions)].copy()
            log.info(
                "  GP pool after test-session exclusion: %d / %d",
                len(fold_aug),
                len(aug_pool),
            )
        if len(fold_aug) > 0:
            dim_scale = parse_dim_aug_scale(args.dim_aug_scale)
            target_t = train_ds.gaze[0].shape[0]
            aug_ds = OrdinalGPDataset(
                fold_aug,
                user2idx,
                session2idx,
                summary_key_order,
                scalers,
                target_t=target_t,
                dim_weight_scale=dim_scale,
            )
        else:
            log.info("  GP augmentation skipped: no leakage-safe pool rows for this split.")

    datasets: list[Dataset] = [train_ds, val_ds, test_ds]
    if aug_ds is not None:
        datasets.append(aug_ds)
    if args.standardize_summary:
        apply_summary_standardization(train_ds, datasets)
    if args.bfi_mode != "none":
        normalize_personality(train_ds, datasets)

    train_loader = build_train_loader(train_ds, aug_ds, args.batch, args.aug_frac)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # FixMatch unlabeled loader: reuses the aug_pool (task-filtered) without GP weights
    use_fixmatch = getattr(args, "fixmatch_arousal", False)
    fixmatch_loader: DataLoader | None = None
    if use_fixmatch:
        if aug_ds is not None and len(aug_ds) > 0:
            fixmatch_loader = DataLoader(
                aug_ds, batch_size=args.batch, shuffle=True, drop_last=True
            )
            log.info(
                "  FixMatch Arousal enabled: %d unlabeled windows, threshold=%.2f, lambda=%.2f",
                len(aug_ds),
                getattr(args, "fixmatch_threshold", 0.80),
                getattr(args, "fixmatch_lambda", 1.0),
            )
        else:
            log.warning("  FixMatch Arousal requested but no aug_pool available — skipped.")

    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = compute_pos_weight(train_ds.targets, train_ds.weights).to(device)
        log.info("  Ordinal pos_weight V=%s", np.round(pos_weight[0].cpu().numpy(), 2).tolist())
        log.info("  Ordinal pos_weight A=%s", np.round(pos_weight[1].cpu().numpy(), 2).tolist())
        log.info("  Ordinal pos_weight D=%s", np.round(pos_weight[2].cpu().numpy(), 2).tolist())

    model = build_ordinal_model(
        arch=args.arch,
        summary_dim=len(summary_key_order),
        hidden=args.hidden,
        dropout=args.dropout,
        bfi_mode=args.bfi_mode,
        use_eda=not exclude_eda,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Model params: %s", f"{n_params:,}")

    loss_fn = OrdinalLoss(
        pos_weight=pos_weight,
        label_smooth=args.label_smooth,
        alpha=args.alpha,
        dim_only=getattr(args, "dim_only", "all"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup)

    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    best_pred_arr: np.ndarray = np.array([])
    best_label_arr: np.ndarray = np.array([])
    patience = 0
    ckpt_path = None
    if args.ckpt_dir:
        ckpt_path = Path(args.ckpt_dir) / f"best_ordinal{fold_tag}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if use_fixmatch and fixmatch_loader is not None:
            tr = train_epoch_with_fixmatch(
                model, train_loader, fixmatch_loader,
                optimizer, loss_fn, device,
                fixmatch_lambda=getattr(args, "fixmatch_lambda", 1.0),
                fixmatch_threshold=getattr(args, "fixmatch_threshold", 0.80),
            )
        else:
            tr = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, loss_fn, device)
            log.info(
                "  ep%03d lr=%.2e tr=%.3f val MAE=%.3f rho=%.3f QWK=%.3f",
                epoch,
                optimizer.param_groups[0]["lr"],
                tr["total"],
                val["mean_mae"],
                val["mean_spearman"],
                val["mean_qwk"],
            )
            if val["mean_mae"] < best_val:
                best_val = val["mean_mae"]
                patience = 0
                best_metrics, best_pred_arr, best_label_arr = evaluate(
                    model, test_loader, loss_fn, device, df=test_df, return_preds=True
                )
                if ckpt_path is not None:
                    torch.save(model.state_dict(), ckpt_path)
            else:
                patience += args.eval_every
                if args.patience > 0 and patience >= args.patience:
                    log.info("  Early stop at epoch %d", epoch)
                    break

    best_metrics["best_val_mae"] = best_val
    best_metrics["n_train"] = float(len(train_df))
    best_metrics["n_val"] = float(len(val_df))
    best_metrics["n_test"] = float(len(test_df))
    log.info(
        "  %s BEST test MAE=%.3f rho=%.3f QWK=%.3f macroF1=%.3f",
        fold_tag or "split",
        best_metrics.get("mean_mae", float("nan")),
        best_metrics.get("mean_spearman", float("nan")),
        best_metrics.get("mean_qwk", float("nan")),
        best_metrics.get("mean_macro_f1", float("nan")),
    )
    if return_preds:
        return best_metrics, best_pred_arr, best_label_arr
    return best_metrics


def run_ensemble_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    fold_tag: str = "",
    aug_pool: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Train one model per VAD dimension with per-dim aug_frac, then ensemble predictions.

    Each dimension uses a model trained with --dim-only <d> and its own aug_frac.
    The ensemble takes V prediction from the V-model, A from A-model, D from D-model.
    This isolates poor-quality GP augmentation (e.g. Arousal) from damaging V/D training.
    """
    import copy

    per_dim_frac = parse_dim_aug_scale(getattr(args, "per_dim_aug_frac", "v=0.3,a=0.0,d=0.3"))
    combined_preds: np.ndarray | None = None
    combined_labels: np.ndarray | None = None

    for d_idx, dim in enumerate(VAD_DIMS):
        dim_aug_frac = per_dim_frac.get(dim, 0.0)
        log.info(
            "=== Ensemble: training %s model (aug_frac=%.2f) fold=%s ===",
            dim, dim_aug_frac, fold_tag or "split",
        )
        dim_args = copy.copy(args)
        dim_args.dim_only = dim
        dim_args.aug_frac = dim_aug_frac
        # Disable aug pool entirely if this dim has no aug
        pool_for_dim = aug_pool if dim_aug_frac > 0 else None

        _, preds, labels = run_split(
            train_df, val_df, test_df,
            dim_args, device,
            fold_tag=f"{fold_tag}_{dim}",
            aug_pool=pool_for_dim,
            return_preds=True,
        )

        if combined_preds is None:
            combined_preds = np.full_like(preds, np.nan)
            combined_labels = labels.copy()
        # Take only this dimension's column from this model
        combined_preds[:, d_idx] = preds[:, d_idx]

    assert combined_preds is not None and combined_labels is not None
    ensemble_metrics = compute_metrics(combined_preds, combined_labels)
    ensemble_metrics["n_train"] = float(len(train_df))
    ensemble_metrics["n_val"] = float(len(val_df))
    ensemble_metrics["n_test"] = float(len(test_df))
    log.info(
        "  Ensemble %s MAE=%.3f rho=%.3f QWK=%.3f acc@1=%.3f acc3=%.3f",
        fold_tag or "split",
        ensemble_metrics.get("mean_mae", float("nan")),
        ensemble_metrics.get("mean_spearman", float("nan")),
        ensemble_metrics.get("mean_qwk", float("nan")),
        ensemble_metrics.get("mean_acc_within_1", float("nan")),
        ensemble_metrics.get("mean_acc_3class", float("nan")),
    )
    return ensemble_metrics


def write_results(path: str, rows: list[dict[str, float | str]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote results: %s", out_path)


def print_summary(rows: list[dict[str, float | str]]) -> None:
    print("\n" + "=" * 98)
    print(f"{'Ordinal Original-Label Results':^98}")
    print("=" * 98)
    print(f"{'Fold':<10} {'MAE':>7} {'rho':>7} {'QWK':>7} {'macroF1':>9} {'acc@1':>7} {'acc3':>7}")
    print("-" * 98)
    for row in rows:
        print(
            f"{str(row.get('fold', 'split')):<10} "
            f"{float(row.get('mean_mae', float('nan'))):>7.3f} "
            f"{float(row.get('mean_spearman', float('nan'))):>7.3f} "
            f"{float(row.get('mean_qwk', float('nan'))):>7.3f} "
            f"{float(row.get('mean_macro_f1', float('nan'))):>9.3f} "
            f"{float(row.get('mean_acc_within_1', float('nan'))):>7.3f} "
            f"{float(row.get('mean_acc_3class', float('nan'))):>7.3f}"
        )
    if len(rows) > 1:
        for metric in ["mean_mae", "mean_spearman", "mean_qwk", "mean_macro_f1",
                       "mean_acc_within_1", "mean_acc_3class"]:
            vals = np.array([float(r.get(metric, np.nan)) for r in rows], dtype=float)
            print(f"{metric}: mean={np.nanmean(vals):.3f} std={np.nanstd(vals):.3f}")
    print("=" * 98)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train standalone ordinal VAD models on original 1-9 SAM labels."
    )
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--augmented-pool", default="", help="Optional GP augmented_pool.pkl")
    parser.add_argument("--aug-frac", type=float, default=0.0)
    parser.add_argument(
        "--dim-aug-scale",
        default="v=1.0,a=0.2,d=0.6",
        help="Per-dimension GP weight scale, e.g. v=1,a=0.2,d=0.6",
    )
    parser.add_argument("--ckpt-dir", default="data/mumt/ordinal_checkpoints")
    parser.add_argument("--output-csv", default="results/ordinal_results.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--label-smooth", type=float, default=0.05)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--arch", choices=["mlp", "pool"], default="pool")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--bfi-mode",
        choices=["none", "concat", "perdim", "taskadapt"],
        default="none",
    )
    parser.add_argument("--standardize-summary", action="store_true", default=True)
    parser.add_argument(
        "--split-mode",
        choices=["task", "per_subject", "subject"],
        default="task",
    )
    parser.add_argument("--test-task", choices=["T2", "T3", "T4"], default="T3")
    parser.add_argument("--task-cv", action="store_true")
    parser.add_argument(
        "--t0-mode",
        choices=["train", "baseline-only", "exclude"],
        default="train",
        help="How to treat T0 (rest) windows: 'train' (default) includes T0 as training data; "
             "'baseline-only' uses T0 only for per-seat physiological normalisation; "
             "'exclude' ignores T0 entirely.",
    )
    parser.add_argument(
        "--dim-only",
        choices=["all", "valence", "arousal", "dominance"],
        default="all",
        help="Train loss for only one VAD dimension. 'all' (default) trains all three jointly. "
             "Use to separate strong augmentation signals (V, D) from weak ones (A).",
    )
    parser.add_argument(
        "--per-dim-ensemble",
        action="store_true",
        default=False,
        help="Train one model per VAD dimension with per-dim aug_frac (see --per-dim-aug-frac), "
             "then ensemble by taking each dim's prediction from its own model. "
             "Isolates poor GP augmentation (Arousal, LOO<chance) from damaging V/D training.",
    )
    parser.add_argument(
        "--per-dim-aug-frac",
        default="v=0.3,a=0.0,d=0.3",
        help="Per-dimension aug_frac for --per-dim-ensemble, e.g. 'v=0.3,a=0.0,d=0.3'. "
             "Valence/Dominance use GP augmentation; Arousal defaults to 0.0 (GP below chance).",
    )
    parser.add_argument(
        "--ordinal-label-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing bandwidth for ordinal label distributions (0=hard targets). "
             "sigma=1.0 applies one-Likert-step unimodal smoothing (Wen et al., ICCV 2023).",
    )
    parser.add_argument(
        "--fixmatch-arousal",
        action="store_true",
        default=False,
        help="Add FixMatch consistency regularisation for Arousal using the unlabeled aug pool. "
             "Clean forward → pseudo-label; strong-aug forward → consistency BCE. "
             "Replaces GP soft labels for Arousal (GP LOO below chance).",
    )
    parser.add_argument(
        "--fixmatch-threshold",
        type=float,
        default=0.45,
        help="3-class confidence threshold for FixMatch pseudo-label inclusion "
             "(max of P_Low/P_Mid/P_High at tertile boundaries 3.5/6.5; typical range 0.4–0.8).",
    )
    parser.add_argument(
        "--fixmatch-lambda",
        type=float,
        default=1.0,
        help="Weight of the FixMatch consistency loss relative to the supervised ordinal loss.",
    )
    parser.add_argument(
        "--gsr-arousal-labels",
        default="",
        help="CSV produced by gsr_arousal_labels.py. When set, replaces the SAM arousal column "
             "with GSR-derived (physiological) arousal labels for all splits.",
    )
    parser.add_argument(
        "--exclude-eda",
        action="store_true",
        default=False,
        help="Remove EDA/GSR features from model inputs (sequences + summary). "
             "Use together with --gsr-arousal-labels to train purely on non-EDA modalities "
             "predicting physiological arousal.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log.info("Device: %s", device)

    df = pd.read_pickle(args.dataset)
    df["_orig_idx"] = np.arange(len(df))  # stable row index for GSR label matching
    log.info(
        "Loaded %d windows | %d subjects | %d sessions",
        len(df),
        df["subject_id"].nunique(),
        df["session_id"].nunique(),
    )
    log.info(
        "Original-label NaNs V=%d A=%d D=%d",
        int(df["valence"].isna().sum()),
        int(df["arousal"].isna().sum()),
        int(df["dominance"].isna().sum()),
    )

    # For per-dim ensemble, load the aug pool if any dim has aug_frac > 0
    _per_dim_fracs = parse_dim_aug_scale(getattr(args, "per_dim_aug_frac", "v=0.3,a=0.0,d=0.3"))
    _need_aug_pool = (
        args.aug_frac > 0
        or (getattr(args, "per_dim_ensemble", False) and any(v > 0 for v in _per_dim_fracs.values()))
    )
    aug_pool = None
    if args.augmented_pool:
        aug_path = Path(args.augmented_pool)
        if aug_path.exists() and _need_aug_pool:
            aug_pool = pd.read_pickle(aug_path)
            log.info("Loaded GP augmented pool: %d windows", len(aug_pool))
            log.info("Using GP ordinal targets from mu/sigma; stored 3-class labels ignored.")
        else:
            log.warning("Augmented pool disabled or not found: %s", aug_path)

    _split_fn = run_ensemble_split if getattr(args, "per_dim_ensemble", False) else run_split
    t0_mode = getattr(args, "t0_mode", "train")
    rows: list[dict[str, float | str]] = []
    if args.task_cv:
        for task in ["T2", "T3", "T4"]:
            log.info("=== Ordinal task-CV fold test=%s ===", task)
            train_df, val_df, test_df = task_split(df, task, t0_mode=t0_mode)
            metrics = _split_fn(
                train_df,
                val_df,
                test_df,
                args,
                device,
                fold_tag=f"_{task}",
                aug_pool=aug_pool,
            )
            metrics["fold"] = task
            rows.append(metrics)
    else:
        if args.split_mode == "task":
            train_df, val_df, test_df = task_split(df, args.test_task, t0_mode=t0_mode)
            fold = args.test_task
        elif args.split_mode == "per_subject":
            train_df, val_df, test_df = per_subject_window_split(df, seed=args.seed)
            fold = "per_subject"
        else:
            train_df, val_df, test_df = subject_split(df, seed=args.seed)
            fold = "subject"
        metrics = _split_fn(
            train_df,
            val_df,
            test_df,
            args,
            device,
            fold_tag=f"_{fold}",
            aug_pool=aug_pool,
        )
        metrics["fold"] = fold
        rows.append(metrics)

    print_summary(rows)
    write_results(args.output_csv, rows)


if __name__ == "__main__":
    main()
