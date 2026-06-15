"""train_simple.py

Single-phase supervised training of MuMTAffectGroupAffect on dataset.pkl.
No SSL pretraining — trains directly on the 292 labeled windows.

Features
--------
* Per-modality StandardScaler fitted on training sequences.
* Data-driven tertile binning for balanced VAD classes.
* Per-dimension inverse-frequency class weights.
* NaN-masked dominance loss (46 / 292 windows have no dominance label).
* Subject-level train / val / test split (default) or LOGO-CV (--logo-cv).
* AdamW + cosine-annealing with warm-up.
* Saves best checkpoint by mean-val macro-F1.

Usage
-----
  python tools/mumt/train_simple.py --dataset data/mumt/dataset.pkl
  python tools/mumt/train_simple.py --dataset data/mumt/dataset.pkl --logo-cv
  python tools/mumt/train_simple.py --dataset data/mumt/dataset.pkl \\
      --epochs 300 --lr 3e-4 --batch 16 --d-enc 32 --d-fuse 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

# ── make local imports work regardless of cwd ─────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import (
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    GroupAffectDataset,
    build_summary_key_order,
    make_session2idx,
    make_user2idx,
    seq_to_array,
)
from model_affectai import MuMTAffectGroupAffect, MuMTAffectLoss
from model_simple import build_simple_model, MODALITY_DIMS

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODALITY_NAMES = ["gaze", "pupil", "eda", "ppg", "imu"]
MODALITY_COLS  = {
    "gaze":  GAZE_SEQ_COLS,
    "pupil": PUPIL_SEQ_COLS,
    "eda":   EDA_SEQ_COLS,
    "ppg":   PPG_SEQ_COLS,
    "imu":   IMU_SEQ_COLS,
}
VAD_DIMS = ["valence", "arousal", "dominance"]


# ── Sequence scaler ──────────────────────────────────────────────────────────

class SequenceScaler:
    """Per-feature StandardScaler that operates on (T, F) arrays.

    Fits on a list of (T, F) arrays from the training set by concatenating
    them along the time axis before fitting, so global per-feature statistics
    are used (not per-window stats).
    """

    def __init__(self) -> None:
        self._sk = StandardScaler()

    def fit(self, arrays: list[np.ndarray]) -> "SequenceScaler":
        """arrays: list of (T, F) float32 arrays."""
        stacked = np.concatenate(arrays, axis=0)  # (N*T, F)
        self._sk.fit(stacked)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """arr: (T, F) → (T, F) scaled float32."""
        return self._sk.transform(arr).astype(np.float32)


def fit_scalers(train_df: pd.DataFrame) -> dict[str, SequenceScaler]:
    """Fit one SequenceScaler per modality from the training split."""
    scalers: dict[str, SequenceScaler] = {}
    for mod, cols in MODALITY_COLS.items():
        arrs = [seq_to_array(row[f"{mod}_seq"], cols) for _, row in train_df.iterrows()]
        sc = SequenceScaler().fit(arrs)
        scalers[mod] = sc
    return scalers


# ── Class-weight helpers ──────────────────────────────────────────────────────

def compute_tertile_thresholds(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute per-dimension balanced tertile thresholds from training labels.

    For integer-scale (Likert) data, clusters of tied values at the boundary can
    make a naive np.percentile threshold collapse into the wrong class.  This
    function instead places the threshold at the *midpoint between the boundary
    group and the preceding group*, so:

        Low  = vals < t1    (~n/3 items)
        Mid  = t1 <= vals < t2  (~n/3 items)
        High = vals >= t2   (~n/3 items)

    Example (valence, 103 windows): {3:1, 5:10, 6:19, 7:36, 8:23, 9:14}
    - Rank boundary at n//3 = 34 falls inside the "7" cluster.
    - Midpoint between preceding group (6) and boundary group (7) -> t1 = 6.5
    - Result: Low=30 (29%), Mid=36 (35%), High=37 (36%) -- near-balanced.
    """
    thresholds: dict[str, tuple[float, float]] = {}
    for dim in VAD_DIMS:
        vals = np.sort(train_df[dim].dropna().values.astype(float))
        n = len(vals)

        def _midpoint_threshold(idx: int) -> float:
            """Return fractional midpoint below vals[idx]'s group."""
            boundary_val = vals[idx]
            # Find largest unique value strictly below the boundary group
            below = vals[vals < boundary_val]
            if len(below) > 0:
                return (below[-1] + boundary_val) / 2.0
            # All values are the same; shift slightly so strict < works
            return boundary_val - 0.5

        idx1 = n // 3           # first index of the boundary group for Low/Mid split
        idx2 = (2 * n) // 3     # first index of the boundary group for Mid/High split
        t1 = _midpoint_threshold(idx1)
        t2 = _midpoint_threshold(idx2)
        # Guard: t2 must be strictly greater than t1 (degenerate when both boundaries
        # fall inside the same cluster, e.g. all 7s span 33%–67%).
        if t2 <= t1:
            # Push t2 to the midpoint between t1's cluster and the NEXT unique value above it
            above = vals[vals > (t1 + 0.5)]   # values strictly above the t1 cluster
            t2 = (t1 + above[0]) / 2.0 if len(above) > 0 else t1 + 1.0
        thresholds[dim] = (t1, t2)

        # Log actual class counts for verification (using <= to match bin_vad_from_thresholds)
        low  = int(np.sum(vals <= t1))
        mid  = int(np.sum((vals > t1) & (vals <= t2)))
        high = int(np.sum(vals >  t2))
        log.info(
            "  %s thresholds: (%.2f, %.2f)  ->  Low=%d Mid=%d High=%d (n=%d)",
            dim, t1, t2, low, mid, high, n,
        )
    return thresholds


def bin_vad_from_thresholds(value: float, t1: float, t2: float) -> int:
    """Map raw Likert score to 0/1/2 using tertile thresholds. NaN -> 1 (Mid).

    Uses non-strict (<=) comparison, which is equivalent to strict (<) for
    data-driven thresholds because those are fractional midpoints (e.g. 6.5)
    that never coincide with integer Likert values.
    Non-strict semantics also preserve the intuitive meaning of fixed thresholds
    (3.0, 6.0): v=3 -> Low, v=6 -> Mid, as intended.
    """
    if np.isnan(value):
        return 1
    v = float(value)
    if v <= t1:
        return 0
    if v <= t2:
        return 1
    return 2


def compute_class_weights(
    train_df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Inverse-frequency class weights per VAD dimension."""
    weights: dict[str, torch.Tensor] = {}
    for dim in VAD_DIMS:
        t1, t2 = thresholds[dim]
        bins = [bin_vad_from_thresholds(v, t1, t2) for v in train_df[dim].values]
        counts = np.bincount(bins, minlength=3).astype(float)
        counts = np.clip(counts, 1, None)
        w = 1.0 / counts
        w = w / w.sum() * 3.0   # normalise so mean weight = 1
        log.info("  %s class counts: %s  weights: %s", dim, counts.astype(int), w.round(3))
        weights[dim] = torch.tensor(w, dtype=torch.float32, device=device)
    return weights


# ── Soft VAD loss ─────────────────────────────────────────────────────────────

import torch.nn.functional as _F


def _soft_ce(
    logits: torch.Tensor,              # (B, 3)
    soft: torch.Tensor,                # (B, 3) — soft label distribution
    weight: torch.Tensor,              # (B,)   — per-sample instance weight (0 = skip)
    class_weight: torch.Tensor | None = None,  # (3,) — inverse-freq class weights
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Weighted soft cross-entropy for one VAD dimension.

    Supports:
      - Per-sample instance weights (0 = exclude; hard NaN dominance, low-conf augmented)
      - Per-class weights (inverse-frequency; restores minority-class gradient)
      - Label smoothing on top of soft labels (blends toward uniform)
    """
    # Apply label smoothing: blend soft distribution toward uniform
    if label_smoothing > 0:
        K = soft.shape[-1]
        soft = (1.0 - label_smoothing) * soft + label_smoothing / K

    log_p = _F.log_softmax(logits.float(), dim=-1)   # (B, 3)

    if class_weight is not None:
        # Upweight minority-class gradients: scale each class component by w_c
        # For one-hot labels this is exactly CrossEntropyLoss(weight=class_weight)
        cw = class_weight.to(logits.device)           # (3,)
        ce = -(soft * cw * log_p).sum(dim=-1)         # (B,)  — not normalized by sum(w_c)
    else:
        ce = -(soft * log_p).sum(dim=-1)              # (B,)

    total_weight = weight.sum()
    if total_weight < 1e-8:
        return torch.tensor(0.0, device=logits.device)
    return (weight * ce).sum() / total_weight


class SoftVADLoss(nn.Module):
    """Unified soft-CE VAD loss for both hard-labeled and soft-labeled windows.

    Restores the two properties from the original MaskedVADLoss that are
    critical for low-data regimes:
      - label_smoothing (default 0.1): prevents overconfidence, regularises
      - class_weights: inverse-frequency weighting keeps minority classes in play

    Hard-labeled windows:
      vad_soft[:, d, label] = 1.0  (one-hot),  vad_weight[:, d] = 1.0
      Dominance NaN:  vad_weight[:, 2] = 0.0   (excluded from loss)

    Soft-labeled (augmented) windows:
      vad_soft[:, d, :] = GP posterior,  vad_weight[:, d] = GP confidence ∈ (0,1]

    Both types can be freely mixed in the same batch.
    """

    def __init__(
        self,
        class_weights: dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.1,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.smooth = label_smoothing
        self.cw_v   = class_weights["valence"]   if class_weights else None
        self.cw_a   = class_weights["arousal"]   if class_weights else None
        self.cw_d   = class_weights["dominance"] if class_weights else None
        self.p_loss = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        vad_labels: torch.Tensor,          # (B, 3) long  — kept for evaluate() compat
        personality_labels: torch.Tensor,  # (B, 5)
        vad_soft: torch.Tensor,            # (B, 3, 3)
        vad_weight: torch.Tensor,          # (B, 3)
    ) -> dict[str, torch.Tensor]:
        l_v = _soft_ce(outputs["valence_logits"],   vad_soft[:, 0], vad_weight[:, 0],
                       self.cw_v, self.smooth)
        l_a = _soft_ce(outputs["arousal_logits"],   vad_soft[:, 1], vad_weight[:, 1],
                       self.cw_a, self.smooth)
        l_d = _soft_ce(outputs["dominance_logits"], vad_soft[:, 2], vad_weight[:, 2],
                       self.cw_d, self.smooth)

        l_e = (l_v + l_a + l_d) / 3.0
        l_p = self.p_loss(outputs["personality_pred"].float(), personality_labels.float())
        total = self.alpha * l_p + (1.0 - self.alpha) * l_e

        return {"total": total, "valence": l_v, "arousal": l_a, "dominance": l_d,
                "emotion": l_e, "personality": l_p}


# Keep the old name as an alias so callers that still reference it don't break.
MaskedVADLoss = SoftVADLoss


# ── Dataset builder ───────────────────────────────────────────────────────────

def make_dataset(
    df: pd.DataFrame,
    user2idx: dict[str, int],
    session2idx: dict[str, int],
    summary_key_order: list[str],
    scalers: dict[str, SequenceScaler],
    thresholds: dict[str, tuple[float, float]],
    augment: bool,
    device: torch.device,
) -> GroupAffectDataset:
    return GroupAffectDataset(
        df=df,
        user2idx=user2idx,
        modality_scalers=scalers,
        augment=augment,
        device=device,
        summary_key_order=summary_key_order,
        vad_thresholds=thresholds,
        session2idx=session2idx,
    )


def collate_vad_with_nan(batch):
    """Custom collate: replaces NaN dominance bin with -1 (masked in loss)."""
    return torch.utils.data.dataloader.default_collate(batch)


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MaskedVADLoss,
    device: torch.device,
    thresholds: dict[str, tuple[float, float]],
    df_split: pd.DataFrame,
) -> dict:
    """Return per-dim macro-F1, accuracy, and mean loss."""
    model.eval()
    all_logits: dict[str, list] = {d: [] for d in VAD_DIMS}
    all_labels: dict[str, list] = {d: [] for d in VAD_DIMS}
    losses = []

    # Recompute true labels using the same thresholds for evaluation
    true_bins: dict[str, np.ndarray] = {}
    for dim in VAD_DIMS:
        t1, t2 = thresholds[dim]
        true_bins[dim] = np.array(
            [bin_vad_from_thresholds(v, t1, t2) for v in df_split[dim].values]
        )

    for batch in loader:
        (gaze, pupil, eda, ppg, imu, personality,
         emotion_bins, uid, summary, sex, task, sid,
         vad_soft, vad_weight) = batch

        if getattr(loss_fn, "_no_subject_embed", False):
            uid = torch.zeros_like(uid)

        out = model(
            gaze_seq=gaze.float(),
            pupil_seq=pupil.float(),
            eda_seq=eda.float(),
            ppg_seq=ppg.float(),
            imu_seq=imu.float(),
            summary=summary.float(),
            personality=personality.float(),
            user_ids=uid,
            task_onehot=task.float(),
        )

        loss_d = loss_fn(
            out, emotion_bins, personality.float(),
            vad_soft.float(), vad_weight.float(),
        )
        losses.append(loss_d["total"].item())

        for i, dim in enumerate(VAD_DIMS):
            key = f"{dim}_logits"
            preds = out[key].argmax(dim=-1).cpu().numpy()
            lbls  = emotion_bins[:, i].cpu().numpy()
            all_logits[dim].extend(preds.tolist())
            all_labels[dim].extend(lbls.tolist())

    metrics: dict = {"loss": float(np.mean(losses))}
    f1s = []
    for dim in VAD_DIMS:
        preds = np.array(all_logits[dim])
        lbls  = np.array(all_labels[dim])
        # For dominance: mask -1 sentinels
        if dim == "dominance":
            valid = lbls >= 0
            preds_v, lbls_v = preds[valid], lbls[valid]
        else:
            preds_v, lbls_v = preds, lbls

        if len(lbls_v) > 0:
            f1 = f1_score(lbls_v, preds_v, average="macro", zero_division=0)
            acc = accuracy_score(lbls_v, preds_v)
        else:
            f1 = 0.0
            acc = 0.0
        metrics[f"{dim}_f1"]  = float(f1)
        metrics[f"{dim}_acc"] = float(acc)
        f1s.append(f1)
    metrics["mean_f1"] = float(np.mean(f1s))
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SoftVADLoss,
    device: torch.device,
    amp_scaler: torch.amp.GradScaler | None = None,
    no_subject_embed: bool = False,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0

    for batch in loader:
        (gaze, pupil, eda, ppg, imu, personality,
         emotion_bins, uid, summary, sex, task, sid,
         vad_soft, vad_weight) = batch

        if no_subject_embed:
            uid = torch.zeros_like(uid)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(amp_scaler is not None)):
            out = model(
                gaze_seq=gaze.float(),
                pupil_seq=pupil.float(),
                eda_seq=eda.float(),
                ppg_seq=ppg.float(),
                imu_seq=imu.float(),
                summary=summary.float(),
                personality=personality.float(),
                user_ids=uid,
                task_onehot=task.float(),
            )
            loss_d = loss_fn(
                out, emotion_bins, personality.float(),
                vad_soft.float(), vad_weight.float(),
            )

        if amp_scaler is not None:
            amp_scaler.scale(loss_d["total"]).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss_d["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        B = gaze.size(0)
        for k, v in loss_d.items():
            totals[k] = totals.get(k, 0.0) + float(v) * B
        n += B

    return {k: v / n for k, v in totals.items()}


# ── NaN-aware label binner for GroupAffectDataset ────────────────────────────

def build_vad_labels_with_nan(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return (N, 3) int array. Dominance NaN → -1 sentinel."""
    labels = np.zeros((len(df), 3), dtype=np.int64)
    for col_idx, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_idx, val in enumerate(df[dim].values):
            if col_idx == 2 and np.isnan(float(val) if val is not None else float("nan")):
                labels[row_idx, col_idx] = -1
            else:
                labels[row_idx, col_idx] = bin_vad_from_thresholds(
                    float(val) if val is not None else float("nan"), t1, t2
                )
    return labels


# ── Session-relative normalisation ───────────────────────────────────────────

def compute_session_summary_stats(
    df: pd.DataFrame,
    summary_key_order: list[str],
    scalers: dict[str, SequenceScaler],
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Compute per-(session_id, seat) mean and std of summary features.

    Returns {(session_id, seat): (mean_vec, std_vec)} using already-scaled
    summary features so the z-score is applied on top of global scaling.
    """
    from dataset_affectai import flatten_features

    stats: dict = {}
    for (ses, seat), grp in df.groupby(["session_id", "seat"]):
        vecs = []
        for _, r in grp.iterrows():
            feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features", "audio_features", "speech_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    feats.update(fd)
            vecs.append(flatten_features(feats, key_order=summary_key_order))
        arr = np.stack(vecs, axis=0)          # (N_win, K)
        mu  = arr.mean(axis=0)
        std = arr.std(axis=0).clip(min=1e-6)
        stats[(str(ses), str(seat))] = (mu, std)
    return stats


# ── Custom Dataset with NaN-safe dominance ───────────────────────────────────

from torch.utils.data import Dataset as _DS

class VADDataset(_DS):
    """Lightweight dataset: returns modality sequences + VAD labels.

    Avoids the complexity of GroupAffectDataset while supporting NaN masking.
    Returns a 12-tuple identical to GroupAffectDataset.__getitem__ so the same
    training loop works for both.

    group_norm: if True, z-score summary features within each (session, seat)
    using statistics computed from *that split's own windows* (no label leakage).
    For test splits this uses the test session's own physiology baseline, which
    is standard in physiological computing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        scalers: dict[str, SequenceScaler],
        thresholds: dict[str, tuple[float, float]],
        augment: bool = False,
        device: torch.device | None = None,
        group_norm: bool = False,
        session_stats: dict | None = None,   # pre-computed stats (from train split)
    ) -> None:
        from dataset_affectai import (
            flatten_features, task_onehot as _task_onehot, noise_injection, time_warp
        )
        from scipy.signal import resample as _resample

        self.device  = device or torch.device("cpu")
        self.augment = augment
        self._noise  = noise_injection
        self._warp   = time_warp
        self._resamp = _resample
        self._task_oh = _task_onehot
        self._flat    = flatten_features
        self.user2idx = user2idx
        self.session2idx = session2idx
        self.key_order = summary_key_order

        # Materialise everything eagerly (292 windows — trivial memory)
        self._gaze   = [seq_to_array(r["gaze_seq"],  GAZE_SEQ_COLS)  for _, r in df.iterrows()]
        self._pupil  = [seq_to_array(r["pupil_seq"], PUPIL_SEQ_COLS) for _, r in df.iterrows()]
        self._eda    = [seq_to_array(r["eda_seq"],   EDA_SEQ_COLS)   for _, r in df.iterrows()]
        self._ppg    = [seq_to_array(r["ppg_seq"],   PPG_SEQ_COLS)   for _, r in df.iterrows()]
        self._imu    = [seq_to_array(r["imu_seq"],   IMU_SEQ_COLS)   for _, r in df.iterrows()]

        # Apply scalers and clip outliers
        for i in range(len(df)):
            self._gaze[i]  = np.clip(scalers["gaze"].transform(self._gaze[i]),   -10, 10)
            self._pupil[i] = np.clip(scalers["pupil"].transform(self._pupil[i]), -10, 10)
            self._eda[i]   = np.clip(scalers["eda"].transform(self._eda[i]),     -10, 10)
            self._ppg[i]   = np.clip(scalers["ppg"].transform(self._ppg[i]),     -10, 10)
            self._imu[i]   = np.clip(scalers["imu"].transform(self._imu[i]),     -10, 10)

        # Summary features (physio + audio + sync)
        self._summary = []
        for _, r in df.iterrows():
            feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features", "audio_features", "speech_features",
                        "sync_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    feats.update(fd)
            self._summary.append(flatten_features(feats, key_order=summary_key_order))

        # Session-relative normalisation: z-score summary within (session, seat)
        if group_norm:
            # Use provided stats (train split stats) or compute from this split
            if session_stats is None:
                session_stats = {}
                for (ses, seat), grp_idx in df.groupby(["session_id", "seat"]).groups.items():
                    vecs = np.stack([self._summary[i] for i in range(len(df))
                                     if df.index[i] in grp_idx or True], axis=0)
                    # Rebuild per-group index properly
                    pass
                # Simpler: compute from df row indices
                session_stats = {}
                df_reset = df.reset_index(drop=True)
                for (ses, seat), grp in df_reset.groupby(["session_id", "seat"]):
                    idxs = list(grp.index)
                    vecs = np.stack([self._summary[i] for i in idxs], axis=0)
                    session_stats[(str(ses), str(seat))] = (
                        vecs.mean(axis=0), vecs.std(axis=0).clip(min=1e-6)
                    )
            # Apply z-score per window
            df_reset = df.reset_index(drop=True)
            for i, r in df_reset.iterrows():
                key = (str(r.get("session_id", "")), str(r.get("seat", "")))
                if key in session_stats:
                    mu, std = session_stats[key]
                    self._summary[i] = ((self._summary[i] - mu) / std).astype(np.float32)

        self._session_stats = session_stats  # expose so test split can reuse

        # VAD labels — dominance NaN → -1
        self._vad     = build_vad_labels_with_nan(df, thresholds)

        # Soft labels (one-hot) + instance weights for hard-labeled windows.
        # Dominance NaN (label == -1) → weight = 0.0 (excluded from loss).
        N = len(df)
        self._vad_soft   = np.zeros((N, 3, 3), dtype=np.float32)
        self._vad_weight = np.ones((N, 3),     dtype=np.float32)
        for i in range(N):
            for d in range(3):
                lbl = int(self._vad[i, d])
                if lbl >= 0:
                    self._vad_soft[i, d, lbl] = 1.0
                else:
                    # NaN dominance: zero weight → excluded from loss
                    self._vad_soft[i, d, 1] = 1.0   # placeholder Mid
                    self._vad_weight[i, d] = 0.0

        self._uid     = np.array([user2idx.get(str(r["subject_id"]), 0) for _, r in df.iterrows()], dtype=np.int64)
        self._sid     = np.array([session2idx.get(str(r.get("session_id","")), 0) for _, r in df.iterrows()], dtype=np.int64)
        self._task    = np.stack([_task_onehot(str(r.get("task","T0"))) for _, r in df.iterrows()])
        self._sex     = np.array([int(str(r.get("sex","-1")).strip()) if str(r.get("sex","-1")).lstrip("-").isdigit()
                                  else -1 for _, r in df.iterrows()], dtype=np.int64)
        self._pers    = np.array([[float(r.get(c, 0.0)) if not pd.isna(r.get(c, float("nan"))) else 0.0
                                   for c in BIG_FIVE_COLS] for _, r in df.iterrows()], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._gaze)

    def __getitem__(self, idx: int):
        dev = self.device
        gaze  = self._gaze[idx].copy()
        pupil = self._pupil[idx].copy()
        eda   = self._eda[idx].copy()
        ppg   = self._ppg[idx].copy()
        imu   = self._imu[idx].copy()

        if self.augment:
            T = gaze.shape[0]
            from scipy.signal import resample as _r
            gaze  = _r(self._noise(self._warp(gaze)),  T, axis=0).astype(np.float32)
            pupil = _r(self._noise(self._warp(pupil)), T, axis=0).astype(np.float32)
            eda   = _r(self._noise(self._warp(eda)),   T, axis=0).astype(np.float32)
            ppg   = _r(self._noise(self._warp(ppg)),   T, axis=0).astype(np.float32)
            imu   = _r(self._noise(self._warp(imu)),   T, axis=0).astype(np.float32)

        return (
            torch.tensor(gaze,           device=dev),                            # 0
            torch.tensor(pupil,          device=dev),                            # 1
            torch.tensor(eda,            device=dev),                            # 2
            torch.tensor(ppg,            device=dev),                            # 3
            torch.tensor(imu,            device=dev),                            # 4
            torch.tensor(self._pers[idx], device=dev),                          # 5
            torch.tensor(self._vad[idx],  dtype=torch.long, device=dev),        # 6 — (3,) vad bins
            torch.tensor(self._uid[idx],  dtype=torch.long, device=dev),        # 7
            torch.tensor(self._summary[idx], device=dev),                       # 8
            torch.tensor(self._sex[idx],  dtype=torch.long, device=dev),        # 9
            torch.tensor(self._task[idx], device=dev),                          # 10
            torch.tensor(self._sid[idx],  dtype=torch.long, device=dev),        # 11
            torch.tensor(self._vad_soft[idx],   device=dev),                    # 12 — (3, 3) soft labels
            torch.tensor(self._vad_weight[idx], device=dev),                    # 13 — (3,) instance weights
        )


# ── Augmented soft-label dataset ─────────────────────────────────────────────

class AugSoftDataset(_DS):
    """Dataset wrapping augmented_pool.pkl windows with GP-posterior soft labels.

    Each sample uses the same 14-tuple format as VADDataset so both can be
    concatenated with WeightedRandomSampler and processed by the same training loop.

    Leakage guarantee: the test session is excluded by the caller before
    constructing this dataset.
    """

    def __init__(
        self,
        aug_df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        scalers: dict[str, SequenceScaler],
        device: torch.device | None = None,
        target_T: int | None = None,  # resample sequences to this length if not None
        dim_weight_scale: dict[str, float] | None = None,
        # Per-dimension multiplier on GP instance weights.
        # Motivation: arousal GP accuracy is 0.30 (below chance) because the OU
        # kernel with sparse task-level observations collapses to the prior mean
        # for fast-reverting dimensions (theta=0.578). Downscaling arousal weights
        # reduces its contribution without eliminating it entirely.
        # Example: {"valence": 1.0, "arousal": 0.2, "dominance": 0.6}
        thresholds: dict[str, tuple[float, float]] | None = None,
        # THRESHOLD ALIGNMENT: if provided, recompute soft labels from stored
        # GP posterior moments (mu, sigma) using these thresholds rather than
        # reading the precomputed {dim}_soft values (which were generated with
        # old hardcoded thresholds (3, 6)).
        # Always pass the current training-set tertile thresholds here.
        # Without this, soft labels are inconsistent with hard labels and aug hurts.
    ) -> None:
        from dataset_affectai import (
            flatten_features, task_onehot as _task_onehot
        )
        from scipy.signal import resample as _scipy_resample

        self.device = device or torch.device("cpu")
        N = len(aug_df)

        self._gaze  = [seq_to_array(r["gaze_seq"],  GAZE_SEQ_COLS)  for _, r in aug_df.iterrows()]
        self._pupil = [seq_to_array(r["pupil_seq"], PUPIL_SEQ_COLS) for _, r in aug_df.iterrows()]
        self._eda   = [seq_to_array(r["eda_seq"],   EDA_SEQ_COLS)   for _, r in aug_df.iterrows()]
        self._ppg   = [seq_to_array(r["ppg_seq"],   PPG_SEQ_COLS)   for _, r in aug_df.iterrows()]
        self._imu   = [seq_to_array(r["imu_seq"],   IMU_SEQ_COLS)   for _, r in aug_df.iterrows()]

        # Resample all sequences to target_T if they differ (mismatch between pretrain and labeled datasets)
        if target_T is not None:
            for lst in (self._gaze, self._pupil, self._eda, self._ppg, self._imu):
                for i in range(N):
                    if lst[i].shape[0] != target_T:
                        lst[i] = _scipy_resample(lst[i], target_T, axis=0).astype(np.float32)

        for i in range(N):
            self._gaze[i]  = np.clip(scalers["gaze"].transform(self._gaze[i]),   -10, 10)
            self._pupil[i] = np.clip(scalers["pupil"].transform(self._pupil[i]), -10, 10)
            self._eda[i]   = np.clip(scalers["eda"].transform(self._eda[i]),     -10, 10)
            self._ppg[i]   = np.clip(scalers["ppg"].transform(self._ppg[i]),     -10, 10)
            self._imu[i]   = np.clip(scalers["imu"].transform(self._imu[i]),     -10, 10)

        # Summary features
        self._summary = []
        for _, r in aug_df.iterrows():
            feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features", "audio_features", "speech_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    feats.update(fd)
            self._summary.append(flatten_features(feats, key_order=summary_key_order))

        # Soft labels and instance weights
        DIMS_IDX = {"valence": 0, "arousal": 1, "dominance": 2}
        self._vad_soft   = np.zeros((N, 3, 3), dtype=np.float32)
        self._vad_weight = np.zeros((N, 3),     dtype=np.float32)
        self._vad_hard   = np.ones((N, 3),      dtype=np.int64)  # argmax placeholder

        # Import norm once for threshold-aligned soft label computation
        from scipy.stats import norm as _norm

        aug_reset = aug_df.reset_index(drop=True)
        for i, row in aug_reset.iterrows():
            for dim, d_idx in DIMS_IDX.items():
                w = float(row.get(f"{dim}_weight", 0.05))

                if thresholds is not None and dim in thresholds:
                    # THRESHOLD ALIGNMENT: recompute soft label from stored GP
                    # posterior moments under the current training thresholds.
                    # The stored {dim}_soft was generated with old fixed thresholds
                    # (3, 6); re-integrating here makes soft labels consistent with
                    # the hard training labels which use balanced tertile thresholds.
                    mu  = float(row.get(f"{dim}_mu",    5.0))
                    sig = float(row.get(f"{dim}_sigma", 1.5))
                    t1, t2 = thresholds[dim]
                    p_low  = float(_norm.cdf(t1, loc=mu, scale=max(sig, 1e-4)))
                    p_high = float(1.0 - _norm.cdf(t2, loc=mu, scale=max(sig, 1e-4)))
                    p_mid  = float(max(1.0 - p_low - p_high, 0.0))
                    s = np.array([p_low, p_mid, p_high], dtype=np.float32)
                    total = s.sum()
                    if total > 1e-6:
                        s /= total
                    else:
                        s = np.array([1/3, 1/3, 1/3], dtype=np.float32)
                else:
                    # Fallback: use precomputed soft labels from pool
                    # (only consistent if pool was generated with the same thresholds)
                    soft = row.get(f"{dim}_soft")
                    if soft is not None and len(soft) == 3:
                        s = np.array(soft, dtype=np.float32)
                        s = np.clip(s, 0, 1)
                        total = s.sum()
                        if total > 1e-6:
                            s /= total
                        else:
                            s = np.array([1/3, 1/3, 1/3], dtype=np.float32)
                    else:
                        s = np.array([1/3, 1/3, 1/3], dtype=np.float32)

                self._vad_soft[i, d_idx] = s
                self._vad_hard[i, d_idx] = int(np.argmax(s))
                self._vad_weight[i, d_idx] = float(np.clip(w, 0.0, 1.0))

        # Apply per-dimension weight scaling (Fix A: dim-specific aug contribution)
        if dim_weight_scale is not None:
            for dim, d_idx in DIMS_IDX.items():
                scale = float(dim_weight_scale.get(dim, 1.0))
                if scale != 1.0:
                    self._vad_weight[:, d_idx] *= scale
                    self._vad_weight[:, d_idx] = np.clip(self._vad_weight[:, d_idx], 0.0, 1.0)

        self._uid  = np.array([user2idx.get(str(r.get("subject_id", "")), 0)
                               for _, r in aug_df.iterrows()], dtype=np.int64)
        self._sid  = np.array([session2idx.get(str(r.get("session_id", "")), 0)
                               for _, r in aug_df.iterrows()], dtype=np.int64)
        self._task = np.stack([_task_onehot(str(r.get("task", "T0")))
                               for _, r in aug_df.iterrows()])
        self._sex  = np.array([int(str(r.get("sex", "-1")).strip())
                               if str(r.get("sex", "-1")).lstrip("-").isdigit()
                               else -1
                               for _, r in aug_df.iterrows()], dtype=np.int64)
        self._pers = np.array([[float(r.get(c, 0.0))
                                if not pd.isna(r.get(c, float("nan"))) else 0.0
                                for c in BIG_FIVE_COLS]
                               for _, r in aug_df.iterrows()], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._gaze)

    def __getitem__(self, idx: int):
        dev = self.device
        return (
            torch.tensor(self._gaze[idx].copy(),  device=dev),                  # 0
            torch.tensor(self._pupil[idx].copy(), device=dev),                  # 1
            torch.tensor(self._eda[idx].copy(),   device=dev),                  # 2
            torch.tensor(self._ppg[idx].copy(),   device=dev),                  # 3
            torch.tensor(self._imu[idx].copy(),   device=dev),                  # 4
            torch.tensor(self._pers[idx],         device=dev),                  # 5
            torch.tensor(self._vad_hard[idx],  dtype=torch.long, device=dev),  # 6 — argmax bins
            torch.tensor(self._uid[idx],       dtype=torch.long, device=dev),  # 7
            torch.tensor(self._summary[idx],      device=dev),                  # 8
            torch.tensor(self._sex[idx],       dtype=torch.long, device=dev),  # 9
            torch.tensor(self._task[idx],         device=dev),                  # 10
            torch.tensor(self._sid[idx],       dtype=torch.long, device=dev),  # 11
            torch.tensor(self._vad_soft[idx],     device=dev),                  # 12 — (3, 3)
            torch.tensor(self._vad_weight[idx],   device=dev),                  # 13 — (3,)
        )


# ── Subject-level split ───────────────────────────────────────────────────────

def subject_split(
    df: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)
    idx = list(rng.permutation(n))
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    test_s  = [subjects[i] for i in idx[:n_test]]
    val_s   = [subjects[i] for i in idx[n_test:n_test + n_val]]
    train_s = [subjects[i] for i in idx[n_test + n_val:]]
    return (
        df[df["subject_id"].isin(train_s)].copy(),
        df[df["subject_id"].isin(val_s)].copy(),
        df[df["subject_id"].isin(test_s)].copy(),
    )


def per_subject_window_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split windows so every subject has samples in train, val, AND test.

    For each subject, their windows are shuffled and partitioned:
      test  = first ⌈n * test_frac⌉  windows
      val   = next  ⌈n * val_frac⌉   windows
      train = remainder

    With 5 windows/subject this gives roughly 1 test + 1 val + 3 train per person.
    The model sees each person's physiological baseline during training,
    eliminating the cold-start distribution shift of LOGO-CV.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for _, grp in df.groupby("subject_id"):
        indices = list(grp.index)
        rng.shuffle(indices)
        n = len(indices)
        n_test  = max(1, round(n * test_frac))
        n_val   = max(1, round(n * val_frac))
        # Ensure at least 1 in train
        if n_test + n_val >= n:
            n_test = 1
            n_val  = 1 if n > 1 else 0
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    return (
        df.loc[train_idx].copy(),
        df.loc[val_idx].copy(),
        df.loc[test_idx].copy(),
    )


def task_split(
    df: pd.DataFrame,
    test_task: str = "T4",
    t0_mode: str = "train",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by task label.  Default: T0–T2 train / T3 val / T4 test.

    Pass test_task="T3" to test on T3 (which has full V/A/D labels; T4 has
    no dominance labels at all).  Val is always the task just before test_task.

    Deterministic and temporally ordered.  Every subject appears in all three
    splits (assuming they completed all 5 tasks).

    t0_mode controls T0 treatment:
      'train'        — T0 included as labelled training data (default)
      'baseline-only'— T0 excluded from training but kept for baseline computation
      'exclude'      — T0 ignored entirely
    """
    task_order  = ["T0", "T1", "T2", "T3", "T4"]
    test_idx_   = task_order.index(test_task)
    # Val = task immediately before test; train = all tasks before val.
    # Strictly temporal: never train on tasks that come after the test task.
    val_task    = task_order[test_idx_ - 1] if test_idx_ > 0 else task_order[0]
    train_tasks = task_order[:test_idx_ - 1]  # everything before val_task
    if not train_tasks:
        train_tasks = [val_task]               # fallback: val is also in train if test=T1
    train_df = df[df["task"].isin(train_tasks)].copy()
    val_df   = df[df["task"] == val_task].copy()
    test_df  = df[df["task"] == test_task].copy()
    if t0_mode in ("baseline-only", "exclude"):
        train_df = train_df[train_df["task"] != "T0"].copy()
    return train_df, val_df, test_df


def logo_splits(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Leave-one-group-out: yields (train_df, test_df) for each session."""
    sessions = sorted(df["session_id"].unique())
    splits = []
    for ses in sessions:
        test  = df[df["session_id"] == ses].copy()
        train = df[df["session_id"] != ses].copy()
        splits.append((train, test))
    return splits


def loso_splits(df: pd.DataFrame, test_task: str = "T3", t0_mode: str = "train"):
    """Leave-one-subject-out: for each subject yields (train_df, val_df, test_df, subject_id).

    Canonical temporal structure is preserved for the held-*in* subjects:
      train = T0+T1 windows from all other subjects
      val   = T2   windows from all other subjects
      test  = ``test_task`` windows from the held-out subject (default: T3)

    Using ``test_task='T3'`` (default) gives a direct cross-person analogue of
    the canonical task-split (test=T3), so the LOSO result is directly comparable
    to the canonical T3 result.  Subjects with no T3 windows are skipped.

    Pass ``test_task='ALL'`` to use all of the held-out subject's windows as test
    (maximises aggregate sample size to 292, but mixes cross-task and cross-person
    generalisation since training covers only T0+T1).

    t0_mode: 'train' | 'baseline-only' | 'exclude'  (same semantics as task_split)
    """
    subjects = sorted(df["subject_id"].unique())
    splits = []
    for sub in subjects:
        rest_df  = df[df["subject_id"] != sub].copy()
        train_df = rest_df[rest_df["task"].isin(["T0", "T1"])].copy()
        val_df   = rest_df[rest_df["task"] == "T2"].copy()
        if len(val_df) == 0:              # rare: fallback to T1 if T2 missing
            val_df = rest_df[rest_df["task"] == "T1"].copy()

        if t0_mode in ("baseline-only", "exclude"):
            train_df = train_df[train_df["task"] != "T0"].copy()

        if test_task == "ALL":
            test_df = df[df["subject_id"] == sub].copy()
        else:
            test_df = df[(df["subject_id"] == sub) & (df["task"] == test_task)].copy()
            if len(test_df) == 0:
                continue   # skip subjects with no windows in the test task

        splits.append((train_df, val_df, test_df, sub))
    return splits


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: dict[str, tuple[float, float]],
    df_split: pd.DataFrame,
) -> dict[str, list]:
    """Return per-window predictions and true labels (no F1 aggregation).

    Returns dict with keys '{dim}_pred' and '{dim}_true' containing int lists.
    Dominance sentinels (-1) are preserved; caller masks them when computing F1.
    """
    model.eval()
    preds_all: dict[str, list] = {d: [] for d in VAD_DIMS}
    trues_all: dict[str, list] = {d: [] for d in VAD_DIMS}

    with torch.no_grad():
        for batch in loader:
            (gaze, pupil, eda, ppg, imu, personality,
             emotion_bins, uid, summary, sex, task, sid,
             vad_soft, vad_weight) = batch

            out = model(
                gaze_seq=gaze.float(),
                pupil_seq=pupil.float(),
                eda_seq=eda.float(),
                ppg_seq=ppg.float(),
                imu_seq=imu.float(),
                summary=summary.float(),
                personality=personality.float(),
                user_ids=uid,
                task_onehot=task.float(),
            )

            for i, dim in enumerate(VAD_DIMS):
                preds = out[f"{dim}_logits"].argmax(dim=-1).cpu().numpy()
                lbls  = emotion_bins[:, i].cpu().numpy()
                preds_all[dim].extend(preds.tolist())
                trues_all[dim].extend(lbls.tolist())

    return {f"{d}_pred": preds_all[d] for d in VAD_DIMS} | \
           {f"{d}_true": trues_all[d] for d in VAD_DIMS}


def _loso_aggregate_f1(
    all_preds: dict[str, list],
    all_trues: dict[str, list],
) -> dict:
    """Compute macro-F1 from aggregated LOSO predictions (all 292 windows)."""
    metrics: dict = {}
    f1s = []
    for dim in VAD_DIMS:
        p = np.array(all_preds[dim])
        t = np.array(all_trues[dim])
        if dim == "dominance":
            valid = t >= 0
            p, t = p[valid], t[valid]
        if len(t) > 0:
            f1 = f1_score(t, p, average="macro", zero_division=0)
        else:
            f1 = 0.0
        metrics[f"{dim}_f1"] = float(f1)
        f1s.append(f1)
    metrics["mean_f1"] = float(np.mean(f1s))
    return metrics


# ── Per-dimension augmentation scale helper ───────────────────────────────────

def _parse_dim_aug_scale(spec: str) -> dict[str, float]:
    """Parse 'v=1.0,a=0.2,d=0.6' or 'valence=1.0,arousal=0.2,dominance=0.6'.

    Returns {} if spec is empty or None.
    Short keys: v=valence, a=arousal, d=dominance.
    Example: 'v=1.0,a=0.2,d=0.6' -> {'valence': 1.0, 'arousal': 0.2, 'dominance': 0.6}
    """
    if not spec:
        return {}
    _ABBREV = {"v": "valence", "a": "arousal", "d": "dominance"}
    result: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        k = _ABBREV.get(k, k)
        try:
            result[k] = float(v.strip())
        except ValueError:
            pass
    return result


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(args, summary_dim: int, n_subjects: int) -> nn.Module:
    arch = getattr(args, "arch", "transformer")
    bfi_mode = getattr(args, "bfi_mode", "none")
    if arch == "transformer":
        return MuMTAffectGroupAffect(
            gaze_dim=len(GAZE_SEQ_COLS),    # 9
            pupil_dim=len(PUPIL_SEQ_COLS),  # 3
            eda_dim=len(EDA_SEQ_COLS),      # 5
            ppg_dim=len(PPG_SEQ_COLS),      # 3
            imu_dim=len(IMU_SEQ_COLS),      # 6
            summary_dim=summary_dim,
            n_subjects=n_subjects,
            d_model_enc=args.d_enc,
            d_model_fuse=args.d_fuse,
            t_out=args.t_out,
            per_dim_queries=True,
            use_se_blocks=False,
            use_scaled_attention=True,
        )
    # mlp / pool / conv
    return build_simple_model(
        arch=arch,
        summary_dim=summary_dim,
        gaze_dim=len(GAZE_SEQ_COLS),
        pupil_dim=len(PUPIL_SEQ_COLS),
        eda_dim=len(EDA_SEQ_COLS),
        ppg_dim=len(PPG_SEQ_COLS),
        imu_dim=len(IMU_SEQ_COLS),
        hidden=args.d_fuse,
        dropout=args.dropout,
        bfi_mode=bfi_mode,
        bfi_dim=5,
        task_dim=5,   # T0–T4
    )


# ── LR schedule ──────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs: int, warmup: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(1, epochs - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Single run ────────────────────────────────────────────────────────────────

def compute_t0_baselines(
    df: pd.DataFrame,
    summary_key_order: list[str],
) -> dict[tuple, np.ndarray]:
    """Thin wrapper — delegates to dataset_affectai.compute_t0_baselines."""
    from dataset_affectai import compute_t0_baselines as _impl
    return _impl(df, summary_key_order)


def apply_t0_baseline(
    summary_list: list[np.ndarray],
    df: pd.DataFrame,
    baselines: dict[tuple, np.ndarray],
) -> list[np.ndarray]:
    """Thin wrapper — delegates to dataset_affectai.apply_t0_baseline."""
    from dataset_affectai import apply_t0_baseline as _impl
    return _impl(summary_list, df, baselines)


def compute_within_group_means(
    df: pd.DataFrame,
    summary_list: list[np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Compute mean summary features across all seats within each (session_id, task).

    Returns {(session_id, task): mean_vec (K,)}.
    Called without labels — no leakage.
    """
    group_means: dict[tuple[str, str], np.ndarray] = {}
    df_reset = df.reset_index(drop=True)
    for (ses, task), grp in df_reset.groupby(["session_id", "task"]):
        idxs = list(grp.index)
        if not idxs:
            continue
        stack = np.stack([summary_list[i] for i in idxs], axis=0)
        group_means[(str(ses), str(task))] = np.nanmean(stack, axis=0).astype(np.float32)
    return group_means


def apply_within_group_contrast(
    summary_list: list[np.ndarray],
    df: pd.DataFrame,
    group_means: dict[tuple[str, str], np.ndarray],
) -> list[np.ndarray]:
    """Subtract group mean from each window: makes features relative to co-participants.

    For a test group never seen during training, the test session's own mean is
    subtracted (computed from test_df, no labels used).  This removes the
    absolute physiological baseline of the unseen group.
    """
    out = []
    df_reset = df.reset_index(drop=True)
    for i, (_, r) in enumerate(df_reset.iterrows()):
        key = (str(r.get("session_id", "")), str(r.get("task", "")))
        if key in group_means:
            out.append((summary_list[i] - group_means[key]).astype(np.float32))
        else:
            out.append(summary_list[i])
    return out


def rebin_aug_soft_labels(
    aug_df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Recompute the *_soft 3-class distributions in the augmented pool using
    the current session's tertile thresholds.

    The pool was originally generated with fixed VAD_THRESHOLDS = (3.0, 6.0).
    When training uses balanced data-driven thresholds (e.g. 6.5/7.5 for valence),
    the stored soft labels are misaligned: they assign most probability to 'High'
    where the new binning splits probability among all three classes.

    Rebinning re-integrates the GP posterior N(mu, sigma) over the new class
    boundaries, producing soft labels consistent with the training hard labels.
    Only dimensions that have both *_mu and *_sigma columns are rebinned.
    """
    from scipy.stats import norm as _norm
    aug_df = aug_df.copy()
    for dim, (t1, t2) in thresholds.items():
        mu_col    = f"{dim}_mu"
        sigma_col = f"{dim}_sigma"
        soft_col  = f"{dim}_soft"
        if mu_col not in aug_df.columns or sigma_col not in aug_df.columns:
            continue
        mu     = aug_df[mu_col].values.astype(float)
        sigma  = np.clip(aug_df[sigma_col].values.astype(float), 1e-4, None)
        p_low  = _norm.cdf(t1, loc=mu, scale=sigma)
        p_high = 1.0 - _norm.cdf(t2, loc=mu, scale=sigma)
        p_mid  = np.clip(1.0 - p_low - p_high, 0.0, 1.0)
        # Renormalise to ensure sum = 1 (numerical guard)
        probs  = np.stack([p_low, p_mid, p_high], axis=1)
        probs  = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        aug_df[soft_col] = [probs[i].astype(np.float32) for i in range(len(probs))]
    log.info("  Rebinned aug pool soft labels to thresholds V=(%.2f,%.2f) "
             "A=(%.2f,%.2f) D=(%.2f,%.2f)",
             thresholds["valence"][0],   thresholds["valence"][1],
             thresholds["arousal"][0],   thresholds["arousal"][1],
             thresholds["dominance"][0], thresholds["dominance"][1])
    return aug_df


def run_split(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    fold_tag: str = "",
    global_thresholds: dict | None = None,    # pre-computed from full dataset
    aug_df: pd.DataFrame | None = None,       # augmented soft-label pool (test session excluded)
    return_preds: bool = False,               # also return per-window preds at best val epoch
) -> "dict | tuple[dict, dict]":
    n_aug = len(aug_df) if aug_df is not None else 0
    log.info("%s Train=%d  Val=%d  Test=%d  Aug=%d",
             fold_tag, len(train_df), len(val_df), len(test_df), n_aug)

    # Build lookup dicts from full dataset to keep indices stable across splits
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2idx    = make_user2idx(full_df)
    session2idx = make_session2idx(full_df)
    summary_key_order = build_summary_key_order(full_df)
    summary_dim = len(summary_key_order)

    # Fit scalers on TRAIN only.
    # Thresholds: use global (consistent across folds) if provided, else per-fold.
    log.info("  Fitting scalers …")
    scalers    = fit_scalers(train_df)
    if getattr(args, "fixed_thresholds", False):
        thresholds = {dim: (3.0, 6.0) for dim in VAD_DIMS}
        log.info("  Using fixed thresholds (3.0, 6.0) for all dims — ablation baseline.")
    elif global_thresholds is not None:
        thresholds = global_thresholds
        log.info("  Using global thresholds (consistent across folds).")
    else:
        thresholds = compute_tertile_thresholds(train_df)
    class_wts  = compute_class_weights(train_df, thresholds, device)
    if getattr(args, "no_class_weights", False):
        class_wts = {dim: None for dim in VAD_DIMS}
        log.info("  Class weights DISABLED (--no-class-weights ablation).")

    group_norm  = getattr(args, "group_norm", False)
    # 'baseline-only' mode implies t0_baseline normalization
    t0_baseline = getattr(args, "t0_baseline", False) or (getattr(args, "t0_mode", "train") == "baseline-only")

    # Datasets
    train_ds = VADDataset(train_df, user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=args.augment, device=device,
                          group_norm=group_norm, session_stats=None)
    train_stats = train_ds._session_stats if group_norm else None
    val_ds   = VADDataset(val_df,   user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=False, device=device,
                          group_norm=group_norm, session_stats=train_stats)
    test_ds  = VADDataset(test_df,  user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=False, device=device,
                          group_norm=group_norm, session_stats=None)

    # T0 baseline normalization: subtract each seat's T0-task mean from all windows.
    # Train uses only training T0 windows; test uses its own T0 windows (no label leak).
    if t0_baseline:
        train_baselines = compute_t0_baselines(train_df, summary_key_order)
        test_baselines  = compute_t0_baselines(test_df,  summary_key_order)
        log.info("  T0 baselines: train=%d keys  test=%d keys",
                 len(train_baselines), len(test_baselines))
        # Apply to train + val using training baselines
        train_ds._summary = apply_t0_baseline(train_ds._summary, train_df, train_baselines)
        val_ds._summary   = apply_t0_baseline(val_ds._summary,   val_df,   train_baselines)
        # Apply to test using its own T0 baseline
        test_ds._summary  = apply_t0_baseline(test_ds._summary,  test_df,  test_baselines)

    # Within-group contrast: subtract (session, task) group mean from each window.
    # Converts absolute physiology into deviation from co-participants' average,
    # removing between-group absolute baseline differences.
    within_group_contrast = getattr(args, "within_group_contrast", False)
    if within_group_contrast:
        # Group means from combined train+val (no label leakage — mean uses features only)
        tv_summaries = train_ds._summary + val_ds._summary
        tv_df = pd.concat([train_df, val_df], ignore_index=True)
        train_group_means = compute_within_group_means(tv_df, tv_summaries)
        log.info("  Within-group contrast: %d (session,task) groups in train",
                 len(train_group_means))
        train_ds._summary = apply_within_group_contrast(
            train_ds._summary, train_df, train_group_means)
        val_ds._summary   = apply_within_group_contrast(
            val_ds._summary,   val_df,   train_group_means)
        # Test: remove test session's own group mean (no label leakage)
        test_group_means = compute_within_group_means(
            test_df.reset_index(drop=True), test_ds._summary)
        test_ds._summary  = apply_within_group_contrast(
            test_ds._summary, test_df, test_group_means)

    # ── BFI normalisation (train-set mean/std; no label leakage) ─────────────
    bfi_mode = getattr(args, "bfi_mode", "none")
    if bfi_mode != "none":
        bfi_mean = train_ds._pers.mean(axis=0, keepdims=True)  # (1, 5)
        bfi_std  = train_ds._pers.std(axis=0, keepdims=True).clip(min=1e-6)
        log.info("  BFI mode=%s  train mean=%s  std=%s",
                 bfi_mode,
                 bfi_mean.round(2).flatten().tolist(),
                 bfi_std.round(2).flatten().tolist())
        train_ds._pers = ((train_ds._pers - bfi_mean) / bfi_std).astype(np.float32)
        val_ds._pers   = ((val_ds._pers   - bfi_mean) / bfi_std).astype(np.float32)
        test_ds._pers  = ((test_ds._pers  - bfi_mean) / bfi_std).astype(np.float32)

    # ── Augmented soft-label pool (optional) ─────────────────────────────────
    if aug_df is not None and len(aug_df) > 0:
        # Rebin soft labels to current session's thresholds.
        # The pool was generated with fixed VAD_THRESHOLDS=(3,6); rebinning
        # re-integrates GP posterior N(mu,sigma) over the balanced boundaries
        # so augmented targets are consistent with hard training labels.
        aug_df = rebin_aug_soft_labels(aug_df, thresholds)

        # Infer target sequence length from the hard-labeled training set
        _target_T = train_ds._gaze[0].shape[0] if train_ds._gaze else None
        # Parse per-dimension weight scale from args
        _dim_scale = _parse_dim_aug_scale(getattr(args, "dim_aug_scale", ""))
        if _dim_scale:
            log.info("  Per-dim aug weight scale: V=%.2f  A=%.2f  D=%.2f",
                     _dim_scale.get("valence", 1.0),
                     _dim_scale.get("arousal", 1.0),
                     _dim_scale.get("dominance", 1.0))
        aug_ds = AugSoftDataset(
            aug_df, user2idx, session2idx, summary_key_order, scalers,
            device=device, target_T=_target_T,
            dim_weight_scale=_dim_scale if _dim_scale else None,
            thresholds=thresholds,  # threshold alignment: recompute soft labels under current training thresholds
        )
        log.info("  Threshold-aligned aug soft labels: recomputed from (mu, sigma) under current thresholds.")
        # Normalise BFI in augmented pool using training stats
        if bfi_mode != "none":
            aug_ds._pers = ((aug_ds._pers - bfi_mean) / bfi_std).astype(np.float32)

        # Apply T0 baseline to augmented pool using training baselines
        if t0_baseline:
            train_baselines_aug = compute_t0_baselines(
                pd.concat([train_df, aug_df], ignore_index=True), summary_key_order
            )
            aug_ds._summary = apply_t0_baseline(aug_ds._summary, aug_df.reset_index(drop=True),
                                                train_baselines_aug)

        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        combined_ds = ConcatDataset([train_ds, aug_ds])

        # Sampling weights: hard-labeled = 1.0, augmented = scale * conf_i
        # We want E[P(aug draw)] = aug_frac.
        # With WeightedRandomSampler: P(aug) = sum(soft_w) / (n_hard + sum(soft_w)).
        # Setting sum(soft_w) = aug_frac * n_hard / (1-aug_frac) achieves this exactly.
        # Since soft_w_i = scale * conf_i, scale = aug_frac*n_hard / ((1-aug_frac)*sum(conf)).
        # BUG FIX: old code used n_soft instead of conf.sum(), which undercounted by ~10x
        # when avg GP confidence ~0.10 — aug actually contributed <5% instead of aug_frac.
        aug_frac = getattr(args, "aug_frac", 0.5)   # target fraction from augmented pool
        n_hard = len(train_ds)
        n_soft = len(aug_ds)

        # Per-sample confidence weights for augmented pool
        aug_conf = aug_ds._vad_weight.mean(axis=1)   # (N_aug,)
        conf_sum = float(aug_conf.sum()) + 1e-8

        if 0 < aug_frac < 1:
            scale = (n_hard * aug_frac) / (conf_sum * (1.0 - aug_frac))
        else:
            scale = 1.0

        hard_w = np.ones(n_hard, dtype=np.float32)
        soft_w = (scale * aug_conf).astype(np.float32)
        all_w  = np.concatenate([hard_w, soft_w])
        all_w  = all_w / (all_w.max() + 1e-8)

        # Verify effective aug fraction
        eff_aug_frac = soft_w.sum() / (hard_w.sum() + soft_w.sum())
        log.info("  Sampling fix: scale=%.4f  eff_aug_frac=%.3f (target=%.2f)  "
                 "avg_conf=%.3f  conf_sum=%.1f",
                 scale, eff_aug_frac, aug_frac, float(aug_conf.mean()), conf_sum)

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(all_w),
            num_samples=n_hard + n_soft,   # one epoch = see all data once on average
            replacement=True,
        )
        train_loader = DataLoader(combined_ds, batch_size=args.batch, sampler=sampler,
                                  drop_last=True)
        log.info("  Combined loader: %d hard + %d soft  (aug_frac=%.2f, scale=%.4f)",
                 n_hard, n_soft, aug_frac, scale)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # Model + loss + optimizer
    model = build_model(args, summary_dim, n_subjects=len(user2idx)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Model params: %s", f"{n_params:,}")

    loss_fn   = SoftVADLoss(class_weights=class_wts,
                              label_smoothing=args.label_smooth,
                              alpha=args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args.epochs, warmup=args.warmup)
    amp_scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda") else None

    best_val_f1  = -1.0
    best_metrics: dict = {}
    best_preds:   dict = {}   # populated when return_preds=True
    patience_cnt = 0

    for ep in range(1, args.epochs + 1):
        _no_se = getattr(args, "no_subject_embed", False)
        loss_fn._no_subject_embed = _no_se   # propagate flag to evaluate() via loss_fn
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, amp_scaler,
                              no_subject_embed=_no_se)
        scheduler.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            val_m = evaluate(model, val_loader, loss_fn, device, thresholds, val_df)
            log.info(
                "  ep%3d | lr=%.2e | tr_loss=%.3f | "
                "val V=%.3f A=%.3f D=%.3f mean=%.3f",
                ep, optimizer.param_groups[0]["lr"],
                tr_loss["total"],
                val_m["valence_f1"], val_m["arousal_f1"],
                val_m["dominance_f1"], val_m["mean_f1"],
            )

            if val_m["mean_f1"] > best_val_f1:
                best_val_f1 = val_m["mean_f1"]
                patience_cnt = 0
                if args.ckpt_dir and args.ckpt_dir.strip():
                    ckpt_path = Path(args.ckpt_dir.strip()) / f"best{fold_tag}.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), ckpt_path)
                # Evaluate on test with best val checkpoint
                best_metrics = evaluate(model, test_loader, loss_fn, device, thresholds, test_df)
                # Capture per-window predictions at best epoch (used by LOSO aggregation)
                if return_preds:
                    best_preds = collect_predictions(
                        model, test_loader, device, thresholds, test_df)
            else:
                patience_cnt += args.eval_every
                if args.patience > 0 and patience_cnt >= args.patience:
                    log.info("  Early stop at epoch %d (patience %d)", ep, args.patience)
                    break

    log.info(
        "  %s BEST test | V=%.3f A=%.3f D=%.3f mean=%.3f (val peak=%.3f)",
        fold_tag,
        best_metrics.get("valence_f1", 0),
        best_metrics.get("arousal_f1", 0),
        best_metrics.get("dominance_f1", 0),
        best_metrics.get("mean_f1", 0),
        best_val_f1,
    )
    if return_preds:
        return best_metrics, best_preds
    return best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MuMTAffectGroupAffect on dataset.pkl (no SSL pretrain)."
    )
    # Paths
    parser.add_argument("--dataset",   default="data/mumt/dataset.pkl")
    parser.add_argument("--augmented-pool", default="data/mumt/augmented_pool.pkl",
                        help="Path to augmented_pool.pkl with GP soft labels. "
                             "If provided (and file exists), training mixes hard-labeled + "
                             "soft-labeled windows. Set to '' to disable.")
    parser.add_argument("--aug-frac",  type=float, default=0.5,
                        help="Target fraction of each batch drawn from the augmented pool "
                             "(0 = no augmented, 1 = only augmented; default: 0.5).")
    parser.add_argument("--dim-aug-scale", type=str, default="",
                        dest="dim_aug_scale",
                        help="Per-dimension multiplier on GP instance weights inside the "
                             "augmented pool. Format: 'v=FLOAT,a=FLOAT,d=FLOAT' or "
                             "'valence=F,arousal=F,dominance=F'. Default: all 1.0 (no scaling). "
                             "Motivation: LOO calibration shows arousal GP accuracy = 0.30 "
                             "(below chance) because OU prior dominates between sparse "
                             "task-level observations (theta=0.578, inter-task gap ~1200s). "
                             "Downscaling arousal (e.g. a=0.2) limits the bias while retaining "
                             "valence augmentation (LOO acc = 0.69). "
                             "Example: --dim-aug-scale v=1.0,a=0.2,d=0.6")
    parser.add_argument("--ckpt-dir",  default="data/mumt/checkpoints",
                        help="Directory to save best checkpoints. Empty = no saving.")
    # Training
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--wd",        type=float, default=1e-3)
    parser.add_argument("--warmup",    type=int,   default=10,
                        help="Linear LR warmup epochs.")
    parser.add_argument("--patience",  type=int,   default=60,
                        help="Early stopping patience in epochs (0 = disabled).")
    parser.add_argument("--eval-every",type=int,   default=5)
    parser.add_argument("--alpha",     type=float, default=0.05,
                        help="Personality loss weight (1-alpha goes to emotion).")
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--augment",   action="store_true",
                        help="Enable time-warp + noise augmentation on training set.")
    # Architecture
    parser.add_argument("--arch",   default="pool",
                        choices=["mlp", "pool", "conv", "transformer"],
                        help="Model architecture (default: pool).")
    parser.add_argument("--d-enc",  type=int, default=32,
                        help="Transformer encoder d_model (only used with --arch transformer).")
    parser.add_argument("--d-fuse", type=int, default=64,
                        help="Fusion/hidden dim (transformer: fusion width; mlp/pool/conv: MLP hidden).")
    parser.add_argument("--t-out",  type=int, default=16,
                        help="Downsampled time steps per modality encoder (transformer only).")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate for simple models (default: 0.5).")
    parser.add_argument("--group-norm", action="store_true",
                        help="Z-score summary features within each (session, seat). "
                             "Removes absolute physiological baseline differences across groups.")
    parser.add_argument("--within-group-contrast", action="store_true",
                        dest="within_group_contrast",
                        help="Subtract (session, task) group mean from each window's summary "
                             "features. Converts absolute physiology into deviation from "
                             "co-participants — the strongest normalisation for LOGO-CV.")
    parser.add_argument("--t0-baseline", action="store_true",
                        help="Subtract each seat's T0-task mean from all window summary features. "
                             "Encodes physiological change from rest rather than absolute levels.")
    parser.add_argument(
        "--t0-mode",
        choices=["train", "baseline-only", "exclude"],
        default="train",
        help="How to treat T0 (baseline/rest) windows. "
             "'train' (default): T0 included in training set as labelled data. "
             "'baseline-only': T0 used only for per-seat physiological normalisation "
             "(implies --t0-baseline); T0 windows excluded from labelled training. "
             "'exclude': T0 ignored entirely.",
    )
    parser.add_argument("--global-thresholds", action="store_true",
                        help="Compute VAD tertile thresholds from the full dataset once "
                             "(not per fold). Ensures consistent class boundaries across LOGO-CV folds.")
    parser.add_argument("--fixed-thresholds", action="store_true",
                        dest="fixed_thresholds",
                        help="Use hardcoded fixed thresholds (3.0, 6.0) for all VAD dims "
                             "instead of data-driven tertile binning. Ablation baseline for C1.")
    parser.add_argument("--no-class-weights", action="store_true",
                        dest="no_class_weights",
                        help="Disable per-dimension inverse-frequency class weights. "
                             "Ablation baseline for C1 step 2.")
    # Evaluation mode
    parser.add_argument("--test-task", default="T4",
                        choices=["T0", "T1", "T2", "T3", "T4"],
                        dest="test_task",
                        help="Task to hold out as test when --split-mode task (default: T4). "
                             "Use T3 to avoid the T4 dominance-NaN issue.")
    parser.add_argument("--split-mode", default="per_subject",
                        choices=["per_subject", "task", "subject"],
                        dest="split_mode",
                        help="Split strategy (default: per_subject). "
                             "per_subject: every subject has windows in train/val/test — "
                             "eliminates cold-start shift. "
                             "task: T0-T2 train / T3 val / T4 test. "
                             "subject: hold out entire subjects (old default).")
    parser.add_argument("--logo-cv", action="store_true",
                        help="Leave-one-group-out cross-validation (10 folds).")
    parser.add_argument("--loso-cv", action="store_true",
                        help="Leave-one-subject-out cross-validation. "
                             "For each subject: train=T0+T1 from others, val=T2 from others, "
                             "test=--loso-test-task windows of that subject (default T3). "
                             "Aggregate F1 over all held-out predictions gives a stable "
                             "cross-subject generalisation estimate. No augmentation used.")
    parser.add_argument("--loso-test-task", default="T3",
                        choices=["T2", "T3", "T4", "ALL"],
                        dest="loso_test_task",
                        help="Which task windows to use as the test set in LOSO-CV. "
                             "T3 (default): direct cross-person analogue of the canonical split. "
                             "ALL: all tasks of the held-out subject (mixes cross-task + cross-person).")
    parser.add_argument("--task-cv", action="store_true",
                        help="Rotating task CV: run one fold per test task in "
                             "{T2, T3, T4} with strict temporal ordering. "
                             "Reports mean ± std across folds for publication-ready estimates.")
    parser.add_argument("--no-subject-embed", action="store_true",
                        dest="no_subject_embed",
                        help="Zero out subject IDs passed to the transformer model. "
                             "Analogue of the no_identity ablation: forces the model to "
                             "learn affect-general rather than subject-specific features.")
    parser.add_argument("--bfi-mode", default="none",
                        choices=["none", "concat", "gate", "perdim", "taskadapt"],
                        dest="bfi_mode",
                        help="Personality (BFI) modulation mode for MLPNet. "
                             "none: BFI ignored (default). "
                             "concat: append normalised BFI-5 to summary features. "
                             "gate: BFI gates summary features via sigmoid (true moderation). "
                             "perdim: per-VAD-head BFI bias (CTSEM-inspired cross-lagged effect).")
    parser.add_argument("--seed",    type=int, default=42)
    # Hardware
    parser.add_argument("--device",  default="auto",
                        help="'auto', 'cpu', 'cuda', 'cuda:0', etc.")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    df = pd.read_pickle(args.dataset)
    log.info("Loaded %d windows  |  %d subjects  |  %d sessions",
             len(df), df["subject_id"].nunique(), df["session_id"].nunique())
    log.info("VAD NaN counts — V:%d  A:%d  D:%d",
             df["valence"].isna().sum(), df["arousal"].isna().sum(),
             df["dominance"].isna().sum())

    # Load augmented soft-label pool (optional)
    import os as _os
    aug_pool: pd.DataFrame | None = None
    aug_pool_path = getattr(args, "augmented_pool", "")
    if aug_pool_path and aug_pool_path.strip() and _os.path.isfile(aug_pool_path.strip()):
        aug_pool = pd.read_pickle(aug_pool_path.strip())
    elif aug_pool_path and aug_pool_path.strip():
        log.warning("  Augmented pool path '%s' not found — running WITHOUT augmentation.",
                    aug_pool_path.strip())
    if aug_pool is not None:
        log.info("Augmented pool: %d windows  |  %d sessions  (aug_frac=%.2f)",
                 len(aug_pool), aug_pool["session_id"].nunique(), args.aug_frac)
        # Median confidence weights for info
        for dim in ["valence", "arousal", "dominance"]:
            med = aug_pool[f"{dim}_weight"].median()
            log.info("  %s median weight: %.3f", dim, med)

    # Global thresholds: computed once from all data so class boundaries are
    # consistent across all LOGO-CV folds.
    global_thresholds: dict | None = None
    if getattr(args, "global_thresholds", False):
        log.info("Computing global VAD thresholds from full dataset …")
        global_thresholds = compute_tertile_thresholds(df)
        log.info("  (will be used for all folds)")

    if args.logo_cv:
        # ── Leave-One-Group-Out CV ─────────────────────────────────────────
        splits = logo_splits(df)
        all_f1s: list[dict] = []
        for fold_i, (train_df, test_df) in enumerate(splits, 1):
            fold_tag = f"_fold{fold_i:02d}"
            test_session = test_df["session_id"].iloc[0]
            log.info("=== LOGO-CV fold %d/%d (test: %s) ===",
                     fold_i, len(splits), test_session)
            # Use 20% of remaining groups as val
            val_subs = sorted(train_df["subject_id"].unique())
            rng = np.random.default_rng(args.seed + fold_i)
            n_val = max(1, int(len(val_subs) * 0.20))
            val_subs_sel = list(rng.choice(val_subs, n_val, replace=False))
            val_df   = train_df[train_df["subject_id"].isin(val_subs_sel)].copy()
            train_df_fold = train_df[~train_df["subject_id"].isin(val_subs_sel)].copy()

            # Exclude test session from augmented pool (leakage prevention)
            fold_aug = None
            if aug_pool is not None:
                fold_aug = aug_pool[aug_pool["session_id"] != test_session].copy()
                log.info("  Aug pool after test-session exclusion: %d / %d windows",
                         len(fold_aug), len(aug_pool))

            m = run_split(train_df_fold, val_df, test_df, args, device, fold_tag,
                          global_thresholds=global_thresholds, aug_df=fold_aug)
            all_f1s.append(m)

        # Summary table
        print("\n" + "=" * 62)
        print(f"{'LOGO-CV Summary':^62}")
        print("=" * 62)
        print(f"{'Fold':<6} {'V F1':>7} {'A F1':>7} {'D F1':>7} {'Mean F1':>9}")
        print("-" * 62)
        fold_means = []
        for i, m in enumerate(all_f1s, 1):
            vf  = m.get("valence_f1",   0)
            af  = m.get("arousal_f1",   0)
            df_ = m.get("dominance_f1", 0)
            mf  = m.get("mean_f1",      0)
            fold_means.append(mf)
            print(f"{i:<6} {vf:>7.3f} {af:>7.3f} {df_:>7.3f} {mf:>9.3f}")
        print("-" * 62)
        all_v  = [m.get("valence_f1",   0) for m in all_f1s]
        all_a  = [m.get("arousal_f1",   0) for m in all_f1s]
        all_d  = [m.get("dominance_f1", 0) for m in all_f1s]
        all_m  = [m.get("mean_f1",      0) for m in all_f1s]
        print(f"{'Mean':<6} {np.mean(all_v):>7.3f} {np.mean(all_a):>7.3f} "
              f"{np.mean(all_d):>7.3f} {np.mean(all_m):>9.3f}")
        print(f"{'Std':<6} {np.std(all_v):>7.3f} {np.std(all_a):>7.3f} "
              f"{np.std(all_d):>7.3f} {np.std(all_m):>9.3f}")
        print("=" * 62)

    elif getattr(args, "loso_cv", False):
        # ── Leave-One-Subject-Out CV ───────────────────────────────────────
        # For each subject i (39 total):
        #   train = T0+T1 from subjects != i   (~100 windows)
        #   val   = T2    from subjects != i   (~76 windows)
        #   test  = ALL   tasks of subject i   (~7.5 windows avg)
        #
        # Per-fold test sets average only 7.5 windows — per-fold F1 is very
        # noisy. Instead we accumulate ALL held-out predictions and compute
        # a single AGGREGATE cross-subject F1 over all 292 windows.
        #
        # Global thresholds ensure consistent class boundaries across folds.
        # No augmentation: prevents GP-label leakage from test subject.

        log.info("Computing global VAD thresholds for LOSO-CV …")
        loso_thresholds = compute_tertile_thresholds(df)
        log.info("  valence: %.2f / %.2f", *loso_thresholds["valence"])
        log.info("  arousal: %.2f / %.2f", *loso_thresholds["arousal"])
        log.info("  dominance: %.2f / %.2f", *loso_thresholds["dominance"])

        loso_test_task = getattr(args, "loso_test_task", "T3")
        log.info("LOSO-CV test task: %s", loso_test_task)
        splits_loso = loso_splits(df, test_task=loso_test_task, t0_mode=getattr(args, "t0_mode", "train"))
        log.info("LOSO-CV: %d folds", len(splits_loso))
        agg_preds: dict[str, list] = {d: [] for d in VAD_DIMS}
        agg_trues: dict[str, list] = {d: [] for d in VAD_DIMS}
        per_fold_f1: list[float] = []
        per_fold_n:  list[int]   = []

        for fold_i, (train_df_ls, val_df_ls, test_df_ls, sub_id) in enumerate(splits_loso, 1):
            fold_tag_ls = f"_loso{fold_i:02d}"
            log.info("=== LOSO-CV fold %d/%d (test: %s, N=%d) ===",
                     fold_i, len(splits_loso), sub_id, len(test_df_ls))

            m, preds = run_split(
                train_df_ls, val_df_ls, test_df_ls,
                args, device, fold_tag_ls,
                global_thresholds=loso_thresholds,
                aug_df=None,         # no aug → no GP leakage
                return_preds=True,
            )

            for dim in VAD_DIMS:
                agg_preds[dim].extend(preds[f"{dim}_pred"])
                agg_trues[dim].extend(preds[f"{dim}_true"])

            # Per-fold F1 from aggregate of this fold's predictions
            fold_agg = _loso_aggregate_f1(
                {d: preds[f"{d}_pred"] for d in VAD_DIMS},
                {d: preds[f"{d}_true"] for d in VAD_DIMS},
            )
            per_fold_f1.append(fold_agg["mean_f1"])
            per_fold_n.append(len(test_df_ls))

        # Global aggregate F1 over all 292 held-out windows
        global_agg = _loso_aggregate_f1(agg_preds, agg_trues)
        n_total    = sum(per_fold_n)

        print("\n" + "=" * 70)
        print(f"{'LOSO-CV Summary':^70}")
        print("=" * 70)
        print(f"  {'Sub':<12} {'N':>3}  {'Mean F1':>8}  (per-fold, noisy due to small N)")
        print("-" * 70)
        for i, (f, n) in enumerate(zip(per_fold_f1, per_fold_n), 1):
            print(f"  Fold {i:2d}       {n:3d}  {f:8.3f}")
        print("-" * 70)
        print(f"  Per-fold mean ± std : "
              f"{np.mean(per_fold_f1):.3f} ± {np.std(per_fold_f1):.3f}  "
              f"(informational only — use aggregate F1)")
        print("-" * 70)
        print(f"  AGGREGATE (N={n_total} held-out windows, cross-subject):")
        print(f"    Valence  F1   : {global_agg['valence_f1']:.3f}")
        print(f"    Arousal  F1   : {global_agg['arousal_f1']:.3f}")
        print(f"    Dominance F1  : {global_agg['dominance_f1']:.3f}")
        print(f"    Mean F1       : {global_agg['mean_f1']:.3f}")
        print("=" * 70)

    elif getattr(args, "task_cv", False):
        # ── Rotating task CV: T2 / T3 / T4 as test tasks ─────────────────
        # T2: val=T1, train=T0          (small train)
        # T3: val=T2, train=T0+T1       (canonical)
        # T4: val=T3, train=T0+T1+T2    (most data; T4 has dominance NaNs)
        task_cv_folds = ["T2", "T3", "T4"]
        all_f1s_tcv: list[dict] = []
        for test_task_cv in task_cv_folds:
            # Re-seed each fold independently so fold results are comparable
            # without being confounded by the random-state evolution of prior folds.
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            log.info("=== Task-CV fold: test=%s ===", test_task_cv)
            train_df_cv, val_df_cv, test_df_cv = task_split(df, test_task=test_task_cv, t0_mode=getattr(args, "t0_mode", "train"))
            log.info("  train:%d  val:%d  test:%d",
                     len(train_df_cv), len(val_df_cv), len(test_df_cv))

            fold_aug_cv = None
            if aug_pool is not None:
                train_tasks_cv = set(train_df_cv["task"].unique())
                fold_aug_cv = aug_pool[aug_pool["task"].isin(train_tasks_cv)].copy()
                log.info("  Aug pool filtered to %s: %d windows",
                         sorted(train_tasks_cv), len(fold_aug_cv))

            m_cv = run_split(train_df_cv, val_df_cv, test_df_cv, args, device,
                             fold_tag=f"_{test_task_cv}",
                             global_thresholds=global_thresholds,
                             aug_df=fold_aug_cv)
            m_cv["test_task"] = test_task_cv
            all_f1s_tcv.append(m_cv)

        # Summary table
        print("\n" + "=" * 66)
        print(f"{'Task-CV Summary (3-fold: T2/T3/T4)':^66}")
        print("=" * 66)
        print(f"{'Test':>6} {'N_tr':>5} {'V F1':>7} {'A F1':>7} {'D F1':>7} {'Mean F1':>9}")
        print("-" * 66)
        task_order = ["T0", "T1", "T2", "T3", "T4"]
        for m_cv in all_f1s_tcv:
            tt   = m_cv["test_task"]
            ti   = task_order.index(tt)
            vi   = task_order[ti - 1] if ti > 0 else task_order[0]
            tr_n = len(df[df["task"].isin(task_order[:ti - 1])]) if ti > 1 else len(df[df["task"] == vi])
            vf   = m_cv.get("valence_f1",   0)
            af   = m_cv.get("arousal_f1",   0)
            df_  = m_cv.get("dominance_f1", 0)
            mf   = m_cv.get("mean_f1",      0)
            print(f"{tt:>6} {tr_n:>5} {vf:>7.3f} {af:>7.3f} {df_:>7.3f} {mf:>9.3f}")
        print("-" * 66)
        all_v_cv = [m.get("valence_f1",   0) for m in all_f1s_tcv]
        all_a_cv = [m.get("arousal_f1",   0) for m in all_f1s_tcv]
        all_d_cv = [m.get("dominance_f1", 0) for m in all_f1s_tcv]
        all_m_cv = [m.get("mean_f1",      0) for m in all_f1s_tcv]
        print(f"{'Mean':>6} {'':>5} {np.mean(all_v_cv):>7.3f} {np.mean(all_a_cv):>7.3f} "
              f"{np.mean(all_d_cv):>7.3f} {np.mean(all_m_cv):>9.3f}")
        print(f"{'Std':>6} {'':>5} {np.std(all_v_cv):>7.3f} {np.std(all_a_cv):>7.3f} "
              f"{np.std(all_d_cv):>7.3f} {np.std(all_m_cv):>9.3f}")
        print("=" * 66)

    else:
        # ── Single split (default: per-subject window split) ───────────────
        split_mode = getattr(args, "split_mode", "per_subject")
        if split_mode == "task":
            test_task = getattr(args, "test_task", "T4")
            train_df, val_df, test_df = task_split(df, test_task=test_task, t0_mode=getattr(args, "t0_mode", "train"))
            log.info("Task split (test=%s) — train:%d  val:%d  test:%d", test_task,
                     len(train_df), len(val_df), len(test_df))
        elif split_mode == "subject":
            train_df, val_df, test_df = subject_split(df, seed=args.seed)
            log.info("Subject split — train:%d  val:%d  test:%d",
                     len(train_df), len(val_df), len(test_df))
        else:  # per_subject (default)
            train_df, val_df, test_df = per_subject_window_split(df, seed=args.seed)
            log.info("Per-subject window split — train:%d  val:%d  test:%d",
                     len(train_df), len(val_df), len(test_df))
            n_train_subj = train_df["subject_id"].nunique()
            n_val_subj   = val_df["subject_id"].nunique()
            n_test_subj  = test_df["subject_id"].nunique()
            log.info("  Subjects in train/val/test: %d / %d / %d",
                     n_train_subj, n_val_subj, n_test_subj)
        # Exclude future/test data from augmented pool.
        # For task splits: filter by train tasks only (session filter would wipe everything
        # since every session participates in every task).
        # For subject/per-subject splits: filter by session as before.
        single_aug = None
        if aug_pool is not None:
            if split_mode == "task":
                train_tasks_set = set(train_df["task"].unique())
                single_aug = aug_pool[aug_pool["task"].isin(train_tasks_set)].copy()
                log.info("  Aug pool filtered to train tasks %s: %d / %d windows",
                         sorted(train_tasks_set), len(single_aug), len(aug_pool))
            else:
                test_sessions = set(test_df["session_id"].unique())
                single_aug = aug_pool[~aug_pool["session_id"].isin(test_sessions)].copy()
                log.info("  Aug pool after test-session exclusion: %d / %d windows",
                         len(single_aug), len(aug_pool))
        m = run_split(train_df, val_df, test_df, args, device,
                      global_thresholds=global_thresholds, aug_df=single_aug)

        print("\n" + "=" * 40)
        print(f"{'Final Test Metrics':^40}")
        print("=" * 40)
        for dim in VAD_DIMS:
            print(f"  {dim:<12}  F1={m.get(dim+'_f1',0):.3f}  "
                  f"Acc={m.get(dim+'_acc',0):.3f}")
        print(f"  {'Mean F1':<12}  {m.get('mean_f1',0):.3f}")
        print("=" * 40)


if __name__ == "__main__":
    main()
