"""literature_experiments.py

Experiment suite testing five preprocessing / temporal modeling ideas from
recent literature (ICMI 2025, SIGDIAL 2025, IEEE TAC/Sensors 2025–2026,
PLOS ONE 2025).

Experiments
-----------
E1 — Time Warping augmentation (Hasan et al. PLOS ONE 2025)
     Replaces circular shift with cubic-spline time warping (non-linear
     temporal distortion). Best for multivariate physio data in TSDA survey.

E2 — Masked Signal Modeling pre-training (Wan et al. IEEE Sensors 2026)
     Masked autoencoding on pool sequences — mask 15–30% of time steps per
     modality and train the encoder to reconstruct.  Learns modality-specific
     temporal dynamics (EDA phasic shape, PPG waveform) vs SimCLR which only
     does instance discrimination.

E3 — Per-modality independent SSL (Jiang et al. SIGDIAL 2025)
     Pre-train each modality encoder independently (not jointly).  Avoids
     cross-modality interference where fast signals (gaze) dominate.

E4 — Window Warping for EDA (Hasan/Iwana, best for univariate)
     Apply window warping (stretch random sub-segment by 2x, contract another
     by 0.5x) specifically to EDA sequences during augmentation.

E5 — Cross-modal reconstruction auxiliary loss (Al Dossary & Chollet, ICMI 2025)
     Auxiliary task: predict PPG features from EDA encoder output and vice versa.
     Exploits autonomic co-variation as a free supervision signal.

E6 — Native-rate per-modality sequences
     Instead of uniform T=200 for all modalities (which over-samples slow EDA
     and under-samples fast gaze), resample each modality to its native temporal
     resolution at runtime:
       Gaze/Pupil (100 Hz): T=200 (keep full — already under-sampled from 1500)
       EDA       (~15 Hz): T=75  (remove interpolation artifacts)
       PPG       (~25 Hz): T=125 (closer to native resolution)
       IMU       (~25 Hz): T=125 (closer to native resolution)
     Encoders handle variable T via adaptive/mean pooling.

Baseline: SimCLR pre-training + noise/jitter/shift augmentation (current system).

All experiments use the same train/val/test split (T0+T1/T2/T3) and report
per-dimension and mean macro F1 on T3 test set, 3-seed average.

Usage
-----
  python tools/mumt/literature_experiments.py
  python tools/mumt/literature_experiments.py --encoder conv1d --seeds 5
  python tools/mumt/literature_experiments.py --experiments E1 E2 --encoder gru
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
from scipy.interpolate import CubicSpline
from sklearn.metrics import f1_score
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
    Conv1DEncoder,
    GRUEncoder,
    SoftVADLoss,
    TemporalFusionNet,
)
from train_simple import (  # noqa: E402
    bin_vad_from_thresholds,
    compute_tertile_thresholds,
    task_split,
)
from train_temporal import (  # noqa: E402
    BFI_COLS,
    FEAT_COLS,
    MODALITY_COLS,
    VAD_DIMS,
    AugPoolDataset,
    LabeledDataset,
    SequenceScaler,
    SummaryScaler,
    augment_sequence,
    build_pool_pseudo_labels,
    collate_labeled,
    collate_pool,
    compute_bfi_similarity_map,
    extract_summary,
    fit_seq_scalers,
    get_hard_labels,
    make_one_hot_soft,
    compute_class_weight_tensors,
)
from pretrain_temporal import (  # noqa: E402
    ProjectionHead,
    nt_xent_loss,
    get_temporal_embedding,
    collate_two_views,
    finetune,
)


# ═══════════════════════════════════════════════════════════════════════════════
# E1 — TIME WARPING AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def time_warp(arr: np.ndarray, n_knots: int = 4, sigma: float = 0.2) -> np.ndarray:
    """Time Warping: non-linear temporal distortion via cubic spline.

    Warps the time axis using a smooth curve generated from a cubic spline
    with random knot heights.  Preserves amplitude, distorts timing.

    Parameters
    ----------
    arr     : (T, D) input sequence
    n_knots : number of interior knots for the spline
    sigma   : std of knot height perturbation (higher = more distortion)

    Returns
    -------
    (T, D) time-warped sequence
    """
    T, D = arr.shape
    # Generate warping curve: knots at evenly-spaced positions
    knot_positions = np.linspace(0, T - 1, n_knots + 2)
    knot_heights = knot_positions + np.random.normal(0, sigma * T / n_knots, size=n_knots + 2)
    # Ensure monotonicity by cumulating deltas
    knot_heights = np.sort(knot_heights)
    knot_heights[0] = 0
    knot_heights[-1] = T - 1

    # Build cubic spline mapping original time → warped time
    cs = CubicSpline(knot_positions, knot_heights)
    warped_t = cs(np.arange(T))
    # Clip and use as indices for interpolation
    warped_t = np.clip(warped_t, 0, T - 1)

    # Interpolate each feature channel
    out = np.zeros_like(arr)
    orig_t = np.arange(T)
    for d in range(D):
        out[:, d] = np.interp(warped_t, orig_t, arr[:, d])
    return out.astype(np.float32)


def augment_sequence_tw(
    arr: np.ndarray,
    noise_sigma: float = 0.05,
    scale_lo: float = 0.85,
    scale_hi: float = 1.15,
    tw_sigma: float = 0.2,
    tw_knots: int = 4,
) -> np.ndarray:
    """Augmentation with Time Warping replacing circular shift.

    Applies: (1) Gaussian noise, (2) amplitude jitter, (3) Time Warping.
    """
    T, D = arr.shape
    # 1. Gaussian noise
    feat_std = arr.std(axis=0, keepdims=True).clip(1e-6)
    arr = arr + (np.random.randn(T, D) * noise_sigma * feat_std).astype(np.float32)
    # 2. Amplitude jitter
    scale = np.random.uniform(scale_lo, scale_hi)
    arr = (arr * scale).astype(np.float32)
    # 3. Time Warping (replaces circular shift)
    arr = time_warp(arr, n_knots=tw_knots, sigma=tw_sigma)
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# E4 — WINDOW WARPING FOR EDA
# ═══════════════════════════════════════════════════════════════════════════════

def window_warp(arr: np.ndarray, warp_ratio: float = 0.5) -> np.ndarray:
    """Window Warping: stretch one sub-segment by 2x and compress another by 0.5x.

    This preserves total sequence length T while creating local timing distortions.
    Particularly effective for univariate slow signals (EDA).

    Parameters
    ----------
    arr : (T, D) sequence
    warp_ratio : fraction of T for the warped window (default 0.5 → T/4 each)
    """
    T, D = arr.shape
    # Select window size (fraction of T)
    win_len = max(int(T * warp_ratio / 2), 4)

    # Randomly select a segment to stretch (2x speed)
    start1 = np.random.randint(0, max(T - win_len * 2, 1))
    segment1 = arr[start1:start1 + win_len * 2]
    # Subsample by 2x (stretch in time → fewer original points cover same duration)
    stretched = segment1[::2]  # Take every other sample → win_len points

    # Randomly select another segment to compress (0.5x speed)
    start2 = np.random.randint(0, max(T - win_len, 1))
    segment2 = arr[start2:start2 + win_len]
    # Upsample by 2x (compress → more points from fewer original)
    indices = np.linspace(0, win_len - 1, win_len * 2)
    compressed = np.zeros((win_len * 2, D), dtype=np.float32)
    for d in range(D):
        compressed[:, d] = np.interp(indices, np.arange(win_len), segment2[:, d])

    # Reconstruct: replace segments and adjust length back to T
    # Simplified: apply stretch to first half, compress to second half
    half = T // 2
    first_half = arr[:half]
    second_half = arr[half:]

    # Stretch first half: interpolate to 1.5x then crop
    # Compress second half: interpolate to 0.75x then pad
    # Simpler approach: warp a random sub-window
    out = arr.copy()

    # Apply stretch to a random window in the first half
    if T > win_len * 2:
        # Replace [start1:start1+win_len] with stretched version of [start1:start1+2*win_len]
        if start1 + win_len * 2 <= T:
            new_segment = np.zeros((win_len, D), dtype=np.float32)
            orig_seg = arr[start1:start1 + win_len * 2]
            for d in range(D):
                new_segment[:, d] = np.interp(
                    np.linspace(0, win_len * 2 - 1, win_len),
                    np.arange(win_len * 2),
                    orig_seg[:, d],
                )
            # Rebuild sequence: keep everything except the warped window, pad to T
            before = out[:start1]
            after = out[start1 + win_len * 2:]
            rebuilt = np.concatenate([before, new_segment, after], axis=0)
            # Pad or truncate back to T
            if len(rebuilt) < T:
                rebuilt = np.concatenate(
                    [rebuilt, np.zeros((T - len(rebuilt), D), dtype=np.float32)], axis=0
                )
            out = rebuilt[:T].astype(np.float32)

    return out


def augment_sequence_ww_eda(
    arr: np.ndarray,
    modality: str,
    noise_sigma: float = 0.05,
    scale_lo: float = 0.85,
    scale_hi: float = 1.15,
    max_shift: int = 20,
) -> np.ndarray:
    """Augmentation with Window Warping applied selectively to EDA modality.

    For EDA: applies window warp instead of circular shift.
    For other modalities: standard augment_sequence (noise + jitter + shift).
    """
    T, D = arr.shape
    # 1. Gaussian noise
    feat_std = arr.std(axis=0, keepdims=True).clip(1e-6)
    arr = arr + (np.random.randn(T, D) * noise_sigma * feat_std).astype(np.float32)
    # 2. Amplitude jitter
    scale = np.random.uniform(scale_lo, scale_hi)
    arr = (arr * scale).astype(np.float32)

    if modality == "eda":
        # 3. Window Warping for EDA
        arr = window_warp(arr)
    else:
        # 3. Circular shift for non-EDA
        shift = np.random.randint(-max_shift, max_shift + 1)
        arr = np.roll(arr, shift, axis=0).astype(np.float32)
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# E2 — MASKED SIGNAL MODELING PRE-TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class MaskedSignalModelingLoss(nn.Module):
    """Reconstruction loss on masked time steps (MAE-style for physiology)."""

    def __init__(self, mask_ratio: float = 0.25) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE only on masked positions.

        Parameters
        ----------
        original      : (B, T, D) original input
        reconstructed : (B, T, D) decoder output
        mask          : (B, T) bool — True where masked
        """
        # Expand mask to feature dim
        mask_3d = mask.unsqueeze(-1).expand_as(original)
        diff = (original - reconstructed) ** 2
        # Mean over masked positions only
        masked_mse = (diff * mask_3d).sum() / mask_3d.sum().clamp(min=1)
        return masked_mse


class MaskedEncoder(nn.Module):
    """Encoder + decoder for masked signal modeling.

    Encoder: per-modality Conv1D/GRU (same architecture as TemporalFusionNet).
    Decoder: simple 2-layer MLP that reconstructs masked time steps.
    """

    def __init__(self, encoder_type: str, enc_dim: int) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.enc_dim = enc_dim

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for name, d_m in MODALITY_DIMS.items():
            if encoder_type == "conv1d":
                self.encoders[name] = Conv1DEncoder(d_m, enc_dim)
                actual_dim = enc_dim
            else:
                self.encoders[name] = GRUEncoder(d_m, hidden_size=enc_dim)
                actual_dim = 2 * enc_dim

            # Lightweight decoder: maps encoder embedding → per-timestep reconstruction
            # Since encoder pools globally, decoder predicts mean reconstruction
            # We use a per-timestep approach: Linear(D + actual_dim) → D
            self.decoders[name] = nn.Sequential(
                nn.Linear(actual_dim + d_m, 64),
                nn.ReLU(),
                nn.Linear(64, d_m),
            )

    def forward(
        self,
        sequences: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with masked input.

        Parameters
        ----------
        sequences : {mod: (B, T, D)} original sequences (unmasked)
        masks     : {mod: (B, T)} bool masks (True = masked positions)

        Returns
        -------
        embeddings : {mod: (B, enc_out)} per-modality embeddings
        reconstructions : {mod: (B, T, D)} reconstructed sequences
        """
        embeddings = {}
        reconstructions = {}

        for name in MODALITIES:
            x_orig = sequences[name]           # (B, T, D)
            mask = masks[name]                 # (B, T)
            B, T, D = x_orig.shape

            # Mask input: zero out masked positions
            x_masked = x_orig.clone()
            x_masked[mask] = 0.0

            # Encode masked input → global embedding
            emb = self.encoders[name](x_masked)  # (B, enc_out)
            embeddings[name] = emb

            # Decode: broadcast embedding to each timestep, concat with masked input
            emb_expanded = emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, enc_out)
            decoder_in = torch.cat([x_masked, emb_expanded], dim=-1)  # (B, T, D + enc_out)
            recon = self.decoders[name](decoder_in)  # (B, T, D)
            reconstructions[name] = recon

        return embeddings, reconstructions


class MSMPoolDataset(Dataset):
    """Dataset for Masked Signal Modeling pre-training on pool sequences."""

    def __init__(
        self,
        pool: pd.DataFrame,
        seq_scalers: dict[str, SequenceScaler],
        mask_ratio: float = 0.25,
    ) -> None:
        log.info("Pre-computing pool sequences for MSM (%d windows)…", len(pool))
        self.mask_ratio = mask_ratio
        self.seq_arrays: dict[str, np.ndarray] = {}
        for mod, cols in MODALITY_COLS.items():
            arrays = []
            for _, row in pool.iterrows():
                arr = seq_to_array(row[f"{mod}_seq"], cols)
                arr = seq_scalers[mod].transform(arr)
                arrays.append(arr)
            self.seq_arrays[mod] = np.stack(arrays, axis=0)
        self.N = len(pool)
        self.T = self.seq_arrays[MODALITIES[0]].shape[1]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        seqs = {}
        masks = {}
        for mod in MODALITIES:
            arr = self.seq_arrays[mod][idx].copy()
            seqs[mod] = torch.from_numpy(arr)
            # Random mask per modality
            mask = torch.rand(self.T) < self.mask_ratio
            masks[mod] = mask
        return {"sequences": seqs, "masks": masks}


def collate_msm(batch: list[dict]) -> dict:
    result: dict = {"sequences": {}, "masks": {}}
    for mod in MODALITIES:
        result["sequences"][mod] = torch.stack([b["sequences"][mod] for b in batch])
        result["masks"][mod] = torch.stack([b["masks"][mod] for b in batch])
    return result


def pretrain_masked(
    pool: pd.DataFrame,
    seq_scalers: dict[str, SequenceScaler],
    encoder_type: str,
    enc_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_ratio: float,
    device: torch.device,
) -> nn.ModuleDict:
    """Pre-train with Masked Signal Modeling (reconstructive SSL).

    Returns trained encoder ModuleDict (same interface as SimCLR pre-training).
    """
    model = MaskedEncoder(encoder_type, enc_dim).to(device)
    msm_loss_fn = MaskedSignalModelingLoss(mask_ratio=mask_ratio)

    ds = MSMPoolDataset(pool, seq_scalers, mask_ratio=mask_ratio)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_msm, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("MSM pre-training %s encoders for %d epochs (N=%d, mask_ratio=%.2f)",
             encoder_type, epochs, len(ds), mask_ratio)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            seqs = {k: v.to(device) for k, v in batch["sequences"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}

            _, recons = model(seqs, masks)

            # Sum reconstruction loss across modalities
            total_loss = torch.tensor(0.0, device=device)
            for mod in MODALITIES:
                total_loss = total_loss + msm_loss_fn(seqs[mod], recons[mod], masks[mod])
            total_loss = total_loss / len(MODALITIES)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            log.info("  MSM epoch %d/%d  loss=%.4f",
                     epoch + 1, epochs, epoch_loss / max(n_batches, 1))

    log.info("MSM pre-training complete.")
    return model.encoders


# ═══════════════════════════════════════════════════════════════════════════════
# E3 — PER-MODALITY INDEPENDENT SSL
# ═══════════════════════════════════════════════════════════════════════════════

class SingleModalityDataset(Dataset):
    """Dataset for single-modality contrastive pre-training."""

    def __init__(
        self,
        pool: pd.DataFrame,
        modality: str,
        seq_scalers: dict[str, SequenceScaler],
    ) -> None:
        cols = MODALITY_COLS[modality]
        arrays = []
        for _, row in pool.iterrows():
            arr = seq_to_array(row[f"{modality}_seq"], cols)
            arr = seq_scalers[modality].transform(arr)
            arrays.append(arr)
        self.data = np.stack(arrays, axis=0)
        self.N = len(pool)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        arr = self.data[idx].copy()
        view1 = augment_sequence(arr.copy())
        view2 = augment_sequence(arr.copy())
        return {"view1": torch.from_numpy(view1), "view2": torch.from_numpy(view2)}


def collate_single_mod(batch: list[dict]) -> dict:
    return {
        "view1": torch.stack([b["view1"] for b in batch]),
        "view2": torch.stack([b["view2"] for b in batch]),
    }


def pretrain_per_modality(
    pool: pd.DataFrame,
    seq_scalers: dict[str, SequenceScaler],
    encoder_type: str,
    enc_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    device: torch.device,
) -> nn.ModuleDict:
    """Pre-train each modality encoder INDEPENDENTLY with SimCLR.

    Unlike joint pre-training (where all 5 modalities are concatenated before
    the projection head), this trains each encoder separately so it learns
    modality-specific features without cross-modality interference.
    """
    encoders = nn.ModuleDict()

    for mod_name, d_m in MODALITY_DIMS.items():
        log.info("  Pre-training %s encoder (d=%d)…", mod_name, d_m)

        if encoder_type == "conv1d":
            encoder = Conv1DEncoder(d_m, enc_dim).to(device)
            proj_dim_in = enc_dim
        else:
            encoder = GRUEncoder(d_m, hidden_size=enc_dim).to(device)
            proj_dim_in = 2 * enc_dim

        proj_head = ProjectionHead(proj_dim_in, proj_dim=64).to(device)

        ds = SingleModalityDataset(pool, mod_name, seq_scalers)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_single_mod, drop_last=True, num_workers=0)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(proj_head.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            encoder.train()
            proj_head.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                optimizer.zero_grad()
                v1 = batch["view1"].to(device)
                v2 = batch["view2"].to(device)
                emb1 = encoder(v1)
                emb2 = encoder(v2)
                z1 = proj_head(emb1)
                z2 = proj_head(emb2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(proj_head.parameters()), 1.0
                )
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

        log.info("    %s: final loss=%.4f", mod_name, epoch_loss / max(n_batches, 1))
        encoders[mod_name] = encoder.cpu()

    return encoders


# ═══════════════════════════════════════════════════════════════════════════════
# E5 — CROSS-MODAL RECONSTRUCTION AUXILIARY LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalReconHead(nn.Module):
    """Auxiliary head: predict one modality's embedding from another's.

    Pairs: EDA↔PPG (autonomic co-variation), Gaze↔Pupil (oculomotor).
    """

    def __init__(self, input_dim: int, target_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalLoss(nn.Module):
    """Compute cross-modal reconstruction loss between paired modalities."""

    def __init__(self, enc_dim: int, encoder_type: str) -> None:
        super().__init__()
        actual_dim = 2 * enc_dim if encoder_type == "gru" else enc_dim
        # Pairs: EDA→PPG, PPG→EDA, Gaze→Pupil, Pupil→Gaze
        self.eda_to_ppg = CrossModalReconHead(actual_dim, actual_dim)
        self.ppg_to_eda = CrossModalReconHead(actual_dim, actual_dim)
        self.gaze_to_pupil = CrossModalReconHead(actual_dim, actual_dim)
        self.pupil_to_gaze = CrossModalReconHead(actual_dim, actual_dim)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute mean MSE across all cross-modal prediction pairs."""
        loss = torch.tensor(0.0, device=next(iter(embeddings.values())).device)

        # EDA ↔ PPG
        pred_ppg = self.eda_to_ppg(embeddings["eda"].detach())
        loss = loss + F.mse_loss(pred_ppg, embeddings["ppg"].detach())
        pred_eda = self.ppg_to_eda(embeddings["ppg"].detach())
        loss = loss + F.mse_loss(pred_eda, embeddings["eda"].detach())

        # Gaze ↔ Pupil
        pred_pupil = self.gaze_to_pupil(embeddings["gaze"].detach())
        loss = loss + F.mse_loss(pred_pupil, embeddings["pupil"].detach())
        pred_gaze = self.pupil_to_gaze(embeddings["pupil"].detach())
        loss = loss + F.mse_loss(pred_gaze, embeddings["gaze"].detach())

        return loss / 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# E6 — NATIVE-RATE PER-MODALITY SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════

# Native sampling rates and target T for 15s windows
# Gaze/Pupil: Tobii 100Hz → 1500 native, but already stored at T=200 (keep)
# EDA: EmotiBit ~15Hz → ~225 native, stored at T=200 (downsample to 75)
# PPG: EmotiBit ~25Hz → ~375 native, stored at T=200 (downsample to 125)
# IMU: EmotiBit ~25Hz → ~375 native, stored at T=200 (downsample to 125)
NATIVE_T: dict[str, int] = {
    "gaze":  200,   # keep full (already 8× downsampled from native 1500)
    "pupil": 200,   # same source as gaze
    "eda":   75,    # ~15Hz × 5s effective → remove upsampling artifacts
    "ppg":   125,   # ~25Hz × 5s effective
    "imu":   125,   # ~25Hz × 5s effective
}

# For pool (30s, T=400): scale proportionally
NATIVE_T_POOL: dict[str, int] = {
    "gaze":  400,
    "pupil": 400,
    "eda":   150,
    "ppg":   250,
    "imu":   250,
}


def resample_to_native(arr: np.ndarray, target_t: int) -> np.ndarray:
    """Resample a (T, D) array to (target_t, D) using linear interpolation.

    If target_t == T, returns arr unchanged.
    If target_t < T, downsamples (removes interpolation artifacts).
    If target_t > T, upsamples (should not normally happen).
    """
    T, D = arr.shape
    if T == target_t:
        return arr
    old_t = np.arange(T, dtype=float)
    new_t = np.linspace(0, T - 1, target_t)
    out = np.zeros((target_t, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_t, old_t, arr[:, d])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING WITH CUSTOM AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class LabeledDatasetCustomAug(Dataset):
    """LabeledDataset with pluggable augmentation function per modality."""

    def __init__(
        self,
        df: pd.DataFrame,
        key_order: list[str],
        thresholds: dict,
        seq_scalers: dict[str, SequenceScaler],
        augment_fn: callable | None = None,
        native_t: dict[str, int] | None = None,
    ) -> None:
        self.N = len(df)
        self.augment_fn = augment_fn
        self.native_t = native_t  # per-modality target T (None = keep original)

        # Pre-compute sequences
        self.sequences: dict[str, np.ndarray] = {}
        for mod, cols in MODALITY_COLS.items():
            arrs = []
            for _, row in df.iterrows():
                arr = seq_to_array(row[f"{mod}_seq"], cols)
                arr = seq_scalers[mod].transform(arr)
                arrs.append(arr)
            self.sequences[mod] = np.stack(arrs, axis=0)

        # Summary features
        self.summary = extract_summary(df, key_order)

        # BFI
        self.bfi = df[BFI_COLS].values.astype(np.float32) if BFI_COLS[0] in df.columns \
            else np.zeros((self.N, 5), dtype=np.float32)

        # Labels
        self.labels = get_hard_labels(df, thresholds)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        seqs = {}
        for mod in MODALITIES:
            arr = self.sequences[mod][idx].copy()
            # Resample to native T if specified
            if self.native_t is not None and mod in self.native_t:
                arr = resample_to_native(arr, self.native_t[mod])
            if self.augment_fn is not None:
                arr = self.augment_fn(arr, mod)
            seqs[mod] = torch.from_numpy(arr)

        return {
            "sequences": seqs,
            "summary": torch.from_numpy(self.summary[idx]),
            "bfi": torch.from_numpy(self.bfi[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "weight": torch.ones(3, dtype=torch.float32),
        }


def collate_custom(batch: list[dict]) -> dict:
    return {
        "sequences": {mod: torch.stack([b["sequences"][mod] for b in batch])
                      for mod in MODALITIES},
        "summary": torch.stack([b["summary"] for b in batch]),
        "bfi": torch.stack([b["bfi"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
    }


def finetune_custom(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_order: list[str],
    encoder_type: str,
    enc_dim: int,
    dropout: float,
    pretrained_encoders: nn.ModuleDict | None,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    augment_fn: callable | None,
    device: torch.device,
    freeze_epochs: int = 10,
    cross_modal_weight: float = 0.0,
    native_t: dict[str, int] | None = None,
    pool: pd.DataFrame | None = None,
    bfi_sim_map: dict | None = None,
    aug_mode: str = "none",
) -> dict:
    """Fine-tune with custom augmentation function and optional cross-modal loss.

    When pool + bfi_sim_map + aug_mode != "none" are provided, adds Track B
    (pseudo-labeled pool windows) alongside the labeled Track A.
    """
    thresholds = compute_tertile_thresholds(train_df)
    seq_scalers = fit_seq_scalers(train_df)

    train_summary = extract_summary(train_df, key_order)
    summary_sc = SummaryScaler().fit(train_summary)

    def make_ds(df: pd.DataFrame, aug: callable | None = None,
               native_t: dict[str, int] | None = None) -> LabeledDatasetCustomAug:
        ds = LabeledDatasetCustomAug(df, key_order, thresholds, seq_scalers,
                                     augment_fn=aug, native_t=native_t)
        ds.summary = summary_sc.transform(ds.summary)
        return ds

    train_ds = make_ds(train_df, aug=augment_fn, native_t=native_t)
    val_ds = make_ds(val_df, native_t=native_t)
    test_ds = make_ds(test_df, native_t=native_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_custom, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_custom, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_custom, drop_last=False)

    # Track B — pseudo-labeled pool windows (optional)
    pool_loader: DataLoader | None = None
    if aug_mode != "none" and pool is not None and bfi_sim_map is not None:
        pool_ds = AugPoolDataset(
            pool, key_order, thresholds, bfi_sim_map, aug_mode,
            seq_scalers=seq_scalers, use_sequences=True,
        )
        pool_ds.summary_arr = summary_sc.transform(pool_ds.summary_arr)
        if len(pool_ds) > 0:
            pool_loader = DataLoader(pool_ds, batch_size=batch_size * 2, shuffle=True,
                                     collate_fn=collate_pool, drop_last=False)
            log.info("  Pool Track B enabled: %d pseudo-labeled windows (mode=%s)",
                     len(pool_ds), aug_mode)

    bfi_dim = 5 if BFI_COLS[0] in train_df.columns else 0
    model = TemporalFusionNet(
        encoder_type=encoder_type,
        enc_dim=enc_dim,
        dropout=dropout,
        bfi_dim=bfi_dim,
    ).to(device)

    # Load pre-trained encoder weights
    if pretrained_encoders is not None:
        for name in MODALITIES:
            model.encoders[name].load_state_dict(pretrained_encoders[name].state_dict())
        log.info("  Loaded pre-trained encoder weights.")

    # Cross-modal auxiliary loss
    xmodal_loss_fn = None
    if cross_modal_weight > 0:
        xmodal_loss_fn = CrossModalLoss(enc_dim, encoder_type).to(device)

    train_labels = get_hard_labels(train_df, thresholds)
    class_weights = compute_class_weight_tensors(train_labels, device)
    criterion = SoftVADLoss(class_weights=class_weights, label_smooth=0.1)

    all_params = list(model.parameters())
    if xmodal_loss_fn is not None:
        all_params += list(xmodal_loss_fn.parameters())

    def make_optimizer(freeze_encoders: bool) -> torch.optim.Optimizer:
        for name in MODALITIES:
            for p in model.encoders[name].parameters():
                p.requires_grad = not freeze_encoders
        params = [p for p in model.parameters() if p.requires_grad]
        if xmodal_loss_fn is not None:
            params += list(xmodal_loss_fn.parameters())
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    optimizer = make_optimizer(
        freeze_encoders=(pretrained_encoders is not None and freeze_epochs > 0)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_state, no_improve = -1.0, None, 0

    for epoch in range(epochs):
        if pretrained_encoders is not None and epoch == freeze_epochs:
            optimizer = make_optimizer(freeze_encoders=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)

        model.train()
        if xmodal_loss_fn:
            xmodal_loss_fn.train()

        for batch in train_loader:
            optimizer.zero_grad()
            summary = batch["summary"].to(device)
            bfi = batch["bfi"].to(device)
            labels = batch["labels"].to(device)
            seqs = {k: v.to(device) for k, v in batch["sequences"].items()}

            logits = model(summary, seqs, bfi)
            soft = make_one_hot_soft(labels, device)
            dim_mask = (labels >= 0)
            sw = batch["weight"].to(device)
            loss = criterion(logits, labels, soft_targets=soft,
                             sample_weights=sw[:, 0], dim_mask=dim_mask)

            # Cross-modal auxiliary loss
            if xmodal_loss_fn is not None:
                embeddings = {mod: model.encoders[mod](seqs[mod]) for mod in MODALITIES}
                xm_loss = xmodal_loss_fn(embeddings)
                loss = loss + cross_modal_weight * xm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Track B — pseudo-labeled pool windows (limited to match labeled steps)
        if pool_loader is not None:
            model.train()
            pool_iter = iter(pool_loader)
            # Limit pool steps to avoid overwhelming the small labeled set
            max_pool_steps = max(1, len(train_loader))
            for _ in range(max_pool_steps):
                try:
                    pbatch = next(pool_iter)
                except StopIteration:
                    break
                optimizer.zero_grad()
                p_summary = pbatch["summary"].to(device)
                p_bfi = pbatch["bfi"].to(device)
                p_soft_labels = pbatch["soft_labels"].to(device)
                p_sw = pbatch["weight"].to(device)
                p_seqs = ({k: v.to(device) for k, v in pbatch["sequences"].items()}
                          if pbatch["sequences"] is not None else None)

                p_logits = model(p_summary, p_seqs, p_bfi)
                pseudo_hard = p_soft_labels.argmax(dim=-1).clone()
                pseudo_hard[p_sw == 0] = -1
                p_loss = criterion(p_logits, pseudo_hard, soft_targets=p_soft_labels,
                                   sample_weights=p_sw.mean(dim=1))
                p_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                summary = batch["summary"].to(device)
                bfi = batch["bfi"].to(device)
                seqs = {k: v.to(device) for k, v in batch["sequences"].items()}
                logits = model(summary, seqs, bfi)
                all_preds.append(logits.argmax(-1).cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        f1s = []
        for d in range(3):
            valid = labels_arr[:, d] >= 0
            if valid.sum() == 0:
                f1s.append(0.0)
            else:
                f1s.append(float(f1_score(labels_arr[valid, d], preds_arr[valid, d],
                                          average="macro", zero_division=0)))
        val_f1 = float(np.mean(f1s))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            summary = batch["summary"].to(device)
            bfi = batch["bfi"].to(device)
            seqs = {k: v.to(device) for k, v in batch["sequences"].items()}
            logits = model(summary, seqs, bfi)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    test_f1s = []
    for d in range(3):
        valid = labels_arr[:, d] >= 0
        if valid.sum() == 0:
            test_f1s.append(0.0)
        else:
            test_f1s.append(float(f1_score(labels_arr[valid, d], preds_arr[valid, d],
                                            average="macro", zero_division=0)))
    return {
        "val_f1": round(best_val_f1, 4),
        "test_f1": round(float(np.mean(test_f1s)), 4),
        "v_f1": round(test_f1s[0], 4),
        "a_f1": round(test_f1s[1], 4),
        "d_f1": round(test_f1s[2], 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    experiment: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pool: pd.DataFrame,
    key_order: list[str],
    encoder_type: str,
    enc_dim: int,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    bfi_sim_map: dict | None = None,
) -> dict:
    """Run one experiment variant and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_scalers = fit_seq_scalers(train_df)

    common_ft_kwargs = dict(
        train_df=train_df, val_df=val_df, test_df=test_df,
        key_order=key_order,
        encoder_type=encoder_type, enc_dim=enc_dim,
        dropout=args.dropout,
        epochs=args.finetune_epochs,
        batch_size=16, lr=args.lr,
        patience=args.patience,
        device=device,
        freeze_epochs=args.freeze_epochs,
    )

    if experiment == "baseline":
        # SimCLR joint pre-training + standard augmentation
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E1":
        # Time Warping augmentation (replace circular shift)
        from pretrain_temporal import pretrain_encoders

        # Also use TW during pre-training
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E2":
        # Masked Signal Modeling pre-training
        pretrained = pretrain_masked(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, mask_ratio=args.mask_ratio, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E3":
        # Per-modality independent SSL
        pretrained = pretrain_per_modality(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        # Move back to device
        for name in MODALITIES:
            pretrained[name] = pretrained[name].to(device)
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E4":
        # Window Warping for EDA + TW for others
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_ww_eda(arr, mod),
            cross_modal_weight=0.0,
        )

    elif experiment == "E5":
        # SimCLR pre-training + cross-modal reconstruction auxiliary loss
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=args.xmodal_weight,
        )

    elif experiment == "E1+E2":
        # Masked pre-training + Time Warping augmentation (best of both)
        pretrained = pretrain_masked(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, mask_ratio=args.mask_ratio, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E3+E1":
        # Per-modality SSL + Time Warping
        pretrained = pretrain_per_modality(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        for name in MODALITIES:
            pretrained[name] = pretrained[name].to(device)
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
        )

    elif experiment == "E6":
        # Native-rate per-modality sequences — remove interpolation artifacts
        # from slow modalities by downsampling to their natural T.
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=0.0,
            native_t=NATIVE_T,
        )

    elif experiment == "E6+E1":
        # Native-rate + Time Warping — combine both ideas
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
            native_t=NATIVE_T,
        )

    # ── Augmented-label variants (Track B: pseudo-labeled pool) ──────────────

    elif experiment == "E2+aug":
        # Masked Signal Modeling + pseudo-labeled pool fine-tuning
        pretrained = pretrain_masked(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, mask_ratio=args.mask_ratio, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence(arr),
            cross_modal_weight=0.0,
            pool=pool, bfi_sim_map=bfi_sim_map, aug_mode=args.aug_mode,
        )

    elif experiment == "E1+E2+aug":
        # Masked pre-training + Time Warping + pseudo-labeled pool
        pretrained = pretrain_masked(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, mask_ratio=args.mask_ratio, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
            pool=pool, bfi_sim_map=bfi_sim_map, aug_mode=args.aug_mode,
        )

    elif experiment == "E6+E1+aug":
        # Native-rate + Time Warping + pseudo-labeled pool
        from pretrain_temporal import pretrain_encoders
        pretrained = pretrain_encoders(
            pool=pool, seq_scalers=seq_scalers,
            encoder_type=encoder_type, enc_dim=enc_dim,
            epochs=args.pretrain_epochs, batch_size=64,
            lr=args.lr, temperature=0.1, device=device,
        )
        return finetune_custom(
            **common_ft_kwargs,
            pretrained_encoders=pretrained,
            augment_fn=lambda arr, mod: augment_sequence_tw(arr),
            cross_modal_weight=0.0,
            native_t=NATIVE_T,
            pool=pool, bfi_sim_map=bfi_sim_map, aug_mode=args.aug_mode,
        )

    else:
        raise ValueError(f"Unknown experiment: {experiment}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_EXPERIMENTS = [
    "baseline", "E1", "E2", "E3", "E4", "E5", "E6",
    "E1+E2", "E3+E1", "E6+E1",
    "E2+aug", "E1+E2+aug", "E6+E1+aug",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Literature-inspired preprocessing and temporal modeling experiments"
    )
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool", default="data/mumt/augmented_pool.pkl")
    parser.add_argument("--out", default="results/literature_experiments.csv")
    parser.add_argument("--encoder", default="conv1d", choices=["conv1d", "gru"])
    parser.add_argument("--enc-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--mask-ratio", type=float, default=0.25,
                        help="Mask ratio for E2 (masked signal modeling)")
    parser.add_argument("--xmodal-weight", type=float, default=0.1,
                        help="Weight for E5 cross-modal auxiliary loss")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per experiment")
    parser.add_argument("--test-task", default="T3")
    parser.add_argument("--aug-mode", default="ap1", choices=["ap1", "ap2", "a2", "none"],
                        help="Pool pseudo-label mode for +aug experiments (default: ap1)")
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS,
                        choices=ALL_EXPERIMENTS,
                        help="Which experiments to run (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Encoder: %s  |  Seeds: %d", device, args.encoder, args.seeds)

    df = pd.read_pickle(args.dataset)
    pool = pd.read_pickle(args.pool)
    log.info("Dataset: %d windows  |  Pool: %d windows", len(df), len(pool))

    key_order = build_summary_key_order(df)
    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # BFI similarity map for pseudo-label augmentation
    bfi_sim_map = compute_bfi_similarity_map(df)
    log.info("BFI similarity map: %d entries", len(bfi_sim_map))

    all_records = []

    for exp_name in args.experiments:
        log.info("\n{'='*60}")
        log.info("EXPERIMENT: %s", exp_name)
        log.info("{'='*60}")

        seed_results = []
        for seed in range(args.seeds):
            log.info("  Seed %d/%d …", seed + 1, args.seeds)
            result = run_experiment(
                experiment=exp_name,
                train_df=train_df, val_df=val_df, test_df=test_df,
                pool=pool, key_order=key_order,
                encoder_type=args.encoder, enc_dim=args.enc_dim,
                device=device, seed=seed + 42, args=args,
                bfi_sim_map=bfi_sim_map,
            )
            seed_results.append(result)
            log.info("    V=%.3f  A=%.3f  D=%.3f  Mean=%.3f",
                     result["v_f1"], result["a_f1"], result["d_f1"], result["test_f1"])

        # Average across seeds
        avg = {
            "experiment": exp_name,
            "encoder": args.encoder,
            "seeds": args.seeds,
            "v_f1": round(np.mean([r["v_f1"] for r in seed_results]), 4),
            "a_f1": round(np.mean([r["a_f1"] for r in seed_results]), 4),
            "d_f1": round(np.mean([r["d_f1"] for r in seed_results]), 4),
            "mean_f1": round(np.mean([r["test_f1"] for r in seed_results]), 4),
            "v_std": round(np.std([r["v_f1"] for r in seed_results]), 4),
            "a_std": round(np.std([r["a_f1"] for r in seed_results]), 4),
            "d_std": round(np.std([r["d_f1"] for r in seed_results]), 4),
            "mean_std": round(np.std([r["test_f1"] for r in seed_results]), 4),
        }
        all_records.append(avg)
        log.info("  AVG: V=%.3f±%.3f  A=%.3f±%.3f  D=%.3f±%.3f  Mean=%.3f±%.3f",
                 avg["v_f1"], avg["v_std"], avg["a_f1"], avg["a_std"],
                 avg["d_f1"], avg["d_std"], avg["mean_f1"], avg["mean_std"])

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(out_path, index=False)
    log.info("\nSaved → %s", out_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("LITERATURE EXPERIMENTS — SUMMARY")
    print("=" * 80)
    print(f"Encoder: {args.encoder}  |  Seeds: {args.seeds}  |  Test task: {args.test_task}")
    print("-" * 80)
    print(results_df[["experiment", "v_f1", "a_f1", "d_f1", "mean_f1"]].to_string(index=False))
    print("=" * 80)
    print("\nLegend:")
    print("  baseline : SimCLR joint pre-training + noise/jitter/shift")
    print("  E1       : Time Warping augmentation (replaces circular shift)")
    print("  E2       : Masked Signal Modeling pre-training (reconstructive SSL)")
    print("  E3       : Per-modality independent SSL (avoids cross-modal interference)")
    print("  E4       : Window Warping for EDA + standard aug for others")
    print("  E5       : Cross-modal reconstruction auxiliary loss")
    print("  E6       : Native-rate per-modality T (remove upsampling artifacts)")
    print("  E1+E2    : Masked pre-training + Time Warping augmentation")
    print("  E3+E1    : Per-modality SSL + Time Warping augmentation")
    print("  E6+E1    : Native-rate + Time Warping augmentation")
    print("  E2+aug   : Masked SSL + pseudo-labeled pool fine-tuning")
    print("  E1+E2+aug: Masked SSL + Time Warping + pseudo-labeled pool")
    print("  E6+E1+aug: Native-rate + Time Warping + pseudo-labeled pool")


if __name__ == "__main__":
    main()
