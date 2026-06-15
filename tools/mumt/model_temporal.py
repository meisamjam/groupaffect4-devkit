"""model_temporal.py

Per-modality temporal encoder architectures for physiological affect sensing.

Two encoder types:
  Conv1D  – two-layer Conv1d with stride + global average pooling
  GRU     – bidirectional GRU with mean pooling over time

Each modality gets its own independent encoder; outputs are concatenated
with the 49-d summary feature vector before the shared VAD trunk.

Architecture:
  ┌─ gaze_seq (200×9) ─→ Encoder_gaze  → 32-d ─┐
  ├─ pupil_seq(200×3) ─→ Encoder_pupil → 32-d  │
  ├─ eda_seq  (200×5) ─→ Encoder_eda   → 32-d  ├─→ concat → 209-d (conv)
  ├─ ppg_seq  (200×3) ─→ Encoder_ppg   → 32-d  │            or 369-d (gru)
  └─ imu_seq  (200×6) ─→ Encoder_imu   → 32-d ─┘
  summary_feat (49-d) ─────────────────────────────────────→ (added above)

  Trunk: Linear(in → 128) → BN → ReLU → Drop(0.3)
       → Linear(128 → 64) → BN → ReLU → Drop(0.3)
  Heads: 3 × Linear(64 → 32) → ReLU → Linear(32 → 3)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Modality specs ─────────────────────────────────────────────────────────────

MODALITY_DIMS: dict[str, int] = {
    "gaze":  9,
    "pupil": 3,
    "eda":   5,
    "ppg":   3,
    "imu":   6,
}
MODALITIES = list(MODALITY_DIMS.keys())
SUMMARY_DIM = 49


# ── Per-modality encoders ──────────────────────────────────────────────────────

class Conv1DEncoder(nn.Module):
    """Lightweight two-layer Conv1D encoder for a single physiological modality.

    Input : (B, T, D) where T=200, D=input_dim
    Output: (B, out_dim)
    """
    def __init__(self, input_dim: int, out_dim: int = 32) -> None:
        super().__init__()
        mid = max(out_dim, 16)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, mid, kernel_size=8, stride=4),
            nn.BatchNorm1d(mid),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid, out_dim, kernel_size=4, stride=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x).squeeze(-1)   # (B, out_dim)
        return x


class GRUEncoder(nn.Module):
    """Bidirectional GRU encoder for a single physiological modality.

    Input : (B, T, D)
    Output: (B, out_dim)  where out_dim = 2 * hidden_size
    """
    def __init__(self, input_dim: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.out_dim = 2 * hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.gru(x)              # (B, T, 2H)
        return out.mean(dim=1)            # global mean pooling


# ── Full network ───────────────────────────────────────────────────────────────

class TemporalFusionNet(nn.Module):
    """Per-modality temporal encoders fused with summary features.

    Two operating modes per forward pass, controlled by the ``sequences`` arg:
    - Full mode   (sequences provided): temporal encoders + summary features
    - Summary mode (sequences=None):    summary features only (skip encoders)

    This allows mixed batches: labeled windows use full mode, augmented pool
    windows (summary features only) use summary mode — a two-track training
    strategy that combines raw-sequence supervision with soft-label augmentation.

    Parameters
    ----------
    encoder_type : 'conv1d' or 'gru'
    enc_dim      : output dimension of each modality encoder (32 for conv1d)
    dropout      : dropout probability in trunk
    bfi_dim      : BFI-44 personality dimension (5). Set to 0 to disable.
    """

    def __init__(
        self,
        encoder_type: str = "conv1d",
        enc_dim: int = 32,
        dropout: float = 0.3,
        bfi_dim: int = 5,
    ) -> None:
        super().__init__()

        self.encoder_type  = encoder_type
        self.enc_dim       = enc_dim
        self.bfi_dim       = bfi_dim
        self.n_modalities  = len(MODALITIES)

        # Build per-modality encoders
        self.encoders = nn.ModuleDict()
        for name, d_m in MODALITY_DIMS.items():
            if encoder_type == "conv1d":
                self.encoders[name] = Conv1DEncoder(d_m, enc_dim)
            else:
                self.encoders[name] = GRUEncoder(d_m, hidden_size=enc_dim)

        # Actual encoder output dim per modality
        if encoder_type == "gru":
            actual_enc_dim = 2 * enc_dim   # bidirectional
        else:
            actual_enc_dim = enc_dim

        # Fusion dimension: temporal embeddings + summary features
        temporal_total = self.n_modalities * actual_enc_dim
        trunk_in_full    = temporal_total + SUMMARY_DIM + bfi_dim
        trunk_in_summary = SUMMARY_DIM + bfi_dim

        self.trunk_in_full    = trunk_in_full
        self.temporal_total   = temporal_total

        # Single projection — always maps full (temporal + summary + bfi) → 128.
        # When sequences are unavailable, temporal part is zero-padded so both
        # training tracks share identical projection weights (no gradient conflict).
        self.proj = nn.Linear(trunk_in_full, 128)

        self.trunk = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Three VAD heads (per-dimension)
        self.vad_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
            for _ in range(3)
        ])

        # BFI personality heads (per VAD dim) — optional
        if bfi_dim > 0:
            self.bfi_proj = nn.ModuleList([
                nn.Linear(bfi_dim, 64, bias=False)
                for _ in range(3)
            ])
            # Zero-init so BFI starts as identity modulation
            for proj in self.bfi_proj:
                nn.init.zeros_(proj.weight)

    def forward(
        self,
        summary: torch.Tensor,
        sequences: dict[str, torch.Tensor] | None = None,
        bfi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        summary   : (B, 49) summary feature vector
        sequences : dict mapping modality name → (B, T, D) raw sequence tensor.
                    If None, skip temporal encoders (summary-only mode).
        bfi       : (B, 5) BFI personality vector, or None.

        Returns
        -------
        logits : (B, 3, 3) — batch × VAD dimension × class
        """
        B = summary.shape[0]
        if sequences is not None:
            enc_out = []
            for name in MODALITIES:
                enc_out.append(self.encoders[name](sequences[name]))
            temporal = torch.cat(enc_out, dim=-1)  # (B, temporal_total)
        else:
            # Zero-pad the temporal part — single proj always used, no gradient conflict
            temporal = torch.zeros(B, self.temporal_total, device=summary.device)

        if self.bfi_dim > 0:
            bfi_in = bfi if bfi is not None else torch.zeros(B, self.bfi_dim, device=summary.device)
            x = torch.cat([temporal, summary, bfi_in], dim=-1)
        else:
            x = torch.cat([temporal, summary], dim=-1)
        x = self.proj(x)

        feat = self.trunk(x)   # (B, 64)

        # Optional per-dim BFI modulation
        if bfi is not None and self.bfi_dim > 0:
            bfi_shifts = [proj(bfi) for proj in self.bfi_proj]  # 3 × (B, 64)
        else:
            bfi_shifts = [None] * 3

        logits = []
        for i, head in enumerate(self.vad_heads):
            f = feat + bfi_shifts[i] if bfi_shifts[i] is not None else feat
            logits.append(head(f))          # (B, 3)

        return torch.stack(logits, dim=1)  # (B, 3, 3)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Soft-VAD loss (reused from main MLP training) ─────────────────────────────

class SoftVADLoss(nn.Module):
    """Weighted soft cross-entropy for VAD.

    Handles both hard labels (one-hot, weight=1) and soft GP pseudo-labels.
    Supports per-dimension class weights and label smoothing.
    """
    def __init__(
        self,
        class_weights: list[torch.Tensor] | None = None,
        label_smooth: float = 0.1,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights  # list of 3 tensors, each (3,)
        self.eps = label_smooth

    def forward(
        self,
        logits: torch.Tensor,          # (B, 3, 3)  — VAD dim × class
        targets: torch.Tensor,          # (B, 3)     — hard class indices
        soft_targets: torch.Tensor | None = None,  # (B, 3, 3) — soft probs
        sample_weights: torch.Tensor | None = None, # (B,)
        dim_mask: torch.Tensor | None = None,       # (B, 3) bool
    ) -> torch.Tensor:
        B = logits.size(0)
        loss = torch.tensor(0.0, device=logits.device)
        n_valid = 0

        for d in range(3):
            log_p = F.log_softmax(logits[:, d, :], dim=-1)  # (B, 3)

            if soft_targets is not None:
                p_target = soft_targets[:, d, :]
            else:
                p_target = F.one_hot(targets[:, d].clamp(min=0), 3).float()

            # Label smoothing
            p_smooth = (1 - self.eps) * p_target + self.eps / 3.0

            # Class weights
            if self.class_weights is not None:
                cw = self.class_weights[d].to(logits.device)
                per_sample = -(p_smooth * log_p * cw.unsqueeze(0)).sum(-1)
            else:
                per_sample = -(p_smooth * log_p).sum(-1)

            # Valid mask (ignore -1 labels and T4 Dominance)
            if dim_mask is not None:
                valid = dim_mask[:, d] & (targets[:, d] >= 0)
            else:
                valid = targets[:, d] >= 0

            if valid.sum() == 0:
                continue

            per_sample = per_sample[valid]
            if sample_weights is not None:
                per_sample = per_sample * sample_weights[valid]

            loss = loss + per_sample.mean()
            n_valid += 1

        return loss / max(n_valid, 1)
