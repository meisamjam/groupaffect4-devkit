"""model_affectai.py

MuMTAffect adapted for GroupAffect-4 group interaction data.

Differences from the original MuMTAffect (Seikavandi et al., 2025):
  - 5 modality encoders instead of 4 (Gaze, Pupil, EDA, PPG, IMU — no Facial AU)
  - 3 emotion-prediction heads (Valence, Arousal, Dominance) instead of 2
  - Fusion transformer input is 5×T segments
  - Full CUDA / mixed-precision compatible
  - Everything else (transformer depths, heads, FFN dims, dropout) unchanged

Architecture overview:
  Gaze encoder (Transformer)  → T=16 latent
  Pupil encoder (Transformer) → T=16 latent
  EDA encoder (Transformer)   → T=16 latent
  PPG encoder (Transformer)   → T=16 latent
  IMU encoder (Transformer)   → T=16 latent
         ↓ concat → Cross-modal Fusion Transformer (5×16=80 tokens)
  TaskAttentionTemporal → personality branch | emotion branch
  CNN pooling (separate per branch)
  Trial-feature MLP (summary statistics)
  ┌─ Personality head: MLP → 5 traits (regression)
  ├─ Valence head: attention + MLP → 3 classes
  ├─ Arousal head: attention + MLP → 3 classes
  └─ Dominance head: attention + MLP → 3 classes
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverseFn(torch.autograd.Function):
    """Gradient reversal layer for adversarial representation learning."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal during backward pass."""
    return _GradReverseFn.apply(x, lambd)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (channel recalibration)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C) with channel-wise scaling."""
        s = x.mean(dim=1)               # (B, C) — temporal squeeze
        w = self.fc(s).unsqueeze(1)      # (B, 1, C) — per-channel excitation
        return x * w                     # (B, T, C) — scale


# ---------------------------------------------------------------------------
# Physiological Contrastive Person Embedding (PCPE) loss
# ---------------------------------------------------------------------------

class PhysiologicalContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss over L2-normalised person embeddings.

    Pulls together windows from the same subject and pushes apart
    windows from different subjects — learning a physiological identity
    signature without BFI labels.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        """z: (B, D) L2-normalised embeddings, subject_idx: (B,) int labels."""
        B = z.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Force float32 for numerical stability under AMP
        z = F.normalize(z.float(), dim=-1)

        # Pairwise cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature   # (B, B) float32

        # Positive mask: same subject, different sample index
        subj_eq = subject_idx.unsqueeze(0) == subject_idx.unsqueeze(1)  # (B, B)
        diag_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = subj_eq & ~diag_mask

        # For each anchor, positives = same person, negatives = everyone else
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=z.device)

        # Mask self-similarities (large negative in float32 range)
        sim = sim.masked_fill(diag_mask, -1e9)
        log_softmax = F.log_softmax(sim, dim=1)

        # Mean log-prob over positives (only for anchors that have positives)
        loss = -(log_softmax * pos_mask.float()).sum(1) / pos_mask.float().sum(1).clamp(min=1)
        loss = loss[has_pos]
        return loss.mean()


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Modality-specific transformer encoder
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Single-layer transformer encoder for one physiological modality.

    Projects raw features → d_model, adds positional encoding, passes through
    one TransformerEncoder layer, then downsamples T=400 → T_out=16 via
    deterministic equal-segment averaging.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        nhead: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.4,
        t_out: int = 16,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.se = SEBlock(d_model, reduction=4) if use_se else None
        self.t_out = t_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, t_out, d_model)."""
        x = self.proj(x)                        # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                 # (B, T, d_model)

        # SE channel recalibration (before downsampling)
        if self.se is not None:
            x = self.se(x)

        # Deterministic downsampling: equal-segment averaging
        B, T, D = x.shape
        segments = torch.stack(
            [x[:, i * T // self.t_out : (i + 1) * T // self.t_out].mean(dim=1)
             for i in range(self.t_out)],
            dim=1,
        )  # (B, t_out, D)
        return segments


class GRUModalityEncoder(nn.Module):
    """GRU-based modality encoder alternative to `ModalityEncoder`.

    Projects raw features → d_model, runs a (bi-)GRU, then downsamples
    to `t_out` via equal-segment averaging to match the transformer's output shape.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        t_out: int = 16,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)
        # Project GRU output to d_model for downstream compatibility
        self.out_proj = nn.Linear(self.out_dim, d_model)
        self.se = SEBlock(d_model, reduction=4) if use_se else None
        self.t_out = t_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, t_out, d_model)."""
        x = self.proj(x)  # (B, T, d_model)
        out, _ = self.gru(x)  # (B, T, out_dim)
        out = self.out_proj(out)  # (B, T, d_model)

        if self.se is not None:
            out = self.se(out)

        B, T, D = out.shape
        segments = torch.stack(
            [out[:, i * T // self.t_out : (i + 1) * T // self.t_out].mean(dim=1)
             for i in range(self.t_out)],
            dim=1,
        )
        return segments


# ---------------------------------------------------------------------------
# Cross-modal fusion transformer
# ---------------------------------------------------------------------------

class FusionTransformer(nn.Module):
    """Fuses latents from N modalities via a single transformer layer.

    Each modality latent (B, t_out, d_in) is projected to d_model/2, then
    concatenated along the time axis before being fed to the transformer.
    """

    def __init__(
        self,
        n_modalities: int,
        d_in: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.25,
        t_out: int = 16,
    ) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d_in, d_model // 2) for _ in range(n_modalities)])
        # After concat: sequence length = n_modalities * t_out
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model // 2, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=min(dropout + 0.1, 0.5), batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model // 2, d_model)
        self.n_modalities = n_modalities
        self.t_out = t_out

    def forward(self, *modality_latents: torch.Tensor) -> torch.Tensor:
        """modality_latents: each (B, t_out, d_in) → (B, n*t_out, d_model)."""
        projected = [self.projs[i](m) for i, m in enumerate(modality_latents)]
        fused = torch.cat(projected, dim=1)     # (B, n*t_out, d_model/2)
        fused = self.transformer(fused)          # (B, n*t_out, d_model/2)
        fused = self.out_proj(fused)             # (B, n*t_out, d_model)
        return fused


class GRUFusion(nn.Module):
    """Fusion module implemented with a GRU over concatenated modality latents.

    Each modality latent (B, t_out, d_in) is projected to d_model/2, concatenated
    along the time axis and passed through a GRU. The output is projected to
    `d_model` to match the interface of `FusionTransformer`.
    """

    def __init__(
        self,
        n_modalities: int,
        d_in: int = 64,
        d_model: int = 128,
        hidden_size: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.1,
        t_out: int = 16,
    ) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d_in, d_model // 2) for _ in range(n_modalities)])
        self.gru = nn.GRU(
            input_size=d_model // 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(hidden_size * (2 if bidirectional else 1), d_model)
        self.n_modalities = n_modalities
        self.t_out = t_out

    def forward(self, *modality_latents: torch.Tensor) -> torch.Tensor:
        projected = [self.projs[i](m) for i, m in enumerate(modality_latents)]
        fused = torch.cat(projected, dim=1)  # (B, n*t_out, d_model/2)
        out, _ = self.gru(fused)              # (B, n*t_out, out_dim)
        out = self.out_proj(out)              # (B, n*t_out, d_model)
        return out


# ---------------------------------------------------------------------------
# Task-specific attention
# ---------------------------------------------------------------------------

class TaskAttentionTemporal(nn.Module):
    """Single-head scaled dot-product attention with learnable queries.

    Produces one context vector per query (personality / emotion branch).
    """

    def __init__(self, d_model: int = 128, n_queries: int = 2) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, S, d_model) → list of n_queries tensors (B, d_model)."""
        # x as keys and values
        scores = torch.einsum("qd,bsd->bqs", self.queries, x) * self.scale
        weights = F.softmax(scores, dim=-1)                     # (B, n_queries, S)
        contexts = torch.einsum("bqs,bsd->bqd", weights, x)    # (B, n_queries, d_model)
        return [contexts[:, i] for i in range(contexts.size(1))]


class TaskAttentionTemporalScaled(nn.Module):
    """Per-query learned temperature for scale-asymmetric attention.

    Motivated by CTSEM arousal half-life (1.2s, fast → sharp attention)
    vs valence half-life (12.3s, slow → smooth attention).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_queries: int = 4,
        init_log_temps: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        # Log-scale temperatures (exp ensures positivity)
        # Default init: personality=neutral, valence=warm(slow), arousal=cold(fast), dom=neutral
        if init_log_temps is None:
            #                  P      V       A       D
            init_log_temps = [0.0, 0.7, -0.5, 0.0]
        self.log_temps = nn.Parameter(
            torch.tensor(init_log_temps[:n_queries], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, S, d_model) → list of n_queries tensors (B, d_model)."""
        # Compute scores: (B, n_queries, S)
        scores = torch.einsum("qd,bsd->bqs", self.queries, x)
        # Per-query temperature scaling: exp(log_temp) is strictly positive
        temps = self.log_temps.exp().unsqueeze(0).unsqueeze(-1)  # (1, n_queries, 1)
        scores = scores / temps
        weights = F.softmax(scores, dim=-1)              # (B, n_queries, S)
        contexts = torch.einsum("bqs,bsd->bqd", weights, x)  # (B, n_queries, d_model)
        return [contexts[:, i] for i in range(contexts.size(1))]


# ---------------------------------------------------------------------------
# CNN branch
# ---------------------------------------------------------------------------

def _make_conv_block(in_ch: int, out_ch: int, k: int, s: int, p: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class PersonalityCNNBranch(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _make_conv_block(d_model, d_model, k=3, s=1, p=1, dropout=dropout),
            _make_conv_block(d_model, d_model, k=3, s=2, p=1, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) → (B, d_model) (unsqueeze/squeeze T dimension)."""
        x = x.unsqueeze(-1)            # (B, d_model, 1)
        x = self.net(x)
        return x.flatten(1)            # (B, d_model * T')


class EmotionCNNBranch(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _make_conv_block(d_model, d_model, k=3, s=2, p=1, dropout=dropout),
            _make_conv_block(d_model, d_model, k=3, s=2, p=1, dropout=dropout),
            _make_conv_block(d_model, d_model, k=3, s=2, p=1, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.net(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Emotion head (one per VAD dimension)
# ---------------------------------------------------------------------------

class EmotionHead(nn.Module):
    """Single-head attention + 2-layer MLP for one emotion dimension."""

    def __init__(self, d_in: int, n_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_in) * 0.02)
        self.scale = d_in ** -0.5
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, n_classes),
        )

    def forward(
        self, emotion_feat: torch.Tensor, personality_embed: torch.Tensor | None = None,
        personality_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """emotion_feat: (B, d_in), personality_embed: (B, d_in) optional.

        Returns logits (B, n_classes).
        """
        if personality_embed is not None:
            gate = personality_gate if personality_gate is not None else 0.1
            feat = emotion_feat + gate * personality_embed
        else:
            feat = emotion_feat
        # Self-attention over single token
        score = (feat * self.q).sum(-1, keepdim=True) * self.scale
        feat = feat * score.sigmoid()
        return self.mlp(feat)


# ---------------------------------------------------------------------------
# Trial-level feature MLP
# ---------------------------------------------------------------------------

class TrialFeatureMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MuMTAffectGroupAffect(nn.Module):
    """MuMTAffect adapted for GroupAffect-4.

    Five physiological modalities (Gaze, Pupil, EDA, PPG, IMU) and three
    emotion outputs (Valence, Arousal, Dominance).  Personality regression is
    used as an auxiliary task to build user-specific embeddings.

    Compatible with CUDA and torch.cuda.amp automatic mixed precision.

    Args:
        gaze_dim:        Number of features in the gaze sequence.
        pupil_dim:       Number of features in the pupil sequence.
        eda_dim:         Number of features in the EDA sequence.
        ppg_dim:         Number of features in the PPG sequence.
        imu_dim:         Number of features in the IMU sequence.
        summary_dim:     Dimension of the trial-level summary features.
        n_subjects:      Total number of unique subjects (for user embedding).
        n_personality:   Number of personality traits to predict (default: 5).
        n_emotion_classes: Number of bins per emotion dimension (default: 3).
        d_model_enc:     Transformer d_model for modality encoders (default: 64).
        d_model_fuse:    Transformer d_model for fusion (default: 128).
        t_out:           Downsampled sequence length per modality (default: 16).
    """

    def __init__(
        self,
        gaze_dim: int = 9,
        pupil_dim: int = 3,
        eda_dim: int = 5,
        ppg_dim: int = 3,
        imu_dim: int = 6,
        summary_dim: int = 40,
        n_subjects: int = 50,
        n_personality: int = 5,
        n_emotion_classes: int = 3,
        d_model_enc: int = 64,
        d_model_fuse: int = 128,
        t_out: int = 16,
        n_tasks: int = 5,
        per_dim_queries: bool = False,
        per_dim_projections: bool = False,
        use_se_blocks: bool = False,
        use_scaled_attention: bool = False,
        use_global_token: bool = False,
        per_dim_gate: bool = False,
        use_gru: bool = False,
        personality_classes: int = 1,
        personality_as_input: bool = False,
    ) -> None:
        super().__init__()

        self.n_modalities = 5
        self.per_dim_queries = per_dim_queries
        self.per_dim_projections = per_dim_projections
        self.use_global_token = use_global_token
        self.n_personality = n_personality
        self.personality_classes = personality_classes
        self.personality_as_input = personality_as_input

        # --- Modality encoders ---
        if use_gru:
            self.gaze_encoder = GRUModalityEncoder(gaze_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.pupil_encoder = GRUModalityEncoder(pupil_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.eda_encoder = GRUModalityEncoder(eda_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.ppg_encoder = GRUModalityEncoder(ppg_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.imu_encoder = GRUModalityEncoder(imu_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
        else:
            self.gaze_encoder = ModalityEncoder(gaze_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.pupil_encoder = ModalityEncoder(pupil_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.eda_encoder = ModalityEncoder(eda_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.ppg_encoder = ModalityEncoder(ppg_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)
            self.imu_encoder = ModalityEncoder(imu_dim, d_model=d_model_enc, t_out=t_out, use_se=use_se_blocks)

        # --- Cross-modal fusion ---
        if use_gru:
            self.fusion = GRUFusion(
                n_modalities=self.n_modalities,
                d_in=d_model_enc,
                d_model=d_model_fuse,
                hidden_size=d_model_fuse // 2,
                bidirectional=True,
                dropout=0.0,
                t_out=t_out,
            )
        else:
            self.fusion = FusionTransformer(
                n_modalities=self.n_modalities,
                d_in=d_model_enc,
                d_model=d_model_fuse,
                nhead=4,
                ffn_dim=2048,
                dropout=0.25,
                t_out=t_out,
            )

        # --- Task-specific attention ---
        # per_dim_queries: 4 queries (personality, valence, arousal, dominance)
        # otherwise: 2 queries (personality, shared emotion)
        n_queries = 4 if per_dim_queries else 2
        if use_scaled_attention:
            self.task_attention = TaskAttentionTemporalScaled(
                d_model=d_model_fuse, n_queries=n_queries
            )
        else:
            self.task_attention = TaskAttentionTemporal(d_model=d_model_fuse, n_queries=n_queries)

        # --- CNN branches ---
        self.personality_cnn = PersonalityCNNBranch(d_model=d_model_fuse, dropout=0.4)
        self.emotion_cnn = EmotionCNNBranch(d_model=d_model_fuse, dropout=0.25)

        # Infer CNN output dim dynamically via a dummy forward pass
        # Use eval mode so BatchNorm works with batch_size=1
        self.personality_cnn.eval()
        self.emotion_cnn.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, d_model_fuse)
            p_out_dim = self.personality_cnn(dummy).shape[-1]
            e_out_dim = self.emotion_cnn(dummy).shape[-1]
        self.personality_cnn.train()
        self.emotion_cnn.train()

        # --- Trial-feature MLP ---
        # If using personality as input, expand summary_dim by 5 (Big Five)
        trial_input_dim = summary_dim + (n_personality if personality_as_input else 0)
        self.trial_mlp = TrialFeatureMLP(trial_input_dim, out_dim=64, dropout=0.3)

        # --- Personality head ---
        p_head_in = p_out_dim + 64  # CNN + trial features
        p_out_units = n_personality * personality_classes if personality_classes > 1 else n_personality
        self.personality_head = nn.Sequential(
            nn.Linear(p_head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, p_out_units),
        )

        # --- Per-dimension projection layers (optional) ---
        if per_dim_projections:
            self.v_proj = nn.Sequential(nn.Linear(e_out_dim, e_out_dim), nn.ReLU(), nn.Dropout(0.2))
            self.a_proj = nn.Sequential(nn.Linear(e_out_dim, e_out_dim), nn.ReLU(), nn.Dropout(0.2))
            self.d_proj = nn.Sequential(nn.Linear(e_out_dim, e_out_dim), nn.ReLU(), nn.Dropout(0.2))

        # --- Emotion heads (one per VAD dimension) ---
        e_head_in = e_out_dim
        self.valence_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)
        self.arousal_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)
        self.dominance_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)

        # --- Per-dimension personality gate ---
        if per_dim_gate:
            # [valence, arousal, dominance] — asymmetric init from CTSEM moderating effects
            self.personality_gate = nn.Parameter(
                torch.tensor([0.05, 0.15, 0.08])
            )
        else:
            self.personality_gate = nn.Parameter(torch.tensor(0.1))

        # --- Subject embedding (kept for pretrain compatibility) ---
        self.subject_embed = nn.Embedding(n_subjects + 2, 16)  # +2 for padding/mixed

        # --- Task context projection (one-hot → d_model_fuse, additive to emotion branch) ---
        self.task_proj = nn.Sequential(
            nn.Linear(n_tasks, d_model_fuse),
            nn.ReLU(),
        )

    def forward(
        self,
        gaze_seq: torch.Tensor,      # (B, T, gaze_dim)
        pupil_seq: torch.Tensor,     # (B, T, pupil_dim)
        eda_seq: torch.Tensor,       # (B, T, eda_dim)
        ppg_seq: torch.Tensor,       # (B, T, ppg_dim)
        imu_seq: torch.Tensor,       # (B, T, imu_dim)
        summary: torch.Tensor,       # (B, summary_dim)
        user_ids: torch.Tensor,      # (B,)
        personality_gt: torch.Tensor | None = None,  # (B, 5) optional teacher forcing or input
        task_onehot: torch.Tensor | None = None,     # (B, n_tasks) one-hot task label
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns a dict with keys:
          'valence_logits'   (B, n_emotion_classes)
          'arousal_logits'   (B, n_emotion_classes)
          'dominance_logits' (B, n_emotion_classes)
          'personality_pred' (B, n_personality)
          'personality_embed' (B, d_model_fuse)  — for auxiliary losses
        """
        # Encode each modality
        gaze_lat = self.gaze_encoder(gaze_seq)     # (B, t_out, d_enc)
        pupil_lat = self.pupil_encoder(pupil_seq)  # (B, t_out, d_enc)
        eda_lat = self.eda_encoder(eda_seq)         # (B, t_out, d_enc)
        ppg_lat = self.ppg_encoder(ppg_seq)         # (B, t_out, d_enc)
        imu_lat = self.imu_encoder(imu_seq)         # (B, t_out, d_enc)

        # Cross-modal fusion
        fused = self.fusion(gaze_lat, pupil_lat, eda_lat, ppg_lat, imu_lat)  # (B, 5*t_out, d_fuse)

        # Optional: append global mean token for valence context
        if self.use_global_token:
            global_token = fused.mean(dim=1, keepdim=True)   # (B, 1, d_fuse)
            fused_aug = torch.cat([fused, global_token], dim=1)  # (B, 5*t_out+1, d_fuse)
        else:
            fused_aug = fused

        # Task-specific attention
        if self.per_dim_queries:
            personality_ctx, valence_ctx, arousal_ctx, dominance_ctx = self.task_attention(fused_aug)
        else:
            personality_ctx, emotion_ctx = self.task_attention(fused_aug)
            valence_ctx = arousal_ctx = dominance_ctx = emotion_ctx

        # Subject embedding (optional, used as conditioning)
        user_ids_clamp = user_ids.clamp(min=0, max=self.subject_embed.num_embeddings - 1)
        subj_emb = self.subject_embed(user_ids_clamp)   # (B, 16)

        # Trial feature summary — optionally augmented with personality
        if self.personality_as_input and personality_gt is not None:
            # Concatenate personality as input features: (B, summary_dim + 5)
            summary_aug = torch.cat([summary, personality_gt.float()], dim=-1)
        else:
            summary_aug = summary
        trial_feat = self.trial_mlp(summary_aug)         # (B, 64)

        # --- Personality branch ---
        p_feat = self.personality_cnn(personality_ctx)   # (B, p_out_dim)
        p_in = torch.cat([p_feat, trial_feat], dim=-1)
        personality_pred = self.personality_head(p_in)   # (B, n_p_out)
        if self.personality_classes > 1:
            personality_pred = personality_pred.view(-1, self.n_personality, self.personality_classes)

        # --- Emotion branch ---
        v_feat = self.emotion_cnn(valence_ctx)
        a_feat = self.emotion_cnn(arousal_ctx)
        d_feat = self.emotion_cnn(dominance_ctx)

        # Per-dimension projections (dimension-specific feature spaces)
        if self.per_dim_projections:
            v_feat = self.v_proj(v_feat)
            a_feat = self.a_proj(a_feat)
            d_feat = self.d_proj(d_feat)

        # Inject task context as additive conditioning on the emotion branch
        if task_onehot is not None:
            task_emb = self.task_proj(task_onehot.float())  # (B, d_model_fuse)
            if task_emb.shape[-1] == v_feat.shape[-1]:
                v_feat = v_feat + task_emb
                a_feat = a_feat + task_emb
                d_feat = d_feat + task_emb

        p_embed = personality_ctx

        # Per-dimension gating: scalar gate broadcasts, 3-vector indexes per-dim
        if self.personality_gate.dim() == 0:
            v_gate = self.personality_gate
            a_gate = self.personality_gate
            d_gate = self.personality_gate
        else:
            v_gate = self.personality_gate[0].sigmoid()
            a_gate = self.personality_gate[1].sigmoid()
            d_gate = self.personality_gate[2].sigmoid()

        valence_logits = self.valence_head(v_feat, p_embed, v_gate)
        arousal_logits = self.arousal_head(a_feat, p_embed, a_gate)
        dominance_logits = self.dominance_head(d_feat, p_embed, d_gate)

        # Global mean pool of fused sequence — used by PretrainingHeads
        fused_pooled = fused.mean(dim=1)  # (B, d_fuse)

        return {
            "valence_logits": valence_logits,
            "arousal_logits": arousal_logits,
            "dominance_logits": dominance_logits,
            "personality_pred": personality_pred,
            "personality_embed": personality_ctx,
            "fused_pooled": fused_pooled,
        }


# ---------------------------------------------------------------------------
# Pre-training heads + loss  (Phase 0)
# ---------------------------------------------------------------------------

class PretrainingHeads(nn.Module):
    """Self-supervised auxiliary heads attached to ``fused_pooled``.

    Targets (all use the global mean-pooled fusion representation):
      task_cls      — 5-way task classification (T0–T4)
      subject_cls   — N_subjects-way subject identity classification
      session_cls   — N_sessions-way session/group classification
      personality   — 5-trait BFI-44 regression
      sex_cls       — binary sex classification (0=female, 1=male)
      age_reg       — continuous age regression
            next_summary  — next-window summary feature regression
    """

    def __init__(
        self,
        d_fuse: int = 128,
        n_tasks: int = 5,
        n_subjects: int = 40,
        n_sessions: int = 10,
        n_personality: int = 5,
        summary_dim: int = 40,
    ) -> None:
        super().__init__()

        self.user_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.2),
        )
        self.moment_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.2),
        )

        # Primary objectives
        self.task_cls = nn.Linear(d_fuse, n_tasks)
        self.subject_cls = nn.Linear(d_fuse, n_subjects)
        self.session_cls = nn.Linear(d_fuse, n_sessions)
        self.personality = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(), nn.Linear(d_fuse // 2, n_personality)
        )
        self.sex_cls = nn.Linear(d_fuse, 2)  # binary: female / male
        self.age_reg = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(), nn.Linear(d_fuse // 2, 1)
        )
        self.next_summary = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Linear(d_fuse, summary_dim)
        )

        # Adversarial objectives
        self.adv_task_from_user = nn.Linear(d_fuse, n_tasks)
        self.adv_session_from_user = nn.Linear(d_fuse, n_sessions)
        self.adv_next_summary_from_user = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Linear(d_fuse, summary_dim)
        )

        self.adv_subject_from_moment = nn.Linear(d_fuse, n_subjects)
        self.adv_personality_from_moment = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(), nn.Linear(d_fuse // 2, n_personality)
        )
        self.adv_sex_from_moment = nn.Linear(d_fuse, 2)
        self.adv_age_from_moment = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(), nn.Linear(d_fuse // 2, 1)
        )

    def forward(self, fused_pooled: torch.Tensor, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        """fused_pooled: (B, d_fuse)."""
        user_feat = self.user_proj(fused_pooled)
        moment_feat = self.moment_proj(fused_pooled)

        adv_user_feat = grad_reverse(user_feat, lambd=grl_lambda)
        adv_moment_feat = grad_reverse(moment_feat, lambd=grl_lambda)

        return {
            "task_logits": self.task_cls(moment_feat),
            "subject_logits": self.subject_cls(user_feat),
            "session_logits": self.session_cls(moment_feat),
            "personality_pred": self.personality(user_feat),
            "sex_logits": self.sex_cls(user_feat),
            "age_pred": self.age_reg(user_feat).squeeze(-1),
            "next_summary_pred": self.next_summary(moment_feat),
            "adv_task_logits_from_user": self.adv_task_from_user(adv_user_feat),
            "adv_session_logits_from_user": self.adv_session_from_user(adv_user_feat),
            "adv_next_summary_pred_from_user": self.adv_next_summary_from_user(adv_user_feat),
            "adv_subject_logits_from_moment": self.adv_subject_from_moment(adv_moment_feat),
            "adv_personality_pred_from_moment": self.adv_personality_from_moment(adv_moment_feat),
            "adv_sex_logits_from_moment": self.adv_sex_from_moment(adv_moment_feat),
            "adv_age_pred_from_moment": self.adv_age_from_moment(adv_moment_feat).squeeze(-1),
        }


class PretrainingLoss(nn.Module):
    """Weighted sum of all pre-training auxiliary losses.

        Loss = w_task * CE(task) + w_subj * CE(subject) + w_ses * CE(session)
            + w_per  * SmoothL1(personality) + w_sex * CE(sex) + w_age * SmoothL1(age)
            + w_next * SmoothL1(next_summary)
    """

    def __init__(
        self,
        w_task: float = 1.0,
        w_subject: float = 1.0,
        w_session: float = 0.5,
        w_personality: float = 0.5,
        w_sex: float = 0.5,
        w_age: float = 0.3,
        w_next: float = 0.5,
        w_adv_user_on_moment: float = 0.0,
        w_adv_moment_on_user: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_task        = w_task
        self.w_subject     = w_subject
        self.w_session     = w_session
        self.w_personality = w_personality
        self.w_sex         = w_sex
        self.w_age         = w_age
        self.w_next        = w_next
        self.w_adv_user_on_moment = w_adv_user_on_moment
        self.w_adv_moment_on_user = w_adv_moment_on_user

        self.ce   = nn.CrossEntropyLoss(ignore_index=-1)
        self.sl1  = nn.SmoothL1Loss()

    def _sex_loss(self, logits: torch.Tensor, sex_label: torch.Tensor) -> torch.Tensor:
        valid_sex = sex_label >= 0
        if valid_sex.any():
            return self.ce(logits[valid_sex].float(), sex_label[valid_sex])
        return torch.tensor(0.0, device=logits.device)

    def _next_loss(
        self,
        pred_next: torch.Tensor,
        next_summary: torch.Tensor,
        has_next: torch.Tensor,
    ) -> torch.Tensor:
        valid_next = has_next > 0.5
        if valid_next.any():
            return self.sl1(
                pred_next[valid_next].float(),
                next_summary[valid_next].float(),
            )
        return torch.tensor(0.0, device=pred_next.device)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        task_idx:    torch.Tensor,   # (B,)  long
        subject_idx: torch.Tensor,   # (B,)  long
        session_idx: torch.Tensor,   # (B,)  long
        personality: torch.Tensor,   # (B,5) float
        sex_label:   torch.Tensor,   # (B,)  long  (−1 = unknown → ignored by CE)
        age:         torch.Tensor,   # (B,)  float
        next_summary: torch.Tensor,  # (B, S) float
        has_next: torch.Tensor,      # (B,) float {0,1}
    ) -> dict[str, torch.Tensor]:
        l_task = self.ce(preds["task_logits"].float(),    task_idx)
        l_subj = self.ce(preds["subject_logits"].float(), subject_idx)
        l_ses  = self.ce(preds["session_logits"].float(), session_idx)
        l_per  = self.sl1(preds["personality_pred"].float(), personality.float())

        # Sex: ignore unknown (−1)
        l_sex = self._sex_loss(preds["sex_logits"], sex_label)

        l_age = self.sl1(preds["age_pred"].float(), age.float())

        l_next = self._next_loss(preds["next_summary_pred"], next_summary, has_next)

        if self.w_adv_user_on_moment > 0.0:
            l_adv_u2m_task = self.ce(preds["adv_task_logits_from_user"].float(), task_idx)
            l_adv_u2m_session = self.ce(preds["adv_session_logits_from_user"].float(), session_idx)
            l_adv_u2m_next = self._next_loss(
                preds["adv_next_summary_pred_from_user"], next_summary, has_next
            )
            l_adv_user_on_moment = (l_adv_u2m_task + l_adv_u2m_session + l_adv_u2m_next) / 3.0
        else:
            l_adv_user_on_moment = torch.tensor(0.0, device=preds["task_logits"].device)

        if self.w_adv_moment_on_user > 0.0:
            l_adv_m2u_subject = self.ce(preds["adv_subject_logits_from_moment"].float(), subject_idx)
            l_adv_m2u_personality = self.sl1(
                preds["adv_personality_pred_from_moment"].float(), personality.float()
            )
            l_adv_m2u_sex = self._sex_loss(preds["adv_sex_logits_from_moment"], sex_label)
            l_adv_m2u_age = self.sl1(preds["adv_age_pred_from_moment"].float(), age.float())
            l_adv_moment_on_user = (
                l_adv_m2u_subject + l_adv_m2u_personality + l_adv_m2u_sex + l_adv_m2u_age
            ) / 4.0
        else:
            l_adv_moment_on_user = torch.tensor(0.0, device=preds["task_logits"].device)

        total = (
            self.w_task    * l_task
            + self.w_subject * l_subj
            + self.w_session * l_ses
            + self.w_personality * l_per
            + self.w_sex   * l_sex
            + self.w_age   * l_age
            + self.w_next  * l_next
            + self.w_adv_user_on_moment * l_adv_user_on_moment
            + self.w_adv_moment_on_user * l_adv_moment_on_user
        )
        return {
            "total":       total,
            "task":        l_task,
            "subject":     l_subj,
            "session":     l_ses,
            "personality": l_per,
            "sex":         l_sex,
            "age":         l_age,
            "next":        l_next,
            "adv_user_on_moment": l_adv_user_on_moment,
            "adv_moment_on_user": l_adv_moment_on_user,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MuMTAffectLoss(nn.Module):
    """Combined personality regression + emotion classification loss.

    L = alpha * L_personality + (1 - alpha) * L_emotion
    L_emotion = mean of cross-entropy losses for V, A, D.
    L_personality = epsilon-insensitive regression (Huber) for each trait,
                    or BCEWithLogitsLoss when personality_binary=True.

    Per-dimension class weights are supported: pass a dict
    ``{"valence": Tensor, "arousal": Tensor, "dominance": Tensor}``
    or a single Tensor applied to all three heads.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        class_weights: torch.Tensor | dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.0,
        personality_binary: bool = False,
        personality_ternary: bool = False,
        personality_thresholds: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.personality_binary = personality_binary
        self.personality_ternary = personality_ternary
        if personality_ternary:
            # thresholds: (5, 2) — p33 and p67 per trait
            self.personality_loss = nn.CrossEntropyLoss()
            self.register_buffer(
                "personality_thresholds",
                personality_thresholds if personality_thresholds is not None
                else torch.zeros(5, 2),
            )
        elif personality_binary:
            self.personality_loss = nn.BCEWithLogitsLoss()
            # thresholds: (5,) median values to binarize on the fly
            self.register_buffer(
                "personality_thresholds",
                personality_thresholds if personality_thresholds is not None
                else torch.zeros(5),
            )
        else:
            self.personality_loss = nn.SmoothL1Loss()

        if isinstance(class_weights, dict):
            self.valence_loss   = nn.CrossEntropyLoss(weight=class_weights.get("valence"), label_smoothing=label_smoothing)
            self.arousal_loss   = nn.CrossEntropyLoss(weight=class_weights.get("arousal"), label_smoothing=label_smoothing)
            self.dominance_loss = nn.CrossEntropyLoss(weight=class_weights.get("dominance"), label_smoothing=label_smoothing)
        else:
            # single tensor or None applied to all three
            self.valence_loss   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            self.arousal_loss   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            self.dominance_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        emotion_labels: torch.Tensor,      # (B, 3) — [valence_bin, arousal_bin, dominance_bin]
        personality_labels: torch.Tensor,  # (B, 5)
    ) -> dict[str, torch.Tensor]:
        # Cast to float32 so CrossEntropyLoss weight tensors (float32) match AMP half-precision logits
        if self.personality_ternary:
            # Discretize continuous labels into 0/1/2 using p33/p67 thresholds
            thresholds = self.personality_thresholds  # (5, 2)
            p_vals = personality_labels.float()
            ternary_labels = torch.zeros_like(p_vals, dtype=torch.long)
            ternary_labels[p_vals > thresholds[:, 0]] = 1
            ternary_labels[p_vals > thresholds[:, 1]] = 2
            # personality_pred shape: (B, 5, 3)
            pred = outputs["personality_pred"].float()
            l_p = sum(
                self.personality_loss(pred[:, i, :], ternary_labels[:, i])
                for i in range(pred.shape[1])
            ) / pred.shape[1]
        elif self.personality_binary:
            binary_targets = (personality_labels.float() > self.personality_thresholds).float()
            l_p = self.personality_loss(outputs["personality_pred"].float(), binary_targets)
        else:
            l_p = self.personality_loss(outputs["personality_pred"].float(), personality_labels.float())

        l_v = self.valence_loss(outputs["valence_logits"].float(),   emotion_labels[:, 0])
        l_a = self.arousal_loss(outputs["arousal_logits"].float(),   emotion_labels[:, 1])
        l_d = self.dominance_loss(outputs["dominance_logits"].float(), emotion_labels[:, 2])
        l_e = (l_v + l_a + l_d) / 3.0

        total = self.alpha * l_p + (1.0 - self.alpha) * l_e
        return {
            "total": total,
            "personality": l_p,
            "emotion": l_e,
            "valence": l_v,
            "arousal": l_a,
            "dominance": l_d,
        }
