"""model_simple.py

Lightweight VAD prediction models designed for N~220 labeled windows.

The core insight: for 200-300 training samples, learned temporal modeling
(Transformers, GRUs) will memorise group-specific physiology artifacts rather
than learning generalizable affect representations.  The 49 pre-computed
summary features already capture per-window statistics; the key missing piece
is light inter-modality fusion with strong regularization.

Three models offered in increasing order of complexity:

  MLPNet   – summary features only (49-dim → MLP → 3 VAD heads).
              ~6K parameters.  The absolute simplest baseline.

  PoolNet  – per-modality mean+std pooling of raw sequences, concatenated
              with summary features, then MLP.
              ~20K parameters.  Uses sequences without any learned temporal
              model (zero overfitting risk from temporal patterns).

  ConvNet  – per-modality lightweight 1-D CNN (kernel=7, stride=4 → global
              average pool), concatenated with summary features, then MLP.
              ~80K parameters.  Captures short-range temporal patterns
              (~0.2 s at 25 Hz) without the full attention complexity.

All three:
  - share the same MLP head (VAD 3-class classification)
  - support personality regression as an auxiliary task
  - use BatchNorm + Dropout throughout
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_DIMS = {
    "gaze":  9,
    "pupil": 3,
    "eda":   5,
    "ppg":   3,
    "imu":   6,
}

# ── Shared VAD head ───────────────────────────────────────────────────────────

class VADHead(nn.Module):
    """Three-class classification head for one VAD dimension."""

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Shared fusion trunk ───────────────────────────────────────────────────────

class FusionTrunk(nn.Module):
    """d_in → [BN → Linear(d_h) → BN → ReLU → Dropout] × n_layers → d_h."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(in_dim)]
        cur = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(cur, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            cur = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: MLPNet — summary features only
# ─────────────────────────────────────────────────────────────────────────────

class MLPNet(nn.Module):
    """Simplest baseline: 49 pre-computed summary features → MLP → V/A/D.

    Supports three personality-modulation modes (paper: Meng & Li 2026,
    "Moderating Roles of the Big Five in Valence-Arousal Dynamics"):

      bfi_mode="none"    — baseline; BFI scores ignored entirely.
      bfi_mode="concat"  — append normalised BFI to summary features before trunk.
                           BFI is a direct feature; model learns additive effects.
      bfi_mode="gate"    — BFI produces a sigmoid gate over summary features.
                           Implements the moderation: personality changes the
                           *sensitivity* of each physiology feature to VAD.
                           gate = sigmoid(W_gate(bfi))  ∈ (0,1)^summary_dim
                           trunk_input = summary * gate
      bfi_mode="perdim"    — BFI produces a separate bias vector for each VAD head.
                             Directly models the CTSEM cross-lagged moderation:
                             V/A/D heads attend to different BFI trait combinations.
                             feat_v = feat + W_bfi_v(bfi)
                             feat_a = feat + W_bfi_a(bfi)
                             feat_d = feat + W_bfi_d(bfi)

      bfi_mode="taskadapt" — Task-adaptive perdim: the BFI projection per VAD head
                             is gated by the current task context.
                             gate_v = sigmoid(W_tg_v(task_onehot))  ∈ (0,1)^hidden
                             feat_v = feat + gate_v ⊙ W_bfi_v(bfi)
                             Addresses the task-sensitivity finding: lets the model
                             learn *which tasks* BFI conditioning is useful for.
                             W_bfi_{v,a,d} zero-initialised (starts as perdim=0).
                             W_tg_{v,a,d} uniform-initialised so initial gate ≈ 0.5.

    Parameters: ~17K (none/concat) / ~17K+sum_dim*5 (gate) /
                ~17K+3*hidden*5 (perdim) / ~17K+6*hidden*5+3*hidden*5 (taskadapt)
    """

    def __init__(
        self,
        summary_dim: int = 49,
        hidden: int = 64,
        n_personality: int = 5,
        dropout: float = 0.5,
        bfi_mode: str = "none",
        bfi_dim: int = 5,
        task_dim: int = 5,   # number of tasks (T0-T4)
    ) -> None:
        super().__init__()
        self.bfi_mode = bfi_mode

        if bfi_mode == "concat":
            trunk_in = summary_dim + bfi_dim
        else:
            trunk_in = summary_dim

        self.trunk = FusionTrunk(trunk_in, hidden=hidden, n_layers=2, dropout=dropout)

        # Modulation modules
        if bfi_mode == "gate":
            # BFI → gate over each summary feature  (true moderation)
            self.bfi_gate = nn.Linear(bfi_dim, summary_dim)
            nn.init.zeros_(self.bfi_gate.weight)   # start as identity gate (sigmoid(0)=0.5)
            nn.init.zeros_(self.bfi_gate.bias)

        elif bfi_mode == "perdim":
            # Separate BFI projection per VAD dimension
            # Paper: N,E most relevant to V; A,O to A; all five to D
            self.bfi_v = nn.Linear(bfi_dim, hidden, bias=True)
            self.bfi_a = nn.Linear(bfi_dim, hidden, bias=True)
            self.bfi_d = nn.Linear(bfi_dim, hidden, bias=True)
            for lin in (self.bfi_v, self.bfi_a, self.bfi_d):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        elif bfi_mode == "taskadapt":
            # Per-dim BFI projections (zero-init — start as no-BFI baseline)
            self.bfi_v = nn.Linear(bfi_dim, hidden, bias=True)
            self.bfi_a = nn.Linear(bfi_dim, hidden, bias=True)
            self.bfi_d = nn.Linear(bfi_dim, hidden, bias=True)
            for lin in (self.bfi_v, self.bfi_a, self.bfi_d):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            # Task-context gates: task one-hot → sigmoid gate ∈ (0,1)^hidden
            # Uniform init so initial gate ≈ sigmoid(0) = 0.5
            self.task_gate_v = nn.Linear(task_dim, hidden, bias=True)
            self.task_gate_a = nn.Linear(task_dim, hidden, bias=True)
            self.task_gate_d = nn.Linear(task_dim, hidden, bias=True)
            for lin in (self.task_gate_v, self.task_gate_a, self.task_gate_d):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        self.valence_head   = VADHead(hidden, hidden=32, dropout=dropout)
        self.arousal_head   = VADHead(hidden, hidden=32, dropout=dropout)
        self.dominance_head = VADHead(hidden, hidden=32, dropout=dropout)
        self.personality_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_personality)
        )

    def forward(
        self,
        gaze_seq: torch.Tensor,
        pupil_seq: torch.Tensor,
        eda_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        imu_seq: torch.Tensor,
        summary: torch.Tensor,
        personality: torch.Tensor | None = None,
        task_onehot: torch.Tensor | None = None,
        **_,
    ) -> dict[str, torch.Tensor]:

        bfi = personality  # (B, 5)  — normalised Big Five scores

        if self.bfi_mode == "concat" and bfi is not None:
            trunk_in = torch.cat([summary, bfi.float()], dim=-1)
        elif self.bfi_mode == "gate" and bfi is not None:
            gate = torch.sigmoid(self.bfi_gate(bfi.float()))  # (B, summary_dim)
            trunk_in = summary * gate
        else:
            trunk_in = summary

        feat = self.trunk(trunk_in)   # (B, hidden)

        if self.bfi_mode == "perdim" and bfi is not None:
            b = bfi.float()
            feat_v = feat + self.bfi_v(b)
            feat_a = feat + self.bfi_a(b)
            feat_d = feat + self.bfi_d(b)

        elif self.bfi_mode == "taskadapt" and bfi is not None:
            b = bfi.float()
            # Task context gate: if task_onehot not provided, gate defaults to 0.5
            if task_onehot is not None:
                t = task_onehot.float()
            else:
                t = torch.zeros(b.size(0), 5, device=b.device)
            gate_v = torch.sigmoid(self.task_gate_v(t))  # (B, hidden)
            gate_a = torch.sigmoid(self.task_gate_a(t))
            gate_d = torch.sigmoid(self.task_gate_d(t))
            feat_v = feat + gate_v * self.bfi_v(b)
            feat_a = feat + gate_a * self.bfi_a(b)
            feat_d = feat + gate_d * self.bfi_d(b)

        else:
            feat_v = feat_a = feat_d = feat

        return {
            "valence_logits":   self.valence_head(feat_v),
            "arousal_logits":   self.arousal_head(feat_a),
            "dominance_logits": self.dominance_head(feat_d),
            "personality_pred": self.personality_head(feat),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: PoolNet — mean+std pooling of raw sequences + summary
# ─────────────────────────────────────────────────────────────────────────────

class PoolNet(nn.Module):
    """Per-modality global mean+std pooling → project → concat with summary → MLP.

    No learned temporal model: pooling is the only temporal operation.
    Captures per-window physiological level and variability.

    Parameters: ~20 K.
    """

    def __init__(
        self,
        gaze_dim: int = 9,
        pupil_dim: int = 3,
        eda_dim: int = 5,
        ppg_dim: int = 3,
        imu_dim: int = 6,
        summary_dim: int = 49,
        proj_dim: int = 16,
        hidden: int = 64,
        n_personality: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Each modality: mean + std → 2*F → linear(proj_dim)
        self.gaze_proj  = nn.Linear(gaze_dim  * 2, proj_dim)
        self.pupil_proj = nn.Linear(pupil_dim * 2, proj_dim)
        self.eda_proj   = nn.Linear(eda_dim   * 2, proj_dim)
        self.ppg_proj   = nn.Linear(ppg_dim   * 2, proj_dim)
        self.imu_proj   = nn.Linear(imu_dim   * 2, proj_dim)

        # 5 × proj_dim + summary_dim
        fusion_dim = 5 * proj_dim + summary_dim
        self.trunk = FusionTrunk(fusion_dim, hidden=hidden, n_layers=2, dropout=dropout)

        self.valence_head   = VADHead(hidden, hidden=32, dropout=dropout)
        self.arousal_head   = VADHead(hidden, hidden=32, dropout=dropout)
        self.dominance_head = VADHead(hidden, hidden=32, dropout=dropout)
        self.personality_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_personality)
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, 2*F) via temporal mean + std."""
        mu  = x.mean(dim=1)                 # (B, F)
        std = x.std(dim=1).clamp(min=1e-6)  # (B, F)
        return torch.cat([mu, std], dim=-1)  # (B, 2F)

    def forward(
        self,
        gaze_seq: torch.Tensor,
        pupil_seq: torch.Tensor,
        eda_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        imu_seq: torch.Tensor,
        summary: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        g = F.relu(self.gaze_proj(self._pool(gaze_seq)))
        p = F.relu(self.pupil_proj(self._pool(pupil_seq)))
        e = F.relu(self.eda_proj(self._pool(eda_seq)))
        q = F.relu(self.ppg_proj(self._pool(ppg_seq)))
        m = F.relu(self.imu_proj(self._pool(imu_seq)))

        feat = self.trunk(torch.cat([g, p, e, q, m, summary], dim=-1))
        return {
            "valence_logits":   self.valence_head(feat),
            "arousal_logits":   self.arousal_head(feat),
            "dominance_logits": self.dominance_head(feat),
            "personality_pred": self.personality_head(feat),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: ConvNet — lightweight 1-D CNN per modality + summary
# ─────────────────────────────────────────────────────────────────────────────

class _ModalityCNN(nn.Module):
    """Very light 1-D CNN for one physiological modality.

    Three conv blocks (k=7, stride=4) halve the sequence three times:
      T=400 → 100 → 25 → 6 (then global-avg-pool → 1)
    Each block doubles channels: in_ch → 16 → 32 → out_ch.

    Global average pooling at the end produces a fixed-size vector
    regardless of original T, making the output completely invariant
    to temporal alignment.
    """

    def __init__(self, in_features: int, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: T=400 → 100
            nn.Conv1d(in_features, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Block 2: T=100 → 25
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Block 3: T=25 → 7
            nn.Conv1d(32, out_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, out_dim)."""
        x = x.permute(0, 2, 1)     # (B, F, T) for Conv1d
        x = self.net(x)             # (B, out_dim, T')
        return x.mean(dim=-1)       # (B, out_dim) — global avg pool


class ConvNet(nn.Module):
    """Per-modality 1-D CNN (shallow, global-pooled) + summary features → MLP.

    Captures short-range temporal patterns (~0.2 s) without the full
    attention complexity of Transformers.

    Parameters: ~80 K.
    """

    def __init__(
        self,
        gaze_dim: int = 9,
        pupil_dim: int = 3,
        eda_dim: int = 5,
        ppg_dim: int = 3,
        imu_dim: int = 6,
        summary_dim: int = 49,
        cnn_out: int = 32,
        hidden: int = 128,
        n_personality: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.gaze_cnn  = _ModalityCNN(gaze_dim,  out_dim=cnn_out)
        self.pupil_cnn = _ModalityCNN(pupil_dim, out_dim=cnn_out)
        self.eda_cnn   = _ModalityCNN(eda_dim,   out_dim=cnn_out)
        self.ppg_cnn   = _ModalityCNN(ppg_dim,   out_dim=cnn_out)
        self.imu_cnn   = _ModalityCNN(imu_dim,   out_dim=cnn_out)

        fusion_dim = 5 * cnn_out + summary_dim
        self.trunk = FusionTrunk(fusion_dim, hidden=hidden, n_layers=2, dropout=dropout)

        self.valence_head   = VADHead(hidden, hidden=64, dropout=dropout)
        self.arousal_head   = VADHead(hidden, hidden=64, dropout=dropout)
        self.dominance_head = VADHead(hidden, hidden=64, dropout=dropout)
        self.personality_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_personality)
        )

    def forward(
        self,
        gaze_seq: torch.Tensor,
        pupil_seq: torch.Tensor,
        eda_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        imu_seq: torch.Tensor,
        summary: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        g = self.gaze_cnn(gaze_seq)
        p = self.pupil_cnn(pupil_seq)
        e = self.eda_cnn(eda_seq)
        q = self.ppg_cnn(ppg_seq)
        m = self.imu_cnn(imu_seq)

        feat = self.trunk(torch.cat([g, p, e, q, m, summary], dim=-1))
        return {
            "valence_logits":   self.valence_head(feat),
            "arousal_logits":   self.arousal_head(feat),
            "dominance_logits": self.dominance_head(feat),
            "personality_pred": self.personality_head(feat),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_simple_model(
    arch: str,
    summary_dim: int,
    gaze_dim: int = 9,
    pupil_dim: int = 3,
    eda_dim: int = 5,
    ppg_dim: int = 3,
    imu_dim: int = 6,
    hidden: int = 64,
    dropout: float = 0.5,
    n_personality: int = 5,
    bfi_mode: str = "none",
    bfi_dim: int = 5,
    task_dim: int = 5,
) -> nn.Module:
    """Return a model by name: 'mlp', 'pool', or 'conv'.

    bfi_mode: one of 'none', 'concat', 'gate', 'perdim', 'taskadapt'.
    See MLPNet docstring for full description of each mode.
    """
    arch = arch.lower()
    if arch == "mlp":
        return MLPNet(
            summary_dim=summary_dim, hidden=hidden,
            n_personality=n_personality, dropout=dropout,
            bfi_mode=bfi_mode, bfi_dim=bfi_dim, task_dim=task_dim,
        )
    if arch == "pool":
        return PoolNet(
            gaze_dim=gaze_dim, pupil_dim=pupil_dim, eda_dim=eda_dim,
            ppg_dim=ppg_dim, imu_dim=imu_dim, summary_dim=summary_dim,
            proj_dim=max(8, hidden // 4), hidden=hidden,
            n_personality=n_personality, dropout=dropout,
        )
    if arch == "conv":
        return ConvNet(
            gaze_dim=gaze_dim, pupil_dim=pupil_dim, eda_dim=eda_dim,
            ppg_dim=ppg_dim, imu_dim=imu_dim, summary_dim=summary_dim,
            cnn_out=32, hidden=hidden,
            n_personality=n_personality, dropout=dropout,
        )
    raise ValueError(f"Unknown arch: {arch!r}. Choose from 'mlp', 'pool', 'conv'.")
