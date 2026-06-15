"""Ordinal VAD models for GroupAffect-4.

This module is intentionally separate from model_simple.py so the original
3-class binned experiments remain unchanged.  The heads predict cumulative
ordinal logits for the original 1-9 SAM labels:

    logit_k ~= logit P(y > k),  k = 1..8

At inference, the expected SAM score is 1 + sum_k sigmoid(logit_k).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_DIMS = {
    "gaze": 9,
    "pupil": 3,
    "eda": 5,
    "ppg": 3,
    "imu": 6,
}


def _inverse_softplus(x: float) -> float:
    return math.log(math.exp(x) - 1.0)


class FusionTrunk(nn.Module):
    """Small MLP trunk with BatchNorm and Dropout."""

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
            layers.extend(
                [
                    nn.Linear(cur, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            cur = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrdinalHead(nn.Module):
    """Cumulative ordinal head with ordered cut points.

    A small network maps the fused feature vector to one latent score.  Eight
    ordered cut points transform that score into logits for P(y > k), k=1..8.
    The ordered parameterisation keeps the cumulative probabilities monotonic.
    """

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.first_cut = nn.Parameter(torch.tensor(-2.0))
        self.raw_increments = nn.Parameter(
            torch.full((7,), _inverse_softplus(0.55), dtype=torch.float32)
        )

    def cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.raw_increments)
        return torch.cat(
            [self.first_cut.view(1), self.first_cut + torch.cumsum(increments, dim=0)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x)
        cuts = self.cutpoints().to(score.device)
        return score - cuts.view(1, -1)


class _OrdinalMixin:
    bfi_mode: str

    def _init_bfi(self, hidden: int, bfi_dim: int, task_dim: int) -> None:
        if self.bfi_mode == "perdim":
            self.bfi_v = nn.Linear(bfi_dim, hidden)
            self.bfi_a = nn.Linear(bfi_dim, hidden)
            self.bfi_d = nn.Linear(bfi_dim, hidden)
            for layer in (self.bfi_v, self.bfi_a, self.bfi_d):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif self.bfi_mode == "taskadapt":
            self.bfi_v = nn.Linear(bfi_dim, hidden)
            self.bfi_a = nn.Linear(bfi_dim, hidden)
            self.bfi_d = nn.Linear(bfi_dim, hidden)
            self.task_gate_v = nn.Linear(task_dim, hidden)
            self.task_gate_a = nn.Linear(task_dim, hidden)
            self.task_gate_d = nn.Linear(task_dim, hidden)
            for layer in (
                self.bfi_v,
                self.bfi_a,
                self.bfi_d,
                self.task_gate_v,
                self.task_gate_a,
                self.task_gate_d,
            ):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _feature_triplet(
        self,
        feat: torch.Tensor,
        personality: torch.Tensor | None,
        task_onehot: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bfi_mode == "perdim" and personality is not None:
            bfi = personality.float()
            return feat + self.bfi_v(bfi), feat + self.bfi_a(bfi), feat + self.bfi_d(bfi)

        if self.bfi_mode == "taskadapt" and personality is not None:
            bfi = personality.float()
            if task_onehot is None:
                task_onehot = torch.zeros(bfi.size(0), 5, device=bfi.device)
            task = task_onehot.float()
            return (
                feat + torch.sigmoid(self.task_gate_v(task)) * self.bfi_v(bfi),
                feat + torch.sigmoid(self.task_gate_a(task)) * self.bfi_a(bfi),
                feat + torch.sigmoid(self.task_gate_d(task)) * self.bfi_d(bfi),
            )

        return feat, feat, feat

    def _outputs(
        self,
        feat: torch.Tensor,
        personality: torch.Tensor | None,
        task_onehot: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        feat_v, feat_a, feat_d = self._feature_triplet(feat, personality, task_onehot)
        return {
            "valence_logits": self.valence_head(feat_v),
            "arousal_logits": self.arousal_head(feat_a),
            "dominance_logits": self.dominance_head(feat_d),
            "personality_pred": self.personality_head(feat),
        }


class OrdinalMLPNet(nn.Module, _OrdinalMixin):
    """Summary-only ordinal model."""

    def __init__(
        self,
        summary_dim: int = 49,
        hidden: int = 64,
        n_personality: int = 5,
        dropout: float = 0.5,
        bfi_mode: str = "none",
        bfi_dim: int = 5,
        task_dim: int = 5,
    ) -> None:
        super().__init__()
        if bfi_mode not in {"none", "concat", "perdim", "taskadapt"}:
            raise ValueError(f"Unsupported bfi_mode for ordinal model: {bfi_mode}")
        self.bfi_mode = bfi_mode
        trunk_in = summary_dim + bfi_dim if bfi_mode == "concat" else summary_dim
        self.trunk = FusionTrunk(trunk_in, hidden=hidden, n_layers=2, dropout=dropout)
        self._init_bfi(hidden=hidden, bfi_dim=bfi_dim, task_dim=task_dim)
        self.valence_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
        self.arousal_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
        self.dominance_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
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
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if self.bfi_mode == "concat" and personality is not None:
            trunk_in = torch.cat([summary, personality.float()], dim=-1)
        else:
            trunk_in = summary
        feat = self.trunk(trunk_in)
        return self._outputs(feat, personality, task_onehot)


class OrdinalPoolNet(nn.Module, _OrdinalMixin):
    """Mean/std pooled sequence features plus summary features.

    Set use_eda=False to exclude the EDA/GSR pathway entirely (e.g. when GSR is
    used as a label source rather than an input modality).  The EDA projection
    layer is omitted and fusion_dim is reduced accordingly.
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
        bfi_mode: str = "none",
        bfi_dim: int = 5,
        task_dim: int = 5,
        use_eda: bool = True,
    ) -> None:
        super().__init__()
        if bfi_mode not in {"none", "perdim", "taskadapt"}:
            raise ValueError("OrdinalPoolNet supports bfi_mode none/perdim/taskadapt")
        self.bfi_mode = bfi_mode
        self.use_eda = use_eda
        self.gaze_proj = nn.Linear(gaze_dim * 2, proj_dim)
        self.pupil_proj = nn.Linear(pupil_dim * 2, proj_dim)
        self.eda_proj = nn.Linear(eda_dim * 2, proj_dim) if use_eda else None
        self.ppg_proj = nn.Linear(ppg_dim * 2, proj_dim)
        self.imu_proj = nn.Linear(imu_dim * 2, proj_dim)
        n_seq_modalities = 5 if use_eda else 4
        fusion_dim = n_seq_modalities * proj_dim + summary_dim
        self.trunk = FusionTrunk(fusion_dim, hidden=hidden, n_layers=2, dropout=dropout)
        self._init_bfi(hidden=hidden, bfi_dim=bfi_dim, task_dim=task_dim)
        self.valence_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
        self.arousal_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
        self.dominance_head = OrdinalHead(hidden, hidden=32, dropout=dropout)
        self.personality_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_personality)
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False).clamp(min=1e-6)
        return torch.cat([mu, std], dim=-1)

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
        **_: object,
    ) -> dict[str, torch.Tensor]:
        parts = [
            F.relu(self.gaze_proj(self._pool(gaze_seq))),
            F.relu(self.pupil_proj(self._pool(pupil_seq))),
        ]
        if self.use_eda:
            parts.append(F.relu(self.eda_proj(self._pool(eda_seq))))
        parts += [
            F.relu(self.ppg_proj(self._pool(ppg_seq))),
            F.relu(self.imu_proj(self._pool(imu_seq))),
            summary,
        ]
        pooled = torch.cat(parts, dim=-1)
        feat = self.trunk(pooled)
        return self._outputs(feat, personality, task_onehot)


def build_ordinal_model(
    arch: str,
    summary_dim: int,
    hidden: int = 64,
    dropout: float = 0.5,
    n_personality: int = 5,
    bfi_mode: str = "none",
    bfi_dim: int = 5,
    task_dim: int = 5,
    use_eda: bool = True,
) -> nn.Module:
    arch = arch.lower()
    if arch == "mlp":
        return OrdinalMLPNet(
            summary_dim=summary_dim,
            hidden=hidden,
            n_personality=n_personality,
            dropout=dropout,
            bfi_mode=bfi_mode,
            bfi_dim=bfi_dim,
            task_dim=task_dim,
        )
    if arch == "pool":
        return OrdinalPoolNet(
            summary_dim=summary_dim,
            proj_dim=max(8, hidden // 4),
            hidden=hidden,
            n_personality=n_personality,
            dropout=dropout,
            bfi_mode=bfi_mode,
            bfi_dim=bfi_dim,
            task_dim=task_dim,
            use_eda=use_eda,
        )
    raise ValueError(f"Unknown ordinal arch: {arch!r}. Choose from 'mlp' or 'pool'.")
