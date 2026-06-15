"""model_v2_affectai.py

Dual-Stream MuMTAffect for GroupAffect-4.

Architecture redesign (v2) based on the following principles:
  1. Modality-specific transformers encode each modality into a shared space
  2. Cross-modal fusion transformer unites modalities (temporal + spatial)
  3. Two parallel streams with asymmetric cross-attention:
     - User Profile Stream (self-attention only) → stable user traits
     - Cognitive State Stream (self-attention + cross-attention from profile) → momentary states
  4. The state stream receives information from the profile stream but NOT vice versa

Architecture:
  Gaze encoder  → T=16 latent ┐
  Pupil encoder → T=16 latent │
  EDA encoder   → T=16 latent ├→ Cross-modal Fusion Transformer (80 tokens)
  PPG encoder   → T=16 latent │           ↓
  IMU encoder   → T=16 latent ┘    ┌──────┴──────┐
                                    ↓              ↓
                          User Profile Stream   Cognitive State Stream
                          (self-attn only)      (self-attn + cross-attn from profile)
                                    ↓              ↓
                          ┌─ Subject cls      ┌─ Task cls
                          ├─ Personality reg   ├─ Session cls
                          ├─ Sex cls           ├─ Next-summary pred
                          └─ Age reg           ├─ Masked reconstruction
                                               └─ Temporal delta pred
  Downstream:
      Profile → personality head
      State (conditioned on profile) → VAD emotion heads
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_affectai import (
    EmotionCNNBranch,
    EmotionHead,
    FusionTransformer,
    ModalityEncoder,
    MuMTAffectLoss,
    PersonalityCNNBranch,
    PositionalEncoding,
    TrialFeatureMLP,
)


# ---------------------------------------------------------------------------
# User Profile Stream — self-attention only (no external conditioning)
# ---------------------------------------------------------------------------

class UserProfileStream(nn.Module):
    """Transformer stream that extracts stable user-level representations.

    Uses self-attention only. The output is NOT conditioned on momentary state.
    Produces both a sequence output (for cross-attention into state stream)
    and a pooled vector (for user-level prediction heads).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learned query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, d_model) → (sequence: (B, S, d_model), pooled: (B, d_model))."""
        seq = self.transformer(x)  # (B, S, d_model)

        # Learned-query pooling
        B = seq.size(0)
        q = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(q, seq, seq)  # (B, 1, d_model)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, d_model)

        return seq, pooled


# ---------------------------------------------------------------------------
# Cognitive State Stream — self-attention + cross-attention from profile
# ---------------------------------------------------------------------------

class CognitiveStateStreamLayer(nn.Module):
    """Single layer: self-attention → cross-attention(profile) → FFN."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (Q=state, K/V=profile)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, profile_seq: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, S, d) state tokens, profile_seq: (B, S', d) from profile stream."""
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]

        # Pre-norm cross-attention (Q=state, K/V=profile)
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, profile_seq, profile_seq)[0]

        # Pre-norm FFN
        x_norm = self.norm3(x)
        x = x + self.ffn(x_norm)

        return x


class CognitiveStateStream(nn.Module):
    """Transformer stream for momentary cognitive state.

    Uses self-attention AND cross-attention from the user profile stream.
    This allows momentary state to be conditioned on who the person is.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CognitiveStateStreamLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        # Learned query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, profile_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, d), profile_seq: (B, S', d) → (seq: (B, S, d), pooled: (B, d))."""
        for layer in self.layers:
            x = layer(x, profile_seq)

        # Learned-query pooling
        B = x.size(0)
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x)
        pooled = self.pool_norm(pooled.squeeze(1))

        return x, pooled


# ---------------------------------------------------------------------------
# Main model (v2 — Dual Stream)
# ---------------------------------------------------------------------------

class MuMTAffectV2(nn.Module):
    """Dual-stream MuMTAffect with asymmetric cross-attention.

    User Profile Stream: extracts stable user traits (self-attention only).
    Cognitive State Stream: momentary state conditioned on profile
                           (self-attention + cross-attention from profile).

    v12 addition: direct emotion attention query on fused sequence as a
    residual bypass path, ensuring emotion heads always access rich temporal
    signal regardless of state stream quality.
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
        stream_n_layers: int = 2,
        stream_nhead: int = 4,
        stream_ffn_dim: int = 512,
        stream_dropout: float = 0.2,
        use_direct_emotion_attn: bool = False,
    ) -> None:
        super().__init__()

        self.n_modalities = 5
        self.d_model_fuse = d_model_fuse
        self.use_direct_emotion_attn = use_direct_emotion_attn

        # --- Modality encoders ---
        self.gaze_encoder = ModalityEncoder(gaze_dim, d_model=d_model_enc, t_out=t_out)
        self.pupil_encoder = ModalityEncoder(pupil_dim, d_model=d_model_enc, t_out=t_out)
        self.eda_encoder = ModalityEncoder(eda_dim, d_model=d_model_enc, t_out=t_out)
        self.ppg_encoder = ModalityEncoder(ppg_dim, d_model=d_model_enc, t_out=t_out)
        self.imu_encoder = ModalityEncoder(imu_dim, d_model=d_model_enc, t_out=t_out)

        # --- Cross-modal fusion ---
        self.fusion = FusionTransformer(
            n_modalities=self.n_modalities,
            d_in=d_model_enc,
            d_model=d_model_fuse,
            nhead=4,
            ffn_dim=2048,
            dropout=0.25,
            t_out=t_out,
        )

        # --- Dual streams ---
        self.profile_stream = UserProfileStream(
            d_model=d_model_fuse, nhead=stream_nhead,
            n_layers=stream_n_layers, ffn_dim=stream_ffn_dim,
            dropout=stream_dropout,
        )
        self.state_stream = CognitiveStateStream(
            d_model=d_model_fuse, nhead=stream_nhead,
            n_layers=stream_n_layers, ffn_dim=stream_ffn_dim,
            dropout=stream_dropout,
        )

        # --- Direct emotion attention (v12 bypass) ---
        # Learned query that attends directly to the fused sequence,
        # bypassing the state stream bottleneck (like v10's TaskAttentionTemporal)
        if use_direct_emotion_attn:
            self.emotion_attn_query = nn.Parameter(
                torch.randn(1, d_model_fuse) * 0.02
            )
            self.emotion_attn_scale = d_model_fuse ** -0.5
            # Gate to blend direct attention with state stream
            self.emotion_gate = nn.Sequential(
                nn.Linear(d_model_fuse * 2, 1),
                nn.Sigmoid(),
            )

        # --- CNN branches (downstream) ---
        self.personality_cnn = PersonalityCNNBranch(d_model=d_model_fuse, dropout=0.4)
        self.emotion_cnn = EmotionCNNBranch(d_model=d_model_fuse, dropout=0.25)

        # Infer CNN output dim
        self.personality_cnn.eval()
        self.emotion_cnn.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, d_model_fuse)
            p_out_dim = self.personality_cnn(dummy).shape[-1]
            e_out_dim = self.emotion_cnn(dummy).shape[-1]
        self.personality_cnn.train()
        self.emotion_cnn.train()

        # --- Trial-feature MLP ---
        self.trial_mlp = TrialFeatureMLP(summary_dim, out_dim=64, dropout=0.3)

        # --- Personality head ---
        p_head_in = p_out_dim + 64
        self.personality_head = nn.Sequential(
            nn.Linear(p_head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_personality),
        )

        # --- Emotion heads ---
        e_head_in = e_out_dim
        self.valence_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)
        self.arousal_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)
        self.dominance_head = EmotionHead(e_head_in, n_classes=n_emotion_classes)

        # --- Subject embedding ---
        self.subject_embed = nn.Embedding(n_subjects + 2, 16)

        # --- Task context projection ---
        self.task_proj = nn.Sequential(
            nn.Linear(n_tasks, d_model_fuse),
            nn.ReLU(),
        )

    def _encode_and_fuse(
        self,
        gaze_seq: torch.Tensor,
        pupil_seq: torch.Tensor,
        eda_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        imu_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Encode modalities and fuse → (B, 5*t_out, d_fuse)."""
        gaze_lat = self.gaze_encoder(gaze_seq)
        pupil_lat = self.pupil_encoder(pupil_seq)
        eda_lat = self.eda_encoder(eda_seq)
        ppg_lat = self.ppg_encoder(ppg_seq)
        imu_lat = self.imu_encoder(imu_seq)
        return self.fusion(gaze_lat, pupil_lat, eda_lat, ppg_lat, imu_lat)

    def forward(
        self,
        gaze_seq: torch.Tensor,
        pupil_seq: torch.Tensor,
        eda_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        imu_seq: torch.Tensor,
        summary: torch.Tensor,
        user_ids: torch.Tensor,
        personality_gt: torch.Tensor | None = None,
        task_onehot: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass — returns predictions and stream embeddings."""
        # Encode + fuse
        fused = self._encode_and_fuse(
            gaze_seq.float(), pupil_seq.float(), eda_seq.float(),
            ppg_seq.float(), imu_seq.float(),
        )  # (B, 80, d_fuse)

        # --- Dual streams (asymmetric) ---
        # Profile stream: self-attention only
        profile_seq, profile_pooled = self.profile_stream(fused)
        # State stream: self-attention + cross-attention from profile
        state_seq, state_pooled = self.state_stream(fused, profile_seq)

        # Subject embedding
        user_ids_clamp = user_ids.clamp(min=0, max=self.subject_embed.num_embeddings - 1)
        subj_emb = self.subject_embed(user_ids_clamp)

        # Trial features
        trial_feat = self.trial_mlp(summary.float())

        # --- Personality branch (from profile stream) ---
        p_feat = self.personality_cnn(profile_pooled)
        p_in = torch.cat([p_feat, trial_feat], dim=-1)
        personality_pred = self.personality_head(p_in)

        # --- Emotion branch ---
        if self.use_direct_emotion_attn:
            # v12: direct attention query on fused sequence (bypasses state bottleneck)
            # Scaled dot-product attention: Q=learned query, K/V=fused
            scores = torch.einsum("qd,bsd->bs", self.emotion_attn_query, fused)
            scores = scores * self.emotion_attn_scale
            weights = F.softmax(scores, dim=-1)  # (B, S)
            direct_ctx = torch.einsum("bs,bsd->bd", weights, fused)  # (B, d_fuse)

            # Learned gate blends direct attention with state stream
            gate_in = torch.cat([direct_ctx, state_pooled], dim=-1)  # (B, 2*d)
            gate = self.emotion_gate(gate_in)  # (B, 1)
            emotion_input = gate * direct_ctx + (1 - gate) * state_pooled
        else:
            # v11 original: emotion only from state_pooled
            emotion_input = state_pooled

        e_feat = self.emotion_cnn(emotion_input)

        # Task context injection
        if task_onehot is not None:
            task_emb = self.task_proj(task_onehot.float())
            if task_emb.shape[-1] == e_feat.shape[-1]:
                e_feat = e_feat + task_emb

        # Profile conditions emotion predictions
        p_embed = profile_pooled

        valence_logits = self.valence_head(e_feat, p_embed)
        arousal_logits = self.arousal_head(e_feat, p_embed)
        dominance_logits = self.dominance_head(e_feat, p_embed)

        return {
            "valence_logits": valence_logits,
            "arousal_logits": arousal_logits,
            "dominance_logits": dominance_logits,
            "personality_pred": personality_pred,
            "personality_embed": profile_pooled,
            # Stream outputs for pretraining
            "profile_pooled": profile_pooled,
            "state_pooled": state_pooled,
            "profile_seq": profile_seq,
            "state_seq": state_seq,
            "fused_pooled": (profile_pooled + state_pooled) / 2,  # backward compat
        }


# ---------------------------------------------------------------------------
# Dual-Stream Pretraining Heads  (Phase 0)
# ---------------------------------------------------------------------------

class DualStreamPretrainingHeads(nn.Module):
    """Pretraining heads for the dual-stream architecture.

    User Profile Stream objectives (stable, trait-level):
      - Subject identity classification
      - Personality regression (BFI-44)
      - Sex classification
      - Age regression

    Cognitive State Stream objectives (momentary, state-level):
      - Task classification (T0–T4)
      - Session classification
      - Next-window summary prediction
      - Temporal delta prediction (summary_t+1 - summary_t)
      - Masked modality reconstruction
    """

    def __init__(
        self,
        d_fuse: int = 128,
        n_tasks: int = 5,
        n_subjects: int = 40,
        n_sessions: int = 10,
        n_personality: int = 5,
        summary_dim: int = 40,
        n_modalities: int = 5,
        d_recon: int = 64,
    ) -> None:
        super().__init__()
        self.d_fuse = d_fuse

        # === User Profile Stream heads ===
        self.subject_cls = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_fuse, n_subjects),
        )
        self.personality = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_fuse // 2, n_personality),
        )
        self.sex_cls = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(),
            nn.Linear(d_fuse // 2, 2),
        )
        self.age_reg = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 2), nn.ReLU(),
            nn.Linear(d_fuse // 2, 1),
        )

        # === Cognitive State Stream heads ===
        self.task_cls = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_fuse, n_tasks),
        )
        self.session_cls = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_fuse, n_sessions),
        )
        self.next_summary = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_fuse, summary_dim),
        )
        # Delta prediction: predict change between t and t+1
        self.delta_summary = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_fuse, summary_dim),
        )
        # Masked modality reconstruction from state tokens
        # Reconstructs per-modality segment means (d_recon per modality)
        self.recon_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_fuse, n_modalities * d_recon),
        )
        self.n_modalities = n_modalities
        self.d_recon = d_recon

    def forward(
        self,
        profile_pooled: torch.Tensor,
        state_pooled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """profile_pooled: (B, d), state_pooled: (B, d)."""
        return {
            # Profile stream
            "subject_logits": self.subject_cls(profile_pooled),
            "personality_pred": self.personality(profile_pooled),
            "sex_logits": self.sex_cls(profile_pooled),
            "age_pred": self.age_reg(profile_pooled).squeeze(-1),
            # State stream
            "task_logits": self.task_cls(state_pooled),
            "session_logits": self.session_cls(state_pooled),
            "next_summary_pred": self.next_summary(state_pooled),
            "delta_summary_pred": self.delta_summary(state_pooled),
            "recon_pred": self.recon_head(state_pooled),
        }


# ---------------------------------------------------------------------------
# Dual-Stream Pretraining Loss
# ---------------------------------------------------------------------------

class DualStreamPretrainingLoss(nn.Module):
    """Loss for dual-stream pretraining.

    Profile objectives: subject, personality, sex, age
    State objectives: task, session, next_summary, delta_summary, masked_recon
    """

    def __init__(
        self,
        # Profile stream weights
        w_subject: float = 1.0,
        w_personality: float = 0.5,
        w_sex: float = 0.5,
        w_age: float = 0.3,
        # State stream weights
        w_task: float = 1.0,
        w_session: float = 0.5,
        w_next: float = 0.5,
        w_delta: float = 0.3,
        w_recon: float = 0.3,
    ) -> None:
        super().__init__()
        # Profile
        self.w_subject = w_subject
        self.w_personality = w_personality
        self.w_sex = w_sex
        self.w_age = w_age
        # State
        self.w_task = w_task
        self.w_session = w_session
        self.w_next = w_next
        self.w_delta = w_delta
        self.w_recon = w_recon

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.sl1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        task_idx: torch.Tensor,
        subject_idx: torch.Tensor,
        session_idx: torch.Tensor,
        personality: torch.Tensor,
        sex_label: torch.Tensor,
        age: torch.Tensor,
        next_summary: torch.Tensor,
        has_next: torch.Tensor,
        delta_summary: torch.Tensor | None = None,
        has_delta: torch.Tensor | None = None,
        recon_target: torch.Tensor | None = None,
        recon_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = preds["task_logits"].device

        # --- Profile stream losses ---
        l_subj = self.ce(preds["subject_logits"].float(), subject_idx)
        l_per = self.sl1(preds["personality_pred"].float(), personality.float())

        # Sex (ignore unknown = -1)
        valid_sex = sex_label >= 0
        if valid_sex.any():
            l_sex = self.ce(preds["sex_logits"][valid_sex].float(), sex_label[valid_sex])
        else:
            l_sex = torch.tensor(0.0, device=device)

        l_age = self.sl1(preds["age_pred"].float(), age.float())

        # --- State stream losses ---
        l_task = self.ce(preds["task_logits"].float(), task_idx)
        l_ses = self.ce(preds["session_logits"].float(), session_idx)

        # Next-summary (masked where no next window)
        valid_next = has_next > 0.5
        if valid_next.any():
            l_next = self.sl1(
                preds["next_summary_pred"][valid_next].float(),
                next_summary[valid_next].float(),
            )
        else:
            l_next = torch.tensor(0.0, device=device)

        # Delta prediction (summary_t+1 - summary_t)
        if delta_summary is not None and has_delta is not None:
            valid_delta = has_delta > 0.5
            if valid_delta.any():
                l_delta = self.sl1(
                    preds["delta_summary_pred"][valid_delta].float(),
                    delta_summary[valid_delta].float(),
                )
            else:
                l_delta = torch.tensor(0.0, device=device)
        else:
            # Fallback: use next_summary - current_summary as delta
            l_delta = torch.tensor(0.0, device=device)

        # Masked reconstruction
        if recon_target is not None and recon_mask is not None:
            # recon_mask: (B, n_modalities) binary, 1 = masked (predict)
            pred_recon = preds["recon_pred"]  # (B, n_modalities * d_recon)
            B = pred_recon.size(0)
            n_mod = recon_mask.size(1)
            d_r = pred_recon.size(1) // n_mod
            pred_recon = pred_recon.view(B, n_mod, d_r)
            target_recon = recon_target.view(B, n_mod, d_r)

            # Only compute loss on masked modalities
            mask_expanded = recon_mask.unsqueeze(-1).float()  # (B, n_mod, 1)
            if mask_expanded.sum() > 0:
                l_recon = (
                    ((pred_recon - target_recon) ** 2) * mask_expanded
                ).sum() / mask_expanded.sum() / d_r
            else:
                l_recon = torch.tensor(0.0, device=device)
        else:
            l_recon = torch.tensor(0.0, device=device)

        # Total
        total_profile = (
            self.w_subject * l_subj
            + self.w_personality * l_per
            + self.w_sex * l_sex
            + self.w_age * l_age
        )
        total_state = (
            self.w_task * l_task
            + self.w_session * l_ses
            + self.w_next * l_next
            + self.w_delta * l_delta
            + self.w_recon * l_recon
        )
        total = total_profile + total_state

        return {
            "total": total,
            "profile_total": total_profile,
            "state_total": total_state,
            # Profile components
            "subject": l_subj,
            "personality": l_per,
            "sex": l_sex,
            "age": l_age,
            # State components
            "task": l_task,
            "session": l_ses,
            "next": l_next,
            "delta": l_delta,
            "recon": l_recon,
        }
