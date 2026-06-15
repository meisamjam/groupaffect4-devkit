"""train_affectai.py

Three-phase training script for MuMTAffect adapted to GroupAffect-4.

Phases (mirrors the original MuMTAffect multiphase_simple.py):
  Phase 1 – Personality pretraining    (alpha=1.0, 80 epochs)
  Phase 2 – Joint multitask training    (alpha=0.3, 200 epochs, cosine LR)
  Phase 3 – Fine-tuning                 (alpha=0.1, 60 epochs)

Improvements over baseline run:
  - Per-dimension inverse-frequency class weights (--class-weights auto)
  - CosineAnnealingWarmRestarts scheduler in Phase 2 for better LR annealing
  - Increased patience and weight decay to reduce overfitting
  - Leave-One-Group-Out cross-validation (--cv flag)

Usage:
  # Standard single-split training
  python tools/mumt/train_affectai.py --data-path data/mumt/dataset.pkl

  # With auto class-weights
  python tools/mumt/train_affectai.py --data-path data/mumt/dataset.pkl --class-weights auto

  # LOGO cross-validation
  python tools/mumt/train_affectai.py --data-path data/mumt/dataset.pkl --cv

  # Resume from phase 2
  python tools/mumt/train_affectai.py --data-path data/mumt/dataset.pkl \\
      --checkpoint data/mumt/runs/checkpoint_phase2.pt --start-phase 3
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset_affectai import (
    BIG_FIVE_COLS,
    GroupAffectDataset,
    N_TASKS,
    PretrainDataset,
    bin_vad_adaptive,
    build_summary_key_order,
    make_session2idx,
    make_user2idx,
    split_by_subject,
    split_by_subject_stratified,
)
from model_affectai import (
    MuMTAffectGroupAffect,
    MuMTAffectLoss,
    PhysiologicalContrastiveLoss,
    PretrainingHeads,
    PretrainingLoss,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASE_CONFIG = {
    1: dict(alpha=1.0,  epochs=60,  lr_base=1e-4,  weight_decay=5e-3, patience=10,
            scheduler="exp",    lr_decay=0.98),
    2: dict(alpha=0.3,  epochs=120, lr_base=5e-4,  weight_decay=5e-3, patience=15,
            scheduler="cosine", lr_decay=0.95, T_0=30, T_mult=2),
    3: dict(alpha=0.1,  epochs=40,  lr_base=5e-5,  weight_decay=5e-3, patience=10,
            scheduler="exp",    lr_decay=0.98),
}

# v13 config: lower Phase 2 LR, no cosine restarts, tighter patience
PHASE_CONFIG_V13 = {
    1: dict(alpha=1.0,  epochs=60,  lr_base=1e-4,  weight_decay=5e-3, patience=10,
            scheduler="exp",    lr_decay=0.98),
    2: dict(alpha=0.3,  epochs=120, lr_base=2e-4,  weight_decay=5e-3, patience=20,
            scheduler="cosine_decay", lr_decay=0.95),
    3: dict(alpha=0.1,  epochs=60,  lr_base=3e-5,  weight_decay=5e-3, patience=15,
            scheduler="cosine_decay", lr_decay=0.98),
}

# v14 config: PCPE architecture — freeze backbone from Phase 1, pure emotion heads
PHASE_CONFIG_V14 = {
    1: dict(alpha=0.0,  epochs=80,  lr_base=2e-4,  weight_decay=5e-3, patience=15,
            scheduler="cosine_decay", lr_decay=0.98),
    2: dict(alpha=0.1,  epochs=120, lr_base=1e-4,  weight_decay=5e-3, patience=25,
            scheduler="cosine_decay", lr_decay=0.95),
    3: dict(alpha=0.0,  epochs=80,  lr_base=3e-5,  weight_decay=5e-3, patience=20,
            scheduler="cosine_decay", lr_decay=0.98),
}

# v15 config: End-to-end with discriminative LR + PCPE regularizer + early stopping
# - Backbone unfrozen but trained at 10× lower LR (ULMFiT-style)
# - PCPE contrastive loss as regularizer during supervised phases
# - Proper early stopping (patience in evaluations)
PHASE_CONFIG_V15 = {
    1: dict(alpha=0.0,  epochs=120, lr_base=2e-4,  lr_backbone=2e-5,
            weight_decay=5e-3, patience=20,
            scheduler="cosine_decay", pcpe_weight=0.3),
    2: dict(alpha=0.05, epochs=80,  lr_base=1e-4,  lr_backbone=1e-5,
            weight_decay=5e-3, patience=15,
            scheduler="cosine_decay", pcpe_weight=0.3),
    3: dict(alpha=0.0,  epochs=60,  lr_base=3e-5,  lr_backbone=3e-6,
            weight_decay=5e-3, patience=15,
            scheduler="cosine_decay", pcpe_weight=0.1),
}

BATCH_SIZE = 32
SUMMARY_DIM = 40   # gaze(6) + pupil(7) + eda(11) + ppg(7) + imu(16) summary features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(labels: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced emotion labels."""
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (n_classes * counts)
    return weights / weights.sum()


def compute_per_dim_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
    n_classes: int = 3,
    vad_thresholds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute separate inverse-frequency class weights for V, A, D."""
    weights = {}
    for col in ("valence", "arousal", "dominance"):
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if vad_thresholds is not None:
            bins = torch.tensor(
                [bin_vad_adaptive(float(v), vad_thresholds[col]) for v in vals],
                dtype=torch.long,
            )
        else:
            from dataset_affectai import bin_vad
            bins = torch.tensor([bin_vad(float(v)) for v in vals], dtype=torch.long)
        weights[col] = compute_class_weights(bins, n_classes).to(device)
        log.info("Class weights %s: %s", col, weights[col].tolist())
    return weights


def fit_scalers(train_df: pd.DataFrame) -> dict:
    """Fit per-modality StandardScalers on the training split."""
    scalers = {}
    for key, col in [
        ("gaze", "gaze_seq"),
        ("pupil", "pupil_seq"),
        ("eda", "eda_seq"),
        ("ppg", "ppg_seq"),
        ("imu", "imu_seq"),
    ]:
        arrays = [df.values for df in train_df[col] if isinstance(df, pd.DataFrame)]
        if arrays:
            stacked = np.vstack(arrays).astype(np.float32)
            sc = StandardScaler()
            sc.fit(stacked)
            scalers[key] = sc
    return scalers


def make_summary_dim(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Return (summary_dim, key_order) computed from the full dataset."""
    key_order = build_summary_key_order(df)
    return len(key_order), key_order


def compute_vad_thresholds(
    train_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Compute 33rd and 67th percentile thresholds for VAD columns from training data.

    Returns a dict mapping column name → (t1, t2) used in bin_vad_adaptive.
    
    Note: If percentiles are NaN or produce missing classes, falls back to fixed bins
    and logs a warning.
    """
    thresholds: dict[str, tuple[float, float]] = {}
    for col in ("valence", "arousal", "dominance"):
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy()
        
        if len(vals) == 0:
            log.warning("No valid values for %s; using fixed bins", col)
            thresholds[col] = (4.0, 6.0)  # fixed fallback
            continue
        
        t1 = float(np.percentile(vals, 33))
        t2 = float(np.percentile(vals, 67))
        
        # Validate: ensure thresholds are not NaN and will create 3 classes
        if np.isnan(t1) or np.isnan(t2):
            log.warning("Percentiles for %s are NaN; using fixed bins (4.0, 6.0)", col)
            thresholds[col] = (4.0, 6.0)
            continue
        
        # Validate that the thresholds actually create 3 distinct classes
        from tools.mumt.dataset_affectai import bin_vad_adaptive
        test_bins = [bin_vad_adaptive(v, (t1, t2)) for v in vals]
        unique_bins = len(set(test_bins))
        
        if unique_bins < 3:
            log.warning(
                "Percentiles for %s only create %d classes; using fixed bins (4.0, 6.0)",
                col, unique_bins
            )
            thresholds[col] = (4.0, 6.0)
        else:
            thresholds[col] = (t1, t2)
            log.info("VAD thresholds %s: p33=%.2f  p67=%.2f  (3 classes OK)", col, t1, t2)
    
    return thresholds


def set_transformers_trainable(model: nn.Module, trainable: bool) -> None:
    """Enable/disable gradients for transformer blocks only.

    This freezes modality/fusion transformer layers while keeping projection
    layers, CNN branches, embeddings, and prediction heads trainable.
    
    For GRU-based models, this is a no-op since GRU modules are not frozen.
    """
    if not isinstance(model, MuMTAffectGroupAffect):
        return

    # Collect transformer modules (skip if using GRU encoders/fusion)
    transformer_modules = []
    for encoder in [model.gaze_encoder, model.pupil_encoder, model.eda_encoder,
                    model.ppg_encoder, model.imu_encoder]:
        if hasattr(encoder, 'transformer'):
            transformer_modules.append(encoder.transformer)
    if hasattr(model.fusion, 'transformer'):
        transformer_modules.append(model.fusion.transformer)
    
    for module in transformer_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of parameters with gradients enabled."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MuMTAffectLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_v_true, all_v_pred = [], []
    all_a_true, all_a_pred = [], []
    all_d_true, all_d_pred = [], []
    all_p_true, all_p_pred = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        gaze, pupil, eda, ppg, imu, personality, emotions, user_ids, summary, _, task_oh, _ses = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
        ]
        personality_gt = personality.float()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(gaze.float(), pupil.float(), eda.float(),
                            ppg.float(), imu.float(),
                            summary.float(), user_ids,
                            personality_gt=personality_gt,
                            task_onehot=task_oh.float())
        losses = criterion(outputs, emotions, personality_gt)
        total_loss += losses["total"].item()
        n_batches += 1

        all_v_true.extend(emotions[:, 0].cpu().numpy())
        all_a_true.extend(emotions[:, 1].cpu().numpy())
        all_d_true.extend(emotions[:, 2].cpu().numpy())

        all_v_pred.extend(outputs["valence_logits"].argmax(-1).cpu().numpy())
        all_a_pred.extend(outputs["arousal_logits"].argmax(-1).cpu().numpy())
        all_d_pred.extend(outputs["dominance_logits"].argmax(-1).cpu().numpy())

        all_p_true.append(personality_gt.cpu().numpy())
        all_p_pred.append(outputs["personality_pred"].cpu().numpy())

    p_true = np.vstack(all_p_true)
    p_pred = np.vstack(all_p_pred) if not criterion.personality_ternary else all_p_pred

    # Ternary personality: compute accuracy per trait (3-class argmax)
    if criterion.personality_ternary:
        # p_pred is list of (B, 5, 3) arrays; p_true is (N, 5) continuous
        p_pred_cat = np.concatenate(all_p_pred, axis=0)  # (N, 5, 3)
        p_pred_class = p_pred_cat.argmax(axis=2)  # (N, 5)
        thresholds = criterion.personality_thresholds.cpu().numpy()  # (5, 2)
        p_true_class = np.zeros_like(p_true, dtype=int)
        p_true_class[p_true > thresholds[:, 0]] = 1
        p_true_class[p_true > thresholds[:, 1]] = 2
        trait_acc = [(p_true_class[:, i] == p_pred_class[:, i]).mean()
                     for i in range(p_true.shape[1])]
        personality_metric = float(np.mean(trait_acc))
        personality_per_trait = trait_acc
    elif criterion.personality_binary:
        thresholds = criterion.personality_thresholds.cpu().numpy()
        p_binary_true = (p_true > thresholds).astype(int)
        p_binary_pred = (p_pred > 0).astype(int)  # logits > 0 → predicted high
        trait_acc = [(p_binary_true[:, i] == p_binary_pred[:, i]).mean()
                     for i in range(p_true.shape[1])]
        personality_metric = float(np.mean(trait_acc))
        personality_per_trait = trait_acc
    else:
        r2_scores = [r2_score(p_true[:, i], p_pred[:, i]) for i in range(p_true.shape[1])]
        personality_metric = float(np.mean(r2_scores))
        personality_per_trait = r2_scores

    labels = list(range(3))
    return {
        "loss": total_loss / max(n_batches, 1),
        "valence_f1": f1_score(all_v_true, all_v_pred, average="macro", labels=labels, zero_division=0),
        "arousal_f1": f1_score(all_a_true, all_a_pred, average="macro", labels=labels, zero_division=0),
        "dominance_f1": f1_score(all_d_true, all_d_pred, average="macro", labels=labels, zero_division=0),
        "personality_r2_mean": personality_metric,
        "personality_r2_per_trait": personality_per_trait,
    }


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def run_phase(
    phase: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    scaler: torch.cuda.amp.GradScaler,
    class_weights: dict[str, torch.Tensor] | None = None,
    freeze_transformers: bool = False,
    freeze_transformers_from_phase: int = 2,
    label_smoothing: float = 0.0,
    grad_accum_steps: int = 1,
    eval_every: int = 5,
    phase_config: dict | None = None,
    personality_binary: bool = False,
    personality_ternary: bool = False,
    personality_thresholds: torch.Tensor | None = None,
) -> None:
    cfg = (phase_config or PHASE_CONFIG)[phase]
    log.info("=" * 60)
    log.info("Phase %d  |  alpha=%.2f  |  epochs=%d  |  lr=%.0e  |  wd=%.0e",
             phase, cfg["alpha"], cfg["epochs"], cfg["lr_base"], cfg["weight_decay"])
    log.info("=" * 60)

    criterion = MuMTAffectLoss(alpha=cfg["alpha"], class_weights=class_weights,
                               label_smoothing=label_smoothing,
                               personality_binary=personality_binary,
                               personality_ternary=personality_ternary,
                               personality_thresholds=personality_thresholds).to(device)

    should_freeze_transformers = freeze_transformers and phase >= freeze_transformers_from_phase
    set_transformers_trainable(model, trainable=not should_freeze_transformers)
    if should_freeze_transformers:
        log.info("Transformer blocks are frozen for phase %d.", phase)
    else:
        log.info("Transformer blocks are trainable for phase %d.", phase)

    # PCPE regularizer during supervised phases (v15)
    pcpe_weight = cfg.get("pcpe_weight", 0.0)
    pcpe_loss_fn = PhysiologicalContrastiveLoss(temperature=0.07).to(device) if pcpe_weight > 0 else None
    if pcpe_weight > 0:
        log.info("PCPE regularizer active: weight=%.2f", pcpe_weight)

    # Per-group learning rates — discriminative LR when lr_backbone is specified
    lr_backbone = cfg.get("lr_backbone", cfg["lr_base"])
    backbone_param_names = set()
    for module_name in ("gaze_encoder", "pupil_encoder", "eda_encoder",
                        "ppg_encoder", "imu_encoder", "fusion"):
        for n, _ in model.named_parameters():
            if n.startswith(module_name + "."):
                backbone_param_names.add(n)

    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "personality" in n],
         "lr": cfg["lr_base"] * 0.5},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and
                    ("valence_head" in n or "arousal_head" in n or "dominance_head" in n)],
         "lr": cfg["lr_base"]},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n in backbone_param_names],
         "lr": lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and
                    n not in backbone_param_names and
                    "personality" not in n and
                    "valence_head" not in n and
                    "arousal_head" not in n and
                    "dominance_head" not in n],
         "lr": cfg["lr_base"]},
    ]
    param_groups = [group for group in param_groups if group["params"]]
    optimizer = torch.optim.Adam(param_groups, weight_decay=cfg["weight_decay"])
    if lr_backbone != cfg["lr_base"]:
        log.info("Discriminative LR: heads=%.0e, backbone=%.0e", cfg["lr_base"], lr_backbone)
    log.info("Trainable parameters this phase: %d", count_trainable_parameters(model))

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg["T_0"], T_mult=cfg.get("T_mult", 1), eta_min=1e-6
        )
    elif cfg["scheduler"] == "cosine_decay":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["lr_decay"])

    best_val_loss = float("inf")
    best_val_f1 = -1.0          # track best macro emotion F1 for checkpoint
    patience_counter = 0
    best_ckpt = output_dir / f"checkpoint_phase{phase}.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_pcpe = 0.0
        optimizer.zero_grad()

        for step_i, batch in enumerate(train_loader):
            gaze, pupil, eda, ppg, imu, personality, emotions, user_ids, summary, _, task_oh, _ses = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(gaze.float(), pupil.float(), eda.float(),
                                ppg.float(), imu.float(),
                                summary.float(), user_ids,
                                personality_gt=personality.float(),
                                task_onehot=task_oh.float())
                losses = criterion(outputs, emotions, personality.float())
                loss = losses["total"] / grad_accum_steps

            # PCPE regularizer — computed in float32 outside autocast (v15)
            if pcpe_loss_fn is not None:
                z = F.normalize(outputs["personality_embed"].float(), dim=-1)
                l_pcpe = pcpe_loss_fn(z, user_ids)
                loss = loss + (pcpe_weight * l_pcpe) / grad_accum_steps
                train_pcpe += l_pcpe.item()

            scaler.scale(loss).backward()

            if (step_i + 1) % grad_accum_steps == 0 or (step_i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += losses["total"].item()

        if cfg["scheduler"] == "cosine":
            scheduler.step(epoch - 1)
        else:
            scheduler.step()
        train_loss /= max(len(train_loader), 1)
        train_pcpe /= max(len(train_loader), 1)

        if epoch % eval_every == 0 or epoch == cfg["epochs"]:
            val_metrics = evaluate(model, val_loader, criterion, device)
            pcpe_str = f" pcpe={train_pcpe:.3f}" if pcpe_weight > 0 else ""
            p_label = "P_acc" if (personality_binary or personality_ternary) else "P_R2"
            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "V_f1=%.3f A_f1=%.3f D_f1=%.3f | %s=%.3f%s",
                epoch, cfg["epochs"], train_loss, val_metrics["loss"],
                val_metrics["valence_f1"], val_metrics["arousal_f1"],
                val_metrics["dominance_f1"], p_label, val_metrics["personality_r2_mean"],
                pcpe_str,
            )
            # Save on best macro emotion F1 (phases 2+3); val_loss for phase 1
            val_f1 = (val_metrics["valence_f1"] + val_metrics["arousal_f1"]
                      + val_metrics["dominance_f1"]) / 3.0
            improved = (phase == 1 and val_metrics["loss"] < best_val_loss) or \
                       (phase != 1 and val_f1 > best_val_f1)
            if improved:
                best_val_loss = val_metrics["loss"]
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
                log.info("  → new best (F1=%.3f, loss=%.4f)", val_f1, val_metrics["loss"])
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    log.info("Early stopping at epoch %d (no improvement for %d evals).",
                             epoch, cfg["patience"])
                    break

    # Reload best weights before returning
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        log.info("Loaded best checkpoint from %s.", best_ckpt)


# ---------------------------------------------------------------------------
# Phase 0 — self-supervised pre-training on pretrain_dataset.pkl
# ---------------------------------------------------------------------------

def run_pretraining(
    model: nn.Module,
    pretrain_path: str,
    summary_key_order: list[str],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    output_dir: Path,
    scalers: dict,
    n_epochs: int = 80,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    participants_tsv: str | None = None,
    w_task: float = 1.0,
    w_subject: float = 1.0,
    w_session: float = 0.5,
    w_personality: float = 0.5,
    w_sex: float = 0.5,
    w_age: float = 0.3,
    w_next: float = 0.5,
    nsp_warmup_epochs: int = 0,
    disentangle_branches: bool = False,
    adv_user_on_moment: float = 0.0,
    adv_moment_on_user: float = 0.0,
    grl_lambda: float = 1.0,
    normalize_targets: bool = True,
    w_contrast: float = 0.0,
    grl_schedule: bool = False,
) -> None:
    """Phase 0: self-supervised pre-training on dense sliding windows.

    Trains PretrainingHeads (task, subject, session, personality, sex, age)
    on the backbone's ``fused_pooled`` vector.  The backbone weights are also
    updated so the shared encoders learn general representations.

    Args:
        participants_tsv: Path to BIDS-root participants.tsv for age/sex labels.
    """
    log.info("=" * 60)
    log.info("Phase 0 — self-supervised pre-training  |  epochs=%d  |  lr=%.0e", n_epochs, lr)
    log.info("Loading pretrain pickle: %s", pretrain_path)

    pt_df = pd.read_pickle(pretrain_path)
    log.info("Pretrain windows: %d  |  subjects: %d  |  sessions: %d",
             len(pt_df), pt_df["subject_id"].nunique(), pt_df["session_id"].nunique())

    pt_user2idx    = make_user2idx(pt_df)
    pt_session2idx = make_session2idx(pt_df)
    n_subjects_pt  = len(pt_user2idx)
    n_sessions_pt  = len(pt_session2idx)

    ds = PretrainDataset(
        pt_df, pt_user2idx, pt_session2idx, summary_key_order,
        device=device, augment=False,   # 8978 samples — no augmentation needed
        participants_tsv=participants_tsv,
        scalers=scalers,
        normalize_targets=normalize_targets,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, num_workers=0)

    heads = PretrainingHeads(
        d_fuse=128,
        n_tasks=N_TASKS,
        n_subjects=n_subjects_pt,
        n_sessions=n_sessions_pt,
        n_personality=5,
        summary_dim=len(summary_key_order),
    ).to(device)

    criterion = PretrainingLoss(
        w_task=w_task, w_subject=w_subject, w_session=w_session,
        w_personality=w_personality, w_sex=w_sex, w_age=w_age, w_next=w_next,
        w_adv_user_on_moment=adv_user_on_moment if disentangle_branches else 0.0,
        w_adv_moment_on_user=adv_moment_on_user if disentangle_branches else 0.0,
    ).to(device)

    # PCPE contrastive loss on personality_ctx
    pcpe_loss_fn = PhysiologicalContrastiveLoss(temperature=0.07).to(device) if w_contrast > 0 else None

    orig_weights = {
        "task": criterion.w_task,
        "subject": criterion.w_subject,
        "session": criterion.w_session,
        "personality": criterion.w_personality,
        "sex": criterion.w_sex,
        "age": criterion.w_age,
        "next": criterion.w_next,
        "adv_user_on_moment": criterion.w_adv_user_on_moment,
        "adv_moment_on_user": criterion.w_adv_moment_on_user,
    }

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    ckpt_path = output_dir / "checkpoint_phase0.pt"
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        heads.train()
        epoch_loss = 0.0
        epoch_task = 0.0
        epoch_subj = 0.0
        epoch_per  = 0.0
        epoch_sex  = 0.0
        epoch_age  = 0.0
        epoch_next = 0.0
        epoch_adv_u2m = 0.0
        epoch_adv_m2u = 0.0
        epoch_contrast = 0.0

        # GRL lambda schedule: Ganin et al. 2016 S-curve ramp 0→1
        if grl_schedule:
            progress = (epoch - 1) / max(n_epochs - 1, 1)
            grl_lambda_eff = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        else:
            grl_lambda_eff = grl_lambda

        if nsp_warmup_epochs > 0 and epoch <= nsp_warmup_epochs:
            criterion.w_task = 0.0
            criterion.w_subject = 0.0
            criterion.w_session = 0.0
            criterion.w_personality = 0.0
            criterion.w_sex = 0.0
            criterion.w_age = 0.0
            criterion.w_next = max(orig_weights["next"], 1.0)
            criterion.w_adv_user_on_moment = 0.0
            criterion.w_adv_moment_on_user = 0.0
        else:
            criterion.w_task = orig_weights["task"]
            criterion.w_subject = orig_weights["subject"]
            criterion.w_session = orig_weights["session"]
            criterion.w_personality = orig_weights["personality"]
            criterion.w_sex = orig_weights["sex"]
            criterion.w_age = orig_weights["age"]
            criterion.w_next = orig_weights["next"]
            criterion.w_adv_user_on_moment = orig_weights["adv_user_on_moment"]
            criterion.w_adv_moment_on_user = orig_weights["adv_moment_on_user"]

        for batch in loader:
            (gaze, pupil, eda, ppg, imu,
             summary, subject_idx, session_idx, task_idx,
             personality, sex_label, age, next_summary, has_next) = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(
                    gaze.float(), pupil.float(), eda.float(),
                    ppg.float(), imu.float(),
                    summary.float(), subject_idx,
                )
                pt_preds = heads(out["fused_pooled"], grl_lambda=grl_lambda_eff)
                losses = criterion(
                    pt_preds, task_idx, subject_idx, session_idx,
                    personality, sex_label, age, next_summary, has_next,
                )
                total_loss = losses["total"]

            # PCPE contrastive loss on personality_ctx — computed in float32 outside autocast
            if pcpe_loss_fn is not None:
                z = F.normalize(out["personality_embed"].float(), dim=-1)
                l_contrast = pcpe_loss_fn(z, subject_idx)
                total_loss = total_loss + w_contrast * l_contrast
            else:
                l_contrast = torch.tensor(0.0, device=device)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss     += total_loss.item()
            epoch_task     += losses["task"].item()
            epoch_subj     += losses["subject"].item()
            epoch_per      += losses["personality"].item()
            epoch_sex      += losses["sex"].item()
            epoch_age      += losses["age"].item()
            epoch_next     += losses["next"].item()
            epoch_adv_u2m  += losses["adv_user_on_moment"].item()
            epoch_adv_m2u  += losses["adv_moment_on_user"].item()
            epoch_contrast += l_contrast.item()

        scheduler.step(epoch - 1)
        n = max(len(loader), 1)
        epoch_loss /= n; epoch_task /= n; epoch_subj /= n
        epoch_per  /= n; epoch_sex  /= n; epoch_age  /= n; epoch_next /= n
        epoch_adv_u2m /= n; epoch_adv_m2u /= n; epoch_contrast /= n

        log.info(
            "Pretrain %3d/%d | loss=%.4f | task=%.3f subj=%.3f per=%.3f sex=%.3f age=%.3f next=%.3f"
            " ctr=%.3f adv_u2m=%.3f adv_m2u=%.3f",
            epoch, n_epochs,
            epoch_loss, epoch_task, epoch_subj, epoch_per, epoch_sex, epoch_age, epoch_next,
            epoch_contrast, epoch_adv_u2m, epoch_adv_m2u,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_path)

    # Reload best backbone weights (heads are discarded)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log.info("Phase 0 complete — backbone saved to %s (best_loss=%.4f)", ckpt_path, best_loss)


# ---------------------------------------------------------------------------
# Leave-One-Group-Out cross-validation
# ---------------------------------------------------------------------------

def run_logo_cv(
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    summary_dim: int,
    summary_key_order: list[str],
    user2idx: dict[str, int],
    sample_row: pd.Series,
    output_dir: Path,
    vad_thresholds: dict[str, tuple[float, float]] | None = None,
    freeze_transformers: bool = False,
    freeze_transformers_from_phase: int = 2,
) -> None:
    """Leave-One-Group-Out cross-validation.

    Each fold holds out one session group.  Metrics are aggregated across folds.
    """
    groups = sorted(df["group_id"].unique())
    log.info("LOGO-CV: %d folds (%s)", len(groups), groups)

    personality_classes = 3 if args.personality_ternary else (2 if args.personality_binary else 1)

    fold_results = []
    for fold_i, held_out in enumerate(groups, 1):
        log.info("-" * 60)
        log.info("Fold %d/%d  held-out group: %s", fold_i, len(groups), held_out)

        train_df = df[df["group_id"] != held_out].copy()
        test_df  = df[df["group_id"] == held_out].copy()

        # Use 15% of train for validation (subject-level or stratified per-subject)
        if args.stratified_split:
            train_df, val_df, _ = split_by_subject_stratified(train_df, test_frac=0.0, val_frac=0.15)
        else:
            train_df, val_df, _ = split_by_subject(train_df, test_frac=0.0, val_frac=0.15)

        fold_user2idx = make_user2idx(df)  # keep global idx so embeddings are consistent
        scalers = fit_scalers(train_df)

        class_weights = None
        if args.class_weights == "auto":
            class_weights = compute_per_dim_class_weights(
                train_df, device, vad_thresholds=vad_thresholds
            )

        # Personality binary thresholds (per-fold median)
        personality_thresholds = None
        if args.personality_ternary:
            from dataset_affectai import BIG_FIVE_COLS
            p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
            p33 = np.nanpercentile(p_values, 33.3, axis=0)
            p67 = np.nanpercentile(p_values, 66.7, axis=0)
            personality_thresholds = torch.tensor(
                np.stack([p33, p67], axis=1), dtype=torch.float32
            )  # (5, 2)
        elif args.personality_binary:
            from dataset_affectai import BIG_FIVE_COLS
            p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
            personality_thresholds = torch.tensor(
                np.nanmedian(p_values, axis=0), dtype=torch.float32
            )

        train_ds = GroupAffectDataset(train_df, fold_user2idx, scalers, augment=True,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))
        val_ds   = GroupAffectDataset(val_df,   fold_user2idx, scalers, augment=False,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))
        test_ds  = GroupAffectDataset(test_df,  fold_user2idx, scalers, augment=False,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        gaze_dim  = len(sample_row["gaze_seq"].columns)  if isinstance(sample_row["gaze_seq"],  pd.DataFrame) else 9
        pupil_dim = len(sample_row["pupil_seq"].columns) if isinstance(sample_row["pupil_seq"], pd.DataFrame) else 3
        eda_dim   = len(sample_row["eda_seq"].columns)   if isinstance(sample_row["eda_seq"],   pd.DataFrame) else 5
        ppg_dim   = len(sample_row["ppg_seq"].columns)   if isinstance(sample_row["ppg_seq"],   pd.DataFrame) else 3
        imu_dim   = len(sample_row["imu_seq"].columns)   if isinstance(sample_row["imu_seq"],   pd.DataFrame) else 6

        fold_model = MuMTAffectGroupAffect(
            gaze_dim=gaze_dim, pupil_dim=pupil_dim, eda_dim=eda_dim,
            ppg_dim=ppg_dim, imu_dim=imu_dim,
            summary_dim=summary_dim, n_subjects=len(fold_user2idx),
            n_personality=5, n_emotion_classes=3,
            d_model_enc=64, d_model_fuse=128, t_out=16,
            n_tasks=N_TASKS,
            per_dim_queries=args.per_dim_queries,
            per_dim_projections=args.per_dim_projections,
            use_se_blocks=args.use_se_blocks,
            use_scaled_attention=args.use_scaled_attention,
            use_global_token=args.use_global_token,
            use_gru=args.use_gru,
            per_dim_gate=args.per_dim_gate,
            personality_classes=personality_classes,
            personality_as_input=args.personality_as_input,
        ).to(device)

        fold_dir = output_dir / f"fold_{fold_i:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_phase_cfg = PHASE_CONFIG_V15 if args.v15_training else (PHASE_CONFIG_V14 if args.v14_training else (PHASE_CONFIG_V13 if args.v13_training else PHASE_CONFIG))
        for phase in range(args.start_phase, 4):
            run_phase(phase, fold_model, train_loader, val_loader,
                      device, fold_dir, amp_scaler, class_weights,
                      freeze_transformers=freeze_transformers,
                      freeze_transformers_from_phase=freeze_transformers_from_phase,
                      label_smoothing=args.label_smoothing,
                      grad_accum_steps=args.grad_accum_steps,
                      eval_every=args.eval_every,
                      phase_config=fold_phase_cfg,
                      personality_binary=args.personality_binary,
                      personality_ternary=args.personality_ternary,
                      personality_thresholds=personality_thresholds)

        criterion_eval = MuMTAffectLoss(alpha=0.1, class_weights=class_weights,
                                        personality_binary=args.personality_binary,
                                        personality_ternary=args.personality_ternary,
                                        personality_thresholds=personality_thresholds).to(device)
        metrics = evaluate(fold_model, test_loader, criterion_eval, device)
        metrics["fold"] = fold_i
        metrics["held_out_group"] = held_out
        metrics["n_test"] = len(test_df)
        fold_results.append(metrics)

        log.info("Fold %d  V_F1=%.3f  A_F1=%.3f  D_F1=%.3f  P_R2=%.3f",
                 fold_i, metrics["valence_f1"], metrics["arousal_f1"],
                 metrics["dominance_f1"], metrics["personality_r2_mean"])

        torch.save(fold_model.state_dict(), fold_dir / "model_final.pt")

    # ---- Aggregate ----
    log.info("=" * 60)
    log.info("LOGO-CV aggregate results (%d folds)", len(fold_results))
    for metric in ["valence_f1", "arousal_f1", "dominance_f1", "personality_r2_mean"]:
        vals = [r[metric] for r in fold_results]
        log.info("  %-25s mean=%.3f  std=%.3f", metric, np.mean(vals), np.std(vals))

    cv_df = pd.DataFrame(fold_results)
    cv_path = output_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    log.info("CV results saved to %s", cv_path)


# ---------------------------------------------------------------------------
# Leave-One-Subject-Out cross-validation
# ---------------------------------------------------------------------------

def run_loso_cv(
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    summary_dim: int,
    summary_key_order: list[str],
    user2idx: dict[str, int],
    sample_row: pd.Series,
    output_dir: Path,
    vad_thresholds: dict[str, tuple[float, float]] | None = None,
    freeze_transformers: bool = False,
    freeze_transformers_from_phase: int = 2,
) -> None:
    """Leave-One-Subject-Out cross-validation.

    Each fold holds out one participant.  Metrics are aggregated across folds.
    """
    subjects = sorted(df["subject_id"].unique())
    log.info("LOSO-CV: %d folds (subjects)", len(subjects))

    personality_classes = 3 if args.personality_ternary else (2 if args.personality_binary else 1)

    fold_results = []
    for fold_i, held_out_subj in enumerate(subjects, 1):
        test_df = df[df["subject_id"] == held_out_subj].copy()
        if len(test_df) == 0:
            continue
        log.info("-" * 60)
        log.info("Fold %d/%d  held-out subject: %s (%d windows)",
                 fold_i, len(subjects), held_out_subj, len(test_df))

        train_df = df[df["subject_id"] != held_out_subj].copy()

        # Use 15% of train for validation (subject-level or stratified per-subject)
        if args.stratified_split:
            train_df, val_df, _ = split_by_subject_stratified(train_df, test_frac=0.0, val_frac=0.15)
        else:
            train_df, val_df, _ = split_by_subject(train_df, test_frac=0.0, val_frac=0.15)

        fold_user2idx = make_user2idx(df)  # keep global idx so embeddings are consistent
        scalers = fit_scalers(train_df)

        class_weights = None
        if args.class_weights == "auto":
            class_weights = compute_per_dim_class_weights(
                train_df, device, vad_thresholds=vad_thresholds
            )

        # Personality thresholds (per-fold)
        personality_thresholds = None
        if args.personality_ternary:
            from dataset_affectai import BIG_FIVE_COLS
            p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
            p33 = np.nanpercentile(p_values, 33.3, axis=0)
            p67 = np.nanpercentile(p_values, 66.7, axis=0)
            personality_thresholds = torch.tensor(
                np.stack([p33, p67], axis=1), dtype=torch.float32
            )
        elif args.personality_binary:
            from dataset_affectai import BIG_FIVE_COLS
            p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
            personality_thresholds = torch.tensor(
                np.nanmedian(p_values, axis=0), dtype=torch.float32
            )

        train_ds = GroupAffectDataset(train_df, fold_user2idx, scalers, augment=True,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))
        val_ds   = GroupAffectDataset(val_df,   fold_user2idx, scalers, augment=False,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))
        test_ds  = GroupAffectDataset(test_df,  fold_user2idx, scalers, augment=False,
                                      device=device, summary_key_order=summary_key_order,
                                      vad_thresholds=vad_thresholds,
                                      session2idx=make_session2idx(df))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        gaze_dim  = len(sample_row["gaze_seq"].columns)  if isinstance(sample_row["gaze_seq"],  pd.DataFrame) else 9
        pupil_dim = len(sample_row["pupil_seq"].columns) if isinstance(sample_row["pupil_seq"], pd.DataFrame) else 3
        eda_dim   = len(sample_row["eda_seq"].columns)   if isinstance(sample_row["eda_seq"],   pd.DataFrame) else 5
        ppg_dim   = len(sample_row["ppg_seq"].columns)   if isinstance(sample_row["ppg_seq"],   pd.DataFrame) else 3
        imu_dim   = len(sample_row["imu_seq"].columns)   if isinstance(sample_row["imu_seq"],   pd.DataFrame) else 6

        fold_model = MuMTAffectGroupAffect(
            gaze_dim=gaze_dim, pupil_dim=pupil_dim, eda_dim=eda_dim,
            ppg_dim=ppg_dim, imu_dim=imu_dim,
            summary_dim=summary_dim, n_subjects=len(fold_user2idx),
            n_personality=5, n_emotion_classes=3,
            d_model_enc=64, d_model_fuse=128, t_out=16,
            n_tasks=N_TASKS,
            per_dim_queries=args.per_dim_queries,
            per_dim_projections=args.per_dim_projections,
            use_se_blocks=args.use_se_blocks,
            use_scaled_attention=args.use_scaled_attention,
            use_global_token=args.use_global_token,
            use_gru=args.use_gru,
            per_dim_gate=args.per_dim_gate,
            personality_classes=personality_classes,
            personality_as_input=args.personality_as_input,
        ).to(device)

        fold_dir = output_dir / f"fold_{fold_i:02d}_{held_out_subj}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_phase_cfg = PHASE_CONFIG_V15 if args.v15_training else (PHASE_CONFIG_V14 if args.v14_training else (PHASE_CONFIG_V13 if args.v13_training else PHASE_CONFIG))
        for phase in range(args.start_phase, 4):
            run_phase(phase, fold_model, train_loader, val_loader,
                      device, fold_dir, amp_scaler, class_weights,
                      freeze_transformers=freeze_transformers,
                      freeze_transformers_from_phase=freeze_transformers_from_phase,
                      label_smoothing=args.label_smoothing,
                      grad_accum_steps=args.grad_accum_steps,
                      eval_every=args.eval_every,
                      phase_config=fold_phase_cfg,
                      personality_binary=args.personality_binary,
                      personality_ternary=args.personality_ternary,
                      personality_thresholds=personality_thresholds)

        criterion_eval = MuMTAffectLoss(alpha=0.1, class_weights=class_weights,
                                        personality_binary=args.personality_binary,
                                        personality_ternary=args.personality_ternary,
                                        personality_thresholds=personality_thresholds).to(device)
        metrics = evaluate(fold_model, test_loader, criterion_eval, device)
        metrics["fold"] = fold_i
        metrics["held_out_subject"] = held_out_subj
        metrics["n_test"] = len(test_df)
        fold_results.append(metrics)

        log.info("Fold %d  subj=%s  V_F1=%.3f  A_F1=%.3f  D_F1=%.3f  P_R2=%.3f",
                 fold_i, held_out_subj, metrics["valence_f1"], metrics["arousal_f1"],
                 metrics["dominance_f1"], metrics["personality_r2_mean"])

        torch.save(fold_model.state_dict(), fold_dir / "model_final.pt")

    # ---- Aggregate ----
    log.info("=" * 60)
    log.info("LOSO-CV aggregate results (%d folds)", len(fold_results))
    for metric in ["valence_f1", "arousal_f1", "dominance_f1", "personality_r2_mean"]:
        vals = [r[metric] for r in fold_results]
        log.info("  %-25s mean=%.3f  std=%.3f", metric, np.mean(vals), np.std(vals))

    cv_df = pd.DataFrame(fold_results)
    cv_path = output_dir / "loso_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    log.info("LOSO-CV results saved to %s", cv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        log.info("GPU: %s (CUDA %s)", gpu_name, torch.version.cuda)

    # AMP grad scaler (no-op on CPU)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading pickle from %s …", args.data_path)
    df = pd.read_pickle(args.data_path)
    log.info("Loaded %d windows.", len(df))

    # --- Subject-level splits ---
    if args.stratified_split:
        log.info("Using stratified per-subject split (each subject in train/val/test).")
        train_df, val_df, test_df = split_by_subject_stratified(df, test_frac=0.15, val_frac=0.15)
    else:
        log.info("Using subject-level split (subjects in one split only).")
        train_df, val_df, test_df = split_by_subject(df, test_frac=0.15, val_frac=0.15)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    user2idx = make_user2idx(df)
    n_subjects = len(user2idx)
    log.info("Unique subjects: %d", n_subjects)

    session2idx = make_session2idx(df)
    log.info("Unique sessions: %d", len(session2idx))

    # --- VAD thresholds (data-driven binning) ---
    vad_thresholds: dict[str, tuple[float, float]] | None = None
    if args.data_driven_bins:
        vad_thresholds = compute_vad_thresholds(train_df)
        log.info("Using data-driven VAD thresholds (tertile split of training set).")
    else:
        log.info("Using fixed VAD thresholds: Low≤3, Mid 4–6, High≥7.")

    # --- Scalers ---
    scalers = fit_scalers(train_df)

    # --- Summary dim ---
    summary_dim, summary_key_order = make_summary_dim(df)
    log.info("Summary feature dim: %d", summary_dim)

    sample_row = df.iloc[0]

    # --- LOGO cross-validation mode ---
    if args.cv:
        if "group_id" not in df.columns:
            # Derive group_id from session_id (e.g. "ses-20260312_grp-07_run01" → "grp-07")
            df["group_id"] = df["session_id"].str.extract(r"(grp-\d+)")
        run_logo_cv(
            df=df,
            args=args,
            device=device,
            amp_scaler=amp_scaler,
            summary_dim=summary_dim,
            summary_key_order=summary_key_order,
            user2idx=user2idx,
            sample_row=sample_row,
            output_dir=output_dir,
            vad_thresholds=vad_thresholds,
            freeze_transformers=args.freeze_transformers,
            freeze_transformers_from_phase=args.freeze_transformers_from_phase,
        )
        return

    # --- LOSO cross-validation mode ---
    if args.loso_cv:
        run_loso_cv(
            df=df,
            args=args,
            device=device,
            amp_scaler=amp_scaler,
            summary_dim=summary_dim,
            summary_key_order=summary_key_order,
            user2idx=user2idx,
            sample_row=sample_row,
            output_dir=output_dir,
            vad_thresholds=vad_thresholds,
            freeze_transformers=args.freeze_transformers,
            freeze_transformers_from_phase=args.freeze_transformers_from_phase,
        )
        return

    # --- Class weights ---
    class_weights = None
    if args.class_weights == "auto":
        class_weights = compute_per_dim_class_weights(
            train_df, device, vad_thresholds=vad_thresholds
        )

    # --- Datasets and loaders ---
    train_ds = GroupAffectDataset(train_df, user2idx, scalers, augment=True,
                                   device=device, summary_key_order=summary_key_order,
                                   vad_thresholds=vad_thresholds,
                                   session2idx=session2idx)
    val_ds   = GroupAffectDataset(val_df,   user2idx, scalers, augment=False,
                                   device=device, summary_key_order=summary_key_order,
                                   vad_thresholds=vad_thresholds,
                                   session2idx=session2idx)
    test_ds  = GroupAffectDataset(test_df,  user2idx, scalers, augment=False,
                                   device=device, summary_key_order=summary_key_order,
                                   vad_thresholds=vad_thresholds,
                                   session2idx=session2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    # Infer actual sequence feature dims from the data
    sample_row = df.iloc[0]
    gaze_dim = len(sample_row["gaze_seq"].columns) if isinstance(sample_row["gaze_seq"], pd.DataFrame) else 9
    pupil_dim = len(sample_row["pupil_seq"].columns) if isinstance(sample_row["pupil_seq"], pd.DataFrame) else 3
    eda_dim = len(sample_row["eda_seq"].columns) if isinstance(sample_row["eda_seq"], pd.DataFrame) else 5
    ppg_dim = len(sample_row["ppg_seq"].columns) if isinstance(sample_row["ppg_seq"], pd.DataFrame) else 3
    imu_dim = len(sample_row["imu_seq"].columns) if isinstance(sample_row["imu_seq"], pd.DataFrame) else 6

    personality_classes = 3 if args.personality_ternary else (2 if args.personality_binary else 1)
    model = MuMTAffectGroupAffect(
        gaze_dim=gaze_dim,
        pupil_dim=pupil_dim,
        eda_dim=eda_dim,
        ppg_dim=ppg_dim,
        imu_dim=imu_dim,
        summary_dim=summary_dim,
        n_subjects=n_subjects,
        n_personality=5,
        n_emotion_classes=3,
        d_model_enc=64,
        d_model_fuse=128,
        t_out=16,
        n_tasks=N_TASKS,
        per_dim_queries=args.per_dim_queries,
        per_dim_projections=args.per_dim_projections,
        use_se_blocks=args.use_se_blocks,
        use_scaled_attention=args.use_scaled_attention,
        use_global_token=args.use_global_token,
        use_gru=args.use_gru,
        per_dim_gate=args.per_dim_gate,
        personality_classes=personality_classes,
        personality_as_input=args.personality_as_input,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d (%.1f K)", n_params, n_params / 1000)

    # --- Optional: load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        strict = (personality_classes == 1)  # allow mismatch when personality head changes
        missing, unexpected = model.load_state_dict(
            torch.load(args.checkpoint, map_location=device), strict=strict
        )
        if missing or unexpected:
            log.info("Checkpoint loaded (non-strict): %d missing, %d unexpected keys",
                     len(missing), len(unexpected))
        log.info("Loaded checkpoint: %s", args.checkpoint)

    # --- Phase 0: self-supervised pre-training (optional) ---
    if args.pretrain_data and Path(args.pretrain_data).exists():
        run_pretraining(
            model=model,
            pretrain_path=args.pretrain_data,
            summary_key_order=summary_key_order,
            device=device,
            scaler=amp_scaler,
            output_dir=output_dir,
            scalers=scalers,
            n_epochs=args.pretrain_epochs,
            participants_tsv=args.participants_tsv,
            w_task=args.pretrain_w_task,
            w_subject=args.pretrain_w_subject,
            w_session=args.pretrain_w_session,
            w_personality=args.pretrain_w_personality,
            w_sex=args.pretrain_w_sex,
            w_age=args.pretrain_w_age,
            w_next=args.pretrain_w_next,
            nsp_warmup_epochs=args.pretrain_nsp_warmup_epochs,
            disentangle_branches=args.pretrain_disentangle_branches,
            adv_user_on_moment=args.pretrain_adv_user_on_moment,
            adv_moment_on_user=args.pretrain_adv_moment_on_user,
            grl_lambda=args.pretrain_grl_lambda,
            normalize_targets=args.pretrain_normalize_targets,
            w_contrast=args.pretrain_w_contrast,
            grl_schedule=args.pretrain_grl_schedule,
        )
    elif args.pretrain_data:
        log.warning("--pretrain-data path not found: %s  (skipping Phase 0)", args.pretrain_data)

    # --- Compute personality thresholds for classification ---
    personality_thresholds = None
    if args.personality_ternary:
        from dataset_affectai import BIG_FIVE_COLS
        p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
        p33 = np.nanpercentile(p_values, 33.3, axis=0)
        p67 = np.nanpercentile(p_values, 66.7, axis=0)
        personality_thresholds = torch.tensor(
            np.stack([p33, p67], axis=1), dtype=torch.float32
        )  # shape (5, 2)
        log.info("Personality ternary mode: p33=%s  p67=%s",
                 p33.tolist(), p67.tolist())
    elif args.personality_binary:
        from dataset_affectai import BIG_FIVE_COLS
        p_values = train_df[BIG_FIVE_COLS].values.astype(np.float32)
        personality_thresholds = torch.tensor(
            np.nanmedian(p_values, axis=0), dtype=torch.float32
        )
        log.info("Personality binary mode: medians=%s", personality_thresholds.tolist())

    # --- Run phases 1–3 ---
    if args.v15_training:
        phase_cfg = PHASE_CONFIG_V15
    elif args.v14_training:
        phase_cfg = PHASE_CONFIG_V14
    elif args.v13_training:
        phase_cfg = PHASE_CONFIG_V13
    else:
        phase_cfg = PHASE_CONFIG
    phases_to_run = range(args.start_phase, 4)
    for phase in phases_to_run:
        run_phase(phase, model, train_loader, val_loader, device, output_dir, amp_scaler,
                  class_weights,
                  freeze_transformers=args.freeze_transformers,
                  freeze_transformers_from_phase=args.freeze_transformers_from_phase,
                  label_smoothing=args.label_smoothing,
                  grad_accum_steps=args.grad_accum_steps,
                  eval_every=args.eval_every,
                  phase_config=phase_cfg,
                  personality_binary=args.personality_binary,
                  personality_ternary=args.personality_ternary,
                  personality_thresholds=personality_thresholds)

    # --- Final test evaluation ---
    log.info("=" * 60)
    log.info("Final test evaluation")
    criterion_eval = MuMTAffectLoss(alpha=0.1, class_weights=class_weights,
                                    personality_binary=args.personality_binary,
                                    personality_ternary=args.personality_ternary,
                                    personality_thresholds=personality_thresholds).to(device)
    test_metrics = evaluate(model, test_loader, criterion_eval, device)

    trait_names = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    log.info("  Valence  macro-F1 : %.3f", test_metrics["valence_f1"])
    log.info("  Arousal  macro-F1 : %.3f", test_metrics["arousal_f1"])
    log.info("  Dominance macro-F1: %.3f", test_metrics["dominance_f1"])
    p_metric_name = "Accuracy" if (args.personality_binary or args.personality_ternary) else "R²"
    log.info("  Personality %s   : %.3f (mean)", p_metric_name, test_metrics["personality_r2_mean"])
    for name, r2 in zip(trait_names, test_metrics["personality_r2_per_trait"]):
        log.info("    %s: %s=%.3f", name, p_metric_name, r2)

    # --- Save results ---
    results_df = pd.DataFrame([test_metrics])
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Results saved to %s", results_path)

    # --- Save final model ---
    final_ckpt = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    log.info("Final model saved to %s", final_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MuMTAffect (GroupAffect-4 adaptation) — VAD + Personality."
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Path to dataset.pkl produced by pickle_generation_affectai.py."
    )
    parser.add_argument(
        "--output-dir", default="data/mumt/runs",
        help="Directory for checkpoints and results (default: data/mumt/runs)."
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="Optional path to a pre-trained checkpoint to load before training."
    )
    parser.add_argument(
        "--start-phase", type=int, default=1, choices=[1, 2, 3],
        help="Which training phase to start from (1=personality pretrain, 2=multitask, 3=finetune)."
    )
    parser.add_argument(
        "--class-weights", default="auto", choices=["auto", "none"],
        help="'auto' computes inverse-frequency weights per VAD dimension (default: auto)."
    )
    parser.add_argument(
        "--data-driven-bins", action="store_true",
        help="Use tertile-based data-driven VAD binning instead of fixed 1–3/4–6/7–9 splits."
    )
    parser.add_argument(
        "--pretrain-data", default="",
        help="Path to pretrain_dataset.pkl (Phase 0 self-supervised pretraining). "
             "If not supplied, Phase 0 is skipped."
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=80,
        help="Number of Phase 0 pretraining epochs (default: 80)."
    )
    parser.add_argument("--pretrain-w-task", type=float, default=1.0,
                        help="Phase 0 weight for task classification objective.")
    parser.add_argument("--pretrain-w-subject", type=float, default=1.0,
                        help="Phase 0 weight for subject classification objective.")
    parser.add_argument("--pretrain-w-session", type=float, default=0.5,
                        help="Phase 0 weight for session classification objective.")
    parser.add_argument("--pretrain-w-personality", type=float, default=0.5,
                        help="Phase 0 weight for personality regression objective.")
    parser.add_argument("--pretrain-w-sex", type=float, default=0.5,
                        help="Phase 0 weight for sex classification objective.")
    parser.add_argument("--pretrain-w-age", type=float, default=0.3,
                        help="Phase 0 weight for age regression objective.")
    parser.add_argument("--pretrain-w-next", type=float, default=0.5,
                        help="Phase 0 weight for next-window summary prediction objective.")
    parser.add_argument("--pretrain-nsp-warmup-epochs", type=int, default=0,
                        help="Optional warmup epochs using only next-step objective before full Phase 0.")
    parser.add_argument(
        "--pretrain-disentangle-branches",
        action="store_true",
        help="Enable branch-level adversarial disentanglement during Phase 0 pretraining.",
    )
    parser.add_argument(
        "--pretrain-adv-user-on-moment",
        type=float,
        default=0.2,
        help="Weight of adversarial penalty for user branch predicting momentary targets.",
    )
    parser.add_argument(
        "--pretrain-adv-moment-on-user",
        type=float,
        default=0.2,
        help="Weight of adversarial penalty for momentary branch predicting user-profile targets.",
    )
    parser.add_argument(
        "--pretrain-grl-lambda",
        type=float,
        default=1.0,
        help="Gradient reversal lambda for adversarial disentanglement heads.",
    )
    parser.add_argument(
        "--pretrain-normalize-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize Phase-0 regression targets (personality, age, next-summary) to reduce loss-scale imbalance.",
    )
    parser.add_argument(
        "--participants-tsv", default="data/zenodo/participants.tsv",
        help="Path to BIDS participants.tsv for age/sex labels in pretraining "
             "(default: data/zenodo/participants.tsv)."
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run Leave-One-Group-Out cross-validation instead of single split."
    )
    parser.add_argument(
        "--loso-cv", action="store_true",
        help="Run Leave-One-Subject-Out cross-validation instead of single split."
    )
    parser.add_argument(
        "--stratified-split", action="store_true",
        help="Use per-subject stratified split (each subject in train/val/test) instead of "
             "subject-level split (subjects in one split only). Recommended for better coverage."
    )
    parser.add_argument(
        "--use-gru", action="store_true",
        help="Use GRU-based modality encoders and fusion instead of Transformers."
    )
    parser.add_argument(
        "--freeze-transformers", action="store_true",
        help="Freeze modality/fusion transformer blocks during fine-tuning phases."
    )
    parser.add_argument(
        "--freeze-transformers-from-phase", type=int, default=2, choices=[1, 2, 3],
        help="First phase where transformer freezing is applied (default: 2)."
    )
    parser.add_argument(
        "--per-dim-queries", action="store_true",
        help="Use separate attention queries for each VAD dimension (4 queries instead of 2)."
    )
    parser.add_argument(
        "--per-dim-projections", action="store_true",
        help="Add per-dimension projection layers after shared emotion CNN."
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing for emotion CE loss (default: 0.0, recommended: 0.1)."
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)."
    )
    parser.add_argument(
        "--eval-every", type=int, default=5,
        help="Evaluate every N epochs (default: 5, use 1 for fine-grained checkpointing)."
    )
    parser.add_argument(
        "--v13-training", action="store_true",
        help="Use v13 training config: lower Phase 2 LR, cosine decay (no restarts), longer Phase 3."
    )
    parser.add_argument(
        "--v14-training", action="store_true",
        help="Use v14 training config: PCPE architecture with cosine_decay all phases, longer patience."
    )
    parser.add_argument(
        "--v15-training", action="store_true",
        help="Use v15 training config: end-to-end with discriminative LR, PCPE regularizer, early stopping."
    )
    parser.add_argument(
        "--personality-binary", action="store_true",
        help="Replace personality regression with binary high/low classification (median split)."
    )
    parser.add_argument(
        "--personality-ternary", action="store_true",
        help="Replace personality regression with 3-class high/medium/low classification (tertile split)."
    )
    parser.add_argument(
        "--use-se-blocks", action="store_true",
        help="Add Squeeze-and-Excitation blocks to modality encoders."
    )
    parser.add_argument(
        "--use-scaled-attention", action="store_true",
        help="Use per-query learned temperature in task attention (CTSEM timescale asymmetry)."
    )
    parser.add_argument(
        "--use-global-token", action="store_true",
        help="Append global mean token to fused sequence before task attention."
    )
    parser.add_argument(
        "--per-dim-gate", action="store_true",
        help="Use per-dimension personality gate (3-vector) instead of scalar."
    )
    parser.add_argument(
        "--pretrain-w-contrast", type=float, default=0.0,
        help="Phase 0 weight for PCPE contrastive loss on personality_ctx (default: 0.0, recommended: 1.0)."
    )
    parser.add_argument(
        "--pretrain-grl-schedule", action="store_true",
        help="Use Ganin et al. 2016 GRL lambda schedule (0→1 S-curve) instead of fixed lambda."
    )
    parser.add_argument(
        "--personality-as-input", action="store_true",
        help="Use Big Five personality traits as input features (concatenated to summary). "
             "Useful for testing if personality improves emotion prediction."
    )
    args = parser.parse_args()
    main(args)
