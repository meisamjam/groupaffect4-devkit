"""train_v2_affectai.py

Dual-Stream MuMTAffect v2 training script.

Architecture improvements over v1:
  - User Profile Stream (self-attention) for stable trait-level objectives
  - Cognitive State Stream (self-attn + cross-attn from profile) for momentary objectives
  - Asymmetric information flow: state sees profile, profile does NOT see state
  - New SSL objectives: temporal delta prediction, masked modality reconstruction
  - No adversarial gradient reversal (architectural separation instead)

Phases:
  Phase 0 – Dual-stream self-supervised pretraining (dense sliding windows)
  Phase 1 – Personality pretraining    (alpha=1.0)
  Phase 2 – Joint multitask training    (alpha=0.3)
  Phase 3 – Fine-tuning                 (alpha=0.1)

Usage:
  python tools/mumt/train_v2_affectai.py \\
      --data-path data/mumt/dataset.pkl \\
      --pretrain-data data/mumt/pretrain_dataset.pkl \\
      --participants-tsv data/zenodo/participants.tsv \\
      --output-dir data/mumt/runs_v11_dualstream \\
      --class-weights auto --data-driven-bins
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
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset_affectai import (
    BIG_FIVE_COLS,
    GroupAffectDataset,
    N_TASKS,
    bin_vad_adaptive,
    build_summary_key_order,
    make_session2idx,
    make_user2idx,
    split_by_subject,
)
from dataset_v2_affectai import PretrainDatasetV2
from model_affectai import MuMTAffectLoss
from model_v2_affectai import (
    DualStreamPretrainingHeads,
    DualStreamPretrainingLoss,
    MuMTAffectV2,
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
    1: dict(alpha=1.0, epochs=60, lr_base=1e-4, weight_decay=5e-3, patience=10,
            scheduler="exp", lr_decay=0.98),
    2: dict(alpha=0.3, epochs=120, lr_base=2e-4, weight_decay=5e-3, patience=15,
            scheduler="cosine", lr_decay=0.95, T_0=30, T_mult=2),
    3: dict(alpha=0.1, epochs=40, lr_base=5e-5, weight_decay=5e-3, patience=10,
            scheduler="exp", lr_decay=0.98),
}

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(labels: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
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
    scalers = {}
    for key, col in [("gaze", "gaze_seq"), ("pupil", "pupil_seq"),
                     ("eda", "eda_seq"), ("ppg", "ppg_seq"), ("imu", "imu_seq")]:
        arrays = [df.values for df in train_df[col] if isinstance(df, pd.DataFrame)]
        if arrays:
            stacked = np.vstack(arrays).astype(np.float32)
            sc = StandardScaler()
            sc.fit(stacked)
            scalers[key] = sc
    return scalers


def make_summary_dim(df: pd.DataFrame) -> tuple[int, list[str]]:
    key_order = build_summary_key_order(df)
    return len(key_order), key_order


def compute_vad_thresholds(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute 33rd and 67th percentile thresholds for VAD columns from training data.

    Note: If percentiles are NaN or produce missing classes, falls back to fixed bins
    and logs a warning.
    """
    thresholds: dict[str, tuple[float, float]] = {}
    for col in ("valence", "arousal", "dominance"):
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy()
        
        if len(vals) == 0:
            log.warning("No valid values for %s; using fixed bins", col)
            thresholds[col] = (4.0, 6.0)
            continue
        
        t1 = float(np.percentile(vals, 33))
        t2 = float(np.percentile(vals, 67))
        
        # Validate: ensure thresholds are not NaN
        if np.isnan(t1) or np.isnan(t2):
            log.warning("Percentiles for %s are NaN; using fixed bins (4.0, 6.0)", col)
            thresholds[col] = (4.0, 6.0)
            continue
        
        thresholds[col] = (t1, t2)
        log.info("VAD thresholds %s: p33=%.2f  p67=%.2f", col, t1, t2)
    return thresholds


def set_streams_trainable(model: MuMTAffectV2, trainable: bool) -> None:
    """Freeze/unfreeze the dual-stream transformer modules."""
    stream_modules = [model.profile_stream, model.state_stream]
    for module in stream_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def set_encoders_trainable(model: MuMTAffectV2, trainable: bool) -> None:
    """Freeze/unfreeze modality encoders and fusion transformer."""
    encoder_modules = [
        model.gaze_encoder.transformer,
        model.pupil_encoder.transformer,
        model.eda_encoder.transformer,
        model.ppg_encoder.transformer,
        model.imu_encoder.transformer,
        model.fusion.transformer,
    ]
    for module in encoder_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        # Evaluate in fp32 to avoid overflow in personality branch
        outputs = model(gaze.float(), pupil.float(), eda.float(),
                        ppg.float(), imu.float(),
                        summary.float(), user_ids,
                        task_onehot=task_oh.float())
        losses = criterion(outputs, emotions, personality.float())
        total_loss += losses["total"].item()
        n_batches += 1

        all_v_true.extend(emotions[:, 0].cpu().numpy())
        all_a_true.extend(emotions[:, 1].cpu().numpy())
        all_d_true.extend(emotions[:, 2].cpu().numpy())
        all_v_pred.extend(outputs["valence_logits"].argmax(-1).cpu().numpy())
        all_a_pred.extend(outputs["arousal_logits"].argmax(-1).cpu().numpy())
        all_d_pred.extend(outputs["dominance_logits"].argmax(-1).cpu().numpy())
        all_p_true.append(personality.float().cpu().numpy())
        all_p_pred.append(outputs["personality_pred"].cpu().numpy())

    p_true = np.vstack(all_p_true)
    p_pred = np.vstack(all_p_pred)
    # Guard against NaN from AMP overflow
    if np.any(np.isnan(p_pred)):
        p_pred = np.nan_to_num(p_pred, nan=0.0, posinf=0.0, neginf=0.0)
    r2_scores_list = [r2_score(p_true[:, i], p_pred[:, i]) for i in range(p_true.shape[1])]

    labels = list(range(3))
    return {
        "loss": total_loss / max(n_batches, 1),
        "valence_f1": f1_score(all_v_true, all_v_pred, average="macro", labels=labels, zero_division=0),
        "arousal_f1": f1_score(all_a_true, all_a_pred, average="macro", labels=labels, zero_division=0),
        "dominance_f1": f1_score(all_d_true, all_d_pred, average="macro", labels=labels, zero_division=0),
        "personality_r2_mean": float(np.mean(r2_scores_list)),
        "personality_r2_per_trait": r2_scores_list,
    }


# ---------------------------------------------------------------------------
# Phase 0 — Dual-stream self-supervised pretraining
# ---------------------------------------------------------------------------

def run_pretraining_v2(
    model: MuMTAffectV2,
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
    w_subject: float = 1.0,
    w_personality: float = 0.5,
    w_sex: float = 0.5,
    w_age: float = 0.3,
    w_task: float = 1.0,
    w_session: float = 0.5,
    w_next: float = 0.5,
    w_delta: float = 0.3,
    w_recon: float = 0.3,
    warmup_epochs: int = 5,
    normalize_targets: bool = True,
    mask_prob: float = 0.15,
) -> None:
    """Phase 0: dual-stream self-supervised pretraining."""
    log.info("=" * 60)
    log.info("Phase 0 — Dual-stream pretraining  |  epochs=%d  |  lr=%.0e", n_epochs, lr)
    log.info("Loading pretrain pickle: %s", pretrain_path)

    pt_df = pd.read_pickle(pretrain_path)
    log.info("Pretrain windows: %d  |  subjects: %d  |  sessions: %d",
             len(pt_df), pt_df["subject_id"].nunique(), pt_df["session_id"].nunique())

    pt_user2idx = make_user2idx(pt_df)
    pt_session2idx = make_session2idx(pt_df)
    n_subjects_pt = len(pt_user2idx)
    n_sessions_pt = len(pt_session2idx)

    ds = PretrainDatasetV2(
        pt_df, pt_user2idx, pt_session2idx, summary_key_order,
        device=device, augment=False,
        participants_tsv=participants_tsv,
        scalers=scalers,
        normalize_targets=normalize_targets,
        mask_prob=mask_prob,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, num_workers=0)

    heads = DualStreamPretrainingHeads(
        d_fuse=model.d_model_fuse,
        n_tasks=N_TASKS,
        n_subjects=n_subjects_pt,
        n_sessions=n_sessions_pt,
        n_personality=5,
        summary_dim=len(summary_key_order),
    ).to(device)

    criterion = DualStreamPretrainingLoss(
        w_subject=w_subject, w_personality=w_personality,
        w_sex=w_sex, w_age=w_age,
        w_task=w_task, w_session=w_session,
        w_next=w_next, w_delta=w_delta, w_recon=w_recon,
    ).to(device)

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

        # Warmup: first few epochs only use profile objectives to bootstrap embeddings
        is_warmup = epoch <= warmup_epochs

        epoch_losses = {
            "total": 0.0, "profile": 0.0, "state": 0.0,
            "subj": 0.0, "per": 0.0, "sex": 0.0, "age": 0.0,
            "task": 0.0, "ses": 0.0, "next": 0.0, "delta": 0.0, "recon": 0.0,
        }

        for batch in loader:
            (gaze, pupil, eda, ppg, imu,
             summary, subject_idx, session_idx, task_idx,
             personality, sex_label, age, next_summary, has_next,
             delta_summary, has_delta, recon_target, recon_mask) = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(
                    gaze.float(), pupil.float(), eda.float(),
                    ppg.float(), imu.float(),
                    summary.float(), subject_idx,
                )
                preds = heads(out["profile_pooled"], out["state_pooled"])

                # During warmup, zero out state stream weights
                if is_warmup:
                    old_w_task = criterion.w_task
                    old_w_ses = criterion.w_session
                    old_w_next = criterion.w_next
                    old_w_delta = criterion.w_delta
                    old_w_recon = criterion.w_recon
                    criterion.w_task = 0.0
                    criterion.w_session = 0.0
                    criterion.w_next = 0.0
                    criterion.w_delta = 0.0
                    criterion.w_recon = 0.0

                losses = criterion(
                    preds, task_idx, subject_idx, session_idx,
                    personality, sex_label, age, next_summary, has_next,
                    delta_summary, has_delta,
                    recon_target.view(recon_target.size(0), -1) if recon_target.dim() == 3 else recon_target,
                    recon_mask,
                )

                if is_warmup:
                    criterion.w_task = old_w_task
                    criterion.w_session = old_w_ses
                    criterion.w_next = old_w_next
                    criterion.w_delta = old_w_delta
                    criterion.w_recon = old_w_recon

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_losses["total"] += losses["total"].item()
            epoch_losses["profile"] += losses["profile_total"].item()
            epoch_losses["state"] += losses["state_total"].item()
            epoch_losses["subj"] += losses["subject"].item()
            epoch_losses["per"] += losses["personality"].item()
            epoch_losses["sex"] += losses["sex"].item()
            epoch_losses["age"] += losses["age"].item()
            epoch_losses["task"] += losses["task"].item()
            epoch_losses["ses"] += losses["session"].item()
            epoch_losses["next"] += losses["next"].item()
            epoch_losses["delta"] += losses["delta"].item()
            epoch_losses["recon"] += losses["recon"].item()

        scheduler.step(epoch - 1)
        n = max(len(loader), 1)
        for k in epoch_losses:
            epoch_losses[k] /= n

        log.info(
            "Pretrain %3d/%d | total=%.4f | profile=%.3f (subj=%.3f per=%.3f sex=%.3f age=%.3f) | "
            "state=%.3f (task=%.3f ses=%.3f next=%.3f delta=%.3f recon=%.3f)%s",
            epoch, n_epochs,
            epoch_losses["total"],
            epoch_losses["profile"], epoch_losses["subj"], epoch_losses["per"],
            epoch_losses["sex"], epoch_losses["age"],
            epoch_losses["state"], epoch_losses["task"], epoch_losses["ses"],
            epoch_losses["next"], epoch_losses["delta"], epoch_losses["recon"],
            " [warmup]" if is_warmup else "",
        )

        # Only save best after warmup completes
        if not is_warmup and epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save(model.state_dict(), ckpt_path)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log.info("Phase 0 complete — backbone saved to %s (best_loss=%.4f)", ckpt_path, best_loss)
    else:
        log.info("Phase 0 complete (no checkpoint saved — all epochs were warmup).")


# ---------------------------------------------------------------------------
# Training phase (phases 1–3, same as v1 but using MuMTAffectV2)
# ---------------------------------------------------------------------------

def run_phase(
    phase: int,
    model: MuMTAffectV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    scaler: torch.cuda.amp.GradScaler,
    class_weights: dict[str, torch.Tensor] | None = None,
    freeze_encoders: bool = False,
    freeze_encoders_from_phase: int = 2,
) -> None:
    cfg = PHASE_CONFIG[phase]
    log.info("=" * 60)
    log.info("Phase %d  |  alpha=%.2f  |  epochs=%d  |  lr=%.0e  |  wd=%.0e",
             phase, cfg["alpha"], cfg["epochs"], cfg["lr_base"], cfg["weight_decay"])
    log.info("=" * 60)

    criterion = MuMTAffectLoss(alpha=cfg["alpha"], class_weights=class_weights).to(device)

    should_freeze = freeze_encoders and phase >= freeze_encoders_from_phase
    set_encoders_trainable(model, trainable=not should_freeze)
    if should_freeze:
        log.info("Encoder/fusion transformers frozen for phase %d.", phase)
    else:
        log.info("All transformer blocks trainable for phase %d.", phase)

    log.info("Trainable parameters this phase: %d", count_trainable_parameters(model))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr_base"], weight_decay=cfg["weight_decay"],
    )

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg["T_0"], T_mult=cfg.get("T_mult", 1), eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["lr_decay"])

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    patience_counter = 0
    best_ckpt = output_dir / f"checkpoint_phase{phase}.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            gaze, pupil, eda, ppg, imu, personality, emotions, user_ids, summary, _, task_oh, _ses = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]
            optimizer.zero_grad()
            # No AMP for downstream phases (tiny dataset, avoids fp16 overflow)
            outputs = model(gaze.float(), pupil.float(), eda.float(),
                            ppg.float(), imu.float(),
                            summary.float(), user_ids,
                            personality_gt=personality.float(),
                            task_onehot=task_oh.float())
            losses = criterion(outputs, emotions, personality.float())
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                continue  # skip corrupted step
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += losses["total"].item()

        if cfg["scheduler"] == "cosine":
            scheduler.step(epoch - 1)
        else:
            scheduler.step()
        train_loss /= max(len(train_loader), 1)

        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            val_metrics = evaluate(model, val_loader, criterion, device)
            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "V_f1=%.3f A_f1=%.3f D_f1=%.3f | P_R2=%.3f",
                epoch, cfg["epochs"], train_loss, val_metrics["loss"],
                val_metrics["valence_f1"], val_metrics["arousal_f1"],
                val_metrics["dominance_f1"], val_metrics["personality_r2_mean"],
            )
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
                patience_counter += 5
                if patience_counter >= cfg["patience"] * 5:
                    log.info("Early stopping at epoch %d.", epoch)
                    break

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        log.info("Loaded best checkpoint from %s.", best_ckpt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log.info("GPU: %s (CUDA %s)", torch.cuda.get_device_name(0), torch.version.cuda)

    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading pickle from %s …", args.data_path)
    df = pd.read_pickle(args.data_path)
    log.info("Loaded %d windows.", len(df))

    train_df, val_df, test_df = split_by_subject(df, test_frac=0.15, val_frac=0.15)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    user2idx = make_user2idx(df)
    n_subjects = len(user2idx)
    log.info("Unique subjects: %d", n_subjects)

    session2idx = make_session2idx(df)
    log.info("Unique sessions: %d", len(session2idx))

    # --- VAD thresholds ---
    vad_thresholds: dict[str, tuple[float, float]] | None = None
    if args.data_driven_bins:
        vad_thresholds = compute_vad_thresholds(train_df)

    # --- Scalers ---
    scalers = fit_scalers(train_df)

    # --- Summary dim ---
    summary_dim, summary_key_order = make_summary_dim(df)
    log.info("Summary feature dim: %d", summary_dim)

    # --- Class weights ---
    class_weights = None
    if args.class_weights == "auto":
        class_weights = compute_per_dim_class_weights(
            train_df, device, vad_thresholds=vad_thresholds
        )

    # --- Datasets ---
    train_ds = GroupAffectDataset(train_df, user2idx, scalers, augment=True,
                                  device=device, summary_key_order=summary_key_order,
                                  vad_thresholds=vad_thresholds, session2idx=session2idx)
    val_ds = GroupAffectDataset(val_df, user2idx, scalers, augment=False,
                                device=device, summary_key_order=summary_key_order,
                                vad_thresholds=vad_thresholds, session2idx=session2idx)
    test_ds = GroupAffectDataset(test_df, user2idx, scalers, augment=False,
                                 device=device, summary_key_order=summary_key_order,
                                 vad_thresholds=vad_thresholds, session2idx=session2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    sample_row = df.iloc[0]
    gaze_dim = len(sample_row["gaze_seq"].columns) if isinstance(sample_row["gaze_seq"], pd.DataFrame) else 9
    pupil_dim = len(sample_row["pupil_seq"].columns) if isinstance(sample_row["pupil_seq"], pd.DataFrame) else 3
    eda_dim = len(sample_row["eda_seq"].columns) if isinstance(sample_row["eda_seq"], pd.DataFrame) else 5
    ppg_dim = len(sample_row["ppg_seq"].columns) if isinstance(sample_row["ppg_seq"], pd.DataFrame) else 3
    imu_dim = len(sample_row["imu_seq"].columns) if isinstance(sample_row["imu_seq"], pd.DataFrame) else 6

    model = MuMTAffectV2(
        gaze_dim=gaze_dim, pupil_dim=pupil_dim, eda_dim=eda_dim,
        ppg_dim=ppg_dim, imu_dim=imu_dim,
        summary_dim=summary_dim, n_subjects=n_subjects,
        n_personality=5, n_emotion_classes=3,
        d_model_enc=64, d_model_fuse=128, t_out=16,
        n_tasks=N_TASKS,
        stream_n_layers=args.stream_layers,
        stream_nhead=4,
        stream_ffn_dim=512,
        stream_dropout=0.2,
        use_direct_emotion_attn=args.use_direct_emotion_attn,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d (%.1f K)", n_params, n_params / 1000)

    # --- Phase 0: dual-stream pretraining ---
    phase0_ckpt = output_dir / "checkpoint_phase0.pt"
    if args.resume_phase0 and phase0_ckpt.exists():
        # strict=False allows new params (e.g. emotion_attn_query) to keep random init
        missing, unexpected = model.load_state_dict(
            torch.load(phase0_ckpt, map_location=device), strict=False
        )
        if missing:
            log.info("Phase 0 load — new params (random init): %s", missing)
        log.info("Resumed from existing Phase 0 checkpoint: %s", phase0_ckpt)
    elif args.pretrain_data and Path(args.pretrain_data).exists():
        run_pretraining_v2(
            model=model,
            pretrain_path=args.pretrain_data,
            summary_key_order=summary_key_order,
            device=device,
            scaler=amp_scaler,
            output_dir=output_dir,
            scalers=scalers,
            n_epochs=args.pretrain_epochs,
            participants_tsv=args.participants_tsv,
            w_subject=args.pretrain_w_subject,
            w_personality=args.pretrain_w_personality,
            w_sex=args.pretrain_w_sex,
            w_age=args.pretrain_w_age,
            w_task=args.pretrain_w_task,
            w_session=args.pretrain_w_session,
            w_next=args.pretrain_w_next,
            w_delta=args.pretrain_w_delta,
            w_recon=args.pretrain_w_recon,
            warmup_epochs=args.pretrain_warmup_epochs,
            normalize_targets=args.pretrain_normalize_targets,
            mask_prob=args.pretrain_mask_prob,
        )
    elif args.pretrain_data:
        log.warning("--pretrain-data path not found: %s  (skipping Phase 0)", args.pretrain_data)

    # --- Phases 1–3 ---
    if args.phase1_alpha is not None:
        PHASE_CONFIG[1]["alpha"] = args.phase1_alpha
        log.info("Phase 1 alpha overridden to %.2f", args.phase1_alpha)

    for phase in range(args.start_phase, 4):
        run_phase(phase, model, train_loader, val_loader, device, output_dir, amp_scaler,
                  class_weights,
                  freeze_encoders=args.freeze_encoders,
                  freeze_encoders_from_phase=args.freeze_encoders_from_phase)

    # --- Final test evaluation ---
    log.info("=" * 60)
    log.info("Final test evaluation")
    criterion_eval = MuMTAffectLoss(alpha=0.1, class_weights=class_weights).to(device)
    test_metrics = evaluate(model, test_loader, criterion_eval, device)

    trait_names = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    log.info("  Valence  macro-F1 : %.3f", test_metrics["valence_f1"])
    log.info("  Arousal  macro-F1 : %.3f", test_metrics["arousal_f1"])
    log.info("  Dominance macro-F1: %.3f", test_metrics["dominance_f1"])
    log.info("  Personality R²    : %.3f (mean)", test_metrics["personality_r2_mean"])
    for name, r2 in zip(trait_names, test_metrics["personality_r2_per_trait"]):
        log.info("    %s: R²=%.3f", name, r2)

    # --- Save results ---
    results_df = pd.DataFrame([test_metrics])
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Results saved to %s", results_path)

    # --- Save final model ---
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    log.info("Final model saved to %s", output_dir / "model_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MuMTAffect v2 (dual-stream) — GroupAffect-4."
    )
    parser.add_argument("--data-path", required=True, help="Path to dataset.pkl.")
    parser.add_argument("--output-dir", default="data/mumt/runs_v11_dualstream",
                        help="Output directory.")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--class-weights", default="auto", choices=["auto", "none"])
    parser.add_argument("--data-driven-bins", action="store_true")

    # Pretraining
    parser.add_argument("--pretrain-data", default="")
    parser.add_argument("--resume-phase0", action="store_true",
                        help="Load existing phase0 checkpoint instead of rerunning pretraining.")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pretrain-warmup-epochs", type=int, default=5)
    parser.add_argument("--pretrain-w-subject", type=float, default=1.0)
    parser.add_argument("--pretrain-w-personality", type=float, default=0.5)
    parser.add_argument("--pretrain-w-sex", type=float, default=0.5)
    parser.add_argument("--pretrain-w-age", type=float, default=0.3)
    parser.add_argument("--pretrain-w-task", type=float, default=1.0)
    parser.add_argument("--pretrain-w-session", type=float, default=0.5)
    parser.add_argument("--pretrain-w-next", type=float, default=0.5)
    parser.add_argument("--pretrain-w-delta", type=float, default=0.3)
    parser.add_argument("--pretrain-w-recon", type=float, default=0.3)
    parser.add_argument("--pretrain-mask-prob", type=float, default=0.15)
    parser.add_argument("--pretrain-normalize-targets",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--participants-tsv", default="data/zenodo/participants.tsv")

    # Architecture
    parser.add_argument("--stream-layers", type=int, default=2,
                        help="Number of transformer layers per stream (default: 2).")
    parser.add_argument("--use-direct-emotion-attn", action="store_true",
                        help="v12: add direct attention bypass for emotion branch.")
    parser.add_argument("--phase1-alpha", type=float, default=None,
                        help="Override Phase 1 alpha (default 1.0). Use 0.8 for v12.")

    # Freezing
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--freeze-encoders-from-phase", type=int, default=2, choices=[1, 2, 3])

    args = parser.parse_args()
    main(args)
