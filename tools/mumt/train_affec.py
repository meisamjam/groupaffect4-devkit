"""train_affec.py

Apply GroupAffect-4 neural models (MLPNet / PoolNet / ConvNet) to the AFFEC
individual FER dataset.

Protocol: within-subject trial-level split (every participant in train/val/test)
  - 60 % train · 20 % val · 20 % test, sorted by task order then onset.
  - T0 baseline normalisation available via --t0-norm flag.

Key adaptations from GroupAffect-4 (train_simple.py)
------------------------------------------------------
| Aspect           | GroupAffect-4          | AFFEC                      |
|------------------|------------------------|----------------------------|
| Seq length       | 400 timesteps          | 200 timesteps              |
| Gaze channels    | 9 (Tobii Pro G3)       | 6 (Gazepoint GP3 HD)       |
| EDA channels     | 5 (w/ HR)              | 4 (no HR; Shimmer3 GSR+)   |
| PPG channels     | 3 (EmotiBit IR/R/G)    | 0 → zero-padded to 3       |
| VAD dims         | V, A, D                | V, A only (D masked)       |
| BFI scale        | per-item 1-5           | item-sum ÷ 40 → [0,1]      |
| Split            | subject-level held-out | within-subject 60/20/20    |

Usage
-----
  # MLPNet (summary features only — fastest)
  python tools/mumt/train_affec.py --dataset data/mumt/dataset_affec.pkl

  # PoolNet (global pool over sequences)
  python tools/mumt/train_affec.py --dataset data/mumt/dataset_affec.pkl --arch pool

  # ConvNet (1-D CNN over sequences)
  python tools/mumt/train_affec.py --dataset data/mumt/dataset_affec.pkl --arch conv

  # All architectures in one run with T0 normalisation
  python tools/mumt/train_affec.py --dataset data/mumt/dataset_affec.pkl \\
      --arch all --t0-norm --epochs 150
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import BIG_FIVE_COLS
from dataset_affec_torch import (
    AFFEC_MODALITY_DIMS,
    AFFEC_TASK_LABELS,
    AFFECDataset,
    build_affec_summary_key_order,
    make_affec_datasets,
)
from model_affectai import MuMTAffectLoss
from model_simple import build_simple_model
from baselines_affec import (
    within_subject_split,
    compute_thresholds,
    compute_t0_means,
    build_X_enhanced,
)
from train_simple import SequenceScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VAD_DIMS = ["valence", "arousal"]   # AFFEC has no dominance
N_AFFEC_TASKS = len(AFFEC_TASK_LABELS)


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    total: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def _fn(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _fn)


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_arch(
    arch: str,
    train_ds: AFFECDataset,
    val_ds:   AFFECDataset,
    test_ds:  AFFECDataset,
    summary_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Train a single architecture; return {dim: macro_f1} on test set."""

    log.info("\n══ arch=%s  summary_dim=%d ══", arch.upper(), summary_dim)

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_simple_model(
        arch=arch,
        summary_dim=summary_dim,
        gaze_dim=AFFEC_MODALITY_DIMS["gaze"],
        pupil_dim=AFFEC_MODALITY_DIMS["pupil"],
        eda_dim=AFFEC_MODALITY_DIMS["eda"],
        ppg_dim=AFFEC_MODALITY_DIMS["ppg"],   # 3 (zero-padded)
        imu_dim=AFFEC_MODALITY_DIMS["imu"],
        hidden=args.d_fuse,
        dropout=args.dropout,
        n_personality=5,
        bfi_mode="none",           # BFI available but disabled by default
        task_dim=N_AFFEC_TASKS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Parameters: %d", n_params)

    # ── Data loaders ────────────────────────────────────────────────────────
    tr_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                           num_workers=0, drop_last=True)
    va_loader = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Loss (with NaN-masking for D=-1) ────────────────────────────────────
    loss_fn = MuMTAffectLoss(label_smoothing=args.label_smooth)

    # ── Optimiser ───────────────────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = _cosine_schedule(opt, warmup=args.warmup, total=args.epochs)

    best_val  = -1.0
    best_ckpt = None
    patience_count = 0

    # ── Training epochs ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tr_loader:
            gaze  = batch["gaze_seq"].to(device)   # (B, T, 6)
            pupil = batch["pupil_seq"].to(device)   # (B, T, 3)
            eda   = batch["eda_seq"].to(device)     # (B, T, 4)
            ppg   = batch["ppg_seq"].to(device)     # (B, T, 3) — zeros
            imu   = batch["imu_seq"].to(device)     # (B, T, 6)
            summ  = batch["summary"].to(device)     # (B, summary_dim)
            pers  = batch["personality"].to(device) # (B, 5)
            targs = batch["emotion_binned"].to(device)  # (B, 3)

            out = model(
                gaze_seq=gaze, pupil_seq=pupil, eda_seq=eda,
                ppg_seq=ppg, imu_seq=imu, summary=summ,
                personality=pers,
            )
            loss_dict = loss_fn(out, targs, batch["personality"].to(device))
            loss = loss_dict["total"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_f1 = _eval_f1(model, va_loader, device)
            mean_val = float(np.mean(list(val_f1.values())))
            if mean_val > best_val:
                best_val  = mean_val
                best_ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += args.eval_every
            if args.patience > 0 and patience_count >= args.patience:
                log.info("  Early stop at epoch %d  best_val=%.3f", epoch, best_val)
                break
            if epoch % (args.eval_every * 5) == 0:
                log.info("  epoch %3d  val_mean=%.3f  best=%.3f", epoch, mean_val, best_val)

    # ── Test with best checkpoint ────────────────────────────────────────────
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt)
    test_f1 = _eval_f1(model, te_loader, device)
    log.info("  Test  V=%.3f  A=%.3f  mean=%.3f",
             test_f1.get("valence", 0), test_f1.get("arousal", 0),
             float(np.mean(list(test_f1.values()))))
    return test_f1


def _eval_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return per-dim macro-F1 on loader (ignores dim if all labels are -1)."""
    model.eval()
    all_preds: list[list[int]] = [[], [], []]
    all_true:  list[list[int]] = [[], [], []]
    DIM_NAMES = ["valence", "arousal", "dominance"]

    _LOGIT_KEYS = ["valence_logits", "arousal_logits", "dominance_logits"]
    with torch.no_grad():
        for batch in loader:
            out = model(
                gaze_seq  = batch["gaze_seq"].to(device),
                pupil_seq = batch["pupil_seq"].to(device),
                eda_seq   = batch["eda_seq"].to(device),
                ppg_seq   = batch["ppg_seq"].to(device),
                imu_seq   = batch["imu_seq"].to(device),
                summary   = batch["summary"].to(device),
                personality = batch["personality"].to(device),
            )
            targs = batch["emotion_binned"]   # (B, 3)
            for d_i, key in enumerate(_LOGIT_KEYS):
                preds = out[key].argmax(dim=-1).cpu().numpy()
                true  = targs[:, d_i].numpy()
                mask  = true >= 0
                all_preds[d_i].extend(preds[mask].tolist())
                all_true[d_i].extend(true[mask].tolist())

    result: dict[str, float] = {}
    for d_i, dim in enumerate(DIM_NAMES[:2]):   # V + A only
        if len(all_true[d_i]) == 0:
            continue
        result[dim] = float(f1_score(all_true[d_i], all_preds[d_i],
                                     average="macro", zero_division=0))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GA4-style neural models on the AFFEC dataset."
    )
    parser.add_argument("--dataset",   default="data/mumt/dataset_affec.pkl")
    parser.add_argument("--arch",      default="mlp",
                        choices=["mlp", "pool", "conv", "all"],
                        help="Architecture. 'all' trains all three sequentially.")
    parser.add_argument("--t0-norm",   action="store_true",
                        help="Subtract per-participant T0 mean from summary features.")
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--wd",        type=float, default=1e-3)
    parser.add_argument("--warmup",    type=int,   default=10)
    parser.add_argument("--patience",  type=int,   default=40)
    parser.add_argument("--eval-every",type=int,   default=5)
    parser.add_argument("--d-fuse",    type=int,   default=128)
    parser.add_argument("--dropout",   type=float, default=0.4)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--augment",   action="store_true")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ────────────────────────────────────────────────────────────
    import pickle
    log.info("Loading %s", args.dataset)
    df = pickle.load(open(args.dataset, "rb"))
    log.info("Dataset: %d trials, %d subjects", len(df), df["subject_id"].nunique())

    # ── Split ────────────────────────────────────────────────────────────────
    tr_df, va_df, te_df = within_subject_split(df)
    log.info("Split: train=%d  val=%d  test=%d", len(tr_df), len(va_df), len(te_df))
    log.info("All 72 subjects in test: %s", te_df["subject_id"].nunique() == df["subject_id"].nunique())

    # ── Thresholds ───────────────────────────────────────────────────────────
    thresh = compute_thresholds(tr_df)
    thresh["dominance"] = (3.0, 6.0)   # dummy — all labels are NaN/masked
    log.info("Thresholds: %s", thresh)

    # ── Optional T0 normalisation ────────────────────────────────────────────
    if args.t0_norm:
        log.info("T0 normalisation: computing participant baselines from T0 train trials...")
        t0_means = compute_t0_means(tr_df)
        # Rewrite feature dicts in-place in each split copy
        for split_df in [tr_df, va_df, te_df]:
            for idx, row in split_df.iterrows():
                subj_t0 = t0_means.get(str(row["subject_id"]), {})
                for feat_col in ["gaze_features", "pupil_features", "eda_features", "imu_features"]:
                    fd = row.get(feat_col, {})
                    if isinstance(fd, dict) and subj_t0:
                        new_fd = {
                            k: v - subj_t0[k] if k in subj_t0 and not np.isnan(float(subj_t0[k])) else v
                            for k, v in fd.items()
                        }
                        split_df.at[idx, feat_col] = new_fd
        log.info("T0 normalisation applied.")

    # ── Build datasets ───────────────────────────────────────────────────────
    tr_ds, va_ds, te_ds = make_affec_datasets(
        tr_df, va_df, te_df, thresh, augment=args.augment
    )
    summary_dim = len(tr_ds.summary_key_order)
    log.info("Summary dim: %d", summary_dim)

    # ── Train ────────────────────────────────────────────────────────────────
    archs = ["mlp", "pool", "conv"] if args.arch == "all" else [args.arch]
    results: dict[str, dict[str, float]] = {}

    for arch in archs:
        results[arch] = train_one_arch(
            arch, tr_ds, va_ds, te_ds, summary_dim, args, device
        )

    # ── Summary table ────────────────────────────────────────────────────────
    log.info("\n\n══ AFFEC Neural Model Comparison (macro-F1, 3-class, within-subject split) ══")
    log.info("%-12s  %-8s  %-8s  %-8s", "Architecture", "Valence", "Arousal", "Mean")
    log.info("-" * 46)
    for arch, f1s in results.items():
        v = f1s.get("valence", float("nan"))
        a = f1s.get("arousal", float("nan"))
        m = float(np.nanmean([v, a]))
        log.info("%-12s  %.3f     %.3f     %.3f", arch.upper(), v, a, m)

    log.info("\nNote: SVM-A0 baseline (Protocol B, raw) = 0.400 mean macro-F1 for reference.")


if __name__ == "__main__":
    main()
