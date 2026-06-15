"""bootstrap_ci.py

Bootstrap 95% confidence intervals for VAD macro-F1 on the T3 test split.

Trains the model once, collects per-window (pred, true) pairs for V/A/D,
then resamples the test set 1000 times to estimate CI.

Usage
-----
  python tools/mumt/bootstrap_ci.py \\
      --dataset data/mumt/dataset_15s.pkl \\
      --augmented-pool data/mumt/augmented_pool.pkl \\
      --aug-frac 0.3 --arch mlp --bfi-mode none
  python tools/mumt/bootstrap_ci.py \\
      --dataset data/mumt/dataset_15s.pkl \\
      --augmented-pool data/mumt/augmented_pool.pkl \\
      --aug-frac 0.3 --arch mlp --bfi-mode perdim
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse helpers from train_simple
from train_simple import (
    VADDataset, AugSoftDataset, SoftVADLoss,
    fit_scalers, compute_tertile_thresholds, compute_class_weights,
    build_model, build_scheduler, train_epoch, evaluate,
    task_split, bin_vad_from_thresholds,
)
from dataset_affectai import (
    build_summary_key_order, make_user2idx, make_session2idx,
)

VAD_DIMS = ["valence", "arousal", "dominance"]


def collect_test_predictions(
    model, loader, device, thresholds
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-dim (preds, labels) arrays over the full test set."""
    model.eval()
    all_preds = {d: [] for d in VAD_DIMS}
    all_labels = {d: [] for d in VAD_DIMS}
    with torch.no_grad():
        for batch in loader:
            (gaze, pupil, eda, ppg, imu, personality,
             emotion_bins, uid, summary, sex, task, sid,
             vad_soft, vad_weight) = batch
            out = model(
                gaze_seq=gaze.float(), pupil_seq=pupil.float(),
                eda_seq=eda.float(), ppg_seq=ppg.float(),
                imu_seq=imu.float(), summary=summary.float(),
                personality=personality.float(),
                user_ids=uid, task_onehot=task.float(),
            )
            for i, dim in enumerate(VAD_DIMS):
                preds = out[f"{dim}_logits"].argmax(dim=-1).cpu().numpy()
                lbls  = emotion_bins[:, i].cpu().numpy()
                all_preds[dim].extend(preds.tolist())
                all_labels[dim].extend(lbls.tolist())
    return {
        d: (np.array(all_preds[d]), np.array(all_labels[d]))
        for d in VAD_DIMS
    }


def bootstrap_f1(preds, labels, n_boot=1000, seed=42) -> tuple[float, float, float]:
    """Return (point_estimate, lower_95, upper_95)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    # Filter out -1 sentinel (NaN dominance)
    valid = labels >= 0
    p, l = preds[valid], labels[valid]
    if len(l) == 0:
        return 0.0, 0.0, 0.0
    point = f1_score(l, p, average="macro", zero_division=0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(l), size=len(l))
        f = f1_score(l[idx], p[idx], average="macro", zero_division=0)
        boots.append(f)
    boots = np.array(boots)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return float(point), lo, hi


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_pickle(args.dataset)
    aug_pool = None
    if getattr(args, "augmented_pool", "") and args.augmented_pool.strip():
        aug_pool = pd.read_pickle(args.augmented_pool.strip())

    train_df, val_df, test_df = task_split(df, test_task="T3")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2idx    = make_user2idx(full_df)
    session2idx = make_session2idx(full_df)
    summary_key_order = build_summary_key_order(full_df)
    summary_dim = len(summary_key_order)

    scalers    = fit_scalers(train_df)
    thresholds = compute_tertile_thresholds(train_df)
    class_wts  = compute_class_weights(train_df, thresholds, device)

    train_ds = VADDataset(train_df, user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=False, device=device)
    val_ds   = VADDataset(val_df,   user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=False, device=device)
    test_ds  = VADDataset(test_df,  user2idx, session2idx, summary_key_order,
                          scalers, thresholds, augment=False, device=device)

    bfi_mode = getattr(args, "bfi_mode", "none")
    if bfi_mode != "none":
        bfi_mean = train_ds._pers.mean(axis=0, keepdims=True)
        bfi_std  = train_ds._pers.std(axis=0, keepdims=True).clip(min=1e-6)
        train_ds._pers = ((train_ds._pers - bfi_mean) / bfi_std).astype(np.float32)
        val_ds._pers   = ((val_ds._pers   - bfi_mean) / bfi_std).astype(np.float32)
        test_ds._pers  = ((test_ds._pers  - bfi_mean) / bfi_std).astype(np.float32)

    if aug_pool is not None:
        train_tasks = set(train_df["task"].unique())
        aug_df = aug_pool[aug_pool["task"].isin(train_tasks)].copy()
        target_T = train_ds._gaze[0].shape[0]
        aug_ds = AugSoftDataset(aug_df, user2idx, session2idx, summary_key_order,
                                scalers, device=device, target_T=target_T)
        if bfi_mode != "none":
            aug_ds._pers = ((aug_ds._pers - bfi_mean) / bfi_std).astype(np.float32)

        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        combined = ConcatDataset([train_ds, aug_ds])
        n_hard, n_soft = len(train_ds), len(aug_ds)
        aug_frac = getattr(args, "aug_frac", 0.3)
        ratio = (n_hard * aug_frac) / (n_soft * (1.0 - aug_frac))
        aug_conf = aug_ds._vad_weight.mean(axis=1)
        all_w = np.concatenate([np.ones(n_hard), ratio * aug_conf])
        all_w /= all_w.max() + 1e-8
        sampler = WeightedRandomSampler(
            torch.from_numpy(all_w.astype(np.float32)),
            num_samples=n_hard + n_soft, replacement=True,
        )
        train_loader = DataLoader(combined, batch_size=16, sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)

    val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_model(args, summary_dim, n_subjects=len(user2idx)).to(device)
    loss_fn   = SoftVADLoss(class_weights=class_wts, label_smoothing=0.1, alpha=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, 200, warmup=10)

    best_val = -1.0
    patience = 0
    for ep in range(1, 201):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        if ep % 5 == 0 or ep == 200:
            m = evaluate(model, val_loader, loss_fn, device, thresholds, val_df)
            if m["mean_f1"] > best_val:
                best_val = m["mean_f1"]
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 5
                if patience >= 60:
                    break

    # Load best model
    model.load_state_dict(best_state)

    # Collect predictions on test set
    preds_dict = collect_test_predictions(model, test_loader, device, thresholds)

    # Bootstrap
    print(f"\n=== Bootstrap CIs (n=1000, 95%) | bfi_mode={bfi_mode} ===")
    print(f"  Test set: N={len(test_df)} windows")
    print(f"  {'Dim':<12} {'F1':>7} {'95% CI':>18}")
    print(f"  {'-'*38}")
    f1s = []
    for dim in VAD_DIMS:
        p, l = preds_dict[dim]
        pt, lo, hi = bootstrap_f1(p, l)
        f1s.append(pt)
        print(f"  {dim:<12} {pt:>7.3f}  [{lo:.3f}, {hi:.3f}]")
    mean_pt = float(np.mean(f1s))
    # Bootstrap mean F1
    all_boot = []
    rng = np.random.default_rng(42)
    n = len(test_df)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        fold_f1s = []
        for dim in VAD_DIMS:
            p_all, l_all = preds_dict[dim]
            valid = l_all >= 0
            p_v, l_v = p_all[valid], l_all[valid]
            if len(l_v) == 0:
                fold_f1s.append(0.0)
                continue
            idx2 = idx[idx < len(l_v)]
            if len(idx2) == 0:
                fold_f1s.append(0.0)
                continue
            idx_v = rng.integers(0, len(l_v), size=len(l_v))
            f = f1_score(l_v[idx_v], p_v[idx_v], average="macro", zero_division=0)
            fold_f1s.append(f)
        all_boot.append(np.mean(fold_f1s))
    all_boot = np.array(all_boot)
    lo_m = float(np.percentile(all_boot, 2.5))
    hi_m = float(np.percentile(all_boot, 97.5))
    print(f"  {'Mean F1':<12} {mean_pt:>7.3f}  [{lo_m:.3f}, {hi_m:.3f}]")
    print(f"  {'='*38}")
    print(f"  val peak = {best_val:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--augmented-pool", default="")
    parser.add_argument("--aug-frac",       type=float, default=0.3)
    parser.add_argument("--arch",           default="mlp",
                        choices=["mlp","pool","conv","transformer"])
    parser.add_argument("--d-enc",          type=int, default=32)
    parser.add_argument("--d-fuse",         type=int, default=64)
    parser.add_argument("--t-out",          type=int, default=16)
    parser.add_argument("--dropout",        type=float, default=0.5)
    parser.add_argument("--bfi-mode",       default="none",
                        choices=["none","concat","gate","perdim"], dest="bfi_mode")
    parser.add_argument("--no-subject-embed", action="store_true", dest="no_subject_embed")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
