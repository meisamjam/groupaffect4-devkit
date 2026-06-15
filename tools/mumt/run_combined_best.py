"""run_combined_best.py — Combined best experiment: MSM + TW + AP1 via run_fold.

Combines the three best ingredients that each improve different VAD dimensions:
  1. MSM pre-training (Masked Signal Modeling) — learns modality-specific dynamics
  2. Time Warping augmentation — non-linear temporal distortion (replaces shift)
  3. AP1 pool augmentation — BFI-similarity pseudo-labels (properly integrated)

Key insight: train_temporal.py's run_fold() handles AP1 properly (two-track training
with confidence thresholds), while pretrain_temporal.py's finetune() handles
pre-trained encoders but has no pool track. Nobody combined them before.

Expected outcome per dimension:
  - Valence:  AP1 helps (SVM: 0.570→0.608), Conv1D v3 got 0.601
  - Arousal:  AP1+TW help (GRU E1+E2: 0.587), SVM physio-smooth: 0.624
  - Dominance: GRU temporal is king (0.520 vs SVM 0.435), MSM+TW preserve this

Usage
-----
  cd tools/mumt
  python run_combined_best.py
  python run_combined_best.py --encoder conv1d --seeds 5
  python run_combined_best.py --no-pretrain  # ablation: TW+AP1 without MSM
  python run_combined_best.py --no-tw        # ablation: MSM+AP1 without TW
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

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import build_summary_key_order  # noqa: E402
from train_simple import task_split  # noqa: E402
from train_temporal import (  # noqa: E402
    compute_bfi_similarity_map,
    fit_seq_scalers,
    run_fold,
)
from literature_experiments import (  # noqa: E402
    augment_sequence_tw,
    pretrain_masked,
    MaskedEncoder,
    MaskedSignalModelingLoss,
    MSMPoolDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined best experiment: MSM pre-training + Time Warping + AP1"
    )
    parser.add_argument(
        "--dataset", default="data/mumt/dataset_15s.pkl",
    )
    parser.add_argument(
        "--pool", default="data/mumt/augmented_pool.pkl",
    )
    parser.add_argument("--encoder", choices=["gru", "conv1d", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--enc-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--msm-epochs", type=int, default=60,
                        help="Number of MSM pre-training epochs")
    parser.add_argument("--msm-lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--freeze-epochs", type=int, default=10,
                        help="Epochs to freeze encoders after loading pre-trained weights")
    parser.add_argument("--aug-mode", default="ap1",
                        choices=["ap1", "ap2", "a2", "none"],
                        help="Pool augmentation mode for run_fold")
    parser.add_argument("--test-task", default="T3")
    # Ablation flags
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Ablation: skip MSM pre-training (random init)")
    parser.add_argument("--no-tw", action="store_true",
                        help="Ablation: use default augmentation (no time warping)")
    parser.add_argument("--no-pool-aug", action="store_true",
                        help="Ablation: disable pool augmentation (no AP1)")
    parser.add_argument("--out", default=str(_HERE / "results" / "combined_best.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d windows", len(df))

    pool: pd.DataFrame | None = None
    if Path(args.pool).exists():
        pool = pd.read_pickle(args.pool)
        log.info("Pool: %d windows", len(pool))
    else:
        log.warning("Pool not found: %s — no pretraining/augmentation", args.pool)

    key_order = build_summary_key_order(df)
    bfi_sim_map = compute_bfi_similarity_map(df)
    log.info("Summary dim: %d  |  BFI map entries: %d", len(key_order), len(bfi_sim_map))

    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Pool restricted to train tasks (prevent leakage)
    pool_train: pd.DataFrame | None = None
    if pool is not None:
        train_tasks = train_df["task"].unique().tolist()
        pool_train = pool[pool["task"].isin(train_tasks)].reset_index(drop=True)
        log.info("Pool (train tasks only): %d", len(pool_train))

    # ── Determine augmentation function ────────────────────────────────────────
    seq_aug_fn = augment_sequence_tw if not args.no_tw else None

    # ── Run experiments ────────────────────────────────────────────────────────
    encoders = ["gru", "conv1d"] if args.encoder == "both" else [args.encoder]
    aug_mode = "none" if args.no_pool_aug else args.aug_mode

    records = []
    for enc in encoders:
        for seed in range(args.seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            tag_parts = []
            pretrained_encoders = None

            # ── MSM pre-training ──────────────────────────────────────────────
            if not args.no_pretrain and pool_train is not None:
                seq_scalers = fit_seq_scalers(train_df)
                pretrained_encoders = pretrain_masked(
                    pool=pool_train,
                    seq_scalers=seq_scalers,
                    encoder_type=enc,
                    enc_dim=args.enc_dim,
                    epochs=args.msm_epochs,
                    batch_size=args.batch,
                    lr=args.msm_lr,
                    mask_ratio=args.mask_ratio,
                    device=device,
                )
                tag_parts.append("MSM")
            elif args.no_pretrain:
                tag_parts.append("noMSM")

            # ── Tags ──────────────────────────────────────────────────────────
            if not args.no_tw:
                tag_parts.append("TW")
            if aug_mode != "none":
                tag_parts.append(aug_mode.upper())

            experiment_name = f"{enc}+{'_'.join(tag_parts)}"
            log.info("=== [seed=%d] %s ===", seed, experiment_name)

            # ── Fine-tune with run_fold (proper AP1 integration) ──────────────
            result = run_fold(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                pool=pool_train if aug_mode != "none" else None,
                bfi_sim_map=bfi_sim_map,
                key_order=key_order,
                encoder_type=enc,
                aug_mode=aug_mode,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                enc_dim=args.enc_dim,
                dropout=args.dropout,
                patience=args.patience,
                device=device,
                seq_aug=True,  # always enable seq augmentation
                pool_seqs=True,
                pretrained_encoders=pretrained_encoders,
                freeze_epochs=args.freeze_epochs if pretrained_encoders else 0,
                seq_aug_fn=seq_aug_fn,
            )

            records.append({
                "experiment": experiment_name,
                "encoder": enc,
                "seed": seed,
                "pretrain": "MSM" if not args.no_pretrain else "none",
                "seq_aug": "TW" if not args.no_tw else "default",
                "pool_aug": aug_mode,
                **result,
            })

            # Log per-seed results
            log.info("  V=%.3f  A=%.3f  D=%.3f  mean=%.3f",
                     result.get("v_f1", 0),
                     result.get("a_f1", 0),
                     result.get("d_f1", 0),
                     result.get("test_f1", 0))

    # ── Aggregate and save ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    log.info("Results saved: %s", out_path)

    # Print summary (seed-averaged)
    print("\n" + "=" * 70)
    print("COMBINED BEST — SEED-AVERAGED RESULTS")
    print("=" * 70)
    for exp_name, grp in results_df.groupby("experiment"):
        v = grp["v_f1"].mean()
        a = grp["a_f1"].mean()
        d = grp["d_f1"].mean()
        m = grp["test_f1"].mean()
        print(f"  {exp_name:<30s}  V={v:.3f}  A={a:.3f}  D={d:.3f}  mean={m:.3f}")

    print("\n--- Reference: SVM + AP1 ---")
    print("  SVM+AP1                         V=0.608  A=0.535  D=0.435")
    print("  SVM+AP1+physio_smooth_45s       V=0.600  A=0.624  D=0.436")
    print("\n--- Reference: Previous temporal ---")
    print("  GRU v3+pretrain+seqaug (paper)  V=0.460  A=0.536  D=0.520")
    print("  Conv1D v3+pretrain+seqaug       V=0.601  A=0.506  D=0.351")
    print("=" * 70)


if __name__ == "__main__":
    main()
