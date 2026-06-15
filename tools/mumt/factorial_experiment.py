"""factorial_experiment.py — Full 3-factor factorial: Encoder × Pretrain × Augmentation.

Systematic evaluation of all combinations of three independent factors:

Factor 1 — Temporal encoder:
  - conv1d: Conv1DEncoder (local patterns, good for Valence)
  - gru:    GRUEncoder (recurrent, good for Dominance)

Factor 2 — Pre-training method (self-supervised on pool):
  - none:   Random initialization
  - simclr: SimCLR contrastive (NT-Xent, instance discrimination)
  - msm:    Masked Signal Modeling (reconstructive, modality-specific)

Factor 3 — Sequence augmentation (applied to labeled windows during training):
  - none:     No augmentation
  - default:  Noise + amplitude jitter + circular shift
  - tw:       Noise + amplitude jitter + Time Warping (cubic spline)

Total: 2 × 3 × 3 = 18 conditions, each run with N seeds.
No pool augmentation (AP1) during fine-tuning — confirmed incompatible with
temporal models due to 30:1 batch imbalance.

Usage
-----
  python tools/mumt/factorial_experiment.py
  python tools/mumt/factorial_experiment.py --seeds 5
  python tools/mumt/factorial_experiment.py --encoder gru --pretrain msm --aug tw
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
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
from pretrain_temporal import pretrain_encoders  # noqa: E402
from literature_experiments import (  # noqa: E402
    augment_sequence_tw,
    pretrain_masked,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Factor levels
# ═══════════════════════════════════════════════════════════════════════════════

ENCODERS = ["conv1d", "gru"]
PRETRAINS = ["none", "simclr", "msm"]
AUGMENTATIONS = ["none", "default", "tw"]


def do_pretrain(
    method: str,
    pool: pd.DataFrame,
    seq_scalers: dict,
    encoder_type: str,
    enc_dim: int,
    device: torch.device,
    pretrain_epochs: int = 60,
    batch_size: int = 32,
):
    """Run pre-training and return encoder ModuleDict or None."""
    if method == "none":
        return None
    elif method == "simclr":
        return pretrain_encoders(
            pool=pool,
            seq_scalers=seq_scalers,
            encoder_type=encoder_type,
            enc_dim=enc_dim,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=1e-3,
            temperature=0.1,
            device=device,
        )
    elif method == "msm":
        return pretrain_masked(
            pool=pool,
            seq_scalers=seq_scalers,
            encoder_type=encoder_type,
            enc_dim=enc_dim,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=1e-3,
            mask_ratio=0.25,
            device=device,
        )
    else:
        raise ValueError(f"Unknown pretrain method: {method}")


def get_aug_fn(aug_name: str):
    """Return (seq_aug: bool, seq_aug_fn: callable|None) for the augmentation level."""
    if aug_name == "none":
        return False, None
    elif aug_name == "default":
        return True, None  # uses built-in augment_sequence (noise/jitter/shift)
    elif aug_name == "tw":
        return True, augment_sequence_tw  # time warping replaces shift
    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-factor factorial: Encoder × Pretrain × Augmentation"
    )
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool", default="data/mumt/augmented_pool.pkl")
    parser.add_argument("--encoder", default="all",
                        choices=["all", "conv1d", "gru"])
    parser.add_argument("--pretrain", default="all",
                        choices=["all", "none", "simclr", "msm"])
    parser.add_argument("--aug", default="all",
                        choices=["all", "none", "default", "tw"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--enc-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--test-task", default="T3")
    parser.add_argument("--out", default="results/factorial_experiment.csv")
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
        log.warning("Pool not found: %s — pre-training disabled", args.pool)

    key_order = build_summary_key_order(df)
    bfi_sim_map = compute_bfi_similarity_map(df)

    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Pool restricted to train tasks
    pool_train: pd.DataFrame | None = None
    if pool is not None:
        train_tasks = train_df["task"].unique().tolist()
        pool_train = pool[pool["task"].isin(train_tasks)].reset_index(drop=True)
        log.info("Pool (train tasks only): %d", len(pool_train))

    # Fit scalers once (shared across all conditions)
    seq_scalers = fit_seq_scalers(train_df)

    # ── Determine factor levels ────────────────────────────────────────────────
    enc_levels = ENCODERS if args.encoder == "all" else [args.encoder]
    pre_levels = PRETRAINS if args.pretrain == "all" else [args.pretrain]
    aug_levels = AUGMENTATIONS if args.aug == "all" else [args.aug]

    conditions = list(itertools.product(enc_levels, pre_levels, aug_levels))
    total_runs = len(conditions) * args.seeds
    log.info("Factorial design: %d conditions × %d seeds = %d total runs",
             len(conditions), args.seeds, total_runs)

    # ── Run experiments ────────────────────────────────────────────────────────
    records = []
    run_idx = 0

    for enc, pre, aug in conditions:
        # Skip pretrain conditions if no pool
        if pre != "none" and pool_train is None:
            log.warning("Skip %s/%s/%s — pool not available", enc, pre, aug)
            continue

        condition_name = f"{enc}|{pre}|{aug}"
        log.info("\n{'='*60}")
        log.info("CONDITION: encoder=%s  pretrain=%s  aug=%s", enc, pre, aug)
        log.info("{'='*60}")

        seq_aug_flag, seq_aug_fn = get_aug_fn(aug)

        for seed in range(args.seeds):
            run_idx += 1
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            log.info("  [%d/%d] seed=%d  %s", run_idx, total_runs, seed, condition_name)
            t0 = time.time()

            # Pre-training (per seed for fair comparison)
            pretrained = None
            if pre != "none" and pool_train is not None:
                pretrained = do_pretrain(
                    method=pre,
                    pool=pool_train,
                    seq_scalers=seq_scalers,
                    encoder_type=enc,
                    enc_dim=args.enc_dim,
                    device=device,
                    pretrain_epochs=args.pretrain_epochs,
                    batch_size=32,
                )

            # Fine-tuning via run_fold (no pool aug during training)
            result = run_fold(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                pool=None,  # NO pool during fine-tuning
                bfi_sim_map=bfi_sim_map,
                key_order=key_order,
                encoder_type=enc,
                aug_mode="none",
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                enc_dim=args.enc_dim,
                dropout=args.dropout,
                patience=args.patience,
                device=device,
                seq_aug=seq_aug_flag,
                pool_seqs=True,
                pretrained_encoders=pretrained,
                freeze_epochs=args.freeze_epochs if pretrained is not None else 0,
                seq_aug_fn=seq_aug_fn,
            )

            elapsed = time.time() - t0
            records.append({
                "encoder": enc,
                "pretrain": pre,
                "augmentation": aug,
                "seed": seed,
                "v_f1": result["v_f1"],
                "a_f1": result["a_f1"],
                "d_f1": result["d_f1"],
                "test_f1": result["test_f1"],
                "val_f1": result["val_f1"],
                "time_s": round(elapsed, 1),
            })

            log.info("    V=%.3f  A=%.3f  D=%.3f  mean=%.3f  (%.0fs)",
                     result["v_f1"], result["a_f1"], result["d_f1"],
                     result["test_f1"], elapsed)

    # ── Save results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    log.info("\nResults saved: %s (%d rows)", out_path, len(results_df))

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FACTORIAL EXPERIMENT — SEED-AVERAGED RESULTS (3-seed mean ± std)")
    print("=" * 80)
    print(f"{'Encoder':<8} {'Pretrain':<8} {'Aug':<8} {'V':>6} {'A':>6} {'D':>6} {'Mean':>6}")
    print("-" * 80)

    for (enc, pre, aug), grp in results_df.groupby(
        ["encoder", "pretrain", "augmentation"], sort=False
    ):
        v = grp["v_f1"].mean()
        a = grp["a_f1"].mean()
        d = grp["d_f1"].mean()
        m = grp["test_f1"].mean()
        print(f"{enc:<8} {pre:<8} {aug:<8} {v:>6.3f} {a:>6.3f} {d:>6.3f} {m:>6.3f}")

    print("-" * 80)
    print("\nREFERENCE (SVM baselines):")
    print("  SVM no-aug:                V=0.570  A=0.530  D=0.385")
    print("  SVM + AP1:                 V=0.608  A=0.535  D=0.435")
    print("  SVM + AP1 + physio-smooth: V=0.600  A=0.624  D=0.436")
    print("=" * 80)

    # ── Factor-level analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FACTOR-LEVEL MARGINAL MEANS")
    print("=" * 80)

    for factor in ["encoder", "pretrain", "augmentation"]:
        print(f"\n  --- {factor.upper()} ---")
        for level, grp in results_df.groupby(factor, sort=False):
            v = grp["v_f1"].mean()
            a = grp["a_f1"].mean()
            d = grp["d_f1"].mean()
            m = grp["test_f1"].mean()
            print(f"    {level:<10s}  V={v:.3f}  A={a:.3f}  D={d:.3f}  mean={m:.3f}")

    # ── Best per dimension ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BEST CONDITION PER DIMENSION (seed-averaged)")
    print("=" * 80)

    agg = results_df.groupby(["encoder", "pretrain", "augmentation"]).agg(
        v_mean=("v_f1", "mean"),
        a_mean=("a_f1", "mean"),
        d_mean=("d_f1", "mean"),
        overall=("test_f1", "mean"),
    ).reset_index()

    for dim, col in [("Valence", "v_mean"), ("Arousal", "a_mean"),
                     ("Dominance", "d_mean"), ("Overall", "overall")]:
        best = agg.loc[agg[col].idxmax()]
        print(f"  {dim:<10s}: {best['encoder']}|{best['pretrain']}|{best['augmentation']}"
              f"  → {best[col]:.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
