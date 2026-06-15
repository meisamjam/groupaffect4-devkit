"""
Enhance augmented_pool.pkl with GSR-derived arousal labels.

The current GP arousal labels are collapsed to the prior (mu~4.9, weight=0.05)
because SCR-event cross-seat GP has effectively zero predictive power for arousal.
This script replaces them with Ridge-regression-derived GSR arousal labels, which
show task-specific correlations with SAM arousal:
  T2 (negotiation)   : rho = -0.266  -> weight = 0.0  (negative, skip)
  T3 (ideation)      : rho = +0.531  -> weight = 0.50
  T4 (public goods)  : rho = +0.500  -> weight = 0.45
  T0 (rest)          : rho = +0.267  -> weight = 0.20
  T1 (info pooling)  : rho = +0.165  -> weight = 0.15

Usage:
    python tools/mumt/enhance_pool_with_gsr.py \
        --pool data/mumt/augmented_pool.pkl \
        --gsr-labels results/gsr_arousal_pool.csv \
        --output data/mumt/augmented_pool_gsr.pkl
"""

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

# Task-specific arousal weights based on LOO cross-check (GSR vs SAM arousal rho)
# T2 is negative -- actively misleading, zeroed out
TASK_AROUSAL_WEIGHT: dict[str, float] = {
    "T0": 0.20,
    "T1": 0.15,
    "T2": 0.00,   # negative correlation in cross-check
    "T3": 0.50,
    "T4": 0.45,
}

# Observation noise (from LOO MAE = 1.47; use slightly tighter 1.3 to account for
# the fact that Ridge already regularises toward the mean, so actual per-row error
# for high-confidence predictions is lower)
GSR_AROUSAL_SIGMA = 1.3

# Tertile boundaries for 3-class soft label recomputation
TERTILE_BINS = np.array([3.5, 6.5])


def mu_sigma_to_soft(mu: float, sigma: float) -> np.ndarray:
    """Convert ordinal GP posterior to 3-class soft probabilities (Low / Mid / High)."""
    p_low = float(norm.cdf(TERTILE_BINS[0], loc=mu, scale=sigma))
    p_high = float(1.0 - norm.cdf(TERTILE_BINS[1], loc=mu, scale=sigma))
    p_mid = max(0.0, 1.0 - p_low - p_high)
    total = p_low + p_mid + p_high
    if total < 1e-8:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)
    return np.array([p_low / total, p_mid / total, p_high / total], dtype=np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Enhance augmented pool with GSR arousal labels.")
    parser.add_argument("--pool", default="data/mumt/augmented_pool.pkl")
    parser.add_argument("--gsr-labels", default="results/gsr_arousal_pool.csv")
    parser.add_argument("--output", default="data/mumt/augmented_pool_gsr.pkl")
    parser.add_argument("--gsr-sigma", type=float, default=GSR_AROUSAL_SIGMA,
                        help="Observation noise for GSR-derived arousal labels (default: 1.3)")
    parser.add_argument("--task-weights", type=str, default="",
                        help="Override task weights: e.g. 'T3=0.5,T4=0.45,T2=0.0'")
    parser.add_argument("--calibrate", action="store_true", default=True,
                        help="Affine-calibrate GSR labels to match SAM arousal distribution "
                             "(mean=6.22, std=1.85). Corrects Ridge mean-regression squeeze.")
    parser.add_argument("--sam-mean", type=float, default=6.22,
                        help="Target mean for calibration (SAM arousal mean on labelled dataset)")
    parser.add_argument("--sam-std", type=float, default=1.85,
                        help="Target std for calibration (SAM arousal std on labelled dataset)")
    args = parser.parse_args()

    task_weights = dict(TASK_AROUSAL_WEIGHT)
    if args.task_weights:
        for kv in args.task_weights.split(","):
            k, v = kv.split("=")
            task_weights[k.strip()] = float(v.strip())

    log.info("Loading pool: %s", args.pool)
    pool = pd.read_pickle(args.pool)
    log.info("  %d rows, tasks: %s", len(pool), pool["task"].value_counts().to_dict())

    log.info("Loading GSR labels: %s", args.gsr_labels)
    gsr_df = pd.read_csv(args.gsr_labels)
    log.info("  %d rows, %d with valid GSR arousal",
             len(gsr_df), gsr_df["gsr_arousal"].notna().sum())

    # Affine calibration: rescale GSR labels from Ridge-squeezed distribution to SAM distribution
    valid_gsr = gsr_df["gsr_arousal"].dropna()
    gsr_src_mean = float(valid_gsr.mean())
    gsr_src_std  = float(valid_gsr.std())
    log.info("GSR label distribution before calibration: mean=%.3f std=%.3f", gsr_src_mean, gsr_src_std)

    def calibrate(val: float) -> float:
        if not args.calibrate or gsr_src_std < 1e-6:
            return val
        scaled = (val - gsr_src_mean) / gsr_src_std * args.sam_std + args.sam_mean
        return float(np.clip(scaled, 1.0, 9.0))

    if args.calibrate:
        log.info("Calibrating: (gsr - %.3f) / %.3f * %.3f + %.3f -> clipped to [1,9]",
                 gsr_src_mean, gsr_src_std, args.sam_std, args.sam_mean)

    # Build lookup: pool row index -> calibrated gsr_arousal
    gsr_map: dict[int, float] = {}
    for _, row in gsr_df.iterrows():
        val = float(row["gsr_arousal"])
        if np.isfinite(val):
            gsr_map[int(row["window_index"])] = calibrate(val)

    log.info("  GSR coverage: %d / %d pool rows", len(gsr_map), len(pool))

    # --- Before stats ---
    print("\n--- Before enhancement ---")
    print(f"arousal_mu:    mean={pool['arousal_mu'].mean():.3f} "
          f"std={pool['arousal_mu'].std():.3f} "
          f"min={pool['arousal_mu'].min():.3f} max={pool['arousal_mu'].max():.3f}")
    print(f"arousal_sigma: mean={pool['arousal_sigma'].mean():.3f}")
    print(f"arousal_weight:mean={pool['arousal_weight'].mean():.4f} "
          f"max={pool['arousal_weight'].max():.3f}")
    for task in sorted(pool["task"].unique()):
        t = pool[pool["task"] == task]
        print(f"  {task}: n={len(t)} mu_mean={t['arousal_mu'].mean():.3f} "
              f"weight_mean={t['arousal_weight'].mean():.4f}")

    # --- Apply GSR labels ---
    pool = pool.copy()
    n_updated = 0
    n_skipped_no_gsr = 0
    n_skipped_zero_weight = 0

    for idx in pool.index:
        task = str(pool.at[idx, "task"])
        task_w = task_weights.get(task, 0.0)
        gsr_val = gsr_map.get(idx)

        if task_w == 0.0:
            # Task has negative or negligible correlation — zero out arousal
            pool.at[idx, "arousal_weight"] = 0.0
            n_skipped_zero_weight += 1
            continue

        if gsr_val is None or not np.isfinite(gsr_val):
            # No GSR for this window — zero out arousal (prior is useless)
            pool.at[idx, "arousal_weight"] = 0.0
            n_skipped_no_gsr += 1
            continue

        # Replace GP arousal with GSR arousal
        pool.at[idx, "arousal_mu"] = gsr_val
        pool.at[idx, "arousal_sigma"] = args.gsr_sigma
        pool.at[idx, "arousal_weight"] = task_w
        pool.at[idx, "arousal_soft"] = mu_sigma_to_soft(gsr_val, args.gsr_sigma)
        n_updated += 1

    log.info("Updated: %d rows with GSR arousal", n_updated)
    log.info("Zeroed (no GSR coverage): %d rows", n_skipped_no_gsr)
    log.info("Zeroed (T2 negative correlation): %d rows", n_skipped_zero_weight)

    # --- After stats ---
    active = pool[pool["arousal_weight"] > 0]
    print("\n--- After enhancement ---")
    print(f"Rows with arousal_weight > 0: {len(active)} / {len(pool)}")
    print(f"arousal_mu (active): mean={active['arousal_mu'].mean():.3f} "
          f"std={active['arousal_mu'].std():.3f} "
          f"min={active['arousal_mu'].min():.3f} max={active['arousal_mu'].max():.3f}")
    print(f"arousal_sigma (active): mean={active['arousal_sigma'].mean():.3f}")
    print(f"arousal_weight (active): mean={active['arousal_weight'].mean():.4f}")
    for task in sorted(pool["task"].unique()):
        t = pool[pool["task"] == task]
        a = t[t["arousal_weight"] > 0]
        w = task_weights.get(str(task), 0.0)
        print(f"  {task}: n_total={len(t)} n_active={len(a)} "
              f"mu_mean={a['arousal_mu'].mean():.3f} if len(a)>0 else 'n/a' "
              f"task_weight={w}")

    print("\nTask-specific arousal weights applied:")
    for task, w in sorted(task_weights.items()):
        print(f"  {task}: {w}")

    pool.to_pickle(args.output)
    log.info("Saved enhanced pool -> %s", args.output)


if __name__ == "__main__":
    main()
