"""eval_label_augmentation.py

Leave-one-out (LOO) calibration evaluation of the GP label augmentation.

For each of the 292 labelled windows, treats it as if unlabelled:
  - Removes it from the GP conditioning observations for that subject
  - Runs the OU-GP to produce a soft 3-class posterior at that window's
    timestamp, conditioned only on the OTHER labelled windows of the same
    subject (same logic as label_augmentation.py's S1 source)
  - Compares the GP posterior to the actual hard label

Metrics computed:
  - Top-1 accuracy   : argmax(soft) == hard_class  (mean over windows)
  - Brier score      : mean squared error between soft and one-hot hard label
  - ECE              : expected calibration error (binned confidence)
  - Rank-1 frequency : true class has highest soft probability
  - Per-dimension breakdown

Usage:
  python tools/mumt/eval_label_augmentation.py \
      --dataset data/mumt/dataset.pkl

Optional -- write a detailed CSV of per-window predictions:
  python tools/mumt/eval_label_augmentation.py \
      --dataset data/mumt/dataset.pkl \
      --output-csv results/label_aug_calibration.csv
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Reuse OU/GP primitives from label_augmentation.py ────────────────────────
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from label_augmentation import (
    DIMS,
    MENG_OU_PARAMS,
    OUParams,
    VAD_THRESHOLDS,
    ou_gp_posterior,
    soft_label_from_posterior,
    estimate_ou_params,
    SIGMA_SELF_REPORT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_tertile_thresholds(vals: np.ndarray) -> tuple[float, float]:
    """Return (p33, p67) tertile thresholds from a clean array."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return (3.0, 6.0)
    return float(np.percentile(vals, 33.33)), float(np.percentile(vals, 66.67))


def soft_to_class(soft: np.ndarray) -> int:
    return int(np.argmax(soft))


def brier_score(soft: np.ndarray, hard_class: int) -> float:
    one_hot = np.zeros(3, dtype=float)
    one_hot[hard_class] = 1.0
    return float(np.mean((soft - one_hot) ** 2))


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute ECE over *n_bins* equal-width confidence bins."""
    conf = np.array(confidences)
    acc  = np.array(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc  = float(acc[mask].mean())
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return ece


# ─────────────────────────────────────────────────────────────────────────────
# LOO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def loo_evaluate(
    df: pd.DataFrame,
    ou_params: dict[str, OUParams],
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """For each row in df, run LOO-GP and return per-window prediction metrics.

    Returns a DataFrame with columns:
      session_id, subject_id, task, dim,
      hard_class, gp_class, gp_top1_prob,
      soft_low, soft_mid, soft_high,
      brier, correct, gp_mu, gp_sigma, n_obs
    """
    records: list[dict] = []

    required = ["valence", "arousal", "dominance",
                "vad_timestamp_lsl", "session_id", "task", "seat"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Dataset missing column: {c!r}")

    # Pre-group by (session_id, seat) for fast LOO indexing
    groups: dict[tuple, pd.DataFrame] = {}
    for key, grp in df.groupby(["session_id", "seat"]):
        groups[key] = grp.sort_values("vad_timestamp_lsl").reset_index(drop=True)

    total = len(df)
    for i_global, (_, row) in enumerate(df.iterrows()):
        if i_global % 50 == 0:
            log.info("  LOO progress: %d / %d", i_global, total)

        ses   = str(row["session_id"])
        seat  = str(row["seat"])
        task  = str(row["task"])
        t_q   = float(row["vad_timestamp_lsl"])

        key = (ses, seat)
        grp = groups.get(key, pd.DataFrame())

        for dim_idx, dim in enumerate(DIMS):
            hard_val = float(row[dim])
            if np.isnan(hard_val):
                continue  # skip NaN-labelled windows

            # Hard class using training-split thresholds
            t1, t2 = thresholds[dim]
            if hard_val <= t1:
                hard_class = 0
            elif hard_val <= t2:
                hard_class = 1
            else:
                hard_class = 2

            params = ou_params[dim]

            # --- LOO: collect ALL obs for this subject EXCEPT the current row ---
            t_obs_list: list[float] = []
            y_obs_list: list[float] = []
            s_obs_list: list[float] = []

            for _, other in grp.iterrows():
                t_other = float(other["vad_timestamp_lsl"])
                if abs(t_other - t_q) < 1e-3:
                    continue  # skip self
                y_other = float(other[dim])
                if np.isnan(y_other):
                    continue
                t_obs_list.append(t_other)
                y_obs_list.append(y_other)
                s_obs_list.append(SIGMA_SELF_REPORT)

            n_obs = len(t_obs_list)

            if n_obs == 0:
                # Fallback: OU prior
                mu_post  = params.mu
                std_post = float(np.sqrt(params.sigma2))
            else:
                mu_post, std_post = ou_gp_posterior(
                    t_q,
                    np.array(t_obs_list),
                    np.array(y_obs_list),
                    np.array(s_obs_list),
                    params,
                )

            soft = soft_label_from_posterior(mu_post, std_post, (t1, t2))
            gp_class   = soft_to_class(soft)
            top1_prob  = float(soft[gp_class])
            brier      = brier_score(soft, hard_class)
            correct    = (gp_class == hard_class)

            records.append({
                "session_id":   ses,
                "subject_id":   str(row.get("subject_id", "")),
                "task":         task,
                "dim":          dim,
                "hard_class":   hard_class,
                "gp_class":     gp_class,
                "gp_top1_prob": top1_prob,
                "soft_low":     float(soft[0]),
                "soft_mid":     float(soft[1]),
                "soft_high":    float(soft[2]),
                "brier":        brier,
                "correct":      correct,
                "gp_mu":        mu_post,
                "gp_sigma":     std_post,
                "n_obs":        n_obs,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("       GP Label Augmentation — LOO Calibration Report")
    print("=" * 70)

    dims_present = results["dim"].unique()

    # ── Per-dimension breakdown ───────────────────────────────────────────
    print(f"\n{'Dim':<12} {'N':>5} {'Top-1 Acc':>10} {'Brier':>8} "
          f"{'ECE':>8} {'No-obs':>8} {'Med n_obs':>10}")
    print("-" * 65)

    for dim in ["valence", "arousal", "dominance"]:
        if dim not in dims_present:
            continue
        sub = results[results["dim"] == dim]
        acc    = sub["correct"].mean()
        brier  = sub["brier"].mean()
        no_obs = (sub["n_obs"] == 0).mean()
        med_n  = sub["n_obs"].median()

        # ECE
        ece = expected_calibration_error(
            sub["gp_top1_prob"].tolist(),
            sub["correct"].tolist(),
        )

        print(f"{dim:<12} {len(sub):>5} {acc:>10.3f} {brier:>8.3f} "
              f"{ece:>8.3f} {no_obs:>8.3f} {med_n:>10.1f}")

    # ── Overall ───────────────────────────────────────────────────────────
    print("-" * 65)
    acc_all   = results["correct"].mean()
    brier_all = results["brier"].mean()
    ece_all   = expected_calibration_error(
        results["gp_top1_prob"].tolist(),
        results["correct"].tolist(),
    )
    print(f"{'OVERALL':<12} {len(results):>5} {acc_all:>10.3f} {brier_all:>8.3f} "
          f"{ece_all:>8.3f}")
    print("=" * 70)

    # ── Per-task breakdown ────────────────────────────────────────────────
    print("\nPer-task top-1 accuracy (all dims):")
    for task, grp in results.groupby("task"):
        print(f"  {task}: {grp['correct'].mean():.3f}  (N={len(grp)})")

    # ── Class distribution vs. GP distribution ────────────────────────────
    print("\nClass distribution: hard labels vs. GP argmax (all dims):")
    for label, name in [(0, "Low"), (1, "Mid"), (2, "High")]:
        hard_frac = (results["hard_class"] == label).mean()
        gp_frac   = (results["gp_class"]   == label).mean()
        print(f"  {name:<5}: hard={hard_frac:.3f}  GP={gp_frac:.3f}")

    # ── Confidence--accuracy table ────────────────────────────────────────
    print("\nConfidence--accuracy (top-1 prob bins):")
    bins = [(0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        mask = (results["gp_top1_prob"] >= lo) & (results["gp_top1_prob"] < hi)
        if mask.sum() == 0:
            continue
        acc_bin = results.loc[mask, "correct"].mean()
        n_bin   = mask.sum()
        print(f"  [{lo:.1f},{hi:.2f}): acc={acc_bin:.3f}  N={n_bin}")

    # ── Interpretation ────────────────────────────────────────────────────
    print()
    print("Interpretation:")
    print("  - Top-1 Acc > 0.50  : GP predicts correct class more than chance")
    print("    (baseline: 0.333 for 3-class uniform).")
    print("  - Brier < 0.222     : better than predicting uniform distribution.")
    print("  - ECE < 0.10        : well-calibrated confidence estimates.")
    print("  - 'No-obs' fraction : windows where GP fell back to OU prior")
    print("    (no other labelled windows for that subject).")
    print()

    # ── Class-bias warning ────────────────────────────────────────────────
    low_frac = (results["gp_class"] == 0).mean()
    if low_frac > 0.7:
        print("WARNING: GP argmax is Low for {:.1f}% of windows.".format(low_frac * 100))
        print("  This is expected: the OU prior mean (mu ~5.3) is below the tertile")
        print("  threshold (~6.0), so between-task intervals revert toward Low affect.")
        print("  The SoftVADLoss instance weights and aug_frac=0.3 limit the impact.")
        print("  Check: if aug_frac > 0.5 hurts performance, this bias is the cause.")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOO calibration evaluation of GP label augmentation."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset.pkl (labelled windows).")
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to write per-window CSV results.")
    parser.add_argument("--no-estimate-ou", action="store_true", default=False,
                        help="Skip local OU estimation; use Meng 2026 defaults only.")
    args = parser.parse_args()

    log.info("Loading dataset from %s …", args.dataset)
    df = pd.read_pickle(args.dataset)
    log.info("  %d labelled windows  |  %d subjects",
             len(df), df["subject_id"].nunique() if "subject_id" in df else -1)

    # ── OU parameters ──────────────────────────────────────────────────────
    if args.no_estimate_ou:
        ou_params = {d: OUParams(**MENG_OU_PARAMS[d]) for d in DIMS}
        log.info("Using Meng 2026 OU defaults (--no-estimate-ou).")
    else:
        log.info("Estimating OU parameters from self-reports …")
        ou_params = estimate_ou_params(df)

    for dim, p in ou_params.items():
        log.info("  %s: θ=%.4f  σ²=%.2f  μ=%.2f", dim, p.theta, p.sigma2, p.mu)

    # ── Tertile thresholds ─────────────────────────────────────────────────
    # Use training-split (T0+T1) thresholds to match actual training procedure.
    # Fallback to full-dataset thresholds if T0/T1 not present.
    train_df = df[df["task"].isin(["T0", "T1"])] if "task" in df.columns else df
    log.info("  Using T0+T1 as threshold reference (%d windows)", len(train_df))
    thresholds: dict[str, tuple[float, float]] = {}
    for dim in DIMS:
        ref_df = train_df if len(train_df) >= 10 else df
        vals = ref_df[dim].dropna().values.astype(float)
        thresholds[dim] = compute_tertile_thresholds(vals)
        log.info("  %s thresholds (train): (%.2f, %.2f)", dim, *thresholds[dim])

    # ── LOO evaluation ─────────────────────────────────────────────────────
    log.info("Running LOO calibration evaluation …")
    results = loo_evaluate(df, ou_params, thresholds)
    log.info("  %d predictions generated.", len(results))

    # ── Report ──────────────────────────────────────────────────────────────
    print_summary(results)

    # ── Optional CSV output ─────────────────────────────────────────────────
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(str(out), index=False)
        log.info("Per-window results written to %s", out)


if __name__ == "__main__":
    main()
