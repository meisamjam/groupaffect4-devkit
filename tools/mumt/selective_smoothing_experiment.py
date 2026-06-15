"""selective_smoothing_experiment.py

Test whether modality-selective feature smoothing can improve V/A/D simultaneously.

Hypothesis: full smoothing helps Arousal (+0.105 with AP1) but hurts Valence
(-0.066 to -0.091) because it blurs fine-grained gaze/pupil patterns.
By smoothing ONLY EDA+PPG (slow phasic signals) while preserving gaze/pupil/IMU,
we may retain Valence while gaining Arousal.

Conditions tested:
  1. 15s baseline (A0, AP1)                   — reference
  2. Full smooth fwd + AP1                    — current best for Arousal
  3. Physio-only smooth fwd (EDA+PPG) + AP1   — hypothesis: V preserved, A improved
  4. Physio-only smooth centered + AP1        — same but 45s context
  5. Physio+IMU smooth fwd + AP1              — IMU may also benefit from smoothing
  6. Gaze/pupil-only smooth fwd + AP1         — control: should HURT Valence

Usage:
  python tools/mumt/selective_smoothing_experiment.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import build_summary_key_order, flatten_features  # noqa: E402
from train_simple import (  # noqa: E402
    bin_vad_from_thresholds,
    compute_tertile_thresholds,
    task_split,
)
from svm_aug_comparison import (  # noqa: E402
    VAD_DIMS,
    FEAT_COLS,
    BFI_COLS,
    extract_X,
    get_hard_labels,
    run_variant,
    get_pool_pseudo_labels_bfi,
    compute_bfi_similarity_map,
)


def selective_smooth(
    df: pd.DataFrame,
    mode: str = "forward",
    smooth_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Smooth only specified feature columns, leaving others untouched.

    Parameters
    ----------
    df : DataFrame with feature dict columns
    mode : "forward" (avg i, i+1) or "centered" (avg i-1, i, i+1)
    smooth_cols : subset of FEAT_COLS to smooth. If None, smooths all.
    """
    if smooth_cols is None:
        smooth_cols = list(FEAT_COLS)

    group_cols = ["session_id", "subject_id", "task"]
    sort_col = "vad_timestamp_lsl"
    df_out = df.copy()

    for _, grp in df.groupby(group_cols, sort=False):
        if sort_col not in grp.columns:
            continue
        grp_sorted = grp.sort_values(sort_col)
        orig_idx = grp_sorted.index.tolist()

        n = len(orig_idx)
        for pos in range(n):
            if mode == "forward":
                positions = [pos, pos + 1] if pos + 1 < n else [pos]
            elif mode == "backward":
                positions = [pos - 1, pos] if pos - 1 >= 0 else [pos]
            else:  # centered
                positions = [p for p in [pos - 1, pos, pos + 1] if 0 <= p < n]

            rows = [grp_sorted.iloc[p] for p in positions]
            # Only smooth the selected columns
            for fc in smooth_cols:
                merged: dict[str, float] = {}
                for row in rows:
                    fd = row.get(fc, {}) or {}
                    for k, v in fd.items():
                        merged[k] = merged.get(k, 0.0) + float(v)
                m = len(rows)
                df_out.at[orig_idx[pos], fc] = {k: v / m for k, v in merged.items()}

    return df_out


def run_condition(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    smooth_cols: list[str] | None,
    mode: str,
    variant: str,
    test_task: str = "T3",
    label: str = "",
) -> dict:
    """Run a single smoothing condition and return per-dim F1."""
    # Apply selective smoothing
    if smooth_cols is not None:
        df_s = selective_smooth(df, mode=mode, smooth_cols=smooth_cols)
        pool_s = selective_smooth(pool, mode=mode, smooth_cols=smooth_cols)
    else:
        df_s = df
        pool_s = pool

    train_df, _, test_df = task_split(df_s, test_task=test_task)
    thresholds = compute_tertile_thresholds(train_df)
    key_order = build_summary_key_order(df_s)
    bfi_sim_map = compute_bfi_similarity_map(train_df)

    train_X = extract_X(train_df, key_order)
    test_X = extract_X(test_df, key_order)
    train_labels = get_hard_labels(train_df, thresholds)
    test_labels = get_hard_labels(test_df, thresholds)

    aug_X, aug_lab, aug_w = None, None, None
    if variant == "AP1":
        p = pool_s[pool_s["task"].isin(set(train_df["task"].unique()))].copy()
        p = p.reset_index(drop=True)
        aug_X = extract_X(p, key_order)
        aug_lab, aug_w, _ = get_pool_pseudo_labels_bfi(
            p, thresholds, 0.5, bfi_sim_map, use_bfi_only=True
        )

    r = run_variant(train_X, train_labels, test_X, test_labels, aug_X, aug_lab, aug_w)
    return {
        "condition": label,
        "smooth_mode": mode,
        "variant": variant,
        "smooth_targets": ",".join(smooth_cols) if smooth_cols else "none",
        "V": round(r["valence"], 3),
        "A": round(r["arousal"], 3),
        "D": round(r["dominance"], 3),
        "mean": round(np.mean([r["valence"], r["arousal"], r["dominance"]]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool", default="data/mumt/augmented_pool_slow.pkl")
    parser.add_argument("--test-task", default="T3")
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    pool = pd.read_pickle(args.pool)
    log.info("Dataset: %d windows | Pool: %d windows", len(df), len(pool))

    conditions: list[dict] = []

    # ── Reference: 15s baselines ──────────────────────────────────────────────
    log.info("Running 15s baselines...")
    conditions.append(run_condition(df, pool, None, "none", "A0", label="15s baseline"))
    conditions.append(run_condition(df, pool, None, "none", "AP1", label="15s + AP1"))

    # ── Full smoothing (all features) — reference ─────────────────────────────
    log.info("Running full smoothing (all features)...")
    conditions.append(run_condition(
        df, pool, list(FEAT_COLS), "forward", "AP1", label="Full-smooth fwd + AP1"
    ))
    conditions.append(run_condition(
        df, pool, list(FEAT_COLS), "centered", "AP1", label="Full-smooth ctr + AP1"
    ))

    # ── Physio-only smoothing (EDA + PPG) ─────────────────────────────────────
    physio_cols = ["eda_features", "ppg_features"]
    log.info("Running physio-only (EDA+PPG) smoothing...")
    conditions.append(run_condition(
        df, pool, physio_cols, "forward", "A0", label="Physio-smooth fwd"
    ))
    conditions.append(run_condition(
        df, pool, physio_cols, "forward", "AP1", label="Physio-smooth fwd + AP1"
    ))
    conditions.append(run_condition(
        df, pool, physio_cols, "centered", "AP1", label="Physio-smooth ctr + AP1"
    ))

    # ── Physio + IMU smoothing ────────────────────────────────────────────────
    physio_imu_cols = ["eda_features", "ppg_features", "imu_features"]
    log.info("Running physio+IMU smoothing...")
    conditions.append(run_condition(
        df, pool, physio_imu_cols, "forward", "AP1", label="Physio+IMU-smooth fwd + AP1"
    ))
    conditions.append(run_condition(
        df, pool, physio_imu_cols, "centered", "AP1", label="Physio+IMU-smooth ctr + AP1"
    ))

    # ── Gaze/pupil-only smoothing (control — expect V hurt) ──────────────────
    gaze_cols = ["gaze_features", "pupil_features"]
    log.info("Running gaze/pupil-only smoothing (control)...")
    conditions.append(run_condition(
        df, pool, gaze_cols, "forward", "AP1", label="Gaze-smooth fwd + AP1 (control)"
    ))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SELECTIVE SMOOTHING EXPERIMENT — Per-Dimension Results")
    print("=" * 75)
    print(f"{'Condition':<35s} {'V':>6s} {'A':>6s} {'D':>6s}  {'Mean':>6s}")
    print("-" * 75)

    ref_v, ref_a, ref_d = None, None, None
    for c in conditions:
        tag = ""
        if ref_v is not None:
            dv = c["V"] - ref_v
            da = c["A"] - ref_a
            dd = c["D"] - ref_d
            tag = f"  Δ({dv:+.3f},{da:+.3f},{dd:+.3f})"
        else:
            ref_v, ref_a, ref_d = c["V"], c["A"], c["D"]

        print(f"{c['condition']:<35s} {c['V']:6.3f} {c['A']:6.3f} {c['D']:6.3f}  {c['mean']:6.3f}{tag}")

    # Save
    out_path = Path("results/selective_smoothing.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conditions).to_csv(out_path, index=False)
    log.info("\nSaved → %s", out_path)

    # ── Pareto analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PARETO-OPTIMAL CONDITIONS (not dominated on all 3 dims)")
    print("=" * 75)
    pareto = []
    for i, ci in enumerate(conditions):
        dominated = False
        for j, cj in enumerate(conditions):
            if i == j:
                continue
            if (cj["V"] >= ci["V"] and cj["A"] >= ci["A"] and cj["D"] >= ci["D"]
                    and (cj["V"] > ci["V"] or cj["A"] > ci["A"] or cj["D"] > ci["D"])):
                dominated = True
                break
        if not dominated:
            pareto.append(ci)

    for c in sorted(pareto, key=lambda x: -x["mean"]):
        print(f"  {c['condition']:<35s} V={c['V']:.3f}  A={c['A']:.3f}  D={c['D']:.3f}")


if __name__ == "__main__":
    main()
