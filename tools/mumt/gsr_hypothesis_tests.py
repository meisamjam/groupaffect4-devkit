"""
GSR-SAM Arousal Hypothesis Tests for MuMT-Affect.

Tests hypotheses for why window-level GSR (rho~0.20) does not predict SAM arousal well:
  H1 - Temporal mismatch: task-level GSR aggregate vs window-level
  H2 - Within-subject consistency: does rank order across tasks align?
  H3 - Peak-end bias: peak-window GSR per task vs SAM
  H4 - T0 normalization: subtract per-subject T0 baseline (confounded by introduction stress)
  H4b - Per-subject global mean normalization: subtract per-subject mean across ALL windows
  H4c - Per-subject 10th-percentile normalization: subtract per-subject p10 across ALL windows
  H4d - Per-subject Z-score normalization: (val - mean) / std across ALL windows
  H5 - SAM-free physiological GSR labels: best normalization -> [1,9] scale, compare to SAM

Also generates normalized SAM-free GSR arousal labels for downstream training.

Usage:
    python tools/mumt/gsr_hypothesis_tests.py \
        --dataset data/mumt/dataset_15s.pkl \
        --output-labels results/gsr_physio_labels.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

EDA_KEYS = [
    "eda_phasic_mean", "eda_phasic_std", "eda_tonic_mean", "eda_tonic_std",
    "scr_peak_count", "scr_amplitude_mean", "scr_amplitude_std",
    "hr_mean_mean", "hrv_rmssd_mean", "temp_skin_mean",
]

# Features that positively track arousal in the literature
AROUSAL_POSITIVE_KEYS = ["eda_phasic_mean", "scr_peak_count", "scr_amplitude_mean", "hr_mean_mean"]
# Features that negatively track arousal (high values = low arousal / parasympathetic)
AROUSAL_NEGATIVE_KEYS = ["hrv_rmssd_mean", "eda_tonic_mean"]


def extract_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Expand eda_features dict into flat columns on a copy of df."""
    out = df.copy()
    for k in EDA_KEYS:
        out[k] = out["eda_features"].apply(
            lambda d: d.get(k, np.nan) if isinstance(d, dict) else np.nan
        )
    return out


def composite_arousal(row: pd.Series) -> float:
    """Simple physiological composite: mean of z-scored positive features minus z-scored negative."""
    pos = np.nanmean([row.get(k, np.nan) for k in AROUSAL_POSITIVE_KEYS])
    neg = np.nanmean([row.get(k, np.nan) for k in AROUSAL_NEGATIVE_KEYS])
    if not np.isfinite(pos):
        return np.nan
    neg = 0.0 if not np.isfinite(neg) else neg
    return pos - neg


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def rho_report(label: str, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        print(f"  {label}: n={mask.sum()} (too few)")
        return
    r, p = spearmanr(x[mask], y[mask])
    print(f"  {label}: rho={r:.3f}  p={p:.3f}  n={mask.sum()}")


def compute_subject_stats(
    df: pd.DataFrame, keys: list[str]
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Per-subject (mean, std, p10) across ALL windows for each key."""
    stats: dict[str, dict[str, tuple[float, float, float]]] = {}
    for subj, g in df.groupby("subject_id"):
        stats[str(subj)] = {}
        for k in keys:
            vals = g[k].values.astype(float)
            valid = vals[np.isfinite(vals)]
            if len(valid) < 3:
                stats[str(subj)][k] = (np.nan, np.nan, np.nan)
            else:
                stats[str(subj)][k] = (
                    float(np.mean(valid)),
                    float(np.std(valid)),
                    float(np.percentile(valid, 10)),
                )
    return stats


def apply_norm(
    df: pd.DataFrame,
    key: str,
    subject_stats: dict[str, dict[str, tuple[float, float, float]]],
    mode: str,  # "mean_sub", "p10_sub", "zscore"
) -> np.ndarray:
    """Apply per-subject normalization to a feature column."""
    out = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        subj = str(row["subject_id"])
        val = float(row[key])
        if not np.isfinite(val):
            continue
        s = subject_stats.get(subj, {}).get(key, (np.nan, np.nan, np.nan))
        mean, std, p10 = s
        if mode == "mean_sub":
            if np.isfinite(mean):
                out[i] = val - mean
        elif mode == "p10_sub":
            if np.isfinite(p10):
                out[i] = val - p10
        elif mode == "zscore":
            if np.isfinite(mean) and np.isfinite(std) and std > 1e-6:
                out[i] = (val - mean) / std
    return out


def physio_labels_from_normed(
    df: pd.DataFrame, normed_arr: np.ndarray
) -> np.ndarray:
    """Quantile-map per-subject T0-free normed values to [1,9] scale."""
    gsr_physio = np.full(len(df), np.nan)
    for subj, g_idx in df.groupby("subject_id").groups.items():
        idx_list = list(g_idx)
        vals = normed_arr[idx_list]
        valid = np.isfinite(vals)
        if valid.sum() < 3:
            continue
        ranks = np.argsort(np.argsort(vals[valid])).astype(float)
        scaled = ranks / max(1.0, valid.sum() - 1) * 8.0 + 1.0
        gsr_physio[np.array(idx_list)[valid]] = scaled
    return gsr_physio


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--output-labels", default="results/gsr_physio_labels.csv",
                        help="CSV of SAM-free T0-normalized GSR arousal labels")
    args = parser.parse_args()

    log.info("Loading dataset: %s", args.dataset)
    df_raw = pd.read_pickle(args.dataset)
    df = extract_eda(df_raw)
    log.info("  %d rows, %d labelled", len(df), df["arousal"].notna().sum())

    # -------------------------------------------------------------------------
    # BASELINE: window-level correlation (reproduce the known result)
    # -------------------------------------------------------------------------
    section("BASELINE: Window-level EDA vs SAM arousal")
    ar = df["arousal"].values.astype(float)
    for k in ["eda_phasic_mean", "scr_peak_count", "scr_amplitude_mean", "hr_mean_mean", "eda_tonic_mean"]:
        rho_report(k, df[k].values.astype(float), ar)

    # Composite
    comp = df.apply(composite_arousal, axis=1).values.astype(float)
    rho_report("composite (pos-neg)", comp, ar)

    # -------------------------------------------------------------------------
    # H1: Task-level aggregation (mean GSR per subject-task vs SAM)
    # -------------------------------------------------------------------------
    section("H1: Task-level aggregation — mean EDA per subject-task vs SAM")
    grp = df.groupby(["subject_id", "task"])
    task_rows = []
    for (subj, task), g in grp:
        sam_ar = float(g["arousal"].dropna().mean()) if g["arousal"].notna().any() else np.nan
        row = {"subject_id": subj, "task": task, "sam_arousal": sam_ar}
        for k in ["eda_phasic_mean", "scr_peak_count", "scr_amplitude_mean", "hr_mean_mean"]:
            row[k] = float(np.nanmean(g[k].values.astype(float)))
        row["composite"] = float(np.nanmean(g.apply(composite_arousal, axis=1).values.astype(float)))
        task_rows.append(row)
    task_df = pd.DataFrame(task_rows)
    ar_task = task_df["sam_arousal"].values.astype(float)
    print(f"  Unit of analysis: {len(task_df)} subject-task pairs")
    for k in ["eda_phasic_mean", "scr_peak_count", "scr_amplitude_mean", "composite"]:
        rho_report(f"task-mean {k}", task_df[k].values.astype(float), ar_task)

    # -------------------------------------------------------------------------
    # H2: Within-subject rank consistency across tasks
    # -------------------------------------------------------------------------
    section("H2: Within-subject rank correlation (EDA rank vs SAM rank across tasks)")
    subj_rhos = []
    for subj, g in task_df.groupby("subject_id"):
        if len(g) < 3:
            continue
        r, p = spearmanr(g["composite"].values, g["sam_arousal"].values)
        if np.isfinite(r):
            subj_rhos.append(r)
    print(f"  N subjects with >=3 tasks: {len(subj_rhos)}")
    if subj_rhos:
        print(f"  Mean within-subject rho: {np.mean(subj_rhos):.3f} +/- {np.std(subj_rhos):.3f}")
        print(f"  Median:                  {np.median(subj_rhos):.3f}")
        print(f"  Subjects rho > 0:        {sum(r > 0 for r in subj_rhos)}/{len(subj_rhos)}")

    # -------------------------------------------------------------------------
    # H3: Peak-end rule — peak-window GSR per subject-task vs SAM
    # -------------------------------------------------------------------------
    section("H3: Peak-end rule — peak GSR window per subject-task vs SAM")
    peak_rows = []
    for (subj, task), g in df.groupby(["subject_id", "task"]):
        sam_ar = float(g["arousal"].dropna().mean()) if g["arousal"].notna().any() else np.nan
        comp_vals = g.apply(composite_arousal, axis=1).values.astype(float)
        peak = float(np.nanmax(comp_vals)) if np.isfinite(comp_vals).any() else np.nan
        peak_rows.append({"subject_id": subj, "task": task, "sam_arousal": sam_ar, "peak_composite": peak})
    peak_df = pd.DataFrame(peak_rows)
    rho_report("peak composite vs SAM", peak_df["peak_composite"].values.astype(float),
               peak_df["sam_arousal"].values.astype(float))
    rho_report("task-mean composite vs SAM (same pairs)", task_df["composite"].values.astype(float), ar_task)

    # -------------------------------------------------------------------------
    # H4: T0 normalization — subtract individual T0 baseline
    # NOTE: T0 is the social-introduction task; EDA reflects meeting stress,
    # not a true neutral baseline. H4b/H4c/H4d test more principled baselines.
    # -------------------------------------------------------------------------
    section("H4: T0 normalization — subtract per-subject T0 EDA mean (confounded)")
    df = df.copy()
    df["composite"] = comp
    t0_baselines: dict[str, dict[str, float]] = {}
    for subj, g_s in df.groupby("subject_id"):
        t0 = g_s[g_s["task"] == "T0"]
        if len(t0) == 0:
            continue
        t0_baselines[str(subj)] = {
            k: float(np.nanmean(t0[k].values.astype(float)))
            for k in ["eda_phasic_mean", "scr_peak_count", "composite"]
        }

    for k in ["eda_phasic_mean", "scr_peak_count", "composite"]:
        def norm_val(row, key=k):
            subj = str(row["subject_id"])
            bl = t0_baselines.get(subj, {})
            baseline = bl.get(key, np.nan)
            raw = row[key]
            if not np.isfinite(raw) or not np.isfinite(baseline):
                return np.nan
            return raw - baseline
        normed = df.apply(norm_val, axis=1).values.astype(float)
        rho_report(f"T0-normed {k} vs SAM", normed, ar)

    print("\n  Task-level T0-normed:")
    t0_task_mean: dict[str, float] = {str(subj): t0_baselines.get(str(subj), {}).get("composite", np.nan)
                                       for subj in task_df["subject_id"]}
    t0_normed_task = np.array([
        row["composite"] - t0_task_mean.get(str(row["subject_id"]), np.nan)
        for _, row in task_df.iterrows()
    ], dtype=float)
    rho_report("task-level T0-normed composite vs SAM", t0_normed_task, ar_task)

    # -------------------------------------------------------------------------
    # H4b: Per-subject global mean normalization (all windows, not T0 only)
    # -------------------------------------------------------------------------
    section("H4b: Per-subject global mean normalization (all windows)")
    norm_keys = ["eda_phasic_mean", "scr_peak_count", "composite"]
    subj_stats = compute_subject_stats(df, norm_keys)
    print(f"  N subjects with stats: {len(subj_stats)}")
    for k in norm_keys:
        normed = apply_norm(df, k, subj_stats, "mean_sub")
        rho_report(f"  mean-sub {k} vs SAM", normed, ar)
    # Task-level
    print("\n  Task-level (composite, mean-sub):")
    normed_comp = apply_norm(df, "composite", subj_stats, "mean_sub")
    task_normed_h4b = np.array([
        float(np.nanmean(normed_comp[(df["subject_id"] == row["subject_id"]) &
                                      (df["task"] == row["task"]).values]))
        for _, row in task_df.iterrows()
    ], dtype=float)
    rho_report("  task-level mean-sub composite vs SAM", task_normed_h4b, ar_task)

    # -------------------------------------------------------------------------
    # H4c: Per-subject 10th-percentile normalization (floor estimate)
    # -------------------------------------------------------------------------
    section("H4c: Per-subject 10th-percentile normalization (quietest moments)")
    for k in norm_keys:
        normed = apply_norm(df, k, subj_stats, "p10_sub")
        rho_report(f"  p10-sub {k} vs SAM", normed, ar)
    print("\n  Task-level (composite, p10-sub):")
    normed_comp_p10 = apply_norm(df, "composite", subj_stats, "p10_sub")
    task_normed_h4c = np.array([
        float(np.nanmean(normed_comp_p10[(df["subject_id"] == row["subject_id"]) &
                                          (df["task"] == row["task"]).values]))
        for _, row in task_df.iterrows()
    ], dtype=float)
    rho_report("  task-level p10-sub composite vs SAM", task_normed_h4c, ar_task)

    # -------------------------------------------------------------------------
    # H4d: Per-subject Z-score normalization (scale-free, most robust)
    # -------------------------------------------------------------------------
    section("H4d: Per-subject Z-score normalization — (val - mean) / std")
    for k in norm_keys:
        normed = apply_norm(df, k, subj_stats, "zscore")
        rho_report(f"  z-score {k} vs SAM", normed, ar)
    print("\n  Task-level (composite, z-score):")
    normed_comp_z = apply_norm(df, "composite", subj_stats, "zscore")
    task_normed_h4d = np.array([
        float(np.nanmean(normed_comp_z[(df["subject_id"] == row["subject_id"]) &
                                        (df["task"] == row["task"]).values]))
        for _, row in task_df.iterrows()
    ], dtype=float)
    rho_report("  task-level z-score composite vs SAM", task_normed_h4d, ar_task)

    # Per-task breakdown for Z-score (most informative)
    print("\n  Per-task Z-score composite vs SAM:")
    print(f"  {'Task':<6} {'N':>5} {'rho':>8} {'p':>8}")
    print("  " + "-" * 30)
    for task_name in sorted(df["task"].unique()):
        mask_t = (df["task"] == task_name).values
        rho_report(f"  {task_name}", normed_comp_z[mask_t], ar[mask_t])

    # -------------------------------------------------------------------------
    # H5: SAM-free physiological GSR labels — compare all normalization methods
    # -------------------------------------------------------------------------
    section("H5: SAM-free GSR labels — all normalizations -> [1,9] scale")

    norm_variants = [
        ("T0-sub",    apply_norm(df, "composite", {
            str(subj): {"composite": (v.get("composite", np.nan), np.nan, np.nan)}
            for subj, v in t0_baselines.items()
        }, "mean_sub")),
        ("mean-sub",  normed_comp),
        ("p10-sub",   normed_comp_p10),
        ("z-score",   normed_comp_z),
    ]

    # For T0-sub, recompute properly from t0_baselines
    t0_normed_comp = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        subj = str(row["subject_id"])
        raw = float(row["composite"])
        bl = t0_baselines.get(subj, {}).get("composite", np.nan)
        if np.isfinite(raw) and np.isfinite(bl):
            t0_normed_comp[i] = raw - bl
    norm_variants[0] = ("T0-sub", t0_normed_comp)

    best_rho = -np.inf
    best_name = ""
    best_labels = None

    for name, normed_arr in norm_variants:
        physio = physio_labels_from_normed(df, normed_arr)
        mask_valid = np.isfinite(physio) & np.isfinite(ar)
        n_valid = mask_valid.sum()
        if n_valid < 5:
            print(f"  {name}: n={n_valid} (too few)")
            continue
        r, p = spearmanr(physio[mask_valid], ar[mask_valid])
        print(f"\n  [{name}] N={n_valid}  physio-labels vs SAM: rho={r:.3f}  p={p:.3f}")
        print(f"  Label stats: mean={np.nanmean(physio):.2f}  std={np.nanstd(physio):.2f}")
        print(f"  Per-task:")
        for task_name in sorted(df["task"].unique()):
            mask_t = (df["task"] == task_name).values
            rho_report(f"    {task_name}", physio[mask_t], ar[mask_t])
        if r > best_rho:
            best_rho = r
            best_name = name
            best_labels = physio

    print(f"\n  Best normalization: {best_name}  (rho={best_rho:.3f})")

    # -------------------------------------------------------------------------
    # Save SAM-free physio labels (best normalization method)
    # -------------------------------------------------------------------------
    out = pd.DataFrame({
        "session_id": df["session_id"].values,
        "seat": df["seat"].values,
        "task": df["task"].values,
        "subject_id": df["subject_id"].values,
        "window_index": np.arange(len(df)),
        "gsr_physio_arousal": best_labels,
        "sam_arousal": ar,
        "composite_raw": comp,
        "norm_method": best_name,
    })
    out.to_csv(args.output_labels, index=False)
    log.info("Saved SAM-free physio labels (%s) -> %s", best_name, args.output_labels)
    print(f"\n  Labels saved to: {args.output_labels}")
    print("  Use with: train_ordinal.py --gsr-arousal-labels <path> --exclude-eda")


if __name__ == "__main__":
    main()
