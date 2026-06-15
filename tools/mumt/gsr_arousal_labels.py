"""
GSR-to-Arousal label generator for MuMT-Affect.

Uses EmotiBit EDA/GSR features as a physiological arousal source:
  - Fits a Ridge regression: EDA features → SAM arousal (1–9 scale) on labelled windows
  - Reports LOO cross-validation performance (cross-check vs user labels)
  - Applies the full model to all windows in dataset + augmented pool
  - Saves CSVs for use with train_ordinal.py --gsr-arousal-labels

Usage:
    python tools/mumt/gsr_arousal_labels.py \
        --dataset data/mumt/dataset_15s.pkl \
        --pool data/mumt/augmented_pool.pkl \
        --output-dataset results/gsr_arousal_dataset.csv \
        --output-pool results/gsr_arousal_pool.csv

The CSVs contain columns: session_id, seat, task, window_index, gsr_arousal
For dataset rows: window_index is the integer row index in the dataset pkl.
For pool rows: window_index matches the pool DataFrame index.
"""

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

# EDA feature keys available in both dataset and pool
EDA_FEATURE_KEYS = [
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_tonic_mean",
    "eda_tonic_std",
    "scr_peak_count",
    "scr_amplitude_mean",
    "scr_amplitude_std",
]

# Strict GSR keys (exclude HR and skin temp which are co-located but distinct physiology)
GSR_ONLY_KEYS = [
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_tonic_mean",
    "eda_tonic_std",
    "scr_peak_count",
    "scr_amplitude_mean",
    "scr_amplitude_std",
]

# All EmotiBit EDA sensor keys (includes HR proxy and skin temp for broader model)
ALL_EDA_KEYS = [
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_tonic_mean",
    "eda_tonic_std",
    "scr_peak_count",
    "scr_amplitude_mean",
    "scr_amplitude_std",
    "hr_mean_mean",
    "hr_mean_std",
    "hrv_rmssd_mean",
    "hrv_rmssd_std",
    "temp_skin_mean",
    "temp_skin_std",
]


def extract_eda_features(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Extract EDA features into an (N, len(keys)) array; NaN where missing."""
    rows = []
    for _, r in df.iterrows():
        ef = r.get("eda_features", {})
        if not isinstance(ef, dict):
            ef = {}
        rows.append([ef.get(k, np.nan) for k in keys])
    return np.array(rows, dtype=np.float64)


def fit_gsr_arousal_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Pipeline:
    """Fit Ridge regression pipeline on complete cases."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(X[mask], y[mask])
    return pipe


def loo_crosscheck(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Leave-one-out evaluation of the GSR → SAM arousal regression."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    Xv, yv = X[mask], y[mask]
    n = len(Xv)
    preds = np.full(n, np.nan)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(Xv):
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(Xv[train_idx], yv[train_idx])
        preds[test_idx] = pipe.predict(Xv[test_idx])

    # Clip predictions to 1–9 Likert range
    preds = np.clip(preds, 1.0, 9.0)
    mae = float(np.mean(np.abs(preds - yv)))
    rho, pval = spearmanr(preds, yv)

    # 3-class accuracy (tertiles at 3.5 / 6.5)
    bins = np.array([3.5, 6.5])
    y_bin = np.digitize(yv, bins)
    p_bin = np.digitize(preds, bins)
    acc3 = float(np.mean(y_bin == p_bin))

    # Acc within 1 Likert step
    acc1 = float(np.mean(np.abs(preds - yv) <= 1.0))

    return {
        "n": n,
        "mae": mae,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "acc3": acc3,
        "acc_within_1": acc1,
        "loo_preds": preds,
        "loo_true": yv,
        "valid_mask": mask,
    }


def generate_labels(
    pipe: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    """Apply fitted pipeline; returns NaN for rows with missing EDA features."""
    labels = np.full(len(X), np.nan)
    valid = np.isfinite(X).all(axis=1)
    if valid.sum() > 0:
        raw = pipe.predict(X[valid])
        labels[valid] = np.clip(raw, 1.0, 9.0)
    return labels


def per_task_crosscheck(df: pd.DataFrame, gsr_pred: np.ndarray) -> None:
    """Print per-task Spearman between GSR arousal and SAM arousal."""
    print("\n--- Per-task cross-check (GSR-predicted vs SAM arousal) ---")
    print(f"{'Task':<6} {'N':>5} {'rho':>8} {'p':>8} {'MAE':>8} {'acc3':>6}")
    print("-" * 46)
    for task in sorted(df["task"].unique()):
        mask_t = (df["task"] == task).values
        ar = df.loc[mask_t, "arousal"].values.astype(float)
        gp = gsr_pred[mask_t]
        valid = np.isfinite(ar) & np.isfinite(gp)
        if valid.sum() < 5:
            continue
        rho, pval = spearmanr(gp[valid], ar[valid])
        mae = float(np.mean(np.abs(gp[valid] - ar[valid])))
        bins = np.array([3.5, 6.5])
        acc3 = float(np.mean(np.digitize(gp[valid], bins) == np.digitize(ar[valid], bins)))
        print(f"{task:<6} {valid.sum():>5} {rho:>8.3f} {pval:>8.3f} {mae:>8.3f} {acc3:>6.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate GSR-based arousal labels for MuMT-Affect.")
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool", default="data/mumt/augmented_pool.pkl")
    parser.add_argument("--output-dataset", default="results/gsr_arousal_dataset.csv")
    parser.add_argument("--output-pool", default="results/gsr_arousal_pool.csv")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    parser.add_argument("--feature-set", choices=["gsr_only", "all_eda"], default="all_eda",
                        help="gsr_only: EDA phasic/tonic/SCR; all_eda: includes HR proxy + skin temp")
    args = parser.parse_args()

    keys = GSR_ONLY_KEYS if args.feature_set == "gsr_only" else ALL_EDA_KEYS
    log.info("Using %d EDA features: %s", len(keys), keys)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    log.info("Loading dataset: %s", args.dataset)
    df = pd.read_pickle(args.dataset)
    log.info("  %d rows, tasks: %s", len(df), df["task"].value_counts().to_dict())

    log.info("Loading pool: %s", args.pool)
    pool = pd.read_pickle(args.pool)
    log.info("  %d rows, tasks: %s", len(pool), pool["task"].value_counts().to_dict())

    # -------------------------------------------------------------------------
    # Extract EDA features
    # -------------------------------------------------------------------------
    X_ds = extract_eda_features(df, keys)
    X_pool = extract_eda_features(pool, keys)

    has_eda_ds = np.isfinite(X_ds).all(axis=1)
    has_eda_pool = np.isfinite(X_pool).all(axis=1)
    log.info("Dataset: %d/%d rows have complete EDA features", has_eda_ds.sum(), len(df))
    log.info("Pool: %d/%d rows have complete EDA features", has_eda_pool.sum(), len(pool))

    # -------------------------------------------------------------------------
    # Cross-check on labelled dataset rows
    # -------------------------------------------------------------------------
    y_ds = df["arousal"].values.astype(float)
    labelled_mask = np.isfinite(y_ds)
    n_labelled = labelled_mask.sum()
    log.info("Labelled rows with SAM arousal: %d", n_labelled)

    print("\n" + "=" * 60)
    print("  GSR -> SAM Arousal Cross-Check (LOO on labelled windows)")
    print("=" * 60)
    print(f"Feature set : {args.feature_set}  ({len(keys)} features)")
    print(f"Ridge alpha : {args.alpha}")
    print(f"N labelled  : {n_labelled}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo_res = loo_crosscheck(X_ds[labelled_mask], y_ds[labelled_mask], alpha=args.alpha)

    print(f"\nLOO performance (GSR pred vs SAM arousal):")
    print(f"  N valid     : {loo_res['n']}")
    print(f"  MAE         : {loo_res['mae']:.3f}")
    print(f"  Spearman ρ  : {loo_res['spearman_rho']:.3f}  (p={loo_res['spearman_pval']:.3f})")
    print(f"  acc@1       : {loo_res['acc_within_1']:.3f}")
    print(f"  acc3        : {loo_res['acc3']:.3f}")
    print()

    # Per-task cross-check
    labelled_df = df[labelled_mask].copy()
    labelled_df = labelled_df.reset_index(drop=True)
    loo_preds_aligned = np.full(len(df), np.nan)
    labelled_idxs = np.where(labelled_mask)[0]
    valid_within_labelled = loo_res["valid_mask"]
    for i, orig_idx in enumerate(labelled_idxs[valid_within_labelled]):
        loo_preds_aligned[orig_idx] = loo_res["loo_preds"][valid_within_labelled][
            np.where(valid_within_labelled)[0] == i
        ][0] if False else loo_res["loo_preds"][valid_within_labelled.cumsum()[i] - 1]

    # Simpler aligned assignment
    loo_full = np.full(len(df), np.nan)
    valid_labelled_orig = labelled_idxs[valid_within_labelled]
    loo_full[valid_labelled_orig] = loo_res["loo_preds"]

    per_task_crosscheck(df, loo_full)

    # -------------------------------------------------------------------------
    # Fit final model on ALL labelled data
    # -------------------------------------------------------------------------
    log.info("Fitting final GSR→arousal model on all %d labelled windows...", n_labelled)
    pipe = fit_gsr_arousal_model(X_ds[labelled_mask], y_ds[labelled_mask], alpha=args.alpha)

    coefs = pipe.named_steps["ridge"].coef_
    print("\n--- Ridge coefficients (scaled features → arousal 1-9) ---")
    for k, c in sorted(zip(keys, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k:<30} {c:+.4f}")

    # -------------------------------------------------------------------------
    # Generate labels for all rows
    # -------------------------------------------------------------------------
    gsr_labels_ds = generate_labels(pipe, X_ds)
    gsr_labels_pool = generate_labels(pipe, X_pool)

    n_ds_valid = np.isfinite(gsr_labels_ds).sum()
    n_pool_valid = np.isfinite(gsr_labels_pool).sum()
    log.info("Generated GSR arousal labels: %d/%d dataset, %d/%d pool",
             n_ds_valid, len(df), n_pool_valid, len(pool))

    print(f"\nGSR arousal label statistics (dataset, valid rows):")
    v = gsr_labels_ds[np.isfinite(gsr_labels_ds)]
    print(f"  mean={v.mean():.2f}  std={v.std():.2f}  min={v.min():.2f}  max={v.max():.2f}")
    print(f"SAM arousal statistics (labelled rows):")
    sv = y_ds[labelled_mask]
    print(f"  mean={sv.mean():.2f}  std={sv.std():.2f}  min={sv.min():.2f}  max={sv.max():.2f}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    ds_out = pd.DataFrame({
        "session_id": df["session_id"].values,
        "seat": df["seat"].values,
        "task": df["task"].values,
        "subject_id": df["subject_id"].values,
        "window_index": np.arange(len(df)),
        "gsr_arousal": gsr_labels_ds,
        "sam_arousal": y_ds,
    })
    ds_out.to_csv(args.output_dataset, index=False)
    log.info("Saved dataset GSR labels → %s", args.output_dataset)

    pool_out = pd.DataFrame({
        "session_id": pool["session_id"].values,
        "seat": pool["seat"].values,
        "task": pool["task"].values,
        "subject_id": pool["subject_id"].values,
        "window_index": pool.index.values,
        "gsr_arousal": gsr_labels_pool,
    })
    pool_out.to_csv(args.output_pool, index=False)
    log.info("Saved pool GSR labels → %s", args.output_pool)

    print(f"\nOutputs written:")
    print(f"  {args.output_dataset}")
    print(f"  {args.output_pool}")


if __name__ == "__main__":
    main()
