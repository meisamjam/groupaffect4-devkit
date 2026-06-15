"""personality_session.py

Session-level personality prediction for MuMT-Affect / GroupAffect-4.

The window-level personality regression (R² < −15 across all models) fails
because 30-second physiological windows cannot reliably encode Big Five traits.
This script tests the FAIR alternative: aggregate all windows per subject,
then predict personality from the aggregated representation.

Two approaches:
  A. Mean-aggregate summary features per subject → Ridge regression per BFI trait.
  B. Mean-aggregate summary features per subject,
     per-task (5 task-level vectors) → Ridge regression.

Both use leave-one-subject-out CV (LOSO-CV) to estimate generalisation.

Output: per-trait R², Pearson r, MAE; comparison with window-level baseline.

Usage:
  python tools/mumt/personality_session.py --dataset data/mumt/dataset_15s.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import build_summary_key_order, flatten_features

BFI_TRAITS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
TRAIT_NAMES = {"bfi44_e": "Extraversion", "bfi44_a": "Agreeableness",
               "bfi44_c": "Conscientiousness", "bfi44_n": "Neuroticism",
               "bfi44_o": "Openness"}


def extract_summary_matrix(df: pd.DataFrame, key_order: list[str]) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features", "audio_features", "speech_features"]:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.nan_to_num(
        np.stack(rows, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def build_subject_features(
    df: pd.DataFrame,
    key_order: list[str],
    per_task: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate summary features per subject (± per task).

    Returns:
        X      : (N_subjects, K) or (N_subjects, 5*K) feature matrix
        Y      : (N_subjects, 5) BFI trait scores
        subj_ids: list of subject IDs
    """
    summary_mat = extract_summary_matrix(df, key_order)
    df_reset = df.reset_index(drop=True)

    subjects = sorted(df_reset["subject_id"].unique())
    tasks    = ["T0", "T1", "T2", "T3", "T4"]
    K = len(key_order)

    X_rows, Y_rows, subj_ids = [], [], []

    for subj in subjects:
        mask = df_reset["subject_id"] == subj
        if not mask.any():
            continue
        bfi_vals = df_reset.loc[mask, BFI_TRAITS].iloc[0].values.astype(float)
        if np.any(np.isnan(bfi_vals)):
            continue

        if per_task:
            feat_vec = []
            for task in tasks:
                task_mask = mask & (df_reset["task"] == task)
                if task_mask.any():
                    feat_vec.append(summary_mat[task_mask.values].mean(axis=0))
                else:
                    feat_vec.append(np.zeros(K, dtype=np.float32))
            X_rows.append(np.concatenate(feat_vec))
        else:
            X_rows.append(summary_mat[mask.values].mean(axis=0))

        Y_rows.append(bfi_vals)
        subj_ids.append(str(subj))

    X = np.stack(X_rows, axis=0)
    Y = np.stack(Y_rows, axis=0)
    return X, Y, subj_ids


def loso_cv_ridge(X: np.ndarray, Y: np.ndarray) -> dict[str, list[float]]:
    """Leave-one-subject-out CV with Ridge regression per BFI trait.

    Returns per-subject predictions stacked for overall R² / r / MAE.
    """
    N = len(X)
    preds = np.zeros_like(Y)

    for i in range(N):
        tr_idx = [j for j in range(N) if j != i]
        X_tr, Y_tr = X[tr_idx], Y[tr_idx]
        X_te = X[[i]]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 500.0])),
        ])
        pipe.fit(X_tr, Y_tr)
        preds[i] = pipe.predict(X_te)[0]

    results: dict[str, list[float]] = {"r2": [], "r": [], "mae": []}
    for t_i, trait in enumerate(BFI_TRAITS):
        y_true = Y[:, t_i]
        y_pred = preds[:, t_i]
        r2  = float(r2_score(y_true, y_pred))
        r   = float(pearsonr(y_true, y_pred)[0]) if np.std(y_true) > 1e-6 else 0.0
        mae = float(mean_absolute_error(y_true, y_pred))
        results["r2"].append(r2)
        results["r"].append(r)
        results["mae"].append(mae)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    log.info("Loaded %d windows, %d subjects", len(df), df["subject_id"].nunique())

    key_order = build_summary_key_order(df)

    print("\n" + "=" * 70)
    print(f"{'Session-Level Personality Prediction (LOSO-CV, Ridge)':^70}")
    print("=" * 70)

    for per_task, label in [(False, "Mean across all tasks"), (True, "Per-task concatenation")]:
        X, Y, subj_ids = build_subject_features(df, key_order, per_task=per_task)
        log.info("%s: X=%s  Y=%s  subjects=%d", label, X.shape, Y.shape, len(subj_ids))

        results = loso_cv_ridge(X, Y)

        print(f"\n-- {label} (N={len(subj_ids)} subjects, K={X.shape[1]} features) --")
        print(f"  {'Trait':<20}  {'R²':>8}  {'r':>8}  {'MAE':>8}")
        print("  " + "-" * 46)
        for t_i, trait in enumerate(BFI_TRAITS):
            name = TRAIT_NAMES[trait]
            r2  = results["r2"][t_i]
            r   = results["r"][t_i]
            mae = results["mae"][t_i]
            flag = " *" if r2 > 0 else ""
            print(f"  {name:<20}  {r2:>8.3f}  {r:>8.3f}  {mae:>8.3f}{flag}")
        print("  " + "-" * 46)
        mean_r2  = float(np.mean(results["r2"]))
        mean_r   = float(np.mean(results["r"]))
        mean_mae = float(np.mean(results["mae"]))
        print(f"  {'Mean':<20}  {mean_r2:>8.3f}  {mean_r:>8.3f}  {mean_mae:>8.3f}")

        n_pos = sum(1 for v in results["r2"] if v > 0)
        print(f"  Traits with R² > 0: {n_pos} / {len(BFI_TRAITS)}")
        if mean_r2 > -1.0:
            status = "RECOVERS" if mean_r2 > 0 else "PARTIALLY RECOVERS"
            print(f"  -> Session-level aggregation {status} "
                  f"personality prediction (vs window-level R2~-30)")

    print("=" * 70)
    print("\nInterpretation:")
    print("  Window-level: R2 ~ -15 to -48 (all models) -- physiology in 30s")
    print("  cannot encode stable personality traits.")
    print("  Session-level: aggregating all windows per person and using LOSO-CV")
    print("  is the minimum-fair experimental design for personality prediction.")
    print("  Positive R2 here would justify reframing personality as a session-level")
    print("  task in the paper rather than dropping it entirely.")


if __name__ == "__main__":
    main()
