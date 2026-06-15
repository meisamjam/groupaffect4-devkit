"""run_v2_experiments.py

Run the full comparison experiment suite on dataset_15s_v2.pkl
(improved gaze/pupil preprocessing + selected speech features).

Experiments:
  1. Unimodal speech SVM  — speech features only (6 features)
  2. Unimodal gaze SVM    — enhanced gaze features only (22)
  3. Unimodal pupil SVM   — enhanced pupil features only (18)
  4. Physio-only SVM      — gaze + pupil + EDA + PPG + IMU (no speech/audio)
  5. Physio + speech SVM  — all physio + 6 speech features
  6. Full (physio+speech) + AP1 augmentation
  7. Best config comparison: v2 vs original

Usage
-----
  python tools/mumt/run_v2_experiments.py
  python tools/mumt/run_v2_experiments.py --out results/v2_experiment_comparison.csv
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import flatten_features
from train_simple import (
    task_split,
    compute_tertile_thresholds,
    bin_vad_from_thresholds,
)
from svm_aug_comparison import (
    compute_bfi_similarity_map,
    get_pool_pseudo_labels_bfi,
    recompute_soft_labels,
)

VAD_DIMS = ["valence", "arousal", "dominance"]


def build_key_order(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    keys: set[str] = set()
    for col in feat_cols:
        if col not in df.columns:
            continue
        for v in df[col]:
            if isinstance(v, dict):
                keys.update(v.keys())
    return sorted(keys)


def build_X(df: pd.DataFrame, feat_cols: list[str], key_order: list[str]) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        merged: dict = {}
        for col in feat_cols:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                merged.update(fd)
        rows.append(flatten_features(merged, key_order=key_order))
    X = np.stack(rows, axis=0).astype(np.float32)
    return np.where(np.isfinite(X), X, np.nan)


def get_hard_labels(df: pd.DataFrame,
                    thresholds: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.zeros((len(df), 3), dtype=np.int64)
    for ci, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for ri, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if ci == 2 and not np.isfinite(v):
                out[ri, ci] = -1
            else:
                out[ri, ci] = bin_vad_from_thresholds(v, t1, t2)
    return out


def run_svm(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    results: dict[str, float] = {}
    for ci, dim in enumerate(VAD_DIMS):
        y_tr = y_train[:, ci]
        y_te = y_test[:, ci]
        tr_valid = y_tr >= 0
        te_valid = y_te >= 0
        X_tr_d, y_tr_d = X_train[tr_valid], y_tr[tr_valid]
        X_te_d, y_te_d = X_test[te_valid], y_te[te_valid]

        if len(np.unique(y_tr_d)) < 2 or len(X_te_d) == 0:
            results[f"{dim}_f1"] = 0.0
            continue

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr_d, y_tr_d)
        preds = pipe.predict(X_te_d)
        results[f"{dim}_f1"] = float(f1_score(y_te_d, preds, average="macro", zero_division=0))
    f1s = [results[f"{d}_f1"] for d in VAD_DIMS]
    results["mean_f1"] = float(np.mean(f1s))
    return results


def get_ap1_augmentation(pool_df: pd.DataFrame, train_df: pd.DataFrame,
                         feat_cols: list[str], key_order: list[str],
                         thresholds: dict[str, tuple[float, float]],
                         conf: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    bfi_sim_map = compute_bfi_similarity_map(train_df)
    pool_train = pool_df[pool_df["task"].isin(["T0", "T1"])].reset_index(drop=True)
    labels, weights, mask = get_pool_pseudo_labels_bfi(
        pool_train, thresholds, conf, bfi_sim_map, use_bfi_only=True,
    )
    X_list, Y_list = [], []
    for i, (_, row) in enumerate(pool_train.iterrows()):
        if not mask[i].any():
            continue
        merged: dict = {}
        for col in feat_cols:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                merged.update(fd)
        x = flatten_features(merged, key_order=key_order).astype(np.float32)
        x = np.where(np.isfinite(x.astype(float)), x, np.nan)
        X_list.append(x)
        Y_list.append(labels[i])
    if not X_list:
        return np.empty((0, len(key_order)), dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    return np.stack(X_list), np.stack(Y_list)


def main() -> None:
    p = argparse.ArgumentParser(description="Run v2 preprocessing comparison experiments")
    p.add_argument("--v2-dataset", default="data/mumt/dataset_15s_v2.pkl")
    p.add_argument("--orig-dataset", default="data/mumt/dataset_15s.pkl")
    p.add_argument("--pool", default="data/mumt/augmented_pool_slow.pkl")
    p.add_argument("--out", default="results/v2_experiment_comparison.csv")
    args = p.parse_args()

    # Load datasets
    log.info("Loading v2 dataset: %s", args.v2_dataset)
    with open(args.v2_dataset, "rb") as f:
        df_v2 = pickle.load(f)

    log.info("Loading original dataset: %s", args.orig_dataset)
    with open(args.orig_dataset, "rb") as f:
        df_orig = pickle.load(f)

    pool_df = None
    if Path(args.pool).exists():
        with open(args.pool, "rb") as f:
            pool_df = pickle.load(f)
        log.info("Pool: %d windows", len(pool_df))

    # Define feature groups
    PHYSIO_COLS = ["gaze_features", "pupil_features", "eda_features", "ppg_features", "imu_features"]
    SPEECH_COLS = ["speech_features"]

    records: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════
    # ORIGINAL DATASET (baseline)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n{'='*60}\nORIGINAL DATASET (physio-only, 49 features)\n{'='*60}")
    train_orig, _, test_orig = task_split(df_orig, test_task="T3")
    thresh_orig = compute_tertile_thresholds(train_orig)
    ko_orig = build_key_order(df_orig, PHYSIO_COLS)

    X_tr_orig = build_X(train_orig, PHYSIO_COLS, ko_orig)
    y_tr_orig = get_hard_labels(train_orig, thresh_orig)
    X_te_orig = build_X(test_orig, PHYSIO_COLS, ko_orig)
    y_te_orig = get_hard_labels(test_orig, thresh_orig)

    res = run_svm(X_tr_orig, y_tr_orig, X_te_orig, y_te_orig)
    records.append({"dataset": "original", "variant": "physio", "augmentation": "none",
                    "n_features": len(ko_orig), **res})
    log.info("  Original physio (no aug): V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # Original + AP1
    if pool_df is not None:
        X_aug, Y_aug = get_ap1_augmentation(pool_df, train_orig, PHYSIO_COLS, ko_orig, thresh_orig)
        if len(X_aug):
            X_tr_ap1 = np.concatenate([X_tr_orig, X_aug], axis=0)
            y_tr_ap1 = np.concatenate([y_tr_orig, Y_aug], axis=0)
            res = run_svm(X_tr_ap1, y_tr_ap1, X_te_orig, y_te_orig)
            records.append({"dataset": "original", "variant": "physio", "augmentation": "AP1",
                            "n_features": len(ko_orig), **res})
            log.info("  Original physio + AP1:    V=%.3f A=%.3f D=%.3f mean=%.3f",
                     res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # ═══════════════════════════════════════════════════════════════════════
    # V2 DATASET — individual modalities
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n{'='*60}\nV2 DATASET (improved preprocessing)\n{'='*60}")
    train_v2, _, test_v2 = task_split(df_v2, test_task="T3")
    thresh_v2 = compute_tertile_thresholds(train_v2)

    # Unimodal: speech only (6 features)
    ko_speech = build_key_order(df_v2, SPEECH_COLS)
    X_tr = build_X(train_v2, SPEECH_COLS, ko_speech)
    y_tr = get_hard_labels(train_v2, thresh_v2)
    X_te = build_X(test_v2, SPEECH_COLS, ko_speech)
    y_te = get_hard_labels(test_v2, thresh_v2)
    res = run_svm(X_tr, y_tr, X_te, y_te)
    records.append({"dataset": "v2", "variant": "speech_only", "augmentation": "none",
                    "n_features": len(ko_speech), **res})
    log.info("  v2 speech only (6 feat):  V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # Unimodal: gaze only
    ko_gaze = build_key_order(df_v2, ["gaze_features"])
    X_tr = build_X(train_v2, ["gaze_features"], ko_gaze)
    X_te = build_X(test_v2, ["gaze_features"], ko_gaze)
    res = run_svm(X_tr, y_tr, X_te, y_te)
    records.append({"dataset": "v2", "variant": "gaze_only", "augmentation": "none",
                    "n_features": len(ko_gaze), **res})
    log.info("  v2 gaze only (22 feat):   V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # Unimodal: pupil only
    ko_pupil = build_key_order(df_v2, ["pupil_features"])
    X_tr = build_X(train_v2, ["pupil_features"], ko_pupil)
    X_te = build_X(test_v2, ["pupil_features"], ko_pupil)
    res = run_svm(X_tr, y_tr, X_te, y_te)
    records.append({"dataset": "v2", "variant": "pupil_only", "augmentation": "none",
                    "n_features": len(ko_pupil), **res})
    log.info("  v2 pupil only (18 feat):  V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # V2 physio only (improved gaze/pupil + same EDA/PPG/IMU)
    ko_v2_physio = build_key_order(df_v2, PHYSIO_COLS)
    X_tr = build_X(train_v2, PHYSIO_COLS, ko_v2_physio)
    X_te = build_X(test_v2, PHYSIO_COLS, ko_v2_physio)
    res = run_svm(X_tr, y_tr, X_te, y_te)
    records.append({"dataset": "v2", "variant": "physio", "augmentation": "none",
                    "n_features": len(ko_v2_physio), **res})
    log.info("  v2 physio (no aug):       V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # V2 physio + speech
    ALL_COLS = PHYSIO_COLS + SPEECH_COLS
    ko_v2_all = build_key_order(df_v2, ALL_COLS)
    X_tr = build_X(train_v2, ALL_COLS, ko_v2_all)
    X_te = build_X(test_v2, ALL_COLS, ko_v2_all)
    res = run_svm(X_tr, y_tr, X_te, y_te)
    records.append({"dataset": "v2", "variant": "physio+speech", "augmentation": "none",
                    "n_features": len(ko_v2_all), **res})
    log.info("  v2 physio+speech (no aug):V=%.3f A=%.3f D=%.3f mean=%.3f",
             res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # V2 physio + AP1
    if pool_df is not None:
        ko_v2_physio = build_key_order(df_v2, PHYSIO_COLS)
        X_tr = build_X(train_v2, PHYSIO_COLS, ko_v2_physio)
        X_te = build_X(test_v2, PHYSIO_COLS, ko_v2_physio)
        X_aug, Y_aug = get_ap1_augmentation(pool_df, train_v2, PHYSIO_COLS, ko_v2_physio, thresh_v2)
        if len(X_aug):
            X_tr_ap1 = np.concatenate([X_tr, X_aug], axis=0)
            y_tr_ap1 = np.concatenate([y_tr, Y_aug], axis=0)
            res = run_svm(X_tr_ap1, y_tr_ap1, X_te, y_te)
            records.append({"dataset": "v2", "variant": "physio", "augmentation": "AP1",
                            "n_features": len(ko_v2_physio), **res})
            log.info("  v2 physio + AP1:          V=%.3f A=%.3f D=%.3f mean=%.3f",
                     res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # V2 physio + speech + AP1
    if pool_df is not None:
        X_tr = build_X(train_v2, ALL_COLS, ko_v2_all)
        X_te = build_X(test_v2, ALL_COLS, ko_v2_all)
        X_aug, Y_aug = get_ap1_augmentation(pool_df, train_v2, ALL_COLS, ko_v2_all, thresh_v2)
        if len(X_aug):
            X_tr_ap1 = np.concatenate([X_tr, X_aug], axis=0)
            y_tr_ap1 = np.concatenate([y_tr, Y_aug], axis=0)
            res = run_svm(X_tr_ap1, y_tr_ap1, X_te, y_te)
            records.append({"dataset": "v2", "variant": "physio+speech", "augmentation": "AP1",
                            "n_features": len(ko_v2_all), **res})
            log.info("  v2 physio+speech + AP1:   V=%.3f A=%.3f D=%.3f mean=%.3f",
                     res["valence_f1"], res["arousal_f1"], res["dominance_f1"], res["mean_f1"])

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(out_path, index=False, float_format="%.4f")
    log.info("\nResults saved to: %s", out_path)

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'DATASET':<10} {'VARIANT':<18} {'AUG':<6} {'V F1':>7} {'A F1':>7} {'D F1':>7} {'Mean':>7} {'N':>4}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<10} {row['variant']:<18} {row['augmentation']:<6} "
              f"{row['valence_f1']:7.3f} {row['arousal_f1']:7.3f} "
              f"{row['dominance_f1']:7.3f} {row['mean_f1']:7.3f} "
              f"{int(row['n_features']):4d}")
    print("=" * 85)

    # Highlight best
    best = results_df.loc[results_df["mean_f1"].idxmax()]
    orig_best = results_df[results_df["dataset"] == "original"]["mean_f1"].max()
    v2_best = results_df[results_df["dataset"] == "v2"]["mean_f1"].max()
    print(f"\nOriginal best: mean F1 = {orig_best:.3f}")
    print(f"V2 best:       mean F1 = {v2_best:.3f}")
    print(f"Improvement:   {v2_best - orig_best:+.3f}")


if __name__ == "__main__":
    main()
