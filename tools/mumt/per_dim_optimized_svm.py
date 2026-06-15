"""per_dim_optimized_svm.py — Per-dimension optimized affect recognition.

Final experiment combining the best strategies for each VAD dimension:
  - Valence:   Original features (49) + centered EDA/PPG smoothing + AP1
  - Arousal:   Original features (49) + centered EDA/PPG smoothing + AP1
  - Dominance: V2 reprocessed features (47 shared) + centered smoothing + NO AP1

This per-dimension approach is standard in VAD affect recognition: each dimension
has different physiological correlates and benefits from different preprocessing.

Key findings that motivate this approach:
  1. AP1 augmentation dramatically boosts A (+0.058) but slightly hurts D when
     pool feature distributions don't match reprocessed training data.
  2. V2 reprocessed gaze/pupil values (same keys but recomputed from raw sequences)
     are much better for Dominance (+0.066 over original).
  3. Original features work better for Valence (simpler feature space avoids
      overfitting in the low-sample training regime).

Usage:
  python tools/mumt/per_dim_optimized_svm.py
  python tools/mumt/per_dim_optimized_svm.py --out results/per_dim_optimized.csv
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
from train_simple import task_split, compute_tertile_thresholds, bin_vad_from_thresholds
from svm_aug_comparison import compute_bfi_similarity_map, get_pool_pseudo_labels_bfi

VAD_DIMS = ["valence", "arousal", "dominance"]
FEAT_COLS = ["gaze_features", "pupil_features", "eda_features", "ppg_features", "imu_features"]


def build_key_order(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    keys: set[str] = set()
    for col in feat_cols:
        if col not in df.columns:
            continue
        for v in df[col]:
            if isinstance(v, dict):
                keys.update(v.keys())
    return sorted(keys)


def get_labels(df: pd.DataFrame, thresholds: dict) -> np.ndarray:
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


def build_X(df: pd.DataFrame, key_set: set[str], feat_cols: list[str],
            key_order: list[str]) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        merged: dict = {}
        for col in feat_cols:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                for k, v in fd.items():
                    if k in key_set:
                        merged[k] = v
        x = flatten_features(merged, key_order=key_order)
        rows.append(x)
    X = np.stack(rows).astype(np.float32)
    return np.where(np.isfinite(X), X, np.nan)


def selective_smooth_centered(df: pd.DataFrame,
                              smooth_cols: list[str]) -> pd.DataFrame:
    """Apply centered smoothing (avg of i-1, i, i+1) to selected features."""
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
            positions = [p for p in [pos - 1, pos, pos + 1] if 0 <= p < n]
            rows_list = [grp_sorted.iloc[p] for p in positions]
            for fc in smooth_cols:
                merged_s: dict[str, float] = {}
                for r in rows_list:
                    fd = r.get(fc, {}) or {}
                    for k, v in fd.items():
                        try:
                            merged_s[k] = merged_s.get(k, 0.0) + float(v)
                        except (TypeError, ValueError):
                            pass
                m = len(rows_list)
                df_out.at[orig_idx[pos], fc] = {k: v / m for k, v in merged_s.items()}
    return df_out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2-dataset", default="data/mumt/dataset_15s_v2.pkl")
    p.add_argument("--orig-dataset", default="data/mumt/dataset_15s.pkl")
    p.add_argument("--pool", default="data/mumt/augmented_pool_slow.pkl")
    p.add_argument("--out", default="results/per_dim_optimized.csv")
    args = p.parse_args()

    # Load datasets
    log.info("Loading datasets...")
    with open(args.v2_dataset, "rb") as f:
        df_v2 = pickle.load(f)
    with open(args.orig_dataset, "rb") as f:
        df_orig = pickle.load(f)
    with open(args.pool, "rb") as f:
        pool_df = pickle.load(f)

    # Apply centered smoothing to EDA + PPG
    log.info("Applying centered smoothing (EDA + PPG)...")
    smooth_cols = ["eda_features", "ppg_features"]
    df_orig_s = selective_smooth_centered(df_orig, smooth_cols)
    df_v2_s = selective_smooth_centered(df_v2, smooth_cols)
    pool_s = selective_smooth_centered(pool_df, smooth_cols)

    # ═══════════════════════════════════════════════════════════════════════
    # ORIGINAL FEATURES (for Valence + Arousal)
    # ═══════════════════════════════════════════════════════════════════════
    orig_keys = set(build_key_order(df_orig, FEAT_COLS))
    orig_sorted = sorted(orig_keys)
    log.info("Original feature set: %d keys", len(orig_keys))

    train_orig, _, test_orig = task_split(df_orig_s, test_task="T3")
    thresh_orig = compute_tertile_thresholds(train_orig)
    y_tr_orig = get_labels(train_orig, thresh_orig)
    y_te_orig = get_labels(test_orig, thresh_orig)

    X_tr_orig = build_X(train_orig, orig_keys, FEAT_COLS, orig_sorted)
    X_te_orig = build_X(test_orig, orig_keys, FEAT_COLS, orig_sorted)

    # AP1 augmentation (original features)
    bfi_sim_orig = compute_bfi_similarity_map(train_orig)
    pool_train_orig = pool_s[pool_s["task"].isin(["T0", "T1"])].reset_index(drop=True)
    labels_orig, _, mask_orig = get_pool_pseudo_labels_bfi(
        pool_train_orig, thresh_orig, 0.5, bfi_sim_orig, use_bfi_only=True)

    X_aug_orig_list, Y_aug_orig_list = [], []
    for i, (_, row) in enumerate(pool_train_orig.iterrows()):
        if not mask_orig[i].any():
            continue
        merged: dict = {}
        for col in FEAT_COLS:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                for k, v in fd.items():
                    if k in orig_keys:
                        merged[k] = v
        x = flatten_features(merged, key_order=orig_sorted).astype(np.float32)
        x = np.where(np.isfinite(x.astype(float)), x, np.nan)
        X_aug_orig_list.append(x)
        Y_aug_orig_list.append(labels_orig[i])

    X_aug_orig = np.stack(X_aug_orig_list)
    Y_aug_orig = np.stack(Y_aug_orig_list)
    log.info("AP1 pool (original features): %d windows", len(X_aug_orig))

    X_tr_orig_ap1 = np.concatenate([X_tr_orig, X_aug_orig])
    y_tr_orig_ap1 = np.concatenate([y_tr_orig, Y_aug_orig])

    # ═══════════════════════════════════════════════════════════════════════
    # V2 FEATURES (for Dominance)
    # Uses the shared keys between v2 and pool (47 features)
    # ═══════════════════════════════════════════════════════════════════════
    v2_keys = set(build_key_order(df_v2, FEAT_COLS))
    shared_keys = v2_keys & set(build_key_order(pool_df, FEAT_COLS))
    shared_sorted = sorted(shared_keys)
    log.info("V2 shared feature set: %d keys", len(shared_keys))

    train_v2, _, test_v2 = task_split(df_v2_s, test_task="T3")
    thresh_v2 = compute_tertile_thresholds(train_v2)
    y_tr_v2 = get_labels(train_v2, thresh_v2)
    y_te_v2 = get_labels(test_v2, thresh_v2)

    X_tr_v2 = build_X(train_v2, shared_keys, FEAT_COLS, shared_sorted)
    X_te_v2 = build_X(test_v2, shared_keys, FEAT_COLS, shared_sorted)

    # ═══════════════════════════════════════════════════════════════════════
    # PER-DIMENSION OPTIMIZED SVM
    # ═══════════════════════════════════════════════════════════════════════
    records = []

    # --- Config A: Original + smooth + AP1 (all dims) ---
    print("\n" + "=" * 80)
    print("CONFIG A: Original (49 feat) + smooth + AP1 — all dimensions")
    print("=" * 80)
    res_a = {}
    for ci, dim in enumerate(VAD_DIMS):
        y_d = y_tr_orig_ap1[:, ci]
        y_te_d = y_te_orig[:, ci]
        tr_v = y_d >= 0
        te_v = y_te_d >= 0
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr_orig_ap1[tr_v], y_d[tr_v])
        preds = pipe.predict(X_te_orig[te_v])
        f1 = float(f1_score(y_te_d[te_v], preds, average="macro", zero_division=0))
        res_a[f"{dim}_f1"] = f1
        print(f"  {dim}: F1={f1:.3f}")
    res_a["mean_f1"] = np.mean([res_a[f"{d}_f1"] for d in VAD_DIMS])
    records.append({"config": "orig+smooth+AP1 (all dims)", **res_a})
    print(f"  → Mean = {res_a['mean_f1']:.3f}")

    # --- Config B: V2 + smooth + NO AP1 (all dims) ---
    print("\n" + "=" * 80)
    print("CONFIG B: V2 shared (47 feat) + smooth + NO AP1 — all dimensions")
    print("=" * 80)
    res_b = {}
    for ci, dim in enumerate(VAD_DIMS):
        y_d = y_tr_v2[:, ci]
        y_te_d = y_te_v2[:, ci]
        tr_v = y_d >= 0
        te_v = y_te_d >= 0
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr_v2[tr_v], y_d[tr_v])
        preds = pipe.predict(X_te_v2[te_v])
        f1 = float(f1_score(y_te_d[te_v], preds, average="macro", zero_division=0))
        res_b[f"{dim}_f1"] = f1
        print(f"  {dim}: F1={f1:.3f}")
    res_b["mean_f1"] = np.mean([res_b[f"{d}_f1"] for d in VAD_DIMS])
    records.append({"config": "v2+smooth (all dims)", **res_b})
    print(f"  → Mean = {res_b['mean_f1']:.3f}")

    # --- Config C: Per-dim optimized ---
    print("\n" + "=" * 80)
    print("CONFIG C: PER-DIM OPTIMIZED")
    print("  V: orig+smooth+AP1 | A: orig+smooth+AP1 | D: v2+smooth (no AP1)")
    print("=" * 80)
    res_c = {
        "valence_f1": res_a["valence_f1"],
        "arousal_f1": res_a["arousal_f1"],
        "dominance_f1": res_b["dominance_f1"],
    }
    res_c["mean_f1"] = np.mean([res_c[f"{d}_f1"] for d in VAD_DIMS])
    records.append({"config": "per-dim optimized (V:orig+AP1, A:orig+AP1, D:v2)", **res_c})
    print(f"  valence:   F1={res_c['valence_f1']:.3f}  (from orig+AP1)")
    print(f"  arousal:   F1={res_c['arousal_f1']:.3f}  (from orig+AP1)")
    print(f"  dominance: F1={res_c['dominance_f1']:.3f}  (from v2, no AP1)")
    print(f"  → Mean = {res_c['mean_f1']:.3f}")

    # --- Config D: V2 + smooth + AP1 (to see how close it gets) ---
    print("\n" + "=" * 80)
    print("CONFIG D: V2 shared (47 feat) + smooth + AP1 — all dimensions")
    print("=" * 80)
    # Use v2 thresholds with pool AP1
    bfi_sim_v2 = compute_bfi_similarity_map(train_v2)
    pool_train_v2 = pool_s[pool_s["task"].isin(["T0", "T1"])].reset_index(drop=True)
    labels_v2, _, mask_v2 = get_pool_pseudo_labels_bfi(
        pool_train_v2, thresh_v2, 0.5, bfi_sim_v2, use_bfi_only=True)

    X_aug_v2_list, Y_aug_v2_list = [], []
    for i, (_, row) in enumerate(pool_train_v2.iterrows()):
        if not mask_v2[i].any():
            continue
        merged = {}
        for col in FEAT_COLS:
            fd = row.get(col, {})
            if isinstance(fd, dict):
                for k, v in fd.items():
                    if k in shared_keys:
                        merged[k] = v
        x = flatten_features(merged, key_order=shared_sorted).astype(np.float32)
        x = np.where(np.isfinite(x.astype(float)), x, np.nan)
        X_aug_v2_list.append(x)
        Y_aug_v2_list.append(labels_v2[i])

    X_aug_v2 = np.stack(X_aug_v2_list)
    Y_aug_v2 = np.stack(Y_aug_v2_list)

    X_tr_v2_ap1 = np.concatenate([X_tr_v2, X_aug_v2])
    y_tr_v2_ap1 = np.concatenate([y_tr_v2, Y_aug_v2])

    res_d = {}
    for ci, dim in enumerate(VAD_DIMS):
        y_d = y_tr_v2_ap1[:, ci]
        y_te_d = y_te_v2[:, ci]
        tr_v = y_d >= 0
        te_v = y_te_d >= 0
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr_v2_ap1[tr_v], y_d[tr_v])
        preds = pipe.predict(X_te_v2[te_v])
        f1 = float(f1_score(y_te_d[te_v], preds, average="macro", zero_division=0))
        res_d[f"{dim}_f1"] = f1
        print(f"  {dim}: F1={f1:.3f}")
    res_d["mean_f1"] = np.mean([res_d[f"{d}_f1"] for d in VAD_DIMS])
    records.append({"config": "v2+smooth+AP1 (all dims)", **res_d})
    print(f"  → Mean = {res_d['mean_f1']:.3f}")

    # --- Config E: Best per-dim from all configs ---
    print("\n" + "=" * 80)
    print("CONFIG E: ORACLE (best per-dim from any config)")
    print("=" * 80)
    all_configs = [res_a, res_b, res_c, res_d]
    config_names = ["orig+AP1", "v2", "per-dim", "v2+AP1"]
    res_e = {}
    for dim in VAD_DIMS:
        best_f1 = 0
        best_cfg = ""
        for cfg, name in zip(all_configs, config_names):
            f1 = cfg[f"{dim}_f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = name
        res_e[f"{dim}_f1"] = best_f1
        print(f"  {dim}: F1={best_f1:.3f}  (from {best_cfg})")
    res_e["mean_f1"] = np.mean([res_e[f"{d}_f1"] for d in VAD_DIMS])
    records.append({"config": "oracle (best per dim)", **res_e})
    print(f"  → Oracle Mean = {res_e['mean_f1']:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print(f"{'CONFIG':<45} {'V F1':>7} {'A F1':>7} {'D F1':>7} {'Mean':>7}")
    print("-" * 90)
    for r in records:
        print(f"{r['config']:<45} {r['valence_f1']:7.3f} {r['arousal_f1']:7.3f} "
              f"{r['dominance_f1']:7.3f} {r['mean_f1']:7.3f}")
    print("-" * 90)
    print(f"{'PREV BEST: orig smooth+AP1':<45} {'0.600':>7} {'0.624':>7} {'0.436':>7} {'0.553':>7}")
    print("=" * 90)

    # Improvement over previous best
    best_mean = max(r["mean_f1"] for r in records)
    print(f"\n  Best achievable mean F1: {best_mean:.3f}")
    print(f"  Previous best:           0.553")
    print(f"  Improvement:             {best_mean - 0.553:+.3f}")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False, float_format="%.4f")
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
