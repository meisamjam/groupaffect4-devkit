"""arousal_enhancement.py

Two targeted experiments to improve Arousal prediction, which is largely
unresponsive to all existing augmentation variants at 15s windows.

Experiment 1 — 30s windows + AP1
---------------------------------
The window ablation shows 30s windows boost Arousal from 0.530 to 0.662
(+0.132) — by far the largest single improvement anywhere in the paper.
This is because a 30s window captures a complete SCR cycle and two full
HR beat groups, making mean/std features more stable arousal proxies.

AP1 has not been tested at 30s. By pooling consecutive 15s windows into 30s
(both dataset and pool), we can apply the existing AP1 augmentation pipeline
to the 30s feature vectors. Expected: Arousal > 0.68.

Experiment 2 — IMU-based arousal pseudo-labels
-----------------------------------------------
IMU (wrist accelerometry + gyroscope) is the dominant Arousal carrier:
removing it drops A from 0.481 to 0.269 in the modality ablation.
EDA-based arousal (A3/A4) failed because phasic EDA is highly person-specific
and the EDA→arousal mapping does not transfer across sessions.
IMU movement intensity is less person-specific: high movement = high arousal
is a more universal mapping across individuals.

Strategy: train an RBF SVC on IMU-only features from labeled data (16 features:
acc/gyr per axis and magnitude, mean+std per 15s window). Apply to pool windows
to generate arousal pseudo-labels. Combine with GP V/D for a full augmentation.

The 30s and IMU experiments are also combined (30s + IMU arousal).

Usage
-----
  python tools/mumt/arousal_enhancement.py
  python tools/mumt/arousal_enhancement.py \\
      --dataset  data/mumt/dataset_15s.pkl \\
      --pool     data/mumt/augmented_pool.pkl \\
      --out      results/arousal_enhancement.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

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
    recompute_soft_labels,
    get_pool_pseudo_labels,
    get_pool_pseudo_labels_bfi,
    compute_bfi_similarity_map,
)

# IMU feature keys used for arousal regression
IMU_AROUSAL_KEYS = [
    "acc_mag_mean", "acc_mag_std",
    "acc_x_mean",   "acc_x_std",
    "acc_y_mean",   "acc_y_std",
    "acc_z_mean",   "acc_z_std",
    "gyr_mag_mean", "gyr_mag_std",
    "gyr_x_mean",   "gyr_x_std",
    "gyr_y_mean",   "gyr_y_std",
    "gyr_z_mean",   "gyr_z_std",
]

# GP columns to average when pooling pool windows
GP_COLS = [
    "valence_mu",    "valence_sigma",    "valence_weight",
    "arousal_mu",    "arousal_sigma",    "arousal_weight",
    "dominance_mu",  "dominance_sigma",  "dominance_weight",
]


# ── Window pooling ─────────────────────────────────────────────────────────────

def pool_df_to_Ns(df: pd.DataFrame, n: int, include_gp: bool = False) -> pd.DataFrame:
    """Pool consecutive n×15s windows into ~(n×15)s windows.

    For each (session_id, subject_id, task) group, consecutive chunks of n
    rows are merged by averaging FEAT_COLS features (and optionally GP columns).
    VAD labels and metadata come from the first window of each chunk.

    Works for both the labeled dataset and the augmented pool.
    """
    group_cols = ["session_id", "subject_id", "task"]
    rows_out = []

    for _, grp in df.groupby(group_cols, sort=False):
        sort_col = "vad_timestamp_lsl" if "vad_timestamp_lsl" in grp.columns else grp.columns[0]
        grp = grp.sort_values(sort_col).reset_index(drop=True)

        i = 0
        while i < len(grp):
            chunk = grp.iloc[i:i + n]
            r0 = chunk.iloc[0].copy()

            # Average feature dicts
            for fc in FEAT_COLS:
                merged: dict[str, float] = {}
                for _, row in chunk.iterrows():
                    fd = row.get(fc, {}) or {}
                    for k, v in fd.items():
                        merged[k] = merged.get(k, 0.0) + float(v)
                m = len(chunk)
                r0[fc] = {k: v / m for k, v in merged.items()}

            # Average GP columns if present
            if include_gp:
                for gc in GP_COLS:
                    if gc in df.columns:
                        vals = pd.to_numeric(chunk[gc], errors="coerce")
                        r0[gc] = float(vals.mean()) if vals.notna().any() else np.nan

            rows_out.append(r0)
            i += n

    return pd.DataFrame(rows_out).reset_index(drop=True)


# ── IMU arousal pseudo-labels ─────────────────────────────────────────────────

def _extract_imu_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        d = r.get("imu_features", {}) or {}
        rows.append([d.get(k, np.nan) for k in IMU_AROUSAL_KEYS])
    return np.array(rows, dtype=float)


def smooth_features(df: pd.DataFrame, mode: str = "forward") -> pd.DataFrame:
    """Smooth FEAT_COLS features using temporal neighbors within each group.

    Approximates longer window context without reducing N or re-extracting raw data.

    Parameters
    ----------
    mode : "forward"  — average window i with window i+1 (~30s context, N preserved)
           "centered" — average window i with i-1 and i+1 (~45s context, N preserved)
           "backward" — average window i with window i-1 (~30s context, causal)

    For windows at group boundaries (no next/prev), only available neighbors are used.
    Labels and metadata remain on their original window; only FEAT_COLS are modified.
    """
    group_cols = ["session_id", "subject_id", "task"]
    sort_col   = "vad_timestamp_lsl"
    df_out     = df.copy()

    for _, grp in df.groupby(group_cols, sort=False):
        if sort_col not in grp.columns:
            continue
        grp_sorted = grp.sort_values(sort_col)
        orig_idx   = grp_sorted.index.tolist()  # indices into df_out

        n = len(orig_idx)
        for pos in range(n):
            if mode == "forward":
                positions = [pos, pos + 1] if pos + 1 < n else [pos]
            elif mode == "backward":
                positions = [pos - 1, pos] if pos - 1 >= 0 else [pos]
            else:  # centered
                positions = [p for p in [pos - 1, pos, pos + 1] if 0 <= p < n]

            rows = [grp_sorted.iloc[p] for p in positions]
            for fc in FEAT_COLS:
                merged: dict[str, float] = {}
                for row in rows:
                    fd = row.get(fc, {}) or {}
                    for k, v in fd.items():
                        merged[k] = merged.get(k, 0.0) + float(v)
                m = len(rows)
                df_out.at[orig_idx[pos], fc] = {k: v / m for k, v in merged.items()}

    return df_out


def generate_imu_arousal_pool(
    train_df: pd.DataFrame,
    pool_subset: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an IMU SVC on training labeled windows and pseudo-label pool_subset.

    Uses 16 IMU summary features (acc/gyr per-axis and magnitude mean+std).
    IMU is the dominant Arousal carrier (dropping it costs A: 0.481→0.269)
    and is less person-specific than EDA, making it a better transfer proxy.

    Returns (labels, weights, mask) shaped (len(pool_subset), 3).
    Only the arousal column (index 1) is filled; V and D remain -1/0/False.
    Confidence weight = max predicted class probability (Platt-scaled SVC).
    """
    N = len(pool_subset)
    labels  = np.full((N, 3), -1, dtype=np.int64)
    weights = np.zeros((N, 3), dtype=np.float32)
    mask    = np.zeros((N, 3), dtype=bool)
    arousal_col = 1

    # Build arousal labels for training fold
    X_train = _extract_imu_features(train_df)
    y_cont  = train_df["arousal"].values.astype(float)
    t1, t2  = thresholds["arousal"]
    y_cls   = np.full(len(train_df), -1, dtype=np.int64)
    for i, v in enumerate(y_cont):
        if np.isfinite(v):
            y_cls[i] = 0 if v < t1 else (2 if v > t2 else 1)

    valid_tr = np.isfinite(X_train).all(axis=1) & (y_cls >= 0)
    classes_present = np.unique(y_cls[valid_tr])
    if valid_tr.sum() < 6 or len(classes_present) < 2:
        log.warning("IMU arousal SVC: too few valid training samples (%d)", valid_tr.sum())
        return labels, weights, mask

    base_svc = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=42)
    cal_svc  = CalibratedClassifierCV(base_svc, cv=min(5, valid_tr.sum() // 3))
    cal_pipe = Pipeline([("sc", RobustScaler()), ("clf", cal_svc)])
    cal_pipe.fit(X_train[valid_tr], y_cls[valid_tr])

    # LOO accuracy on training fold
    from sklearn.model_selection import cross_val_score
    loo_scores = cross_val_score(
        Pipeline([("sc", RobustScaler()),
                  ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                              class_weight="balanced", random_state=42))]),
        X_train[valid_tr], y_cls[valid_tr],
        cv=min(5, valid_tr.sum() // 3), scoring="f1_macro",
    )
    log.info("  IMU SVC LOO arousal F1: %.3f ± %.3f (n=%d)",
             loo_scores.mean(), loo_scores.std(), valid_tr.sum())

    # Apply to pool
    X_pool    = _extract_imu_features(pool_subset)
    valid_pool = np.isfinite(X_pool).all(axis=1)
    if valid_pool.sum() == 0:
        return labels, weights, mask

    proba     = cal_pipe.predict_proba(X_pool[valid_pool])
    cls_order = cal_pipe.named_steps["clf"].classes_

    pool_positions = np.where(valid_pool)[0]
    for j, pos in enumerate(pool_positions):
        proba_full = np.zeros(3, dtype=float)
        for k, c in enumerate(cls_order):
            proba_full[c] = proba[j, k]
        pseudo = int(np.argmax(proba_full))
        conf   = float(proba_full[pseudo])
        labels[pos, arousal_col]  = pseudo
        weights[pos, arousal_col] = conf
        mask[pos, arousal_col]    = True

    pseudo_cls = labels[mask[:, arousal_col], arousal_col]
    cts = np.bincount(pseudo_cls, minlength=3)
    acc = mask[:, arousal_col] & (weights[:, arousal_col] >= 0.5)
    log.info("  IMU arousal: %d labelled pool  conf≥0.5: %d  [L=%d M=%d H=%d]",
             valid_pool.sum(), acc.sum(), cts[0], cts[1], cts[2])
    return labels, weights, mask


# ── Core evaluation ────────────────────────────────────────────────────────────

def run_experiments(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    key_order_15: list[str],
    key_order_30: list[str],
    test_task: str = "T3",
) -> list[dict]:
    records = []

    train_df_15, _, test_df_15 = task_split(df, test_task=test_task)
    train_tasks = set(train_df_15["task"].unique())
    thresholds_15 = compute_tertile_thresholds(train_df_15)

    # 30s pooled versions
    log.info("Pooling dataset and pool to 30s …")
    df_30   = pool_df_to_Ns(df,   n=2, include_gp=False)
    pool_30 = pool_df_to_Ns(pool, n=2, include_gp=True)

    train_df_30, _, test_df_30 = task_split(df_30, test_task=test_task)
    thresholds_30 = compute_tertile_thresholds(train_df_30)

    log.info("15s: train=%d test=%d | 30s: train=%d test=%d",
             len(train_df_15), len(test_df_15),
             len(train_df_30), len(test_df_30))

    bfi_sim_map_15 = compute_bfi_similarity_map(train_df_15)
    bfi_sim_map_30 = compute_bfi_similarity_map(train_df_30)

    def _eval(train_df, test_df, thresholds, key_order, bfi_sim_map, pool_sub,
              variant_name, window_s):
        train_X      = extract_X(train_df, key_order)
        test_X       = extract_X(test_df,  key_order)
        train_labels = get_hard_labels(train_df, thresholds)
        test_labels  = get_hard_labels(test_df,  thresholds)

        aug_X, aug_lab, aug_w = None, None, None

        if pool_sub is not None:
            p = pool_sub[pool_sub["task"].isin(set(train_df["task"].unique()))].copy()
            p = p.reset_index(drop=True)
            aug_X = extract_X(p, key_order)

            if variant_name == "AP1":
                aug_lab, aug_w, _ = get_pool_pseudo_labels_bfi(
                    p, thresholds, 0.5, bfi_sim_map, use_bfi_only=True
                )
            elif variant_name == "IMU":
                # IMU arousal only; no GP V/D (isolate IMU signal)
                imu_lab, imu_w, imu_mask = generate_imu_arousal_pool(
                    train_df, p, thresholds
                )
                # V and D from GP; A from IMU
                gp_lab, gp_w, _ = get_pool_pseudo_labels(
                    p, thresholds, 0.5, disable_arousal=True
                )
                aug_lab = gp_lab.copy()
                aug_w   = gp_w.copy()
                arousal_col = 1
                acc = imu_mask[:, arousal_col] & (imu_w[:, arousal_col] >= 0.5)
                aug_lab[acc, arousal_col] = imu_lab[acc, arousal_col]
                aug_w[acc, arousal_col]   = imu_w[acc, arousal_col]
            elif variant_name == "IMU+AP1":
                # IMU arousal + BFI-weighted GP V/D
                imu_lab, imu_w, imu_mask = generate_imu_arousal_pool(
                    train_df, p, thresholds
                )
                bfi_lab, bfi_w, _ = get_pool_pseudo_labels_bfi(
                    p, thresholds, 0.5, bfi_sim_map, use_bfi_only=True
                )
                aug_lab = bfi_lab.copy()
                aug_w   = bfi_w.copy()
                arousal_col = 1
                acc = imu_mask[:, arousal_col] & (imu_w[:, arousal_col] >= 0.5)
                aug_lab[acc, arousal_col] = imu_lab[acc, arousal_col]
                aug_w[acc, arousal_col]   = imu_w[acc, arousal_col]

        n_aug = int(np.sum(aug_lab >= 0)) if aug_lab is not None else 0
        log.info("  %s @ %ds: n_train=%d  n_aug=%d  n_test=%d",
                 variant_name, window_s, len(train_df), n_aug, len(test_df))

        r = run_variant(train_X, train_labels, test_X, test_labels, aug_X, aug_lab, aug_w)
        return {
            "window_s": window_s,
            "variant":  variant_name,
            "v_f1":  round(r["valence"],   4),
            "a_f1":  round(r["arousal"],   4),
            "d_f1":  round(r["dominance"], 4),
        }

    # ── 15s baselines (reference) ──────────────────────────────────────────────
    for variant in ["A0", "AP1"]:
        pool_sub = pool if variant != "A0" else None
        rec = _eval(train_df_15, test_df_15, thresholds_15,
                    key_order_15, bfi_sim_map_15, pool_sub, variant, 15)
        records.append(rec)

    # ── 30s pooled ────────────────────────────────────────────────────────────
    for variant in ["A0", "AP1"]:
        pool_sub = pool_30 if variant != "A0" else None
        rec = _eval(train_df_30, test_df_30, thresholds_30,
                    key_order_30, bfi_sim_map_30, pool_sub, variant, 30)
        records.append(rec)

    # ── Feature-level smoothing on 15s (preserves N=103) ─────────────────────
    log.info("Smoothing 15s features …")
    for smooth_mode in ("forward", "centered"):
        label = "30s" if smooth_mode == "forward" else "45s"
        df_smooth = smooth_features(df, mode=smooth_mode)
        pool_smooth = smooth_features(pool, mode=smooth_mode)

        train_sm, _, test_sm = task_split(df_smooth, test_task=test_task)
        thresh_sm = compute_tertile_thresholds(train_sm)
        key_sm    = build_summary_key_order(df_smooth)
        bfi_sm    = compute_bfi_similarity_map(train_sm)

        log.info("  smooth-%s: train=%d test=%d", smooth_mode, len(train_sm), len(test_sm))

        for variant in ["A0", "AP1"]:
            pool_sub = pool_smooth if variant != "A0" else None
            rec = _eval(train_sm, test_sm, thresh_sm,
                        key_sm, bfi_sm, pool_sub, variant, f"smooth-{label}")
            records.append(rec)

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool",      default="data/mumt/augmented_pool_slow.pkl")
    parser.add_argument("--out",       default="results/arousal_enhancement.csv")
    parser.add_argument("--test-task", default="T3")
    args = parser.parse_args()

    df   = pd.read_pickle(args.dataset)
    pool = pd.read_pickle(args.pool)
    log.info("Dataset: %d windows  |  Pool: %d windows", len(df), len(pool))

    key_order_15 = build_summary_key_order(df)
    df_30        = pool_df_to_Ns(df, n=2, include_gp=False)
    key_order_30 = build_summary_key_order(df_30)

    records = run_experiments(df, pool, key_order_15, key_order_30, args.test_task)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(records)
    result_df.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    print("\n=== Arousal Enhancement Results ===")
    print(result_df[["window_s", "variant", "v_f1", "a_f1", "d_f1"]].to_string(index=False))

    print("\n--- Arousal column only (pivot) ---")
    print(result_df.pivot_table(
        index="window_s", columns="variant", values="a_f1", aggfunc="first"
    ).to_string())


if __name__ == "__main__":
    main()
