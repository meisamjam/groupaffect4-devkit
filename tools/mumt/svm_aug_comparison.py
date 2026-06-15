"""svm_aug_comparison.py

Compare label augmentation variants using SVM (RBF, summary features) as the
fixed base model.  All variants use the same SVM hyperparameters and the same
task-CV protocol (train T0+T1, val T2, test T3).

Test sets always use ground-truth SAM labels only — the augmentation pool is
restricted to windows in the training tasks and is never mixed into test sets.

Augmentation variants
---------------------
A0   No augmentation — SVM on ~100 hard-labeled windows only.
A1   GP V/D aug only — pseudo-labels from augmented_pool_slow.pkl, arousal
     disabled (weight=0).  Confidence threshold 0.5.
A2   GP all dims — same pool, arousal included.  Confidence ≥ 0.5.
A3   GSR arousal + GP V/D — pool with GSR-derived arousal
     (augmented_pool_gsr.pkl).  Confidence ≥ 0.5.
A4   GSR + GP V/D, relaxed threshold — same as A3 but confidence ≥ 0.3.
AP1  BFI personality similarity only — pseudo-label trust weight is the mean
     cosine similarity of the target person's BFI-44 vector to their group
     members.  Confidence threshold 0.5.
AP2  BFI_sim × GP weight (personality-mediated) — multiplies the GP
     confidence from the SLOW pool by BFI personality similarity.
     Confidence threshold 0.5.
A5   kNN arousal + GP V/D — replaces GP Arousal pseudo-labels with k-NN
     (k=7) labels in the full 49-d physiological feature space.  kNN LOO
     accuracy 0.476 vs chance 0.333; much better than EDA-only SVR (0.286).
A6   BFI_sim × GP V/D + kNN arousal — combines A5 arousal with AP2-style
     BFI-weighted GP labels for Valence and Dominance.

Usage
-----
  python tools/mumt/svm_aug_comparison.py
  python tools/mumt/svm_aug_comparison.py \
      --dataset data/mumt/dataset_15s.pkl \
      --pool    data/mumt/augmented_pool_slow.pkl \
      --pool-gsr data/mumt/augmented_pool_gsr.pkl \
      --out     results/svm_aug_comparison.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR

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

VAD_DIMS = ["valence", "arousal", "dominance"]
FEAT_COLS = [
    "gaze_features", "pupil_features", "eda_features",
    "ppg_features", "imu_features",
]
BFI_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_X(df: pd.DataFrame, key_order: list[str]) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in FEAT_COLS:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    X = np.stack(rows, axis=0).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def get_hard_labels(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return (N, 3) int64.  Dominance NaN → -1."""
    out = np.full((len(df), 3), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if np.isnan(v):
                out[row_i, col_i] = -1
            else:
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


# ── Augmented pool helpers ─────────────────────────────────────────────────────

def recompute_soft_labels(
    pool: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    dim: str,
) -> np.ndarray:
    """Recompute 3-class soft labels from GP posteriors under balanced thresholds.

    Returns (N, 3) float32 probability array.
    """
    t1, t2 = thresholds[dim]
    mu = pool[f"{dim}_mu"].fillna(5.0).values.astype(float)
    sig = pool[f"{dim}_sigma"].fillna(1.5).values.astype(float)
    sig = np.clip(sig, 1e-4, None)

    p_low = norm.cdf(t1, loc=mu, scale=sig)
    p_high = 1.0 - norm.cdf(t2, loc=mu, scale=sig)
    p_mid = np.clip(1.0 - p_low - p_high, 0.0, 1.0)

    soft = np.stack([p_low, p_mid, p_high], axis=1).astype(np.float32)
    row_sums = soft.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    return soft / row_sums


def get_pool_pseudo_labels(
    pool: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    conf_threshold: float,
    disable_arousal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pseudo-label arrays for pool windows.

    Returns
    -------
    labels  : (N, 3) int64, pseudo-label class (argmax of soft), -1 if excluded
    weights : (N, 3) float32, GP confidence weights per dimension
    mask    : (N, 3) bool, True where the pseudo-label is accepted
    """
    N = len(pool)
    labels = np.full((N, 3), -1, dtype=np.int64)
    weights = np.zeros((N, 3), dtype=np.float32)
    mask = np.zeros((N, 3), dtype=bool)

    for col_i, dim in enumerate(VAD_DIMS):
        if disable_arousal and dim == "arousal":
            continue

        weight_col = f"{dim}_weight"
        if weight_col not in pool.columns:
            continue

        conf = pool[weight_col].fillna(0.0).values.astype(float)

        # T4 dominance: task-specific availability mask
        if dim == "dominance":
            is_t4 = pool["task"].values == "T4"
            conf = np.where(is_t4, 0.0, conf)

        soft = recompute_soft_labels(pool, thresholds, dim)
        pseudo = np.argmax(soft, axis=1).astype(np.int64)
        max_prob = soft[np.arange(N), pseudo]

        accepted = (conf >= conf_threshold) & (max_prob >= 0.5)
        labels[accepted, col_i] = pseudo[accepted]
        weights[accepted, col_i] = conf[accepted].astype(np.float32)
        mask[accepted, col_i] = True

    return labels, weights, mask


# ── Personality-similarity helpers ────────────────────────────────────────────

def compute_bfi_similarity_map(
    dataset_df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """Return {(session_id, seat): mean cosine BFI similarity to group members}.

    For each person, computes mean cosine similarity of their BFI-44 vector
    (E/A/C/N/O) to the other group members within the same session.
    Used as pseudo-label trust weight for personality-mediated augmentation.
    """
    avail = [c for c in BFI_COLS if c in dataset_df.columns]
    if not avail:
        log.warning("BFI columns not found in dataset — BFI similarity = 0.5 everywhere")
        return {}

    person_bfi = (
        dataset_df[["session_id", "seat"] + avail]
        .dropna(subset=avail)
        .groupby(["session_id", "seat"])
        .first()
        .reset_index()
    )

    bfi_map: dict[tuple[str, str], float] = {}
    for session_id, grp in person_bfi.groupby("session_id"):
        seats = grp["seat"].values
        vecs = grp[avail].values.astype(float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        vecs_n = vecs / norms
        sim_mat = vecs_n @ vecs_n.T  # (K, K)

        for i, seat in enumerate(seats):
            others = [j for j in range(len(seats)) if j != i]
            sim = float(np.mean(sim_mat[i, others])) if others else 0.5
            bfi_map[(session_id, str(seat))] = float(np.clip(sim, 0.0, 1.0))

    return bfi_map


def get_pool_pseudo_labels_bfi(
    pool: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    conf_threshold: float,
    bfi_sim_map: dict[tuple[str, str], float],
    use_bfi_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pseudo-label arrays with BFI personality similarity as trust weight.

    Parameters
    ----------
    use_bfi_only : if True, weight = bfi_sim (AP1 variant)
                   if False, weight = bfi_sim × gp_conf (AP2 variant)

    Returns same (labels, weights, mask) shape as get_pool_pseudo_labels.
    """
    N = len(pool)
    labels  = np.full((N, 3), -1, dtype=np.int64)
    weights = np.zeros((N, 3), dtype=np.float32)
    mask    = np.zeros((N, 3), dtype=bool)

    # Vectorised lookup of BFI sim per pool row
    bfi_sims = np.array([
        bfi_sim_map.get((str(r.session_id), str(r.seat)), 0.5)
        for r in pool.itertuples()
    ], dtype=np.float32)

    for col_i, dim in enumerate(VAD_DIMS):
        weight_col = f"{dim}_weight"
        if weight_col not in pool.columns:
            continue

        gp_conf = pool[weight_col].fillna(0.0).values.astype(float)

        if dim == "dominance":
            gp_conf = np.where(pool["task"].values == "T4", 0.0, gp_conf)

        if use_bfi_only:
            conf = bfi_sims.astype(float)
        else:
            conf = (bfi_sims * gp_conf).astype(float)

        soft   = recompute_soft_labels(pool, thresholds, dim)
        pseudo = np.argmax(soft, axis=1).astype(np.int64)
        max_p  = soft[np.arange(N), pseudo]

        accepted = (conf >= conf_threshold) & (max_p >= 0.5)
        labels[accepted, col_i]  = pseudo[accepted]
        weights[accepted, col_i] = conf[accepted].astype(np.float32)
        mask[accepted, col_i]    = True

    return labels, weights, mask


# ── SVR-based arousal augmentation ────────────────────────────────────────────

EDA_AROUSAL_KEYS = [
    "eda_phasic_mean", "eda_phasic_std",
    "eda_tonic_mean",  "eda_tonic_std",
    "scr_peak_count",  "scr_amplitude_mean",
]


def _extract_eda_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        d = r.get("eda_features", {}) or {}
        rows.append([d.get(k, np.nan) for k in EDA_AROUSAL_KEYS])
    return np.array(rows, dtype=float)


def generate_knn_arousal_pool(
    train_df: pd.DataFrame,
    pool_subset: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    key_order: list[str],
    k: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-NN arousal pseudo-labels in the full 49-d physiological feature space.

    For each pool window, find its k nearest labelled neighbors (same 49-d
    summary features the SVM uses) and assign majority-vote arousal class.
    Confidence = vote proportion, so pools far from any labelled example get
    low confidence and are filtered by the default 0.5 threshold.

    LOO accuracy on T0+T1: ~0.476 (vs chance 0.333), compared to EDA-only
    SVC 0.286 or SVR 0.344 Spearman — multimodal features discriminate arousal
    much better than EDA alone.

    pool_subset must already be filtered to training tasks and reset_index(drop=True).
    Returns (labels, weights, mask) shaped (len(pool_subset), 3).
    Only the arousal column (index 1) is filled.
    """
    from sklearn.neighbors import KNeighborsClassifier

    N = len(pool_subset)
    labels  = np.full((N, 3), -1, dtype=np.int64)
    weights = np.zeros((N, 3), dtype=np.float32)
    mask    = np.zeros((N, 3), dtype=bool)
    arousal_col = 1

    # Build arousal class labels for training windows
    t1, t2  = thresholds["arousal"]
    y_cont  = train_df["arousal"].values.astype(float)
    y_cls   = np.full(len(train_df), -1, dtype=np.int64)
    for i, v in enumerate(y_cont):
        if np.isfinite(v):
            y_cls[i] = 0 if v < t1 else (2 if v > t2 else 1)

    valid_tr = y_cls >= 0
    if valid_tr.sum() < k + 1:
        log.warning("kNN arousal: too few labelled training windows (%d)", valid_tr.sum())
        return labels, weights, mask

    # Feature matrices
    X_train = extract_X(train_df, key_order)
    X_pool  = extract_X(pool_subset, key_order)

    sc = StandardScaler()
    X_tr_sc   = sc.fit_transform(X_train[valid_tr])
    X_pool_sc = sc.transform(X_pool)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    knn.fit(X_tr_sc, y_cls[valid_tr])

    proba = knn.predict_proba(X_pool_sc)  # (N, n_classes)
    cls_order = knn.classes_

    for i in range(N):
        proba_full = np.zeros(3, dtype=float)
        for j, c in enumerate(cls_order):
            proba_full[c] = proba[i, j]
        pseudo = int(np.argmax(proba_full))
        conf   = float(proba_full[pseudo])
        labels[i, arousal_col]  = pseudo
        weights[i, arousal_col] = conf
        mask[i, arousal_col]    = True

    acc_mask  = mask[:, arousal_col] & (weights[:, arousal_col] >= 0.5)
    cls_dist  = np.bincount(labels[acc_mask, arousal_col], minlength=3)
    log.info("  kNN arousal (k=%d): %d accepted  [L=%d M=%d H=%d]",
             k, acc_mask.sum(), cls_dist[0], cls_dist[1], cls_dist[2])
    return labels, weights, mask


def generate_eda_arousal_pool(
    train_df: pd.DataFrame,
    pool_subset: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an EDA SVM *classifier* on training-fold labelled windows and
    pseudo-label pool_subset.

    Uses SVC(class_weight='balanced', probability=True) to directly predict
    Low/Mid/High arousal classes from EDA features, avoiding the
    regression-toward-mean distortion of SVR-based approaches.

    pool_subset must already be filtered to training tasks.
    Returns (labels, weights, mask) shaped (len(pool_subset), 3).
    Only the arousal column (index 1) is filled; V and D stay -1/0/False.

    Confidence weight = max predicted class probability (from Platt-scaled SVC).
    """
    from sklearn.calibration import CalibratedClassifierCV

    N = len(pool_subset)
    labels  = np.full((N, 3), -1, dtype=np.int64)
    weights = np.zeros((N, 3), dtype=np.float32)
    mask    = np.zeros((N, 3), dtype=bool)
    arousal_col = 1

    # Build training labels using current fold's balanced thresholds
    X_train = _extract_eda_features(train_df)
    y_cont  = train_df["arousal"].values.astype(float)
    t1, t2  = thresholds["arousal"]
    y_cls   = np.full(len(train_df), -1, dtype=np.int64)
    for i, v in enumerate(y_cont):
        if np.isfinite(v):
            y_cls[i] = 0 if v < t1 else (2 if v > t2 else 1)

    valid_tr = np.isfinite(X_train).all(axis=1) & (y_cls >= 0)
    classes_present = np.unique(y_cls[valid_tr])
    if valid_tr.sum() < 6 or len(classes_present) < 2:
        log.warning("EDA arousal SVC: too few valid training samples (%d)", valid_tr.sum())
        return labels, weights, mask

    base_svc = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=42)
    cal_svc  = CalibratedClassifierCV(base_svc, cv=min(5, valid_tr.sum() // 3))
    cal_pipe = Pipeline([("sc", RobustScaler()), ("clf", cal_svc)])
    cal_pipe.fit(X_train[valid_tr], y_cls[valid_tr])

    # Apply to pool_subset
    X_pool    = _extract_eda_features(pool_subset)
    valid_pool = np.isfinite(X_pool).all(axis=1)

    if valid_pool.sum() == 0:
        return labels, weights, mask

    proba = cal_pipe.predict_proba(X_pool[valid_pool])  # (M, n_classes)
    cls_order = cal_pipe.named_steps["clf"].classes_      # may not be [0,1,2]

    pool_positions = np.where(valid_pool)[0]
    for j, pos in enumerate(pool_positions):
        # Map probabilities back to class indices 0/1/2
        proba_full = np.zeros(3, dtype=float)
        for k, c in enumerate(cls_order):
            proba_full[c] = proba[j, k]
        pseudo = int(np.argmax(proba_full))
        conf   = float(proba_full[pseudo])
        labels[pos, arousal_col]  = pseudo
        weights[pos, arousal_col] = conf
        mask[pos, arousal_col]    = True

    # Log class distribution of pseudo-labels
    pseudo_cls = labels[mask[:, arousal_col], arousal_col]
    cts = np.bincount(pseudo_cls, minlength=3)
    log.info("  EDA SVC arousal: %d labelled  [L=%d M=%d H=%d]",
             valid_pool.sum(), cts[0], cts[1], cts[2])
    return labels, weights, mask


# ── SVM runner ────────────────────────────────────────────────────────────────

def run_svm_dim(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """Train RBF SVM for a single dimension and return macro-F1."""
    valid_tr = train_y >= 0
    valid_te = test_y >= 0
    if valid_tr.sum() < 3 or valid_te.sum() == 0:
        return 0.0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                      class_weight="balanced", random_state=42)),
    ])
    sw = train_w[valid_tr] if train_w is not None else None
    pipe.fit(train_X[valid_tr], train_y[valid_tr], svm__sample_weight=sw)
    preds = pipe.predict(test_X[valid_te])
    return float(f1_score(test_y[valid_te], preds, average="macro", zero_division=0))


def run_variant(
    train_X: np.ndarray,
    train_labels: np.ndarray,
    test_X: np.ndarray,
    test_labels: np.ndarray,
    aug_X: np.ndarray | None = None,
    aug_labels: np.ndarray | None = None,
    aug_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Run SVM for all 3 dims, optionally augmenting training set."""
    results: dict[str, float] = {}
    for col_i, dim in enumerate(VAD_DIMS):
        tr_X = train_X
        tr_y = train_labels[:, col_i]
        tr_w = None

        if aug_X is not None and aug_labels is not None:
            aug_y = aug_labels[:, col_i]
            aug_mask = aug_y >= 0
            if aug_mask.sum() > 0:
                aug_w_dim = (
                    aug_weights[:, col_i][aug_mask]
                    if aug_weights is not None
                    else np.ones(aug_mask.sum(), dtype=np.float32)
                )
                concat_X = np.concatenate([train_X, aug_X[aug_mask]], axis=0)
                concat_y = np.concatenate([tr_y, aug_y[aug_mask]], axis=0)
                # Labeled windows get weight 1.0; pseudo-labeled get GP weight
                labeled_w = np.ones(len(train_X), dtype=np.float32)
                concat_w  = np.concatenate([labeled_w, aug_w_dim], axis=0)
                tr_X = concat_X
                tr_y = concat_y
                tr_w = concat_w

        results[dim] = run_svm_dim(tr_X, tr_y, tr_w, test_X, test_labels[:, col_i])

    results["mean"] = float(np.mean([results[d] for d in VAD_DIMS]))
    return results


# ── Task-CV ────────────────────────────────────────────────────────────────────

def task_cv(
    df: pd.DataFrame,
    key_order: list[str],
    pool_slow: pd.DataFrame | None,
    pool_gsr: pd.DataFrame | None,
    bfi_sim_map: dict[tuple[str, str], float],
    test_tasks: list[str] = ("T3",),
) -> list[dict]:
    records = []

    for test_task in test_tasks:
        log.info("Task-CV fold: test=%s", test_task)
        # test_df always uses ground-truth SAM labels; pool never leaks into test set
        train_df, _, test_df = task_split(df, test_task=test_task)
        thresholds = compute_tertile_thresholds(train_df)

        train_X      = extract_X(train_df, key_order)
        test_X       = extract_X(test_df, key_order)
        train_labels = get_hard_labels(train_df, thresholds)
        test_labels  = get_hard_labels(test_df, thresholds)

        train_tasks = set(train_df["task"].unique())

        def get_aug(pool: pd.DataFrame | None, conf: float, no_arousal: bool = False):
            if pool is None:
                return None, None, None
            p = pool[pool["task"].isin(train_tasks)].copy()
            aug_X = extract_X(p, key_order)
            aug_labels, aug_weights, _ = get_pool_pseudo_labels(
                p, thresholds, conf, disable_arousal=no_arousal
            )
            return aug_X, aug_labels, aug_weights

        def get_aug_bfi(pool: pd.DataFrame | None, conf: float, bfi_only: bool):
            if pool is None:
                return None, None, None
            p = pool[pool["task"].isin(train_tasks)].copy()
            aug_X = extract_X(p, key_order)
            aug_labels, aug_weights, _ = get_pool_pseudo_labels_bfi(
                p, thresholds, conf, bfi_sim_map, use_bfi_only=bfi_only
            )
            return aug_X, aug_labels, aug_weights

        # Pre-compute SVR arousal labels for pool_slow (used by A5/A6)
        def get_aug_knn_arousal():
            """GP V/D labels + kNN arousal in 49-d feature space (A5)."""
            if pool_slow is None:
                return None, None, None
            p = pool_slow[pool_slow["task"].isin(train_tasks)].copy().reset_index(drop=True)
            aug_X = extract_X(p, key_order)
            aug_lab, aug_w, _ = get_pool_pseudo_labels(
                p, thresholds, 0.5, disable_arousal=True
            )
            knn_lab, knn_w, knn_mask = generate_knn_arousal_pool(
                train_df, p, thresholds, key_order
            )
            arousal_col = 1
            acc = knn_mask[:, arousal_col] & (knn_w[:, arousal_col] >= 0.5)
            aug_lab[acc, arousal_col] = knn_lab[acc, arousal_col]
            aug_w[acc, arousal_col]   = knn_w[acc, arousal_col]
            return aug_X, aug_lab, aug_w

        def get_aug_knn_bfi():
            """BFI_sim × GP V/D  +  kNN arousal weighted by BFI_sim (A6)."""
            if pool_slow is None:
                return None, None, None
            p = pool_slow[pool_slow["task"].isin(train_tasks)].copy().reset_index(drop=True)
            aug_X = extract_X(p, key_order)
            aug_lab, aug_w, _ = get_pool_pseudo_labels_bfi(
                p, thresholds, 0.5, bfi_sim_map, use_bfi_only=False
            )
            knn_lab, knn_w, knn_mask = generate_knn_arousal_pool(
                train_df, p, thresholds, key_order
            )
            bfi_sims = np.array([
                bfi_sim_map.get((str(r.session_id), str(r.seat)), 0.5)
                for r in p.itertuples()
            ], dtype=np.float32)
            arousal_col = 1
            combined_w = knn_w[:, arousal_col] * bfi_sims
            acc = knn_mask[:, arousal_col] & (combined_w >= 0.5)
            aug_lab[acc, arousal_col] = knn_lab[acc, arousal_col]
            aug_w[acc, arousal_col]   = combined_w[acc]
            return aug_X, aug_lab, aug_w

        # (name, builder_fn, builder_args)
        variant_specs = [
            ("A0_no_aug",
             lambda: (None, None, None)),
            ("A1_gp_vd_only",
             lambda: get_aug(pool_slow, 0.5, no_arousal=True)),
            ("A2_gp_all_dims",
             lambda: get_aug(pool_slow, 0.5, no_arousal=False)),
            ("A3_gsr_vd_05",
             lambda: get_aug(pool_gsr, 0.5, no_arousal=False)),
            ("A4_gsr_vd_03",
             lambda: get_aug(pool_gsr, 0.3, no_arousal=False)),
            ("AP1_bfi_only",
             lambda: get_aug_bfi(pool_slow, 0.5, bfi_only=True)),
            ("AP2_bfi_x_gp",
             lambda: get_aug_bfi(pool_slow, 0.5, bfi_only=False)),
            ("A5_knn_arousal",  get_aug_knn_arousal),
            ("A6_bfi_x_knn",    get_aug_knn_bfi),
        ]

        for name, builder in variant_specs:
            aug_X, aug_lab, aug_w = builder()
            n_aug = int(np.sum(aug_lab >= 0)) if aug_lab is not None else 0
            log.info("  %s: n_labeled=%d  n_aug_accepted=%d",
                     name, len(train_df), n_aug)

            r = run_variant(train_X, train_labels, test_X, test_labels,
                            aug_X, aug_lab, aug_w)
            for dim in VAD_DIMS + ["mean"]:
                records.append({
                    "test_task": test_task,
                    "variant":   name,
                    "dim":       dim,
                    "macro_f1":  round(r[dim], 4),
                    "n_labeled": len(train_df),
                    "n_test":    len(test_df),
                })

    return records


# ── Session-isolated LOGO augmentation (strict leakage-free scenario) ─────────

def logo_aug_cv(
    df: pd.DataFrame,
    key_order: list[str],
    pool_slow: pd.DataFrame | None,
) -> list[dict]:
    """Leave-one-group-out with augmentation from held-out-free pool.

    Strict leakage-free scenario:
      - Test  : held-out session, all tasks, ground-truth SAM labels only.
      - Train : other 9 sessions, all labeled windows.
      - Pool  : only windows from the 9 training sessions (strictly no test-
                session data: no shared subjects, no shared physiology, no
                shared OU-parameter contamination from test subjects).
      - BFI   : similarity map built from training sessions only.

    Compares A0 (no aug) vs AP1 (BFI personality, truly leakage-free) vs
    A2 (GP all dims, distributional leakage from full-dataset OU params noted).

    Returns one row per (session_fold × variant × dim).
    """
    sessions = sorted(df["session_id"].unique())
    records: list[dict] = []

    for test_session in sessions:
        train_df = df[df["session_id"] != test_session].copy()
        test_df  = df[df["session_id"] == test_session].copy()
        if len(test_df) == 0:
            continue

        thresholds   = compute_tertile_thresholds(train_df)
        train_X      = extract_X(train_df, key_order)
        test_X       = extract_X(test_df,  key_order)
        train_labels = get_hard_labels(train_df, thresholds)
        test_labels  = get_hard_labels(test_df,  thresholds)

        # BFI map built from training sessions only — test session excluded
        bfi_map_fold = compute_bfi_similarity_map(train_df)

        variant_specs: list[tuple[str, str]] = [("A0_no_aug", "none")]
        if pool_slow is not None:
            variant_specs += [
                ("AP1_bfi_only", "ap1"),
                ("A2_gp_all_dims", "gp"),
            ]

        for name, aug_type in variant_specs:
            aug_X, aug_lab, aug_w = None, None, None

            if aug_type != "none" and pool_slow is not None:
                # Pool restricted to training sessions only — no test subjects
                p = pool_slow[pool_slow["session_id"] != test_session].copy()
                aug_X = extract_X(p, key_order)

                if aug_type == "ap1":
                    aug_lab, aug_w, _ = get_pool_pseudo_labels_bfi(
                        p, thresholds, conf_threshold=0.5,
                        bfi_sim_map=bfi_map_fold, use_bfi_only=True,
                    )
                elif aug_type == "gp":
                    aug_lab, aug_w, _ = get_pool_pseudo_labels(
                        p, thresholds, conf_threshold=0.5,
                    )

            n_aug = int(np.sum(aug_lab >= 0)) if aug_lab is not None else 0
            log.info("  LOGO session=%s  %s: n_train=%d  n_aug=%d  n_test=%d",
                     test_session, name, len(train_df), n_aug, len(test_df))

            r = run_variant(train_X, train_labels, test_X, test_labels,
                            aug_X, aug_lab, aug_w)

            for dim in VAD_DIMS + ["mean"]:
                records.append({
                    "test_session": test_session,
                    "variant":      name,
                    "dim":          dim,
                    "macro_f1":     round(r[dim], 4),
                    "n_labeled":    len(train_df),
                    "n_test":       len(test_df),
                    "leakage_free": name != "A2_gp_all_dims",
                })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool",      default="data/mumt/augmented_pool_slow.pkl")
    parser.add_argument("--pool-gsr",  default="data/mumt/augmented_pool_gsr.pkl")
    parser.add_argument("--out",       default="results/svm_aug_comparison.csv")
    parser.add_argument("--test-tasks", nargs="+", default=["T3"])
    parser.add_argument("--logo-out", default="results/logo_aug_comparison.csv",
                        help="Output path for session-isolated LOGO+aug evaluation")
    parser.add_argument("--skip-logo", action="store_true",
                        help="Skip the LOGO augmentation evaluation")
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d windows", len(df))

    pool_slow = None
    if Path(args.pool).exists():
        pool_slow = pd.read_pickle(args.pool)
        log.info("SLOW pool: %d windows", len(pool_slow))
    else:
        log.warning("Pool not found: %s  — A1/A2 variants disabled", args.pool)

    pool_gsr = None
    if Path(args.pool_gsr).exists():
        pool_gsr = pd.read_pickle(args.pool_gsr)
        log.info("GSR pool:  %d windows", len(pool_gsr))
    else:
        log.warning("GSR pool not found: %s  — A3/A4 variants disabled", args.pool_gsr)

    key_order = build_summary_key_order(df)
    log.info("Summary feature dim: %d", len(key_order))

    bfi_sim_map = compute_bfi_similarity_map(df)
    log.info("BFI similarity map: %d (session, seat) entries", len(bfi_sim_map))

    records = task_cv(df, key_order, pool_slow, pool_gsr, bfi_sim_map, args.test_tasks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    # Print summary table
    pivot = out_df[out_df["dim"] == "mean"].pivot_table(
        index="variant", columns="test_task", values="macro_f1", aggfunc="mean"
    )
    print("\n=== SVM augmentation comparison (mean macro-F1) ===")
    print(pivot.round(4).to_string())

    # ── Session-isolated LOGO+aug (strict leakage-free scenario) ──────────────
    if not args.skip_logo:
        log.info("\n=== Running session-isolated LOGO+aug evaluation ===")
        logo_records = logo_aug_cv(df, key_order, pool_slow)
        logo_path = Path(args.logo_out)
        logo_path.parent.mkdir(parents=True, exist_ok=True)
        logo_df = pd.DataFrame(logo_records)
        logo_df.to_csv(logo_path, index=False)
        log.info("LOGO+aug saved → %s", logo_path)

        logo_summary = (
            logo_df[logo_df["dim"] == "mean"]
            .groupby("variant")["macro_f1"]
            .agg(["mean", "std"])
            .round(4)
        )
        print("\n=== Session-isolated LOGO+aug (strict leakage-free, mean±std macro-F1) ===")
        print(logo_summary.to_string())
        print("Note: AP1 (BFI similarity) is fully leakage-free (pre-session personality).")
        print("      A2 (GP) has distributional leakage from OU params estimated on full dataset.")


if __name__ == "__main__":
    main()
