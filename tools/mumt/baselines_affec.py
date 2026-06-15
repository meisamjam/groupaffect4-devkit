"""baselines_affec.py

Method comparison for the AFFEC dataset (individual-level evaluation).

Protocol A  — run-based temporal split (T0+T1 train / T2 val / T3 test):
  1. Majority-class
  2. Random
  3. SVM (all features)
  4. Per-modality SVM ablation
  5. 5-fold subject CV (cross-subject protocol)

Protocol B  — within-subject trial-level split (every participant in all sets):
  B0. Majority-class
  B1. SVM (all features, raw)
  B2. SVM + T0 baseline normalisation
  B3. SVM + trial-position feature
  B4. SVM + T0 norm + trial position   ← preprocessing + temporal modelling

Usage:
  python tools/mumt/baselines_affec.py --dataset data/mumt/dataset_affec.pkl
  python tools/mumt/baselines_affec.py --dataset data/mumt/dataset_affec.pkl --kfold 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import build_summary_key_order, flatten_features
from train_simple import bin_vad_from_thresholds

VAD_DIMS = ["valence", "arousal"]   # Dominance not in AFFEC


def compute_thresholds(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Tertile thresholds for V+A only (Dominance absent in AFFEC)."""
    out: dict[str, tuple[float, float]] = {}
    for dim in VAD_DIMS:
        vals = np.sort(train_df[dim].dropna().values.astype(float))
        n = len(vals)
        if n < 3:
            out[dim] = (3.5, 6.5)
            continue

        def _mid(idx: int) -> float:
            bv = vals[idx]
            below = vals[vals < bv]
            return (below[-1] + bv) / 2.0 if len(below) > 0 else bv - 0.5

        t1, t2 = _mid(n // 3), _mid((2 * n) // 3)
        if t2 <= t1:
            above = vals[vals > (t1 + 0.5)]
            t2 = (t1 + above[0]) / 2.0 if len(above) > 0 else t1 + 1.0
        out[dim] = (t1, t2)
        low  = int(np.sum(vals <= t1))
        mid  = int(np.sum((vals > t1) & (vals <= t2)))
        high = int(np.sum(vals >  t2))
        log.info("  %s: (%.2f, %.2f)  Low=%d Mid=%d High=%d N=%d",
                 dim, t1, t2, low, mid, high, n)
    return out


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_X(df: pd.DataFrame, key_order: list[str]) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features"]:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.stack(rows, axis=0).astype(np.float32)


def get_labels(df: pd.DataFrame,
               thresholds: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.full((len(df), len(VAD_DIMS)), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if np.isnan(v):
                out[row_i, col_i] = -1
            else:
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


# Modality name → feature dict column
MODALITY_FEAT_COLS: dict[str, list[str]] = {
    "eda":   ["eda_features"],
    "gaze":  ["gaze_features"],
    "pupil": ["pupil_features"],
    "imu":   ["imu_features"],
    "eye":   ["gaze_features", "pupil_features"],
    "physio":["eda_features",  "imu_features"],
}


def build_X_modality(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix using only the specified modality columns.

    Returns (X, key_order) so callers can reuse the same key_order on other splits.
    """
    key_sets: set[str] = set()
    for col in feat_cols:
        for feat in df[col]:
            if isinstance(feat, dict):
                key_sets.update(feat.keys())
    key_order = sorted(key_sets)
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in feat_cols:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.stack(rows, axis=0).astype(np.float32), key_order


# ---------------------------------------------------------------------------
# Within-subject trial-level split helpers
# ---------------------------------------------------------------------------

_TASK_ORD: dict[str, int] = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}


def within_subject_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at trial level so every participant appears in all three sets.

    Trials are sorted chronologically (run order then onset) per participant.
    First *train_frac* → train, next *val_frac* → val, remainder → test.
    """
    train_idx: list = []
    val_idx:   list = []
    test_idx:  list = []
    for _, sdf in df.groupby("subject_id", sort=False):
        sdf = sdf.copy()
        sdf["_to"] = sdf["task"].map(_TASK_ORD).fillna(99).astype(int)
        sdf = sdf.sort_values(["_to", "task_start_lsl"])
        indices = sdf.index.tolist()
        n  = len(indices)
        nt = max(1, int(n * train_frac))
        nv = max(1, int(n * val_frac))
        if n - nt - nv < 1:          # very short: all to train
            train_idx.extend(indices)
        else:
            train_idx.extend(indices[:nt])
            val_idx.extend(indices[nt:nt + nv])
            test_idx.extend(indices[nt + nv:])
    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def compute_t0_means(
    train_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-participant mean physio features from T0-labelled trials in train_df.

    If a participant has no T0 trials in the training set, falls back to their
    overall training mean (avoids NaN normalisation offsets).
    """
    FEAT_COLS = ["gaze_features", "pupil_features", "eda_features",
                 "ppg_features", "imu_features"]
    t0_df = train_df[train_df["task"] == "T0"]
    t0_means: dict[str, dict[str, float]] = {}
    for subj in train_df["subject_id"].unique():
        src = t0_df[t0_df["subject_id"] == subj]
        if src.empty:
            src = train_df[train_df["subject_id"] == subj]  # fallback
        accum: dict[str, list[float]] = {}
        for _, r in src.iterrows():
            for col in FEAT_COLS:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    for k, v in fd.items():
                        try:
                            fv = float(v)
                            if not np.isnan(fv):
                                accum.setdefault(k, []).append(fv)
                        except (TypeError, ValueError):
                            pass
        t0_means[subj] = {k: float(np.mean(v)) for k, v in accum.items()}
    return t0_means


def compute_trial_positions(df: pd.DataFrame) -> dict:
    """Normalised position (0→1) of each trial within its participant's session.

    Computed on the *full* dataset so the mapping is consistent across splits.
    """
    pos_map: dict = {}
    for _, sdf in df.groupby("subject_id", sort=False):
        sdf = sdf.copy()
        sdf["_to"] = sdf["task"].map(_TASK_ORD).fillna(99).astype(int)
        sdf = sdf.sort_values(["_to", "task_start_lsl"])
        n = len(sdf)
        for rank, idx in enumerate(sdf.index):
            pos_map[idx] = rank / max(1, n - 1)
    return pos_map


def build_X_enhanced(
    df: pd.DataFrame,
    key_order: list[str],
    t0_means:     dict[str, dict[str, float]] | None = None,
    trial_pos_map: dict | None = None,
) -> np.ndarray:
    """Feature matrix with optional T0 normalisation and/or trial-position feature."""
    rows = []
    for idx, r in df.iterrows():
        feats: dict[str, float] = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features"]:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update({k: float(v) for k, v in fd.items()})
        if t0_means is not None:
            subj_t0 = t0_means.get(str(r["subject_id"]), {})
            for k in list(feats.keys()):
                if k in subj_t0 and not np.isnan(subj_t0[k]):
                    feats[k] = feats[k] - subj_t0[k]
        if trial_pos_map is not None:
            feats["trial_position"] = trial_pos_map.get(idx, 0.0)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.stack(rows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal task split
# ---------------------------------------------------------------------------

def task_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """T0+T1 → train, T2 → val, T3 → test."""
    train = df[df["task"].isin(["T0", "T1"])].copy()
    val   = df[df["task"] == "T2"].copy()
    test  = df[df["task"] == "T3"].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# k-fold subject CV (cross-subject protocol)
# ---------------------------------------------------------------------------

def kfold_subject_cv(df: pd.DataFrame, n_splits: int = 5) -> dict[str, float]:
    """Leave-N-subjects-out CV across subjects."""
    from sklearn.model_selection import GroupKFold
    subjects = df["subject_id"].to_numpy()
    X_full = None
    scores: dict[str, list[float]] = {d: [] for d in VAD_DIMS}

    key_order = build_summary_key_order(df)

    gkf = GroupKFold(n_splits=n_splits)
    X = build_X(df, key_order)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=subjects)):
        train_df = df.iloc[tr_idx]
        test_df  = df.iloc[te_idx]
        thresh = compute_thresholds(train_df)
        y_tr = get_labels(train_df, thresh)
        y_te = get_labels(test_df, thresh)
        X_tr = X[tr_idx]
        X_te = X[te_idx]

        for d_i, dim in enumerate(VAD_DIMS):
            mask_tr = y_tr[:, d_i] >= 0
            mask_te = y_te[:, d_i] >= 0
            if mask_tr.sum() < 3 or mask_te.sum() < 3:
                continue
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
            ])
            pipe.fit(X_tr[mask_tr], y_tr[mask_tr, d_i])
            pred = pipe.predict(X_te[mask_te])
            scores[dim].append(f1_score(y_te[mask_te, d_i], pred, average="macro", zero_division=0))

    return {d: float(np.mean(v)) if v else float("nan") for d, v in scores.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="AFFEC SVM baselines (A0 individual pipeline)")
    ap.add_argument("--dataset", default="data/mumt/dataset_affec.pkl", type=Path)
    ap.add_argument("--kfold", type=int, default=5,
                    help="Run k-fold subject CV as cross-subject protocol (0 = skip)")
    args = ap.parse_args()

    log.info("Loading %s", args.dataset)
    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d trials, %d subjects", len(df), df["subject_id"].nunique())
    log.info("Tasks: %s", df["task"].value_counts().to_dict())

    key_order = build_summary_key_order(df)
    log.info("Feature dimension: %d", len(key_order))

    # ── Temporal split (mirrors GroupAffect-4 T0+T1/T2/T3) ──────────────────
    train_df, val_df, test_df = task_split(df)
    log.info("Split — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))

    thresh = compute_thresholds(train_df)
    log.info("Thresholds: %s", {d: (round(t[0], 2), round(t[1], 2)) for d, t in thresh.items()})

    X_tr = build_X(train_df, key_order)
    X_va = build_X(val_df,   key_order)
    X_te = build_X(test_df,  key_order)
    y_tr = get_labels(train_df, thresh)
    y_va = get_labels(val_df,   thresh)
    y_te = get_labels(test_df,  thresh)

    results: list[dict] = []

    # ── Majority baseline ────────────────────────────────────────────────────
    log.info("\n── Majority-class baseline ──────────────────────────────────────")
    for d_i, dim in enumerate(VAD_DIMS):
        mask_tr = y_tr[:, d_i] >= 0
        mask_te = y_te[:, d_i] >= 0
        if not mask_tr.any() or not mask_te.any():
            continue
        maj = np.bincount(y_tr[mask_tr, d_i]).argmax()
        pred = np.full(mask_te.sum(), maj)
        f1 = f1_score(y_te[mask_te, d_i], pred, average="macro", zero_division=0)
        log.info("  %s: %.3f", dim, f1)
        results.append({"model": "Majority", "dim": dim, "split": "test", "macro_f1": f1})

    # ── Random baseline ──────────────────────────────────────────────────────
    log.info("\n── Random baseline ──────────────────────────────────────────────")
    rng = np.random.default_rng(42)
    for d_i, dim in enumerate(VAD_DIMS):
        mask_te = y_te[:, d_i] >= 0
        if not mask_te.any():
            continue
        rand_preds = rng.integers(0, 3, size=mask_te.sum())
        f1 = f1_score(y_te[mask_te, d_i], rand_preds, average="macro", zero_division=0)
        log.info("  %s: %.3f", dim, f1)
        results.append({"model": "Random", "dim": dim, "split": "test", "macro_f1": f1})

    # SVM A0 (no augmentation)
    log.info("\n── RBF-SVM A0 (no augmentation, temporal split) ─────────────────")
    svm_test: dict[str, float] = {}
    svm_val:  dict[str, float] = {}
    for d_i, dim in enumerate(VAD_DIMS):
        mask_tr = y_tr[:, d_i] >= 0
        mask_va = y_va[:, d_i] >= 0
        mask_te = y_te[:, d_i] >= 0
        if not mask_tr.any():
            continue
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
        ])
        pipe.fit(X_tr[mask_tr], y_tr[mask_tr, d_i])

        if mask_va.any():
            pv = pipe.predict(X_va[mask_va])
            svm_val[dim] = f1_score(y_va[mask_va, d_i], pv, average="macro", zero_division=0)

        if mask_te.any():
            pt = pipe.predict(X_te[mask_te])
            svm_test[dim] = f1_score(y_te[mask_te, d_i], pt, average="macro", zero_division=0)
            log.info("  %s — val: %.3f  test: %.3f", dim,
                     svm_val.get(dim, float("nan")), svm_test[dim])
            results.append({"model": "SVM-A0", "dim": dim, "split": "test", "macro_f1": svm_test[dim]})

    mean_f1 = float(np.nanmean(list(svm_test.values())))
    log.info("  Mean test macro-F1: %.3f", mean_f1)

    # ── Per-modality SVM ablation ─────────────────────────────────────────────
    log.info("\n── Per-modality SVM ablation (test split) ───────────────────────────────")
    for mod_name, feat_cols in MODALITY_FEAT_COLS.items():
        Xm_tr, mod_key_order = build_X_modality(train_df, feat_cols)
        if not mod_key_order:
            continue
        Xm_te, _ = build_X_modality(test_df, feat_cols)
        for d_i, dim in enumerate(VAD_DIMS):
            mask_tr = y_tr[:, d_i] >= 0
            mask_te = y_te[:, d_i] >= 0
            if not mask_tr.any() or not mask_te.any():
                continue
            pipe = Pipeline([
                ("sc",  StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
            ])
            pipe.fit(Xm_tr[mask_tr], y_tr[mask_tr, d_i])
            pred = pipe.predict(Xm_te[mask_te])
            f1 = f1_score(y_te[mask_te, d_i], pred, average="macro", zero_division=0)
            results.append({"model": f"SVM-{mod_name}", "dim": dim,
                             "split": "test", "macro_f1": f1})
        # log mean across dims for this modality
        mod_rows = [r for r in results if r["model"] == f"SVM-{mod_name}" and r["split"] == "test"]
        mean_mod = float(np.mean([r["macro_f1"] for r in mod_rows]))
        log.info("  SVM-%-10s  mean=%.3f", mod_name, mean_mod)

    # k-fold CV (for devkit comparison)
    if args.kfold > 0:
        log.info("\n── %d-fold subject CV (cross-subject protocol) ──────────────────", args.kfold)
        cv_scores = kfold_subject_cv(df, n_splits=args.kfold)
        for dim, sc in cv_scores.items():
            log.info("  %s: %.3f", dim, sc)
            results.append({"model": "SVM-all (CV)", "dim": dim,
                             "split": "cv", "macro_f1": sc})
        log.info("  Mean CV macro-F1: %.3f", float(np.nanmean(list(cv_scores.values()))))

    # ── Summary: Protocol A ───────────────────────────────────────────────
    log.info("\n══ Protocol A  AFFEC method comparison (macro-F1, 3-class) ══")
    res_df = pd.DataFrame([r for r in results if r["split"] in ("test", "cv")])
    pivot = res_df.pivot_table(index="model", columns=["dim", "split"],
                               values="macro_f1", aggfunc="first")
    for split in ["test", "cv"]:
        cols = [(d, split) for d in VAD_DIMS if (d, split) in pivot.columns]
        if cols:
            pivot[("mean", split)] = pivot[cols].mean(axis=1)
    log.info("\n%s", pivot.sort_index().to_string())

    # ============================================================
    # Protocol B: within-subject split + preprocessing + temporal
    # ============================================================
    log.info("\n══ Protocol B: within-subject trial-level split ══")
    tr_ws, va_ws, te_ws = within_subject_split(df)
    log.info("Split  train=%d  val=%d  test=%d", len(tr_ws), len(va_ws), len(te_ws))
    log.info("Subjects in test: %d / %d", te_ws["subject_id"].nunique(), df["subject_id"].nunique())
    log.info("T0 trials in train portion: %d", int((tr_ws["task"] == "T0").sum()))

    thresh_ws = compute_thresholds(tr_ws)
    y_tr_ws = get_labels(tr_ws, thresh_ws)
    y_va_ws = get_labels(va_ws, thresh_ws)
    y_te_ws = get_labels(te_ws, thresh_ws)

    # Pre-compute enhancements once
    t0_means      = compute_t0_means(tr_ws)
    trial_pos_map = compute_trial_positions(df)      # full df for consistent positions
    key_order_tp  = sorted(set(key_order) | {"trial_position"})

    ws_results: list[dict] = []

    def _svm_ws(X_tr: np.ndarray, X_te: np.ndarray, label: str) -> None:
        for d_i, dim in enumerate(VAD_DIMS):
            m_tr = y_tr_ws[:, d_i] >= 0
            m_te = y_te_ws[:, d_i] >= 0
            if not m_tr.any() or not m_te.any():
                continue
            pipe = Pipeline([("sc", StandardScaler()),
                             ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
            pipe.fit(X_tr[m_tr], y_tr_ws[m_tr, d_i])
            pred = pipe.predict(X_te[m_te])
            f1 = f1_score(y_te_ws[m_te, d_i], pred, average="macro", zero_division=0)
            ws_results.append({"model": label, "dim": dim, "macro_f1": f1})
        mean = np.mean([r["macro_f1"] for r in ws_results if r["model"] == label])
        log.info("  %-35s  mean=%.3f", label, mean)

    # B0: Majority
    log.info("\n── B0 Majority (within-subject) ─────────────────────────────")
    for d_i, dim in enumerate(VAD_DIMS):
        m_tr = y_tr_ws[:, d_i] >= 0
        m_te = y_te_ws[:, d_i] >= 0
        if not m_tr.any() or not m_te.any():
            continue
        maj  = np.bincount(y_tr_ws[m_tr, d_i]).argmax()
        pred = np.full(m_te.sum(), maj)
        f1   = f1_score(y_te_ws[m_te, d_i], pred, average="macro", zero_division=0)
        ws_results.append({"model": "B0 Majority", "dim": dim, "macro_f1": f1})
    log.info("  mean=%.3f", np.mean([r["macro_f1"] for r in ws_results if r["model"] == "B0 Majority"]))

    # B1: SVM raw (no preprocessing)
    log.info("\n── B1–B4 SVM variants (within-subject) ─────────────────────")
    X_tr_raw = build_X(tr_ws, key_order)
    X_te_raw = build_X(te_ws, key_order)
    _svm_ws(X_tr_raw, X_te_raw, "B1 SVM raw")

    # B2: SVM + T0 baseline normalisation
    X_tr_t0 = build_X_enhanced(tr_ws, key_order, t0_means=t0_means)
    X_te_t0 = build_X_enhanced(te_ws, key_order, t0_means=t0_means)
    _svm_ws(X_tr_t0, X_te_t0, "B2 SVM + T0 norm")

    # B3: SVM + trial-position feature
    X_tr_tp = build_X_enhanced(tr_ws, key_order_tp, trial_pos_map=trial_pos_map)
    X_te_tp = build_X_enhanced(te_ws, key_order_tp, trial_pos_map=trial_pos_map)
    _svm_ws(X_tr_tp, X_te_tp, "B3 SVM + TrialPos")

    # B4: SVM + T0 norm + trial position (full preprocessing + temporal)
    X_tr_full = build_X_enhanced(tr_ws, key_order_tp, t0_means=t0_means, trial_pos_map=trial_pos_map)
    X_te_full = build_X_enhanced(te_ws, key_order_tp, t0_means=t0_means, trial_pos_map=trial_pos_map)
    _svm_ws(X_tr_full, X_te_full, "B4 SVM + T0norm + TrialPos")

    # ── Summary: Protocol B ───────────────────────────────────────────────
    log.info("\n══ Protocol B  within-subject comparison (macro-F1, 3-class) ══")
    ws_df = pd.DataFrame(ws_results)
    pivot_b = ws_df.pivot_table(index="model", columns="dim",
                                values="macro_f1", aggfunc="mean")
    pivot_b["mean"] = pivot_b[list(VAD_DIMS)].mean(axis=1)
    log.info("\n%s", pivot_b.sort_index().to_string())


if __name__ == "__main__":
    main()
