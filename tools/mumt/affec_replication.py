"""affec_replication.py

Replicates the AFFEC paper (itubrainlab/AFFEC, Zenodo 10.5281/zenodo.14794876)
baseline protocol, then tests AP1 personality-weighted augmentation.

AFFEC paper protocol (from GitHub):
  - 5-fold subject-disjoint grouped CV (GroupKFold on participant_id)
  - RandomForest classifier (or AutoGluon with --autogluon flag)
  - Fixed label thresholds:
      felt_arousal : low ≤ 4.6 < mid ≤ 6.0 < high
      felt_valence : low ≤ 4.3 < mid ≤ 5.2 < high
  - Modalities: gaze + pupil + GSR/EDA + IMU + BFI personality traits
  - Targets: felt valence, felt arousal (no dominance in AFFEC)

Expected AFFEC paper baselines (full multimodal):
  felt_arousal  ≈ 0.478 ± 0.014 macro F1
  felt_valence  ≈ 0.460 ± 0.019 macro F1

AP1 augmentation (our addition, tested in same 5-fold protocol):
  For each test-fold subject S, generate pseudo-labels for each stimulus
  using BFI cosine-similarity–weighted averages of training subjects'
  labels on the same stimulus category.  Pseudo-labelled data are added
  to the training set; actual S labels remain held-out for evaluation.
  Since AFFEC BFI similarities cluster tightly (mean ≈ 0.97, std ≈ 0.02)
  this acts as a near-uniform population mean.

Usage:
  python tools/mumt/affec_replication.py --dataset data/mumt/dataset_affec.pkl
  python tools/mumt/affec_replication.py --dataset data/mumt/dataset_affec.pkl --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def flatten_features(feat_dict: dict, key_order: list[str] | None = None) -> np.ndarray:
    if key_order is not None:
        vals = [float(feat_dict.get(k, 0.0) or 0.0) for k in key_order]
        return np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.array([float(v or 0.0) for v in feat_dict.values()], dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------------------------------------------------------
# Label thresholds — from AFFEC paper (itubrainlab/AFFEC GitHub)
# ---------------------------------------------------------------------------

AFFEC_THRESHOLDS = {
    "arousal": (4.6, 6.0),   # felt_arousal
    "valence": (4.3, 5.2),   # felt_valence
}

VAD_DIMS = ["valence", "arousal"]

BFI_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
FEAT_COLS = ["gaze_features", "pupil_features", "eda_features", "imu_features"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bin_affec(val: float, thresholds: tuple[float, float]) -> int:
    """Map continuous 1-9 rating to 3-class label using AFFEC fixed thresholds."""
    lo, hi = thresholds
    if val <= lo:
        return 0
    if val <= hi:
        return 1
    return 2


def build_key_order(df: pd.DataFrame, include_bfi: bool = True) -> list[str]:
    keys: set[str] = set()
    for col in FEAT_COLS:
        for feat in df[col]:
            if isinstance(feat, dict):
                keys.update(feat.keys())
    if include_bfi:
        for c in BFI_COLS:
            keys.add(c)
    return sorted(keys)


def build_X(df: pd.DataFrame, key_order: list[str], include_bfi: bool = True) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        feats: dict[str, float] = {}
        for col in FEAT_COLS:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update({k: float(v) for k, v in fd.items()})
        if include_bfi:
            for c in BFI_COLS:
                feats[c] = float(r.get(c, 0.0) or 0.0)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.stack(rows, axis=0).astype(np.float32)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) label array [valence, arousal] using AFFEC fixed thresholds."""
    out = np.full((len(df), len(VAD_DIMS)), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t = AFFEC_THRESHOLDS[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if not np.isnan(v):
                out[row_i, col_i] = bin_affec(v, t)
    return out


def bfi_cosine(a: np.ndarray, b: np.ndarray, floor: float = 0.05) -> float:
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)
    if n1 < 1e-9 or n2 < 1e-9:
        return floor
    return max(floor, float(np.dot(a, b) / (n1 * n2)))


def build_subject_bfi(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {subject_id: bfi_vector} from the dataset."""
    subj_bfi: dict[str, np.ndarray] = {}
    for subj, sdf in df.groupby("subject_id"):
        row = sdf.iloc[0]
        vec = np.array([float(row.get(c, 0.0) or 0.0) for c in BFI_COLS], dtype=np.float64)
        subj_bfi[str(subj)] = vec
    return subj_bfi


# ---------------------------------------------------------------------------
# AP1 augmentation: BFI-weighted cross-subject pseudo-labels
# ---------------------------------------------------------------------------

def ap1_augment(
    train_df: pd.DataFrame,
    test_subjects: list[str],
    df_all: pd.DataFrame,
    key_order: list[str],
    include_bfi: bool = True,
    sim_floor: float = 0.05,
    same_stim_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate AP1 pseudo-labelled data for held-out test subjects.

    For each test subject S and each stimulus category V they observed:
      pseudo_label[S, V, dim] = sum_T( w(S,T) * label(T, V, dim) ) / sum_T( w(S,T) )
    where T ranges over training subjects and w(S, T) is BFI cosine similarity.

    Returns (X_aug, y_aug) — feature matrix and label matrix for pseudo-labelled
    data (one row per test-subject × stimulus-group).  These are added to the
    training set before fitting.

    Parameters
    ----------
    same_stim_only: if True, only aggregate over training trials with matching
        trial_type (stimulus category); otherwise aggregate over all training data.
    """
    subj_bfi = build_subject_bfi(df_all)
    train_subj_bfi = {s: subj_bfi[s] for s in train_df["subject_id"].unique() if s in subj_bfi}

    aug_X: list[np.ndarray] = []
    aug_y: list[np.ndarray] = []

    for test_subj in test_subjects:
        s_bfi = subj_bfi.get(str(test_subj))
        if s_bfi is None:
            continue
        test_rows = df_all[df_all["subject_id"] == test_subj]
        if test_rows.empty:
            continue

        # Compute similarity weights to all training subjects
        weights = {
            t: bfi_cosine(s_bfi, t_bfi, floor=sim_floor)
            for t, t_bfi in train_subj_bfi.items()
        }

        # Group training data by stimulus category
        stim_groups = train_df.groupby("trial_type") if same_stim_only else {"_all": train_df}

        for stim_cat, stim_rows in (
            train_df.groupby("trial_type") if same_stim_only else [("_all", train_df)]
        ):
            # Get test rows for this stimulus
            if same_stim_only:
                s_stim = test_rows[test_rows["trial_type"] == stim_cat]
            else:
                s_stim = test_rows

            if s_stim.empty:
                continue

            # Compute weighted pseudo-label for this stimulus group
            total_w = 0.0
            weighted_labels = np.zeros(len(VAD_DIMS), dtype=np.float64)
            for t_subj, t_rows in stim_rows.groupby("subject_id"):
                w = weights.get(str(t_subj), sim_floor)
                for col_i, dim in enumerate(VAD_DIMS):
                    valid = pd.to_numeric(t_rows[dim], errors="coerce").dropna()
                    if not valid.empty:
                        weighted_labels[col_i] += w * valid.mean()
                total_w += w

            if total_w < 1e-9:
                continue

            pseudo_label_raw = weighted_labels / total_w  # continuous weighted mean

            # Bin the pseudo-label using AFFEC thresholds
            pseudo_label_binned = np.array([
                bin_affec(float(pseudo_label_raw[col_i]), AFFEC_THRESHOLDS[dim])
                for col_i, dim in enumerate(VAD_DIMS)
            ], dtype=np.int64)

            # One synthetic training row per test-subject stimulus occurrence
            for _, row in s_stim.iterrows():
                row_df = pd.DataFrame([row])
                x = build_X(row_df, key_order, include_bfi=include_bfi)
                aug_X.append(x[0])
                aug_y.append(pseudo_label_binned)

    if not aug_X:
        return np.zeros((0, len(key_order)), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)

    return np.stack(aug_X), np.stack(aug_y)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    classifier: str = "rf",
    augment: str = "none",
    include_bfi: bool = True,
    same_stim_only: bool = True,
) -> dict[str, list[float]]:
    """5-fold subject-disjoint GroupKFold CV matching the AFFEC paper protocol.

    Parameters
    ----------
    augment : 'none' | 'ap1'
    """
    key_order = build_key_order(df, include_bfi=include_bfi)
    log.info("Feature dim: %d (bfi=%s)", len(key_order), include_bfi)

    groups = df["subject_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_folds)

    scores: dict[str, list[float]] = {d: [] for d in VAD_DIMS}
    label_dist: dict[str, list] = {d: [] for d in VAD_DIMS}

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[tr_idx].copy()
        test_df  = df.iloc[te_idx].copy()

        X_tr = build_X(train_df, key_order, include_bfi)
        X_te = build_X(test_df,  key_order, include_bfi)
        y_tr = get_labels(train_df)
        y_te = get_labels(test_df)

        # AP1 augmentation
        if augment == "ap1":
            test_subjects = list(test_df["subject_id"].unique())
            X_aug, y_aug = ap1_augment(
                train_df, test_subjects, df, key_order,
                include_bfi=include_bfi, same_stim_only=same_stim_only,
            )
            if len(X_aug) > 0:
                X_tr = np.vstack([X_tr, X_aug])
                y_tr = np.vstack([y_tr, y_aug])
                log.info("  Fold %d: added %d AP1 pseudo-labelled rows", fold_i, len(X_aug))

        for d_i, dim in enumerate(VAD_DIMS):
            mask_tr = y_tr[:, d_i] >= 0
            mask_te = y_te[:, d_i] >= 0
            if mask_tr.sum() < 3 or mask_te.sum() < 3:
                continue

            if classifier == "rf":
                clf = Pipeline([
                    ("clf", RandomForestClassifier(
                        n_estimators=200, max_depth=None,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )),
                ])
            else:  # svm
                clf = Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
                ])

            clf.fit(X_tr[mask_tr], y_tr[mask_tr, d_i])
            pred = clf.predict(X_te[mask_te])
            f1 = f1_score(y_te[mask_te, d_i], pred, average="macro", zero_division=0)
            scores[dim].append(f1)

            # Label distribution in test fold
            y_te_valid = y_te[mask_te, d_i]
            unique, counts = np.unique(y_te_valid, return_counts=True)
            label_dist[dim].append(dict(zip(unique.tolist(), counts.tolist())))

        log.info("  Fold %d | V=%.3f A=%.3f  (test subj=%d, n_train=%d, n_test=%d)",
                 fold_i,
                 scores["valence"][-1] if scores["valence"] else float("nan"),
                 scores["arousal"][-1]  if scores["arousal"]  else float("nan"),
                 test_df["subject_id"].nunique(), len(train_df), len(test_df))

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AFFEC replication: 5-fold grouped CV + RandomForest + AP1 augmentation"
    )
    ap.add_argument("--dataset", default="data/mumt/dataset_affec.pkl", type=Path)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--classifier", choices=["rf", "svm"], default="rf",
                    help="rf = RandomForest (paper default); svm = RBF-SVM")
    ap.add_argument("--no-bfi", action="store_true",
                    help="Exclude BFI personality from features (ablation)")
    args = ap.parse_args()

    log.info("Loading %s", args.dataset)
    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d trials, %d subjects", len(df), df["subject_id"].nunique())

    # Label distribution using AFFEC thresholds
    for dim in VAD_DIMS:
        t = AFFEC_THRESHOLDS[dim]
        vals = pd.to_numeric(df[dim], errors="coerce").dropna()
        binned = vals.apply(lambda v: bin_affec(v, t))
        log.info("  %s thresholds=(%.1f,%.1f)  Low=%d Mid=%d High=%d",
                 dim, t[0], t[1],
                 int((binned==0).sum()), int((binned==1).sum()), int((binned==2).sum()))

    # BFI similarity distribution (confirms near-uniform for AFFEC)
    subj_bfi = build_subject_bfi(df)
    bfi_mat = np.stack(list(subj_bfi.values()))
    sims = []
    for i in range(len(bfi_mat)):
        for j in range(i+1, len(bfi_mat)):
            a, b = bfi_mat[i], bfi_mat[j]
            sims.append(bfi_cosine(a, b))
    log.info("BFI cosine sim: mean=%.4f std=%.4f min=%.4f max=%.4f  (near-uniform = AP1 ~ mean-label)",
             np.mean(sims), np.std(sims), np.min(sims), np.max(sims))

    include_bfi = not args.no_bfi
    results = {}

    # ── A0: baseline (no augmentation) ──────────────────────────────────────
    log.info("\n── A0 %s baseline (%d-fold grouped CV) ────────────────────────",
             args.classifier.upper(), args.n_folds)
    scores_a0 = run_cv(df, n_folds=args.n_folds, classifier=args.classifier,
                       augment="none", include_bfi=include_bfi)
    results["A0"] = scores_a0

    # ── AP1_stim: same-stimulus BFI-weighted augmentation ───────────────────
    log.info("\n── AP1_stim %s (%d-fold, same-stimulus BFI weights) ────────────",
             args.classifier.upper(), args.n_folds)
    scores_ap1 = run_cv(df, n_folds=args.n_folds, classifier=args.classifier,
                        augment="ap1", include_bfi=include_bfi, same_stim_only=True)
    results["AP1_stim"] = scores_ap1

    # ── AP1_global: cross-all-stimuli BFI augmentation ──────────────────────
    log.info("\n── AP1_global %s (%d-fold, all-stim BFI weights) ────────────────",
             args.classifier.upper(), args.n_folds)
    scores_ap1g = run_cv(df, n_folds=args.n_folds, classifier=args.classifier,
                         augment="ap1", include_bfi=include_bfi, same_stim_only=False)
    results["AP1_global"] = scores_ap1g

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n══ AFFEC replication results (%s, %d-fold grouped CV) ══",
             args.classifier.upper(), args.n_folds)
    log.info("Fixed thresholds:  arousal=(4.6,6.0)  valence=(4.3,5.2)")
    log.info("AFFEC paper target: felt_arousal≈0.478  felt_valence≈0.460")
    log.info("")

    header = f"{'Variant':<14} {'Valence':>10} {'Arousal':>10} {'Mean':>10}"
    log.info(header)
    log.info("-" * len(header))

    for name, sc in results.items():
        v_scores = sc.get("valence", [])
        a_scores = sc.get("arousal", [])
        v_mean = np.mean(v_scores) if v_scores else float("nan")
        v_std  = np.std(v_scores)  if v_scores else float("nan")
        a_mean = np.mean(a_scores) if a_scores else float("nan")
        a_std  = np.std(a_scores)  if a_scores else float("nan")
        mean_f1 = np.mean([v_mean, a_mean]) if v_scores and a_scores else float("nan")
        log.info("%-14s  %5.3f±%.3f  %5.3f±%.3f  %5.3f",
                 name, v_mean, v_std, a_mean, a_std, mean_f1)

    log.info("")
    log.info("AP1 delta over A0:")
    for aug_name in ("AP1_stim", "AP1_global"):
        sc_a = results["A0"]
        sc_b = results[aug_name]
        for dim in VAD_DIMS:
            a0 = np.mean(sc_a.get(dim, [float("nan")]))
            ap = np.mean(sc_b.get(dim, [float("nan")]))
            log.info("  %-12s %-9s  Δ=%+.3f", aug_name, dim, ap - a0)


if __name__ == "__main__":
    main()
