"""audio_modality_experiment.py

Ablation experiment: contribution of the DPA microphone audio modality
to VAD (Valence / Arousal / Dominance) classification.

The current model pipeline uses five physiological modalities from Tobii and
EmotiBit: gaze, pupil, EDA, PPG, IMU.  The DPA microphones capture per-seat
audio that was processed into:
  - 34 acoustic features (MFCC 1-13 mean/std, RMS energy, ZCR,
    spectral centroid/rolloff, speech activity fraction)
  - 7 speech transcript features (speaking time, utterance count, word count,
    energy mean/ratio, backchannel fraction, speech fraction)

This script runs a systematic SVM (RBF kernel, 49+41=90-d features max)
ablation measuring each modality group's incremental contribution.

Feature combinations tested
----------------------------
physio        — 5 physiological modalities only (baseline, matches svm_aug A0)
audio         — 34 acoustic (librosa) features only
speech        — 7 transcript features only
audio+speech  — 41 audio+speech features only
physio+audio  — physio + 34 acoustic features  ← key comparison
physio+speech — physio + 7 transcript features
physio+all    — physio + audio + speech (full feature set)

Each variant is evaluated with and without AP1 (BFI-weighted) augmentation.

Protocol
--------
  Train:  T0 + T1 tasks (same as all other experiments)
  Val:    T2 (used here for early-stopping / debug only)
  Test:   T3 (held-out)
  Model:  RBF-SVM with StandardScaler + SimpleImputer (handles NaN audio rows)

Missing audio handling
-----------------------
~11% of windows (34/296) have NaN audio features because the corresponding
audio file was unavailable (lost/corrupted during recording).  These are
imputed with the training-set mean after scaling (= 0 in standardised space).

Usage
-----
  python tools/mumt/audio_modality_experiment.py
  python tools/mumt/audio_modality_experiment.py \\
      --dataset  data/mumt/dataset_15s_speech.pkl \\
      --pool     data/mumt/augmented_pool_slow.pkl \\
      --out      results/audio_modality_experiment.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import flatten_features
from train_simple import (
    task_split,
    compute_tertile_thresholds,
    bin_vad_from_thresholds,
    MODALITY_COLS,
)
from svm_aug_comparison import (
    compute_bfi_similarity_map,
    get_pool_pseudo_labels_bfi,
    recompute_soft_labels,
)

VAD_DIMS = ["valence", "arousal", "dominance"]

# ── Feature group definitions ────────────────────────────────────────────────

PHYSIO_COLS = ["gaze_features", "pupil_features", "eda_features", "ppg_features", "imu_features"]
AUDIO_COLS  = ["audio_features"]
SPEECH_COLS = ["speech_features"]
ALL_COLS    = PHYSIO_COLS + AUDIO_COLS + SPEECH_COLS

FEATURE_GROUPS: dict[str, list[str]] = {
    "physio":       PHYSIO_COLS,
    "audio":        AUDIO_COLS,
    "speech":       SPEECH_COLS,
    "audio+speech": AUDIO_COLS + SPEECH_COLS,
    "physio+audio": PHYSIO_COLS + AUDIO_COLS,
    "physio+speech": PHYSIO_COLS + SPEECH_COLS,
    "physio+all":   ALL_COLS,
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def collect_feats(row: pd.Series, feat_cols: list[str], key_order: list[str]) -> np.ndarray:
    """Merge feature dicts from *feat_cols* and flatten to fixed-length vector."""
    merged: dict = {}
    for col in feat_cols:
        fd = row.get(col, {})
        if isinstance(fd, dict):
            merged.update(fd)
    return flatten_features(merged, key_order=key_order)


def build_key_order(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """Collect the union of all feature keys from *feat_cols* in *df*."""
    keys: set[str] = set()
    for col in feat_cols:
        if col not in df.columns:
            continue
        for v in df[col]:
            if isinstance(v, dict):
                keys.update(v.keys())
    return sorted(keys)


def build_X(df: pd.DataFrame, feat_cols: list[str], key_order: list[str]) -> np.ndarray:
    """Return (N, K) feature matrix.  NaN where feature is missing."""
    rows = [collect_feats(row, feat_cols, key_order) for _, row in df.iterrows()]
    X = np.stack(rows, axis=0).astype(np.float32)
    # Replace inf/−inf with NaN so SimpleImputer handles them
    X = np.where(np.isfinite(X), X, np.nan)
    return X


def get_hard_labels(df: pd.DataFrame,
                    thresholds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Return (N, 3) int64.  Dominance NaN → -1."""
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


# ── AP1 augmentation helpers ──────────────────────────────────────────────────

def get_ap1_aug_arrays(
    pool_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feat_cols: list[str],
    key_order: list[str],
    thresholds: dict[str, tuple[float, float]],
    conf_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_aug, Y_aug) for AP1 augmentation.

    Uses the correct BFI cosine similarity weighting from svm_aug_comparison.py.
    Pool rows that lack audio/speech features get NaN values; SimpleImputer
    in the SVM pipeline fills those with training-set mean (= 0 in std space).
    """
    bfi_sim_map = compute_bfi_similarity_map(train_df)

    # Filter pool to training tasks only
    pool_train = pool_df[pool_df["task"].isin(["T0", "T1"])].reset_index(drop=True)

    labels, weights, mask = get_pool_pseudo_labels_bfi(
        pool_train, thresholds, conf_threshold, bfi_sim_map, use_bfi_only=True,
    )

    # Build feature rows for pool windows — no audio/speech in pool, so those
    # dimensions will be NaN and filled by imputer with training-set mean.
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    n_added = 0
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
        n_added += 1

    log.info("  AP1 pool: %d / %d windows accepted", n_added, len(pool_train))
    if not X_list:
        return np.empty((0, len(key_order)), dtype=np.float32), np.empty((0, 3), dtype=np.int64)
    return np.stack(X_list), np.stack(Y_list)


# ── Run one SVM variant ───────────────────────────────────────────────────────

def run_svm_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    key_order: list[str],
    thresholds: dict[str, tuple[float, float]],
    pool_df: pd.DataFrame | None = None,
    ap1_weight_threshold: float = 0.5,
) -> dict[str, float]:
    """Train RBF-SVM and return per-dim macro-F1."""
    X_train = build_X(train_df, feat_cols, key_order)
    y_train = get_hard_labels(train_df, thresholds)
    X_test  = build_X(test_df,  feat_cols, key_order)
    y_test  = get_hard_labels(test_df,  thresholds)

    if pool_df is not None:
        X_aug, Y_aug = get_ap1_aug_arrays(
            pool_df, train_df, feat_cols, key_order, thresholds, ap1_weight_threshold,
        )
        if len(X_aug):
            X_train = np.concatenate([X_train, X_aug], axis=0)
            y_train = np.concatenate([y_train, Y_aug], axis=0)

    results: dict[str, float] = {}
    for ci, dim in enumerate(VAD_DIMS):
        y_tr = y_train[:, ci]
        y_te = y_test[:, ci]

        # Mask dominance sentinels (-1) from both train and test
        tr_valid = y_tr >= 0
        te_valid = y_te >= 0
        X_tr_d, y_tr_d = X_train[tr_valid], y_tr[tr_valid]
        X_te_d, y_te_d = X_test[te_valid],  y_te[te_valid]

        if len(np.unique(y_tr_d)) < 2 or len(X_te_d) == 0:
            results[f"{dim}_f1"] = 0.0
            continue

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("svm",     SVC(kernel="rbf", C=1.0, gamma="scale",
                            class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr_d, y_tr_d)
        preds = pipe.predict(X_te_d)
        f1 = float(f1_score(y_te_d, preds, average="macro", zero_division=0))
        results[f"{dim}_f1"] = f1

    f1s = [results[f"{d}_f1"] for d in VAD_DIMS]
    results["mean_f1"] = float(np.mean(f1s))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Audio modality ablation: contribution of DPA-mic features to VAD",
    )
    p.add_argument("--dataset",  default="data/mumt/dataset_15s_speech.pkl",
                   help="Pickle with audio_features + speech_features columns")
    p.add_argument("--audio-only-dataset", default="data/mumt/dataset_15s_audio.pkl",
                   help="Pickle with audio_features only (no speech); used as fallback")
    p.add_argument("--pool",     default="data/mumt/augmented_pool_slow.pkl",
                   help="AP1 augmented pool (physio-only; audio features imputed as 0)")
    p.add_argument("--out",      default="results/audio_modality_experiment.csv",
                   help="Output CSV path")
    p.add_argument("--no-ap1",   action="store_true",
                   help="Skip AP1 augmentation variants")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="AP1 confidence threshold (default 0.5)")
    args = p.parse_args()

    import pickle
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.warning("Speech dataset not found, falling back to audio-only dataset")
        dataset_path = Path(args.audio_only_dataset)
    if not dataset_path.exists():
        log.error("No suitable dataset found at %s or %s", args.dataset, args.audio_only_dataset)
        sys.exit(1)

    log.info("Loading dataset: %s", dataset_path)
    with open(dataset_path, "rb") as f:
        df = pickle.load(f)
    log.info("Dataset: %d rows, columns: %s", len(df),
             [c for c in df.columns if c.endswith("_features") or c.endswith("_seq")])

    # Check which feature groups are available
    available_groups: dict[str, list[str]] = {}
    for name, cols in FEATURE_GROUPS.items():
        avail_cols = [c for c in cols if c in df.columns]
        if avail_cols:
            available_groups[name] = avail_cols
            missing = [c for c in cols if c not in df.columns]
            if missing:
                log.warning("Group '%s': missing columns %s (will zero-pad)", name, missing)
        else:
            log.warning("Skipping group '%s' — no columns found", name)
    log.info("Available groups: %s", list(available_groups.keys()))

    # Task split: test on T3 (full V/A/D labels); T4 has no Dominance labels
    train_df, val_df, test_df = task_split(df, test_task="T3")
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Compute thresholds from training set only
    thresholds = compute_tertile_thresholds(train_df)

    # Load augmented pool for AP1
    pool_df: pd.DataFrame | None = None
    pool_path = Path(args.pool)
    if not args.no_ap1 and pool_path.exists():
        with open(pool_path, "rb") as f:
            pool_df = pickle.load(f)
        log.info("AP1 pool: %d windows", len(pool_df))
    elif not args.no_ap1:
        log.warning("Pool not found at %s — skipping AP1 variants", args.pool)

    # Run all variants
    records: list[dict] = []
    for group_name, feat_cols in available_groups.items():
        # Build key order from both train and pool data
        key_order = build_key_order(df, feat_cols)
        log.info("\n[%s] %d features | columns: %s", group_name, len(key_order), feat_cols)

        # No augmentation
        log.info("  Running: %s (no aug)...", group_name)
        res_noapt = run_svm_variant(
            train_df, test_df, feat_cols, key_order, thresholds,
            pool_df=None,
        )
        records.append({"variant": group_name, "augmentation": "none", **res_noapt,
                        "n_features": len(key_order)})
        log.info("  → V=%.3f  A=%.3f  D=%.3f  mean=%.3f",
                 res_noapt["valence_f1"], res_noapt["arousal_f1"],
                 res_noapt["dominance_f1"], res_noapt["mean_f1"])

        # AP1 augmentation
        if pool_df is not None:
            log.info("  Running: %s + AP1...", group_name)
            res_ap1 = run_svm_variant(
                train_df, test_df, feat_cols, key_order, thresholds,
                pool_df=pool_df,
                ap1_weight_threshold=args.conf_threshold,
            )
            records.append({"variant": group_name, "augmentation": "AP1", **res_ap1,
                            "n_features": len(key_order)})
            log.info("  → V=%.3f  A=%.3f  D=%.3f  mean=%.3f",
                     res_ap1["valence_f1"], res_ap1["arousal_f1"],
                     res_ap1["dominance_f1"], res_ap1["mean_f1"])

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(["augmentation", "mean_f1"], ascending=[True, False])
    results_df.to_csv(out_path, index=False, float_format="%.4f")
    log.info("\nResults saved to: %s", out_path)

    # Pretty-print summary table
    print("\n" + "=" * 75)
    print(f"{'VARIANT':<20} {'AUG':<6} {'V F1':>7} {'A F1':>7} {'D F1':>7} {'Mean':>7} {'N':>5}")
    print("-" * 75)
    for _, row in results_df.iterrows():
        print(f"{row['variant']:<20} {row['augmentation']:<6} "
              f"{row['valence_f1']:7.3f} {row['arousal_f1']:7.3f} "
              f"{row['dominance_f1']:7.3f} {row['mean_f1']:7.3f} "
              f"{int(row['n_features']):5d}")
    print("=" * 75)

    # Highlight best
    best = results_df.loc[results_df["mean_f1"].idxmax()]
    print(f"\nBest: {best['variant']} + {best['augmentation']} "
          f"→ mean F1 = {best['mean_f1']:.3f}")
    # Delta vs physio baseline
    physio_rows = results_df[results_df["variant"] == "physio"]
    if not physio_rows.empty:
        physio_noapt = physio_rows[physio_rows["augmentation"] == "none"]["mean_f1"].values
        if len(physio_noapt):
            print(f"Physio-only baseline (no aug): {physio_noapt[0]:.3f}")
            print(f"Best delta over physio baseline: "
                  f"+{best['mean_f1'] - physio_noapt[0]:+.3f}")


if __name__ == "__main__":
    main()
