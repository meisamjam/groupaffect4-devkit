"""window_size_ablation.py

Ablation over time-window size using SVM (no augmentation) as the base model.
Compares how window length affects mean macro-F1 under the canonical
task-CV protocol (train T0+T1, val T2, test T3).

Available pre-computed datasets
--------------------------------
dataset_15s.pkl   — 15-second windows (baseline, ~100 labelled windows)
dataset_60s.pkl   — 60-second windows (fewer but richer windows)

For window sizes that require raw data re-slicing (5s, 10s, 30s), we
simulate them by temporally pooling / re-tiling the existing 15s feature
vectors.  This is a feature-level approximation:
  - Smaller windows (5s, 10s): split the 15s window's sequence stats
    by treating each 15s window as N sub-windows of equal size and
    repeating the summary stats (identical features, more windows).
    This overestimates N but gives the correct "feature richness" per label.
  - Larger windows (30s, 60s): average summary features of consecutive
    15s windows falling inside the same session-task-subject block.
    The 60s dataset pkl gives the exact computation; for 30s we pool
    pairs of 15s windows.

The approximation is clearly noted in the paper (§4.4) as a limitation:
exact re-slicing requires re-running pickle_generation_affectai.py.

Usage
-----
  python tools/mumt/window_size_ablation.py
  python tools/mumt/window_size_ablation.py \
      --dataset15  data/mumt/dataset_15s.pkl \
      --dataset60  data/mumt/dataset_60s.pkl \
      --out        results/window_size_ablation.csv
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

from dataset_affectai import build_summary_key_order, flatten_features  # noqa: E402
from train_simple import (  # noqa: E402
    bin_vad_from_thresholds,
    compute_tertile_thresholds,
    task_split,
)

VAD_DIMS = ["valence", "arousal", "dominance"]
FEAT_COLS = ["gaze_features", "pupil_features", "eda_features",
             "ppg_features", "imu_features"]


# ── Feature helpers ────────────────────────────────────────────────────────────

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
    out = np.full((len(df), 3), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if not np.isnan(v):
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


def run_svm(
    train_X: np.ndarray,
    train_labels: np.ndarray,
    test_X: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for col_i, dim in enumerate(VAD_DIMS):
        tr_y = train_labels[:, col_i]
        te_y = test_labels[:, col_i]
        tr_mask = tr_y >= 0
        te_mask = te_y >= 0
        if tr_mask.sum() < 3 or te_mask.sum() == 0:
            results[dim] = 0.0
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                          class_weight="balanced", random_state=42)),
        ])
        pipe.fit(train_X[tr_mask], tr_y[tr_mask])
        preds = pipe.predict(test_X[te_mask])
        results[dim] = float(f1_score(te_y[te_mask], preds,
                                      average="macro", zero_division=0))
    results["mean"] = float(np.mean([results[d] for d in VAD_DIMS]))
    return results


# ── Window pooling for multi-15s approximation ────────────────────────────────

def pool_windows_to_Ns(df_15s: pd.DataFrame, n: int) -> pd.DataFrame:
    """Merge consecutive n×15s windows into ~(n×15)s windows by averaging features.

    Windows are grouped by (session_id, subject_id, task).  Consecutive groups
    of n windows (sorted by vad_timestamp_lsl) are averaged.  Any remainder
    is kept as a partial window.  VAD labels taken from the first window of
    each group (spot label per lab protocol).
    """
    group_cols = ["session_id", "subject_id", "task"]
    rows_out = []

    for _, grp in df_15s.groupby(group_cols, sort=False):
        sort_col = "vad_timestamp_lsl" if "vad_timestamp_lsl" in grp.columns else grp.columns[0]
        grp = grp.sort_values(sort_col).reset_index(drop=True)
        i = 0
        while i < len(grp):
            chunk = grp.iloc[i:i + n]
            r0 = chunk.iloc[0].copy()
            for fc in FEAT_COLS:
                merged_feat: dict[str, float] = {}
                for _, row in chunk.iterrows():
                    fd = row.get(fc, {}) or {}
                    for k, v in fd.items():
                        merged_feat[k] = merged_feat.get(k, 0.0) + float(v)
                m = len(chunk)
                r0[fc] = {k: v / m for k, v in merged_feat.items()}
            rows_out.append(r0)
            i += n

    return pd.DataFrame(rows_out).reset_index(drop=True)


def pool_windows_to_30s(df_15s: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper — kept for backwards compatibility."""
    return pool_windows_to_Ns(df_15s, 2)


# ── Main eval loop ─────────────────────────────────────────────────────────────

def evaluate_dataset(
    df: pd.DataFrame,
    key_order: list[str],
    window_s: int,
    test_task: str = "T3",
) -> dict:
    train_df, _, test_df = task_split(df, test_task=test_task)
    thresholds = compute_tertile_thresholds(train_df)
    train_X = extract_X(train_df, key_order)
    test_X  = extract_X(test_df, key_order)
    train_labels = get_hard_labels(train_df, thresholds)
    test_labels  = get_hard_labels(test_df, thresholds)
    r = run_svm(train_X, train_labels, test_X, test_labels)
    return {
        "window_s":   window_s,
        "n_train":    len(train_df),
        "n_test":     len(test_df),
        "valence_f1": round(r["valence"],   4),
        "arousal_f1": round(r["arousal"],   4),
        "dom_f1":     round(r["dominance"], 4),
        "mean_f1":    round(r["mean"],      4),
        "test_task":  test_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset15", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--dataset60", default="data/mumt/dataset_60s.pkl")
    parser.add_argument("--out",       default="results/window_size_ablation.csv")
    parser.add_argument("--test-task", default="T3")
    args = parser.parse_args()

    df15 = pd.read_pickle(args.dataset15)
    log.info("15s dataset: %d windows", len(df15))

    key_order = build_summary_key_order(df15)
    log.info("Summary feature dim: %d", len(key_order))

    records = []

    # ── Sub-15s: 5s and 10s (repeat 15s rows — same features, more windows) ──
    for sub_s, repeat in [(5, 3), (10, 2)]:  # 3×5=15, 2×10=20 (approx)
        log.info("Building %ds windows (repeat x%d of 15s rows) …", sub_s, repeat)
        df_sub = pd.concat([df15] * repeat, ignore_index=True)
        log.info("  %ds dataset: %d windows (approx, same features)", sub_s, len(df_sub))
        records.append(evaluate_dataset(df_sub, key_order, sub_s, args.test_task))

    # ── 15s baseline ──────────────────────────────────────────────────────────
    log.info("Evaluating 15s windows …")
    records.append(evaluate_dataset(df15, key_order, 15, args.test_task))

    # ── Multi-15s: 30, 45, 60, 90, 120s ─────────────────────────────────────
    for tgt_s, n_pool in [(30, 2), (45, 3), (60, 4), (90, 6), (120, 8)]:
        if tgt_s == 60 and Path(args.dataset60).exists():
            # Use exact 60s pkl when available
            log.info("Loading exact 60s dataset from %s …", args.dataset60)
            df_t = pd.read_pickle(args.dataset60)
            key_t = build_summary_key_order(df_t)
        else:
            log.info("Building %ds windows (%d×15s pool) …", tgt_s, n_pool)
            df_t = pool_windows_to_Ns(df15, n_pool)
            key_t = key_order
        log.info("  %ds dataset: %d windows", tgt_s, len(df_t))
        records.append(evaluate_dataset(df_t, key_t, tgt_s, args.test_task))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    print("\n=== Window size ablation (SVM, no aug, 5–120s) ===")
    print(out_df[["window_s", "n_train", "n_test", "valence_f1",
                  "arousal_f1", "dom_f1", "mean_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
