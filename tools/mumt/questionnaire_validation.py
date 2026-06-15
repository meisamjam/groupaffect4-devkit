"""questionnaire_validation.py

Validate SVM-predicted VAD against post-block questionnaire items.

For each participant × task, we compute the mean predicted VAD class
(0/1/2) across their ~10-15 windows, then correlate those session-level
predictions with their post-block questionnaire ratings (engagement,
overall_valence, mental_demand, satisfaction, perceived_control).

Two variants are compared:
  A0  No augmentation (standard SVM)
  A3  GSR arousal + GP V/D augmentation (augmented_pool_gsr.pkl)

Questionnaire data source:
  metadata/extracted stimuli answers/sub-01_ses-*_stimuli_answers_extracted.tsv
  Columns: wall_clock, lsl_clock, task, phase, response_type,
           participant, device_id, item_key, item_value

session_id in the TSV filename: sub-01_ses-{date}_grp-{N}_run01_...
session_id in dataset.pkl: "ses-{date}_grp-{N}_run01"  (directory name)

Usage
-----
  python tools/mumt/questionnaire_validation.py
  python tools/mumt/questionnaire_validation.py \
      --dataset  data/mumt/dataset_15s.pkl \
      --pool-gsr data/mumt/augmented_pool_gsr.pkl \
      --q-dir    "metadata/extracted stimuli answers" \
      --out      results/questionnaire_validation.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
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

# Post-block questionnaire items to correlate with each predicted VAD dimension
Q_HYPOTHESES = {
    "valence":   ["overall_valence", "satisfaction", "voice_inclusion"],
    "arousal":   ["engagement", "mental_demand"],
    "dominance": ["perceived_control", "equality_of_contribution"],
}

# Tasks that have post-block questionnaires
Q_TASKS = ["T1", "T2", "T3", "T4"]


# ── Questionnaire loading ──────────────────────────────────────────────────────

def _session_id_from_filename(fname: str) -> str | None:
    """Extract session_id (e.g. 'ses-20260309_grp-03_run01') from TSV filename."""
    m = re.search(r"(ses-\d+_grp-[^_]+_run\d+)", fname)
    return m.group(1) if m else None


def load_questionnaire_data(q_dir: Path) -> pd.DataFrame:
    """Load all post-block questionnaire responses from extracted TSV files.

    Returns long-format DataFrame with columns:
      session_id, task, participant, item_key, item_value (float)
    """
    frames = []
    for tsv in sorted(q_dir.glob("sub-01_ses-*_stimuli_answers_extracted.tsv")):
        ses_id = _session_id_from_filename(tsv.name)
        if ses_id is None:
            log.warning("Cannot parse session_id from %s — skipping", tsv.name)
            continue
        df = pd.read_csv(tsv, sep="\t")
        pb = df[df["response_type"] == "postblock"].copy()
        pb["session_id"] = ses_id
        pb["item_value"] = pd.to_numeric(pb["item_value"], errors="coerce")
        frames.append(pb[["session_id", "task", "participant", "item_key", "item_value"]])

    if not frames:
        raise FileNotFoundError(f"No TSV files found in {q_dir}")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d post-block questionnaire rows from %d sessions",
             len(combined), combined["session_id"].nunique())
    return combined


def pivot_questionnaire(q_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format: one row per (session_id, task, participant).

    Returns DataFrame with session_id, task, participant, and one column
    per questionnaire item (NaN if the item was not asked in that task).
    """
    pivot = q_long.pivot_table(
        index=["session_id", "task", "participant"],
        columns="item_key",
        values="item_value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


# ── Feature / label helpers ───────────────────────────────────────────────────

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


def get_hard_labels(df, thresholds):
    out = np.full((len(df), 3), -1, dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if not np.isnan(v):
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


def recompute_soft(pool, thresholds, dim):
    t1, t2 = thresholds[dim]
    mu  = pool[f"{dim}_mu"].fillna(5.0).values.astype(float)
    sig = np.clip(pool[f"{dim}_sigma"].fillna(1.5).values.astype(float), 1e-4, None)
    p_low  = norm.cdf(t1, loc=mu, scale=sig)
    p_high = 1.0 - norm.cdf(t2, loc=mu, scale=sig)
    p_mid  = np.clip(1.0 - p_low - p_high, 0.0, 1.0)
    soft   = np.stack([p_low, p_mid, p_high], axis=1).astype(np.float32)
    row_sums = np.where(soft.sum(1, keepdims=True) < 1e-8, 1.0, soft.sum(1, keepdims=True))
    return soft / row_sums


# ── SVM with optional pseudo-label augmentation ───────────────────────────────

def train_predict_svm(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    key_order: list[str],
    thresholds: dict,
    pool: pd.DataFrame | None = None,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """Train SVM and return predicted class matrix (N_predict, 3)."""
    train_X      = extract_X(train_df, key_order)
    train_labels = get_hard_labels(train_df, thresholds)
    predict_X    = extract_X(predict_df, key_order)

    preds = np.full((len(predict_df), 3), 1, dtype=np.int64)  # default Mid

    for col_i, dim in enumerate(VAD_DIMS):
        tr_y   = train_labels[:, col_i]
        tr_mask = tr_y >= 0

        tr_X_dim = train_X[tr_mask]
        tr_y_dim = tr_y[tr_mask]
        tr_w_dim = np.ones(tr_mask.sum(), dtype=np.float32)

        if pool is not None:
            train_tasks = set(train_df["task"].unique())
            p = pool[pool["task"].isin(train_tasks)].copy()
            wkey = f"{dim}_weight"
            if wkey in p.columns:
                conf = p[wkey].fillna(0.0).values.astype(float)
                if dim == "dominance":
                    conf = np.where(p["task"].values == "T4", 0.0, conf)
                soft = recompute_soft(p, thresholds, dim)
                pseudo = np.argmax(soft, axis=1).astype(np.int64)
                max_p  = soft[np.arange(len(p)), pseudo]
                accepted = (conf >= conf_threshold) & (max_p >= 0.5)
                if accepted.sum() > 0:
                    aug_X = extract_X(p[accepted], key_order)
                    tr_X_dim = np.concatenate([tr_X_dim, aug_X], axis=0)
                    tr_y_dim = np.concatenate([tr_y_dim, pseudo[accepted]], axis=0)
                    tr_w_dim = np.concatenate([
                        tr_w_dim, conf[accepted].astype(np.float32)
                    ], axis=0)

        if tr_mask.sum() < 3:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                          class_weight="balanced", random_state=42)),
        ])
        pipe.fit(tr_X_dim, tr_y_dim, svm__sample_weight=tr_w_dim)
        preds[:, col_i] = pipe.predict(predict_X)

    return preds


# ── Session-level aggregation ─────────────────────────────────────────────────

def aggregate_predictions(
    predict_df: pd.DataFrame,
    preds: np.ndarray,
) -> pd.DataFrame:
    """Aggregate per-window predictions to participant×task level.

    Returns DataFrame with: session_id, task, seat, valence_pred, arousal_pred, dominance_pred.
    """
    df = predict_df[["session_id", "task", "seat"]].copy().reset_index(drop=True)
    for col_i, dim in enumerate(VAD_DIMS):
        df[f"{dim}_pred"] = preds[:, col_i].astype(float)

    agg = df.groupby(["session_id", "task", "seat"]).mean(numeric_only=True).reset_index()
    return agg


# ── Correlation analysis ───────────────────────────────────────────────────────

def compute_correlations(
    pred_agg: pd.DataFrame,
    q_wide: pd.DataFrame,
    variant: str,
) -> list[dict]:
    """Pearson r between predicted VAD and questionnaire items.

    Join key: (session_id, task, seat ↔ participant).
    """
    # participant in questionnaire is P1/P2/P3/P4 == seat in dataset
    q = q_wide.rename(columns={"participant": "seat"})
    merged = pred_agg.merge(q, on=["session_id", "task", "seat"], how="inner")
    log.info("  Variant %s: %d matched rows", variant, len(merged))

    records = []
    for pred_dim, q_items in Q_HYPOTHESES.items():
        pred_col = f"{pred_dim}_pred"
        if pred_col not in merged.columns:
            continue
        for item in q_items:
            if item not in merged.columns:
                continue
            sub = merged[[pred_col, item]].dropna()
            if len(sub) < 5:
                continue
            r, p = pearsonr(sub[pred_col].values, sub[item].values)
            records.append({
                "variant":       variant,
                "pred_dim":      pred_dim,
                "q_item":        item,
                "pearson_r":     round(float(r), 4),
                "p_value":       round(float(p), 4),
                "n":             len(sub),
            })
            log.info("    %-10s ↔ %-30s  r=%+.3f  p=%.3f  n=%d",
                     pred_dim, item, r, p, len(sub))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool-gsr",  default="data/mumt/augmented_pool_gsr.pkl")
    parser.add_argument("--q-dir",     default="metadata/extracted stimuli answers")
    parser.add_argument("--out",       default="results/questionnaire_validation.csv")
    parser.add_argument("--test-tasks", nargs="+", default=["T1", "T2", "T3"])
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    log.info("Dataset: %d windows", len(df))

    pool_gsr = None
    if Path(args.pool_gsr).exists():
        pool_gsr = pd.read_pickle(args.pool_gsr)
        log.info("GSR pool: %d windows", len(pool_gsr))
    else:
        log.warning("GSR pool not found; A3 variant will be same as A0")

    q_long = load_questionnaire_data(Path(args.q_dir))
    q_wide = pivot_questionnaire(q_long)

    key_order = build_summary_key_order(df)

    all_records = []

    # For each test task we train on prior tasks and predict on the test task.
    # The test task windows are the ones we correlate with questionnaires.
    for test_task in args.test_tasks:
        if test_task not in Q_TASKS:
            log.warning("Task %s has no questionnaire data — skipping", test_task)
            continue
        log.info("\n── Test task: %s ──────────────────────────────", test_task)
        train_df, _, test_df = task_split(df, test_task=test_task)
        thresholds = compute_tertile_thresholds(train_df)

        for variant, pool in [("A0_no_aug", None), ("A3_gsr_vd", pool_gsr)]:
            log.info("  Variant: %s", variant)
            preds = train_predict_svm(
                train_df, test_df, key_order, thresholds,
                pool=pool, conf_threshold=0.5,
            )
            pred_agg = aggregate_predictions(test_df, preds)
            # Filter questionnaire to the test task
            q_task = q_wide[q_wide["task"] == test_task].copy()
            records = compute_correlations(pred_agg, q_task, f"{variant}_{test_task}")
            all_records.extend(records)

    if not all_records:
        log.error("No correlation results produced — check session_id format matching")
        return

    out_df = pd.DataFrame(all_records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("\nSaved → %s", out_path)

    print("\n=== Post-block questionnaire correlations ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
