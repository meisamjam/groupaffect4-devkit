"""baselines.py

Publication-required baselines for the MuMT-Affect GroupAffect-4 dataset.

Baselines (all evaluated on T3-test task split):
  1. Majority-class  — always predict the most frequent class in training.
  2. Random          — random 3-class predictions (uniform).
  3. SVM             — StandardScaler + RBF SVM on flattened summary features.
  4. Per-modality ablation — MLP trained with one modality's summary features
                             zeroed out. Identifies which modalities carry affect signal.

Usage:
  python tools/mumt/baselines.py --dataset data/mumt/dataset_15s.pkl
  python tools/mumt/baselines.py --dataset data/mumt/dataset_15s.pkl \
      --augmented-pool data/mumt/augmented_pool.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import build_summary_key_order, flatten_features
from train_simple import (
    task_split, compute_tertile_thresholds, bin_vad_from_thresholds,
    fit_scalers, VADDataset, run_split, SoftVADLoss,
    MODALITY_COLS, MODALITY_NAMES,
)

VAD_DIMS = ["valence", "arousal", "dominance"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_summary_matrix(df: pd.DataFrame, key_order: list[str]) -> np.ndarray:
    """Return (N, K) float32 summary feature matrix."""
    rows = []
    for _, r in df.iterrows():
        feats: dict = {}
        for col in ["gaze_features", "pupil_features", "eda_features",
                    "ppg_features", "imu_features", "audio_features", "speech_features"]:
            fd = r.get(col, {})
            if isinstance(fd, dict):
                feats.update(fd)
        rows.append(flatten_features(feats, key_order=key_order))
    return np.stack(rows, axis=0).astype(np.float32)


def get_hard_labels(df: pd.DataFrame,
                    thresholds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Return (N, 3) int array.  Dominance NaN → -1."""
    out = np.zeros((len(df), 3), dtype=np.int64)
    for col_i, dim in enumerate(VAD_DIMS):
        t1, t2 = thresholds[dim]
        for row_i, val in enumerate(df[dim].values):
            v = float(val) if val is not None else float("nan")
            if col_i == 2 and np.isnan(v):
                out[row_i, col_i] = -1
            else:
                out[row_i, col_i] = bin_vad_from_thresholds(v, t1, t2)
    return out


def macro_f1_masked(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-F1 ignoring -1 dominance sentinels."""
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0))


# ── Baseline 1 & 2: Majority-class and Random ────────────────────────────────

def run_majority_random(train_labels: np.ndarray,
                        test_labels: np.ndarray) -> dict[str, dict]:
    results: dict[str, dict] = {"majority": {}, "random": {}}
    rng = np.random.default_rng(42)

    for col_i, dim in enumerate(VAD_DIMS):
        tr = train_labels[:, col_i]
        te = test_labels[:, col_i]
        valid_tr = tr[tr >= 0]
        valid_te = te[te >= 0]
        if len(valid_tr) == 0 or len(valid_te) == 0:
            results["majority"][dim] = 0.0
            results["random"][dim]   = 0.0
            continue

        counts = np.bincount(valid_tr.astype(int), minlength=3)
        maj_class = int(np.argmax(counts))
        maj_preds = np.full(len(valid_te), maj_class)
        results["majority"][dim] = float(
            f1_score(valid_te, maj_preds, average="macro", zero_division=0)
        )

        rand_preds = rng.integers(0, 3, size=len(valid_te))
        results["random"][dim] = float(
            f1_score(valid_te, rand_preds, average="macro", zero_division=0)
        )

    for name in ("majority", "random"):
        vals = list(results[name].values())
        results[name]["mean"] = float(np.mean(vals))

    return results


# ── Baseline 3: SVM ───────────────────────────────────────────────────────────

def run_svm(train_X: np.ndarray, train_labels: np.ndarray,
            test_X: np.ndarray, test_labels: np.ndarray) -> dict:
    results: dict = {}
    for col_i, dim in enumerate(VAD_DIMS):
        tr_y = train_labels[:, col_i]
        te_y = test_labels[:, col_i]
        # Mask dominance NaNs
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


# ── Baseline 4: Per-modality ablation ────────────────────────────────────────

def zero_modality_features(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return a copy of df with one modality's summary features zeroed out."""
    feat_col = f"{modality}_features"
    df2 = df.copy()
    df2[feat_col] = [{} for _ in range(len(df2))]  # empty dict → all zeros when flattened
    return df2


def run_modality_ablation(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    args: argparse.Namespace,
    device,
    aug_df: pd.DataFrame | None,
) -> dict[str, dict]:
    """Train one MLP per ablated modality, return per-modality test F1s."""
    import torch
    results: dict[str, dict] = {}

    for mod in MODALITY_NAMES:
        log.info("  Ablating modality: %s", mod)
        tr2   = zero_modality_features(train_df, mod)
        val2  = zero_modality_features(val_df,   mod)
        test2 = zero_modality_features(test_df,  mod)
        aug2  = zero_modality_features(aug_df, mod) if aug_df is not None else None

        m = run_split(tr2, val2, test2, args, device,
                      fold_tag=f"_ablate_{mod}", aug_df=aug2)
        results[mod] = {
            "valence":   m.get("valence_f1",   0),
            "arousal":   m.get("arousal_f1",   0),
            "dominance": m.get("dominance_f1", 0),
            "mean":      m.get("mean_f1",      0),
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def print_results_table(name: str, r: dict) -> None:
    v  = r.get("valence",   r.get("valence_f1",   0))
    a  = r.get("arousal",   r.get("arousal_f1",   0))
    d  = r.get("dominance", r.get("dominance_f1", 0))
    mn = r.get("mean",      r.get("mean_f1",      0))
    print(f"  {name:<28}  V={v:.3f}  A={a:.3f}  D={d:.3f}  Mean={mn:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--augmented-pool",  default="")
    parser.add_argument("--aug-frac",        type=float, default=0.5)
    parser.add_argument("--test-task",       default="T3")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch",           type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--wd",              type=float, default=1e-3)
    parser.add_argument("--warmup",          type=int,   default=10)
    parser.add_argument("--patience",        type=int,   default=60)
    parser.add_argument("--eval-every",      type=int,   default=5)
    parser.add_argument("--alpha",           type=float, default=0.05)
    parser.add_argument("--label-smooth",    type=float, default=0.1)
    parser.add_argument("--augment",         action="store_true")
    parser.add_argument("--arch",            default="mlp")
    parser.add_argument("--d-enc",           type=int,   default=32)
    parser.add_argument("--d-fuse",          type=int,   default=64)
    parser.add_argument("--t-out",           type=int,   default=16)
    parser.add_argument("--dropout",         type=float, default=0.5)
    parser.add_argument("--group-norm",      action="store_true")
    parser.add_argument("--t0-baseline",     action="store_true")
    parser.add_argument("--within-group-contrast", action="store_true",
                        dest="within_group_contrast")
    parser.add_argument("--ckpt-dir",        default="")
    parser.add_argument("--device",          default="auto")
    parser.add_argument("--skip-mlp-ablation", action="store_true",
                        help="Skip per-modality MLP ablation (fast mode).")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_pickle(args.dataset)
    log.info("Loaded %d windows from %s", len(df), args.dataset)

    aug_pool = None
    if args.augmented_pool and args.augmented_pool.strip():
        aug_pool = pd.read_pickle(args.augmented_pool.strip())
        log.info("Augmented pool: %d windows", len(aug_pool))

    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Task split (test=%s): train=%d  val=%d  test=%d",
             args.test_task, len(train_df), len(val_df), len(test_df))

    thresholds  = compute_tertile_thresholds(train_df)
    key_order   = build_summary_key_order(df)

    # Augmented pool filtered to train tasks only
    aug_train = None
    if aug_pool is not None:
        train_tasks = set(train_df["task"].unique())
        aug_train = aug_pool[aug_pool["task"].isin(train_tasks)].copy()
        log.info("Aug pool (train tasks only): %d windows", len(aug_train))

    # ── Feature matrices for SVM ──────────────────────────────────────────────
    log.info("Extracting summary features …")
    train_X = extract_summary_matrix(train_df, key_order)
    test_X  = extract_summary_matrix(test_df,  key_order)
    # Replace NaN/inf with 0
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_X  = np.nan_to_num(test_X,  nan=0.0, posinf=0.0, neginf=0.0)

    train_labels = get_hard_labels(train_df, thresholds)
    test_labels  = get_hard_labels(test_df,  thresholds)

    # ── Run baselines ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"{'Baselines — T3-test task split':^64}")
    print("=" * 64)

    # 1 & 2: Majority / Random
    log.info("Running majority-class and random baselines …")
    bl = run_majority_random(train_labels, test_labels)
    print_results_table("Random (uniform 3-class)", bl["random"])
    print_results_table("Majority-class",           bl["majority"])

    # 3: SVM
    log.info("Running SVM baseline …")
    svm_r = run_svm(train_X, train_labels, test_X, test_labels)
    print_results_table("SVM (RBF, summary features)", svm_r)

    # Reference: best MLP result from prior runs
    print(f"  {'MLP physio-only (reference)':<28}  V=0.418  A=0.385  D=0.355  Mean=0.386")
    print(f"  {'MLP + GP-aug (best, reference)':<28}  V=0.383  A=0.580  D=0.397  Mean=0.454")

    # 4: Per-modality ablation
    if not args.skip_mlp_ablation:
        log.info("Running per-modality MLP ablation (5 × full training run) …")
        print("\n" + "-" * 64)
        print("Per-modality ablation (MLP, drop one modality's summary features):")
        print("-" * 64)

        # Full model F1 for reference (quick re-run)
        full_m = run_split(train_df, val_df, test_df, args, device,
                           fold_tag="_full", aug_df=aug_train)
        print_results_table("Full (all modalities)", full_m)

        ablation_r = run_modality_ablation(
            train_df, val_df, test_df, args, device, aug_train
        )
        for mod, r in ablation_r.items():
            delta = r["mean"] - full_m.get("mean_f1", 0)
            tag = f"Drop {mod:5s}  (d={delta:+.3f})"
            print_results_table(tag, r)

    print("=" * 64)


if __name__ == "__main__":
    main()
