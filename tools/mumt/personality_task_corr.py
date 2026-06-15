"""personality_task_corr.py

Spearman correlations between BFI-44 traits and VAD self-reports,
computed separately per task (T1-T4) and across all tasks combined.

Shows WHICH traits moderate WHICH VAD dimensions in WHICH task contexts —
directly grounding the perdim model design and explaining the task-sensitivity
of BFI conditioning.

Output:
  - Console table: BFI trait × VAD dim × task (r, p-value)
  - results/personality_task_correlations.tsv

Usage
-----
  python tools/mumt/personality_task_corr.py \\
      --dataset data/mumt/dataset_15s.pkl
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_affectai import BIG_FIVE_COLS

BFI_LABELS = {
    "bfi44_e": "E (Extraversion)",
    "bfi44_a": "A (Agreeableness)",
    "bfi44_c": "C (Conscientiousness)",
    "bfi44_n": "N (Neuroticism)",
    "bfi44_o": "O (Openness)",
}
VAD_DIMS  = ["valence", "arousal", "dominance"]
TASKS     = ["T1", "T2", "T3", "T4"]


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman r and p per (trait, vad_dim, task) and all-tasks."""
    rows = []
    for trait in BIG_FIVE_COLS:
        for vad in VAD_DIMS:
            # All tasks (participant-level to avoid pseudo-replication)
            subj = df.groupby("subject_id")[[trait, vad]].mean().dropna()
            if len(subj) >= 5:
                r, p = stats.spearmanr(subj[trait], subj[vad])
            else:
                r, p = float("nan"), float("nan")
            rows.append(dict(trait=trait, vad=vad, task="ALL",
                             r=r, p=p, n=len(subj)))

            # Per task (window-level, but n is small so also show participant-level)
            for task in TASKS:
                t_df = df[df["task"] == task]
                # window-level (all windows)
                sub_w = t_df[[trait, vad]].dropna()
                if len(sub_w) >= 5:
                    r_w, p_w = stats.spearmanr(sub_w[trait], sub_w[vad])
                else:
                    r_w, p_w = float("nan"), float("nan")
                rows.append(dict(trait=trait, vad=vad, task=task,
                                 r=r_w, p=p_w, n=len(sub_w)))
    return pd.DataFrame(rows)


def print_heatmap(corr_df: pd.DataFrame, task: str) -> None:
    """Print a trait × VAD correlation matrix for one task."""
    print(f"\n  Task {task}  (r = Spearman, * p<.05, ** p<.01)")
    print(f"  {'Trait':<26} {'Valence':>10} {'Arousal':>10} {'Dominance':>10}")
    print(f"  {'-'*56}")
    for trait in BIG_FIVE_COLS:
        label = BFI_LABELS[trait]
        row = []
        for vad in VAD_DIMS:
            sub = corr_df[(corr_df.trait == trait) & (corr_df.vad == vad)
                          & (corr_df.task == task)]
            if len(sub) == 0 or np.isnan(sub.iloc[0]["r"]):
                row.append("     ---")
            else:
                r = sub.iloc[0]["r"]
                p = sub.iloc[0]["p"]
                stars = "**" if p < .01 else ("*" if p < .05 else "  ")
                row.append(f"{r:+.3f}{stars}")
        print(f"  {label:<26} {row[0]:>10} {row[1]:>10} {row[2]:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--out",     default="results/personality_task_correlations.tsv")
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)

    # Check BFI columns
    missing = [c for c in BIG_FIVE_COLS if c not in df.columns]
    if missing:
        print(f"Missing BFI columns: {missing}")
        sys.exit(1)

    print(f"Dataset: {len(df)} windows | {df['subject_id'].nunique()} subjects | "
          f"{df['session_id'].nunique()} sessions")

    corr_df = corr_table(df)

    # Print heatmaps
    for task in ["ALL"] + TASKS:
        print_heatmap(corr_df, task)

    # Summary: top significant findings
    sig = corr_df[(corr_df["p"] < 0.05) & (corr_df["task"] != "ALL")].copy()
    sig["abs_r"] = sig["r"].abs()
    sig = sig.sort_values("abs_r", ascending=False)

    print(f"\n  === Significant trait-VAD correlations (p<.05, window-level) ===")
    print(f"  {'Trait':<26} {'VAD':<12} {'Task':<6} {'r':>8} {'p':>8} {'n':>5}")
    print(f"  {'-'*68}")
    for _, row in sig.head(20).iterrows():
        trait_label = BFI_LABELS.get(row["trait"], row["trait"])
        print(f"  {trait_label:<26} {row['vad']:<12} {row['task']:<6} "
              f"{row['r']:>+8.3f} {row['p']:>8.3f} {int(row['n']):>5}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
