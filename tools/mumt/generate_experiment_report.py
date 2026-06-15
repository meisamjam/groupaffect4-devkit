"""generate_experiment_report.py

Generate a detailed Markdown report comparing all literature experiment results
across temporal encoder architectures.

Usage:
    python tools/mumt/generate_experiment_report.py \
        --conv1d-results results/comprehensive_conv1d.csv \
        --gru-results results/comprehensive_gru.csv \
        --output results/EXPERIMENT_REPORT.md
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EXPERIMENT_DESCRIPTIONS = {
    "baseline": "SimCLR contrastive pre-training + standard augmentation (noise/jitter/shift)",
    "E1": "Time Warping augmentation (replaces circular shift with local time distortion)",
    "E2": "Masked Signal Modeling pre-training (reconstructive SSL, replaces SimCLR)",
    "E3": "Per-modality independent SSL (avoids cross-modal interference during pre-training)",
    "E4": "Window Warping for EDA + standard augmentation for other modalities",
    "E5": "Cross-modal reconstruction auxiliary loss during fine-tuning",
    "E6": "Native-rate per-modality T (removes upsampling artifacts from slow signals)",
    "E1+E2": "Masked Signal Modeling + Time Warping (best SSL + best augmentation)",
    "E3+E1": "Per-modality SSL + Time Warping augmentation",
    "E6+E1": "Native-rate sequences + Time Warping augmentation",
    "E2+aug": "Masked SSL + pseudo-labeled pool fine-tuning (Track B)",
    "E1+E2+aug": "Masked SSL + Time Warping + pseudo-labeled pool (Track B)",
    "E6+E1+aug": "Native-rate + Time Warping + pseudo-labeled pool (Track B)",
}

EXPERIMENT_CATEGORIES = {
    "Pre-training Method": ["baseline", "E2", "E3"],
    "Augmentation Strategy": ["baseline", "E1", "E4"],
    "Auxiliary Loss": ["baseline", "E5"],
    "Temporal Resolution": ["baseline", "E6"],
    "Best Combinations": ["baseline", "E1+E2", "E6+E1", "E3+E1"],
    "With Pseudo-Label Augmentation": ["baseline", "E2+aug", "E1+E2+aug", "E6+E1+aug"],
}


def load_results(path: Path) -> pd.DataFrame | None:
    """Load CSV results, return None if file doesn't exist."""
    if not path.exists():
        log.warning("Results file not found: %s", path)
        return None
    df = pd.read_csv(path)
    return df


def format_f1(val: float) -> str:
    """Format F1 score with 3 decimal places."""
    return f"{val:.3f}"


def format_delta(val: float, baseline: float) -> str:
    """Format delta from baseline with +/- sign."""
    delta = val - baseline
    if abs(delta) < 0.001:
        return "—"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.3f}"


def make_results_table(df: pd.DataFrame, encoder: str) -> str:
    """Generate a markdown table for one encoder's results."""
    lines = []
    lines.append(f"| Experiment | V F1 | A F1 | D F1 | **Mean F1** | Δ baseline | V std | A std | D std |")
    lines.append("|:-----------|:----:|:----:|:----:|:-----------:|:----------:|:-----:|:-----:|:-----:|")

    baseline_mean = None
    for _, row in df.iterrows():
        if row["experiment"] == "baseline":
            baseline_mean = row["mean_f1"]
            break

    if baseline_mean is None:
        baseline_mean = 0.0

    for _, row in df.iterrows():
        exp = row["experiment"]
        delta = format_delta(row["mean_f1"], baseline_mean)
        bold_start = "**" if row["mean_f1"] == df["mean_f1"].max() else ""
        bold_end = "**" if row["mean_f1"] == df["mean_f1"].max() else ""
        lines.append(
            f"| {exp} | {format_f1(row['v_f1'])} | {format_f1(row['a_f1'])} | "
            f"{format_f1(row['d_f1'])} | {bold_start}{format_f1(row['mean_f1'])}{bold_end} | "
            f"{delta} | {format_f1(row.get('v_std', 0))} | "
            f"{format_f1(row.get('a_std', 0))} | {format_f1(row.get('d_std', 0))} |"
        )

    return "\n".join(lines)


def make_cross_encoder_table(conv1d_df: pd.DataFrame | None, gru_df: pd.DataFrame | None) -> str:
    """Generate a cross-encoder comparison table."""
    lines = []
    lines.append("| Experiment | Conv1D Mean | GRU Mean | Best | Winner |")
    lines.append("|:-----------|:-----------:|:--------:|:----:|:------:|")

    # Get common experiments
    experiments = []
    if conv1d_df is not None:
        experiments = conv1d_df["experiment"].tolist()
    elif gru_df is not None:
        experiments = gru_df["experiment"].tolist()

    for exp in experiments:
        conv1d_val = None
        gru_val = None

        if conv1d_df is not None:
            row = conv1d_df[conv1d_df["experiment"] == exp]
            if not row.empty:
                conv1d_val = row.iloc[0]["mean_f1"]

        if gru_df is not None:
            row = gru_df[gru_df["experiment"] == exp]
            if not row.empty:
                gru_val = row.iloc[0]["mean_f1"]

        c_str = format_f1(conv1d_val) if conv1d_val is not None else "—"
        g_str = format_f1(gru_val) if gru_val is not None else "—"

        if conv1d_val is not None and gru_val is not None:
            best = max(conv1d_val, gru_val)
            winner = "Conv1D" if conv1d_val > gru_val else "GRU"
            if abs(conv1d_val - gru_val) < 0.005:
                winner = "Tie"
        elif conv1d_val is not None:
            best = conv1d_val
            winner = "Conv1D"
        elif gru_val is not None:
            best = gru_val
            winner = "GRU"
        else:
            best = 0.0
            winner = "—"

        lines.append(f"| {exp} | {c_str} | {g_str} | **{format_f1(best)}** | {winner} |")

    return "\n".join(lines)


def make_category_analysis(conv1d_df: pd.DataFrame | None, gru_df: pd.DataFrame | None) -> str:
    """Generate per-category analysis sections."""
    sections = []

    for category, experiments in EXPERIMENT_CATEGORIES.items():
        sections.append(f"\n#### {category}\n")

        for encoder_name, df in [("Conv1D", conv1d_df), ("GRU", gru_df)]:
            if df is None:
                continue
            subset = df[df["experiment"].isin(experiments)]
            if subset.empty:
                continue

            baseline_row = subset[subset["experiment"] == "baseline"]
            baseline_mean = baseline_row.iloc[0]["mean_f1"] if not baseline_row.empty else 0

            best_row = subset.loc[subset["mean_f1"].idxmax()]
            if best_row["experiment"] != "baseline":
                delta = best_row["mean_f1"] - baseline_mean
                sections.append(
                    f"- **{encoder_name}**: Best = `{best_row['experiment']}` "
                    f"(Mean F1={format_f1(best_row['mean_f1'])}, "
                    f"+{delta:.3f} over baseline)"
                )
            else:
                sections.append(
                    f"- **{encoder_name}**: No improvement over baseline "
                    f"(Mean F1={format_f1(baseline_mean)})"
                )

    return "\n".join(sections)


def generate_report(
    conv1d_df: pd.DataFrame | None,
    gru_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Generate the full markdown report."""
    lines: list[str] = []

    # Header
    lines.append("# Comprehensive Experiment Report: Preprocessing & Temporal Modeling")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset**: GroupAffect-4 (15s windows, T=200)")
    lines.append(f"**Split**: Task-based (T0+T1 train=103, T2 val=78, T3 test=65)")
    lines.append(f"**Seeds**: 3 (averaged)")
    lines.append(f"**Pre-training**: 100 epochs on 8,221 unlabeled pool windows")
    lines.append(f"**Fine-tuning**: 200 epochs, patience=40")
    lines.append("")

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("1. [Executive Summary](#executive-summary)")
    lines.append("2. [Experiment Descriptions](#experiment-descriptions)")
    lines.append("3. [Conv1D Results](#conv1d-results)")
    lines.append("4. [GRU Results](#gru-results)")
    lines.append("5. [Cross-Encoder Comparison](#cross-encoder-comparison)")
    lines.append("6. [Category Analysis](#category-analysis)")
    lines.append("7. [Key Findings & Recommendations](#key-findings--recommendations)")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    best_overall = None
    best_encoder = None
    best_exp = None

    for encoder_name, df in [("Conv1D", conv1d_df), ("GRU", gru_df)]:
        if df is None:
            continue
        best_row = df.loc[df["mean_f1"].idxmax()]
        if best_overall is None or best_row["mean_f1"] > best_overall:
            best_overall = best_row["mean_f1"]
            best_encoder = encoder_name
            best_exp = best_row["experiment"]

    if best_overall is not None:
        lines.append(f"**Best overall result**: `{best_exp}` with **{best_encoder}** encoder "
                     f"→ Mean F1 = **{format_f1(best_overall)}**")
        lines.append("")

    # Per-dimension bests
    for dim_col, dim_name in [("v_f1", "Valence"), ("a_f1", "Arousal"), ("d_f1", "Dominance")]:
        best_val = 0.0
        best_info = ""
        for encoder_name, df in [("Conv1D", conv1d_df), ("GRU", gru_df)]:
            if df is None:
                continue
            row = df.loc[df[dim_col].idxmax()]
            if row[dim_col] > best_val:
                best_val = row[dim_col]
                best_info = f"`{row['experiment']}` ({encoder_name}) = {format_f1(best_val)}"
        if best_info:
            lines.append(f"- **{dim_name}**: {best_info}")

    lines.append("")

    # Experiment Descriptions
    lines.append("## Experiment Descriptions")
    lines.append("")
    lines.append("| ID | Method | Description |")
    lines.append("|:---|:-------|:------------|")
    for exp_id, desc in EXPERIMENT_DESCRIPTIONS.items():
        lines.append(f"| `{exp_id}` | {exp_id} | {desc} |")
    lines.append("")

    # Literature sources
    lines.append("### Literature Sources")
    lines.append("")
    lines.append("| Method | Inspired By |")
    lines.append("|:-------|:-----------|")
    lines.append("| Time Warping (E1) | PLOS ONE 2025 — TSDA survey (time-series augmentation) |")
    lines.append("| Masked Signal Modeling (E2) | IEEE Sensors 2026 — self-supervised physiological signals |")
    lines.append("| Per-modality SSL (E3) | SIGDIAL 2025 — modality-specific pre-training |")
    lines.append("| Window Warping (E4) | Le Guennec et al. — best for univariate slow signals |")
    lines.append("| Cross-modal recon (E5) | ICMI 2025 — auxiliary cross-modal objectives |")
    lines.append("| Native-rate T (E6) | Signal processing — avoid upsampling artifacts |")
    lines.append("| Pseudo-labels (+aug) | Semi-supervised learning — BFI-similarity confidence |")
    lines.append("")

    # Conv1D Results
    lines.append("## Conv1D Results")
    lines.append("")
    if conv1d_df is not None:
        lines.append(make_results_table(conv1d_df, "conv1d"))
    else:
        lines.append("*No Conv1D results available.*")
    lines.append("")

    # GRU Results
    lines.append("## GRU Results")
    lines.append("")
    if gru_df is not None:
        lines.append(make_results_table(gru_df, "gru"))
    else:
        lines.append("*No GRU results available.*")
    lines.append("")

    # Cross-Encoder Comparison
    lines.append("## Cross-Encoder Comparison")
    lines.append("")
    lines.append(make_cross_encoder_table(conv1d_df, gru_df))
    lines.append("")

    # Category Analysis
    lines.append("## Category Analysis")
    lines.append(make_category_analysis(conv1d_df, gru_df))
    lines.append("")

    # Key Findings
    lines.append("## Key Findings & Recommendations")
    lines.append("")

    findings = []

    # Finding 1: Best pre-training
    if conv1d_df is not None:
        pretrain_exps = conv1d_df[conv1d_df["experiment"].isin(["baseline", "E2", "E3"])]
        if not pretrain_exps.empty:
            best_pt = pretrain_exps.loc[pretrain_exps["mean_f1"].idxmax()]
            findings.append(
                f"1. **Pre-training**: `{best_pt['experiment']}` is the best pre-training strategy "
                f"(Mean F1={format_f1(best_pt['mean_f1'])}). "
                + ("Masked Signal Modeling (reconstructive) outperforms SimCLR (contrastive)."
                   if best_pt["experiment"] == "E2" else "")
            )

    # Finding 2: Best augmentation
    if conv1d_df is not None:
        aug_exps = conv1d_df[conv1d_df["experiment"].isin(["baseline", "E1", "E4"])]
        if not aug_exps.empty:
            best_aug = aug_exps.loc[aug_exps["mean_f1"].idxmax()]
            findings.append(
                f"2. **Augmentation**: `{best_aug['experiment']}` is the best augmentation "
                f"(Mean F1={format_f1(best_aug['mean_f1'])}). "
                + ("Time Warping provides more realistic temporal variations than noise/jitter."
                   if best_aug["experiment"] == "E1" else "")
            )

    # Finding 3: Combination
    if conv1d_df is not None:
        combo_exps = conv1d_df[conv1d_df["experiment"].isin(["baseline", "E1+E2", "E3+E1", "E6+E1"])]
        if not combo_exps.empty:
            best_combo = combo_exps.loc[combo_exps["mean_f1"].idxmax()]
            findings.append(
                f"3. **Best combination**: `{best_combo['experiment']}` "
                f"(Mean F1={format_f1(best_combo['mean_f1'])})"
            )

    # Finding 4: Encoder
    if conv1d_df is not None and gru_df is not None:
        conv1d_best = conv1d_df["mean_f1"].max()
        gru_best = gru_df["mean_f1"].max()
        winner = "Conv1D" if conv1d_best > gru_best else "GRU"
        findings.append(
            f"4. **Encoder**: {winner} achieves the highest peak performance "
            f"(Conv1D={format_f1(conv1d_best)}, GRU={format_f1(gru_best)})"
        )

    # Finding 5: Pseudo-labels
    if conv1d_df is not None:
        aug_pool = conv1d_df[conv1d_df["experiment"].str.contains("aug", na=False)]
        no_aug = conv1d_df[~conv1d_df["experiment"].str.contains("aug", na=False)]
        if not aug_pool.empty and not no_aug.empty:
            aug_best = aug_pool["mean_f1"].max()
            no_aug_best = no_aug["mean_f1"].max()
            helps = "helps" if aug_best > no_aug_best else "does not help"
            findings.append(
                f"5. **Pseudo-label augmentation**: Pool Track B {helps} "
                f"(best with aug={format_f1(aug_best)}, best without={format_f1(no_aug_best)})"
            )

    # Finding 6: Dominance challenge
    if conv1d_df is not None:
        d_best = conv1d_df["d_f1"].max()
        v_best = conv1d_df["v_f1"].max()
        findings.append(
            f"6. **Per-dimension difficulty**: Dominance remains hardest "
            f"(best D={format_f1(d_best)} vs best V={format_f1(v_best)})"
        )

    for f in findings:
        lines.append(f)
        lines.append("")

    # Recommendations
    lines.append("### Recommendations for Final Model")
    lines.append("")
    lines.append("Based on the comprehensive results above:")
    lines.append("")
    if best_exp and best_encoder:
        lines.append(f"1. Use **`{best_exp}`** configuration with **{best_encoder}** encoder")
    lines.append("2. Consider ensemble of top-2 configurations for robustness")
    lines.append("3. The pool pseudo-label results indicate whether semi-supervised "
                 "approaches are viable for this dataset size")
    lines.append("4. For the paper, report the best configuration and its per-dimension breakdown")
    lines.append("")

    # Reproducibility
    lines.append("---")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# Conv1D full run")
    lines.append("python tools/mumt/literature_experiments.py --encoder conv1d --seeds 3 "
                 "--out results/comprehensive_conv1d.csv")
    lines.append("")
    lines.append("# GRU full run")
    lines.append("python tools/mumt/literature_experiments.py --encoder gru --seeds 3 "
                 "--out results/comprehensive_gru.csv")
    lines.append("```")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s (%d lines)", output_path, len(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment comparison report")
    parser.add_argument("--conv1d-results", type=Path, default=None,
                        help="Path to conv1d results CSV")
    parser.add_argument("--gru-results", type=Path, default=None,
                        help="Path to GRU results CSV")
    parser.add_argument("--output", type=Path, default=Path("results/EXPERIMENT_REPORT.md"))
    args = parser.parse_args()

    conv1d_df = load_results(args.conv1d_results) if args.conv1d_results else None
    gru_df = load_results(args.gru_results) if args.gru_results else None

    if conv1d_df is None and gru_df is None:
        log.error("No results files found. Run experiments first.")
        raise SystemExit(1)

    generate_report(conv1d_df, gru_df, args.output)


if __name__ == "__main__":
    main()
