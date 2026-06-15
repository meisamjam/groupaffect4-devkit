"""Create quick visual overviews from derived feature tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("plot_feature_overview")


def _read_tsv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOG.warning("Missing input (skipping): %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot feature and semantic biomarker overviews.")
    p.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing feature TSV outputs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <features-dir>/figures).",
    )
    p.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional session_id for window trend plots (default: first available).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    LOG.info("Saved %s", path)


def plot_semantic_by_task(sem_task: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
        "biomarker_recovery_capacity",
        "biomarker_fatigue_depletion",
    ]
    cols = [c for c in cols if c in sem_task.columns]
    if not cols:
        return
    task_order = ["T0", "T1", "T2", "T3", "T4"]
    grp = sem_task.groupby("task")[cols].mean(numeric_only=True).reindex(task_order)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(grp.index))
    for col in cols:
        ax.plot(x, grp[col].to_numpy(), marker="o", label=col.replace("biomarker_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index.tolist())
    ax.set_ylabel("Mean z-like score")
    ax.set_title("Semantic Biomarkers by Task")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / "semantic_biomarkers_by_task.png")


def plot_semantic_distributions(sem_task: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
    ]
    cols = [c for c in cols if c in sem_task.columns]
    if not cols:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        vals = pd.to_numeric(sem_task[col], errors="coerce").dropna()
        if vals.empty:
            continue
        axes[i].hist(vals.to_numpy(), bins=30, alpha=0.9, color="#3a6ea5")
        axes[i].set_title(col.replace("biomarker_", "").replace("_", " "))
        axes[i].grid(alpha=0.2)
    _save(fig, out_dir / "semantic_biomarker_histograms.png")


def plot_window_trends(
    sem_window: pd.DataFrame,
    out_dir: Path,
    session_id: str | None,
) -> None:
    cols = ["biomarker_arousal_stress", "biomarker_cognitive_load", "biomarker_attention"]
    cols = [c for c in cols if c in sem_window.columns]
    if not cols or sem_window.empty:
        return
    sessions = sem_window["session_id"].dropna().unique().tolist()
    if not sessions:
        return
    session_id = session_id or sessions[0]
    sdf = sem_window[sem_window["session_id"] == session_id].copy()
    if sdf.empty:
        return
    tasks = [t for t in ["T1", "T2", "T3", "T4"] if t in sdf["task"].unique()]
    if not tasks:
        tasks = sorted(sdf["task"].dropna().unique().tolist())

    for task in tasks:
        tdf = sdf[sdf["task"] == task].copy()
        if tdf.empty:
            continue
        metric = None
        for c in cols:
            if pd.to_numeric(tdf[c], errors="coerce").notna().any():
                metric = c
                break
        if metric is None:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for pid, pdf in tdf.groupby("participant_id"):
            x = pd.to_numeric(pdf["window_index"], errors="coerce")
            y = pd.to_numeric(pdf[metric], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.any():
                ax.plot(x[valid], y[valid], marker="o", markersize=2.5, linewidth=1.4, label=str(pid))
        ax.set_title(f"{metric.replace('biomarker_', '').replace('_', ' ').title()} Trend - {session_id} - {task}")
        ax.set_xlabel("Window index")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        suffix = metric.replace("biomarker_", "")
        _save(fig, out_dir / f"trend_{session_id}_{task}_{suffix}.png")


def plot_dyad_sync(dyn_task: pd.DataFrame, out_dir: Path) -> None:
    if dyn_task.empty or "metric" not in dyn_task.columns or "corr" not in dyn_task.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = sorted(dyn_task["metric"].dropna().unique().tolist())
    data = []
    labels = []
    for metric in metrics:
        vals = pd.to_numeric(dyn_task[dyn_task["metric"] == metric]["corr"], errors="coerce").dropna()
        if vals.empty:
            continue
        data.append(vals.to_numpy())
        labels.append(metric)
    if not data:
        return
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Dyad Synchrony Correlation by Metric")
    ax.set_ylabel("Pearson r")
    ax.grid(alpha=0.2)
    _save(fig, out_dir / "dyad_synchrony_boxplot.png")


def plot_participant_biomarker_profiles(participant_df: pd.DataFrame, out_dir: Path, session_id: str | None) -> None:
    if participant_df.empty:
        return
    cols = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
        "biomarker_recovery_capacity",
        "biomarker_fatigue_depletion",
    ]
    cols = [c for c in cols if c in participant_df.columns]
    if not cols:
        return
    sessions = participant_df["session_id"].dropna().unique().tolist()
    if not sessions:
        return
    session_id = session_id or sessions[0]
    sdf = participant_df[participant_df["session_id"] == session_id].copy()
    if sdf.empty:
        return
    task_order = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in sdf["task"].unique()]
    if not task_order:
        task_order = sorted(sdf["task"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        for pid, pdf in sdf.groupby("participant_id"):
            g = pdf.groupby("task")[col].mean(numeric_only=True).reindex(task_order)
            ax.plot(np.arange(len(task_order)), g.to_numpy(), marker="o", linewidth=1.3, label=str(pid))
        ax.set_title(col.replace("biomarker_", "").replace("_", " "))
        ax.set_xticks(np.arange(len(task_order)))
        ax.set_xticklabels(task_order)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(f"Participant Biomarker Profiles - {session_id}", y=1.02)
    _save(fig, out_dir / f"participant_biomarker_profiles_{session_id}.png")


def plot_group_pool_biomarkers(group_pool_df: pd.DataFrame, out_dir: Path) -> None:
    if group_pool_df.empty:
        return
    base = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
        "biomarker_recovery_capacity",
        "biomarker_fatigue_depletion",
    ]
    mean_cols = [f"{c}_group_mean" for c in base if f"{c}_group_mean" in group_pool_df.columns]
    if not mean_cols:
        return
    task_order = ["T0", "T1", "T2", "T3", "T4"]
    grp = group_pool_df.groupby("task")[mean_cols].mean(numeric_only=True).reindex(task_order)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(grp.index))
    for col in mean_cols:
        label = col.replace("biomarker_", "").replace("_group_mean", "")
        ax.plot(x, grp[col].to_numpy(), marker="o", linewidth=1.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index.tolist())
    ax.set_title("Group-Pooled Biomarkers by Task")
    ax.set_ylabel("Mean score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir / "group_pool_biomarkers_by_task.png")


def plot_participant_vs_group_z(compare_df: pd.DataFrame, out_dir: Path) -> None:
    if compare_df.empty:
        return
    z_cols = [
        "biomarker_cognitive_load_z_vs_group",
        "biomarker_arousal_stress_z_vs_group",
        "biomarker_attention_z_vs_group",
        "biomarker_decision_pressure_z_vs_group",
        "biomarker_recovery_capacity_z_vs_group",
        "biomarker_fatigue_depletion_z_vs_group",
    ]
    z_cols = [c for c in z_cols if c in compare_df.columns]
    if not z_cols:
        return

    data = []
    labels = []
    for col in z_cols:
        vals = pd.to_numeric(compare_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        data.append(vals.to_numpy())
        labels.append(col.replace("_z_vs_group", "").replace("biomarker_", ""))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Participant vs Group Biomarker Deviations (z-score)")
    ax.set_ylabel("z vs group")
    ax.grid(alpha=0.2)
    _save(fig, out_dir / "participant_vs_group_biomarker_z_boxplot.png")


def _draw_heatmap(ax: plt.Axes, data: np.ndarray, xlabels: list[str], ylabels: list[str], title: str) -> None:
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    finite = data[np.isfinite(data)]
    if finite.size:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_biomarker_task_heatmap(sem_task: pd.DataFrame, out_dir: Path) -> None:
    if sem_task.empty:
        return
    biomarker_cols = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
        "biomarker_recovery_capacity",
        "biomarker_fatigue_depletion",
    ]
    cols = [c for c in biomarker_cols if c in sem_task.columns]
    if not cols:
        return
    task_order = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in sem_task["task"].unique()]
    table = sem_task.groupby("task")[cols].mean(numeric_only=True).reindex(task_order)
    mat = table.to_numpy(dtype=float, copy=False).T
    ylabels = [c.replace("biomarker_", "") for c in cols]
    xlabels = table.index.tolist()

    fig, ax = plt.subplots(figsize=(9.5, 5))
    _draw_heatmap(ax, mat, xlabels, ylabels, "Task x Biomarker Mean Heatmap")
    _save(fig, out_dir / "task_biomarker_mean_heatmap.png")


def plot_participant_task_deviation_heatmap(compare_df: pd.DataFrame, out_dir: Path, session_id: str | None) -> None:
    if compare_df.empty:
        return
    z_col = "biomarker_arousal_stress_z_vs_group"
    if z_col not in compare_df.columns:
        z_fallback = [c for c in compare_df.columns if c.endswith("_z_vs_group") and "biomarker_" in c]
        if not z_fallback:
            return
        z_col = z_fallback[0]

    sessions = compare_df["session_id"].dropna().unique().tolist()
    if not sessions:
        return
    session_id = session_id or sessions[0]
    sdf = compare_df[compare_df["session_id"] == session_id].copy()
    if sdf.empty:
        return
    task_order = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in sdf["task"].unique()]
    participants = sorted(sdf["participant_id"].dropna().unique().tolist())
    if not participants or not task_order:
        return
    pivot = (
        sdf.pivot_table(index="participant_id", columns="task", values=z_col, aggfunc="mean")
        .reindex(index=participants, columns=task_order)
    )
    mat = pivot.to_numpy(dtype=float, copy=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    title = f"Participant vs Group Deviation Heatmap ({z_col.replace('biomarker_', '').replace('_z_vs_group', '')}) - {session_id}"
    _draw_heatmap(ax, mat, task_order, participants, title)
    _save(fig, out_dir / f"participant_task_deviation_heatmap_{session_id}.png")


def plot_answer_biomarker_correlation(participant_df: pd.DataFrame, out_dir: Path) -> None:
    if participant_df.empty:
        return
    biomarker_cols = [
        "biomarker_cognitive_load",
        "biomarker_arousal_stress",
        "biomarker_attention",
        "biomarker_decision_pressure",
        "biomarker_recovery_capacity",
        "biomarker_fatigue_depletion",
    ]
    biomarker_cols = [c for c in biomarker_cols if c in participant_df.columns]
    answer_cols = [c for c in participant_df.columns if c.startswith("ans_") and not c.startswith("ans_text_")]
    answer_cols = [c for c in answer_cols if pd.api.types.is_numeric_dtype(participant_df[c])]
    if not biomarker_cols or not answer_cols:
        return

    coverage = []
    for c in answer_cols:
        vals = pd.to_numeric(participant_df[c], errors="coerce")
        coverage.append((int(vals.notna().sum()), c))
    coverage.sort(reverse=True)
    answer_cols = [c for n, c in coverage if n >= 12][:12]
    if not answer_cols:
        return

    corr = pd.DataFrame(index=biomarker_cols, columns=answer_cols, dtype=float)
    for b in biomarker_cols:
        x = pd.to_numeric(participant_df[b], errors="coerce")
        for a in answer_cols:
            y = pd.to_numeric(participant_df[a], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 8:
                corr.loc[b, a] = np.nan
            else:
                corr.loc[b, a] = float(np.corrcoef(x[valid], y[valid])[0, 1])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _draw_heatmap(
        ax,
        corr.to_numpy(dtype=float, copy=False),
        [c.replace("ans_", "") for c in answer_cols],
        [c.replace("biomarker_", "") for c in biomarker_cols],
        "Biomarker vs Answer Correlation Heatmap",
    )
    _save(fig, out_dir / "biomarker_answer_correlation_heatmap.png")


def plot_annotation_context_overview(participant_df: pd.DataFrame, out_dir: Path) -> None:
    if participant_df.empty:
        return
    needed = ["task", "annotation_entries_count", "task_events_count", "biomarker_arousal_stress"]
    if any(c not in participant_df.columns for c in needed):
        return
    task_order = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in participant_df["task"].unique()]
    if not task_order:
        return
    grp = (
        participant_df.groupby("task", as_index=False)[
            ["annotation_entries_count", "task_events_count", "biomarker_arousal_stress"]
        ]
        .mean(numeric_only=True)
        .set_index("task")
        .reindex(task_order)
    )
    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(task_order))
    w = 0.38
    ax1.bar(x - w / 2, grp["annotation_entries_count"].to_numpy(), width=w, label="annotation entries")
    ax1.bar(x + w / 2, grp["task_events_count"].to_numpy(), width=w, label="task events")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_order)
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.2, axis="y")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        grp["biomarker_arousal_stress"].to_numpy(),
        color="#222222",
        marker="o",
        linewidth=1.8,
        label="arousal_stress",
    )
    ax2.set_ylabel("Arousal stress score")
    ax1.set_title("Annotation/Event Context vs Arousal Stress by Task")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    _save(fig, out_dir / "annotation_event_vs_arousal_by_task.png")


def plot_vad_performance(vad_perf: pd.DataFrame, out_dir: Path) -> None:
    if vad_perf.empty:
        return
    needed = {"participant_id", "dimension", "label_agreement"}
    if not needed.issubset(vad_perf.columns):
        return
    df = vad_perf.copy()
    df["label_agreement"] = pd.to_numeric(df["label_agreement"], errors="coerce")
    df = df.dropna(subset=["label_agreement"])
    if df.empty:
        return
    dims = [d for d in ["valence", "arousal", "dominance"] if d in df["dimension"].unique()]
    participants = sorted(df["participant_id"].dropna().unique().tolist())
    if not dims or not participants:
        return
    pivot = (
        df.pivot_table(index="participant_id", columns="dimension", values="label_agreement", aggfunc="mean")
        .reindex(index=participants, columns=dims)
    )
    mat = pivot.to_numpy(dtype=float, copy=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _draw_heatmap(ax, mat, dims, participants, "Biomarker vs Self-VAD Label Agreement")
    _save(fig, out_dir / "performance_vad_agreement_heatmap.png")


def plot_vad_confusion(vad_cmp: pd.DataFrame, out_dir: Path) -> None:
    if vad_cmp.empty:
        return
    labels = ["Low", "Moderate", "High"]
    dims = ["valence", "arousal", "dominance"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    any_plot = False
    for i, dim in enumerate(dims):
        self_col = f"vad_{dim}_self_label"
        pred_col = f"vad_{dim}_pred_label"
        ax = axes[i]
        if self_col not in vad_cmp.columns or pred_col not in vad_cmp.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dim)
            continue
        sub = vad_cmp[(vad_cmp[self_col] != "") & (vad_cmp[pred_col] != "")]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dim)
            continue
        m = np.zeros((3, 3), dtype=float)
        for r, sl in enumerate(labels):
            for c, pl in enumerate(labels):
                m[r, c] = float(((sub[self_col] == sl) & (sub[pred_col] == pl)).sum())
        any_plot = True
        _draw_heatmap(ax, m, labels, labels, f"{dim.title()} Confusion")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Self-report")
    if any_plot:
        _save(fig, out_dir / "performance_vad_confusion_matrices.png")
    else:
        plt.close(fig)


def plot_annotation_alignment_performance(ann_perf: pd.DataFrame, out_dir: Path) -> None:
    if ann_perf.empty:
        return
    needed = {"participant_id", "label_agreement"}
    if not needed.issubset(ann_perf.columns):
        return
    df = ann_perf.copy()
    df["label_agreement"] = pd.to_numeric(df["label_agreement"], errors="coerce")
    df = df.dropna(subset=["label_agreement"])
    if df.empty:
        return
    g = df.groupby("participant_id", as_index=False)["label_agreement"].mean()
    g = g.sort_values("participant_id")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(g["participant_id"].astype(str), g["label_agreement"].to_numpy(), color="#3a6ea5")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement")
    ax.set_title("Arousal Biomarker vs Annotation-Derived Activity Label")
    ax.grid(alpha=0.2, axis="y")
    _save(fig, out_dir / "performance_annotation_alignment_bar.png")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    features_dir = args.features_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else features_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_task = _read_tsv_optional(features_dir / "semantic_biomarkers_participant_task.tsv")
    sem_window = _read_tsv_optional(features_dir / "semantic_biomarkers_window_30s.tsv")
    dyn_task = _read_tsv_optional(features_dir / "features_group_dynamics_task.tsv")
    participant = _read_tsv_optional(features_dir / "participant_features_answers_annotations.tsv")
    group_pool = _read_tsv_optional(features_dir / "group_pool_task_summary.tsv")
    compare = _read_tsv_optional(features_dir / "participant_vs_group_comparison.tsv")
    vad_cmp = _read_tsv_optional(features_dir / "biomarker_vad_label_comparison.tsv")
    vad_perf = _read_tsv_optional(features_dir / "biomarker_vad_performance_by_participant.tsv")
    ann_perf = _read_tsv_optional(features_dir / "biomarker_annotation_performance_by_participant.tsv")

    plot_semantic_by_task(sem_task, out_dir)
    plot_semantic_distributions(sem_task, out_dir)
    plot_window_trends(sem_window, out_dir, args.session)
    plot_dyad_sync(dyn_task, out_dir)
    plot_participant_biomarker_profiles(participant, out_dir, args.session)
    plot_group_pool_biomarkers(group_pool, out_dir)
    plot_participant_vs_group_z(compare, out_dir)
    plot_biomarker_task_heatmap(sem_task, out_dir)
    plot_participant_task_deviation_heatmap(compare, out_dir, args.session)
    plot_answer_biomarker_correlation(participant, out_dir)
    plot_annotation_context_overview(participant, out_dir)
    plot_vad_performance(vad_perf, out_dir)
    plot_vad_confusion(vad_cmp, out_dir)
    plot_annotation_alignment_performance(ann_perf, out_dir)
    LOG.info("Visualization complete. Output dir: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
