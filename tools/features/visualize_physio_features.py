"""Create quick paper-facing visualizations from extracted EmotiBit features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("visualize_physio_features")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
DELTA_FEATURES = [
    ("hr_mean_bpm_delta_t0", "Heart rate change vs T0", "bpm"),
    ("hrv_rmssd_ms_delta_t0", "HRV RMSSD change vs T0", "ms"),
    ("eda_tonic_mean_delta_t0", "EDA tonic change vs T0", "a.u."),
    ("eda_phasic_rate_hz_delta_t0", "EDA phasic-rate change vs T0", "peaks/s"),
    ("temp_mean_delta_t0", "Skin temperature change vs T0", "deg C"),
]

EMOTIBIT_DELTA_FEATURES = [
    ("ppg_green_mean_delta_t0", "PPG green mean", "raw"),
    ("ppg_red_mean_delta_t0", "PPG red mean", "raw"),
    ("ppg_ir_mean_delta_t0", "PPG infrared mean", "raw"),
    ("eda_scr_amplitude_mean_delta_t0", "EDA SCR amplitude", "a.u."),
    ("thermopile_mean_delta_t0", "Thermopile-like channel", "deg C"),
    ("temp_aux_mean_delta_t0", "Aux temperature-like channel", "deg C"),
    ("accel_dynamic_mean_delta_t0", "Accel dynamic movement", "g"),
    ("gyro_motion_mean_delta_t0", "Gyro movement", "deg/s"),
    ("mag_motion_mean_delta_t0", "Magnetometer magnitude", "a.u."),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize paper-ready physiology feature and QC tables."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing physio_participant_task.tsv and physio_qc_summary.tsv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "physio",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def _read_tables(features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_path = features_dir / "physio_participant_task.tsv"
    qc_path = features_dir / "physio_qc_summary.tsv"
    if not task_path.exists():
        raise FileNotFoundError(f"Missing participant-task table: {task_path}")
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing QC table: {qc_path}")
    task = pd.read_csv(task_path, sep="\t")
    qc = pd.read_csv(qc_path, sep="\t")
    return task, qc


def _task_delta_summary(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = task[task["task_id"].isin(["T1", "T2", "T3", "T4"])].copy()
    ncols = 2
    nrows = int(np.ceil(len(DELTA_FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), constrained_layout=True)
    fig.suptitle("Task-level physiology changes relative to T0", fontsize=14, fontweight="bold")
    task_labels = ["T1", "T2", "T3", "T4"]
    x = np.arange(len(task_labels))

    colors = {
        "hr_mean_bpm_delta_t0": "#2f6f9f",
        "hrv_rmssd_ms_delta_t0": "#5b5f97",
        "eda_tonic_mean_delta_t0": "#a33f2f",
        "eda_phasic_rate_hz_delta_t0": "#6b7f2a",
        "temp_mean_delta_t0": "#7c4d9e",
    }

    flat_axes = axes.ravel()
    for ax, (column, title, unit) in zip(flat_axes, DELTA_FEATURES, strict=False):
        if column not in plot_df.columns:
            ax.set_axis_off()
            continue
        grouped = plot_df.groupby("task_id")[column]
        means = grouped.mean().reindex(task_labels)
        sems = grouped.sem().reindex(task_labels)
        counts = grouped.count().reindex(task_labels).fillna(0).astype(int)
        ax.axhline(0.0, color="#6f6f6f", linewidth=0.9, linestyle="--")
        ax.errorbar(
            x,
            means.to_numpy(dtype=float),
            yerr=sems.to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=colors[column],
        )
        ax.set_xticks(x, [f"{task}\nn={counts.loc[task]}" for task in task_labels])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in flat_axes[len(DELTA_FEATURES) :]:
        ax.set_axis_off()

    out_path = out_dir / "physio_task_delta_summary.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _ppg_hrv_quality(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = task.copy()
    plot_df["hrv_usable"] = plot_df["hrv_rmssd_ms"].notna()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    fig.suptitle("PPG-derived HRV method evaluation", fontsize=14, fontweight="bold")

    colors = plot_df["hrv_usable"].map({True: "#2f7d5c", False: "#b95f45"}).to_numpy()
    axes[0].scatter(
        plot_df["ppg_hr_agreement_bpm"],
        plot_df["hrv_quality_score"],
        c=colors,
        alpha=0.75,
        s=32,
        edgecolors="none",
    )
    axes[0].axvline(20.0, color="#6f6f6f", linewidth=0.9, linestyle="--")
    axes[0].axhline(0.65, color="#6f6f6f", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("PPG vs device-HR difference (bpm)")
    axes[0].set_ylabel("HRV quality score")
    axes[0].set_title("Quality gates")

    usable = plot_df.loc[plot_df["hrv_usable"], "hrv_rmssd_ms"].dropna()
    axes[1].hist(usable, bins=16, color="#5b5f97", alpha=0.9)
    axes[1].set_xlabel("RMSSD (ms)")
    axes[1].set_ylabel("participant-task rows")
    axes[1].set_title(f"Usable HRV distribution (n={len(usable)})")

    by_session = plot_df.groupby("session_id")["hrv_usable"].mean().sort_index() * 100.0
    axes[2].barh(np.arange(len(by_session)), by_session.to_numpy(), color="#2f6f9f")
    axes[2].set_yticks(np.arange(len(by_session)), by_session.index)
    axes[2].tick_params(axis="y", labelsize=7)
    axes[2].set_xlabel("usable HRV rows (%)")
    axes[2].set_title("Usable HRV by session")
    axes[2].set_xlim(0, 100)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    out_path = out_dir / "physio_ppg_hrv_quality.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _emotibit_signal_delta_grid(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = task[task["task_id"].isin(["T1", "T2", "T3", "T4"])].copy()
    task_labels = ["T1", "T2", "T3", "T4"]
    ncols = 3
    nrows = int(np.ceil(len(EMOTIBIT_DELTA_FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), constrained_layout=True)
    fig.suptitle("EmotiBit signal changes relative to T0", fontsize=14, fontweight="bold")
    x = np.arange(len(task_labels))
    palette = [
        "#2f6f9f",
        "#8e5a2b",
        "#6b7f2a",
        "#a33f2f",
        "#7c4d9e",
        "#5a8f7b",
        "#4e6fae",
        "#9b5f73",
        "#6b6b6b",
    ]

    flat_axes = axes.ravel()
    for ax, (column, title, unit), color in zip(
        flat_axes,
        EMOTIBIT_DELTA_FEATURES,
        palette,
        strict=False,
    ):
        if column not in plot_df.columns:
            ax.set_axis_off()
            continue
        grouped = plot_df.groupby("task_id")[column]
        means = grouped.mean().reindex(task_labels)
        sems = grouped.sem().reindex(task_labels)
        counts = grouped.count().reindex(task_labels).fillna(0).astype(int)
        ax.axhline(0.0, color="#6f6f6f", linewidth=0.8, linestyle="--")
        ax.errorbar(
            x,
            means.to_numpy(dtype=float),
            yerr=sems.to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
        )
        ax.set_xticks(x, [f"{task}\nn={counts.loc[task]}" for task in task_labels])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in flat_axes[len(EMOTIBIT_DELTA_FEATURES) :]:
        ax.set_axis_off()

    out_path = out_dir / "physio_emotibit_signal_delta_grid.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _session_task_coverage(qc: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    sessions = sorted(qc["session_id"].dropna().unique().tolist())
    tasks = [task for task in TASK_ORDER if task in set(qc["task_id"].dropna())]
    pivot = (
        qc.assign(available=qc["physio_available"].astype(str).str.lower().eq("true"))
        .groupby(["session_id", "task_id"])["available"]
        .sum()
        .unstack("task_id")
        .reindex(index=sessions, columns=tasks)
        .fillna(0)
    )

    fig_height = max(5.0, 0.38 * len(sessions) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=4, cmap="viridis")
    ax.set_title("EmotiBit participant coverage by session and task", fontsize=13, fontweight="bold")
    ax.set_xlabel("Task")
    ax.set_ylabel("Session")
    ax.set_xticks(np.arange(len(tasks)), tasks)
    ax.set_yticks(np.arange(len(sessions)), sessions)
    ax.tick_params(axis="y", labelsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iloc[i, j])
            text_color = "white" if value <= 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("participants with physio")
    out_path = out_dir / "physio_session_task_coverage.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _qc_overview(qc: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    qc = qc.copy()
    qc["available"] = qc["physio_available"].astype(str).str.lower().eq("true")
    session_counts = qc.groupby("session_id")["available"].agg(["sum", "count"]).sort_index()
    session_counts["missing"] = session_counts["count"] - session_counts["sum"]

    flags = (
        qc["qc_flag"]
        .fillna("missing_physio")
        .str.split(";")
        .explode()
        .value_counts()
        .drop(labels=["ok"], errors="ignore")
        .head(8)
        .sort_values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    fig.suptitle("Physio QC overview", fontsize=14, fontweight="bold")

    y = np.arange(len(session_counts))
    axes[0].barh(y, session_counts["sum"], color="#2f7d5c", label="available")
    axes[0].barh(
        y,
        session_counts["missing"],
        left=session_counts["sum"],
        color="#b95f45",
        label="missing",
    )
    axes[0].set_yticks(y, session_counts.index)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].set_xlabel("participant-task rows")
    axes[0].set_title("Availability in expected QC grid")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].barh(np.arange(len(flags)), flags.to_numpy(), color="#4e6fae")
    axes[1].set_yticks(np.arange(len(flags)), flags.index)
    axes[1].set_xlabel("row count")
    axes[1].set_title("Most common QC flags")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    out_path = out_dir / "physio_qc_overview.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    features_dir = args.features_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task, qc = _read_tables(features_dir)
    paths = [
        _task_delta_summary(task, out_dir, args.dpi),
        _ppg_hrv_quality(task, out_dir, args.dpi),
        _emotibit_signal_delta_grid(task, out_dir, args.dpi),
        _session_task_coverage(qc, out_dir, args.dpi),
        _qc_overview(qc, out_dir, args.dpi),
    ]
    for path in paths:
        LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
