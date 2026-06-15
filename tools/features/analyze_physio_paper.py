"""Build paper-ready descriptive analyses from extracted EmotiBit features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_physio_paper")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
ACTIVE_TASKS = ["T1", "T2", "T3", "T4"]

PAPER_FEATURES = [
    ("hr_mean_bpm_delta_t0", "HR change", "bpm"),
    ("hrv_rmssd_ms_delta_t0", "HRV RMSSD change", "ms"),
    ("eda_tonic_mean_delta_t0", "EDA tonic change", "a.u."),
    ("eda_phasic_rate_hz_delta_t0", "EDA SCR rate change", "peaks/s"),
    ("eda_scr_amplitude_mean_delta_t0", "EDA SCR amplitude change", "a.u."),
    ("temp_mean_delta_t0", "Skin temp change", "deg C"),
    ("temp_aux_mean_delta_t0", "Aux temp change", "deg C"),
    ("accel_dynamic_mean_delta_t0", "Accel dynamic change", "g"),
    ("gyro_motion_mean_delta_t0", "Gyro movement change", "deg/s"),
]

WINDOW_FEATURES = [
    ("hr_mean_bpm", "HR", "bpm"),
    ("eda_tonic_mean", "EDA tonic", "a.u."),
    ("eda_phasic_rate_hz", "EDA SCR rate", "peaks/s"),
    ("temp_mean", "Skin temp", "deg C"),
    ("accel_dynamic_mean", "Accel dynamic", "g"),
    ("gyro_motion_mean", "Gyro movement", "deg/s"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-ready physio analysis tables and figures."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing physio feature and QC TSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "physio",
        help="Directory where analysis TSV outputs are written.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures") / "physio",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure output DPI.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def _read_tables(features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_path = features_dir / "physio_participant_task.tsv"
    window_path = features_dir / "physio_window_30s.tsv"
    qc_path = features_dir / "physio_qc_summary.tsv"
    missing = [str(path) for path in [task_path, window_path, qc_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required physio table(s): " + ", ".join(missing))
    return (
        pd.read_csv(task_path, sep="\t"),
        pd.read_csv(window_path, sep="\t"),
        pd.read_csv(qc_path, sep="\t"),
    )


def _series_stats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(x))
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "cohen_dz_vs_zero": np.nan,
        }
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    sem = float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan
    ci = 1.96 * sem if np.isfinite(sem) else np.nan
    return {
        "n": n,
        "mean": mean,
        "median": float(x.median()),
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - ci if np.isfinite(ci) else np.nan,
        "ci95_high": mean + ci if np.isfinite(ci) else np.nan,
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "cohen_dz_vs_zero": float(mean / sd) if np.isfinite(sd) and sd > 0 else np.nan,
    }


def build_feature_usability(task: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature availability and QC usability for paper tables."""
    actual_rows = len(task)
    expected_rows = len(qc)
    rows: list[dict[str, object]] = []
    for column, label, unit in PAPER_FEATURES:
        if column not in task.columns:
            continue
        n_nonnull = int(pd.to_numeric(task[column], errors="coerce").notna().sum())
        rows.append(
            {
                "feature": column,
                "label": label,
                "unit": unit,
                "nonnull_rows": n_nonnull,
                "actual_feature_rows": actual_rows,
                "expected_qc_rows": expected_rows,
                "nonnull_pct_of_actual": n_nonnull / actual_rows * 100.0 if actual_rows else np.nan,
                "nonnull_pct_of_expected": n_nonnull / expected_rows * 100.0 if expected_rows else np.nan,
            }
        )

    if not qc.empty:
        rows.extend(
            [
                {
                    "feature": "ppg_usable",
                    "label": "PPG usable",
                    "unit": "rows",
                    "nonnull_rows": int(qc["ppg_usable"].astype(str).str.lower().eq("true").sum()),
                    "actual_feature_rows": actual_rows,
                    "expected_qc_rows": expected_rows,
                    "nonnull_pct_of_actual": np.nan,
                    "nonnull_pct_of_expected": float(
                        qc["ppg_usable"].astype(str).str.lower().eq("true").mean() * 100.0
                    ),
                },
                {
                    "feature": "eda_usable",
                    "label": "EDA usable",
                    "unit": "rows",
                    "nonnull_rows": int(qc["eda_usable"].astype(str).str.lower().eq("true").sum()),
                    "actual_feature_rows": actual_rows,
                    "expected_qc_rows": expected_rows,
                    "nonnull_pct_of_actual": np.nan,
                    "nonnull_pct_of_expected": float(
                        qc["eda_usable"].astype(str).str.lower().eq("true").mean() * 100.0
                    ),
                },
                {
                    "feature": "temp_usable",
                    "label": "Temperature usable",
                    "unit": "rows",
                    "nonnull_rows": int(qc["temp_usable"].astype(str).str.lower().eq("true").sum()),
                    "actual_feature_rows": actual_rows,
                    "expected_qc_rows": expected_rows,
                    "nonnull_pct_of_actual": np.nan,
                    "nonnull_pct_of_expected": float(
                        qc["temp_usable"].astype(str).str.lower().eq("true").mean() * 100.0
                    ),
                },
                {
                    "feature": "imu_usable",
                    "label": "IMU usable",
                    "unit": "rows",
                    "nonnull_rows": int(qc["imu_usable"].astype(str).str.lower().eq("true").sum()),
                    "actual_feature_rows": actual_rows,
                    "expected_qc_rows": expected_rows,
                    "nonnull_pct_of_actual": np.nan,
                    "nonnull_pct_of_expected": float(
                        qc["imu_usable"].astype(str).str.lower().eq("true").mean() * 100.0
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_task_delta_stats(task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active = task[task["task_id"].isin(ACTIVE_TASKS)].copy()
    for column, label, unit in PAPER_FEATURES:
        if column not in active.columns:
            continue
        for task_id in ACTIVE_TASKS:
            subset = active.loc[active["task_id"] == task_id, column]
            rows.append(
                {
                    "feature": column,
                    "label": label,
                    "unit": unit,
                    "task_id": task_id,
                    **_series_stats(subset),
                }
            )
    return pd.DataFrame(rows)


def build_session_task_summary(task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary_features = [column for column, _label, _unit in PAPER_FEATURES if column in task.columns]
    base_cols = ["session_id", "task_id"]
    for (session_id, task_id), group in task.groupby(base_cols):
        row: dict[str, object] = {
            "session_id": session_id,
            "task_id": task_id,
            "n_participants": int(group["participant_id"].nunique()),
            "n_rows": int(len(group)),
        }
        for feature in summary_features:
            row[f"{feature}_mean"] = pd.to_numeric(group[feature], errors="coerce").mean()
            row[f"{feature}_median"] = pd.to_numeric(group[feature], errors="coerce").median()
            row[f"{feature}_n"] = int(pd.to_numeric(group[feature], errors="coerce").notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["session_id", "task_id"])
    return out


def build_qc_flag_counts(qc: pd.DataFrame) -> pd.DataFrame:
    flags = qc["qc_flag"].fillna("missing").str.split(";").explode()
    out = flags.value_counts().rename_axis("qc_flag").reset_index(name="row_count")
    out["row_pct"] = out["row_count"] / len(qc) * 100.0 if len(qc) else np.nan
    return out


def build_feature_correlations(task: pd.DataFrame) -> pd.DataFrame:
    active = task[task["task_id"].isin(ACTIVE_TASKS)].copy()
    columns = [column for column, _label, _unit in PAPER_FEATURES if column in active.columns]
    rows: list[dict[str, object]] = []
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            pair = active[[feature_a, feature_b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 5:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(pair[feature_a].corr(pair[feature_b], method="pearson"))
                spearman = float(pair[feature_a].corr(pair[feature_b], method="spearman"))
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "n": int(len(pair)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_temporal_profile(window: pd.DataFrame) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    data = window[window["task_id"].isin(ACTIVE_TASKS)].copy()
    if data.empty:
        return pd.DataFrame()
    data["window_start_lsl"] = pd.to_numeric(data["window_start_lsl"], errors="coerce")
    data["window_end_lsl"] = pd.to_numeric(data["window_end_lsl"], errors="coerce")
    gcols = ["session_id", "task_id", "participant_id"]
    data["task_progress"] = np.nan
    for _key, group in data.groupby(gcols):
        start = float(group["window_start_lsl"].min())
        end = float(group["window_end_lsl"].max())
        denom = max(end - start, 1e-9)
        data.loc[group.index, "task_progress"] = (group["window_start_lsl"] - start) / denom
    bins = [-0.001, 1 / 3, 2 / 3, 1.001]
    labels = ["early", "middle", "late"]
    data["task_segment"] = pd.cut(data["task_progress"], bins=bins, labels=labels)

    rows: list[dict[str, object]] = []
    for column, label, unit in WINDOW_FEATURES:
        if column not in data.columns:
            continue
        for (task_id, segment), group in data.groupby(["task_id", "task_segment"], observed=False):
            stats = _series_stats(group[column])
            rows.append(
                {
                    "feature": column,
                    "label": label,
                    "unit": unit,
                    "task_id": task_id,
                    "task_segment": str(segment),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _plot_task_effect_heatmap(stats: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    pivot = stats.pivot(index="label", columns="task_id", values="cohen_dz_vs_zero").reindex(
        [label for _col, label, _unit in PAPER_FEATURES]
    )
    pivot = pivot[[task for task in ACTIVE_TASKS if task in pivot.columns]]
    fig_height = max(5.0, 0.45 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height), constrained_layout=True)
    values = np.abs(pivot.to_numpy(dtype=float)) if not pivot.empty else np.array([])
    finite_values = values[np.isfinite(values)]
    max_abs = float(finite_values.max()) if finite_values.size else 1.0
    max_abs = max(max_abs, 0.5)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_title("Task-level physiology effect sizes vs T0", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cohen dz")
    out_path = figures_dir / "physio_paper_task_effect_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_feature_usability(usability: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = usability[usability["feature"].isin([f[0] for f in PAPER_FEATURES])].copy()
    plot_df = plot_df.sort_values("nonnull_pct_of_actual")
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    ax.barh(plot_df["label"], plot_df["nonnull_pct_of_actual"], color="#2f6f9f")
    ax.set_xlim(0, 100)
    ax.set_xlabel("non-missing rows among available physio rows (%)")
    ax.set_title("Physio feature usability", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, value in enumerate(plot_df["nonnull_pct_of_actual"]):
        ax.text(value + 1, i, f"{value:.0f}%", va="center", fontsize=8)
    out_path = figures_dir / "physio_paper_feature_usability.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_correlation_matrix(corr: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    labels = [label for _col, label, _unit in PAPER_FEATURES]
    feature_to_label = {column: label for column, label, _unit in PAPER_FEATURES}
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels, dtype=float)
    for row in corr.itertuples(index=False):
        a = feature_to_label.get(row.feature_a, row.feature_a)
        b = feature_to_label.get(row.feature_b, row.feature_b)
        matrix.loc[a, b] = row.spearman_rho
        matrix.loc[b, a] = row.spearman_rho
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Physio feature correlation structure", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Spearman rho")
    out_path = figures_dir / "physio_paper_feature_correlation.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_temporal_profile(profile: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    selected = ["hr_mean_bpm", "eda_tonic_mean", "accel_dynamic_mean", "gyro_motion_mean"]
    plot_df = profile[profile["feature"].isin(selected)].copy()
    if plot_df.empty:
        out_path = figures_dir / "physio_paper_temporal_profiles.png"
        return out_path
    segments = ["early", "middle", "late"]
    tasks = ACTIVE_TASKS
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Within-task physio temporal profiles", fontsize=14, fontweight="bold")
    colors = {"T1": "#2f6f9f", "T2": "#a33f2f", "T3": "#6b7f2a", "T4": "#7c4d9e"}
    for ax, feature in zip(axes.ravel(), selected, strict=True):
        fdf = plot_df[plot_df["feature"] == feature]
        label = fdf["label"].iloc[0] if not fdf.empty else feature
        unit = fdf["unit"].iloc[0] if not fdf.empty else ""
        for task_id in tasks:
            tdf = fdf[fdf["task_id"] == task_id].set_index("task_segment").reindex(segments)
            ax.errorbar(
                np.arange(len(segments)),
                tdf["mean"].to_numpy(dtype=float),
                yerr=tdf["sem"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[task_id],
                label=task_id,
            )
        ax.set_xticks(np.arange(len(segments)), segments)
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[0].legend(frameon=False, ncols=4, fontsize=8)
    out_path = figures_dir / "physio_paper_temporal_profiles.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_outputs(
    task: pd.DataFrame,
    window: pd.DataFrame,
    qc: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    usability = build_feature_usability(task, qc)
    task_stats = build_task_delta_stats(task)
    session_task = build_session_task_summary(task)
    qc_counts = build_qc_flag_counts(qc)
    correlations = build_feature_correlations(task)
    temporal = build_temporal_profile(window)

    outputs = [
        results_dir / "physio_paper_feature_usability.tsv",
        results_dir / "physio_task_delta_stats.tsv",
        results_dir / "physio_session_task_summary.tsv",
        results_dir / "physio_qc_flag_counts.tsv",
        results_dir / "physio_feature_correlations.tsv",
        results_dir / "physio_temporal_profile.tsv",
    ]
    for df, path in [
        (usability, outputs[0]),
        (task_stats, outputs[1]),
        (session_task, outputs[2]),
        (qc_counts, outputs[3]),
        (correlations, outputs[4]),
        (temporal, outputs[5]),
    ]:
        df.to_csv(path, sep="\t", index=False)

    outputs.extend(
        [
            _plot_feature_usability(usability, figures_dir, dpi),
            _plot_task_effect_heatmap(task_stats, figures_dir, dpi),
            _plot_correlation_matrix(correlations, figures_dir, dpi),
            _plot_temporal_profile(temporal, figures_dir, dpi),
        ]
    )
    return outputs


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    task, window, qc = _read_tables(args.features_dir.resolve())
    outputs = write_outputs(
        task,
        window,
        qc,
        args.results_dir.resolve(),
        args.figures_dir.resolve(),
        args.dpi,
    )
    for path in outputs:
        LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
