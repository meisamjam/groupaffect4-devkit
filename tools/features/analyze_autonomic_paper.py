"""Build paper-ready autonomic analyses from EmotiBit and Tobii pupil features."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_autonomic_paper")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
ACTIVE_TASKS = ["T1", "T2", "T3", "T4"]
KEYS = ["session_id", "task_id", "participant_id"]


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    label: str
    unit: str
    modality: str
    arousal_direction: float = 1.0


AUTONOMIC_FEATURES = [
    FeatureSpec("hr_mean_bpm_delta_t0", "HR increase", "bpm", "cardiac", 1.0),
    FeatureSpec("hrv_rmssd_ms_delta_t0", "HRV RMSSD decrease", "ms", "cardiac", -1.0),
    FeatureSpec("eda_tonic_mean_delta_t0", "EDA tonic increase", "a.u.", "eda", 1.0),
    FeatureSpec("eda_phasic_rate_hz_delta_t0", "SCR rate increase", "peaks/s", "eda", 1.0),
    FeatureSpec("eda_scr_amplitude_mean_delta_t0", "SCR amplitude increase", "a.u.", "eda", 1.0),
    FeatureSpec("pupil_mean_delta_t0", "Pupil diameter change", "mm", "pupil", 1.0),
    FeatureSpec("pupil_std_delta_t0", "Pupil variability change", "mm", "pupil", 1.0),
    FeatureSpec("temp_mean_delta_t0", "Skin temp change", "deg C", "temperature", 1.0),
    FeatureSpec("accel_dynamic_mean_delta_t0", "Accel movement", "g", "motion", 1.0),
    FeatureSpec("gyro_motion_mean_delta_t0", "Gyro movement", "deg/s", "motion", 1.0),
]

PHYSIO_AROUSAL_FEATURES = [
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_tonic_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
]

MOTION_FEATURES = ["accel_dynamic_mean_delta_t0", "gyro_motion_mean_delta_t0"]

TEMPORAL_FEATURES = [
    FeatureSpec("hr_mean_bpm", "HR change", "bpm", "cardiac", 1.0),
    FeatureSpec("eda_tonic_mean", "EDA tonic change", "a.u.", "eda", 1.0),
    FeatureSpec("pupil_mean", "Pupil diameter change", "mm", "pupil", 1.0),
    FeatureSpec("accel_dynamic_mean", "Accel movement change", "g", "motion", 1.0),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create combined EmotiBit + Tobii pupil analyses for the dataset paper."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing physio and pupil feature TSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "autonomic",
        help="Directory where analysis TSV outputs are written.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures") / "autonomic",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--min-pupil-valid-frac",
        type=float,
        default=0.70,
        help="Minimum gaze-valid fraction for pupil rows counted as usable.",
    )
    parser.add_argument(
        "--max-pupil-missing-frac",
        type=float,
        default=0.30,
        help="Maximum missing-pupil fraction for pupil rows counted as usable.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure output DPI.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def _read_tsv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required table: {path}")
        LOG.warning("Optional table not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


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
            "p25": np.nan,
            "p75": np.nan,
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
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "cohen_dz_vs_zero": float(mean / sd) if np.isfinite(sd) and sd > 0 else np.nan,
    }


def _corr_pair(data: pd.DataFrame, feature_a: str, feature_b: str) -> dict[str, float]:
    pair = data[[feature_a, feature_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 5:
        return {"n": int(len(pair)), "pearson_r": np.nan, "spearman_rho": np.nan}
    return {
        "n": int(len(pair)),
        "pearson_r": float(pair[feature_a].corr(pair[feature_b], method="pearson")),
        "spearman_rho": float(pair[feature_a].corr(pair[feature_b], method="spearman")),
    }


def _residualize(y: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    frame = pd.concat([y.rename("y"), covariates], axis=1).apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < covariates.shape[1] + 3:
        return pd.Series(dtype=float)
    design = np.column_stack([np.ones(len(frame)), frame[covariates.columns].to_numpy(dtype=float)])
    values = frame["y"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ beta
    return pd.Series(residuals, index=frame.index, dtype=float)


def load_tables(features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_dir = features_dir.resolve()
    physio_task = _read_tsv(features_dir / "physio_participant_task.tsv")
    physio_window = _read_tsv(features_dir / "physio_window_30s.tsv")
    physio_qc = _read_tsv(features_dir / "physio_qc_summary.tsv")
    pupil_task = _read_tsv(features_dir / "features_pupil_participant_task.tsv", required=False)
    pupil_window = _read_tsv(features_dir / "features_pupil_window_30s.tsv", required=False)
    return physio_task, physio_window, physio_qc, pupil_task, pupil_window


def prepare_pupil_task(
    pupil_task: pd.DataFrame,
    min_valid_frac: float,
    max_missing_frac: float,
) -> pd.DataFrame:
    """Normalize pupil task rows and add T0-centered pupil changes."""
    if pupil_task.empty:
        return pupil_task
    out = pupil_task.copy()
    if "task" in out.columns and "task_id" not in out.columns:
        out = out.rename(columns={"task": "task_id"})
    for column in ["pupil_mean", "pupil_std", "pupil_slope_per_s", "pupil_missing_frac", "gaze_valid_frac"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    baseline_cols = [c for c in ["pupil_mean", "pupil_std", "pupil_slope_per_s"] if c in out.columns]
    if baseline_cols:
        baseline = out[out["task_id"] == "T0"][KEYS[:1] + KEYS[2:] + baseline_cols].copy()
        baseline = baseline.rename(columns={c: f"{c}_t0" for c in baseline_cols})
        out = out.merge(baseline, on=["session_id", "participant_id"], how="left")
        for column in baseline_cols:
            out[f"{column}_delta_t0"] = out[column] - out[f"{column}_t0"]
    if {"pupil_mean", "pupil_missing_frac", "gaze_valid_frac"}.issubset(out.columns):
        out["pupil_usable"] = (
            out["pupil_mean"].notna()
            & (out["gaze_valid_frac"] >= min_valid_frac)
            & (out["pupil_missing_frac"] <= max_missing_frac)
        )
    else:
        out["pupil_usable"] = False
    return out


def build_joined_task(physio_task: pd.DataFrame, pupil_task: pd.DataFrame) -> pd.DataFrame:
    physio = physio_task.copy()
    if "task" in physio.columns and "task_id" not in physio.columns:
        physio = physio.rename(columns={"task": "task_id"})
    pupil = pupil_task.copy()
    if pupil.empty:
        return physio

    pupil_cols = [
        c
        for c in [
            *KEYS,
            "pupil_mean",
            "pupil_std",
            "pupil_slope_per_s",
            "pupil_mean_delta_t0",
            "pupil_std_delta_t0",
            "pupil_slope_per_s_delta_t0",
            "pupil_missing_frac",
            "gaze_valid_frac",
            "pupil_usable",
        ]
        if c in pupil.columns
    ]
    return physio.merge(pupil[pupil_cols], on=KEYS, how="outer")


def build_feature_usability(
    joined: pd.DataFrame,
    physio_qc: pd.DataFrame,
    pupil_task: pd.DataFrame,
) -> pd.DataFrame:
    expected_rows = len(physio_qc) if not physio_qc.empty else len(joined)
    rows: list[dict[str, object]] = []
    for spec in AUTONOMIC_FEATURES:
        if spec.column not in joined.columns:
            continue
        nonnull = int(pd.to_numeric(joined[spec.column], errors="coerce").notna().sum())
        rows.append(
            {
                "feature": spec.column,
                "label": spec.label,
                "modality": spec.modality,
                "unit": spec.unit,
                "nonnull_rows": nonnull,
                "observed_rows": len(joined),
                "expected_rows": expected_rows,
                "nonnull_pct_observed": nonnull / len(joined) * 100.0 if len(joined) else np.nan,
                "nonnull_pct_expected": nonnull / expected_rows * 100.0 if expected_rows else np.nan,
            }
        )
    if "pupil_usable" in pupil_task.columns:
        usable = int(_bool_series(pupil_task["pupil_usable"]).sum())
        rows.append(
            {
                "feature": "pupil_usable",
                "label": "Pupil usable",
                "modality": "pupil",
                "unit": "rows",
                "nonnull_rows": usable,
                "observed_rows": len(pupil_task),
                "expected_rows": expected_rows,
                "nonnull_pct_observed": usable / len(pupil_task) * 100.0 if len(pupil_task) else np.nan,
                "nonnull_pct_expected": usable / expected_rows * 100.0 if expected_rows else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_task_delta_stats(joined: pd.DataFrame) -> pd.DataFrame:
    active = joined[joined["task_id"].isin(ACTIVE_TASKS)].copy()
    rows: list[dict[str, object]] = []
    for spec in AUTONOMIC_FEATURES:
        if spec.column not in active.columns:
            continue
        for task_id in ACTIVE_TASKS:
            stats = _series_stats(active.loc[active["task_id"] == task_id, spec.column])
            dz = stats["cohen_dz_vs_zero"]
            rows.append(
                {
                    "feature": spec.column,
                    "label": spec.label,
                    "modality": spec.modality,
                    "unit": spec.unit,
                    "task_id": task_id,
                    "arousal_direction": spec.arousal_direction,
                    "arousal_dz": dz * spec.arousal_direction if np.isfinite(dz) else np.nan,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def build_composite_scores(joined: pd.DataFrame) -> pd.DataFrame:
    active = joined[joined["task_id"].isin(ACTIVE_TASKS)].copy()
    if active.empty:
        return pd.DataFrame()
    out = active[KEYS].copy()
    spec_by_col = {spec.column: spec for spec in AUTONOMIC_FEATURES}

    arousal_z_cols: list[str] = []
    for feature in PHYSIO_AROUSAL_FEATURES:
        if feature not in active.columns:
            continue
        x = pd.to_numeric(active[feature], errors="coerce")
        sd = x.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            continue
        z_col = f"{feature}_arousal_z"
        out[z_col] = ((x - x.mean(skipna=True)) / sd) * spec_by_col[feature].arousal_direction
        arousal_z_cols.append(z_col)

    motion_z_cols: list[str] = []
    for feature in MOTION_FEATURES:
        if feature not in active.columns:
            continue
        x = pd.to_numeric(active[feature], errors="coerce")
        sd = x.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            continue
        z_col = f"{feature}_z"
        out[z_col] = (x - x.mean(skipna=True)) / sd
        motion_z_cols.append(z_col)

    if arousal_z_cols:
        out["physio_arousal_feature_count"] = out[arousal_z_cols].notna().sum(axis=1)
        out["physio_arousal_index"] = out[arousal_z_cols].mean(axis=1, skipna=True)
        out.loc[out["physio_arousal_feature_count"] < 2, "physio_arousal_index"] = np.nan

    if "pupil_mean_delta_t0" in active.columns:
        x = pd.to_numeric(active["pupil_mean_delta_t0"], errors="coerce")
        sd = x.std(skipna=True, ddof=0)
        if np.isfinite(sd) and sd > 0:
            out["pupil_diameter_index"] = (x - x.mean(skipna=True)) / sd
    if motion_z_cols:
        out["motion_feature_count"] = out[motion_z_cols].notna().sum(axis=1)
        out["motion_context_index"] = out[motion_z_cols].mean(axis=1, skipna=True)
    return out


def build_composite_task_stats(composite: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, label in [
        ("physio_arousal_index", "Physio arousal index"),
        ("pupil_diameter_index", "Pupil diameter index"),
        ("motion_context_index", "Motion context index"),
    ]:
        if column not in composite.columns:
            continue
        for task_id in ACTIVE_TASKS:
            rows.append({"metric": column, "label": label, "task_id": task_id, **_series_stats(
                composite.loc[composite["task_id"] == task_id, column]
            )})
    return pd.DataFrame(rows)


def build_modality_coverage(
    physio_qc: pd.DataFrame,
    physio_task: pd.DataFrame,
    pupil_task: pd.DataFrame,
) -> pd.DataFrame:
    if not physio_qc.empty:
        expected = physio_qc[KEYS].drop_duplicates().copy()
    else:
        expected = pd.concat(
            [
                physio_task[[c for c in KEYS if c in physio_task.columns]],
                pupil_task[[c for c in KEYS if c in pupil_task.columns]],
            ],
            ignore_index=True,
        ).drop_duplicates()

    rows = expected.copy()
    if not physio_qc.empty:
        qc = physio_qc.copy()
        qc["physio_available"] = _bool_series(qc["physio_available"])
        usable_cols = [c for c in ["ppg_usable", "eda_usable", "temp_usable", "imu_usable"] if c in qc]
        if usable_cols:
            qc["physio_usable"] = False
            for column in usable_cols:
                qc["physio_usable"] = qc["physio_usable"] | _bool_series(qc[column])
        else:
            qc["physio_usable"] = qc["physio_available"]
        rows = rows.merge(qc[KEYS + ["physio_available", "physio_usable"]], on=KEYS, how="left")
    else:
        present = physio_task[KEYS].drop_duplicates()
        present["physio_available"] = True
        present["physio_usable"] = True
        rows = rows.merge(present, on=KEYS, how="left")

    if not pupil_task.empty:
        pupil = pupil_task[KEYS + ["pupil_usable"]].drop_duplicates().copy()
        pupil["pupil_available"] = True
        pupil["pupil_usable"] = _bool_series(pupil["pupil_usable"])
        rows = rows.merge(pupil, on=KEYS, how="left")
    else:
        rows["pupil_available"] = False
        rows["pupil_usable"] = False

    for column in ["physio_available", "physio_usable", "pupil_available", "pupil_usable"]:
        if column not in rows.columns:
            rows[column] = False
        rows[column] = _bool_series(rows[column])
    rows["both_available"] = rows["physio_available"] & rows["pupil_available"]
    rows["both_usable"] = rows["physio_usable"] & rows["pupil_usable"]

    agg_rows: list[dict[str, object]] = []
    for (session_id, task_id), group in rows.groupby(["session_id", "task_id"], sort=True):
        expected_count = int(len(group))
        row = {
            "session_id": session_id,
            "task_id": task_id,
            "expected_participant_rows": expected_count,
        }
        for column in [
            "physio_available",
            "physio_usable",
            "pupil_available",
            "pupil_usable",
            "both_available",
            "both_usable",
        ]:
            count = int(group[column].sum())
            row[f"{column}_count"] = count
            row[f"{column}_pct"] = count / expected_count * 100.0 if expected_count else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def build_feature_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    active = joined[joined["task_id"].isin(ACTIVE_TASKS)].copy()
    features = [spec for spec in AUTONOMIC_FEATURES if spec.column in active.columns]
    rows: list[dict[str, object]] = []
    for i, spec_a in enumerate(features):
        for spec_b in features[i + 1 :]:
            corr = _corr_pair(active, spec_a.column, spec_b.column)
            rows.append(
                {
                    "feature_a": spec_a.column,
                    "feature_b": spec_b.column,
                    "label_a": spec_a.label,
                    "label_b": spec_b.label,
                    "modality_a": spec_a.modality,
                    "modality_b": spec_b.modality,
                    **corr,
                }
            )
    return pd.DataFrame(rows)


def build_pupil_physio_links(joined: pd.DataFrame) -> pd.DataFrame:
    active = joined[joined["task_id"].isin(ACTIVE_TASKS)].copy()
    if "pupil_mean_delta_t0" not in active.columns:
        return pd.DataFrame()
    target_features = [
        "hr_mean_bpm_delta_t0",
        "hrv_rmssd_ms_delta_t0",
        "eda_tonic_mean_delta_t0",
        "eda_phasic_rate_hz_delta_t0",
        "eda_scr_amplitude_mean_delta_t0",
        "accel_dynamic_mean_delta_t0",
        "gyro_motion_mean_delta_t0",
    ]
    spec_by_col = {spec.column: spec for spec in AUTONOMIC_FEATURES}
    rows: list[dict[str, object]] = []
    for feature in [f for f in target_features if f in active.columns]:
        raw = _corr_pair(active, "pupil_mean_delta_t0", feature)
        row: dict[str, object] = {
            "pupil_feature": "pupil_mean_delta_t0",
            "physio_feature": feature,
            "physio_label": spec_by_col[feature].label,
            **raw,
        }
        covariate_cols = [c for c in MOTION_FEATURES if c in active.columns and c != feature]
        if feature in MOTION_FEATURES or not covariate_cols:
            row["motion_adjusted_n"] = np.nan
            row["motion_adjusted_pearson_r"] = np.nan
        else:
            cols = ["pupil_mean_delta_t0", feature, *covariate_cols]
            frame = active[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(frame) >= 8:
                x_resid = _residualize(frame["pupil_mean_delta_t0"], frame[covariate_cols])
                y_resid = _residualize(frame[feature], frame[covariate_cols])
                common = x_resid.index.intersection(y_resid.index)
                row["motion_adjusted_n"] = int(len(common))
                row["motion_adjusted_pearson_r"] = (
                    float(x_resid.loc[common].corr(y_resid.loc[common], method="pearson"))
                    if len(common) >= 5
                    else np.nan
                )
            else:
                row["motion_adjusted_n"] = int(len(frame))
                row["motion_adjusted_pearson_r"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _add_task_progress(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["window_start_lsl"] = pd.to_numeric(out["window_start_lsl"], errors="coerce")
    out["window_end_lsl"] = pd.to_numeric(out["window_end_lsl"], errors="coerce")
    out["task_progress"] = np.nan
    for _key, group in out.groupby(KEYS):
        start = float(group["window_start_lsl"].min())
        end = float(group["window_end_lsl"].max())
        denom = max(end - start, 1e-9)
        out.loc[group.index, "task_progress"] = (group["window_start_lsl"] - start) / denom
    return out


def _window_feature_profile(
    window: pd.DataFrame,
    specs: list[FeatureSpec],
) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame()
    data = window.copy()
    if "task" in data.columns and "task_id" not in data.columns:
        data = data.rename(columns={"task": "task_id"})
    data = _add_task_progress(data)
    bins = [-0.001, 1 / 3, 2 / 3, 1.001]
    labels = ["early", "middle", "late"]
    data["task_segment"] = pd.cut(data["task_progress"], bins=bins, labels=labels)

    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.column not in data.columns:
            continue
        base = (
            data[data["task_id"] == "T0"]
            .groupby(["session_id", "participant_id"])[spec.column]
            .median()
            .rename("baseline")
            .reset_index()
        )
        active = data[data["task_id"].isin(ACTIVE_TASKS)].merge(
            base,
            on=["session_id", "participant_id"],
            how="left",
        )
        active["feature_delta_t0"] = pd.to_numeric(active[spec.column], errors="coerce") - active[
            "baseline"
        ]
        for (task_id, segment), group in active.groupby(["task_id", "task_segment"], observed=False):
            rows.append(
                {
                    "feature": spec.column,
                    "label": spec.label,
                    "unit": spec.unit,
                    "modality": spec.modality,
                    "task_id": task_id,
                    "task_segment": str(segment),
                    **_series_stats(group["feature_delta_t0"]),
                }
            )
    return pd.DataFrame(rows)


def build_temporal_profile(physio_window: pd.DataFrame, pupil_window: pd.DataFrame) -> pd.DataFrame:
    physio_specs = [s for s in TEMPORAL_FEATURES if s.modality != "pupil"]
    pupil_specs = [s for s in TEMPORAL_FEATURES if s.modality == "pupil"]
    pieces = [
        _window_feature_profile(physio_window, physio_specs),
        _window_feature_profile(pupil_window, pupil_specs),
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_key_findings(
    task_stats: pd.DataFrame,
    pupil_links: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not task_stats.empty:
        top = task_stats.dropna(subset=["arousal_dz"]).copy()
        top["abs_arousal_dz"] = top["arousal_dz"].abs()
        for row in top.sort_values("abs_arousal_dz", ascending=False).head(8).itertuples(index=False):
            rows.append(
                {
                    "finding_type": "task_effect",
                    "label": f"{row.task_id}: {row.label}",
                    "n": row.n,
                    "value": row.mean,
                    "effect": row.arousal_dz,
                    "note": "direction-coded standardized change from T0",
                }
            )
    if not pupil_links.empty:
        links = pupil_links.dropna(subset=["spearman_rho"]).copy()
        links["abs_rho"] = links["spearman_rho"].abs()
        for row in links.sort_values("abs_rho", ascending=False).head(5).itertuples(index=False):
            rows.append(
                {
                    "finding_type": "pupil_physio_link",
                    "label": f"Pupil diameter vs {row.physio_label}",
                    "n": row.n,
                    "value": row.spearman_rho,
                    "effect": row.motion_adjusted_pearson_r,
                    "note": "value=Spearman rho; effect=motion-adjusted Pearson r when available",
                }
            )
    if not coverage.empty:
        for task_id, group in coverage[coverage["task_id"].isin(ACTIVE_TASKS)].groupby("task_id"):
            rows.append(
                {
                    "finding_type": "coverage",
                    "label": f"{task_id}: both modalities usable",
                    "n": int(group["expected_participant_rows"].sum()),
                    "value": float(group["both_usable_count"].sum()),
                    "effect": float(group["both_usable_count"].sum() / group[
                        "expected_participant_rows"
                    ].sum() * 100.0),
                    "note": "effect is percent of expected participant-task rows",
                }
            )
    return pd.DataFrame(rows)


def _plot_task_fingerprint(task_stats: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    labels = [spec.label for spec in AUTONOMIC_FEATURES]
    pivot = task_stats.pivot(index="label", columns="task_id", values="arousal_dz").reindex(labels)
    pivot = pivot[[task for task in ACTIVE_TASKS if task in pivot.columns]]
    values = np.abs(pivot.to_numpy(dtype=float)) if not pivot.empty else np.array([])
    finite = values[np.isfinite(values)]
    max_abs = max(float(finite.max()) if finite.size else 1.0, 0.5)

    fig, ax = plt.subplots(figsize=(8.8, 6.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_title("Autonomic task fingerprint vs T0", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("signed Cohen dz (HRV row inverted)")
    out_path = figures_dir / "autonomic_task_fingerprint_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_composite_task_summary(composite_stats: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = figures_dir / "autonomic_composite_task_summary.png"
    if composite_stats.empty:
        return out_path
    metrics = ["physio_arousal_index", "pupil_diameter_index", "motion_context_index"]
    labels = ["Physio", "Pupil", "Motion"]
    colors = ["#2f6f9f", "#6b7f2a", "#8a5a24"]
    x = np.arange(len(ACTIVE_TASKS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        sub = composite_stats[composite_stats["metric"] == metric].set_index("task_id")
        sub = sub.reindex(ACTIVE_TASKS)
        xpos = x + (idx - 1.0) * width
        ax.bar(xpos, sub["mean"], yerr=sub["sem"], width=width, label=label, color=color, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x, ACTIVE_TASKS)
    ax.set_ylabel("z-scored index")
    ax.set_title(
        "Physio response, pupil diameter, and motion context",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_modality_coverage(coverage: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = figures_dir / "autonomic_modality_coverage.png"
    if coverage.empty:
        return out_path
    pivot = coverage.pivot(index="session_id", columns="task_id", values="both_usable_pct")
    pivot = pivot[[task for task in TASK_ORDER if task in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(pivot) * 0.32 + 1.5)), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=100)
    ax.set_title("Rows with usable physio and pupil data", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    lookup = coverage.set_index(["session_id", "task_id"])
    for i, session_id in enumerate(pivot.index):
        for j, task_id in enumerate(pivot.columns):
            if (session_id, task_id) not in lookup.index:
                continue
            row = lookup.loc[(session_id, task_id)]
            ax.text(
                j,
                i,
                f"{int(row['both_usable_count'])}/{int(row['expected_participant_rows'])}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if row["both_usable_pct"] < 55 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("usable both modalities (%)")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_correlation_matrix(corr: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = figures_dir / "autonomic_cross_modal_correlation.png"
    if corr.empty:
        return out_path
    labels = [spec.label for spec in AUTONOMIC_FEATURES]
    col_to_label = {spec.column: spec.label for spec in AUTONOMIC_FEATURES}
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels, dtype=float)
    for row in corr.itertuples(index=False):
        a = col_to_label.get(row.feature_a, row.feature_a)
        b = col_to_label.get(row.feature_b, row.feature_b)
        matrix.loc[a, b] = row.spearman_rho
        matrix.loc[b, a] = row.spearman_rho
    fig, ax = plt.subplots(figsize=(9.4, 8.2), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Cross-signal correlation structure", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Spearman rho")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_pupil_physio_scatter(joined: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = figures_dir / "autonomic_pupil_physio_links.png"
    panels = [
        ("eda_phasic_rate_hz_delta_t0", "SCR rate change"),
        ("hr_mean_bpm_delta_t0", "HR change"),
    ]
    if "pupil_mean_delta_t0" not in joined.columns:
        return out_path
    active = joined[joined["task_id"].isin(ACTIVE_TASKS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    colors = {"T1": "#2f6f9f", "T2": "#a33f2f", "T3": "#6b7f2a", "T4": "#7c4d9e"}
    for ax, (feature, label) in zip(axes, panels, strict=True):
        if feature not in active.columns:
            ax.set_visible(False)
            continue
        pair = active[["task_id", "pupil_mean_delta_t0", feature]].copy()
        pair["pupil_mean_delta_t0"] = pd.to_numeric(pair["pupil_mean_delta_t0"], errors="coerce")
        pair[feature] = pd.to_numeric(pair[feature], errors="coerce")
        pair = pair.dropna(subset=["pupil_mean_delta_t0", feature])
        for task_id in ACTIVE_TASKS:
            sub = pair[pair["task_id"] == task_id]
            ax.scatter(
                sub["pupil_mean_delta_t0"],
                sub[feature],
                s=24,
                alpha=0.72,
                label=task_id,
                color=colors[task_id],
            )
        if len(pair) >= 5:
            x = pair["pupil_mean_delta_t0"].to_numpy(dtype=float)
            y = pair[feature].to_numpy(dtype=float)
            beta = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            ax.plot(xs, beta[0] * xs + beta[1], color="black", linewidth=1.2, alpha=0.75)
            rho = pd.Series(x).corr(pd.Series(y), method="spearman")
            ax.text(0.03, 0.95, f"rho={rho:.2f}, n={len(pair)}", transform=ax.transAxes, va="top")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.35)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_xlabel("Pupil diameter change from T0 (mm)")
        ax.set_ylabel(label)
        ax.set_title(f"Pupil vs {label}")
        ax.grid(alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_temporal_profile(profile: pd.DataFrame, figures_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out_path = figures_dir / "autonomic_temporal_profiles.png"
    if profile.empty:
        return out_path
    selected = [spec.column for spec in TEMPORAL_FEATURES]
    plot_df = profile[profile["feature"].isin(selected)].copy()
    segments = ["early", "middle", "late"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Baseline-centered within-task autonomic profiles", fontsize=14, fontweight="bold")
    colors = {"T1": "#2f6f9f", "T2": "#a33f2f", "T3": "#6b7f2a", "T4": "#7c4d9e"}
    for ax, feature in zip(axes.ravel(), selected, strict=True):
        fdf = plot_df[plot_df["feature"] == feature]
        if fdf.empty:
            ax.set_visible(False)
            continue
        label = fdf["label"].iloc[0]
        unit = fdf["unit"].iloc[0]
        for task_id in ACTIVE_TASKS:
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
        ax.axhline(0, color="black", linewidth=0.75, alpha=0.45)
        ax.set_xticks(np.arange(len(segments)), segments)
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[0].legend(frameon=False, ncols=4, fontsize=8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_outputs(
    physio_task: pd.DataFrame,
    physio_window: pd.DataFrame,
    physio_qc: pd.DataFrame,
    pupil_task: pd.DataFrame,
    pupil_window: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    joined = build_joined_task(physio_task, pupil_task)
    usability = build_feature_usability(joined, physio_qc, pupil_task)
    task_stats = build_task_delta_stats(joined)
    composite = build_composite_scores(joined)
    composite_stats = build_composite_task_stats(composite)
    coverage = build_modality_coverage(physio_qc, physio_task, pupil_task)
    correlations = build_feature_correlations(joined)
    pupil_links = build_pupil_physio_links(joined)
    temporal = build_temporal_profile(physio_window, pupil_window)
    findings = build_key_findings(task_stats, pupil_links, coverage)

    table_outputs = [
        (joined, results_dir / "autonomic_joined_participant_task.tsv"),
        (usability, results_dir / "autonomic_feature_usability.tsv"),
        (task_stats, results_dir / "autonomic_task_delta_stats.tsv"),
        (composite, results_dir / "autonomic_composite_scores.tsv"),
        (composite_stats, results_dir / "autonomic_composite_task_stats.tsv"),
        (coverage, results_dir / "autonomic_modality_coverage.tsv"),
        (correlations, results_dir / "autonomic_cross_modal_correlations.tsv"),
        (pupil_links, results_dir / "autonomic_pupil_physio_links.tsv"),
        (temporal, results_dir / "autonomic_temporal_profile.tsv"),
        (findings, results_dir / "autonomic_paper_key_findings.tsv"),
    ]
    for df, path in table_outputs:
        df.to_csv(path, sep="\t", index=False)

    figure_outputs = [
        _plot_task_fingerprint(task_stats, figures_dir, dpi),
        _plot_composite_task_summary(composite_stats, figures_dir, dpi),
        _plot_modality_coverage(coverage, figures_dir, dpi),
        _plot_correlation_matrix(correlations, figures_dir, dpi),
        _plot_pupil_physio_scatter(joined, figures_dir, dpi),
        _plot_temporal_profile(temporal, figures_dir, dpi),
    ]
    return [path for _df, path in table_outputs] + figure_outputs


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    physio_task, physio_window, physio_qc, pupil_raw, pupil_window = load_tables(args.features_dir)
    pupil_task = prepare_pupil_task(
        pupil_raw,
        min_valid_frac=args.min_pupil_valid_frac,
        max_missing_frac=args.max_pupil_missing_frac,
    )
    outputs = write_outputs(
        physio_task=physio_task,
        physio_window=physio_window,
        physio_qc=physio_qc,
        pupil_task=pupil_task,
        pupil_window=pupil_window,
        results_dir=args.results_dir.resolve(),
        figures_dir=args.figures_dir.resolve(),
        dpi=args.dpi,
    )
    for path in outputs:
        LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
