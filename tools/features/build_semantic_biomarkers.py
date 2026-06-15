"""Build semantic biomarker tables from extracted physio/pupil/dynamics features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_semantic_biomarkers")

TASK_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}


def _z_within(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df[col], errors="coerce")
    g = values.groupby([df[c] for c in group_cols])
    mean = g.transform("mean")
    std = g.transform("std")
    z = (values - mean) / std.replace(0.0, np.nan)
    # If variance is zero inside a group, treat finite samples as neutral (z=0) instead of NaN.
    zero_var = (std.isna() | (std == 0.0)) & values.notna()
    z = z.where(~zero_var, 0.0)
    return z


def _baseline_z(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df[col], errors="coerce")
    base = (
        df[df["task"] == "T0"][["session_id", "participant_id"]].assign(**{f"{col}_baseline": values[df["task"] == "T0"]})
        .copy()
    )
    out = df.copy()
    out[col] = values
    out = out.merge(base, on=["session_id", "participant_id"], how="left")
    baseline = out[f"{col}_baseline"]
    centered = out[col] - baseline
    scale = out.groupby(["session_id", "participant_id"])[col].transform("std")
    z = centered / scale.replace(0.0, np.nan)
    zero_var = (scale.isna() | (scale == 0.0)) & out[col].notna() & baseline.notna()
    z = z.where(~zero_var, 0.0)
    # If baseline task T0 is missing for a participant, fall back to within-participant z.
    fallback = _z_within(out, col, ["session_id", "participant_id"])
    return z.where(baseline.notna(), fallback)


def _score_to_label(value: float) -> str:
    if not np.isfinite(value):
        return "Unknown"
    if value >= 1.0:
        return "High"
    if value <= -1.0:
        return "Low"
    return "Moderate"


def _composite_mean(*parts: pd.Series) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    arr = np.column_stack([p.to_numpy(dtype=float, copy=False) for p in parts])
    valid = np.isfinite(arr)
    counts = valid.sum(axis=1)
    sums = np.where(valid, arr, 0.0).sum(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return pd.Series(out, index=parts[0].index)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build semantic biomarker composites from derived features.")
    p.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory with extracted feature TSV files.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def _participant_task_semantics(features_dir: Path) -> pd.DataFrame:
    physio_path = features_dir / "features_physio_participant_task.tsv"
    pupil_path = features_dir / "features_pupil_participant_task.tsv"
    if not physio_path.exists() or not pupil_path.exists():
        return pd.DataFrame()
    physio = pd.read_csv(physio_path, sep="\t")
    pupil = pd.read_csv(pupil_path, sep="\t")
    keys = ["session_id", "task", "participant_id"]
    merged = physio.merge(pupil, on=keys, how="outer", suffixes=("_physio", "_pupil"))
    merged["task_index"] = merged["task"].map(TASK_ORDER)

    for col in ["eda_mean", "scr_rate_hz", "ppg_rate_proxy_bpm", "ppg_rmssd_ms", "pupil_mean", "pupil_std"]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    z_eda = _baseline_z(merged, "eda_mean")
    z_scr = _baseline_z(merged, "scr_rate_hz")
    z_bpm = _baseline_z(merged, "ppg_rate_proxy_bpm")
    z_rmssd = _baseline_z(merged, "ppg_rmssd_ms")
    z_pupil = _baseline_z(merged, "pupil_mean")
    z_pupil_std = _baseline_z(merged, "pupil_std")
    z_missing = _baseline_z(merged, "pupil_missing_frac")
    z_gaze_valid = _baseline_z(merged, "gaze_valid_frac")

    merged["biomarker_cognitive_load"] = _composite_mean(z_pupil, z_eda, -z_rmssd)
    merged["biomarker_arousal_stress"] = _composite_mean(z_scr, z_bpm, z_pupil_std)
    merged["biomarker_attention"] = _composite_mean(z_gaze_valid, -z_missing, -z_pupil_std)
    merged["biomarker_decision_pressure"] = _composite_mean(merged["biomarker_arousal_stress"], z_eda)
    merged["biomarker_recovery_capacity"] = _composite_mean(-z_eda, -z_bpm, z_rmssd)
    merged["biomarker_fatigue_depletion"] = np.where(
        merged["task_index"].fillna(0) >= 1,
        merged["task_index"].fillna(0) * (-z_pupil) / 4.0,
        np.nan,
    )
    merged["state_load_label"] = merged["biomarker_cognitive_load"].map(_score_to_label)
    merged["state_arousal_label"] = merged["biomarker_arousal_stress"].map(_score_to_label)
    merged["state_attention_label"] = merged["biomarker_attention"].map(_score_to_label)
    return merged


def _window_semantics(features_dir: Path) -> pd.DataFrame:
    physio_w_path = features_dir / "features_physio_window_30s.tsv"
    pupil_w_path = features_dir / "features_pupil_window_30s.tsv"
    if not physio_w_path.exists() or not pupil_w_path.exists():
        return pd.DataFrame()
    physio_w = pd.read_csv(physio_w_path, sep="\t")
    pupil_w = pd.read_csv(pupil_w_path, sep="\t")
    group_w_path = features_dir / "features_group_dynamics_window_30s.tsv"
    keys = ["session_id", "task", "window_index", "window_start_lsl", "window_end_lsl", "participant_id"]
    merged = physio_w.merge(pupil_w, on=keys, how="outer", suffixes=("_physio", "_pupil"))
    if group_w_path.exists():
        group_w = pd.read_csv(group_w_path, sep="\t")
        merged = merged.merge(
            group_w[["session_id", "task", "window_index", "n_participants"]],
            on=["session_id", "task", "window_index"],
            how="left",
        )

    for col in ["eda_mean", "scr_rate_hz", "ppg_rate_proxy_bpm", "ppg_rmssd_ms", "pupil_mean", "pupil_std"]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    gcols = ["session_id", "participant_id", "task"]
    z_eda = _z_within(merged, "eda_mean", gcols)
    z_scr = _z_within(merged, "scr_rate_hz", gcols)
    z_bpm = _z_within(merged, "ppg_rate_proxy_bpm", gcols)
    z_rmssd = _z_within(merged, "ppg_rmssd_ms", gcols)
    z_pupil = _z_within(merged, "pupil_mean", gcols)
    z_pupil_std = _z_within(merged, "pupil_std", gcols)
    z_missing = _z_within(merged, "pupil_missing_frac", gcols)
    z_gaze_valid = _z_within(merged, "gaze_valid_frac", gcols)

    merged["biomarker_cognitive_load"] = _composite_mean(z_pupil, z_eda, -z_rmssd)
    merged["biomarker_arousal_stress"] = _composite_mean(z_scr, z_bpm, z_pupil_std)
    merged["biomarker_attention"] = _composite_mean(z_gaze_valid, -z_missing, -z_pupil_std)
    merged["biomarker_decision_pressure"] = _composite_mean(merged["biomarker_arousal_stress"], z_eda)
    merged["state_load_label"] = merged["biomarker_cognitive_load"].map(_score_to_label)
    merged["state_arousal_label"] = merged["biomarker_arousal_stress"].map(_score_to_label)
    return merged


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    features_dir = args.features_dir.resolve()
    task_df = _participant_task_semantics(features_dir)
    window_df = _window_semantics(features_dir)

    task_out = features_dir / "semantic_biomarkers_participant_task.tsv"
    window_out = features_dir / "semantic_biomarkers_window_30s.tsv"
    if not task_df.empty:
        task_df = task_df.sort_values(["session_id", "task", "participant_id"])
    if not window_df.empty:
        window_df = window_df.sort_values(["session_id", "task", "participant_id", "window_index"])
    task_df.to_csv(task_out, sep="\t", index=False)
    window_df.to_csv(window_out, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", task_out, len(task_df))
    LOG.info("Wrote %s (%d rows)", window_out, len(window_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
