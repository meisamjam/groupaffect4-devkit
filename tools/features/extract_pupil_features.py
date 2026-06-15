"""Extract participant-level and rolling-window Tobii pupil features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tools.features.common import (
        add_common_io_args,
        discover_session_dirs,
        linear_slope,
        numeric_columns,
        parse_participant_from_name,
        parse_session_from_path,
        parse_task_from_name,
        read_tsv,
        rolling_windows,
        sample_rate_hz,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.features.common import (  # type: ignore[no-redef]
        add_common_io_args,
        discover_session_dirs,
        linear_slope,
        numeric_columns,
        parse_participant_from_name,
        parse_session_from_path,
        parse_task_from_name,
        read_tsv,
        rolling_windows,
        sample_rate_hz,
    )

LOG = logging.getLogger("extract_pupil_features")


def _safe_nanmean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask]))


def _safe_nanstd(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mask = np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    return float(np.std(x[mask], ddof=0))


def _pairwise_nanmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 and b.size == 0:
        return np.array([], dtype=float)
    if a.size == 0:
        return b.astype(float, copy=True)
    if b.size == 0:
        return a.astype(float, copy=True)
    n = min(len(a), len(b))
    stack = np.vstack([a[:n], b[:n]]).astype(float, copy=False)
    mask = np.isfinite(stack)
    counts = mask.sum(axis=0)
    sums = np.where(mask, stack, 0.0).sum(axis=0)
    out = np.full(n, np.nan, dtype=float)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _value(df: pd.DataFrame, idx: int) -> np.ndarray:
    col = f"value_{idx}"
    if col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float, copy=False)


def compute_pupil_features(
    df: pd.DataFrame,
    pupil_left_idx: int,
    pupil_right_idx: int,
    gaze_valid_idx: int,
) -> dict[str, float]:
    t = df["lsl_time"].to_numpy(dtype=float, copy=False)
    left = _value(df, pupil_left_idx)
    right = _value(df, pupil_right_idx)
    gaze_valid = _value(df, gaze_valid_idx)
    pupil = _pairwise_nanmean(left, right)

    duration_s = float(np.nanmax(t) - np.nanmin(t)) if len(t) else float("nan")
    missing_frac = float(np.mean(~np.isfinite(pupil))) if len(pupil) else 1.0
    valid_frac = _safe_nanmean(gaze_valid)

    return {
        "sample_rate_hz": sample_rate_hz(df) or float("nan"),
        "duration_s": duration_s,
        "pupil_left_mean": _safe_nanmean(left),
        "pupil_right_mean": _safe_nanmean(right),
        "pupil_mean": _safe_nanmean(pupil),
        "pupil_std": _safe_nanstd(pupil),
        "pupil_slope_per_s": linear_slope(t, pupil) if len(pupil) else float("nan"),
        "pupil_missing_frac": missing_frac,
        "gaze_valid_frac": valid_frac,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract Tobii pupil features from task-split TSV files.")
    add_common_io_args(p)
    p.add_argument("--window-s", type=float, default=30.0, help="Rolling window length (seconds).")
    p.add_argument("--step-s", type=float, default=15.0, help="Rolling window step (seconds).")
    p.add_argument("--pupil-left-idx", type=int, default=2, help="Left pupil index in value_*.")
    p.add_argument("--pupil-right-idx", type=int, default=3, help="Right pupil index in value_*.")
    p.add_argument("--gaze-valid-idx", type=int, default=4, help="Gaze-valid index in value_*.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    session_dirs = discover_session_dirs(args.data_root, args.sessions)
    LOG.info("Found %d session directories.", len(session_dirs))

    task_rows: list[dict] = []
    window_rows: list[dict] = []

    for session_dir in session_dirs:
        et_dir = session_dir / "et"
        if not et_dir.exists():
            continue
        files = sorted(et_dir.glob("*_acq-P*_tobii.tsv.gz"))
        for path in files:
            task = parse_task_from_name(path.name)
            participant = parse_participant_from_name(path.name)
            if task is None or participant is None:
                continue
            df = read_tsv(path)
            numeric_columns(df)
            if "lsl_time" not in df.columns:
                continue
            df = df.dropna(subset=["lsl_time"]).sort_values("lsl_time").reset_index(drop=True)
            if df.empty:
                continue
            meta = {
                "session_id": parse_session_from_path(path),
                "task": task,
                "participant_id": participant,
                "source_file": str(path),
            }
            feat = compute_pupil_features(
                df,
                args.pupil_left_idx,
                args.pupil_right_idx,
                args.gaze_valid_idx,
            )
            task_rows.append({**meta, **feat})

            for w in rolling_windows(df, args.window_s, args.step_s):
                w_meta = {
                    **meta,
                    "window_index": int(w["window_index"].iloc[0]),
                    "window_start_lsl": float(w["window_start_lsl"].iloc[0]),
                    "window_end_lsl": float(w["window_end_lsl"].iloc[0]),
                }
                w_feat = compute_pupil_features(
                    w,
                    args.pupil_left_idx,
                    args.pupil_right_idx,
                    args.gaze_valid_idx,
                )
                window_rows.append({**w_meta, **w_feat})

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / "features_pupil_participant_task.tsv"
    window_path = out_dir / "features_pupil_window_30s.tsv"
    task_df = pd.DataFrame(task_rows)
    window_df = pd.DataFrame(window_rows)
    if not task_df.empty:
        task_df = task_df.sort_values(["session_id", "task", "participant_id"])
    if not window_df.empty:
        window_df = window_df.sort_values(["session_id", "task", "participant_id", "window_index"])
    task_df.to_csv(task_path, sep="\t", index=False)
    window_df.to_csv(window_path, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", task_path, len(task_rows))
    LOG.info("Wrote %s (%d rows)", window_path, len(window_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
