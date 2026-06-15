"""Shared helpers for task-aware physiological feature extraction tools."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_RE_TASK = re.compile(r"_task-(T0|T1|T2|T3|T4)")
_RE_PARTICIPANT = re.compile(r"_acq-(P[1-4])_")
_RE_SESSION = re.compile(r"ses-[^\\/]+")


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root (e.g., affectai-data-processing-seed/data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "derived_features",
        help="Output directory for derived feature tables.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional session IDs to include (e.g., ses-20260317_grp-09_run01).",
    )


def discover_session_dirs(data_root: Path, sessions: list[str] | None = None) -> list[Path]:
    data_root = data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    candidates = sorted(data_root.glob("sub-*/ses-*"))
    if not candidates and data_root.name.startswith("ses-"):
        candidates = [data_root]
    if not candidates:
        candidates = sorted(data_root.glob("ses-*"))

    if sessions:
        wanted = set(sessions)
        candidates = [p for p in candidates if p.name in wanted]
    return candidates


def read_tsv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression)


def parse_task_from_name(name: str) -> str | None:
    m = _RE_TASK.search(name)
    return m.group(1) if m else None


def parse_participant_from_name(name: str) -> str | None:
    m = _RE_PARTICIPANT.search(name)
    return m.group(1) if m else None


def parse_session_from_path(path: Path) -> str:
    m = _RE_SESSION.search(str(path))
    return m.group(0) if m else path.parent.name


def numeric_columns(df: pd.DataFrame, prefix: str = "value_") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "lsl_time" in df.columns:
        df["lsl_time"] = pd.to_numeric(df["lsl_time"], errors="coerce")
    return cols


def sample_rate_hz(df: pd.DataFrame) -> float | None:
    if "lsl_time" not in df.columns:
        return None
    s = df["lsl_time"].dropna().sort_values()
    if len(s) < 5:
        return None
    d = np.diff(s.to_numpy(dtype=float))
    d = d[(d > 0) & np.isfinite(d)]
    if len(d) == 0:
        return None
    return float(1.0 / np.median(d))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    x0 = xv - xv.min()
    denom = np.dot(x0, x0)
    if denom <= 0:
        return float("nan")
    return float(np.dot(x0, yv - yv.mean()) / denom)


def rolling_windows(df: pd.DataFrame, window_s: float, step_s: float) -> Iterable[pd.DataFrame]:
    if "lsl_time" not in df.columns:
        return []
    s = df["lsl_time"].dropna()
    if s.empty:
        return []
    t_min = float(s.min())
    t_max = float(s.max())
    if t_max - t_min < window_s:
        return []
    out = []
    i = 0
    start = t_min
    while start + window_s <= t_max + 1e-6:
        end = start + window_s
        w = df[(df["lsl_time"] >= start) & (df["lsl_time"] < end)].copy()
        if not w.empty:
            w["window_index"] = i
            w["window_start_lsl"] = start
            w["window_end_lsl"] = end
            out.append(w)
        start += step_s
        i += 1
    return out


def pairs(values: list[str]) -> Iterable[tuple[str, str]]:
    return itertools.combinations(sorted(values), 2)

