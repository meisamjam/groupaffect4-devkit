"""Compute dyad/group conversation-dynamics features from windowed physiology."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tools.features.common import pairs
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.features.common import pairs  # type: ignore[no-redef]

LOG = logging.getLogger("compute_group_dynamics")


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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _best_lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int = 2) -> tuple[float, int]:
    best = float("nan")
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = _pearson(x[:lag], y[-lag:])
        elif lag > 0:
            c = _pearson(x[lag:], y[:-lag])
        else:
            c = _pearson(x, y)
        if np.isnan(best) or (np.isfinite(c) and c > best):
            best = c
            best_lag = lag
    return best, best_lag


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute group-dynamics features from window-level tables.")
    p.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing features_physio_window_30s.tsv and features_pupil_window_30s.tsv.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def _load_window_tables(features_dir: Path) -> pd.DataFrame:
    physio = pd.read_csv(features_dir / "features_physio_window_30s.tsv", sep="\t")
    pupil = pd.read_csv(features_dir / "features_pupil_window_30s.tsv", sep="\t")
    keys = ["session_id", "task", "participant_id", "window_index", "window_start_lsl", "window_end_lsl"]
    keep_physio = keys + [c for c in ["eda_mean", "ppg_rate_proxy_bpm"] if c in physio.columns]
    keep_pupil = keys + [c for c in ["pupil_mean"] if c in pupil.columns]
    merged = physio[keep_physio].merge(pupil[keep_pupil], on=keys, how="outer")
    return merged


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    features_dir = args.features_dir.resolve()
    merged = _load_window_tables(features_dir)
    LOG.info("Loaded %d window rows.", len(merged))

    metrics = [c for c in ["eda_mean", "ppg_rate_proxy_bpm", "pupil_mean"] if c in merged.columns]
    if merged.empty or not metrics:
        out_window = features_dir / "features_group_dynamics_window_30s.tsv"
        out_dyad = features_dir / "features_group_dynamics_task.tsv"
        pd.DataFrame().to_csv(out_window, sep="\t", index=False)
        pd.DataFrame().to_csv(out_dyad, sep="\t", index=False)
        LOG.warning("No usable window metrics found; wrote empty dynamics tables.")
        return 0

    group_window_rows: list[dict] = []
    dyad_rows: list[dict] = []

    for (session_id, task, window_index), wdf in merged.groupby(["session_id", "task", "window_index"]):
        row = {"session_id": session_id, "task": task, "window_index": int(window_index)}
        row["n_participants"] = int(wdf["participant_id"].nunique())
        row["window_start_lsl"] = float(wdf["window_start_lsl"].iloc[0])
        row["window_end_lsl"] = float(wdf["window_end_lsl"].iloc[0])
        for metric in metrics:
            vals = pd.to_numeric(wdf[metric], errors="coerce").to_numpy(dtype=float)
            row[f"{metric}_group_mean"] = _safe_nanmean(vals)
            row[f"{metric}_group_std"] = _safe_nanstd(vals)
        group_window_rows.append(row)

    for (session_id, task), tdf in merged.groupby(["session_id", "task"]):
        participants = sorted(tdf["participant_id"].dropna().unique().tolist())
        for p1, p2 in pairs(participants):
            pair = tdf[tdf["participant_id"].isin([p1, p2])].copy()
            for metric in metrics:
                piv = pair.pivot_table(
                    index="window_index",
                    columns="participant_id",
                    values=metric,
                    aggfunc="mean",
                )
                if p1 not in piv.columns or p2 not in piv.columns:
                    continue
                x = pd.to_numeric(piv[p1], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(piv[p2], errors="coerce").to_numpy(dtype=float)
                corr = _pearson(x, y)
                best_corr, best_lag = _best_lag_corr(x, y, max_lag=2)
                dyad_rows.append(
                    {
                        "session_id": session_id,
                        "task": task,
                        "participant_a": p1,
                        "participant_b": p2,
                        "metric": metric,
                        "window_count": int(len(piv)),
                        "corr": corr,
                        "best_lag_corr": best_corr,
                        "best_lag_windows": int(best_lag),
                    }
                )

    out_window = features_dir / "features_group_dynamics_window_30s.tsv"
    out_dyad = features_dir / "features_group_dynamics_task.tsv"
    gw_df = pd.DataFrame(group_window_rows)
    dy_df = pd.DataFrame(dyad_rows)
    if not gw_df.empty:
        gw_df = gw_df.sort_values(["session_id", "task", "window_index"])
    if not dy_df.empty:
        dy_df = dy_df.sort_values(["session_id", "task", "participant_a", "participant_b", "metric"])
    gw_df.to_csv(out_window, sep="\t", index=False)
    dy_df.to_csv(out_dyad, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", out_window, len(group_window_rows))
    LOG.info("Wrote %s (%d rows)", out_dyad, len(dyad_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
