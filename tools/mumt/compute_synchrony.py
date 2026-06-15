"""compute_synchrony.py

Compute group physiological synchrony features for each 15-second window.

Strategy: within each (session_id, task), sort windows per seat by timestamp
and assign a relative window index. Match windows across seats by index
(window 0 of P1 with window 0 of P2, etc.), only pairing windows whose
timestamps are within 30 seconds of each other (15-second windows at 50%
overlap are ~7.5 s apart sequentially).

Synchrony features added per window (12 total):
  sync_eda_mean / _std / _max    — pairwise Pearson r of EDA mean time-series
  sync_pupil_mean / _std / _max  — pairwise Pearson r of pupil mean time-series
  sync_imu_mean / _std           — pairwise Pearson r of IMU magnitude
  sync_hr_mean / _std            — pairwise Pearson r of PPG mean
  n_sync_partners                — number of co-present partners with valid windows
  sync_global_mean               — mean across all 4 modality synchrony means

Usage
-----
  python tools/mumt/compute_synchrony.py \\
      --dataset  data/mumt/dataset_15s.pkl \\
      --out      data/mumt/dataset_15s_sync.pkl
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample as _resample

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_affectai import (
    EDA_SEQ_COLS, PPG_SEQ_COLS, PUPIL_SEQ_COLS, IMU_SEQ_COLS,
    seq_to_array,
)

T_COMMON = 200   # resample target
MAX_TIME_GAP = 30.0   # seconds: maximum timestamp gap for cross-seat pairing


def _extract(row, mod: str, cols: list[str]) -> np.ndarray:
    """Extract (T, F) sequence for a modality from a row, resample to T_COMMON."""
    raw = row.get(f"{mod}_seq")
    if raw is None:
        return np.zeros((T_COMMON, len(cols)), dtype=np.float32)
    arr = seq_to_array(raw, cols)
    if arr.shape[0] != T_COMMON:
        arr = _resample(arr, T_COMMON, axis=0).astype(np.float32)
    return arr.astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between temporal mean signals of two (T, F) arrays."""
    ta = a.mean(axis=1) if a.ndim == 2 else a
    tb = b.mean(axis=1) if b.ndim == 2 else b
    if np.std(ta) < 1e-8 or np.std(tb) < 1e-8:
        return 0.0
    return float(np.corrcoef(ta, tb)[0, 1])


ZERO_FEATS = {
    "sync_eda_mean": 0.0, "sync_eda_std": 0.0, "sync_eda_max": 0.0,
    "sync_pupil_mean": 0.0, "sync_pupil_std": 0.0, "sync_pupil_max": 0.0,
    "sync_imu_mean": 0.0, "sync_imu_std": 0.0,
    "sync_hr_mean": 0.0, "sync_hr_std": 0.0,
    "n_sync_partners": 0.0, "sync_global_mean": 0.0,
}


def compute_group_synchrony(session_task_df: pd.DataFrame) -> dict[int, dict]:
    """Given all rows for one (session_id, task), compute synchrony per row.

    Returns {original_df_index: sync_feature_dict}.
    """
    # Sort each seat's windows by timestamp, assign within-seat window index
    df = session_task_df.copy()
    ts_col = "vad_timestamp_lsl" if "vad_timestamp_lsl" in df.columns else None

    if ts_col:
        df = df.sort_values([ts_col])
    df["_seat_win_idx"] = df.groupby("seat").cumcount()

    # Build lookup: seat → {win_idx: (orig_index, timestamp, row)}
    seat_windows: dict[str, dict[int, tuple]] = {}
    for orig_idx, row in df.iterrows():
        seat = str(row.get("seat", "unknown"))
        widx = int(row["_seat_win_idx"])
        ts   = float(row[ts_col]) if ts_col else 0.0
        if seat not in seat_windows:
            seat_windows[seat] = {}
        seat_windows[seat][widx] = (orig_idx, ts, row)

    seats = sorted(seat_windows.keys())
    results: dict[int, dict] = {}

    for seat in seats:
        for widx, (orig_idx, ts_self, row_self) in seat_windows[seat].items():
            # Find matching windows from other seats (same widx, timestamp within gap)
            partner_rows = []
            for other_seat in seats:
                if other_seat == seat:
                    continue
                # Try same index first, then adjacent
                for candidate_idx in [widx, widx - 1, widx + 1]:
                    if candidate_idx in seat_windows[other_seat]:
                        o_idx, ts_other, row_other = seat_windows[other_seat][candidate_idx]
                        if ts_col and abs(ts_self - ts_other) > MAX_TIME_GAP:
                            continue
                        partner_rows.append(row_other)
                        break   # one partner per seat

            if not partner_rows:
                results[orig_idx] = dict(ZERO_FEATS)
                continue

            # Extract sequences
            self_seqs = {
                "eda":   _extract(row_self, "eda",   EDA_SEQ_COLS),
                "pupil": _extract(row_self, "pupil", PUPIL_SEQ_COLS),
                "imu":   _extract(row_self, "imu",   IMU_SEQ_COLS),
                "ppg":   _extract(row_self, "ppg",   PPG_SEQ_COLS),
            }
            partner_seqs_list = [{
                "eda":   _extract(p, "eda",   EDA_SEQ_COLS),
                "pupil": _extract(p, "pupil", PUPIL_SEQ_COLS),
                "imu":   _extract(p, "imu",   IMU_SEQ_COLS),
                "ppg":   _extract(p, "ppg",   PPG_SEQ_COLS),
            } for p in partner_rows]

            feats: dict[str, float] = {}
            for mod, rename in [("eda","eda"), ("pupil","pupil"), ("imu","imu"), ("ppg","hr")]:
                pairs = [_safe_corr(self_seqs[mod], ps[mod]) for ps in partner_seqs_list]
                # Also all partner-partner pairs
                for i in range(len(partner_rows)):
                    for j in range(i + 1, len(partner_rows)):
                        pairs.append(_safe_corr(partner_seqs_list[i][mod],
                                                partner_seqs_list[j][mod]))
                if pairs:
                    feats[f"sync_{rename}_mean"] = float(np.mean(pairs))
                    feats[f"sync_{rename}_std"]  = float(np.std(pairs))
                    feats[f"sync_{rename}_max"]  = float(np.max(pairs))
                else:
                    feats[f"sync_{rename}_mean"] = 0.0
                    feats[f"sync_{rename}_std"]  = 0.0
                    feats[f"sync_{rename}_max"]  = 0.0

            feats.pop("sync_hr_max", None)   # keep eda+pupil max only
            feats["n_sync_partners"] = float(len(partner_rows))
            feats["sync_global_mean"] = float(np.mean([
                feats["sync_eda_mean"], feats["sync_pupil_mean"],
                feats["sync_imu_mean"], feats["sync_hr_mean"],
            ]))
            results[orig_idx] = feats

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--out",     default="data/mumt/dataset_15s_sync.pkl")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    df = pd.read_pickle(args.dataset)
    n = len(df)
    print(f"  {n} windows | {df['subject_id'].nunique()} subjects | "
          f"{df['session_id'].nunique()} sessions")

    all_sync: list[dict] = [dict(ZERO_FEATS)] * n
    df_reset = df.reset_index(drop=True)

    groups = list(df_reset.groupby(["session_id", "task"]))
    print(f"  Processing {len(groups)} (session, task) groups ...")

    for gi, ((ses, task), grp) in enumerate(groups):
        if gi % 10 == 0:
            print(f"    {gi}/{len(groups)} ...", end="\r", flush=True)
        sync_dict = compute_group_synchrony(grp)
        for orig_idx, feats in sync_dict.items():
            all_sync[orig_idx] = feats

    print(f"    {len(groups)}/{len(groups)} ... done    ")

    df_reset["sync_features"] = all_sync

    # Report statistics
    sync_df = pd.DataFrame(all_sync)
    print(f"\n  Non-zero windows: {(sync_df['n_sync_partners'] > 0).sum()} / {n}")
    print(f"\n  Synchrony feature statistics (non-zero only):")
    nonzero = sync_df[sync_df["n_sync_partners"] > 0]
    if len(nonzero) > 0:
        print(nonzero.describe().round(3).to_string())
    else:
        print("  All zeros — no partner windows found")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_reset.to_pickle(out_path)
    print(f"\n  Saved: {out_path}  ({len(df_reset)} rows)")


if __name__ == "__main__":
    main()
