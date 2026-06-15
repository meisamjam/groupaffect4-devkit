"""pickle_generation_pretrain.py

Extract dense sliding windows from GroupAffect-4 sensor files for
self-supervised pre-training of the MuMTAffect backbone.

Windows are extracted within each task's LSL time range (from task_run_windows.tsv)
so windows never span task boundaries or break periods. Each window stores the
absolute LSL timestamps so label_augmentation.py can align them to self-reports.

Output: pretrain_dataset.pkl
  Same schema as dataset.pkl but without VAD labels. Extra columns:
    window_index   — sequential index within (session, seat, task)
    window_t_start — LSL time of first sample in the window
    window_t_center— LSL time of window midpoint
    window_t_end   — LSL time of last sample in the window

Usage:
  python tools/mumt/pickle_generation_pretrain.py --dataset-path data/zenodo
  python tools/mumt/pickle_generation_pretrain.py \
      --dataset-path data/zenodo \
      --output data/mumt/pretrain_dataset.pkl \
      --window-sec 30 --step-sec 15
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pickle_generation_affectai import (
    ALL_EMOTIBIT_COLS,
    EDA_SEQ_STORED,
    FIXED_LENGTH,
    GAZE_COLS,
    IMU_COLS,
    PPG_COLS,
    PUPIL_COLS,
    SEATS,
    SEX_MAP,
    TASKS,
    WINDOW_SEC,
    add_eda_decomposition,
    compute_eda_features,
    compute_gaze_features,
    compute_imu_features,
    compute_ppg_features,
    compute_pupil_features,
    downsample_to_fixed,
    load_group_participants,
    load_task_windows,
    load_tsv_gz,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from neurokit2.misc import NeuroKitWarning
    warnings.filterwarnings("ignore", category=NeuroKitWarning)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding window extraction
# ---------------------------------------------------------------------------

def sliding_windows(
    df: pd.DataFrame,
    t_start_bound: float,
    t_end_bound: float,
    window_sec: float,
    step_sec: float,
    min_samples: int = 20,
) -> list[dict]:
    """Slide a window over *df* within [t_start_bound, t_end_bound].

    Returns list of dicts: {window_index, t_start, t_end, t_center, data}
    Windows that cross the task boundary are discarded.
    """
    windows = []
    t_cursor = t_start_bound
    idx = 0
    while t_cursor + window_sec <= t_end_bound + 1e-3:
        t_win_end = t_cursor + window_sec
        win = df[(df["lsl_time"] >= t_cursor) & (df["lsl_time"] < t_win_end)].copy()
        if len(win) >= min_samples:
            windows.append({
                "window_index":  idx,
                "t_start":  float(win["lsl_time"].min()),
                "t_end":    float(win["lsl_time"].max()),
                "t_center": float(win["lsl_time"].mean()),
                "data":     win.reset_index(drop=True),
            })
        t_cursor += step_sec
        idx += 1
    return windows


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def process_session(
    session_dir: Path,
    window_sec: float,
    step_sec: float,
) -> list[dict]:
    ses_id = session_dir.name

    annot_dir  = session_dir / "annot"
    et_dir     = session_dir / "et"
    physio_dir = session_dir / "physio"

    task_windows = load_task_windows(annot_dir)
    group_parts  = load_group_participants(annot_dir)

    if group_parts.empty:
        log.warning("No group_participants in %s — skipping.", ses_id)
        return []

    records: list[dict] = []

    for seat in SEATS:
        row = group_parts[group_parts["seat"] == seat]
        if row.empty:
            continue
        row = row.iloc[0]

        participant_id = str(row["participant_id"])
        personality = {
            "bfi44_e": float(row.get("bfi44_e", float("nan"))),
            "bfi44_a": float(row.get("bfi44_a", float("nan"))),
            "bfi44_c": float(row.get("bfi44_c", float("nan"))),
            "bfi44_n": float(row.get("bfi44_n", float("nan"))),
            "bfi44_o": float(row.get("bfi44_o", float("nan"))),
        }
        sex_int = SEX_MAP.get(str(row.get("sex", "")).lower(), -1)
        age_val = int(row.get("age", -1)) if pd.notna(row.get("age")) else -1

        for task in TASKS:
            if task not in task_windows:
                log.debug("  No task window for %s / %s", ses_id, task)
                continue

            t_task_start, t_task_end = task_windows[task]

            # Load sensor files for this seat + task
            tobii_files = list(et_dir.glob(f"*task-{task}_run-01_acq-{seat}_tobii.tsv.gz"))
            emoti_files = list(physio_dir.glob(f"*task-{task}_run-01_acq-{seat}_emotibit.tsv.gz"))

            tobii_df = load_tsv_gz(tobii_files[0], GAZE_COLS + PUPIL_COLS) if tobii_files else None
            emoti_df = load_tsv_gz(emoti_files[0], ALL_EMOTIBIT_COLS) if emoti_files else None

            if tobii_df is None and emoti_df is None:
                continue

            # Clip sensor data to task boundaries
            if tobii_df is not None:
                tobii_df = tobii_df[
                    (tobii_df["lsl_time"] >= t_task_start) &
                    (tobii_df["lsl_time"] <= t_task_end)
                ].reset_index(drop=True)

            if emoti_df is not None:
                emoti_df = emoti_df[
                    (emoti_df["lsl_time"] >= t_task_start) &
                    (emoti_df["lsl_time"] <= t_task_end)
                ].reset_index(drop=True)

            # Use emotibit as the master timeline for window placement
            # (higher and more regular sampling rate than Tobii for this purpose)
            master_df = emoti_df if emoti_df is not None else tobii_df
            win_list  = sliding_windows(master_df, t_task_start, t_task_end,
                                        window_sec, step_sec)

            for win_meta in win_list:
                wi     = win_meta["window_index"]
                t_s    = win_meta["t_start"]
                t_e    = win_meta["t_end"]
                t_c    = win_meta["t_center"]

                record: dict = {
                    "session_id":    ses_id,
                    "subject_id":    participant_id,
                    "seat":          seat,
                    "task":          task,
                    "window_index":  wi,
                    "window_t_start":  t_s,
                    "window_t_end":    t_e,
                    "window_t_center": t_c,
                    "task_start_lsl":  t_task_start,
                    "task_end_lsl":    t_task_end,
                    "sex":           sex_int,
                    "age":           age_val,
                    **personality,
                }

                # ── Gaze / Pupil ──────────────────────────────────────
                if tobii_df is not None:
                    gaze_win = tobii_df[
                        (tobii_df["lsl_time"] >= t_s) & (tobii_df["lsl_time"] < t_e)
                    ]
                    if len(gaze_win) >= 10:
                        record["gaze_seq"]       = downsample_to_fixed(gaze_win, GAZE_COLS)
                        record["pupil_seq"]      = downsample_to_fixed(gaze_win, PUPIL_COLS)
                        record["gaze_features"]  = compute_gaze_features(gaze_win)
                        record["pupil_features"] = compute_pupil_features(gaze_win)
                    else:
                        record["gaze_seq"]       = pd.DataFrame(np.zeros((FIXED_LENGTH, len(GAZE_COLS))),  columns=GAZE_COLS)
                        record["pupil_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PUPIL_COLS))), columns=PUPIL_COLS)
                        record["gaze_features"]  = {}
                        record["pupil_features"] = {}
                else:
                    record["gaze_seq"]       = pd.DataFrame(np.zeros((FIXED_LENGTH, len(GAZE_COLS))),  columns=GAZE_COLS)
                    record["pupil_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PUPIL_COLS))), columns=PUPIL_COLS)
                    record["gaze_features"]  = {}
                    record["pupil_features"] = {}

                # ── EmotiBit ──────────────────────────────────────────
                if emoti_df is not None:
                    emoti_win = emoti_df[
                        (emoti_df["lsl_time"] >= t_s) & (emoti_df["lsl_time"] < t_e)
                    ]
                    if len(emoti_win) >= 10:
                        sr_em = max(1, int(len(emoti_win) / window_sec))
                        eda_decomp = add_eda_decomposition(emoti_win, sr_em)
                        record["eda_seq"]      = downsample_to_fixed(eda_decomp,  EDA_SEQ_STORED)
                        record["ppg_seq"]      = downsample_to_fixed(emoti_win,   PPG_COLS)
                        record["imu_seq"]      = downsample_to_fixed(emoti_win,   IMU_COLS)
                        record["eda_features"] = compute_eda_features(emoti_win, sr_em)
                        record["ppg_features"] = compute_ppg_features(emoti_win, sr_em)
                        record["imu_features"] = compute_imu_features(emoti_win)
                    else:
                        record["eda_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(EDA_SEQ_STORED))), columns=EDA_SEQ_STORED)
                        record["ppg_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PPG_COLS))),        columns=PPG_COLS)
                        record["imu_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(IMU_COLS))),        columns=IMU_COLS)
                        record["eda_features"] = {}
                        record["ppg_features"] = {}
                        record["imu_features"] = {}
                else:
                    record["eda_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(EDA_SEQ_STORED))), columns=EDA_SEQ_STORED)
                    record["ppg_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(PPG_COLS))),        columns=PPG_COLS)
                    record["imu_seq"]      = pd.DataFrame(np.zeros((FIXED_LENGTH, len(IMU_COLS))),        columns=IMU_COLS)
                    record["eda_features"] = {}
                    record["ppg_features"] = {}
                    record["imu_features"] = {}

                records.append(record)

        log.debug("  %s / %s: %d windows so far", ses_id, seat, len(records))

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    root = Path(args.dataset_path)

    session_dirs = sorted(root.rglob("ses-*"))
    session_dirs = [s for s in session_dirs if s.is_dir() and (s / "annot").exists()]
    log.info("Found %d session directories.", len(session_dirs))
    log.info("Window %.0fs  |  Step %.0fs", args.window_sec, args.step_sec)

    all_records: list[dict] = []
    for i, ses_dir in enumerate(session_dirs, 1):
        log.info("[%d/%d] %s …", i, len(session_dirs), ses_dir.name)
        recs = process_session(ses_dir, args.window_sec, args.step_sec)
        log.info("  → %d windows", len(recs))
        all_records.extend(recs)

    if not all_records:
        log.error("No windows extracted. Check dataset path.")
        return

    df = pd.DataFrame(all_records)
    log.info("Total: %d windows  |  %d subjects  |  %d sessions",
             len(df), df["subject_id"].nunique(), df["session_id"].nunique())
    log.info("Task distribution: %s", df["task"].value_counts().to_dict())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(str(out))
    log.info("Saved to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sliding pretrain windows from GroupAffect-4 (zenodo format)."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Root of the zenodo BIDS dataset.")
    parser.add_argument("--output", default="data/mumt/pretrain_dataset.pkl",
                        help="Output pickle path.")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--step-sec",   type=float, default=15.0)
    args = parser.parse_args()
    main(args)
