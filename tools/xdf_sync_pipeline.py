#!/usr/bin/env python3
"""XDF-based Synchronization Pipeline.

Synchronizes all multimodal data using XDF files (LSL recordings) as the
timing backbone for each session.  Extracts streams, derives T0–T4 task
windows from stimuli markers, and writes only the **synchronized processed
data** — no source files are copied.

Videos are **excluded** from the output but their clock anchors (start times
and durations from LSL sync streams) are preserved so downstream pipelines
can align video later.

Workflow per session
--------------------
1. Locate XDF file(s) (CurrentStudy or recording-session dirs)
2. Locate stimuli experiment events for task-window derivation
3. Locate AV session dir for video/audio clock anchors (optional)
4. Load XDF via ``pyxdf`` → categorize every LSL stream
5. Derive T0–T4 task windows from experiment markers
6. Split every extracted stream TSV by task window
7. Record video clock anchors (no video/audio files copied)
8. Write events.tsv, sync_metadata.json, participant_signal_map.tsv

Output structure (processed_data/sub-01/ses-{id}/)
---------------------------------------------------
::

    annot/
        *_task_run_windows.tsv
        *_segment_windows.tsv
        *_video_clock_anchors.tsv
        *_participant_signal_map.tsv
        *_sync_metadata.json
    beh/
        *_task-T{n}_run-01_events.tsv   (per task)
        *_stimuli_answers.tsv
    et/
        *_task-T{n}_run-01_acq-lsl_tobii.tsv.gz   (per task)
    physio/
        *_task-T{n}_run-01_acq-lsl_emotibit.tsv.gz (per task)
    eeg/
        *_task-T{n}_run-01_acq-lsl_vicon.tsv.gz    (per task, if present)
    events.tsv  ← authoritative timeline spine

Usage
-----
::

    python tools/xdf_sync_pipeline.py \\
        --data-root affectai-data-processing-seed/data \\
        --output-dir E:/processed_data \\
        [--sessions ses-20260312_grp-07_run01 ...] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import array
import bisect
import csv
import gzip
import importlib.util
import json
import logging
import re
import statistics
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xdf_sync_pipeline")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]

# ---------------------------------------------------------------------------
# Stream classification helpers
# ---------------------------------------------------------------------------

_RE_TOBII = re.compile(r"^Tobii[_\-]", re.IGNORECASE)
_RE_EMOTIBIT = re.compile(r"^Emotibit[_\-]", re.IGNORECASE)
_RE_VICON = re.compile(r"^Vicon", re.IGNORECASE)
_RE_AFFECTAI = re.compile(r"^AffectAI[_\-]", re.IGNORECASE)
_RE_SYNC = re.compile(r"^ffmpeg[_\-]|^dpa[_\-]", re.IGNORECASE)
_RE_PARTICIPANT_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:p|participant|tablet)[_\- ]?([1-4])(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


def _stream_name(stream: dict) -> str:
    info = stream.get("info", {})
    if isinstance(info, dict):
        name = info.get("name", [""])[0] if isinstance(info.get("name"), list) else info.get("name", "")
    else:
        name = ""
    return str(name or "").strip()


def _stream_type(stream: dict) -> str:
    info = stream.get("info", {})
    if isinstance(info, dict):
        stype = info.get("type", [""])[0] if isinstance(info.get("type"), list) else info.get("type", "")
    else:
        stype = ""
    return str(stype or "").strip()


def _classify_stream(name: str, stype: str) -> str:
    """Return a category tag for an LSL stream."""
    if _RE_TOBII.search(name):
        return "tobii"
    if _RE_EMOTIBIT.search(name):
        return "emotibit"
    if _RE_VICON.search(name):
        return "vicon"
    if _RE_AFFECTAI.search(name) or stype.lower() == "markers":
        return "markers"
    if _RE_SYNC.search(name) or stype.lower() in {"clock", "ffmpeg_progress"}:
        return "sync"
    # fallback: eye-tracker via Vicon
    if "eyetracker" in name.lower() or "tobii" in name.lower():
        return "tobii"
    return "other"


def _flatten_sample(sample: Any) -> list[str]:
    if isinstance(sample, str):
        return [sample]
    if isinstance(sample, (int, float)):
        return [str(sample)]
    if isinstance(sample, (list, tuple)):
        return [str(v) for v in sample]
    try:
        import numpy as np
        if isinstance(sample, np.ndarray):
            return [str(v) for v in sample.flat]
    except ImportError:
        pass
    return [str(sample)]


def _participant_from_text(text: str | None) -> str | None:
    if not text:
        return None
    m = _RE_PARTICIPANT_TOKEN.search(str(text).strip().lower())
    return f"P{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# XDF extraction
# ---------------------------------------------------------------------------


def _load_xdf(xdf_path: Path) -> list[dict]:
    """Load XDF file via pyxdf, raising if unavailable."""
    if importlib.util.find_spec("pyxdf") is None:
        raise RuntimeError(
            "pyxdf is not installed. Install via: pip install pyxdf"
        )
    pyxdf = __import__("pyxdf")
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    return streams


def extract_xdf_streams(
    xdf_paths: list[Path],
    return_raw: bool = False,
) -> dict[str, list[list[str]]] | tuple[dict[str, list[list[str]]], list[dict]]:
    """Extract all streams from XDF file(s) into categorized row lists.

    Returns a dict with keys: tobii, emotibit, vicon, markers, sync, other.
    Each value is a list of rows: [lsl_time, stream_name, stream_type, value_0, ...].

    If *return_raw* is True, also returns the raw pyxdf stream dicts so
    callers can use them for clock bridge computations without reloading.
    """
    categories: dict[str, list[list[str]]] = {
        "tobii": [],
        "emotibit": [],
        "vicon": [],
        "markers": [],
        "sync": [],
        "other": [],
    }
    raw_streams: list[dict] = []

    for xdf_path in xdf_paths:
        logger.info("Loading XDF: %s", xdf_path)
        try:
            streams = _load_xdf(xdf_path)
        except Exception as exc:
            logger.error("  Failed to load %s: %s", xdf_path.name, exc)
            continue
        logger.info("  Found %d streams", len(streams))
        if return_raw:
            raw_streams.extend(streams)

        for stream in streams:
            name = _stream_name(stream)
            stype = _stream_type(stream)
            cat = _classify_stream(name, stype)
            stamps = stream.get("time_stamps")
            series = stream.get("time_series")
            if stamps is None or not hasattr(stamps, '__len__') or len(stamps) == 0:
                continue
            if series is None or not hasattr(series, '__len__') or len(series) == 0:
                continue
            count = 0
            for ts, sample in zip(stamps, series, strict=False):
                flat = _flatten_sample(sample)
                categories[cat].append([f"{float(ts):.6f}", name, stype, *flat])
                count += 1
            if count:
                logger.info("    %s → %s (%d samples)", name, cat, count)

    if return_raw:
        return categories, raw_streams
    return categories


# ---------------------------------------------------------------------------
# TSV I/O
# ---------------------------------------------------------------------------


def _write_tsv(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    gzip_out: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_out else open
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def _read_lsl_rows(path: Path) -> tuple[list[str], list[list[str]], bool]:
    gz = path.suffix.lower() == ".gz"
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return [], [], gz
    return rows[0], rows[1:], gz


# ---------------------------------------------------------------------------
# Write extracted streams to categorized TSVs
# ---------------------------------------------------------------------------


def _normalise_header(rows: list[list[str]], base_cols: int = 3) -> tuple[list[str], list[list[str]]]:
    if not rows:
        return [], []
    max_cols = max(len(r) for r in rows)
    header = ["lsl_time", "stream_name", "stream_type"] + [
        f"value_{i}" for i in range(max_cols - base_cols)
    ]
    norm = [r + [""] * (max_cols - len(r)) for r in rows]
    return header, norm


def write_stream_tables(
    session_dir: Path,
    base: str,
    categories: dict[str, list[list[str]]],
) -> dict[str, Path | None]:
    """Write categorized stream rows to BIDS-organized TSV files."""
    written: dict[str, Path | None] = {}

    # Tobii → et/
    if categories["tobii"]:
        header, norm = _normalise_header(categories["tobii"])
        p = session_dir / "et" / f"{base}_acq-lsl_tobii.tsv.gz"
        _write_tsv(p, header, norm, gzip_out=True)
        written["tobii"] = p
        logger.info("  Tobii → %s (%d rows)", p.name, len(norm))
    else:
        written["tobii"] = None

    # EmotiBit → physio/
    if categories["emotibit"]:
        header, norm = _normalise_header(categories["emotibit"])
        p = session_dir / "physio" / f"{base}_acq-lsl_emotibit.tsv.gz"
        _write_tsv(p, header, norm, gzip_out=True)
        written["emotibit"] = p
        logger.info("  EmotiBit → %s (%d rows)", p.name, len(norm))
    else:
        written["emotibit"] = None

    # Vicon → eeg/ (uses eeg following existing BIDS convention in this project)
    if categories["vicon"]:
        header, norm = _normalise_header(categories["vicon"])
        p = session_dir / "eeg" / f"{base}_acq-lsl_vicon.tsv.gz"
        _write_tsv(p, header, norm, gzip_out=True)
        written["vicon"] = p
        logger.info("  Vicon → %s (%d rows)", p.name, len(norm))
    else:
        written["vicon"] = None

    # Markers/Events → beh/
    if categories["markers"]:
        header = ["lsl_time", "stream_name", "stream_type", "value"]
        rows = []
        for r in categories["markers"]:
            value = r[3] if len(r) > 3 else ""
            rows.append([r[0], r[1], r[2], value])
        p = session_dir / "beh" / f"{base}_recording-lsl_events.tsv"
        _write_tsv(p, header, rows)
        written["markers"] = p
        logger.info("  Markers → %s (%d rows)", p.name, len(rows))
    else:
        written["markers"] = None

    # Sync streams → annot/
    if categories["sync"]:
        header, norm = _normalise_header(categories["sync"])
        p = session_dir / "annot" / f"{base}_acq-lsl_sync.tsv"
        _write_tsv(p, header, norm)
        written["sync"] = p
        logger.info("  Sync → %s (%d rows)", p.name, len(norm))
    else:
        written["sync"] = None

    # Other → annot/
    if categories["other"]:
        header, norm = _normalise_header(categories["other"])
        p = session_dir / "annot" / f"{base}_acq-lsl_other.tsv"
        _write_tsv(p, header, norm)
        written["other"] = p
    else:
        written["other"] = None

    return written


# ---------------------------------------------------------------------------
# Task-window derivation (from stimuli experiment events)
# ---------------------------------------------------------------------------


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _find_experiment_events(stimuli_dir: Path) -> Path | None:
    candidates = sorted(stimuli_dir.rglob("events_*_experiment.tsv"))
    return candidates[-1] if candidates else None


def compute_task_windows(
    events_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], float | None]:
    """Derive T0–T4 task windows from stimuli experiment events.

    Returns (windows, wall_minus_lsl_offset).
    """
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    fallback_starts: dict[str, float] = {}

    def _event_type(row: dict[str, str]) -> str:
        return str(row.get("event_type", "") or "").strip().lower()

    def _phase(row: dict[str, str]) -> str:
        p = str(row.get("phase", "") or "").strip().lower()
        if p and p != "n/a":
            return p
        try:
            detail = json.loads(row.get("detail") or "{}")
            return str(detail.get("phase", "") or "").strip().lower()
        except Exception:
            return p

    def _is_t0_intro_start(row: dict[str, str]) -> bool:
        task = (row.get("task") or "").strip().upper()
        if task != "T0":
            return False
        phase = _phase(row)
        et = _event_type(row)
        if phase in {
            "welcome", "study_introduction", "vad_introduction",
            "postblock_introduction", "intro",
        }:
            return et in {"push_content", "phase_start"}
        return False

    def _is_tn_tobii_start(row: dict[str, str]) -> bool:
        task = (row.get("task") or "").strip().upper()
        if task not in {"T1", "T2", "T3", "T4"}:
            return False
        et = _event_type(row)
        phase = _phase(row)
        return et == "tobii_calibration" or (
            et in {"push_content", "phase_start"} and phase == "tobii_calibration"
        )

    def _is_task_finish(row: dict[str, str]) -> bool:
        task = (row.get("task") or "").strip().upper()
        if task not in TASK_ORDER:
            return False
        et = _event_type(row)
        phase = _phase(row)
        return et == "task_end" or (
            et in {"push_content", "phase_end"} and phase == "finish"
        )

    offsets: list[float] = []
    all_wall: list[float] = []

    for row in events_rows:
        wall = _parse_float(row.get("wall_clock", ""), default=-1.0)
        lsl = _parse_float(row.get("lsl_clock", ""), default=-1.0)
        if wall >= 0:
            all_wall.append(wall)
        if wall >= 0 and lsl >= 0:
            offsets.append(wall - lsl)
        task = (row.get("task") or "").strip().upper()
        if task in TASK_ORDER and wall >= 0 and task not in fallback_starts:
            fallback_starts[task] = wall
        if wall >= 0 and task in TASK_ORDER:
            if _is_t0_intro_start(row) or _is_tn_tobii_start(row):
                if task not in starts:
                    starts[task] = wall
            if _is_task_finish(row):
                ends[task] = wall

    if not all_wall:
        return [], None

    offset = statistics.median(offsets) if offsets else None
    windows: list[dict[str, Any]] = []

    for idx, task in enumerate(TASK_ORDER):
        start = starts.get(task) or fallback_starts.get(task)
        if start is None:
            continue
        end = ends.get(task)
        if end is None or end <= start:
            next_start = None
            for t2 in TASK_ORDER[idx + 1:]:
                t2_start = starts.get(t2) or fallback_starts.get(t2)
                if t2_start is not None:
                    next_start = t2_start
                    break
            end = next_start if next_start is not None else max(all_wall)
        if end <= start:
            continue
        start_lsl = (start - offset) if offset is not None else None
        end_lsl = (end - offset) if offset is not None else None
        windows.append({
            "task": task,
            "run": "01",
            "start_wall_clock": start,
            "end_wall_clock": end,
            "duration_s": max(0.0, end - start),
            "start_lsl": start_lsl,
            "end_lsl": end_lsl,
        })

    return windows, offset


def compute_break_windows(
    events_rows: list[dict[str, str]],
    task_windows: list[dict[str, Any]],
    offset: float | None,
) -> list[dict[str, Any]]:
    """Compute inter-task break windows (PRE, BREAK_T0_T1, ..., POST)."""
    if not task_windows:
        return []
    all_wall = sorted(
        v for v in (_parse_float(r.get("wall_clock", ""), -1.0) for r in events_rows) if v >= 0
    )
    if not all_wall:
        return []

    session_start = min(all_wall)
    session_end = max(all_wall)
    sorted_tasks = sorted(task_windows, key=lambda w: float(w["start_wall_clock"]))

    def _mk(label: str, start: float, end: float) -> dict[str, Any] | None:
        if end <= start:
            return None
        return {
            "task": label,
            "run": "01",
            "start_wall_clock": start,
            "end_wall_clock": end,
            "duration_s": max(0.0, end - start),
            "start_lsl": (start - offset) if offset is not None else None,
            "end_lsl": (end - offset) if offset is not None else None,
        }

    windows: list[dict[str, Any]] = []
    pre = _mk("PRE", session_start, float(sorted_tasks[0]["start_wall_clock"]))
    if pre:
        windows.append(pre)
    for left, right in zip(sorted_tasks, sorted_tasks[1:], strict=False):
        brk = _mk(
            f"BREAK_{left['task']}_{right['task']}",
            float(left["end_wall_clock"]),
            float(right["start_wall_clock"]),
        )
        if brk:
            windows.append(brk)
    post = _mk("POST", float(sorted_tasks[-1]["end_wall_clock"]), session_end)
    if post:
        windows.append(post)
    return windows


# ---------------------------------------------------------------------------
# Write task windows & events
# ---------------------------------------------------------------------------


def _write_windows_tsv(
    path: Path,
    windows: list[dict[str, Any]],
    offset: float | None,
) -> None:
    _write_tsv(
        path,
        ["task", "run", "start_wall_clock", "end_wall_clock",
         "duration_s", "start_lsl", "end_lsl", "wall_minus_lsl_offset"],
        [
            [
                w["task"],
                w["run"],
                f"{w['start_wall_clock']:.6f}",
                f"{w['end_wall_clock']:.6f}",
                f"{w['duration_s']:.6f}",
                "" if w["start_lsl"] is None else f"{w['start_lsl']:.6f}",
                "" if w["end_lsl"] is None else f"{w['end_lsl']:.6f}",
                "" if offset is None else f"{offset:.6f}",
            ]
            for w in windows
        ],
    )


def write_session_events(
    session_dir: Path,
    events_rows: list[dict[str, str]],
) -> Path:
    """Write BIDS events.tsv (onset relative to first event)."""
    out = session_dir / "events.tsv"
    wall_values = [
        v for v in (_parse_float(r.get("wall_clock", ""), -1.0) for r in events_rows)
        if v >= 0
    ]
    if not wall_values:
        _write_tsv(out, ["onset", "duration", "trial_type", "value", "description"], [])
        return out

    t0 = min(wall_values)
    rows = []
    for row in events_rows:
        wall = _parse_float(row.get("wall_clock", ""), -1.0)
        if wall < 0:
            continue
        onset = wall - t0
        rows.append([
            f"{onset:.6f}",
            "0.0",
            row.get("event_type", ""),
            row.get("detail", ""),
            (
                f"task={row.get('task', '')};phase={row.get('phase', '')};"
                f"stream={row.get('stream', '')};participant={row.get('participant', '')}"
            ),
        ])
    _write_tsv(out, ["onset", "duration", "trial_type", "value", "description"], rows)
    return out


def write_task_beh_events(
    session_dir: Path,
    sub_label: str,
    ses_label: str,
    events_rows: list[dict[str, str]],
    windows: list[dict[str, Any]],
) -> list[Path]:
    """Write per-task events TSV under beh/."""
    beh_dir = session_dir / "beh"
    out_files: list[Path] = []
    for w in windows:
        task = w["task"]
        start = float(w["start_wall_clock"])
        end = float(w["end_wall_clock"])
        rows = []
        for row in events_rows:
            wall = _parse_float(row.get("wall_clock", ""), -1.0)
            if wall < start or wall >= end:
                continue
            rows.append([
                f"{wall - start:.6f}",
                "0.0",
                row.get("event_type", ""),
                row.get("detail", ""),
                (
                    f"task={row.get('task', '')};phase={row.get('phase', '')};"
                    f"stream={row.get('stream', '')};participant={row.get('participant', '')}"
                ),
            ])
        out = beh_dir / f"sub-{sub_label}_ses-{ses_label}_task-{task}_run-01_events.tsv"
        _write_tsv(out, ["onset", "duration", "trial_type", "value", "description"], rows)
        out_files.append(out)
    return out_files


# ---------------------------------------------------------------------------
# Split LSL tables by task windows
# ---------------------------------------------------------------------------


def split_lsl_table_by_windows(
    input_file: Path,
    output_dir: Path,
    sub_label: str,
    ses_label: str,
    suffix: str,
    windows: list[dict[str, Any]],
) -> list[Path]:
    """Split a full-session LSL TSV into per-task files."""
    header, rows, gz = _read_lsl_rows(input_file)
    if not header or not rows:
        return []
    if "lsl_time" not in header:
        return []
    time_idx = header.index("lsl_time")
    outputs: list[Path] = []
    for w in windows:
        if w["start_lsl"] is None or w["end_lsl"] is None:
            continue
        start = float(w["start_lsl"])
        end = float(w["end_lsl"])
        keep = [
            row for row in rows
            if len(row) > time_idx and start <= _parse_float(row[time_idx], -1.0) < end
        ]
        label = str(w.get("task", "UNK"))
        out = output_dir / f"sub-{sub_label}_ses-{ses_label}_task-{label}_run-01_{suffix}"
        _write_tsv(out, header, keep, gzip_out=gz)
        outputs.append(out)
        if keep:
            logger.info("    %s → %d rows", out.name, len(keep))
    return outputs


def split_lsl_by_participant(
    input_file: Path,
    output_dir: Path,
    sub_label: str,
    ses_label: str,
    task_label: str,
    stream_prefix: str,
    acq_suffix: str,
) -> list[Path]:
    """Split a per-task LSL TSV by participant (P1–P4).

    Reads ``input_file``, filters rows where ``stream_name`` starts with
    ``stream_prefix`` followed by the participant number (e.g. ``Tobii_P1_stream``),
    and writes one output file per participant.

    Parameters
    ----------
    input_file:
        Per-task TSV/TSV.GZ from ``split_lsl_table_by_windows()``.
    output_dir:
        Target directory (e.g. ``et/`` or ``physio/``).
    sub_label / ses_label / task_label:
        BIDS entity labels for output filename.
    stream_prefix:
        Prefix used to match participant streams (e.g. ``"Tobii_P"`` or
        ``"Emotibit_P"``).  The full stream name pattern is
        ``{stream_prefix}{N}_stream`` for N in 1–4.
    acq_suffix:
        Acquisition label used in the output filename (e.g. ``"tobii"`` or
        ``"emotibit"``).

    Returns
    -------
    list of written Paths.
    """
    header, rows, gz = _read_lsl_rows(input_file)
    if not header or not rows:
        return []
    if "stream_name" not in header:
        return []
    name_idx = header.index("stream_name")
    outputs: list[Path] = []
    for n in range(1, 5):
        stream_name = f"{stream_prefix}{n}_stream"
        keep = [row for row in rows if len(row) > name_idx and row[name_idx] == stream_name]
        if not keep:
            continue
        out_name = (
            f"sub-{sub_label}_ses-{ses_label}_task-{task_label}_run-01"
            f"_acq-P{n}_{acq_suffix}"
        )
        if gz:
            out_name += ".tsv.gz"
        else:
            out_name += ".tsv"
        out = output_dir / out_name
        _write_tsv(out, header, keep, gzip_out=gz)
        outputs.append(out)
        logger.info("    %s → %d rows", out.name, len(keep))
    return outputs


# ---------------------------------------------------------------------------
# Video clock anchors (no video copying)
# ---------------------------------------------------------------------------


def extract_video_clock_anchors(
    categories: dict[str, list[list[str]]],
) -> list[dict[str, str]]:
    """Derive video/audio clock anchor rows from LSL sync streams."""
    anchors: list[dict[str, str]] = []
    if not categories.get("sync"):
        return anchors

    # Group sync samples by stream_name to estimate start times
    by_stream: dict[str, list[float]] = {}
    for row in categories["sync"]:
        if len(row) < 2:
            continue
        name = row[1]
        ts = _parse_float(row[0], -1.0)
        if ts >= 0:
            by_stream.setdefault(name, []).append(ts)

    for name, times in sorted(by_stream.items()):
        if not times:
            continue
        start = min(times)
        end = max(times)
        duration = end - start if end > start else 0.0
        anchors.append({
            "stream_name": name,
            "device": name,
            "first_lsl_time": f"{start:.6f}",
            "last_lsl_time": f"{end:.6f}",
            "duration_s": f"{duration:.6f}",
            "sample_count": str(len(times)),
        })

    return anchors


def write_video_clock_anchors(
    session_dir: Path,
    base: str,
    anchors: list[dict[str, str]],
    av_session_dir: Path | None = None,
) -> Path | None:
    """Write video/audio clock anchor TSV.

    If an AV session directory is provided, also scan for frame-log and
    progress-TSV anchors from the raw AV data.

    .. note:: DPA Audio Recording Desync

       DPA mics sharing the same RME Fireface DirectShow device are captured
       by separate ffmpeg processes, causing 50–250 ms start-time jitter.
       Post-hoc alignment uses **per-mic anchors** with equal-duration cuts
       to produce correctly aligned, equal-length clips.  See
       ``_compute_dpa_anchors()``, ``split_av_audio_by_windows()``, and
       docs/recording_sync_calibration_pipeline.md §3.3 for details.
    """
    # Also scan AV session dir for frame-log anchors if available
    if av_session_dir and av_session_dir.exists():
        for sync_dir in [av_session_dir / "sourcedata" / "sync", av_session_dir / "sync"]:
            if not sync_dir.exists():
                continue
            for tsv in sorted(sync_dir.glob("*_ffmpeg_progress.tsv")):
                try:
                    offsets = []
                    with tsv.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for row in reader:
                            try:
                                host_t = float(row.get("host_time_sec", ""))
                                out_t = float(row.get("out_time_sec", ""))
                                offsets.append(host_t - out_t)
                            except (ValueError, TypeError):
                                continue
                    if offsets:
                        start_wall = statistics.median(offsets)
                        anchors.append({
                            "stream_name": tsv.stem,
                            "device": tsv.stem.replace("_ffmpeg_progress", ""),
                            "first_lsl_time": "",
                            "last_lsl_time": "",
                            "duration_s": "",
                            "sample_count": str(len(offsets)),
                            "start_wall_clock": f"{start_wall:.6f}",
                            "anchor_source": "progress_tsv",
                        })
                except Exception:
                    continue

    if not anchors:
        return None

    out = session_dir / "annot" / f"{base}_video_clock_anchors.tsv"
    fields = sorted({k for a in anchors for k in a})
    _write_tsv(out, fields, [[a.get(f, "") for f in fields] for a in anchors])
    return out


# ---------------------------------------------------------------------------
# Participant signal map
# ---------------------------------------------------------------------------


def write_participant_signal_map(
    session_dir: Path,
    base: str,
    categories: dict[str, list[list[str]]],
    emotibit_cfg_path: Path | None = None,
) -> Path:
    """Write participant ↔ signal mapping TSV."""
    rows: list[list[str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(signal: str, participant: str | None, src_type: str, reason: str) -> None:
        key = (signal, src_type)
        if key in seen:
            return
        seen.add(key)
        rows.append([signal, participant or "", src_type, reason])

    for cat, label in [("tobii", "tobii_lsl"), ("emotibit", "emotibit_lsl"), ("markers", "markers")]:
        stream_names: set[str] = set()
        for r in categories.get(cat, []):
            if len(r) > 1:
                stream_names.add(r[1])
        for name in sorted(stream_names):
            part = _participant_from_text(name)
            _add(name, part, label, "stream_name")

    # EmotiBit config lookup
    if emotibit_cfg_path and emotibit_cfg_path.exists():
        try:
            cfg = json.loads(emotibit_cfg_path.read_text(encoding="utf-8"))
            for p, hwid in sorted((cfg.get("participants") or {}).items()):
                _add(str(hwid), f"P{p[-1]}" if p[-1].isdigit() else p, "emotibit_config", "hardware_id")
            for source, p in sorted((cfg.get("by_source") or {}).items()):
                _add(str(source), f"P{p[-1]}" if p[-1].isdigit() else p, "emotibit_config", "source_ip")
        except Exception:
            pass

    out = session_dir / "annot" / f"{base}_participant_signal_map.tsv"
    _write_tsv(out, ["signal", "participant", "source_type", "mapping_reason"], rows)
    return out


# ---------------------------------------------------------------------------
# Stimuli answer extraction
# ---------------------------------------------------------------------------


_RESPONSE_META_KEYS = {
    "device_id", "participant", "task", "phase", "type", "probe_name",
    "probe_schema", "session_id", "received_at", "server_received_lsl",
    "client_timestamp", "client_perf_ms", "clock_offset_ms", "block_id",
}


def write_stimuli_answers(
    session_dir: Path,
    sub_label: str,
    ses_label: str,
    stimuli_dir: Path,
) -> Path | None:
    """Extract stimuli tablet responses into a single TSV."""
    files = sorted(p for p in stimuli_dir.rglob("responses_*.jsonl") if p.is_file())
    if not files:
        return None

    header = [
        "wall_clock", "lsl_clock", "task", "phase", "response_type",
        "participant", "device_id", "item_key", "item_value",
    ]
    out = session_dir / "beh" / f"sub-{sub_label}_ses-{ses_label}_task-T0T1T2T3T4_stimuli_answers.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for src in files:
            with src.open("r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    wall = _parse_float(str(payload.get("received_at", "")), -1.0)
                    lsl = _parse_float(str(payload.get("server_received_lsl", "")), -1.0)
                    task = str(payload.get("task", "") or "")
                    phase = str(payload.get("phase", "") or "")
                    rtype = str(payload.get("type", "form") or "form")
                    device_id = str(payload.get("device_id", "") or "")
                    participant = _participant_from_text(
                        str(payload.get("participant", ""))
                    ) or _participant_from_text(device_id) or ""

                    answers = _extract_answer_items(payload)
                    if not answers:
                        answers = [("response", "")]

                    for key, value in answers:
                        writer.writerow([
                            "" if wall < 0 else f"{wall:.6f}",
                            "" if lsl < 0 else f"{lsl:.6f}",
                            task, phase, rtype, participant, device_id,
                            str(key), json.dumps(value) if not isinstance(value, str) else value,
                        ])
                        row_count += 1

    return out if row_count > 0 else None


def _extract_answer_items(payload: dict) -> list[tuple[str, Any]]:
    rtype = str(payload.get("type", "") or "").strip().lower()
    rows: list[tuple[str, Any]] = []
    if rtype == "vad":
        for k in ("valence", "arousal", "dominance"):
            if payload.get(k) is not None:
                rows.append((k, payload[k]))
        return rows
    if rtype == "postblock":
        resp = payload.get("responses")
        if isinstance(resp, dict):
            for k, v in sorted(resp.items()):
                rows.append((str(k), v))
        return rows
    resp = payload.get("responses")
    if isinstance(resp, dict):
        for k, v in sorted(resp.items()):
            rows.append((str(k), v))
    for k, v in payload.items():
        if k in _RESPONSE_META_KEYS or k == "responses":
            continue
        rows.append((str(k), v))
    return rows


# ---------------------------------------------------------------------------
# AV audio splitting helpers
# ---------------------------------------------------------------------------

_RE_DPA_MIC = re.compile(r"^dpa_mic(?:9|1[0-2])(?:[_\-]|$)", re.IGNORECASE)
_RE_DPA_PROGRESS = re.compile(
    r"^ffmpeg_progress_dpa_mic(?:9|1[0-2])_aud$", re.IGNORECASE
)


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple OLS linear regression.  Returns ``(slope, intercept)``."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if abs(den) < 1e-12:
        return 0.0, y_mean
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _regression_rmse(
    xs: list[float], ys: list[float], slope: float, intercept: float
) -> float:
    """Root-mean-square of linear regression residuals."""
    mse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / len(xs)
    return mse ** 0.5


def _compute_dpa_anchors_from_xdf(
    raw_streams: list[dict],
) -> dict[str, tuple[float, float, float]]:
    """Compute time-varying anchor for each DPA mic from XDF progress streams.

    Returns dict mapping label → ``(slope, intercept, rmse)``.

    The audio card's sample clock drifts relative to the LSL/XDF clock
    (measured at ~0.1 ms/s for the RME Fireface 802).  A single median
    anchor introduces 50–100 ms error for tasks far from the temporal
    centre of the recording.

    Instead we fit a **linear regression**::

        anchor(t) = slope * t + intercept

    using the per-sample corrected XDF timestamps from
    ``ffmpeg_progress_dpa_mic*_aud`` streams.  This is more accurate than
    the progress-TSV + ``ffmpeg_clock`` bridge because each sample's
    timestamp is individually clock-corrected by pyxdf.

    Returns dict mapping label (e.g. ``dpa_mic9_aud``) →
    ``(slope, intercept)``.
    """
    anchors: dict[str, tuple[float, float]] = {}
    for s in raw_streams:
        info = s.get("info", {})
        name = (
            info.get("name", [""])[0]
            if isinstance(info.get("name"), list)
            else info.get("name", "")
        )
        if not _RE_DPA_PROGRESS.match(name.strip()):
            continue
        ts = s.get("time_stamps")
        series = s.get("time_series")
        if ts is None or not hasattr(ts, "__len__") or len(ts) < 10:
            continue
        label = name.strip().replace("ffmpeg_progress_", "")
        diffs = [float(ts[i]) - float(series[i][0]) for i in range(len(ts))]
        xdf_times = [float(ts[i]) for i in range(len(ts))]

        if len(diffs) < 20:
            med = statistics.median(diffs)
            rmse = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            anchors[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xdf_times, diffs)
            rmse = _regression_rmse(xdf_times, diffs, slope, intercept)
            anchors[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (XDF): linreg slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(diffs),
            )
    return anchors


def _compute_dpa_anchors_from_jsonl(
    capture_dir: Path,
    wall_minus_xdf_lsl: float,
) -> dict[str, tuple[float, float, float]]:
    """Compute time-varying anchor for each DPA mic from JSONL progress files.

    Fallback for sessions where the AV XDF is missing or corrupted but raw
    JSONL logs from the LSL recorder are available under ``<capture_dir>/lsl/``.

    Each JSONL line::

        {"stream_time": <av_lsl_t>, "received_time": "<ISO wall>", "values": [out_time_sec, ...]}

    ``stream_time`` is the **AV PC's** local LSL clock — a different epoch than
    the unified EEG XDF clock.  Instead we use ``received_time`` (the wall
    clock of the LSL Lab Recorder when the sample arrived) and convert it to
    unified XDF time::

        xdf_t = received_time_epoch - wall_minus_xdf_lsl

    This aligns with the same clock used in ``split_av_audio_by_windows``.

    Returns dict mapping label (e.g. ``dpa_mic9_aud``) →
    ``(slope, intercept, rmse)``.
    """
    from datetime import datetime, timezone

    lsl_dir = capture_dir / "lsl"
    if not lsl_dir.exists():
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for jsonl_path in sorted(lsl_dir.glob("ffmpeg_progress_dpa_*.jsonl")):
        label = jsonl_path.stem.replace("ffmpeg_progress_", "")
        if not _RE_DPA_MIC.match(label):
            continue
        xs: list[float] = []
        ys: list[float] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    received_str = rec["received_time"]
                    out_t = float(rec["values"][0])
                    # Parse ISO wall-clock → UTC epoch seconds
                    dt = datetime.fromisoformat(received_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    wall_epoch = dt.timestamp()
                    xdf_t = wall_epoch - wall_minus_xdf_lsl
                except (KeyError, IndexError, ValueError,
                        json.JSONDecodeError, AttributeError):
                    continue
                xs.append(xdf_t)
                ys.append(xdf_t - out_t)
        if not xs:
            continue
        if len(xs) < 20:
            med = statistics.median(ys)
            rmse = statistics.stdev(ys) if len(ys) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xs, ys)
            rmse = _regression_rmse(xs, ys, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (JSONL): linreg slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xs),
            )
    return result


def _compute_av_to_xdf_offset(
    streams: list[dict],
) -> float | None:
    """Compute offset to convert AV PC local_clock to XDF-unified LSL time.

    Uses the ``ffmpeg_clock`` stream present in XDF recordings that bridges
    the AV PC's ``local_clock()`` to the unified (clock-corrected) LSL
    timeline.

    Returns *av_to_xdf_offset* such that::

        xdf_lsl = av_local_clock + av_to_xdf_offset
    """
    for s in streams:
        info = s.get("info", {})
        name = (
            info.get("name", [""])[0]
            if isinstance(info.get("name"), list)
            else info.get("name", "")
        )
        if str(name).strip() != "ffmpeg_clock":
            continue
        ts = s.get("time_stamps")
        series = s.get("time_series")
        if ts is None or not hasattr(ts, "__len__") or len(ts) < 2:
            continue
        offsets = [float(ts[i]) - float(series[i][0]) for i in range(len(ts))]
        return statistics.median(offsets)
    return None


def _find_av_capture_dir(av_session_dir: Path) -> Path | None:
    """Find the main capture subdirectory containing audio/ and sourcedata/sync/.

    The AV session layout is::

        ses-.../sourcedata/<capture_name>/audio/*.wav
        ses-.../sourcedata/<capture_name>/sourcedata/sync/*_ffmpeg_progress.tsv

    We skip calibration subdirectories and pick the capture with the largest
    WAV files (= the actual session, not a short test).
    """
    sd = av_session_dir / "sourcedata"
    if not sd.exists():
        return None
    best: Path | None = None
    best_size = 0
    for child in sd.iterdir():
        if not child.is_dir():
            continue
        # Skip calibration captures and AV subdirectories
        if "calibration" in child.name.lower() or child.name == "av":
            continue
        audio_dir = child / "audio"
        if not audio_dir.exists():
            continue
        total = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
        if total > best_size:
            best_size = total
            best = child
    return best


def _compute_dpa_anchors(
    capture_dir: Path,
    av_to_xdf_offset: float,
) -> dict[str, tuple[float, float]]:
    """Compute anchor regression for each DPA mic from progress TSV.

    This is the **fallback** path used when XDF ``ffmpeg_progress_*``
    streams are unavailable.  It uses the ``ffmpeg_clock`` bridge
    (``av_to_xdf_offset``) to convert host_time_sec → corrected XDF time,
    then fits a linear regression of ``(corrected_xdf_t − out_time)`` over
    time to capture audio-clock drift.

    Returns dict mapping label → ``(slope, intercept)`` where
    ``anchor(t) = slope * t + intercept``.
    """
    sync_dir = capture_dir / "sourcedata" / "sync"
    if not sync_dir.exists():
        return {}

    # Collect per-mic (xdf_time, anchor_diff) pairs for regression
    per_mic_data: dict[str, tuple[list[float], list[float]]] = {}
    any_data = False
    for tsv in sorted(sync_dir.glob("*_ffmpeg_progress.tsv")):
        label = tsv.stem.replace("_ffmpeg_progress", "")
        if not _RE_DPA_MIC.match(label):
            continue
        xs: list[float] = []
        ys: list[float] = []
        with tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    host_t = float(row["host_time_sec"])
                    out_t = float(row["out_time_sec"])
                except (ValueError, TypeError, KeyError):
                    continue
                xdf_t = host_t + av_to_xdf_offset
                xs.append(xdf_t)
                ys.append(xdf_t - out_t)
        if xs:
            per_mic_data[label] = (xs, ys)
            any_data = True

    if not any_data:
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for label, (xs, ys) in per_mic_data.items():
        if len(xs) < 20:
            med = statistics.median(ys)
            rmse = statistics.stdev(ys) if len(ys) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xs, ys)
            rmse = _regression_rmse(xs, ys, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (TSV): linreg slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xs),
            )
    return result


def _select_best_dpa_anchors(
    xdf_anchors: dict[str, tuple[float, float, float]],
    tsv_anchors: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float]]:
    """Per-mic: pick the anchor regression with the lower residual RMSE.

    XDF streams have per-sample pyxdf clock correction so usually win.
    TSV + ffmpeg_clock bridge applies a global median offset, inflating
    residuals by the bridge stdev (~0.4 s for grp-12).  When both are
    available the better fit is chosen empirically rather than assumed.

    Returns dict mapping label → ``(slope, intercept)``.
    """
    result: dict[str, tuple[float, float]] = {}
    all_labels = sorted(set(xdf_anchors) | set(tsv_anchors))
    for label in all_labels:
        has_xdf = label in xdf_anchors
        has_tsv = label in tsv_anchors
        if has_xdf and has_tsv:
            xdf_slope, xdf_int, xdf_rmse = xdf_anchors[label]
            tsv_slope, tsv_int, tsv_rmse = tsv_anchors[label]
            if xdf_rmse <= tsv_rmse:
                result[label] = (xdf_slope, xdf_int)
                logger.info(
                    "    %s anchor: XDF wins  (xdf_rmse=%.6f s  tsv_rmse=%.6f s)",
                    label, xdf_rmse, tsv_rmse,
                )
            else:
                result[label] = (tsv_slope, tsv_int)
                logger.info(
                    "    %s anchor: TSV wins  (tsv_rmse=%.6f s  xdf_rmse=%.6f s)",
                    label, tsv_rmse, xdf_rmse,
                )
        elif has_xdf:
            xdf_slope, xdf_int, xdf_rmse = xdf_anchors[label]
            result[label] = (xdf_slope, xdf_int)
            logger.info("    %s anchor: XDF only (rmse=%.6f s)", label, xdf_rmse)
        else:
            tsv_slope, tsv_int, tsv_rmse = tsv_anchors[label]
            result[label] = (tsv_slope, tsv_int)
            logger.info("    %s anchor: TSV only (rmse=%.6f s)", label, tsv_rmse)
    return result


def split_av_audio_by_windows(
    capture_dir: Path,
    output_dir: Path,
    sub_label: str,
    ses_label: str,
    windows: list[dict[str, Any]],
    dpa_anchors: dict[str, tuple[float, float]],
    wall_minus_xdf_lsl: float,
) -> list[Path]:
    """Crop DPA mic WAV files into per-task segments using ffmpeg.

    Uses **time-varying anchors** (linear regression) to compensate for
    audio-clock drift relative to the XDF/LSL clock.  Each mic's
    ``anchor(t) = slope * t + intercept`` maps XDF time *t* to the media
    time offset at that instant.

    ffmpeg is invoked with ``-ss`` **after** ``-i`` and re-encodes to
    ``pcm_s16le`` (lossless PCM→PCM) so that seeking is sample-accurate.
    The original sample rate and bit depth are preserved.

    Parameters
    ----------
    capture_dir : Path
        AV capture subdirectory containing ``audio/*.wav``.
    output_dir : Path
        BIDS session output directory (files go under ``audio/``).
    sub_label, ses_label : str
        BIDS subject / session labels.
    windows : list
        Task windows with ``start_wall_clock`` and ``end_wall_clock``.
    dpa_anchors : dict
        Device label -> ``(slope, intercept)`` anchor regression.
    wall_minus_xdf_lsl : float
        Offset: ``stimuli_wall_clock - xdf_lsl``.

    Returns
    -------
    list[Path]
        Paths of written WAV files.
    """
    audio_dir = capture_dir / "audio"
    if not audio_dir.exists():
        return []

    out_audio = output_dir / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Resolve which labels actually have WAV files
    active_labels = []
    for label in sorted(dpa_anchors):
        wav_path = audio_dir / f"{label}.wav"
        if wav_path.exists():
            active_labels.append(label)
        else:
            logger.warning("    WAV not found: %s", wav_path)

    if not active_labels:
        return []

    # Probe original sample rate once (all DPA WAVs use same rate)
    sample_rate: int | None = None
    probe_path = audio_dir / f"{active_labels[0]}.wav"
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,sample_fmt",
             "-of", "csv=p=0", str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        parts = probe.stdout.strip().split(",")
        if parts:
            sample_rate = int(parts[0])
    except Exception:
        pass

    def _anchor_at(label: str, xdf_t: float) -> float:
        """Evaluate linear anchor at XDF time *t*."""
        slope, intercept = dpa_anchors[label]
        return slope * xdf_t + intercept

    def _run_ffmpeg(wav_path: Path, out_path: Path, ss: float, duration: float,
                    label: str, task: str) -> bool:
        """Run ffmpeg with -ss after -i for sample-accurate seeking."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-ss", f"{ss:.6f}",
            "-t", f"{duration:.6f}",
            "-c:a", "pcm_s16le",
        ]
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        cmd.append(str(out_path))
        logger.info("    %s %s: -ss %.6f -t %.6f", label, task, ss, duration)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "    ffmpeg failed for %s %s: %s",
                label, task, exc.stderr.decode(errors="replace")[:200],
            )
        except FileNotFoundError:
            logger.error("    ffmpeg not found on PATH")
        return False

    for w in windows:
        task = w["task"]
        xdf_start = w["start_wall_clock"] - wall_minus_xdf_lsl
        xdf_end = w["end_wall_clock"] - wall_minus_xdf_lsl
        task_duration = xdf_end - xdf_start

        # Step 1: compute raw media_start for each mic using anchor at
        # the task's start time (compensates for clock drift).
        raw_media_starts: dict[str, float] = {}
        for label in active_labels:
            anchor = _anchor_at(label, xdf_start)
            raw_media_starts[label] = xdf_start - anchor

        # Step 2: handle negative media_start (task before recording start)
        min_media_start = min(raw_media_starts.values())
        if min_media_start < 0:
            max_media_start = max(raw_media_starts.values())
            if max_media_start < 0:
                # ALL mics started after the task began
                common_duration = task_duration + max_media_start
                logger.warning(
                    "    %s: all mics start before recording (worst=%.3fs), "
                    "trimming to t=0, duration=%.3f",
                    task, min_media_start, common_duration,
                )
                for label in active_labels:
                    if common_duration <= 0:
                        logger.warning("    %s %s: duration <= 0, skipping", label, task)
                        continue
                    wav_path = audio_dir / f"{label}.wav"
                    out_name = (
                        f"sub-{sub_label}_ses-{ses_label}"
                        f"_task-{task}_run-01_acq-{label}.wav"
                    )
                    out_path = out_audio / out_name
                    if _run_ffmpeg(wav_path, out_path, 0.0, common_duration, label, task):
                        outputs.append(out_path)
                continue

            # Mixed: trim to shortest usable duration
            common_duration = task_duration + min_media_start
            for label in active_labels:
                ms = raw_media_starts[label]
                effective_start = max(0.0, ms)
                if common_duration <= 0:
                    logger.warning("    %s %s: duration <= 0, skipping", label, task)
                    continue
                wav_path = audio_dir / f"{label}.wav"
                out_name = (
                    f"sub-{sub_label}_ses-{ses_label}"
                    f"_task-{task}_run-01_acq-{label}.wav"
                )
                out_path = out_audio / out_name
                if _run_ffmpeg(wav_path, out_path, effective_start, common_duration, label, task):
                    outputs.append(out_path)
            continue

        # Step 3: normal case — per-mic -ss with common -t.
        for label in active_labels:
            media_start = raw_media_starts[label]
            wav_path = audio_dir / f"{label}.wav"
            out_name = (
                f"sub-{sub_label}_ses-{ses_label}"
                f"_task-{task}_run-01_acq-{label}.wav"
            )
            out_path = out_audio / out_name
            if _run_ffmpeg(wav_path, out_path, media_start, task_duration, label, task):
                outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------------
# Tobii world video splitting per participant / task
# ---------------------------------------------------------------------------

_FOLDER_SIZE_RE = re.compile(r"^(.*?)\(\d+MB\)$")


def _load_tobii_video_map(
    metadata_root: Path,
    session_id: str,
) -> dict[str, list[str]]:
    """Load per-participant Tobii video folder names from session_metadata_report.tsv.

    Returns a dict mapping ``"P1"``–``"P4"`` to a list of folder names
    (MB annotations stripped).  Participants with no entry are omitted.
    """
    tsv_path = metadata_root / "session_metadata_report.tsv"
    if not tsv_path.exists():
        return {}
    with tsv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("session") != session_id:
                continue
            result: dict[str, list[str]] = {}
            for n in range(1, 5):
                raw = row.get(f"tobii_video_P{n}", "").strip()
                if not raw:
                    continue
                folders: list[str] = []
                for part in raw.split(";"):
                    part = part.strip()
                    m = _FOLDER_SIZE_RE.match(part)
                    folders.append(m.group(1).strip() if m else part)
                result[f"P{n}"] = [f for f in folders if f]
            return result
    return {}


def _tobii_recording_covers_task(
    recording_start_xdf: float,
    recording_duration_s: float,
    task_start_xdf: float,
    task_end_xdf: float,
) -> bool:
    """Return True if the Tobii recording window overlaps the task window."""
    rec_end = recording_start_xdf + recording_duration_s
    return recording_start_xdf < task_end_xdf and rec_end > task_start_xdf


def _tobii_overlap_duration(
    recording_start_xdf: float,
    recording_duration_s: float,
    window_start_xdf: float,
    window_end_xdf: float,
) -> float:
    """Return overlap duration in seconds between recording and target window."""
    rec_end = recording_start_xdf + recording_duration_s
    overlap_start = max(recording_start_xdf, window_start_xdf)
    overlap_end = min(rec_end, window_end_xdf)
    return max(0.0, overlap_end - overlap_start)


def _normalize_tobii_recording_start(
    recording_start_xdf: float,
    recording_duration_s: float,
    session_start_xdf: float,
    session_end_xdf: float,
    max_hour_shift: int = 14,
) -> tuple[float, int]:
    """Apply a whole-hour correction maximizing overlap with the session window."""
    best_start = recording_start_xdf
    best_shift_h = 0
    best_overlap = _tobii_overlap_duration(
        recording_start_xdf,
        recording_duration_s,
        session_start_xdf,
        session_end_xdf,
    )
    for shift_h in range(-max_hour_shift, max_hour_shift + 1):
        candidate_start = recording_start_xdf + shift_h * 3600.0
        overlap = _tobii_overlap_duration(
            candidate_start,
            recording_duration_s,
            session_start_xdf,
            session_end_xdf,
        )
        if overlap > best_overlap or (
            abs(overlap - best_overlap) <= 1e-6 and abs(shift_h) < abs(best_shift_h)
        ):
            best_start = candidate_start
            best_shift_h = shift_h
            best_overlap = overlap
    return best_start, best_shift_h


def _load_task_lsl_gaze_samples(
    et_dir: Path,
    sub_label: str,
    ses_label: str,
    task: str,
    run: str,
    participant: str,
    max_rows: int = 400,
) -> list[tuple[float, float, float]]:
    """Load early per-task Tobii LSL gaze samples as (lsl_time, gaze_x, gaze_y)."""
    path = (
        et_dir
        / f"sub-{sub_label}_ses-{ses_label}_task-{task}_run-{run}_acq-{participant}_tobii.tsv.gz"
    )
    if not path.exists():
        return []
    out: list[tuple[float, float, float]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    lsl_t = float(row.get("lsl_time", ""))
                    gx = float(row.get("value_0", ""))
                    gy = float(row.get("value_1", ""))
                except Exception:
                    continue
                # Keep only valid normalized 2D gaze.
                if not (0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0):
                    continue
                out.append((lsl_t, gx, gy))
                if len(out) >= max_rows:
                    break
    except Exception:
        return []
    return out


def _load_raw_gaze_samples_from_recording(
    recording_dir: Path,
    max_rows: int = 120000,
) -> list[tuple[float, float, float]]:
    """Load Tobii raw gaze samples as (recording_time_s, gaze_x, gaze_y)."""
    gz_path = recording_dir / "gazedata.gz"
    if not gz_path.exists():
        return []
    out: list[tuple[float, float, float]] = []
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") != "gaze":
                        continue
                    ts = float(rec.get("timestamp", "nan"))
                    g2d = rec.get("data", {}).get("gaze2d", [])
                    gx = float(g2d[0])
                    gy = float(g2d[1])
                except Exception:
                    continue
                if ts < 0:
                    continue
                if not (0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0):
                    continue
                out.append((ts, gx, gy))
                if len(out) >= max_rows:
                    break
    except Exception:
        return []
    return out


def _estimate_tobii_start_delta_from_gaze(
    recording_start_xdf: float,
    raw_gaze: list[tuple[float, float, float]],
    lsl_gaze: list[tuple[float, float, float]],
    search_s: float = 3.0,
    step_s: float = 0.02,
) -> tuple[float, float] | None:
    """Estimate start-time correction from gaze matching.

    Returns ``(delta_s, rmse)`` where updated start is ``recording_start_xdf + delta_s``.
    """
    if len(raw_gaze) < 20 or len(lsl_gaze) < 20 or step_s <= 0:
        return None
    raw_t = [r[0] for r in raw_gaze]
    raw_x = [r[1] for r in raw_gaze]
    raw_y = [r[2] for r in raw_gaze]

    # Use a short early window to keep matching fast and local.
    lsl_win = lsl_gaze[: min(len(lsl_gaze), 220)]
    deltas = int(search_s / step_s)
    best_delta = 0.0
    best_rmse = float("inf")

    for k in range(-deltas, deltas + 1):
        delta = k * step_s
        err = 0.0
        n = 0
        for lsl_t, gx, gy in lsl_win:
            # Candidate raw timestamp under this start correction.
            rt = lsl_t - (recording_start_xdf + delta)
            idx = bisect.bisect_left(raw_t, rt)
            if idx <= 0:
                j = 0
            elif idx >= len(raw_t):
                j = len(raw_t) - 1
            else:
                j = idx if abs(raw_t[idx] - rt) < abs(raw_t[idx - 1] - rt) else idx - 1
            dx = raw_x[j] - gx
            dy = raw_y[j] - gy
            err += dx * dx + dy * dy
            n += 1
        if n < 10:
            continue
        rmse = (err / float(n)) ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_delta = delta

    if best_rmse == float("inf"):
        return None
    return best_delta, best_rmse


def _mean_abs_envelope_from_pcm16(
    pcm: array.array,
    sample_rate: int,
    bucket_hz: int = 20,
) -> list[float]:
    """Convert mono PCM16 to mean-absolute amplitude envelope."""
    if sample_rate <= 0 or bucket_hz <= 0 or len(pcm) == 0:
        return []
    bucket = max(1, int(sample_rate / bucket_hz))
    out: list[float] = []
    i = 0
    n = len(pcm)
    while i < n:
        j = min(n, i + bucket)
        s = 0.0
        for k in range(i, j):
            s += abs(float(pcm[k]))
        out.append(s / float(j - i))
        i = j
    return out


def _load_wav_envelope(path: Path, max_seconds: float = 90.0, bucket_hz: int = 20) -> list[float]:
    """Load a low-rate audio envelope from WAV for coarse lag estimation."""
    if not path.exists():
        return []
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            if sr <= 0 or nch <= 0 or sw != 2:
                return []
            max_frames = int(max_seconds * sr) if max_seconds > 0 else wf.getnframes()
            raw = wf.readframes(max_frames)
            pcm = array.array("h")
            pcm.frombytes(raw)
            if nch > 1:
                mono = array.array("h")
                for i in range(0, len(pcm), nch):
                    mono.append(pcm[i])
                pcm = mono
            return _mean_abs_envelope_from_pcm16(pcm, sr, bucket_hz=bucket_hz)
    except Exception:
        return []


def _load_video_audio_envelope(
    ffmpeg_bin: str,
    video_path: Path,
    media_start_s: float,
    max_seconds: float = 90.0,
    sample_rate: int = 8000,
    bucket_hz: int = 20,
) -> list[float]:
    """Extract video audio envelope via ffmpeg pipe for coarse lag estimation."""
    if not video_path.exists():
        return []
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-ss",
        f"{max(0.0, media_start_s):.6f}",
        "-t",
        f"{max_seconds:.6f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            return []
        pcm = array.array("h")
        pcm.frombytes(proc.stdout)
        return _mean_abs_envelope_from_pcm16(pcm, sample_rate, bucket_hz=bucket_hz)
    except Exception:
        return []


def _estimate_audio_lag_seconds(
    ref_env: list[float],
    cand_env: list[float],
    bucket_hz: int = 20,
    max_lag_s: float = 2.0,
) -> tuple[float, float] | None:
    """Estimate lag using envelope cross-correlation.

    Returns ``(lag_seconds, corr_score)`` where positive lag means
    candidate audio is delayed relative to reference.
    """
    n = min(len(ref_env), len(cand_env))
    if n < 40 or bucket_hz <= 0:
        return None
    r = ref_env[:n]
    c = cand_env[:n]
    # Normalize to zero-mean to reduce gain dependence.
    mr = sum(r) / n
    mc = sum(c) / n
    r = [x - mr for x in r]
    c = [x - mc for x in c]
    max_lag = int(max_lag_s * bucket_hz)
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag, max_lag + 1):
        sxy = 0.0
        sxx = 0.0
        syy = 0.0
        m = 0
        for i in range(n):
            j = i + lag
            if j < 0 or j >= n:
                continue
            x = r[i]
            y = c[j]
            sxy += x * y
            sxx += x * x
            syy += y * y
            m += 1
        if m < 20 or sxx <= 0 or syy <= 0:
            continue
        score = sxy / ((sxx * syy) ** 0.5)
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_score < -0.5:
        return None
    return best_lag / float(bucket_hz), best_score


def split_tobii_video_by_task(
    tobii_root: Path,
    tobii_video_map: dict[str, list[str]],
    session_dir: Path,
    sub_label: str,
    ses_label: str,
    windows: list[dict[str, Any]],
    wall_minus_xdf_lsl: float,
) -> list[Path]:
    """Clip Tobii world video (scenevideo.mp4) per participant per task.

    For each participant (P1–P4) and each task window the function:

    1. Iterates the ``tobii_video_map`` folder list for that participant.
    2. Reads ``recording.g3`` to obtain ``created`` (wall-clock) and
       ``duration`` (seconds).
    3. Converts ``created`` to XDF time via ``wall_minus_xdf_lsl``.
    4. Checks whether the recording covers the task window.
    5. Runs ffmpeg ``-ss <offset> -t <duration> -c copy`` to clip.

    Output is written to ``<session_dir>/et/`` as::

        sub-{sub}_ses-{ses}_task-{task}_run-01_acq-P{N}_tobii.mp4

    Parameters
    ----------
    tobii_root:
        Parent directory containing all Tobii recording folders.
    tobii_video_map:
        Mapping from ``"P1"``–``"P4"`` to list of recording folder names,
        as returned by ``_load_tobii_video_map()``.
    session_dir:
        Session output directory (``processed_data/sub-01/ses-…/``).
    sub_label / ses_label:
        BIDS entities.
    windows:
        Task windows list from ``compute_task_windows()``.  Each entry has
        ``start_wall_clock`` and ``end_wall_clock`` keys.
    wall_minus_xdf_lsl:
        Offset so that ``xdf_t = wall_epoch - wall_minus_xdf_lsl``.

    Returns
    -------
    List of Paths of written mp4 clips.
    """
    if not tobii_root.exists():
        logger.warning("  Tobii root not found: %s", tobii_root)
        return []
    if not windows:
        return []

    et_dir = session_dir / "et"
    audio_dir = session_dir / "audio"
    et_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    session_start_xdf = min(float(w["start_wall_clock"]) for w in windows) - wall_minus_xdf_lsl
    session_end_xdf = max(float(w["end_wall_clock"]) for w in windows) - wall_minus_xdf_lsl
    # Default seat-to-DPA mapping used in this dataset.
    dpa_by_participant = {
        "P1": "dpa_mic9_aud",
        "P2": "dpa_mic10_aud",
        "P3": "dpa_mic11_aud",
        "P4": "dpa_mic12_aud",
    }

    for participant, folders in sorted(tobii_video_map.items()):
        # Pre-load recording.g3 metadata for all known folders
        recordings: list[tuple[float, float, Path, str, int]] = []
        # (xdf_start, duration, video_path, folder_name, applied_hour_shift)
        for folder_name in folders:
            folder = tobii_root / folder_name
            g3 = folder / "recording.g3"
            video = folder / "scenevideo.mp4"
            if not g3.exists() or not video.exists():
                logger.debug("    %s/%s: missing recording.g3 or scenevideo.mp4", participant, folder_name)
                continue
            try:
                meta = json.loads(g3.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("    %s/%s: cannot read recording.g3 — %s", participant, folder_name, exc)
                continue
            created_iso = meta.get("created", "")
            duration_s = float(meta.get("duration", 0.0))
            if not created_iso or duration_s <= 0:
                continue
            # Convert created UTC ISO → XDF time
            wall_epoch = datetime.fromisoformat(
                created_iso.replace("Z", "+00:00")
            ).timestamp()
            xdf_start = wall_epoch - wall_minus_xdf_lsl
            normalized_start, shift_h = _normalize_tobii_recording_start(
                recording_start_xdf=xdf_start,
                recording_duration_s=duration_s,
                session_start_xdf=session_start_xdf,
                session_end_xdf=session_end_xdf,
            )
            recordings.append((normalized_start, duration_s, video, folder_name, shift_h))
            logger.debug(
                "    %s/%s: xdf_start=%.1f duration=%.1fs shift=%+dh",
                participant, folder_name, normalized_start, duration_s, shift_h,
            )

        if not recordings:
            logger.warning("  %s: no valid Tobii recordings found", participant)
            continue

        raw_gaze_cache: dict[str, list[tuple[float, float, float]]] = {}
        task_lsl_gaze_cache: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        dpa_env_cache: dict[tuple[str, str], list[float]] = {}

        for w in windows:
            task = str(w.get("task", "UNK"))
            run = str(w.get("run", "01"))
            task_start_xdf = float(w["start_wall_clock"]) - wall_minus_xdf_lsl
            task_end_xdf = float(w["end_wall_clock"]) - wall_minus_xdf_lsl
            task_duration = task_end_xdf - task_start_xdf

            # Select recording with the largest overlap for this task window.
            matched: tuple[float, float, Path, str, int] | None = None
            best_overlap = 0.0
            for rec_start, rec_dur, video, folder_name, shift_h in recordings:
                overlap = _tobii_overlap_duration(rec_start, rec_dur, task_start_xdf, task_end_xdf)
                if overlap <= 0:
                    continue
                if (
                    matched is None
                    or overlap > best_overlap
                    or (abs(overlap - best_overlap) <= 1e-6 and abs(shift_h) < abs(matched[4]))
                ):
                    matched = (rec_start, rec_dur, video, folder_name, shift_h)
                    best_overlap = overlap

            if matched is None:
                logger.warning(
                    "  %s %s: no Tobii recording covers task window (xdf %.1f–%.1f)",
                    participant, task, task_start_xdf, task_end_xdf,
                )
                continue

            rec_start, rec_dur, video, folder_name, shift_h = matched
            refined_from_gaze = False
            # Refine recording start using shared gaze (raw Tobii file <-> LSL task file).
            cache_key = (task, run)
            if cache_key not in task_lsl_gaze_cache:
                task_lsl_gaze_cache[cache_key] = _load_task_lsl_gaze_samples(
                    et_dir=et_dir,
                    sub_label=sub_label,
                    ses_label=ses_label,
                    task=task,
                    run=run,
                    participant=participant,
                )
            lsl_gaze = task_lsl_gaze_cache[cache_key]
            if lsl_gaze:
                raw_key = str(video.parent)
                if raw_key not in raw_gaze_cache:
                    raw_gaze_cache[raw_key] = _load_raw_gaze_samples_from_recording(video.parent)
                raw_gaze = raw_gaze_cache[raw_key]
                est = _estimate_tobii_start_delta_from_gaze(rec_start, raw_gaze, lsl_gaze)
                if est is not None:
                    delta_s, rmse = est
                    # Apply only conservative corrections with strong fit.
                    if abs(delta_s) <= 3.0 and rmse <= 0.08:
                        rec_start += delta_s
                        refined_from_gaze = True
                        logger.info(
                            "  %s %s: gaze-refined Tobii start by %.3fs (rmse=%.4f, src=%s, shift=%+dh)",
                            participant, task, delta_s, rmse, folder_name, shift_h,
                        )
                    else:
                        logger.debug(
                            "  %s %s: rejected gaze refinement delta=%.3fs rmse=%.4f (src=%s)",
                            participant, task, delta_s, rmse, folder_name,
                        )

            # Secondary mediator: match Tobii embedded audio to participant DPA audio.
            if not refined_from_gaze and audio_dir.exists():
                dpa_label = dpa_by_participant.get(participant, "")
                dpa_path = (
                    audio_dir
                    / f"sub-{sub_label}_ses-{ses_label}_task-{task}_run-{run}_acq-{dpa_label}.wav"
                )
                if dpa_label and dpa_path.exists():
                    if cache_key not in dpa_env_cache:
                        dpa_env_cache[cache_key] = _load_wav_envelope(dpa_path, max_seconds=90.0, bucket_hz=20)
                    dpa_env = dpa_env_cache.get(cache_key, [])
                    media_start_rough = task_start_xdf - rec_start
                    tobii_env = _load_video_audio_envelope(
                        "ffmpeg",
                        video,
                        media_start_s=max(0.0, media_start_rough),
                        max_seconds=90.0,
                        sample_rate=8000,
                        bucket_hz=20,
                    )
                    lag_est = _estimate_audio_lag_seconds(dpa_env, tobii_env, bucket_hz=20, max_lag_s=2.0)
                    if lag_est is not None:
                        lag_s, corr = lag_est
                        if abs(lag_s) <= 2.0 and corr >= 0.20:
                            rec_start += lag_s
                            logger.info(
                                "  %s %s: audio-refined Tobii start by %.3fs (corr=%.3f, dpa=%s)",
                                participant, task, lag_s, corr, dpa_label,
                            )
            media_start = task_start_xdf - rec_start
            # Clamp: recording may start slightly after task begins
            if media_start < 0:
                clip_duration = task_duration + media_start
                media_start = 0.0
            else:
                clip_duration = task_duration

            # Clamp to recording length
            clip_duration = min(clip_duration, rec_dur - media_start)
            if clip_duration <= 0:
                logger.warning("  %s %s: clip duration <= 0, skipping", participant, task)
                continue

            n = participant[1]  # "P1" → "1"
            out_name = (
                f"sub-{sub_label}_ses-{ses_label}"
                f"_task-{task}_run-01_acq-P{n}_tobii.mp4"
            )
            out_path = et_dir / out_name

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video),
                "-ss", f"{media_start:.6f}",
                "-t", f"{clip_duration:.6f}",
                "-c", "copy",
                str(out_path),
            ]
            logger.info(
                "  Tobii video %s %s: offset=%.3fs  duration=%.3fs → %s",
                participant, task, media_start, clip_duration, out_path.name,
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                logger.error(
                    "  ffmpeg failed for %s %s: %s",
                    participant, task, proc.stderr[-300:],
                )
            elif out_path.exists():
                outputs.append(out_path)
            else:
                logger.error(
                    "  ffmpeg produced no output for %s %s",
                    participant, task,
                )

    return outputs


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def _load_inventory(data_root: Path) -> dict[str, Any]:
    inv_path = data_root / "high_level_data_inventory.json"
    if not inv_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inv_path}")
    return json.loads(inv_path.read_text(encoding="utf-8"))


def _find_xdf_files(data_root: Path, group_id: str, session_id: str = "") -> list[Path]:
    """Locate XDF files for a group in CurrentStudy or recording-session dirs."""
    found: list[Path] = []

    # CurrentStudy: sub-grp-{NN}/ses-S001/eeg/*.xdf
    cs_root = data_root / "CurrentStudy"
    if cs_root.exists():
        # Try exact match then fuzzy
        grp_num = group_id.replace("grp-", "")
        candidates = [
            cs_root / f"sub-{group_id}",
            cs_root / f"sub-grp-{grp_num}",
        ]
        for cand in candidates:
            if cand.exists():
                found.extend(sorted(cand.rglob("*.xdf")))
        if not found:
            # Fuzzy match on group number
            for d in cs_root.iterdir():
                if d.is_dir() and grp_num in d.name:
                    found.extend(sorted(d.rglob("*.xdf")))
        if not found and session_id:
            # Date-based fallback: extract YYYYMMDD from session_id and scan all
            # CurrentStudy folders for XDF files whose names or paths contain it.
            # Handles non-standard subject labels like sub-P001.
            date_str = ""
            m = re.search(r"ses-(\d{8})", session_id)
            if m:
                date_str = m.group(1)
            if date_str:
                for d in cs_root.iterdir():
                    if not d.is_dir():
                        continue
                    for xdf in d.rglob("*.xdf"):
                        if date_str in xdf.name or date_str in xdf.parent.name:
                            found.append(xdf)
                if found:
                    logger.info(
                        "  XDF: group-name scan empty; found %d file(s) via date %s",
                        len(found), date_str,
                    )

    # Also check recording sessions
    rec_root = data_root / "affectai-capture-recording" / "sessions"
    if rec_root.exists():
        for xdf in rec_root.rglob("*.xdf"):
            if group_id in str(xdf) or group_id.replace("grp-", "") in xdf.parent.name:
                found.append(xdf)

    # Deduplicate, filter out *_old* backups
    unique: list[Path] = []
    seen_names: set[str] = set()
    for p in found:
        if "_old" in p.stem:
            continue
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique.append(p)

    return sorted(unique)


def _find_stimuli_dir(
    data_root: Path,
    session_info: dict,
    extra_root: Path | None = None,
) -> Path | None:
    """Find stimuli events directory for a session.

    Checks the standard location under ``data_root`` first, then
    ``extra_root`` if provided (e.g. a stimuli app data directory on a
    different drive).
    """
    group_id = session_info.get("group_id", "")
    session_name = session_info.get("session", "")
    candidates = session_info.get("stimuli_candidates", [])

    roots_to_check: list[Path] = []
    stim_root = data_root / "affectai-capture-recording" / "stimuli" / "data"
    if stim_root.exists():
        roots_to_check.append(stim_root)
    if extra_root is not None and extra_root.exists():
        roots_to_check.append(extra_root)

    for root in roots_to_check:
        for cand in candidates:
            subdir = root / cand
            if subdir.exists():
                return root
            for tsv in root.glob(f"events_{cand}*_experiment.tsv"):
                return root

        # Fallback: subdirectory contains group/session identifiers
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            if group_id and group_id in subdir.name:
                return root
            date_part = session_name.replace("ses-", "").split("_")[0]
            if date_part and date_part in subdir.name:
                return root

        # Last resort: loose experiment TSVs in root
        for tsv in root.glob("events_*_experiment.tsv"):
            if group_id in tsv.name or session_name.split("_")[0].replace("ses-", "") in tsv.name:
                return root

    return stim_root if stim_root.exists() else None


def _find_all_stimuli_experiment_events(
    stimuli_dir: Path,
    session_info: dict,
    extra_root: Path | None = None,
) -> list[Path]:
    """Locate **all** matching experiment events TSVs for a session (may span restarts).

    Searches ``stimuli_dir`` and optionally ``extra_root`` (e.g. events on a
    different drive).
    """
    found: list[Path] = []
    candidates = session_info.get("stimuli_candidates", [])
    group_id = session_info.get("group_id", "")
    session_stem = session_info.get("session", "").replace("ses-", "").split("_run")[0]

    roots_to_search: list[Path] = [stimuli_dir]
    if extra_root is not None and extra_root.exists() and extra_root != stimuli_dir:
        roots_to_search.append(extra_root)

    for root in roots_to_search:
        # Each candidate is a subdirectory name — look INSIDE it for experiment TSVs
        for cand in candidates:
            cand_dir = root / cand
            if cand_dir.is_dir():
                found.extend(sorted(cand_dir.glob("events_*_experiment.tsv")))

        if found:
            continue  # found via candidates, still check extra_root

        # Fallback: match subdirectories by session stem or group_id
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            if session_stem and session_stem in subdir.name:
                found.extend(sorted(subdir.glob("events_*_experiment.tsv")))
            elif group_id and group_id in subdir.name:
                found.extend(sorted(subdir.glob("events_*_experiment.tsv")))

    return sorted(set(found))


def _find_av_session_dir(data_root: Path, session_info: dict) -> Path | None:
    """Find AV session directory for video clock anchors."""
    session_name = session_info.get("session", "")
    av_splits = session_info.get("av_splits", [])

    # Search both conventional AV/ layout and affectai-capture-av/ repo layout
    search_roots: list[Path] = []
    for phase in av_splits or ["final", "pilot", "test"]:
        search_roots.append(data_root / "AV" / phase / "sub-01")
        search_roots.append(
            data_root / "affectai-capture-av" / "sessions" / phase / "sub-01"
        )

    for av_root in search_roots:
        if not av_root.exists():
            continue
        for d in av_root.iterdir():
            if d.is_dir() and session_name.replace("ses-", "") in d.name:
                return d
        exact = av_root / session_name
        if exact.exists():
            return exact

    return None


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def _session_entities(session_id: str) -> tuple[str, str, str]:
    ses_label = session_id[4:] if session_id.startswith("ses-") else session_id
    sub_label = "01"
    base = f"sub-{sub_label}_ses-{ses_label}_task-T0T1T2T3T4"
    return sub_label, ses_label, base


_TASK_LABEL_RE = re.compile(r"_task-([^_]+)_")


def _task_label_from_path(p: Path) -> str:
    """Extract the task label (e.g. ``"T1"``) from a BIDS filename."""
    m = _TASK_LABEL_RE.search(p.name)
    return m.group(1) if m else "UNK"


def process_session(
    session_info: dict,
    data_root: Path,
    output_root: Path,
    dry_run: bool = False,
    tobii_root: Path | None = None,
    stimuli_root: Path | None = None,
) -> dict[str, Any]:
    """Process a single session: extract XDF, derive T0-T4 windows, split streams."""
    session_id = session_info["session"]
    group_id = session_info.get("group_id", "")
    sub_label, ses_label, base = _session_entities(session_id)

    result: dict[str, Any] = {
        "session": session_id,
        "group_id": group_id,
        "success": False,
        "error": None,
        "xdf_files": [],
        "streams": {},
        "task_windows": [],
        "split_files": [],
    }

    logger.info("=" * 70)
    logger.info("Processing: %s (group=%s)", session_id, group_id)
    logger.info("=" * 70)

    # 1. Locate XDF files
    xdf_files = _find_xdf_files(data_root, group_id, session_id)
    result["xdf_files"] = [str(p) for p in xdf_files]
    if not xdf_files:
        msg = f"No XDF files found for group {group_id}"
        logger.warning("  SKIP: %s", msg)
        result["error"] = msg
        return result

    logger.info("  XDF files: %s", [p.name for p in xdf_files])

    # 2. Locate stimuli events (may span multiple dirs from restarts)
    stimuli_dir = _find_stimuli_dir(data_root, session_info, extra_root=stimuli_root)
    experiment_events_paths: list[Path] = []
    if stimuli_dir:
        experiment_events_paths = _find_all_stimuli_experiment_events(
            stimuli_dir, session_info, extra_root=stimuli_root
        )
    if experiment_events_paths:
        logger.info("  Stimuli events: %s", [p.name for p in experiment_events_paths])
    else:
        logger.warning("  No stimuli experiment events found")

    # 3. Locate AV session for video anchors
    av_session_dir = _find_av_session_dir(data_root, session_info)
    if av_session_dir:
        logger.info("  AV session: %s", av_session_dir.name)

    if dry_run:
        logger.info("  [DRY-RUN] Would process this session")
        result["success"] = True
        return result

    # 4. Create output directory
    session_dir = output_root / f"sub-{sub_label}" / f"ses-{ses_label}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 5. Extract all streams from XDF
    raw_streams: list[dict] = []
    try:
        categories, raw_streams = extract_xdf_streams(xdf_files, return_raw=True)
    except Exception as exc:
        msg = f"XDF extraction failed: {exc}"
        logger.error("  %s", msg)
        result["error"] = msg
        return result

    stream_counts = {k: len(v) for k, v in categories.items() if v}
    result["streams"] = stream_counts
    logger.info("  Stream counts: %s", stream_counts)

    # 6. Write full-session stream tables
    stream_paths = write_stream_tables(session_dir, base, categories)

    # 7. Derive task windows from stimuli events
    windows: list[dict[str, Any]] = []
    offset: float | None = None
    events_rows: list[dict[str, str]] = []

    if experiment_events_paths:
        # Compute windows per-TSV independently, then merge the best per task.
        # This avoids inflated durations when wall clocks span app restarts.
        best_windows: dict[str, dict[str, Any]] = {}
        best_offset: float | None = None
        best_events: list[dict[str, str]] = []
        for ep in experiment_events_paths:
            ep_rows = _read_tsv(ep)
            ep_windows, ep_offset = compute_task_windows(ep_rows)
            for w in ep_windows:
                task = w["task"]
                prev = best_windows.get(task)
                if prev is None or w["duration_s"] > prev["duration_s"]:
                    best_windows[task] = w
            # Keep offset & events from the file that produced the most windows
            if len(ep_windows) > len(windows):
                best_offset = ep_offset
                best_events = ep_rows
                windows = ep_windows
        # Rebuild windows list in canonical TASK_ORDER
        windows = [best_windows[t] for t in TASK_ORDER if t in best_windows]
        offset = best_offset
        events_rows = best_events

    if not windows:
        # Fallback: try to derive windows from XDF marker streams
        logger.warning("  No task windows from stimuli; attempting fallback from XDF markers")
        if categories["markers"]:
            # Convert marker rows to pseudo-events format
            marker_events = _markers_to_pseudo_events(categories["markers"])
            if marker_events:
                windows, offset = compute_task_windows(marker_events)

    if windows:
        logger.info("  Task windows: %s", [w["task"] for w in windows])
        for w in windows:
            logger.info(
                "    %s: %.1fs (wall %.0f–%.0f)",
                w["task"], w["duration_s"],
                w["start_wall_clock"], w["end_wall_clock"],
            )
        result["task_windows"] = [
            {"task": w["task"], "duration_s": round(w["duration_s"], 1)} for w in windows
        ]
    else:
        logger.warning("  No task windows derived — writing full-session data only")

    # 8. Write task window TSVs
    if windows:
        task_win_path = session_dir / "annot" / f"{base}_task_run_windows.tsv"
        _write_windows_tsv(task_win_path, windows, offset)

        break_windows = compute_break_windows(events_rows, windows, offset)
        if break_windows:
            seg_win_path = session_dir / "annot" / f"{base}_segment_windows.tsv"
            _write_windows_tsv(seg_win_path, break_windows, offset)

    # 9. Write events.tsv
    if events_rows:
        write_session_events(session_dir, events_rows)

    # 10. Write per-task beh events
    if windows and events_rows:
        write_task_beh_events(session_dir, sub_label, ses_label, events_rows, windows)

    # 11. Split LSL stream tables by T0-T4 windows
    split_files: list[str] = []
    if windows:
        split_specs = [
            (stream_paths.get("tobii"), "et", "acq-lsl_tobii.tsv.gz"),
            (stream_paths.get("emotibit"), "physio", "acq-lsl_emotibit.tsv.gz"),
            (stream_paths.get("vicon"), "eeg", "acq-lsl_vicon.tsv.gz"),
            (stream_paths.get("sync"), "annot", "acq-lsl_sync.tsv"),
            (stream_paths.get("markers"), "beh", "recording-lsl_events.tsv"),
        ]
        for src_path, out_subdir, suffix in split_specs:
            if src_path and src_path.exists():
                logger.info("  Splitting %s by T0–T4...", src_path.name)
                splits = split_lsl_table_by_windows(
                    src_path,
                    session_dir / out_subdir,
                    sub_label, ses_label, suffix,
                    windows,
                )
                split_files.extend(str(p) for p in splits)

    result["split_files"] = split_files

    # 12. Video clock anchors (no video copied)
    video_anchors = extract_video_clock_anchors(categories)
    write_video_clock_anchors(session_dir, base, video_anchors, av_session_dir)

    # 12b. Split DPA mic audio by task windows
    audio_split_files: list[str] = []
    if windows and av_session_dir:
        # Run BOTH methods independently, then pick the better anchor per mic
        # based on regression residual RMSE.
        #
        # Method 1: XDF ffmpeg_progress streams (per-sample clock-corrected)
        xdf_anchors_raw = _compute_dpa_anchors_from_xdf(raw_streams)

        # Method 2: Progress TSV + ffmpeg_clock bridge
        tsv_anchors_raw: dict[str, tuple[float, float, float]] = {}
        capture_dir = _find_av_capture_dir(av_session_dir)
        if capture_dir:
            av_to_xdf = _compute_av_to_xdf_offset(raw_streams)
            if av_to_xdf is not None:
                tsv_anchors_raw = _compute_dpa_anchors(capture_dir, av_to_xdf)
            else:
                logger.warning("  No ffmpeg_clock stream in XDF — TSV method skipped")
        else:
            logger.warning("  No AV capture directory found in %s", av_session_dir)

        # Method 3: Raw JSONL logs (fallback when AV XDF is corrupted/missing)
        # Uses received_time (wall clock of LSL recorder) → XDF time via offset.
        jsonl_anchors_raw: dict[str, tuple[float, float, float]] = {}
        wall_minus_xdf = offset if offset is not None else 0.0
        if not xdf_anchors_raw and capture_dir and wall_minus_xdf != 0.0:
            jsonl_anchors_raw = _compute_dpa_anchors_from_jsonl(capture_dir, wall_minus_xdf)
            if jsonl_anchors_raw:
                logger.info(
                    "  DPA anchors: XDF empty, using JSONL logs (%d mics)",
                    len(jsonl_anchors_raw),
                )

        combined_xdf = xdf_anchors_raw or jsonl_anchors_raw
        if combined_xdf or tsv_anchors_raw:
            logger.info(
                "  DPA anchor comparison: XDF/JSONL mics=%d  TSV mics=%d",
                len(combined_xdf), len(tsv_anchors_raw),
            )
            dpa_anchors = _select_best_dpa_anchors(combined_xdf, tsv_anchors_raw)
        else:
            dpa_anchors: dict[str, tuple[float, float]] = {}
            logger.warning("  No DPA mic progress data found from XDF, JSONL, or TSV")

        if dpa_anchors:
            if capture_dir is None:
                capture_dir = _find_av_capture_dir(av_session_dir)
            if capture_dir:
                logger.info(
                    "  Splitting DPA audio: %d mics, offset=%.6f",
                    len(dpa_anchors), wall_minus_xdf,
                )
                audio_files = split_av_audio_by_windows(
                    capture_dir, session_dir,
                    sub_label, ses_label,
                    windows, dpa_anchors, wall_minus_xdf,
                )
                audio_split_files = [str(p) for p in audio_files]
                split_files.extend(audio_split_files)

    # 12c. Per-participant Tobii LSL and EmotiBit splits
    if windows:
        # Tobii: et/ directory — split rows by Tobii_P{N}_stream
        tobii_task_files = [
            session_dir / "et" / f"sub-{sub_label}_ses-{ses_label}_task-{w['task']}_run-01_acq-lsl_tobii.tsv.gz"
            for w in windows
        ]
        for task_file in tobii_task_files:
            if task_file.exists():
                task_label = _task_label_from_path(task_file)
                per_p = split_lsl_by_participant(
                    task_file,
                    session_dir / "et",
                    sub_label, ses_label, task_label,
                    stream_prefix="Tobii_P",
                    acq_suffix="tobii",
                )
                split_files.extend(str(p) for p in per_p)

        # EmotiBit: physio/ directory — split rows by Emotibit_P{N}_stream
        emotibit_task_files = [
            session_dir / "physio" / f"sub-{sub_label}_ses-{ses_label}_task-{w['task']}_run-01_acq-lsl_emotibit.tsv.gz"
            for w in windows
        ]
        for task_file in emotibit_task_files:
            if task_file.exists():
                task_label = _task_label_from_path(task_file)
                per_p = split_lsl_by_participant(
                    task_file,
                    session_dir / "physio",
                    sub_label, ses_label, task_label,
                    stream_prefix="Emotibit_P",
                    acq_suffix="emotibit",
                )
                split_files.extend(str(p) for p in per_p)

    # 12d. Per-participant Tobii world video clips
    if windows and tobii_root is not None:
        metadata_root = data_root.parent / "metadata"
        if not metadata_root.exists():
            metadata_root = Path(__file__).resolve().parents[1] / "metadata"
        session_id = session_info["session"]
        tobii_video_map = _load_tobii_video_map(metadata_root, session_id)
        if tobii_video_map:
            logger.info("  Splitting Tobii world video for %d participants", len(tobii_video_map))
            video_files = split_tobii_video_by_task(
                tobii_root,
                tobii_video_map,
                session_dir,
                sub_label, ses_label,
                windows,
                wall_minus_xdf if wall_minus_xdf != 0.0 else (offset or 0.0),
            )
            split_files.extend(str(p) for p in video_files)
        else:
            logger.info("  No Tobii video map found for %s — skipping world video clips", session_id)

    # 13. Participant signal map
    emotibit_cfg = data_root.parent / "configs" / "emotibit_participants_by_source.json"
    if not emotibit_cfg.exists():
        emotibit_cfg = Path(__file__).resolve().parents[1] / "configs" / "emotibit_participants_by_source.json"
    write_participant_signal_map(
        session_dir, base, categories,
        emotibit_cfg if emotibit_cfg.exists() else None,
    )

    # 14. Stimuli answers
    if stimuli_dir:
        write_stimuli_answers(session_dir, sub_label, ses_label, stimuli_dir)

    # 15. Sync metadata summary
    sync_meta = {
        "session_id": session_id,
        "group_id": group_id,
        "pipeline": "xdf_sync_pipeline",
        "pipeline_version": "1.0",
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "xdf_files": [str(p) for p in xdf_files],
        "stimuli_events": [str(p) for p in experiment_events_paths] if experiment_events_paths else None,
        "av_session_dir": str(av_session_dir) if av_session_dir else None,
        "wall_minus_lsl_offset": offset,
        "stream_counts": stream_counts,
        "task_windows": [
            {"task": w["task"], "duration_s": w["duration_s"]} for w in windows
        ],
        "split_file_count": len(split_files),
        "video_files_copied": False,
        "audio_files_split": len(audio_split_files) > 0,
        "audio_split_count": len(audio_split_files),
        "notes": [
            "Videos excluded -- clock anchors preserved in annot/video_clock_anchors.tsv",
            "All timestamps are LSL-synchronized via XDF",
            "DPA mic audio split by task windows via ffmpeg_clock bridge"
            if audio_split_files
            else "No DPA audio split (missing AV data or ffmpeg_clock stream)",
        ],
    }
    meta_path = session_dir / "annot" / f"{base}_sync_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(sync_meta, indent=2), encoding="utf-8")

    result["success"] = True
    logger.info("  Done: %d split files written to %s", len(split_files), session_dir)
    return result


def _markers_to_pseudo_events(marker_rows: list[list[str]]) -> list[dict[str, str]]:
    """Convert XDF marker rows to pseudo stimuli-events format for window derivation."""
    pseudo: list[dict[str, str]] = []
    for row in marker_rows:
        if len(row) < 4:
            continue
        lsl_time = row[0]
        stream_name = row[1]
        value = row[3] if len(row) > 3 else ""

        # Try to parse JSON payload from marker value
        detail = {}
        try:
            detail = json.loads(value) if value.startswith("{") else {}
        except Exception:
            pass

        task = str(detail.get("task", "") or "")
        phase_val = str(detail.get("phase", "") or "")
        event_type = str(detail.get("event_type", "") or "")

        # Extract wall_clock from JSON payload if available
        wall_str = str(detail.get("wall_clock", "") or "")

        # Also check stream_name for task info
        if not task and "experiment" in stream_name.lower():
            # Try extracting from value text
            for t in TASK_ORDER:
                if t in value.upper():
                    task = t
                    break

        pseudo.append({
            "wall_clock": wall_str if wall_str else lsl_time,
            "lsl_clock": lsl_time,
            "task": task,
            "phase": phase_val,
            "event_type": event_type,
            "stream": stream_name,
            "detail": value,
            "participant": str(detail.get("participant", "") or ""),
        })

    return pseudo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="XDF-based synchronization pipeline: extract, sync, and split into T0–T4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory containing high_level_data_inventory.json "
             "(default: <repo>/data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed synchronized data",
    )
    p.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Process only these session IDs (e.g. ses-20260312_grp-07_run01). "
             "Default: all sessions with XDF files",
    )
    p.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Process only these group IDs (e.g. grp-07 grp-09). "
             "Default: all groups",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing files",
    )
    p.add_argument(
        "--tobii-root",
        type=Path,
        default=None,
        help="Root directory containing Tobii recording folders "
             "(e.g. D:/data_witout-video/Tobii). "
             "When provided, Tobii world video (scenevideo.mp4) is clipped "
             "per participant per task and written to et/ as acq-P{N}_tobii.mp4. "
             "If omitted, video clipping is skipped.",
    )
    p.add_argument(
        "--stimuli-root",
        type=Path,
        default=None,
        help="Extra stimuli/data directory to search for experiment events TSVs. "
             "Use when the stimuli app data lives outside the main data root "
             "(e.g. C:/Users/AffectAI/Documents/Codes/.../affectai-capture/stimuli/data). "
             "The directory should contain one sub-folder per session named after "
             "the session (e.g. 20260320_grp-16_run01_20260320_095857/).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    tobii_root = args.tobii_root.resolve() if args.tobii_root else None
    stimuli_root = args.stimuli_root.resolve() if args.stimuli_root else None

    logger.info("Data root:  %s", data_root)
    logger.info("Output dir: %s", output_dir)
    if tobii_root:
        logger.info("Tobii root: %s", tobii_root)
    if stimuli_root:
        logger.info("Stimuli root: %s", stimuli_root)

    # Load inventory
    inventory = _load_inventory(data_root)
    sessions = inventory.get("sessions", [])
    logger.info("Inventory: %d sessions", len(sessions))

    # Filter sessions
    if args.sessions:
        sessions = [s for s in sessions if s["session"] in args.sessions]
    if args.groups:
        sessions = [s for s in sessions if s.get("group_id") in args.groups]

    # Only process sessions that have XDF files available
    processable = []
    for s in sessions:
        xdf_files = _find_xdf_files(data_root, s.get("group_id", ""), s.get("session", ""))
        if xdf_files:
            processable.append(s)
        else:
            logger.debug("Skipping %s — no XDF files", s["session"])

    logger.info("Sessions with XDF: %d", len(processable))

    if not processable:
        logger.warning("No sessions with XDF files found. Nothing to process.")
        return 0

    if args.dry_run:
        logger.info("--- DRY RUN ---")
        for s in processable:
            xdf_files = _find_xdf_files(data_root, s.get("group_id", ""), s.get("session", ""))
            logger.info(
                "  %s (group=%s): %d XDF file(s)",
                s["session"], s.get("group_id", "?"), len(xdf_files),
            )
        return 0

    # Process each session
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for s in processable:
        try:
            result = process_session(
                s, data_root, output_dir,
                tobii_root=tobii_root,
                stimuli_root=stimuli_root,
            )
            results.append(result)
        except Exception as exc:
            logger.error("FAILED: %s — %s", s["session"], exc)
            results.append({
                "session": s["session"],
                "success": False,
                "error": str(exc),
            })

    # Write pipeline summary
    summary = {
        "pipeline": "xdf_sync_pipeline",
        "version": "1.0",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "total_sessions": len(processable),
        "succeeded": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }
    summary_path = output_dir / "xdf_sync_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Pipeline complete: %d/%d succeeded. Summary: %s",
        summary["succeeded"], summary["total_sessions"], summary_path,
    )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
