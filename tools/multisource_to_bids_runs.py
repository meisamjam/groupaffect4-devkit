#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_raw_to_bids_module():
    module_path = Path(__file__).resolve().parent / "raw_to_bids.py"
    spec = importlib.util.spec_from_file_location("raw_to_bids", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


raw_to_bids = _load_raw_to_bids_module()

_sync_video_cache = None


def _get_sync_video_module():
    """Lazy-load sync_video module (only when media splitting is needed)."""
    global _sync_video_cache
    if _sync_video_cache is None:
        module_path = Path(__file__).resolve().parent / "create_sync_test_video.py"
        spec = importlib.util.spec_from_file_location("create_sync_test_video", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sync_video_cache = mod
    return _sync_video_cache


TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
AUDIO_EXTS = {".wav", ".flac", ".aac", ".m4a"}

_RE_PARTICIPANT_TOKEN = re.compile(r"(?:^|[^a-z0-9])(?:p|participant|tablet)[_\- ]?([1-4])(?:$|[^a-z0-9])", re.IGNORECASE)
_RE_PARTICIPANT_FILE = re.compile(r"_p[1-4]\.jsonl$", re.IGNORECASE)
_RE_DPA_AUDIO_LABEL = re.compile(r"^dpa(?:[_\-]|$)", re.IGNORECASE)
_RE_CAMERA_AUDIO_LABEL = re.compile(r"(?:jabra|panacast).*_audio", re.IGNORECASE)

_RESPONSE_META_KEYS = {
    "device_id",
    "participant",
    "task",
    "phase",
    "type",
    "probe_name",
    "probe_schema",
    "session_id",
    "received_at",
    "server_received_lsl",
    "client_timestamp",
    "client_perf_ms",
    "clock_offset_ms",
    "block_id",
}


def _copy_or_link(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _copy_tree(src_root: Path, dst_root: Path, link: bool) -> int:
    if not src_root.exists():
        return 0
    copied = 0
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)
        _copy_or_link(src, dst_root / rel, link)
        copied += 1
    return copied


def _session_entities(session_dir: Path) -> tuple[str, str, str]:
    ses = session_dir.name
    sub = session_dir.parent.name
    ses_label = ses[4:] if ses.startswith("ses-") else ses
    sub_label = sub[4:] if sub.startswith("sub-") else sub
    base = f"sub-{sub_label}_ses-{ses_label}_task-T0T1T2T3T4"
    return sub_label, ses_label, base


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _find_experiment_events(stimuli_dir: Path) -> Path:
    candidates = sorted(stimuli_dir.rglob("events_*_experiment.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No experiment events TSV found under: {stimuli_dir}")
    return candidates[-1]


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _participant_label(value: Any) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    m = re.fullmatch(r"(?:p|participant|tablet)?[_\- ]?([1-4])", txt, flags=re.IGNORECASE)
    if m:
        return f"P{m.group(1)}"
    return None


def _participant_from_text(text: str | None) -> str | None:
    if not text:
        return None
    m = _RE_PARTICIPANT_TOKEN.search(str(text).strip().lower())
    if not m:
        return None
    return f"P{m.group(1)}"


def _json_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _find_stimuli_response_files(stimuli_dir: Path) -> list[Path]:
    files = sorted(p for p in stimuli_dir.rglob("responses_*.jsonl") if p.is_file())
    if not files:
        return []
    session_level = [p for p in files if not _RE_PARTICIPANT_FILE.search(p.name)]
    return session_level if session_level else files


def _iter_stimuli_answer_rows(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    response_type = str(payload.get("type", "") or "").strip().lower()
    rows: list[tuple[str, Any]] = []

    if response_type == "vad":
        for k in ("valence", "arousal", "dominance"):
            if payload.get(k) is not None:
                rows.append((k, payload.get(k)))
        return rows

    if response_type == "postblock":
        resp = payload.get("responses")
        if isinstance(resp, dict):
            for k, v in sorted(resp.items()):
                rows.append((str(k), v))
        return rows

    if response_type == "probe":
        probe_name = payload.get("probe_name")
        if probe_name is not None:
            rows.append(("probe_name", probe_name))
        for k, v in payload.items():
            if k in _RESPONSE_META_KEYS:
                continue
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


def _write_stimuli_answers_table(
    session_dir: Path,
    stimuli_dir: Path,
    sub_label: str,
    ses_label: str,
) -> tuple[Path | None, dict[str, Any]]:
    files = _find_stimuli_response_files(stimuli_dir)
    if not files:
        return None, {"response_files": [], "rows": 0, "participants": {}}

    out = session_dir / "beh" / f"sub-{sub_label}_ses-{ses_label}_task-T0T1T2T3T4_stimuli_answers.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "wall_clock",
        "lsl_clock",
        "task",
        "phase",
        "response_type",
        "participant",
        "device_id",
        "item_key",
        "item_value",
        "source_file",
    ]

    row_count = 0
    participant_counts: dict[str, int] = {}

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        for src in files:
            rel_src = src.relative_to(stimuli_dir)
            with src.open("r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue

                    participant = (
                        _participant_label(payload.get("participant"))
                        or _participant_from_text(payload.get("device_id"))
                    )
                    if participant:
                        participant_counts[participant] = participant_counts.get(participant, 0) + 1

                    wall = _parse_float(str(payload.get("received_at", "")), -1.0)
                    lsl = _parse_float(str(payload.get("server_received_lsl", "")), -1.0)
                    task = str(payload.get("task", "") or "")
                    phase = str(payload.get("phase", "") or "")
                    rtype = str(payload.get("type", "form") or "form")
                    device_id = str(payload.get("device_id", "") or "")
                    answers = _iter_stimuli_answer_rows(payload)

                    if not answers:
                        answers = [("response", "")]

                    for key, value in answers:
                        writer.writerow(
                            [
                                "" if wall < 0 else f"{wall:.6f}",
                                "" if lsl < 0 else f"{lsl:.6f}",
                                task,
                                phase,
                                rtype,
                                participant or "",
                                device_id,
                                str(key),
                                _json_scalar(value),
                                str(rel_src).replace("\\", "/"),
                            ]
                        )
                        row_count += 1

    return out, {
        "response_files": [str(p.relative_to(stimuli_dir)).replace("\\", "/") for p in files],
        "rows": row_count,
        "participants": participant_counts,
    }


def _read_table_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _participant_signal_rows(session_dir: Path, answers_tsv: Path | None) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[tuple[str, str]] = set()

    def add_row(signal: str, participant: str | None, source_type: str, source_file: str, reason: str) -> None:
        key = (signal, source_file)
        if key in seen:
            return
        seen.add(key)
        rows.append([signal, participant or "", source_type, source_file, reason])

    tobii_tsv = next(iter((session_dir / "et").glob("*_acq-lsl_tobii.tsv.gz")), None)
    if tobii_tsv and tobii_tsv.exists():
        header, data, _ = _read_lsl_rows(tobii_tsv)
        if "stream_name" in header:
            idx = header.index("stream_name")
            for row in data:
                if len(row) <= idx:
                    continue
                name = row[idx]
                part = _participant_from_text(name)
                add_row(name, part, "tobii_lsl", str(tobii_tsv.relative_to(session_dir)), "stream_name")

    emotibit_tsv = next(iter((session_dir / "physio").glob("*_acq-lsl_emotibit.tsv.gz")), None)
    if emotibit_tsv and emotibit_tsv.exists():
        header, data, _ = _read_lsl_rows(emotibit_tsv)
        if "stream_name" in header:
            idx = header.index("stream_name")
            for row in data:
                if len(row) <= idx:
                    continue
                name = row[idx]
                part = _participant_from_text(name)
                add_row(name, part, "emotibit_lsl", str(emotibit_tsv.relative_to(session_dir)), "stream_name")

    if answers_tsv is not None and answers_tsv.exists():
        for rec in _read_table_rows(answers_tsv):
            participant = (rec.get("participant") or "").strip()
            device_id = (rec.get("device_id") or "").strip()
            if not participant:
                participant = _participant_from_text(device_id) or ""
            if not device_id:
                device_id = "stimuli_response"
            add_row(device_id, participant or None, "stimuli", str(answers_tsv.relative_to(session_dir)), "response_device")

    repo_root = Path(__file__).resolve().parents[1]
    emotibit_cfg = repo_root / "configs" / "emotibit_participants_by_source.json"
    if emotibit_cfg.exists():
        try:
            raw = json.loads(emotibit_cfg.read_text(encoding="utf-8"))
            participants = raw.get("participants", {}) if isinstance(raw, dict) else {}
            by_source = raw.get("by_source", {}) if isinstance(raw, dict) else {}
            if isinstance(participants, dict):
                for p, hwid in sorted(participants.items()):
                    part = _participant_label(p)
                    add_row(str(hwid), part, "emotibit_config", "configs/emotibit_participants_by_source.json", "participant_hardware_id")
            if isinstance(by_source, dict):
                for source, p in sorted(by_source.items()):
                    part = _participant_label(p)
                    add_row(str(source), part, "emotibit_config", "configs/emotibit_participants_by_source.json", "participant_source")
        except Exception:
            pass

    return sorted(rows, key=lambda r: (r[2], r[0], r[1], r[3]))


def _write_participant_signal_map(
    session_dir: Path,
    sub_label: str,
    ses_label: str,
    answers_tsv: Path | None,
) -> Path:
    out = session_dir / "annot" / f"sub-{sub_label}_ses-{ses_label}_participant_signal_map.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _participant_signal_rows(session_dir, answers_tsv)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["signal", "participant", "source_type", "source_file", "mapping_reason"])
        writer.writerows(rows)
    return out


def _compute_task_windows(
    events_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], float | None]:
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
        et = _event_type(row)
        phase = _phase(row)
        if phase in {"welcome", "study_introduction", "vad_introduction", "postblock_introduction", "intro"}:
            return et in {"push_content", "phase_start"}
        return False

    def _is_tn_tobii_start(row: dict[str, str]) -> bool:
        task = (row.get("task") or "").strip().upper()
        if task not in {"T1", "T2", "T3", "T4"}:
            return False
        et = _event_type(row)
        phase = _phase(row)
        return et == "tobii_calibration" or (et in {"push_content", "phase_start"} and phase == "tobii_calibration")

    def _is_task_finish(row: dict[str, str]) -> bool:
        task = (row.get("task") or "").strip().upper()
        if task not in TASK_ORDER:
            return False
        et = _event_type(row)
        phase = _phase(row)
        return et == "task_end" or (et in {"push_content", "phase_end"} and phase == "finish")

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
        start = starts.get(task)
        if start is None:
            start = fallback_starts.get(task)
        if start is None:
            continue

        end = ends.get(task)
        if end is None or end <= start:
            # Fallback endpoint: next task's resolved start or last wall clock.
            next_start = None
            for t2 in TASK_ORDER[idx + 1 :]:
                t2_start = starts.get(t2) or fallback_starts.get(t2)
                if t2_start is not None:
                    next_start = t2_start
                    break
            end = next_start if next_start is not None else max(all_wall)

        # Guard against malformed logs where end can still be <= start.
        if end <= start:
            continue

        start_lsl = (start - offset) if offset is not None else None
        end_lsl = (end - offset) if offset is not None else None
        windows.append(
            {
                "task": task,
                "run": "01",
                "start_wall_clock": start,
                "end_wall_clock": end,
                "duration_s": max(0.0, end - start),
                "start_lsl": start_lsl,
                "end_lsl": end_lsl,
            }
        )
    return windows, offset


def _write_windows(
    session_dir: Path, base: str, windows: list[dict[str, Any]], offset: float | None
) -> Path:
    out = session_dir / "annot" / f"{base}_task_run_windows.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "task",
                "run",
                "start_wall_clock",
                "end_wall_clock",
                "duration_s",
                "start_lsl",
                "end_lsl",
                "wall_minus_lsl_offset",
            ]
        )
        for w in windows:
            writer.writerow(
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
            )
    return out


def _write_segment_windows(
    session_dir: Path,
    base: str,
    windows: list[dict[str, Any]],
    offset: float | None,
) -> Path:
    out = session_dir / "annot" / f"{base}_segment_windows.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "segment",
                "run",
                "start_wall_clock",
                "end_wall_clock",
                "duration_s",
                "start_lsl",
                "end_lsl",
                "wall_minus_lsl_offset",
            ]
        )
        for w in windows:
            writer.writerow(
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
            )
    return out


def _compute_break_windows(
    events_rows: list[dict[str, str]],
    task_windows: list[dict[str, Any]],
    offset: float | None,
) -> list[dict[str, Any]]:
    if not task_windows:
        return []
    all_wall = sorted(
        _parse_float(r.get("wall_clock", ""), -1.0) for r in events_rows if r.get("wall_clock")
    )
    all_wall = [v for v in all_wall if v >= 0]
    if not all_wall:
        return []

    session_start = min(all_wall)
    session_end = max(all_wall)
    sorted_tasks = sorted(task_windows, key=lambda w: float(w["start_wall_clock"]))

    def _mk_window(label: str, start: float, end: float) -> dict[str, Any] | None:
        if end <= start:
            return None
        start_lsl = (start - offset) if offset is not None else None
        end_lsl = (end - offset) if offset is not None else None
        return {
            "task": label,
            "run": "01",
            "start_wall_clock": start,
            "end_wall_clock": end,
            "duration_s": max(0.0, end - start),
            "start_lsl": start_lsl,
            "end_lsl": end_lsl,
        }

    windows: list[dict[str, Any]] = []

    first = sorted_tasks[0]
    pre = _mk_window("PRE", session_start, float(first["start_wall_clock"]))
    if pre is not None:
        windows.append(pre)

    for left, right in zip(sorted_tasks, sorted_tasks[1:], strict=False):
        left_label = str(left.get("task", ""))
        right_label = str(right.get("task", ""))
        brk = _mk_window(
            f"BREAK_{left_label}_{right_label}",
            float(left["end_wall_clock"]),
            float(right["start_wall_clock"]),
        )
        if brk is not None:
            windows.append(brk)

    last = sorted_tasks[-1]
    post = _mk_window("POST", float(last["end_wall_clock"]), session_end)
    if post is not None:
        windows.append(post)

    return windows


def _write_session_events_from_stimuli(
    session_dir: Path, events_rows: list[dict[str, str]]
) -> Path:
    out = session_dir / "events.tsv"
    wall_values = [_parse_float(r.get("wall_clock", ""), -1.0) for r in events_rows]
    wall_values = [v for v in wall_values if v >= 0]
    if not wall_values:
        return out
    t0 = min(wall_values)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["onset", "duration", "trial_type", "value", "description"])
        for row in events_rows:
            wall = _parse_float(row.get("wall_clock", ""), -1.0)
            if wall < 0:
                continue
            onset = wall - t0
            trial_type = row.get("event_type", "")
            value = row.get("detail", "")
            desc = (
                f"task={row.get('task', '')};phase={row.get('phase', '')};"
                f"stream={row.get('stream', '')};participant={row.get('participant', '')}"
            )
            writer.writerow([f"{onset:.6f}", "0.0", trial_type, value, desc])
    return out


def _write_task_beh_events(
    session_dir: Path,
    sub_label: str,
    ses_label: str,
    events_rows: list[dict[str, str]],
    windows: list[dict[str, Any]],
    filter_event_task: bool,
) -> list[Path]:
    out_files: list[Path] = []
    beh_dir = session_dir / "beh"
    beh_dir.mkdir(parents=True, exist_ok=True)
    for w in windows:
        task = w["task"]
        start = float(w["start_wall_clock"])
        end = float(w["end_wall_clock"])
        out = beh_dir / f"sub-{sub_label}_ses-{ses_label}_task-{task}_run-01_events.tsv"
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["onset", "duration", "trial_type", "value", "description"])
            for row in events_rows:
                if filter_event_task and (row.get("task") or "").strip() != task:
                    continue
                wall = _parse_float(row.get("wall_clock", ""), -1.0)
                if wall < start or wall >= end:
                    continue
                onset = wall - start
                trial_type = row.get("event_type", "")
                value = row.get("detail", "")
                desc = (
                    f"task={row.get('task', '')};phase={row.get('phase', '')};"
                    f"stream={row.get('stream', '')};participant={row.get('participant', '')}"
                )
                writer.writerow([f"{onset:.6f}", "0.0", trial_type, value, desc])
        out_files.append(out)
    return out_files


def _read_lsl_rows(path: Path) -> tuple[list[str], list[list[str]], bool]:
    gz = path.suffix.lower() == ".gz"
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return [], [], gz
    return rows[0], rows[1:], gz


def _write_lsl_rows(path: Path, header: list[str], rows: list[list[str]], gzip_out: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_out else open
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _split_lsl_table_by_windows(
    input_file: Path,
    output_dir: Path,
    sub_label: str,
    ses_label: str,
    suffix: str,
    windows: list[dict[str, Any]],
) -> list[Path]:
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
        keep: list[list[str]] = []
        for row in rows:
            if len(row) <= time_idx:
                continue
            t = _parse_float(row[time_idx], -1.0)
            if start <= t < end:
                keep.append(row)
        label = str(w.get("task", "UNK"))
        out = output_dir / f"sub-{sub_label}_ses-{ses_label}_task-{label}_run-01_{suffix}"
        _write_lsl_rows(out, header, keep, gzip_out=gz)
        outputs.append(out)
    return outputs


def _load_windows_from_tsv(tsv: Path) -> tuple[list[dict[str, Any]], float | None]:
    windows: list[dict[str, Any]] = []
    offset: float | None = None
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            off_val = row.get("wall_minus_lsl_offset", "")
            if off_val and offset is None:
                try:
                    offset = float(off_val)
                except ValueError:
                    pass
            windows.append({
                "task": row.get("task") or row.get("segment") or "UNK",
                "run": row.get("run", "01"),
                "start_wall_clock": float(row["start_wall_clock"]),
                "end_wall_clock": float(row["end_wall_clock"]),
                "duration_s": float(row.get("duration_s") or 0),
                "start_lsl": float(row["start_lsl"]) if row.get("start_lsl") else None,
                "end_lsl": float(row["end_lsl"]) if row.get("end_lsl") else None,
            })
    return windows, offset


def _build_run_chunks(
    session_dir: Path,
    stimuli_dir: Path,
    write_session_events: bool,
    override_windows_tsv: Path | None = None,
) -> dict[str, Any]:
    sub_label, ses_label, base = _session_entities(session_dir)
    exp_events = _find_experiment_events(stimuli_dir)
    rows = _read_tsv(exp_events)

    if override_windows_tsv is not None and override_windows_tsv.exists():
        windows, offset = _load_windows_from_tsv(override_windows_tsv)
        if not windows:
            raise RuntimeError(f"No windows in override TSV: {override_windows_tsv}")
    else:
        windows, offset = _compute_task_windows(rows)
        if not windows:
            raise RuntimeError("No task windows found from stimuli events")
    break_windows = _compute_break_windows(rows, windows, offset)

    windows_path = _write_windows(session_dir, base, windows, offset)
    segment_windows_path = _write_segment_windows(session_dir, base, break_windows, offset)
    session_events_path = None
    if write_session_events:
        session_events_path = _write_session_events_from_stimuli(session_dir, rows)

    task_event_files = _write_task_beh_events(
        session_dir,
        sub_label,
        ses_label,
        rows,
        windows,
        True,
    )
    segment_event_files = _write_task_beh_events(
        session_dir,
        sub_label,
        ses_label,
        rows,
        break_windows,
        False,
    )

    lsl_outputs: list[Path] = []
    lsl_specs = [
        (
            session_dir / "et" / f"{base}_acq-lsl_tobii.tsv.gz",
            session_dir / "et",
            "acq-lsl_tobii.tsv.gz",
        ),
        (
            session_dir / "physio" / f"{base}_acq-lsl_emotibit.tsv.gz",
            session_dir / "physio",
            "acq-lsl_emotibit.tsv.gz",
        ),
        (
            session_dir / "annot" / f"{base}_acq-lsl_sync.tsv",
            session_dir / "annot",
            "acq-lsl_sync.tsv",
        ),
        (
            session_dir / "beh" / f"{base}_recording-lsl_events.tsv",
            session_dir / "beh",
            "recording-lsl_events.tsv",
        ),
    ]
    for src, out_dir, suffix in lsl_specs:
        if src.exists():
            lsl_outputs.extend(
                _split_lsl_table_by_windows(src, out_dir, sub_label, ses_label, suffix, windows)
            )
            lsl_outputs.extend(
                _split_lsl_table_by_windows(src, out_dir, sub_label, ses_label, suffix, break_windows)
            )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stimuli_events": str(exp_events),
        "task_windows_tsv": str(windows_path),
        "segment_windows_tsv": str(segment_windows_path),
        "session_events_tsv": None if session_events_path is None else str(session_events_path),
        "task_event_files": [str(p) for p in task_event_files],
        "segment_event_files": [str(p) for p in segment_event_files],
        "lsl_chunk_files": [str(p) for p in lsl_outputs],
    }
    out = session_dir / "annot" / f"{base}_task_chunking_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _copy_if_exists(src: Path | None, dst: Path, link: bool) -> bool:
    if src is None or not src.exists() or not src.is_file():
        return False
    _copy_or_link(src, dst, link)
    return True


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _parse_iso_utc(value: str) -> float | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _load_progress_start(run_dir: Path, label: str) -> float | None:
    tsv = run_dir / "sourcedata" / "sync" / f"{label}_ffmpeg_progress.tsv"
    if not tsv.exists():
        return None
    anchors: list[float] = []
    try:
        with tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    host_t = float(row.get("host_time_sec", ""))
                    out_t = float(row.get("out_time_sec", ""))
                except Exception:
                    continue
                anchors.append(host_t - out_t)
    except Exception:
        return None
    return _median(anchors)


def _load_lsl_start(run_dir: Path, label: str) -> float | None:
    lsl_dir = run_dir / "lsl"
    candidates = [lsl_dir / f"ffmpeg_progress_{label}.jsonl", lsl_dir / f"{label}.jsonl"]
    for path in candidates:
        if not path.exists():
            continue
        sync_video = _get_sync_video_module()
        vals = sync_video.load_lsl_anchor_candidates(path)
        t = _median(vals)
        if t is not None:
            return t
    return None


def _load_event_start(run_dir: Path, label: str) -> float | None:
    ev_path = run_dir / "video" / "ffmpeg_multicap_events.jsonl"
    if not ev_path.exists():
        return None
    values: list[float] = []
    try:
        with ev_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event_type") != "capture_started":
                    continue
                if rec.get("device_id") != label:
                    continue
                if "unix_time_s" in rec:
                    try:
                        values.append(float(rec["unix_time_s"]))
                        continue
                    except Exception:
                        pass
                ts = rec.get("timestamp")
                if isinstance(ts, str):
                    parsed = _parse_iso_utc(ts)
                    if parsed is not None:
                        values.append(parsed)
    except Exception:
        return None
    return values[-1] if values else None


def _load_av_lsl_to_wall_offset(run_dir: Path) -> float | None:
    """Return the offset (wall_clock - AV_PC_LSL_clock) from any frame_log in this session.

    The AV PC records host_time_sec in its own local_clock() (LSL clock, ~564000s),
    but w_start in _split_media_runs is Unix wall_clock (~1773835xxxs).  This offset
    bridges the two so that LSL-domain progress-TSV anchors can be converted to wall time.
    Derived from frame_log entries which record both unix_time_s and lsl_time.
    """
    frame_logs_dir = run_dir / "frame_logs"
    if not frame_logs_dir.exists():
        return None
    offsets: list[float] = []
    for fl in sorted(frame_logs_dir.glob("*_frames.jsonl")):
        try:
            with fl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    unix_t = rec.get("unix_time_s") or rec.get("unix_time")
                    lsl_t = rec.get("lsl_time")
                    if unix_t is not None and lsl_t is not None and float(unix_t) > 0:
                        offsets.append(float(unix_t) - float(lsl_t))
                    if len(offsets) >= 5:
                        break
        except Exception:
            continue
        if len(offsets) >= 5:
            break
    return _median(offsets) if offsets else None


def _estimate_av_media_start(
    run_dir: Path, media_path: Path
) -> tuple[float | None, str | None, str]:
    stem = media_path.stem
    base_label = stem[:-6] if stem.endswith("_video") else stem

    # For camera audio files (e.g. jabra_panacast_20_cam2_vid_audio.wav), the timing
    # anchor is stored under the companion video label (without the _audio suffix).
    # Build a list of labels to try: original first, then with _audio stripped.
    labels_to_try = [base_label]
    if base_label.endswith("_audio"):
        labels_to_try.append(base_label[:-6])  # strip trailing "_audio"

    if media_path.suffix.lower() in VIDEO_EXTS:
        frame_log = run_dir / "frame_logs" / f"{base_label}_frames.jsonl"
        if frame_log.exists():
            sync_video = _get_sync_video_module()
            t = sync_video.load_frame_log_start_estimate(frame_log)
            if t is not None:
                return t, "frame_log", base_label

    # progress_tsv and lsl_jsonl both return AV-PC LSL clock time (host_time_sec /
    # stream_time are local_clock(), NOT Unix wall time).  We must convert to wall
    # time before subtracting from w_start (which is Unix wall_clock).
    # The conversion offset is derived from any frame_log in this session which
    # records both unix_time_s and lsl_time for the same moment.
    av_offset: float | None = None

    for lbl in labels_to_try:
        t = _load_progress_start(run_dir, lbl)
        if t is not None:
            if av_offset is None:
                av_offset = _load_av_lsl_to_wall_offset(run_dir)
            if av_offset is not None:
                return t + av_offset, "progress_tsv", base_label
            break  # no offset available; fall through to events

    for lbl in labels_to_try:
        t = _load_lsl_start(run_dir, lbl)
        if t is not None:
            if av_offset is None:
                av_offset = _load_av_lsl_to_wall_offset(run_dir)
            if av_offset is not None:
                return t + av_offset, "lsl_jsonl", base_label
            break

    t = _load_event_start(run_dir, base_label)
    if t is not None:
        return t, "events", base_label

    return None, None, base_label


def _load_tobii_scene_start(scene_video: Path) -> tuple[float | None, str | None, str]:
    device_dir = scene_video.parent
    rec = device_dir / "recording.g3"
    if not rec.exists():
        return None, None, device_dir.name
    try:
        data = json.loads(rec.read_text(encoding="utf-8"))
        created = data.get("created")
        if isinstance(created, str):
            ts = _parse_iso_utc(created)
            if ts is not None:
                return ts, "recording.g3", device_dir.name
    except Exception:
        return None, None, device_dir.name
    return None, None, device_dir.name


def _ffprobe_duration(path: Path, ffprobe_bin: str) -> float | None:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        try:
            if path.suffix.lower() == ".wav":
                with wave.open(str(path), "rb") as wf:
                    sr = wf.getframerate()
                    frames = wf.getnframes()
                    if sr > 0:
                        return float(frames) / float(sr)
        except Exception:
            pass
        # Fallback: derive duration via ffmpeg stderr when ffprobe is unavailable.
        ffmpeg_cmd = [
            ffprobe_bin,
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
        try:
            out = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            text = (out.stderr or "") + "\n" + (out.stdout or "")
            m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", text)
            if not m:
                return None
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = float(m.group(3))
            return hh * 3600 + mm * 60 + ss
        except Exception:
            return None


def _split_wav_pcm(src: Path, dst: Path, start_s: float, dur_s: float) -> bool:
    try:
        with wave.open(str(src), "rb") as rf:
            sr = rf.getframerate()
            nch = rf.getnchannels()
            sw = rf.getsampwidth()
            total_frames = rf.getnframes()
            if sr <= 0:
                return False
            start_frame = max(0, int(round(start_s * sr)))
            dur_frames = max(0, int(round(dur_s * sr)))
            if dur_frames <= 0:
                return False
            end_frame = min(total_frames, start_frame + dur_frames)
            if end_frame <= start_frame:
                return False
            rf.setpos(start_frame)
            frames = rf.readframes(end_frame - start_frame)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(nch)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(frames)
        return True
    except Exception:
        return False


def _write_video_clock_anchors(session_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    out = session_dir / "annot" / "video_clock_anchors.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "device",
        "label",
        "anchor_source",
        "start_wall_clock",
        "duration_s",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out


def _split_one_media(
    src: Path,
    dst: Path,
    start_s: float,
    dur_s: float,
    ffmpeg_bin: str,
    ffmpeg_threads: int = 0,
) -> bool:
    if src.suffix.lower() == ".wav" and dst.suffix.lower() == ".wav":
        return _split_wav_pcm(src, dst, start_s, dur_s)

    dst.parent.mkdir(parents=True, exist_ok=True)
    threads_args = ["-threads", str(ffmpeg_threads)] if ffmpeg_threads > 0 else []
    is_video_target_mp4 = dst.suffix.lower() == ".mp4"
    if is_video_target_mp4:
        # Re-encode to H.264/AAC so clips are directly usable by 3D pipelines expecting MP4.
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            str(src),
            "-t",
            f"{dur_s:.6f}",
            *threads_args,
            "-map",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            str(dst),
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            str(src),
            "-t",
            f"{dur_s:.6f}",
            *threads_args,
            "-map",
            "0",
            "-c",
            "copy",
            str(dst),
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def _is_dpa_audio_label(label: str) -> bool:
    return bool(_RE_DPA_AUDIO_LABEL.search((label or "").strip()))


def _is_windowed_audio_label(label: str) -> bool:
    """Return True for audio sources that should be split by task windows."""

    normalized = (label or "").strip()
    return bool(_RE_DPA_AUDIO_LABEL.search(normalized) or _RE_CAMERA_AUDIO_LABEL.search(normalized))


def _split_media_runs(
    session_dir: Path,
    windows: list[dict[str, Any]],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    split_video: bool,
    split_audio: bool,
    av_session_dir: Path | None = None,
    tobii_dirs: list[Path] | None = None,
    ffmpeg_threads: int = 0,
) -> dict[str, Any]:
    sub_label, ses_label, _ = _session_entities(session_dir)
    if av_session_dir is not None:
        av_root = av_session_dir / "sourcedata"
        if not av_root.exists():
            av_root = av_session_dir
    else:
        av_root = session_dir / "sourcedata" / "av" / "sourcedata"

    tobii_roots: list[Path] = []
    if tobii_dirs:
        tobii_roots.extend([p for p in tobii_dirs if p.exists()])
    else:
        default_tobii = session_dir / "sourcedata" / "tobii_device"
        if default_tobii.exists():
            tobii_roots.append(default_tobii)

    media_items: list[dict[str, Any]] = []

    if av_root.exists():
        for run_dir in sorted(p for p in av_root.iterdir() if p.is_dir()):
            for rel in [run_dir / "video", run_dir / "audio"]:
                if not rel.exists():
                    continue
                for media in sorted(p for p in rel.glob("*") if p.is_file()):
                    ext = media.suffix.lower()
                    if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
                        continue
                    start, src_name, label = _estimate_av_media_start(run_dir, media)
                    media_items.append(
                        {
                            "path": media,
                            "label": label,
                            "source": "av",
                            "device": label,
                            "modality": "video" if ext in VIDEO_EXTS else "audio",
                            "start_wall": start,
                            "start_source": src_name,
                        }
                    )

    for tobii_root in tobii_roots:
        for scene in sorted(tobii_root.rglob("scenevideo.mp4")):
            start, src_name, device = _load_tobii_scene_start(scene)
            media_items.append(
                {
                    "path": scene,
                    "label": "scenevideo",
                    "source": "tobii",
                    "device": device,
                    "modality": "video",
                    "start_wall": start,
                    "start_source": src_name,
                }
            )

    # If we have repeated captures for the same logical source, encode capture id in acq
    # so run clips stay deterministic and non-overwriting.
    duplicates: dict[tuple[str, str, str, str], int] = {}
    for item in media_items:
        cap = ""
        path_obj = Path(item["path"])
        if item["source"] == "av":
            # .../av/sourcedata/<capture_run>/(video|audio)/file
            try:
                cap = path_obj.parent.parent.name
            except Exception:
                cap = ""
        elif item["source"] == "tobii":
            # .../tobii_device/<device>/<recording>/scenevideo.mp4
            cap = path_obj.parent.name
        item["capture_id"] = cap
        key = (item["source"], item["device"], item["label"], item["modality"])
        duplicates[key] = duplicates.get(key, 0) + 1

    outputs: list[str] = []
    skipped: list[dict[str, Any]] = []
    video_clock_rows: list[dict[str, Any]] = []
    for item in media_items:
        start_wall = item.get("start_wall")
        if start_wall is None:
            skipped.append({"path": str(item["path"]), "reason": "no_start_anchor"})
            continue

        src_path = Path(item["path"])
        duration = _ffprobe_duration(src_path, ffprobe_bin)

        if item["modality"] == "video":
            video_clock_rows.append(
                {
                    "source_path": str(src_path),
                    "device": str(item.get("device", "")),
                    "label": str(item.get("label", "")),
                    "anchor_source": str(item.get("start_source", "")),
                    "start_wall_clock": f"{float(start_wall):.6f}",
                    "duration_s": "" if duration is None else f"{float(duration):.6f}",
                }
            )

        if item["modality"] == "video" and not split_video:
            skipped.append({"path": str(item["path"]), "reason": "video_split_disabled"})
            continue
        if item["modality"] == "audio" and not split_audio:
            skipped.append({"path": str(item["path"]), "reason": "audio_split_disabled"})
            continue
        if item["modality"] == "audio" and not _is_windowed_audio_label(str(item.get("label", ""))):
            skipped.append({"path": str(item["path"]), "reason": "non_windowed_audio"})
            continue

        if duration is None or duration <= 0:
            skipped.append({"path": str(item["path"]), "reason": "no_duration"})
            continue

        ext = src_path.suffix.lower()
        acq_raw = f"{item['source']}-{item['device']}-{item['label']}"
        key = (item["source"], item["device"], item["label"], item["modality"])
        if duplicates.get(key, 0) > 1 and item.get("capture_id"):
            acq_raw = f"{acq_raw}-cap-{item['capture_id']}"
        acq = raw_to_bids._slug(acq_raw)
        out_dir = session_dir / ("video" if item["modality"] == "video" else "audio")
        suffix = "video" if item["modality"] == "video" else "audio"
        if item["modality"] == "video":
            ext = ".mp4"

        for w in windows:
            task = w["task"]
            run = w["run"]
            w_start = float(w["start_wall_clock"])
            w_end = float(w["end_wall_clock"])

            media_rel_start = w_start - float(start_wall)
            media_rel_end = w_end - float(start_wall)
            clip_start = max(0.0, media_rel_start)
            clip_end = min(float(duration), media_rel_end)
            clip_dur = clip_end - clip_start
            if clip_dur <= 0.05:
                continue

            out = (
                out_dir
                / f"sub-{sub_label}_ses-{ses_label}_task-{task}_run-{run}_acq-{acq}_{suffix}{ext}"
            )
            ok = _split_one_media(src_path, out, clip_start, clip_dur, ffmpeg_bin, ffmpeg_threads)
            if ok:
                outputs.append(str(out))
            else:
                skipped.append({"path": str(src_path), "task": task, "reason": "ffmpeg_failed"})

    video_clock_tsv = _write_video_clock_anchors(session_dir, video_clock_rows)

    return {
        "media_items": len(media_items),
        "generated_clips": outputs,
        "skipped": skipped,
        "video_clock_anchors_tsv": None if video_clock_tsv is None else str(video_clock_tsv),
    }


def merge_and_convert(
    av_session_dir: Path,
    recording_session_dir: Path,
    stimuli_dir: Path,
    output_session_dir: Path,
    tobii_dirs: list[Path],
    link: bool,
    write_session_events: bool,
    split_media: bool,
    processed_only: bool,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    split_video: bool,
    split_audio: bool,
    require_xdf_extraction: bool,
    no_source_ingest: bool,
    override_windows_tsv: Path | None = None,
    ffmpeg_threads: int = 0,
) -> Path:
    if not av_session_dir.exists():
        raise FileNotFoundError(f"AV session dir not found: {av_session_dir}")
    if not recording_session_dir.exists():
        raise FileNotFoundError(f"Recording session dir not found: {recording_session_dir}")
    if not stimuli_dir.exists():
        raise FileNotFoundError(f"Stimuli dir not found: {stimuli_dir}")

    ingest_root = output_session_dir / "sourcedata"
    av_ingest = ingest_root / "av"
    rec_ingest = ingest_root / "recording"
    stim_ingest = ingest_root / "stimuli" / stimuli_dir.name
    tobii_ingest = ingest_root / "tobii_device"

    if no_source_ingest:
        counts = {
            "av_files": 0,
            "recording_files": 0,
            "stimuli_files": 0,
            "tobii_files": 0,
        }
        stim_for_processing = stimuli_dir
    else:
        counts = {
            "av_files": _copy_tree(av_session_dir, av_ingest, link),
            "recording_files": _copy_tree(recording_session_dir, rec_ingest, link),
            "stimuli_files": _copy_tree(stimuli_dir, stim_ingest, link),
            "tobii_files": 0,
        }
        for tobii_dir in tobii_dirs:
            counts["tobii_files"] += _copy_tree(tobii_dir, tobii_ingest / tobii_dir.name, link)
        stim_for_processing = stim_ingest

    xdf_src = _first_existing(
        [
            recording_session_dir / "sourcedata" / "lsl" / f"{recording_session_dir.name[4:]}.xdf",
            *sorted((recording_session_dir / "sourcedata" / "lsl").glob("*.xdf")),
            *sorted(recording_session_dir.glob("*.xdf")),
        ]
    )
    if xdf_src is not None and not no_source_ingest:
        _copy_or_link(xdf_src, output_session_dir / xdf_src.name, link)

    tobii_lsl_src = recording_session_dir / "sourcedata" / "tobii_lsl"
    if tobii_lsl_src.exists() and not no_source_ingest:
        _copy_tree(tobii_lsl_src, output_session_dir / "sourcedata" / "tobii_lsl", link)

    if not no_source_ingest:
        _copy_if_exists(
            _first_existing(
                [recording_session_dir / "participants.json", av_session_dir / "participants.json"]
            ),
            output_session_dir / "participants.json",
            link,
        )

    raw_summary = raw_to_bids.convert(
        output_session_dir,
        link=link,
        require_xdf_extraction=require_xdf_extraction,
        xdf_files=[] if xdf_src is None else [xdf_src],
    )
    chunk_summary = _build_run_chunks(output_session_dir, stim_for_processing, write_session_events, override_windows_tsv)
    sub_label, ses_label, _ = _session_entities(output_session_dir)
    stimuli_answers_tsv, stimuli_answers_summary = _write_stimuli_answers_table(
        session_dir=output_session_dir,
        stimuli_dir=stim_for_processing,
        sub_label=sub_label,
        ses_label=ses_label,
    )
    participant_signal_map = _write_participant_signal_map(
        session_dir=output_session_dir,
        sub_label=sub_label,
        ses_label=ses_label,
        answers_tsv=stimuli_answers_tsv,
    )
    media_summary = None
    if split_media:
        # Media clips must be split strictly by T0-T4 task windows to match sync/task markers.
        windows_tsv = Path(chunk_summary["task_windows_tsv"])
        windows: list[dict[str, Any]] = []
        with windows_tsv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                label = row.get("task") or row.get("segment") or "UNK"
                windows.append(
                    {
                        "task": label,
                        "run": row.get("run", "01"),
                        "start_wall_clock": float(row.get("start_wall_clock", "0") or 0.0),
                        "end_wall_clock": float(row.get("end_wall_clock", "0") or 0.0),
                    }
                )
        media_summary = _split_media_runs(
            session_dir=output_session_dir,
            windows=windows,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            split_video=split_video,
            split_audio=split_audio,
            av_session_dir=av_session_dir if no_source_ingest else None,
            tobii_dirs=tobii_dirs if no_source_ingest else None,
            ffmpeg_threads=ffmpeg_threads,
        )

    removed_raw_artifacts: dict[str, Any] | None = None
    if processed_only:
        removed = {
            "sourcedata_removed": False,
            "xdf_removed": [],
        }
        sourcedata = output_session_dir / "sourcedata"
        if sourcedata.exists():
            shutil.rmtree(sourcedata)
            removed["sourcedata_removed"] = True
        for xdf in sorted(output_session_dir.glob("*.xdf")):
            try:
                xdf.unlink()
                removed["xdf_removed"].append(str(xdf))
            except Exception:
                pass
        removed_raw_artifacts = removed

    final_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_session_dir": str(output_session_dir),
        "mode": "link" if link else "copy",
        "source_counts": counts,
        "raw_to_bids_summary": str(raw_summary),
        "task_chunking_summary": chunk_summary,
        "stimuli_answers_summary": stimuli_answers_summary,
        "stimuli_answers_tsv": None if stimuli_answers_tsv is None else str(stimuli_answers_tsv),
        "participant_signal_map_tsv": str(participant_signal_map),
        "media_run_splitting": media_summary,
        "processed_only": bool(processed_only),
        "removed_raw_artifacts": removed_raw_artifacts,
    }
    out = output_session_dir / "annot" / "multisource_to_bids_runs_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge AV/Recording/Stimuli/Tobii raw folders and emit BIDS run chunks"
    )
    p.add_argument("--av-session-dir", type=Path, required=True)
    p.add_argument("--recording-session-dir", type=Path, required=True)
    p.add_argument("--stimuli-dir", type=Path, required=True)
    p.add_argument("--output-session-dir", type=Path, required=True)
    p.add_argument("--tobii-dir", type=Path, action="append", default=[])
    p.add_argument("--link", action="store_true")
    p.add_argument(
        "--split-media", action="store_true", help="Split AV/Tobii media into task run clips"
    )
    p.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable path")
    p.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable path")
    p.add_argument(
        "--skip-video-splitting",
        action="store_true",
        help="When --split-media is enabled, skip video clip generation and only keep video clock anchors.",
    )
    p.add_argument(
        "--skip-audio-splitting",
        action="store_true",
        help="When --split-media is enabled, skip audio clip generation.",
    )
    p.add_argument(
        "--allow-missing-xdf",
        action="store_true",
        help="Do not fail if XDF extraction cannot be performed.",
    )
    p.add_argument(
        "--no-source-ingest",
        action="store_true",
        help="Do not copy/link source raw folders into output; process directly from source paths.",
    )
    p.add_argument(
        "--processed-only",
        action="store_true",
        default=True,
        help="Keep only processed outputs in output session; remove copied raw sourcedata/*.xdf. Default: enabled.",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep sourcedata and copied XDF files in output session (overrides --processed-only default).",
    )
    p.add_argument(
        "--no-write-session-events",
        action="store_true",
        help="Do not replace session events.tsv from stimuli logs",
    )
    p.add_argument(
        "--task-windows-tsv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Use an existing task_run_windows.tsv instead of computing from stimuli events. "
             "Useful for sessions with incomplete stimuli logs.",
    )
    p.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        metavar="N",
        help="Number of threads per ffmpeg call (0 = ffmpeg default). "
             "Set to e.g. 4 to improve CPU utilization when workers < CPU count.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = merge_and_convert(
            av_session_dir=args.av_session_dir,
            recording_session_dir=args.recording_session_dir,
            stimuli_dir=args.stimuli_dir,
            output_session_dir=args.output_session_dir,
            tobii_dirs=list(args.tobii_dir),
            link=bool(args.link),
            write_session_events=not bool(args.no_write_session_events),
            split_media=bool(args.split_media),
            processed_only=(not bool(args.keep_raw)) and bool(args.processed_only),
            ffmpeg_bin=str(args.ffmpeg_bin),
            ffprobe_bin=str(args.ffprobe_bin),
            split_video=not bool(args.skip_video_splitting),
            split_audio=not bool(args.skip_audio_splitting),
            require_xdf_extraction=not bool(args.allow_missing_xdf),
            no_source_ingest=bool(args.no_source_ingest),
            override_windows_tsv=args.task_windows_tsv,
            ffmpeg_threads=int(args.ffmpeg_threads),
        )
        print(f"[multisource_to_bids_runs] summary: {summary}")
        return 0
    except Exception as exc:
        print(f"[multisource_to_bids_runs] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
