#!/usr/bin/env python3
"""Generate a comprehensive per-session metadata report.

Cross-checks **all** data sources (two recording pathways, XDF archive,
AV media, Tobii Pro raw recordings, stimuli events, central mic) and
attributes every stream to P1–P4 using device-serial configs.

Outputs a TSV with one row per session covering:
- participants (names),
- available streams per participant (Tobii LSL, EmotiBit LSL, Tobii raw video),
- task presence/duration from stimuli events,
- data availability across every source pathway,
- gaps / cross-check notes.

Usage::

    python tools/generate_session_metadata_report.py \\
        [--data-root data] \\
        [--output session_metadata_report.tsv]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("session_metadata_report")

# ── Device → participant maps ────────────────────────────────────────────

TOBII_SERIAL_TO_P: dict[str, str] = {}
EMOTIBIT_SERIAL_TO_P: dict[str, str] = {}
EMOTIBIT_IP_TO_P: dict[str, str] = {}


def _load_device_maps(cfg_root: Path) -> None:
    """Populate global serial-to-participant maps from config files."""
    # Tobii glasses serials
    tobii_cfg = cfg_root / "tobii_glasses_streams.yaml"
    if tobii_cfg.exists():
        with tobii_cfg.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for dev in data.get("devices", []):
            serial = dev.get("serial", "").strip()
            pid = dev.get("id", "").strip()
            if serial and pid:
                TOBII_SERIAL_TO_P[serial] = pid
        logger.info("Tobii serial map: %s", TOBII_SERIAL_TO_P)

    # EmotiBit device serials + IPs
    eb_cfg = cfg_root / "emotibit_participants_by_source.json"
    if eb_cfg.exists():
        data = json.loads(eb_cfg.read_text(encoding="utf-8"))
        for pid, hw in (data.get("participants") or {}).items():
            EMOTIBIT_SERIAL_TO_P[hw] = pid
        for ip, pid in (data.get("by_source") or {}).items():
            EMOTIBIT_IP_TO_P[ip] = pid
        logger.info("EmotiBit serial map: %s", EMOTIBIT_SERIAL_TO_P)
    else:
        eb_cfg2 = cfg_root / "emotibit_participants.json"
        if eb_cfg2.exists():
            data = json.loads(eb_cfg2.read_text(encoding="utf-8"))
            for pid, hw in data.items():
                EMOTIBIT_SERIAL_TO_P[hw] = pid


def _participant_from_serial(serial: str) -> str:
    """Look up participant from a Tobii or EmotiBit serial."""
    if serial in TOBII_SERIAL_TO_P:
        return TOBII_SERIAL_TO_P[serial]
    if serial in EMOTIBIT_SERIAL_TO_P:
        return EMOTIBIT_SERIAL_TO_P[serial]
    return ""


# ── Stimuli event parsing ────────────────────────────────────────────────

def _parse_float(v: str, default: float = -1.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]


def _read_stimuli_events(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def _task_durations_from_events(
    rows: list[dict[str, str]],
) -> dict[str, float]:
    """Derive per-task durations from stimuli experiment events."""
    task_walls: dict[str, list[float]] = {}
    for row in rows:
        task = (row.get("task") or "").strip().upper()
        wall = _parse_float(row.get("wall_clock", ""))
        if task in TASK_ORDER and wall >= 0:
            task_walls.setdefault(task, []).append(wall)
    durations: dict[str, float] = {}
    for task in TASK_ORDER:
        walls = task_walls.get(task, [])
        if len(walls) >= 2:
            durations[task] = max(walls) - min(walls)
    return durations


def _total_session_duration(rows: list[dict[str, str]]) -> float:
    walls = [_parse_float(r.get("wall_clock", "")) for r in rows]
    walls = [w for w in walls if w >= 0]
    return (max(walls) - min(walls)) if len(walls) >= 2 else 0.0


# ── Source discovery helpers ─────────────────────────────────────────────

def _find_recording_session_dir(
    data_root: Path, session_id: str
) -> Path | None:
    rec_root = data_root / "affectai-capture-recording" / "sessions"
    for phase in ("final", "test", "pilot"):
        d = rec_root / phase / "sub-01" / session_id
        if d.exists():
            return d
    return None


def _find_currentstudy_xdf(
    data_root: Path, group_id: str
) -> list[Path]:
    cs = data_root / "CurrentStudy"
    if not cs.exists():
        return []
    grp_num = group_id.replace("grp-", "")
    found: list[Path] = []
    for d in cs.iterdir():
        if d.is_dir() and (group_id in d.name or grp_num in d.name):
            for xdf in d.rglob("*.xdf"):
                if "_old" not in xdf.stem:
                    found.append(xdf)
    return sorted(set(found))


def _find_all_stimuli_experiment_tsvs(
    data_root: Path, session_info: dict
) -> list[Path]:
    """Find **all** stimuli experiment TSVs for a session (may span multiple dirs)."""
    stim_root = data_root / "affectai-capture-recording" / "stimuli" / "data"
    if not stim_root.exists():
        return []
    found: list[Path] = []
    candidates = session_info.get("stimuli_candidates", [])

    # Each candidate is a subdirectory name — look INSIDE it for experiment TSVs
    for cand in candidates:
        cand_dir = stim_root / cand
        if cand_dir.is_dir():
            found.extend(sorted(cand_dir.glob("events_*_experiment.tsv")))

    if found:
        return sorted(set(found))

    # Fallback: match subdirectories by group_id or session_id stem
    group_id = session_info.get("group_id", "")
    session_stem = session_info.get("session", "").replace("ses-", "").split("_run")[0]

    for subdir in sorted(stim_root.iterdir()):
        if not subdir.is_dir():
            continue
        if session_stem and session_stem in subdir.name:
            found.extend(sorted(subdir.glob("events_*_experiment.tsv")))
        elif group_id and group_id in subdir.name:
            found.extend(sorted(subdir.glob("events_*_experiment.tsv")))

    return sorted(set(found))


def _find_av_session_dir(
    data_root: Path, session_info: dict
) -> Path | None:
    av_root = data_root / "AV"
    session_id = session_info.get("session", "")
    for phase in ("final", "pilot", "test"):
        d = av_root / phase / "sub-01" / session_id
        if d.exists():
            return d
    return None


def _tobii_recordings_for_session(
    data_root: Path, session_info: dict
) -> list[dict[str, Any]]:
    """Match Tobii raw recordings to a session by candidate list."""
    tobii_root = data_root / "Tobii"
    if not tobii_root.exists():
        return []
    candidates = set(session_info.get("tobii_candidates", []))
    results = []
    for d in sorted(tobii_root.iterdir()):
        if not d.is_dir() or d.name == "aborted":
            continue
        if d.name not in candidates:
            continue
        ru = d / "meta" / "RuSerial"
        serial = ru.read_bytes().decode("utf-8", errors="replace").strip() if ru.exists() else ""
        participant = _participant_from_serial(serial)
        scene = d / "scenevideo.mp4"
        scene_mb = scene.stat().st_size / (1024 * 1024) if scene.exists() else 0
        gaze = d / "gazedata.gz"
        imu = d / "imudata.gz"
        results.append({
            "recording": d.name,
            "serial": serial,
            "participant": participant,
            "has_scene_video": scene.exists(),
            "scene_video_mb": round(scene_mb, 0),
            "has_gaze": gaze.exists(),
            "has_imu": imu.exists(),
        })
    return results


# ── XDF stream probing (lightweight — header only via pyxdf) ─────────────

def _probe_xdf_streams(xdf_path: Path) -> list[dict[str, str]]:
    """Load XDF and extract unique stream metadata (name, type, source_id, participant, sample_count, duration)."""
    try:
        import pyxdf
    except ImportError:
        return []
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        logger.warning("  XDF load failed %s: %s", xdf_path.name, exc)
        return []

    results: list[dict[str, str]] = []
    for s in streams:
        info = s.get("info", {})
        name = (info.get("name", [""])[0] if isinstance(info.get("name"), list)
                else info.get("name", ""))
        stype = (info.get("type", [""])[0] if isinstance(info.get("type"), list)
                 else info.get("type", ""))
        src_id = (info.get("source_id", [""])[0] if isinstance(info.get("source_id"), list)
                  else info.get("source_id", ""))

        stamps = s.get("time_stamps")
        n = len(stamps) if stamps is not None and hasattr(stamps, "__len__") else 0
        dur = 0.0
        if n > 1:
            dur = float(stamps[-1]) - float(stamps[0])

        # Attribute participant
        participant = ""
        # Tobii streams: name contains P1-P4 or source_id contains serial
        if re.search(r"Tobii[_\-]P([1-4])", name, re.IGNORECASE):
            m = re.search(r"P([1-4])", name)
            participant = f"P{m.group(1)}" if m else ""
        elif re.search(r"Emotibit[_\-]P([1-4])", name, re.IGNORECASE):
            m = re.search(r"P([1-4])", name)
            participant = f"P{m.group(1)}" if m else ""
        elif re.search(r"Participant[_\-]?([1-4])", name, re.IGNORECASE):
            m = re.search(r"([1-4])", name)
            participant = f"P{m.group(1)}" if m else ""
        else:
            # Try source_id serial lookup
            for serial, pid in {**TOBII_SERIAL_TO_P, **EMOTIBIT_SERIAL_TO_P}.items():
                if serial in str(src_id):
                    participant = pid
                    break

        results.append({
            "name": str(name),
            "type": str(stype),
            "source_id": str(src_id),
            "participant": participant,
            "sample_count": str(n),
            "duration_s": f"{dur:.1f}",
        })
    return results


def _classify_xdf_stream(name: str, stype: str) -> str:
    """Classify stream into broad category."""
    nl = name.lower()
    tl = stype.lower()
    if re.search(r"^tobii", nl):
        return "tobii_lsl"
    if re.search(r"^emotibit", nl):
        return "emotibit_aggregate"
    if re.search(r"^affectai", nl):
        return "markers"
    if re.search(r"^evetns_tobii|^events_tobii", nl):
        return "tobii_events"
    if tl == "eyetracking":
        return "tobii_lsl"
    if tl == "emotibit":
        return "emotibit_aggregate"
    if tl == "markers":
        return "markers"
    # EmotiBit individual sensor streams (EDA, PPG, ACC, GYRO, etc.)
    emotibit_types = {
        "eda", "ppggreen", "ppgred", "ppginfrared", "heartrate",
        "temperature", "thermopile", "accelerometerx", "accelerometery",
        "accelerometerz", "gyroscopex", "gyroscopey", "gyroscopez",
        "magnetometerx", "magnetometery", "magnetometerz",
        "scrfrequency", "scramplitude", "scrrisetime",
    }
    if tl in emotibit_types:
        return "emotibit_sensor"
    return "other"


# ── Row builder ──────────────────────────────────────────────────────────

REPORT_FIELDS = [
    "session",
    "group_id",
    "phase",
    "participants_names",
    # Schedule
    "schedule_date",
    "schedule_start_time",
    "schedule_end_time",
    "date_match",
    "names_match",
    # Per-participant Tobii LSL
    "tobii_lsl_P1", "tobii_lsl_P2", "tobii_lsl_P3", "tobii_lsl_P4",
    # Per-participant EmotiBit LSL (sensor streams)
    "emotibit_lsl_P1", "emotibit_lsl_P2", "emotibit_lsl_P3", "emotibit_lsl_P4",
    # Per-participant Tobii raw scene video (Tobii Pro G3)
    "tobii_video_P1", "tobii_video_P2", "tobii_video_P3", "tobii_video_P4",
    # Per-participant Tobii NDJSON (recording-session pathway)
    "tobii_ndjson_P1", "tobii_ndjson_P2", "tobii_ndjson_P3", "tobii_ndjson_P4",
    # Marker streams
    "markers_experiment", "markers_moderator", "markers_bigscreen",
    "markers_P1", "markers_P2", "markers_P3", "markers_P4",
    # AV
    "av_video_count", "av_audio_count",
    # Central mic
    "central_mic_files",
    # XDF sources
    "xdf_recording_session", "xdf_currentstudy",
    # Stimuli
    "stimuli_events_found",
    # Tasks
    "tasks_found",
    "T0_duration_s", "T1_duration_s", "T2_duration_s",
    "T3_duration_s", "T4_duration_s",
    "total_session_duration_s",
    # XDF stream counts by category
    "xdf_tobii_streams", "xdf_emotibit_sensor_streams",
    "xdf_emotibit_aggregate_streams", "xdf_marker_streams",
    "xdf_other_streams", "xdf_total_streams",
    # Duration from XDF
    "xdf_max_stream_duration_s",
    # Cross-check notes
    "notes",
]


def _load_schedule(cfg_root: Path) -> dict[str, dict[str, str]]:
    """Load session_schedule.tsv into a dict keyed by group_id."""
    path = cfg_root / "session_schedule.tsv"
    if not path.exists():
        return {}
    schedule: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gid = (r.get("group_id") or "").strip()
            if gid:
                schedule[gid] = {k: (v or "").strip() for k, v in r.items()}
    return schedule


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison (fix mojibake, NFC, lowercase)."""
    import unicodedata
    s = name.strip()
    # Fix double-encoded UTF-8 (mojibake): try latin-1 → utf-8 roundtrip
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return unicodedata.normalize("NFC", s.lower())


def _build_session_row(
    session_info: dict,
    data_root: Path,
    probe_xdf: bool = False,
    schedule: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Build one report row for a session."""
    row: dict[str, str] = {f: "" for f in REPORT_FIELDS}
    notes: list[str] = []

    session_id = session_info["session"]
    group_id = session_info.get("group_id", "")
    row["session"] = session_id
    row["group_id"] = group_id
    row["phase"] = ";".join(session_info.get("phase_tags", []))
    row["participants_names"] = "; ".join(session_info.get("participants_names", []))

    # ── Schedule cross-check ──
    sched = (schedule or {}).get(group_id)
    if sched:
        row["schedule_date"] = sched.get("date", "")
        row["schedule_start_time"] = sched.get("start_time", "")
        row["schedule_end_time"] = sched.get("end_time", "")
        # Date match
        raw_date = session_id.replace("ses-", "").split("_")[0]
        actual_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        if sched.get("date", "") == actual_date:
            row["date_match"] = "yes"
        else:
            row["date_match"] = "no"
            notes.append(f"date mismatch: scheduled {sched.get('date','')}, actual {actual_date}")
        # Names match
        sched_names = {_normalize_name(sched.get(f"name_{i}", "")) for i in range(1, 5)}
        sched_names.discard("")
        report_names = {_normalize_name(n) for n in session_info.get("participants_names", [])}
        report_names.discard("")
        if sched_names == report_names:
            row["names_match"] = "yes"
        elif sched_names & report_names:
            row["names_match"] = "partial"
            diff = sched_names.symmetric_difference(report_names)
            if diff:
                notes.append(f"name diff: {', '.join(sorted(diff))}")
        elif not report_names:
            row["names_match"] = "no_names"
        else:
            row["names_match"] = "mismatch"
            notes.append(f"names mismatch vs schedule")

    # ── Recording-session sourcedata ──
    rec_dir = _find_recording_session_dir(data_root, session_id)
    if rec_dir:
        sd = rec_dir / "sourcedata"

        # XDF from recording-session
        lsl_dir = sd / "lsl"
        rec_xdfs = sorted(lsl_dir.glob("*.xdf")) if lsl_dir.exists() else []
        row["xdf_recording_session"] = str(len(rec_xdfs))
        if not rec_xdfs:
            notes.append("no recording-session XDF")

        # Tobii NDJSON per participant
        tobii_lsl_dir = sd / "tobii_lsl"
        if tobii_lsl_dir.exists():
            for ndjson in sorted(tobii_lsl_dir.glob("*.ndjson")):
                # Filename is P1.ndjson, P2.ndjson, etc.
                pname = ndjson.stem.upper()
                if pname in ("P1", "P2", "P3", "P4"):
                    sz_mb = ndjson.stat().st_size / (1024 * 1024)
                    row[f"tobii_ndjson_{pname}"] = f"{sz_mb:.0f}MB"
    else:
        row["xdf_recording_session"] = "0"
        notes.append("no recording-session dir")

    # ── CurrentStudy XDF ──
    cs_xdfs = _find_currentstudy_xdf(data_root, group_id)
    row["xdf_currentstudy"] = str(len(cs_xdfs))

    # ── Probe XDF streams (from best available XDF) ──
    best_xdf = None
    if cs_xdfs:
        # Prefer largest CurrentStudy XDF
        best_xdf = max(cs_xdfs, key=lambda p: p.stat().st_size)
    elif rec_dir:
        lsl_dir = rec_dir / "sourcedata" / "lsl"
        rec_xdfs_full = sorted(lsl_dir.glob("*.xdf")) if lsl_dir.exists() else []
        if rec_xdfs_full:
            best_xdf = max(rec_xdfs_full, key=lambda p: p.stat().st_size)

    xdf_streams: list[dict[str, str]] = []
    if best_xdf and probe_xdf:
        logger.info("  Probing XDF: %s (%.1fMB)", best_xdf.name,
                     best_xdf.stat().st_size / (1024 * 1024))
        xdf_streams = _probe_xdf_streams(best_xdf)

    if xdf_streams:
        # Classify and count
        cats: dict[str, int] = {}
        max_dur = 0.0
        tobii_by_p: dict[str, bool] = {}
        emotibit_by_p: dict[str, int] = {}
        markers_by_role: dict[str, bool] = {}

        for s in xdf_streams:
            cat = _classify_xdf_stream(s["name"], s["type"])
            cats[cat] = cats.get(cat, 0) + 1
            dur = _parse_float(s["duration_s"])
            if dur > max_dur:
                max_dur = dur
            participant = s.get("participant", "")

            if cat == "tobii_lsl" and participant:
                tobii_by_p[participant] = True
            if cat in ("emotibit_sensor", "emotibit_aggregate") and participant:
                emotibit_by_p[participant] = emotibit_by_p.get(participant, 0) + 1
            if cat == "markers":
                nl = s["name"].lower()
                if "experiment" in nl:
                    markers_by_role["experiment"] = True
                elif "moderator" in nl:
                    markers_by_role["moderator"] = True
                elif "bigscreen" in nl:
                    markers_by_role["bigscreen"] = True
                elif "participant" in nl and participant:
                    markers_by_role[participant] = True

        for p in ("P1", "P2", "P3", "P4"):
            row[f"tobii_lsl_{p}"] = "yes" if tobii_by_p.get(p) else "no"
            cnt = emotibit_by_p.get(p, 0)
            row[f"emotibit_lsl_{p}"] = str(cnt) if cnt else "no"

        row["markers_experiment"] = "yes" if markers_by_role.get("experiment") else "no"
        row["markers_moderator"] = "yes" if markers_by_role.get("moderator") else "no"
        row["markers_bigscreen"] = "yes" if markers_by_role.get("bigscreen") else "no"
        for p in ("P1", "P2", "P3", "P4"):
            row[f"markers_{p}"] = "yes" if markers_by_role.get(p) else "no"

        row["xdf_tobii_streams"] = str(cats.get("tobii_lsl", 0))
        row["xdf_emotibit_sensor_streams"] = str(cats.get("emotibit_sensor", 0))
        row["xdf_emotibit_aggregate_streams"] = str(cats.get("emotibit_aggregate", 0))
        row["xdf_marker_streams"] = str(cats.get("markers", 0))
        row["xdf_other_streams"] = str(cats.get("other", 0) + cats.get("tobii_events", 0))
        row["xdf_total_streams"] = str(len(xdf_streams))
        row["xdf_max_stream_duration_s"] = f"{max_dur:.0f}"

        # Check for missing participants
        for p in ("P1", "P2", "P3", "P4"):
            if not tobii_by_p.get(p):
                notes.append(f"missing Tobii LSL {p}")
            if not emotibit_by_p.get(p):
                notes.append(f"missing EmotiBit {p}")

    elif best_xdf:
        # XDF exists but not probed
        xdf_sz = best_xdf.stat().st_size / (1024 * 1024)
        row["xdf_total_streams"] = f"(not probed, {xdf_sz:.0f}MB)"

    # ── Tobii raw recordings (scene videos) ──
    tobii_recs = _tobii_recordings_for_session(data_root, session_info)
    for rec in tobii_recs:
        p = rec.get("participant", "")
        if p in ("P1", "P2", "P3", "P4") and rec.get("has_scene_video"):
            existing = row.get(f"tobii_video_{p}", "")
            entry = f"{rec['recording']}({rec['scene_video_mb']:.0f}MB)"
            row[f"tobii_video_{p}"] = f"{existing};{entry}" if existing else entry
    # Check for missing Tobii video
    for p in ("P1", "P2", "P3", "P4"):
        if not row.get(f"tobii_video_{p}"):
            # Check if we have tobii candidates at all
            if session_info.get("tobii_candidate_count", 0) > 0:
                notes.append(f"tobii video {p} unmatched")

    # ── AV media ──
    av_dir = _find_av_session_dir(data_root, session_info)
    if av_dir:
        vids = list(av_dir.rglob("*.mkv")) + list(av_dir.rglob("*.mp4"))
        auds = list(av_dir.rglob("*.wav"))
        row["av_video_count"] = str(len(vids))
        row["av_audio_count"] = str(len(auds))
    else:
        row["av_video_count"] = "0"
        row["av_audio_count"] = "0"
        if session_info.get("raw_av"):
            notes.append("inventory says AV but dir not found")

    # ── Central mic ──
    row["central_mic_files"] = str(session_info.get("central_mic_file_count", 0))

    # ── Stimuli events (tasks & durations) — merge ALL candidate dirs ──
    stim_tsvs = _find_all_stimuli_experiment_tsvs(data_root, session_info)
    if stim_tsvs:
        row["stimuli_events_found"] = "yes"
        # Compute per-task durations INDEPENDENTLY per TSV, then keep
        # the longest clean run per task (avoids inflated wall-clock spans
        # across app restarts).
        merged_durs: dict[str, float] = {}
        best_total: float = 0.0
        for tsv in stim_tsvs:
            events = _read_stimuli_events(tsv)
            durs = _task_durations_from_events(events)
            for task, dur in durs.items():
                if dur > merged_durs.get(task, 0.0):
                    merged_durs[task] = dur
            total = _total_session_duration(events)
            if total > best_total:
                best_total = total

        tasks_present = sorted(merged_durs.keys())
        row["tasks_found"] = ";".join(tasks_present)
        for task in TASK_ORDER:
            if task in merged_durs:
                row[f"{task}_duration_s"] = f"{merged_durs[task]:.0f}"
        row["total_session_duration_s"] = f"{best_total:.0f}"

        missing_tasks = [t for t in TASK_ORDER if t not in merged_durs]
        if missing_tasks:
            notes.append(f"missing tasks: {','.join(missing_tasks)}")
        if len(stim_tsvs) > 1:
            notes.append(f"stimuli merged from {len(stim_tsvs)} dirs")
    else:
        row["stimuli_events_found"] = "no"
        notes.append("no stimuli experiment events")

    row["notes"] = "; ".join(notes) if notes else ""
    return row


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive session metadata report TSV",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "metadata" / "session_metadata_report.tsv",
    )
    parser.add_argument(
        "--probe-xdf", action="store_true",
        help="Load XDF files to probe individual stream metadata (slow but thorough)",
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Process only these session IDs",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    cfg_root = data_root.parent / "configs"
    logger.info("Data root: %s", data_root)
    logger.info("Config root: %s", cfg_root)

    # Load device maps and schedule
    _load_device_maps(cfg_root)
    schedule = _load_schedule(cfg_root)
    logger.info("Schedule: %d groups loaded", len(schedule))

    # Load inventory
    inv_path = data_root / "high_level_data_inventory.json"
    if not inv_path.exists():
        logger.error("Inventory not found: %s", inv_path)
        return 1
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    sessions = inventory.get("sessions", [])

    if args.sessions:
        sessions = [s for s in sessions if s["session"] in args.sessions]

    logger.info("Processing %d sessions (probe_xdf=%s)", len(sessions), args.probe_xdf)

    rows: list[dict[str, str]] = []
    for i, session_info in enumerate(sessions, 1):
        sid = session_info["session"]
        logger.info("[%d/%d] %s", i, len(sessions), sid)
        row = _build_session_row(session_info, data_root, probe_xdf=args.probe_xdf, schedule=schedule)
        rows.append(row)

    # Write TSV
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Report written: %s (%d rows)", output, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
