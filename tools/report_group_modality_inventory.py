#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MEDIA_EXTS = {
    "video": {".mp4", ".mkv", ".mov", ".avi"},
    "audio": {".wav", ".flac", ".aac", ".m4a", ".mp3"},
}


@dataclass
class ModalitySummary:
    available: bool
    file_count: int
    total_bytes: int
    duration_s: float | None


def _parse_group_id(session_name: str) -> str:
    m = re.search(r"_(grp-[^_]+)_", session_name)
    return m.group(1) if m else "unknown"


def _parse_session_tokens(session_name: str) -> tuple[str | None, str | None, str | None]:
    m = re.match(r"ses-(\d{8})_(grp-[^_]+)_(run\d+)$", session_name)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def _group_to_num(group_id: str) -> str | None:
    m = re.search(r"grp-(\d+)", group_id)
    if not m:
        return None
    return f"{int(m.group(1)):02d}"


def _read_schedule(schedule_path: Path) -> dict[str, list[str]]:
    if not schedule_path.exists():
        return {}
    mapping: dict[str, list[str]] = {}
    with schedule_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gid = (row.get("group_id") or "").strip()
            if not gid:
                continue
            names = []
            for idx in range(1, 5):
                val = (row.get(f"name_{idx}") or "").strip()
                if val:
                    names.append(val)
            mapping[gid] = names
    return mapping


def _to_tobii_date_keys(yyyymmdd: str) -> tuple[str, str]:
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    return yyyymmdd, dt.strftime("%y-%m-%d")


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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    try:
        return float(out)
    except Exception:
        return None


def _read_tsv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _duration_from_lsl_table(path: Path) -> float | None:
    gz = path.suffix.lower() == ".gz"
    opener = gzip.open if gz else open
    first: float | None = None
    last: float | None = None
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "lsl_time" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                t = float(row.get("lsl_time", ""))
            except Exception:
                continue
            if first is None:
                first = t
            last = t
    if first is None or last is None:
        return None
    return max(0.0, last - first)


def _duration_from_events(events_tsv: Path) -> float | None:
    if not events_tsv.exists():
        return None
    onsets: list[float] = []
    max_end = 0.0
    has_any = False
    for row in _read_tsv_dicts(events_tsv):
        try:
            onset = float(row.get("onset", "0") or 0.0)
            dur = float(row.get("duration", "0") or 0.0)
        except Exception:
            continue
        has_any = True
        onsets.append(onset)
        max_end = max(max_end, onset + dur)
    if not has_any:
        return None
    if not onsets:
        return max_end
    absolute_onsets = [v for v in onsets if v > 1_000_000.0]
    # Some events.tsv files mix a 0.0 init marker with absolute wall-clock onsets.
    if absolute_onsets:
        return max(0.0, max_end - min(absolute_onsets))
    return max_end


def _duration_from_windows(session_dir: Path) -> float | None:
    annot = session_dir / "annot"
    if not annot.exists():
        return None
    windows = sorted(annot.glob("*_task_run_windows.tsv"))
    if not windows:
        return None
    total = 0.0
    has_any = False
    for row in _read_tsv_dicts(windows[0]):
        try:
            total += float(row.get("duration_s", "0") or 0.0)
            has_any = True
        except Exception:
            continue
    return total if has_any else None


def _scan_modality(session_dir: Path, modality: str, ffprobe_bin: str, session_duration_s: float | None) -> ModalitySummary:
    root = session_dir / modality
    if not root.exists():
        return ModalitySummary(False, 0, 0, None)

    files = [p for p in root.rglob("*") if p.is_file()]
    file_count = len(files)
    total_bytes = sum(p.stat().st_size for p in files)
    available = file_count > 0
    duration_s: float | None = None

    if not available:
        return ModalitySummary(False, 0, 0, None)

    if modality in {"video", "audio"}:
        exts = MEDIA_EXTS[modality]
        media_files = [p for p in files if p.suffix.lower() in exts]
        durs = [_ffprobe_duration(p, ffprobe_bin) for p in media_files[:8]]
        durs = [d for d in durs if d is not None]
        duration_s = max(durs) if durs else session_duration_s
    elif modality == "et":
        candidates = sorted(root.glob("*_acq-lsl_tobii.tsv.gz"))
        if candidates:
            duration_s = _duration_from_lsl_table(candidates[0])
        if duration_s is None:
            duration_s = session_duration_s
    elif modality == "physio":
        candidates = sorted(root.glob("*_acq-lsl_emotibit.tsv.gz"))
        if candidates:
            duration_s = _duration_from_lsl_table(candidates[0])
        if duration_s is None:
            duration_s = session_duration_s
    elif modality == "beh":
        events_files = sorted(root.glob("*_events.tsv"))
        if events_files:
            duration_s = _duration_from_events(events_files[0])
        if duration_s is None:
            duration_s = session_duration_s
    else:
        duration_s = session_duration_s

    return ModalitySummary(available, file_count, total_bytes, duration_s)


def _scan_raw_modality(session_dir: Path, raw_name: str, session_duration_s: float | None) -> ModalitySummary:
    root = session_dir / "sourcedata" / raw_name
    if not root.exists():
        return ModalitySummary(False, 0, 0, None)
    files = [p for p in root.rglob("*") if p.is_file()]
    file_count = len(files)
    total_bytes = sum(p.stat().st_size for p in files)
    return ModalitySummary(file_count > 0, file_count, total_bytes, session_duration_s)


def _count_files_and_bytes(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _scan_session_root_modalities(session_dir: Path, ffprobe_bin: str, session_duration_s: float | None) -> dict[str, ModalitySummary]:
    return {
        m: _scan_modality(session_dir, m, ffprobe_bin, session_duration_s)
        for m in ["video", "audio", "et", "physio", "eeg", "mocap", "beh", "annot"]
    }


def _scan_stimuli_matches(stimuli_root: Path, session_name: str) -> list[Path]:
    yyyymmdd, group_id, run = _parse_session_tokens(session_name)
    if not yyyymmdd or not group_id or not run:
        return []
    group_num = _group_to_num(group_id)
    if group_num is None:
        return []
    key = f"{yyyymmdd}_grp-{group_num}_{run}"
    return sorted(p for p in stimuli_root.glob(f"{key}_*") if p.is_dir())


def _index_currentstudy_by_group(currentstudy_root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not currentstudy_root.exists():
        return out
    for subdir in sorted(p for p in currentstudy_root.iterdir() if p.is_dir()):
        name = subdir.name.lower()
        m = re.search(r"grp[-_ ]?(\d+)", name)
        if not m:
            continue
        grp = f"grp-{int(m.group(1)):02d}"
        xdfs = list(subdir.rglob("*.xdf"))
        if grp not in out:
            out[grp] = {"folder_count": 0, "xdf_count": 0}
        out[grp]["folder_count"] += 1
        out[grp]["xdf_count"] += len(xdfs)
    return out


def _scan_tobii_date_matches(tobii_root: Path, yyyymmdd: str | None) -> list[Path]:
    if yyyymmdd is None or not tobii_root.exists():
        return []
    key_a, key_b = _to_tobii_date_keys(yyyymmdd)
    matched: list[Path] = []
    for p in sorted(x for x in tobii_root.iterdir() if x.is_dir()):
        n = p.name
        if n.startswith(key_a) or n.startswith(key_b):
            matched.append(p)
    return matched


def _central_mic_summary(central_mic_root: Path) -> tuple[bool, int, int]:
    if not central_mic_root.exists():
        return False, 0, 0
    files = [p for p in central_mic_root.iterdir() if p.is_file()]
    return len(files) > 0, len(files), sum(p.stat().st_size for p in files)


def _load_participant_ids(session_dir: Path) -> list[str]:
    path = session_dir / "participants.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    seats = payload.get("seats", []) if isinstance(payload, dict) else []
    ids: list[str] = []
    for seat in seats:
        if not isinstance(seat, dict):
            continue
        pid = str(seat.get("participant_id", "")).strip()
        if pid:
            ids.append(pid)
    return ids


def build_inventory(
    sessions_root: Path,
    av_sessions_root: Path,
    stimuli_root: Path,
    currentstudy_root: Path,
    tobii_root: Path,
    central_mic_root: Path,
    schedule_tsv: Path,
    out_csv: Path,
    out_json: Path,
    ffprobe_bin: str,
) -> dict[str, Any]:
    schedule = _read_schedule(schedule_tsv)
    rec_sessions = {
        p.name: p for p in sessions_root.iterdir() if p.is_dir() and p.name.startswith("ses-")
    } if sessions_root.exists() else {}
    av_sessions = {
        p.name: p for p in av_sessions_root.iterdir() if p.is_dir() and p.name.startswith("ses-")
    } if av_sessions_root.exists() else {}
    all_session_names = sorted(set(rec_sessions) | set(av_sessions))
    currentstudy_index = _index_currentstudy_by_group(currentstudy_root)
    central_mic_available, central_mic_file_count, central_mic_total_bytes = _central_mic_summary(central_mic_root)

    records: list[dict[str, Any]] = []
    for session_name in all_session_names:
        rec_dir = rec_sessions.get(session_name)
        av_dir = av_sessions.get(session_name)
        session_dir = rec_dir or av_dir
        if session_dir is None:
            continue

        group_id = _parse_group_id(session_name)
        participant_ids = _load_participant_ids(session_dir)
        participant_names = schedule.get(group_id, [])
        session_date, _, _ = _parse_session_tokens(session_name)

        session_duration: float | None = None
        if rec_dir is not None:
            session_duration = _duration_from_windows(rec_dir)
            if session_duration is None:
                session_duration = _duration_from_events(rec_dir / "events.tsv")
        if session_duration is None and av_dir is not None:
            session_duration = _duration_from_windows(av_dir)
            if session_duration is None:
                session_duration = _duration_from_events(av_dir / "events.tsv")

        recording_modalities = _scan_session_root_modalities(rec_dir, ffprobe_bin, session_duration) if rec_dir else {}
        av_modalities = _scan_session_root_modalities(av_dir, ffprobe_bin, session_duration) if av_dir else {}

        # Canonical modality availability: prefer recording-final if present, otherwise AV-final.
        modalities: dict[str, ModalitySummary] = {}
        for m in ["video", "audio", "et", "physio", "eeg", "mocap", "beh", "annot"]:
            if m in recording_modalities and recording_modalities[m].available:
                modalities[m] = recording_modalities[m]
            elif m in av_modalities:
                modalities[m] = av_modalities[m]
            elif m in recording_modalities:
                modalities[m] = recording_modalities[m]
            else:
                modalities[m] = ModalitySummary(False, 0, 0, None)

        recording_raw_modalities = {
            f"recording_raw_{m}": _scan_raw_modality(rec_dir, m, session_duration)
            for m in ["av", "lsl", "tobii_lsl", "sync"]
        } if rec_dir else {
            f"recording_raw_{m}": ModalitySummary(False, 0, 0, None)
            for m in ["av", "lsl", "tobii_lsl", "sync"]
        }
        av_raw_modalities = {
            f"av_raw_{m}": _scan_raw_modality(av_dir, m, session_duration)
            for m in ["av", "lsl", "tobii_lsl", "sync"]
        } if av_dir else {
            f"av_raw_{m}": ModalitySummary(False, 0, 0, None)
            for m in ["av", "lsl", "tobii_lsl", "sync"]
        }

        stimuli_matches = _scan_stimuli_matches(stimuli_root, session_name)
        tobii_matches = _scan_tobii_date_matches(tobii_root, session_date)
        currentstudy_info = currentstudy_index.get(group_id, {"folder_count": 0, "xdf_count": 0})
        tobii_scenevideo_count = sum((p / "scenevideo.mp4").exists() for p in tobii_matches)
        tobii_gazedata_count = sum((p / "gazedata.gz").exists() for p in tobii_matches)
        tobii_note = "date-only match"
        if len(tobii_matches) == 0:
            tobii_note = "none"
        elif len(tobii_matches) == 1:
            tobii_note = "single candidate"

        rec: dict[str, Any] = {
            "session": session_name,
            "group_id": group_id,
            "participants_ids": participant_ids,
            "participants_names": participant_names,
            "session_duration_s": session_duration,
            "recording_session_exists": rec_dir is not None,
            "av_session_exists": av_dir is not None,
            "stimuli_match_count": len(stimuli_matches),
            "stimuli_matched_runs": [p.name for p in stimuli_matches],
            "currentstudy_group_folder_count": currentstudy_info["folder_count"],
            "currentstudy_group_xdf_count": currentstudy_info["xdf_count"],
            "tobii_date_match_count": len(tobii_matches),
            "tobii_scenevideo_count": tobii_scenevideo_count,
            "tobii_gazedata_count": tobii_gazedata_count,
            "tobii_match_note": tobii_note,
            "central_mic_available": central_mic_available,
            "central_mic_file_count_total": central_mic_file_count,
            "central_mic_total_bytes": central_mic_total_bytes,
            "central_mic_match_note": "global pool; no session key in filenames",
        }

        for m, info in modalities.items():
            rec[f"{m}_available"] = info.available
            rec[f"{m}_file_count"] = info.file_count
            rec[f"{m}_total_bytes"] = info.total_bytes
            rec[f"{m}_duration_s"] = info.duration_s

        for m, info in recording_raw_modalities.items():
            rec[f"{m}_available"] = info.available
            rec[f"{m}_file_count"] = info.file_count
            rec[f"{m}_total_bytes"] = info.total_bytes
            rec[f"{m}_duration_s"] = info.duration_s

        for m, info in av_raw_modalities.items():
            rec[f"{m}_available"] = info.available
            rec[f"{m}_file_count"] = info.file_count
            rec[f"{m}_total_bytes"] = info.total_bytes
            rec[f"{m}_duration_s"] = info.duration_s

        records.append(rec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "session",
        "group_id",
        "participants_ids",
        "participants_names",
        "session_duration_s",
        "recording_session_exists",
        "av_session_exists",
        "stimuli_match_count",
        "stimuli_matched_runs",
        "currentstudy_group_folder_count",
        "currentstudy_group_xdf_count",
        "tobii_date_match_count",
        "tobii_scenevideo_count",
        "tobii_gazedata_count",
        "tobii_match_note",
        "central_mic_available",
        "central_mic_file_count_total",
        "central_mic_total_bytes",
        "central_mic_match_note",
    ]
    for m in ["video", "audio", "et", "physio", "eeg", "mocap", "beh", "annot"]:
        csv_fields.extend([
            f"{m}_available",
            f"{m}_file_count",
            f"{m}_total_bytes",
            f"{m}_duration_s",
        ])
    for m in ["recording_raw_av", "recording_raw_lsl", "recording_raw_tobii_lsl", "recording_raw_sync"]:
        csv_fields.extend([
            f"{m}_available",
            f"{m}_file_count",
            f"{m}_total_bytes",
            f"{m}_duration_s",
        ])
    for m in ["av_raw_av", "av_raw_lsl", "av_raw_tobii_lsl", "av_raw_sync"]:
        csv_fields.extend([
            f"{m}_available",
            f"{m}_file_count",
            f"{m}_total_bytes",
            f"{m}_duration_s",
        ])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["participants_ids"] = ";".join(rec.get("participants_ids", []))
            row["participants_names"] = ";".join(rec.get("participants_names", []))
            row["stimuli_matched_runs"] = ";".join(rec.get("stimuli_matched_runs", []))
            writer.writerow(row)

    summary = {
        "sessions_root": str(sessions_root),
        "av_sessions_root": str(av_sessions_root),
        "stimuli_root": str(stimuli_root),
        "currentstudy_root": str(currentstudy_root),
        "tobii_root": str(tobii_root),
        "central_mic_root": str(central_mic_root),
        "schedule_tsv": str(schedule_tsv),
        "session_count": len(records),
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build per-session group/modality inventory with availability, duration and participant names"
    )
    p.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("data/affectai-capture-recording/sessions/final/sub-01"),
        help="Directory containing recording-final ses-* folders",
    )
    p.add_argument(
        "--av-sessions-root",
        type=Path,
        default=Path("data/AV/final/sub-01"),
        help="Directory containing AV-final ses-* folders",
    )
    p.add_argument(
        "--stimuli-root",
        type=Path,
        default=Path("data/affectai-capture-recording/stimuli/data"),
        help="Stimuli logs root folder",
    )
    p.add_argument(
        "--currentstudy-root",
        type=Path,
        default=Path("data/CurrentStudy"),
        help="Legacy/duplicate CurrentStudy root",
    )
    p.add_argument(
        "--tobii-root",
        type=Path,
        default=Path("data/Tobii"),
        help="Tobii device export root",
    )
    p.add_argument(
        "--central-mic-root",
        type=Path,
        default=Path("data/Central mic"),
        help="Central microphone root",
    )
    p.add_argument(
        "--schedule-tsv",
        type=Path,
        default=Path("data/affectai-capture-recording/configs/session_schedule.tsv"),
        help="Schedule TSV used to map group_id -> participant names",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/affectai-capture-recording/sessions/final/group_modality_inventory.csv"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/affectai-capture-recording/sessions/final/group_modality_inventory.json"),
    )
    p.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable path")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = build_inventory(
            sessions_root=args.sessions_root,
            av_sessions_root=args.av_sessions_root,
            stimuli_root=args.stimuli_root,
            currentstudy_root=args.currentstudy_root,
            tobii_root=args.tobii_root,
            central_mic_root=args.central_mic_root,
            schedule_tsv=args.schedule_tsv,
            out_csv=args.out_csv,
            out_json=args.out_json,
            ffprobe_bin=str(args.ffprobe_bin),
        )
    except Exception as exc:
        print(f"[report_group_modality_inventory] ERROR: {exc}")
        return 1

    print(
        "[report_group_modality_inventory] "
        f"sessions={summary.get('session_count', 0)} "
        f"csv={args.out_csv} json={args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
