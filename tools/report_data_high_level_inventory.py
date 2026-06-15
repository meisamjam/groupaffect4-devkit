#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROCESSED_MODALITIES = ["video", "audio", "et", "physio", "eeg", "mocap", "beh", "annot"]
RAW_MODALITIES = ["av", "lsl", "tobii_lsl", "sync"]
TRACKED_SPLITS = ["pilot", "test", "final"]


def _parse_group_id(session_name: str) -> str:
    m = re.search(r"_(grp-[^_]+)_", session_name)
    return m.group(1) if m else "unknown"


def _normalize_group_token(token: str) -> str:
    t = token.strip().lower().replace("group-", "grp-").replace("gpr-", "grp-")
    m_num = re.match(r"grp-(\d+)$", t)
    if m_num:
        return f"grp-{int(m_num.group(1)):02d}"
    m_alpha = re.match(r"grp-([a-z])$", t)
    if m_alpha:
        return f"grp-{m_alpha.group(1).upper()}"
    return token


def _normalize_run_token(token: str) -> str:
    m = re.match(r"run(\d+)$", token.lower())
    if not m:
        return token
    return f"run{int(m.group(1)):02d}"


def _session_signature(session_name: str) -> tuple[str, str, str] | None:
    m = re.match(r"^ses-(\d{8})_((?:grp|gpr|group)-[^_]+)_(run\d+)$", session_name, flags=re.IGNORECASE)
    if not m:
        return None
    date_token = m.group(1)
    group_token = _normalize_group_token(m.group(2))
    run_token = _normalize_run_token(m.group(3))
    return (date_token, group_token, run_token)


def _signature_to_session_name(sig: tuple[str, str, str]) -> str:
    return f"ses-{sig[0]}_{sig[1]}_{sig[2]}"


def _group_for_stimuli(group_id: str) -> str | None:
    m_num = re.match(r"grp-(\d+)$", group_id)
    if m_num:
        return f"{int(m_num.group(1)):02d}"
    m_alpha = re.match(r"grp-([A-Z])$", group_id)
    if m_alpha:
        return m_alpha.group(1)
    return None


def _scan_stimuli_candidates(stimuli_root: Path, session_sig: tuple[str, str, str] | None) -> list[str]:
    if session_sig is None or not stimuli_root.exists():
        return []
    date_token, group_id, run_token = session_sig
    grp_token = _group_for_stimuli(group_id)
    if grp_token is None:
        return []
    run_num = re.sub(r"^run0*", "", run_token)
    pat = re.compile(rf"^{date_token}_grp-{re.escape(grp_token)}_run0*{run_num}_.+$", flags=re.IGNORECASE)
    return sorted(p.name for p in stimuli_root.iterdir() if p.is_dir() and pat.match(p.name))


def _scan_tobii_candidates(tobii_root: Path, date_token: str | None) -> list[str]:
    if not date_token or not tobii_root.exists():
        return []
    short = f"{date_token[2:4]}-{date_token[4:6]}-{date_token[6:8]}"
    out: list[str] = []
    for p in tobii_root.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n.startswith(date_token) or n.startswith(short):
            out.append(n)
    return sorted(out)


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


def _central_mic_summary(central_mic_root: Path) -> tuple[bool, int]:
    if not central_mic_root.exists():
        return False, 0
    files = [p for p in central_mic_root.iterdir() if p.is_file()]
    return len(files) > 0, len(files)


def _read_schedule(schedule_path: Path) -> dict[str, list[str]]:
    if not schedule_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with schedule_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gid = (row.get("group_id") or "").strip()
            if not gid:
                continue
            names: list[str] = []
            for idx in range(1, 5):
                val = (row.get(f"name_{idx}") or "").strip()
                if val:
                    names.append(val)
            out[gid] = names
    return out


def _list_session_dirs(root: Path, split: str) -> dict[str, Path]:
    split_root = root / split / "sub-01"
    if not split_root.exists():
        return {}
    return {
        p.name: p
        for p in split_root.iterdir()
        if p.is_dir() and p.name.startswith("ses-")
    }


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


def _has_any_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for _ in path.rglob("*"):
        return True
    return False


def _scan_modalities(session_dir: Path) -> tuple[list[str], list[str]]:
    processed: list[str] = []
    raw: list[str] = []
    for m in PROCESSED_MODALITIES:
        if _has_any_file(session_dir / m):
            processed.append(m)

    sourcedata = session_dir / "sourcedata"
    for m in RAW_MODALITIES:
        if _has_any_file(sourcedata / m):
            raw.append(m)

    # AV sessions often store raw media under sourcedata/<capture_id>/{video,audio,...}
    # instead of sourcedata/av. If such capture folders exist, mark raw AV present.
    if "av" not in raw and sourcedata.exists():
        for cap_dir in [p for p in sourcedata.iterdir() if p.is_dir()]:
            if _has_any_file(cap_dir / "video") or _has_any_file(cap_dir / "audio"):
                raw.append("av")
                break

            # Some layouts nest sync as sourcedata/<capture_id>/sourcedata/sync.
            if "sync" not in raw and _has_any_file(cap_dir / "sourcedata" / "sync"):
                raw.append("sync")

            # Some layouts place lsl logs directly under sourcedata/<capture_id>/lsl.
            if "lsl" not in raw and _has_any_file(cap_dir / "lsl"):
                raw.append("lsl")

    raw = sorted(set(raw))
    return processed, raw


def _pick_participant_source(
    rec_dirs: dict[str, Path],
    av_dirs: dict[str, Path],
) -> list[str]:
    for split in ["final", "pilot", "test"]:
        if split in rec_dirs:
            ids = _load_participant_ids(rec_dirs[split])
            if ids:
                return ids
    for split in ["final", "pilot", "test"]:
        if split in av_dirs:
            ids = _load_participant_ids(av_dirs[split])
            if ids:
                return ids
    return []


def _collect_participant_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        ids.update(_load_participant_ids(p))
    return ids


def _compute_match_quality(
    rec_names: set[str],
    av_names: set[str],
    rec_paths: list[Path],
    av_paths: list[Path],
    by_signature: bool,
) -> tuple[str, list[str]]:
    cues: list[str] = []
    exact_name_match = len(rec_names.intersection(av_names)) > 0
    if exact_name_match:
        cues.append("exact_session_name")
    if by_signature:
        cues.append("normalized_date_group_run")

    rec_ids = _collect_participant_ids(rec_paths)
    av_ids = _collect_participant_ids(av_paths)
    if rec_ids and av_ids and rec_ids.intersection(av_ids):
        cues.append("participant_id_overlap")

    has_both_sources = bool(rec_names) and bool(av_names)
    if not has_both_sources:
        return "single-source", cues
    if exact_name_match or ("normalized_date_group_run" in cues and "participant_id_overlap" in cues):
        return "high", cues
    if "normalized_date_group_run" in cues:
        return "medium", cues
    return "low", cues


def build_inventory(
    recording_root: Path,
    av_root: Path,
    stimuli_root: Path,
    tobii_root: Path,
    currentstudy_root: Path,
    central_mic_root: Path,
    schedule_tsv: Path,
    out_session_csv: Path,
    out_group_csv: Path,
    out_json: Path,
) -> dict[str, Any]:
    schedule = _read_schedule(schedule_tsv)

    rec_index: dict[str, dict[str, Path]] = defaultdict(dict)
    av_index: dict[str, dict[str, Path]] = defaultdict(dict)
    currentstudy_index = _index_currentstudy_by_group(currentstudy_root)
    central_mic_available, central_mic_file_count = _central_mic_summary(central_mic_root)

    for split in TRACKED_SPLITS:
        for sname, sdir in _list_session_dirs(recording_root, split).items():
            rec_index[sname][split] = sdir
        for sname, sdir in _list_session_dirs(av_root, split).items():
            av_index[sname][split] = sdir

    # Build robust canonical grouping: exact names first, then signature-based aliases.
    sig_to_rec_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    sig_to_av_names: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for name in rec_index.keys():
        sig = _session_signature(name)
        if sig is not None:
            sig_to_rec_names[sig].add(name)
    for name in av_index.keys():
        sig = _session_signature(name)
        if sig is not None:
            sig_to_av_names[sig].add(name)

    canonical_items: list[tuple[str, set[str], set[str], bool]] = []
    consumed_rec: set[str] = set()
    consumed_av: set[str] = set()

    all_sigs = sorted(set(sig_to_rec_names.keys()) | set(sig_to_av_names.keys()))
    for sig in all_sigs:
        rec_names = set(sig_to_rec_names.get(sig, set()))
        av_names = set(sig_to_av_names.get(sig, set()))
        if not rec_names and not av_names:
            continue
        canonical_items.append((_signature_to_session_name(sig), rec_names, av_names, True))
        consumed_rec.update(rec_names)
        consumed_av.update(av_names)

    for name in sorted(set(rec_index.keys()) - consumed_rec):
        canonical_items.append((name, {name}, set(), False))
    for name in sorted(set(av_index.keys()) - consumed_av):
        canonical_items.append((name, set(), {name}, False))

    session_records: list[dict[str, Any]] = []

    for session, rec_names, av_names, by_signature in canonical_items:
        rec_dirs: dict[str, Path] = {}
        av_dirs: dict[str, Path] = {}
        for name in rec_names:
            rec_dirs.update(rec_index.get(name, {}))
        for name in av_names:
            av_dirs.update(av_index.get(name, {}))

        group_id = _parse_group_id(session)
        session_sig = _session_signature(session)
        session_date = session_sig[0] if session_sig else None
        participants_ids = _pick_participant_source(rec_dirs, av_dirs)
        participants_names = schedule.get(group_id, [])

        stimuli_candidates = _scan_stimuli_candidates(stimuli_root, session_sig)
        tobii_candidates = _scan_tobii_candidates(tobii_root, session_date)
        currentstudy_info = currentstudy_index.get(group_id, {"folder_count": 0, "xdf_count": 0})

        processed_available = set()
        raw_available = set()

        for split, sdir in rec_dirs.items():
            processed, raw = _scan_modalities(sdir)
            processed_available.update(processed)
            raw_available.update(raw)

        for split, sdir in av_dirs.items():
            processed, raw = _scan_modalities(sdir)
            processed_available.update(processed)
            raw_available.update(raw)

        rec_splits = sorted(rec_dirs.keys())
        av_splits = sorted(av_dirs.keys())
        phase_tags = sorted(set(rec_splits) | set(av_splits))

        rec_paths = [rec_dirs[k] for k in rec_splits]
        av_paths = [av_dirs[k] for k in av_splits]
        match_quality, match_cues = _compute_match_quality(
            rec_names=rec_names,
            av_names=av_names,
            rec_paths=rec_paths,
            av_paths=av_paths,
            by_signature=by_signature,
        )

        rec = {
            "session": session,
            "group_id": group_id,
            "phase_tags": phase_tags,
            "recording_session_names": sorted(rec_names),
            "av_session_names": sorted(av_names),
            "recording_splits": rec_splits,
            "av_splits": av_splits,
            "match_quality": match_quality,
            "match_cues": match_cues,
            "participants_ids": participants_ids,
            "participants_names": participants_names,
            "processed_modalities": sorted(processed_available),
            "raw_modalities": sorted(raw_available),
            "stimuli_candidates": stimuli_candidates,
            "stimuli_candidate_count": len(stimuli_candidates),
            "tobii_candidates": tobii_candidates,
            "tobii_candidate_count": len(tobii_candidates),
            "currentstudy_group_folder_count": currentstudy_info["folder_count"],
            "currentstudy_group_xdf_count": currentstudy_info["xdf_count"],
            "central_mic_available": central_mic_available,
            "central_mic_file_count": central_mic_file_count,
        }

        for split in TRACKED_SPLITS:
            rec[f"recording_{split}"] = split in rec_dirs
            rec[f"av_{split}"] = split in av_dirs

        for m in PROCESSED_MODALITIES:
            rec[f"proc_{m}"] = m in processed_available
        for m in RAW_MODALITIES:
            rec[f"raw_{m}"] = m in raw_available

        session_records.append(rec)

    groups: dict[str, dict[str, Any]] = {}
    for rec in session_records:
        gid = rec["group_id"]
        if gid not in groups:
            groups[gid] = {
                "group_id": gid,
                "sessions": [],
                "phase_tags": set(),
                "recording_splits": set(),
                "av_splits": set(),
                "participants_ids": set(),
                "participants_names": set(),
                "processed_modalities": set(),
                "raw_modalities": set(),
                "stimuli_candidate_count": 0,
                "tobii_candidate_count": 0,
                "currentstudy_group_folder_count": 0,
                "currentstudy_group_xdf_count": 0,
                "central_mic_available": False,
                "central_mic_file_count": 0,
            }
        g = groups[gid]
        g["sessions"].append(rec["session"])
        g["phase_tags"].update(rec["phase_tags"])
        g["recording_splits"].update(rec["recording_splits"])
        g["av_splits"].update(rec["av_splits"])
        g["participants_ids"].update(rec["participants_ids"])
        g["participants_names"].update(rec["participants_names"])
        g["processed_modalities"].update(rec["processed_modalities"])
        g["raw_modalities"].update(rec["raw_modalities"])
        g["stimuli_candidate_count"] += rec["stimuli_candidate_count"]
        g["tobii_candidate_count"] += rec["tobii_candidate_count"]
        g["currentstudy_group_folder_count"] = max(g["currentstudy_group_folder_count"], rec["currentstudy_group_folder_count"])
        g["currentstudy_group_xdf_count"] = max(g["currentstudy_group_xdf_count"], rec["currentstudy_group_xdf_count"])
        g["central_mic_available"] = g["central_mic_available"] or rec["central_mic_available"]
        g["central_mic_file_count"] = max(g["central_mic_file_count"], rec["central_mic_file_count"])

    group_records: list[dict[str, Any]] = []
    for gid in sorted(groups.keys()):
        g = groups[gid]
        group_records.append(
            {
                "group_id": gid,
                "session_count": len(g["sessions"]),
                "sessions": sorted(g["sessions"]),
                "phase_tags": sorted(g["phase_tags"]),
                "recording_splits": sorted(g["recording_splits"]),
                "av_splits": sorted(g["av_splits"]),
                "participants_ids": sorted(g["participants_ids"]),
                "participants_names": sorted(g["participants_names"]),
                "processed_modalities": sorted(g["processed_modalities"]),
                "raw_modalities": sorted(g["raw_modalities"]),
                "stimuli_candidate_count": g["stimuli_candidate_count"],
                "tobii_candidate_count": g["tobii_candidate_count"],
                "currentstudy_group_folder_count": g["currentstudy_group_folder_count"],
                "currentstudy_group_xdf_count": g["currentstudy_group_xdf_count"],
                "central_mic_available": g["central_mic_available"],
                "central_mic_file_count": g["central_mic_file_count"],
            }
        )

    out_session_csv.parent.mkdir(parents=True, exist_ok=True)
    session_fields = [
        "session",
        "group_id",
        "phase_tags",
        "recording_session_names",
        "av_session_names",
        "recording_splits",
        "av_splits",
        "match_quality",
        "match_cues",
        "participants_ids",
        "participants_names",
        "processed_modalities",
        "raw_modalities",
        "stimuli_candidate_count",
        "stimuli_candidates",
        "tobii_candidate_count",
        "tobii_candidates",
        "currentstudy_group_folder_count",
        "currentstudy_group_xdf_count",
        "central_mic_available",
        "central_mic_file_count",
    ]
    for split in TRACKED_SPLITS:
        session_fields.extend([f"recording_{split}", f"av_{split}"])
    for m in PROCESSED_MODALITIES:
        session_fields.append(f"proc_{m}")
    for m in RAW_MODALITIES:
        session_fields.append(f"raw_{m}")

    with out_session_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=session_fields)
        writer.writeheader()
        for rec in session_records:
            row = dict(rec)
            row["phase_tags"] = ";".join(rec["phase_tags"])
            row["recording_session_names"] = ";".join(rec["recording_session_names"])
            row["av_session_names"] = ";".join(rec["av_session_names"])
            row["recording_splits"] = ";".join(rec["recording_splits"])
            row["av_splits"] = ";".join(rec["av_splits"])
            row["match_cues"] = ";".join(rec["match_cues"])
            row["participants_ids"] = ";".join(rec["participants_ids"])
            row["participants_names"] = ";".join(rec["participants_names"])
            row["processed_modalities"] = ";".join(rec["processed_modalities"])
            row["raw_modalities"] = ";".join(rec["raw_modalities"])
            row["stimuli_candidates"] = ";".join(rec["stimuli_candidates"])
            row["tobii_candidates"] = ";".join(rec["tobii_candidates"])
            writer.writerow(row)

    group_fields = [
        "group_id",
        "session_count",
        "sessions",
        "phase_tags",
        "recording_splits",
        "av_splits",
        "participants_ids",
        "participants_names",
        "processed_modalities",
        "raw_modalities",
        "stimuli_candidate_count",
        "tobii_candidate_count",
        "currentstudy_group_folder_count",
        "currentstudy_group_xdf_count",
        "central_mic_available",
        "central_mic_file_count",
    ]
    with out_group_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        for rec in group_records:
            row = dict(rec)
            row["sessions"] = ";".join(rec["sessions"])
            row["phase_tags"] = ";".join(rec["phase_tags"])
            row["recording_splits"] = ";".join(rec["recording_splits"])
            row["av_splits"] = ";".join(rec["av_splits"])
            row["participants_ids"] = ";".join(rec["participants_ids"])
            row["participants_names"] = ";".join(rec["participants_names"])
            row["processed_modalities"] = ";".join(rec["processed_modalities"])
            row["raw_modalities"] = ";".join(rec["raw_modalities"])
            writer.writerow(row)

    payload = {
        "recording_root": str(recording_root),
        "av_root": str(av_root),
        "stimuli_root": str(stimuli_root),
        "tobii_root": str(tobii_root),
        "currentstudy_root": str(currentstudy_root),
        "central_mic_root": str(central_mic_root),
        "schedule_tsv": str(schedule_tsv),
        "session_count": len(session_records),
        "group_count": len(group_records),
        "sessions": session_records,
        "groups": group_records,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build high-level data inventory across pilot/test/final with participants and modalities"
    )
    p.add_argument(
        "--recording-root",
        type=Path,
        default=Path("data/affectai-capture-recording/sessions"),
    )
    p.add_argument(
        "--av-root",
        type=Path,
        default=Path("data/AV"),
    )
    p.add_argument(
        "--stimuli-root",
        type=Path,
        default=Path("data/affectai-capture-recording/stimuli/data"),
    )
    p.add_argument(
        "--tobii-root",
        type=Path,
        default=Path("data/Tobii"),
    )
    p.add_argument(
        "--currentstudy-root",
        type=Path,
        default=Path("data/CurrentStudy"),
    )
    p.add_argument(
        "--central-mic-root",
        type=Path,
        default=Path("data/Central mic"),
    )
    p.add_argument(
        "--schedule-tsv",
        type=Path,
        default=Path("data/affectai-capture-recording/configs/session_schedule.tsv"),
    )
    p.add_argument(
        "--out-session-csv",
        type=Path,
        default=Path("data/high_level_session_inventory.csv"),
    )
    p.add_argument(
        "--out-group-csv",
        type=Path,
        default=Path("data/high_level_group_inventory.csv"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/high_level_data_inventory.json"),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = build_inventory(
            recording_root=args.recording_root,
            av_root=args.av_root,
            stimuli_root=args.stimuli_root,
            tobii_root=args.tobii_root,
            currentstudy_root=args.currentstudy_root,
            central_mic_root=args.central_mic_root,
            schedule_tsv=args.schedule_tsv,
            out_session_csv=args.out_session_csv,
            out_group_csv=args.out_group_csv,
            out_json=args.out_json,
        )
    except Exception as exc:
        print(f"[report_data_high_level_inventory] ERROR: {exc}")
        return 1

    print(
        "[report_data_high_level_inventory] "
        f"sessions={payload.get('session_count', 0)} "
        f"groups={payload.get('group_count', 0)} "
        f"session_csv={args.out_session_csv} "
        f"group_csv={args.out_group_csv} "
        f"json={args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
