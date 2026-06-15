#!/usr/bin/env python3
"""Refresh per-task Tobii scene videos from an external Tobii source root.

This script updates:
  affectai-data-processing-seed/data/sub-01/ses-*/et/*_acq-P*_tobii.mp4

It uses:
  - session task windows (annot/*_task_run_windows.tsv)
  - session wall-minus-lsl offset (from windows TSV or sync_metadata JSON)
  - Tobii folder mapping from metadata/session_metadata_report.tsv
  - Tobii recordings mapping (configs/tobii_recordings_mapping.json)
  - raw Tobii files from --tobii-source-root (scenevideo.mp4, recording.g3, gazedata.gz)
  - existing session audio clips for optional audio-mediated refinement
  - microphone audio (mic9-mic12) for video-audio sync verification

Internally this calls xdf_sync_pipeline.split_tobii_video_by_task().
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import xdf_sync_pipeline as xsp

# Microphone to participant mapping
MIC_PARTICIPANT_MAP = {
    "mic9": "P1",
    "mic10": "P2",
    "mic11": "P3",
    "mic12": "P4",
}


def _parse_float(value: str, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _session_entities(session_dir: Path) -> tuple[str, str]:
    ses = session_dir.name
    sub = session_dir.parent.name
    ses_label = ses[4:] if ses.startswith("ses-") else ses
    sub_label = sub[4:] if sub.startswith("sub-") else sub
    return sub_label, ses_label


def _find_task_windows_tsv(session_dir: Path) -> Path | None:
    annot = session_dir / "annot"
    if not annot.exists():
        return None
    candidates = sorted(annot.glob("*_task_run_windows.tsv"))
    if not candidates:
        return None
    # Prefer canonical T0..T4 aggregate if present.
    preferred = [p for p in candidates if "_task-T0T1T2T3T4_" in p.name]
    return preferred[0] if preferred else candidates[0]


def _read_windows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            task = str(row.get("task", "")).strip()
            run = str(row.get("run", "01")).strip() or "01"
            start = _parse_float(str(row.get("start_wall_clock", "")))
            end = _parse_float(str(row.get("end_wall_clock", "")))
            if not task or start is None or end is None:
                continue
            rows.append(
                {
                    "task": task,
                    "run": run,
                    "start_wall_clock": start,
                    "end_wall_clock": end,
                }
            )
    return rows


def _load_offset_from_windows(path: Path) -> float | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            off = _parse_float(str(row.get("wall_minus_lsl_offset", "")))
            if off is not None:
                return off
    return None


def _load_offset_from_sync_metadata(session_dir: Path) -> float | None:
    annot = session_dir / "annot"
    if not annot.exists():
        return None
    candidates = sorted(annot.glob("*_sync_metadata.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        off = _parse_float(str(payload.get("wall_minus_lsl_offset", "")))
        if off is not None:
            return off
    return None


def _load_tobii_recordings_mapping(metadata_root: Path) -> dict[str, Any] | None:
    """Load Tobii recordings mapping from configs/tobii_recordings_mapping.json"""
    # Try multiple possible locations
    possible_paths = [
        metadata_root.parent.parent / "configs" / "tobii_recordings_mapping.json",  # Project root
        metadata_root.parent / "configs" / "tobii_recordings_mapping.json",  # One level up
        Path("configs") / "tobii_recordings_mapping.json",  # Current directory
    ]

    for mapping_path in possible_paths:
        if mapping_path.exists():
            try:
                return json.loads(mapping_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: Failed to load tobii recordings mapping from {mapping_path}: {e}")
                continue

    print(f"Warning: tobii_recordings_mapping.json not found. Tried: {possible_paths}")
    return None


def _get_group_info_from_mapping(
    mapping: dict[str, Any], tobii_folder: str
) -> dict[str, Any] | None:
    """Extract group info for a given tobii folder name from the mapping."""
    if not mapping or "groups" not in mapping:
        return None

    for group_data in mapping.get("groups", []):
        for recording in group_data.get("recordings", []):
            folder_names = recording.get("folder_names", [])
            if not isinstance(folder_names, list):
                folder_names = [folder_names] if folder_names else []

            if tobii_folder in folder_names:
                return {
                    "group_id": group_data.get("group_id"),
                    "date": group_data.get("date"),
                    "participant": recording.get("participant"),
                    "device_serial": recording.get("device_serial"),
                }
    return None


def _verify_video_audio_sync(
    video_path: Path,
    audio_path: Path,
    participant: str,
) -> dict[str, Any]:
    """Verify that video and audio are synced by checking their durations and metadata.

    Returns a dict with:
      - synced: bool - whether they appear to be synced
      - video_duration: float - video duration in seconds
      - audio_duration: float - audio duration in seconds
      - duration_diff: float - absolute difference in seconds
      - reason: str - explanation if not synced
    """
    result = {
        "synced": False,
        "video_duration": None,
        "audio_duration": None,
        "duration_diff": None,
        "participant": participant,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "reason": None,
    }

    if not video_path.exists():
        result["reason"] = f"Video file not found: {video_path}"
        return result

    if not audio_path.exists():
        result["reason"] = f"Audio file not found: {audio_path}"
        return result

    try:
        # Get video duration
        cmd_video = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            str(video_path),
        ]
        video_duration = float(subprocess.check_output(cmd_video).decode().strip())
        result["video_duration"] = video_duration
    except Exception as e:
        result["reason"] = f"Failed to get video duration: {e}"
        return result

    try:
        # Get audio duration
        cmd_audio = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            str(audio_path),
        ]
        audio_duration = float(subprocess.check_output(cmd_audio).decode().strip())
        result["audio_duration"] = audio_duration
    except Exception as e:
        result["reason"] = f"Failed to get audio duration: {e}"
        return result

    # Check if durations match (within 1 second tolerance for processing artifacts)
    duration_diff = abs(video_duration - audio_duration)
    result["duration_diff"] = duration_diff

    if duration_diff <= 1.0:
        result["synced"] = True
        result["reason"] = f"Durations match within tolerance (diff={duration_diff:.2f}s)"
    else:
        result["synced"] = False
        result["reason"] = f"Duration mismatch: video={video_duration:.2f}s, audio={audio_duration:.2f}s (diff={duration_diff:.2f}s)"

    return result


def _build_video_map_from_tobii_mapping(
    tobii_mapping: dict[str, Any], session_id: str
) -> dict[str, list[str]] | None:
    """Build video_map from tobii_recordings_mapping.json using session ID.

    Extracts group ID from session_id and returns mapping of participant -> list of tobii_folders.
    Format: {"P1": ["folder1", "folder2"], "P2": [...], ...}
    """
    if not tobii_mapping or "groups" not in tobii_mapping:
        return None

    # Extract group ID from session_id (e.g., "ses-20260318_grp-12_run01" -> "grp-12")
    group_id = None
    for part in session_id.split("_"):
        if part.startswith("grp-"):
            group_id = part
            break

    if not group_id:
        return None

    # Find group in mapping
    for group_data in tobii_mapping.get("groups", []):
        if group_data.get("group_id") == group_id:
            video_map = {}
            for recording in group_data.get("recordings", []):
                participant = recording.get("participant")
                folder_names = recording.get("folder_names", [])

                # Ensure folder_names is a list and filter out None/empty entries
                if isinstance(folder_names, str):
                    folder_names = [folder_names] if folder_names else []
                elif not isinstance(folder_names, list):
                    folder_names = []

                # Filter out None and empty strings
                valid_folders = [f for f in folder_names if f]

                if valid_folders:
                    video_map[participant] = valid_folders

            return video_map if video_map else None

    return None


def refresh_one_session(
    session_dir: Path,
    metadata_root: Path,
    tobii_source_root: Path,
    dry_run: bool = False,
    verify_audio_sync: bool = False,
) -> dict[str, Any]:
    session_id = session_dir.name
    sub_label, ses_label = _session_entities(session_dir)

    windows_tsv = _find_task_windows_tsv(session_dir)
    if windows_tsv is None:
        return {"session": session_id, "status": "skip", "reason": "missing_task_windows_tsv"}

    windows = _read_windows(windows_tsv)
    if not windows:
        return {"session": session_id, "status": "skip", "reason": "empty_windows"}

    offset = _load_offset_from_windows(windows_tsv)
    if offset is None:
        offset = _load_offset_from_sync_metadata(session_dir)
    if offset is None:
        return {"session": session_id, "status": "skip", "reason": "missing_wall_minus_lsl_offset"}

    # Load Tobii recordings mapping (FORCED PRIMARY SOURCE)
    tobii_mapping = _load_tobii_recordings_mapping(metadata_root)
    if not tobii_mapping:
        return {"session": session_id, "status": "skip", "reason": "missing_tobii_recordings_mapping"}

    # Build video_map from tobii_recordings_mapping.json
    video_map = _build_video_map_from_tobii_mapping(tobii_mapping, session_id)
    if not video_map:
        return {
            "session": session_id,
            "status": "skip",
            "reason": "unable_to_build_video_map_from_tobii_mapping",
            "note": "Check tobii_recordings_mapping.json for group matching session_id",
        }

    # Build lookup for group info
    tobii_folder_to_group = {}
    for group_data in tobii_mapping.get("groups", []):
        for recording in group_data.get("recordings", []):
            folder_names = recording.get("folder_names", [])
            if isinstance(folder_names, str):
                folder_names = [folder_names]
            for folder in folder_names:
                tobii_folder_to_group[folder] = {
                    "group_id": group_data.get("group_id"),
                    "date": group_data.get("date"),
                    "participant": recording.get("participant"),
                    "device_serial": recording.get("device_serial"),
                }

    if dry_run:
        return {
            "session": session_id,
            "status": "dry_run",
            "windows": len(windows),
            "participants": sorted(video_map.keys()),
            "tobii_mapping_source": "configs/tobii_recordings_mapping.json",
            "video_map": {p: str(v) for p, v in video_map.items()},
        }

    # Verify audio sync if requested
    audio_sync_results = []
    if verify_audio_sync:
        audio_dir = session_dir / "aud"
        if audio_dir.exists():
            for participant in sorted(video_map.keys()):
                # Map participant to microphone
                mic_id = None
                for mic_label, p in MIC_PARTICIPANT_MAP.items():
                    if p == participant:
                        mic_id = mic_label
                        break

                if not mic_id:
                    continue

                # Find audio file for this microphone
                audio_files = list(audio_dir.glob(f"*{mic_id}*.wav")) + \
                             list(audio_dir.glob(f"*{mic_id}*.mp3"))

                if video_map[participant] and audio_files:
                    sync_check = _verify_video_audio_sync(
                        video_path=tobii_source_root / video_map[participant] / "scenevideo.mp4",
                        audio_path=audio_files[0],
                        participant=participant,
                    )
                    audio_sync_results.append(sync_check)

    outputs = xsp.split_tobii_video_by_task(
        tobii_root=tobii_source_root,
        tobii_video_map=video_map,
        session_dir=session_dir,
        sub_label=sub_label,
        ses_label=ses_label,
        windows=windows,
        wall_minus_xdf_lsl=offset,
    )

    result = {
        "session": session_id,
        "status": "ok",
        "generated": len(outputs),
        "generated_files": [str(p) for p in outputs],
        "tobii_mapping_source": "configs/tobii_recordings_mapping.json",
        "video_map_used": {p: str(v) for p, v in video_map.items()},
    }

    if audio_sync_results:
        result["audio_sync_verification"] = audio_sync_results
        result["all_audio_synced"] = all(r.get("synced", False) for r in audio_sync_results)

    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Refresh Tobii task videos for sub-01 sessions from source Tobii folders"
    )
    p.add_argument(
        "--tobii-source-root",
        type=Path,
        required=True,
        help="Root containing Tobii recording folders referenced by session_metadata_report.tsv",
    )
    p.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("affectai-data-processing-seed/data/sub-01"),
        help="Root containing ses-* folders (default: affectai-data-processing-seed/data/sub-01)",
    )
    p.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("affectai-data-processing-seed/metadata"),
        help="Metadata root containing session_metadata_report.tsv",
    )
    p.add_argument(
        "--session-glob",
        default="ses-*",
        help="Session folder glob under --sessions-root (default: ses-*)",
    )
    p.add_argument(
        "--verify-audio-sync",
        action="store_true",
        help="Verify video-audio sync by comparing durations (requires ffprobe)",
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would be processed")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tobii_source_root = args.tobii_source_root.resolve()
    sessions_root = args.sessions_root.resolve()
    metadata_root = args.metadata_root.resolve()

    if not tobii_source_root.exists():
        raise FileNotFoundError(f"Tobii source root not found: {tobii_source_root}")
    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions root not found: {sessions_root}")
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root not found: {metadata_root}")

    session_dirs = sorted(p for p in sessions_root.glob(args.session_glob) if p.is_dir())
    if not session_dirs:
        print("No session folders matched.")
        return 0

    results: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        res = refresh_one_session(
            session_dir=session_dir,
            metadata_root=metadata_root,
            tobii_source_root=tobii_source_root,
            dry_run=args.dry_run,
            verify_audio_sync=args.verify_audio_sync,
        )
        results.append(res)
        print(json.dumps(res, ensure_ascii=True))

    out = sessions_root / "_tobii_refresh_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

