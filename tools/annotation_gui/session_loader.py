"""Discover BIDS task-run media and metadata for the annotation GUI.

BIDS layout (per session):
    sub-XX/ses-YYYYMMDD_grp-NN_runNN/
        audio/  sub-XX_ses-..._task-T1_run-01_acq-*_audio.wav
        video/  sub-XX_ses-..._task-T1_run-01_acq-*_video.mkv
        beh/    sub-XX_ses-..._task-T1_run-01_events.tsv
        et/     *_task-T1_run-01_acq-tobii_gaze.tsv
        physio/ *_task-T1_run-01_acq-emotibit_physio.tsv
        annot/  *_task_run_windows.tsv, *_participant_signal_map.tsv
        mocap/  gestures_events.ndjson (optional)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_TASK_RUN_RE = re.compile(r"task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)")
_ACQ_RE = re.compile(r"acq-(?P<acq>[A-Za-z0-9\-]+)")


@dataclass(frozen=True)
class MediaFile:
    path: Path
    acq: str  # acquisition label from BIDS filename, or ""
    kind: str  # "audio" | "video"


@dataclass
class TaskRun:
    sub: str
    ses: str
    task: str
    run: str
    root: Path  # sub-XX/ses-.../
    audio: list[MediaFile] = field(default_factory=list)
    video: list[MediaFile] = field(default_factory=list)
    events_tsv: Path | None = None
    task_windows_tsv: Path | None = None
    participant_map_tsv: Path | None = None
    gaze_tsvs: list[Path] = field(default_factory=list)
    physio_tsvs: list[Path] = field(default_factory=list)
    gestures_ndjson: Path | None = None
    annotations_json: Path | None = None  # set by the GUI when editing

    @property
    def label(self) -> str:
        return f"{self.sub}/{self.ses}/task-{self.task}_run-{self.run}"

    def default_annotations_path(self) -> Path:
        return self.root / "annot" / f"{self.sub}_{self.ses}_task-{self.task}_run-{self.run}_annotations.json"

    def default_sync_offsets_path(self) -> Path:
        return self.root / "annot" / f"{self.sub}_{self.ses}_task-{self.task}_run-{self.run}_sync_offsets.json"


def _parse_task_run(name: str) -> tuple[str, str] | None:
    m = _TASK_RUN_RE.search(name)
    if not m:
        return None
    return m.group("task"), m.group("run")


def _parse_acq(name: str) -> str:
    m = _ACQ_RE.search(name)
    return m.group("acq") if m else ""


def discover_sessions(bids_root: Path) -> list[Path]:
    """Return session directories: <bids_root>/sub-*/ses-*."""
    if not bids_root.is_dir():
        return []
    sessions: list[Path] = []
    for sub in sorted(bids_root.glob("sub-*")):
        if not sub.is_dir():
            continue
        sessions.extend(sorted(p for p in sub.glob("ses-*") if p.is_dir()))
    return sessions


def discover_task_runs(session_dir: Path) -> list[TaskRun]:
    """Enumerate task runs in a session by scanning audio/ and video/ filenames."""
    sub = session_dir.parent.name
    ses = session_dir.name
    audio_dir = session_dir / "audio"
    video_dir = session_dir / "video"
    et = session_dir / "et"

    found: dict[tuple[str, str], TaskRun] = {}

    def _get(task: str, run: str) -> TaskRun:
        key = (task, run)
        if key not in found:
            found[key] = TaskRun(sub=sub, ses=ses, task=task, run=run, root=session_dir)
        return found[key]

    if audio_dir.is_dir():
        # Accept both BIDS `_audio.wav` and the project's `_aud.wav` convention.
        audio_candidates = list(audio_dir.glob("*_audio.*")) + list(audio_dir.glob("*_aud.*"))
        for p in sorted(set(audio_candidates)):
            if p.suffix.lower() not in {".wav", ".flac"}:
                continue
            tr = _parse_task_run(p.name)
            if not tr:
                continue
            _get(*tr).audio.append(MediaFile(path=p, acq=_parse_acq(p.name), kind="audio"))

    if video_dir.is_dir():
        for p in sorted(video_dir.glob("*_video.*")):
            if p.suffix.lower() not in {".mkv", ".mp4", ".avi", ".mov"}:
                continue
            tr = _parse_task_run(p.name)
            if not tr:
                continue
            _get(*tr).video.append(MediaFile(path=p, acq=_parse_acq(p.name), kind="video"))
    # Include Tobii scene videos that live in et/ as additional video feeds.
    if et.is_dir():
        for p in sorted(et.glob("*_task-*_run-*_acq-*_tobii.*")):
            if p.suffix.lower() not in {".mkv", ".mp4", ".avi", ".mov"}:
                continue
            tr = _parse_task_run(p.name)
            if not tr:
                continue
            _get(*tr).video.append(MediaFile(path=p, acq=_parse_acq(p.name), kind="video"))

    # attach per-task metadata and task-agnostic session files
    beh = session_dir / "beh"
    physio = session_dir / "physio"
    annot = session_dir / "annot"
    mocap = session_dir / "mocap"

    task_windows = next(annot.glob("*_task_run_windows.tsv"), None) if annot.is_dir() else None
    participant_map = (
        next(annot.glob("*_participant_signal_map.tsv"), None) if annot.is_dir() else None
    )
    # Fall back to session-root participant_map.tsv if the annot one is absent.
    if participant_map is None:
        fallback = session_dir / "participant_map.tsv"
        participant_map = fallback if fallback.is_file() else None
    gestures = next(mocap.glob("gestures_events.ndjson"), None) if mocap.is_dir() else None

    for tr in found.values():
        tr.task_windows_tsv = task_windows
        tr.participant_map_tsv = participant_map
        tr.gestures_ndjson = gestures
        if beh.is_dir():
            stem = f"*_task-{tr.task}_run-{tr.run}_events.tsv"
            tr.events_tsv = next(beh.glob(stem), None)
        if et.is_dir():
            # Eye-tracking is stored as P{n}_task-Tx_gaze.ndjson or task-scoped TSVs.
            tr.gaze_tsvs = sorted(et.glob(f"*_task-{tr.task}_run-{tr.run}_*.tsv")) + sorted(
                et.glob(f"*task-{tr.task}_gaze.ndjson")
            )
        if physio.is_dir():
            tr.physio_tsvs = sorted(physio.glob(f"*_task-{tr.task}_run-{tr.run}_*.tsv")) + sorted(
                physio.glob(f"*_task-{tr.task}_run-{tr.run}_*.tsv.gz")
            )

    return sorted(found.values(), key=lambda t: (t.task, t.run))


def load_participant_map(path: Path | None) -> pd.DataFrame:
    """Participant ↔ signal map. Returns an empty DF if the file is missing."""
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_task_windows(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")
