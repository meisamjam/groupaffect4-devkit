"""Time synchronization offset management for annotation feeds."""

from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class SyncOffsets:
    """Per-stream time offset adjustments (in seconds)."""

    video: dict[str, float] = field(default_factory=dict)  # acq/stem → offset
    audio: dict[str, float] = field(default_factory=dict)  # mic name → offset
    transcript: float = 0.0


def load_sync(path: Path) -> SyncOffsets:
    """Load sync offsets from JSON file; return zeroed defaults if absent."""
    if not path.is_file():
        return SyncOffsets()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return SyncOffsets(
            video=data.get("video", {}),
            audio=data.get("audio", {}),
            transcript=float(data.get("transcript", 0.0)),
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return SyncOffsets()


def save_sync(path: Path, offsets: SyncOffsets) -> None:
    """Save sync offsets to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(offsets)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


_JABRA_KEY_RE = re.compile(r"(jabra[-_]panacast[-_](?:20|50)(?:[-_]cam\d+)?[-_]vid)")
_DPA_KEY_RE = re.compile(r"(dpa[-_][a-z0-9]+(?:[-_][a-z0-9]+)?(?:[-_]aud)?)")
_PARTICIPANT_RE = re.compile(r"(?:^|[_-])p([1-4])(?:[_-]|$)", re.IGNORECASE)


def _canonical_stream_key(label: str) -> str:
    """Normalize stream/media labels for robust ffmpeg_progress matching."""
    raw = (label or "").strip().lower()
    if raw.startswith("ffmpeg_progress_"):
        raw = raw[len("ffmpeg_progress_") :]

    jabra = _JABRA_KEY_RE.search(raw)
    if jabra:
        raw = jabra.group(1)
    else:
        dpa = _DPA_KEY_RE.search(raw)
        if dpa:
            raw = dpa.group(1)
        elif raw.startswith("acq-"):
            raw = raw[4:]

    raw = raw.replace("-", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def infer_sync_offsets_from_lsl_sync(
    lsl_sync_tsv: Path,
    video_labels: list[str],
    audio_labels: list[str],
    max_abs_offset_s: float = 5.0,
    lsl_ref_time: float | None = None,
) -> SyncOffsets:
    """Estimate per-feed offsets from task `*_acq-lsl_sync.tsv` using LSL clock.

    We project each stream's media clock (`value_0`) to a shared LSL reference
    time (global earliest `lsl_time`) and align feeds by the median projected
    media time at that reference:
    offset = median(projected_media_at_lsl_ref) - stream_projected_media_at_lsl_ref
    """
    if not lsl_sync_tsv.is_file():
        return SyncOffsets()

    samples_by_key: dict[str, list[tuple[float, float]]] = {}
    try:
        with open(lsl_sync_tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                stream_name = str(row.get("stream_name", "")).strip()
                if not stream_name.startswith("ffmpeg_progress_"):
                    continue
                key = _canonical_stream_key(stream_name)
                if not key:
                    continue
                try:
                    lsl_time = float(row.get("lsl_time", ""))
                    value_0 = float(row.get("value_0", ""))
                except (TypeError, ValueError):
                    continue
                samples_by_key.setdefault(key, []).append((lsl_time, value_0))
    except OSError:
        return SyncOffsets()

    if not samples_by_key:
        return SyncOffsets()

    if lsl_ref_time is None:
        all_lsl = [lsl for rows in samples_by_key.values() for lsl, _ in rows]
        if not all_lsl:
            return SyncOffsets()
        lsl_ref = min(all_lsl)
    else:
        lsl_ref = float(lsl_ref_time)

    projected_value_by_key: dict[str, float] = {}
    for key, rows in samples_by_key.items():
        # Project media time to the shared LSL reference using slope ~1:
        # media(lsl_ref) ~= value_0 - (lsl_time - lsl_ref)
        projected = [v0 - (lsl - lsl_ref) for lsl, v0 in rows]
        if not projected:
            continue
        projected_value_by_key[key] = float(statistics.median(projected))

    if not projected_value_by_key:
        return SyncOffsets()

    matched_video: dict[str, float] = {}
    matched_audio: dict[str, float] = {}
    for label in video_labels:
        key = _canonical_stream_key(label)
        if key in projected_value_by_key:
            matched_video[label] = projected_value_by_key[key]
    for label in audio_labels:
        key = _canonical_stream_key(label)
        if key in projected_value_by_key:
            matched_audio[label] = projected_value_by_key[key]

    matched_values = list(matched_video.values()) + list(matched_audio.values())
    if len(matched_values) < 2:
        return SyncOffsets()

    reference = float(statistics.median(matched_values))
    out = SyncOffsets()
    for label, stream_t in matched_video.items():
        offset = round(reference - stream_t, 3)
        if abs(offset) <= max_abs_offset_s:
            out.video[label] = offset
    for label, stream_t in matched_audio.items():
        offset = round(reference - stream_t, 3)
        if abs(offset) <= max_abs_offset_s:
            out.audio[label] = offset
    return out


def load_task_start_lsl(task_windows_tsv: Path | None, task: str, run: str) -> float | None:
    """Read task-run `start_lsl` from `*_task_run_windows.tsv` when available."""
    if task_windows_tsv is None or not task_windows_tsv.is_file():
        return None
    task_norm = str(task).strip().upper()
    run_norm = _normalize_run(run)
    try:
        with open(task_windows_tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row_task = str(row.get("task", "")).strip().upper()
                row_run = _normalize_run(row.get("run", ""))
                if row_task != task_norm or row_run != run_norm:
                    continue
                try:
                    return float(row.get("start_lsl", ""))
                except (TypeError, ValueError):
                    return None
    except OSError:
        return None
    return None


def _normalize_run(run: object) -> str:
    text = str(run).strip()
    if not text:
        return ""
    text = text.lstrip("0")
    return text or "0"


def infer_tobii_video_offsets_from_et(
    et_dir: Path,
    task: str,
    run: str,
    video_labels: list[str],
    max_abs_offset_s: float = 5.0,
    lsl_ref_time: float | None = None,
) -> dict[str, float]:
    """Infer Tobii scene-video offsets using per-participant ET LSL start times.

    Reads per-participant files:
    `*_task-{task}_run-{run}_acq-P*_tobii.tsv[.gz]`
    and uses the first `lsl_time` as stream start anchor.
    """
    if not et_dir.is_dir():
        return {}

    run_norm = _normalize_run(run)
    task_norm = str(task).strip().upper()
    first_lsl_by_participant: dict[str, float] = {}
    patterns = [
        f"*_task-{task_norm}_run-*_acq-P*_tobii.tsv",
        f"*_task-{task_norm}_run-*_acq-P*_tobii.tsv.gz",
    ]
    for pattern in patterns:
        for path in sorted(et_dir.glob(pattern)):
            m = _PARTICIPANT_RE.search(path.name)
            if not m:
                continue
            participant = f"P{m.group(1)}"
            # Run filter from filename.
            tr = re.search(r"task-[A-Za-z0-9]+_run-(\d+)", path.name)
            if tr is None or _normalize_run(tr.group(1)) != run_norm:
                continue
            first_lsl = _read_first_lsl_time(path)
            if first_lsl is None:
                continue
            if participant not in first_lsl_by_participant:
                first_lsl_by_participant[participant] = first_lsl

    if len(first_lsl_by_participant) < 2:
        return {}

    lsl_ref = float(lsl_ref_time) if lsl_ref_time is not None else min(first_lsl_by_participant.values())
    projected_media_at_ref = {
        p: (lsl_ref - start_lsl) for p, start_lsl in first_lsl_by_participant.items()
    }
    reference = float(statistics.median(projected_media_at_ref.values()))

    offsets: dict[str, float] = {}
    for label in video_labels:
        pm = _participant_from_label(label)
        if pm is None or pm not in projected_media_at_ref:
            continue
        offset = round(reference - projected_media_at_ref[pm], 3)
        if abs(offset) <= max_abs_offset_s:
            offsets[label] = offset
    return offsets


def _participant_from_label(label: str) -> str | None:
    m = _PARTICIPANT_RE.search((label or "").strip())
    if not m:
        return None
    return f"P{m.group(1)}"


def _read_first_lsl_time(path: Path) -> float | None:
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, mode="rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    return float(row.get("lsl_time", ""))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return None
    return None


def infer_sync_offsets_from_source_ffmpeg_lsl(
    source_session_dir: Path,
    video_labels: list[str],
    audio_labels: list[str],
    max_abs_offset_s: float = 5.0,
    lsl_ref_time: float | None = None,
) -> SyncOffsets:
    """Infer offsets from source capture `lsl/ffmpeg_progress_*.jsonl` logs.

    Uses `(stream_time, values[0])` as `(lsl_time, media_time)` samples and
    applies the same projection-to-reference logic as `infer_sync_offsets_from_lsl_sync`.
    """
    if not source_session_dir.is_dir():
        return SyncOffsets()

    samples_by_key: dict[str, list[tuple[float, float]]] = {}
    for p in source_session_dir.glob("sourcedata/*/lsl/ffmpeg_progress_*.jsonl"):
        key = _canonical_stream_key(p.stem)
        if not key:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        lsl_time = float(obj.get("stream_time"))
                        values = obj.get("values") or []
                        if not isinstance(values, list) or not values:
                            continue
                        media_time = float(values[0])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    samples_by_key.setdefault(key, []).append((lsl_time, media_time))
        except OSError:
            continue

    if not samples_by_key:
        return SyncOffsets()

    if lsl_ref_time is None:
        all_lsl = [lsl for rows in samples_by_key.values() for lsl, _ in rows]
        if not all_lsl:
            return SyncOffsets()
        lsl_ref = min(all_lsl)
    else:
        lsl_ref = float(lsl_ref_time)

    projected_by_key: dict[str, float] = {}
    for key, rows in samples_by_key.items():
        projected = [v0 - (lsl - lsl_ref) for lsl, v0 in rows]
        if projected:
            projected_by_key[key] = float(statistics.median(projected))

    if not projected_by_key:
        return SyncOffsets()

    matched_video: dict[str, float] = {}
    matched_audio: dict[str, float] = {}
    for label in video_labels:
        key = _canonical_stream_key(label)
        if key in projected_by_key:
            matched_video[label] = projected_by_key[key]
    for label in audio_labels:
        key = _canonical_stream_key(label)
        if key in projected_by_key:
            matched_audio[label] = projected_by_key[key]

    all_vals = list(matched_video.values()) + list(matched_audio.values())
    if len(all_vals) < 2:
        return SyncOffsets()
    reference = float(statistics.median(all_vals))

    out = SyncOffsets()
    for label, value in matched_video.items():
        off = round(reference - value, 3)
        if abs(off) <= max_abs_offset_s:
            out.video[label] = off
    for label, value in matched_audio.items():
        off = round(reference - value, 3)
        if abs(off) <= max_abs_offset_s:
            out.audio[label] = off
    return out
