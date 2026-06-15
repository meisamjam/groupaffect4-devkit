#!/usr/bin/env python3
"""Convert session raw sources into BIDS-oriented modality folders.

Inputs used:
- AV raw files under ``sourcedata/av``
- Recording PC XDF (``*.xdf`` in session root)
- Tobii bridge dumps under ``sourcedata/tobii_lsl``
- Manually downloaded Tobii files under ``sourcedata/tobii_device``

This tool does not overwrite vendor raw data. It writes derived/canonicalised
outputs under ``video/``, ``audio/``, ``et/``, ``physio/``, ``beh/``, and
``annot/`` in the same session directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import fromstring

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
AUDIO_EXTS = {".wav", ".flac", ".aac", ".m4a"}
TEXT_EXTS = {".json", ".ndjson", ".tsv", ".csv"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower() or "unk"


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


def _session_entities(session_dir: Path) -> tuple[str, str, str]:
    ses = session_dir.name
    sub = session_dir.parent.name
    ses_label = ses[4:] if ses.startswith("ses-") else ses
    sub_label = sub[4:] if sub.startswith("sub-") else sub
    base = f"sub-{sub_label}_ses-{ses_label}_task-T0T1T2T3T4"
    return sub_label, ses_label, base


def _collect_av_raw(session_dir: Path) -> list[Path]:
    av_root = session_dir / "sourcedata" / "av"
    if not av_root.exists():
        return []
    return sorted(p for p in av_root.rglob("*") if p.is_file())


def _collect_tobii_device_raw(session_dir: Path) -> list[Path]:
    root = session_dir / "sourcedata" / "tobii_device"
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def _collect_tobii_lsl_raw(session_dir: Path) -> list[Path]:
    root = session_dir / "sourcedata" / "tobii_lsl"
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.ndjson"))


def _collect_xdf_files(session_dir: Path) -> list[Path]:
    roots = [
        session_dir,
        session_dir / "sourcedata" / "lsl",
    ]
    seen: set[Path] = set()
    xdf_files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.xdf")):
            if path in seen:
                continue
            seen.add(path)
            xdf_files.append(path)
    return xdf_files


def _tolerant_xdf_stream_ids(xdf_path: Path, prefix_re: re.Pattern[str]) -> list[int]:
    boundary = bytes(
        [
            0x43,
            0xA5,
            0x46,
            0xDC,
            0xCB,
            0xF5,
            0x41,
            0x0F,
            0xB3,
            0x0E,
            0xD5,
            0x46,
            0x73,
            0x83,
            0xCB,
            0xE4,
        ]
    )

    def read_varlen_int(file_obj: Any) -> int:
        first = file_obj.read(1)
        if not first:
            raise EOFError()
        nbytes = first[0]
        if nbytes == 1:
            return file_obj.read(1)[0]
        if nbytes == 2:
            raw = file_obj.read(2)
            if len(raw) != 2:
                raise EOFError()
            return struct.unpack("<H", raw)[0]
        if nbytes == 4:
            return struct.unpack("<I", file_obj.read(4))[0]
        if nbytes == 8:
            return struct.unpack("<Q", file_obj.read(8))[0]
        raise RuntimeError(f"invalid variable-length integer encountered: {nbytes}")

    def scan_forward(file_obj: Any) -> bool:
        block_size = 1024 * 1024
        while True:
            current = file_obj.tell()
            block = file_obj.read(block_size)
            if not block:
                return False
            match = block.find(boundary)
            if match != -1:
                file_obj.seek(current + match + len(boundary))
                return True

    stream_ids: list[int] = []
    with xdf_path.open("rb") as file_obj:
        if file_obj.read(4) != b"XDF:":
            return stream_ids
        while True:
            try:
                chunklen = read_varlen_int(file_obj)
            except EOFError:
                break
            except Exception:
                if not scan_forward(file_obj):
                    break
                continue

            try:
                tag = struct.unpack("<H", file_obj.read(2))[0]
            except struct.error:
                if not scan_forward(file_obj):
                    break
                continue

            stream_id = None
            if tag in (2, 3, 4, 6):
                raw = file_obj.read(4)
                if len(raw) != 4:
                    if not scan_forward(file_obj):
                        break
                    continue
                stream_id = struct.unpack("<I", raw)[0]

            try:
                if tag == 2:
                    xml_bytes = file_obj.read(chunklen - 6)
                    header = fromstring(xml_bytes.decode("utf-8", "replace"))
                    name = header.findtext("name") or ""
                    if prefix_re.search(name):
                        stream_ids.append(stream_id)
                elif tag in (3, 4, 6):
                    file_obj.seek(chunklen - 6, 1)
                else:
                    file_obj.seek(chunklen - 2, 1)
            except Exception:
                if not scan_forward(file_obj):
                    break

    seen: set[int] = set()
    ordered: list[int] = []
    for stream_id in stream_ids:
        if stream_id is None or stream_id in seen:
            continue
        seen.add(stream_id)
        ordered.append(stream_id)
    return ordered


def _load_tolerant_xdf_streams(xdf_path: Path, stream_ids: list[int]) -> list[dict[str, Any]]:
    if not stream_ids or importlib.util.find_spec("pyxdf") is None:
        return []

    pyxdf = __import__("pyxdf")
    module = pyxdf.pyxdf
    original = module._read_varlen_int

    def patched_read_varlen_int(file_obj: Any) -> int:
        first = file_obj.read(1)
        if not first:
            raise EOFError()
        nbytes = first[0]
        if nbytes == 1:
            return file_obj.read(1)[0]
        if nbytes == 2:
            raw = file_obj.read(2)
            if len(raw) != 2:
                raise EOFError()
            return struct.unpack("<H", raw)[0]
        if nbytes == 4:
            return struct.unpack("<I", file_obj.read(4))[0]
        if nbytes == 8:
            return struct.unpack("<Q", file_obj.read(8))[0]
        raise RuntimeError(f"invalid variable-length integer encountered: {nbytes}")

    module._read_varlen_int = patched_read_varlen_int
    try:
        streams, _ = pyxdf.load_xdf(
            str(xdf_path),
            select_streams=stream_ids,
            synchronize_clocks=False,
            dejitter_timestamps=False,
        )
        return streams
    finally:
        module._read_varlen_int = original


def _copy_local_raw_into_modalities(session_dir: Path, base: str, link: bool) -> dict[str, int]:
    counts = {"video": 0, "audio": 0, "et": 0, "annot": 0}

    for p in _collect_av_raw(session_dir):
        ext = p.suffix.lower()
        acq = _slug(p.stem)
        if ext in VIDEO_EXTS:
            out = session_dir / "video" / f"{base}_acq-av-{acq}_video{ext}"
            _copy_or_link(p, out, link)
            counts["video"] += 1
        elif ext in AUDIO_EXTS:
            out = session_dir / "audio" / f"{base}_acq-av-{acq}_audio{ext}"
            _copy_or_link(p, out, link)
            counts["audio"] += 1
        elif ext in {".jsonl", ".json", ".tsv"}:
            out = session_dir / "annot" / f"{base}_acq-av-{acq}_sync{ext}"
            _copy_or_link(p, out, link)
            counts["annot"] += 1

    for p in _collect_tobii_device_raw(session_dir):
        rel_parts = p.relative_to(session_dir / "sourcedata" / "tobii_device").parts
        device = _slug(rel_parts[0]) if rel_parts else "device"
        ext = p.suffix.lower()
        stem = _slug(p.stem)
        if ext in VIDEO_EXTS:
            out = session_dir / "video" / f"{base}_acq-tobii-{device}-{stem}_video{ext}"
            _copy_or_link(p, out, link)
            counts["video"] += 1
        elif ext in TEXT_EXTS:
            out = session_dir / "et" / f"{base}_acq-tobii-{device}-{stem}_recording{ext}"
            _copy_or_link(p, out, link)
            counts["et"] += 1

    for p in _collect_tobii_lsl_raw(session_dir):
        device = _slug(p.stem)
        out = session_dir / "et" / f"{base}_acq-tobii-lsl-{device}_recording.ndjson"
        _copy_or_link(p, out, link)
        counts["et"] += 1

    return counts


def _stream_name(stream: dict[str, Any]) -> str:
    info = stream.get("info", {})
    name = info.get("name", ["unknown"])
    if isinstance(name, list) and name:
        return str(name[0])
    return str(name)


def _stream_type(stream: dict[str, Any]) -> str:
    info = stream.get("info", {})
    stype = info.get("type", ["unknown"])
    if isinstance(stype, list) and stype:
        return str(stype[0])
    return str(stype)


def _is_tobii_stream_name(name: str) -> bool:
    return name.startswith("Tobii_") or name.startswith("TobiiGlasses")


def _is_emotibit_stream_name(name: str) -> bool:
    return name.startswith("EmotiBit_") or name.startswith("Emotibit_")


def _flatten_sample(sample: Any) -> list[str]:
    if isinstance(sample, list | tuple):
        return [str(v) for v in sample]
    return [str(sample)]


def _write_tsv(path: Path, header: list[str], rows: list[list[str]], gzip_out: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_out:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)


def _extract_from_xdf(
    session_dir: Path,
    base: str,
    require_xdf_extraction: bool = False,
    xdf_files: list[Path] | None = None,
) -> dict[str, int]:
    counts = {"beh": 0, "et": 0, "physio": 0, "annot": 0}
    if xdf_files is None:
        xdf_files = _collect_xdf_files(session_dir)
    else:
        xdf_files = sorted(Path(p) for p in xdf_files if Path(p).exists())
    if not xdf_files:
        if require_xdf_extraction:
            raise RuntimeError(f"No XDF files found in session: {session_dir}")
        return counts

    if importlib.util.find_spec("pyxdf") is None:
        if require_xdf_extraction:
            raise RuntimeError(
                "pyxdf is not installed; cannot open XDF files for synchronized sequence extraction"
            )
        return counts

    pyxdf = __import__("pyxdf")

    event_rows: list[list[str]] = []
    tobii_rows: list[list[str]] = []
    emotibit_rows: list[list[str]] = []
    sync_rows: list[list[str]] = []

    for xdf_path in xdf_files:
        streams: list[dict[str, Any]] = []
        try:
            streams, _ = pyxdf.load_xdf(str(xdf_path))
        except Exception as exc:
            print(f"[raw_to_bids] WARNING: standard XDF load failed for {xdf_path}: {exc}")

        if not streams:
            emotibit_ids = _tolerant_xdf_stream_ids(xdf_path, re.compile(r"^Emotibit[_\-]|^EmotiBit[_\-]", re.IGNORECASE))
            if emotibit_ids:
                streams = _load_tolerant_xdf_streams(xdf_path, emotibit_ids)
                if streams:
                    print(
                        f"[raw_to_bids] INFO: recovered {len(streams)} EmotiBit stream(s) from {xdf_path.name}"
                    )
        for stream in streams:
            name = _stream_name(stream)
            stype = _stream_type(stream)
            stamps = stream.get("time_stamps", [])
            series = stream.get("time_series", [])
            if stamps is None:
                stamps = []
            if series is None:
                series = []

            for ts, sample in zip(stamps, series, strict=False):
                flat = _flatten_sample(sample)
                if name.startswith("AffectAI_") or stype.lower() == "markers":
                    value = flat[0] if flat else ""
                    event_rows.append([f"{float(ts):.6f}", name, stype, value])
                elif _is_tobii_stream_name(name):
                    tobii_rows.append([f"{float(ts):.6f}", name, stype, *flat])
                elif _is_emotibit_stream_name(name):
                    emotibit_rows.append([f"{float(ts):.6f}", name, stype, *flat])
                elif name.startswith("ffmpeg_") or stype in {"clock", "ffmpeg_progress"}:
                    sync_rows.append([f"{float(ts):.6f}", name, stype, *flat])

    if event_rows:
        beh_path = session_dir / "beh" / f"{base}_recording-lsl_events.tsv"
        _write_tsv(beh_path, ["lsl_time", "stream_name", "stream_type", "value"], event_rows)
        counts["beh"] = len(event_rows)

    if tobii_rows:
        max_cols = max(len(r) for r in tobii_rows)
        header = ["lsl_time", "stream_name", "stream_type"] + [f"value_{i}" for i in range(max_cols - 3)]
        norm_rows = [r + [""] * (max_cols - len(r)) for r in tobii_rows]
        et_path = session_dir / "et" / f"{base}_acq-lsl_tobii.tsv.gz"
        _write_tsv(et_path, header, norm_rows, gzip_out=True)
        counts["et"] = len(tobii_rows)

    if emotibit_rows:
        max_cols = max(len(r) for r in emotibit_rows)
        header = ["lsl_time", "stream_name", "stream_type"] + [f"value_{i}" for i in range(max_cols - 3)]
        norm_rows = [r + [""] * (max_cols - len(r)) for r in emotibit_rows]
        phys_path = session_dir / "physio" / f"{base}_acq-lsl_emotibit.tsv.gz"
        _write_tsv(phys_path, header, norm_rows, gzip_out=True)
        counts["physio"] = len(emotibit_rows)

    if sync_rows:
        max_cols = max(len(r) for r in sync_rows)
        header = ["lsl_time", "stream_name", "stream_type"] + [f"value_{i}" for i in range(max_cols - 3)]
        norm_rows = [r + [""] * (max_cols - len(r)) for r in sync_rows]
        sync_path = session_dir / "annot" / f"{base}_acq-lsl_sync.tsv"
        _write_tsv(sync_path, header, norm_rows)
        counts["annot"] = len(sync_rows)

    return counts


def convert(
    session_dir: Path,
    link: bool = False,
    require_xdf_extraction: bool = False,
    xdf_files: list[Path] | None = None,
) -> Path:
    if not session_dir.exists():
        raise FileNotFoundError(f"session-dir not found: {session_dir}")

    _, _, base = _session_entities(session_dir)
    local_counts = _copy_local_raw_into_modalities(session_dir, base, link)
    xdf_counts = _extract_from_xdf(
        session_dir,
        base,
        require_xdf_extraction=require_xdf_extraction,
        xdf_files=xdf_files,
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "mode": "link" if link else "copy",
        "local_raw_copies": local_counts,
        "xdf_extracted_rows": xdf_counts,
        "notes": [
            "Raw vendor files remain untouched under sourcedata/.",
            "XDF extraction runs only when pyxdf is available.",
        ],
    }

    out = session_dir / "annot" / f"{base}_raw_to_bids_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert raw AV/Recording/Tobii data into BIDS-oriented outputs")
    p.add_argument("--session-dir", type=Path, required=True, help="Path to ses-* directory")
    p.add_argument("--link", action="store_true", help="Use hard-links when possible instead of file copies")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = convert(args.session_dir, link=bool(args.link))
        print(f"[raw_to_bids] summary: {summary}")
        return 0
    except Exception as exc:
        print(f"[raw_to_bids] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
