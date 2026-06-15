#!/usr/bin/env python3
"""Ingest manually downloaded Tobii recordings into a session folder.

Use this when Tobii scene videos/recordings are downloaded after the session.
The tool copies downloaded files into:

    <session>/sourcedata/tobii_device/<device_id>/

and writes an index file:

    <session>/sourcedata/tobii_device/tobii_download_index.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov"}
JSON_EXTS = {".json", ".ndjson"}


def _copy_tree(src: Path, dst: Path) -> int:
    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied += 1
    return copied


def ingest_downloads(session_dir: Path, download_root: Path, device_id: str) -> Path:
    if not session_dir.exists():
        raise FileNotFoundError(f"session-dir not found: {session_dir}")
    if not download_root.exists():
        raise FileNotFoundError(f"download-root not found: {download_root}")

    target = session_dir / "sourcedata" / "tobii_device" / device_id
    copied = _copy_tree(download_root, target)

    videos = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    metadata = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() and p.suffix.lower() in JSON_EXTS)

    index_path = session_dir / "sourcedata" / "tobii_device" / "tobii_download_index.json"
    existing = {"devices": []}
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"devices": []}

    devices = [d for d in existing.get("devices", []) if d.get("device_id") != device_id]
    devices.append(
        {
            "device_id": device_id,
            "ingested_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(download_root),
            "target": str(target),
            "files_copied": copied,
            "video_files": videos,
            "metadata_files": metadata,
        }
    )

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "devices": sorted(devices, key=lambda d: str(d.get("device_id", ""))),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest downloaded Tobii files into session sourcedata")
    p.add_argument("--session-dir", type=Path, required=True, help="Session directory (ses-*)")
    p.add_argument("--download-root", type=Path, required=True, help="Local folder with downloaded Tobii files")
    p.add_argument("--device-id", required=True, help="Device label (e.g., p1, p2, tobii-01)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        index_path = ingest_downloads(args.session_dir, args.download_root, args.device_id)
        print(f"[ingest_tobii] index written: {index_path}")
        return 0
    except Exception as exc:
        print(f"[ingest_tobii] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
