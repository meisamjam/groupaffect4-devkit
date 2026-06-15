#!/usr/bin/env python3
"""Upload one session's raw data to Azure Blob storage.

This tool is intended to run on either AV PC or Recording PC right after a
session. It uploads the whole session directory (including sourcedata and sync
artifacts) to a role-scoped path in Azure Blob using ``azcopy``.

Example:
    python tools/upload_raw_data.py \
        --session-dir sessions/sub-01/ses-20260307_grpA_run01 \
        --destination-url "https://account.blob.core.windows.net/affectai?<SAS>" \
        --role av-pc
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _parse_session_id(session_dir: Path) -> str:
    name = session_dir.name
    if name.startswith("ses-"):
        return name[4:]
    return name


def _session_sync_manifest(session_dir: Path, role: str) -> Path:
    sync_dir = session_dir / "sourcedata" / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    out_path = sync_dir / f"raw_upload_manifest_{role}.json"

    xdf_files = sorted(str(p.relative_to(session_dir)) for p in session_dir.glob("*.xdf"))
    ffmpeg_lsl = sorted(str(p.relative_to(session_dir)) for p in (session_dir / "sourcedata" / "av").rglob("lsl/*.jsonl"))
    frame_logs = sorted(str(p.relative_to(session_dir)) for p in (session_dir / "sourcedata" / "av").rglob("frame_logs/*.jsonl"))
    progress_logs = sorted(str(p.relative_to(session_dir)) for p in (session_dir / "sourcedata" / "av").rglob("progress_logs/*.tsv"))
    tobii_lsl = sorted(str(p.relative_to(session_dir)) for p in (session_dir / "sourcedata" / "tobii_lsl").glob("*.ndjson"))
    events_tsv = session_dir / "events.tsv"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "host": socket.gethostname(),
        "session_dir": str(session_dir),
        "sync_artifacts": {
            "events_tsv": {
                "path": str(events_tsv.relative_to(session_dir)) if events_tsv.exists() else "events.tsv",
                "exists": events_tsv.exists(),
                "size": events_tsv.stat().st_size if events_tsv.exists() else 0,
            },
            "xdf_files": xdf_files,
            "ffmpeg_lsl_jsonl": ffmpeg_lsl,
            "ffmpeg_frame_logs": frame_logs,
            "ffmpeg_progress_logs": progress_logs,
            "tobii_lsl_ndjson": tobii_lsl,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _azcopy_destination(destination_url: str, session_id: str, role: str) -> str:
    parsed = urlsplit(destination_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("destination-url must be a full Azure Blob URL")
    base_path = parsed.path.rstrip("/")
    dest_path = f"{base_path}/{session_id}/{role}"
    return urlunsplit((parsed.scheme, parsed.netloc, dest_path, parsed.query, parsed.fragment))


def _upload_session_via_azcopy(
    session_dir: Path,
    destination_url: str,
    role: str,
    session_id: str,
    dry_run: bool,
) -> int:
    azcopy = shutil.which("azcopy")
    if not azcopy:
        raise RuntimeError("azcopy is not installed or not on PATH")

    dest = _azcopy_destination(destination_url, session_id, role)

    cmd = [
        azcopy,
        "copy",
        str(session_dir),
        dest,
        "--recursive=true",
        "--overwrite=ifSourceNewer",
    ]

    print("[upload] backend=azcopy")
    print(f"[upload] destination={dest}")
    print(f"[upload] command={' '.join(cmd)}")

    if dry_run:
        return 0

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _upload_session_via_sdk(
    session_dir: Path,
    account_name: str,
    account_key: str,
    container_name: str,
    role: str,
    session_id: str,
    dry_run: bool,
) -> int:
    files = sorted(p for p in session_dir.rglob("*") if p.is_file())
    print("[upload] backend=azure-sdk")
    print(
        f"[upload] destination=https://{account_name}.blob.core.windows.net/{container_name}/{session_id}/{role}"
    )
    print(f"[upload] files={len(files)}")

    if dry_run:
        return 0

    try:
        from azure.storage.blob import BlobServiceClient
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "azure-storage-blob is required for SDK upload mode. "
            "Install it in your active environment: pip install azure-storage-blob"
        ) from exc

    connection_string = (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        "EndpointSuffix=core.windows.net"
    )
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
        print(f"[upload] created container: {container_name}")
    except Exception:
        pass

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(session_dir).as_posix()
        blob_name = f"{session_id}/{role}/{rel}"
        print(f"[upload] ({idx}/{len(files)}) {blob_name}")
        with path.open("rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True, max_concurrency=4)
    return 0


def upload_session(
    session_dir: Path,
    role: str,
    destination_url: str | None = None,
    account_name: str | None = None,
    account_key: str | None = None,
    container_name: str | None = None,
    dry_run: bool = False,
) -> int:
    _must_exist(session_dir, "Session directory")
    session_id = _parse_session_id(session_dir)

    manifest_path = _session_sync_manifest(session_dir, role)

    print(f"[upload] session_dir={session_dir}")
    print(f"[upload] manifest={manifest_path}")

    if destination_url:
        return _upload_session_via_azcopy(
            session_dir=session_dir,
            destination_url=destination_url,
            role=role,
            session_id=session_id,
            dry_run=dry_run,
        )

    if account_name and account_key and container_name:
        return _upload_session_via_sdk(
            session_dir=session_dir,
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
            role=role,
            session_id=session_id,
            dry_run=dry_run,
        )

    raise ValueError(
        "Provide either --destination-url (azcopy mode) or "
        "--account-name + --account-key-env + --container-name (SDK mode)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload session raw data to Azure Blob (via azcopy)")
    p.add_argument("--session-dir", type=Path, required=True, help="Path to ses-* directory")
    p.add_argument("--destination-url", help="Azure Blob container URL (optionally with SAS); uses azcopy")
    p.add_argument("--account-name", help="Azure Storage account name; uses SDK mode")
    p.add_argument("--container-name", help="Azure Blob container name; uses SDK mode")
    p.add_argument(
        "--account-key-env",
        default="AFFECTAI_AZURE_ACCOUNT_KEY",
        help="Environment variable name that stores the Azure account key for SDK mode",
    )
    p.add_argument("--role", choices=["av-pc", "recording-pc"], required=True, help="Uploader role")
    p.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        account_key = ""
        if args.account_name or args.container_name:
            account_key = os.getenv(args.account_key_env, "")
            if not account_key:
                account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")

        return upload_session(
            session_dir=args.session_dir,
            role=args.role,
            destination_url=args.destination_url,
            account_name=args.account_name,
            account_key=account_key or None,
            container_name=args.container_name,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[upload] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
