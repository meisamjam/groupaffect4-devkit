#!/usr/bin/env python3
"""Offline compute stack helper for staging, planning, and execution.

This tool codifies the workflow documented in docs/offline_compute_stack.md for
the two-workstation post-collection processing setup.

Subcommands:
- init: create stack directories and an optional queue TSV file.
- queue-upsert: add or update one session row in the queue TSV.
- plan: emit a per-session command plan (JSON) for BIDS, pose, and QC stages.
- run: execute one stage command for a session.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:  # pyyaml is in project deps but keep import soft for tests
    _yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

QUEUE_COLUMNS = ["session_id", "status", "assigned_to", "notes", "updated_at"]


@dataclass(frozen=True)
class StackPaths:
    """Top-level stack paths used across planning and execution."""

    repo_root: Path
    data_root: Path
    config_dir: Path
    work_root: Path
    processed_root: Path
    queue_tsv: Path


def load_stack_config(config_path: Path) -> dict:
    """Load offline_compute_stack.yaml and return the parsed dict."""
    if _yaml is None:
        raise ImportError("pyyaml is required to load a stack config file. Run: pip install pyyaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Stack config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return _yaml.safe_load(fh)


def _stack_from_config(config: dict, role: str) -> StackPaths:
    """Build StackPaths from a parsed offline_compute_stack.yaml for the given role node."""
    nodes = config.get("nodes", {})
    if role not in nodes:
        available = ", ".join(nodes.keys())
        raise ValueError(f"Role '{role}' not found in stack config. Available: {available}")

    node = nodes[role]
    node_paths = node.get("paths", {})

    # gpu_main does not own a staging root; fall back to the share mount
    share = node_paths.get("share_mount", "Z:")
    staging = node_paths.get("staging_root", share)

    repo_root = Path(node_paths.get("repo_root", Path(__file__).resolve().parents[1]))
    data_root = Path(node_paths.get("data_root", f"{staging}/data"))
    config_dir = Path(node_paths.get("config_dir", f"{staging}/configs"))
    work_root = Path(node_paths.get("work_root", f"{staging}/work"))
    processed_root = Path(node_paths.get("processed_root", f"{staging}/processed"))
    queue_tsv = Path(node_paths.get("queue_tsv", f"{staging}/work_queue.tsv"))

    return StackPaths(
        repo_root=repo_root,
        data_root=data_root,
        config_dir=config_dir,
        work_root=work_root,
        processed_root=processed_root,
        queue_tsv=queue_tsv,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_session_id(session_id: str) -> str:
    if session_id.startswith("ses-"):
        return session_id
    return f"ses-{session_id}"


def _read_queue(queue_tsv: Path) -> list[dict[str, str]]:
    if not queue_tsv.exists():
        return []
    with queue_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({col: row.get(col, "") for col in QUEUE_COLUMNS})
        return rows


def _write_queue(queue_tsv: Path, rows: list[dict[str, str]]) -> None:
    queue_tsv.parent.mkdir(parents=True, exist_ok=True)
    with queue_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def upsert_queue_row(
    queue_tsv: Path,
    session_id: str,
    status: str,
    assigned_to: str,
    notes: str,
) -> dict[str, str]:
    """Add or update one queue row keyed by session_id."""
    normalized = _normalize_session_id(session_id)
    now = _utc_now_iso()
    rows = _read_queue(queue_tsv)

    new_row = {
        "session_id": normalized,
        "status": status,
        "assigned_to": assigned_to,
        "notes": notes,
        "updated_at": now,
    }

    for idx, row in enumerate(rows):
        if row.get("session_id") == normalized:
            rows[idx] = new_row
            _write_queue(queue_tsv, rows)
            return new_row

    rows.append(new_row)
    _write_queue(queue_tsv, rows)
    return new_row


def _find_first_session_dir(base: Path, session_id: str) -> Path | None:
    if not base.exists():
        return None
    needle = _normalize_session_id(session_id)
    for path in sorted(base.rglob(f"*{needle}*")):
        if path.is_dir():
            return path
    return None


def infer_session_inputs(data_root: Path, session_id: str) -> dict[str, Path | None]:
    """Infer source directories for a session from standard staging layout."""
    normalized = _normalize_session_id(session_id)
    return {
        "recording_session_dir": _find_first_session_dir(
            data_root / "affectai-capture-recording" / "sessions", normalized
        ),
        "av_session_dir": _find_first_session_dir(data_root / "AV", normalized),
        "stimuli_dir": _find_first_session_dir(data_root / "stimuli", normalized)
        or _find_first_session_dir(data_root / "affectai-capture-recording" / "stimuli" / "data", normalized),
        "tobii_dir": _find_first_session_dir(data_root / "Tobii", normalized),
    }


def build_stage_commands(
    stack: StackPaths,
    session_id: str,
    split_media: bool,
    max_workers: int,
    sub_id: str,
) -> dict[str, list[str]]:
    """Build command vectors for the offline stack stages."""
    normalized = _normalize_session_id(session_id)
    python_exe = sys.executable
    inputs = infer_session_inputs(stack.data_root, normalized)

    bids_session_dir = stack.processed_root / "bids" / sub_id / normalized
    work_session_dir = stack.work_root / normalized
    video_dir = work_session_dir / "video"
    pose_root = work_session_dir / "mediapipe"
    pipeline_out_dir = work_session_dir / "video_only_3d"
    pipeline_calibration = video_dir / "video_camera_calibration.toml"
    tracker_config = stack.repo_root / "configs" / "tobii_multicam_glasses_tracker.example.yaml"

    bids_cmd = [
        python_exe,
        str(stack.repo_root / "tools" / "multisource_to_bids_runs.py"),
        "--recording-session-dir",
        str(inputs["recording_session_dir"] or ""),
        "--av-session-dir",
        str(inputs["av_session_dir"] or ""),
        "--stimuli-dir",
        str(inputs["stimuli_dir"] or ""),
        "--output-session-dir",
        str(bids_session_dir),
    ]
    if inputs["tobii_dir"] is not None:
        bids_cmd.extend(["--tobii-dir", str(inputs["tobii_dir"])])
    if split_media:
        bids_cmd.append("--split-media")

    return {
        "bids": bids_cmd,
        "bids_batch": [
            python_exe,
            str(stack.repo_root / "tools" / "bids_processing_pipeline.py"),
            "--data-root",
            str(stack.data_root),
            "--output-root",
            str(stack.processed_root / "bids"),
            "--inventory",
            str(stack.data_root / "high_level_session_inventory.csv"),
            "--config-dir",
            str(stack.config_dir),
            "--max-workers",
            str(max_workers),
            "--split-media",
        ],
        "pose_json": [
            python_exe,
            str(stack.repo_root / "tools" / "test_mediapipe_pose.py"),
            "--session-dir",
            str(video_dir),
            "--write-json",
            str(pose_root),
            "--max-frames",
            "0",
            "--model-complexity",
            "2",
        ],
        "pose3d_dryrun": [
            python_exe,
            str(stack.repo_root / "tools" / "video_only_3d_pipeline.py"),
            "--calibration",
            str(pipeline_calibration),
            "--videos-dir",
            str(video_dir),
            "--tracker-config",
            str(tracker_config),
            "--pose-root",
            str(pose_root),
            "--output-dir",
            str(pipeline_out_dir),
            "--camera-zones",
            "cam1+cam4:0,1",
            "cam2+cam3:2,3",
            "--flip-cameras",
            "cam_0",
            "cam_1",
            "cam_2",
            "cam_3",
            "--refine-skeleton",
            "--dry-run",
        ],
        "pose3d": [
            python_exe,
            str(stack.repo_root / "tools" / "video_only_3d_pipeline.py"),
            "--calibration",
            str(pipeline_calibration),
            "--videos-dir",
            str(video_dir),
            "--tracker-config",
            str(tracker_config),
            "--pose-root",
            str(pose_root),
            "--output-dir",
            str(pipeline_out_dir),
            "--camera-zones",
            "cam1+cam4:0,1",
            "cam2+cam3:2,3",
            "--flip-cameras",
            "cam_0",
            "cam_1",
            "cam_2",
            "cam_3",
            "--refine-skeleton",
        ],
        "qc_sync": [
            python_exe,
            str(stack.repo_root / "tools" / "qc" / "qc_sync_report.py"),
            "--session-dir",
            str(bids_session_dir),
        ],
        "qc_gaze": [
            python_exe,
            str(stack.repo_root / "tools" / "qc" / "qc_tobii_world_gaze.py"),
            "--input-dir",
            str(pipeline_out_dir / "tobii_world"),
            "--output-dir",
            str(stack.processed_root / "qc" / normalized / "tobii_world"),
        ],
    }


def _stack_from_args(args: argparse.Namespace) -> StackPaths:
    # If a YAML config and role are supplied, load paths from there;
    # any individual CLI flags override the config values.
    if getattr(args, "stack_config", None) and args.stack_config is not None:
        cfg = load_stack_config(args.stack_config)
        role = getattr(args, "role", None) or "storage_worker"
        base = _stack_from_config(cfg, role)
        # CLI overrides: only apply if value differs from the argparse default
        defaults = {
            "repo_root": Path(__file__).resolve().parents[1],
            "data_root": Path("D:/affectai_stage/data"),
            "config_dir": Path("D:/affectai_stage/configs"),
            "work_root": Path("D:/affectai_work"),
            "processed_root": Path("D:/affectai_stage/processed"),
            "queue_tsv": Path("D:/affectai_stage/work_queue.tsv"),
        }
        return StackPaths(
            repo_root=(args.repo_root.resolve() if args.repo_root != defaults["repo_root"] else base.repo_root),
            data_root=(args.data_root.resolve() if args.data_root != defaults["data_root"] else base.data_root),
            config_dir=(args.config_dir.resolve() if args.config_dir != defaults["config_dir"] else base.config_dir),
            work_root=(args.work_root.resolve() if args.work_root != defaults["work_root"] else base.work_root),
            processed_root=(args.processed_root.resolve() if args.processed_root != defaults["processed_root"] else base.processed_root),
            queue_tsv=(args.queue_tsv.resolve() if args.queue_tsv != defaults["queue_tsv"] else base.queue_tsv),
        )

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    config_dir = args.config_dir.resolve()
    work_root = args.work_root.resolve()
    processed_root = args.processed_root.resolve()
    queue_tsv = args.queue_tsv.resolve()
    return StackPaths(
        repo_root=repo_root,
        data_root=data_root,
        config_dir=config_dir,
        work_root=work_root,
        processed_root=processed_root,
        queue_tsv=queue_tsv,
    )


def _init_stack(stack: StackPaths) -> None:
    for path in [
        stack.data_root,
        stack.config_dir,
        stack.work_root,
        stack.processed_root,
        stack.processed_root / "bids",
        stack.processed_root / "mocap",
        stack.processed_root / "qc",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if not stack.queue_tsv.exists():
        _write_queue(stack.queue_tsv, [])


def _run_command(command: list[str], cwd: Path) -> int:
    logger.info("Executing command: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(cwd), check=False)
    return int(result.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline compute stack helper for the AffectAI post-collection workflow."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing tools/ and configs/.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("D:/affectai_stage/data"),
        help="Staging data root used by the offline stack.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("D:/affectai_stage/configs"),
        help="Config directory used by BIDS packaging tools.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("D:/affectai_work"),
        help="Local GPU scratch directory for active sessions.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("D:/affectai_stage/processed"),
        help="Root for processed outputs (bids, mocap, qc).",
    )
    parser.add_argument(
        "--queue-tsv",
        type=Path,
        default=Path("D:/affectai_stage/work_queue.tsv"),
        help="TSV queue path for manual batch tracking.",
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to configs/offline_compute_stack.yaml. When supplied, paths are loaded from the config for the given --role.",
    )
    parser.add_argument(
        "--role",
        choices=["storage_worker", "gpu_main"],
        default="storage_worker",
        help="Which node's paths to use when loading from --stack-config (default: storage_worker).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create stack directories and queue TSV.")

    queue_upsert = subparsers.add_parser("queue-upsert", help="Add/update one queue row.")
    queue_upsert.add_argument("--session-id", required=True, help="Session ID, with or without ses-.")
    queue_upsert.add_argument("--status", required=True, help="Queue status label.")
    queue_upsert.add_argument("--assigned-to", default="", help="Worker name (e.g., gpu-main).")
    queue_upsert.add_argument("--notes", default="", help="Free-text notes.")

    plan = subparsers.add_parser("plan", help="Write per-session stack command plan JSON.")
    plan.add_argument("--session-id", required=True, help="Session ID, with or without ses-.")
    plan.add_argument("--sub-id", default="sub-01", help="BIDS participant folder to target.")
    plan.add_argument("--split-media", action="store_true", help="Include --split-media in BIDS step.")
    plan.add_argument("--max-workers", type=int, default=2, help="Worker count for bids_batch command.")
    plan.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the generated plan JSON.",
    )

    run = subparsers.add_parser("run", help="Run one stage command for a single session.")
    run.add_argument("--session-id", required=True, help="Session ID, with or without ses-.")
    run.add_argument(
        "--stage",
        required=True,
        choices=["bids", "bids_batch", "pose_json", "pose3d_dryrun", "pose3d", "qc_sync", "qc_gaze"],
        help="Stage to run.",
    )
    run.add_argument("--sub-id", default="sub-01", help="BIDS participant folder to target.")
    run.add_argument("--split-media", action="store_true", help="Include --split-media in BIDS step.")
    run.add_argument("--max-workers", type=int, default=2, help="Worker count for bids_batch command.")
    run.add_argument(
        "--execute",
        action="store_true",
        help="Execute command. Without this flag, only print the command JSON payload.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stack = _stack_from_args(args)

    if args.command == "init":
        _init_stack(stack)
        logger.info("Stack initialized at data=%s processed=%s", stack.data_root, stack.processed_root)
        logger.info("Queue TSV: %s", stack.queue_tsv)
        return 0

    if args.command == "queue-upsert":
        row = upsert_queue_row(
            queue_tsv=stack.queue_tsv,
            session_id=args.session_id,
            status=args.status,
            assigned_to=args.assigned_to,
            notes=args.notes,
        )
        logger.info("Queue row upserted: %s", json.dumps(row, ensure_ascii=True))
        return 0

    if args.command in {"plan", "run"}:
        commands = build_stage_commands(
            stack=stack,
            session_id=args.session_id,
            split_media=bool(args.split_media),
            max_workers=int(args.max_workers),
            sub_id=args.sub_id,
        )
        normalized = _normalize_session_id(args.session_id)
        payload = {
            "session_id": normalized,
            "paths": {
                "repo_root": str(stack.repo_root),
                "data_root": str(stack.data_root),
                "work_root": str(stack.work_root),
                "processed_root": str(stack.processed_root),
                "queue_tsv": str(stack.queue_tsv),
            },
            "commands": commands,
        }

        if args.command == "plan":
            output_path = args.output_json or (
                stack.processed_root / "plans" / f"{normalized}_offline_compute_plan.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Plan written: %s", output_path)
            return 0

        stage_command = commands[args.stage]
        logger.info("Stage '%s' command: %s", args.stage, json.dumps(stage_command, ensure_ascii=True))
        if not args.execute:
            logger.info("Dry-run mode (no execution). Use --execute to run the stage command.")
            return 0
        return _run_command(stage_command, cwd=stack.repo_root)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())