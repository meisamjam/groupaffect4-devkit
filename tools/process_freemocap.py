#!/usr/bin/env python3
"""
FreeMoCap post-hoc processing tool for AffectAI Capture sessions.

Processes video recordings (e.g., from Jabra cameras) to extract 3D skeleton data.

Usage:
    python tools/process_freemocap.py \
        --session-dir data/sub-01/ses-01 \
        --video-label 'jabra_panacast' \
        --task T1 --task T2

    # Process all task videos in a session
    python tools/process_freemocap.py \
        --session-dir data/sub-01/ses-01 \
        --auto-discover

    # Specify video file directly
    python tools/process_freemocap.py \
        --session-dir data/sub-01/ses-01 \
        --video-file data/sub-01/ses-01/video/task-T1.mp4 \
        --task T1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src/ to path for imports
TOOLS_DIR = Path(__file__).resolve().parent
SRC_DIR = TOOLS_DIR.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_video_files(
    session_dir: Path,
    video_label: str | None = None,
    task_list: list[str] | None = None,
) -> dict[str, Path]:
    """
    Discover video files in session directory.

    Args:
        session_dir: Path to session directory (sub-XX/ses-YY/)
        video_label: Optional acquisition label filter (e.g., 'jabra_panacast')
        task_list: Optional list of task names (e.g., ['T1', 'T2'])

    Returns:
        Dictionary mapping task_id -> video_path

    Looks for files matching BIDS pattern:
        sub-XX_ses-YY_task-T[N](_run-[N])?_video.mp4
    """
    video_dir = session_dir / "video"
    if not video_dir.exists():
        logger.warning(f"Video directory not found: {video_dir}")
        return {}

    videos = {}

    # Find all video files
    from itertools import chain
    for video_file in chain(video_dir.glob("*.mp4"), video_dir.glob("*.mov"), video_dir.glob("*.avi")):
        # Extract task info from filename
        filename = video_file.stem

        # Skip if label filter specified and doesn't match
        if video_label and video_label not in filename:
            continue

        # Extract task name (look for 'task-T[N]' pattern)
        task_name = None
        if "task-" in filename:
            parts = filename.split("task-")[1].split("_")[0]
            task_name = f"T{parts}" if not parts.startswith("T") else parts

        # Skip if task list specified and task not in list
        if task_list and task_name not in task_list:
            continue

        if task_name:
            # Extract run number if present
            run_num = "01"
            if "run-" in filename:
                run_str = filename.split("run-")[1].split("_")[0]
                run_num = run_str.zfill(2)

            task_id = f"{task_name}_run{run_num}"
            videos[task_id] = video_file

    logger.info(f"Found {len(videos)} video files in {video_dir}")
    for task_id, path in videos.items():
        logger.info(f"  - {task_id}: {path.name}")

    return videos


def process_session(
    session_dir: Path,
    video_files: dict[str, Path] | None = None,
    video_label: str | None = None,
    task_list: list[str] | None = None,
    auto_discover: bool = False,
    output_label: str = "freemocap",
    confidence_threshold: float = 0.5,
    save_config: bool = True,
) -> dict[str, Any]:
    """
    Process motion capture for a session.

    Args:
        session_dir: Path to session directory
        video_files: Optional pre-defined mapping of task -> video path
        video_label: Filter videos by acquisition label
        task_list: Process only specified tasks
        auto_discover: Auto-discover video files
        output_label: Output acquisition label
        confidence_threshold: Confidence threshold for keypoints
        save_config: Save processing config as JSON

    Returns:
        Results dictionary with processing status per task
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    logger.info(f"Processing session: {session_dir}")

    # Discover video files if not provided
    if video_files is None:
        video_files = get_video_files(
            session_dir, video_label=video_label, task_list=task_list
        )

    if not video_files:
        logger.warning(
            "No video files found. "
            "Specify --video-file or use --auto-discover"
        )
        return {"error": "No video files found"}

    # Import processor
    try:
        from affectai_capture.devices.freemocap_processor import (
            FreeMoCapProcessor,
        )
    except ImportError as e:
        raise ImportError(
            "Could not import FreeMoCapProcessor. "
            "Ensure src/ is in Python path."
        ) from e

    # Initialize processor
    processor = FreeMoCapProcessor(session_dir)

    # Process each video
    results = {}
    for task_id, video_path in video_files.items():
        try:
            task_name, run = _parse_task_id_from_path(task_id)

            logger.info(f"Processing {task_id}...")
            outputs = processor.process_video(
                video_path=video_path,
                output_label=output_label,
                task_name=task_name,
                run=run,
                confidence_threshold=confidence_threshold,
            )

            results[task_id] = {
                "status": "completed",
                "outputs": {k: str(v) for k, v in outputs.items()},
            }
            logger.info(f"  ✓ {task_id} completed")

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"  ✗ {task_id} failed: {e}")
            results[task_id] = {
                "status": "failed",
                "error": str(e),
            }

    # Save processing config
    if save_config:
        config = {
            "session_dir": str(session_dir),
            "output_label": output_label,
            "confidence_threshold": confidence_threshold,
            "processed_videos": len([v for v in results.values() if v["status"] == "completed"]),
            "failed_videos": len([v for v in results.values() if v["status"] == "failed"]),
            "results": results,
        }

        config_file = session_dir / "mocap" / "freemocap_processing_log.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved processing log: {config_file}")

    return results


def _parse_task_id_from_path(task_id: str) -> tuple[str, int]:
    """Parse task_id into task name and run number."""
    if "_run" in task_id:
        task_part, run_part = task_id.rsplit("_run", 1)
        return task_part, int(run_part)
    return task_id, 1


def main() -> int:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="FreeMoCap post-hoc processor for AffectAI Capture sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to session directory (e.g., data/sub-01/ses-01)",
    )

    parser.add_argument(
        "--video-file",
        type=Path,
        help="Specific video file to process",
    )

    parser.add_argument(
        "--video-label",
        type=str,
        help="Filter videos by acquisition label (e.g., 'jabra_panacast')",
    )

    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task(s) to process (e.g., T1, T2). Can specify multiple times.",
    )

    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover all video files in session",
    )

    parser.add_argument(
        "--output-label",
        type=str,
        default="freemocap",
        help="Acquisition label for outputs (default: freemocap)",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold (default: 0.5)",
    )

    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Don't save processing config JSON",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Handle single video file option
        video_files = None
        if args.video_file:
            if not args.video_file.exists():
                raise FileNotFoundError(f"Video file not found: {args.video_file}")

            task_name = args.tasks[0] if args.tasks else "unknown"
            run = 1
            if "_run" in args.video_file.stem:
                parts = args.video_file.stem.split("_run")
                run = int(parts[1].split("_")[0])

            video_files = {f"{task_name}_run{run:02d}": args.video_file}

        # Process session
        results = process_session(
            session_dir=args.session_dir,
            video_files=video_files,
            video_label=args.video_label,
            task_list=args.tasks,
            auto_discover=args.auto_discover,
            output_label=args.output_label,
            confidence_threshold=args.confidence_threshold,
            save_config=not args.no_save_config,
        )

        # Print summary
        completed = sum(
            1 for v in results.values() if isinstance(v, dict) and v.get("status") == "completed"
        )
        failed = sum(
            1 for v in results.values() if isinstance(v, dict) and v.get("status") == "failed"
        )

        print("\n" + "=" * 60)
        print("FreeMoCap Processing Summary")
        print("=" * 60)
        print(f"Session: {args.session_dir}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print("=" * 60 + "\n")

        return 0 if failed == 0 else 1

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
