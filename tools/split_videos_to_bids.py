#!/usr/bin/env python3
"""Split raw AV-PC camera recordings into per-task BIDS video clips.

This script is a post-hoc companion to ``multisource_to_bids_runs.py``:
it handles only the video-splitting step for sessions that have already been
through the BIDS pipeline but whose ``video/`` folder is missing or incomplete.

The per-task task windows are read from the ``annot/*_task_run_windows.tsv``
file that ``multisource_to_bids_runs.py`` already wrote.  Raw camera MKVs
are read from the AV-PC session tree; task clips (.mp4) are written to
``<bids_session>/video/``.

Usage:
    python tools/split_videos_to_bids.py \\
        --bids-root  F:\\processed_data \\
        --av-root    F:\\affectai-capture-av\\sessions \\
        [--sub sub-01] \\
        [--session ses-20260312_grp-07_run01] \\
        [--ffmpeg ffmpeg] \\
        [--ffprobe ffprobe] \\
        [--dry-run]

Arguments:
    --bids-root   Root of the processed BIDS output tree (contains sub-*/ses-*).
    --av-root     Root of the AV-PC session tree.  Searched under
                  ``<av-root>/final/sub-*``, ``<av-root>/pilot/sub-*``,
                  ``<av-root>/test/sub-*``, and ``<av-root>/sub-*`` directly.
    --sub         Limit to a single BIDS subject directory (e.g. ``sub-01``).
                  Defaults to all subject directories found.
    --session     Limit to a single BIDS session directory name
                  (e.g. ``ses-20260312_grp-07_run01``).
    --ffmpeg      Path to ffmpeg binary.  Default: ``ffmpeg``.
    --ffprobe     Path to ffprobe binary.  Default: ``ffprobe``.
    --dry-run     Print what would be done without writing any files.
    --verbose     Enable DEBUG-level logging.
    --skip-existing
                  Skip a session entirely if its video/ folder already contains
                  at least one .mp4 clip.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-load multisource_to_bids_runs.py to reuse its video-splitting logic.
# ---------------------------------------------------------------------------

def _load_multisource_module():
    module_path = Path(__file__).resolve().parent / "multisource_to_bids_runs.py"
    spec = importlib.util.spec_from_file_location("multisource_to_bids_runs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Session discovery helpers
# ---------------------------------------------------------------------------

def _find_av_session(bids_ses_dir: Path, av_root: Path) -> Path | None:
    """Locate the AV-PC session directory matching *bids_ses_dir*.

    Searches ``<av_root>/{final,pilot,test,""}/sub-*/ses-<name>`` where
    ``ses-<name>`` equals the BIDS session directory name.
    """
    ses_name = bids_ses_dir.name          # e.g. ses-20260312_grp-07_run01
    sub_name = bids_ses_dir.parent.name   # e.g. sub-01

    # Tiers to search under av_root
    tiers = ["final", "pilot", "test", ""]
    for tier in tiers:
        if tier:
            candidate = av_root / tier / sub_name / ses_name
        else:
            candidate = av_root / sub_name / ses_name
        if candidate.is_dir():
            logger.debug("Found AV session: %s", candidate)
            return candidate

    return None


def _find_task_windows_tsv(bids_ses_dir: Path) -> Path | None:
    """Return the authoritative *_task_run_windows.tsv inside annot/."""
    annot = bids_ses_dir / "annot"
    if not annot.is_dir():
        return None
    candidates = sorted(annot.glob("*_task_run_windows.tsv"))
    return candidates[-1] if candidates else None


def _read_task_windows(tsv: Path) -> list[dict[str, Any]]:
    """Parse task_run_windows.tsv into window dicts for _split_media_runs."""
    windows: list[dict[str, Any]] = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("task") or row.get("segment") or "UNK"
            try:
                start = float(row.get("start_wall_clock", "0") or 0.0)
                end = float(row.get("end_wall_clock", "0") or 0.0)
            except ValueError:
                continue
            windows.append(
                {
                    "task": label,
                    "run": row.get("run", "01"),
                    "start_wall_clock": start,
                    "end_wall_clock": end,
                }
            )
    return windows


def _discover_sessions(
    bids_root: Path,
    sub_filter: str | None,
    ses_filter: str | None,
) -> list[Path]:
    """Return sorted list of BIDS session dirs matching the optional filters."""
    sessions: list[Path] = []
    for sub_dir in sorted(bids_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        if sub_filter and sub_dir.name != sub_filter:
            continue
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue
            if ses_filter and ses_dir.name != ses_filter:
                continue
            sessions.append(ses_dir)
    return sessions


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def _process_session(
    bids_ses_dir: Path,
    av_ses_dir: Path,
    windows: list[dict[str, Any]],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    dry_run: bool,
    skip_existing: bool,
    multisource: Any,
) -> dict[str, Any]:
    ses_label = bids_ses_dir.name
    video_dir = bids_ses_dir / "video"

    if skip_existing and video_dir.is_dir():
        clips = list(video_dir.glob("*.mp4"))
        if clips:
            logger.info(
                "[%s] SKIP — video/ already has %d clip(s)", ses_label, len(clips)
            )
            return {"session": str(bids_ses_dir), "status": "skipped_existing", "clips": []}

    logger.info("[%s] Processing — task windows: %d, AV: %s", ses_label, len(windows), av_ses_dir)

    if dry_run:
        logger.info("[%s] DRY-RUN — would call _split_media_runs", ses_label)
        return {"session": str(bids_ses_dir), "status": "dry_run", "clips": []}

    result = multisource._split_media_runs(
        session_dir=bids_ses_dir,
        windows=windows,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        split_video=True,
        split_audio=False,
        av_session_dir=av_ses_dir,
        tobii_dirs=None,  # Tobii scene clips are handled by the full pipeline
    )

    n_clips = len(result.get("generated_clips", []))
    n_skipped = len(result.get("skipped", []))
    logger.info(
        "[%s] Done — %d clip(s) written, %d skipped", ses_label, n_clips, n_skipped
    )

    if n_skipped:
        for s in result.get("skipped", []):
            logger.debug("[%s] Skipped: %s — %s", ses_label, s.get("path"), s.get("reason"))

    return {
        "session": str(bids_ses_dir),
        "status": "ok",
        "clips": result.get("generated_clips", []),
        "skipped": result.get("skipped", []),
        "video_clock_anchors_tsv": result.get("video_clock_anchors_tsv"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split raw AV-PC camera recordings into per-task BIDS video clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bids-root",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root of the processed BIDS output tree (contains sub-*/ses-*).",
    )
    parser.add_argument(
        "--av-root",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Root of the AV-PC session tree. "
            "Searched under <av-root>/final/sub-*, /pilot/sub-*, /test/sub-*, and "
            "<av-root>/sub-* directly."
        ),
    )
    parser.add_argument(
        "--sub",
        default=None,
        metavar="SUB",
        help="Limit to a single subject directory name, e.g. sub-01.",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="SES",
        help="Limit to a single session directory name, e.g. ses-20260312_grp-07_run01.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="BIN",
        help="Path to ffmpeg binary (default: ffmpeg).",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="BIN",
        help="Path to ffprobe binary (default: ffprobe).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions whose video/ folder already contains at least one .mp4 clip.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    bids_root: Path = args.bids_root.resolve()
    av_root: Path = args.av_root.resolve()

    if not bids_root.is_dir():
        logger.error("--bids-root does not exist: %s", bids_root)
        return 1
    if not av_root.is_dir():
        logger.error("--av-root does not exist: %s", av_root)
        return 1

    # Load multisource module once.
    multisource = _load_multisource_module()

    sessions = _discover_sessions(bids_root, args.sub, args.session)
    if not sessions:
        logger.error("No matching sessions found under %s", bids_root)
        return 1

    logger.info("Found %d session(s) to process", len(sessions))

    report: list[dict[str, Any]] = []
    errors = 0

    for bids_ses_dir in sessions:
        ses_label = f"{bids_ses_dir.parent.name}/{bids_ses_dir.name}"

        # 1. Find matching AV session.
        av_ses_dir = _find_av_session(bids_ses_dir, av_root)
        if av_ses_dir is None:
            logger.warning("[%s] No AV session found — skipping", ses_label)
            report.append({"session": str(bids_ses_dir), "status": "no_av_session"})
            continue

        # 2. Load task windows from existing annot/.
        windows_tsv = _find_task_windows_tsv(bids_ses_dir)
        if windows_tsv is None:
            logger.warning(
                "[%s] No *_task_run_windows.tsv found in annot/ — skipping", ses_label
            )
            report.append({"session": str(bids_ses_dir), "status": "no_task_windows"})
            continue

        windows = _read_task_windows(windows_tsv)
        if not windows:
            logger.warning("[%s] task_run_windows.tsv is empty — skipping", ses_label)
            report.append({"session": str(bids_ses_dir), "status": "empty_task_windows"})
            continue

        # 3. Split videos.
        try:
            result = _process_session(
                bids_ses_dir=bids_ses_dir,
                av_ses_dir=av_ses_dir,
                windows=windows,
                ffmpeg_bin=args.ffmpeg,
                ffprobe_bin=args.ffprobe,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                multisource=multisource,
            )
            report.append(result)
        except Exception as exc:
            logger.error("[%s] Error: %s", ses_label, exc, exc_info=args.verbose)
            report.append({"session": str(bids_ses_dir), "status": "error", "error": str(exc)})
            errors += 1

    # Summary
    ok = sum(1 for r in report if r.get("status") == "ok")
    skipped = sum(1 for r in report if r.get("status") in {"skipped_existing", "dry_run"})
    no_av = sum(1 for r in report if r.get("status") == "no_av_session")
    errs = sum(1 for r in report if r.get("status") == "error")

    total_clips = sum(len(r.get("clips", [])) for r in report)

    logger.info(
        "Summary: %d processed, %d skipped, %d no-AV-match, %d errors | %d clips written",
        ok, skipped, no_av, errs, total_clips,
    )

    if not args.dry_run:
        summary_path = bids_root / f"split_videos_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Summary written to: %s", summary_path)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
