#!/usr/bin/env python3
"""
Simplified BIDS Pipeline Launcher - Works with actual data structure.
Processes all sessions with multiprocessing and proper argument handling.
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Session metadata from inventory."""
    session_id: str
    group_id: str
    participants: list[str]
    raw_modalities: list[str]
    phase: str


def load_sessions(inventory_path: Path) -> list[SessionInfo]:
    """Load sessions from CSV inventory."""
    sessions = []
    
    with open(inventory_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            session_id = row.get("session", f"session-{idx}")
            group_id = row.get("group_id", "")
            participants = row.get("participants_ids", "").split(";") if row.get("participants_ids") else []
            raw_modalities = row.get("raw_modalities", "").split(";") if row.get("raw_modalities") else []
            
            # Detect phase from phase_tags
            phase_tags = row.get("phase_tags", "").split(";") if row.get("phase_tags") else []
            phase = phase_tags[0] if phase_tags else "final"
            
            sessions.append(SessionInfo(
                session_id=session_id,
                group_id=group_id,
                participants=[p.strip() for p in participants if p.strip()],
                raw_modalities=[m.strip() for m in raw_modalities if m.strip()],
                phase=phase,
            ))
    
    return sessions


def find_session_sources(
    session: SessionInfo,
    data_root: Path
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find source directories for a session."""
    
    # Recording dir (by phase and session name)
    recording_root = data_root / "affectai-capture-recording" / "sessions" / session.phase
    recording_sessions = list(recording_root.glob(f"*{session.session_id}*"))
    recording_dir = recording_sessions[0] if recording_sessions else None
    
    # AV dir (from AV folder by group)
    av_root = data_root / "AV" / session.phase
    av_sessions = list(av_root.glob(f"*{session.group_id}*")) if av_root.exists() else []
    av_dir = av_sessions[0] if av_sessions else None
    
    # Stimuli dir
    stimuli_root = data_root / "affectai-capture-recording" / "stimuli" / "data"
    stimuli_sessions = list(stimuli_root.glob(f"*{session.session_id}*")) if stimuli_root.exists() else []
    stimuli_dir = stimuli_sessions[0] if stimuli_sessions else (stimuli_root if stimuli_root.exists() else None)
    
    return recording_dir, av_dir, stimuli_dir


def process_session(
    session: SessionInfo,
    data_root: Path,
    output_root: Path,
    python_exe: str,
    tools_path: Path,
) -> bool:
    """Process a single session."""
    
    logger.info(f"Processing {session.session_id} ({session.group_id})...")
    
    # Find source directories
    recording_dir, av_dir, stimuli_dir = find_session_sources(session, data_root)
    
    # Validate we have required sources
    if not recording_dir or not recording_dir.exists():
        logger.warning(f"  ⚠ Recording dir not found: {recording_dir}")
        # Continue anyway - might have AV-only or other modalities
    
    # Create output directory
    output_dir = output_root / f"sub-01" / f"ses-{session.session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Build command for multisource_to_bids_runs.py
        cmd = [
            python_exe,
            str(tools_path / "multisource_to_bids_runs.py"),
            "--recording-session-dir", str(recording_dir or ""),
            "--av-session-dir", str(av_dir or ""),
            "--stimuli-dir", str(stimuli_dir or ""),
            "--output-session-dir", str(output_dir),
        ]
        
        # Remove empty directories
        cmd = [c for c in cmd if c]
        
        logger.info(f"  Running: {' '.join(cmd[-3:])}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.warning(f"  ⚠ Processing failed: {result.stderr[:200]}")
            return False
        
        logger.info(f"  ✓ {session.session_id} completed")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ {session.session_id} timeout")
        return False
    except Exception as e:
        logger.error(f"  ✗ {session.session_id} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simplified BIDS Pipeline for AffectAI Data"
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root data directory (contains affectai-capture-recording, AV, etc.)",
    )
    
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for BIDS data",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Path to session inventory CSV (default: data-root/high_level_session_inventory.csv)",
    )
    
    args = parser.parse_args()
    
    # Get Python exe and tools path
    python_exe = sys.executable
    tools_path = Path(__file__).parent
    
    # Load inventory
    inventory_path = args.inventory or (args.data_root / "high_level_session_inventory.csv")
    
    if not inventory_path.exists():
        logger.error(f"Inventory not found: {inventory_path}")
        return 1
    
    logger.info(f"Loading sessions from {inventory_path}")
    sessions = load_sessions(inventory_path)
    logger.info(f"Loaded {len(sessions)} sessions")
    
    # Create output root
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Process sessions
    logger.info(f"Starting processing with {args.max_workers} workers...")
    
    with mp.Pool(processes=args.max_workers) as pool:
        tasks = [
            (session, args.data_root, args.output_root, python_exe, tools_path)
            for session in sessions
        ]
        
        results = pool.starmap(process_session, tasks)
    
    # Summary
    successful = sum(results)
    failed = len(results) - successful
    
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total sessions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/len(results):.1f}%")
    print("=" * 70 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
