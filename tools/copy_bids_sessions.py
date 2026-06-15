#!/usr/bin/env python3
"""
Copy/Link existing BIDS-organized sessions to output directory.
Fast parallel copying of already-processed data.
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def copy_or_link_file(src: Path, dst: Path, use_links: bool = False) -> bool:
    """Copy or link a single file."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if dst.exists():
            return True  # Already exists
        
        if use_links:
            try:
                os.link(src, dst)
                return True
            except OSError:
                pass  # Fall through to copy
        
        # Copy file
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.warning(f"Failed to copy {src.name}: {e}")
        return False


def copy_session_tree(src_session: Path, dst_session: Path, use_links: bool = False) -> int:
    """Copy entire session tree, return count of files copied."""
    if not src_session.exists():
        return 0
    
    copied = 0
    for src_file in src_session.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_session)
            dst_file = dst_session / rel_path
            
            if copy_or_link_file(src_file, dst_file, use_links):
                copied += 1
    
    return copied


def process_subject_session(src_root: Path, dst_root: Path, subject_session: Path, use_links: bool) -> str:
    """Process a single subject-session combination."""
    
    src_path = src_root / subject_session
    dst_path = dst_root / subject_session
    
    if not src_path.exists():
        return f"⚠ SKIP {subject_session}: source not found"
    
    try:
        copied = copy_session_tree(src_path, dst_path, use_links)
        return f"✓ {subject_session}: {copied} files"
    except Exception as e:
        return f"✗ {subject_session}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Copy existing BIDS sessions to output directory"
    )
    
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing BIDS sessions",
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
        "--use-links",
        action="store_true",
        help="Use hard links instead of copying (faster, saves space)",
    )
    
    args = parser.parse_args()
    
    # Create output root
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Find all subject-session directories (handle nested structure)
    sessions = []
    for session_dir in sorted(args.input_root.rglob("ses-*")):
        if session_dir.is_dir():
            rel_path = session_dir.relative_to(args.input_root)
            sessions.append(rel_path)
    
    logger.info(f"Found {len(sessions)} sessions to copy")
    logger.info(f"Using {'hard links' if args.use_links else 'copying'}")
    
    if not sessions:
        logger.error("No sessions found!")
        return 1
    
    # Copy sessions in parallel
    with mp.Pool(processes=args.max_workers) as pool:
        tasks = [
            (args.input_root, args.output_root, session, args.use_links)
            for session in sessions
        ]
        
        results = pool.starmap(process_subject_session, tasks)
    
    # Print results
    print("\n" + "=" * 70)
    print("BIDS DATA COPY SUMMARY")
    print("=" * 70)
    for result in results:
        print(result)
    
    successful = sum(1 for r in results if r.startswith("✓"))
    skipped = sum(1 for r in results if r.startswith("⚠"))
    failed = sum(1 for r in results if r.startswith("✗"))
    
    print("=" * 70)
    print(f"Successful: {successful}/{len(sessions)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_root}")
    print("=" * 70 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
