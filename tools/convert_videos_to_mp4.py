#!/usr/bin/env python3
"""
Convert split BIDS videos to MP4 format for 3D modeling pipelines.

This tool uses ffmpeg (from imageio-ffmpeg) to:
1. Read all video files in a BIDS session (MKV, AVI, MOV, etc)
2. Convert to H.264 + AAC MP4 format with optimized settings
3. Replace original files with MP4 versions

Installation:
    pip install imageio-ffmpeg

Usage:
    python convert_videos_to_mp4.py --session-dir <path> [--preset medium]

Performance:
  - Uses codec copy (stream copy) mode for fastest conversion
  - For full re-encoding with quality control, set --codec-copy false
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("convert_videos_to_mp4")


def _get_ffmpeg_exe() -> str:
    """Get ffmpeg executable from imageio-ffmpeg package."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        log.info(f"Using FFmpeg from imageio-ffmpeg: {exe}")
        return exe
    except ImportError:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")
    except Exception as e:
        raise RuntimeError(f"Could not locate ffmpeg: {e}")


def _convert_video_to_mp4(
    src_video: Path,
    output_path: Path,
    ffmpeg_exe: str,
    use_codec_copy: bool = True,
) -> bool:
    """
    Convert video file to MP4 (H.264 + AAC) using ffmpeg.
    
    Args:
        src_video: Source video file (any format supported by ffmpeg)
        output_path: Output MP4 file path
        ffmpeg_exe: Path to ffmpeg executable
        use_codec_copy: If True, use stream copy (fast, no re-encoding).
                       If False, re-encode with H.264/AAC (slower, better compatibility).
    
    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        output_path.unlink()
    
    if use_codec_copy:
        # Fast mode: just re-container to MP4 without re-encoding
        # This is much faster (~100x) but may have compatibility issues
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(src_video),
            "-c", "copy",            # Copy codecs as-is
            "-f", "mp4",             # Force MP4 format
            str(output_path),
        ]
    else:
        # High-quality mode: re-encode with H.264 + AAC
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(src_video),
            "-c:v", "libx264",       # H.264 video codec
            "-preset", "fast",       # Encoding speed (fast/medium/slow)
            "-crf", "23",            # Quality (lower = better, 0-51)
            "-c:a", "aac",           # AAC audio codec
            "-q:a", "5",             # Audio quality
            str(output_path),
        ]
    
    try:
        log.info(f"Converting: {src_video.name} -> {output_path.name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        if result.returncode == 0:
            log.info(f"✓ Created: {output_path.name}")
            return True
        else:
            log.error(f"FFmpeg error for {output_path.name}:")
            if result.stderr:
                log.error(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        log.error(f"Timeout processing {output_path.name}")
        return False
    except Exception as e:
        log.error(f"Error converting {src_video.name}: {e}")
        return False


def process_session(
    session_dir: Path,
    ffmpeg_exe: str,
    skip_existing: bool = True,
    use_codec_copy: bool = True,
) -> dict[str, Any]:
    """
    Process all video files in a BIDS session, converting to MP4.
    
    Args:
        session_dir: Path to BIDS session directory
        ffmpeg_exe: Path to ffmpeg executable
        skip_existing: Skip if MP4 already exists
        use_codec_copy: Use fast codec copy mode (stream copy) vs re-encoding
    
    Returns dict with processing summary.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    
    summary = {
        "session_dir": str(session_dir),
        "codec_copy_mode": use_codec_copy,
        "processed_files": [],
        "skipped_files": [],
        "failed_conversions": [],
    }
    
    video_dir = session_dir / "video"
    if not video_dir.exists():
        log.warning(f"No video directory found in {session_dir}")
        return summary
    
    # Find all video files (originally MKV, AVI, MOV, etc - anything except MP4)
    total_size_gb = 0
    for video_file in sorted(video_dir.glob("**/*")):
        if not video_file.is_file():
            continue
        
        ext = video_file.suffix.lower()
        
        # Skip if already MP4
        if ext == ".mp4":
            log.debug(f"Already MP4: {video_file.name}")
            continue
        
        # Process only video/audio files
        if ext not in {".mkv", ".avi", ".mov", ".flv", ".wav", ".webm", ".m4v", ".mts", ".m2ts"}:
            log.debug(f"Skipping unsupported file: {video_file.name}")
            continue
        
        # Generate MP4 output path
        mp4_path = video_file.with_suffix(".mp4")
        
        if mp4_path.exists() and skip_existing:
            log.debug(f"Skipping existing MP4: {mp4_path.name}")
            summary["skipped_files"].append(str(mp4_path))
            continue
        
        # Convert to MP4
        if _convert_video_to_mp4(video_file, mp4_path, ffmpeg_exe, use_codec_copy):
            file_size_gb = video_file.stat().st_size / (1024**3)
            summary["processed_files"].append({
                "source": str(video_file),
                "output": str(mp4_path),
                "size_gb": round(file_size_gb, 2),
            })
            total_size_gb += file_size_gb
            
            # Remove original file
            try:
                video_file.unlink()
                log.info(f"Removed original: {video_file.name}")
            except Exception as e:
                log.warning(f"Could not remove {video_file.name}: {e}")
        else:
            summary["failed_conversions"].append({
                "file": video_file.name,
                "reason": "conversion_failed",
            })
    
    # Write summary
    summary_path = session_dir / "annot" / "mp4_conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({**summary, "total_size_gb": round(total_size_gb, 2)}, f, indent=2)
    
    log.info(f"Summary saved to: {summary_path}")
    return summary



def main():
    p = argparse.ArgumentParser(
        description="Convert BIDS video files to MP4 format for 3D modeling",
        epilog="Requirements: pip install imageio-ffmpeg",
    )
    p.add_argument("--session-dir", required=True, help="Path to BIDS session directory")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip if MP4 already exists (default: True)")
    p.add_argument("--codec-copy", action="store_true", default=True,
                   help="Use fast codec copy mode (default: True). Set to False for H.264/AAC re-encoding.")
    
    args = p.parse_args()
    session_dir = Path(args.session_dir)
    
    try:
        ffmpeg_exe = _get_ffmpeg_exe()
    except RuntimeError as e:
        log.error(f"FFmpeg setup failed: {e}")
        return 1
    
    log.info(f"Processing session: {session_dir}")
    
    try:
        summary = process_session(
            session_dir,
            ffmpeg_exe,
            skip_existing=args.skip_existing,
            use_codec_copy=args.codec_copy,
        )
        
        if summary["processed_files"]:
            log.info(f"✓ Converted {len(summary['processed_files'])} file(s) to MP4")
            for item in summary["processed_files"]:
                log.info(f"  - {Path(item['source']).name} ({item['size_gb']} GB)")
        
        if summary["failed_conversions"]:
            log.warning(f"✗ Failed to convert {len(summary['failed_conversions'])} file(s)")
            for item in summary["failed_conversions"]:
                log.warning(f"  - {item['file']}: {item['reason']}")
        
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
        
        return 0
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
