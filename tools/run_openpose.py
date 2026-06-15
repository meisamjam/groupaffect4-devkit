#!/usr/bin/env python3
"""
Wrapper script to run local OpenPose installation with standardized CLI.

Usage:
    python run_openpose.py --video <path_to_mp4> --output <json_output_dir>

Example:
    python run_openpose.py \
        --video "data/sub-meisam/ses-20260202_test/video/_charuco_mp4/jabra_panacast_20_cam1_vid_video.mp4" \
        --output "openpose_output/output_cam1_json"
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_openpose_exe():
    """Auto-detect OpenPose installation in tools/ directory."""
    openpose_dir = Path(__file__).parent / "openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended"
    exe_path = openpose_dir / "openpose" / "bin" / "OpenPoseDemo.exe"
    
    if not exe_path.exists():
        raise FileNotFoundError(
            f"OpenPose executable not found at:\n  {exe_path}\n"
            f"Expected installation at: {openpose_dir}"
        )
    return str(exe_path.resolve())


def run_openpose(video_path, output_dir, max_people=10):
    """
    Run OpenPose on a video file.
    
    Args:
        video_path: Path to input MP4 video
        output_dir: Directory to save JSON pose output
        max_people: Maximum number of people to detect per frame
    
    Returns:
        Exit code from OpenPose
    """
    # Validate input
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find OpenPose executable
    openpose_exe = find_openpose_exe()
    
    # Build command
    cmd = [
        openpose_exe,
        "--video", video_path,
        "--write_json", output_dir,
        "--number_people_max", str(max_people),
        "--display", "0",
        "--render_pose", "0",  # Disable pose rendering (required without GUI)
    ]
    
    print("=" * 70)
    print("RUNNING OPENPOSE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output JSON: {output_dir}")
    print(f"Max people: {max_people}")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    # Run OpenPose from its root directory (where models/ folder is located)
    # Convert to absolute paths before changing directory
    video_abs = os.path.abspath(video_path)
    output_abs = os.path.abspath(output_dir)
    
    # Update command with absolute paths
    cmd = [
        openpose_exe,
        "--video", video_abs,
        "--write_json", output_abs,
        "--number_people_max", str(max_people),
        "--display", "0",
        "--render_pose", "0",  # Disable pose rendering (required without GUI)
    ]
    
    try:
        # Change to OpenPose root directory (where models/ folder is located)
        # exe is at: .../openpose/bin/OpenPoseDemo.exe
        # root is: .../openpose/
        openpose_root = Path(openpose_exe).parent.parent  # From bin/ up to openpose/
        original_cwd = os.getcwd()
        os.chdir(openpose_root)
        result = subprocess.run(cmd, check=False)
        os.chdir(original_cwd)
        return result.returncode
    except Exception as e:
        original_cwd = os.getcwd()
        os.chdir(original_cwd)
        print(f"ERROR: Failed to execute OpenPose: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run local OpenPose installation on a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single camera
    python run_openpose.py \\
        --video "data/sub-meisam/ses-20260202_test/video/_charuco_mp4/jabra_panacast_20_cam1_vid_video.mp4" \\
        --output "openpose_output/output_cam1_json"
    
    # List all available videos
    Get-ChildItem "data/sub-meisam/ses-20260202_test/video/_charuco_mp4/*.mp4"
        """
    )
    
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input MP4 video file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save JSON pose output"
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=10,
        help="Maximum number of people to detect per frame (default: 10)"
    )
    
    args = parser.parse_args()
    
    exit_code = run_openpose(args.video, args.output, args.max_people)
    
    if exit_code == 0:
        print()
        print("=" * 70)
        print("SUCCESS: OpenPose completed without errors")
        print("=" * 70)
        print(f"Pose JSON saved to: {args.output}")
        
        # Count output files
        json_dir = Path(args.output)
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            print(f"Generated {len(json_files)} JSON files ({len(json_files)} frames)")
    else:
        print()
        print("=" * 70)
        print(f"ERROR: OpenPose exited with code {exit_code}")
        print("=" * 70)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
