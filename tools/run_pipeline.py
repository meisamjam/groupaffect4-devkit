#!/usr/bin/env python3
"""
Convenience launcher for Master BIDS Pipeline with preset configurations.

Provides easy-to-use commands for common processing scenarios.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "quick": {
        "description": "Quick BIDS-only processing (no 3D pose/face-hand)",
        "args": [
            "--max-workers", "4",
            "--gpu-devices", "0",
        ]
    },
    "standard": {
        "description": "Standard processing with 3D pose (recommended)",
        "args": [
            "--max-workers", "4",
            "--gpu-devices", "0",
            "--enable-3d-pose",
        ]
    },
    "full": {
        "description": "Full processing with 3D pose and face/hand landmarks",
        "args": [
            "--max-workers", "4",
            "--gpu-devices", "0",
            "--enable-3d-pose",
            "--enable-face-hand",
        ]
    },
    "dual_gpu": {
        "description": "Full processing with dual GPU acceleration",
        "args": [
            "--max-workers", "8",
            "--gpu-devices", "0", "1",
            "--enable-3d-pose",
            "--enable-face-hand",
        ]
    },
    "single_session": {
        "description": "Process single session with full features",
        "args": [
            "--max-workers", "1",
            "--gpu-devices", "0",
            "--enable-3d-pose",
            "--enable-face-hand",
        ]
    }
}


def print_presets():
    """Print available presets."""
    print("\nAvailable Presets:")
    print("-" * 70)
    for name, config in PRESETS.items():
        print(f"\n{name.upper()}")
        print(f"  {config['description']}")
        print(f"  Args: {' '.join(config['args'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Master BIDS Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/run_pipeline.py --data-dir data --output-dir E:\\processed_data --preset standard
  python tools/run_pipeline.py --data-dir data --output-dir E:\\processed_data --preset full --verbose
  python tools/run_pipeline.py --data-dir data --output-dir E:\\processed_data --max-workers 8 --enable-3d-pose
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use preset configuration (overridden by explicit args)",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Number of parallel workers",
    )
    
    parser.add_argument(
        "--gpu-devices",
        type=int,
        nargs="+",
        help="GPU device IDs",
    )
    
    parser.add_argument(
        "--enable-3d-pose",
        action="store_true",
        help="Enable 3D pose reconstruction",
    )
    
    parser.add_argument(
        "--enable-face-hand",
        action="store_true",
        help="Enable face/hand landmark detection",
    )
    
    parser.add_argument(
        "--enable-physio",
        action="store_true",
        help="Enable physiological processing",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show available presets and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_presets:
        print_presets()
        return 0
    
    # Build command
    cmd = [
        sys.executable,
        "tools/master_bids_pipeline.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
    ]
    
    # Apply preset if specified
    if args.preset:
        print(f"Using preset: {args.preset}")
        cmd.extend(PRESETS[args.preset]["args"])
    
    # Override with explicit arguments
    if args.max_workers:
        # Remove existing max-workers from cmd if present
        if "--max-workers" in cmd:
            idx = cmd.index("--max-workers")
            del cmd[idx:idx+2]
        cmd.extend(["--max-workers", str(args.max_workers)])
    
    if args.gpu_devices:
        # Remove existing gpu-devices from cmd if present
        if "--gpu-devices" in cmd:
            idx = cmd.index("--gpu-devices")
            # Find how many values follow
            count = 1
            while idx + count + 1 < len(cmd) and not cmd[idx + count + 1].startswith("--"):
                count += 1
            del cmd[idx:idx+count+1]
        cmd.extend(["--gpu-devices"] + [str(d) for d in args.gpu_devices])
    
    if args.enable_3d_pose and "--enable-3d-pose" not in cmd:
        cmd.append("--enable-3d-pose")
    
    if args.enable_face_hand and "--enable-face-hand" not in cmd:
        cmd.append("--enable-face-hand")
    
    if args.enable_physio and "--enable-physio" not in cmd:
        cmd.append("--enable-physio")
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"\nLaunching pipeline with command:")
    print(f"  {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
