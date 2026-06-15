#!/usr/bin/env python3
"""
3D skeleton reconstruction using FreeMoCap and calibration file.

This script takes multi-camera videos and a spatial calibration .toml file,
runs pose detection (MediaPipe), then triangulates 3D skeleton positions.

Usage
-----
  python tools/reconstruct_3d.py \\
      --videos-dir data/sub-meisam/ses-20260202_test/video \\
      --calibration data/sub-meisam/ses-20260202_test/video/video_camera_calibration.toml \\
      --output data/sub-meisam/ses-20260202_test/output_data/skeleton_3d

Dependencies
------------
Requires: freemocap>=1.3.0, mediapipe, numpy, pandas, toml
  conda activate affectai-freemocap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# CRITICAL: Do NOT call logging.basicConfig() at module level.
# FreeMoCap's configure_logging() has circular import issues.


def _setup_logging(level: int = logging.INFO) -> None:
    """Setup logging after importing FreeMoCap to avoid circular imports."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _discover_video_files(videos_dir: Path) -> dict[str, list[Path]]:
    """
    Discover video files grouped by camera.
    
    FreeMoCap expects .mp4 files. Returns dict like:
      {
        'Camera_0': [path1.mp4, path2.mp4, ...],
        'Camera_1': [path1.mp4, ...],
      }
    """
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    # Discover .mp4 files (FreeMoCap requirement)
    videos = sorted(videos_dir.glob('*.mp4'))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")
    
    # Group by camera name (assumes naming like "cam1_video.mp4", "cam1_video.mkv", etc.)
    camera_groups = {}
    for video_path in videos:
        # Extract camera identifier (e.g., "cam1" from "jabra_panacast_20_cam1_vid_video.mp4")
        stem = video_path.stem
        if 'cam' in stem.lower():
            # Find the camera part: look for patterns like "cam1", "camera_0", etc.
            parts = stem.lower().split('_')
            cam_id = None
            for i, part in enumerate(parts):
                if 'cam' in part:
                    cam_id = part
                    break
            if cam_id is None:
                cam_id = f"camera_{len(camera_groups)}"
        else:
            cam_id = f"camera_{len(camera_groups)}"
        
        if cam_id not in camera_groups:
            camera_groups[cam_id] = []
        camera_groups[cam_id].append(video_path)
    
    if not camera_groups:
        raise ValueError(f"Could not identify cameras in videos. Found: {[v.name for v in videos]}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Discovered {len(videos)} video file(s) across {len(camera_groups)} camera(s):")
    for cam_id, paths in sorted(camera_groups.items()):
        logger.info(f"  {cam_id}: {', '.join(p.name for p in paths)}")
    
    return camera_groups


def cmd_reconstruct_3d(args) -> None:
    """
    Main 3D reconstruction command.
    
    Steps:
    1. Load calibration file (.toml with camera intrinsics/extrinsics)
    2. Discover video files
    3. Run pose detection (MediaPipe) on all videos
    4. Triangulate 3D skeleton positions using calibration
    5. Save output (numpy, JSON, or CSV)
    """
    _setup_logging()
    logger = logging.getLogger(__name__)
    
    videos_dir = Path(args.videos_dir)
    calibration_file = Path(args.calibration)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("3D SKELETON RECONSTRUCTION")
    logger.info("=" * 70)
    logger.info(f"Videos directory: {videos_dir}")
    logger.info(f"Calibration file: {calibration_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate calibration file exists
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    # Discover videos
    camera_groups = _discover_video_files(videos_dir)
    
    # Convert .mkv to .mp4 if needed
    _charuco_mp4_dir = videos_dir / '_charuco_mp4'
    fps_target = 30
    
    needs_conversion = False
    for paths in camera_groups.values():
        for path in paths:
            if path.suffix.lower() != '.mp4':
                needs_conversion = True
                break
    
    if needs_conversion:
        import subprocess
        _charuco_mp4_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nConverting non-MP4 videos to MP4 in {_charuco_mp4_dir}")
        for cam_id, paths in camera_groups.items():
            for path in paths:
                if path.suffix.lower() != '.mp4':
                    output_path = _charuco_mp4_dir / path.with_suffix('.mp4').name
                    if not output_path.exists():
                        cmd = [
                            'ffmpeg', '-y', '-i', str(path),
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                            '-c:a', 'aac', str(output_path)
                        ]
                        logger.info(f"  {path.name} -> {output_path.name}")
                        subprocess.run(cmd, check=True, capture_output=True)
        
        # Update camera_groups to use converted videos
        camera_groups = _discover_video_files(_charuco_mp4_dir)
    
    try:
        # ===== IMPORT FREEMOCAP (now that logging is configured) =====
        from freemocap import FreeMoCapObject
        from freemocap.system.marker_detector.video_marker_detection import MarkerDetectionResults
        
        logger.info("\n" + "=" * 70)
        logger.info("LOADING CALIBRATION")
        logger.info("=" * 70)
        
        # Load calibration
        import tomllib
        with open(calibration_file, 'rb') as f:
            calibration_dict = tomllib.load(f)
        
        logger.info(f"Loaded calibration with {len(calibration_dict.get('camera_names', []))} camera(s)")
        camera_names = calibration_dict.get('camera_names', list(camera_groups.keys()))
        
        # ===== INITIALIZE FREEMOCAP OBJECT =====
        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING FREEMOCAP")
        logger.info("=" * 70)
        
        videos_list = []
        for cam_id in sorted(camera_groups.keys()):
            videos_list.extend(camera_groups[cam_id])
        
        logger.info(f"Processing {len(videos_list)} video file(s)")
        
        # Create FreeMoCapObject with video paths
        freemocap_object = FreeMoCapObject(
            session_id=videos_dir.parent.name,
            output_data_folder_path=str(output_dir),
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("POSE DETECTION (MEDIAPIPE)")
        logger.info("=" * 70)
        
        # Run pose detection
        # FreeMoCap will automatically detect poses using MediaPipe
        marker_presence_list_of_lists = freemocap_object.detect_poses_in_videos(
            video_path_list=[str(v) for v in videos_list],
        )
        
        logger.info(f"✓ Pose detection complete")
        logger.info(f"  Detected markers shape: {[m.shape for m in marker_presence_list_of_lists]}")
        
        logger.info("\n" + "=" * 70)
        logger.info("3D TRIANGULATION")
        logger.info("=" * 70)
        
        # Triangulate 3D positions using calibration
        marker_positions_3d = freemocap_object.triangulate_3d_data(
            marker_detection_results=marker_presence_list_of_lists,
            camera_calibration_object=freemocap_object.calibration_object,
        )
        
        logger.info(f"✓ Triangulation complete")
        logger.info(f"  3D marker positions shape: {marker_positions_3d.shape if hasattr(marker_positions_3d, 'shape') else type(marker_positions_3d)}")
        
        logger.info("\n" + "=" * 70)
        logger.info("SAVING OUTPUT")
        logger.info("=" * 70)
        
        # Save as numpy file
        output_npy = output_dir / 'skeleton_3d.npy'
        if isinstance(marker_positions_3d, np.ndarray):
            np.save(output_npy, marker_positions_3d)
            logger.info(f"✓ Saved 3D skeleton to: {output_npy}")
            logger.info(f"  Shape: {marker_positions_3d.shape} (frames, markers, xyz)")
        
        # Save metadata
        metadata = {
            'calibration_file': str(calibration_file),
            'videos_processed': [str(v) for v in videos_list],
            'output_shape': str(marker_positions_3d.shape) if hasattr(marker_positions_3d, 'shape') else 'unknown',
            'timestamp': str(Path.cwd()),
        }
        metadata_file = output_dir / 'reconstruction_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata to: {metadata_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("RECONSTRUCTION SUCCESSFUL")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"\n❌ RECONSTRUCTION FAILED")
        logger.error(f"Error: {e}")
        logger.error("\nTroubleshooting checklist:")
        logger.error("  1. Ensure calibration file is valid: validate with `calibrate_charuco.py validate --toml ...`")
        logger.error("  2. Check that all videos are present and readable")
        logger.error("  3. Verify FreeMoCap environment: `pip list | grep freemocap`")
        logger.error("  4. If pose detection fails, check MediaPipe is installed: `conda install -c conda-forge mediapipe`")
        logger.error("  5. Review FreeMoCap logs in output directory")
        raise


def cmd_validate_reconstruction(args) -> None:
    """Validate and inspect 3D reconstruction output."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    
    output_file = Path(args.output_file)
    
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")
    
    logger.info(f"Loading reconstruction: {output_file}")
    
    if output_file.suffix == '.npy':
        data = np.load(output_file)
        logger.info(f"\nShape: {data.shape}")
        logger.info(f"Data type: {data.dtype}")
        logger.info(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
        logger.info(f"NaN count: {np.isnan(data).sum()}")
        
        if len(data.shape) == 3:
            n_frames, n_markers, n_dims = data.shape
            logger.info(f"\nFrames: {n_frames}, Markers: {n_markers}, Dims: {n_dims}")
            logger.info(f"Assuming format: (time, marker_index, [x,y,z])")
        
        # Show a sample frame
        logger.info(f"\nSample (frame 0, first 3 markers):")
        logger.info(f"{data[0, :3, :]}")


def main():
    parser = argparse.ArgumentParser(
        description='3D skeleton reconstruction using FreeMoCap and calibration'
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-command')
    
    # reconstruct sub-command
    p_recon = subparsers.add_parser(
        'reconstruct',
        help='Run 3D reconstruction on videos using calibration'
    )
    p_recon.add_argument(
        '--videos-dir', required=True, type=Path,
        help='Directory containing .mp4/.mkv videos'
    )
    p_recon.add_argument(
        '--calibration', required=True, type=Path,
        help='Path to calibration .toml file'
    )
    p_recon.add_argument(
        '--output', required=True, type=Path,
        help='Output directory for 3D skeleton data'
    )
    p_recon.set_defaults(func=cmd_reconstruct_3d)
    
    # validate sub-command
    p_val = subparsers.add_parser(
        'validate',
        help='Inspect 3D reconstruction output'
    )
    p_val.add_argument(
        '--output-file', required=True, type=Path,
        help='Path to reconstruction output file (.npy)'
    )
    p_val.set_defaults(func=cmd_validate_reconstruction)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        sys.exit(1)


if __name__ == '__main__':
    main()
