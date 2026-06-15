#!/usr/bin/env python3
"""
Multi-person 3D skeleton triangulation from OpenPose JSON outputs.

This tool takes 2D poses detected by OpenPose (multiple people, multiple cameras)
and triangulates them to 3D using your camera spatial calibration file.

Workflow
--------
1. Run OpenPose on all camera videos:
     openpose.exe --video_path cam1.mp4 --write_json output_cam0/ --write_video output_cam0_video.avi
     openpose.exe --video_path cam3.mp4 --write_json output_cam1/ --write_video output_cam1_video.avi

2. Triangulate poses to 3D:
     python tools/triangulate_openpose.py \\
         --calibration data/.../video_camera_calibration.toml \\
         --pose-dirs output_cam0/ output_cam1/ \\
         --output 3d_skeleton.npy

3. Analyze results:
     python tools/triangulate_openpose.py validate --file 3d_skeleton.npy

Dependencies
------------
- OpenPose: Install from https://github.com/CMU-Perceptual-Computing-Lab/openpose
- numpy, scipy, toml (for triangulation)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Do NOT call logging.basicConfig() at module level
def _setup_logging(level: int = logging.INFO) -> None:
    """Setup logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_calibration_toml(toml_path: Path) -> dict:
    """Load calibration from .toml file."""
    try:
        import tomllib
        with open(toml_path, 'rb') as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml
            with open(toml_path, 'r') as f:
                return toml.load(f)
        except ImportError:
            raise ImportError("Install: pip install toml")


def extract_camera_matrices(calib_dict: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Extract intrinsic and extrinsic matrices from calibration.
    
    Returns dict mapping camera_id → (K, [R|t])
      K: 3×3 intrinsic matrix
      [R|t]: 3×4 projection matrix (rotation-translation)
    """
    matrices = {}
    
    camera_keys = [k for k in calib_dict.keys() if k.startswith('cam_')]
    for cam_key in sorted(camera_keys):
        cam_data = calib_dict[cam_key]
        
        # Intrinsic matrix
        K = np.array(cam_data['matrix'], dtype=np.float64)

        # Distortion coefficients (needed for undistortPoints)
        dist = np.array(
            cam_data.get('distortions', [0, 0, 0, 0, 0]), dtype=np.float64
        )
        
        # Extrinsic: rotation (Rodrigues vector) + translation
        rvec = np.array(cam_data.get('rotation', [0, 0, 0]), dtype=np.float64)
        tvec = np.array(cam_data.get('translation', [0, 0, 0]), dtype=np.float64)
        
        # Convert rotation vector to rotation matrix
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(rvec).as_matrix()  # 3×3
        Rt = np.hstack([R, tvec.reshape(3, 1)])  # 3×4 [R|t]
        
        matrices[cam_key] = (K, Rt, dist)
    
    return matrices


def load_openpose_json(json_path: Path) -> dict[str, Any]:
    """
    Load OpenPose JSON output.
    
    Structure:
    {
      "people": [
        {
          "person_id": [id],
          "pose_keypoints_2d": [x0, y0, conf0, x1, y1, conf1, ...]
        },
        ...
      ],
      "version": 1.8
    }
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_people_poses_2d(json_data: dict) -> dict[int, np.ndarray]:
    """
    Extract 2D poses from OpenPose JSON.
    
    Returns:
      person_id → (n_keypoints, 3) array of [x, y, confidence]
    """
    people = {}
    
    for person in json_data.get('people', []):
        person_id = person.get('person_id', [None])[0]
        if person_id is None:
            person_id = len(people)  # Fallback ID
        
        keypoints = np.array(person.get('pose_keypoints_2d', []), dtype=np.float64)
        if len(keypoints) > 0:
            # Reshape to (n_keypoints, 3) where each row is [x, y, confidence]
            keypoints = keypoints.reshape(-1, 3)
            people[person_id] = keypoints
    
    return people


def _distortion_is_monotonic_op(
    K: np.ndarray, dist: np.ndarray, pt2d: np.ndarray,
) -> bool:
    """Check if radial distortion is monotonic at *pt2d*.

    For k1 < 0 the mapping r_d = r_u*(1 + k1*r_u^2) has a critical radius
    r_crit = 1/sqrt(-3*k1).  Beyond that, ``cv2.undistortPoints`` silently
    diverges.  Returns True when it is safe to use undistortPoints.
    """
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    if k1 >= 0:
        return True
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xn = (float(pt2d[0]) - cx) / fx
    yn = (float(pt2d[1]) - cy) / fy
    r2 = xn**2 + yn**2
    return (1.0 + 3.0 * k1 * r2) > 0.0


def triangulate_point_multi_camera(
    point_2d_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    min_confidence: float = 0.1
) -> tuple[np.ndarray, float]:
    """
    Triangulate a single 3D point from multiple 2D observations.
    
    Args:
        point_2d_list: List of (K, Rt, dist, [x, y, conf]) for each camera.
            ``dist`` is the distortion coefficient vector (5 elements).
        min_confidence: Minimum confidence to include observation
    
    Returns:
        (point_3d, mean_reprojection_error)
    """
    valid_cameras = []
    
    for K, Rt, dist, pt_2d in point_2d_list:
        x, y, conf = pt_2d
        if conf > min_confidence:
            valid_cameras.append((K, Rt, dist, np.array([x, y])))
    
    if len(valid_cameras) < 2:
        # Not enough cameras see this joint
        return np.array([np.nan, np.nan, np.nan]), np.nan

    # Filter cameras with non-monotonic distortion at observation radius
    mono = [
        (K, Rt, dist, pt)
        for K, Rt, dist, pt in valid_cameras
        if _distortion_is_monotonic_op(K, dist, pt)
    ]
    if len(mono) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan

    # Undistort 2D points + build DLT system
    A = []
    for K, Rt, dist, pt_2d in mono:
        # Undistort using calibration distortion coefficients
        if np.allclose(dist, 0):
            ud = pt_2d
        else:
            ud = cv2.undistortPoints(
                pt_2d.reshape(1, 1, 2).astype(np.float32),
                K, dist, P=K,
            ).reshape(2)
        P = K @ Rt  # 3×4 projection matrix
        x, y = ud
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # Solve using SVD: find null space of A
    _, _, Vh = np.linalg.svd(A)
    X_homogeneous = Vh[-1, :]  # Last row = null space
    
    # Convert from homogeneous to 3D
    point_3d = X_homogeneous[:3] / X_homogeneous[3]
    
    # Compute reprojection error (against undistorted coords)
    reprojection_errors = []
    for K, Rt, dist, pt_2d in mono:
        if np.allclose(dist, 0):
            ud = pt_2d
        else:
            ud = cv2.undistortPoints(
                pt_2d.reshape(1, 1, 2).astype(np.float32),
                K, dist, P=K,
            ).reshape(2)
        P = K @ Rt
        pt_proj_h = P @ np.append(point_3d, 1)
        pt_proj = pt_proj_h[:2] / pt_proj_h[2]
        error = np.linalg.norm(pt_proj - ud)
        reprojection_errors.append(error)
    
    mean_error = np.mean(reprojection_errors) if reprojection_errors else np.nan
    
    return point_3d, mean_error


def triangulate_frame(
    frame_idx: int,
    pose_frames: list[dict[int, np.ndarray]],
    camera_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    n_keypoints: int = 25
) -> dict[int, np.ndarray]:
    """
    Triangulate all people in a single frame.
    
    Args:
        frame_idx: Frame number
        pose_frames: List of {person_id → 2D pose} for each camera
        camera_matrices: {camera_id → (K, Rt)}
        n_keypoints: Number of expected keypoints (25 for COCO, 17 for OpenPose-BODY17)
    
    Returns:
        {person_id → (n_keypoints, 3) array of 3D joint positions}
    """
    camera_ids = sorted(camera_matrices.keys())
    
    # Collect all person IDs in this frame
    all_person_ids = set()
    for frame_poses in pose_frames:
        all_person_ids.update(frame_poses.keys())
    
    result_3d = {}
    
    for person_id in all_person_ids:
        pose_3d = np.zeros((n_keypoints, 4))  # (n, 4) for [x, y, z, conf]
        
        for joint_idx in range(n_keypoints):
            # Collect 2D observations of this joint across cameras
            point_2d_list = []
            
            for cam_idx, cam_id in enumerate(camera_ids):
                if cam_idx < len(pose_frames):
                    frame_poses = pose_frames[cam_idx]
                    if person_id in frame_poses:
                        pose_2d = frame_poses[person_id]
                        if joint_idx < len(pose_2d):
                            K, Rt, dist = camera_matrices[cam_id]
                            pt_2d = pose_2d[joint_idx]  # [x, y, conf]
                            point_2d_list.append((K, Rt, dist, pt_2d))
            
            if point_2d_list:
                point_3d, reprojection_error = triangulate_point_multi_camera(
                    point_2d_list, min_confidence=0.1
                )
                pose_3d[joint_idx, :3] = point_3d
                # Use average confidence from 2D observations
                confidences = [pt_2d[2] for K, Rt, dist, pt_2d in point_2d_list]
                pose_3d[joint_idx, 3] = np.mean(confidences) if confidences else 0.0
        
        result_3d[person_id] = pose_3d
    
    return result_3d


def cmd_triangulate(args) -> None:
    """Triangulate 2D poses from multiple OpenPose outputs to 3D."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    
    calibration_file = Path(args.calibration)
    pose_dirs = [Path(d) for d in args.pose_dirs]
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MULTI-PERSON 3D TRIANGULATION")
    logger.info("=" * 70)
    logger.info(f"Calibration: {calibration_file}")
    logger.info(f"Pose directories: {pose_dirs}")
    logger.info(f"Output: {output_file}")
    
    # Load calibration
    try:
        calib_dict = load_calibration_toml(calibration_file)
        logger.info("✓ Loaded calibration")
    except Exception as e:
        logger.error(f"✗ Failed to load calibration: {e}")
        sys.exit(1)
    
    # Extract camera matrices
    try:
        camera_matrices = extract_camera_matrices(calib_dict)
        logger.info(f"✓ Extracted {len(camera_matrices)} camera(s)")
        for cam_id in sorted(camera_matrices.keys()):
            logger.info(f"  {cam_id}")
    except Exception as e:
        logger.error(f"✗ Failed to extract camera matrices: {e}")
        sys.exit(1)
    
    # Load OpenPose JSON outputs
    logger.info("\n" + "=" * 70)
    logger.info("LOADING OPENPOSE OUTPUT")
    logger.info("=" * 70)
    
    pose_sequences = []
    for pose_dir in pose_dirs:
        if not pose_dir.exists():
            logger.warning(f"Pose directory not found: {pose_dir}")
            continue
        
        json_files = sorted(pose_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} frames in {pose_dir.name}")
        
        # Load all frames for this camera
        frames = []
        for json_file in json_files:
            try:
                json_data = load_openpose_json(json_file)
                people_poses = get_people_poses_2d(json_data)
                frames.append(people_poses)
            except Exception as e:
                logger.warning(f"  ✗ {json_file.name}: {e}")
        
        pose_sequences.append(frames)
    
    if not pose_sequences or not pose_sequences[0]:
        logger.error("✗ No OpenPose data loaded")
        sys.exit(1)
    
    n_frames = min(len(seq) for seq in pose_sequences if seq)
    logger.info(f"✓ Processing {n_frames} frames")
    
    # Triangulate all frames
    logger.info("\n" + "=" * 70)
    logger.info("TRIANGULATION")
    logger.info("=" * 70)
    
    triangulated_data = []
    person_ids_set = set()
    
    for frame_idx in range(n_frames):
        if frame_idx % max(1, n_frames // 10) == 0:
            logger.info(f"  Frame {frame_idx}/{n_frames}")
        
        # Get poses for this frame from all cameras
        frame_poses = []
        for seq_idx, seq in enumerate(pose_sequences):
            if seq_idx < len(pose_sequences) and frame_idx < len(seq):
                frame_poses.append(seq[frame_idx])
            else:
                frame_poses.append({})
        
        # Triangulate
        result_3d = triangulate_frame(
            frame_idx, frame_poses, camera_matrices, n_keypoints=25
        )
        triangulated_data.append(result_3d)
        person_ids_set.update(result_3d.keys())
    
    logger.info(f"✓ Triangulation complete")
    logger.info(f"  Found {len(person_ids_set)} people")
    logger.info(f"  Frames: {n_frames}")
    
    # Save output
    logger.info("\n" + "=" * 70)
    logger.info("SAVING OUTPUT")
    logger.info("=" * 70)
    
    # Convert to numpy array: (n_frames, n_people, n_keypoints, 4)
    # Where 4 = [x, y, z, confidence]
    person_ids = sorted(person_ids_set)
    n_people = len(person_ids)
    n_keypoints = 25
    
    output_array = np.zeros((n_frames, n_people, n_keypoints, 4))
    
    for frame_idx, frame_data in enumerate(triangulated_data):
        for person_idx, person_id in enumerate(person_ids):
            if person_id in frame_data:
                output_array[frame_idx, person_idx, :, :] = frame_data[person_id]
    
    np.save(output_file, output_array)
    logger.info(f"✓ Saved 3D skeleton to: {output_file}")
    logger.info(f"  Shape: {output_array.shape} (frames, people, keypoints, [x,y,z,conf])")
    logger.info(f"  Person IDs: {person_ids}")
    
    # Save metadata
    metadata = {
        'n_frames': int(n_frames),
        'n_people': int(n_people),
        'person_ids': sorted(person_ids),
        'n_keypoints': int(n_keypoints),
        'calibration_file': str(calibration_file),
        'pose_directories': [str(d) for d in pose_dirs],
    }
    
    import json
    metadata_file = output_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata to: {metadata_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ TRIANGULATION COMPLETE")
    logger.info("=" * 70)


def cmd_validate(args) -> None:
    """Validate triangulated 3D skeleton output."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    
    output_file = Path(args.file)
    
    if not output_file.exists():
        logger.error(f"File not found: {output_file}")
        sys.exit(1)
    
    data = np.load(output_file)
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    
    if len(data.shape) == 4:
        n_frames, n_people, n_keypoints, n_dims = data.shape
        logger.info(f"\nFrames: {n_frames}")
        logger.info(f"People: {n_people}")
        logger.info(f"Keypoints per person: {n_keypoints}")
        logger.info(f"Dims per keypoint: {n_dims} ([x, y, z, confidence])")
        
        # Check for valid data
        valid_frames = 0
        for frame_idx in range(n_frames):
            frame = data[frame_idx]
            # Count keypoints with valid (non-NaN) positions
            valid_keypoints = np.sum(~np.isnan(frame[:, :, :3]))
            if valid_keypoints > 0:
                valid_frames += 1
        
        logger.info(f"\nValid frames: {valid_frames}/{n_frames}")
        
        # Show sample
        logger.info(f"\nSample (frame 0, person 0, first 5 keypoints):")
        logger.info(f"{data[0, 0, :5, :]}")


def main():
    parser = argparse.ArgumentParser(
        description='Triangulate OpenPose multi-person 2D poses to 3D using camera calibration'
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-command')
    
    # triangulate sub-command
    p_tri = subparsers.add_parser(
        'triangulate',
        help='Triangulate 2D poses to 3D'
    )
    p_tri.add_argument(
        '--calibration', required=True, type=Path,
        help='Path to camera calibration .toml file'
    )
    p_tri.add_argument(
        '--pose-dirs', required=True, nargs='+', type=Path,
        help='OpenPose JSON output directories (one per camera)'
    )
    p_tri.add_argument(
        '--output', required=True, type=Path,
        help='Output .npy file for 3D skeleton data'
    )
    p_tri.set_defaults(func=cmd_triangulate)
    
    # validate sub-command
    p_val = subparsers.add_parser(
        'validate',
        help='Inspect triangulated output'
    )
    p_val.add_argument(
        '--file', required=True, type=Path,
        help='Path to .npy output file'
    )
    p_val.set_defaults(func=cmd_validate)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
