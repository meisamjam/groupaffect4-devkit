#!/usr/bin/env python3
"""Track Tobii Glasses 6-DoF pose via small ArUco markers + fixed cameras.

This tool uses the lab's 6 fixed cameras (with multicam calibration) to track
small ArUco markers attached to each Tobii Glasses frame.  The tracked pose is
then combined with Tobii gaze data to produce world-aligned gaze coordinates.

Marker setup
------------
- **Table markers** (optional): Large ArUco markers at table corners/centre to
  verify the world frame aligns with the ground-plane calibration.
- **Glasses markers**: Two small ArUco markers per glasses (left + right temple).
  Markers should be rigidly attached at known offsets from the glasses eye centre.

Workflow
--------
1. Run multicam calibration (tools/calibrate_charuco.py) → TOML file.
2. Attach small ArUco markers to each glasses frame.
3. Record session with all 6 cameras + Tobii streams.
4. Run this tool:

   python tools/tobii_multicam_glasses_tracker.py \\
       --calibration data/session/video_camera_calibration.toml \\
       --videos-dir data/session/video \\
       --config configs/tobii_multicam_glasses_tracker.example.yaml \\
       --output-dir data/session/glasses_poses

Outputs
-------
- ``{glasses_id}_pose.ndjson``: Per-frame 6-DoF pose (position + quaternion)
- ``{glasses_id}_gaze_world.ndjson``: Gaze projected to world coordinates
- ``summary.json``: Detection rates, sync stats, QC metrics

Dependencies: numpy, scipy, opencv-python, pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration loading (reused from multicam_pose3d.py)
# ---------------------------------------------------------------------------

def load_calibration_toml(toml_path: Path) -> dict:
    """Load calibration from .toml file."""
    try:
        import tomllib
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml
            with open(toml_path) as f:
                return toml.load(f)
        except ImportError as exc:
            raise ImportError("Install: pip install toml  (or use Python >=3.11)") from exc


class CameraCalibration:
    """Single camera's intrinsic + extrinsic parameters."""

    def __init__(self, cam_key: str, cam_data: dict):
        self.key = cam_key
        self.name: str = cam_data.get("name", cam_key)
        self.size = tuple(cam_data.get("size", [1920, 1080]))

        # Intrinsic
        self.K = np.array(cam_data["matrix"], dtype=np.float64)
        self.dist = np.array(
            cam_data.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64
        )

        # Extrinsic: rotation (Rodrigues vector) + translation
        rvec = np.array(cam_data.get("rotation", [0, 0, 0]), dtype=np.float64)
        tvec = np.array(cam_data.get("translation", [0, 0, 0]), dtype=np.float64)
        R = Rotation.from_rotvec(rvec).as_matrix()
        self.R = R
        self.t = tvec.reshape(3, 1)
        self.Rt = np.hstack([R, self.t])  # 3x4
        self.P = self.K @ self.Rt  # 3x4 projection matrix

        # Camera position in world
        self.world_pos = -R.T @ self.t

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Undistort 2D points using calibration distortion coefficients."""
        if np.allclose(self.dist, 0):
            return pts
        pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts_f32, self.K, self.dist, P=self.K)
        return undist.reshape(-1, 2).astype(np.float64)

    def project(self, pts_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points -> 2D pixel coordinates."""
        N = pts_3d.shape[0]
        h = np.hstack([pts_3d, np.ones((N, 1))])  # (N, 4) homogeneous
        proj = (self.P @ h.T).T  # (N, 3)
        return proj[:, :2] / proj[:, 2:3]


def load_all_cameras(toml_path: Path) -> dict[str, CameraCalibration]:
    """Load calibration TOML -> dict of camera_key -> CameraCalibration."""
    calib_dict = load_calibration_toml(toml_path)
    cameras: dict[str, CameraCalibration] = {}
    for k in sorted(calib_dict.keys()):
        if k.startswith("cam_"):
            cameras[k] = CameraCalibration(k, calib_dict[k])
    return cameras


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GlassesMarkerConfig:
    """Configuration for one glasses frame's markers."""
    glasses_id: str
    left_marker_id: int
    right_marker_id: int
    marker_size_m: float
    # Offset from marker centre to glasses eye centre (in marker local frame)
    # These should be measured once when attaching markers
    left_marker_offset_mm: list[float]   # [x, y, z]
    right_marker_offset_mm: list[float]  # [x, y, z]
    # Tobii data paths
    gaze_ndjson: Path | None = None
    # Per-device time offset (Tobii time - video time)
    time_offset_s: float = 0.0


@dataclass
class TableMarkerConfig:
    """World reference marker on the table."""
    marker_id: int
    corners_m: np.ndarray  # (4, 3) world coordinates of corners


@dataclass
class TrackerConfig:
    """Full configuration for the tracker."""
    aruco_dictionary: str
    glasses: list[GlassesMarkerConfig]
    table_markers: list[TableMarkerConfig]
    min_cameras_for_triangulation: int = 2
    max_reproj_error_px: float = 10.0
    video_fps: float = 30.0


def load_tracker_config(config_path: Path) -> TrackerConfig:
    """Load tracker configuration from YAML file."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    world_cfg = raw.get("world", {}) if isinstance(raw, dict) else {}
    aruco_dict = raw.get("aruco_dictionary") or world_cfg.get("aruco_dictionary") or "DICT_4X4_50"
    video_fps = float(raw.get("video_fps", 30.0))
    min_cams = int(raw.get("min_cameras_for_triangulation", 2))
    max_reproj = float(raw.get("max_reproj_error_px", 10.0))

    # Parse glasses configs
    glasses_list: list[GlassesMarkerConfig] = []
    for g in raw.get("glasses", []):
        gaze_path = g.get("gaze_ndjson")
        glasses_list.append(GlassesMarkerConfig(
            glasses_id=str(g["id"]),
            left_marker_id=int(g["left_marker_id"]),
            right_marker_id=int(g["right_marker_id"]),
            marker_size_m=float(g.get("marker_size_m", 0.02)),
            left_marker_offset_mm=list(g.get("left_marker_offset_mm", [0, 0, 0])),
            right_marker_offset_mm=list(g.get("right_marker_offset_mm", [0, 0, 0])),
            gaze_ndjson=Path(gaze_path) if gaze_path else None,
            time_offset_s=float(g.get("time_offset_s", 0.0)),
        ))

    # Parse table markers. Accept either:
    # - tracker-native: table_markers: [{id, corners_m}, ...]
    # - marker-map style: world.marker_map: [{id, corners_m}, ...]
    marker_defs = raw.get("table_markers")
    if marker_defs is None:
        marker_defs = world_cfg.get("marker_map", [])

    table_markers: list[TableMarkerConfig] = []
    for tm in marker_defs:
        corners = np.array(tm["corners_m"], dtype=np.float64)
        table_markers.append(TableMarkerConfig(
            marker_id=int(tm["id"]),
            corners_m=corners,
        ))

    return TrackerConfig(
        aruco_dictionary=aruco_dict,
        glasses=glasses_list,
        table_markers=table_markers,
        min_cameras_for_triangulation=min_cams,
        max_reproj_error_px=max_reproj,
        video_fps=video_fps,
    )


# ---------------------------------------------------------------------------
# ArUco detection
# ---------------------------------------------------------------------------

def create_aruco_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    """Create ArUco detector with given dictionary."""
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    dictionary_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    # Tune for small markers
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


@dataclass
class MarkerDetection:
    """Single marker detection in one camera view."""
    marker_id: int
    corners_px: np.ndarray  # (4, 2) pixel coordinates, ordered TL,TR,BR,BL
    camera_key: str


def detect_markers_in_frame(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_key: str,
) -> list[MarkerDetection]:
    """Detect all ArUco markers in a single camera frame."""
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return []

    detections = []
    for i, marker_id in enumerate(ids.flatten()):
        detections.append(MarkerDetection(
            marker_id=int(marker_id),
            corners_px=corners[i].reshape(4, 2),
            camera_key=camera_key,
        ))
    return detections


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulate_point_dlt(
    observations: list[tuple[CameraCalibration, np.ndarray]],
) -> tuple[np.ndarray, float] | None:
    """DLT triangulation of a single 3D point from multiple 2D views.

    Args:
        observations: List of (camera, point_2d) tuples.

    Returns:
        (point_3d, mean_reproj_error) or None if insufficient observations.
    """
    if len(observations) < 2:
        return None

    # Build DLT matrix
    A = []
    for cam, pt in observations:
        x, y = pt
        P = cam.P
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)

    # SVD solve
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]  # Dehomogenize

    # Compute reprojection error
    errors = []
    for cam, pt in observations:
        proj = cam.project(X.reshape(1, 3))[0]
        errors.append(np.linalg.norm(proj - pt))
    mean_err = float(np.mean(errors))

    return X, mean_err


def triangulate_marker_corners(
    detections: list[MarkerDetection],
    cameras: dict[str, CameraCalibration],
    min_cameras: int = 2,
    max_reproj: float = 10.0,
) -> tuple[np.ndarray | None, float]:
    """Triangulate all 4 corners of a marker from multiple camera views.

    Returns:
        (corners_3d [4,3], mean_reproj_error) or (None, inf) if failed.
    """
    if len(detections) < min_cameras:
        return None, float("inf")

    corners_3d = []
    total_error = 0.0

    for corner_idx in range(4):
        obs = []
        for det in detections:
            cam = cameras.get(det.camera_key)
            if cam is None:
                continue
            # Undistort the corner
            corner_px = det.corners_px[corner_idx:corner_idx + 1]
            corner_undist = cam.undistort_points(corner_px)[0]
            obs.append((cam, corner_undist))

        result = triangulate_point_dlt(obs)
        if result is None:
            return None, float("inf")

        pt_3d, err = result
        if err > max_reproj:
            return None, err

        corners_3d.append(pt_3d)
        total_error += err

    mean_error = total_error / 4.0
    return np.array(corners_3d), mean_error


# ---------------------------------------------------------------------------
# Glasses pose estimation
# ---------------------------------------------------------------------------

@dataclass
class GlassesPose:
    """6-DoF pose of a glasses frame."""
    frame_idx: int
    frame_time_s: float
    glasses_id: str
    position: np.ndarray  # (3,) world position of glasses centre
    quaternion: np.ndarray  # (4,) [x, y, z, w] orientation
    reproj_error: float
    cameras_used: int
    markers_detected: int  # 0, 1, or 2


def estimate_glasses_pose_from_markers(
    left_corners_3d: np.ndarray | None,  # (4, 3)
    right_corners_3d: np.ndarray | None,  # (4, 3)
    left_offset_mm: list[float],
    right_offset_mm: list[float],
    marker_size_m: float,
    frame_idx: int,
    frame_time_s: float,
    glasses_id: str,
    reproj_error: float,
    cameras_used: int,
) -> GlassesPose | None:
    """Estimate glasses 6-DoF pose from triangulated marker corners.

    Each marker provides:
    - Centre: mean of 4 corners
    - Local X axis: right edge direction
    - Local Y axis: up edge direction
    - Local Z axis: normal (facing out from marker)

    With both markers, we compute a robust pose. With one marker, we use
    the single marker's pose directly (less robust for roll).
    """
    def marker_pose(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract position and rotation matrix from marker corners."""
        centre = corners.mean(axis=0)
        # Marker corners are ordered: TL, TR, BR, BL
        # X axis: TL -> TR
        x_axis = corners[1] - corners[0]
        x_axis /= np.linalg.norm(x_axis)
        # Y axis: TL -> BL (marker Y points down in image, but up in 3D)
        y_temp = corners[0] - corners[3]
        y_temp /= np.linalg.norm(y_temp)
        # Z axis: cross product
        z_axis = np.cross(x_axis, y_temp)
        z_axis /= np.linalg.norm(z_axis)
        # Re-orthogonalize Y
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        return centre, R

    markers_detected = sum([left_corners_3d is not None, right_corners_3d is not None])
    if markers_detected == 0:
        return None

    if left_corners_3d is not None and right_corners_3d is not None:
        # Both markers detected - best case
        left_centre, left_R = marker_pose(left_corners_3d)
        right_centre, right_R = marker_pose(right_corners_3d)

        # Apply offsets (convert mm to m)
        left_offset = np.array(left_offset_mm) / 1000.0
        right_offset = np.array(right_offset_mm) / 1000.0
        left_eye = left_centre + left_R @ left_offset
        right_eye = right_centre + right_R @ right_offset

        # Glasses centre is midpoint between eye positions
        glasses_centre = (left_eye + right_eye) / 2.0

        # Glasses orientation: X axis from right to left eye, Y up, Z forward
        x_axis = left_eye - right_eye
        x_axis /= np.linalg.norm(x_axis)
        # Use average of marker Z axes for forward direction
        z_avg = (left_R[:, 2] + right_R[:, 2]) / 2.0
        z_avg /= np.linalg.norm(z_avg)
        # Y axis: cross of Z and X
        y_axis = np.cross(z_avg, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        # Re-orthogonalize Z
        z_axis = np.cross(x_axis, y_axis)
        glasses_R = np.column_stack([x_axis, y_axis, z_axis])

    else:
        # Only one marker detected
        if left_corners_3d is not None:
            centre, R = marker_pose(left_corners_3d)
            offset = np.array(left_offset_mm) / 1000.0
        else:
            centre, R = marker_pose(right_corners_3d)
            offset = np.array(right_offset_mm) / 1000.0

        glasses_centre = centre + R @ offset
        glasses_R = R

    # Convert rotation matrix to quaternion
    rot = Rotation.from_matrix(glasses_R)
    quat = rot.as_quat()  # [x, y, z, w]

    return GlassesPose(
        frame_idx=frame_idx,
        frame_time_s=frame_time_s,
        glasses_id=glasses_id,
        position=glasses_centre,
        quaternion=quat,
        reproj_error=reproj_error,
        cameras_used=cameras_used,
        markers_detected=markers_detected,
    )


# ---------------------------------------------------------------------------
# Gaze transformation
# ---------------------------------------------------------------------------

@dataclass
class GazeSample:
    """Single gaze sample from Tobii."""
    sample_time_s: float
    gaze3d: np.ndarray | None  # Gaze point in Tobii coordinate system
    left_origin: np.ndarray | None
    right_origin: np.ndarray | None
    left_direction: np.ndarray | None
    right_direction: np.ndarray | None


def load_gaze_samples(gaze_path: Path) -> list[GazeSample]:
    """Load gaze samples from Tobii NDJSON file."""
    samples = []
    with gaze_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Extract timestamp
            timestamp = data.get("timestamp_s")
            if timestamp is None:
                ticks = data.get("timestamp_ticks", 0)
                tps = data.get("ticks_per_second", 1000000)
                timestamp = float(ticks) / float(tps)

            # Extract gaze vectors
            gaze3d = data.get("gaze3d")
            left_eye = data.get("left_eye", {})
            right_eye = data.get("right_eye", {})

            samples.append(GazeSample(
                sample_time_s=float(timestamp),
                gaze3d=np.array(gaze3d) if gaze3d else None,
                left_origin=np.array(left_eye.get("gaze_origin")) if left_eye.get("gaze_origin") else None,
                right_origin=np.array(right_eye.get("gaze_origin")) if right_eye.get("gaze_origin") else None,
                left_direction=np.array(left_eye.get("gaze_direction")) if left_eye.get("gaze_direction") else None,
                right_direction=np.array(right_eye.get("gaze_direction")) if right_eye.get("gaze_direction") else None,
            ))

    return samples


@dataclass
class WorldGaze:
    """Gaze sample transformed to world coordinates."""
    frame_idx: int
    frame_time_s: float
    glasses_id: str
    world_x: float
    world_y: float
    world_z: float
    confidence: float  # Based on reproj error and markers detected


def transform_gaze_to_world(
    gaze: GazeSample,
    pose: GlassesPose,
    plane_z: float = 0.0,
) -> WorldGaze:
    """Transform a Tobii gaze sample to world coordinates using glasses pose.

    The Tobii gaze is in the glasses-local coordinate system (camera frame).
    We transform it using the tracked pose from the fixed cameras.
    """
    # Get world transform from pose
    R = Rotation.from_quat(pose.quaternion).as_matrix()
    t = pose.position

    # Determine gaze ray
    if gaze.left_origin is not None and gaze.left_direction is not None:
        origin_local = gaze.left_origin.copy()
        direction_local = gaze.left_direction.copy()
    elif gaze.right_origin is not None and gaze.right_direction is not None:
        origin_local = gaze.right_origin.copy()
        direction_local = gaze.right_direction.copy()
    elif gaze.gaze3d is not None:
        # Use gaze3d as the target point
        origin_local = np.array([0.0, 0.0, 0.0])
        direction_local = gaze.gaze3d / np.linalg.norm(gaze.gaze3d)
    else:
        # No valid gaze data
        return WorldGaze(
            frame_idx=pose.frame_idx,
            frame_time_s=pose.frame_time_s,
            glasses_id=pose.glasses_id,
            world_x=float("nan"),
            world_y=float("nan"),
            world_z=float("nan"),
            confidence=0.0,
        )

    # Transform to world
    origin_world = (R @ origin_local) + t
    direction_world = R @ direction_local
    direction_world /= np.linalg.norm(direction_world)

    # Intersect with z=plane_z plane
    if abs(direction_world[2]) < 1e-9:
        # Nearly parallel to plane
        world_point = origin_world + direction_world * 1.0  # 1m ahead
    else:
        t_intersect = (plane_z - origin_world[2]) / direction_world[2]
        if t_intersect < 0:
            # Looking away from plane - project 1m ahead
            t_intersect = 1.0
        world_point = origin_world + direction_world * t_intersect

    # Confidence based on pose quality
    conf_markers = pose.markers_detected / 2.0  # 0.5 for 1 marker, 1.0 for 2
    conf_reproj = max(0.0, 1.0 - pose.reproj_error / 20.0)
    confidence = conf_markers * conf_reproj

    return WorldGaze(
        frame_idx=pose.frame_idx,
        frame_time_s=pose.frame_time_s,
        glasses_id=pose.glasses_id,
        world_x=float(world_point[0]),
        world_y=float(world_point[1]),
        world_z=float(world_point[2]),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def discover_video_files(videos_dir: Path) -> list[tuple[str, Path]]:
    """Discover video files and extract camera labels."""
    video_exts = {".mp4", ".mkv", ".avi", ".mov"}
    exclude = {"sync_grid", "combined", "mosaic"}

    videos = []
    for p in sorted(videos_dir.iterdir()):
        if p.suffix.lower() in video_exts and p.stem.lower() not in exclude:
            # Extract camera label from filename
            label = p.stem.replace("_video", "")
            videos.append((label, p))
    return videos


def map_videos_to_cameras(
    video_files: list[tuple[str, Path]],
    cameras: dict[str, CameraCalibration],
) -> dict[str, Path]:
    """Map video files to calibration cameras by name matching."""

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    mapping = {}
    for cam_key, cam in cameras.items():
        cam_name_norm = _norm(cam.name)
        for label, video_path in video_files:
            label_norm = _norm(label)
            if (
                cam.name in label
                or label in cam.name
                or cam_name_norm in label_norm
                or label_norm in cam_name_norm
            ):
                mapping[cam_key] = video_path
                break
    return mapping


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_session(
    cameras: dict[str, CameraCalibration],
    video_map: dict[str, Path],
    config: TrackerConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process entire session: detect markers, track glasses, transform gaze."""

    detector = create_aruco_detector(config.aruco_dictionary)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build marker ID -> glasses mapping
    marker_to_glasses: dict[int, tuple[GlassesMarkerConfig, str]] = {}
    for g in config.glasses:
        marker_to_glasses[g.left_marker_id] = (g, "left")
        marker_to_glasses[g.right_marker_id] = (g, "right")

    # Open all video captures
    caps: dict[str, cv2.VideoCapture] = {}
    for cam_key, video_path in video_map.items():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            caps[cam_key] = cap
            logger.info(f"Opened camera {cam_key}: {video_path}")
        else:
            logger.warning(f"Failed to open {video_path}")

    if not caps:
        raise RuntimeError("No videos could be opened")

    # Get frame count from first video
    first_cap = list(caps.values())[0]
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = config.video_fps

    # Initialize output files
    pose_writers: dict[str, Any] = {}
    gaze_writers: dict[str, Any] = {}
    for g in config.glasses:
        pose_path = output_dir / f"{g.glasses_id}_pose.ndjson"
        gaze_path = output_dir / f"{g.glasses_id}_gaze_world.ndjson"
        pose_writers[g.glasses_id] = pose_path.open("w", encoding="utf-8")
        gaze_writers[g.glasses_id] = gaze_path.open("w", encoding="utf-8")

    # Load gaze samples for each glasses
    gaze_samples: dict[str, list[GazeSample]] = {}
    for g in config.glasses:
        if g.gaze_ndjson and g.gaze_ndjson.exists():
            gaze_samples[g.glasses_id] = load_gaze_samples(g.gaze_ndjson)
            logger.info(f"Loaded {len(gaze_samples[g.glasses_id])} gaze samples for {g.glasses_id}")
        else:
            gaze_samples[g.glasses_id] = []

    # Statistics
    stats: dict[str, Any] = {
        "total_frames": total_frames,
        "cameras_used": list(caps.keys()),
        "glasses": {},
    }
    for g in config.glasses:
        stats["glasses"][g.glasses_id] = {
            "poses_detected": 0,
            "two_marker_frames": 0,
            "one_marker_frames": 0,
            "gaze_samples_processed": 0,
        }

    # Process frame by frame
    frame_idx = 0
    while True:
        # Read frames from all cameras
        frames: dict[str, np.ndarray] = {}
        for cam_key, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_key] = frame

        if not frames:
            break

        frame_time_s = frame_idx / fps

        # Detect markers in all camera views
        all_detections: list[MarkerDetection] = []
        for cam_key, frame in frames.items():
            dets = detect_markers_in_frame(frame, detector, cam_key)
            all_detections.extend(dets)

        # Group detections by marker ID
        by_marker_id: dict[int, list[MarkerDetection]] = {}
        for det in all_detections:
            by_marker_id.setdefault(det.marker_id, []).append(det)

        # Process each glasses
        for g in config.glasses:
            # Triangulate left marker
            left_dets = by_marker_id.get(g.left_marker_id, [])
            left_corners, left_err = triangulate_marker_corners(
                left_dets, cameras,
                config.min_cameras_for_triangulation,
                config.max_reproj_error_px,
            )

            # Triangulate right marker
            right_dets = by_marker_id.get(g.right_marker_id, [])
            right_corners, right_err = triangulate_marker_corners(
                right_dets, cameras,
                config.min_cameras_for_triangulation,
                config.max_reproj_error_px,
            )

            # Compute average error
            errors = [e for e in [left_err, right_err] if e < float("inf")]
            avg_err = sum(errors) / len(errors) if errors else float("inf")
            n_cams = max(len(left_dets), len(right_dets))

            # Estimate glasses pose
            pose = estimate_glasses_pose_from_markers(
                left_corners, right_corners,
                g.left_marker_offset_mm, g.right_marker_offset_mm,
                g.marker_size_m,
                frame_idx, frame_time_s, g.glasses_id,
                avg_err, n_cams,
            )

            if pose is not None:
                # Write pose
                pose_record = {
                    "frame_idx": pose.frame_idx,
                    "frame_time_s": pose.frame_time_s,
                    "glasses_id": pose.glasses_id,
                    "position": pose.position.tolist(),
                    "quaternion": pose.quaternion.tolist(),
                    "reproj_error_px": pose.reproj_error,
                    "cameras_used": pose.cameras_used,
                    "markers_detected": pose.markers_detected,
                }
                pose_writers[g.glasses_id].write(
                    json.dumps(pose_record, default=float) + "\n"
                )

                # Update stats
                stats["glasses"][g.glasses_id]["poses_detected"] += 1
                if pose.markers_detected == 2:
                    stats["glasses"][g.glasses_id]["two_marker_frames"] += 1
                else:
                    stats["glasses"][g.glasses_id]["one_marker_frames"] += 1

                # Find and transform nearest gaze samples
                g_samples = gaze_samples.get(g.glasses_id, [])
                adjusted_time = frame_time_s + g.time_offset_s

                # Find samples within +/- 0.5 frame duration
                half_frame = 0.5 / fps
                for gaze in g_samples:
                    if abs(gaze.sample_time_s - adjusted_time) < half_frame:
                        world_gaze = transform_gaze_to_world(gaze, pose)
                        gaze_record = {
                            "frame_idx": world_gaze.frame_idx,
                            "frame_time_s": world_gaze.frame_time_s,
                            "glasses_id": world_gaze.glasses_id,
                            "world_x": world_gaze.world_x,
                            "world_y": world_gaze.world_y,
                            "world_z": world_gaze.world_z,
                            "confidence": world_gaze.confidence,
                        }
                        gaze_writers[g.glasses_id].write(
                            json.dumps(gaze_record, default=float) + "\n"
                        )
                        stats["glasses"][g.glasses_id]["gaze_samples_processed"] += 1

        frame_idx += 1
        if frame_idx % 300 == 0:
            logger.info(f"Processed frame {frame_idx}/{total_frames}")

    # Cleanup
    for cap in caps.values():
        cap.release()
    for f in pose_writers.values():
        f.close()
    for f in gaze_writers.values():
        f.close()

    # Write summary
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Processing complete. Summary: {summary_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track Tobii Glasses 6-DoF pose via ArUco markers + fixed cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibration", "-c", required=True, type=Path,
        help="Multi-camera calibration TOML file",
    )
    parser.add_argument(
        "--videos-dir", "-v", required=True, type=Path,
        help="Directory containing video files from fixed cameras",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Tracker configuration YAML file",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path,
        help="Output directory for pose and gaze files",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate inputs
    if not args.calibration.exists():
        logger.error(f"Calibration file not found: {args.calibration}")
        return 1
    if not args.videos_dir.is_dir():
        logger.error(f"Videos directory not found: {args.videos_dir}")
        return 1
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    # Load calibration and config
    logger.info(f"Loading calibration from {args.calibration}")
    cameras = load_all_cameras(args.calibration)
    logger.info(f"Loaded {len(cameras)} cameras")

    logger.info(f"Loading tracker config from {args.config}")
    config = load_tracker_config(args.config)
    logger.info(f"Tracking {len(config.glasses)} glasses devices")

    # Discover and map videos
    video_files = discover_video_files(args.videos_dir)
    logger.info(f"Found {len(video_files)} video files")

    video_map = map_videos_to_cameras(video_files, cameras)
    logger.info(f"Mapped {len(video_map)} videos to cameras")

    if not video_map:
        logger.error("No videos could be mapped to calibration cameras")
        return 1

    # Process session
    try:
        stats = process_session(cameras, video_map, config, args.output_dir)
        print("\n=== Summary ===")
        print(f"Total frames processed: {stats['total_frames']}")
        for gid, gstats in stats["glasses"].items():
            print(f"\n{gid}:")
            print(f"  Poses detected: {gstats['poses_detected']}")
            print(f"  Two-marker frames: {gstats['two_marker_frames']}")
            print(f"  One-marker frames: {gstats['one_marker_frames']}")
            print(f"  Gaze samples processed: {gstats['gaze_samples_processed']}")
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
