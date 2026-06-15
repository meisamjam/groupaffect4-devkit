#!/usr/bin/env python3
"""Robust calibration validation that works without FreeMoCap dependency.

This provides a standalone validation report generator for camera calibration
TOML files that includes:
- Camera intrinsics summary
- Focal length validation against known camera specs
- Inter-camera geometry analysis
- Distortion coefficient analysis  
- Quality scoring and recommendations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import toml
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_SPECS = Path(__file__).parent.parent / "configs" / "camera_specs.json"
DEFAULT_CAMERA_CONFIG = Path(__file__).parent.parent / "configs" / "ffmpeg_multicap.json"


def load_flipped_camera_patterns(camera_config_path: Path) -> tuple[str, ...]:
    """Load camera-label patterns flagged as rotate_180 from camera config."""
    if not camera_config_path.exists():
        return tuple()

    import json

    try:
        data = json.loads(camera_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read camera config %s: %s", camera_config_path, exc)
        return tuple()

    patterns: list[str] = []
    for dev in data.get("devices", []):
        if dev.get("audio_only"):
            continue
        if not bool(dev.get("rotate_180", False)):
            continue
        label = str(dev.get("label", "")).strip().lower()
        if label:
            patterns.append(label)

    return tuple(patterns)


def load_toml_calibration(toml_path: Path) -> dict[str, Any]:
    """Load calibration TOML file."""
    with open(toml_path, encoding="utf-8") as f:
        return toml.load(f)


def load_camera_specs(specs_path: Path) -> dict[str, Any]:
    """Load camera specifications JSON."""
    import json
    with open(specs_path, encoding="utf-8") as f:
        return json.load(f)


def match_camera_model(camera_name: str, camera_specs: dict) -> dict[str, Any] | None:
    """Match camera name to specifications using regex patterns.

    Returns a model spec dict, with any per-camera overrides (from
    ``camera_specs["camera_overrides"]``) merged on top of the base model values.
    Override keys whose names start with ``_`` are skipped (comments/notes).
    """
    import re

    name_lower = camera_name.lower()

    # 1. Resolve base model via camera_name_patterns
    base_model: dict[str, Any] | None = None
    patterns = camera_specs.get("camera_name_patterns", {})
    for pattern, model_key in patterns.items():
        if pattern.startswith("_"):
            continue
        try:
            if re.search(pattern, name_lower, re.IGNORECASE):
                model = camera_specs.get("models", {}).get(model_key)
                if model:
                    base_model = model
                    break
        except re.error:
            pass

    if base_model is None:
        # Fallback: simple substring matching on model names
        for model_name, specs in camera_specs.get("models", {}).items():
            norm = model_name.lower().replace("_", "").replace("-", "")
            if norm in name_lower.replace("_", "").replace("-", ""):
                base_model = specs
                break

    if base_model is None:
        return None

    # 2. Check camera_overrides for a per-camera override (substring match)
    overrides_map = camera_specs.get("camera_overrides", {})
    matched_override: dict[str, Any] = {}
    for override_key, override_vals in overrides_map.items():
        if override_key.startswith("_"):
            continue
        if override_key.lower() in name_lower:
            for k, v in override_vals.items():
                if not k.startswith("_"):
                    matched_override[k] = v
            break  # first match wins

    if not matched_override:
        return base_model

    # Merge: base model values + per-camera overrides (override wins on conflict)
    return {**base_model, **matched_override}


def analyze_intrinsics(calib: dict[str, Any]) -> dict[str, Any]:
    """Analyze camera intrinsic parameters."""
    cameras = []
    for cam_key, cam_data in calib.items():
        if not cam_key.startswith("cam_"):
            continue
        
        matrix = np.array(cam_data["matrix"])
        distortions = np.array(cam_data["distortions"])
        size = cam_data["size"]
        name = cam_data.get("name", cam_key)
        
        fx, fy = matrix[0, 0], matrix[1, 1]
        cx, cy = matrix[0, 2], matrix[1, 2]
        
        # Distortion analysis
        k1 = distortions[0] if len(distortions) > 0 else 0.0
        k2 = distortions[1] if len(distortions) > 1 else 0.0
        dist_mag = float(np.linalg.norm(distortions))
        
        # Check monotonicity for barrel distortion
        monotonic = True
        r_crit = None
        if k1 < 0:
            # For barrel distortion, check if field of view exceeds critical radius
            w, h = size
            corners_r = [
                np.sqrt(((0 - cx) / fx) ** 2 + ((0 - cy) / fy) ** 2),
                np.sqrt(((w - cx) / fx) ** 2 + ((0 - cy) / fy) ** 2),
                np.sqrt(((0 - cx) / fx) ** 2 + ((h - cy) / fy) ** 2),
                np.sqrt(((w - cx) / fx) ** 2 + ((h - cy) / fy) ** 2),
            ]
            r_max = max(corners_r)
            r_crit = 1.0 / np.sqrt(-3.0 * k1)
            if r_max > r_crit:
                monotonic = False
        
        cameras.append({
            "key": cam_key,
            "name": name,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "size": size,
            "k1": k1,
            "k2": k2,
            "dist_mag": dist_mag,
            "monotonic": monotonic,
            "r_crit": r_crit,
        })
    
    return {"cameras": cameras}


def analyze_extrinsics(calib: dict[str, Any]) -> dict[str, Any]:
    """Analyze camera extrinsic parameters (positions and geometry)."""
    cameras = []
    for cam_key, cam_data in calib.items():
        if not cam_key.startswith("cam_"):
            continue
        
        translation = np.array(cam_data["translation"])
        rotation = np.array(cam_data["rotation"])
        name = cam_data.get("name", cam_key)
        
        cameras.append({
            "key": cam_key,
            "name": name,
            "translation": translation,
            "rotation": rotation,
        })
    
    # Compute inter-camera distances
    distances = []
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            dist = np.linalg.norm(cameras[i]["translation"] - cameras[j]["translation"])
            distances.append({
                "cam1": cameras[i]["name"],
                "cam2": cameras[j]["name"],
                "distance_mm": float(dist),
            })
    
    return {
        "cameras": cameras,
        "inter_camera_distances": distances,
    }


def validate_against_specs(
    intrinsics: dict[str, Any],
    camera_specs: dict[str, Any],
) -> dict[str, Any]:
    """Validate focal lengths against known camera specifications."""
    validations = []
    warnings = []
    
    for cam in intrinsics["cameras"]:
        model = match_camera_model(cam["name"], camera_specs)
        if model and "expected_fx_1080p" in model:
            fx_spec = float(model["expected_fx_1080p"])
            fx_cal = cam["fx"]
            ratio = fx_cal / fx_spec
            
            # Use per-model nominal range when provided; otherwise fall back to
            # tight ±40%/±15% defaults.
            nominal_range = model.get("expected_fx_1080p_nominal_range")
            if nominal_range and len(nominal_range) == 2:
                bad_low  = float(nominal_range[0]) / fx_spec
                bad_high = float(nominal_range[1]) / fx_spec
                suspect_low  = max(bad_low,  0.85)
                suspect_high = min(bad_high, 1.15)
                # Widen suspect bounds to fill the gap between ok and bad
                suspect_low  = bad_low
                suspect_high = bad_high
            else:
                bad_low, bad_high = 0.7, 1.4
                suspect_low, suspect_high = 0.85, 1.15

            status = "ok"
            if ratio > bad_high or ratio < bad_low:
                status = "bad"
                warnings.append({
                    "camera": cam["name"],
                    "type": "focal_length_mismatch",
                    "severity": "error",
                    "message": f"fx_calibrated={fx_cal:.1f} vs fx_expected={fx_spec:.1f} (ratio={ratio:.2f}). Likely miscalibrated.",
                })
            elif ratio > suspect_high or ratio < suspect_low:
                status = "suspect"
                warnings.append({
                    "camera": cam["name"],
                    "type": "focal_length_suspicious",
                    "severity": "warning",
                    "message": f"fx ratio={ratio:.2f} is outside normal range ({suspect_low:.2f}-{suspect_high:.2f}).",
                })
            
            validations.append({
                "camera": cam["name"],
                "fx_calibrated": fx_cal,
                "fx_expected": fx_spec,
                "ratio": ratio,
                "status": status,
            })
        else:
            validations.append({
                "camera": cam["name"],
                "fx_calibrated": cam["fx"],
                "fx_expected": None,
                "ratio": None,
                "status": "no_spec",
            })
    
    return {
        "validations": validations,
        "warnings": warnings,
    }


def check_distortion_issues(intrinsics: dict[str, Any]) -> list[dict[str, Any]]:
    """Check for distortion-related issues."""
    issues = []
    
    for cam in intrinsics["cameras"]:
        if not cam["monotonic"]:
            issues.append({
                "camera": cam["name"],
                "type": "non_monotonic_distortion",
                "severity": "error",
                "message": (
                    f"k1={cam['k1']:.4f} causes non-monotonic distortion. "
                    f"cv2.undistortPoints() will fail near image edges. "
                    f"r_crit={cam['r_crit']:.2f}. "
                    "Use nonlinear triangulation or re-calibrate."
                ),
            })
    
    return issues


def load_marker_map(marker_map_path: Path) -> dict[str, Any]:
    """Load marker map from YAML file.

    Supported formats:
    - world.marker_map (online_calibration export style)
    - table_markers (tobii_multicam_glasses_tracker style)
    """
    with open(marker_map_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid marker map format: expected a mapping at YAML root")

    world_cfg = data.get("world", {})
    marker_defs = None
    if isinstance(world_cfg, dict) and "marker_map" in world_cfg:
        marker_defs = world_cfg.get("marker_map")
    elif "table_markers" in data:
        marker_defs = data.get("table_markers")

    if marker_defs is None:
        raise ValueError("Invalid marker map format: expected world.marker_map or table_markers")

    marker_map = {}
    for marker_def in marker_defs:
        marker_id = marker_def["id"]
        corners = np.array(marker_def["corners_m"], dtype=np.float64)
        marker_map[marker_id] = corners

    aruco_dict = data.get("aruco_dictionary") or world_cfg.get("aruco_dictionary", "DICT_4X4_50")

    return {
        "marker_map": marker_map,
        "aruco_dictionary": aruco_dict,
    }


def _should_rotate_180(camera_name: str, patterns: tuple[str, ...]) -> bool:
    """Return True if camera should be corrected by 180-degree rotation."""
    name_lower = camera_name.lower()
    return any(pattern in name_lower for pattern in patterns)


def _normalize_aruco_dict_names(
    primary_dict_name: str,
    extra_dict_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Build ordered, de-duplicated list of valid ArUco dictionary names."""
    ordered: list[str] = []
    seen: set[str] = set()

    for raw_name in (primary_dict_name, *extra_dict_names):
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        if not hasattr(cv2.aruco, name):
            logger.warning("Ignoring unknown ArUco dictionary: %s", name)
            continue
        seen.add(name)
        ordered.append(name)

    if not ordered:
        raise ValueError("No valid ArUco dictionary names were provided")

    return tuple(ordered)


def _build_aruco_detectors(dict_names: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Create one ArUco detector per dictionary name."""
    detector_params = cv2.aruco.DetectorParameters()
    detectors: list[tuple[str, Any]] = []
    for name in dict_names:
        aruco_dict_id = getattr(cv2.aruco, name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        detectors.append((name, cv2.aruco.ArucoDetector(aruco_dict, detector_params)))
    return detectors


def _detect_markers_with_multiple_dicts(
    frame: np.ndarray,
    detectors: list[tuple[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Detect markers across dictionaries and merge unique marker ids."""
    merged_by_id: dict[int, np.ndarray] = {}

    for _, detector in detectors:
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None or len(ids) == 0:
            continue
        for idx, marker_id_raw in enumerate(ids.flatten()):
            marker_id = int(marker_id_raw)
            if marker_id not in merged_by_id:
                merged_by_id[marker_id] = corners[idx]

    if not merged_by_id:
        return [], None

    marker_ids = np.array(sorted(merged_by_id.keys()), dtype=np.int32).reshape(-1, 1)
    marker_corners = [merged_by_id[int(marker_id)] for marker_id in marker_ids.flatten()]
    return marker_corners, marker_ids


def validate_with_desk_markers(
    calib: dict[str, Any],
    marker_map_path: Path,
    videos_dir: Path,
    max_frames: int = 100,
    sample_stride: int = 10,
    flipped_camera_patterns: tuple[str, ...] = tuple(),
    aruco_dicts: tuple[str, ...] = tuple(),
) -> dict[str, Any]:
    """Validate calibration by detecting ArUco markers and computing reprojection error.
    
    Args:
        calib: Calibration dict from TOML
        marker_map_path: Path to marker map YAML (table_marker_map.yaml format)
        videos_dir: Directory containing video files for each camera
        max_frames: Maximum frames to analyze per camera
        sample_stride: Frame sampling stride (e.g., 10 = every 10th frame)
    
    Returns:
        Dictionary with validation results:
        - per_camera_results: list of per-camera results
        - mean_reprojection_error_px: overall mean reprojection error
        - max_reprojection_error_px: worst reprojection error
        - total_detections: total marker detections across all cameras
        - success: boolean indicating if validation passed
    """
    logger.info(f"Loading marker map from {marker_map_path}")
    marker_data = load_marker_map(marker_map_path)
    marker_map = marker_data["marker_map"]
    marker_map_dict = marker_data["aruco_dictionary"]

    detection_dicts = _normalize_aruco_dict_names(marker_map_dict, aruco_dicts)
    aruco_detectors = _build_aruco_detectors(detection_dicts)
    logger.info("Marker detection dictionaries: %s", ", ".join(detection_dicts))
    
    per_camera_results = []
    all_errors = []
    total_detections = 0
    
    # Process each camera
    for cam_key, cam_data in calib.items():
        if not cam_key.startswith("cam_"):
            continue
        
        cam_name = cam_data.get("name", cam_key)
        rotate_180 = _should_rotate_180(cam_name, flipped_camera_patterns)
        logger.info(
            "Validating camera: %s%s",
            cam_name,
            " (applying 180deg correction)" if rotate_180 else "",
        )
        
        # Find video file for this camera
        # Try common patterns: exact name, name with extensions
        video_path = None
        for ext in [".mkv", ".mp4", ".avi"]:
            candidate = videos_dir / f"{cam_name}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            logger.warning(f"No video found for {cam_name} in {videos_dir}")
            continue
        
        # Load camera parameters
        camera_matrix = np.array(cam_data["matrix"], dtype=np.float64)
        dist_coeffs = np.array(cam_data["distortions"], dtype=np.float64)
        rotation = np.array(cam_data["rotation"], dtype=np.float64)
        translation = np.array(cam_data["translation"], dtype=np.float64)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_analyze = min(max_frames, total_frames // sample_stride)
        
        reproj_errors = []
        detections_count = 0
        
        for frame_idx in range(0, min(max_frames * sample_stride, total_frames), sample_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Detect markers using one or more dictionaries.
            corners, ids = _detect_markers_with_multiple_dicts(frame, aruco_detectors)
            if ids is None or len(ids) == 0:
                continue
            
            # Process each detected marker
            for i, marker_id_raw in enumerate(ids.flatten()):
                marker_id = int(marker_id_raw)
                if marker_id not in marker_map:
                    continue
                
                # Get 3D marker corners from map
                marker_3d = marker_map[marker_id]
                
                # Get 2D detected corners
                marker_2d = corners[i].reshape(4, 2)
                
                # Project 3D points using calibration
                projected_2d, _ = cv2.projectPoints(
                    marker_3d,
                    rotation,
                    translation,
                    camera_matrix,
                    dist_coeffs,
                )
                projected_2d = projected_2d.reshape(-1, 2)
                
                # Compute reprojection error
                error = np.linalg.norm(marker_2d - projected_2d, axis=1).mean()
                reproj_errors.append(error)
                detections_count += 1
        
        cap.release()
        
        # Compute statistics for this camera
        if reproj_errors:
            mean_error = float(np.mean(reproj_errors))
            max_error = float(np.max(reproj_errors))
            std_error = float(np.std(reproj_errors))
            
            per_camera_results.append({
                "camera": cam_name,
                "video_path": str(video_path),
                "detections": detections_count,
                "frames_analyzed": frames_to_analyze,
                "frame_rotation_correction": "180deg" if rotate_180 else "none",
                "mean_reprojection_error_px": mean_error,
                "max_reprojection_error_px": max_error,
                "std_reprojection_error_px": std_error,
            })
            
            all_errors.extend(reproj_errors)
            total_detections += detections_count
            
            logger.info(
                f"  {cam_name}: {detections_count} detections, "
                f"mean_error={mean_error:.2f}px, max_error={max_error:.2f}px"
            )
        else:
            logger.warning(f"  {cam_name}: no marker detections found")
            per_camera_results.append({
                "camera": cam_name,
                "video_path": str(video_path),
                "detections": 0,
                "frames_analyzed": frames_to_analyze,
                "frame_rotation_correction": "180deg" if rotate_180 else "none",
                "mean_reprojection_error_px": None,
                "max_reprojection_error_px": None,
                "std_reprojection_error_px": None,
            })
    
    # Overall statistics
    if all_errors:
        mean_error = float(np.mean(all_errors))
        max_error = float(np.max(all_errors))
        success = mean_error < 5.0  # Accept if mean error < 5 pixels
    else:
        mean_error = None
        max_error = None
        success = False
    
    return {
        "per_camera_results": per_camera_results,
        "mean_reprojection_error_px": mean_error,
        "max_reprojection_error_px": max_error,
        "total_detections": total_detections,
        "success": success,
        "threshold_px": 5.0,
        "flipped_camera_patterns": list(flipped_camera_patterns),
        "aruco_dictionaries": list(detection_dicts),
    }


def compute_quality_score(
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    spec_validation: dict[str, Any],
    distortion_issues: list[dict[str, Any]],
    marker_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute overall calibration quality score (0-100)."""
    score = 100.0
    reasons = []
    
    # Penalize focal length mismatches
    for val in spec_validation["validations"]:
        if val["status"] == "bad":
            score -= 30
            reasons.append(f"{val['camera']}: severe focal length mismatch")
        elif val["status"] == "suspect":
            score -= 10
            reasons.append(f"{val['camera']}: suspicious focal length")
    
    # Penalize non-monotonic distortion
    for issue in distortion_issues:
        if issue["severity"] == "error":
            score -= 20
            reasons.append(f"{issue['camera']}: non-monotonic distortion")
    
    # Penalize if cameras are too close or too far
    if extrinsics["inter_camera_distances"]:
        for dist_info in extrinsics["inter_camera_distances"]:
            dist_m = dist_info["distance_mm"] / 1000.0
            if dist_m < 0.5:
                score -= 10
                reasons.append(
                    f"{dist_info['cam1']} and {dist_info['cam2']}: "
                    f"too close ({dist_m:.2f}m) for good triangulation"
                )
            elif dist_m > 10.0:
                score -= 5
                reasons.append(
                    f"{dist_info['cam1']} and {dist_info['cam2']}: "
                    f"very far apart ({dist_m:.2f}m), check if intentional"
                )
    
    # Marker validation (if available)
    if marker_validation is not None:
        mean_error = marker_validation.get("mean_reprojection_error_px")
        if mean_error is not None:
            if mean_error > 10.0:
                score -= 30
                reasons.append(f"High marker reprojection error ({mean_error:.1f}px > 10px)")
            elif mean_error > 5.0:
                score -= 15
                reasons.append(f"Elevated marker reprojection error ({mean_error:.1f}px > 5px)")
            elif mean_error > 2.5:
                score -= 5
                reasons.append(f"Moderate marker reprojection error ({mean_error:.1f}px > 2.5px)")
        
        if marker_validation.get("total_detections", 0) == 0:
            score -= 20
            reasons.append("No desk markers detected in videos")
    
    # Ensure score is in [0, 100]
    score = max(0.0, min(100.0, score))
    
    # Quality grade
    if score >= 90:
        grade = "Excellent"
        recommendation = "Accept calibration"
    elif score >= 75:
        grade = "Good"
        recommendation = "Accept calibration"
    elif score >= 60:
        grade = "Fair"
        recommendation = "Accept with caution - consider re-calibrating if critical"
    elif score >= 40:
        grade = "Poor"
        recommendation = "Reject - re-calibration recommended"
    else:
        grade = "Failed"
        recommendation = "Reject - re-calibration required"
    
    return {
        "score": score,
        "grade": grade,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def generate_report_text(
    toml_path: Path,
    intrinsics: dict[str, Any],
    extrinsics: dict[str, Any],
    spec_validation: dict[str, Any],
    distortion_issues: list[dict[str, Any]],
    quality_score: dict[str, Any],
    marker_validation: dict[str, Any] | None = None,
) -> str:
    """Generate human-readable validation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CAMERA CALIBRATION VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Calibration file: {toml_path}")
    lines.append(f"Number of cameras: {len(intrinsics['cameras'])}")
    lines.append("")
    
    # Quality score
    lines.append("=" * 70)
    lines.append("OVERALL QUALITY")
    lines.append("=" * 70)
    lines.append(f"Score: {quality_score['score']:.1f}/100")
    lines.append(f"Grade: {quality_score['grade']}")
    lines.append(f"Recommendation: {quality_score['recommendation']}")
    if quality_score["reasons"]:
        lines.append("\nIssues affecting score:")
        for reason in quality_score["reasons"]:
            lines.append(f"  - {reason}")
    else:
        lines.append("\n✓ No significant issues detected")
    lines.append("")
    
    # Marker validation results (if available)
    if marker_validation is not None:
        lines.append("=" * 70)
        lines.append("DESK MARKER VALIDATION")
        lines.append("=" * 70)
        
        mean_error = marker_validation.get("mean_reprojection_error_px")
        max_error = marker_validation.get("max_reprojection_error_px")
        total_detections = marker_validation.get("total_detections", 0)
        threshold = marker_validation.get("threshold_px", 5.0)
        corrected_patterns = marker_validation.get("flipped_camera_patterns", [])
        aruco_dicts = marker_validation.get("aruco_dictionaries", [])

        if corrected_patterns:
            lines.append(
                "180deg correction patterns: "
                + ", ".join(corrected_patterns)
            )
        if aruco_dicts:
            lines.append("ArUco dictionaries: " + ", ".join(aruco_dicts))
        
        if mean_error is not None:
            lines.append(f"Total marker detections: {total_detections}")
            lines.append(f"Mean reprojection error: {mean_error:.2f} pixels")
            lines.append(f"Max reprojection error: {max_error:.2f} pixels")
            lines.append(f"Acceptance threshold: < {threshold:.1f} pixels")
            
            if marker_validation.get("success"):
                lines.append("Status: ✓ PASSED")
            else:
                lines.append("Status: ✗ FAILED")
            
            lines.append("\nPer-camera results:")
            for cam_result in marker_validation.get("per_camera_results", []):
                cam_name = cam_result["camera"]
                detections = cam_result["detections"]
                cam_mean = cam_result.get("mean_reprojection_error_px")
                rotation_note = " [rot180]" if cam_result.get("frame_rotation_correction") == "180deg" else ""
                if cam_mean is not None:
                    lines.append(
                        f"  {cam_name}{rotation_note}: {detections} detections, "
                        f"error={cam_mean:.2f}px"
                    )
                else:
                    lines.append(f"  {cam_name}{rotation_note}: no detections")
        else:
            lines.append("Status: No marker detections found")
            lines.append("\nEnsure:")
            lines.append("  - Desk markers are visible in videos")
            lines.append("  - Marker map file matches actual marker layout")
            lines.append("  - Video files exist in the specified directory")
        
        lines.append("")
    
    # Intrinsics summary
    lines.append("=" * 70)
    lines.append("INTRINSIC PARAMETERS")
    lines.append("=" * 70)
    lines.append(f"{'Camera':<30} {'fx':>10} {'fy':>10} {'cx':>10} {'cy':>10}")
    lines.append("-" * 70)
    for cam in intrinsics["cameras"]:
        lines.append(
            f"{cam['name']:<30} {cam['fx']:>10.1f} {cam['fy']:>10.1f} "
            f"{cam['cx']:>10.1f} {cam['cy']:>10.1f}"
        )
    lines.append("")
    
    # Focal length validation
    if spec_validation["validations"]:
        lines.append("=" * 70)
        lines.append("FOCAL LENGTH VALIDATION")
        lines.append("=" * 70)
        lines.append(f"{'Camera':<30} {'fx_cal':>10} {'fx_spec':>10} {'Ratio':>8} {'Status':>10}")
        lines.append("-" * 68)
        for val in spec_validation["validations"]:
            fx_spec_str = f"{val['fx_expected']:.1f}" if val["fx_expected"] else "?"
            ratio_str = f"{val['ratio']:.2f}" if val["ratio"] else "?"
            lines.append(
                f"{val['camera']:<30} {val['fx_calibrated']:>10.1f} "
                f"{fx_spec_str:>10} {ratio_str:>8} {val['status']:>10}"
            )
        lines.append("")
    
    # Distortion analysis
    lines.append("=" * 70)
    lines.append("DISTORTION COEFFICIENTS")
    lines.append("=" * 70)
    lines.append(f"{'Camera':<30} {'k1':>10} {'k2':>10} {'|dist|':>10} {'Monotonic':>12}")
    lines.append("-" * 72)
    for cam in intrinsics["cameras"]:
        monotonic_str = "yes" if cam["monotonic"] else f"NO (r_crit={cam['r_crit']:.2f})"
        lines.append(
            f"{cam['name']:<30} {cam['k1']:>10.4f} {cam['k2']:>10.4f} "
            f"{cam['dist_mag']:>10.3f} {monotonic_str:>12}"
        )
    lines.append("")
    
    # Extrinsics summary
    lines.append("=" * 70)
    lines.append("EXTRINSIC PARAMETERS (Camera Positions)")
    lines.append("=" * 70)
    lines.append(f"{'Camera':<30} {'tx (mm)':>12} {'ty (mm)':>12} {'tz (mm)':>12}")
    lines.append("-" * 72)
    for cam in extrinsics["cameras"]:
        tx, ty, tz = cam["translation"]
        lines.append(f"{cam['name']:<30} {tx:>12.1f} {ty:>12.1f} {tz:>12.1f}")
    lines.append("")
    
    # Inter-camera distances
    if extrinsics["inter_camera_distances"]:
        lines.append("=" * 70)
        lines.append("INTER-CAMERA GEOMETRY")
        lines.append("=" * 70)
        lines.append(f"{'Camera 1':<25} {'Camera 2':<25} {'Distance (m)':>15}")
        lines.append("-" * 70)
        for dist_info in extrinsics["inter_camera_distances"]:
            dist_m = dist_info["distance_mm"] / 1000.0
            lines.append(
                f"{dist_info['cam1']:<25} {dist_info['cam2']:<25} {dist_m:>15.2f}"
            )
        lines.append("")
    
    # Warnings and recommendations
    all_warnings = spec_validation["warnings"] + distortion_issues
    if all_warnings:
        lines.append("=" * 70)
        lines.append("WARNINGS AND RECOMMENDATIONS")
        lines.append("=" * 70)
        for i, warning in enumerate(all_warnings, 1):
            severity = warning["severity"].upper()
            lines.append(f"{i}. [{severity}] {warning['camera']}: {warning['type']}")
            lines.append(f"   {warning['message']}")
            lines.append("")
        
        # Add specific recommendations
        lines.append("RECOMMENDATIONS:")
        if any(w["type"] == "focal_length_mismatch" for w in all_warnings):
            lines.append("  • Re-calibrate with --init-focal to seed expected focal length")
            lines.append("    from camera specs (configs/camera_specs.json)")
        if any(w["type"] == "non_monotonic_distortion" for w in all_warnings):
            lines.append("  • Use multicam_pose3d.py which has built-in distortion handling")
            lines.append("  • OR re-calibrate with more frames and better board visibility")
            lines.append("  • Accept that edge regions may have unreliable triangulation")
        lines.append("")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate camera calibration TOML file without FreeMoCap dependency"
    )
    parser.add_argument(
        "--toml",
        type=Path,
        required=True,
        help="Path to calibration .toml file",
    )
    parser.add_argument(
        "--camera-specs",
        type=Path,
        default=DEFAULT_CAMERA_SPECS,
        help="Path to camera specifications JSON (default: configs/camera_specs.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output report file (default: print to stdout)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Also write JSON report to this path",
    )
    parser.add_argument(
        "--marker-map",
        type=Path,
        help="Path to marker map YAML file (e.g., table_marker_map.yaml) for desk marker validation",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        help="Directory containing video files for marker validation (required if --marker-map is used)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to analyze per camera for marker validation (default: 100)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=10,
        help="Frame sampling stride for marker validation (default: 10 = every 10th frame)",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="Camera config JSON with rotate_180 metadata (default: configs/ffmpeg_multicap.json)",
    )
    parser.add_argument(
        "--flipped-camera-patterns",
        type=str,
        default=None,
        help=(
            "Optional comma-separated camera-name patterns to rotate 180deg before marker detection. "
            "If omitted, patterns are loaded from --camera-config devices where rotate_180=true. "
            "Use empty string to disable all rotation correction."
        ),
    )
    parser.add_argument(
        "--aruco-dicts",
        type=str,
        default="",
        help=(
            "Optional comma-separated extra ArUco dictionaries for marker detection "
            "(for example DICT_4X4_50,DICT_4X4_250). Marker-map dictionary is always included."
        ),
    )
    
    args = parser.parse_args()

    if args.flipped_camera_patterns is None:
        flipped_camera_patterns = load_flipped_camera_patterns(args.camera_config)
    else:
        flipped_camera_patterns = tuple(
            token.strip().lower()
            for token in args.flipped_camera_patterns.split(",")
            if token.strip()
        )

    extra_aruco_dicts = tuple(
        token.strip()
        for token in args.aruco_dicts.split(",")
        if token.strip()
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    if not args.toml.exists():
        logger.error(f"Calibration TOML not found: {args.toml}")
        return 1
    
    # Validate marker validation arguments
    if args.marker_map and not args.videos_dir:
        logger.error("--videos-dir is required when --marker-map is specified")
        return 1
    
    if args.marker_map and not args.marker_map.exists():
        logger.error(f"Marker map not found: {args.marker_map}")
        return 1
    
    if args.videos_dir and not args.videos_dir.exists():
        logger.error(f"Videos directory not found: {args.videos_dir}")
        return 1
    
    # Load calibration
    logger.info(f"Loading calibration from {args.toml}")
    calib = load_toml_calibration(args.toml)
    
    # Load camera specs if available
    camera_specs = {}
    if args.camera_specs.exists():
        logger.info(f"Loading camera specs from {args.camera_specs}")
        camera_specs = load_camera_specs(args.camera_specs)
    else:
        logger.warning(f"Camera specs not found at {args.camera_specs}, skipping focal length validation")
    
    # Run analyses
    logger.info("Analyzing intrinsic parameters...")
    intrinsics = analyze_intrinsics(calib)
    
    logger.info("Analyzing extrinsic parameters...")
    extrinsics = analyze_extrinsics(calib)
    
    logger.info("Validating against camera specifications...")
    spec_validation = validate_against_specs(intrinsics, camera_specs)
    
    logger.info("Checking for distortion issues...")
    distortion_issues = check_distortion_issues(intrinsics)
    
    # Optional marker validation
    marker_validation = None
    if args.marker_map and args.videos_dir:
        logger.info("Running desk marker validation...")
        try:
            marker_validation = validate_with_desk_markers(
                calib,
                args.marker_map,
                args.videos_dir,
                max_frames=args.max_frames,
                sample_stride=args.sample_stride,
                flipped_camera_patterns=flipped_camera_patterns,
                aruco_dicts=extra_aruco_dicts,
            )
        except Exception as exc:
            logger.error(f"Marker validation failed: {exc}")
            logger.exception("Full traceback:")
    
    logger.info("Computing quality score...")
    quality_score = compute_quality_score(
        intrinsics, extrinsics, spec_validation, distortion_issues, marker_validation
    )
    
    # Generate report
    report_text = generate_report_text(
        args.toml,
        intrinsics,
        extrinsics,
        spec_validation,
        distortion_issues,
        quality_score,
        marker_validation,
    )
    
    # Output report
    if args.output:
        logger.info(f"Writing report to {args.output}")
        args.output.write_text(report_text, encoding="utf-8")
    else:
        print(report_text)
    
    # Optional JSON output
    if args.json:
        import json
        json_data = {
            "toml_path": str(args.toml),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "spec_validation": spec_validation,
            "distortion_issues": distortion_issues,
            "marker_validation": marker_validation,
            "quality_score": quality_score,
        }
        logger.info(f"Writing JSON report to {args.json}")
        args.json.write_text(
            json.dumps(json_data, indent=2, default=str),
            encoding="utf-8",
        )
    
    # Return exit code based on quality and recommendation
    if quality_score["score"] < 40:
        logger.error(f"Calibration quality is poor (score={quality_score['score']:.1f}) - {quality_score['recommendation']}")
        return 1
    elif quality_score["score"] < 60:
        logger.warning(f"Calibration quality is fair (score={quality_score['score']:.1f}) - {quality_score['recommendation']}")
        return 0
    else:
        logger.info(f"Calibration quality is {quality_score['grade'].lower()} (score={quality_score['score']:.1f}) - {quality_score['recommendation']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
