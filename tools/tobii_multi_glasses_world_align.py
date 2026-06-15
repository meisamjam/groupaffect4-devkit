#!/usr/bin/env python3
"""Offline world alignment for multiple Tobii Glasses recordings.

Given per-glasses scene video + gaze NDJSON and a marker-map definition,
this tool estimates camera pose per frame (via ArUco), then projects gaze to a
shared world frame (board plane z=0 by default).

Typical workflow
----------------
1) Record with each Tobii scene camera seeing at least one marker at all times.
2) Export or collect gaze NDJSON and scene videos per glasses device.
3) Prepare config YAML (see configs/tobii_offline_world_align.example.yaml).
4) Run:
   python tools/tobii_multi_glasses_world_align.py \
       --config configs/tobii_offline_world_align.example.yaml \
       --output-dir data/derived/tobii_world
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float]

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)


@dataclass(slots=True)
class DeviceConfig:
    device_id: str
    scene_video: Path
    gaze_ndjson: Path
    intrinsics: CameraIntrinsics
    video_fps_override: float | None
    video_time_offset_s: float


@dataclass(slots=True)
class PoseRecord:
    frame_idx: int
    frame_time_s: float
    has_pose: bool
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    marker_count: int
    reproj_error_px: float | None


@dataclass(slots=True)
class GazeSample:
    sample_time_s: float
    gaze_x: float
    gaze_y: float
    gaze3d: list[float] | None
    left_origin: list[float] | None
    right_origin: list[float] | None
    left_direction: list[float] | None
    right_direction: list[float] | None
    raw: dict[str, Any]


def _load_aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    dictionary_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _parse_intrinsics(raw: dict[str, Any]) -> CameraIntrinsics:
    required = ["width", "height", "fx", "fy", "cx", "cy"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing intrinsics fields: {missing}")

    dist_coeffs = raw.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
    if not isinstance(dist_coeffs, list) or len(dist_coeffs) < 4:
        raise ValueError("intrinsics.dist_coeffs must be a list of at least 4 values")

    return CameraIntrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        dist_coeffs=[float(value) for value in dist_coeffs],
    )


def _load_config(config_path: Path) -> tuple[list[DeviceConfig], dict[int, np.ndarray], dict[str, Any]]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    world = raw_config.get("world", {})
    devices_raw = raw_config.get("devices", [])

    if not isinstance(devices_raw, list) or not devices_raw:
        raise ValueError("Config must include non-empty 'devices' list")

    marker_map_raw = world.get("marker_map", [])
    if not isinstance(marker_map_raw, list) or not marker_map_raw:
        raise ValueError("Config must include world.marker_map")

    marker_map: dict[int, np.ndarray] = {}
    for marker_entry in marker_map_raw:
        marker_id = int(marker_entry["id"])
        corners = marker_entry["corners_m"]
        if not isinstance(corners, list) or len(corners) != 4:
            raise ValueError(f"Marker {marker_id}: corners_m must have 4 corners")
        corner_array = np.array(corners, dtype=np.float64)
        if corner_array.shape != (4, 3):
            raise ValueError(f"Marker {marker_id}: corners_m must be shape [4][3]")
        marker_map[marker_id] = corner_array

    devices: list[DeviceConfig] = []
    for raw_device in devices_raw:
        intrinsics = _parse_intrinsics(raw_device["intrinsics"])
        device = DeviceConfig(
            device_id=str(raw_device["id"]),
            scene_video=Path(raw_device["scene_video"]),
            gaze_ndjson=Path(raw_device["gaze_ndjson"]),
            intrinsics=intrinsics,
            video_fps_override=(
                float(raw_device["video_fps_override"])
                if raw_device.get("video_fps_override") is not None
                else None
            ),
            video_time_offset_s=float(raw_device.get("video_time_offset_s", 0.0)),
        )
        devices.append(device)

    return devices, marker_map, world


def _estimate_pose_for_frame(
    frame_bgr: np.ndarray,
    aruco_detector: cv2.aruco.ArucoDetector,
    marker_map: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int, float | None]:
    corners, marker_ids, _ = aruco_detector.detectMarkers(frame_bgr)
    if marker_ids is None or len(marker_ids) == 0:
        return False, None, None, 0, None

    image_points_list: list[np.ndarray] = []
    object_points_list: list[np.ndarray] = []

    for marker_index, marker_id_raw in enumerate(marker_ids.flatten().tolist()):
        marker_id = int(marker_id_raw)
        if marker_id not in marker_map:
            continue
        world_corners = marker_map[marker_id]
        image_corners = np.asarray(corners[marker_index], dtype=np.float64).reshape(4, 2)
        object_points_list.append(world_corners)
        image_points_list.append(image_corners)

    if not object_points_list:
        return False, None, None, 0, None

    object_points = np.vstack(object_points_list)
    image_points = np.vstack(image_points_list)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return False, None, None, len(object_points_list), None

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vec,
        translation_vec,
        camera_matrix,
        distortion,
    )
    projected_2d = projected.reshape(-1, 2)
    reproj_error = float(np.sqrt(np.mean(np.sum((projected_2d - image_points) ** 2, axis=1))))

    return True, rotation_vec, translation_vec, len(object_points_list), reproj_error


def _collect_frame_poses(
    device: DeviceConfig,
    marker_map: dict[int, np.ndarray],
    aruco_dictionary_name: str,
) -> list[PoseRecord]:
    capture = cv2.VideoCapture(str(device.scene_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open scene video: {device.scene_video}")

    fps = device.video_fps_override
    if fps is None:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps or fps <= 0:
            raise RuntimeError(
                f"Invalid FPS from video metadata for {device.scene_video}. "
                "Set video_fps_override in config."
            )

    aruco_dict = _load_aruco_dictionary(aruco_dictionary_name)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    pose_records: list[PoseRecord] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_time_s = (frame_index / fps) + device.video_time_offset_s
        has_pose, rotation_vec, translation_vec, marker_count, reproj_error = _estimate_pose_for_frame(
            frame,
            aruco_detector,
            marker_map,
            device.intrinsics.matrix,
            device.intrinsics.distortion,
        )

        pose_records.append(
            PoseRecord(
                frame_idx=frame_index,
                frame_time_s=frame_time_s,
                has_pose=has_pose,
                rvec=rotation_vec,
                tvec=translation_vec,
                marker_count=marker_count,
                reproj_error_px=reproj_error,
            )
        )
        frame_index += 1

    capture.release()
    return pose_records


def _extract_sample_time_seconds(packet: dict[str, Any], ticks_per_second: float) -> float | None:
    timestamp_ticks = packet.get("timestamp_ticks")
    if isinstance(timestamp_ticks, (int, float)):
        return float(timestamp_ticks) / ticks_per_second

    received_at = packet.get("received_at_utc")
    if isinstance(received_at, str):
        # fallback: no absolute parse needed, relative time is computed later
        return math.nan

    return None


def _parse_vec3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        vec = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in vec):
        return None
    return vec


def _load_gaze_samples(gaze_path: Path, ticks_per_second: float) -> list[GazeSample]:
    samples: list[GazeSample] = []
    with gaze_path.open("r", encoding="utf-8") as file_pointer:
        for line in file_pointer:
            line = line.strip()
            if not line:
                continue
            envelope = json.loads(line)
            packet = envelope.get("packet", {})
            gaze = packet.get("gaze2d")
            if not isinstance(gaze, list) or len(gaze) < 2:
                continue
            try:
                gaze_x = float(gaze[0])
                gaze_y = float(gaze[1])
            except (TypeError, ValueError):
                continue

            sample_time_s = _extract_sample_time_seconds(packet, ticks_per_second)
            if sample_time_s is None:
                continue

            left_eye = packet.get("left_eye") if isinstance(packet.get("left_eye"), dict) else None
            right_eye = packet.get("right_eye") if isinstance(packet.get("right_eye"), dict) else None

            left_origin = _parse_vec3(left_eye.get("gaze_origin")) if left_eye else None
            right_origin = _parse_vec3(right_eye.get("gaze_origin")) if right_eye else None
            left_direction = _parse_vec3(left_eye.get("gaze_direction")) if left_eye else None
            right_direction = _parse_vec3(right_eye.get("gaze_direction")) if right_eye else None
            gaze3d = _parse_vec3(packet.get("gaze3d"))

            samples.append(
                GazeSample(
                    sample_time_s=sample_time_s,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    gaze3d=gaze3d,
                    left_origin=left_origin,
                    right_origin=right_origin,
                    left_direction=left_direction,
                    right_direction=right_direction,
                    raw=packet,
                )
            )

    if not samples:
        return samples

    first_finite = next((sample.sample_time_s for sample in samples if math.isfinite(sample.sample_time_s)), None)
    if first_finite is not None:
        for sample in samples:
            if math.isfinite(sample.sample_time_s):
                sample.sample_time_s -= first_finite

    return samples


def _is_normalized_gaze(gaze_x: float, gaze_y: float) -> bool:
    return -0.2 <= gaze_x <= 1.2 and -0.2 <= gaze_y <= 1.2


def _gaze_to_pixel(gaze_x: float, gaze_y: float, intrinsics: CameraIntrinsics) -> tuple[float, float]:
    if _is_normalized_gaze(gaze_x, gaze_y):
        return gaze_x * intrinsics.width, gaze_y * intrinsics.height
    return gaze_x, gaze_y


def _build_time_index(pose_records: list[PoseRecord]) -> np.ndarray:
    if not pose_records:
        return np.array([], dtype=np.float64)
    return np.array([record.frame_time_s for record in pose_records], dtype=np.float64)


def _nearest_pose_record(pose_records: list[PoseRecord], time_index: np.ndarray, sample_time_s: float) -> PoseRecord | None:
    if len(pose_records) == 0:
        return None
    index = int(np.searchsorted(time_index, sample_time_s, side="left"))
    if index <= 0:
        return pose_records[0]
    if index >= len(pose_records):
        return pose_records[-1]

    prev_record = pose_records[index - 1]
    next_record = pose_records[index]
    prev_delta = abs(prev_record.frame_time_s - sample_time_s)
    next_delta = abs(next_record.frame_time_s - sample_time_s)
    return prev_record if prev_delta <= next_delta else next_record


def _ray_from_pixel(
    pixel_x: float,
    pixel_y: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    undistorted = cv2.undistortPoints(
        np.array([[[pixel_x, pixel_y]]], dtype=np.float64),
        camera_matrix,
        distortion,
    )
    x_norm = float(undistorted[0, 0, 0])
    y_norm = float(undistorted[0, 0, 1])
    ray = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    norm = np.linalg.norm(ray)
    if norm <= 1e-12:
        return ray
    return ray / norm


def _intersect_ray_with_plane_z0(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    if abs(direction[2]) <= 1e-9:
        return None
    distance = -origin[2] / direction[2]
    if distance <= 0:
        return None
    return origin + (distance * direction)


def _project_gaze_to_world(
    sample: GazeSample,
    pose_record: PoseRecord,
    intrinsics: CameraIntrinsics,
    ray_source: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    assert pose_record.rvec is not None
    assert pose_record.tvec is not None

    pixel_x = float("nan")
    pixel_y = float("nan")

    if ray_source == "gaze3d":
        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []

        if sample.left_origin is not None:
            origins.append(np.array(sample.left_origin, dtype=np.float64))
        if sample.right_origin is not None:
            origins.append(np.array(sample.right_origin, dtype=np.float64))
        if sample.left_direction is not None:
            directions.append(np.array(sample.left_direction, dtype=np.float64))
        if sample.right_direction is not None:
            directions.append(np.array(sample.right_direction, dtype=np.float64))

        if origins:
            origin_cam = np.mean(np.vstack(origins), axis=0)
        else:
            origin_cam = np.zeros(3, dtype=np.float64)

        if directions:
            ray_cam = np.mean(np.vstack(directions), axis=0)
        elif sample.gaze3d is not None:
            gaze3d_point = np.array(sample.gaze3d, dtype=np.float64)
            ray_cam = gaze3d_point - origin_cam
        else:
            pixel_x, pixel_y = _gaze_to_pixel(sample.gaze_x, sample.gaze_y, intrinsics)
            origin_cam = np.zeros(3, dtype=np.float64)
            ray_cam = _ray_from_pixel(pixel_x, pixel_y, intrinsics.matrix, intrinsics.distortion)
    else:
        pixel_x, pixel_y = _gaze_to_pixel(sample.gaze_x, sample.gaze_y, intrinsics)
        origin_cam = np.zeros(3, dtype=np.float64)
        ray_cam = _ray_from_pixel(pixel_x, pixel_y, intrinsics.matrix, intrinsics.distortion)

    ray_cam_norm = np.linalg.norm(ray_cam)
    if ray_cam_norm > 1e-12:
        ray_cam = ray_cam / ray_cam_norm

    rotation_matrix, _ = cv2.Rodrigues(pose_record.rvec)
    rotation_world_from_camera = rotation_matrix.T
    camera_origin_world = (-rotation_world_from_camera @ pose_record.tvec).reshape(3)
    ray_origin_world = camera_origin_world + (rotation_world_from_camera @ origin_cam.reshape(3, 1)).reshape(3)
    ray_world = rotation_world_from_camera @ ray_cam
    ray_world = ray_world.reshape(3)
    ray_norm = np.linalg.norm(ray_world)
    if ray_norm > 1e-12:
        ray_world = ray_world / ray_norm

    intersection = _intersect_ray_with_plane_z0(ray_origin_world, ray_world)

    debug = {
        "ray_source": ray_source,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "camera_origin_world": camera_origin_world.tolist(),
        "ray_origin_world": ray_origin_world.tolist(),
        "ray_origin_cam": origin_cam.tolist(),
        "ray_world": ray_world.tolist(),
    }
    return intersection, debug


def _write_pose_records(output_path: Path, pose_records: list[PoseRecord]) -> None:
    with output_path.open("w", encoding="utf-8") as file_pointer:
        for record in pose_records:
            payload = {
                "frame_idx": record.frame_idx,
                "frame_time_s": record.frame_time_s,
                "has_pose": record.has_pose,
                "marker_count": record.marker_count,
                "reproj_error_px": record.reproj_error_px,
                "rvec": record.rvec.reshape(-1).tolist() if record.rvec is not None else None,
                "tvec": record.tvec.reshape(-1).tolist() if record.tvec is not None else None,
            }
            file_pointer.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_world_gaze(
    output_ndjson: Path,
    output_csv: Path,
    device_id: str,
    aligned_rows: list[dict[str, Any]],
) -> None:
    with output_ndjson.open("w", encoding="utf-8") as file_pointer:
        for row in aligned_rows:
            file_pointer.write(json.dumps(row, ensure_ascii=False) + "\n")

    with output_csv.open("w", encoding="utf-8", newline="") as csv_pointer:
        writer = csv.DictWriter(
            csv_pointer,
            fieldnames=[
                "device_id",
                "sample_time_s",
                "frame_idx",
                "frame_time_s",
                "time_delta_s",
                "gaze_x",
                "gaze_y",
                "world_x",
                "world_y",
                "world_z",
                "marker_count",
                "reproj_error_px",
            ],
        )
        writer.writeheader()
        for row in aligned_rows:
            writer.writerow(
                {
                    "device_id": device_id,
                    "sample_time_s": row["sample_time_s"],
                    "frame_idx": row["frame_idx"],
                    "frame_time_s": row["frame_time_s"],
                    "time_delta_s": row["time_delta_s"],
                    "gaze_x": row["gaze_x"],
                    "gaze_y": row["gaze_y"],
                    "world_x": row["world_point_m"][0] if row["world_point_m"] is not None else None,
                    "world_y": row["world_point_m"][1] if row["world_point_m"] is not None else None,
                    "world_z": row["world_point_m"][2] if row["world_point_m"] is not None else None,
                    "marker_count": row["marker_count"],
                    "reproj_error_px": row["reproj_error_px"],
                }
            )


def _run_device(
    device: DeviceConfig,
    marker_map: dict[int, np.ndarray],
    aruco_dictionary_name: str,
    ticks_per_second: float,
    max_time_delta_s: float,
    ray_source: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_records = _collect_frame_poses(device, marker_map, aruco_dictionary_name)
    pose_output_path = output_dir / f"{device.device_id}_frame_pose.ndjson"
    _write_pose_records(pose_output_path, pose_records)

    gaze_samples = _load_gaze_samples(device.gaze_ndjson, ticks_per_second)
    time_index = _build_time_index(pose_records)

    aligned_rows: list[dict[str, Any]] = []
    dropped_no_pose = 0
    dropped_time_gap = 0

    for sample in gaze_samples:
        nearest = _nearest_pose_record(pose_records, time_index, sample.sample_time_s)
        if nearest is None:
            dropped_no_pose += 1
            continue

        time_delta_s = abs(nearest.frame_time_s - sample.sample_time_s)
        if time_delta_s > max_time_delta_s:
            dropped_time_gap += 1
            continue
        if not nearest.has_pose:
            dropped_no_pose += 1
            continue

        world_point, debug = _project_gaze_to_world(sample, nearest, device.intrinsics, ray_source)
        row = {
            "device_id": device.device_id,
            "sample_time_s": sample.sample_time_s,
            "frame_idx": nearest.frame_idx,
            "frame_time_s": nearest.frame_time_s,
            "time_delta_s": time_delta_s,
            "gaze_x": sample.gaze_x,
            "gaze_y": sample.gaze_y,
            "world_point_m": world_point.tolist() if world_point is not None else None,
            "marker_count": nearest.marker_count,
            "reproj_error_px": nearest.reproj_error_px,
            "debug": debug,
        }
        aligned_rows.append(row)

    world_ndjson = output_dir / f"{device.device_id}_gaze_world.ndjson"
    world_csv = output_dir / f"{device.device_id}_gaze_world.csv"
    _write_world_gaze(world_ndjson, world_csv, device.device_id, aligned_rows)

    return {
        "device_id": device.device_id,
        "pose_frames_total": len(pose_records),
        "pose_frames_with_solution": sum(1 for item in pose_records if item.has_pose),
        "gaze_samples_total": len(gaze_samples),
        "gaze_samples_aligned": len(aligned_rows),
        "dropped_no_pose": dropped_no_pose,
        "dropped_time_gap": dropped_time_gap,
        "pose_output": str(pose_output_path),
        "world_output_ndjson": str(world_ndjson),
        "world_output_csv": str(world_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to alignment YAML config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for derived outputs")
    parser.add_argument(
        "--ticks-per-second",
        type=float,
        default=10_000_000.0,
        help="Conversion for Tobii timestamp_ticks to seconds (default assumes .NET ticks)",
    )
    parser.add_argument(
        "--max-time-delta-s",
        type=float,
        default=0.100,
        help="Maximum allowed time gap between gaze sample and nearest pose frame",
    )
    parser.add_argument(
        "--ray-source",
        choices=["gaze2d", "gaze3d"],
        default="gaze2d",
        help=(
            "Ray construction mode: gaze2d uses image gaze + intrinsics; "
            "gaze3d uses eye origin/direction or gaze3d vectors when available"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices, marker_map, world = _load_config(args.config)
    aruco_dictionary_name = str(world.get("aruco_dictionary", "DICT_4X4_50"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(args.config),
        "aruco_dictionary": aruco_dictionary_name,
        "ticks_per_second": args.ticks_per_second,
        "max_time_delta_s": args.max_time_delta_s,
        "ray_source": args.ray_source,
        "devices": [],
    }

    for device in devices:
        result = _run_device(
            device=device,
            marker_map=marker_map,
            aruco_dictionary_name=aruco_dictionary_name,
            ticks_per_second=args.ticks_per_second,
            max_time_delta_s=args.max_time_delta_s,
            ray_source=args.ray_source,
            output_dir=args.output_dir,
        )
        summary["devices"].append(result)
        print(
            f"[{device.device_id}] aligned {result['gaze_samples_aligned']}/{result['gaze_samples_total']} "
            f"samples (pose frames: {result['pose_frames_with_solution']}/{result['pose_frames_total']})"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
