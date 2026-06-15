#!/usr/bin/env python3
"""Video-only 3D pipeline for desk-centered gaze + participant gestures.

This pipeline orchestrates three existing toolchains:
1) Track Tobii glasses pose from fixed multicam video + marker geometry.
2) Reconstruct participant 3D body skeletons from multicam 2D pose detections.
3) Derive per-participant gesture events from 3D skeleton trajectories.

The shared coordinate system is inherited from the calibration TOML / table markers
used by the tracker and reconstruction tools. For desk-centered analyses, calibrate
with the board centered at the desk and keep table marker definitions consistent.

Outputs are written to a dedicated folder and are fully derived artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# BODY_25 indices used by this script
NOSE = 0
R_SHOULDER = 2
R_ELBOW = 3
R_WRIST = 4
L_SHOULDER = 5
L_ELBOW = 6
L_WRIST = 7
MID_HIP = 8


@dataclass
class GestureThresholds:
    """Heuristic thresholds for gesture extraction."""

    hand_near_head_m: float = 0.18
    hand_near_chest_m: float = 0.20
    arm_extension_ratio: float = 1.20
    min_event_frames: int = 6


def _safe_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two 3D points, inf if either is invalid."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return float("inf")
    return float(np.linalg.norm(a - b))


def _joint_xyz(person_frame: np.ndarray, kp_idx: int) -> np.ndarray:
    """Return xyz for keypoint or NaNs when missing."""
    if kp_idx >= person_frame.shape[0]:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return person_frame[kp_idx, :3].astype(np.float64)


def _is_valid_joint(person_frame: np.ndarray, kp_idx: int, min_conf: float) -> bool:
    if kp_idx >= person_frame.shape[0]:
        return False
    conf = float(person_frame[kp_idx, 3]) if person_frame.shape[1] > 3 else 0.0
    xyz = person_frame[kp_idx, :3]
    return conf >= min_conf and not np.any(np.isnan(xyz))


def _frame_binary_gestures(
    person_frame: np.ndarray,
    min_conf: float,
    th: GestureThresholds,
) -> dict[str, bool]:
    """Compute per-frame binary gesture flags for one person."""
    nose = _joint_xyz(person_frame, NOSE)
    mid_hip = _joint_xyz(person_frame, MID_HIP)

    l_sh = _joint_xyz(person_frame, L_SHOULDER)
    l_el = _joint_xyz(person_frame, L_ELBOW)
    l_wr = _joint_xyz(person_frame, L_WRIST)

    r_sh = _joint_xyz(person_frame, R_SHOULDER)
    r_el = _joint_xyz(person_frame, R_ELBOW)
    r_wr = _joint_xyz(person_frame, R_WRIST)

    left_valid = (
        _is_valid_joint(person_frame, L_SHOULDER, min_conf)
        and _is_valid_joint(person_frame, L_ELBOW, min_conf)
        and _is_valid_joint(person_frame, L_WRIST, min_conf)
    )
    right_valid = (
        _is_valid_joint(person_frame, R_SHOULDER, min_conf)
        and _is_valid_joint(person_frame, R_ELBOW, min_conf)
        and _is_valid_joint(person_frame, R_WRIST, min_conf)
    )

    left_hand_to_head = left_valid and _safe_dist(l_wr, nose) <= th.hand_near_head_m
    right_hand_to_head = right_valid and _safe_dist(r_wr, nose) <= th.hand_near_head_m

    left_hand_to_chest = left_valid and _safe_dist(l_wr, mid_hip) <= th.hand_near_chest_m
    right_hand_to_chest = right_valid and _safe_dist(r_wr, mid_hip) <= th.hand_near_chest_m

    left_ext = False
    right_ext = False
    if left_valid:
        seg1 = _safe_dist(l_sh, l_el)
        seg2 = _safe_dist(l_el, l_wr)
        reach = _safe_dist(l_sh, l_wr)
        if math.isfinite(seg1) and math.isfinite(seg2) and (seg1 + seg2) > 1e-6:
            left_ext = reach / (seg1 + seg2) >= th.arm_extension_ratio
    if right_valid:
        seg1 = _safe_dist(r_sh, r_el)
        seg2 = _safe_dist(r_el, r_wr)
        reach = _safe_dist(r_sh, r_wr)
        if math.isfinite(seg1) and math.isfinite(seg2) and (seg1 + seg2) > 1e-6:
            right_ext = reach / (seg1 + seg2) >= th.arm_extension_ratio

    return {
        "left_hand_to_head": bool(left_hand_to_head),
        "right_hand_to_head": bool(right_hand_to_head),
        "left_hand_to_chest": bool(left_hand_to_chest),
        "right_hand_to_chest": bool(right_hand_to_chest),
        "left_arm_extended": bool(left_ext),
        "right_arm_extended": bool(right_ext),
        "both_hands_to_head": bool(left_hand_to_head and right_hand_to_head),
    }


def _collapse_binary_runs(values: list[bool], min_len: int) -> list[tuple[int, int]]:
    """Return inclusive [start, end] runs where values are true."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            end = i - 1
            if (end - start + 1) >= min_len:
                runs.append((start, end))
            start = None
    if start is not None:
        end = len(values) - 1
        if (end - start + 1) >= min_len:
            runs.append((start, end))
    return runs


def extract_gesture_events(
    skeleton_path: Path,
    output_ndjson: Path,
    summary_json: Path,
    fps: float,
    min_confidence: float,
    thresholds: GestureThresholds,
) -> dict[str, Any]:
    """Extract gesture events from a 3D skeleton file.

    Input shape expected: (frames, people, keypoints, dims>=4).
    """
    data = np.load(skeleton_path, allow_pickle=False)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D skeleton array, got shape {data.shape}")

    n_frames, n_people, _n_kp, n_dims = data.shape
    if n_dims < 4:
        raise ValueError("Skeleton must include at least [x,y,z,confidence]")

    per_person_flags: dict[int, dict[str, list[bool]]] = {}
    for p in range(n_people):
        per_person_flags[p] = {
            "left_hand_to_head": [],
            "right_hand_to_head": [],
            "left_hand_to_chest": [],
            "right_hand_to_chest": [],
            "left_arm_extended": [],
            "right_arm_extended": [],
            "both_hands_to_head": [],
        }
        for f in range(n_frames):
            person_frame = data[f, p]
            flags = _frame_binary_gestures(person_frame, min_confidence, thresholds)
            for k, v in flags.items():
                per_person_flags[p][k].append(v)

    events: list[dict[str, Any]] = []
    for p in range(n_people):
        participant = f"P{p + 1}"
        for gesture_name, series in per_person_flags[p].items():
            runs = _collapse_binary_runs(series, thresholds.min_event_frames)
            for start, end in runs:
                event = {
                    "participant": participant,
                    "person_index": p,
                    "gesture": gesture_name,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time_s": start / fps,
                    "end_time_s": end / fps,
                    "duration_s": (end - start + 1) / fps,
                    "n_frames": end - start + 1,
                }
                events.append(event)

    events.sort(key=lambda e: (e["start_frame"], e["person_index"], e["gesture"]))

    output_ndjson.parent.mkdir(parents=True, exist_ok=True)
    with output_ndjson.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    summary: dict[str, Any] = {
        "skeleton": str(skeleton_path),
        "fps": fps,
        "frames": n_frames,
        "people": n_people,
        "min_confidence": min_confidence,
        "thresholds": {
            "hand_near_head_m": thresholds.hand_near_head_m,
            "hand_near_chest_m": thresholds.hand_near_chest_m,
            "arm_extension_ratio": thresholds.arm_extension_ratio,
            "min_event_frames": thresholds.min_event_frames,
        },
        "n_events": len(events),
        "events_per_participant": {},
        "events_per_gesture": {},
    }

    for e in events:
        participant = e["participant"]
        gesture = e["gesture"]
        summary["events_per_participant"][participant] = (
            summary["events_per_participant"].get(participant, 0) + 1
        )
        summary["events_per_gesture"][gesture] = (
            summary["events_per_gesture"].get(gesture, 0) + 1
        )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _run_step(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def _discover_p20_videos(videos_dir: Path) -> list[Path]:
    """Find six P20 camera videos under videos_dir.

    Matches names containing "panacast-20-cam" and keeps one file per cam index.
    """
    if not videos_dir.exists() or not videos_dir.is_dir():
        return []

    candidates = sorted(
        [
            p
            for p in videos_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".mkv", ".mp4", ".mov", ".avi"}
            and "panacast-20-cam" in p.name.lower()
        ]
    )

    by_cam: dict[str, Path] = {}
    for p in candidates:
        name = p.name.lower()
        cam_id = None
        for i in range(1, 7):
            if f"panacast-20-cam{i}" in name:
                cam_id = f"cam{i}"
                break
        if cam_id is None:
            continue
        if cam_id not in by_cam:
            by_cam[cam_id] = p

    ordered = [by_cam[k] for k in sorted(by_cam.keys())]
    return ordered


def _ensure_calibration(args: argparse.Namespace) -> Path:
    """Return a usable calibration path, auto-generating if allowed and missing."""
    if args.calibration.exists():
        return args.calibration

    if not args.auto_calibrate_missing:
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")

    p20_videos = _discover_p20_videos(args.videos_dir)
    if len(p20_videos) < 6:
        raise RuntimeError(
            "Auto-calibration requires six P20 videos but found "
            f"{len(p20_videos)} in {args.videos_dir}"
        )

    calib_stage_dir = args.output_dir / "_auto_calibration" / "videos"
    calib_stage_dir.mkdir(parents=True, exist_ok=True)

    staged: list[Path] = []
    for src in p20_videos:
        dst = calib_stage_dir / src.name
        if not dst.exists():
            try:
                dst.hardlink_to(src)
            except Exception:
                import shutil

                shutil.copy2(src, dst)
        staged.append(dst)

    auto_calib_path = args.output_dir / "auto_calibration_charuco.toml"
    cmd = [
        sys.executable,
        str(Path("tools") / "calibrate_charuco.py"),
        "calibrate",
        "--videos-dir",
        str(calib_stage_dir),
        "--board-type",
        "auto",
        "--board-config",
        str(args.board_config),
        "--camera-specs",
        str(args.camera_specs),
        "--output",
        str(auto_calib_path),
        "--init-focal",
    ]
    _run_step(cmd)

    if not auto_calib_path.exists():
        raise RuntimeError(f"Auto-calibration did not produce {auto_calib_path}")

    return auto_calib_path


def _pose_json_dirs(pose_root: Path) -> list[Path]:
    """Return sorted *_json directories under pose_root."""
    if not pose_root.exists() or not pose_root.is_dir():
        return []
    return sorted([p for p in pose_root.iterdir() if p.is_dir() and p.name.endswith("_json")])


def _prereq_report(args: argparse.Namespace) -> dict[str, Any]:
    """Collect prerequisite status for a pipeline run."""
    pose_dirs = _pose_json_dirs(args.pose_root)
    report: dict[str, Any] = {
        "calibration_exists": bool(args.calibration.exists()),
        "auto_calibrate_missing": bool(args.auto_calibrate_missing),
        "videos_dir_exists": bool(args.videos_dir.exists() and args.videos_dir.is_dir()),
        "tracker_config_exists": bool(args.tracker_config.exists()),
        "pose_root_exists": bool(args.pose_root.exists() and args.pose_root.is_dir()),
        "pose_json_dir_count": len(pose_dirs),
        "pose_json_dirs": [str(p) for p in pose_dirs],
        "p20_videos_for_autocalibration": [str(p) for p in _discover_p20_videos(args.videos_dir)],
        "ready": False,
        "missing": [],
    }

    if not report["calibration_exists"] and not report["auto_calibrate_missing"]:
        report["missing"].append("calibration")
    if not report["calibration_exists"] and report["auto_calibrate_missing"]:
        if len(report["p20_videos_for_autocalibration"]) < 6:
            report["missing"].append("p20_videos_for_autocalibration")
    if not report["videos_dir_exists"]:
        report["missing"].append("videos_dir")
    if not report["tracker_config_exists"]:
        report["missing"].append("tracker_config")
    if not report["pose_root_exists"]:
        report["missing"].append("pose_root")
    if report["pose_json_dir_count"] < 2:
        report["missing"].append("pose_json_dirs")

    report["ready"] = len(report["missing"]) == 0
    return report


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run tracker -> 3D reconstruction -> optional refinement -> gestures."""
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prereq = _prereq_report(args)
    if args.dry_run:
        dry = {
            "mode": "dry_run",
            "prerequisites": prereq,
        }
        with (out_dir / "pipeline_dry_run.json").open("w", encoding="utf-8") as f:
            json.dump(dry, f, indent=2)
        return dry

    if not prereq["ready"]:
        raise RuntimeError(
            "Pipeline prerequisites not met. Missing: "
            + ", ".join(prereq["missing"])
            + ". Use --dry-run to inspect details."
        )

    calibration_path = _ensure_calibration(args)

    py = sys.executable

    tobii_out = out_dir / "tobii_world"
    tracker_cmd = [
        py,
        str(Path("tools") / "tobii_multicam_glasses_tracker.py"),
        "--calibration",
        str(calibration_path),
        "--videos-dir",
        str(args.videos_dir),
        "--config",
        str(args.tracker_config),
        "--output-dir",
        str(tobii_out),
    ]
    if args.verbose:
        tracker_cmd.append("--verbose")
    _run_step(tracker_cmd)

    skeleton_raw = out_dir / "skeleton_3d.npy"
    reconstruct_cmd = [
        py,
        str(Path("tools") / "multicam_pose3d.py"),
        "reconstruct",
        "--calibration",
        str(calibration_path),
        "--pose-root",
        str(args.pose_root),
        "--output",
        str(skeleton_raw),
        "--fps",
        str(args.fps),
        "--max-epipolar-px",
        str(args.max_epipolar_px),
        "--max-reproj-px",
        str(args.max_reproj_px),
    ]
    if args.events_jsonl:
        reconstruct_cmd.extend(["--events-jsonl", str(args.events_jsonl)])
    if args.frame_log_dir:
        reconstruct_cmd.extend(["--frame-log-dir", str(args.frame_log_dir)])
    if args.lsl_dir:
        reconstruct_cmd.extend(["--lsl-dir", str(args.lsl_dir)])
    if args.camera_zones:
        reconstruct_cmd.append("--camera-zones")
        reconstruct_cmd.extend(args.camera_zones)
    if args.flip_cameras:
        reconstruct_cmd.append("--flip-cameras")
        reconstruct_cmd.extend(args.flip_cameras)
    if args.face_cameras:
        reconstruct_cmd.append("--face-cameras")
        reconstruct_cmd.extend(args.face_cameras)
    if args.no_front_facing_filter:
        reconstruct_cmd.append("--no-front-facing-filter")
    reconstruct_cmd.extend(["--min-face-conf", str(args.min_face_conf)])

    _run_step(reconstruct_cmd)

    skeleton_for_gestures = skeleton_raw
    if args.refine_skeleton:
        refined = out_dir / "skeleton_3d_refined.npy"
        refine_cmd = [
            py,
            str(Path("tools") / "refine_skeleton_3d.py"),
            "--input",
            str(skeleton_raw),
            "--output",
            str(refined),
            "--min-confidence",
            str(args.refine_min_confidence),
            "--max-reproj",
            str(args.refine_max_reproj_px),
            "--min-cameras",
            str(args.refine_min_cameras),
            "--smooth-cutoff",
            str(args.refine_smooth_cutoff_hz),
            "--fps",
            str(args.fps),
        ]
        _run_step(refine_cmd)
        skeleton_for_gestures = refined

    gestures_ndjson = out_dir / "gestures_events.ndjson"
    gestures_summary_json = out_dir / "gestures_summary.json"
    gesture_summary = extract_gesture_events(
        skeleton_path=skeleton_for_gestures,
        output_ndjson=gestures_ndjson,
        summary_json=gestures_summary_json,
        fps=args.fps,
        min_confidence=args.gesture_min_confidence,
        thresholds=GestureThresholds(
            hand_near_head_m=args.gesture_hand_near_head_m,
            hand_near_chest_m=args.gesture_hand_near_chest_m,
            arm_extension_ratio=args.gesture_arm_extension_ratio,
            min_event_frames=args.gesture_min_event_frames,
        ),
    )

    pipeline_summary = {
        "calibration_used": str(calibration_path),
        "tobii_world_dir": str(tobii_out),
        "skeleton_raw": str(skeleton_raw),
        "skeleton_for_gesture": str(skeleton_for_gestures),
        "gestures_ndjson": str(gestures_ndjson),
        "gestures_summary": str(gestures_summary_json),
        "n_gesture_events": int(gesture_summary.get("n_events", 0)),
    }
    with (out_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)

    return pipeline_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video-only pipeline: Tobii world gaze + multicam 3D + gesture extraction",
    )
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--videos-dir", type=Path, required=True,
                   help="Fixed multicam videos used by Tobii glasses tracker")
    p.add_argument("--tracker-config", type=Path, required=True,
                   help="YAML config for tobii_multicam_glasses_tracker")
    p.add_argument("--pose-root", type=Path, required=True,
                   help="Root folder containing *_json pose folders for multicam_pose3d")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Pipeline output folder")

    p.add_argument("--events-jsonl", type=Path, default=None)
    p.add_argument("--frame-log-dir", type=Path, default=None)
    p.add_argument("--lsl-dir", type=Path, default=None)

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-epipolar-px", type=float, default=40.0)
    p.add_argument("--max-reproj-px", type=float, default=30.0)

    p.add_argument("--camera-zones", nargs="+", default=None,
                   help="Zone mapping for stable person indices, e.g. cam1+cam4:0,1 cam2+cam3:2,3")
    p.add_argument("--flip-cameras", nargs="*", default=None)
    p.add_argument("--face-cameras", nargs="+", default=None)
    p.add_argument("--no-front-facing-filter", action="store_true")
    p.add_argument("--min-face-conf", type=float, default=0.3)

    p.add_argument("--refine-skeleton", action="store_true",
                   help="Run refine_skeleton_3d before gesture extraction")
    p.add_argument("--refine-min-confidence", type=float, default=0.3)
    p.add_argument("--refine-max-reproj-px", type=float, default=20.0)
    p.add_argument("--refine-min-cameras", type=int, default=2)
    p.add_argument("--refine-smooth-cutoff-hz", type=float, default=6.0)

    p.add_argument("--gesture-min-confidence", type=float, default=0.3)
    p.add_argument("--gesture-hand-near-head-m", type=float, default=0.18)
    p.add_argument("--gesture-hand-near-chest-m", type=float, default=0.20)
    p.add_argument("--gesture-arm-extension-ratio", type=float, default=1.20)
    p.add_argument("--gesture-min-event-frames", type=int, default=6)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and write pipeline_dry_run.json without executing stages")
    p.add_argument("--auto-calibrate-missing", action="store_true", default=True,
                   help="If --calibration does not exist, auto-calibrate from six P20 videos in --videos-dir")
    p.add_argument("--no-auto-calibrate-missing", dest="auto_calibrate_missing", action="store_false",
                   help="Disable auto-calibration fallback when --calibration is missing")
    p.add_argument("--board-config", type=Path, default=Path("configs/desk_markers_large.yaml"),
                   help="Board config used by auto-calibration fallback")
    p.add_argument("--camera-specs", type=Path, default=Path("configs/camera_specs.json"),
                   help="Camera specs used by auto-calibration fallback")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        summary = run_pipeline(args)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    logger.info("Pipeline completed")
    logger.info("Output summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
