#!/usr/bin/env python3
"""
Quick MediaPipe Pose test on multi-camera videos.

Processes a short clip from each camera, reports detection quality,
and outputs annotated frames + JSON keypoints (OpenPose-compatible format).

Usage
-----
  # Test one camera (first 100 frames)
  python tools/test_mediapipe_pose.py \
      --video new_data/ses-20260202_test/video/jabra_panacast_20_cam1_vid_video.mkv

  # Test all cameras in a session
  python tools/test_mediapipe_pose.py \
      --session-dir new_data/ses-20260202_test/video

  # Full video (no frame limit)
  python tools/test_mediapipe_pose.py \
      --video new_data/.../cam1.mkv --max-frames 0

  # Write OpenPose-compatible JSON per frame
  python tools/test_mediapipe_pose.py \
      --video new_data/.../cam1.mkv --write-json output_json/

  # Write annotated video
  python tools/test_mediapipe_pose.py \
      --video new_data/.../cam1.mkv --write-video output_annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# MediaPipe Pose landmark indices → OpenPose BODY_25 mapping
# MediaPipe has 33 landmarks; OpenPose BODY_25 has 25.
# We map the closest equivalents.
MEDIAPIPE_TO_OPENPOSE_25 = {
    0: 0,    # Nose
    # 1: left eye inner (no direct OP equivalent)
    # 2: left eye (OP 15 = left eye)
    # 3: left eye outer
    # 4: right eye inner
    # 5: right eye (OP 16 = right eye)
    # 6: right eye outer
    # 7: left ear (OP 17)
    # 8: right ear (OP 18)
    11: 5,   # Left shoulder → OP 5
    12: 2,   # Right shoulder → OP 2
    13: 6,   # Left elbow → OP 6
    14: 3,   # Right elbow → OP 3
    15: 7,   # Left wrist → OP 7
    16: 4,   # Right wrist → OP 4
    23: 12,  # Left hip → OP 12
    24: 9,   # Right hip → OP 9
    25: 13,  # Left knee → OP 13
    26: 10,  # Right knee → OP 10
    27: 14,  # Left ankle → OP 14
    28: 11,  # Right ankle → OP 11
    31: 21,  # Left foot index → OP 21 (left big toe)
    32: 24,  # Right foot index → OP 24 (right big toe)
}

# Additional mappings for eyes/ears
MEDIAPIPE_EXTRA = {
    2: 15,   # Left eye → OP 15
    5: 16,   # Right eye → OP 16
    7: 17,   # Left ear → OP 17
    8: 18,   # Right ear → OP 18
}


def mediapipe_to_openpose_keypoints(
    landmarks, image_width: int, image_height: int
) -> list[float]:
    """
    Convert MediaPipe Pose landmarks to OpenPose BODY_25 format.

    Returns flat list: [x0, y0, conf0, x1, y1, conf1, ...] with 25 keypoints.
    """
    # Initialize 25 keypoints with zeros
    keypoints = [0.0] * (25 * 3)

    all_mappings = {**MEDIAPIPE_TO_OPENPOSE_25, **MEDIAPIPE_EXTRA}

    for mp_idx, op_idx in all_mappings.items():
        if mp_idx < len(landmarks):
            lm = landmarks[mp_idx]
            x = lm.x * image_width
            y = lm.y * image_height
            conf = lm.visibility
            keypoints[op_idx * 3] = x
            keypoints[op_idx * 3 + 1] = y
            keypoints[op_idx * 3 + 2] = conf

    # Neck (OP index 1) = midpoint of shoulders
    l_shoulder_idx = 5   # OP left shoulder
    r_shoulder_idx = 2   # OP right shoulder
    lx, ly, lc = keypoints[l_shoulder_idx*3:l_shoulder_idx*3+3]
    rx, ry, rc = keypoints[r_shoulder_idx*3:r_shoulder_idx*3+3]
    if lc > 0 and rc > 0:
        keypoints[1*3] = (lx + rx) / 2
        keypoints[1*3+1] = (ly + ry) / 2
        keypoints[1*3+2] = min(lc, rc)

    # Mid-hip (OP index 8) = midpoint of hips
    l_hip_idx = 12
    r_hip_idx = 9
    lx, ly, lc = keypoints[l_hip_idx*3:l_hip_idx*3+3]
    rx, ry, rc = keypoints[r_hip_idx*3:r_hip_idx*3+3]
    if lc > 0 and rc > 0:
        keypoints[8*3] = (lx + rx) / 2
        keypoints[8*3+1] = (ly + ry) / 2
        keypoints[8*3+2] = min(lc, rc)

    return keypoints


def _ensure_pose_model(model_complexity: int = 1) -> Path:
    """Download the MediaPipe Pose Landmarker model if not cached."""
    import urllib.request

    complexity_map = {
        0: ("pose_landmarker_lite", "pose_landmarker_lite.task"),
        1: ("pose_landmarker_full", "pose_landmarker_full.task"),
        2: ("pose_landmarker_heavy", "pose_landmarker_heavy.task"),
    }
    model_name, filename = complexity_map.get(model_complexity, complexity_map[1])

    cache_dir = Path(__file__).resolve().parent / ".mediapipe_models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / filename

    if model_path.exists():
        return model_path

    url = (
        f"https://storage.googleapis.com/mediapipe-models/"
        f"pose_landmarker/{model_name}/float16/latest/{filename}"
    )
    logger.info(f"Downloading model: {url}")
    urllib.request.urlretrieve(url, model_path)
    logger.info(f"Saved to: {model_path}")
    return model_path


def _draw_pose_landmarks(frame, landmarks, width, height):
    """Draw pose skeleton on frame using OpenCV (no mp.solutions dependency)."""
    # MediaPipe Pose connections (pairs of landmark indices)
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),   # left face
        (0, 4), (4, 5), (5, 6), (6, 8),   # right face
        (9, 10),                            # mouth
        (11, 12),                           # shoulders
        (11, 13), (13, 15),                 # left arm
        (12, 14), (14, 16),                 # right arm
        (15, 17), (15, 19), (15, 21),       # left hand
        (16, 18), (16, 20), (16, 22),       # right hand
        (11, 23), (12, 24),                 # torso
        (23, 24),                           # hips
        (23, 25), (25, 27),                 # left leg
        (24, 26), (26, 28),                 # right leg
        (27, 29), (27, 31),                 # left foot
        (28, 30), (28, 32),                 # right foot
    ]

    points = []
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py, lm.visibility))

    # Draw connections
    for i, j in POSE_CONNECTIONS:
        if i < len(points) and j < len(points):
            if points[i][2] > 0.3 and points[j][2] > 0.3:
                cv2.line(frame, points[i][:2], points[j][:2], (0, 255, 0), 2)

    # Draw keypoints
    for px, py, vis in points:
        if vis > 0.3:
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


def process_video(
    video_path: Path,
    max_frames: int = 100,
    write_json_dir: Path | None = None,
    json_file_prefix: str | None = None,
    write_video_path: Path | None = None,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> dict:
    """
    Process a video with MediaPipe Pose Landmarker (Tasks API).

    Returns detection statistics dict.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    # Download model if needed
    model_path = _ensure_pose_model(model_complexity)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {"error": f"Cannot open {video_path}"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames > 0:
        n_to_process = min(max_frames, total_frames)
    else:
        n_to_process = total_frames

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Resolution: {width}x{height} @ {fps:.1f} fps")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Processing: {n_to_process} frames")

    # Setup output dirs
    if write_json_dir:
        write_json_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if write_video_path:
        write_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(write_video_path), fourcc, fps, (width, height)
        )

    # Stats
    frames_with_detection = 0
    total_confidence = []
    keypoint_visibility = np.zeros(33)  # MediaPipe has 33 landmarks
    frame_count = 0

    # Create Pose Landmarker
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while frame_count < n_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to MediaPipe Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect (VIDEO mode needs timestamp in ms)
            timestamp_ms = int(frame_count * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected = len(results.pose_landmarks) > 0

            if detected:
                frames_with_detection += 1
                # Use first detected person for stats
                lms = results.pose_landmarks[0]

                # Accumulate stats
                confs = [lm.visibility for lm in lms]
                total_confidence.append(np.mean(confs))
                for i, lm in enumerate(lms):
                    if lm.visibility > 0.5:
                        keypoint_visibility[i] += 1

                # Write OpenPose-compatible JSON (all detected people)
                if write_json_dir:
                    people_list = []
                    for person_idx, person_lms in enumerate(results.pose_landmarks):
                        op_keypoints = mediapipe_to_openpose_keypoints(
                            person_lms, width, height
                        )
                        people_list.append({
                            "person_id": [person_idx],
                            "pose_keypoints_2d": op_keypoints,
                            "face_keypoints_2d": [],
                            "hand_left_keypoints_2d": [],
                            "hand_right_keypoints_2d": [],
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d": [],
                            "hand_right_keypoints_3d": [],
                        })
                    json_data = {"version": 1.3, "people": people_list}
                    prefix = json_file_prefix or video_path.stem
                    json_path = (
                        write_json_dir
                        / f"{prefix}_{frame_count:012d}_keypoints.json"
                    )
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(json_path, "w") as f:
                        json.dump(json_data, f)

                # Draw skeleton on frame for all detected people
                if video_writer:
                    for person_lms in results.pose_landmarks:
                        _draw_pose_landmarks(frame, person_lms, width, height)
            else:
                # Write empty JSON
                if write_json_dir:
                    json_data = {"version": 1.3, "people": []}
                    prefix = json_file_prefix or video_path.stem
                    json_path = (
                        write_json_dir
                        / f"{prefix}_{frame_count:012d}_keypoints.json"
                    )
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(json_path, "w") as f:
                        json.dump(json_data, f)

            if video_writer:
                video_writer.write(frame)

            frame_count += 1
            if frame_count % 50 == 0:
                pct = frame_count / n_to_process * 100
                det_pct = frames_with_detection / max(frame_count, 1) * 100
                logger.info(
                    f"  Frame {frame_count}/{n_to_process} "
                    f"({pct:.0f}%) — detection rate: {det_pct:.0f}%"
                )

    cap.release()
    if video_writer:
        video_writer.release()

    # Compute summary
    detection_rate = frames_with_detection / max(frame_count, 1) * 100
    avg_confidence = float(np.mean(total_confidence)) if total_confidence else 0.0

    # Keypoint visibility ranking (top-5, bottom-5)
    visibility_pct = keypoint_visibility / max(frames_with_detection, 1) * 100
    mp_landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    stats = {
        "video": video_path.name,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_count,
        "frames_with_detection": frames_with_detection,
        "detection_rate_pct": round(detection_rate, 1),
        "avg_confidence": round(avg_confidence, 3),
        "top_visible_keypoints": [],
        "least_visible_keypoints": [],
    }

    if frames_with_detection > 0:
        sorted_kps = sorted(
            range(len(visibility_pct)),
            key=lambda i: visibility_pct[i],
            reverse=True,
        )
        stats["top_visible_keypoints"] = [
            (mp_landmark_names[i], round(visibility_pct[i], 1))
            for i in sorted_kps[:5]
        ]
        stats["least_visible_keypoints"] = [
            (mp_landmark_names[i], round(visibility_pct[i], 1))
            for i in sorted_kps[-5:]
        ]

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Test MediaPipe Pose on multi-camera videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", type=Path, help="Single video file to process"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Directory with camera videos (processes all .mkv/.mp4 files)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to process per video (0=all, default=100)",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Output directory for OpenPose-compatible JSON per frame",
    )
    parser.add_argument(
        "--write-video",
        type=Path,
        help="Output annotated video file",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe model complexity (0=lite, 1=full, 2=heavy; default=1)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Min detection confidence (default=0.5)",
    )
    args = parser.parse_args()

    if not args.video and not args.session_dir:
        parser.error("Provide --video or --session-dir")

    videos = []
    if args.video:
        if not args.video.exists():
            logger.error(f"Video not found: {args.video}")
            sys.exit(1)
        videos.append(args.video)
    elif args.session_dir:
        if not args.session_dir.exists():
            logger.error(f"Directory not found: {args.session_dir}")
            sys.exit(1)
        for ext in ("*.mkv", "*.mp4", "*.avi"):
            videos.extend(sorted(args.session_dir.glob(ext)))
        if not videos:
            logger.error(f"No video files found in {args.session_dir}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("MEDIAPIPE POSE DETECTION TEST")
    logger.info("=" * 70)
    logger.info(f"Videos: {len(videos)}")
    logger.info(f"Max frames/video: {args.max_frames if args.max_frames > 0 else 'all'}")
    logger.info(f"Model complexity: {args.model_complexity}")
    logger.info("")

    all_stats = []
    for i, video_path in enumerate(videos):
        logger.info(f"\n--- Camera {i+1}/{len(videos)}: {video_path.name} ---")

        # Per-video JSON output dir
        json_dir = None
        json_prefix = video_path.stem
        if args.write_json:
            m = re.search(r"panacast-20-cam(\d+)", video_path.stem.lower())
            cam_label = f"cam{m.group(1)}" if m else f"cam{i+1}"
            json_dir = args.write_json / f"{cam_label}_json"
            json_prefix = cam_label

        # Per-video annotated output
        vid_out = None
        if args.write_video and len(videos) == 1:
            vid_out = args.write_video
        elif args.write_video and len(videos) > 1:
            vid_out = args.write_video.parent / f"{video_path.stem}_annotated.mp4"

        stats = process_video(
            video_path=video_path,
            max_frames=args.max_frames,
            write_json_dir=json_dir,
            json_file_prefix=json_prefix,
            write_video_path=vid_out,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
        )
        all_stats.append(stats)

    # Print summary table
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(f"{'Camera':<45} {'Detect%':>8} {'AvgConf':>8} {'Frames':>8}")
    print("-" * 70)
    for s in all_stats:
        if "error" in s:
            print(f"{s.get('video', '?'):<45} {'ERROR':>8}")
        else:
            print(
                f"{s['video']:<45} "
                f"{s['detection_rate_pct']:>7.1f}% "
                f"{s['avg_confidence']:>7.3f} "
                f"{s['frames_with_detection']:>4}/{s['processed_frames']:<3}"
            )

    # Detail for each camera
    for s in all_stats:
        if "error" in s or not s.get("top_visible_keypoints"):
            continue
        print(f"\n  {s['video']}:")
        print(f"    Top visible:   {', '.join(f'{n} ({v}%)' for n, v in s['top_visible_keypoints'])}")
        print(f"    Least visible: {', '.join(f'{n} ({v}%)' for n, v in s['least_visible_keypoints'])}")

    print("\n" + "=" * 70)
    overall = np.mean([s["detection_rate_pct"] for s in all_stats if "error" not in s])
    print(f"Overall detection rate: {overall:.1f}%")
    if overall > 80:
        print("GOOD: MediaPipe detects people reliably. Ready for triangulation.")
    elif overall > 40:
        print("FAIR: Partial detections. Some cameras may have occlusion/lighting issues.")
    else:
        print("POOR: Low detection rate. Check video quality, lighting, and framing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
