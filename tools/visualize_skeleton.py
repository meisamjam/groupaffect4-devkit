#!/usr/bin/env python3
"""
Visualize 3D skeleton data from triangulation output.

Two modes:
  1. plot3d   – Animated 3D matplotlib figure (saved as MP4 or shown interactively)
  2. overlay  – Draw 2D OpenPose skeletons on a source video (annotated MP4)

Usage
-----
  # 3D animation (interactive or saved)
  python tools/visualize_skeleton.py plot3d \
      --skeleton data/.../skeleton_3d.npy \
      --output data/.../skeleton_3d_anim.mp4 \
      --fps 30

  # 2D overlay on a camera video
  python tools/visualize_skeleton.py overlay \
      --video data/.../cam1_video.mkv \
      --pose-dir data/.../openpose/cam1_json \
      --output data/.../cam1_annotated.mp4 \
      --fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── BODY_25 skeleton definition ──────────────────────────────────────
# Index → name
BODY_25_NAMES = [
    "Nose",          # 0
    "Neck",          # 1
    "RShoulder",     # 2
    "RElbow",        # 3
    "RWrist",        # 4
    "LShoulder",     # 5
    "LElbow",        # 6
    "LWrist",        # 7
    "MidHip",        # 8
    "RHip",          # 9
    "RKnee",         # 10
    "RAnkle",        # 11
    "LHip",          # 12
    "LKnee",         # 13
    "LAnkle",        # 14
    "REye",          # 15
    "LEye",          # 16
    "REar",          # 17
    "LEar",          # 18
    "LBigToe",       # 19
    "LSmallToe",     # 20
    "LHeel",         # 21
    "RBigToe",       # 22
    "RSmallToe",     # 23
    "RHeel",         # 24
]

# Bone connections: (parent, child)
BODY_25_BONES = [
    (0, 1),    # Nose → Neck
    (1, 2),    # Neck → RShoulder
    (2, 3),    # RShoulder → RElbow
    (3, 4),    # RElbow → RWrist
    (1, 5),    # Neck → LShoulder
    (5, 6),    # LShoulder → LElbow
    (6, 7),    # LElbow → LWrist
    (1, 8),    # Neck → MidHip
    (8, 9),    # MidHip → RHip
    (9, 10),   # RHip → RKnee
    (10, 11),  # RKnee → RAnkle
    (8, 12),   # MidHip → LHip
    (12, 13),  # LHip → LKnee
    (13, 14),  # LKnee → LAnkle
    (0, 15),   # Nose → REye
    (0, 16),   # Nose → LEye
    (15, 17),  # REye → REar
    (16, 18),  # LEye → LEar
    (14, 19),  # LAnkle → LBigToe
    (14, 20),  # LAnkle → LSmallToe
    (14, 21),  # LAnkle → LHeel
    (11, 22),  # RAnkle → RBigToe
    (11, 23),  # RAnkle → RSmallToe
    (11, 24),  # RAnkle → RHeel
]

# Colors per limb group (BGR for OpenCV, RGB for matplotlib)
BONE_COLORS_BGR = {
    "head":      (0, 255, 255),    # yellow
    "torso":     (255, 255, 0),    # cyan
    "right_arm": (0, 0, 255),      # red
    "left_arm":  (0, 255, 0),      # green
    "right_leg": (255, 0, 255),    # magenta
    "left_leg":  (255, 165, 0),    # orange-ish
    "foot":      (200, 200, 200),  # grey
}

def _bone_color_bgr(i: int, j: int) -> tuple[int, int, int]:
    """Return a BGR color for a bone based on its body region."""
    if i in (0, 15, 16, 17, 18) or j in (15, 16, 17, 18):
        return BONE_COLORS_BGR["head"]
    if (i, j) == (1, 8) or (i, j) == (0, 1):
        return BONE_COLORS_BGR["torso"]
    if i in (2, 3, 4) or j in (2, 3, 4):
        return BONE_COLORS_BGR["right_arm"]
    if i in (5, 6, 7) or j in (5, 6, 7):
        return BONE_COLORS_BGR["left_arm"]
    if i in (9, 10, 11, 22, 23, 24) or j in (9, 10, 11, 22, 23, 24):
        return BONE_COLORS_BGR["right_leg"]
    if i in (12, 13, 14, 19, 20, 21) or j in (12, 13, 14, 19, 20, 21):
        return BONE_COLORS_BGR["left_leg"]
    return BONE_COLORS_BGR["foot"]


def _bone_color_rgb_norm(i: int, j: int) -> tuple[float, float, float]:
    """Return normalized RGB for matplotlib."""
    bgr = _bone_color_bgr(i, j)
    return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)


# ── 3D PLOT ──────────────────────────────────────────────────────────

def cmd_plot3d(args) -> None:
    """Create a 3D skeleton animation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    skeleton = np.load(args.skeleton)
    print(f"Loaded skeleton: {skeleton.shape}")
    n_frames, n_people, n_kp, _ = skeleton.shape

    fps = args.fps
    step = max(1, args.step)

    # Filter valid frames (at least some non-NaN keypoints)
    frame_indices = list(range(0, n_frames, step))

    # Compute axis limits from valid data (use median ± 3*IQR to reject outliers)
    all_xyz = skeleton[:, :, :, :3].reshape(-1, 3)
    valid = ~np.isnan(all_xyz[:, 0])
    all_xyz = all_xyz[valid]

    if len(all_xyz) == 0:
        print("No valid 3D data to plot.")
        return

    def robust_limits(vals):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo = med - 3 * iqr
        hi = med + 3 * iqr
        # Ensure minimum range
        if hi - lo < 100:
            lo, hi = med - 500, med + 500
        return lo, hi

    xlim = robust_limits(all_xyz[:, 0])
    ylim = robust_limits(all_xyz[:, 1])
    zlim = robust_limits(all_xyz[:, 2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title("Frame 0")

    def update(fi):
        ax.cla()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Frame {fi} / {n_frames}")

        for pid in range(n_people):
            kps = skeleton[fi, pid, :, :3]   # (25, 3)
            conf = skeleton[fi, pid, :, 3]

            # Draw bones
            for (a, b) in BODY_25_BONES:
                if np.isnan(kps[a]).any() or np.isnan(kps[b]).any():
                    continue
                if conf[a] < 0.1 or conf[b] < 0.1:
                    continue
                color = _bone_color_rgb_norm(a, b)
                ax.plot(
                    [kps[a, 0], kps[b, 0]],
                    [kps[a, 1], kps[b, 1]],
                    [kps[a, 2], kps[b, 2]],
                    color=color, linewidth=2
                )

            # Draw keypoints
            valid_mask = (~np.isnan(kps[:, 0])) & (conf > 0.1)
            if valid_mask.any():
                ax.scatter(
                    kps[valid_mask, 0],
                    kps[valid_mask, 1],
                    kps[valid_mask, 2],
                    c="white", edgecolors="black", s=30, depthshade=True
                )

    if args.output:
        print(f"Rendering {len(frame_indices)} frames to {args.output} ...")
        anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps)
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(args.output), writer=writer)
        print(f"Saved: {args.output}")
    else:
        anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps)
        plt.show()

    plt.close(fig)


# ── 2D OVERLAY ───────────────────────────────────────────────────────

def cmd_overlay(args) -> None:
    """Overlay OpenPose 2D skeletons on source video."""
    video_path = Path(args.video)
    pose_dir = Path(args.pose_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Find available codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        print("Cannot open VideoWriter — trying MJPG fallback")
        output_path = output_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    json_files = sorted(pose_dir.glob("*.json"))
    print(f"Video: {video_path.name}  ({w}x{h}, {total} frames)")
    print(f"Poses: {len(json_files)} JSON files")
    print(f"Output: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(json_files):
            with open(json_files[frame_idx], "r") as f:
                data = json.load(f)

            for person in data.get("people", []):
                kps_flat = person.get("pose_keypoints_2d", [])
                if not kps_flat:
                    continue
                kps = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)

                # Draw bones
                for (a, b) in BODY_25_BONES:
                    if a >= len(kps) or b >= len(kps):
                        continue
                    if kps[a, 2] < 0.05 or kps[b, 2] < 0.05:
                        continue
                    pt1 = (int(kps[a, 0]), int(kps[a, 1]))
                    pt2 = (int(kps[b, 0]), int(kps[b, 1]))
                    color = _bone_color_bgr(a, b)
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

                # Draw keypoints
                for k in range(len(kps)):
                    if kps[k, 2] < 0.05:
                        continue
                    pt = (int(kps[k, 0]), int(kps[k, 1]))
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(
            frame, f"Frame {frame_idx}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        out.write(frame)
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total} frames ...")

    cap.release()
    out.release()
    print(f"Done: {frame_idx} frames written to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D skeletons or annotate 2D video with OpenPose overlays"
    )
    sub = parser.add_subparsers(dest="command")

    # plot3d
    p3 = sub.add_parser("plot3d", help="3D skeleton animation")
    p3.add_argument("--skeleton", required=True, help="Path to skeleton_3d.npy")
    p3.add_argument("--output", default=None, help="Save MP4 (omit to show interactive)")
    p3.add_argument("--fps", type=float, default=30)
    p3.add_argument("--step", type=int, default=1, help="Frame step (e.g. 5 = every 5th)")
    p3.set_defaults(func=cmd_plot3d)

    # overlay
    po = sub.add_parser("overlay", help="2D skeleton overlay on video")
    po.add_argument("--video", required=True, help="Source video file")
    po.add_argument("--pose-dir", required=True, help="OpenPose JSON directory")
    po.add_argument("--output", required=True, help="Output annotated video")
    po.add_argument("--fps", type=float, default=None, help="FPS (default: from video)")
    po.set_defaults(func=cmd_overlay)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
