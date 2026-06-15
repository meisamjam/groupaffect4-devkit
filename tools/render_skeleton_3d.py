#!/usr/bin/env python3
"""
Render a 3D skeleton video from multicam_pose3d.py output.

Produces an MP4 video showing animated 3D skeletons from multiple
viewpoints using matplotlib. Optionally overlays a side-by-side
camera view for visual verification.

Usage
-----
    # Simple 3D-only render (two views)
    python tools/render_skeleton_3d.py \
        --skeleton new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \
        --output new_data/ses-20260202_test/skeleton_3d_video.mp4

    # With camera overlay
    python tools/render_skeleton_3d.py \
        --skeleton new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \
        --camera-video new_data/ses-20260202_test/video/jabra_panacast_20_cam2_vid_video.mkv \
        --output new_data/ses-20260202_test/skeleton_3d_video.mp4

    # Custom view angles and FPS
    python tools/render_skeleton_3d.py \
        --skeleton new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \
        --output skeleton.mp4 --fps 15 --elevation 25 --azimuth -60

Dependencies: numpy, matplotlib, opencv-python
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for rendering
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — imported for 3D projection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenPose BODY_25 skeleton definition
# ---------------------------------------------------------------------------

BODY_25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]

# Bone connections as (joint_a, joint_b)
BODY_25_BONES = [
    # Head
    (0, 15), (0, 16), (15, 17), (16, 18),
    # Spine
    (0, 1), (1, 8),
    # Right arm
    (1, 2), (2, 3), (3, 4),
    # Left arm
    (1, 5), (5, 6), (6, 7),
    # Right leg
    (8, 9), (9, 10), (10, 11),
    (11, 22), (11, 24), (22, 23),
    # Left leg
    (8, 12), (12, 13), (13, 14),
    (14, 19), (14, 21), (19, 20),
]

# Color scheme per body region
BONE_COLORS = {
    "head": "#FFD700",      # gold
    "spine": "#FFFFFF",     # white
    "right_arm": "#FF4444", # red
    "left_arm": "#4444FF",  # blue
    "right_leg": "#FF8844", # orange
    "left_leg": "#44AAFF",  # light blue
}

BONE_REGION = [
    "head", "head", "head", "head",  # head bones
    "spine", "spine",                 # spine
    "right_arm", "right_arm", "right_arm",  # R arm
    "left_arm", "left_arm", "left_arm",     # L arm
    "right_leg", "right_leg", "right_leg",  # R leg
    "right_leg", "right_leg", "right_leg",  # R foot
    "left_leg", "left_leg", "left_leg",     # L leg
    "left_leg", "left_leg", "left_leg",     # L foot
]

# Person colors for multi-person rendering
PERSON_COLORS = [
    ("#00FF88", "#00CC66"),  # green
    ("#FF6644", "#CC4422"),  # red-orange
    ("#4488FF", "#2266CC"),  # blue
    ("#FFCC00", "#CC9900"),  # yellow
]


def compute_axis_limits(data: np.ndarray) -> tuple[tuple, tuple, tuple]:
    """Compute axis limits from all valid 3D points with padding."""
    valid = ~np.isnan(data[:, :, :, 0])
    xs = data[:, :, :, 0][valid]
    ys = data[:, :, :, 1][valid]
    zs = data[:, :, :, 2][valid]

    if len(xs) == 0:
        return (-1, 1), (-1, 1), (-1, 1)

    # Use percentile to exclude outliers
    pct_lo, pct_hi = 2, 98
    x_range = (np.percentile(xs, pct_lo), np.percentile(xs, pct_hi))
    y_range = (np.percentile(ys, pct_lo), np.percentile(ys, pct_hi))
    z_range = (np.percentile(zs, pct_lo), np.percentile(zs, pct_hi))

    # Make axes equal-scale
    ranges = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
    max_range = max(ranges) * 0.6  # add padding

    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2
    z_mid = (z_range[0] + z_range[1]) / 2

    return (
        (x_mid - max_range, x_mid + max_range),
        (y_mid - max_range, y_mid + max_range),
        (z_mid - max_range, z_mid + max_range),
    )


def draw_skeleton_3d(
    ax: plt.Axes,
    joints: np.ndarray,
    person_idx: int = 0,
    alpha: float = 1.0,
) -> None:
    """
    Draw a single skeleton on a 3D axes.

    Args:
        ax: matplotlib 3D axes
        joints: (25, 7) array [x, y, z, conf, reproj, n_cams, group]
        person_idx: index for color selection
        alpha: transparency
    """
    pcolor = PERSON_COLORS[person_idx % len(PERSON_COLORS)]

    # Draw bones
    for bi, (ja, jb) in enumerate(BODY_25_BONES):
        if ja >= len(joints) or jb >= len(joints):
            continue
        if np.isnan(joints[ja, 0]) or np.isnan(joints[jb, 0]):
            continue

        region = BONE_REGION[bi]
        color = BONE_COLORS.get(region, pcolor[0])
        ax.plot3D(
            [joints[ja, 0], joints[jb, 0]],
            [joints[ja, 1], joints[jb, 1]],
            [joints[ja, 2], joints[jb, 2]],
            color=color, linewidth=2.5, alpha=alpha,
        )

    # Draw joints
    valid = ~np.isnan(joints[:, 0])
    if valid.any():
        ax.scatter3D(
            joints[valid, 0], joints[valid, 1], joints[valid, 2],
            c=pcolor[0], s=25, alpha=alpha, edgecolors=pcolor[1],
            linewidths=0.8, depthshade=True,
        )


def render_frame(
    fig: plt.Figure,
    axes: list[plt.Axes],
    frame_data: np.ndarray,
    frame_idx: int,
    n_frames: int,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    views: list[tuple[float, float]],
    camera_frame: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render a single frame to a numpy image array.

    Args:
        fig: matplotlib figure
        axes: list of 3D axes (one per view)
        frame_data: (n_people, 25, 7) skeleton data for this frame
        frame_idx: current frame number
        n_frames: total frames
        xlim, ylim, zlim: axis limits
        views: list of (elevation, azimuth) tuples
        camera_frame: optional camera image to show in a panel

    Returns:
        (H, W, 3) uint8 image
    """
    for ax_idx, ax in enumerate(axes):
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=views[ax_idx][0], azim=views[ax_idx][1])
        ax.set_xlabel("X", fontsize=8, color="#888888")
        ax.set_ylabel("Y", fontsize=8, color="#888888")
        ax.set_zlabel("Z", fontsize=8, color="#888888")
        ax.tick_params(labelsize=6, colors="#666666")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.grid(True, alpha=0.2)

        # Draw all people
        n_people = frame_data.shape[0]
        for pi in range(n_people):
            if np.all(np.isnan(frame_data[pi, :, 0])):
                continue
            draw_skeleton_3d(ax, frame_data[pi], person_idx=pi)

        view_label = f"View {ax_idx+1} (elev={views[ax_idx][0]}°, az={views[ax_idx][1]}°)"
        ax.set_title(view_label, fontsize=9, color="#CCCCCC", pad=2)

    # Frame info text
    n_valid_people = sum(
        1 for pi in range(frame_data.shape[0])
        if not np.all(np.isnan(frame_data[pi, :, 0]))
    )
    n_valid_joints = int(np.sum(~np.isnan(frame_data[:, :, 0])))

    # Get reproj stats for valid joints
    reproj_vals = frame_data[:, :, 4]
    valid_reproj = reproj_vals[(~np.isnan(reproj_vals)) & (reproj_vals >= 0)]
    reproj_str = f"{np.mean(valid_reproj):.1f}px" if len(valid_reproj) > 0 else "n/a"

    info = (
        f"Frame {frame_idx}/{n_frames}  |  "
        f"People: {n_valid_people}  |  "
        f"Joints: {n_valid_joints}  |  "
        f"Reproj: {reproj_str}"
    )
    fig.suptitle(info, fontsize=10, color="#FFFFFF", y=0.98)

    # Render to image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()  # drop alpha, RGBA → RGB

    # If camera overlay, composite it
    if camera_frame is not None:
        h, w = img.shape[:2]
        cam_h = h // 4  # camera view takes bottom quarter
        cam_w = int(camera_frame.shape[1] * cam_h / camera_frame.shape[0])
        cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        # Place at bottom center
        x_off = (w - cam_w) // 2
        if x_off >= 0 and x_off + cam_w <= w:
            img[h - cam_h:h, x_off:x_off + cam_w] = cam_resized

    return img


def render_video(
    skeleton_path: Path,
    output_path: Path,
    camera_video_path: Path | None = None,
    fps: float = 30.0,
    elevation: float = 20.0,
    azimuth: float = -60.0,
    width: int = 1920,
    height: int = 1080,
    max_frames: int = 0,
    person_filter: int = -1,
) -> None:
    """
    Render 3D skeleton video.

    Args:
        skeleton_path: path to .npy from multicam_pose3d.py
        output_path: output .mp4 path
        camera_video_path: optional camera video for overlay
        fps: output frame rate
        elevation: default camera elevation angle
        azimuth: default camera azimuth angle
        width, height: output resolution
        max_frames: limit frames (0 = all)
        person_filter: render only this person (-1 = all)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    data = np.load(skeleton_path, allow_pickle=False)
    logger.info(f"Loaded skeleton: {data.shape}  (frames, people, keypoints, dims)")
    n_frames = data.shape[0]

    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
        data = data[:n_frames]

    if person_filter >= 0:
        data = data[:, person_filter:person_filter + 1, :, :]
        logger.info(f"Filtering to person {person_filter}")

    # Compute axis limits from all data
    xlim, ylim, zlim = compute_axis_limits(data)
    logger.info(f"Axis limits: X={xlim}, Y={ylim}, Z={zlim}")

    # Camera video (if provided)
    cam_cap = None
    if camera_video_path and camera_video_path.exists():
        cam_cap = cv2.VideoCapture(str(camera_video_path))
        logger.info(f"Camera overlay: {camera_video_path.name}")

    # Two views: front-ish and top-down
    views = [
        (elevation, azimuth),           # primary view
        (elevation, azimuth + 90),      # rotated 90°
    ]

    # Setup matplotlib figure
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#1a1a2e")

    if cam_cap:
        # 2 skeleton views on top, camera at bottom
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#1a1a2e")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#1a1a2e")
    else:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#1a1a2e")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#1a1a2e")

    axes = [ax1, ax2]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.05)

    # Setup video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error(f"Failed to open video writer: {output_path}")
        sys.exit(1)

    logger.info(f"Rendering {n_frames} frames @ {fps} fps → {output_path}")
    logger.info(f"Resolution: {width}x{height}")

    for fi in range(n_frames):
        # Camera frame
        cam_frame = None
        if cam_cap:
            ret, raw = cam_cap.read()
            if ret:
                cam_frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # Render
        img = render_frame(
            fig, axes, data[fi], fi, n_frames,
            xlim, ylim, zlim, views, cam_frame,
        )

        # Convert RGB → BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        if fi % max(1, n_frames // 20) == 0:
            pct = fi / n_frames * 100
            logger.info(f"  Frame {fi:>5}/{n_frames} ({pct:4.0f}%)")

    writer.release()
    if cam_cap:
        cam_cap.release()
    plt.close(fig)

    file_mb = output_path.stat().st_size / (1024 * 1024)
    duration_s = n_frames / fps
    logger.info(f"Done: {output_path}  ({file_mb:.1f} MB, {duration_s:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 3D skeleton video from multicam_pose3d.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skeleton", type=Path, required=True,
        help="Path to skeleton .npy file",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output .mp4 video path",
    )
    parser.add_argument(
        "--camera-video", type=Path, default=None,
        help="Camera video for overlay (optional)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Camera elevation angle (degrees)")
    parser.add_argument("--azimuth", type=float, default=-60.0,
                        help="Camera azimuth angle (degrees)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit frames (0 = all)")
    parser.add_argument("--person", type=int, default=-1,
                        help="Render only this person index (-1 = all)")

    args = parser.parse_args()

    render_video(
        skeleton_path=args.skeleton,
        output_path=args.output,
        camera_video_path=args.camera_video,
        fps=args.fps,
        elevation=args.elevation,
        azimuth=args.azimuth,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        person_filter=args.person,
    )


if __name__ == "__main__":
    main()
