#!/usr/bin/env python3
"""
Create a layout video combining multi-camera 2D pose views with 3D skeleton.

Produces a single MP4 with a grid of camera feeds (raw or annotated) plus
a 3D skeleton view rendered from ``multicam_pose3d.py`` output.

Layout (default 3x2):
  ┌────────┬────────┬────────┐
  │ cam1   │ cam2   │ cam3   │
  ├────────┼────────┼────────┤
  │ cam4   │ cam5   │ 3D skel│
  └────────┴────────┴────────┘

The 3D panel is rendered frame-by-frame via matplotlib and composited
with the camera tiles by OpenCV (no ffmpeg filter_complex needed).

Usage
-----
    # Auto-discover annotated videos + 3D skeleton
    python tools/layout_video_3d.py \\
        --session new_data/ses-20260202_test \\
        --output new_data/ses-20260202_test/layout_3d.mp4

    # Specify skeleton file explicitly
    python tools/layout_video_3d.py \\
        --session new_data/ses-20260202_test \\
        --skeleton new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \\
        --output layout.mp4

    # Use raw camera videos instead of annotated
    python tools/layout_video_3d.py \\
        --session new_data/ses-20260202_test \\
        --raw-video \\
        --output layout.mp4

    # Limit to first 300 frames for quick test
    python tools/layout_video_3d.py \\
        --session new_data/ses-20260202_test \\
        --max-frames 300 \\
        --output layout_test.mp4

Dependencies: numpy, matplotlib, opencv-python
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenPose BODY_25 skeleton
# ---------------------------------------------------------------------------

BODY_25_BONES = [
    (0, 15), (0, 16), (15, 17), (16, 18),              # head
    (0, 1), (1, 8),                                      # spine
    (1, 2), (2, 3), (3, 4),                              # R arm
    (1, 5), (5, 6), (6, 7),                              # L arm
    (8, 9), (9, 10), (10, 11), (11, 22), (11, 24), (22, 23),  # R leg
    (8, 12), (12, 13), (13, 14), (14, 19), (14, 21), (19, 20),  # L leg
]

BONE_COLORS_RGB = {
    "head": (255, 215, 0),
    "spine": (255, 255, 255),
    "right_arm": (255, 68, 68),
    "left_arm": (68, 68, 255),
    "right_leg": (255, 136, 68),
    "left_leg": (68, 170, 255),
}

BONE_REGION = [
    "head", "head", "head", "head",
    "spine", "spine",
    "right_arm", "right_arm", "right_arm",
    "left_arm", "left_arm", "left_arm",
    "right_leg", "right_leg", "right_leg",
    "right_leg", "right_leg", "right_leg",
    "left_leg", "left_leg", "left_leg",
    "left_leg", "left_leg", "left_leg",
]

# Upper-body subset (head + spine + arms, no legs)
UPPER_BODY_KP = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18}

UPPER_BODY_BONES = [
    (0, 15), (0, 16), (15, 17), (16, 18),  # head
    (0, 1), (1, 8),                          # spine
    (1, 2), (2, 3), (3, 4),                  # R arm
    (1, 5), (5, 6), (6, 7),                  # L arm
]

UPPER_BODY_BONE_REGION = [
    "head", "head", "head", "head",
    "spine", "spine",
    "right_arm", "right_arm", "right_arm",
    "left_arm", "left_arm", "left_arm",
]

PERSON_COLORS = [
    "#00FF88", "#FF6644", "#4488FF", "#FFCC00",
]

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_annotated_videos(session_dir: Path) -> list[Path]:
    """Find MediaPipe-annotated MP4s (preferred) under mediapipe/."""
    mediapipe_dir = session_dir / "mediapipe"
    if mediapipe_dir.exists():
        vids = sorted(mediapipe_dir.glob("*_annotated.mp4"))
        if vids:
            return vids
    return []


def discover_raw_videos(session_dir: Path) -> list[Path]:
    """Find raw camera MKVs/MP4s under video/."""
    video_dir = session_dir / "video"
    if not video_dir.exists():
        return []
    from itertools import chain
    exts = chain(video_dir.glob("*.mkv"), video_dir.glob("*.mp4"))
    # Exclude event logs and calibration artifacts
    return sorted(
        p for p in exts
        if not p.name.startswith("ffmpeg") and "calibration" not in p.name
    )


def discover_skeleton(session_dir: Path) -> Path | None:
    """Find best skeleton .npy (prefer mediapipe variant)."""
    mp = session_dir / "skeleton_3d_mediapipe.npy"
    if mp.exists():
        return mp
    op = session_dir / "skeleton_3d.npy"
    if op.exists():
        return op
    return None


# ---------------------------------------------------------------------------
# 3D skeleton renderer (matplotlib, single lightweight figure)
# ---------------------------------------------------------------------------


class Skeleton3DRenderer:
    """Renders 3D skeleton frames to numpy images via matplotlib."""

    def __init__(
        self,
        data: np.ndarray,
        cell_w: int = 640,
        cell_h: int = 360,
        elevation: float = 20.0,
        azimuth: float = -60.0,
        upper_body: bool = False,
    ):
        self.data = data  # (frames, people, 25, 7)
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.elevation = elevation
        self.azimuth = azimuth
        self.upper_body = upper_body
        # Select bone set based on mode
        self.bones = UPPER_BODY_BONES if upper_body else BODY_25_BONES
        self.bone_regions = UPPER_BODY_BONE_REGION if upper_body else BONE_REGION
        self.kp_mask = UPPER_BODY_KP if upper_body else None

        # Compute global axis limits
        self.xlim, self.ylim, self.zlim = self._axis_limits()

        # Persistent figure
        dpi = 100
        self.fig = plt.figure(
            figsize=(cell_w / dpi, cell_h / dpi),
            dpi=dpi,
            facecolor="#1a1a2e",
        )
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
        self.fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0)

    def _axis_limits(self):
        if self.kp_mask is not None:
            kp_idx = sorted(self.kp_mask)
            sub = self.data[:, :, kp_idx, :]
        else:
            sub = self.data
        valid = ~np.isnan(sub[:, :, :, 0])
        xs = sub[:, :, :, 0][valid]
        ys = sub[:, :, :, 1][valid]
        zs = sub[:, :, :, 2][valid]
        if len(xs) == 0:
            return (-1, 1), (-1, 1), (-1, 1)
        pct_lo, pct_hi = 2, 98
        x_r = (np.percentile(xs, pct_lo), np.percentile(xs, pct_hi))
        y_r = (np.percentile(ys, pct_lo), np.percentile(ys, pct_hi))
        z_r = (np.percentile(zs, pct_lo), np.percentile(zs, pct_hi))
        max_range = max(x_r[1] - x_r[0], y_r[1] - y_r[0], z_r[1] - z_r[0]) * 0.6
        xm = sum(x_r) / 2
        ym = sum(y_r) / 2
        zm = sum(z_r) / 2
        return (xm - max_range, xm + max_range), (ym - max_range, ym + max_range), (zm - max_range, zm + max_range)

    def render(self, frame_idx: int) -> np.ndarray:
        """Return (cell_h, cell_w, 3) BGR image for one frame."""
        ax = self.ax
        ax.clear()
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_zlim(*self.zlim)
        ax.view_init(elev=self.elevation, azim=self.azimuth)
        ax.set_xlabel("X", fontsize=7, color="#888888")
        ax.set_ylabel("Y", fontsize=7, color="#888888")
        ax.set_zlabel("Z", fontsize=7, color="#888888")
        ax.tick_params(labelsize=5, colors="#666666")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.grid(True, alpha=0.2)

        if frame_idx < self.data.shape[0]:
            fd = self.data[frame_idx]
            n_people_vis = 0
            for pi in range(fd.shape[0]):
                if np.all(np.isnan(fd[pi, :, 0])):
                    continue
                n_people_vis += 1
                self._draw_skeleton(ax, fd[pi], pi)

            n_valid = int(np.sum(~np.isnan(fd[:, :, 0])))
            reproj_vals = fd[:, :, 4]
            vr = reproj_vals[(~np.isnan(reproj_vals)) & (reproj_vals >= 0)]
            rstr = f"{np.mean(vr):.1f}px" if len(vr) > 0 else "n/a"
            title = f"3D  F{frame_idx}  P:{n_people_vis} J:{n_valid} R:{rstr}"
        else:
            title = f"3D  F{frame_idx}  (no data)"

        self.fig.suptitle(title, fontsize=8, color="#FFFFFF", y=0.98)
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        # Resize to exact cell size
        img = cv2.resize(img, (self.cell_w, self.cell_h))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _draw_skeleton(self, ax, joints, person_idx):
        pc = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
        for bi, (ja, jb) in enumerate(self.bones):
            if ja >= len(joints) or jb >= len(joints):
                continue
            if np.isnan(joints[ja, 0]) or np.isnan(joints[jb, 0]):
                continue
            region = self.bone_regions[bi]
            c = "#{:02x}{:02x}{:02x}".format(*BONE_COLORS_RGB.get(region, (0, 255, 136)))
            ax.plot3D(
                [joints[ja, 0], joints[jb, 0]],
                [joints[ja, 1], joints[jb, 1]],
                [joints[ja, 2], joints[jb, 2]],
                color=c, linewidth=2, alpha=0.9,
            )
        # Scatter only the active keypoints
        if self.kp_mask is not None:
            kp_idx = sorted(self.kp_mask)
            sub = joints[kp_idx]
        else:
            sub = joints
        valid = ~np.isnan(sub[:, 0])
        if valid.any():
            ax.scatter3D(
                sub[valid, 0], sub[valid, 1], sub[valid, 2],
                c=pc, s=18, alpha=0.9, edgecolors="white",
                linewidths=0.5, depthshade=True,
            )

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Layout compositor
# ---------------------------------------------------------------------------


def create_layout_video(
    camera_videos: list[Path],
    skeleton_path: Path,
    output_path: Path,
    fps: float = 30.0,
    cell_w: int = 640,
    cell_h: int = 360,
    max_frames: int = 0,
    elevation: float = 20.0,
    azimuth: float = -60.0,
    flip_p20: bool = False,
    upper_body: bool = False,
    flip_cameras: list[str] | None = None,
) -> None:
    """
    Create a layout video with camera tiles + 3D skeleton panel.

    The grid auto-sizes: N cameras + 1 skeleton panel arranged in
    cols x rows where cols = ceil(sqrt(n_panels)).

    Args:
        camera_videos: list of camera video paths (raw or annotated)
        skeleton_path: path to skeleton .npy
        output_path: output .mp4
        fps: output frame rate
        cell_w, cell_h: per-cell resolution
        max_frames: limit (0 = min of all inputs)
        elevation, azimuth: 3D view angles
        flip_p20: shorthand to flip all "panacast_20" feeds 180°
        flip_cameras: list of substrings — any camera whose filename
            contains one of these strings will be flipped 180°.
            Takes precedence over flip_p20.
    """
    import math

    n_cams = len(camera_videos)
    n_panels = n_cams + 1  # cameras + 3D panel
    cols = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / cols)

    out_w = cols * cell_w
    out_h = rows * cell_h

    logger.info(f"Layout: {n_cams} cameras + 1 3D panel = {cols}x{rows} grid ({out_w}x{out_h})")

    # Open camera captures
    caps = []
    for vp in camera_videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            logger.warning(f"Cannot open {vp}, skipping")
            continue
        caps.append((vp.stem, cap))

    if not caps:
        logger.error("No camera videos could be opened")
        return

    # Load skeleton
    skeleton_data = np.load(skeleton_path, allow_pickle=False)
    logger.info(f"Skeleton: {skeleton_data.shape}")

    # Determine frame count
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for _, c in caps]
    frame_counts.append(skeleton_data.shape[0])
    total_frames = min(frame_counts)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    logger.info(f"Rendering {total_frames} frames @ {fps} fps")

    # Build unified flip-match list
    _flip_substrings: list[str] = list(flip_cameras or [])
    if flip_p20 and "panacast_20" not in _flip_substrings:
        _flip_substrings.append("panacast_20")

    if _flip_substrings:
        logger.info(f"Cameras matching {_flip_substrings} will be flipped 180°")
    if upper_body:
        logger.info("3D skeleton: upper-body only")

    # Init 3D renderer
    renderer = Skeleton3DRenderer(
        skeleton_data, cell_w=cell_w, cell_h=cell_h,
        elevation=elevation, azimuth=azimuth,
        upper_body=upper_body,
    )

    # Output writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        logger.error(f"Cannot open video writer: {output_path}")
        renderer.close()
        return

    # Label font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_bg = (0, 0, 0)

    for fi in range(total_frames):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Camera panels
        for ci, (label, cap) in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            else:
                # Flip cameras whose label matches any flip substring
                if _flip_substrings and any(s in label for s in _flip_substrings):
                    frame = cv2.flip(frame, -1)
                frame = cv2.resize(frame, (cell_w, cell_h))

            # Label overlay
            short_label = label.replace("jabra_panacast_20_", "").replace("jabra_panacast_50_", "P50_")
            short_label = short_label.replace("_vid_video", "")
            cv2.putText(frame, short_label, (8, 22), font, font_scale, font_bg, 3, cv2.LINE_AA)
            cv2.putText(frame, short_label, (8, 22), font, font_scale, font_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"F{fi}", (cell_w - 60, 22), font, font_scale, font_bg, 3, cv2.LINE_AA)
            cv2.putText(frame, f"F{fi}", (cell_w - 60, 22), font, font_scale, font_color, 1, cv2.LINE_AA)

            row = ci // cols
            col = ci % cols
            y0 = row * cell_h
            x0 = col * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = frame

        # 3D skeleton panel (fills last slot)
        skel_img = renderer.render(fi)
        skel_row = (n_cams) // cols
        skel_col = (n_cams) % cols
        y0 = skel_row * cell_h
        x0 = skel_col * cell_w
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = skel_img

        writer.write(canvas)

        if fi % max(1, total_frames // 20) == 0:
            pct = fi / total_frames * 100
            logger.info(f"  Frame {fi:>5}/{total_frames} ({pct:4.0f}%)")

    writer.release()
    for _, cap in caps:
        cap.release()
    renderer.close()

    mb = output_path.stat().st_size / (1024 * 1024)
    dur = total_frames / fps
    logger.info(f"Done: {output_path}  ({mb:.1f} MB, {dur:.1f}s, {total_frames} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layout video: multi-camera views + 3D skeleton panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session", type=Path, required=True,
        help="Session directory (e.g. new_data/ses-20260202_test)",
    )
    parser.add_argument(
        "--skeleton", type=Path, default=None,
        help="Skeleton .npy file (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .mp4 path (default: <session>/layout_3d.mp4)",
    )
    parser.add_argument(
        "--raw-video", action="store_true",
        help="Use raw camera videos instead of annotated 2D pose overlays",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=360)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit frames (0 = all)")
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument(
        "--flip-p20", action="store_true",
        help="Flip Panacast-20 camera feeds 180\u00b0 (mounted upside-down)",
    )
    parser.add_argument(        "--flip-cameras", nargs="+", metavar="SUBSTR",
        help="Flip camera feeds 180° whose filename contains any of "
             "these substrings (e.g. 'panacast_20 face_cam1'). "
             "More flexible than --flip-p20.",
    )
    parser.add_argument(        "--upper-body", action="store_true",
        help="Show only upper-body keypoints in 3D skeleton (head+spine+arms)",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    session = args.session
    if not session.exists():
        logger.error(f"Session directory not found: {session}")
        sys.exit(1)

    # Discover camera videos
    if args.raw_video:
        camera_videos = discover_raw_videos(session)
        logger.info(f"Using raw camera videos: {len(camera_videos)}")
    else:
        camera_videos = discover_annotated_videos(session)
        if not camera_videos:
            logger.info("No annotated videos found, falling back to raw")
            camera_videos = discover_raw_videos(session)
        else:
            logger.info(f"Using annotated videos: {len(camera_videos)}")

    if not camera_videos:
        logger.error("No camera videos found in session")
        sys.exit(1)

    for v in camera_videos:
        logger.info(f"  - {v.name}")

    # Skeleton
    skeleton_path = args.skeleton or discover_skeleton(session)
    if skeleton_path is None or not skeleton_path.exists():
        logger.error(f"Skeleton .npy not found (tried: {skeleton_path})")
        sys.exit(1)
    logger.info(f"Skeleton: {skeleton_path.name}")

    # Output
    output_path = args.output or (session / "layout_3d.mp4")

    create_layout_video(
        camera_videos=camera_videos,
        skeleton_path=skeleton_path,
        output_path=output_path,
        fps=args.fps,
        cell_w=args.cell_width,
        cell_h=args.cell_height,
        max_frames=args.max_frames,
        elevation=args.elevation,
        azimuth=args.azimuth,
        flip_p20=args.flip_p20,
        upper_body=args.upper_body,
        flip_cameras=args.flip_cameras,
    )


if __name__ == "__main__":
    main()
