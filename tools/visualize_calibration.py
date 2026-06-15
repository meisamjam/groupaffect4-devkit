#!/usr/bin/env python3
"""Visualize camera calibration geometry in 3D.

Creates an interactive 3D plot showing:
- Camera positions and orientations
- Inter-camera distances
- Field of view cones
- Quality indicators
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_toml_calibration(toml_path: Path) -> dict[str, Any]:
    """Load calibration TOML file."""
    with open(toml_path, encoding="utf-8") as f:
        return toml.load(f)


def rotation_matrix_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix using Rodrigues formula."""
    import cv2
    return cv2.Rodrigues(rvec)[0]


def plot_camera_rig(
    calib: dict[str, Any],
    output_path: Path | None = None,
    focal_specs: dict[str, float] | None = None,
) -> None:
    """Create 3D visualization of camera rig."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    cameras = []
    for cam_key, cam_data in calib.items():
        if not cam_key.startswith("cam_"):
            continue

        translation = np.array(cam_data["translation"])
        rotation = np.array(cam_data["rotation"])
        matrix = np.array(cam_data["matrix"])
        name = cam_data.get("name", cam_key)
        
        fx = matrix[0, 0]
        cameras.append({
            "name": name,
            "position": translation,
            "rotation": rotation,
            "fx": fx,
        })

    if not cameras:
        print("No cameras found in calibration file")
        return

    # Convert to meters for visualization
    positions = np.array([cam["position"] for cam in cameras]) / 1000.0

    # Plot camera positions
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="red",
        marker="o",
        s=100,
        label="Cameras",
    )

    # Add camera labels
    for i, cam in enumerate(cameras):
        pos = positions[i]
        label = cam["name"].replace("jabra_panacast_", "").replace("_vid_video", "")
        ax.text(pos[0], pos[1], pos[2], f"  {label}", fontsize=8)

    # Plot camera orientations (Z-axis of each camera)
    scale = 0.5  # Arrow length in meters
    for i, cam in enumerate(cameras):
        pos = positions[i]
        R = rotation_matrix_from_rvec(cam["rotation"])
        
        # Camera looks along Z-axis in camera coordinate frame
        # After rotation, world Z-axis direction is third column of R
        z_axis = R[:, 2]
        
        arrow = Arrow3D(
            [pos[0], pos[0] + scale * z_axis[0]],
            [pos[1], pos[1] + scale * z_axis[1]],
            [pos[2], pos[2] + scale * z_axis[2]],
            mutation_scale=20,
            lw=2,
            arrowstyle="->",
            color="blue",
            alpha=0.6,
        )
        ax.add_artist(arrow)

    # Draw lines between cameras
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            p1 = positions[i]
            p2 = positions[j]
            dist = np.linalg.norm(p1 - p2)
            
            # Color code by distance
            if dist < 0.5:
                color = "red"
                alpha = 0.8
                linewidth = 2
            elif dist > 5.0:
                color = "orange"
                alpha = 0.5
                linewidth = 1
            else:
                color = "gray"
                alpha = 0.3
                linewidth = 1
            
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
            
            # Add distance annotation
            mid = (p1 + p2) / 2
            ax.text(mid[0], mid[1], mid[2], f"{dist:.2f}m", fontsize=7, alpha=0.7)

    # Add coordinate system at origin
    origin = np.array([0, 0, 0])
    axis_scale = 0.8
    for i, (color, label) in enumerate([("r", "X"), ("g", "Y"), ("b", "Z")]):
        end = origin.copy()
        end[i] = axis_scale
        arrow = Arrow3D(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            mutation_scale=15,
            lw=2,
            arrowstyle="->",
            color=color,
        )
        ax.add_artist(arrow)
        ax.text(end[0] * 1.1, end[1] * 1.1, end[2] * 1.1, label, fontsize=10, color=color)

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Rig Geometry\n(Reference: cam_0 at origin)", fontsize=14)

    # Add legend with focal length info
    legend_text = "Camera Orientations (blue arrows)\n"
    legend_text += "Inter-camera distances:\n"
    legend_text += "  Red: < 0.5m (too close)\n"
    legend_text += "  Orange: > 5.0m (very far)\n"
    legend_text += "  Gray: good separation\n\n"
    legend_text += "Focal lengths (1080p):\n"
    for cam in cameras:
        label = cam["name"].replace("jabra_panacast_", "").replace("_vid_video", "")
        legend_text += f"  {label}: fx={cam['fx']:.0f}px"
        if focal_specs and cam["name"] in focal_specs:
            expected = focal_specs[cam["name"]]
            ratio = cam["fx"] / expected
            legend_text += f" ({ratio:.2f}x)"
        legend_text += "\n"
    
    ax.text2D(
        0.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        family="monospace",
    )

    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize camera calibration geometry in 3D"
    )
    parser.add_argument(
        "--toml",
        type=Path,
        required=True,
        help="Path to calibration .toml file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image file (default: show interactive plot)",
    )

    args = parser.parse_args()

    if not args.toml.exists():
        print(f"ERROR: Calibration TOML not found: {args.toml}")
        return 1

    print(f"Loading calibration from {args.toml}")
    calib = load_toml_calibration(args.toml)

    plot_camera_rig(calib, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
