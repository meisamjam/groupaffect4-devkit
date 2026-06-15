#!/usr/bin/env python3
"""Visualize a short segment of world-gaze output to sanity-check alignment.

Produces a single figure with three panels:
  1. Desk top-view scatter: gaze landing points in world X/Y, colour-coded by
     participant; desk outline and marker positions overlaid.
  2. Gaze timeseries: world_x and world_y over time for each participant.
  3. Annotated scene-video frame: one representative frame from each scene
     video with the ArUco marker detections and the gaze circle drawn.

Usage::

    python tools/visualize_gaze_world_sample.py \\
        --gaze-csvs \\
            "F:/.../gaze_world_T1/tobii_P2_gaze_world.csv" \\
            "F:/.../gaze_world_T1/tobii_P4_gaze_world.csv" \\
        --scene-videos \\
            "F:/.../et/sub-01_..._task-T1_run-01_acq-P2_tobii.mp4" \\
            "F:/.../et/sub-01_..._task-T1_run-01_acq-P4_tobii.mp4" \\
        --config configs/tobii_offline_world_align_grp12_T1.yaml \\
        --output  "F:/.../gaze_world_T1/sanity_check.png" \\
        --t-start 60 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ── matplotlib import (non-interactive backend for headless render) ──────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── colour palette ────────────────────────────────────────────────────────────
_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]  # P1-P4


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_gaze_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({
                    "sample_time_s": float(row["sample_time_s"]),
                    "frame_idx":     int(row["frame_idx"]),
                    "world_x":       float(row["world_x"]),
                    "world_y":       float(row["world_y"]),
                    "world_z":       float(row["world_z"]),
                    "gaze_x":        float(row["gaze_x"]),
                    "gaze_y":        float(row["gaze_y"]),
                    "marker_count":  int(row["marker_count"]),
                    "reproj_error_px": float(row["reproj_error_px"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def _filter_time(rows: list[dict], t_start: float, duration: float) -> list[dict]:
    """Keep rows whose sample_time_s falls in [t_start, t_start+duration).

    If sample_time_s is small (< 1000 s) assume it's already relative.
    Otherwise normalise to the first observed time.
    """
    if not rows:
        return rows
    t0 = rows[0]["sample_time_s"]
    # If times look like LSL absolute (> 1000 s) make them relative
    if t0 > 1000:
        for r in rows:
            r["sample_time_s"] = r["sample_time_s"] - t0
    return [r for r in rows if t_start <= r["sample_time_s"] < t_start + duration]


def _load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _desk_outline(cfg: dict) -> tuple[float, float, float, float]:
    """Return (half_w, half_d) in metres from config desk block."""
    desk = cfg.get("desk", {})
    w = float(desk.get("width_m", 1.8)) / 2
    d = float(desk.get("depth_m", 0.8)) / 2
    return w, d


def _marker_centres(cfg: dict) -> list[tuple[float, float, int]]:
    """Return list of (x, y, id) for marker map centres."""
    result = []
    for m in cfg.get("world", {}).get("marker_map", []):
        corners = m.get("corners_m", [])
        if len(corners) != 4:
            continue
        cx = float(np.mean([c[0] for c in corners]))
        cy = float(np.mean([c[1] for c in corners]))
        result.append((cx, cy, int(m["id"])))
    return result


def _annotate_frame(
    video_path: Path,
    frame_idx: int,
    gaze_x_norm: float,
    gaze_y_norm: float,
    aruco_dict_name: str = "DICT_4X4_50",
    colour: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray | None:
    """Grab *frame_idx* from video, draw ArUco detections + gaze circle."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    h, w = frame.shape[:2]

    # Draw ArUco detections
    try:
        aruco_dict_id = getattr(cv2.aruco, aruco_dict_name, cv2.aruco.DICT_4X4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 200, 0))
    except Exception:
        pass

    # Draw gaze circle (gaze_x/y are normalised 0-1)
    gx = int(gaze_x_norm * w)
    gy = int(gaze_y_norm * h)
    cv2.circle(frame, (gx, gy), 30, colour, 3)
    cv2.circle(frame, (gx, gy), 5, colour, -1)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── main plot ─────────────────────────────────────────────────────────────────

def make_figure(
    gaze_csvs: list[Path],
    scene_videos: list[Path],
    config_path: Path,
    output: Path,
    t_start: float,
    duration: float,
    min_markers: int = 1,
    max_reproj_px: float = 20.0,
) -> None:
    cfg = _load_config(config_path)
    half_w, half_d = _desk_outline(cfg)
    marker_centres = _marker_centres(cfg)
    aruco_dict_name = cfg.get("world", {}).get("aruco_dictionary", "DICT_4X4_50")

    # Load and filter gaze data
    datasets: list[tuple[str, list[dict]]] = []
    for path in gaze_csvs:
        rows = _load_gaze_csv(path)
        pid = path.stem.split("_")[0]  # e.g. "tobii" → strip further below
        # pid from filename like "tobii_P2_gaze_world" → take "P2"
        parts = path.stem.split("_")
        pid = next((p for p in parts if p.startswith("P") and p[1:].isdigit()), parts[0])
        filtered = _filter_time(rows, t_start, duration)
        # Quality filter: drop rows with too few markers or high reproj error
        n_before = len(filtered)
        filtered = [r for r in filtered
                    if r["marker_count"] >= min_markers
                    and r["reproj_error_px"] <= max_reproj_px]
        n_dropped = n_before - len(filtered)
        if n_dropped:
            print(f"  {pid}: dropped {n_dropped}/{n_before} samples "
                  f"(min_markers={min_markers}, max_reproj={max_reproj_px}px)")
        datasets.append((pid, filtered))

    n_videos = min(len(scene_videos), 2)  # show at most 2 scene frames

    # Figure layout: top row = scatter + timeseries; bottom row = scene frames + reproj hist
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(2, max(3, n_videos + 1), figure=fig,
                  height_ratios=[1.3, 1], hspace=0.4, wspace=0.3)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_ts      = fig.add_subplot(gs[0, 1:])

    # ── Panel 1: desk top-view scatter ────────────────────────────────────────
    desk_rect = mpatches.Rectangle(
        (-half_w, -half_d), 2 * half_w, 2 * half_d,
        linewidth=1.5, edgecolor="#555", facecolor="#f9f9e8", zorder=0,
    )
    ax_scatter.add_patch(desk_rect)

    for (cx, cy, mid) in marker_centres:
        ax_scatter.plot(cx, cy, "ks", markersize=8, zorder=2)
        ax_scatter.text(cx + 0.03, cy + 0.03, str(mid), fontsize=7, color="#333")

    for i, (pid, rows) in enumerate(datasets):
        colour = _COLOURS[i % len(_COLOURS)]
        wx = [r["world_x"] for r in rows]
        wy = [r["world_y"] for r in rows]
        # Alpha driven by marker count: more markers = more opaque
        alphas = np.clip([r["marker_count"] / 6 for r in rows], 0.05, 0.8)
        ax_scatter.scatter(wx, wy, s=4, alpha=alphas.mean(), color=colour, label=pid, zorder=3)

    # Indicate desk extent lines
    ax_scatter.set_xlim(-half_w * 1.3, half_w * 1.3)
    ax_scatter.set_ylim(-half_d * 2.5, half_d * 2.5)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("World X (m)  ← left | right →")
    ax_scatter.set_ylabel("World Y (m)  front ↓ | back ↑")
    ax_scatter.set_title(f"Gaze landing points (desk top view)\nt={t_start}–{t_start+duration}s")
    ax_scatter.legend(loc="upper right", markerscale=3, fontsize=8)
    ax_scatter.grid(True, alpha=0.3)

    # ── Panel 2: gaze timeseries ──────────────────────────────────────────────
    for i, (pid, rows) in enumerate(datasets):
        colour = _COLOURS[i % len(_COLOURS)]
        ts = [r["sample_time_s"] for r in rows]
        wx = [r["world_x"] for r in rows]
        wy = [r["world_y"] for r in rows]
        ax_ts.plot(ts, wx, color=colour, lw=0.8, alpha=0.8, label=f"{pid} X")
        ax_ts.plot(ts, wy, color=colour, lw=0.8, alpha=0.4, ls="--", label=f"{pid} Y")

    ax_ts.axhline(0, color="#aaa", lw=0.8, ls=":")
    ax_ts.axhline(-half_d, color="#aaa", lw=0.5, ls=":")
    ax_ts.axhline(half_d, color="#aaa", lw=0.5, ls=":")
    ax_ts.set_xlabel("Time (s, relative)")
    ax_ts.set_ylabel("World coordinate (m)")
    ax_ts.set_title("World-gaze X (solid) and Y (dashed) timeseries")
    ax_ts.legend(fontsize=7, ncol=4, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # ── Bottom row: annotated scene frames ────────────────────────────────────
    scene_video_map: dict[str, Path] = {}
    for sv in scene_videos:
        # Extract participant ID from BIDS filename (acq-P2 pattern) or from stem
        import re
        m = re.search(r"acq-([^_]+)_tobii", str(sv))
        if m:
            scene_video_map[m.group(1)] = sv
        else:
            scene_video_map[sv.stem] = sv

    for col, (pid, rows) in enumerate(datasets[:n_videos]):
        ax_frame = fig.add_subplot(gs[1, col])
        sv = scene_video_map.get(pid)
        if sv is None or not sv.exists() or not rows:
            ax_frame.text(0.5, 0.5, f"No scene video\nfor {pid}",
                          ha="center", va="center", transform=ax_frame.transAxes)
            ax_frame.axis("off")
            continue

    # Pick best scene frame = row with most markers (tie-break: lowest reproj error)
        if sv is None or not sv.exists() or not rows:
            ax_frame.text(0.5, 0.5, f"No scene video\nfor {pid}",
                          ha="center", va="center", transform=ax_frame.transAxes)
            ax_frame.axis("off")
            continue

        best = max(rows, key=lambda r: (r["marker_count"], -r["reproj_error_px"]))
        frame_img = _annotate_frame(
            sv, best["frame_idx"],
            best["gaze_x"], best["gaze_y"],
            aruco_dict_name=aruco_dict_name,
            colour=(255, 50, 50),
        )
        if frame_img is None:
            ax_frame.text(0.5, 0.5, f"Frame read failed\n(frame {best['frame_idx']})",
                          ha="center", va="center", transform=ax_frame.transAxes)
            ax_frame.axis("off")
        else:
            ax_frame.imshow(frame_img)
            ax_frame.set_title(
                f"{pid} — frame {best['frame_idx']}\n"
                f"markers={best['marker_count']}  reproj={best['reproj_error_px']:.1f}px",
                fontsize=8,
            )
            ax_frame.axis("off")

    # ── Bottom-right: reproj error histogram ─────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, n_videos])
    for i, (pid, rows) in enumerate(datasets):
        colour = _COLOURS[i % len(_COLOURS)]
        errs = [r["reproj_error_px"] for r in rows]
        if errs:
            cap = min(np.percentile(errs, 98), 200)
            ax_hist.hist([min(e, cap) for e in errs], bins=40,
                         color=colour, alpha=0.5, label=pid, density=True)
    ax_hist.set_xlabel("Reproj error (px, capped at p98)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("PnP reprojection error\ndistribution")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        f"World-gaze sanity check | grp-12 T1 | t={t_start}–{t_start+duration}s",
        fontsize=13, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick sanity-check figure for world-gaze alignment output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gaze-csvs", nargs="+", required=True, type=Path,
                   metavar="CSV", help="One or more gaze_world.csv files")
    p.add_argument("--scene-videos", nargs="*", default=[], type=Path,
                   metavar="MP4", help="Tobii scene videos (optional, for frame panel)")
    p.add_argument("--config", required=True, type=Path,
                   help="World-align YAML config (for desk/marker geometry)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output PNG path")
    p.add_argument("--t-start", type=float, default=60.0,
                   help="Start time in seconds (relative, default 60)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Duration in seconds (default 30)")
    p.add_argument("--min-markers", type=int, default=1,
                   help="Minimum marker_count to include a sample (default 1; use 2+ for stricter QC)")
    p.add_argument("--max-reproj-px", type=float, default=20.0,
                   help="Maximum reprojection error (px) to include a sample (default 20)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    missing = [p for p in args.gaze_csvs if not p.exists()]
    if missing:
        print(f"ERROR: files not found: {missing}", file=sys.stderr)
        return 1

    make_figure(
        gaze_csvs=args.gaze_csvs,
        scene_videos=args.scene_videos,
        config_path=args.config,
        output=args.output,
        t_start=args.t_start,
        duration=args.duration,
        min_markers=args.min_markers,
        max_reproj_px=args.max_reproj_px,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
