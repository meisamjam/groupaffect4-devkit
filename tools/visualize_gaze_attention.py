#!/usr/bin/env python3
"""Visualize world-gaze with interest-area zones and shared-attention analysis.

Produces a multi-panel figure:
  1. Desk top-view scatter: gaze landing points in world X/Y, overlaid with
     interest-area zone patches (P1–P4 person zones, screen, moderator, desk).
     Points are coloured by the zone they land in.
  2. Per-participant gaze-zone dwell heatmap over time (5-second bins).
  3. Shared-attention timeline: time bins where ≥2 participants look at the
     same zone simultaneously.
  4. Zone-dwell summary bar chart (% time per zone per participant).

Zone centres default to seat positions from desk_markers_large.yaml
(P1=back_right, P2=front_right, P3=front_left, P4=back_left) and can be
overridden with --zone-config.

If --skeleton NPY is supplied, person-zone centres update per frame from
BODY_25 keypoints (0=nose for head, 4/7=wrists for hands).

Usage::

    python tools/visualize_gaze_attention.py \\
        --gaze-csvs \\
            "F:/.../gaze_world_T1/tobii_P1_gaze_world.csv" \\
            "F:/.../gaze_world_T1/tobii_P2_gaze_world.csv" \\
            "F:/.../gaze_world_T1/tobii_P3_gaze_world.csv" \\
            "F:/.../gaze_world_T1/tobii_P4_gaze_world.csv" \\
        --config configs/tobii_offline_world_align_grp13_T1.yaml \\
        --output "F:/.../gaze_world_T1/attention_zones.png" \\
        --t-start 0 --duration 300 \\
        --min-markers 1 --max-reproj-px 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


# ── Colour palettes ───────────────────────────────────────────────────────────
# Per-participant colours (P1–P4)
_PART_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

# Zone colours (named, semi-transparent fills)
_ZONE_PALETTE = {
    "desk":       "#d9d9d9",
    "P1":         "#fbb4ae",
    "P2":         "#b3cde3",
    "P3":         "#ccebc5",
    "P4":         "#fed9a6",
    "screen":     "#decbe4",
    "moderator":  "#ffffcc",
    "other":      "#f0f0f0",
}


# ── Camera positions in world frame (metres) ──────────────────────────────────
# Source: docs/camera_layout_and_positions.md (derived from calibration TOML +
# ChArUco board origin + physical mounting measurements).
#
# World frame: origin = desk-centre ChArUco board, x = right, y = back, z = up.
#
# | Camera | x (m) | y (m) | z (m) | Focus       |
# |--------|-------|-------|-------|-------------|
# | cam1   | −0.90 | −0.20 | 0.88  | P1, P2      |
# | cam2   | +0.90 | −0.20 | 0.88  | P3, P4      |
# | cam3   | +0.90 | +0.20 | 0.88  | P3, P4      |
# | cam4   | −0.90 | +0.20 | 0.88  | P1, P2      |
# | cam5   | −0.90 | −0.40 | 0.00  | table ovrvw |
# | cam6   |  0.00 | +0.40 | 0.88  | all         |
# | P50    |  0.00 | −0.40 | 0.90  | all (wide)  |
#
_CAMERA_WORLD_POS: dict[str, list[float]] = {
    "cam1": [-0.900, -0.200, 0.880],
    "cam2": [+0.900, -0.200, 0.880],
    "cam3": [+0.900, +0.200, 0.880],
    "cam4": [-0.900, +0.200, 0.880],
    "cam5": [-0.900, -0.400, 0.000],
    "cam6": [+0.000, +0.400, 0.880],
    "P50":  [+0.000, -0.400, 0.900],
}

# Camera → participant zone mapping (from calibration layout):
#   cam1+cam4 on LEFT side (x ≈ −0.9) look RIGHT → see P1, P2 on the right side
#   cam2+cam3 on RIGHT side (x ≈ +0.9) look LEFT → see P3, P4 on the left side
_CAM_PARTICIPANT_ZONES: dict[str, list[str]] = {
    "cam1": ["P1", "P2"],
    "cam4": ["P1", "P2"],
    "cam2": ["P3", "P4"],
    "cam3": ["P3", "P4"],
}

# Tobii glasses → participant mapping (from device config):
_GLASSES_PARTICIPANT: dict[str, str] = {
    "tobii_P1": "P1",
    "tobii_P2": "P2",
    "tobii_P3": "P3",
    "tobii_P4": "P4",
}


def _build_zones_from_config(cfg: dict) -> dict[str, dict[str, Any]]:
    """Compute interest-area zones from config data (marker map + desk geometry).

    Sources used:
    - Desk extents: from ArUco marker_map corner positions in the config YAML
    - Participant positions: inferred from desk extents + camera-participant zone
      assignment (participants sit ~0.30 m outside the desk long edges)
    - Screen: between P50 and cam5 world positions (front short end)
    - Moderator: back short end (behind cam6)

    All coordinates in world frame: x = right, y = back, z = up.
    """
    # ── 1. Desk extents from marker map ───────────────────────────────────────
    markers = cfg.get("world", {}).get("marker_map", [])
    if markers:
        all_x = [c[0] for m in markers for c in m["corners_m"]]
        all_y = [c[1] for m in markers for c in m["corners_m"]]
        desk_xmin, desk_xmax = min(all_x), max(all_x)
        desk_ymin, desk_ymax = min(all_y), max(all_y)
    else:
        desk_block = cfg.get("desk", {})
        half_w = float(desk_block.get("width_m", 1.80)) / 2
        half_d = float(desk_block.get("depth_m", 0.80)) / 2
        desk_xmin, desk_xmax = -half_w, +half_w
        desk_ymin, desk_ymax = -half_d, +half_d

    desk_cx = (desk_xmin + desk_xmax) / 2
    desk_cy = (desk_ymin + desk_ymax) / 2

    # ── 2. Participant positions ──────────────────────────────────────────────
    # The desk is wider in x (long sides at x ≈ ±0.925) than in y (short ends
    # at y ≈ ±0.425).  Participants sit ~0.30 m outside the long edges:
    #   RIGHT side (x+): P1 back-right, P2 front-right
    #   LEFT  side (x−): P3 front-left,  P4 back-left
    seat_offset = 0.30
    y_back  = desk_cy + (desk_ymax - desk_cy) * 0.5   # +0.21
    y_front = desk_cy - (desk_cy - desk_ymin) * 0.5   # −0.21

    p1_xy = [desk_xmax + seat_offset, y_back]   # back-right
    p2_xy = [desk_xmax + seat_offset, y_front]  # front-right
    p3_xy = [desk_xmin - seat_offset, y_front]  # front-left
    p4_xy = [desk_xmin - seat_offset, y_back]   # back-left

    # ── 3. Screen zone — between P50 and cam5 (front side, y < desk_ymin) ────
    p50 = _CAMERA_WORLD_POS["P50"]
    c5  = _CAMERA_WORLD_POS["cam5"]
    screen_cx = (p50[0] + c5[0]) / 2           # x midpoint
    screen_half_w = abs(p50[0] - c5[0]) / 2 + 0.15  # extend slightly
    screen_y_inner = desk_ymin - 0.05
    screen_y_outer = desk_ymin - 0.80

    # ── 4. Moderator zone — back side, behind cam6 (y > desk_ymax) ───────────
    c6 = _CAMERA_WORLD_POS["cam6"]
    mod_cx = c6[0]
    mod_half_w = 0.40
    mod_y_inner = desk_ymax + 0.05
    mod_y_outer = desk_ymax + 0.80

    return {
        "desk": {
            "type": "rect",
            "x": [desk_xmin, desk_xmax],
            "y": [desk_ymin, desk_ymax],
            "priority": 0,
        },
        "P1": {
            "type": "circle",
            "center": p1_xy,
            "radius": 0.25,
            "priority": 1,
            "label": "P1 (back-right)",
        },
        "P2": {
            "type": "circle",
            "center": p2_xy,
            "radius": 0.25,
            "priority": 1,
            "label": "P2 (front-right)",
        },
        "P3": {
            "type": "circle",
            "center": p3_xy,
            "radius": 0.25,
            "priority": 1,
            "label": "P3 (front-left)",
        },
        "P4": {
            "type": "circle",
            "center": p4_xy,
            "radius": 0.25,
            "priority": 1,
            "label": "P4 (back-left)",
        },
        "screen": {
            "type": "rect",
            "x": [screen_cx - screen_half_w, screen_cx + screen_half_w],
            "y": [min(screen_y_inner, screen_y_outer),
                  max(screen_y_inner, screen_y_outer)],
            "priority": 2,
            "label": "Screen",
        },
        "moderator": {
            "type": "rect",
            "x": [mod_cx - mod_half_w, mod_cx + mod_half_w],
            "y": [min(mod_y_inner, mod_y_outer),
                  max(mod_y_inner, mod_y_outer)],
            "priority": 2,
            "label": "Moderator",
        },
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_gaze_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({
                    "sample_time_s":   float(row["sample_time_s"]),
                    "frame_idx":       int(row["frame_idx"]),
                    "world_x":         float(row["world_x"]),
                    "world_y":         float(row["world_y"]),
                    "world_z":         float(row.get("world_z", 0.0)),
                    "gaze_x":          float(row["gaze_x"]),
                    "gaze_y":          float(row["gaze_y"]),
                    "marker_count":    int(row["marker_count"]),
                    "reproj_error_px": float(row["reproj_error_px"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def _normalise_times(rows: list[dict]) -> list[dict]:
    """Make sample_time_s relative to the first row's timestamp."""
    if not rows:
        return rows
    t0 = rows[0]["sample_time_s"]
    if t0 > 1000:
        for r in rows:
            r["sample_time_s"] = r["sample_time_s"] - t0
    return rows


def _filter_time(rows: list[dict], t_start: float, duration: float) -> list[dict]:
    return [r for r in rows if t_start <= r["sample_time_s"] < t_start + duration]


def _quality_filter(
    rows: list[dict],
    min_markers: int,
    max_reproj_px: float,
    pid: str,
) -> list[dict]:
    before = len(rows)
    out = [r for r in rows
           if r["marker_count"] >= min_markers and r["reproj_error_px"] <= max_reproj_px]
    dropped = before - len(out)
    if dropped:
        print(f"  {pid}: dropped {dropped}/{before} samples "
              f"(min_markers={min_markers}, max_reproj={max_reproj_px:.0f}px)")
    return out


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_zones(
    zone_config_path: Path | None,
    cfg: dict,
) -> dict[str, dict[str, Any]]:
    if zone_config_path is not None:
        with zone_config_path.open(encoding="utf-8") as fh:
            custom = yaml.safe_load(fh)
        zones = _build_zones_from_config(cfg)
        zones.update(custom.get("zones", {}))
        return zones
    return _build_zones_from_config(cfg)


def _pid_from_stem(stem: str) -> str:
    """Extract P1–P4 participant ID from a CSV / video filename stem."""
    parts = stem.replace("-", "_").split("_")
    for p in parts:
        if re.fullmatch(r"P\d+", p):
            return p
    return stem


# ── Fixation detection (IVT – velocity threshold) ────────────────────────────

def detect_fixations(
    rows: list[dict],
    velocity_threshold_m_s: float = 0.40,
    min_fixation_samples: int = 3,
) -> list[dict]:
    """IVT fixation detection in world XY coordinates.

    Consecutive gaze samples whose instantaneous velocity (world-space m/s)
    falls below *velocity_threshold_m_s* are grouped and collapsed into a
    single fixation record whose position is the centroid of the group.
    A group must contain at least *min_fixation_samples* to be kept.

    Returns one dict per fixation event, with the same fields as the input
    rows plus ``fixation_duration_s`` and ``fixation_n_samples``.
    """
    if len(rows) < 2:
        return list(rows)

    # Compute instantaneous velocities
    velocities: list[float] = [0.0]
    for i in range(1, len(rows)):
        dt = rows[i]["sample_time_s"] - rows[i - 1]["sample_time_s"]
        if dt <= 0 or dt > 0.5:          # gap >0.5 s = discontinuity
            velocities.append(float("inf"))
            continue
        dx = rows[i]["world_x"] - rows[i - 1]["world_x"]
        dy = rows[i]["world_y"] - rows[i - 1]["world_y"]
        velocities.append(((dx ** 2) + (dy ** 2)) ** 0.5 / dt)

    # Group consecutive fixation samples
    fixations: list[dict] = []
    group: list[dict] = []

    def _flush(grp: list[dict]) -> None:
        if len(grp) < min_fixation_samples:
            return
        c = dict(grp[len(grp) // 2])   # take middle row as base
        c["world_x"]             = float(np.mean([g["world_x"] for g in grp]))
        c["world_y"]             = float(np.mean([g["world_y"] for g in grp]))
        c["world_z"]             = float(np.mean([g["world_z"] for g in grp]))
        c["sample_time_s"]       = grp[len(grp) // 2]["sample_time_s"]
        c["marker_count"]        = int(np.mean([g["marker_count"] for g in grp]))
        c["reproj_error_px"]     = float(np.mean([g["reproj_error_px"] for g in grp]))
        c["fixation_duration_s"] = grp[-1]["sample_time_s"] - grp[0]["sample_time_s"]
        c["fixation_n_samples"]  = len(grp)
        fixations.append(c)

    for row, vel in zip(rows, velocities):
        if vel <= velocity_threshold_m_s:
            group.append(row)
        else:
            _flush(group)
            group = []
    _flush(group)

    return fixations


# ── Zone classification ───────────────────────────────────────────────────────

def _classify_point(
    wx: float,
    wy: float,
    zones: dict[str, dict[str, Any]],
) -> str:
    """Return the name of the highest-priority zone containing (wx, wy)."""
    hits: list[tuple[int, str]] = []
    for name, z in zones.items():
        if z["type"] == "circle":
            cx, cy = z["center"]
            r = z["radius"]
            if (wx - cx) ** 2 + (wy - cy) ** 2 <= r ** 2:
                hits.append((z.get("priority", 5), name))
        elif z["type"] == "rect":
            x0, x1 = sorted(z["x"])
            y0, y1 = sorted(z["y"])
            if x0 <= wx <= x1 and y0 <= wy <= y1:
                hits.append((z.get("priority", 5), name))
    if not hits:
        return "other"
    hits.sort()
    return hits[0][1]


def classify_dataset(rows: list[dict], zones: dict[str, dict, Any]) -> list[str]:
    return [_classify_point(r["world_x"], r["world_y"], zones) for r in rows]


# ── Shared attention ──────────────────────────────────────────────────────────

def compute_shared_attention(
    datasets: list[tuple[str, list[dict], list[str]]],
    t_start: float,
    duration: float,
    bin_sec: float = 5.0,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Compute per-bin shared-attention scores.

    Returns
    -------
    shared_mat : (n_zones × n_bins) float array.
        Each cell = number of participants that looked at that zone in that bin
        (fractional: uses fraction of samples in bin that land in zone).
    zone_names : list of zone names (row labels)
    bin_edges  : list of bin-start times (column labels)
    """
    n_bins = max(1, int(duration / bin_sec))
    bin_edges = [t_start + i * bin_sec for i in range(n_bins)]

    # Collect zone names across all datasets
    all_zones: list[str] = []
    for _, _, labels in datasets:
        for z in labels:
            if z not in all_zones:
                all_zones.append(z)
    all_zones = sorted(set(all_zones))

    shared_mat = np.zeros((len(all_zones), n_bins), dtype=float)

    for pid, rows, labels in datasets:
        for bi, bt in enumerate(bin_edges):
            bin_end = bt + bin_sec
            in_bin = [
                labels[j] for j, r in enumerate(rows)
                if bt <= r["sample_time_s"] < bin_end
            ]
            if not in_bin:
                continue
            total = len(in_bin)
            for zi, zname in enumerate(all_zones):
                frac = in_bin.count(zname) / total
                shared_mat[zi, bi] += frac

    return shared_mat, all_zones, bin_edges


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _draw_zones(ax: plt.Axes, zones: dict[str, dict[str, Any]], alpha: float = 0.18) -> None:
    """Draw zone patches on *ax* (world X/Y)."""
    for name, z in sorted(zones.items(), key=lambda kv: -kv[1].get("priority", 5)):
        colour = _ZONE_PALETTE.get(name, "#cccccc")
        label = z.get("label", name)
        if z["type"] == "circle":
            cx, cy = z["center"]
            r = z["radius"]
            patch = mpatches.Circle((cx, cy), r, color=colour, alpha=alpha, zorder=1)
            ax.add_patch(patch)
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=6.5, color="#333", zorder=5, style="italic")
            # Outline
            outline = mpatches.Circle((cx, cy), r, fill=False,
                                      edgecolor=colour, linewidth=1.2, alpha=0.7, zorder=2)
            ax.add_patch(outline)
        elif z["type"] == "rect":
            x0, x1 = sorted(z["x"])
            y0, y1 = sorted(z["y"])
            patch = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                color=colour, alpha=alpha, zorder=1,
            )
            ax.add_patch(patch)
            outline = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=colour, linewidth=1.2, alpha=0.7, zorder=2,
            )
            ax.add_patch(outline)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label,
                    ha="center", va="center",
                    fontsize=6.5, color="#333", zorder=5, style="italic")


def _zone_colour_for_point(zone_name: str) -> str:
    return _ZONE_PALETTE.get(zone_name, "#888888")


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(
    gaze_csvs: list[Path],
    config_path: Path,
    output: Path,
    t_start: float,
    duration: float,
    min_markers: int = 1,
    max_reproj_px: float = 50.0,
    bin_sec: float = 5.0,
    zone_config: Path | None = None,
    session_label: str = "",
    use_fixations: bool = True,
    fixation_velocity_threshold: float = 0.40,
    fixation_min_samples: int = 3,
) -> None:
    cfg = _load_config(config_path)

    zones = _load_zones(zone_config, cfg)

    # ── Load & preprocess gaze ────────────────────────────────────────────────
    datasets: list[tuple[str, list[dict], list[str]]] = []
    for csv_path in gaze_csvs:
        pid = _pid_from_stem(csv_path.stem)
        rows = _load_gaze_csv(csv_path)
        rows = _normalise_times(rows)
        rows = _filter_time(rows, t_start, duration)
        rows = _quality_filter(rows, min_markers, max_reproj_px, pid)
        if use_fixations:
            rows = detect_fixations(
                rows,
                velocity_threshold_m_s=fixation_velocity_threshold,
                min_fixation_samples=fixation_min_samples,
            )
        labels = classify_dataset(rows, zones)
        datasets.append((pid, rows, labels))
        zone_counts = {z: labels.count(z) for z in set(labels)}
        print(f"  {pid}: {len(rows)} {'fixations' if use_fixations else 'samples'} | zones: "
              + " | ".join(f"{k}={v}" for k, v in sorted(zone_counts.items())))

    if not any(rows for _, rows, _ in datasets):
        print("ERROR: no gaze data after filtering.", file=sys.stderr)
        return

    n_parts = len(datasets)
    all_zone_names_ordered = list(zones.keys()) + ["other"]
    # Only keep zones that actually appear in data
    present_zones = [z for z in all_zone_names_ordered
                     if any(z in labels for _, _, labels in datasets)]

    shared_mat, shared_zones, bin_edges = compute_shared_attention(
        datasets, t_start, duration, bin_sec
    )
    n_bins = len(bin_edges)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: scatter (large) + shared-attention heatmap
    # Row 1: per-participant zone-dwell heatmaps (one row per participant)
    # Row 2: zone-dwell summary bar chart
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(
        3, 4,
        figure=fig,
        height_ratios=[1.8, 1.0, 0.9],
        width_ratios=[1.4, 1.4, 1.2, 1.0],
        hspace=0.50,
        wspace=0.38,
    )

    ax_scatter = fig.add_subplot(gs[0, :2])
    ax_shared  = fig.add_subplot(gs[0, 2:])

    # ── Panel 1: desk top-view scatter ────────────────────────────────────────
    _draw_zones(ax_scatter, zones)

    # Desk outline from zone definition (world frame: x=right, y=back)
    _desk_z = zones.get("desk", {})
    _dx0, _dx1 = sorted(_desk_z.get("x", [-0.925, 0.925]))
    _dy0, _dy1 = sorted(_desk_z.get("y", [-0.425, 0.425]))
    desk_rect = mpatches.Rectangle(
        (_dx0, _dy0), _dx1 - _dx0, _dy1 - _dy0,
        linewidth=2.0, edgecolor="#444", facecolor="none", zorder=3,
    )
    ax_scatter.add_patch(desk_rect)

    # Gaze points coloured by zone
    for i, (pid, rows, labels) in enumerate(datasets):
        if not rows:
            continue
        part_col = _PART_COLOURS[i % len(_PART_COLOURS)]
        wx = np.array([r["world_x"] for r in rows])
        wy = np.array([r["world_y"] for r in rows])
        if use_fixations:
            # Size dots by fixation duration (longer fixation → larger dot)
            durations = np.array([r.get("fixation_duration_s", 0.05) for r in rows])
            sizes = np.clip(durations * 120, 8, 180)
            alpha = 0.55
        else:
            sizes = 5
            alpha = 0.35
        ax_scatter.scatter(
            wx, wy,
            s=sizes, alpha=alpha,
            color=part_col,
            edgecolors="none",
            label=pid,
            zorder=4,
        )

    # Direction labels — world frame:  x = right, y = back
    # Left side (x−) → P3/P4;  Right side (x+) → P1/P2
    # Front (y−) → Screen;  Back (y+) → Moderator/Big Screen
    ax_scatter.text(0, _dy1 + 0.10, "BACK — Moderator / Big Screen",
                    ha="center", va="bottom", fontsize=7, color="#888800", style="italic")
    ax_scatter.text(0, _dy0 - 0.10, "FRONT — Screen (between P50 & cam5)",
                    ha="center", va="top",    fontsize=7, color="#6633aa", style="italic")
    ax_scatter.text(_dx0 - 0.10, 0, "LEFT side\nP4 (back) · P3 (front)",
                    ha="right", va="center", fontsize=6, color="#2980b9")
    ax_scatter.text(_dx1 + 0.10, 0, "RIGHT side\nP1 (back) · P2 (front)",
                    ha="left",  va="center", fontsize=6, color="#c0392b")

    # World frame: x = right, y = back.  Standard orientation.
    ax_scatter.set_xlim(-2.00, +2.00)
    ax_scatter.set_ylim(-1.60, +1.60)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel(
        "← LEFT (P3/P4)   x (right)   RIGHT (P1/P2) →",
        fontsize=7,
    )
    ax_scatter.set_ylabel(
        "↓ FRONT (Screen)   y (back)   BACK (Moderator) ↑",
        fontsize=7,
    )
    point_label = "fixations" if use_fixations else "gaze samples"
    ax_scatter.set_title(
        f"{'Fixation' if use_fixations else 'Gaze'} landing points by participant\n"
        f"t={t_start:.0f}–{t_start + duration:.0f}s  |  "
        f"dot size ∝ fixation duration" if use_fixations else
        f"t={t_start:.0f}–{t_start + duration:.0f}s  |  zones: colour patches",
        fontsize=10,
    )

    # Participant-colour legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_PART_COLOURS[i % 4], markersize=8,
               label=pid)
        for i, (pid, _, _) in enumerate(datasets)
    ]
    ax_scatter.legend(handles=handles, loc="lower right", fontsize=8,
                      framealpha=0.8)
    ax_scatter.grid(True, alpha=0.25)

    # ── Panel 2: shared-attention heatmap ─────────────────────────────────────
    # shared_mat shape: (n_zones, n_bins) — value = sum of per-participant fractions
    # A cell value ≥ 2.0 means ≥ 2 participants looked at the zone ≥ 100% of the bin
    # Cap at n_parts for normalisation
    shared_disp = shared_mat[[shared_zones.index(z) for z in present_zones
                               if z in shared_zones], :]

    if shared_disp.size > 0:
        bin_tick_labels = [f"{bt:.0f}" for bt in bin_edges]
        # Downsample tick labels for readability
        tick_every = max(1, n_bins // 10)
        tick_pos = list(range(0, n_bins, tick_every))

        im = ax_shared.imshow(
            shared_disp,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=max(2.0, float(shared_disp.max())),
            interpolation="nearest",
        )
        ax_shared.set_yticks(range(len(present_zones)))
        ax_shared.set_yticklabels(
            [zones.get(z, {}).get("label", z) for z in present_zones],
            fontsize=8,
        )
        ax_shared.set_xticks(tick_pos)
        ax_shared.set_xticklabels(
            [f"{bin_edges[i]:.0f}s" for i in tick_pos],
            fontsize=7, rotation=45, ha="right",
        )
        ax_shared.set_xlabel("Time (s)", fontsize=9)
        ax_shared.set_title(
            "Shared attention\n(sum of gaze-fraction per zone; ≥2 = multiple participants)",
            fontsize=9,
        )
        cbar = fig.colorbar(im, ax=ax_shared, fraction=0.04, pad=0.02)
        cbar.set_label("Σ participant gaze fraction", fontsize=7)
        # Draw threshold lines
        ax_shared.axhline(-0.5, color="k", lw=0.4)
    else:
        ax_shared.text(0.5, 0.5, "No shared-attention data",
                       ha="center", va="center", transform=ax_shared.transAxes)
        ax_shared.axis("off")

    # ── Row 1: per-participant zone-dwell heatmaps ────────────────────────────
    dwell_axes: list[plt.Axes] = []
    for i in range(min(n_parts, 4)):
        ax_dw = fig.add_subplot(gs[1, i])
        dwell_axes.append(ax_dw)

    for i, (pid, rows, labels) in enumerate(datasets[:4]):
        ax_dw = dwell_axes[i]
        if not rows:
            ax_dw.text(0.5, 0.5, f"{pid}\nno data",
                       ha="center", va="center", transform=ax_dw.transAxes)
            ax_dw.axis("off")
            continue

        # Build (n_zones × n_bins) dwell matrix for this participant
        dwell = np.zeros((len(present_zones), n_bins), dtype=float)
        for bi, bt in enumerate(bin_edges):
            bin_end = bt + bin_sec
            in_bin = [
                labels[j] for j, r in enumerate(rows)
                if bt <= r["sample_time_s"] < bin_end
            ]
            if not in_bin:
                continue
            total = len(in_bin)
            for zi, zname in enumerate(present_zones):
                dwell[zi, bi] = in_bin.count(zname) / total

        part_cmap = LinearSegmentedColormap.from_list(
            "white_to_part",
            ["white", _PART_COLOURS[i % len(_PART_COLOURS)]],
        )
        ax_dw.imshow(
            dwell, aspect="auto", cmap=part_cmap,
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax_dw.set_yticks(range(len(present_zones)))
        ax_dw.set_yticklabels(
            [zones.get(z, {}).get("label", z) for z in present_zones],
            fontsize=7,
        )
        ax_dw.set_xticks(tick_pos)
        ax_dw.set_xticklabels(
            [f"{bin_edges[i_]:.0f}" for i_ in tick_pos],
            fontsize=6, rotation=45, ha="right",
        )
        ax_dw.set_title(f"{pid} gaze zone dwell\n(fraction per {bin_sec:.0f}s bin)",
                        fontsize=8)
        ax_dw.set_xlabel("Time (s)", fontsize=7)

    # ── Row 2: zone-dwell summary bar chart ───────────────────────────────────
    ax_bar = fig.add_subplot(gs[2, :3])

    bar_width = 0.15
    x_pos = np.arange(len(present_zones))

    for i, (pid, rows, labels) in enumerate(datasets):
        if not labels:
            continue
        total = len(labels)
        fracs = [labels.count(z) / total for z in present_zones]
        ax_bar.bar(
            x_pos + i * bar_width,
            fracs,
            bar_width,
            color=_PART_COLOURS[i % len(_PART_COLOURS)],
            alpha=0.8,
            label=pid,
        )

    ax_bar.set_xticks(x_pos + bar_width * (n_parts - 1) / 2)
    ax_bar.set_xticklabels(
        [zones.get(z, {}).get("label", z) for z in present_zones],
        rotation=30, ha="right", fontsize=8,
    )
    ax_bar.set_ylabel("Fraction of samples", fontsize=9)
    ax_bar.set_title("Overall gaze dwell by zone", fontsize=10)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.set_ylim(0, 1.05)

    # ── Zone legend (bottom-right) ────────────────────────────────────────────
    ax_legend = fig.add_subplot(gs[2, 3])
    zone_legend_handles = [
        mpatches.Patch(
            facecolor=_ZONE_PALETTE.get(z, "#cccccc"),
            edgecolor="#555",
            alpha=0.7,
            label=zones.get(z, {}).get("label", z),
        )
        for z in present_zones
    ]
    ax_legend.legend(handles=zone_legend_handles, loc="center",
                     fontsize=7.5, frameon=True, title="Interest areas",
                     title_fontsize=8)
    ax_legend.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    title = session_label or "Gaze attention zones"
    mode_str = "fixations" if use_fixations else "raw gaze"
    fig.suptitle(
        f"{title} | t={t_start:.0f}–{t_start + duration:.0f}s | "
        f"{mode_str} | ≥{min_markers} markers, ≤{max_reproj_px:.0f}px reproj",
        fontsize=12, fontweight="bold", y=1.01,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="World-gaze interest-area and shared-attention visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--gaze-csvs", nargs="+", required=True, type=Path,
        metavar="CSV",
        help="gaze_world.csv files (one per participant)",
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="World-align YAML config (for desk geometry)",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output PNG path",
    )
    p.add_argument(
        "--zone-config", type=Path, default=None,
        metavar="YAML",
        help=(
            "Optional YAML file with custom zone definitions. "
            "Must have a top-level 'zones:' key whose value overrides defaults."
        ),
    )
    p.add_argument(
        "--raw-gaze", action="store_true", default=False,
        help="Use raw gaze samples instead of fixation centroids",
    )
    p.add_argument(
        "--fixation-velocity", type=float, default=0.40, metavar="M_S",
        help="IVT fixation velocity threshold in world m/s (default 0.40)",
    )
    p.add_argument(
        "--fixation-min-samples", type=int, default=3,
        help="Minimum consecutive samples to constitute a fixation (default 3)",
    )
    p.add_argument("--t-start", type=float, default=0.0,
                   help="Start time in seconds relative to session (default 0)")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Duration in seconds (default 300 = 5 min)")
    p.add_argument("--bin-sec", type=float, default=5.0,
                   help="Time-bin size in seconds for dwell heatmap (default 5)")
    p.add_argument("--min-markers", type=int, default=1,
                   help="Min ArUco marker count to accept a sample (default 1)")
    p.add_argument("--max-reproj-px", type=float, default=50.0,
                   help="Max PnP reprojection error (px) to accept a sample (default 50)")
    p.add_argument("--session-label", type=str, default="",
                   help="Session label for the figure title, e.g. 'grp-13 T1'")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    missing = [p for p in args.gaze_csvs if not p.exists()]
    if missing:
        print(f"ERROR: files not found: {missing}", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    make_figure(
        gaze_csvs=args.gaze_csvs,
        config_path=args.config,
        output=args.output,
        t_start=args.t_start,
        duration=args.duration,
        min_markers=args.min_markers,
        max_reproj_px=args.max_reproj_px,
        bin_sec=args.bin_sec,
        zone_config=args.zone_config,
        session_label=args.session_label,
        use_fixations=not args.raw_gaze,
        fixation_velocity_threshold=args.fixation_velocity,
        fixation_min_samples=args.fixation_min_samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
