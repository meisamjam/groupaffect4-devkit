#!/usr/bin/env python3
"""QC plotter for offline Tobii multi-glasses world-aligned gaze outputs.

Reads per-device `*_gaze_world.ndjson` files produced by
`tools/tobii_multi_glasses_world_align.py`, then writes:
- summary JSON (`tobii_world_gaze_summary.json`)
- summary CSV (`tobii_world_gaze_summary.csv`)
- world-plane scatter plot (`tobii_world_gaze_scatter.png`)
- world x/y over time plot (`tobii_world_gaze_timeseries.png`)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(slots=True)
class DeviceSeries:
    device_id: str
    times: np.ndarray
    world_x: np.ndarray
    world_y: np.ndarray
    world_z: np.ndarray
    marker_count: np.ndarray
    reproj_error_px: np.ndarray
    total_rows: int
    valid_world_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC plots for world-aligned Tobii gaze")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_gaze_world.ndjson files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/qc)",
    )
    parser.add_argument(
        "--pattern",
        default="*_gaze_world.ndjson",
        help="Glob pattern for world-gaze files",
    )
    parser.add_argument(
        "--max-points-per-device",
        type=int,
        default=10000,
        help="Maximum plotted points per device (uniform downsample)",
    )
    parser.add_argument(
        "--align-config",
        type=Path,
        default=None,
        help="Optional alignment config YAML to overlay marker polygons/bounds",
    )
    return parser.parse_args()


def _load_marker_polygons(align_config_path: Path | None) -> list[np.ndarray]:
    if align_config_path is None:
        return []
    if not align_config_path.exists():
        raise SystemExit(f"Alignment config not found: {align_config_path}")

    raw_config = yaml.safe_load(align_config_path.read_text(encoding="utf-8"))
    world = raw_config.get("world", {}) if isinstance(raw_config, dict) else {}
    marker_map = world.get("marker_map", [])
    if not isinstance(marker_map, list):
        return []

    polygons: list[np.ndarray] = []
    for marker in marker_map:
        if not isinstance(marker, dict):
            continue
        corners = marker.get("corners_m")
        if not isinstance(corners, list) or len(corners) != 4:
            continue
        polygon_xy: list[list[float]] = []
        is_valid = True
        for corner in corners:
            if not isinstance(corner, list) or len(corner) < 2:
                is_valid = False
                break
            x = _safe_float(corner[0])
            y = _safe_float(corner[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                is_valid = False
                break
            polygon_xy.append([x, y])
        if is_valid:
            polygons.append(np.asarray(polygon_xy, dtype=np.float64))

    return polygons


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_device_series(path: Path) -> DeviceSeries:
    device_id = path.name.replace("_gaze_world.ndjson", "")
    times: list[float] = []
    world_x: list[float] = []
    world_y: list[float] = []
    world_z: list[float] = []
    marker_count: list[float] = []
    reproj_error_px: list[float] = []

    total_rows = 0
    valid_world_rows = 0

    with path.open("r", encoding="utf-8") as file_pointer:
        for line in file_pointer:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            payload = json.loads(line)

            sample_time = _safe_float(payload.get("sample_time_s"))
            marker_value = _safe_float(payload.get("marker_count"))
            reproj_value = _safe_float(payload.get("reproj_error_px"))

            world_point = payload.get("world_point_m")
            if not isinstance(world_point, list) or len(world_point) < 3:
                continue

            wx = _safe_float(world_point[0])
            wy = _safe_float(world_point[1])
            wz = _safe_float(world_point[2])
            if not (np.isfinite(wx) and np.isfinite(wy) and np.isfinite(wz) and np.isfinite(sample_time)):
                continue

            valid_world_rows += 1
            times.append(sample_time)
            world_x.append(wx)
            world_y.append(wy)
            world_z.append(wz)
            marker_count.append(marker_value)
            reproj_error_px.append(reproj_value)

    return DeviceSeries(
        device_id=device_id,
        times=np.asarray(times, dtype=np.float64),
        world_x=np.asarray(world_x, dtype=np.float64),
        world_y=np.asarray(world_y, dtype=np.float64),
        world_z=np.asarray(world_z, dtype=np.float64),
        marker_count=np.asarray(marker_count, dtype=np.float64),
        reproj_error_px=np.asarray(reproj_error_px, dtype=np.float64),
        total_rows=total_rows,
        valid_world_rows=valid_world_rows,
    )


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, num=max_points, dtype=np.int64)


def _write_summary(series_list: list[DeviceSeries], output_dir: Path) -> None:
    summary: dict[str, dict] = {}

    for series in series_list:
        if len(series.times) == 0:
            summary[series.device_id] = {
                "rows_total": series.total_rows,
                "rows_with_world_point": series.valid_world_rows,
                "time_start_s": None,
                "time_end_s": None,
                "x_min_m": None,
                "x_max_m": None,
                "y_min_m": None,
                "y_max_m": None,
                "z_median_m": None,
                "marker_count_median": None,
                "reproj_error_median_px": None,
            }
            continue

        summary[series.device_id] = {
            "rows_total": series.total_rows,
            "rows_with_world_point": series.valid_world_rows,
            "time_start_s": float(np.min(series.times)),
            "time_end_s": float(np.max(series.times)),
            "x_min_m": float(np.min(series.world_x)),
            "x_max_m": float(np.max(series.world_x)),
            "y_min_m": float(np.min(series.world_y)),
            "y_max_m": float(np.max(series.world_y)),
            "z_median_m": float(np.median(series.world_z)),
            "marker_count_median": float(np.nanmedian(series.marker_count)),
            "reproj_error_median_px": float(np.nanmedian(series.reproj_error_px)),
        }

    summary_path = output_dir / "tobii_world_gaze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "tobii_world_gaze_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "device_id",
            "rows_total",
            "rows_with_world_point",
            "time_start_s",
            "time_end_s",
            "x_min_m",
            "x_max_m",
            "y_min_m",
            "y_max_m",
            "z_median_m",
            "marker_count_median",
            "reproj_error_median_px",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for device_id, device_summary in summary.items():
            writer.writerow({"device_id": device_id, **device_summary})


def _plot(
    series_list: list[DeviceSeries],
    output_dir: Path,
    max_points_per_device: int,
    marker_polygons: list[np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    valid_series = [series for series in series_list if len(series.times) > 0]
    if not valid_series:
        raise SystemExit("No valid world-gaze points found to plot.")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 7))
    for index, series in enumerate(valid_series):
        idx = _downsample_indices(len(series.times), max_points_per_device)
        scatter_ax.scatter(
            series.world_x[idx],
            series.world_y[idx],
            s=6,
            alpha=0.35,
            color=colors[index % len(colors)],
            label=series.device_id,
        )

    if marker_polygons:
        for marker_index, polygon in enumerate(marker_polygons):
            closed = np.vstack([polygon, polygon[0]])
            scatter_ax.plot(
                closed[:, 0],
                closed[:, 1],
                color="#111111",
                linewidth=1.0,
                alpha=0.8,
                label="marker" if marker_index == 0 else None,
            )

        all_points = np.vstack(marker_polygons)
        x_min = float(np.min(all_points[:, 0]))
        x_max = float(np.max(all_points[:, 0]))
        y_min = float(np.min(all_points[:, 1]))
        y_max = float(np.max(all_points[:, 1]))

        board_x = [x_min, x_max, x_max, x_min, x_min]
        board_y = [y_min, y_min, y_max, y_max, y_min]
        scatter_ax.plot(
            board_x,
            board_y,
            color="#000000",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label="board bounds",
        )

    scatter_ax.set_title("World-Gaze Scatter (Table/Board Plane)")
    scatter_ax.set_xlabel("world x (m)")
    scatter_ax.set_ylabel("world y (m)")
    scatter_ax.grid(True, alpha=0.3)
    scatter_ax.legend(loc="best")
    scatter_fig.tight_layout()
    scatter_fig.savefig(output_dir / "tobii_world_gaze_scatter.png", dpi=180)
    plt.close(scatter_fig)

    ts_fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for index, series in enumerate(valid_series):
        idx = _downsample_indices(len(series.times), max_points_per_device)
        color = colors[index % len(colors)]
        ax_x.plot(series.times[idx], series.world_x[idx], color=color, linewidth=0.8, label=series.device_id)
        ax_y.plot(series.times[idx], series.world_y[idx], color=color, linewidth=0.8, label=series.device_id)

    ax_x.set_ylabel("world x (m)")
    ax_y.set_ylabel("world y (m)")
    ax_y.set_xlabel("time (s)")
    ax_x.set_title("World-Gaze Trajectories Over Time")
    ax_x.grid(True, alpha=0.3)
    ax_y.grid(True, alpha=0.3)
    ax_x.legend(loc="best")
    ts_fig.tight_layout()
    ts_fig.savefig(output_dir / "tobii_world_gaze_timeseries.png", dpi=180)
    plt.close(ts_fig)


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else (input_dir / "qc")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {input_dir} with pattern {args.pattern}")

    series_list = [_load_device_series(path) for path in files]
    marker_polygons = _load_marker_polygons(args.align_config)

    _write_summary(series_list, output_dir)
    _plot(series_list, output_dir, args.max_points_per_device, marker_polygons)

    print(f"Processed {len(series_list)} devices")
    if marker_polygons:
        print(f"Overlayed {len(marker_polygons)} marker polygons from: {args.align_config}")
    print(f"Wrote QC outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
