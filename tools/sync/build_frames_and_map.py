#!/usr/bin/env python3
"""Build frame timing tables and LSL sync maps from ffmpeg progress logs."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MappingSegment:
    t0_src: float
    t0_tgt: float
    scale: float
    offset: float

    def apply(self, t_src: float) -> float:
        return (self.scale * t_src) + self.offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frame tables and sync maps")
    parser.add_argument("--session-dir", required=True, help="Session directory")
    parser.add_argument("--label", required=True, help="Camera/device label")
    parser.add_argument("--video-path", help="Override path to video file")
    parser.add_argument("--progress-path", help="Override path to ffmpeg progress TSV")
    parser.add_argument("--xdf-path", help="Optional XDF path for LSL timestamps")
    parser.add_argument("--window", type=int, default=300, help="Window size for piecewise fit")
    parser.add_argument("--residual-threshold", type=float, default=0.05, help="Seconds residual threshold")
    return parser.parse_args()


def run_ffprobe(video_path: Path) -> list[float]:
    cmd = [
        "ffprobe",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    pts_times: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pts_times.append(float(line))
        except ValueError:
            continue
    return pts_times


def read_progress_tsv(progress_path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    segments: list[tuple[list[float], list[float], list[float], list[float]]] = [
        ([], [], [], [])
    ]
    prev_out_time: float | None = None
    with progress_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                host_time = float(row["host_time_sec"])
                out_time = float(row["out_time_sec"])
                drop = float(row.get("drop_frames", 0.0))
                dup = float(row.get("dup_frames", 0.0))
            except (TypeError, ValueError, KeyError):
                continue
            if prev_out_time is not None and out_time + 0.5 < prev_out_time:
                segments.append(([], [], [], []))
            seg_host, seg_out, seg_drop, seg_dup = segments[-1]
            seg_host.append(host_time)
            seg_out.append(out_time)
            seg_drop.append(drop)
            seg_dup.append(dup)
            prev_out_time = out_time

    for host_times, out_times, drop_frames, dup_frames in reversed(segments):
        if host_times:
            return host_times, out_times, drop_frames, dup_frames
    return [], [], [], []


def linear_fit(xs: Iterable[float], ys: Iterable[float]) -> tuple[float, float]:
    xs_list = list(xs)
    ys_list = list(ys)
    n = len(xs_list)
    if n == 0:
        return 1.0, 0.0
    mean_x = sum(xs_list) / n
    mean_y = sum(ys_list) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_list, ys_list, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs_list)
    if den == 0:
        return 1.0, mean_y - mean_x
    scale = num / den
    offset = mean_y - (scale * mean_x)
    return scale, offset


def residual_stats(xs: list[float], ys: list[float], scale: float, offset: float) -> float:
    if not xs:
        return 0.0
    residuals = [abs((scale * x + offset) - y) for x, y in zip(xs, ys, strict=True)]
    return sum(residuals) / len(residuals)


def piecewise_fit(xs: list[float], ys: list[float], window: int) -> list[MappingSegment]:
    segments: list[MappingSegment] = []
    if not xs:
        return segments
    for start in range(0, len(xs), window):
        end = min(start + window, len(xs))
        segment_x = xs[start:end]
        segment_y = ys[start:end]
        scale, offset = linear_fit(segment_x, segment_y)
        t0_src = segment_x[0]
        t0_tgt = segment_y[0]
        segments.append(
            MappingSegment(t0_src=t0_src, t0_tgt=t0_tgt, scale=scale, offset=offset)
        )
    return segments


def apply_segments(segments: list[MappingSegment], t_src: float) -> float:
    if not segments:
        return t_src
    segment = segments[0]
    for candidate in segments:
        if t_src >= candidate.t0_src:
            segment = candidate
        else:
            break
    return segment.apply(t_src)


def load_xdf_mapping(xdf_path: Path, label: str) -> tuple[list[float], list[float]]:
    if not xdf_path.exists():
        return [], []
    if importlib.util.find_spec("pyxdf") is None:
        return [], []
    pyxdf = importlib.import_module("pyxdf")
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    target_name = f"ffmpeg_progress_{label}"
    for stream in streams:
        if stream.get("info", {}).get("name", [""])[0] == target_name:
            times = stream.get("time_stamps", [])
            series = stream.get("time_series", [])
            out_times = [float(sample[0]) for sample in series]
            return out_times, list(times)
    return [], []


def build_sync_map(
    out_times: list[float],
    target_times: list[float],
    window: int,
    residual_threshold: float,
) -> tuple[list[MappingSegment], dict]:
    scale, offset = linear_fit(out_times, target_times)
    residual = residual_stats(out_times, target_times, scale, offset)
    use_piecewise = residual > residual_threshold and len(out_times) > window
    if use_piecewise:
        segments = piecewise_fit(out_times, target_times, window)
        notes = "USB jitter detected; used piecewise mapping"
    else:
        segments = [MappingSegment(t0_src=out_times[0], t0_tgt=target_times[0], scale=scale, offset=offset)]
        notes = "Linear mapping"
    drift_ppm = (segments[0].scale - 1.0) * 1_000_000
    qc = {
        "progress_points": len(out_times),
        "estimated_drift_ppm": drift_ppm,
        "notes": notes,
    }
    return segments, qc


def write_frames_tsv(output_path: Path, pts_times: list[float], segments: list[MappingSegment]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["frame_idx", "pts_time_sec", "lsl_time_sec", "is_missing", "flags"])
        for idx, pts in enumerate(pts_times):
            lsl_time = apply_segments(segments, pts)
            writer.writerow([idx, f"{pts:.6f}", f"{lsl_time:.6f}", 0, ""])


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    label = args.label
    video_path = Path(args.video_path) if args.video_path else session_dir / "video" / f"{label}_video.mkv"
    progress_path = (
        Path(args.progress_path)
        if args.progress_path
        else session_dir / "sourcedata" / "sync" / f"{label}_ffmpeg_progress.tsv"
    )
    xdf_path = Path(args.xdf_path) if args.xdf_path else session_dir / "sourcedata" / "lsl" / "session.xdf"

    pts_times = run_ffprobe(video_path)
    host_times, out_times, _, _ = read_progress_tsv(progress_path)

    xdf_out_times, xdf_times = load_xdf_mapping(xdf_path, label)
    if xdf_out_times and xdf_times:
        map_out_times = xdf_out_times
        map_target_times = xdf_times
        target_time_label = "lsl_time"
    else:
        map_out_times = out_times
        map_target_times = host_times
        target_time_label = "host_time"

    if not map_out_times or not map_target_times:
        raise SystemExit("No progress timestamps available to build mapping")

    segments, qc = build_sync_map(map_out_times, map_target_times, args.window, args.residual_threshold)

    sync_map = {
        "device": label,
        "method": "piecewise_linear_from_progress" if len(segments) > 1 else "linear_from_progress",
        "source_time": "ffmpeg_out_time_sec",
        "target_time": target_time_label,
        "segments": [
            {
                "t0_src": s.t0_src,
                "t0_tgt": s.t0_tgt,
                "scale": s.scale,
                "offset": s.offset,
            }
            for s in segments
        ],
        "qc": qc,
    }

    sync_dir = session_dir / "sourcedata" / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    sync_map_path = sync_dir / f"{label}_sync_map.json"
    sync_map_path.write_text(json.dumps(sync_map, indent=2))

    frames_path = sync_dir / f"{label}_frames.tsv"
    write_frames_tsv(frames_path, pts_times, segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
