#!/usr/bin/env python3
"""Generate QC metrics for sync logs and frame tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sync QC report")
    parser.add_argument("--session-dir", required=True, help="Session directory")
    return parser.parse_args()


def read_progress(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append(
                    {
                        "host_time_sec": float(row["host_time_sec"]),
                        "out_time_sec": float(row["out_time_sec"]),
                        "frame": float(row.get("frame", 0.0)),
                        "drop_frames": float(row.get("drop_frames", 0.0)),
                        "dup_frames": float(row.get("dup_frames", 0.0)),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue
    return rows


def read_frames(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append(
                    {
                        "frame_idx": float(row["frame_idx"]),
                        "pts_time_sec": float(row["pts_time_sec"]),
                        "lsl_time_sec": float(row["lsl_time_sec"]),
                        "is_missing": float(row.get("is_missing", 0.0)),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue
    return rows


def summarize_progress(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {}
    out_times = [r["out_time_sec"] for r in rows]
    gaps = [out_times[i+1] - out_times[i] for i in range(len(out_times)-1)]
    monotonic = all(g >= 0 for g in gaps)
    max_gap = max(gaps) if gaps else 0.0
    last = rows[-1]
    return {
        "progress_points": len(rows),
        "monotonic_out_time": monotonic,
        "max_out_time_gap_sec": max_gap,
        "drop_frames": last.get("drop_frames", 0.0),
        "dup_frames": last.get("dup_frames", 0.0),
    }


def summarize_frames(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {}
    pts_times = [r["pts_time_sec"] for r in rows]
    gaps = [pts_times[i+1] - pts_times[i] for i in range(len(pts_times)-1) if pts_times[i+1] > pts_times[i]]
    fps_values = [1.0 / g for g in gaps if g > 0]
    mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    max_gap = max(gaps) if gaps else 0.0
    missing_rate = sum(r["is_missing"] for r in rows) / len(rows)
    return {
        "frames": len(rows),
        "mean_fps": mean_fps,
        "max_frame_gap_sec": max_gap,
        "missing_rate": missing_rate,
    }


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    sync_dir = session_dir / "sourcedata" / "sync"
    qc_dir = session_dir / "sourcedata" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}

    for progress_path in sync_dir.glob("*_ffmpeg_progress.tsv"):
        label = progress_path.name.replace("_ffmpeg_progress.tsv", "")
        progress_rows = read_progress(progress_path)
        frames_path = sync_dir / f"{label}_frames.tsv"
        frames_rows = read_frames(frames_path) if frames_path.exists() else []
        report[label] = {
            "progress": summarize_progress(progress_rows),
            "frames": summarize_frames(frames_rows),
        }

    json_path = qc_dir / "sync_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_lines = ["# Sync QC Report", ""]
    for label, metrics in report.items():
        md_lines.append(f"## {label}")
        progress = metrics.get("progress", {})
        frames = metrics.get("frames", {})
        if progress:
            md_lines.append("- Progress")
            md_lines.append(f"  - points: {progress.get('progress_points', 0)}")
            md_lines.append(f"  - monotonic out_time: {progress.get('monotonic_out_time', False)}")
            md_lines.append(f"  - max out_time gap (s): {progress.get('max_out_time_gap_sec', 0):.4f}")
            md_lines.append(f"  - drop frames: {progress.get('drop_frames', 0)}")
            md_lines.append(f"  - dup frames: {progress.get('dup_frames', 0)}")
        if frames:
            md_lines.append("- Frames")
            md_lines.append(f"  - frame count: {frames.get('frames', 0)}")
            md_lines.append(f"  - mean FPS: {frames.get('mean_fps', 0):.2f}")
            md_lines.append(f"  - max frame gap (s): {frames.get('max_frame_gap_sec', 0):.4f}")
            md_lines.append(f"  - missing rate: {frames.get('missing_rate', 0):.2%}")
        md_lines.append("")

    md_path = qc_dir / "sync_report.md"
    md_path.write_text("\n".join(md_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
