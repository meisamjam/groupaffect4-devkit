#!/usr/bin/env python3
"""Resample frame timestamps onto a fixed LSL time grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample frame table to fixed rate")
    parser.add_argument("--input", required=True, help="Input frames TSV")
    parser.add_argument("--output", required=True, help="Output resampled TSV")
    parser.add_argument("--rate", type=float, default=30.0, help="Target rate in Hz")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Max seconds for nearest frame")
    return parser.parse_args()


def read_frames(path: Path) -> list[tuple[int, float]]:
    frames: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                idx = int(row["frame_idx"])
                lsl_time = float(row["lsl_time_sec"])
            except (TypeError, ValueError, KeyError):
                continue
            frames.append((idx, lsl_time))
    return frames


def main() -> int:
    args = parse_args()
    frames = read_frames(Path(args.input))
    if not frames:
        raise SystemExit("No frames found to resample")

    frames.sort(key=lambda x: x[1])
    start = frames[0][1]
    end = frames[-1][1]
    step = 1.0 / args.rate

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["target_lsl_time", "frame_idx", "source_lsl_time", "is_missing"])

        frame_iter = iter(frames)
        current = next(frame_iter)
        next_frame = next(frame_iter, None)

        t = start
        while t <= end:
            while next_frame and next_frame[1] < t:
                current = next_frame
                next_frame = next(frame_iter, None)

            candidates = [current]
            if next_frame:
                candidates.append(next_frame)

            best = min(candidates, key=lambda x: abs(x[1] - t))
            delta = abs(best[1] - t)
            if delta <= args.tolerance:
                writer.writerow([f"{t:.6f}", best[0], f"{best[1]:.6f}", 0])
            else:
                writer.writerow([f"{t:.6f}", "", "", 1])
            t += step

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
