"""Compare LSL stream timestamps with frame log timestamps.

Analyzes timing differences between LSL progress-based timestamps and
showinfo PTS-based frame logs to diagnose synchronization accuracy.

Usage:
    python tools/compare_lsl_frame_logs.py --session data/sub-001/ses-001 --device jabra_panacast_20_vid
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def load_lsl_data(lsl_path: Path) -> list[dict]:
    """Load LSL JSONL records."""
    records = []
    if not lsl_path.exists():
        return records
    
    with lsl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # LSL formats:
                # - Legacy: values=[timestamp, frame_number]
                # - Anchor: values=[lsl_time, media_time_us, frame]
                stream_time = float(data.get("stream_time", 0))
                values = data.get("values", [])
                timestamp = float(values[0]) if len(values) > 0 else 0
                if len(values) >= 3:
                    media_time_us = float(values[1])
                    frame_number = int(values[2])
                else:
                    media_time_us = None
                    frame_number = int(values[1]) if len(values) > 1 else 0
                records.append({
                    "stream_time": stream_time,
                    "lsl_timestamp": timestamp,
                    "frame": frame_number,
                    "media_time_us": media_time_us,
                })
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
    
    return records


def load_frame_logs(frame_log_path: Path) -> list[dict]:
    """Load frame log JSONL records."""
    records = []
    if not frame_log_path.exists():
        return records
    
    with frame_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Frame log format: {frame, pts_time, lsl_clock, unix_time}
                records.append({
                    "frame": int(data.get("frame", 0)),
                    "pts_time": float(data.get("pts_time", 0)),
                    "lsl_clock": float(data.get("lsl_time", data.get("lsl_clock", 0))),
                    "unix_time": float(data.get("unix_time", 0)),
                    "unix_time_s": float(data.get("unix_time_s", 0)),
                })
            except (json.JSONDecodeError, ValueError):
                continue
    
    return records


def load_progress_logs(progress_tsv_path: Path, progress_json_path: Path | None = None) -> list[dict]:
    """Load ffmpeg progress records from TSV (current) or JSONL (legacy)."""
    records = []
    if progress_tsv_path.exists():
        segments: list[list[dict]] = [[]]
        prev_out_time = None
        with progress_tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    host_time = float(row["host_time_sec"])
                    out_time = float(row["out_time_sec"])
                except (TypeError, ValueError, KeyError):
                    continue
                if prev_out_time is not None and out_time + 0.5 < prev_out_time:
                    segments.append([])
                try:
                    frame = int(float(row.get("frame", 0)))
                except (TypeError, ValueError):
                    frame = None
                segments[-1].append(
                    {
                        "lsl_time": host_time,
                        "media_time_us": int(out_time * 1_000_000),
                        "frame": frame,
                    }
                )
                prev_out_time = out_time
        for segment in reversed(segments):
            if segment:
                return segment

    if progress_json_path and progress_json_path.exists():
        with progress_json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "media_time_us" in data:
                        media_time_us = int(data["media_time_us"])
                    else:
                        media_time_us = None
                    frame = None
                    if "frame" in data:
                        try:
                            frame = int(data["frame"])
                        except ValueError:
                            frame = None
                    records.append({
                        "lsl_time": float(data.get("lsl_time", 0)),
                        "media_time_us": media_time_us,
                        "frame": frame,
                    })
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

    return records


def median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    med = median(values)
    return median([abs(v - med) for v in values])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    return values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)


def linear_regression_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def compare_timing(session_dir: Path, device_label: str, lsl_prefix: str = "ffmpeg_progress_", emit_json: bool = False) -> dict | None:
    """Compare LSL and frame log timing for a device."""
    lsl_dir = session_dir / "lsl"
    frame_dir = session_dir / "frame_logs"
    
    # Load LSL data
    lsl_candidates = [
        lsl_dir / f"{lsl_prefix}{device_label}.jsonl",
        lsl_dir / f"ffmpeg_progress_{device_label}.jsonl",
        lsl_dir / f"{device_label}.jsonl",
    ]
    lsl_path = next((p for p in lsl_candidates if p.exists()), lsl_candidates[0])
    lsl_data = load_lsl_data(lsl_path)

    # Load anchor stream data (optional)
    anchor_path = lsl_dir / f"{lsl_prefix}{device_label}_anchor.jsonl"
    anchor_data = load_lsl_data(anchor_path) if anchor_path.exists() else []
    
    # Load frame logs
    frame_log_path = frame_dir / f"{device_label}_frames.jsonl"
    frame_logs = load_frame_logs(frame_log_path)

    # Load progress logs
    progress_tsv_path = session_dir / "sourcedata" / "sync" / f"{device_label}_ffmpeg_progress.tsv"
    progress_json_path = session_dir / "progress_logs" / f"{device_label}_progress.jsonl"
    progress_logs = load_progress_logs(progress_tsv_path, progress_json_path)
    
    print("=" * 80)
    print(f"LSL vs Frame Log Comparison: {device_label}")
    print("=" * 80)
    
    if not lsl_data:
        print(f"\n⚠ No LSL data found at {lsl_path}")
    else:
        print(f"\n✓ LSL records: {len(lsl_data)}")
        print(f"  First frame: {lsl_data[0]['frame']}")
        print(f"  Last frame: {lsl_data[-1]['frame']}")
        print(f"  Stream time range: {lsl_data[0]['stream_time']:.3f}s → {lsl_data[-1]['stream_time']:.3f}s")
    
    if not frame_logs:
        print(f"\n⚠ No frame logs found at {frame_log_path}")
    else:
        print(f"\n✓ Frame log records: {len(frame_logs)}")
        print(f"  First frame: {frame_logs[0]['frame']}")
        print(f"  Last frame: {frame_logs[-1]['frame']}")
        print(f"  PTS time range: {frame_logs[0]['pts_time']:.3f}s → {frame_logs[-1]['pts_time']:.3f}s")
        print(f"  Unix time range: {frame_logs[0]['unix_time']:.3f} → {frame_logs[-1]['unix_time']:.3f}")
    
    if not frame_logs:
        print("\n✗ Cannot compare: missing frame logs")
        return None
    
    # Build frame lookup for LSL data
    lsl_by_frame = {rec["frame"]: rec for rec in lsl_data}
    
    # Find matching frames and compute differences
    matches = []
    for flog in frame_logs:
        frame_num = flog["frame"]
        if frame_num in lsl_by_frame:
            lsl_rec = lsl_by_frame[frame_num]
            matches.append({
                "frame": frame_num,
                "pts_time": flog["pts_time"],
                "lsl_clock": flog["lsl_clock"],
                "unix_time": flog["unix_time"],
                "lsl_stream_time": lsl_rec["stream_time"],
                "lsl_timestamp": lsl_rec["lsl_timestamp"],
                "media_time_us": lsl_rec.get("media_time_us"),
            })
    
    if not matches:
        print("\n⚠ No overlapping frames found between LSL and frame logs")
    
    print(f"\n✓ Matched frames: {len(matches)}")
    
    # Analyze timing differences
    print("\n" + "=" * 80)
    print("Timing Analysis")
    print("=" * 80)
    
    # Compare LSL clock from frame log vs LSL stream time
    lsl_diffs = []
    for m in matches:
        # Both are in LSL clock domain
        diff = m["lsl_clock"] - m["lsl_stream_time"]
        lsl_diffs.append(diff)
    
    report: dict = {
        "device": device_label,
        "matched_frames": len(matches),
        "anchor_method": None,
        "median_delay_s": None,
        "mad_s": None,
        "drift_ms_per_min": None,
        "recommended_offset_s": None,
    }

    if lsl_diffs:
        avg_diff = sum(lsl_diffs) / len(lsl_diffs)
        min_diff = min(lsl_diffs)
        max_diff = max(lsl_diffs)
        
        print("\nLSL clock (frame log) vs LSL stream_time (LSL JSONL):")
        print(f"  Average difference: {avg_diff*1000:.1f} ms")
        print(f"  Min difference: {min_diff*1000:.1f} ms")
        print(f"  Max difference: {max_diff*1000:.1f} ms")
        print(f"  Range (drift): {(max_diff - min_diff)*1000:.1f} ms")
        
        if abs(avg_diff) < 0.001:
            print("  ✓ Clocks are well aligned (< 1ms average)")
        elif abs(avg_diff) < 0.05:
            print("  ⚠ Small systematic offset detected")
        else:
            print("  ✗ Significant offset detected")
    
    # Show sample comparison
    print("\n" + "=" * 80)
    print("Sample Frame Comparison (first 10 matches)")
    print("=" * 80)
    print(f"{'Frame':<8} {'PTS (s)':<10} {'LSL_clock':<12} {'LSL_stream':<12} {'Diff (ms)':<10}")
    print("-" * 80)
    
    for m in matches[:10]:
        diff_ms = (m["lsl_clock"] - m["lsl_stream_time"]) * 1000
        print(f"{m['frame']:<8} {m['pts_time']:<10.3f} {m['lsl_clock']:<12.3f} {m['lsl_stream_time']:<12.3f} {diff_ms:>9.1f}")

    # Compare media_time_us vs pts_time if available
    media_matches = [m for m in matches if m.get("media_time_us") is not None]
    if media_matches:
        diffs = [((m["media_time_us"] / 1_000_000) - m["pts_time"]) for m in media_matches]
        avg = sum(diffs) / len(diffs)
        min_d = min(diffs)
        max_d = max(diffs)
        print("\n" + "=" * 80)
        print("Media time vs PTS (anchor streams)")
        print("=" * 80)
        print(f"  Average difference: {avg*1000:.1f} ms")
        print(f"  Min difference: {min_d*1000:.1f} ms")
        print(f"  Max difference: {max_d*1000:.1f} ms")
    
    # Anchor-based analysis (progress logs or anchor streams)
    anchor_candidates = []
    for rec in progress_logs:
        if rec.get("media_time_us") is None:
            continue
        anchor_candidates.append(rec["lsl_time"] - (rec["media_time_us"] / 1_000_000))
    for rec in anchor_data:
        if rec.get("media_time_us") is None:
            continue
        anchor_candidates.append(rec["stream_time"] - (rec["media_time_us"] / 1_000_000))

    if anchor_candidates:
        anchor_start_lsl = median(anchor_candidates)
        delays = [fl["lsl_clock"] - (anchor_start_lsl + fl["pts_time"]) for fl in frame_logs]
        drift_slope = linear_regression_slope(
            [fl["pts_time"] for fl in frame_logs],
            [fl["lsl_clock"] - fl["pts_time"] for fl in frame_logs],
        )

        print("\n" + "=" * 80)
        print("Anchor-based Delay/Drift")
        print("=" * 80)
        med_delay_s = median(delays)
        mad_s = median_abs_deviation(delays)
        print(f"Median delay: {med_delay_s*1000:.1f} ms")
        print(f"MAD: {mad_s*1000:.1f} ms")
        print(
            f"P5/P95: {percentile(delays, 0.05)*1000:.1f} / {percentile(delays, 0.95)*1000:.1f} ms"
        )
        drift_ms_min = drift_slope*1000*60 if drift_slope is not None else None
        if drift_ms_min is not None:
            print(f"Drift: {drift_ms_min:.2f} ms/min")
        else:
            print("Drift: insufficient data")

        # Recommended offset: trim by -median_delay to correct early/late stream
        # If median delay is negative (stream early), offset is positive trim.
        report.update({
            "anchor_method": "anchor",
            "median_delay_s": float(med_delay_s),
            "mad_s": float(mad_s),
            "drift_ms_per_min": float(drift_ms_min) if drift_ms_min is not None else None,
            "recommended_offset_s": float(-med_delay_s),
        })
    else:
        # Fallback: estimate offset from frame log start (unix_time - pts_time) stability
        starts = [fl["unix_time_s"] - fl["pts_time"] for fl in frame_logs if fl.get("unix_time_s")]
        if starts:
            med_start = median(starts)
            mad = median_abs_deviation(starts)
            print("\n" + "=" * 80)
            print("Frame-log Start Offset (fallback)")
            print("=" * 80)
            print(f"Median start (unix-pts): {med_start:.3f} s")
            print(f"MAD: {mad:.3f} s")
            # Without a cross-device anchor, set recommended_offset_s to 0 here; the
            # multi-device alignment uses relative starts.
            report.update({
                "anchor_method": "frame",
                "median_delay_s": None,
                "mad_s": float(mad),
                "recommended_offset_s": 0.0,
            })

    if emit_json:
        print("\nJSON:")
        print(json.dumps(report, indent=2))
        return report
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare LSL stream timing with frame log timing"
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Session directory (e.g., data/sub-001/ses-001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device label (e.g., jabra_panacast_20_vid)",
    )
    parser.add_argument(
        "--lsl-prefix",
        type=str,
        default="ffmpeg_progress_",
        help="LSL stream file prefix (default: ffmpeg_progress_)",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Emit JSON with recommended offset and quality metrics",
    )
    
    args = parser.parse_args()
    
    if not args.session.exists():
        print(f"Session directory not found: {args.session}")
        return 1
    
    compare_timing(args.session, args.device, args.lsl_prefix, emit_json=args.emit_json)
    return 0


if __name__ == "__main__":
    exit(main())
