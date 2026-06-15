"""Analyze video/audio synchronization using LSL timestamps.

Compares LSL stream timestamps from device LSL streams to verify synchronization.
Useful for debugging timing drift and validating multi-device capture alignment.

Usage:
    python tools/analyze_sync.py --session data/sub-001/ses-001
"""

import argparse
import json
from pathlib import Path


def load_events(events_file: Path) -> list[dict]:
    """Load events from JSONL file."""
    events: list[dict] = []
    if events_file.exists():
        with events_file.open() as f:
            for line in f:
                events.append(json.loads(line))
    return events


def _latest_run_start_times(events: list[dict]) -> dict[str, str]:
    """Return the most recent capture_started timestamps per device from the latest run.

    Runs are segmented by gaps > 5 minutes between capture_started events.
    This avoids mixing multiple runs in a single report (which can yield huge offsets).
    """
    starts = [e for e in events if e.get("event_type") == "capture_started" and e.get("timestamp")]
    if not starts:
        return {}

    # Sort chronologically
    starts.sort(key=lambda e: e["timestamp"])

    # Segment into runs: gap > 300s => new run
    runs: list[list[dict]] = []
    current: list[dict] = []
    prev_ts = None
    for e in starts:
        ts = e["timestamp"]
        if prev_ts is None:
            current.append(e)
            prev_ts = ts
            continue
        try:
            from datetime import datetime

            delta = datetime.fromisoformat(ts) - datetime.fromisoformat(prev_ts)
            gap = delta.total_seconds()
        except Exception:
            gap = 0

        if gap > 300 and current:
            runs.append(current)
            current = [e]
        else:
            current.append(e)
        prev_ts = ts

    if current:
        runs.append(current)

    latest_run = runs[-1]

    # Keep the latest start per device within the latest run (in case of retries)
    latest_per_device: dict[str, str] = {}
    for e in latest_run:
        dev = e.get("device_id")
        ts = e.get("timestamp")
        if not dev or not ts:
            continue
        latest_per_device[dev] = ts

    return latest_per_device


def analyze_synchronization(session_dir: Path) -> None:
    """Analyze synchronization across captured streams."""
    events_file = session_dir / "video" / "ffmpeg_multicap_events.jsonl"
    
    if not events_file.exists():
        print(f"No events file found: {events_file}")
        return
    
    events = load_events(events_file)

    # Restrict to latest run start times to avoid mixing multiple sessions
    latest_starts = _latest_run_start_times(events)

    # Group events by device, keeping only latest run start times
    devices: dict[str, dict] = {}
    for event in events:
        device_id = event.get("device_id")
        event_type = event.get("event_type")
        ts = event.get("timestamp")
        if not device_id:
            continue

        if device_id not in devices:
            devices[device_id] = {"started": None, "completed": None}

        if event_type == "capture_started" and ts == latest_starts.get(device_id):
            devices[device_id]["started"] = ts
        elif event_type == "capture_completed" and devices[device_id].get("started"):
            # keep completion if it follows a tracked start in latest run
            devices[device_id]["completed"] = ts
    
    # Display synchronization report
    print("\n" + "=" * 80)
    print("Capture Synchronization Report")
    print("=" * 80)
    
    if not devices:
        print("No capture events found")
        return
    
    # Find first start time for reference
    start_times = [dev["started"] for dev in devices.values() if dev["started"]]
    if not start_times:
        print("No start events found")
        return
    
    reference_time = min(start_times)
    
    print(f"\nReference start time: {reference_time}")
    print("\nDevice timing (relative to first device):")
    print(f"{'Device':<40} {'Start Offset':<15} {'Duration':<15} {'Status':<10}")
    print("-" * 80)
    
    for device_id, times in sorted(devices.items()):
        if times["started"]:
            from datetime import datetime
            ref_dt = datetime.fromisoformat(reference_time)
            start_dt = datetime.fromisoformat(times["started"])
            offset_ms = (start_dt - ref_dt).total_seconds() * 1000
            
            duration = "N/A"
            status = "Running" if not times["completed"] else "Completed"
            
            if times["completed"]:
                end_dt = datetime.fromisoformat(times["completed"])
                duration_sec = (end_dt - start_dt).total_seconds()
                duration = f"{duration_sec:.2f}s"
            
            print(f"{device_id:<40} {offset_ms:>10.1f} ms   {duration:<15} {status:<10}")
        else:
            print(f"{device_id:<40} {'N/A':<15} {'N/A':<15} {'No start':<10}")
    
    # Synchronization assessment
    print("\n" + "=" * 80)
    print("Synchronization Assessment")
    print("=" * 80)
    
    offsets = []
    for _device_id, times in devices.items():
        if times["started"]:
            from datetime import datetime
            ref_dt = datetime.fromisoformat(reference_time)
            start_dt = datetime.fromisoformat(times["started"])
            offset_ms = abs((start_dt - ref_dt).total_seconds() * 1000)
            offsets.append(offset_ms)
    
    if offsets:
        max_offset = max(offsets)
        print(f"\nMaximum start time difference: {max_offset:.1f} ms")
        
        if max_offset < 100:
            print("✓ Excellent synchronization (< 100 ms)")
        elif max_offset < 500:
            print("⚠ Good synchronization (< 500 ms)")
        elif max_offset < 1000:
            print("⚠ Acceptable synchronization (< 1 second)")
        else:
            print("✗ Poor synchronization (> 1 second)")

        print("\nNote: For frame-accurate sync analysis, check LSL stream timestamps")
        print("      LSL streams provide microsecond-precision timing for each frame/sample")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze synchronization across captured video/audio streams"
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Session directory (e.g., data/sub-001/ses-001)",
    )
    
    args = parser.parse_args()
    
    if not args.session.exists():
        print(f"Session directory not found: {args.session}")
        return 1
    
    analyze_synchronization(args.session)
    return 0


if __name__ == "__main__":
    exit(main())
