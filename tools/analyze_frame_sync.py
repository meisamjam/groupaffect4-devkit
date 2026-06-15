"""Analyze frame log synchronization accuracy across devices.

Computes start offsets between devices using unix_time anchors and
reports maximum offset to verify sub-10ms synchronization.
"""

import json
from pathlib import Path


def analyze_sync(frame_dir: Path):
    """Analyze synchronization from frame logs."""
    frame_logs = list(frame_dir.glob("*_frames.jsonl"))
    
    if not frame_logs:
        print(f"No frame logs found in {frame_dir}")
        return
    
    # Collect video start times (unix_time at pts_time=0)
    starts = {}
    
    for log_path in frame_logs:
        device = log_path.stem.replace("_frames", "")
        
        # Read first frame
        with log_path.open() as f:
            first_line = f.readline().strip()
            if not first_line:
                print(f"⚠️  {device}: empty log")
                continue
                
            data = json.loads(first_line)
            pts_time = data["pts_time"]
            unix_time = data["unix_time"]
            
            # Compute when pts_time=0 occurred
            video_start = unix_time - pts_time
            starts[device] = video_start
            
            print(f"📹 {device:30s} start: {video_start:.9f} (first pts={pts_time:.6f})")
    
    if len(starts) < 2:
        print("\n⚠️  Need at least 2 devices to compute sync offset")
        return
    
    # Compute offsets relative to earliest
    base = min(starts.values())
    offsets = {dev: (t - base) * 1000 for dev, t in starts.items()}  # Convert to ms
    
    print("\n📊 Synchronization Analysis (relative to earliest device):")
    print("=" * 70)
    
    for dev in sorted(offsets.keys()):
        offset_ms = offsets[dev]
        status = "✅" if offset_ms < 10 else "⚠️"
        print(f"{status} {dev:30s}: {offset_ms:8.3f} ms")
    
    max_offset = max(offsets.values())
    min_offset = min(offsets.values())
    spread = max_offset - min_offset
    
    print("=" * 70)
    print(f"Max offset: {max_offset:.3f} ms")
    print(f"Min offset: {min_offset:.3f} ms")
    print(f"Spread:     {spread:.3f} ms")
    
    if spread < 10:
        print(f"\n✅ SUCCESS: All devices within {spread:.3f} ms (target: <10ms)")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: {spread:.3f} ms spread (target: <10ms)")
        print("\nRecommendations:")
        print("  1. Ensure all devices use wall-clock timestamps (-use_wallclock_as_timestamps 1)")
        print("  2. Check that processes start in truly parallel threads (no sequential delay)")
        print("  3. Verify system time synchronization (NTP) across capture machines")
        print("  4. Consider hardware sync signals (genlock) for sub-millisecond accuracy")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze frame log synchronization")
    parser.add_argument(
        "--session",
        type=Path,
        default=Path("data/sub-001/ses-001"),
        help="Session directory containing frame_logs/",
    )
    
    args = parser.parse_args()
    frame_dir = args.session / "frame_logs"
    
    analyze_sync(frame_dir)
