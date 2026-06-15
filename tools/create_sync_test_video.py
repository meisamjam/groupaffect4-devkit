 #!/usr/bin/env python3
"""
Sync video module for frame timestamp synchronization.

This module provides helper functions to extract and synchronize video frame
timestamps with LSL (LabStreamingLayer) timestamps. If specific sync information
is not available, functions return empty lists, allowing the caller to fall back
to alternative sync methods (progress TSV, event logs, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_lsl_anchor_candidates(lsl_path: Path | str) -> list[float]:
    """
    Load LSL timestamp anchors from a JSONL file.

    Args:
        lsl_path: Path to LSL JSONL file (e.g., ffmpeg_progress_<label>.jsonl)

    Returns:
        List of float timestamps. Returns empty list if file is missing,
        unreadable, or contains no valid timestamps.
    """
    lsl_path = Path(lsl_path)
    if not lsl_path.exists():
        return []

    try:
        anchors: list[float] = []
        with lsl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Try to extract timestamp from LSL record
                    if "timestamp" in record and isinstance(record["timestamp"], (int, float)):
                        anchors.append(float(record["timestamp"]))
                    elif "server_received_lsl" in record and isinstance(record["server_received_lsl"], (int, float)):
                        anchors.append(float(record["server_received_lsl"]))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        return anchors
    except Exception:
        return []


def load_frame_log_start_estimate(
    frame_log_path: Path | str, max_samples: int = 30, min_valid: int = 3
) -> float | None:
    """Estimate video start unix time from frame log samples.

    Computes start_time = unix_time_s - pts_time across multiple samples and
    returns the median to reduce jitter.  Returns None if insufficient samples.

    Args:
        frame_log_path: Path to frame log JSONL file (e.g., <label>_frames.jsonl)
        max_samples: Maximum number of frame-log samples to use for estimation.
        min_valid: Minimum number of valid samples required.

    Returns:
        Median Unix-epoch start timestamp (seconds), or None if unavailable.
    """
    frame_log_path = Path(frame_log_path)
    if not frame_log_path.exists():
        return None

    starts: list[float] = []
    try:
        with frame_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(starts) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pts_time = record.get("pts_time")
                    unix_time = record.get("unix_time_s") or record.get("unix_time")
                    if pts_time is None or unix_time is None:
                        continue
                    pts_time = float(pts_time)
                    unix_time = float(unix_time)
                    if unix_time <= 0:
                        continue
                    starts.append(unix_time - pts_time)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except Exception:
        return None

    if len(starts) < min_valid:
        return None

    starts.sort()
    mid = len(starts) // 2
    if len(starts) % 2 == 1:
        return starts[mid]
    return (starts[mid - 1] + starts[mid]) / 2.0
