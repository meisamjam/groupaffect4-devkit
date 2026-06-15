#!/usr/bin/env python3
"""
Charuco-based spatial camera calibration for FreeMoCap post-hoc processing.

This tool calibrates multi-camera setups (Jabra PanaCast etc.) so that
FreeMoCap can triangulate 3D skeleton data from pre-recorded videos.

Workflow
--------
1. Print charuco board:
     python tools/calibrate_charuco.py print-board --output charuco_board.png

2. Record calibration videos (wave the board in front of all cameras):
     python tools/calibrate_charuco.py record \\
         --config configs/ffmpeg_multicap.json --duration 60

3. Run anipose calibration on recorded charuco videos:
     python tools/calibrate_charuco.py calibrate \\
         --videos-dir data/calibration/charuco_YYYYMMDD_HHMMSS/video \\
         --square-size 39

   For better results with wide-angle cameras (Jabra PanaCast), seed the
   calibration with known focal lengths from camera specs:
     python tools/calibrate_charuco.py calibrate \\
         --videos-dir data/calibration/charuco_YYYYMMDD_HHMMSS/video \\
         --square-size 39 --init-focal

4. (Optional) Ground-plane calibration — board lying flat on the table:
     python tools/calibrate_charuco.py ground-plane \\
         --videos-dir data/calibration/charuco_YYYYMMDD_HHMMSS/video \\
         --toml calibration_charuco.toml --square-size 39

5. Validate calibration quality:
     python tools/calibrate_charuco.py validate \\
         --toml calibration_charuco.toml

The output .toml file can then be used for FreeMoCap 3D reconstruction.

Dependencies
------------
Requires: freemocap>=1.3.0, opencv-python>=4.8
  conda activate affectai-freemocap
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# NOTE: Do NOT call logging.basicConfig() at module level.
# FreeMoCap's configure_logging() has a circular import bug:
# if root logger already has handlers, it does `from freemocap import logger`
# while freemocap.__init__ is still loading → ImportError.
# We configure logging lazily inside main() instead.
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging (call AFTER freemocap imports, or for non-freemocap commands)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHARUCO_BOARDS = {
    "5x3": {"width": 5, "height": 3, "name": "5x3 Charuco"},
    "7x5": {"width": 7, "height": 5, "name": "7x5 Charuco"},
}
# 7x5 recommended for wide-angle cameras (Jabra) — more corners = better
# coverage across the full field of view.  5x3 is fine for narrow-angle setups.
DEFAULT_BOARD = "7x5"
DEFAULT_SQUARE_SIZE_MM = 39  # Measure YOUR printed board's black square edge!
DEFAULT_CALIBRATION_DIR = Path("data/calibration")
DEFAULT_RECORD_DURATION_S = 75
DEFAULT_BEEP_INTERVAL_S = 15
BOARD_MODE_AUTO = "auto"
BOARD_MODE_CHOICES = [*CHARUCO_BOARDS.keys(), BOARD_MODE_AUTO]

# Video files to exclude from calibration (known non-calibration outputs)
_EXCLUDE_PATTERNS = {"sync_grid", "sync_grid_cfr", "combined", "mosaic"}


# ---------------------------------------------------------------------------
# Duplicate-marker-ID filtering (charuco + table ArUco coexistence)
# ---------------------------------------------------------------------------

def _filter_duplicate_marker_ids(
    corners: tuple,
    ids: np.ndarray,
    board_marker_ids: set[int],
) -> tuple[tuple, np.ndarray]:
    """Resolve duplicate ArUco marker IDs caused by non-charuco markers
    (e.g. table tracking markers from ``DICT_4X4_50``) sharing the same
    dictionary and ID range as the charuco board.

    Strategy: keep only markers whose IDs are on the board.  When an ID
    appears more than once, pick the instance that is closest to the
    spatial centroid of the tightest cluster containing all required IDs.

    Returns filtered ``(corners, ids)`` ready for
    ``cv2.aruco.interpolateCornersCharuco``.
    """
    if ids is None or len(ids) == 0:
        return corners, ids

    flat_ids = ids.ravel()
    # Quick path: no duplicates among board IDs → nothing to do
    board_mask = np.array([mid in board_marker_ids for mid in flat_ids])
    board_flat = flat_ids[board_mask]
    if len(board_flat) == len(set(board_flat)):
        # Keep only board-ID markers, no duplicates
        idx = np.where(board_mask)[0]
        return tuple(corners[i] for i in idx), ids[idx]

    # -- Duplicate board IDs detected → spatial clustering --
    # Collect per-ID candidates: (original_index, centroid_xy)
    from collections import defaultdict
    candidates: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for i, mid in enumerate(flat_ids):
        if mid in board_marker_ids:
            cx, cy = corners[i][0].mean(axis=0)
            candidates[mid].append((i, np.array([cx, cy])))

    # Find the cluster of markers (one per board ID) with smallest bbox
    # For efficiency, use greedy approach: pick the majority cluster via
    # centroid proximity.
    # Step 1: compute centroid of ALL board-ID marker positions
    all_pts = []
    all_idx_mid = []
    for mid, cands in candidates.items():
        for orig_i, pt in cands:
            all_pts.append(pt)
            all_idx_mid.append((orig_i, mid))
    all_pts_arr = np.array(all_pts)

    # Step 2: for each candidate centroid, score how many unique IDs
    # cluster around it within a radius.  Pick the centroid that covers
    # the most unique IDs with smallest spread.
    best_selection: dict[int, int] = {}
    # Simple approach: for each ID with duplicates, pick the instance
    # closest to the median position of all single-instance IDs.
    single_pts = []
    for mid, cands in candidates.items():
        if len(cands) == 1:
            single_pts.append(cands[0][1])
            best_selection[mid] = cands[0][0]
    if single_pts:
        ref = np.median(single_pts, axis=0)
    else:
        ref = np.median(all_pts_arr, axis=0)

    for mid, cands in candidates.items():
        if mid in best_selection:
            continue
        # Pick candidate closest to reference centroid
        dists = [np.linalg.norm(pt - ref) for _, pt in cands]
        best_i = int(np.argmin(dists))
        best_selection[mid] = cands[best_i][0]

    keep_idx = sorted(best_selection.values())
    return tuple(corners[i] for i in keep_idx), ids[keep_idx]


def _detect_charuco_robust(
    gray: np.ndarray,
    aruco_detector: cv2.aruco.ArucoDetector,
    board: cv2.aruco.CharucoBoard,
    board_marker_ids: set[int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect charuco corners with duplicate-marker-ID filtering.

    Works around the issue where non-charuco ArUco markers (e.g. table
    tracking markers) share the same dictionary and ID range, confusing
    ``CharucoDetector.detectBoard()``.

    Returns ``(charuco_corners, charuco_ids)`` or ``(None, None)``.
    """
    # Step 1: detect all ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None

    # Step 2: filter duplicate / non-board marker IDs
    filt_corners, filt_ids = _filter_duplicate_marker_ids(
        corners, ids, board_marker_ids,
    )
    if filt_ids is None or len(filt_ids) == 0:
        return None, None

    # Step 3: interpolate charuco corners from filtered markers
    ret_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        filt_corners, filt_ids, gray, board,
    )
    if ret_count > 0 and charuco_ids is not None:
        return charuco_corners, charuco_ids
    return None, None


def _board_marker_id_set(spec: dict) -> set[int]:
    """Return the set of ArUco marker IDs used by a charuco board.

    For a W×H charuco board, markers fill the 'white' squares
    of the checkerboard pattern.  Total markers = floor(W*H / 2).
    IDs are sequential starting from 0.
    """
    n_markers = (spec["width"] * spec["height"]) // 2
    return set(range(n_markers))


# ---------------------------------------------------------------------------
# Camera-spec helpers (for --init-focal)
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_SPECS = Path(__file__).resolve().parent.parent / "configs" / "camera_specs.json"


def _load_camera_specs(path: Path) -> dict[str, Any]:
    """Load camera specifications from a JSON file.

    Returns the parsed dict with keys "models" and "camera_name_patterns".
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _match_camera_model(
    camera_name: str,
    specs: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the model spec dict for *camera_name*, or ``None`` if no
    pattern matches.

    Patterns in ``specs["camera_name_patterns"]`` are tested as
    case-insensitive full-match regexes (anchored with ``$``).

    Per-camera overrides in ``specs["camera_overrides"]`` are applied on
    top of the base model when a key (substring match, case-insensitive)
    is found in *camera_name*. This allows cameras with non-default firmware
    settings (e.g., Intelligent Zoom active, manual focus locked, wide-angle
    mode) to declare different expected focal lengths without forking the
    whole model entry.
    """
    patterns = specs.get("camera_name_patterns", {})
    models = specs.get("models", {})
    base_model: dict[str, Any] | None = None
    for pattern, model_key in patterns.items():
        if pattern.startswith("_"):
            continue  # skip comment keys
        if re.match(pattern + "$", camera_name, re.IGNORECASE):
            base_model = models.get(model_key)
            break

    if base_model is None:
        return None

    # Apply per-camera overrides (substring match on camera_name)
    name_lower = camera_name.lower()
    overrides_map = specs.get("camera_overrides", {})
    matched_override: dict[str, Any] = {}
    for override_key, override_vals in overrides_map.items():
        if override_key.startswith("_"):
            continue
        if override_key.lower() in name_lower:
            for k, v in override_vals.items():
                if not k.startswith("_"):
                    matched_override[k] = v
            break  # first match wins

    return {**base_model, **matched_override} if matched_override else base_model


def _build_intrinsic_matrix(
    model: dict[str, Any],
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix from model specs.

    Uses ``expected_fx_1080p`` / ``expected_fy_1080p`` if present, otherwise
    computes from ``hfov_deg`` and the given resolution.

    The principal point is placed at the image centre.
    """
    # Focal length —— prefer pre-computed, else derive from HFOV
    if "expected_fx_1080p" in model:
        fx = float(model["expected_fx_1080p"]) * (width / 1920.0)
        fy = float(model.get("expected_fy_1080p", model["expected_fx_1080p"])) * (height / 1080.0)
    elif "hfov_deg" in model:
        hfov_rad = float(model["hfov_deg"]) * math.pi / 180.0
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx * (height / width) * (width / height)  # square pixels → fy = fx
    else:
        # Fallback: identity-like — no improvement over anipose default
        fx = fy = float(width)

    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _find_ffmpeg() -> str:
    """Return the path to an ffmpeg binary.

    Resolution order:
      1. ``ffmpeg`` on PATH (system install)
      2. ``imageio_ffmpeg.get_ffmpeg_exe()`` (bundled with imageio)
    """
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"  # last resort — let subprocess raise if missing


def _detect_h264_encoder() -> list[str]:
    """Return ffmpeg encoder args for the best available H.264 encoder.

    Preference order:
      1. libx264 (GPL build) — best quality/control
      2. h264_mf (MediaFoundation, Windows conda-forge) — HW-accelerated
      3. mjpeg fallback — always available
    """
    try:
        result = subprocess.run(
            [_find_ffmpeg(), "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        encoders = result.stdout + result.stderr
    except Exception:
        encoders = ""

    if "libx264" in encoders:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if "h264_mf" in encoders:
        # MediaFoundation H.264 — quality via bitrate (no CRF support)
        return ["-c:v", "h264_mf", "-b:v", "8M"]
    # Last resort: MJPEG (large files, but universally available)
    return ["-c:v", "mjpeg", "-q:v", "3"]


def _discover_video_files(videos_dir: Path) -> list[Path]:
    """Find video files in directory, excluding known non-camera files."""
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    return sorted(
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in video_exts
        and p.stem.lower() not in _EXCLUDE_PATTERNS
        and not p.stem.lower().startswith("sync_grid")
    )


def _load_structured_config(path: Path) -> dict[str, Any]:
    """Load JSON/YAML config file into a dict."""
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    # Fallback: try JSON then YAML.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_square_size_mm(board_def: dict[str, Any]) -> float | None:
    """Extract square size in mm from a board definition."""
    if "square_size_mm" in board_def:
        try:
            return float(board_def["square_size_mm"])
        except (TypeError, ValueError):
            return None
    if "square_size_m" in board_def:
        try:
            return float(board_def["square_size_m"]) * 1000.0
        except (TypeError, ValueError):
            return None
    return None


def _load_multicam_sync_offsets(videos_dir: Path) -> dict[str, float]:
    """Load per-camera sync offsets (seconds) from ffmpeg_multicap_events.jsonl.
    
    Reads capture_started timestamps and computes relative offsets (in seconds)
    for each camera so that temporal alignment accounts for staggered start times.
    
    Returns:
        Dict mapping video filename stem to offset in seconds (0.0 for earliest camera).
        Empty dict if events file not found.
    """
    events_file = videos_dir / "ffmpeg_multicap_events.jsonl"
    if not events_file.exists():
        return {}
    
    offsets: dict[str, float] = {}
    capture_starts: dict[str, float] = {}
    
    try:
        with events_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("event_type") != "capture_started":
                        continue
                    
                    device_id = rec.get("device_id", "")
                    if not device_id:
                        continue
                    
                    unix_time_s = float(rec.get("unix_time_s", 0))
                    if unix_time_s <= 0:
                        continue
                    
                    capture_starts[device_id] = unix_time_s
                except Exception:
                    continue
    except Exception:
        return {}
    
    if not capture_starts:
        return {}
    
    # Find earliest start time as reference
    ref_time = min(capture_starts.values())
    
    # Compute offset for each camera (in seconds, relative to earliest)
    for device_id, start_time in capture_starts.items():
        offset_s = start_time - ref_time
        # Extract the video label (remove common suffixes)
        label = device_id
        for suffix in ["_vid", "_audio"]:
            if label.endswith(suffix):
                label = label[:-len(suffix)]
        offsets[label] = offset_s
    
    if offsets:
        print("  Sync offsets from ffmpeg_multicap_events.jsonl:")
        for label, offset_s in sorted(offsets.items()):
            print(f"    {label:40s} : {offset_s:+.6f}s ({offset_s*1000:+.2f}ms)")
    
    return offsets


def _load_calibration_options(config_path: Path) -> dict[str, Any]:
    """Load calibration board + recording options from config.

    Supported schemas:
    - calibration.boards: [{board_type, square_size_mm|square_size_m}, ...]
    - calibration_boards: [{...}, ...]
    - fixed_charuco_board: {board_type, square_size_m|square_size_mm}
    - calibration.record_duration_s / beep_interval_s / board_setting
    """
    raw = _load_structured_config(config_path)
    if not raw:
        return {
            "board_sizes_mm": {},
            "record_duration_s": None,
            "beep_interval_s": None,
            "board_setting": None,
        }

    calibration_cfg = raw.get("calibration", {}) if isinstance(raw.get("calibration", {}), dict) else {}

    board_defs: list[dict[str, Any]] = []

    cal_boards = calibration_cfg.get("boards")
    if isinstance(cal_boards, list):
        board_defs.extend([b for b in cal_boards if isinstance(b, dict)])

    root_boards = raw.get("calibration_boards")
    if isinstance(root_boards, list):
        board_defs.extend([b for b in root_boards if isinstance(b, dict)])

    fixed_board = raw.get("fixed_charuco_board")
    if isinstance(fixed_board, dict):
        board_defs.append(fixed_board)

    board_sizes_mm: dict[str, float] = {}
    for bd in board_defs:
        board_type = bd.get("board_type")
        if not isinstance(board_type, str):
            width = bd.get("board_width")
            height = bd.get("board_height")
            if isinstance(width, int) and isinstance(height, int):
                board_type = f"{width}x{height}"
        if not isinstance(board_type, str):
            continue
        if board_type not in CHARUCO_BOARDS:
            continue
        sq = _read_square_size_mm(bd)
        if sq is None:
            continue
        board_sizes_mm[board_type] = float(sq)

    return {
        "board_sizes_mm": board_sizes_mm,
        "record_duration_s": calibration_cfg.get("record_duration_s"),
        "beep_interval_s": calibration_cfg.get("beep_interval_s"),
        "board_setting": calibration_cfg.get("board_setting"),
    }


def _stop_process_tree(process: subprocess.Popen, timeout_s: float = 20.0) -> None:
    """Stop recorder process and all children (Windows-safe)."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        # ffmpeg_multicap traps CTRL_BREAK_EVENT and stops child ffmpeg captures.
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=timeout_s)
            return
        except Exception:
            pass

        # Fallback hard kill process tree.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        return

    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _emit_beep() -> None:
    """Play a short cue beep (Windows first, terminal bell fallback)."""
    try:
        import winsound  # type: ignore

        winsound.Beep(1200, 180)
        return
    except Exception:
        pass
    print("\a", end="", flush=True)


def _run_timed_recording_with_beeps(
    process: subprocess.Popen,
    duration_s: int,
    beep_interval_s: int,
    cue_text: str,
) -> None:
    """Run a timed recording and emit periodic cue beeps."""
    start = time.monotonic()
    next_beep = beep_interval_s if beep_interval_s > 0 else None

    _emit_beep()  # start cue
    print("[BEEP] START - begin board motion now")

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"ffmpeg_multicap exited early (rc={process.returncode})"
            )

        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break

        if next_beep is not None and elapsed >= next_beep:
            print(f"[BEEP] t={int(next_beep)}s - {cue_text}")
            _emit_beep()
            next_beep += beep_interval_s

        time.sleep(0.2)

    # completion cue
    _emit_beep()
    _emit_beep()
    print("[BEEP] COMPLETE - stopping recording")


def _score_board_spec(
    video_files: list[Path],
    board_spec: dict[str, Any],
    sample_frames: int = 8,
    max_videos: int = 3,
) -> tuple[int, int]:
    """Return (total_detected_corners, hit_frames) for a board spec."""
    if not video_files:
        return 0, 0

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=[board_spec["width"], board_spec["height"]],
        squareLength=1,
        markerLength=0.8,
        dictionary=aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    board_ids = _board_marker_id_set(board_spec)

    score = 0
    hits = 0

    for vf in video_files[:max_videos]:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
        step = max(1, total_frames // sample_frames)

        for fi in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids = _detect_charuco_robust(
                gray,
                aruco_detector,
                board,
                board_ids,
            )
            if charuco_ids is not None and len(charuco_ids) > 0:
                hits += 1
                score += int(len(charuco_ids))

        # MKV GOP structure makes seek unreliable — if seek produced nothing,
        # fall back to sequential reading (same sample count).
        if hits == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            seq_idx = 0
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    charuco_corners, charuco_ids = _detect_charuco_robust(
                        gray,
                        aruco_detector,
                        board,
                        board_ids,
                    )
                    if charuco_ids is not None and len(charuco_ids) > 0:
                        hits += 1
                        score += int(len(charuco_ids))
                    seq_idx += 1
                    if seq_idx >= sample_frames:
                        break

        cap.release()

    return score, hits


def _resolve_board_spec(
    board_type: str,
    video_files: list[Path] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve board type, including auto-selection across 5x3 and 7x5."""
    if board_type != BOARD_MODE_AUTO:
        return board_type, CHARUCO_BOARDS[board_type]

    files = video_files or []
    if not files:
        logger.warning("--board-type auto requested without videos; falling back to %s", DEFAULT_BOARD)
        return DEFAULT_BOARD, CHARUCO_BOARDS[DEFAULT_BOARD]

    scores: dict[str, tuple[int, int]] = {}
    for candidate in CHARUCO_BOARDS:
        scores[candidate] = _score_board_spec(files, CHARUCO_BOARDS[candidate])

    best = max(
        scores.items(),
        key=lambda item: (item[1][0], item[1][1]),
    )[0]
    score, hits = scores[best]
    logger.info("Auto board select: %s (score=%d, hits=%d); alternatives=%s", best, score, hits, scores)
    if score == 0 and hits == 0:
        logger.warning("Auto board detection found no corners; using default board %s", DEFAULT_BOARD)
        return DEFAULT_BOARD, CHARUCO_BOARDS[DEFAULT_BOARD]
    return best, CHARUCO_BOARDS[best]


# ===================================================================
# Sub-command: print-board
# ===================================================================
def cmd_print_board(args: argparse.Namespace) -> None:
    """Generate and save a printable Charuco board image."""
    _setup_logging()
    spec = CHARUCO_BOARDS[args.board_type]
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=[spec["width"], spec["height"]],
        squareLength=1,
        markerLength=0.8,
        dictionary=aruco_dict,
    )
    img = board.generateImage((args.width, args.height))
    out = Path(args.output)
    cv2.imwrite(str(out), img)

    corners = (spec["width"] - 1) * (spec["height"] - 1)
    print(f"\nCharuco board saved  : {out}")
    print(f"  Type               : {spec['name']}")
    print(f"  Grid               : {spec['width']} x {spec['height']} squares")
    print(f"  Inner corners      : {corners}")
    print(f"  ArUco dictionary   : DICT_4X4_250")
    print(f"  Image size (px)    : {args.width} x {args.height}")
    print()
    print("IMPORTANT:")
    print("  1. Print this image WITHOUT any scaling (100%, no fit-to-page).")
    print("  2. Mount on a RIGID, FLAT surface (cardboard, foam board).")
    print("  3. Measure the black square edge length in mm and pass it")
    print("     as --square-size when running 'calibrate'.")
    print()
    print("Recommended: print on A3/tabloid for larger capture volumes.")


# ===================================================================
# Sub-command: record
# ===================================================================
def cmd_record(args: argparse.Namespace) -> None:
    """Record charuco calibration videos using the existing ffmpeg pipeline."""
    _setup_logging()
    output_dir = Path(args.output_dir)
    session_hint = output_dir / "charuco_recording"

    # Load and patch config to point at calibration session dir
    config = json.loads(Path(args.config).read_text())
    config["session_dir"] = str(session_hint)

    # Optional board/recording defaults from config file.
    board_cfg_path = Path(args.board_config) if args.board_config else Path(args.config)
    cal_opts = _load_calibration_options(board_cfg_path)
    board_sizes_mm = cal_opts.get("board_sizes_mm", {})
    if args.duration == DEFAULT_RECORD_DURATION_S and cal_opts.get("record_duration_s") is not None:
        args.duration = int(cal_opts["record_duration_s"])
    if args.beep_interval == DEFAULT_BEEP_INTERVAL_S and cal_opts.get("beep_interval_s") is not None:
        args.beep_interval = int(cal_opts["beep_interval_s"])
    if args.board_setting == "two-board" and isinstance(cal_opts.get("board_setting"), str):
        args.board_setting = str(cal_opts["board_setting"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    temp_config = output_dir / f"_tmp_charuco_config_{timestamp}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_config.write_text(json.dumps(config, indent=2))

    print()
    print("=" * 60)
    print("CHARUCO CALIBRATION RECORDING")
    print("=" * 60)
    print(f"  Output root : {output_dir}")
    print(f"  Group ID    : {args.group_id}")
    print(f"  Duration    : {args.duration}s")
    print(f"  Beep every  : {args.beep_interval}s")
    print(f"  Board mode  : {args.board_setting}")
    print(f"  Config      : {args.config}")
    print(f"  Board cfg   : {board_cfg_path}")
    if board_sizes_mm:
        print("  Board sizes : " + ", ".join(f"{k}={v:.1f}mm" for k, v in sorted(board_sizes_mm.items())))
    print()
    print("INSTRUCTIONS:")
    print("  1. Hold the printed ChArUco board in view of ALL cameras.")
    print("  2. At each beep, move/rotate the DYNAMIC board to a new pose.")
    print("  3. Keep board visible in at least TWO cameras simultaneously.")
    print("  4. Cover entire volume: up/down/left/right/near/far and tilt angles.")
    if args.board_setting == "two-board":
        if board_sizes_mm:
            print("  5. Two-board mode: alternate boards using config sizes above.")
        else:
            print("  5. Two-board mode: alternate between 5x3 and 7x5 boards during cues.")
    elif args.board_setting == "fixed":
        print("  5. Fixed-board mode: keep one board type, vary only pose/position.")
    print()
    input("Press ENTER when ready to start recording...")

    cmd = [
        sys.executable,
        "tools/ffmpeg_multicap.py",
        "--config", str(temp_config),
        "--group-id", args.group_id,
        "--frame-log",
        "--record-lsl",
        "--stabilization-delay", "2.0",
        "--sequential-start-delay", "0.3",
        "--lsl-stream-name", "ffmpeg_clock",
        "--lsl-rate", "100",
        "--lsl-prefixes", "ffmpeg_progress_",
        "--enable-markers",
    ]

    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(cmd, **popen_kwargs)

    try:
        logger.info("Recording for %s seconds with cue beeps every %s seconds...", args.duration, args.beep_interval)
        _run_timed_recording_with_beeps(
            process,
            duration_s=args.duration,
            beep_interval_s=args.beep_interval,
            cue_text="move dynamic board to a new pose",
        )
    finally:
        logger.info("Stopping recording...")
        _stop_process_tree(process, timeout_s=20.0)

    temp_config.unlink(missing_ok=True)

    # ffmpeg_multicap writes timestamped sessions under output root when group-id is used.
    candidates = sorted(
        [p for p in output_dir.glob(f"{args.group_id}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    session_dir = candidates[0] if candidates else session_hint

    video_dir = session_dir / "video"
    if video_dir.exists():
        vids = list(video_dir.glob("*.*"))
        print(f"\nRecorded {len(vids)} video file(s) in: {video_dir}")
        print(f"Session dir: {session_dir}")
    else:
        print(f"\nWARNING: No video directory found at {video_dir}")
        print("Check ffmpeg_multicap output for errors.")
        return

    print()
    print("Next step — run calibration:")
    print(f"  python tools/calibrate_charuco.py calibrate \\")
    print(f"      --videos-dir \"{video_dir}\" \\")
    print(f"      --board-type auto --board-config \"{board_cfg_path}\"")


# ===================================================================
# Sub-command: detect  (diagnostic — check if board is visible)
# ===================================================================
def cmd_detect(args: argparse.Namespace) -> None:
    """Sample frames from each video and report charuco corner detection."""
    _setup_logging()
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        sys.exit(f"ERROR: Directory not found: {videos_dir}")

    video_files = _discover_video_files(videos_dir)
    if not video_files:
        sys.exit(f"ERROR: No video files found in {videos_dir}")

    resolved_board_type, spec = _resolve_board_spec(args.board_type, video_files)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=[spec["width"], spec["height"]],
        squareLength=1,
        markerLength=0.8,
        dictionary=aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    board_ids = _board_marker_id_set(spec)
    expected_corners = (spec["width"] - 1) * (spec["height"] - 1)

    sample_count = args.frames  # frames to sample per video

    print()
    print("=" * 60)
    print("CHARUCO DETECTION DIAGNOSTIC")
    print("=" * 60)
    print(f"  Board type      : {resolved_board_type} -> {spec['name']} ({expected_corners} inner corners)")
    print(f"  ArUco dictionary: DICT_4X4_250")
    print(f"  Sampling        : {sample_count} frames per video")
    print()

    all_ok = True
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            print(f"  {vf.name}: CANNOT OPEN")
            all_ok = False
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < 1:
            print(f"  {vf.name}: EMPTY (0 frames)")
            cap.release()
            all_ok = False
            continue

        # Sample evenly spaced frames via seeking
        step = max(1, total_frames // sample_count)
        sample_indices = list(range(0, total_frames, step))[:sample_count]

        detected_count = 0
        max_corners_found = 0

        # Spatial coverage tracking: divide frame into 3×3 grid
        region_hits = np.zeros((3, 3), dtype=int)

        def _process_gray(gray: np.ndarray) -> None:
            nonlocal detected_count, max_corners_found
            charuco_corners, charuco_ids = _detect_charuco_robust(
                gray, aruco_detector, board, board_ids,
            )
            if charuco_ids is not None and len(charuco_ids) > 0:
                detected_count += 1
                max_corners_found = max(max_corners_found, len(charuco_ids))
                for corner in charuco_corners:
                    cx, cy = corner.ravel()[:2]
                    gi = min(2, int(cx / w * 3))
                    gj = min(2, int(cy / h * 3))
                    region_hits[gj, gi] += 1

        # First pass: frame-seeking (fast but unreliable for some codecs)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            _process_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Fallback: if seeking produced 0 detections, try sequential reading
        # (ffmpeg MKV recordings may have long GOP intervals that defeat seeking)
        if detected_count == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            seq_read = 0
            seq_step = max(1, total_frames // sample_count)
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % seq_step == 0 and seq_read < sample_count:
                    seq_read += 1
                    _process_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if detected_count > 0:
                print(f"    NOTE: Frame-seeking returned 0; sequential fallback found detections.")

        cap.release()

        pct = 100 * detected_count / len(sample_indices) if sample_indices else 0
        if detected_count == 0:
            status = "NO BOARD DETECTED"
            all_ok = False
        elif max_corners_found < 4:
            status = f"WEAK ({max_corners_found} corners max)"
            all_ok = False
        else:
            status = f"OK (up to {max_corners_found}/{expected_corners} corners)"

        print(f"  {vf.name}")
        print(f"    Resolution : {w}x{h} @ {fps:.1f} fps, {total_frames} frames")
        print(f"    Detection  : {detected_count}/{len(sample_indices)} sampled frames -> {status}")

        # Spatial coverage report (3×3 grid)
        if detected_count > 0:
            total_hits = region_hits.sum()
            labels = ["left", "center", "right"]
            rows_l = ["top", "mid", "bottom"]
            empty_regions = []
            for rj in range(3):
                for ri in range(3):
                    if region_hits[rj, ri] == 0:
                        empty_regions.append(f"{rows_l[rj]}-{labels[ri]}")
            coverage_pct = int(100 * np.count_nonzero(region_hits) / 9)
            print(f"    Coverage   : {coverage_pct}% of image regions "
                  f"({np.count_nonzero(region_hits)}/9 zones)")
            if empty_regions:
                print(f"    Empty zones: {', '.join(empty_regions)}")
                if any("top" in r or "bottom" in r for r in empty_regions):
                    print("    TIP: Move board to cover top/bottom edges "
                          "(wide-angle distortion is worst at periphery)")
                if any("left" in r or "right" in r for r in empty_regions):
                    print("    TIP: Move board to cover left/right edges "
                          "for better distortion calibration")

    print()
    if all_ok:
        print("All cameras see the charuco board. Ready for calibration.")
    else:
        print("ISSUES FOUND. Checklist:")
        print("  1. Did you print charuco_board.png from THIS tool (5x3 or 7x5)?")
        print("     A different charuco layout / ArUco dictionary won't be detected.")
        print("  2. Is the board FLAT and rigid? Bent paper won't calibrate well.")
        print("  3. Is the board FULLY visible in frame? Even partial occlusion hurts.")
        print("  4. Lighting: avoid glare / reflections on the paper.")
        print("  5. Distance: board should be large enough to fill ~20-50% of frame.")
    print()


# ===================================================================
# Seeded-intrinsics calibration (used by --init-focal)
# ===================================================================

def _robust_charuco_2d_data(
    video_files: list[Path],
    board_spec: dict,
    progress_callback: Any = None,
) -> np.ndarray:
    """Pre-compute charuco 2D detections for all cameras using robust
    duplicate-ID filtering.

    This replaces anipose's internal multiprocessing charuco detection
    (``process_list_of_videos``) which cannot handle duplicate ArUco
    marker IDs from non-charuco markers sharing the same dictionary.

    Returns array of shape ``(n_cameras, n_frames, n_corners, 2)``
    with NaN for undetected corners — the same format as
    ``CameraGroup.charuco_2d_data``.
    """
    cb = progress_callback or (lambda _: None)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=[board_spec["width"], board_spec["height"]],
        squareLength=1,
        markerLength=0.8,
        dictionary=aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    board_ids = _board_marker_id_set(board_spec)
    n_corners = (board_spec["width"] - 1) * (board_spec["height"] - 1)

    # First pass: determine frame counts
    frame_counts = []
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    n_frames = min(frame_counts)
    n_cameras = len(video_files)

    # Allocate output array (NaN = undetected)
    data = np.full((n_cameras, n_frames, n_corners, 2), np.nan, dtype=np.float64)

    for cam_idx, vf in enumerate(video_files):
        cap = cv2.VideoCapture(str(vf))
        detected = 0
        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids = _detect_charuco_robust(
                gray, aruco_detector, board, board_ids,
            )
            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                detected += 1
                for corner, cid in zip(charuco_corners, charuco_ids):
                    idx = int(cid.ravel()[0])
                    if 0 <= idx < n_corners:
                        data[cam_idx, frame_idx, idx, :] = corner.ravel()[:2]

        cap.release()
        cb(f"Camera {cam_idx} ({vf.stem}): {detected}/{n_frames} frames with charuco")
        print(f"    {vf.stem}: {detected}/{n_frames} frames with charuco corners")

    return data


def _calibrate_with_seeded_intrinsics(
    video_files: list[Path],
    videos_dir: Path,
    charuco_board_def: Any,
    square_size: float,
    camera_specs: dict[str, Any],
    pin_camera_0_to_origin: bool = True,
    use_groundplane: bool = False,
    progress_callback: Any = None,
    min_charuco_frames: int = 15,
) -> tuple[Path, Any]:
    """Run anipose calibration with factory-spec focal lengths as initial
    intrinsic guesses.

    Bypasses anipose's multiprocessing charuco detection (which fails when
    non-charuco ArUco markers share the same dictionary/ID range) by
    pre-computing charuco 2D data with our robust duplicate-ID-filtered
    detector, then feeding it directly to ``calibrate_rows``.

    1. Pre-compute charuco 2D detections with duplicate-ID filtering.
    2. Create a ``CameraGroup`` and seed intrinsic matrices from specs.
    3. Build anipose row format from our 2D data.
    4. Call ``calibrate_rows`` (skipping ``calibrate_videos`` which uses
       multiprocessing detection internally).
    """
    from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import (
        freemocap_anipose,
    )

    cb = progress_callback or (lambda _: None)

    # --- Build camera names (stems of the MP4 files anipose will see) ---
    camera_names = [v.stem for v in video_files]

    # --- Create CameraGroup and seed intrinsics ---
    cam_group = freemocap_anipose.CameraGroup.from_names(camera_names)

    seeded_count = 0
    fallback_names: list[str] = []
    for cam in cam_group.cameras:
        name = cam.get_name()
        # Read actual video dimensions for this camera
        vid_path = videos_dir / (name + ".mp4")
        if vid_path.exists():
            cap = cv2.VideoCapture(str(vid_path))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            w, h = 1920, 1080
        cam.set_size((w, h))

        model = _match_camera_model(name, camera_specs)
        if model is not None:
            mtx = _build_intrinsic_matrix(model, w, h)
            cam.set_camera_matrix(mtx)
            seeded_count += 1
            logger.info(
                f"Seeded {name} with fx={mtx[0,0]:.1f}, fy={mtx[1,1]:.1f} "
                f"from model '{model.get('description', '?')}'"
            )
        else:
            fallback_names.append(name)

    if fallback_names:
        logger.warning(
            f"No camera-spec match for: {fallback_names}. "
            f"These will use anipose's default init (cv2.initCameraMatrix2D)."
        )

    # Decide init_intrinsics: only skip if ALL cameras were seeded
    init_intrinsics = bool(fallback_names)
    mode = "seeded" if not init_intrinsics else "mixed (seeded + auto)"
    print(f"  Init-focal mode  : {mode} ({seeded_count}/{len(cam_group.cameras)} cameras seeded)")
    cb(f"Intrinsics mode: {mode}")

    # --- Build anipose board ---
    anipose_board = freemocap_anipose.AniposeCharucoBoard(
        charuco_board_def.number_of_squares_width,
        charuco_board_def.number_of_squares_height,
        square_length=square_size,
        marker_length=square_size * 0.8,
        marker_bits=4,
        dict_size=250,
    )

    # --- Pre-compute charuco 2D data with robust detection ---
    # Bypass anipose's multiprocessing detection which cannot handle
    # duplicate ArUco IDs from non-charuco markers (table tracking).
    board_spec = {
        "width": charuco_board_def.number_of_squares_width,
        "height": charuco_board_def.number_of_squares_height,
    }
    print("  Detecting charuco board (robust duplicate-ID filtering)...")
    charuco_2d = _robust_charuco_2d_data(video_files, board_spec, cb)

    # Set camera sizes from videos
    for i, vf in enumerate(video_files):
        cap = cv2.VideoCapture(str(vf))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        cam_group.cameras[i].set_size((w, h))

    # Build anipose row format from our 2D data
    n_cameras, n_frames, n_corners, _ = charuco_2d.shape
    all_rows = []
    for cam_idx in range(n_cameras):
        camera_rows = []
        for frame_idx in range(n_frames):
            filled = charuco_2d[cam_idx, frame_idx, :, :].astype(np.float32)
            filled = np.reshape(filled, (n_corners, 1, 2))
            mask = (~np.isnan(filled[:, :, 0])) & (~np.isnan(filled[:, :, 1]))
            non_empty_ids = np.where(mask)[0]
            corners = filled[non_empty_ids, :, :]
            non_empty_ids = non_empty_ids.reshape(-1, 1)
            if corners.shape[0] != 0:
                camera_rows.append({
                    "framenum": (0, frame_idx),
                    "corners": corners,
                    "ids": non_empty_ids,
                    "filled": filled,
                })
        all_rows.append(camera_rows)

    print("  Charuco detection results (robust):")
    for i, rows in enumerate(all_rows):
        print(f"    Camera {i} ({video_files[i].stem}): {len(rows)} frames")

    # --- Auto-exclude cameras with too few charuco detections to form graph edges ---
    sparse = [
        (i, video_files[i].stem, len(all_rows[i]))
        for i in range(len(all_rows))
        if len(all_rows[i]) < min_charuco_frames
    ]
    if sparse:
        print(
            f"\n  WARNING: Auto-excluding {len(sparse)} camera(s) with "
            f"< {min_charuco_frames} charuco frames:"
        )
        for _, cname, cnt in sparse:
            print(f"    EXCLUDED {cname}: {cnt} frames (threshold: {min_charuco_frames})")
            logger.warning(
                "Auto-excluding camera %s: only %d charuco frames (< %d)",
                cname, cnt, min_charuco_frames,
            )
        keep = [i for i in range(len(all_rows)) if len(all_rows[i]) >= min_charuco_frames]
        if len(keep) < 2:
            raise ValueError(
                f"Only {len(keep)} camera(s) have >= {min_charuco_frames} charuco frames. "
                "Cannot calibrate. Re-record with the board visible in more cameras, "
                f"or lower --min-charuco-frames (currently {min_charuco_frames})."
            )
        video_files = [video_files[k] for k in keep]
        all_rows = [all_rows[k] for k in keep]
        # Rebuild cam_group with only the surviving cameras and re-seed intrinsics
        _fnames = [v.stem for v in video_files]
        cam_group = freemocap_anipose.CameraGroup.from_names(_fnames)
        _no_seed: list[str] = []
        for cam in cam_group.cameras:
            _cn = cam.get_name()
            _vp = videos_dir / (_cn + ".mp4")
            if _vp.exists():
                _vc = cv2.VideoCapture(str(_vp))
                _w2 = int(_vc.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h2 = int(_vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _vc.release()
            else:
                _w2, _h2 = 1920, 1080
            cam.set_size((_w2, _h2))
            _mdl = _match_camera_model(_cn, camera_specs)
            if _mdl is not None:
                cam.set_camera_matrix(_build_intrinsic_matrix(_mdl, _w2, _h2))
            else:
                _no_seed.append(_cn)
        init_intrinsics = bool(_no_seed)
        print(f"  Proceeding with {len(video_files)} camera(s) after exclusion.")

    # --- Run calibration (directly via calibrate_rows) ---
    error, merged, charuco_frames = cam_group.calibrate_rows(
        all_rows,
        anipose_board,
        init_intrinsics=init_intrinsics,
        init_extrinsics=True,
        verbose=True,
    )
    cb(f"Calibration done, reprojection error = {error:.4f}")

    # --- Pin camera 0 to origin ---
    groundplane_result = None
    if pin_camera_0_to_origin:
        rvecs = cam_group.get_rotations()
        tvecs = cam_group.get_translations()
        # Transform so camera 0 is at the origin
        R0 = cv2.Rodrigues(rvecs[0])[0]
        t0 = tvecs[0].reshape(3, 1)
        for i in range(len(cam_group.cameras)):
            Ri = cv2.Rodrigues(rvecs[i])[0]
            ti = tvecs[i].reshape(3, 1)
            Ri_new = Ri @ R0.T
            ti_new = ti - Ri_new @ t0
            cam_group.cameras[i].set_rotation(cv2.Rodrigues(Ri_new)[0].flatten())
            cam_group.cameras[i].set_translation(ti_new.flatten())

    # --- Ground-plane (using charuco) ---
    if use_groundplane:
        try:
            from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration.anipose_camera_calibrator import (
                AniposeCameraCalibrator,
            )
            # Use the static method for ground-plane calculation
            cam_group, groundplane_result = AniposeCameraCalibrator.set_charuco_board_as_groundplane(
                cam_group,
            )
        except Exception as exc:
            logger.warning(f"Ground-plane calibration failed: {exc}")
            from dataclasses import dataclass

            @dataclass
            class _GPFail:
                success: bool = False
                error: str = ""
            groundplane_result = _GPFail(success=False, error=str(exc))

    # --- Save TOML ---
    cam_group.metadata["calibration_method"] = "seeded_intrinsics"
    cam_group.metadata["camera_specs_file"] = str(
        getattr(camera_specs, "_source_path", "camera_specs.json")
    )
    cam_group.metadata["date_time_calibrated"] = str(np.datetime64("now"))
    cam_group.metadata["charuco_square_size"] = square_size
    cam_group.metadata["reprojection_error"] = float(error)

    toml_path = videos_dir / "calibration_seeded.toml"
    cam_group.dump(str(toml_path))
    cb(f"Saved calibration to {toml_path}")

    return toml_path, groundplane_result


# ===================================================================
# Sub-command: calibrate
# ===================================================================
def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run anipose Charuco calibration on pre-recorded videos."""
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        sys.exit(f"ERROR: Videos directory not found: {videos_dir}")

    # Discover video files
    video_files = _discover_video_files(videos_dir)
    if len(video_files) < 2:
        sys.exit(
            f"ERROR: Need at least 2 video files for multi-camera calibration, "
            f"found {len(video_files)} in {videos_dir}"
        )

    print()
    print("=" * 60)
    print("ANIPOSE CHARUCO CALIBRATION")
    print("=" * 60)
    print(f"  Videos dir       : {videos_dir}")
    print(f"  Video files      : {len(video_files)}")
    for v in video_files:
        print(f"    - {v.name}")
    print(f"  Board type       : {args.board_type}")
    print(f"  Square size (mm) : {args.square_size}")
    print(f"  Ground-plane     : {args.groundplane}")
    print(f"  Init-focal       : {getattr(args, 'init_focal', False)}")
    print()

    # FreeMoCap's get_video_paths() only recognises .mp4 files.
    # If inputs are .mkv (our default ffmpeg_multicap format), convert first.
    # Also: anipose requires all videos to have IDENTICAL frame counts,
    # so we always convert through a temp dir with CFR + trimmed duration.
    non_mp4 = [v for v in video_files if v.suffix.lower() != ".mp4"]
    needs_conversion = bool(non_mp4)

    # Check frame counts — even for .mp4 files, mismatched counts will crash anipose
    if not needs_conversion:
        counts = []
        for v in video_files:
            cap = cv2.VideoCapture(str(v))
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        if len(set(counts)) > 1:
            needs_conversion = True
            print(f"Frame count mismatch detected: {dict(zip([v.name for v in video_files], counts))}")

    if needs_conversion:
        calibration_mp4_dir = videos_dir / "_charuco_mp4"
        calibration_mp4_dir.mkdir(exist_ok=True)

        # Load per-camera sync offsets from ffmpeg_multicap_events.jsonl
        print("\nLoading sync offsets from ffmpeg_multicap_events.jsonl...")
        sync_offsets = _load_multicam_sync_offsets(videos_dir)

        # Find shortest duration to use as trim target
        min_duration = float("inf")
        fps_map: dict[Path, float] = {}
        for v in video_files:
            cap = cv2.VideoCapture(str(v))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            fps_map[v] = fps
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = nframes / fps
            cap.release()
            min_duration = min(min_duration, dur)

        print(f"\nConverting {len(video_files)} video(s) to MP4 (FreeMoCap requires .mp4, equal frame counts)")
        
        # Calculate max offset to know how much to trim for alignment
        max_offset = max(sync_offsets.values()) if sync_offsets else 0.0
        effective_duration = min_duration - max_offset
        
        if max_offset > 0:
            print(f"  Temporal alignment detected: max offset={max_offset:+.6f}s ({max_offset*1000:+.2f}ms)")
        print(f"  Trimming all to {effective_duration:.2f}s (after sync offset correction)")
        
        converted = []
        for src in video_files:
            dst = calibration_mp4_dir / (src.stem + ".mp4")
            if not dst.exists():
                print(f"  ffmpeg: {src.name} -> {dst.name}")
                # Detect available H.264 encoder: prefer libx264 (GPL), fall
                # back to h264_mf (MediaFoundation, Windows) or mjpeg.
                encoder_args = _detect_h264_encoder()
                
                # Get per-camera offset
                label = src.stem
                for suffix in ["_video", "_audio"]:
                    if label.endswith(suffix):
                        label = label[:-len(suffix)]
                camera_offset = sync_offsets.get(label, 0.0)
                
                # Build ffmpeg command with temporal alignment
                ffmpeg_cmd = [
                    _find_ffmpeg(), "-y",
                    "-i", str(src),
                ]
                
                # Apply trim with offset: skip camera_offset seconds, then take effective_duration
                if camera_offset > 0:
                    ffmpeg_cmd.extend(["-ss", f"{camera_offset:.6f}"])
                ffmpeg_cmd.extend(["-t", f"{effective_duration:.6f}"])
                
                ffmpeg_cmd.extend([
                    *encoder_args,
                    "-r", "30",  # force constant frame rate
                    "-an",  # drop audio — not needed for calibration
                    str(dst),
                ])
                
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    WARNING: ffmpeg failed for {src.name}")
                    print(f"    stderr: {result.stderr[:300]}")
                    continue
            else:
                print(f"  (cached) {dst.name}")
            converted.append(dst)
        if len(converted) < 2:
            sys.exit("ERROR: Need at least 2 converted MP4 files.")
        videos_dir = calibration_mp4_dir
        video_files = converted
        print(f"  Calibration will use: {videos_dir}")
        print()

    # Import FreeMoCap / anipose calibration machinery
    try:
        from freemocap.core_processes.capture_volume_calibration.charuco_stuff.charuco_board_definition import (
            CharucoBoardDefinition,
        )
        from freemocap.core_processes.capture_volume_calibration.run_anipose_capture_volume_calibration import (
            run_anipose_capture_volume_calibration,
        )
    except ImportError as exc:
        _setup_logging()  # fallback if freemocap unavailable
        sys.exit(
            f"ERROR: Could not import FreeMoCap calibration modules.\n"
            f"  Make sure freemocap>=1.3.0 is installed:\n"
            f"    conda activate affectai-freemocap\n"
            f"  Detail: {exc}"
        )

    # FreeMoCap configured logging; set up ours now
    _setup_logging()

    resolved_board_type, spec = _resolve_board_spec(args.board_type, video_files)
    board_cfg_path = Path(args.board_config) if args.board_config else None
    board_sizes_mm: dict[str, float] = {}
    if board_cfg_path:
        board_sizes_mm = _load_calibration_options(board_cfg_path).get("board_sizes_mm", {})
    if (
        board_sizes_mm
        and resolved_board_type in board_sizes_mm
        and not getattr(args, "_square_size_set", False)
    ):
        args.square_size = float(board_sizes_mm[resolved_board_type])
        print(f"  Square size      : {args.square_size} mm (from {board_cfg_path})")
    if args.board_type == BOARD_MODE_AUTO:
        print(f"  Auto board pick  : {resolved_board_type} ({spec['name']})")

    # --- Monkey-patch anipose's detect_markers at the class level ---
    # When non-charuco ArUco markers (e.g. table tracking markers from
    # tobii_multicam_glasses_tracker) share the same DICT_4X4 dictionary
    # and ID range as the charuco board, duplicate IDs confuse
    # interpolateCornersCharuco().  We patch detect_markers to filter
    # duplicates using spatial clustering before they reach anipose.
    from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import (
        freemocap_anipose,
    )
    board_ids = _board_marker_id_set(spec)
    _orig_dm = freemocap_anipose.AniposeCharucoBoard.detect_markers

    def _patched_dm(self, image, camera=None, refine=True):
        corners, ids = _orig_dm(self, image, camera, refine)
        if len(corners) == 0 or len(ids) == 0:
            return corners, ids
        ids_arr = np.array(ids) if not isinstance(ids, np.ndarray) else ids
        if ids_arr.ndim == 1:
            ids_arr = ids_arr.reshape(-1, 1)
        filt_corners, filt_ids = _filter_duplicate_marker_ids(
            tuple(corners), ids_arr, board_ids,
        )
        return list(filt_corners), filt_ids

    freemocap_anipose.AniposeCharucoBoard.detect_markers = _patched_dm
    logger.info(
        f"Patched anipose detect_markers for duplicate-ID filtering "
        f"(board IDs: {sorted(board_ids)})"
    )

    charuco_board = CharucoBoardDefinition(
        name=spec["name"],
        number_of_squares_width=spec["width"],
        number_of_squares_height=spec["height"],
        black_square_side_length=1,
        aruco_marker_length_proportional=0.8,
    )

    # ------------------------------------------------------------------
    # Decide calibration path: seeded intrinsics vs. default anipose
    # ------------------------------------------------------------------
    camera_specs = None
    use_init_focal = getattr(args, "init_focal", False)
    specs_path = getattr(args, "camera_specs", None)
    if specs_path:
        specs_path = Path(specs_path)
        if specs_path.exists():
            camera_specs = _load_camera_specs(specs_path)
            if use_init_focal:
                print(f"  Camera specs     : {specs_path}")
        else:
            if use_init_focal:
                sys.exit(
                    f"ERROR: --init-focal requires camera specs file, "
                    f"but not found: {specs_path}"
                )

    def _progress(msg: str) -> None:
        logger.info(f"[anipose] {msg}")

    logger.info("Starting anipose calibration (this may take a few minutes)...")
    try:
        if use_init_focal and camera_specs is not None:
            # ----- Seeded-intrinsics path (bypass run_anipose_capture_volume_calibration) -----
            toml_path, groundplane_result = _calibrate_with_seeded_intrinsics(
                video_files=video_files,
                videos_dir=videos_dir,
                charuco_board_def=charuco_board,
                square_size=args.square_size,
                camera_specs=camera_specs,
                pin_camera_0_to_origin=True,
                use_groundplane=args.groundplane,
                progress_callback=_progress,
                min_charuco_frames=args.min_charuco_frames,
            )
        else:
            # ----- Default anipose path (unchanged) -----
            toml_path, groundplane_result = run_anipose_capture_volume_calibration(
                charuco_board_definition=charuco_board,
                charuco_square_size=args.square_size,
                calibration_videos_folder_path=videos_dir,
                pin_camera_0_to_origin=True,
                use_charuco_as_groundplane=args.groundplane,
                progress_callback=_progress,
            )
    except Exception as exc:
        import traceback
        print()
        print("=" * 60)
        print("CALIBRATION FAILED")
        print("=" * 60)
        print(f"  Error: {exc}")
        print()
        print("Traceback:")
        traceback.print_exc()
        print()
        print("Common causes:")
        print("  1. Charuco board not detected in enough frames.")
        print("     Run 'detect' first to check:")
        print(f"       python tools/calibrate_charuco.py detect --videos-dir \"{videos_dir}\"")
        print("  2. Board must be visible in at least 2 cameras simultaneously.")
        print("  3. Make sure you printed the board from THIS tool (5x3, DICT_4X4_250).")
        print("     A different charuco layout will NOT be detected.")
        print("  4. Check lighting (no glare) and that the board is flat/rigid.")
        sys.exit(1)

    # Copy toml to a convenient location
    output_toml = Path(args.output)
    if toml_path and Path(toml_path).exists():
        import shutil
        shutil.copy2(toml_path, output_toml)
        print()
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"  Calibration file : {output_toml}")
        print(f"  Anipose toml     : {toml_path}")
        if args.groundplane and groundplane_result:
            status = "OK" if groundplane_result.success else f"FAILED ({groundplane_result.error})"
            print(f"  Ground-plane     : {status}")
        print()
        print("You can now use this calibration with FreeMoCap for 3D reconstruction.")
        print("See: docs/freemocap_quickstart.md")
    else:
        sys.exit("ERROR: Calibration completed but no .toml file was produced.")


# ===================================================================
# Sub-command: validate
# ===================================================================
def cmd_validate(args: argparse.Namespace) -> None:
    """Validate an existing calibration .toml file."""
    _setup_logging()
    toml_path = Path(args.toml)
    if not toml_path.exists():
        sys.exit(f"ERROR: File not found: {toml_path}")

    try:
        from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import (
            freemocap_anipose,
        )
    except ImportError as exc:
        sys.exit(f"ERROR: Cannot import freemocap anipose module: {exc}")

    cam_group = freemocap_anipose.CameraGroup.load(str(toml_path))

    print()
    print("=" * 60)
    print("CALIBRATION VALIDATION")
    print("=" * 60)
    print(f"  File     : {toml_path}")
    print(f"  Cameras  : {len(cam_group.cameras)}")
    print()

    # Print per-camera intrinsics summary
    print(f"{'Camera':<30} {'fx':>10} {'fy':>10} {'cx':>10} {'cy':>10}")
    print("-" * 70)

    for cam in cam_group.cameras:
        mtx = cam.get_camera_matrix()
        if mtx is not None:
            fx, fy = mtx[0, 0], mtx[1, 1]
            cx, cy = mtx[0, 2], mtx[1, 2]
            print(f"{cam.get_name():<30} {fx:>10.1f} {fy:>10.1f} {cx:>10.1f} {cy:>10.1f}")

    # --- Focal length vs camera specs comparison ---
    specs_path = DEFAULT_CAMERA_SPECS
    focal_warnings: list[str] = []
    if specs_path.exists():
        camera_specs = _load_camera_specs(specs_path)
        print()
        print(f"{'Camera':<30} {'fx_cal':>10} {'fx_spec':>10} {'Ratio':>8} {'Status':>10}")
        print("-" * 68)
        for cam in cam_group.cameras:
            name = cam.get_name()
            mtx = cam.get_camera_matrix()
            if mtx is None:
                continue
            fx_cal = mtx[0, 0]
            model = _match_camera_model(name, camera_specs)
            if model and "expected_fx_1080p" in model:
                fx_spec = float(model["expected_fx_1080p"])
                ratio = fx_cal / fx_spec
                if ratio > 1.4 or ratio < 0.7:
                    status = "BAD"
                    focal_warnings.append(
                        f"  {name}: fx_calibrated={fx_cal:.1f} vs "
                        f"fx_expected={fx_spec:.1f} (ratio={ratio:.2f}). "
                        f"Likely miscalibrated — try --init-focal."
                    )
                elif ratio > 1.15 or ratio < 0.85:
                    status = "SUSPECT"
                else:
                    status = "ok"
                print(f"{name:<30} {fx_cal:>10.1f} {fx_spec:>10.1f} {ratio:>8.2f} {status:>10}")
            else:
                print(f"{name:<30} {fx_cal:>10.1f} {'?':>10} {'?':>8} {'no spec':>10}")

    # Print per-camera extrinsics summary (world positions)
    rvecs = cam_group.get_rotations()
    tvecs = cam_group.get_translations()
    print()
    print(f"{'Camera':<30} {'tx':>10} {'ty':>10} {'tz':>10}")
    print("-" * 70)
    names = [cam.get_name() for cam in cam_group.cameras]
    for i, name in enumerate(names):
        tx, ty, tz = tvecs[i]
        print(f"{name:<30} {tx:>10.3f} {ty:>10.3f} {tz:>10.3f}")

    # Reprojection error (from metadata if available)
    meta = cam_group.metadata
    if meta:
        print()
        print("Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    # ---- Distortion coefficient analysis (lesson from face/hand pipeline) ----
    print()
    print(f"{'Camera':<30} {'k1':>10} {'k2':>10} {'|dist|':>10} {'Monotonic?':>12}")
    print("-" * 72)
    warnings = []
    for cam in cam_group.cameras:
        dist = getattr(cam, 'dist', None)
        if dist is None:
            # Anipose Camera.get_distortions() may also work
            dist = cam.get_distortions() if hasattr(cam, 'get_distortions') else None
        name = cam.get_name()
        if dist is not None and len(dist) > 0:
            k1 = float(dist[0])
            k2 = float(dist[1]) if len(dist) > 1 else 0.0
            dist_mag = float(np.linalg.norm(dist))
            # Monotonicity: for k1 < 0, critical at r_crit = 1/sqrt(-3*k1).
            # For a 1920×1080 sensor with corners at ~r_max, check if
            # the full field of view stays monotonic.
            mtx = cam.get_camera_matrix()
            if mtx is not None and k1 < 0:
                fx, fy = mtx[0, 0], mtx[1, 1]
                cx, cy = mtx[0, 2], mtx[1, 2]
                # Worst-case radius: image corner
                w, h = 1920, 1080  # default; exact size not in TOML
                corners = [
                    ((0 - cx) / fx, (0 - cy) / fy),
                    ((w - cx) / fx, (0 - cy) / fy),
                    ((0 - cx) / fx, (h - cy) / fy),
                    ((w - cx) / fx, (h - cy) / fy),
                ]
                r_max = max(np.sqrt(xn**2 + yn**2) for xn, yn in corners)
                r_crit = 1.0 / np.sqrt(-3.0 * k1)
                if r_max > r_crit:
                    monotonic = f"NO (r_max={r_max:.2f} > r_crit={r_crit:.2f})"
                    warnings.append(
                        f"  ⚠ {name}: k1={k1:.3f} causes NON-MONOTONIC distortion.\n"
                        f"    cv2.undistortPoints() will FAIL for points near image edges.\n"
                        f"    Use nonlinear triangulation (multicam_pose3d.py has this built-in)."
                    )
                else:
                    monotonic = "yes"
            else:
                monotonic = "yes"
            print(f"{name:<30} {k1:>10.4f} {k2:>10.4f} {dist_mag:>10.3f} {monotonic:>12}")
        else:
            print(f"{name:<30} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>12}")

    if warnings:
        print()
        print("=" * 60)
        print("DISTORTION WARNINGS")
        print("=" * 60)
        for w in warnings:
            print(w)
        print()
        print("  These cameras have extreme barrel distortion (k1 << 0).")
        print("  The radial distortion function r_d = r*(1 + k1*r^2) is")
        print("  non-monotonic beyond r_crit = 1/sqrt(-3*k1), which makes")
        print("  cv2.undistortPoints() diverge silently.")
        print()
        print("  Options:")
        print("    1. Re-calibrate with more frames / better board visibility.")
        print("    2. Use multicam_pose3d.py which has a monotonicity filter")
        print("       and nonlinear triangulation to handle this robustly.")
        print("    3. Accept that these cameras may produce unreliable")
        print("       triangulation near image edges.")

    if focal_warnings:
        print()
        print("=" * 60)
        print("FOCAL LENGTH WARNINGS")
        print("=" * 60)
        for fw in focal_warnings:
            print(fw)
        print()
        print("  Re-calibrate with --init-focal to seed anipose with the")
        print("  expected focal length from camera specs:")
        print("    python tools/calibrate_charuco.py calibrate \\")
        print("        --videos-dir <DIR> --square-size 39 --init-focal")

    print()
    print("Tip: If reprojection errors are high, re-record with")
    print("  - better lighting")
    print("  - slower board movement")
    print("  - board visible in at least 2 cameras simultaneously")


# ===================================================================
# Sub-command: ground-plane  (dedicated ground-plane calibration step)
# ===================================================================
def cmd_ground_plane(args: argparse.Namespace) -> None:
    """Compute ground-plane transform from a short clip of the ChArUco board
    lying flat on the table/floor.

    The board defines the Z=0 plane. The transform is saved alongside
    the calibration TOML so that 3D reconstructions are expressed in a
    coordinate system where:
      - X/Y lie on the table surface
      - Z points upward (away from the table)
      - Origin is at the board centre
    """
    _setup_logging()
    import toml as _toml  # anipose writes TOML; reuse the same library

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        sys.exit(f"ERROR: Videos directory not found: {videos_dir}")

    toml_path = Path(args.toml)
    if not toml_path.exists():
        sys.exit(f"ERROR: Calibration TOML not found: {toml_path}")

    video_files = _discover_video_files(videos_dir)
    if not video_files:
        sys.exit(f"ERROR: No video files in {videos_dir}")

    resolved_board_type, spec = _resolve_board_spec(args.board_type, video_files)
    board_cfg_path = Path(args.board_config) if args.board_config else None
    if board_cfg_path:
        board_sizes_mm = _load_calibration_options(board_cfg_path).get("board_sizes_mm", {})
        if (
            board_sizes_mm
            and resolved_board_type in board_sizes_mm
            and not getattr(args, "_square_size_set", False)
        ):
            args.square_size = float(board_sizes_mm[resolved_board_type])
            print(f"  Square size      : {args.square_size} mm (from {board_cfg_path})")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=[spec["width"], spec["height"]],
        squareLength=args.square_size,
        markerLength=args.square_size * 0.8,
        dictionary=aruco_dict,
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)
    expected_corners = (spec["width"] - 1) * (spec["height"] - 1)

    print()
    print("=" * 60)
    print("GROUND-PLANE CALIBRATION")
    print("=" * 60)
    print(f"  Videos dir  : {videos_dir}")
    print(f"  Calib TOML  : {toml_path}")
    print(f"  Board       : {resolved_board_type} -> {spec['name']} ({expected_corners} corners)")
    print(f"  Square size : {args.square_size} mm")
    print()

    # Load calibration to get camera intrinsics
    try:
        from freemocap.core_processes.capture_volume_calibration.anipose_camera_calibration import (
            freemocap_anipose,
        )
    except ImportError as exc:
        sys.exit(f"ERROR: Cannot import freemocap anipose module: {exc}")

    cam_group = freemocap_anipose.CameraGroup.load(str(toml_path))
    cam_names = [cam.get_name() for cam in cam_group.cameras]

    # For each camera, find a frame with a good charuco detection and compute
    # the board-to-camera transform via solvePnP.
    all_transforms = []  # list of (cam_name, R_board2cam, t_board2cam, n_corners, reproj_err)

    for vf in video_files:
        # Match video file to calibration camera by stem
        cam_match = None
        for ci, cn in enumerate(cam_names):
            if cn in vf.stem or vf.stem in cn:
                cam_match = ci
                break
        if cam_match is None:
            logger.warning(f"  {vf.name}: no matching calibration camera, skipping")
            continue

        cam = cam_group.cameras[cam_match]
        mtx = cam.get_camera_matrix()
        dist = cam.get_distortions() if hasattr(cam, 'get_distortions') else np.zeros(5)
        if mtx is None:
            logger.warning(f"  {cam_names[cam_match]}: no intrinsic matrix, skipping")
            continue

        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            logger.warning(f"  {vf.name}: cannot open")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        best_n = 0
        best_result = None

        # Sample frames to find best charuco detection
        n_samples = min(args.frames, total_frames)
        step = max(1, total_frames // n_samples)
        for fi in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

            if charuco_ids is not None and len(charuco_ids) >= 6 and len(charuco_ids) > best_n:
                # solvePnP needs object points for detected corners
                obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
                ret_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts, charuco_corners, mtx, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ret_pnp:
                    # Compute reprojection error
                    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
                    reproj = float(np.mean(np.linalg.norm(
                        proj.reshape(-1, 2) - charuco_corners.reshape(-1, 2), axis=1
                    )))
                    R, _ = cv2.Rodrigues(rvec)
                    best_n = len(charuco_ids)
                    best_result = (cam_names[cam_match], R, tvec.flatten(), best_n, reproj)

        cap.release()

        if best_result:
            cn, R, t, nc, re = best_result
            print(f"  {cn}: {nc}/{expected_corners} corners, reproj {re:.2f} px")
            all_transforms.append(best_result)
        else:
            print(f"  {vf.name}: no usable charuco detection (board must be visible & flat)")

    if not all_transforms:
        sys.exit("\nERROR: No cameras detected the charuco board. "
                 "Place the board flat on the table and record a short clip.")

    # Pick the camera with the most corners (best view of the flat board)
    best = max(all_transforms, key=lambda x: (x[3], -x[4]))
    best_cam, R_bc, t_bc, n_corners, reproj = best
    print(f"\n  Best view: {best_cam} ({n_corners} corners, {reproj:.2f} px reproj)")

    # The board coordinate system has Z perpendicular to the board surface.
    # R_bc @ [0,0,1]^T = the board's Z-axis in camera frame.
    # To define a world frame where Z is up (board normal pointing up):
    #   R_world = R_bc^T  (board frame = world frame)
    #   t_world = -R_bc^T @ t_bc

    # Compute world-frame transform for all cameras
    R_world_inv = R_bc.T  # world-to-board = board-to-world inverse = R_bc^T
    t_world_origin = -R_bc.T @ t_bc

    ground_plane_info = {
        "reference_camera": best_cam,
        "n_corners_detected": n_corners,
        "reprojection_error_px": round(reproj, 3),
        "board_type": resolved_board_type,
        "square_size_mm": args.square_size,
        "R_board_to_camera": R_bc.tolist(),
        "t_board_to_camera_mm": t_bc.tolist(),
        "R_world_rotation": R_world_inv.tolist(),
        "t_world_origin_mm": t_world_origin.tolist(),
        "description": (
            "Ground plane defined by ChArUco board lying flat on table. "
            "World frame: X/Y on table surface, Z pointing up (away from table), "
            "origin at board centre."
        ),
    }

    # Save ground-plane data as a companion JSON alongside the TOML
    gp_path = toml_path.with_name(toml_path.stem + "_groundplane.json")
    import json as _json
    gp_path.write_text(_json.dumps(ground_plane_info, indent=2))

    # Also try to inject into the TOML metadata section
    try:
        toml_data = _toml.load(str(toml_path))
        if "metadata" not in toml_data:
            toml_data["metadata"] = {}
        toml_data["metadata"]["groundplane_reference_camera"] = best_cam
        toml_data["metadata"]["groundplane_reproj_px"] = round(reproj, 3)
        toml_data["metadata"]["groundplane_file"] = gp_path.name
        with open(toml_path, "w") as f:
            _toml.dump(toml_data, f)
        logger.info("Updated TOML metadata with ground-plane reference")
    except Exception as e:
        logger.warning(f"Could not update TOML metadata: {e} (ground-plane JSON still saved)")

    print()
    print("=" * 60)
    print("GROUND-PLANE CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"  Ground-plane file : {gp_path}")
    print(f"  Reference camera  : {best_cam}")
    print(f"  Board corners     : {n_corners}/{expected_corners}")
    print(f"  Reproj error      : {reproj:.2f} px")
    print()
    print("  The ground plane defines Z=0 at the table surface.")
    print("  3D reconstructions can use this to orient skeleton data")
    print("  so that Y is 'up' and the table is at Z=0.")
    print()
    print("  To apply: pass --groundplane-json to multicam_pose3d.py")
    print("  or transform 3D output by R_world_rotation / t_world_origin_mm.")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibrate_charuco",
        description=(
            "Charuco-based spatial calibration for multi-camera FreeMoCap "
            "post-hoc 3D reconstruction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical workflow:
  1. python tools/calibrate_charuco.py print-board --output charuco_board.png
    2. python tools/calibrate_charuco.py record --config configs/ffmpeg_multicap.json --duration 75 --beep-interval 15
  3. python tools/calibrate_charuco.py detect --videos-dir data/calibration/charuco_<ts>/video
  4. python tools/calibrate_charuco.py calibrate --videos-dir data/calibration/charuco_<ts>/video --square-size 39
  5. python tools/calibrate_charuco.py ground-plane --videos-dir data/calibration/gp_<ts>/video --toml calibration_charuco.toml --square-size 39
  6. python tools/calibrate_charuco.py validate --toml calibration_charuco.toml
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- detect ---
    det = sub.add_parser(
        "detect",
        help="Check if the charuco board is detectable in your videos (run BEFORE calibrate)",
    )
    det.add_argument(
        "--videos-dir", required=True, type=str,
        help="Directory containing video files to check",
    )
    det.add_argument(
        "--board-type", choices=BOARD_MODE_CHOICES,
        default=DEFAULT_BOARD,
        help="Board type to look for (5x3/7x5/auto; default: 7x5)",
    )
    det.add_argument(
        "--frames", type=int, default=20,
        help="Number of frames to sample per video (default: 20)",
    )

    # --- print-board ---
    pb = sub.add_parser(
        "print-board",
        help="Generate a printable Charuco board image",
    )
    pb.add_argument(
        "--output", default="charuco_board.png",
        help="Output image path (default: charuco_board.png)",
    )
    pb.add_argument(
        "--board-type", choices=list(CHARUCO_BOARDS.keys()),
        default=DEFAULT_BOARD,
        help="Board type (default: 5x3, recommended for larger spaces)",
    )
    pb.add_argument("--width", type=int, default=1200, help="Image width in pixels")
    pb.add_argument("--height", type=int, default=800, help="Image height in pixels")

    # --- record ---
    rec = sub.add_parser(
        "record",
        help="Record charuco calibration videos via ffmpeg_multicap",
    )
    rec.add_argument(
        "--config", required=True, type=str,
        help="ffmpeg_multicap config JSON (e.g. configs/ffmpeg_multicap.json)",
    )
    rec.add_argument(
        "--board-config", type=str, default="configs/desk_markers_large.yaml",
        help="JSON/YAML config containing calibration board sizes/record defaults (default: configs/desk_markers_large.yaml)",
    )
    rec.add_argument(
        "--duration", type=int, default=DEFAULT_RECORD_DURATION_S,
        help=f"Recording duration in seconds (default: {DEFAULT_RECORD_DURATION_S})",
    )
    rec.add_argument(
        "--beep-interval", type=int, default=DEFAULT_BEEP_INTERVAL_S,
        help=f"Cue beep interval in seconds for board motion (default: {DEFAULT_BEEP_INTERVAL_S})",
    )
    rec.add_argument(
        "--group-id", default="calibration",
        help="Group ID prefix used by ffmpeg_multicap timestamped session naming",
    )
    rec.add_argument(
        "--board-setting", choices=["dynamic", "fixed", "two-board"], default="two-board",
        help="Recording protocol hint: dynamic board, fixed board, or alternating two-board setup",
    )
    rec.add_argument(
        "--output-dir", default=str(DEFAULT_CALIBRATION_DIR),
        help="Directory for calibration sessions",
    )

    # --- calibrate ---
    cal = sub.add_parser(
        "calibrate",
        help="Run anipose calibration on charuco videos",
    )
    cal.add_argument(
        "--videos-dir", required=True, type=str,
        help="Directory containing one video file per camera",
    )
    cal.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE_MM,
        help=f"Black square edge length in mm (default: {DEFAULT_SQUARE_SIZE_MM}). "
             "MEASURE your printed board!",
    )
    cal.add_argument(
        "--board-type", choices=BOARD_MODE_CHOICES,
        default=DEFAULT_BOARD,
        help=f"Board type matching your printed board (5x3/7x5/auto; default: {DEFAULT_BOARD})",
    )
    cal.add_argument(
        "--groundplane", action="store_true", default=True,
        help="Use charuco board position to define the ground plane (default: ON)",
    )
    cal.add_argument(
        "--no-groundplane", dest="groundplane", action="store_false",
        help="Disable ground-plane calibration",
    )
    cal.add_argument(
        "--output", default="calibration_charuco.toml",
        help="Output .toml path (default: calibration_charuco.toml)",
    )
    cal.add_argument(
        "--init-focal", action="store_true", default=False,
        help="Seed anipose with expected focal lengths from camera specs "
             "(improves calibration for wide-angle cameras like Jabra PanaCast). "
             "Requires --camera-specs.",
    )
    cal.add_argument(
        "--camera-specs", type=str,
        default=str(DEFAULT_CAMERA_SPECS),
        help="Path to camera_specs.json with known FOV/focal length per model "
             f"(default: {DEFAULT_CAMERA_SPECS})",
    )
    cal.add_argument(
        "--board-config", type=str, default="configs/desk_markers_large.yaml",
        help="JSON/YAML config for board square sizes and marker settings (default: configs/desk_markers_large.yaml)",
    )
    cal.add_argument(
        "--min-charuco-frames", type=int, default=15,
        help="Auto-exclude cameras with fewer than this many detected charuco frames (default: 15). "
             "Cameras below this threshold cannot form calibration graph edges and will cause failure.",
    )

    # --- validate ---
    val = sub.add_parser(
        "validate",
        help="Validate / inspect a calibration .toml file",
    )
    val.add_argument(
        "--toml", required=True, type=str,
        help="Path to calibration .toml file",
    )

    # --- ground-plane ---
    gp = sub.add_parser(
        "ground-plane",
        help="Compute ground-plane transform from a board lying flat on the table",
    )
    gp.add_argument(
        "--videos-dir", required=True, type=str,
        help="Directory containing video files of the board lying flat on the table",
    )
    gp.add_argument(
        "--toml", required=True, type=str,
        help="Path to existing calibration .toml file (intrinsics needed)",
    )
    gp.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE_MM,
        help=f"Black square edge length in mm (default: {DEFAULT_SQUARE_SIZE_MM})",
    )
    gp.add_argument(
        "--board-type", choices=BOARD_MODE_CHOICES,
        default=DEFAULT_BOARD,
        help=f"Board type (5x3/7x5/auto; default: {DEFAULT_BOARD})",
    )
    gp.add_argument(
        "--frames", type=int, default=30,
        help="Frames to sample per video for board detection (default: 30)",
    )
    gp.add_argument(
        "--board-config", type=str, default="configs/desk_markers_large.yaml",
        help="JSON/YAML config for board square sizes and marker settings (default: configs/desk_markers_large.yaml)",
    )

    args = parser.parse_args()
    args._square_size_set = "--square-size" in sys.argv

    dispatch = {
        "detect": cmd_detect,
        "print-board": cmd_print_board,
        "record": cmd_record,
        "calibrate": cmd_calibrate,
        "validate": cmd_validate,
        "ground-plane": cmd_ground_plane,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
