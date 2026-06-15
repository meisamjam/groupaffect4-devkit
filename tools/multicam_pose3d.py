#!/usr/bin/env python3
"""
Calibration-aware multi-camera 3D pose reconstruction.

Exploits synchronised multicam recording + spatial calibration:
  1. Auto-maps video/JSON dirs → calibration cameras via TOML ``name`` field
  2. Aligns frames using per-camera start offsets from events JSONL
  3. Undistorts 2D key-points with calibration distortion coefficients
  4. Matches persons across cameras using epipolar geometry
  5. Triangulates with confidence weighting + reprojection-error filtering
  6. Emits per-frame QC (reprojection error, # cameras, confidence)

Usage
-----
    # Full pipeline (auto-discover everything in a session)
    python tools/multicam_pose3d.py \\
        --session-dir new_data/ses-20260202_test \\
        --pose-root new_data/ses-20260202_test/mediapipe \\
        --output new_data/ses-20260202_test/skeleton_3d_mediapipe.npy

    # Explicit paths
    python tools/multicam_pose3d.py \\
        --calibration  .../video_camera_calibration.toml \\
        --events-jsonl .../ffmpeg_multicap_events.jsonl \\
        --pose-dirs cam1_json cam2_json cam3_json cam4_json p50_json \\
        --output skeleton.npy

    # Validate existing output
    python tools/multicam_pose3d.py validate --file skeleton.npy

Dependencies: numpy, scipy, opencv-python
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Calibration loading + camera matrix extraction
# ---------------------------------------------------------------------------

def load_calibration_toml(toml_path: Path) -> dict:
    """Load calibration from .toml file."""
    try:
        import tomllib
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml  # type: ignore[import-untyped]
            with open(toml_path) as f:
                return toml.load(f)
        except ImportError as exc:
            raise ImportError("Install: pip install toml  (or use Python ≥3.11)") from exc


# Upper-body + head keypoints (BODY_25):
# 0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist 5=LShoulder 6=LElbow 7=LWrist
# 8=MidHip 15=REye 16=LEye 17=REar 18=LEar
FACE_UPPER_KP: set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18}


class CameraCalibration:
    """Single camera's intrinsic + extrinsic parameters."""

    def __init__(self, cam_key: str, cam_data: dict):
        self.key = cam_key
        self.name: str = cam_data.get("name", cam_key)
        self.size = tuple(cam_data.get("size", [1920, 1080]))

        # Intrinsic
        self.K = np.array(cam_data["matrix"], dtype=np.float64)
        self.dist = np.array(
            cam_data.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64
        )

        # Extrinsic: rotation (Rodrigues vector) + translation
        from scipy.spatial.transform import Rotation

        rvec = np.array(cam_data.get("rotation", [0, 0, 0]), dtype=np.float64)
        tvec = np.array(cam_data.get("translation", [0, 0, 0]), dtype=np.float64)
        R = Rotation.from_rotvec(rvec).as_matrix()
        self.R = R
        self.t = tvec.reshape(3, 1)
        self.Rt = np.hstack([R, self.t])  # 3×4
        self.P = self.K @ self.Rt  # 3×4 projection matrix

        # World position (for debugging / visualisation)
        self.world_pos = np.array(
            cam_data.get("world_position", [0, 0, 0]), dtype=np.float64
        )

        # Camera role: 'scene' (default wide-angle) or 'face' (close-up)
        # kp_mask: set of BODY_25 keypoint indices this camera contributes to
        # (None means all keypoints).  Face cameras default to FACE_UPPER_KP.
        self.role: str = "scene"
        self.kp_mask: set[int] | None = None  # None = all keypoints

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Undistort 2D points using calibration distortion coefficients.

        Args:
            pts: (N, 2) array of distorted pixel coordinates

        Returns:
            (N, 2) array of undistorted pixel coordinates
        """
        if np.allclose(self.dist, 0):
            return pts

        # cv2.undistortPoints returns normalised coords; we re-project with K
        pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts_f32, self.K, self.dist, P=self.K)
        return undist.reshape(-1, 2).astype(np.float64)

    def reproject(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D world points → 2D undistorted pixel coordinates.

        Args:
            pts_3d: (N, 3) world coordinates

        Returns:
            (N, 2) pixel coordinates
        """
        N = pts_3d.shape[0]
        h = np.hstack([pts_3d, np.ones((N, 1))])  # (N, 4) homogeneous
        proj = (self.P @ h.T).T  # (N, 3)
        return proj[:, :2] / proj[:, 2:3]


def load_all_cameras(toml_path: Path) -> dict[str, CameraCalibration]:
    """Load calibration TOML → dict of camera_key → CameraCalibration."""
    calib_dict = load_calibration_toml(toml_path)
    cameras: dict[str, CameraCalibration] = {}
    for k in sorted(calib_dict.keys()):
        if k.startswith("cam_"):
            cameras[k] = CameraCalibration(k, calib_dict[k])
    return cameras


# ---------------------------------------------------------------------------
# 2. Camera ↔ pose-directory auto-mapping
# ---------------------------------------------------------------------------

def auto_map_cameras_to_pose_dirs(
    cameras: dict[str, CameraCalibration],
    pose_dirs: list[Path],
) -> dict[str, Path]:
    """
    Map each calibration camera to the matching pose-JSON directory.

    Matching heuristic: calibration ``name`` field is a substring of the
    directory name, or vice versa.

    Returns:
        cam_key → pose_dir   (only for matched cameras)
    """
    mapping: dict[str, Path] = {}

    for cam_key, cam in cameras.items():
        best: Path | None = None
        for pd in pose_dirs:
            dir_stem = pd.name.replace("_json", "")
            if cam.name in dir_stem or dir_stem in cam.name:
                best = pd
                break
        if best is not None:
            mapping[cam_key] = best
        else:
            logger.warning(
                f"  No pose directory found for camera {cam_key} "
                f"(name={cam.name!r})"
            )

    return mapping


# ---------------------------------------------------------------------------
# 3. Temporal synchronisation (multi-tier)
# ---------------------------------------------------------------------------
#
# Sync tiers (tried in priority order, best accuracy first):
#   Tier 1: Frame logs  — per-frame (unix_time - pts_time) median
#   Tier 2: LSL JSONL   — median(stream_time - out_time_sec) @ ~10 Hz
#   Tier 3: Progress TSV — median(host_time_sec - out_time_sec)
#   Tier 4: Events JSONL — capture_started unix timestamps
#
# All tiers produce a "start time" (the wall-clock/LSL instant when
# pts_time=0 would have occurred).  The reference is the latest-starting
# camera; each other camera skips N frames to align.
# ---------------------------------------------------------------------------


def _label_from_cam_name(cam_name: str) -> str:
    """Derive device label from calibration camera name.

    cam_name like ``jabra_panacast_20_cam1_vid_video``
    → label ``jabra_panacast_20_cam1_vid``
    """
    if cam_name.endswith("_video"):
        return cam_name[: -len("_video")]
    return cam_name


def _list_median(values: list[float]) -> float | None:
    """Median of a list, or None if empty."""
    if not values:
        return None
    vs = sorted(values)
    m = len(vs) // 2
    return vs[m] if len(vs) % 2 else (vs[m - 1] + vs[m]) / 2.0


# --- Tier 1: frame logs ---

def _load_frame_log_start(
    frame_log: Path, max_samples: int = 30, min_valid: int = 3,
) -> tuple[float | None, float | None]:
    """Estimate video start time from ``showinfo`` frame log.

    Computes ``start_time = unix_time_s - pts_time`` across multiple
    samples and returns ``(median, MAD)``.  Returns ``(None, None)`` if
    insufficient valid samples.
    """
    starts: list[float] = []
    try:
        with frame_log.open("r", encoding="utf-8") as f:
            for line in f:
                if len(starts) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                try:
                    pts_time = float(data["pts_time"])
                    unix_time = float(data.get("unix_time_s", data.get("unix_time", 0)))
                except (TypeError, ValueError, KeyError):
                    continue
                if unix_time <= 0:
                    continue
                starts.append(unix_time - pts_time)
    except Exception:
        return (None, None)

    if len(starts) < min_valid:
        return (None, None)

    median = _list_median(starts)
    if median is None:
        return (None, None)
    deviations = sorted(abs(s - median) for s in starts)
    mad = _list_median(deviations)
    return (median, mad)


def load_frame_log_starts(
    cameras: dict[str, "CameraCalibration"],
    camera_pose_map: dict[str, Path],
    frame_log_dir: Path | None,
) -> dict[str, float]:
    """Tier 1 sync: per-camera start times from frame logs.

    Looks for ``{label}_frames.jsonl`` in *frame_log_dir*.
    Returns cam_key → start_time (unix seconds at pts_time=0).
    """
    if frame_log_dir is None or not frame_log_dir.is_dir():
        return {}

    starts: dict[str, float] = {}
    for cam_key in camera_pose_map:
        label = _label_from_cam_name(cameras[cam_key].name)
        fl = frame_log_dir / f"{label}_frames.jsonl"
        if not fl.exists():
            continue
        median, mad = _load_frame_log_start(fl)
        if median is not None:
            starts[cam_key] = median
            mad_ms = (mad or 0) * 1000
            logger.info(f"    {cam_key}: frame-log start={median:.6f}  MAD={mad_ms:.2f}ms")
    return starts


# --- Tier 2: LSL progress JSONL ---

def _load_lsl_anchor_candidates(lsl_path: Path) -> list[float]:
    """Load (stream_time − out_time_sec) from an LSL progress JSONL.

    Works with 3/4/5-channel formats (values[0] is always out_time_sec).
    """
    candidates: list[float] = []
    try:
        segments: list[list[float]] = [[]]
        prev_out: float | None = None
        with lsl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    st = float(data.get("stream_time", 0))
                    vals = data.get("values", [])
                    if not vals:
                        continue
                    ot = float(vals[0])
                    if prev_out is not None and ot + 0.5 < prev_out:
                        segments.append([])
                    segments[-1].append(st - ot)
                    prev_out = ot
                except Exception:
                    continue
        # Use last (most recent) segment
        for seg in reversed(segments):
            if seg:
                candidates.extend(seg)
                break
    except Exception:
        pass
    return candidates


def load_lsl_starts(
    cameras: dict[str, "CameraCalibration"],
    camera_pose_map: dict[str, Path],
    lsl_dir: Path | None,
    lsl_prefix: str = "ffmpeg_progress_",
) -> dict[str, float]:
    """Tier 2 sync: per-camera start times from LSL progress JSONL.

    Uses ``median(stream_time − out_time_sec)`` which equals the LSL clock
    instant when ``out_time_sec=0`` (i.e. video start in the shared LSL
    clock domain).

    Returns cam_key → start_time (LSL clock at pts=0).
    """
    if lsl_dir is None or not lsl_dir.is_dir():
        return {}

    starts: dict[str, float] = {}
    for cam_key in camera_pose_map:
        label = _label_from_cam_name(cameras[cam_key].name)
        candidates_paths = [
            lsl_dir / f"{lsl_prefix}{label}.jsonl",
            lsl_dir / f"{label}.jsonl",
        ]
        lsl_path = next((p for p in candidates_paths if p.exists()), None)
        if lsl_path is None:
            continue
        cands = _load_lsl_anchor_candidates(lsl_path)
        median = _list_median(cands)
        if median is not None:
            starts[cam_key] = median
            logger.info(f"    {cam_key}: LSL start={median:.6f}  ({len(cands)} samples)")
    return starts


# --- Tier 3: progress TSV ---

def _load_tsv_anchor_candidates(
    session_dir: Path, label: str,
) -> list[float]:
    """Load (host_time_sec − out_time_sec) from progress TSV."""
    candidates: list[float] = []
    tsv = session_dir / "sourcedata" / "sync" / f"{label}_ffmpeg_progress.tsv"
    if not tsv.exists():
        return candidates
    try:
        segments: list[list[float]] = [[]]
        prev_out: float | None = None
        with tsv.open("r", encoding="utf-8") as f:
            import csv as _csv
            reader = _csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    ht = float(row["host_time_sec"])
                    ot = float(row["out_time_sec"])
                except (TypeError, ValueError, KeyError):
                    continue
                if prev_out is not None and ot + 0.5 < prev_out:
                    segments.append([])
                segments[-1].append(ht - ot)
                prev_out = ot
        for seg in reversed(segments):
            if seg:
                candidates.extend(seg)
                break
    except Exception:
        pass
    return candidates


def load_tsv_starts(
    cameras: dict[str, "CameraCalibration"],
    camera_pose_map: dict[str, Path],
    session_dir: Path | None,
) -> dict[str, float]:
    """Tier 3 sync: per-camera start times from progress TSV.

    Uses ``median(host_time_sec − out_time_sec)`` from
    ``sourcedata/sync/{label}_ffmpeg_progress.tsv``.

    Returns cam_key → start_time (host clock at pts=0).
    """
    if session_dir is None or not session_dir.is_dir():
        return {}

    starts: dict[str, float] = {}
    for cam_key in camera_pose_map:
        label = _label_from_cam_name(cameras[cam_key].name)
        cands = _load_tsv_anchor_candidates(session_dir, label)
        median = _list_median(cands)
        if median is not None:
            starts[cam_key] = median
            logger.info(f"    {cam_key}: TSV start={median:.6f}  ({len(cands)} samples)")
    return starts


# --- Tier 4: events JSONL ---

def load_event_starts(
    cameras: dict[str, "CameraCalibration"],
    camera_pose_map: dict[str, Path],
    events_jsonl: Path | None,
) -> dict[str, float]:
    """Tier 4 sync: per-camera start from ``capture_started`` events.

    Least accurate — single unix timestamp per camera.
    Returns cam_key → start_time (unix seconds).
    """
    if events_jsonl is None or not events_jsonl.exists():
        return {}

    raw_starts: dict[str, float] = {}
    with open(events_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("event_type") == "capture_started" and "video_output" in evt:
                raw_starts[evt["device_id"]] = evt["unix_time_s"]

    starts: dict[str, float] = {}
    for cam_key, cam in cameras.items():
        if cam_key not in camera_pose_map:
            continue
        label = _label_from_cam_name(cam.name)
        for dev_id, ts in raw_starts.items():
            if dev_id == label or dev_id in cam.name or cam.name.startswith(dev_id):
                starts[cam_key] = ts
                logger.info(f"    {cam_key}: event start={ts:.6f}")
                break
    return starts


# --- Unified sync ---

def compute_frame_sync_offsets(
    cameras: dict[str, "CameraCalibration"],
    camera_pose_map: dict[str, Path],
    fps: float = 30.0,
    frame_log_dir: Path | None = None,
    lsl_dir: Path | None = None,
    session_dir: Path | None = None,
    events_jsonl: Path | None = None,
) -> dict[str, int]:
    """Compute per-camera frame offsets using tiered sync sources.

    Tries sync sources in priority order (frame logs → LSL → TSV → events).
    Uses the first tier that provides data for **all** mapped cameras.
    If no tier covers all cameras, falls back to the tier with the most
    coverage.

    A positive offset means "skip this many frames at the beginning" to
    align with the camera that started last.

    Returns cam_key → frame_offset (int).
    """
    tiers: list[tuple[str, dict[str, float]]] = []

    # Tier 1: frame logs
    fl_starts = load_frame_log_starts(cameras, camera_pose_map, frame_log_dir)
    if fl_starts:
        tiers.append(("frame-log", fl_starts))

    # Tier 2: LSL JSONL
    lsl_starts = load_lsl_starts(cameras, camera_pose_map, lsl_dir)
    if lsl_starts:
        tiers.append(("LSL", lsl_starts))

    # Tier 3: progress TSV
    tsv_starts = load_tsv_starts(cameras, camera_pose_map, session_dir)
    if tsv_starts:
        tiers.append(("TSV", tsv_starts))

    # Tier 4: events JSONL
    ev_starts = load_event_starts(cameras, camera_pose_map, events_jsonl)
    if ev_starts:
        tiers.append(("events", ev_starts))

    if not tiers:
        logger.info("  No sync data found; assuming zero frame offsets.")
        return {ck: 0 for ck in camera_pose_map}

    # Pick best tier: prefer full coverage, then most cameras
    needed = set(camera_pose_map.keys())
    chosen_name, chosen_starts = tiers[0]  # default to first (highest priority)
    for name, starts in tiers:
        if set(starts.keys()) >= needed:
            chosen_name, chosen_starts = name, starts
            break
    else:
        # No tier covers all cameras; pick one with most coverage
        chosen_name, chosen_starts = max(tiers, key=lambda t: len(t[1]))

    logger.info(f"  Using sync tier: {chosen_name}"
                f"  ({len(chosen_starts)}/{len(needed)} cameras)")

    # Compute offsets relative to latest-starting camera
    t_ref = max(chosen_starts.values())
    offsets: dict[str, int] = {}
    for cam_key in camera_pose_map:
        if cam_key in chosen_starts:
            delta_s = t_ref - chosen_starts[cam_key]
            offsets[cam_key] = round(delta_s * fps)
        else:
            offsets[cam_key] = 0

    for ck, off in sorted(offsets.items()):
        logger.info(f"  {ck}: skip {off} frames ({off / fps:.3f} s)")

    return offsets


# ---------------------------------------------------------------------------
# 4. Pose-JSON loader (OpenPose-compatible)
# ---------------------------------------------------------------------------

def load_poses_from_dir(
    pose_dir: Path,
) -> list[list[dict[str, Any]]]:
    """
    Load all per-frame JSONs from a directory.

    Returns list-of-frames, each frame being a list of person dicts with
    ``pose_keypoints_2d`` already reshaped to (n_kp, 3).
    """
    json_files = sorted(pose_dir.glob("*.json"))
    frames: list[list[dict[str, Any]]] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        people = []
        for p in data.get("people", []):
            kps = np.array(p.get("pose_keypoints_2d", []), dtype=np.float64)
            if len(kps) == 0:
                continue
            kps = kps.reshape(-1, 3)  # (n_kp, 3)  [x, y, conf]
            pid = p.get("person_id", [-1])
            if isinstance(pid, list):
                pid = pid[0] if pid else -1
            people.append({"person_id": pid, "keypoints": kps})
        frames.append(people)
    return frames


# ---------------------------------------------------------------------------
# 5. Cross-camera person matching via epipolar geometry
# ---------------------------------------------------------------------------

def fundamental_matrix(cam_a: CameraCalibration, cam_b: CameraCalibration) -> np.ndarray:
    """Compute the fundamental matrix F_ab such that x_b^T F x_a = 0."""
    # F = K_b^{-T} [t_ab]_x R_ab K_a^{-1}
    # where R_ab = R_b R_a^T,  t_ab = t_b - R_ab t_a
    R_ab = cam_b.R @ cam_a.R.T
    t_ab = cam_b.t - R_ab @ cam_a.t
    # Skew-symmetric [t_ab]_x
    tx = np.array(
        [
            [0, -t_ab[2, 0], t_ab[1, 0]],
            [t_ab[2, 0], 0, -t_ab[0, 0]],
            [-t_ab[1, 0], t_ab[0, 0], 0],
        ]
    )
    E = tx @ R_ab  # Essential matrix
    F = np.linalg.inv(cam_b.K).T @ E @ np.linalg.inv(cam_a.K)
    return F / (F[2, 2] + 1e-12)  # normalise


def epipolar_distance(F: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """Symmetric epipolar distance (pixels) between pt_a in cam_a and pt_b in cam_b."""
    pa = np.array([pt_a[0], pt_a[1], 1.0])
    pb = np.array([pt_b[0], pt_b[1], 1.0])
    epi_line = F @ pa  # epipolar line in cam_b
    d_b = abs(pb @ epi_line) / (np.sqrt(epi_line[0] ** 2 + epi_line[1] ** 2) + 1e-12)
    epi_line2 = F.T @ pb  # epipolar line in cam_a
    d_a = abs(pa @ epi_line2) / (np.sqrt(epi_line2[0] ** 2 + epi_line2[1] ** 2) + 1e-12)
    return float((d_a + d_b) / 2)


def match_persons_across_cameras(
    frame_people: dict[str, list[dict[str, Any]]],
    cameras: dict[str, CameraCalibration],
    max_epipolar_px: float = 40.0,
) -> list[dict[str, int]]:
    """
    Match detected persons across cameras using epipolar geometry.

    For each camera pair, compute mean epipolar distance over visible joints.
    Build a cost matrix and solve via greedy matching.

    Args:
        frame_people: cam_key → list of person dicts (``keypoints``: Nx3)
        cameras: calibration objects
        max_epipolar_px: reject matches above this threshold

    Returns:
        list of assignment dicts: [{cam_key: local_person_idx, ...}, ...]
    """
    cam_keys = sorted(frame_people.keys())
    if len(cam_keys) < 2:
        # Single camera — each person is its own group
        if cam_keys:
            ck = cam_keys[0]
            return [{ck: i} for i in range(len(frame_people[ck]))]
        return []

    # Start from the camera with the most detections
    anchor_cam = max(cam_keys, key=lambda c: len(frame_people[c]))
    n_anchor = len(frame_people[anchor_cam])
    if n_anchor == 0:
        return []

    # Initialise groups: one per anchor person
    groups: list[dict[str, int]] = [{anchor_cam: i} for i in range(n_anchor)]

    for other_cam in cam_keys:
        if other_cam == anchor_cam:
            continue

        n_other = len(frame_people[other_cam])
        if n_other == 0:
            continue

        F = fundamental_matrix(cameras[anchor_cam], cameras[other_cam])

        # Cost matrix (n_anchor × n_other)
        cost = np.full((n_anchor, n_other), np.inf)
        for ai in range(n_anchor):
            kps_a = frame_people[anchor_cam][ai]["keypoints"]
            for bi in range(n_other):
                kps_b = frame_people[other_cam][bi]["keypoints"]
                dists = []
                for j in range(min(len(kps_a), len(kps_b))):
                    if kps_a[j, 2] > 0.3 and kps_b[j, 2] > 0.3:
                        d = epipolar_distance(F, kps_a[j, :2], kps_b[j, :2])
                        dists.append(d)
                if dists:
                    cost[ai, bi] = np.median(dists)

        # Greedy assignment  (Hungarian would be better for >4 ppl,
        # but typical lab has 1–4 participants)
        used_other: set[int] = set()
        for ai in range(n_anchor):
            row = cost[ai, :]
            order = np.argsort(row)
            for bi in order:
                if bi in used_other:
                    continue
                if row[bi] < max_epipolar_px:
                    groups[ai][other_cam] = int(bi)
                    used_other.add(int(bi))
                    break

    return groups


# ---------------------------------------------------------------------------
# 5b. Zone-aware person matching (multi-person seated setup)
# ---------------------------------------------------------------------------

@dataclass
class CameraZone:
    """Group of cameras that observe the same subset of people."""

    cam_keys: list[str]      # calibration camera keys, e.g. ["cam_0", "cam_3"]
    person_ids: list[int]    # global person indices, e.g. [0, 1]
    name: str = ""           # optional label for logging


def _resolve_camera_key(
    friendly: str,
    cameras: dict[str, CameraCalibration],
) -> str | None:
    """Resolve a friendly camera identifier to a calibration key.

    Accepted forms:
      - Calibration keys: ``cam_0``, ``cam_3``
      - Friendly short names: ``cam1``, ``cam4``, ``p50``
      - Full camera names: ``jabra_panacast_20_cam1_vid_video``
    """
    friendly = friendly.strip()
    # Exact match on key
    if friendly in cameras:
        return friendly
    # Exact match on camera name
    for k, c in cameras.items():
        if c.name == friendly:
            return k
    # Substring match ("cam1" in "jabra_panacast_20_cam1_vid_video")
    for k, c in cameras.items():
        if friendly in c.name or friendly in k:
            return k
    return None


def parse_camera_zones(
    zone_specs: list[str],
    cameras: dict[str, CameraCalibration],
) -> list[CameraZone]:
    """Parse CLI zone specs into CameraZone objects.

    Format: ``"cam1+cam4:0,1"`` means cameras cam1 and cam4 are
    primary for global person IDs 0 and 1.

    Example::

        --camera-zones cam1+cam4:0,1  cam2+cam3:2,3
    """
    zones: list[CameraZone] = []
    for idx, spec in enumerate(zone_specs):
        if ":" not in spec:
            raise ValueError(
                f"Zone spec {spec!r} must be 'cam_a+cam_b:pid1,pid2'"
            )
        cams_str, ids_str = spec.split(":", 1)
        cam_names = [n.strip() for n in cams_str.split("+")]
        person_ids = [int(x.strip()) for x in ids_str.split(",")]

        cam_keys: list[str] = []
        for cn in cam_names:
            key = _resolve_camera_key(cn, cameras)
            if key is not None:
                cam_keys.append(key)
            else:
                logger.warning(f"  Zone camera {cn!r} not found in calibration")

        if cam_keys:
            zones.append(CameraZone(
                cam_keys=cam_keys,
                person_ids=person_ids,
                name=f"zone_{idx}",
            ))
    return zones


# ---------------------------------------------------------------------------
# 5c. Face-camera assignments (close-up cameras for individual people)
# ---------------------------------------------------------------------------

@dataclass
class FaceCameraAssignment:
    """Maps a face/close-up camera to a specific person."""

    cam_key: str           # calibration key, e.g. "cam_5"
    person_id: int         # global person index this camera is dedicated to
    kp_mask: set[int]      # keypoint indices this camera contributes to


def parse_face_cameras(
    specs: list[str],
    cameras: dict[str, CameraCalibration],
    kp_mask: set[int] | None = None,
) -> list[FaceCameraAssignment]:
    """Parse ``--face-cameras`` CLI specs.

    Format: ``"cam5:0"``  — camera *cam5* is a face close-up for person 0.
    Multiple cameras per person are fine: ``"cam5:0 cam6:0 cam7:1"``.

    The keypoint mask defaults to ``FACE_UPPER_KP`` (head + upper body).
    Override with ``--face-kp-mask`` to specify a different set.

    Returns:
        list of FaceCameraAssignment
    """
    kp_mask = kp_mask if kp_mask is not None else FACE_UPPER_KP
    assignments: list[FaceCameraAssignment] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Face-camera spec {spec!r} must be 'cam_name:person_id'"
            )
        cam_str, pid_str = spec.split(":", 1)
        cam_key = _resolve_camera_key(cam_str.strip(), cameras)
        if cam_key is None:
            logger.warning(f"  Face camera {cam_str!r} not found in calibration")
            continue
        person_id = int(pid_str.strip())
        # Tag the camera calibration object
        cameras[cam_key].role = "face"
        cameras[cam_key].kp_mask = kp_mask
        assignments.append(FaceCameraAssignment(
            cam_key=cam_key,
            person_id=person_id,
            kp_mask=kp_mask,
        ))
    return assignments


# Face keypoints used for front-facing detection (BODY_25 indices).
_FACE_KP_NOSE = 0
_FACE_KP_REYE = 15
_FACE_KP_LEYE = 16
_FACE_KP_REAR = 17
_FACE_KP_LEAR = 18


def is_front_facing(
    keypoints: np.ndarray,
    min_face_conf: float = 0.3,
) -> bool:
    """Return True if a detection is front-facing (face visible).

    A person seen from behind will have very low confidence on Nose
    and Eyes because those landmarks are occluded.  We require *at
    least* the Nose **plus** one Eye to be visible.

    Args:
        keypoints: (N, 3) array ``[x, y, confidence]`` per keypoint.
        min_face_conf: minimum confidence to consider a face keypoint
            as visible.
    """
    if len(keypoints) <= max(_FACE_KP_NOSE, _FACE_KP_REYE, _FACE_KP_LEYE):
        return True  # cannot evaluate; assume front-facing

    nose_ok = keypoints[_FACE_KP_NOSE, 2] >= min_face_conf
    reye_ok = keypoints[_FACE_KP_REYE, 2] >= min_face_conf
    leye_ok = keypoints[_FACE_KP_LEYE, 2] >= min_face_conf

    return bool(nose_ok and (reye_ok or leye_ok))


def _filter_front_facing(
    people: list[dict[str, Any]],
    min_face_conf: float = 0.3,
) -> list[dict[str, Any]]:
    """Keep only front-facing detections, preserving original indices.

    Each returned dict gains an ``_orig_idx`` key so that the caller
    can map back to the original detection index.
    """
    result: list[dict[str, Any]] = []
    for idx, p in enumerate(people):
        if is_front_facing(p["keypoints"], min_face_conf):
            result.append({**p, "_orig_idx": idx})
    return result


def match_persons_zonewise(
    frame_people: dict[str, list[dict[str, Any]]],
    cameras: dict[str, CameraCalibration],
    zones: list[CameraZone],
    max_epipolar_px: float = 40.0,
    front_facing_filter: bool = True,
    min_face_conf: float = 0.3,
) -> list[dict[str, int]]:
    """Zone-aware person matching for multi-person seated setups.

    1. (Optional) Filter out back-facing detections in zone cameras
       so that people visible from behind (opposite side of the desk)
       are not matched as zone targets.
    2. Within each zone, match remaining persons via standard epipolar
       matching.
    3. Order zone-local persons by horizontal centroid (left→right)
       for consistent person-ID assignment across frames.
    4. Auto-detect shared cameras (not in any zone) and assign their
       detections to the closest zone person via epipolar distance.

    Args:
        front_facing_filter: if True, reject back-facing detections
            (Nose + at least one Eye must be visible at ``min_face_conf``).
        min_face_conf: confidence threshold for face keypoints.

    Returns:
        list of assignment dicts indexed by global person_id:
        ``[{cam_key: local_person_idx, ...}, ...]``
    """
    # Determine max person ID → output size
    max_pid = max(pid for z in zones for pid in z.person_ids) + 1
    groups: list[dict[str, int]] = [{} for _ in range(max_pid)]

    # Cameras assigned to zones vs. shared (e.g. P50)
    zoned_cams: set[str] = set()
    for z in zones:
        zoned_cams.update(z.cam_keys)
    shared_cams = [
        ck for ck in frame_people if ck not in zoned_cams and ck in cameras
    ]

    # --- Step 1: within-zone matching (with optional front-facing filter) ---
    for zone in zones:
        # Build per-camera detection lists; optionally drop back-facing.
        # We keep a mapping from filtered index → original index so the
        # final group dicts reference the original ``frame_people`` lists.
        zone_people_filt: dict[str, list[dict[str, Any]]] = {}
        filt_to_orig: dict[str, dict[int, int]] = {}  # cam → {filt_idx: orig_idx}

        for ck in zone.cam_keys:
            raw = frame_people.get(ck, [])
            if front_facing_filter:
                ff = _filter_front_facing(raw, min_face_conf)
                zone_people_filt[ck] = ff
                filt_to_orig[ck] = {
                    fi: p["_orig_idx"] for fi, p in enumerate(ff)
                }
            else:
                zone_people_filt[ck] = raw
                filt_to_orig[ck] = {i: i for i in range(len(raw))}

        zone_cams = {ck: cameras[ck] for ck in zone.cam_keys if ck in cameras}
        if not zone_cams:
            continue

        zone_groups = match_persons_across_cameras(
            zone_people_filt, zone_cams, max_epipolar_px=max_epipolar_px,
        )

        # Sort matched persons by horizontal centroid (left→right)
        # so that person assignment is spatially consistent.
        centroids: list[float] = []
        for zg in zone_groups:
            xs: list[float] = []
            for ck, li in zg.items():
                people = zone_people_filt.get(ck, [])
                if li < len(people):
                    kps = people[li]["keypoints"]
                    valid_x = kps[kps[:, 2] > 0.3, 0]
                    if len(valid_x) > 0:
                        xs.append(float(np.mean(valid_x)))
            centroids.append(np.mean(xs) if xs else 0.0)

        order = list(np.argsort(centroids))  # left-to-right

        for rank, zg_idx in enumerate(order):
            if rank < len(zone.person_ids):
                global_pid = zone.person_ids[rank]
                for ck, li in zone_groups[zg_idx].items():
                    # Map filtered index back to original
                    orig_idx = filt_to_orig.get(ck, {}).get(li, li)
                    groups[global_pid][ck] = orig_idx

    # --- Step 2: assign shared-camera detections to zone persons ---
    for shared_ck in shared_cams:
        shared_persons = frame_people.get(shared_ck, [])
        if not shared_persons:
            continue

        # Build cost matrix: (n_global_people × n_shared_detections)
        cost = np.full((max_pid, len(shared_persons)), np.inf)
        for gpid in range(max_pid):
            if not groups[gpid]:  # no zone data for this person
                continue
            for si in range(len(shared_persons)):
                epi_dists: list[float] = []
                for zone_ck, zone_li in groups[gpid].items():
                    if zone_ck not in cameras:
                        continue
                    zone_persons = frame_people.get(zone_ck, [])
                    if zone_li >= len(zone_persons):
                        continue
                    F = fundamental_matrix(cameras[zone_ck], cameras[shared_ck])
                    kps_z = zone_persons[zone_li]["keypoints"]
                    kps_s = shared_persons[si]["keypoints"]
                    jdists: list[float] = []
                    for j in range(min(len(kps_z), len(kps_s))):
                        if kps_z[j, 2] > 0.3 and kps_s[j, 2] > 0.3:
                            jdists.append(
                                epipolar_distance(F, kps_z[j, :2], kps_s[j, :2])
                            )
                    if jdists:
                        epi_dists.append(float(np.median(jdists)))
                if epi_dists:
                    cost[gpid, si] = float(np.mean(epi_dists))

        # Greedy assignment (low cost first)
        used_shared: set[int] = set()
        used_global: set[int] = set()
        flat_order = np.argsort(cost.ravel())
        for flat_idx in flat_order:
            gpid = int(flat_idx // len(shared_persons))
            si = int(flat_idx % len(shared_persons))
            if cost[gpid, si] >= max_epipolar_px:
                break
            if gpid in used_global or si in used_shared:
                continue
            groups[gpid][shared_ck] = si
            used_global.add(gpid)
            used_shared.add(si)

    return groups


# ---------------------------------------------------------------------------
# 6. Triangulation (with distortion-aware nonlinear refinement)
# ---------------------------------------------------------------------------

def _project_distorted(cam: CameraCalibration, pt3d: np.ndarray) -> np.ndarray:
    """Project a 3D point to *distorted* pixel coords (forward model).

    Uses the full OpenCV distortion model (k1, k2, p1, p2, k3).
    Unlike ``cv2.undistortPoints`` (inverse), this is always well-defined
    regardless of distortion magnitude or monotonicity.
    """
    pt_cam = cam.R @ pt3d.reshape(3, 1) + cam.t
    xc, yc, zc = pt_cam.flatten()
    if abs(zc) < 1e-12:
        return np.array([np.nan, np.nan])
    xn, yn = xc / zc, yc / zc
    r2 = xn**2 + yn**2

    d = cam.dist
    k1 = float(d[0]) if len(d) > 0 else 0.0
    k2 = float(d[1]) if len(d) > 1 else 0.0
    p1 = float(d[2]) if len(d) > 2 else 0.0
    p2 = float(d[3]) if len(d) > 3 else 0.0
    k3 = float(d[4]) if len(d) > 4 else 0.0

    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn

    px = cam.K[0, 0] * xd + cam.K[0, 2]
    py = cam.K[1, 1] * yd + cam.K[1, 2]
    return np.array([px, py])


def _distortion_is_monotonic(cam: CameraCalibration, pt2d: np.ndarray) -> bool:
    """Check whether radial distortion is monotonic at *pt2d*.

    For k1 < 0 (barrel distortion), the mapping
        r_d = r_u * (1 + k1*r_u**2)
    becomes non-monotonic beyond r_crit = 1 / sqrt(-3*k1).
    ``cv2.undistortPoints`` fails silently at those radii.

    Returns True when the observation is safe for ``undistortPoints``.
    """
    k1 = float(cam.dist[0]) if len(cam.dist) > 0 else 0.0
    if k1 >= 0:
        return True  # pincushion / zero — always monotonic
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    xn = (float(pt2d[0]) - cx) / fx
    yn = (float(pt2d[1]) - cy) / fy
    r2 = xn**2 + yn**2
    return (1.0 + 3.0 * k1 * r2) > 0.0


def triangulate_point(
    observations: list[tuple[CameraCalibration, np.ndarray]],
    min_confidence: float = 0.1,
    max_reproj_px: float = 30.0,
) -> tuple[np.ndarray, float, int]:
    """Triangulate one 3D joint from multiple 2D observations.

    Each observation is ``(CameraCalibration, [x, y, conf])``.

    Uses a two-step approach (lesson from face/hand pipeline):
      1. Filter cameras whose distortion is non-monotonic at the
         observation's radius — ``cv2.undistortPoints`` diverges there.
      2. DLT initial estimate from surviving (monotonic) cameras.
      3. Levenberg-Marquardt refinement against the full forward
         distortion model for ALL monotonic-safe cameras.

    Step 3 is skipped when all cameras have negligible distortion.

    Returns ``(point_3d, mean_reproj_error, n_cameras_used)``.
    """
    from scipy.optimize import least_squares

    # --- confidence gate ---
    valid: list[tuple[CameraCalibration, np.ndarray]] = []
    for cam, pt in observations:
        if pt[2] > min_confidence:
            valid.append((cam, pt))
    if len(valid) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan, 0

    # --- monotonicity filter ---
    mono: list[tuple[CameraCalibration, np.ndarray]] = [
        (cam, pt)
        for cam, pt in valid
        if _distortion_is_monotonic(cam, pt)
    ]
    if len(mono) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan, 0

    # --- Step 1: DLT initial estimate (undistorted coords) ---
    undist_pts = []
    for cam, pt in mono:
        ud = cam.undistort_points(pt[:2].reshape(1, 2))[0]
        undist_pts.append(ud)

    A = []
    for (cam, _), ud in zip(mono, undist_pts, strict=True):
        P = cam.P
        x, y = ud
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A_mat = np.array(A)

    _, _, Vh = np.linalg.svd(A_mat)
    X_h = Vh[-1, :]
    X0 = X_h[:3] / X_h[3]

    # --- decide whether LM refinement adds value ---
    has_distortion = any(not np.allclose(cam.dist, 0) for cam, _ in mono)

    if not has_distortion:
        # Pure DLT is sufficient when there is no distortion
        errors = []
        for (cam, _), ud in zip(mono, undist_pts, strict=True):
            proj = cam.reproject(X0.reshape(1, 3))[0]
            errors.append(np.linalg.norm(proj - ud))
        mean_err = float(np.mean(errors))
        if mean_err > max_reproj_px:
            return np.array([np.nan, np.nan, np.nan]), mean_err, len(mono)
        return X0, mean_err, len(mono)

    # --- Step 2: LM refinement with forward distortion model ---
    def residuals(x3d: np.ndarray) -> np.ndarray:
        res = np.empty(len(mono) * 2)
        for i, (cam, pt) in enumerate(mono):
            proj = _project_distorted(cam, x3d)
            res[2 * i] = proj[0] - pt[0]
            res[2 * i + 1] = proj[1] - pt[1]
        return res

    result = least_squares(residuals, X0, method="lm", max_nfev=100)
    pt3d = result.x

    # Reprojection error in distorted pixel space
    r = result.fun
    reproj_errs = [
        float(np.sqrt(r[2 * i] ** 2 + r[2 * i + 1] ** 2))
        for i in range(len(mono))
    ]
    mean_err = float(np.mean(reproj_errs))

    if mean_err > max_reproj_px:
        return np.array([np.nan, np.nan, np.nan]), mean_err, len(mono)

    return pt3d, mean_err, len(mono)


# ---------------------------------------------------------------------------
# 7. Full-frame pipeline
# ---------------------------------------------------------------------------

def process_frame(
    frame_idx: int,
    all_poses: dict[str, list[dict[str, Any]]],
    cameras: dict[str, CameraCalibration],
    n_keypoints: int = 25,
    max_epipolar_px: float = 40.0,
    max_reproj_px: float = 30.0,
    min_kp_conf: float = 0.1,
    zones: list[CameraZone] | None = None,
    n_people_fixed: int | None = None,
    front_facing_filter: bool = True,
    min_face_conf: float = 0.3,
    face_assignments: list[FaceCameraAssignment] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Process one synchronised frame across all cameras.

    Args:
        zones: if provided, use zone-aware matching instead of
            global epipolar matching.
        n_people_fixed: force output to this many person slots
            (used with zones to keep consistent array shape).
        front_facing_filter: reject back-of-head detections when
            using zone matching (default True).
        min_face_conf: face keypoint confidence threshold for the
            front-facing filter.
        face_assignments: list of FaceCameraAssignment — each face camera
            is pre-assigned to a specific person and contributes only to
            the keypoints in its ``kp_mask``.  Face cameras pick the
            highest-confidence detection (single-person close-up).

    Returns:
        skeleton: (n_people, n_keypoints, 7)
            columns: [x, y, z, confidence, reproj_err, n_cams, person_group_id]
        qc: dict with per-frame quality metrics
    """
    # 1. Cross-camera person matching (scene cameras only — face cameras
    #    are excluded from matching and injected directly below)
    scene_poses = all_poses
    if face_assignments:
        face_cam_keys = {fa.cam_key for fa in face_assignments}
        scene_poses = {
            ck: ps for ck, ps in all_poses.items() if ck not in face_cam_keys
        }

    if zones:
        groups = match_persons_zonewise(
            scene_poses, cameras, zones,
            max_epipolar_px=max_epipolar_px,
            front_facing_filter=front_facing_filter,
            min_face_conf=min_face_conf,
        )
    else:
        groups = match_persons_across_cameras(
            scene_poses, cameras, max_epipolar_px=max_epipolar_px
        )

    n_people = n_people_fixed or len(groups)
    skeleton = np.full((max(n_people, 1), n_keypoints, 7), np.nan)

    # 1b. Build per-person face-camera observation lookup.
    # For each face camera, pick the best (highest mean confidence)
    # detection — face cameras typically see only one person.
    face_obs_per_person: dict[int, list[tuple[str, int, set[int]]]] = {}
    if face_assignments:
        for fa in face_assignments:
            people = all_poses.get(fa.cam_key, [])
            if not people:
                continue
            # Pick best detection by mean confidence
            best_idx = 0
            best_conf = -1.0
            for di, det in enumerate(people):
                mc = float(np.mean(det["keypoints"][:, 2]))
                if mc > best_conf:
                    best_conf = mc
                    best_idx = di
            face_obs_per_person.setdefault(fa.person_id, []).append(
                (fa.cam_key, best_idx, fa.kp_mask)
            )

    n_people = n_people_fixed or len(groups)
    skeleton = np.full((max(n_people, 1), n_keypoints, 7), np.nan)

    accepted_reproj = []  # reproj for joints that passed threshold
    rejected_reproj = []  # reproj for joints that failed threshold
    accepted_cams = []

    for pi, group in enumerate(groups):
        for ji in range(n_keypoints):
            observations: list[tuple[CameraCalibration, np.ndarray]] = []
            # Scene camera observations (from zone/epipolar matching)
            for cam_key, local_idx in group.items():
                if cam_key not in cameras:
                    continue
                cam = cameras[cam_key]
                # Respect per-camera kp_mask (scene cameras default: all)
                if cam.kp_mask is not None and ji not in cam.kp_mask:
                    continue
                people = all_poses.get(cam_key, [])
                if local_idx < len(people):
                    kps = people[local_idx]["keypoints"]
                    if ji < len(kps):
                        observations.append((cam, kps[ji]))

            # Face-camera observations (pre-assigned, kp_mask filtered)
            if face_obs_per_person and pi in face_obs_per_person:
                for fc_key, fc_idx, fc_kp_mask in face_obs_per_person[pi]:
                    if ji not in fc_kp_mask:
                        continue
                    if fc_key not in cameras:
                        continue
                    cam = cameras[fc_key]
                    people = all_poses.get(fc_key, [])
                    if fc_idx < len(people):
                        kps = people[fc_idx]["keypoints"]
                        if ji < len(kps):
                            observations.append((cam, kps[ji]))

            if observations:
                pt3d, reproj, n_c = triangulate_point(
                    observations,
                    min_confidence=min_kp_conf,
                    max_reproj_px=max_reproj_px,
                )
                confs = [pt[2] for _, pt in observations if pt[2] > min_kp_conf]
                avg_conf = float(np.mean(confs)) if confs else 0.0

                skeleton[pi, ji, :3] = pt3d
                skeleton[pi, ji, 3] = avg_conf
                skeleton[pi, ji, 4] = reproj if not np.isnan(reproj) else -1
                skeleton[pi, ji, 5] = n_c
                skeleton[pi, ji, 6] = pi

                if not np.isnan(reproj):
                    if not np.isnan(pt3d[0]):  # accepted
                        accepted_reproj.append(reproj)
                        accepted_cams.append(n_c)
                    else:  # rejected by reproj threshold
                        rejected_reproj.append(reproj)

    qc = {
        "n_people": n_people,
        "mean_reproj_accepted_px": float(np.mean(accepted_reproj)) if accepted_reproj else np.nan,
        "median_reproj_accepted_px": float(np.median(accepted_reproj)) if accepted_reproj else np.nan,
        "mean_reproj_rejected_px": float(np.mean(rejected_reproj)) if rejected_reproj else np.nan,
        "n_accepted_joints": len(accepted_reproj),
        "n_rejected_joints": len(rejected_reproj),
        "mean_n_cameras": float(np.mean(accepted_cams)) if accepted_cams else 0,
        "pct_valid_joints": (
            float(np.mean(~np.isnan(skeleton[:n_people, :, 0]))) * 100
            if n_people > 0
            else 0.0
        ),
    }
    return skeleton[:n_people], qc


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def _flip_keypoints_180(
    cam_poses: dict[str, list[list[dict]]],
    cameras: dict[str, CameraCalibration],
    flip_keys: set[str],
) -> None:
    """Flip 2D keypoints 180° for cameras mounted upside-down.

    Transforms ``(x, y) → (W-1-x, H-1-y)`` in-place for each flagged
    camera.  This must be paired with a calibration that has been corrected
    for the flip (see ``recenter_calibration.py --flip-cameras``).
    """
    for ck in flip_keys:
        if ck not in cam_poses or ck not in cameras:
            continue
        W, H = cameras[ck].size  # (width, height)
        for frame_people in cam_poses[ck]:
            for person in frame_people:
                kps = person["keypoints"]  # (K, 3)  [x, y, conf]
                mask = kps[:, 2] > 0  # only flip valid keypoints
                kps[mask, 0] = (W - 1) - kps[mask, 0]
                kps[mask, 1] = (H - 1) - kps[mask, 1]


def run_pipeline(
    calibration_path: Path,
    pose_dirs: list[Path],
    output_path: Path,
    events_jsonl: Path | None = None,
    fps: float = 30.0,
    max_epipolar_px: float = 40.0,
    max_reproj_px: float = 30.0,
    min_kp_conf: float = 0.1,
    camera_zone_specs: list[str] | None = None,
    front_facing_filter: bool = True,
    min_face_conf: float = 0.3,
    frame_log_dir: Path | None = None,
    lsl_dir: Path | None = None,
    session_dir: Path | None = None,
    flip_camera_keys: set[str] | None = None,
    face_camera_specs: list[str] | None = None,
) -> None:
    """End-to-end calibration-aware 3D reconstruction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("CALIBRATION-AWARE MULTI-CAMERA 3D POSE RECONSTRUCTION")
    logger.info("=" * 70)

    # --- load calibration ---
    cameras = load_all_cameras(calibration_path)
    logger.info(f"Loaded {len(cameras)} cameras from calibration:")
    for ck, c in cameras.items():
        dist_mag = np.linalg.norm(c.dist)
        logger.info(f"  {ck}: name={c.name!r}  f={c.K[0,0]:.0f}px  |dist|={dist_mag:.2f}")

    # --- parse camera zones (if defined) ---
    zones: list[CameraZone] | None = None
    n_people_fixed: int | None = None
    if camera_zone_specs:
        zones = parse_camera_zones(camera_zone_specs, cameras)
        n_people_fixed = max(pid for z in zones for pid in z.person_ids) + 1
        zoned_cams = {ck for z in zones for ck in z.cam_keys}
        logger.info(f"\nCamera zones ({n_people_fixed} people):")
        for z in zones:
            cam_names = [cameras[ck].name for ck in z.cam_keys if ck in cameras]
            logger.info(f"  {z.name}: {cam_names} → persons {z.person_ids}")
        shared = [cameras[ck].name for ck in cameras if ck not in zoned_cams]
        if shared:
            logger.info(f"  shared: {shared} → assigned per-frame via epipolar")
        if front_facing_filter:
            logger.info(f"  front-facing filter: ON (min_face_conf={min_face_conf})")
        else:
            logger.info(f"  front-facing filter: OFF")

    # --- parse face-camera assignments ---
    face_assignments: list[FaceCameraAssignment] | None = None
    if face_camera_specs:
        face_assignments = parse_face_cameras(face_camera_specs, cameras)
        if face_assignments:
            logger.info(f"\nFace cameras ({len(face_assignments)}):")
            for fa in face_assignments:
                cname = cameras[fa.cam_key].name if fa.cam_key in cameras else fa.cam_key
                logger.info(
                    f"  {fa.cam_key} ({cname}) → person {fa.person_id}  "
                    f"kp_mask={sorted(fa.kp_mask)}"
                )
            # Ensure n_people_fixed covers face-camera person IDs
            max_face_pid = max(fa.person_id for fa in face_assignments) + 1
            if n_people_fixed is None:
                n_people_fixed = max_face_pid
            else:
                n_people_fixed = max(n_people_fixed, max_face_pid)

    # --- auto-map cameras → pose dirs ---
    logger.info("\nAuto-mapping cameras → pose directories:")
    cam_pose_map = auto_map_cameras_to_pose_dirs(cameras, pose_dirs)
    for ck, pd in sorted(cam_pose_map.items()):
        logger.info(f"  {ck} ({cameras[ck].name}) → {pd.name}")

    if len(cam_pose_map) < 2:
        logger.error("Need ≥2 matched cameras for triangulation.")
        sys.exit(1)

    # --- temporal sync (multi-tier) ---
    logger.info("\nFrame synchronisation:")
    frame_offsets = compute_frame_sync_offsets(
        cameras, cam_pose_map, fps=fps,
        frame_log_dir=frame_log_dir,
        lsl_dir=lsl_dir,
        session_dir=session_dir,
        events_jsonl=events_jsonl,
    )

    # --- load poses ---
    logger.info("\nLoading 2D pose data:")
    cam_poses: dict[str, list[list[dict]]] = {}
    for ck, pd in cam_pose_map.items():
        frames = load_poses_from_dir(pd)
        cam_poses[ck] = frames
        det_frames = sum(1 for f in frames if len(f) > 0)
        logger.info(f"  {ck}: {len(frames)} frames, {det_frames} with detections ({100*det_frames/max(len(frames),1):.0f}%)")

    # --- flip 2D keypoints for upside-down cameras ---
    if flip_camera_keys:
        matched_flips = flip_camera_keys & set(cam_pose_map.keys())
        if matched_flips:
            logger.info(f"\nFlipping 2D keypoints 180° for: {sorted(matched_flips)}")
            _flip_keypoints_180(cam_poses, cameras, matched_flips)

    # --- determine frame range ---
    effective_lengths = {}
    for ck in cam_pose_map:
        offset = frame_offsets.get(ck, 0)
        n_raw = len(cam_poses[ck])
        effective_lengths[ck] = n_raw - offset

    n_frames = min(effective_lengths.values())
    logger.info(f"\nSynchronised frame count: {n_frames}")

    # --- triangulate ---
    logger.info("\n" + "=" * 70)
    logger.info("TRIANGULATION  (undistort → match → DLT → reproject QC)")
    logger.info("=" * 70)

    all_skeletons: list[np.ndarray] = []
    all_qc: list[dict] = []

    for fi in range(n_frames):
        # Gather synchronised poses
        frame_data: dict[str, list[dict]] = {}
        for ck in cam_pose_map:
            raw_idx = fi + frame_offsets.get(ck, 0)
            if raw_idx < len(cam_poses[ck]):
                frame_data[ck] = cam_poses[ck][raw_idx]
            else:
                frame_data[ck] = []

        skel, qc = process_frame(
            fi, frame_data, cameras,
            n_keypoints=25,
            max_epipolar_px=max_epipolar_px,
            max_reproj_px=max_reproj_px,
            min_kp_conf=min_kp_conf,
            zones=zones,
            n_people_fixed=n_people_fixed,
            front_facing_filter=front_facing_filter,
            min_face_conf=min_face_conf,
            face_assignments=face_assignments,
        )
        all_skeletons.append(skel)
        all_qc.append(qc)

        if fi % max(1, n_frames // 20) == 0:
            pct = fi / n_frames * 100
            rp = qc["mean_reproj_accepted_px"]
            rp_str = f"{rp:.1f}px" if not np.isnan(rp) else "n/a"
            logger.info(
                f"  Frame {fi:>5}/{n_frames} ({pct:4.0f}%)  "
                f"people={qc['n_people']}  reproj={rp_str}  "
                f"cams={qc['mean_n_cameras']:.1f}  "
                f"valid={qc['pct_valid_joints']:.0f}%  "
                f"reject={qc['n_rejected_joints']}"
            )

    # --- pack into array ---
    max_people = max(s.shape[0] for s in all_skeletons) if all_skeletons else 1
    output = np.full((n_frames, max_people, 25, 7), np.nan)
    for fi, skel in enumerate(all_skeletons):
        output[fi, : skel.shape[0], :, :] = skel

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output)
    logger.info(f"\nSaved: {output_path}")
    logger.info(f"  Shape: {output.shape}  (frames, people, keypoints, [x,y,z,conf,reproj,n_cams,group])")

    # --- QC summary ---
    logger.info("\n" + "=" * 70)
    logger.info("QC SUMMARY")
    logger.info("=" * 70)

    reproj_accepted = [q["mean_reproj_accepted_px"] for q in all_qc if not np.isnan(q["mean_reproj_accepted_px"])]
    reproj_rejected = [q["mean_reproj_rejected_px"] for q in all_qc if not np.isnan(q["mean_reproj_rejected_px"])]
    valid_all = [q["pct_valid_joints"] for q in all_qc if q["n_people"] > 0]
    cams_all = [q["mean_n_cameras"] for q in all_qc if q["mean_n_cameras"] > 0]
    n_ppl = [q["n_people"] for q in all_qc]

    if reproj_accepted:
        logger.info(f"  Reproj (accepted)    — mean: {np.mean(reproj_accepted):.1f}px  "
                     f"median: {np.median(reproj_accepted):.1f}px  "
                     f"95th: {np.percentile(reproj_accepted, 95):.1f}px")
    if reproj_rejected:
        logger.info(f"  Reproj (rejected)    — mean: {np.mean(reproj_rejected):.1f}px  "
                     f"(joints rejected by >{max_reproj_px}px threshold)")
    if valid_all:
        logger.info(f"  Valid joints         — mean: {np.mean(valid_all):.0f}%")
    if cams_all:
        logger.info(f"  Cameras/joint        — mean: {np.mean(cams_all):.1f}")
    logger.info(f"  People detected      — max: {max(n_ppl)}  mean: {np.mean(n_ppl):.1f}")
    logger.info(f"  Frames w/ detection  — {sum(1 for n in n_ppl if n > 0)}/{n_frames} "
                f"({100*sum(1 for n in n_ppl if n > 0)/max(n_frames,1):.0f}%)")

    if reproj_accepted and np.median(reproj_accepted) > 20:
        logger.warning("  ⚠ High reprojection error — check calibration quality or camera mapping.")
    elif reproj_accepted:
        logger.info("  ✓ Reprojection errors look reasonable.")

    # --- save metadata ---
    meta = {
        "n_frames": n_frames,
        "max_people": max_people,
        "n_cameras": len(cam_pose_map),
        "camera_mapping": {ck: str(pd) for ck, pd in cam_pose_map.items()},
        "frame_offsets": frame_offsets,
        "calibration_file": str(calibration_path),
        "fps": fps,
        "max_epipolar_px": max_epipolar_px,
        "max_reproj_px": max_reproj_px,
        "camera_zones": (
            [{"cameras": z.cam_keys, "person_ids": z.person_ids} for z in zones]
            if zones else None
        ),
        "qc_summary": {
            "mean_reproj_accepted_px": float(np.mean(reproj_accepted)) if reproj_accepted else None,
            "median_reproj_accepted_px": float(np.median(reproj_accepted)) if reproj_accepted else None,
            "mean_reproj_rejected_px": float(np.mean(reproj_rejected)) if reproj_rejected else None,
            "mean_valid_joints_pct": float(np.mean(valid_all)) if valid_all else None,
            "mean_cameras_per_joint": float(np.mean(cams_all)) if cams_all else None,
            "frames_with_detection": sum(1 for n in n_ppl if n > 0),
        },
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata → {meta_path}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ CALIBRATION-AWARE 3D RECONSTRUCTION COMPLETE")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# 9. Validate sub-command
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    """Inspect a 3D skeleton .npy file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fpath = Path(args.file)
    if not fpath.exists():
        logger.error(f"Not found: {fpath}")
        sys.exit(1)

    data = np.load(fpath, allow_pickle=False)
    logger.info(f"File:   {fpath}")
    logger.info(f"Shape:  {data.shape}  (frames, people, keypoints, dims)")
    logger.info(f"Dtype:  {data.dtype}")

    if data.ndim != 4:
        logger.warning("Expected 4-D array.")
        return

    nf, np_, nk, nd = data.shape

    # Validity
    valid_mask = ~np.isnan(data[:, :, :, 0])
    frames_with_data = np.any(valid_mask, axis=(1, 2)).sum()
    logger.info(f"\nFrames with any data: {frames_with_data}/{nf}")

    if nd >= 5:  # has reproj column
        reproj = data[:, :, :, 4]
        good = reproj[~np.isnan(reproj) & (reproj >= 0)]
        if len(good) > 0:
            logger.info(f"Reprojection error: mean={np.mean(good):.1f}px  "
                        f"median={np.median(good):.1f}px  "
                        f"95th={np.percentile(good, 95):.1f}px")
    if nd >= 6:
        ncams = data[:, :, :, 5]
        good = ncams[~np.isnan(ncams) & (ncams > 0)]
        if len(good) > 0:
            logger.info(f"Cameras per joint:  mean={np.mean(good):.1f}  "
                        f"max={np.max(good):.0f}")

    # Sample
    logger.info("\nSample (frame 0, person 0, first 5 keypoints):")
    logger.info(str(data[0, 0, :5, :]))

    # Load companion metadata if exists
    meta_path = fpath.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(f"\nMetadata: {json.dumps(meta.get('qc_summary', {}), indent=2)}")


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration-aware multi-camera 3D pose reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- reconstruct (default) ---
    p_rec = sub.add_parser("reconstruct", help="Run full 3D pipeline")
    p_rec.add_argument("--calibration", type=Path,
                        help="Path to camera calibration .toml")
    p_rec.add_argument("--pose-dirs", type=Path, nargs="+",
                        help="Pose JSON directories (one per camera)")
    p_rec.add_argument("--pose-root", type=Path,
                        help="Root dir containing *_json/ sub-dirs (auto-discover)")
    p_rec.add_argument("--session-dir", type=Path,
                        help="Session directory (auto-discover calibration + events)")
    p_rec.add_argument("--events-jsonl", type=Path,
                        help="Path to ffmpeg_multicap_events.jsonl for sync (tier 4)")
    p_rec.add_argument("--frame-log-dir", type=Path,
                        help="Dir with per-camera frame logs "
                             "({label}_frames.jsonl); best sync accuracy (tier 1)")
    p_rec.add_argument("--lsl-dir", type=Path,
                        help="Dir with LSL progress JSONL "
                             "(ffmpeg_progress_{label}.jsonl); tier 2 sync")
    p_rec.add_argument("--output", type=Path, required=True,
                        help="Output .npy file")
    p_rec.add_argument("--fps", type=float, default=30.0)
    p_rec.add_argument("--max-epipolar-px", type=float, default=40.0,
                        help="Max epipolar distance for person matching (px)")
    p_rec.add_argument("--max-reproj-px", type=float, default=30.0,
                        help="Max reprojection error to accept a joint (px)")
    p_rec.add_argument("--camera-zones", nargs="+", metavar="SPEC",
                        help="Zone-aware person matching.  Each spec: "
                             "'camA+camB:pid1,pid2'.  Camera names accept "
                             "calibration keys (cam_0), short names (cam1), "
                             "or substrings.  Cameras not in any zone are "
                             "shared (e.g. P50).  "
                             "Example: --camera-zones cam1+cam4:0,1 cam2+cam3:2,3")
    p_rec.add_argument("--no-front-facing-filter", action="store_true",
                        help="Disable the front-facing filter that rejects "
                             "back-of-head detections in zone cameras "
                             "(enabled by default when --camera-zones is used)")
    p_rec.add_argument("--min-face-conf", type=float, default=0.3,
                        help="Min confidence on Nose+Eye keypoints to accept "
                             "a detection as front-facing (default: 0.3)")
    p_rec.add_argument("--flip-cameras", nargs="*", metavar="CAM",
                        help="Flip 2D keypoints 180° for specified cameras "
                             "(e.g. cam_0 cam_1 cam_2 cam_3 for upside-down P20s). "
                             "Must be paired with a calibration TOML that was "
                             "also flip-corrected (see recenter_calibration.py).")
    p_rec.add_argument("--face-cameras", nargs="+", metavar="SPEC",
                        help="Dedicated face/upper-body close-up cameras.  "
                             "Each spec: 'cam_name:person_id'.  These cameras "
                             "are pre-assigned to one person and contribute only "
                             "head+upper-body keypoints (indices 0-8, 15-18).  "
                             "Bypasses epipolar matching (picks best detection).  "
                             "Example: --face-cameras cam5:0 cam6:1")

    # --- validate ---
    p_val = sub.add_parser("validate", help="Inspect 3D skeleton .npy")
    p_val.add_argument("--file", type=Path, required=True)

    args = parser.parse_args()

    # Default to "reconstruct" if no sub-command
    if args.command is None:
        if hasattr(args, "output"):
            args.command = "reconstruct"
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "validate":
        cmd_validate(args)
        return

    # --- auto-discover from session-dir ---
    frame_log_dir = getattr(args, 'frame_log_dir', None)
    lsl_dir = getattr(args, 'lsl_dir', None)
    session_dir = args.session_dir

    if args.session_dir:
        sd = args.session_dir
        if not args.calibration:
            cands = list(sd.rglob("*calibration*.toml"))
            if cands:
                args.calibration = cands[0]
        if not args.events_jsonl:
            cands = list(sd.rglob("*events*.jsonl"))
            if cands:
                args.events_jsonl = cands[0]
        # Auto-discover sync dirs
        if not frame_log_dir:
            fl_cand = sd / "frame_logs"
            if fl_cand.is_dir():
                frame_log_dir = fl_cand
        if not lsl_dir:
            lsl_cand = sd / "lsl"
            if lsl_cand.is_dir():
                lsl_dir = lsl_cand

    if not args.calibration or not args.calibration.exists():
        parser.error("--calibration required (or discoverable via --session-dir)")

    # Pose dirs
    pose_dirs: list[Path] = []
    if args.pose_dirs:
        pose_dirs = args.pose_dirs
    elif args.pose_root:
        pose_dirs = sorted(
            [d for d in args.pose_root.iterdir() if d.is_dir() and d.name.endswith("_json")]
        )
    if not pose_dirs:
        parser.error("Provide --pose-dirs, --pose-root, or ensure *_json/ dirs exist")

    # Build flip-camera set
    flip_camera_keys: set[str] | None = None
    if getattr(args, 'flip_cameras', None):
        flip_camera_keys = set(args.flip_cameras)

    run_pipeline(
        calibration_path=args.calibration,
        pose_dirs=pose_dirs,
        output_path=args.output,
        events_jsonl=args.events_jsonl,
        fps=args.fps,
        max_epipolar_px=args.max_epipolar_px,
        max_reproj_px=args.max_reproj_px,
        camera_zone_specs=args.camera_zones,
        front_facing_filter=not args.no_front_facing_filter,
        min_face_conf=args.min_face_conf,
        frame_log_dir=frame_log_dir,
        lsl_dir=lsl_dir,
        session_dir=session_dir,
        flip_camera_keys=flip_camera_keys,
        face_camera_specs=getattr(args, 'face_cameras', None),
    )


if __name__ == "__main__":
    main()
