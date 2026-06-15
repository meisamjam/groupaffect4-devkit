#!/usr/bin/env python3
"""
Multi-camera face-landmark + hand-landmark detection and 3D reconstruction.

Uses the **same** calibrated, synchronised multi-camera setup as
``multicam_pose3d.py`` but runs MediaPipe **FaceLandmarker** (468 mesh
landmarks + 52 ARKit blendshapes) and **HandLandmarker** (21 landmarks
per hand) instead of body-pose detection.

Subcommands
-----------
detect
    Run 2D face + hand detection on each camera video independently.
    Outputs per-frame JSON under ``{output_dir}/{camera_label}_facehand_json/``.

reconstruct
    Load per-camera 2D detections + calibration TOML.
    Associate face/hand detections with persons via body-pose proximity.
    Triangulate face & hand landmarks across cameras.
    Aggregate blendshapes (confidence-weighted).
    Output ``.npz`` with ``face_3d``, ``hand_3d``, ``blendshapes``.

Examples
--------
    # Detect 2D on all videos (first 300 frames only)
    python tools/face_hand_pipeline.py detect \\
        --video-dir new_data/ses-20260202_test/video \\
        --output-dir new_data/ses-20260202_test/facehand \\
        --max-frames 300 \\
        --flip-cameras cam1 cam2 cam3 cam4

    # 3D reconstruction with body-skeleton anchoring
    python tools/face_hand_pipeline.py reconstruct \\
        --calib new_data/ses-20260202_test/video/video_camera_calibration_p50.toml \\
        --detect-dir new_data/ses-20260202_test/facehand \\
        --body-skeleton new_data/ses-20260202_test/video/skeleton_3d_facetest.npy \\
        --output new_data/ses-20260202_test/facehand/face_hand_3d.npz

Dependencies: numpy, scipy, opencv-python, mediapipe ≥0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

N_FACE_LANDMARKS = 478  # 468 mesh + 10 iris landmarks
N_HAND_LANDMARKS = 21
N_BLENDSHAPES = 52

# Canonical blendshape names (ARKit / MediaPipe ordering)
BLENDSHAPE_NAMES: list[str] = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


# ======================================================================
# Model download helpers
# ======================================================================

_MODEL_CACHE = Path(__file__).resolve().parent / ".mediapipe_models"


def _ensure_model(task_name: str, model_file: str, base_url: str) -> Path:
    """Download a MediaPipe task model if not already cached."""
    _MODEL_CACHE.mkdir(exist_ok=True)
    path = _MODEL_CACHE / model_file
    if path.exists():
        return path
    url = f"{base_url}/{model_file}"
    logger.info(f"Downloading {task_name} model: {url}")
    urllib.request.urlretrieve(url, path)
    logger.info(f"  Saved to {path}")
    return path


def _face_model_path() -> Path:
    return _ensure_model(
        "FaceLandmarker",
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models"
        "/face_landmarker/face_landmarker/float16/latest",
    )


def _hand_model_path() -> Path:
    return _ensure_model(
        "HandLandmarker",
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models"
        "/hand_landmarker/hand_landmarker/float16/latest",
    )


# ======================================================================
# 2D Detection (per-camera)
# ======================================================================

@dataclass
class _UndistortHelper:
    """Pre-computed undistortion maps for a single camera."""
    map1: np.ndarray
    map2: np.ndarray
    new_K: np.ndarray  # optimal new camera matrix

    @classmethod
    def from_calib(cls, K: np.ndarray, dist: np.ndarray,
                   size: tuple[int, int]) -> _UndistortHelper:
        w, h = size
        # Use K as the output camera matrix so that undistorted pixel
        # coordinates live in the same intrinsic space as the original K.
        # This avoids having to swap projection matrices during
        # triangulation: P = K @ [R|t] stays correct.
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (w, h), cv2.CV_32FC1,
        )
        return cls(map1=map1, map2=map2, new_K=K.copy())

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def undistort_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a distorted-image pixel to the undistorted image.

        Uses bilinear interpolation on the remap tables (forward map).
        map1/map2 give: for each undistorted pixel, the source distorted x/y.
        So we need to INVERT.  Quick approach: search in a local window.
        """
        # The remap tables map undistorted→distorted.
        # For a fast inverse, build a KD-tree would be ideal, but for
        # per-point usage we use a local search around the expected area.
        h, w = self.map1.shape[:2]
        ix, iy = int(round(x)), int(round(y))

        best_err = float("inf")
        best_xu, best_yu = float(x), float(y)
        pad = 80  # search radius

        x1s = max(0, ix - pad)
        x2s = min(w, ix + pad)
        y1s = max(0, iy - pad)
        y2s = min(h, iy + pad)

        # Vectorised search in local patch
        patch_x = self.map1[y1s:y2s, x1s:x2s]
        patch_y = self.map2[y1s:y2s, x1s:x2s]
        err = (patch_x - x) ** 2 + (patch_y - y) ** 2
        min_idx = np.unravel_index(np.argmin(err), err.shape)
        best_yu = float(y1s + min_idx[0])
        best_xu = float(x1s + min_idx[1])
        best_err = float(err[min_idx])

        if best_err > 25.0:  # More than 5px off, expand search
            x1s = max(0, ix - 200)
            x2s = min(w, ix + 200)
            y1s = max(0, iy - 200)
            y2s = min(h, iy + 200)
            patch_x = self.map1[y1s:y2s, x1s:x2s]
            patch_y = self.map2[y1s:y2s, x1s:x2s]
            err = (patch_x - x) ** 2 + (patch_y - y) ** 2
            min_idx = np.unravel_index(np.argmin(err), err.shape)
            best_yu = float(y1s + min_idx[0])
            best_xu = float(x1s + min_idx[1])

        return best_xu, best_yu


def _load_openpose_head_rois(
    openpose_dir: Path,
    frame_idx: int,
    width: int,
    height: int,
    flip_180: bool = False,
    crop_scale: float = 3.0,
    min_crop_px: int = 200,
    min_head_conf: float = 0.2,
    undistort: _UndistortHelper | None = None,
) -> list[tuple[int, int, int, int]]:
    """Extract head ROI bounding boxes from OpenPose BODY_25 JSON.

    Uses keypoints Nose(0), REye(15), LEye(16), REar(17), LEar(18)
    to compute a generous crop around each detected head.

    When ``undistort`` is provided, keypoints are remapped from
    distorted → undistorted coordinates before computing the crop
    (since the frame will also be undistorted).

    Returns list of (x1, y1, x2, y2) in *post-flip, post-undistort* pixel
    coordinates.
    """
    # OpenPose files are typically like 00000000_keypoints.json
    json_path = openpose_dir / f"{frame_idx:012d}_keypoints.json"
    if not json_path.exists():
        # Try alternate naming: 8-digit
        json_path = openpose_dir / f"{frame_idx:08d}_keypoints.json"
    if not json_path.exists():
        # Try listing the directory for the n-th file
        jsons = sorted(openpose_dir.glob("*_keypoints.json"))
        if frame_idx < len(jsons):
            json_path = jsons[frame_idx]
        else:
            return []

    with open(json_path) as f:
        data = json.load(f)

    rois: list[tuple[int, int, int, int]] = []
    for person in data.get("people", []):
        kps_flat = person.get("pose_keypoints_2d", [])
        if len(kps_flat) < 19 * 3:
            continue
        kps = np.array(kps_flat, dtype=np.float64).reshape(-1, 3)

        head_idx = [0, 15, 16, 17, 18]
        head_pts: list[list[float]] = []
        for ki in head_idx:
            if kps[ki, 2] > min_head_conf:
                px, py = float(kps[ki, 0]), float(kps[ki, 1])
                # Undistort BEFORE flip: remap tables are in raw-camera
                # orientation, so feed raw-orientation coordinates first.
                if undistort is not None:
                    px, py = undistort.undistort_point(px, py)
                if flip_180:
                    px, py = width - 1 - px, height - 1 - py
                head_pts.append([px, py])

        if len(head_pts) < 2:
            continue

        pts = np.array(head_pts)
        cx, cy = np.mean(pts, axis=0)
        head_span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        crop_sz = int(max(head_span * crop_scale, min_crop_px))

        x1 = max(0, int(cx - crop_sz // 2))
        y1 = max(0, int(cy - crop_sz // 2))
        x2 = min(width, x1 + crop_sz)
        y2 = min(height, y1 + crop_sz)

        if x2 - x1 >= 80 and y2 - y1 >= 80:
            rois.append((x1, y1, x2, y2))

    return rois


def detect_faces_and_hands(
    video_path: Path,
    output_dir: Path,
    *,
    max_frames: int = 0,
    num_faces: int = 4,
    num_hands: int = 8,
    min_face_det_conf: float = 0.4,
    min_hand_det_conf: float = 0.4,
    flip_180: bool = False,
    openpose_dir: Path | None = None,
    camera_K: np.ndarray | None = None,
    camera_dist: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run FaceLandmarker + HandLandmarker on a video, write per-frame JSON.

    When ``openpose_dir`` is provided, face detection uses body-pose-guided
    head crops instead of the full frame.  This dramatically improves
    detection on wide-angle / barrel-distorted cameras (Jabra PanaCast)
    where the face occupies a small portion of the frame.

    When ``camera_K`` and ``camera_dist`` are provided, each frame is
    undistorted before detection.  This ensures face/hand landmark
    coordinates are in pinhole (undistorted) image space — essential for
    accurate multi-camera triangulation with extreme-distortion lenses.

    Hand detection always runs on the full frame (VIDEO mode) since it
    works reliably without cropping.

    Returns detection-rate statistics.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Face: use IMAGE mode with crops when openpose is available,
    # else VIDEO mode on full frame (less reliable on wide-angle).
    use_face_crops = openpose_dir is not None and openpose_dir.is_dir()
    face_mode = RunningMode.IMAGE if use_face_crops else RunningMode.VIDEO
    face_opts_kwargs: dict[str, Any] = {
        "base_options": BaseOptions(model_asset_path=str(_face_model_path())),
        "running_mode": face_mode,
        "num_faces": 1 if use_face_crops else num_faces,  # 1 per crop
        "min_face_detection_confidence": 0.15 if use_face_crops else min_face_det_conf,
        "min_face_presence_confidence": 0.15 if use_face_crops else min_face_det_conf,
        "output_face_blendshapes": True,
        "output_facial_transformation_matrixes": True,
    }
    if not use_face_crops:
        face_opts_kwargs["min_tracking_confidence"] = 0.4
    face_opts = FaceLandmarkerOptions(**face_opts_kwargs)

    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_hand_model_path())),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_det_conf,
        min_hand_presence_confidence=min_hand_det_conf,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {"error": str(video_path)}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_process = min(max_frames, total) if max_frames > 0 else total

    # Prepare optional image undistortion
    ud_helper: _UndistortHelper | None = None
    if camera_K is not None and camera_dist is not None:
        if not np.allclose(camera_dist, 0):
            ud_helper = _UndistortHelper.from_calib(
                camera_K, camera_dist, (width, height),
            )
            logger.info(f"    Undistortion enabled (k1={camera_dist[0]:.4f})")

    mode_label = "CROP" if use_face_crops else "FULL"
    ud_label = "+UNDIST" if ud_helper else ""
    logger.info(f"  {video_path.name}: {width}x{height} @ {fps:.0f}fps, "
                f"processing {n_process}/{total} frames "
                f"[face={mode_label}{ud_label}]"
                + (" [FLIP 180°]" if flip_180 else ""))

    frames_with_face = 0
    frames_with_hand = 0
    frame_idx = 0

    with (
        FaceLandmarker.create_from_options(face_opts) as face_lm,
        HandLandmarker.create_from_options(hand_opts) as hand_lm,
    ):
        while frame_idx < n_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort BEFORE flip: remap tables are computed for the
            # raw camera orientation (matching K, dist).  The flip is
            # only needed so MediaPipe sees upright faces.
            if ud_helper is not None:
                frame = ud_helper.undistort_image(frame)

            if flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = int(frame_idx * 1000 / fps)

            # --- Face detection ---
            faces_out: list[dict[str, Any]] = []

            if use_face_crops:
                # Body-pose-guided face crops
                rois = _load_openpose_head_rois(
                    openpose_dir, frame_idx, width, height,
                    flip_180=flip_180, undistort=ud_helper,
                )
                for (x1, y1, x2, y2) in rois:
                    crop = frame[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=crop_rgb,
                    )
                    face_result = face_lm.detect(crop_img)

                    crop_w, crop_h = x2 - x1, y2 - y1
                    for fi_det in range(len(face_result.face_landmarks)):
                        lms = face_result.face_landmarks[fi_det]
                        # Remap crop-relative coords → full-frame coords
                        lm_flat: list[float] = []
                        for lm in lms:
                            px = x1 + lm.x * crop_w
                            py = y1 + lm.y * crop_h
                            pz = lm.z * crop_w  # z uses x-scale
                            # Un-flip back to raw camera orientation so
                            # coordinates match K, R, t for triangulation.
                            if flip_180:
                                px = width - 1 - px
                                py = height - 1 - py
                            # FaceLandmarker: visibility is always 0;
                            # use presence instead (indicates landmark quality).
                            pres = getattr(lm, "presence", None)
                            vis = pres if (pres is not None and pres > 0) else 1.0
                            lm_flat.extend([px, py, pz, vis])

                        bs_dict: dict[str, float] = {}
                        if fi_det < len(face_result.face_blendshapes):
                            for cat in face_result.face_blendshapes[fi_det]:
                                bs_dict[cat.category_name] = round(cat.score, 4)

                        tf_matrix: list[list[float]] | None = None
                        if fi_det < len(face_result.facial_transformation_matrixes):
                            tf_matrix = (
                                face_result.facial_transformation_matrixes[fi_det]
                                .tolist()
                            )

                        faces_out.append({
                            "landmarks_2d": lm_flat,
                            "blendshapes": bs_dict,
                            "transform_matrix": tf_matrix,
                            "crop_roi": [x1, y1, x2, y2],
                        })
            else:
                # Full-frame face detection (fallback)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                face_result = face_lm.detect_for_video(mp_image, ts_ms)
                for fi_det in range(len(face_result.face_landmarks)):
                    lms = face_result.face_landmarks[fi_det]
                    lm_flat = []
                    for lm in lms:
                        pres = getattr(lm, "presence", None)
                        vis = pres if (pres is not None and pres > 0) else 1.0
                        px = lm.x * width
                        py = lm.y * height
                        if flip_180:
                            px = width - 1 - px
                            py = height - 1 - py
                        lm_flat.extend([px, py, lm.z * width, vis])

                    bs_dict = {}
                    if fi_det < len(face_result.face_blendshapes):
                        for cat in face_result.face_blendshapes[fi_det]:
                            bs_dict[cat.category_name] = round(cat.score, 4)

                    tf_matrix = None
                    if fi_det < len(face_result.facial_transformation_matrixes):
                        tf_matrix = (
                            face_result.facial_transformation_matrixes[fi_det]
                            .tolist()
                        )

                    faces_out.append({
                        "landmarks_2d": lm_flat,
                        "blendshapes": bs_dict,
                        "transform_matrix": tf_matrix,
                    })

            # --- Hand detection (always full-frame VIDEO mode) ---
            mp_image_hand = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_lm.detect_for_video(mp_image_hand, ts_ms)
            hands_out: list[dict[str, Any]] = []
            for hi in range(len(hand_result.hand_landmarks)):
                lms = hand_result.hand_landmarks[hi]
                lm_flat = []
                for lm in lms:
                    px = lm.x * width
                    py = lm.y * height
                    if flip_180:
                        px = width - 1 - px
                        py = height - 1 - py
                    # HandLandmarker: visibility is always 0; use
                    # presence (like FaceLandmarker) or default to 1.0.
                    pres = getattr(lm, "presence", None)
                    vis = pres if (pres is not None and pres > 0) else 1.0
                    lm_flat.extend([
                        px, py,
                        lm.z * width,
                        vis,
                    ])

                handedness_label = "Unknown"
                handedness_score = 0.0
                if hi < len(hand_result.handedness):
                    cats = hand_result.handedness[hi]
                    if cats:
                        handedness_label = cats[0].category_name
                        handedness_score = round(cats[0].score, 4)

                hands_out.append({
                    "landmarks_2d": lm_flat,
                    "handedness": handedness_label,
                    "handedness_score": handedness_score,
                })

            if faces_out:
                frames_with_face += 1
            if hands_out:
                frames_with_hand += 1

            # Write JSON
            out_data = {
                "version": 1.1,
                "frame": frame_idx,
                "image_size": [width, height],
                "face_mode": mode_label,
                "undistorted": ud_helper is not None,
                "faces": faces_out,
                "hands": hands_out,
            }
            json_path = output_dir / f"{frame_idx:08d}_facehand.json"
            with open(json_path, "w") as f:
                json.dump(out_data, f)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(
                    f"    frame {frame_idx}/{n_process} "
                    f"({frame_idx * 100 // n_process}%)  "
                    f"faces={frames_with_face}  hands={frames_with_hand}"
                )

    cap.release()

    stats = {
        "video": video_path.name,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "processed_frames": frame_idx,
        "total_frames": total,
        "frames_with_face": frames_with_face,
        "frames_with_hand": frames_with_hand,
        "face_detect_pct": round(frames_with_face / max(frame_idx, 1) * 100, 1),
        "hand_detect_pct": round(frames_with_hand / max(frame_idx, 1) * 100, 1),
        "face_mode": mode_label,
    }
    logger.info(
        f"  Done: face={stats['face_detect_pct']}% [{mode_label}]  "
        f"hand={stats['hand_detect_pct']}%  "
        f"({frame_idx} frames)"
    )
    return stats


# ======================================================================
# 2D detection loader
# ======================================================================

def load_facehand_frames(
    json_dir: Path,
) -> list[dict[str, Any]]:
    """Load per-frame face/hand JSONs from a detection directory.

    Returns list indexed by frame number.  Each entry has ``faces`` and
    ``hands`` lists with landmarks parsed to numpy arrays, plus
    ``undistorted`` flag indicating coordinate space.
    """
    json_files = sorted(json_dir.glob("*_facehand.json"))
    frames: list[dict[str, Any]] = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        # Preserve undistorted flag from detection
        data.setdefault("undistorted", False)

        # Parse face landmarks to (N, 4) numpy
        for face in data.get("faces", []):
            flat = face.get("landmarks_2d", [])
            if flat:
                face["lm_np"] = np.array(flat, dtype=np.float64).reshape(-1, 4)
            else:
                face["lm_np"] = np.zeros((0, 4), dtype=np.float64)

        # Parse hand landmarks to (21, 4) numpy
        for hand in data.get("hands", []):
            flat = hand.get("landmarks_2d", [])
            if flat:
                hand["lm_np"] = np.array(flat, dtype=np.float64).reshape(-1, 4)
            else:
                hand["lm_np"] = np.zeros((0, 4), dtype=np.float64)

        frames.append(data)
    return frames


# ======================================================================
# Calibration loading (reuse from multicam_pose3d)
# ======================================================================

def _load_toml(toml_path: Path) -> dict:
    """Load a TOML file (Python 3.11+ or toml package)."""
    try:
        import tomllib
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import toml  # type: ignore[import-untyped]
        with open(toml_path) as f:
            return toml.load(f)


@dataclass
class CameraCalib:
    """Lightweight camera calibration for face/hand pipeline."""
    key: str
    name: str
    size: tuple[int, int]
    K: np.ndarray       # 3×3 intrinsic
    dist: np.ndarray     # distortion coeffs
    R: np.ndarray        # 3×3 rotation
    t: np.ndarray        # 3×1 translation
    P: np.ndarray        # 3×4 projection

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        if np.allclose(self.dist, 0):
            return pts
        pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
        ud = cv2.undistortPoints(pts_f32, self.K, self.dist, P=self.K)
        return ud.reshape(-1, 2).astype(np.float64)

    def reproject(self, pts_3d: np.ndarray) -> np.ndarray:
        N = pts_3d.shape[0]
        h = np.hstack([pts_3d, np.ones((N, 1))])
        proj = (self.P @ h.T).T
        return proj[:, :2] / proj[:, 2:3]


def load_cameras(toml_path: Path) -> dict[str, CameraCalib]:
    """Load calibration TOML → dict of camera_key → CameraCalib."""
    from scipy.spatial.transform import Rotation

    data = _load_toml(toml_path)
    cameras: dict[str, CameraCalib] = {}
    for k in sorted(data.keys()):
        if not k.startswith("cam_"):
            continue
        cd = data[k]
        K = np.array(cd["matrix"], dtype=np.float64)
        dist = np.array(cd.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64)
        rvec = np.array(cd.get("rotation", [0, 0, 0]), dtype=np.float64)
        tvec = np.array(cd.get("translation", [0, 0, 0]), dtype=np.float64)
        R = Rotation.from_rotvec(rvec).as_matrix()
        t = tvec.reshape(3, 1)
        Rt = np.hstack([R, t])
        P = K @ Rt
        cameras[k] = CameraCalib(
            key=k, name=cd.get("name", k),
            size=tuple(cd.get("size", [1920, 1080])),
            K=K, dist=dist, R=R, t=t, P=P,
        )
    return cameras


# ======================================================================
# Person ↔ detection association (body skeleton anchoring)
# ======================================================================

def _body_head_centroid(skeleton_frame: np.ndarray, person_id: int) -> np.ndarray | None:
    """Get 2D head centroid from body skeleton for a person.

    ``skeleton_frame`` is ``(P, K, 7)``.  Uses BODY_25 keypoints:
    0=Nose, 15=REye, 16=LEye, 17=REar, 18=LEar.
    Returns (x, y, z) world-coords centroid or None.
    """
    if person_id >= skeleton_frame.shape[0]:
        return None
    head_kps = [0, 15, 16, 17, 18]
    pts = []
    for ki in head_kps:
        if ki < skeleton_frame.shape[1]:
            conf = skeleton_frame[person_id, ki, 3]
            if conf > 0.1 and not np.isnan(skeleton_frame[person_id, ki, 0]):
                pts.append(skeleton_frame[person_id, ki, :3])
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _body_wrist_3d(
    skeleton_frame: np.ndarray, person_id: int, side: str,
) -> np.ndarray | None:
    """Get 3D wrist position from body skeleton.

    side: 'Left' or 'Right' (BODY_25: 4=RWrist, 7=LWrist).
    """
    if person_id >= skeleton_frame.shape[0]:
        return None
    ki = 7 if side == "Left" else 4
    if ki >= skeleton_frame.shape[1]:
        return None
    conf = skeleton_frame[person_id, ki, 3]
    if conf > 0.1 and not np.isnan(skeleton_frame[person_id, ki, 0]):
        return skeleton_frame[person_id, ki, :3]
    return None


def associate_faces_to_persons(
    face_detections: list[dict[str, Any]],
    camera: CameraCalib,
    skeleton_frame: np.ndarray,
    n_people: int,
) -> dict[int, int]:
    """Map face detections → person IDs using body-skeleton head projection.

    Projects each person's 3D head centroid into this camera's image
    and finds the closest face detection (by distance to nose landmark #1).

    Returns {person_id: face_detection_index}.
    """
    if not face_detections:
        return {}

    # Get face centroids (use nose tip = landmark index 1, or centroid of all)
    face_centres: list[np.ndarray] = []
    for fd in face_detections:
        lm = fd.get("lm_np")
        if lm is not None and len(lm) > 4:
            # Use landmarks around nose/eyes for centroid
            center_idxs = [1, 4, 5, 6, 168]  # nose tip, inner eye corners, bridge
            pts = lm[center_idxs, :2]
            face_centres.append(np.mean(pts, axis=0))
        else:
            face_centres.append(np.array([0.0, 0.0]))

    assignments: dict[int, int] = {}
    used_faces: set[int] = set()

    for pid in range(n_people):
        head_3d = _body_head_centroid(skeleton_frame, pid)
        if head_3d is None:
            continue
        # Project head centroid into camera 2D
        head_2d = camera.reproject(head_3d.reshape(1, 3))[0]

        # Find nearest unassigned face
        best_fi = -1
        best_dist = float("inf")
        for fi, fc in enumerate(face_centres):
            if fi in used_faces:
                continue
            d = float(np.linalg.norm(head_2d - fc))
            if d < best_dist:
                best_dist = d
                best_fi = fi

        if best_fi >= 0 and best_dist < max(camera.size) * 0.3:
            assignments[pid] = best_fi
            used_faces.add(best_fi)

    return assignments


def associate_hands_to_persons(
    hand_detections: list[dict[str, Any]],
    camera: CameraCalib,
    skeleton_frame: np.ndarray,
    n_people: int,
) -> dict[int, list[int]]:
    """Map hand detections → person IDs using body-skeleton wrist projection.

    Returns {person_id: [hand_det_idx, ...]} (up to 2 hands per person).
    """
    if not hand_detections:
        return {}

    # Get hand wrist 2D position (landmark 0 = wrist)
    hand_centres: list[np.ndarray] = []
    for hd in hand_detections:
        lm = hd.get("lm_np")
        if lm is not None and len(lm) > 0:
            hand_centres.append(lm[0, :2])  # wrist
        else:
            hand_centres.append(np.array([0.0, 0.0]))

    assignments: dict[int, list[int]] = {}
    used_hands: set[int] = set()

    for pid in range(n_people):
        pid_hands: list[int] = []
        for side in ("Left", "Right"):
            wrist_3d = _body_wrist_3d(skeleton_frame, pid, side)
            if wrist_3d is None:
                continue
            wrist_2d = camera.reproject(wrist_3d.reshape(1, 3))[0]

            best_hi = -1
            best_dist = float("inf")
            for hi, hc in enumerate(hand_centres):
                if hi in used_hands:
                    continue
                d = float(np.linalg.norm(wrist_2d - hc))
                if d < best_dist:
                    best_dist = d
                    best_hi = hi

            if best_hi >= 0 and best_dist < max(camera.size) * 0.25:
                pid_hands.append(best_hi)
                used_hands.add(best_hi)

        if pid_hands:
            assignments[pid] = pid_hands

    return assignments


# ======================================================================
# Nonlinear triangulation (handles extreme lens distortion)
# ======================================================================

def _project_distorted(cam: CameraCalib, pt3d: np.ndarray) -> np.ndarray:
    """Project a 3D point to distorted pixel coordinates.

    Uses the full OpenCV distortion model (k1, k2, p1, p2, k3) in the
    FORWARD direction — always well-defined regardless of distortion
    magnitude or monotonicity.
    """
    pt_cam = cam.R @ pt3d.reshape(3, 1) + cam.t
    xc, yc, zc = pt_cam.flatten()
    if abs(zc) < 1e-12:
        return np.array([np.nan, np.nan])
    xn, yn = xc / zc, yc / zc
    r2 = xn ** 2 + yn ** 2

    d = cam.dist
    k1 = float(d[0]) if len(d) > 0 else 0.0
    k2 = float(d[1]) if len(d) > 1 else 0.0
    p1 = float(d[2]) if len(d) > 2 else 0.0
    p2 = float(d[3]) if len(d) > 3 else 0.0
    k3 = float(d[4]) if len(d) > 4 else 0.0

    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
    yd = yn * radial + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn

    px = cam.K[0, 0] * xd + cam.K[0, 2]
    py = cam.K[1, 1] * yd + cam.K[1, 2]
    return np.array([px, py])


def _distortion_is_monotonic(cam: CameraCalib, pt2d: np.ndarray) -> bool:
    """Check whether the radial distortion function is monotonic at *pt2d*.

    For k1 < 0 (barrel distortion), the function
        r_d = r_u × (1 + k1·r_u²)
    has a critical radius  r_crit = 1 / √(−3k1).
    Points with normalized radius > r_crit lie in the non-monotonic
    region where ``cv2.undistortPoints`` diverges and the forward
    distortion model can fold the image.

    Returns True when it is safe to use the camera for this observation.
    """
    k1 = float(cam.dist[0]) if len(cam.dist) > 0 else 0.0
    if k1 >= 0:
        return True  # pincushion / zero distortion is always monotonic
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    xn = (float(pt2d[0]) - cx) / fx
    yn = (float(pt2d[1]) - cy) / fy
    r2 = xn ** 2 + yn ** 2
    # Monotonicity condition: 1 + 3·k1·r² > 0
    return (1.0 + 3.0 * k1 * r2) > 0.0


def triangulate_point(
    observations: list[tuple[CameraCalib, np.ndarray]],
    max_reproj_px: float = 30.0,
    already_undistorted: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Triangulate a single 3D point from ≥2 camera observations.

    Uses a two-step approach:
    1. Filter out cameras whose distortion model is non-monotonic at the
       observed radius (their calibration cannot be trusted there).
    2. DLT for initial estimate using properly undistorted observations.
    3. Levenberg-Marquardt refinement against the full forward distortion
       model for the remaining cameras.

    When ``already_undistorted`` is True, both the DLT and reprojection
    error are computed in undistorted pixel space (no distortion applied).

    Returns (xyz, mean_reproj_error, n_cameras).
    """
    from scipy.optimize import least_squares

    if len(observations) < 2:
        return np.full(3, np.nan), np.nan, 0

    # ---- Filter cameras with non-monotonic distortion ----
    if not already_undistorted:
        good_obs = [
            (cam, pt) for cam, pt in observations
            if _distortion_is_monotonic(cam, pt)
        ]
        if len(good_obs) < 2:
            return np.full(3, np.nan), np.nan, 0
    else:
        good_obs = observations

    # ---- Step 1: DLT initial estimate ----
    A_rows: list[np.ndarray] = []
    for cam, pt2d in good_obs:
        if already_undistorted or np.allclose(cam.dist, 0):
            ud = pt2d[:2].astype(np.float64)
        else:
            ud = cv2.undistortPoints(
                pt2d[:2].reshape(1, 1, 2).astype(np.float32),
                cam.K, cam.dist, P=cam.K,
            ).reshape(2).astype(np.float64)
        P = cam.P
        A_rows.append(ud[0] * P[2, :] - P[0, :])
        A_rows.append(ud[1] * P[2, :] - P[1, :])

    A_mat = np.array(A_rows)
    _, _, Vh = np.linalg.svd(A_mat)
    X_h = Vh[-1, :]
    X0 = X_h[:3] / X_h[3]

    if already_undistorted:
        # No distortion to model — DLT result is final.
        reproj_errs = []
        for cam, pt2d in good_obs:
            proj = cam.reproject(X0.reshape(1, 3))[0]
            reproj_errs.append(float(np.linalg.norm(proj - pt2d[:2])))
        mean_err = float(np.mean(reproj_errs))
        if mean_err > max_reproj_px:
            return np.full(3, np.nan), mean_err, len(good_obs)
        return X0, mean_err, len(good_obs)

    # ---- Step 2: LM refinement with full distortion model ----
    def residuals(x3d: np.ndarray) -> np.ndarray:
        res = np.empty(len(good_obs) * 2)
        for i, (cam, pt2d) in enumerate(good_obs):
            proj = _project_distorted(cam, x3d)
            res[2 * i] = proj[0] - pt2d[0]
            res[2 * i + 1] = proj[1] - pt2d[1]
        return res

    result = least_squares(residuals, X0, method="lm", max_nfev=100)
    pt3d = result.x

    # Compute mean reprojection error in distorted pixel space
    r = result.fun
    reproj_errs = [
        float(np.sqrt(r[2 * i] ** 2 + r[2 * i + 1] ** 2))
        for i in range(len(good_obs))
    ]
    mean_err = float(np.mean(reproj_errs))

    if mean_err > max_reproj_px:
        return np.full(3, np.nan), mean_err, len(good_obs)

    return pt3d, mean_err, len(good_obs)


# ======================================================================
# 3D Reconstruction
# ======================================================================

def reconstruct_3d(
    cameras: dict[str, CameraCalib],
    detect_dirs: dict[str, Path],
    body_skeleton: np.ndarray | None,
    n_people: int = 4,
    max_reproj_px: float = 40.0,
) -> dict[str, np.ndarray]:
    """Multi-camera face/hand 3D reconstruction.

    Args:
        cameras: cam_key → CameraCalib
        detect_dirs: cam_key → Path to detection JSON directory
        body_skeleton: (F, P, K, 7) body skeleton for person anchoring
        n_people: number of people to track
        max_reproj_px: max reprojection error for triangulation

    Returns dict with:
        face_3d:      (F, P, 468, 4) = [x, y, z, reproj_err]
        hand_3d:      (F, P, 2, 21, 4)  (dim 2 = left/right)
        blendshapes:  (F, P, 52)
        metadata:     dict with stats
    """
    # Load all per-camera detections
    cam_keys = sorted(detect_dirs.keys())
    cam_frames: dict[str, list[dict[str, Any]]] = {}
    cam_undistorted: dict[str, bool] = {}
    min_frames = None
    for ck in cam_keys:
        frames = load_facehand_frames(detect_dirs[ck])
        cam_frames[ck] = frames
        # Check if detections were run on undistorted images
        is_ud = frames[0].get("undistorted", False) if frames else False
        cam_undistorted[ck] = is_ud
        if min_frames is None or len(frames) < min_frames:
            min_frames = len(frames)
        logger.info(f"  {ck}: {len(frames)} frames loaded"
                     f" (undistorted={is_ud})")

    if min_frames is None or min_frames == 0:
        logger.error("No detection data found")
        return {}

    n_frames = min_frames
    if body_skeleton is not None:
        n_frames = min(n_frames, body_skeleton.shape[0])

    logger.info(f"  Reconstructing {n_frames} frames, {n_people} people")

    # Allocate output arrays
    face_3d = np.full((n_frames, n_people, N_FACE_LANDMARKS, 4), np.nan, dtype=np.float32)
    hand_3d = np.full((n_frames, n_people, 2, N_HAND_LANDMARKS, 4), np.nan, dtype=np.float32)
    blendshapes = np.full((n_frames, n_people, N_BLENDSHAPES), np.nan, dtype=np.float32)

    # Stats
    face_tri_count = 0
    hand_tri_count = 0
    bs_count = 0

    for fi in range(n_frames):
        skel_frame = body_skeleton[fi] if body_skeleton is not None else None

        # --- Per-camera: associate faces and hands with persons ---
        # face_obs[person_id] = list of (cam, face_landmarks_np)
        face_obs: dict[int, list[tuple[CameraCalib, np.ndarray]]] = {
            pid: [] for pid in range(n_people)
        }
        # hand_obs[person_id][side_idx] = list of (cam, hand_landmarks_np)
        #   side_idx: 0=Left, 1=Right
        hand_obs: dict[int, dict[int, list[tuple[CameraCalib, np.ndarray]]]] = {
            pid: {0: [], 1: []} for pid in range(n_people)
        }
        # blendshape_obs[person_id] = list of (confidence, bs_vector)
        bs_obs: dict[int, list[tuple[float, np.ndarray]]] = {
            pid: [] for pid in range(n_people)
        }

        for ck in cam_keys:
            if fi >= len(cam_frames[ck]):
                continue
            frame_data = cam_frames[ck][fi]
            cam = cameras[ck]
            faces = frame_data.get("faces", [])
            hands = frame_data.get("hands", [])

            # Decide association strategy: use skeleton if it has
            # valid data for this frame, otherwise fall back to
            # sequential assignment.
            use_skeleton = False
            if skel_frame is not None:
                # Check if ANY person has a valid head or wrist keypoint
                for pid in range(min(n_people, skel_frame.shape[0])):
                    for ki in [0, 4, 7, 15, 16]:  # nose, Rwrist, Lwrist, Reye, Leye
                        if ki < skel_frame.shape[1]:
                            val = skel_frame[pid, ki, 0]
                            conf = skel_frame[pid, ki, 3]
                            if not np.isnan(val) and conf > 0.1:
                                use_skeleton = True
                                break
                    if use_skeleton:
                        break

            if use_skeleton:
                face_assign = associate_faces_to_persons(
                    faces, cam, skel_frame, n_people,
                )
                hand_assign = associate_hands_to_persons(
                    hands, cam, skel_frame, n_people,
                )
            else:
                # Fallback: assign faces sequentially (1 person = 1 face)
                face_assign = {i: i for i in range(min(len(faces), n_people))}

                # Associate hands to persons via face centroid proximity.
                # For each person with an assigned face, find the
                # nearest hand detections (within a generous radius).
                hand_assign: dict[int, list[int]] = {}
                used_hands: set[int] = set()
                for pid, fdi in face_assign.items():
                    if fdi >= len(faces):
                        continue
                    face = faces[fdi]
                    face_lm = face.get("lm_np")
                    if face_lm is None or len(face_lm) == 0:
                        continue
                    # Face centroid
                    face_cx = float(np.mean(face_lm[:, 0]))
                    face_cy = float(np.mean(face_lm[:, 1]))

                    # Score each hand by distance to face centroid
                    scored: list[tuple[float, int]] = []
                    for hi, hand in enumerate(hands):
                        if hi in used_hands:
                            continue
                        hand_lm = hand.get("lm_np")
                        if hand_lm is None or len(hand_lm) == 0:
                            continue
                        # Use wrist (landmark 0) position
                        wx = float(hand_lm[0, 0])
                        wy = float(hand_lm[0, 1])
                        dist = np.sqrt(
                            (face_cx - wx) ** 2 + (face_cy - wy) ** 2
                        )
                        scored.append((dist, hi))

                    scored.sort()
                    pid_hands: list[int] = []
                    # Take up to 2 closest hands within 40 % of image diagonal
                    max_dist = 0.4 * np.sqrt(
                        cam.size[0] ** 2 + cam.size[1] ** 2
                    )
                    for d, hi in scored:
                        if d > max_dist:
                            break
                        pid_hands.append(hi)
                        used_hands.add(hi)
                        if len(pid_hands) >= 2:
                            break
                    if pid_hands:
                        hand_assign[pid] = pid_hands

            # Collect face observations
            for pid, fdi in face_assign.items():
                if fdi >= len(faces):
                    continue
                face = faces[fdi]
                lm = face.get("lm_np")
                if lm is None or len(lm) == 0:
                    continue
                face_obs[pid].append((cam, lm))

                # Collect blendshape
                bs_dict = face.get("blendshapes", {})
                if bs_dict:
                    bs_vec = np.zeros(N_BLENDSHAPES, dtype=np.float32)
                    for bi, name in enumerate(BLENDSHAPE_NAMES):
                        bs_vec[bi] = bs_dict.get(name, 0.0)
                    # Confidence = mean visibility of face landmarks
                    conf = float(np.mean(lm[:, 3]))
                    bs_obs[pid].append((conf, bs_vec))

            # Collect hand observations
            for pid, hdi_list in hand_assign.items():
                for hdi in hdi_list:
                    if hdi >= len(hands):
                        continue
                    hand = hands[hdi]
                    lm = hand.get("lm_np")
                    if lm is None or len(lm) == 0:
                        continue
                    handedness = hand.get("handedness", "Unknown")
                    side_idx = 0 if handedness == "Left" else 1
                    hand_obs[pid][side_idx].append((cam, lm))

        # Determine if ALL cameras have undistorted data (skip undistort in triangulation)
        all_undist = all(cam_undistorted.get(ck, False) for ck in cam_keys)

        # --- Triangulate face landmarks per person ---
        for pid in range(n_people):
            obs_list = face_obs[pid]
            if len(obs_list) >= 2:
                for li in range(N_FACE_LANDMARKS):
                    pt_obs = []
                    for cam, lm in obs_list:
                        if li < len(lm) and lm[li, 3] > 0.1:
                            pt_obs.append((cam, lm[li, :2]))
                    if len(pt_obs) >= 2:
                        pt3d, err, nc = triangulate_point(
                            pt_obs, max_reproj_px=max_reproj_px,
                            already_undistorted=all_undist,
                        )
                        if not np.isnan(pt3d[0]):
                            face_3d[fi, pid, li, :3] = pt3d
                            face_3d[fi, pid, li, 3] = err
                            face_tri_count += 1

            # Aggregate blendshapes (confidence-weighted mean)
            bp = bs_obs[pid]
            if bp:
                total_conf = sum(c for c, _ in bp)
                if total_conf > 0:
                    weighted = np.zeros(N_BLENDSHAPES, dtype=np.float32)
                    for conf, bv in bp:
                        weighted += conf * bv
                    blendshapes[fi, pid, :] = weighted / total_conf
                    bs_count += 1

        # --- Triangulate hand landmarks per person ---
        for pid in range(n_people):
            for side_idx in (0, 1):
                obs_list = hand_obs[pid][side_idx]
                if len(obs_list) >= 2:
                    for li in range(N_HAND_LANDMARKS):
                        pt_obs = []
                        for cam, lm in obs_list:
                            if li < len(lm) and lm[li, 3] > 0.1:
                                pt_obs.append((cam, lm[li, :2]))
                        if len(pt_obs) >= 2:
                            pt3d, err, nc = triangulate_point(
                                pt_obs, max_reproj_px=max_reproj_px,
                                already_undistorted=all_undist,
                            )
                            if not np.isnan(pt3d[0]):
                                hand_3d[fi, pid, side_idx, li, :3] = pt3d
                                hand_3d[fi, pid, side_idx, li, 3] = err
                                hand_tri_count += 1

        # Progress
        if fi % 100 == 0 or fi == n_frames - 1:
            logger.info(
                f"  Frame {fi}/{n_frames} ({fi * 100 // n_frames}%)  "
                f"face_pts={face_tri_count}  hand_pts={hand_tri_count}  "
                f"bs={bs_count}"
            )

    # Compute coverage stats
    face_valid = np.isfinite(face_3d[:, :, :, 0]).sum()
    face_total = n_frames * n_people * N_FACE_LANDMARKS
    hand_valid = np.isfinite(hand_3d[:, :, :, :, 0]).sum()
    hand_total = n_frames * n_people * 2 * N_HAND_LANDMARKS
    bs_valid = np.isfinite(blendshapes[:, :, 0]).sum()

    metadata = {
        "n_frames": n_frames,
        "n_people": n_people,
        "face_landmarks_valid": int(face_valid),
        "face_landmarks_total": int(face_total),
        "face_coverage_pct": round(float(face_valid / max(face_total, 1) * 100), 2),
        "hand_landmarks_valid": int(hand_valid),
        "hand_landmarks_total": int(hand_total),
        "hand_coverage_pct": round(float(hand_valid / max(hand_total, 1) * 100), 2),
        "blendshape_frames_valid": int(bs_valid),
        "blendshape_frames_total": n_frames * n_people,
        "blendshape_coverage_pct": round(
            float(bs_valid / max(n_frames * n_people, 1) * 100), 2,
        ),
        "blendshape_names": BLENDSHAPE_NAMES,
    }

    return {
        "face_3d": face_3d,
        "hand_3d": hand_3d,
        "blendshapes": blendshapes,
        "metadata": metadata,
    }


# ======================================================================
# CLI: detect
# ======================================================================

def _match_openpose_dir(
    video_stem: str,
    openpose_base: Path,
) -> Path | None:
    """Find OpenPose JSON directory matching a video file stem.

    Mapping logic:
      jabra_panacast_20_cam1_vid_video → cam1_json
      jabra_panacast_20_cam2_vid_video → cam2_json
      jabra_panacast_50_vid_video      → p50_json
    """
    if not openpose_base or not openpose_base.is_dir():
        return None

    stem_lower = video_stem.lower()

    # Try direct substring matches
    candidate_dirs = sorted(openpose_base.iterdir())
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        dname = d.name.lower()
        # cam1 ↔ cam1_json, cam2 ↔ cam2_json, etc.
        if "cam1" in stem_lower and "cam1" in dname:
            return d
        if "cam2" in stem_lower and "cam2" in dname:
            return d
        if "cam3" in stem_lower and "cam3" in dname:
            return d
        if "cam4" in stem_lower and "cam4" in dname:
            return d
        if ("cam_5" in stem_lower or "panacast_50" in stem_lower) and "p50" in dname:
            return d

    return None


def cmd_detect(args: argparse.Namespace) -> None:
    """Run 2D face + hand detection on camera videos."""
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openpose_base = Path(args.openpose_dir) if args.openpose_dir else None

    # Load calibration for image undistortion (optional but recommended)
    cam_calibs: dict[str, CameraCalib] | None = None
    if args.calib:
        calib_path = Path(args.calib)
        if calib_path.exists():
            cam_calibs = load_cameras(calib_path)
            logger.info(f"Loaded calibration ({len(cam_calibs)} cameras) for undistortion")
        else:
            logger.warning(f"Calibration file not found: {calib_path}")

    # Find videos
    videos: list[Path] = []
    for ext in ("*.mkv", "*.mp4", "*.avi"):
        videos.extend(sorted(video_dir.glob(ext)))
    if not videos:
        logger.error(f"No videos found in {video_dir}")
        sys.exit(1)

    flip_subs = args.flip_cameras or []

    logger.info("=" * 70)
    logger.info("FACE + HAND 2D DETECTION")
    logger.info("=" * 70)
    logger.info(f"Videos: {len(videos)} in {video_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max frames: {args.max_frames if args.max_frames > 0 else 'all'}")
    if flip_subs:
        logger.info(f"Flip 180°: {flip_subs}")
    if openpose_base:
        logger.info(f"OpenPose dir: {openpose_base}  (body-pose-guided face crops)")
    logger.info("")

    all_stats: list[dict[str, Any]] = []
    for vp in videos:
        label = vp.stem
        cam_json_dir = output_dir / f"{label}_facehand_json"

        do_flip = bool(flip_subs and any(s in label for s in flip_subs))

        # Find matching OpenPose dir for this camera
        op_dir = _match_openpose_dir(label, openpose_base) if openpose_base else None
        if openpose_base:
            if op_dir:
                logger.info(f"  {label} → OpenPose: {op_dir.name}")
            else:
                logger.warning(f"  {label} → no matching OpenPose dir (full-frame fallback)")

        # Find matching calibration for undistortion
        cam_K: np.ndarray | None = None
        cam_dist: np.ndarray | None = None
        if cam_calibs:
            for ck, cal in cam_calibs.items():
                if cal.name in label or label in cal.name:
                    cam_K = cal.K
                    cam_dist = cal.dist
                    logger.info(f"  {label} → Calibration: {ck} (k1={cal.dist[0]:.4f})")
                    break

        stats = detect_faces_and_hands(
            video_path=vp,
            output_dir=cam_json_dir,
            max_frames=args.max_frames,
            num_faces=args.num_faces,
            num_hands=args.num_hands,
            min_face_det_conf=args.min_face_conf,
            min_hand_det_conf=args.min_hand_conf,
            flip_180=do_flip,
            openpose_dir=op_dir,
            camera_K=cam_K,
            camera_dist=cam_dist,
        )
        all_stats.append(stats)

    # Summary table
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(f"{'Camera':<45} {'Mode':>5} {'Face%':>7} {'Hand%':>7} {'Frames':>8}")
    print("-" * 75)
    for s in all_stats:
        if "error" in s:
            print(f"{s.get('video', '?'):<45} {'':>5} {'ERR':>7} {'ERR':>7}")
        else:
            print(
                f"{s['video']:<45} "
                f"{s.get('face_mode', '?'):>5} "
                f"{s['face_detect_pct']:>6.1f}% "
                f"{s['hand_detect_pct']:>6.1f}% "
                f"{s['processed_frames']:>8}"
            )
    print("=" * 70)


# ======================================================================
# CLI: reconstruct
# ======================================================================

def cmd_reconstruct(args: argparse.Namespace) -> None:
    """3D reconstruction from multi-camera face/hand detections."""
    calib_path = Path(args.calib)
    detect_dir = Path(args.detect_dir)
    output_path = Path(args.output)

    logger.info("=" * 70)
    logger.info("FACE + HAND 3D RECONSTRUCTION")
    logger.info("=" * 70)

    # Load cameras
    cameras = load_cameras(calib_path)
    logger.info(f"Loaded {len(cameras)} cameras from calibration")

    # Auto-map cameras → detection directories
    detect_subdirs = sorted(detect_dir.glob("*_facehand_json"))
    cam_detect_map: dict[str, Path] = {}
    for ck, cam in cameras.items():
        for sd in detect_subdirs:
            dir_label = sd.name.replace("_facehand_json", "")
            if cam.name in dir_label or dir_label in cam.name:
                cam_detect_map[ck] = sd
                logger.info(f"  {ck} ({cam.name}) → {sd.name}")
                break
        else:
            logger.warning(f"  {ck} ({cam.name}): no detection dir found")

    if not cam_detect_map:
        logger.error("No camera → detection mappings found")
        sys.exit(1)

    # Load body skeleton (optional)
    body_skel = None
    if args.body_skeleton:
        bp = Path(args.body_skeleton)
        if bp.exists():
            body_skel = np.load(bp)
            logger.info(f"Body skeleton: {bp.name}, shape={body_skel.shape}")
        else:
            logger.warning(f"Body skeleton not found: {bp}")

    n_people = args.n_people or (body_skel.shape[1] if body_skel is not None else 1)

    # Reconstruct
    results = reconstruct_3d(
        cameras=cameras,
        detect_dirs=cam_detect_map,
        body_skeleton=body_skel,
        n_people=n_people,
        max_reproj_px=args.max_reproj_px,
    )

    if not results:
        logger.error("Reconstruction failed")
        sys.exit(1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        face_3d=results["face_3d"],
        hand_3d=results["hand_3d"],
        blendshapes=results["blendshapes"],
    )

    # Save metadata JSON alongside
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(results["metadata"], f, indent=2)

    meta = results["metadata"]
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Face 3D:      shape=({meta['n_frames']}, {meta['n_people']}, 468, 4)")
    logger.info(f"                coverage={meta['face_coverage_pct']}%  "
                f"({meta['face_landmarks_valid']}/{meta['face_landmarks_total']} landmarks)")
    logger.info(f"  Hand 3D:      shape=({meta['n_frames']}, {meta['n_people']}, 2, 21, 4)")
    logger.info(f"                coverage={meta['hand_coverage_pct']}%  "
                f"({meta['hand_landmarks_valid']}/{meta['hand_landmarks_total']} landmarks)")
    logger.info(f"  Blendshapes:  shape=({meta['n_frames']}, {meta['n_people']}, 52)")
    logger.info(f"                coverage={meta['blendshape_coverage_pct']}%  "
                f"({meta['blendshape_frames_valid']}/{meta['blendshape_frames_total']} frames)")
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Metadata: {meta_path}")
    logger.info("=" * 70)


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-camera face and hand landmark pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- detect ---
    p_det = sub.add_parser(
        "detect",
        help="Run 2D face + hand detection on camera videos",
    )
    p_det.add_argument(
        "--video-dir", required=True,
        help="Directory containing camera video files (.mkv/.mp4)",
    )
    p_det.add_argument(
        "--output-dir", required=True,
        help="Output directory for per-camera detection JSON",
    )
    p_det.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process per video (0=all)",
    )
    p_det.add_argument(
        "--num-faces", type=int, default=4,
        help="Max faces to detect per frame (default: 4)",
    )
    p_det.add_argument(
        "--num-hands", type=int, default=8,
        help="Max hands per frame (default: 8 = 4 people × 2)",
    )
    p_det.add_argument(
        "--min-face-conf", type=float, default=0.4,
        help="Min face detection confidence (default: 0.4)",
    )
    p_det.add_argument(
        "--min-hand-conf", type=float, default=0.4,
        help="Min hand detection confidence (default: 0.4)",
    )
    p_det.add_argument(
        "--flip-cameras", nargs="*", default=[],
        help="Flip 180° for cameras matching these substrings "
             "(e.g. cam1 cam2 cam3 cam4 for upside-down P20s)",
    )
    p_det.add_argument(
        "--openpose-dir",
        help="OpenPose output directory (body-pose-guided face crops). "
             "Contains per-camera subdirs like cam1_json/, cam2_json/, p50_json/.",
    )
    p_det.add_argument(
        "--calib",
        help="Calibration .toml file for image undistortion. "
             "Required for accurate landmark coordinates on wide-angle lenses.",
    )
    p_det.set_defaults(func=cmd_detect)

    # --- reconstruct ---
    p_rec = sub.add_parser(
        "reconstruct",
        help="3D reconstruction from multi-camera face/hand detections",
    )
    p_rec.add_argument(
        "--calib", required=True,
        help="Calibration .toml file",
    )
    p_rec.add_argument(
        "--detect-dir", required=True,
        help="Directory containing *_facehand_json/ subdirs",
    )
    p_rec.add_argument(
        "--body-skeleton",
        help="Body skeleton .npy (for person anchoring)",
    )
    p_rec.add_argument(
        "--n-people", type=int, default=None,
        help="Number of people (default: from body skeleton)",
    )
    p_rec.add_argument(
        "--max-reproj-px", type=float, default=40.0,
        help="Max reprojection error for triangulation (default: 40px)",
    )
    p_rec.add_argument(
        "--output", required=True,
        help="Output .npz file",
    )
    p_rec.set_defaults(func=cmd_reconstruct)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
