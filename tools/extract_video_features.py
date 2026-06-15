#!/usr/bin/env python3
"""Extract compact video-surrogate features from AffectAI camera recordings.

The extractor decodes each camera video once and writes derived features that
downstream calibration, 3D gaze, 3D skeleton, and gesture pipelines can reuse
without repeatedly scanning raw video.

Outputs are session-derived artifacts only:
- per-camera frame sync JSONL
- sparse ArUco marker detections JSONL
- optional dense body, face, and hand arrays in compressed NPZ files
- feature_manifest.json with provenance and extraction settings

Examples
--------
    python tools/extract_video_features.py \\
        --videos-dir sessions/Final/merged/sub-01/ses-.../video \\
        --output-dir sessions/Final/merged/sub-01/ses-.../features_video \\
        --frame-log-dir sessions/Final/merged/sub-01/ses-.../sourcedata/av/frame_logs \\
        --body --hands --faces \\
        --aruco-dicts DICT_4X4_50,DICT_4X4_250

Notes
-----
Heavy imports (OpenCV, MediaPipe, MMPose) are loaded only when extraction runs, so
``python tools/extract_video_features.py --help`` stays fast.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\. Please use message_factory\.GetMessageClass\(\) instead\..*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

VIDEO_EXTS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_ARUCO_DICTS = ("DICT_4X4_50", "DICT_4X4_250")
_TASK_RE = re.compile(r"_task-(T[0-9A-Za-z]+)")
_RUN_RE = re.compile(r"_run-([0-9]+)")
_ACQ_RE = re.compile(r"_acq-([^_]+)")

N_BODY_LANDMARKS = 33
N_COCO_LANDMARKS = 17
N_WHOLEBODY_LANDMARKS = 133
N_FACE_LANDMARKS = 478
N_HAND_LANDMARKS = 21

BODY_BACKBONES = ("mediapipe-pose", "rtmpose-mmpose", "none")
FACE_BACKBONES = ("mediapipe-face", "none")
HAND_BACKBONES = ("mediapipe-hands", "none")
MARKER_BACKBONES = ("opencv-aruco", "none")


@dataclass(frozen=True)
class ExtractorConfig:
    """Serializable extraction settings written to the manifest."""

    max_people: int
    max_faces: int
    max_hands: int
    body: bool
    faces: bool
    hands: bool
    markers: bool
    body_backbone: str
    face_backbone: str
    hand_backbone: str
    marker_backbone: str
    body_stride: int
    face_stride: int
    hand_stride: int
    marker_stride: int
    max_frames: int
    resize_width: int | None
    aruco_dicts: tuple[str, ...]
    float_dtype: str


@dataclass(frozen=True)
class MarkerInstance:
    """Configured semantic identity for a marker ID."""

    marker_id: int
    role: str
    instance_id: str
    dictionary: str | None = None


@dataclass(frozen=True)
class ClipTimingContext:
    """Timing metadata for a task-split BIDS video clip."""

    task: str
    run: str
    acq: str | None
    clip_start_unix_time_s: float | None
    clip_start_lsl: float | None
    wall_minus_lsl_offset: float | None
    source: str


def _json_default(value: Any) -> Any:
    """Convert numpy scalars/arrays for JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _camera_label_from_path(path: Path) -> str:
    """Create a stable, filesystem-safe camera label from a video filename."""
    stem = path.stem.lower()
    for token in ("_vid_video", "_video", "_vid"):
        stem = stem.replace(token, "")
    stem = stem.replace("jabra_panacast_20_", "panacast-20-")
    stem = stem.replace("jabra_panacast_50_", "panacast-50-")
    safe = []
    for char in stem:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char in {" ", "."}:
            safe.append("-")
    label = "".join(safe).strip("-_")
    return label or path.stem


def _discover_videos(videos_dir: Path) -> list[Path]:
    """Return sorted video files directly under videos_dir.
    
    On Windows, uses UNC long-path notation (\\?\) to support paths >260 chars.
    """
    unc_path = _to_windows_long_path(videos_dir)
    return sorted(p for p in unc_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _to_windows_long_path(p: Path) -> Path:
    """Convert a Path to Windows UNC long-path format if needed.
    
    On Windows, converts absolute paths to \\?\<abs_path> format to bypass
    the MAX_PATH (260 char) limit. On other platforms, returns unchanged.
    """
    import sys
    
    if sys.platform != "win32":
        return p
    
    abs_path = p.resolve()
    path_str = str(abs_path)
    
    # Already in UNC format
    if path_str.startswith("\\\\?\\"):
        return p
    
    # UNC network path—convert to \\?\UNC\<server>\<share>
    if path_str.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{path_str[2:]}")
    
    # Regular path—convert to \\?\<drive>:\<path>
    return Path(f"\\\\?\\{path_str}")


def _normalise_aruco_dict_names(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize comma/list ArUco dictionary names and remove duplicates."""
    if raw is None:
        names = list(DEFAULT_ARUCO_DICTS)
    elif isinstance(raw, str):
        names = [part.strip() for part in raw.split(",")]
    else:
        names = [str(part).strip() for part in raw]

    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name:
            continue
        if name not in seen:
            out.append(name)
            seen.add(name)
    return tuple(out)


def _body_landmark_count(backbone: str, rtmpose_model: str) -> int:
    """Return expected body landmark count for the configured body backbone."""
    if backbone == "mediapipe-pose":
        return N_BODY_LANDMARKS
    if backbone == "rtmpose-mmpose":
        model_name = rtmpose_model.lower()
        if "wholebody" in model_name or "rtmw" in model_name:
            return N_WHOLEBODY_LANDMARKS
        return N_COCO_LANDMARKS
    return 0


def _enabled_backbone(enabled: bool, backbone: str) -> bool:
    """True when a feature family is enabled and has an active backbone."""
    return enabled and backbone != "none"


def _sample_count(total_frames: int, stride: int, max_frames: int) -> int:
    """Return number of sampled frames for an extraction stream."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    usable = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    if usable <= 0:
        return 0
    return (usable + stride - 1) // stride


def _load_frame_log(frame_log_path: Path | None) -> dict[int, dict[str, Any]]:
    """Load frame-log records keyed by frame index."""
    if frame_log_path is None or not frame_log_path.exists():
        return {}

    records: dict[int, dict[str, Any]] = {}
    with frame_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_idx = payload.get("frame_idx", payload.get("frame"))
            if frame_idx is None:
                frame_idx = len(records)
            try:
                records[int(frame_idx)] = payload
            except (TypeError, ValueError):
                continue
    return records


def _load_structured_config(path: Path | None) -> dict[str, Any]:
    """Load JSON/YAML config, returning an empty dict when no path is provided."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _marker_instance_lookup(
    marker_config: dict[str, Any],
    default_dictionary: str | None = None,
) -> dict[tuple[str | None, int], list[MarkerInstance]]:
    """Build marker-ID lookup while preserving duplicate/ambiguous instances."""
    world = marker_config.get("world", {}) if isinstance(marker_config, dict) else {}
    dictionary = (
        marker_config.get("aruco_dictionary")
        or world.get("aruco_dictionary")
        or default_dictionary
    )

    instances: list[MarkerInstance] = []

    marker_defs = marker_config.get("table_markers")
    if marker_defs is None:
        marker_defs = world.get("marker_map", [])
    for index, marker in enumerate(marker_defs or []):
        marker_id = marker.get("id", marker.get("marker_id"))
        if marker_id is None:
            continue
        name = marker.get("name") or marker.get("label") or f"desk-{marker_id}-{index}"
        instances.append(
            MarkerInstance(
                marker_id=int(marker_id),
                role=str(marker.get("role", "desk")),
                instance_id=str(name),
                dictionary=marker.get("dictionary", dictionary),
            )
        )

    for glasses in marker_config.get("glasses", []) or []:
        participant = str(glasses.get("participant", glasses.get("id", "glasses")))
        for side in ("left", "right"):
            marker_id = glasses.get(f"{side}_marker_id")
            if marker_id is None:
                continue
            instances.append(
                MarkerInstance(
                    marker_id=int(marker_id),
                    role="glasses",
                    instance_id=f"{participant}:{side}",
                    dictionary=glasses.get("aruco_dictionary", dictionary),
                )
            )

    fixed_board = marker_config.get("fixed_charuco_board") or {}
    board_marker_ids = fixed_board.get("marker_ids", [])
    for marker_id in board_marker_ids:
        instances.append(
            MarkerInstance(
                marker_id=int(marker_id),
                role="charuco_board",
                instance_id=f"charuco:{marker_id}",
                dictionary=fixed_board.get("aruco_dictionary", "DICT_4X4_250"),
            )
        )

    lookup: dict[tuple[str | None, int], list[MarkerInstance]] = {}
    for instance in instances:
        keys = [(instance.dictionary, instance.marker_id)]
        if instance.dictionary is not None:
            keys.append((None, instance.marker_id))
        for key in keys:
            lookup.setdefault(key, []).append(instance)
    return lookup


def _find_frame_log(frame_log_dir: Path | None, video_path: Path, camera_label: str) -> Path | None:
    """Find the most likely frame-log JSONL for a camera video."""
    if frame_log_dir is None or not frame_log_dir.exists():
        return None

    candidates: list[Path] = []
    label_tokens = {
        camera_label.lower(),
        video_path.stem.lower(),
        camera_label.lower().replace("-", "_"),
        camera_label.lower().replace("_", "-"),
    }
    for path in sorted(frame_log_dir.rglob("*.jsonl")):
        low_name = path.name.lower()
        if any(token and token in low_name for token in label_tokens):
            candidates.append(path)
    return candidates[0] if candidates else None


def _parse_video_entities(video_path: Path) -> dict[str, str | None]:
    """Extract common BIDS entities from a split video filename."""
    name = video_path.name
    task_match = _TASK_RE.search(name)
    run_match = _RUN_RE.search(name)
    acq_match = _ACQ_RE.search(name)
    return {
        "task": None if task_match is None else task_match.group(1),
        "run": None if run_match is None else run_match.group(1),
        "acq": None if acq_match is None else acq_match.group(1),
    }


def _find_session_dir(video_path: Path) -> Path | None:
    """Return the enclosing BIDS session dir for a session/video/<file> path."""
    if video_path.parent.name != "video":
        return None
    session_dir = video_path.parent.parent
    annot_dir = session_dir / "annot"
    if annot_dir.is_dir():
        return session_dir
    return None


def _find_task_windows_tsv(session_dir: Path | None) -> Path | None:
    """Return the authoritative task-run windows TSV inside annot/."""
    if session_dir is None:
        return None
    annot_dir = session_dir / "annot"
    if not annot_dir.is_dir():
        return None
    preferred = sorted(annot_dir.glob("*_task-T0T1T2T3T4_task_run_windows.tsv"))
    if preferred:
        return preferred[0]
    candidates = sorted(annot_dir.glob("*_task_run_windows.tsv"))
    return candidates[0] if candidates else None


def _parse_optional_float(raw: str | None) -> float | None:
    """Convert a TSV cell to float when possible."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=128)
def _load_task_windows_lookup(task_windows_tsv: str) -> dict[tuple[str, str], dict[str, float | None]]:
    """Load task-run timing rows keyed by (task, run)."""
    lookup: dict[tuple[str, str], dict[str, float | None]] = {}
    path = Path(task_windows_tsv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            task = (row.get("task") or "").strip()
            run = str((row.get("run") or "").strip()).zfill(2)
            if not task or not run:
                continue
            lookup[(task, run)] = {
                "start_wall_clock": _parse_optional_float(row.get("start_wall_clock")),
                "start_lsl": _parse_optional_float(row.get("start_lsl")),
                "wall_minus_lsl_offset": _parse_optional_float(row.get("wall_minus_lsl_offset")),
            }
    return lookup


def _resolve_clip_timing_context(video_path: Path) -> ClipTimingContext | None:
    """Return split-clip timing metadata from annot/*_task_run_windows.tsv when present."""
    session_dir = _find_session_dir(video_path)
    task_windows_tsv = _find_task_windows_tsv(session_dir)
    if task_windows_tsv is None:
        return None

    entities = _parse_video_entities(video_path)
    task = entities.get("task")
    run = entities.get("run")
    if task is None or run is None:
        return None

    row = _load_task_windows_lookup(str(task_windows_tsv)).get((task, str(run).zfill(2)))
    if row is None:
        return None

    return ClipTimingContext(
        task=task,
        run=str(run).zfill(2),
        acq=entities.get("acq"),
        clip_start_unix_time_s=row.get("start_wall_clock"),
        clip_start_lsl=row.get("start_lsl"),
        wall_minus_lsl_offset=row.get("wall_minus_lsl_offset"),
        source="task_run_windows",
    )


def _sha256_prefix(
    path: Path,
    chunk_size: int = 1024 * 1024,
    max_bytes: int = 64 * 1024 * 1024,
) -> str:
    """Hash the first max_bytes of a source video for provenance without full-file IO."""
    hasher = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


def _import_cv2() -> Any:
    """Import OpenCV with a clearer compatibility error for broken envs."""
    try:
        import cv2
    except (AttributeError, ImportError) as exc:
        message = str(exc)
        if "_ARRAY_API" in message or "numpy.core.multiarray failed to import" in message:
            raise ImportError(
                "OpenCV could not import because it is incompatible with the installed NumPy. "
                "This usually means cv2 was built against NumPy 1.x but the environment has "
                "NumPy 2.x. In the target environment, reinstall a compatible combination such "
                "as `conda install \"numpy<2\"` and then verify `python -c \"import cv2, numpy\"`."
            ) from exc
        raise
    return cv2


def _ensure_mediapipe_model(task_name: str, model_file: str, base_url: str) -> Path:
    """Download a MediaPipe task model if not already cached."""
    cache_dir = Path(__file__).resolve().parent / ".mediapipe_models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / model_file
    if model_path.exists():
        return model_path
    url = f"{base_url}/{model_file}"
    logger.info("Downloading %s model: %s", task_name, url)
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _pose_model_path(model_complexity: int) -> Path:
    model_map = {
        0: ("pose_landmarker_lite", "pose_landmarker_lite.task"),
        1: ("pose_landmarker_full", "pose_landmarker_full.task"),
        2: ("pose_landmarker_heavy", "pose_landmarker_heavy.task"),
    }
    model_name, model_file = model_map.get(model_complexity, model_map[1])
    base_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        f"pose_landmarker/{model_name}/float16/latest"
    )
    return _ensure_mediapipe_model("PoseLandmarker", model_file, base_url)


def _face_model_path() -> Path:
    base_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest"
    )
    return _ensure_mediapipe_model(
        "FaceLandmarker",
        "face_landmarker.task",
        base_url,
    )


def _hand_model_path() -> Path:
    base_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest"
    )
    return _ensure_mediapipe_model(
        "HandLandmarker",
        "hand_landmarker.task",
        base_url,
    )


class ArrayFeatureWriter:
    """Bounded-memory writer for dense per-frame landmark arrays."""

    def __init__(
        self,
        output_npz: Path,
        shape: tuple[int, ...],
        dtype: np.dtype,
        fill_value: float = math.nan,
    ) -> None:
        self.output_npz = output_npz
        self.output_npz.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_dir_obj = tempfile.TemporaryDirectory(
            prefix="affectai_features_", ignore_cleanup_errors=True
        )
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.data_path = self.tmp_dir / "data.npy"
        self.frame_path = self.tmp_dir / "frame_idx.npy"
        self.time_path = self.tmp_dir / "time_s.npy"
        self.data = np.lib.format.open_memmap(
            self.data_path,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        self.data[...] = fill_value
        self.frame_idx = np.lib.format.open_memmap(
            self.frame_path, mode="w+", dtype=np.int64, shape=(shape[0],)
        )
        self.frame_idx[...] = -1
        self.time_s = np.lib.format.open_memmap(
            self.time_path, mode="w+", dtype=np.float64, shape=(shape[0],)
        )
        self.time_s[...] = np.nan
        self.count = 0

    def write(self, frame_idx: int, time_s: float, values: np.ndarray) -> None:
        """Write one sampled frame."""
        if self.count >= self.data.shape[0]:
            return
        self.frame_idx[self.count] = frame_idx
        self.time_s[self.count] = time_s
        self.data[self.count, ...] = values.astype(self.data.dtype, copy=False)
        self.count += 1

    def close(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Flush and compress the arrays to NPZ."""
        self.data.flush()
        self.frame_idx.flush()
        self.time_s.flush()
        # Release writable memmap handles before reopening files for read/compress on Windows.
        del self.data
        del self.frame_idx
        del self.time_s
        data = np.load(self.data_path, mmap_mode="r")
        frames = np.load(self.frame_path, mmap_mode="r")
        times = np.load(self.time_path, mmap_mode="r")
        trimmed = slice(0, self.count)
        out_shape = list(data[trimmed].shape)
        out_dtype = str(data.dtype)
        np.savez_compressed(
            self.output_npz,
            data=data[trimmed],
            frame_idx=frames[trimmed],
            time_s=times[trimmed],
            metadata=json.dumps(metadata, default=_json_default),
        )
        del data
        del frames
        del times
        try:
            self.tmp_dir_obj.cleanup()
        except (PermissionError, FileNotFoundError, NotADirectoryError) as exc:
            logger.warning("Temporary feature cleanup warning (%s): %s", self.tmp_dir, exc)
        return {
            "path": str(self.output_npz),
            "sampled_frames": self.count,
            "shape": out_shape,
            "dtype": out_dtype,
        }


def _make_aruco_detectors(dict_names: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Create OpenCV ArUco detectors for each configured dictionary."""
    cv2 = _import_cv2()

    detectors: list[tuple[str, Any]] = []
    for name in dict_names:
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"Unknown ArUco dictionary: {name}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detectors.append((name, cv2.aruco.ArucoDetector(dictionary, params)))
    return detectors


def _landmarks_to_array(
    landmarks: list[Any],
    max_items: int,
    n_landmarks: int,
    dims: int,
) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to a fixed-size array."""
    arr = np.full((max_items, n_landmarks, dims), np.nan, dtype=np.float32)
    for item_idx, item_landmarks in enumerate(landmarks[:max_items]):
        for lm_idx, lm in enumerate(item_landmarks[:n_landmarks]):
            arr[item_idx, lm_idx, 0] = float(lm.x)
            arr[item_idx, lm_idx, 1] = float(lm.y)
            if dims >= 3:
                arr[item_idx, lm_idx, 2] = float(getattr(lm, "z", 0.0))
            if dims >= 4:
                arr[item_idx, lm_idx, 3] = float(
                    getattr(lm, "visibility", getattr(lm, "presence", 1.0))
                )
    return arr


def _mmpose_result_to_array(
    result: dict[str, Any],
    max_people: int,
    n_landmarks: int,
    scale_to_source: float,
) -> np.ndarray:
    """Convert an MMPoseInferencer result to fixed source-pixel keypoints."""
    arr = np.full((max_people, n_landmarks, 4), np.nan, dtype=np.float32)
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        people = predictions[0]
    elif isinstance(predictions, list):
        people = predictions
    else:
        people = []

    for person_idx, person in enumerate(people[:max_people]):
        keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
        scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            continue
        n = min(n_landmarks, keypoints.shape[0])
        arr[person_idx, :n, 0:2] = keypoints[:n, :2] * float(scale_to_source)
        arr[person_idx, :n, 2] = np.nan
        if scores.ndim == 1 and scores.shape[0] >= n:
            arr[person_idx, :n, 3] = scores[:n]
        else:
            arr[person_idx, :n, 3] = 1.0
    return arr


def _create_mmpose_inferencer(model: str, device: str) -> Any:
    """Create an MMPose inferencer, raising a clear optional-dependency error."""
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise ImportError(
            "RTMPose requires optional MMPose dependencies. Install an environment "
            "with torch, mmcv, mmdet, and mmpose, then rerun with "
            "`--body-backbone rtmpose-mmpose`."
        ) from exc
    try:
        return MMPoseInferencer(pose2d=model, device=device)
    except TypeError:
        return MMPoseInferencer(model, device=device)


def _resize_frame(frame: np.ndarray, resize_width: int | None) -> tuple[np.ndarray, float]:
    """Resize frame for inference and return scale from output pixels to source pixels."""
    if resize_width is None or resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame, 1.0
    cv2 = _import_cv2()

    scale = frame.shape[1] / float(resize_width)
    height = max(1, int(round(frame.shape[0] / scale)))
    return cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA), scale


def _frame_time_s(
    cap: Any,
    frame_idx: int,
    frame_log: dict[int, dict[str, Any]],
    fps: float,
) -> float:
    """Return best available frame time in seconds relative to video start."""
    log_row = frame_log.get(frame_idx, {})
    for key in ("pts_time", "time_s"):
        value = log_row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    pos_msec = cap.get(0)  # cv2.CAP_PROP_POS_MSEC, kept numeric to avoid top-level cv2 import
    if pos_msec and pos_msec > 0:
        return float(pos_msec) / 1000.0
    return frame_idx / fps if fps > 0 else float(frame_idx)


def _write_frame_sync(
    handle: Any,
    camera_label: str,
    frame_idx: int,
    time_s: float,
    frame_log: dict[int, dict[str, Any]],
    timing_context: ClipTimingContext | None = None,
) -> None:
    """Write one compact frame sync record."""
    log_row = frame_log.get(frame_idx, {})
    unix_time_s = log_row.get("unix_time_s", log_row.get("unix_time"))
    lsl_time = log_row.get("lsl_time", log_row.get("server_received_lsl"))
    source = "frame_log" if log_row else "video_pts"
    if not log_row and timing_context is not None:
        if timing_context.clip_start_unix_time_s is not None:
            unix_time_s = timing_context.clip_start_unix_time_s + time_s
        if timing_context.clip_start_lsl is not None:
            lsl_time = timing_context.clip_start_lsl + time_s
        source = "task_run_windows+video_pts"
    record = {
        "camera_id": camera_label,
        "frame_idx": frame_idx,
        "pts_time": time_s,
        "task": None if timing_context is None else timing_context.task,
        "run": None if timing_context is None else timing_context.run,
        "acq": None if timing_context is None else timing_context.acq,
        "unix_time_s": unix_time_s,
        "wall_time_s": unix_time_s,
        "lsl_time": lsl_time,
        "source": source,
    }
    handle.write(json.dumps(record, separators=(",", ":"), default=_json_default) + "\n")


def _write_marker_detections(
    handle: Any,
    frame: np.ndarray,
    camera_label: str,
    frame_idx: int,
    time_s: float,
    detectors: list[tuple[str, Any]],
    scale_to_source: float,
    marker_lookup: dict[tuple[str | None, int], list[MarkerInstance]],
) -> int:
    """Detect and write sparse ArUco marker detections for one frame."""
    count = 0
    for dict_name, detector in detectors:
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        for det_idx, marker_id in enumerate(ids.flatten()):
            int_marker_id = int(marker_id)
            instances = marker_lookup.get((dict_name, int_marker_id), [])
            if not instances:
                instances = marker_lookup.get((None, int_marker_id), [])
            corners_px = corners[det_idx].reshape(4, 2).astype(np.float32) * scale_to_source
            record = {
                "camera_id": camera_label,
                "frame_idx": frame_idx,
                "time_s": time_s,
                "dictionary": dict_name,
                "marker_id": int_marker_id,
                "marker_instances": [asdict(instance) for instance in instances],
                "ambiguous_marker_id": len(instances) > 1,
                "corners_px": np.round(corners_px, 3).tolist(),
            }
            handle.write(json.dumps(record, separators=(",", ":"), default=_json_default) + "\n")
            count += 1
    return count


def process_video(
    video_path: Path,
    output_dir: Path,
    config: ExtractorConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Extract configured features from one camera video."""
    cv2 = _import_cv2()

    camera_label = _camera_label_from_path(video_path)
    camera_dir = output_dir / camera_label
    camera_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = int(args.max_frames) if args.max_frames > 0 else 0

    frame_log_path = _find_frame_log(args.frame_log_dir, video_path, camera_label)
    frame_log = _load_frame_log(frame_log_path)
    timing_context = _resolve_clip_timing_context(video_path)
    marker_config = _load_structured_config(args.marker_config)
    marker_lookup = _marker_instance_lookup(
        marker_config,
        default_dictionary=config.aruco_dicts[0] if config.aruco_dicts else None,
    )
    body_enabled = _enabled_backbone(config.body, config.body_backbone)
    face_enabled = _enabled_backbone(config.faces, config.face_backbone)
    hand_enabled = _enabled_backbone(config.hands, config.hand_backbone)
    marker_enabled = _enabled_backbone(config.markers, config.marker_backbone)
    body_landmarks = _body_landmark_count(config.body_backbone, args.rtmpose_model)

    body_writer: ArrayFeatureWriter | None = None
    face_writer: ArrayFeatureWriter | None = None
    hand_writer: ArrayFeatureWriter | None = None
    dtype = np.dtype(config.float_dtype)

    if body_enabled:
        body_shape = (
            _sample_count(total_frames, config.body_stride, config.max_frames),
            config.max_people,
            body_landmarks,
            4,
        )
        body_writer = ArrayFeatureWriter(
            camera_dir / "body_2d.npz",
            body_shape,
            dtype,
        )
    if face_enabled:
        face_shape = (
            _sample_count(total_frames, config.face_stride, config.max_frames),
            config.max_faces,
            N_FACE_LANDMARKS,
            3,
        )
        face_writer = ArrayFeatureWriter(
            camera_dir / "face_2d.npz",
            face_shape,
            dtype,
        )
    if hand_enabled:
        hand_shape = (
            _sample_count(total_frames, config.hand_stride, config.max_frames),
            config.max_hands,
            N_HAND_LANDMARKS,
            3,
        )
        hand_writer = ArrayFeatureWriter(
            camera_dir / "hands_2d.npz",
            hand_shape,
            dtype,
        )

    aruco_detectors = _make_aruco_detectors(config.aruco_dicts) if marker_enabled else []

    pose_lm = None
    face_lm = None
    hand_lm = None
    mmpose_inferencer = None
    use_mediapipe = (
        (body_enabled and config.body_backbone == "mediapipe-pose")
        or (face_enabled and config.face_backbone == "mediapipe-face")
        or (hand_enabled and config.hand_backbone == "mediapipe-hands")
    )
    if use_mediapipe:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            HandLandmarker,
            HandLandmarkerOptions,
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        if body_enabled and config.body_backbone == "mediapipe-pose":
            pose_options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(_pose_model_path(args.pose_model_complexity))
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=config.max_people,
                min_pose_detection_confidence=args.min_body_confidence,
                min_pose_presence_confidence=args.min_body_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            pose_lm = PoseLandmarker.create_from_options(pose_options)
        if face_enabled and config.face_backbone == "mediapipe-face":
            face_lm = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(_face_model_path())),
                    running_mode=RunningMode.VIDEO,
                    num_faces=config.max_faces,
                    min_face_detection_confidence=args.min_face_confidence,
                    min_face_presence_confidence=args.min_face_confidence,
                    min_tracking_confidence=args.min_tracking_confidence,
                )
            )
        if hand_enabled and config.hand_backbone == "mediapipe-hands":
            hand_lm = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(_hand_model_path())),
                    running_mode=RunningMode.VIDEO,
                    num_hands=config.max_hands,
                    min_hand_detection_confidence=args.min_hand_confidence,
                    min_hand_presence_confidence=args.min_hand_confidence,
                    min_tracking_confidence=args.min_tracking_confidence,
                )
            )
    else:
        mp = None
    if body_enabled and config.body_backbone == "rtmpose-mmpose":
        mmpose_inferencer = _create_mmpose_inferencer(args.rtmpose_model, args.device)

    frame_sync_path = camera_dir / "frame_sync.jsonl"
    marker_path = camera_dir / "marker_detections_2d.jsonl"
    marker_handle = marker_path.open("w", encoding="utf-8") if marker_enabled else None

    processed = 0
    marker_count = 0
    body_frames = 0
    face_frames = 0
    hand_frames = 0
    start_wall = time.time()

    with frame_sync_path.open("w", encoding="utf-8") as sync_handle:
        while True:
            if config.max_frames > 0 and processed >= config.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx = processed
            time_s = _frame_time_s(cap, frame_idx, frame_log, fps)
            _write_frame_sync(
                sync_handle,
                camera_label,
                frame_idx,
                time_s,
                frame_log,
                timing_context=timing_context,
            )

            inference_frame, scale_to_source = _resize_frame(frame, config.resize_width)

            run_marker_frame = (
                marker_enabled
                and frame_idx % config.marker_stride == 0
                and marker_handle is not None
            )
            if run_marker_frame:
                marker_count += _write_marker_detections(
                    marker_handle,
                    inference_frame,
                    camera_label,
                    frame_idx,
                    time_s,
                    aruco_detectors,
                    scale_to_source,
                    marker_lookup,
                )

            run_body_frame = (
                body_enabled
                and body_writer is not None
                and frame_idx % config.body_stride == 0
            )
            if (
                run_body_frame
                and config.body_backbone == "rtmpose-mmpose"
                and mmpose_inferencer is not None
            ):
                mmpose_result = next(mmpose_inferencer(inference_frame, return_vis=False))
                arr = _mmpose_result_to_array(
                    mmpose_result,
                    config.max_people,
                    body_landmarks,
                    scale_to_source,
                )
                body_writer.write(frame_idx, time_s, arr)
                body_frames += int(np.any(np.isfinite(arr[..., 3])))

            if use_mediapipe and mp is not None:
                rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(round(time_s * 1000.0))

                if run_body_frame and config.body_backbone == "mediapipe-pose":
                    result = pose_lm.detect_for_video(mp_image, timestamp_ms)
                    arr = _landmarks_to_array(
                        result.pose_landmarks,
                        config.max_people,
                        body_landmarks,
                        4,
                    )
                    body_writer.write(frame_idx, time_s, arr)
                    body_frames += int(len(result.pose_landmarks) > 0)

                run_face_frame = (
                    face_enabled
                    and face_writer is not None
                    and frame_idx % config.face_stride == 0
                )
                if run_face_frame and config.face_backbone == "mediapipe-face":
                    result = face_lm.detect_for_video(mp_image, timestamp_ms)
                    arr = _landmarks_to_array(
                        result.face_landmarks,
                        config.max_faces,
                        N_FACE_LANDMARKS,
                        3,
                    )
                    face_writer.write(frame_idx, time_s, arr)
                    face_frames += int(len(result.face_landmarks) > 0)

                run_hand_frame = (
                    hand_enabled
                    and hand_writer is not None
                    and frame_idx % config.hand_stride == 0
                )
                if run_hand_frame and config.hand_backbone == "mediapipe-hands":
                    result = hand_lm.detect_for_video(mp_image, timestamp_ms)
                    arr = _landmarks_to_array(
                        result.hand_landmarks,
                        config.max_hands,
                        N_HAND_LANDMARKS,
                        3,
                    )
                    hand_writer.write(frame_idx, time_s, arr)
                    hand_frames += int(len(result.hand_landmarks) > 0)

            processed += 1
            if args.progress_interval > 0 and processed % args.progress_interval == 0:
                logger.info("%s: processed %d frames", camera_label, processed)

    cap.release()
    if marker_handle is not None:
        marker_handle.close()
    if pose_lm is not None:
        pose_lm.close()
    if face_lm is not None:
        face_lm.close()
    if hand_lm is not None:
        hand_lm.close()

    common_metadata = {
        "camera_id": camera_label,
        "source_video": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames_reported": total_frames,
        "coordinate_space": "normalized_image",
        "dtype": config.float_dtype,
        "task": None if timing_context is None else timing_context.task,
        "run": None if timing_context is None else timing_context.run,
        "acq": None if timing_context is None else timing_context.acq,
        "timing_source": "frame_log" if frame_log_path else (
            "video_pts" if timing_context is None else timing_context.source
        ),
        "clip_start_unix_time_s": (
            None if timing_context is None else timing_context.clip_start_unix_time_s
        ),
        "clip_start_lsl": None if timing_context is None else timing_context.clip_start_lsl,
        "wall_minus_lsl_offset": (
            None if timing_context is None else timing_context.wall_minus_lsl_offset
        ),
    }
    body_coordinate_space = (
        "source_pixel" if config.body_backbone == "rtmpose-mmpose" else "normalized_image"
    )
    outputs: dict[str, Any] = {
        "frame_sync": str(frame_sync_path),
        "marker_detections_2d": str(marker_path) if marker_enabled else None,
    }
    if body_writer is not None:
        outputs["body_2d"] = body_writer.close(
            {
                **common_metadata,
                "coordinate_space": body_coordinate_space,
                "backbone": config.body_backbone,
                "model": (
                    args.rtmpose_model
                    if config.body_backbone == "rtmpose-mmpose"
                    else f"mediapipe_pose_landmarker_complexity_{args.pose_model_complexity}"
                ),
                "landmarks": (
                    f"mmpose_{body_landmarks}"
                    if config.body_backbone == "rtmpose-mmpose"
                    else "mediapipe_pose_33"
                ),
            }
        )
    if face_writer is not None:
        outputs["face_2d"] = face_writer.close(
            {
                **common_metadata,
                "backbone": config.face_backbone,
                "model": "mediapipe_face_landmarker",
                "landmarks": "mediapipe_face_478",
            }
        )
    if hand_writer is not None:
        outputs["hands_2d"] = hand_writer.close(
            {
                **common_metadata,
                "backbone": config.hand_backbone,
                "model": "mediapipe_hand_landmarker",
                "landmarks": "mediapipe_hand_21",
            }
        )

    return {
        "camera_id": camera_label,
        "source_video": str(video_path),
        "source_sha256_first64mb": _sha256_prefix(video_path),
        "resolution": [width, height],
        "fps": fps,
        "frames_processed": processed,
        "frame_log": str(frame_log_path) if frame_log_path else None,
        "task": None if timing_context is None else timing_context.task,
        "run": None if timing_context is None else timing_context.run,
        "acq": None if timing_context is None else timing_context.acq,
        "timing_source": "frame_log" if frame_log_path else (
            "video_pts" if timing_context is None else timing_context.source
        ),
        "clip_start_unix_time_s": (
            None if timing_context is None else timing_context.clip_start_unix_time_s
        ),
        "clip_start_lsl": None if timing_context is None else timing_context.clip_start_lsl,
        "marker_detections": marker_count,
        "body_detected_frames": body_frames,
        "face_detected_frames": face_frames,
        "hand_detected_frames": hand_frames,
        "elapsed_s": round(time.time() - start_wall, 3),
        "outputs": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Extract compact video-surrogate features from multicamera recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos-dir", type=Path, required=True, help="Directory containing camera videos"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for derived feature outputs"
    )
    parser.add_argument(
        "--frame-log-dir",
        type=Path,
        default=None,
        help="Optional directory containing frame-log JSONL files",
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=None,
        help="Specific video(s) to process instead of all videos in --videos-dir",
    )
    parser.add_argument(
        "--marker-config",
        type=Path,
        default=None,
        help="Optional YAML/JSON marker map for desk/glasses/board instance labels",
    )

    parser.add_argument(
        "--body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract body pose landmarks",
    )
    parser.add_argument(
        "--faces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract face landmarks",
    )
    parser.add_argument(
        "--hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract hand landmarks",
    )
    parser.add_argument(
        "--markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract ArUco marker detections",
    )
    parser.add_argument(
        "--body-backbone",
        choices=BODY_BACKBONES,
        default="mediapipe-pose",
        help="Body pose backbone",
    )
    parser.add_argument(
        "--face-backbone",
        choices=FACE_BACKBONES,
        default="mediapipe-face",
        help="Face landmark backbone",
    )
    parser.add_argument(
        "--hand-backbone",
        choices=HAND_BACKBONES,
        default="mediapipe-hands",
        help="Hand landmark backbone",
    )
    parser.add_argument(
        "--marker-backbone",
        choices=MARKER_BACKBONES,
        default="opencv-aruco",
        help="Marker detection backend",
    )

    parser.add_argument(
        "--max-people", type=int, default=5, help="Maximum body pose tracks per frame"
    )
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum faces per frame")
    parser.add_argument("--max-hands", type=int, default=10, help="Maximum hands per frame")
    parser.add_argument(
        "--body-stride", type=int, default=1, help="Run body detector every N frames"
    )
    parser.add_argument(
        "--face-stride", type=int, default=1, help="Run face detector every N frames"
    )
    parser.add_argument(
        "--hand-stride", type=int, default=1, help="Run hand detector every N frames"
    )
    parser.add_argument(
        "--marker-stride", type=int, default=1, help="Run marker detector every N frames"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Maximum frames per video; 0 means full video"
    )
    parser.add_argument(
        "--resize-width", type=int, default=None, help="Optional inference width to reduce compute"
    )
    parser.add_argument(
        "--float-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dense landmark storage dtype",
    )

    parser.add_argument(
        "--aruco-dicts",
        default=",".join(DEFAULT_ARUCO_DICTS),
        help="Comma-separated OpenCV ArUco dictionaries",
    )
    parser.add_argument(
        "--pose-model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe pose model complexity",
    )
    parser.add_argument(
        "--rtmpose-model",
        default="rtmw-l",
        help=(
            "MMPoseInferencer model alias/config for --body-backbone rtmpose-mmpose. "
            "Use an RTMW/wholebody alias for 133 landmarks or RTMPose body alias for 17."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device for optional backbones such as MMPose/RTMPose",
    )
    parser.add_argument(
        "--min-body-confidence",
        type=float,
        default=0.5,
        help="Minimum body detection/presence confidence",
    )
    parser.add_argument(
        "--min-face-confidence",
        type=float,
        default=0.4,
        help="Minimum face detection/presence confidence",
    )
    parser.add_argument(
        "--min-hand-confidence",
        type=float,
        default=0.4,
        help="Minimum hand detection/presence confidence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe tracking confidence",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Log progress every N frames; 0 disables",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a dry-run summary without decoding videos",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of videos to process in parallel within one session",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _resolve_backbone_enabled(enabled: bool, backbone: str) -> str:
    """Disable a backbone when its feature family is disabled."""
    return backbone if enabled else "none"


def _build_dry_run_summary(
    videos: list[Path],
    frame_log_dir: Path | None,
    output_dir: Path,
    config: ExtractorConfig,
) -> dict[str, Any]:
    """Create a lightweight preflight summary without decoding videos."""
    cameras: list[dict[str, Any]] = []
    for video in videos:
        camera_label = _camera_label_from_path(video)
        frame_log_path = _find_frame_log(frame_log_dir, video, camera_label)
        timing_context = _resolve_clip_timing_context(video)
        cameras.append(
            {
                "camera_id": camera_label,
                "source_video": str(video),
                "source_exists": video.exists(),
                "source_size_bytes": int(video.stat().st_size) if video.exists() else None,
                "frame_log": str(frame_log_path) if frame_log_path else None,
                "task": None if timing_context is None else timing_context.task,
                "run": None if timing_context is None else timing_context.run,
                "acq": None if timing_context is None else timing_context.acq,
                "timing_source": "frame_log" if frame_log_path else (
                    None if timing_context is None else timing_context.source
                ),
                "clip_start_unix_time_s": (
                    None if timing_context is None else timing_context.clip_start_unix_time_s
                ),
                "clip_start_lsl": None if timing_context is None else timing_context.clip_start_lsl,
            }
        )
    return {
        "schema_version": "affectai.video_features.dry_run.v1",
        "created_unix_s": time.time(),
        "output_dir": str(output_dir),
        "frame_log_dir": str(frame_log_dir) if frame_log_dir else None,
        "config": asdict(config),
        "video_count": len(videos),
        "cameras": cameras,
    }


def _resolve_worker_count(requested_workers: int, num_videos: int) -> int:
    """Clamp worker count to a sensible range for the current batch."""
    if num_videos <= 0:
        return 1
    if requested_workers <= 1:
        return 1
    cpu_cap = max(1, os.cpu_count() or 1)
    return max(1, min(requested_workers, num_videos, cpu_cap))


def _process_video_worker(
    video_path: Path,
    output_dir: Path,
    config: ExtractorConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Top-level worker wrapper for multiprocessing."""
    return process_video(video_path, output_dir, config, args)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.video:
        videos = [Path(p) for p in args.video]
    else:
        videos = _discover_videos(args.videos_dir)
    if not videos:
        parser.error(f"No videos found in {args.videos_dir}")

    config = ExtractorConfig(
        max_people=args.max_people,
        max_faces=args.max_faces,
        max_hands=args.max_hands,
        body=bool(args.body),
        faces=bool(args.faces),
        hands=bool(args.hands),
        markers=bool(args.markers),
        body_backbone=_resolve_backbone_enabled(bool(args.body), args.body_backbone),
        face_backbone=_resolve_backbone_enabled(bool(args.faces), args.face_backbone),
        hand_backbone=_resolve_backbone_enabled(bool(args.hands), args.hand_backbone),
        marker_backbone=_resolve_backbone_enabled(bool(args.markers), args.marker_backbone),
        body_stride=args.body_stride,
        face_stride=args.face_stride,
        hand_stride=args.hand_stride,
        marker_stride=args.marker_stride,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        aruco_dicts=_normalise_aruco_dict_names(args.aruco_dicts),
        float_dtype=args.float_dtype,
    )

    # Convert to UNC long paths on Windows if needed to bypass MAX_PATH limit
    output_dir = _to_windows_long_path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run_path = output_dir / "feature_extraction_dry_run.json"
        dry_run_path.write_text(
            json.dumps(
                _build_dry_run_summary(videos, args.frame_log_dir, output_dir, config),
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote dry-run summary: %s", dry_run_path)
        return 0

    worker_count = _resolve_worker_count(args.workers, len(videos))
    logger.info("Processing %d video(s)", len(videos))
    if worker_count > 1:
        logger.info(
            "Using %d workers across split clips. MediaPipe defaults to CPU/XNNPACK; "
            "GPU is only used by optional backbones such as RTMPose/MMPose.",
            worker_count,
        )

    cameras: list[dict[str, Any]] = []
    if worker_count == 1:
        for video in videos:
            logger.info("Extracting features: %s", video)
            cameras.append(process_video(video, output_dir, config, args))
    else:
        futures: dict[Any, Path] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for video in videos:
                logger.info("Queueing features: %s", video)
                future = executor.submit(_process_video_worker, video, output_dir, config, args)
                futures[future] = video

            for future in as_completed(futures):
                video = futures[future]
                logger.info("Collecting features: %s", video)
                cameras.append(future.result())

        cameras.sort(key=lambda row: str(row.get("source_video", "")))

    manifest = {
        "schema_version": "affectai.video_features.v1",
        "created_unix_s": time.time(),
        "videos_dir": str(args.videos_dir),
        "output_dir": str(args.output_dir),
        "config": asdict(config),
        "cameras": cameras,
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Wrote manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
