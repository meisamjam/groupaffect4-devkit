from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.video_only_3d_pipeline import (
    GestureThresholds,
    _discover_p20_videos,
    _pose_json_dirs,
    _prereq_report,
    extract_gesture_events,
)


def _make_empty_skeleton(frames: int, people: int, keypoints: int = 25) -> np.ndarray:
    data = np.full((frames, people, keypoints, 7), np.nan, dtype=np.float64)
    return data


def _set_joint(data: np.ndarray, f: int, p: int, k: int, xyz: tuple[float, float, float], conf: float = 1.0) -> None:
    data[f, p, k, 0:3] = np.array(xyz, dtype=np.float64)
    data[f, p, k, 3] = conf


def test_extract_gesture_events_detects_hand_to_head(tmp_path: Path) -> None:
    frames = 12
    skel = _make_empty_skeleton(frames=frames, people=1)

    # Keep shoulder/elbow/wrist and nose valid for all frames.
    for f in range(frames):
        _set_joint(skel, f, 0, 0, (0.0, 0.0, 0.0))   # nose
        _set_joint(skel, f, 0, 5, (-0.2, -0.1, 0.0)) # left shoulder
        _set_joint(skel, f, 0, 6, (-0.1, -0.05, 0.0))# left elbow

        # Frames 3..8: left wrist near head (gesture true)
        if 3 <= f <= 8:
            _set_joint(skel, f, 0, 7, (-0.02, 0.01, 0.0))
        else:
            _set_joint(skel, f, 0, 7, (-0.6, -0.6, 0.0))

        # Right arm valid but not near head
        _set_joint(skel, f, 0, 2, (0.2, -0.1, 0.0))
        _set_joint(skel, f, 0, 3, (0.3, -0.1, 0.0))
        _set_joint(skel, f, 0, 4, (0.4, -0.1, 0.0))

        _set_joint(skel, f, 0, 8, (0.0, -0.4, 0.0))  # mid hip

    skel_path = tmp_path / "skeleton.npy"
    np.save(skel_path, skel)

    events_path = tmp_path / "events.ndjson"
    summary_path = tmp_path / "summary.json"

    summary = extract_gesture_events(
        skeleton_path=skel_path,
        output_ndjson=events_path,
        summary_json=summary_path,
        fps=30.0,
        min_confidence=0.2,
        thresholds=GestureThresholds(min_event_frames=4),
    )

    assert summary["n_events"] >= 1

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(x) for x in lines if x.strip()]

    left_head = [e for e in parsed if e["gesture"] == "left_hand_to_head"]
    assert len(left_head) == 1
    assert left_head[0]["start_frame"] == 3
    assert left_head[0]["end_frame"] == 8


def test_extract_gesture_events_empty_when_no_valid_joints(tmp_path: Path) -> None:
    skel = _make_empty_skeleton(frames=5, people=2)
    skel_path = tmp_path / "skeleton.npy"
    np.save(skel_path, skel)

    events_path = tmp_path / "events.ndjson"
    summary_path = tmp_path / "summary.json"

    summary = extract_gesture_events(
        skeleton_path=skel_path,
        output_ndjson=events_path,
        summary_json=summary_path,
        fps=30.0,
        min_confidence=0.2,
        thresholds=GestureThresholds(min_event_frames=3),
    )

    assert summary["n_events"] == 0
    assert events_path.read_text(encoding="utf-8") == ""


def test_prereq_report_detects_missing_pose_dirs(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.toml"
    calibration.write_text("", encoding="utf-8")
    videos_dir = tmp_path / "video"
    videos_dir.mkdir(parents=True)
    tracker_cfg = tmp_path / "tracker.yaml"
    tracker_cfg.write_text("", encoding="utf-8")
    pose_root = tmp_path / "poses"
    pose_root.mkdir(parents=True)

    args = SimpleNamespace(
        calibration=calibration,
        videos_dir=videos_dir,
        tracker_config=tracker_cfg,
        pose_root=pose_root,
        auto_calibrate_missing=True,
    )
    report = _prereq_report(args)
    assert report["pose_json_dir_count"] == 0
    assert "pose_json_dirs" in report["missing"]
    assert report["ready"] is False


def test_pose_json_dirs_filters_suffix(tmp_path: Path) -> None:
    pose_root = tmp_path / "poses"
    pose_root.mkdir(parents=True)
    (pose_root / "cam1_json").mkdir()
    (pose_root / "cam2_json").mkdir()
    (pose_root / "notes").mkdir()

    found = _pose_json_dirs(pose_root)
    names = [p.name for p in found]
    assert names == ["cam1_json", "cam2_json"]


def test_discover_p20_videos_selects_six_unique_cams(tmp_path: Path) -> None:
    videos = tmp_path / "video"
    videos.mkdir(parents=True)
    for i in range(1, 7):
        (videos / f"cam-p20_panacast-20-cam{i}_video.mkv").write_text("", encoding="utf-8")

    # Duplicate for cam1 should be ignored because first sorted one is kept.
    (videos / "z_cam-p20_panacast-20-cam1_video.mp4").write_text("", encoding="utf-8")

    selected = _discover_p20_videos(videos)
    assert len(selected) == 6
    assert [p.name for p in selected] == [
        "cam-p20_panacast-20-cam1_video.mkv",
        "cam-p20_panacast-20-cam2_video.mkv",
        "cam-p20_panacast-20-cam3_video.mkv",
        "cam-p20_panacast-20-cam4_video.mkv",
        "cam-p20_panacast-20-cam5_video.mkv",
        "cam-p20_panacast-20-cam6_video.mkv",
    ]


def test_prereq_report_requires_p20_when_auto_calibration_enabled(tmp_path: Path) -> None:
    calibration = tmp_path / "missing.toml"
    videos_dir = tmp_path / "video"
    videos_dir.mkdir(parents=True)
    for i in range(1, 5):
        (videos_dir / f"cam-p20_panacast-20-cam{i}_video.mkv").write_text("", encoding="utf-8")

    tracker_cfg = tmp_path / "tracker.yaml"
    tracker_cfg.write_text("", encoding="utf-8")
    pose_root = tmp_path / "poses"
    pose_root.mkdir(parents=True)
    (pose_root / "cam1_json").mkdir()

    args = SimpleNamespace(
        calibration=calibration,
        videos_dir=videos_dir,
        tracker_config=tracker_cfg,
        pose_root=pose_root,
        auto_calibrate_missing=True,
    )
    report = _prereq_report(args)

    assert "calibration" not in report["missing"]
    assert "p20_videos_for_autocalibration" in report["missing"]
    assert report["ready"] is False
