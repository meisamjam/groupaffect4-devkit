"""Tests for compact video feature extraction helpers."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import extract_video_features as evf  # noqa: E402, I001


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("jabra_panacast_20_cam1_vid_video.mkv", "panacast-20-cam1"),
        ("jabra_panacast_50_room_video.mp4", "panacast-50-room"),
        ("sub-01_ses-test_task-T1_acq-cam_3_video.mkv", "sub-01_ses-test_task-t1_acq-cam_3"),
    ],
)
def test_camera_label_from_path(filename: str, expected: str) -> None:
    assert evf._camera_label_from_path(Path(filename)) == expected


@pytest.mark.parametrize(
    ("total_frames", "stride", "max_frames", "expected"),
    [
        (10, 1, 0, 10),
        (10, 3, 0, 4),
        (10, 3, 5, 2),
        (0, 1, 0, 0),
    ],
)
def test_sample_count(total_frames: int, stride: int, max_frames: int, expected: int) -> None:
    assert evf._sample_count(total_frames, stride, max_frames) == expected


def test_sample_count_rejects_bad_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        evf._sample_count(10, 0, 0)


def test_normalise_aruco_dict_names_removes_blanks_and_duplicates() -> None:
    names = evf._normalise_aruco_dict_names("DICT_4X4_50, DICT_4X4_250,DICT_4X4_50,,")
    assert names == ("DICT_4X4_50", "DICT_4X4_250")


@pytest.mark.parametrize(
    ("backbone", "model", "expected"),
    [
        ("mediapipe-pose", "unused", 33),
        ("rtmpose-mmpose", "rtmpose-m", 17),
        ("rtmpose-mmpose", "rtmw-l", 133),
        ("rtmpose-mmpose", "wholebody", 133),
        ("none", "unused", 0),
    ],
)
def test_body_landmark_count(backbone: str, model: str, expected: int) -> None:
    assert evf._body_landmark_count(backbone, model) == expected


def test_resolve_backbone_enabled_turns_off_disabled_feature() -> None:
    assert evf._resolve_backbone_enabled(False, "mediapipe-pose") == "none"
    assert evf._resolve_backbone_enabled(True, "mediapipe-pose") == "mediapipe-pose"


def test_mmpose_result_to_array_scales_to_source_pixels() -> None:
    result = {
        "predictions": [
            [
                {
                    "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                    "keypoint_scores": [0.8, 0.6],
                }
            ]
        ]
    }

    arr = evf._mmpose_result_to_array(result, max_people=2, n_landmarks=3, scale_to_source=2.0)

    assert arr.shape == (2, 3, 4)
    np.testing.assert_allclose(arr[0, 0], [20.0, 40.0, np.nan, 0.8], equal_nan=True)
    np.testing.assert_allclose(arr[0, 1], [60.0, 80.0, np.nan, 0.6], equal_nan=True)
    assert np.isnan(arr[1]).all()


def test_discover_videos_filters_known_video_extensions(tmp_path: Path) -> None:
    keep = [
        tmp_path / "cam1.mkv",
        tmp_path / "cam2.MP4",
    ]
    skip = [
        tmp_path / "notes.txt",
        tmp_path / "frame_log.jsonl",
    ]
    for path in keep + skip:
        path.write_text("", encoding="utf-8")

    assert evf._discover_videos(tmp_path) == sorted(keep)


def test_marker_instance_lookup_preserves_duplicate_marker_ids() -> None:
    cfg = {
        "aruco_dictionary": "DICT_4X4_50",
        "table_markers": [{"id": 10, "name": "desk-front-left"}],
        "glasses": [{"id": "P1", "left_marker_id": 10, "right_marker_id": 11}],
    }

    lookup = evf._marker_instance_lookup(cfg)

    instances = lookup[("DICT_4X4_50", 10)]
    assert len(instances) == 2
    assert {instance.role for instance in instances} == {"desk", "glasses"}
    assert {instance.instance_id for instance in instances} == {"desk-front-left", "P1:left"}


def test_build_dry_run_summary_lists_videos_and_frame_logs(tmp_path: Path) -> None:
    videos_dir = tmp_path / "video"
    logs_dir = tmp_path / "frame_logs"
    out_dir = tmp_path / "features_video"
    videos_dir.mkdir()
    logs_dir.mkdir()

    video_path = videos_dir / "jabra_panacast_20_cam1_vid_video.mkv"
    video_path.write_bytes(b"dummy")
    frame_log_path = logs_dir / "ffmpeg_frames_panacast-20-cam1.jsonl"
    frame_log_path.write_text("", encoding="utf-8")

    cfg = evf.ExtractorConfig(
        max_people=5,
        max_faces=5,
        max_hands=10,
        body=True,
        faces=False,
        hands=True,
        markers=True,
        body_backbone="mediapipe-pose",
        face_backbone="none",
        hand_backbone="mediapipe-hands",
        marker_backbone="opencv-aruco",
        body_stride=1,
        face_stride=1,
        hand_stride=1,
        marker_stride=1,
        max_frames=0,
        resize_width=None,
        aruco_dicts=("DICT_4X4_50", "DICT_4X4_250"),
        float_dtype="float16",
    )

    summary = evf._build_dry_run_summary([video_path], logs_dir, out_dir, cfg)

    assert summary["schema_version"] == "affectai.video_features.dry_run.v1"
    assert summary["video_count"] == 1
    assert summary["cameras"][0]["camera_id"] == "panacast-20-cam1"
    assert summary["cameras"][0]["source_size_bytes"] == 5
    assert summary["cameras"][0]["frame_log"] == str(frame_log_path)


def test_resolve_clip_timing_context_from_task_run_windows(tmp_path: Path) -> None:
    session_dir = tmp_path / "sub-01" / "ses-demo"
    video_dir = session_dir / "video"
    annot_dir = session_dir / "annot"
    video_dir.mkdir(parents=True)
    annot_dir.mkdir()

    video_path = (
        video_dir
        / "sub-01_ses-demo_task-T1_run-01_acq-av-jabra-panacast-20-cam1-vid_video.mp4"
    )
    video_path.write_bytes(b"")
    task_windows = annot_dir / "sub-01_ses-demo_task-T0T1T2T3T4_task_run_windows.tsv"
    task_windows.write_text(
        "\n".join(
            [
                "task\trun\tstart_wall_clock\tend_wall_clock\tduration_s\tstart_lsl\tend_lsl\twall_minus_lsl_offset",
                "T1\t01\t100.5\t120.5\t20.0\t10.25\t30.25\t90.25",
            ]
        ),
        encoding="utf-8",
    )

    ctx = evf._resolve_clip_timing_context(video_path)

    assert ctx is not None
    assert ctx.task == "T1"
    assert ctx.run == "01"
    assert ctx.acq == "av-jabra-panacast-20-cam1-vid"
    assert ctx.clip_start_unix_time_s == pytest.approx(100.5)
    assert ctx.clip_start_lsl == pytest.approx(10.25)


def test_write_frame_sync_uses_task_window_timestamps_without_frame_logs() -> None:
    handle = StringIO()
    ctx = evf.ClipTimingContext(
        task="T1",
        run="01",
        acq="av-jabra-panacast-20-cam1-vid",
        clip_start_unix_time_s=100.5,
        clip_start_lsl=10.25,
        wall_minus_lsl_offset=90.25,
        source="task_run_windows",
    )

    evf._write_frame_sync(
        handle,
        camera_label="cam1",
        frame_idx=3,
        time_s=1.5,
        frame_log={},
        timing_context=ctx,
    )

    record = json.loads(handle.getvalue())
    assert record["task"] == "T1"
    assert record["run"] == "01"
    assert record["acq"] == "av-jabra-panacast-20-cam1-vid"
    assert record["unix_time_s"] == pytest.approx(102.0)
    assert record["wall_time_s"] == pytest.approx(102.0)
    assert record["lsl_time"] == pytest.approx(11.75)
    assert record["source"] == "task_run_windows+video_pts"


def test_build_dry_run_summary_includes_split_clip_timing_context(tmp_path: Path) -> None:
    session_dir = tmp_path / "sub-01" / "ses-demo"
    video_dir = session_dir / "video"
    annot_dir = session_dir / "annot"
    out_dir = tmp_path / "features_video"
    video_dir.mkdir(parents=True)
    annot_dir.mkdir()

    video_path = (
        video_dir
        / "sub-01_ses-demo_task-T1_run-01_acq-av-jabra-panacast-20-cam1-vid_video.mp4"
    )
    video_path.write_bytes(b"abc")
    (annot_dir / "sub-01_ses-demo_task-T0T1T2T3T4_task_run_windows.tsv").write_text(
        "\n".join(
            [
                "task\trun\tstart_wall_clock\tend_wall_clock\tduration_s\tstart_lsl\tend_lsl\twall_minus_lsl_offset",
                "T1\t01\t100.5\t120.5\t20.0\t10.25\t30.25\t90.25",
            ]
        ),
        encoding="utf-8",
    )

    cfg = evf.ExtractorConfig(
        max_people=5,
        max_faces=5,
        max_hands=10,
        body=True,
        faces=False,
        hands=True,
        markers=True,
        body_backbone="mediapipe-pose",
        face_backbone="none",
        hand_backbone="mediapipe-hands",
        marker_backbone="opencv-aruco",
        body_stride=1,
        face_stride=1,
        hand_stride=1,
        marker_stride=1,
        max_frames=0,
        resize_width=None,
        aruco_dicts=("DICT_4X4_50", "DICT_4X4_250"),
        float_dtype="float16",
    )

    summary = evf._build_dry_run_summary([video_path], None, out_dir, cfg)

    assert summary["cameras"][0]["task"] == "T1"
    assert summary["cameras"][0]["run"] == "01"
    assert summary["cameras"][0]["timing_source"] == "task_run_windows"
    assert summary["cameras"][0]["clip_start_unix_time_s"] == pytest.approx(100.5)
    assert summary["cameras"][0]["clip_start_lsl"] == pytest.approx(10.25)


def test_parser_supports_dry_run_flag() -> None:
    parser = evf.build_parser()
    args = parser.parse_args(["--videos-dir", "video", "--output-dir", "out", "--dry-run"])
    assert args.dry_run is True
