"""
Tests for FreeMoCap integration.

Covers:
  - FreeMoCapProcessor instantiation & BIDS output structure
  - Video file discovery (get_video_files)
  - task_id parsing helpers
  - Smoke-level import of freemocap package
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a minimal BIDS-like session directory tree."""
    ses = tmp_path / "sub-01" / "ses-01"
    (ses / "video").mkdir(parents=True)
    (ses / "mocap").mkdir(parents=True)
    return ses


@pytest.fixture()
def dummy_video(session_dir: Path) -> Path:
    """Create a zero-byte .mp4 so path-existence checks pass."""
    vid = session_dir / "video" / "sub-01_ses-01_task-T1_run-01_video.mp4"
    vid.write_bytes(b"")
    return vid


# ---------------------------------------------------------------------------
# Import / smoke tests
# ---------------------------------------------------------------------------

def test_freemocap_import():
    """FreeMoCap package is importable (may warn about Blender/circular-import, but must not hard-fail)."""
    try:
        import freemocap  # noqa: F401
    except ImportError as exc:
        # FreeMoCap 1.7.x has a known circular-import on first load when
        # logging handlers already exist; treat as skip rather than fail.
        if "circular import" in str(exc):
            pytest.skip(f"FreeMoCap circular-import bug: {exc}")
        raise


def test_processor_import():
    """FreeMoCapProcessor class is importable."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor  # noqa: F401


# ---------------------------------------------------------------------------
# FreeMoCapProcessor unit tests
# ---------------------------------------------------------------------------

def test_processor_creates_mocap_dir(tmp_path: Path):
    """Processor __init__ creates mocap/ inside session_dir."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    ses = tmp_path / "sub-02" / "ses-01"
    ses.mkdir(parents=True)
    proc = FreeMoCapProcessor(ses)
    assert proc.mocap_dir.exists()
    assert proc.mocap_dir == ses / "mocap"


def test_processor_metadata_structure(session_dir: Path):
    """_create_metadata returns expected BIDS metadata keys."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    proc = FreeMoCapProcessor(session_dir)
    meta = proc._create_metadata({}, "freemocap")
    assert meta["Source"] == "FreeMoCap"
    assert meta["SkeletonModel"] == "mediapipe_holistic"
    assert "Description" in meta
    assert "AcquisitionLabel" in meta


def test_processor_metadata_with_frame_info(session_dir: Path):
    """Metadata includes frame_rate/number_of_frames when present."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    proc = FreeMoCapProcessor(session_dir)
    output_dict = {"frame_rate": 30.0, "number_of_frames": 900}
    meta = proc._create_metadata(output_dict, "freemocap")
    assert meta["FrameRate"] == 30.0
    assert meta["NumberOfFrames"] == 900


def test_processor_save_metadata_json(session_dir: Path):
    """_save_metadata writes valid JSON sidecar."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    proc = FreeMoCapProcessor(session_dir)
    meta = {"Description": "test", "Source": "FreeMoCap"}
    out = proc._save_metadata(meta, "freemocap", "T1", 1)
    assert out.exists()
    assert out.suffix == ".json"
    data = json.loads(out.read_text())
    assert data["Source"] == "FreeMoCap"


def test_processor_extract_skeleton_data(session_dir: Path):
    """_extract_skeleton_data returns correct structure for known keys."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    proc = FreeMoCapProcessor(session_dir)

    # With marker_data key
    result = proc._extract_skeleton_data({"marker_data": [1, 2, 3]}, 0.5)
    assert result["markers"] == [1, 2, 3]
    assert result["confidence_threshold"] == 0.5

    # With body_markers key
    result = proc._extract_skeleton_data({"body_markers": [4, 5]}, 0.3)
    assert result["markers"] == [4, 5]

    # With skeletal_data key
    result = proc._extract_skeleton_data({"skeletal_data": [6]}, 0.9)
    assert result["markers"] == [6]

    # Empty output dict
    result = proc._extract_skeleton_data({}, 0.5)
    assert "markers" not in result
    assert result["confidence_threshold"] == 0.5


# ---------------------------------------------------------------------------
# Video discovery tests
# ---------------------------------------------------------------------------

def test_get_video_files_discovers_mp4(session_dir: Path, dummy_video: Path):
    """get_video_files finds BIDS-named .mp4 in video/ dir."""
    from tools.process_freemocap import get_video_files

    videos = get_video_files(session_dir)
    assert len(videos) >= 1
    # Should contain a key with 'T1'
    assert any("T1" in k for k in videos), f"Expected T1 key, got {list(videos.keys())}"


def test_get_video_files_filters_by_task(session_dir: Path, dummy_video: Path):
    """get_video_files respects task_list filter."""
    from tools.process_freemocap import get_video_files

    # Create a second video for T2
    vid2 = session_dir / "video" / "sub-01_ses-01_task-T2_run-01_video.mp4"
    vid2.write_bytes(b"")

    # Only request T2
    videos = get_video_files(session_dir, task_list=["T2"])
    assert all("T2" in k for k in videos)
    assert not any("T1" in k for k in videos)


def test_get_video_files_filters_by_label(session_dir: Path):
    """get_video_files respects video_label filter."""
    from tools.process_freemocap import get_video_files

    # Create videos with different labels in filename
    (session_dir / "video" / "sub-01_ses-01_task-T1_acq-jabra_panacast_video.mp4").write_bytes(b"")
    (session_dir / "video" / "sub-01_ses-01_task-T1_acq-webcam_video.mp4").write_bytes(b"")

    videos = get_video_files(session_dir, video_label="jabra_panacast")
    assert len(videos) >= 1
    # All discovered paths should contain 'jabra_panacast'
    for path in videos.values():
        assert "jabra_panacast" in path.name


def test_get_video_files_empty_dir(tmp_path: Path):
    """get_video_files returns {} when video/ dir is missing."""
    from tools.process_freemocap import get_video_files

    result = get_video_files(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# Task-ID parsing helpers
# ---------------------------------------------------------------------------

def test_parse_task_id_with_run():
    """Parse 'T1_run01' style task IDs."""
    from tools.process_freemocap import _parse_task_id_from_path

    task, run = _parse_task_id_from_path("T1_run01")
    assert task == "T1"
    assert run == 1


def test_parse_task_id_without_run():
    """Parse bare task name."""
    from tools.process_freemocap import _parse_task_id_from_path

    task, run = _parse_task_id_from_path("T3")
    assert task == "T3"
    assert run == 1


def test_parse_task_id_from_processor_module():
    """_parse_task_id in freemocap_processor handles multiple formats."""
    from affectai_capture.devices.freemocap_processor import _parse_task_id

    assert _parse_task_id("T1_run1") == ("T1", 1)
    assert _parse_task_id("task-T2") == ("T2", 1)
    assert _parse_task_id("T3") == ("T3", 1)


# ---------------------------------------------------------------------------
# Skeleton TSV output
# ---------------------------------------------------------------------------

def test_save_skeleton_tsv_creates_file(session_dir: Path):
    """_save_skeleton_tsv creates a valid TSV with header."""
    from affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

    proc = FreeMoCapProcessor(session_dir)
    skeleton_data = {"markers": [], "confidence_threshold": 0.5}
    tsv_path = proc._save_skeleton_tsv(skeleton_data, "freemocap", "T1", 1)
    assert tsv_path.exists()
    assert tsv_path.suffix == ".tsv"

    content = tsv_path.read_text()
    # Header must contain expected columns
    for col in ("frame", "joint", "x", "y", "z", "confidence"):
        assert col in content.split("\n")[0]
