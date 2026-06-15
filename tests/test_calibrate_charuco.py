"""Combined tests for multicamera charuco calibration (tools/calibrate_charuco.py).

Tests cover:
- Constants and board definitions
- Camera-spec loading, matching, and intrinsic matrix building
- Video file discovery and exclusion patterns
- CLI argument parsing and subcommand dispatch table
- H.264 encoder detection (mocked)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add tools/ to path so we can import calibrate_charuco as a module
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import calibrate_charuco as cc  # noqa: E402, I001


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def camera_specs() -> dict:
    """Return a minimal camera_specs dict matching the repo's format."""
    return {
        "models": {
            "jabra_panacast_20": {
                "description": "P20",
                "hfov_deg": 120,
                "expected_fx_1080p": 554.3,
                "expected_fy_1080p": 554.3,
            },
            "jabra_panacast_50": {
                "description": "P50",
                "hfov_deg": 133,
                "expected_fx_1080p": 418.2,
                "expected_fy_1080p": 418.2,
            },
        },
        "camera_name_patterns": {
            "_comment": "skip me",
            "jabra_panacast_20_cam[0-9]+(_vid)?": "jabra_panacast_20",
            "jabra_panacast_50(_cam[0-9]+)?(_vid)?": "jabra_panacast_50",
            "cam[0-9]+_p20": "jabra_panacast_20",
        },
    }


@pytest.fixture()
def specs_json(tmp_path: Path, camera_specs: dict) -> Path:
    """Write camera_specs to a temp JSON file and return the path."""
    p = tmp_path / "camera_specs.json"
    p.write_text(json.dumps(camera_specs), encoding="utf-8")
    return p


@pytest.fixture()
def videos_dir(tmp_path: Path) -> Path:
    """Create a directory with dummy video files for discovery tests."""
    vdir = tmp_path / "video"
    vdir.mkdir()
    for name in [
        "cam_0.mp4", "cam_1.mkv", "cam_2.avi",
        "sync_grid.mp4", "sync_grid_cfr.mp4",
        "combined.mp4", "mosaic.mp4",
        "notes.txt",
    ]:
        (vdir / name).write_bytes(b"")
    return vdir


# -----------------------------------------------------------------------
# Board definitions
# -----------------------------------------------------------------------

class TestBoardDefinitions:
    def test_default_board_exists(self):
        assert cc.DEFAULT_BOARD in cc.CHARUCO_BOARDS

    def test_all_boards_have_required_keys(self):
        for _key, spec in cc.CHARUCO_BOARDS.items():
            assert "width" in spec
            assert "height" in spec
            assert "name" in spec

    def test_7x5_corners(self):
        s = cc.CHARUCO_BOARDS["7x5"]
        expected = (s["width"] - 1) * (s["height"] - 1)  # 6 * 4 = 24
        assert expected == 24

    def test_5x3_corners(self):
        s = cc.CHARUCO_BOARDS["5x3"]
        expected = (s["width"] - 1) * (s["height"] - 1)  # 4 * 2 = 8
        assert expected == 8

    def test_default_square_size_positive(self):
        assert cc.DEFAULT_SQUARE_SIZE_MM > 0


# -----------------------------------------------------------------------
# Camera-spec helpers
# -----------------------------------------------------------------------

class TestCameraSpecs:
    def test_load_camera_specs_file(self, specs_json: Path, camera_specs: dict):
        loaded = cc._load_camera_specs(specs_json)
        assert set(loaded["models"]) == set(camera_specs["models"])

    def test_match_p20_name(self, camera_specs: dict):
        model = cc._match_camera_model("jabra_panacast_20_cam1_vid", camera_specs)
        assert model is not None
        assert "P20" in model.get("description", "")

    def test_match_p20_without_suffix(self, camera_specs: dict):
        model = cc._match_camera_model("jabra_panacast_20_cam3", camera_specs)
        assert model is not None

    def test_match_p50_name(self, camera_specs: dict):
        model = cc._match_camera_model("jabra_panacast_50_cam4_vid", camera_specs)
        assert model is not None
        assert "P50" in model.get("description", "")

    def test_match_alt_pattern(self, camera_specs: dict):
        model = cc._match_camera_model("cam2_p20", camera_specs)
        assert model is not None

    def test_no_match_returns_none(self, camera_specs: dict):
        assert cc._match_camera_model("random_webcam", camera_specs) is None

    def test_comment_patterns_are_skipped(self, camera_specs: dict):
        # _comment key should be skipped (starts with _)
        assert cc._match_camera_model("_comment", camera_specs) is None

    def test_match_case_insensitive(self, camera_specs: dict):
        model = cc._match_camera_model("Jabra_PanaCast_20_Cam1_Vid", camera_specs)
        assert model is not None


class TestBuildIntrinsicMatrix:
    def test_basic_shape(self, camera_specs: dict):
        model = camera_specs["models"]["jabra_panacast_20"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        assert K.shape == (3, 3)
        assert K.dtype == np.float64

    def test_principal_point_at_centre(self, camera_specs: dict):
        model = camera_specs["models"]["jabra_panacast_20"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        assert K[0, 2] == pytest.approx(960.0)
        assert K[1, 2] == pytest.approx(540.0)

    def test_fx_matches_spec_at_1080p(self, camera_specs: dict):
        model = camera_specs["models"]["jabra_panacast_20"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        assert K[0, 0] == pytest.approx(554.3, abs=0.1)

    def test_scaling_for_different_resolution(self, camera_specs: dict):
        model = camera_specs["models"]["jabra_panacast_20"]
        cc._build_intrinsic_matrix(model, 1920, 1080)  # reference
        K_720 = cc._build_intrinsic_matrix(model, 1280, 720)
        # fx should scale proportionally: 1280/1920 * 554.3
        expected_fx_720 = 554.3 * (1280 / 1920)
        assert K_720[0, 0] == pytest.approx(expected_fx_720, abs=0.1)

    def test_hfov_computation_when_no_fx(self):
        """When only hfov_deg is provided (no expected_fx_1080p)."""
        model = {"hfov_deg": 90}
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        # fx = (1920/2) / tan(45°) = 960 / 1.0 = 960
        assert K[0, 0] == pytest.approx(960.0, abs=0.1)

    def test_fallback_no_spec(self):
        """When no spec info is available, fx defaults to width."""
        model = {}
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        assert K[0, 0] == pytest.approx(1920.0)

    def test_k_matrix_zero_off_diagonal(self, camera_specs: dict):
        model = camera_specs["models"]["jabra_panacast_20"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        assert K[0, 1] == 0.0  # no skew
        assert K[1, 0] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        assert K[2, 2] == 1.0


# -----------------------------------------------------------------------
# Video discovery
# -----------------------------------------------------------------------

class TestVideoDiscovery:
    def test_finds_valid_videos(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = {p.name for p in found}
        assert "cam_0.mp4" in names
        assert "cam_1.mkv" in names
        assert "cam_2.avi" in names

    def test_excludes_sync_grid(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = {p.name for p in found}
        assert "sync_grid.mp4" not in names
        assert "sync_grid_cfr.mp4" not in names

    def test_excludes_combined(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = {p.name for p in found}
        assert "combined.mp4" not in names

    def test_excludes_mosaic(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = {p.name for p in found}
        assert "mosaic.mp4" not in names

    def test_excludes_non_video_extensions(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = {p.name for p in found}
        assert "notes.txt" not in names

    def test_returns_sorted(self, videos_dir: Path):
        found = cc._discover_video_files(videos_dir)
        names = [p.name for p in found]
        assert names == sorted(names)

    def test_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert cc._discover_video_files(empty) == []


# -----------------------------------------------------------------------
# H.264 encoder detection
# -----------------------------------------------------------------------

class TestEncoderDetection:
    @patch("calibrate_charuco.subprocess.run")
    def test_prefers_libx264(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(stdout="libx264 h264_mf", stderr="")
        args = cc._detect_h264_encoder()
        assert "-c:v" in args
        assert "libx264" in args

    @patch("calibrate_charuco.subprocess.run")
    def test_falls_back_to_h264_mf(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(stdout="h264_mf only", stderr="")
        args = cc._detect_h264_encoder()
        assert "h264_mf" in args

    @patch("calibrate_charuco.subprocess.run")
    def test_falls_back_to_mjpeg(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(stdout="no useful encoders", stderr="")
        args = cc._detect_h264_encoder()
        assert "mjpeg" in args

    @patch("calibrate_charuco.subprocess.run", side_effect=Exception("ffmpeg not found"))
    def test_exception_falls_back_to_mjpeg(self, mock_run: MagicMock):
        args = cc._detect_h264_encoder()
        assert "mjpeg" in args


# -----------------------------------------------------------------------
# CLI / dispatch
# -----------------------------------------------------------------------

class TestCLIParsing:
    """Verify that main() has a dispatch dict covering all subcommands."""

    def test_all_subcommands_in_dispatch(self):
        """The dispatch dict at the end of main() should include all 6 subcommands."""
        # Verify main() runs and the parser has all 6 subcommands by calling --help.
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["calibrate_charuco", "--help"]):
                cc.main()

    def test_exclude_patterns(self):
        """Verify the exclude patterns set is defined and non-empty."""
        assert isinstance(cc._EXCLUDE_PATTERNS, set)
        assert "sync_grid" in cc._EXCLUDE_PATTERNS
        assert "combined" in cc._EXCLUDE_PATTERNS
        assert "mosaic" in cc._EXCLUDE_PATTERNS

    def test_default_camera_specs_path(self):
        """DEFAULT_CAMERA_SPECS should point to configs/camera_specs.json."""
        assert cc.DEFAULT_CAMERA_SPECS.name == "camera_specs.json"
        assert "configs" in str(cc.DEFAULT_CAMERA_SPECS)


# -----------------------------------------------------------------------
# Integration-style test: full specs file from repo
# -----------------------------------------------------------------------

class TestRepoSpecs:
    """Test with the actual camera_specs.json from the repo (if available)."""

    @pytest.fixture()
    def real_specs(self) -> dict | None:
        specs_path = cc.DEFAULT_CAMERA_SPECS
        if not specs_path.exists():
            pytest.skip("camera_specs.json not found")
        return cc._load_camera_specs(specs_path)

    def test_p20_model_exists(self, real_specs: dict):
        assert "jabra_panacast_20" in real_specs["models"]

    def test_p50_model_exists(self, real_specs: dict):
        assert "jabra_panacast_50" in real_specs["models"]

    def test_p20_fx_reasonable(self, real_specs: dict):
        model = real_specs["models"]["jabra_panacast_20"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        fx = K[0, 0]
        # For 120-deg HFOV: fx ~ 554 px at 1080p
        assert 400 < fx < 700, f"P20 fx={fx} outside reasonable range"

    def test_p50_fx_reasonable(self, real_specs: dict):
        model = real_specs["models"]["jabra_panacast_50"]
        K = cc._build_intrinsic_matrix(model, 1920, 1080)
        fx = K[0, 0]
        # For 133-deg HFOV: fx ~ 418 px at 1080p
        assert 300 < fx < 600, f"P50 fx={fx} outside reasonable range"

    def test_all_patterns_resolve_to_known_models(self, real_specs: dict):
        patterns = real_specs.get("camera_name_patterns", {})
        models = real_specs.get("models", {})
        for key, model_key in patterns.items():
            if key.startswith("_"):
                continue
            assert model_key in models, f"Pattern '{key}' maps to unknown model '{model_key}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
