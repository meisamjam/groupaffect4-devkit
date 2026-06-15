"""Combined tests for Tobii multicam glasses pose tracking
(tools/tobii_multicam_glasses_tracker.py).

Tests cover:
- CameraCalibration: construction, projection, undistortion
- Configuration parsing: GlassesMarkerConfig, TableMarkerConfig, TrackerConfig
- ArUco detection helpers: MarkerDetection dataclass
- DLT triangulation: synthetic multi-view geometry
- Marker corner triangulation pipeline
- Glasses 6-DoF pose estimation from marker corners
- Gaze-to-world transformation (ray-plane intersection)
- Video discovery and camera-to-video mapping
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import tobii_multicam_glasses_tracker as tg  # noqa: E402, I001


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_identity_camera(key: str = "cam_0") -> tg.CameraCalibration:
    """Create a camera at the world origin with identity extrinsics."""
    data = {
        "name": key,
        "size": [1920, 1080],
        "matrix": [[500, 0, 960], [0, 500, 540], [0, 0, 1]],
        "distortions": [0, 0, 0, 0, 0],
        "rotation": [0, 0, 0],
        "translation": [0, 0, 0],
    }
    return tg.CameraCalibration(key, data)


def _make_translated_camera(key: str, tx: float) -> tg.CameraCalibration:
    """Create a camera translated along X by *tx* (baseline for stereo)."""
    data = {
        "name": key,
        "size": [1920, 1080],
        "matrix": [[500, 0, 960], [0, 500, 540], [0, 0, 1]],
        "distortions": [0, 0, 0, 0, 0],
        "rotation": [0, 0, 0],
        "translation": [tx, 0, 0],
    }
    return tg.CameraCalibration(key, data)


@pytest.fixture()
def stereo_cameras() -> dict[str, tg.CameraCalibration]:
    """Two cameras 0.5m apart along X, both looking down +Z."""
    return {
        "cam_0": _make_identity_camera("cam_0"),
        "cam_1": _make_translated_camera("cam_1", 0.5),
    }


@pytest.fixture()
def tracker_yaml(tmp_path: Path) -> Path:
    """Write a minimal tracker config YAML and return its path."""
    cfg = {
        "aruco_dictionary": "DICT_4X4_50",
        "video_fps": 30.0,
        "min_cameras_for_triangulation": 2,
        "max_reproj_error_px": 10.0,
        "table_markers": [
            {
                "id": 0,
                "corners_m": [
                    [-0.6, -0.4, 0.0],
                    [-0.55, -0.4, 0.0],
                    [-0.55, -0.45, 0.0],
                    [-0.6, -0.45, 0.0],
                ],
            },
        ],
        "glasses": [
            {
                "id": "tobii_p1",
                "left_marker_id": 10,
                "right_marker_id": 11,
                "marker_size_m": 0.015,
                "left_marker_offset_mm": [40.0, 0.0, 10.0],
                "right_marker_offset_mm": [-40.0, 0.0, 10.0],
                "gaze_ndjson": None,
                "time_offset_s": 0.0,
            },
        ],
    }
    p = tmp_path / "tracker_cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


@pytest.fixture()
def tracker_yaml_world_marker_map(tmp_path: Path) -> Path:
    """Write tracker config using world.marker_map schema and return its path."""
    cfg = {
        "world": {
            "aruco_dictionary": "DICT_4X4_50",
            "marker_map": [
                {
                    "id": 0,
                    "corners_m": [
                        [-0.6, -0.4, 0.0],
                        [-0.55, -0.4, 0.0],
                        [-0.55, -0.45, 0.0],
                        [-0.6, -0.45, 0.0],
                    ],
                },
            ],
        },
        "glasses": [
            {
                "id": "tobii_p1",
                "left_marker_id": 10,
                "right_marker_id": 11,
                "marker_size_m": 0.015,
                "left_marker_offset_mm": [40.0, 0.0, 10.0],
                "right_marker_offset_mm": [-40.0, 0.0, 10.0],
            },
        ],
    }
    p = tmp_path / "tracker_cfg_world_marker_map.yaml"
    import yaml
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# -----------------------------------------------------------------------
# CameraCalibration
# -----------------------------------------------------------------------

class TestCameraCalibration:
    def test_identity_camera_at_origin(self):
        cam = _make_identity_camera()
        np.testing.assert_allclose(cam.world_pos.flatten(), [0, 0, 0], atol=1e-9)

    def test_intrinsic_matrix_shape(self):
        cam = _make_identity_camera()
        assert cam.K.shape == (3, 3)

    def test_projection_matrix_shape(self):
        cam = _make_identity_camera()
        assert cam.P.shape == (3, 4)

    def test_project_origin_gives_principal_point(self):
        """A 3D point at [0,0,1] should project near the principal point."""
        cam = _make_identity_camera()
        pt = np.array([[0.0, 0.0, 1.0]])
        px = cam.project(pt)
        np.testing.assert_allclose(px[0], [960, 540], atol=1)

    def test_project_off_centre(self):
        cam = _make_identity_camera()
        # Point 1m right, at depth 1m → projects to 960 + 500 = 1460
        pt = np.array([[1.0, 0.0, 1.0]])
        px = cam.project(pt)
        assert px[0, 0] == pytest.approx(1460, abs=1)
        assert px[0, 1] == pytest.approx(540, abs=1)

    def test_undistort_with_zero_distortion(self):
        cam = _make_identity_camera()
        pts = np.array([[500.0, 300.0], [960.0, 540.0]])
        undist = cam.undistort_points(pts)
        np.testing.assert_allclose(undist, pts, atol=0.1)

    def test_translated_camera_world_pos(self):
        cam = _make_translated_camera("cam_1", 2.0)
        # Translation [2, 0, 0] with identity rotation → world_pos = -R^T @ t = [-2, 0, 0]
        np.testing.assert_allclose(cam.world_pos.flatten(), [-2, 0, 0], atol=1e-9)

    def test_camera_name(self):
        cam = _make_identity_camera("my_cam")
        assert cam.name == "my_cam"
        assert cam.key == "my_cam"

    def test_camera_size(self):
        cam = _make_identity_camera()
        assert cam.size == (1920, 1080)


# -----------------------------------------------------------------------
# Config parsing
# -----------------------------------------------------------------------

class TestConfigParsing:
    def test_load_tracker_config(self, tracker_yaml: Path):
        cfg = tg.load_tracker_config(tracker_yaml)
        assert isinstance(cfg, tg.TrackerConfig)
        assert cfg.aruco_dictionary == "DICT_4X4_50"
        assert cfg.video_fps == 30.0
        assert cfg.min_cameras_for_triangulation == 2
        assert cfg.max_reproj_error_px == 10.0

    def test_glasses_parsed(self, tracker_yaml: Path):
        cfg = tg.load_tracker_config(tracker_yaml)
        assert len(cfg.glasses) == 1
        g = cfg.glasses[0]
        assert g.glasses_id == "tobii_p1"
        assert g.left_marker_id == 10
        assert g.right_marker_id == 11
        assert g.marker_size_m == pytest.approx(0.015)

    def test_glasses_offsets(self, tracker_yaml: Path):
        cfg = tg.load_tracker_config(tracker_yaml)
        g = cfg.glasses[0]
        assert g.left_marker_offset_mm == [40.0, 0.0, 10.0]
        assert g.right_marker_offset_mm == [-40.0, 0.0, 10.0]

    def test_table_markers_parsed(self, tracker_yaml: Path):
        cfg = tg.load_tracker_config(tracker_yaml)
        assert len(cfg.table_markers) == 1
        tm = cfg.table_markers[0]
        assert tm.marker_id == 0
        assert tm.corners_m.shape == (4, 3)

    def test_world_marker_map_format_parsed(self, tracker_yaml_world_marker_map: Path):
        cfg = tg.load_tracker_config(tracker_yaml_world_marker_map)
        assert cfg.aruco_dictionary == "DICT_4X4_50"
        assert len(cfg.table_markers) == 1
        assert cfg.table_markers[0].marker_id == 0

    def test_glasses_marker_config_defaults(self):
        g = tg.GlassesMarkerConfig(
            glasses_id="test",
            left_marker_id=10,
            right_marker_id=11,
            marker_size_m=0.02,
            left_marker_offset_mm=[0, 0, 0],
            right_marker_offset_mm=[0, 0, 0],
        )
        assert g.gaze_ndjson is None
        assert g.time_offset_s == 0.0


# -----------------------------------------------------------------------
# Marker detection dataclass
# -----------------------------------------------------------------------

class TestMarkerDetection:
    def test_creation(self):
        corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        det = tg.MarkerDetection(marker_id=5, corners_px=corners, camera_key="cam_0")
        assert det.marker_id == 5
        assert det.camera_key == "cam_0"
        assert det.corners_px.shape == (4, 2)


# -----------------------------------------------------------------------
# DLT triangulation
# -----------------------------------------------------------------------

class TestTriangulateDLT:
    def test_triangulate_identity_stereo(self, stereo_cameras):
        """Two cameras, both project a known 3D point → triangulate recovers it."""
        true_pt = np.array([0.1, 0.2, 2.0])
        cam0 = stereo_cameras["cam_0"]
        cam1 = stereo_cameras["cam_1"]
        px0 = cam0.project(true_pt.reshape(1, 3))[0]
        px1 = cam1.project(true_pt.reshape(1, 3))[0]

        result = tg.triangulate_point_dlt([(cam0, px0), (cam1, px1)])
        assert result is not None
        pt_3d, err = result
        np.testing.assert_allclose(pt_3d, true_pt, atol=0.01)
        assert err < 1.0  # reprojection error should be small

    def test_insufficient_observations(self, stereo_cameras):
        cam0 = stereo_cameras["cam_0"]
        px = np.array([960.0, 540.0])
        result = tg.triangulate_point_dlt([(cam0, px)])
        assert result is None

    def test_empty_observations(self):
        result = tg.triangulate_point_dlt([])
        assert result is None

    def test_three_camera_triangulation(self):
        """Three cameras should yield more accurate triangulation."""
        true_pt = np.array([0.3, -0.1, 3.0])
        cams = [
            _make_identity_camera("cam_0"),
            _make_translated_camera("cam_1", 0.5),
            _make_translated_camera("cam_2", -0.5),
        ]
        observations = []
        for c in cams:
            px = c.project(true_pt.reshape(1, 3))[0]
            observations.append((c, px))
        result = tg.triangulate_point_dlt(observations)
        assert result is not None
        pt_3d, err = result
        np.testing.assert_allclose(pt_3d, true_pt, atol=0.01)


# -----------------------------------------------------------------------
# Marker corner triangulation
# -----------------------------------------------------------------------

class TestTriangulateMarkerCorners:
    def test_triangulate_square_marker(self, stereo_cameras):
        """Triangulate a 2cm square marker at Z=1m."""
        s = 0.01  # half-side = 10mm
        corners_3d_true = np.array([
            [-s,  s, 1.0],  # TL
            [ s,  s, 1.0],  # TR
            [ s, -s, 1.0],  # BR
            [-s, -s, 1.0],  # BL
        ])
        cam0 = stereo_cameras["cam_0"]
        cam1 = stereo_cameras["cam_1"]

        # Create synthetic detections
        det0_corners = np.array([cam0.project(c.reshape(1, 3))[0] for c in corners_3d_true])
        det1_corners = np.array([cam1.project(c.reshape(1, 3))[0] for c in corners_3d_true])

        dets = [
            tg.MarkerDetection(marker_id=10, corners_px=det0_corners, camera_key="cam_0"),
            tg.MarkerDetection(marker_id=10, corners_px=det1_corners, camera_key="cam_1"),
        ]

        corners_3d, err = tg.triangulate_marker_corners(
            dets, stereo_cameras, min_cameras=2, max_reproj=20.0,
        )
        assert corners_3d is not None
        assert corners_3d.shape == (4, 3)
        np.testing.assert_allclose(corners_3d, corners_3d_true, atol=0.02)
        assert err < 2.0

    def test_insufficient_detections(self, stereo_cameras):
        det = tg.MarkerDetection(
            marker_id=10,
            corners_px=np.zeros((4, 2)),
            camera_key="cam_0",
        )
        corners, err = tg.triangulate_marker_corners(
            [det], stereo_cameras, min_cameras=2,
        )
        assert corners is None
        assert err == float("inf")


# -----------------------------------------------------------------------
# Glasses pose estimation
# -----------------------------------------------------------------------

class TestGlassesPoseEstimation:
    def _make_square_corners(self, cx: float, cz: float, s: float = 0.01) -> np.ndarray:
        """Create 4 corners for a marker centred at (cx, 0, cz)."""
        return np.array([
            [cx - s,  s, cz],  # TL
            [cx + s,  s, cz],  # TR
            [cx + s, -s, cz],  # BR
            [cx - s, -s, cz],  # BL
        ])

    def test_two_markers_gives_pose(self):
        left_corners = self._make_square_corners(-0.04, 1.0)
        right_corners = self._make_square_corners(0.04, 1.0)

        pose = tg.estimate_glasses_pose_from_markers(
            left_corners_3d=left_corners,
            right_corners_3d=right_corners,
            left_offset_mm=[0, 0, 0],
            right_offset_mm=[0, 0, 0],
            marker_size_m=0.02,
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="test",
            reproj_error=1.0,
            cameras_used=3,
        )
        assert pose is not None
        assert pose.markers_detected == 2
        assert pose.glasses_id == "test"
        assert pose.position.shape == (3,)
        assert pose.quaternion.shape == (4,)

    def test_one_marker_gives_pose(self):
        left_corners = self._make_square_corners(-0.04, 1.0)

        pose = tg.estimate_glasses_pose_from_markers(
            left_corners_3d=left_corners,
            right_corners_3d=None,
            left_offset_mm=[0, 0, 0],
            right_offset_mm=[0, 0, 0],
            marker_size_m=0.02,
            frame_idx=5,
            frame_time_s=0.167,
            glasses_id="p1",
            reproj_error=2.5,
            cameras_used=2,
        )
        assert pose is not None
        assert pose.markers_detected == 1
        assert pose.frame_idx == 5

    def test_no_markers_returns_none(self):
        pose = tg.estimate_glasses_pose_from_markers(
            left_corners_3d=None,
            right_corners_3d=None,
            left_offset_mm=[0, 0, 0],
            right_offset_mm=[0, 0, 0],
            marker_size_m=0.02,
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="test",
            reproj_error=0.0,
            cameras_used=0,
        )
        assert pose is None

    def test_two_marker_centre_between_eyes(self):
        """Glasses centre should be approximately the midpoint when offsets are zero."""
        left_corners = self._make_square_corners(-0.05, 1.0)
        right_corners = self._make_square_corners(0.05, 1.0)

        pose = tg.estimate_glasses_pose_from_markers(
            left_corners_3d=left_corners,
            right_corners_3d=right_corners,
            left_offset_mm=[0, 0, 0],
            right_offset_mm=[0, 0, 0],
            marker_size_m=0.02,
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="test",
            reproj_error=1.0,
            cameras_used=3,
        )
        assert pose is not None
        # Centre should be near x=0 (midpoint of -0.05 and 0.05)
        assert abs(pose.position[0]) < 0.01

    def test_quaternion_is_unit(self):
        left_corners = self._make_square_corners(-0.04, 1.0)
        right_corners = self._make_square_corners(0.04, 1.0)

        pose = tg.estimate_glasses_pose_from_markers(
            left_corners_3d=left_corners,
            right_corners_3d=right_corners,
            left_offset_mm=[0, 0, 0],
            right_offset_mm=[0, 0, 0],
            marker_size_m=0.02,
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="test",
            reproj_error=1.0,
            cameras_used=3,
        )
        assert pose is not None
        assert np.linalg.norm(pose.quaternion) == pytest.approx(1.0, abs=1e-6)


# -----------------------------------------------------------------------
# Gaze transformation
# -----------------------------------------------------------------------

class TestGazeTransformation:
    def _make_pose_at_origin(self) -> tg.GlassesPose:
        return tg.GlassesPose(
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="test",
            position=np.array([0.0, 0.0, 0.5]),  # 0.5m above z=0 plane
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),  # identity rotation
            reproj_error=1.0,
            cameras_used=3,
            markers_detected=2,
        )

    def test_gaze_hits_plane(self):
        """Gaze directed downward from z=0.5 should intersect z=0 plane."""
        pose = self._make_pose_at_origin()
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=np.array([0.0, 0.0, -1.0]),  # looking straight down
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose, plane_z=0.0)
        assert isinstance(result, tg.WorldGaze)
        assert result.world_z == pytest.approx(0.0, abs=0.01)

    def test_gaze_with_left_eye(self):
        pose = self._make_pose_at_origin()
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=None,
            left_origin=np.array([0.0, 0.0, 0.0]),
            left_direction=np.array([0.0, 0.0, -1.0]),
            right_origin=None,
            right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose, plane_z=0.0)
        assert not math.isnan(result.world_x)
        assert result.world_z == pytest.approx(0.0, abs=0.01)

    def test_no_gaze_data_returns_nan(self):
        pose = self._make_pose_at_origin()
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=None,
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose, plane_z=0.0)
        assert math.isnan(result.world_x)
        assert result.confidence == 0.0

    def test_confidence_based_on_markers(self):
        """Confidence should be higher with 2 markers and low reproj error."""
        pose_2m = self._make_pose_at_origin()
        pose_2m.markers_detected = 2
        pose_2m.reproj_error = 0.0
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=np.array([0.0, 0.0, -1.0]),
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose_2m)
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_lower_with_one_marker(self):
        pose_1m = self._make_pose_at_origin()
        pose_1m.markers_detected = 1
        pose_1m.reproj_error = 0.0
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=np.array([0.0, 0.0, -1.0]),
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose_1m)
        assert result.confidence == pytest.approx(0.5)

    def test_gaze_forward_at_angle(self):
        """Gaze at 45° should hit the plane at x offset from origin."""
        pose = self._make_pose_at_origin()
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=np.array([1.0, 0.0, -1.0]),  # 45° to the right and down
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose, plane_z=0.0)
        # At 45° from height 0.5m, hits z=0 at x=0.5m
        assert result.world_x == pytest.approx(0.5, abs=0.1)
        assert result.world_z == pytest.approx(0.0, abs=0.01)

    def test_looking_away_from_plane(self):
        """Looking upward (away from z=0 plane) should not produce nan but use fallback."""
        pose = self._make_pose_at_origin()
        gaze = tg.GazeSample(
            sample_time_s=0.0,
            gaze3d=np.array([0.0, 0.0, 1.0]),  # looking up
            left_origin=None, right_origin=None,
            left_direction=None, right_direction=None,
        )
        result = tg.transform_gaze_to_world(gaze, pose, plane_z=0.0)
        # Should not be nan — fallback projects 1m ahead
        assert not math.isnan(result.world_x)


# -----------------------------------------------------------------------
# Gaze sample loading
# -----------------------------------------------------------------------

class TestGazeSampleLoading:
    def test_load_gaze_samples(self, tmp_path: Path):
        gaze_file = tmp_path / "gaze.ndjson"
        samples = [
            {"timestamp_s": 0.0, "gaze3d": [0.1, 0.2, 1.0],
             "left_eye": {"gaze_origin": [0, 0, 0], "gaze_direction": [0, 0, 1]},
             "right_eye": {}},
            {"timestamp_s": 0.033, "gaze3d": [0.15, 0.25, 1.1],
             "left_eye": {}, "right_eye": {}},
        ]
        gaze_file.write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n",
            encoding="utf-8",
        )
        loaded = tg.load_gaze_samples(gaze_file)
        assert len(loaded) == 2
        assert loaded[0].sample_time_s == 0.0
        assert loaded[0].gaze3d is not None
        np.testing.assert_allclose(loaded[0].gaze3d, [0.1, 0.2, 1.0])
        assert loaded[0].left_origin is not None
        assert loaded[1].left_origin is None

    def test_load_gaze_with_ticks(self, tmp_path: Path):
        gaze_file = tmp_path / "gaze_ticks.ndjson"
        sample = {"timestamp_ticks": 1000000, "ticks_per_second": 1000000}
        gaze_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        loaded = tg.load_gaze_samples(gaze_file)
        assert len(loaded) == 1
        assert loaded[0].sample_time_s == pytest.approx(1.0)

    def test_load_empty_file(self, tmp_path: Path):
        gaze_file = tmp_path / "empty.ndjson"
        gaze_file.write_text("", encoding="utf-8")
        loaded = tg.load_gaze_samples(gaze_file)
        assert loaded == []


# -----------------------------------------------------------------------
# Video discovery + camera mapping
# -----------------------------------------------------------------------

class TestVideoDiscovery:
    def test_discover_video_files(self, tmp_path: Path):
        vdir = tmp_path / "video"
        vdir.mkdir()
        (vdir / "cam_0_video.mp4").write_bytes(b"")
        (vdir / "cam_1_video.mkv").write_bytes(b"")
        (vdir / "sync_grid.mp4").write_bytes(b"")
        (vdir / "readme.txt").write_bytes(b"")

        found = tg.discover_video_files(vdir)
        labels = [label for label, _ in found]
        assert "cam_0" in labels
        assert "cam_1" in labels
        assert len(found) == 2  # excludes sync_grid and txt

    def test_map_videos_to_cameras(self, stereo_cameras):
        videos = [("cam_0_video", Path("cam_0_video.mp4")),
                  ("cam_1_video", Path("cam_1_video.mp4"))]
        mapping = tg.map_videos_to_cameras(videos, stereo_cameras)
        assert "cam_0" in mapping
        assert "cam_1" in mapping

    def test_map_no_match(self, stereo_cameras):
        videos = [("webcam_front", Path("webcam_front.mp4"))]
        mapping = tg.map_videos_to_cameras(videos, stereo_cameras)
        assert len(mapping) == 0

    def test_map_videos_to_cameras_mixed_separators(self):
        # Real capture filenames often use dashes while calibration names use underscores.
        cameras = {
            "cam_0": tg.CameraCalibration(
                "cam_0",
                {
                    "name": "jabra_panacast_20_cam1_vid",
                    "size": [1920, 1080],
                    "matrix": [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
                },
            ),
            "cam_1": tg.CameraCalibration(
                "cam_1",
                {
                    "name": "jabra_panacast_20_cam2_vid",
                    "size": [1920, 1080],
                    "matrix": [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]],
                },
            ),
        }
        videos = [
            (
                "sub99_taskt0_acqavjabrapanacast20cam1vidjabrapanacast20cam1vid",
                Path("sub99_taskt0_acq-av-jabra-panacast-20-cam1-vid_video.mkv"),
            ),
            (
                "sub99_taskt0_acqavjabrapanacast20cam2vidjabrapanacast20cam2vid",
                Path("sub99_taskt0_acq-av-jabra-panacast-20-cam2-vid_video.mkv"),
            ),
        ]
        mapping = tg.map_videos_to_cameras(videos, cameras)
        assert "cam_0" in mapping
        assert "cam_1" in mapping


# -----------------------------------------------------------------------
# load_all_cameras (requires TOML)
# -----------------------------------------------------------------------

class TestLoadAllCameras:
    def test_load_synthetic_toml(self, tmp_path: Path):
        """Write a small TOML by hand, load it, verify camera objects."""
        # The toml library on Python 3.10 requires homogeneous arrays,
        # so all numbers must be floats (matching real anipose output).
        toml_text = """\
[cam_0]
name = "cam_0"
size = [1920, 1080]
matrix = [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.0, 0.0, 0.0, 0.0, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [0.0, 0.0, 0.0]

[cam_1]
name = "cam_1"
size = [1920, 1080]
matrix = [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]]
distortions = [0.0, 0.0, 0.0, 0.0, 0.0]
rotation = [0.0, 0.0, 0.0]
translation = [0.5, 0.0, 0.0]

[metadata]
note = "test"
"""
        toml_path = tmp_path / "test_calib.toml"
        toml_path.write_text(toml_text, encoding="utf-8")

        cameras = tg.load_all_cameras(toml_path)
        assert "cam_0" in cameras
        assert "cam_1" in cameras
        assert len(cameras) == 2  # metadata not loaded as camera


# -----------------------------------------------------------------------
# WorldGaze dataclass
# -----------------------------------------------------------------------

class TestWorldGaze:
    def test_creation(self):
        wg = tg.WorldGaze(
            frame_idx=10,
            frame_time_s=0.333,
            glasses_id="tobii_p1",
            world_x=0.1, world_y=0.2, world_z=0.0,
            confidence=0.95,
        )
        assert wg.frame_idx == 10
        assert wg.confidence == pytest.approx(0.95)


# -----------------------------------------------------------------------
# GlassesPose dataclass
# -----------------------------------------------------------------------

class TestGlassesPose:
    def test_creation(self):
        gp = tg.GlassesPose(
            frame_idx=0,
            frame_time_s=0.0,
            glasses_id="tobii_p1",
            position=np.zeros(3),
            quaternion=np.array([0, 0, 0, 1.0]),
            reproj_error=1.5,
            cameras_used=3,
            markers_detected=2,
        )
        assert gp.markers_detected == 2
        assert gp.reproj_error == 1.5


# -----------------------------------------------------------------------
# Integration: example config from repo
# -----------------------------------------------------------------------

class TestRepoConfig:
    """Load the actual example config from the repo if available."""

    @pytest.fixture()
    def example_cfg_path(self) -> Path:
        p = Path(__file__).resolve().parent.parent / "configs" / "tobii_multicam_glasses_tracker.example.yaml"
        if not p.exists():
            pytest.skip("Example config YAML not found")
        return p

    def test_example_config_loads(self, example_cfg_path: Path):
        cfg = tg.load_tracker_config(example_cfg_path)
        assert cfg.aruco_dictionary == "DICT_4X4_50"
        assert len(cfg.glasses) == 4
        assert len(cfg.table_markers) == 6

    def test_example_config_marker_ids(self, example_cfg_path: Path):
        cfg = tg.load_tracker_config(example_cfg_path)
        marker_ids = set()
        for g in cfg.glasses:
            marker_ids.add(g.left_marker_id)
            marker_ids.add(g.right_marker_id)
        # Glasses markers should be 10-17
        assert marker_ids == {10, 11, 12, 13, 14, 15, 16, 17}

    def test_example_config_table_marker_ids(self, example_cfg_path: Path):
        cfg = tg.load_tracker_config(example_cfg_path)
        ids = {tm.marker_id for tm in cfg.table_markers}
        assert ids == {0, 1, 2, 3, 4, 5}

    def test_example_config_all_corners_on_z0(self, example_cfg_path: Path):
        cfg = tg.load_tracker_config(example_cfg_path)
        for tm in cfg.table_markers:
            assert tm.corners_m.shape == (4, 3)
            np.testing.assert_allclose(tm.corners_m[:, 2], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
