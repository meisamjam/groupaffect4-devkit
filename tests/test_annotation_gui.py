"""Tests for session_loader and MasterClock (no Qt/GUI imports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.annotation_gui.io import annotations as anns
from tools.annotation_gui.io.sync import (
    infer_sync_offsets_from_lsl_sync,
    infer_sync_offsets_from_source_ffmpeg_lsl,
    infer_tobii_video_offsets_from_et,
    load_task_start_lsl,
)
from tools.annotation_gui.media.clock import MasterClock
from tools.annotation_gui.media.video_engine import (
    _box_iou,
    _decode_res10_dnn_detections,
    _flip_frame_180,
    _is_cam5_feed,
    _is_p1_p4_feed,
    _is_tobii_feed,
    _offset_boxes,
    _select_face_detection_roi,
    _should_flip_180,
    _stabilize_face_boxes,
)
from tools.annotation_gui.session_loader import discover_sessions, discover_task_runs


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_sessions(tmp_path: Path) -> None:
    root = tmp_path / "bids"
    for sub in ("sub-01", "sub-02"):
        for ses in ("ses-20260311_grp-06_run01", "ses-20260311_grp-07_run01"):
            (root / sub / ses).mkdir(parents=True)
    # A non-session sibling should be ignored.
    (root / "sub-01" / "derivatives").mkdir()

    sessions = discover_sessions(root)
    assert len(sessions) == 4
    assert all(s.name.startswith("ses-") for s in sessions)


def test_discover_task_runs_groups_by_task(tmp_path: Path) -> None:
    ses = tmp_path / "bids" / "sub-01" / "ses-20260311_grp-06_run01"
    prefix = "sub-01_ses-20260311_grp-06_run01"
    # Two tasks, each with 2 audio + 2 video acquisitions.
    for task in ("T1", "T2"):
        for acq in ("dpa-close-talk", "dpa-room"):
            _touch(ses / "audio" / f"{prefix}_task-{task}_run-01_acq-{acq}_audio.wav")
        for acq in ("jabra-panacast-20-cam1", "jabra-panacast-20-cam2"):
            _touch(ses / "video" / f"{prefix}_task-{task}_run-01_acq-{acq}_video.mkv")
    _touch(ses / "beh" / f"{prefix}_task-T1_run-01_events.tsv")
    _touch(ses / "annot" / f"{prefix}_task_run_windows.tsv")
    _touch(ses / "annot" / f"{prefix}_participant_signal_map.tsv")

    runs = discover_task_runs(ses)
    assert [(r.task, r.run) for r in runs] == [("T1", "01"), ("T2", "01")]

    t1 = runs[0]
    assert len(t1.audio) == 2
    assert {m.acq for m in t1.audio} == {"dpa-close-talk", "dpa-room"}
    assert len(t1.video) == 2
    assert t1.events_tsv is not None and t1.events_tsv.name.endswith("_events.tsv")
    assert t1.task_windows_tsv is not None
    assert t1.participant_map_tsv is not None

    t2 = runs[1]
    assert t2.events_tsv is None  # only T1 got an events.tsv above


def test_discover_task_runs_real_seed_layout(tmp_path: Path) -> None:
    """Match the filenames used in affectai-data-processing-seed/data/sub-01/*."""
    ses = tmp_path / "data" / "sub-01" / "ses-20260318_grp-13_run01"
    prefix = "sub-01_ses-20260318_grp-13_run01"
    # _aud.wav (not _audio.wav), 4 close-talk mics
    for mic in ("mic9", "mic10", "mic11", "mic12"):
        _touch(ses / "audio" / f"{prefix}_task-T1_run-01_acq-dpa_{mic}_aud.wav")
    # .mp4 videos with long composite acq labels
    for cam in ("cam1", "cam2", "cam3", "cam4", "cam5", "cam6"):
        _touch(
            ses
            / "video"
            / f"{prefix}_task-T1_run-01_acq-av-jabra-panacast-20-{cam}-vid-jabra-panacast-20-{cam}-vid_video.mp4"
        )
    # Tobii scene video can also live under et/.
    _touch(ses / "et" / f"{prefix}_task-T1_run-01_acq-P1_tobii.mp4")
    _touch(ses / "beh" / f"{prefix}_task-T1_run-01_events.tsv")
    _touch(ses / "annot" / f"{prefix}_task-T0T1T2T3T4_task_run_windows.tsv")
    _touch(ses / "annot" / f"{prefix}_task-T0T1T2T3T4_participant_signal_map.tsv")
    # Physio as gzipped TSV
    _touch(ses / "physio" / f"{prefix}_task-T1_run-01_acq-P1_emotibit.tsv.gz")
    # Eye-tracking as NDJSON per participant per task
    _touch(ses / "et" / "P1_task-T1_gaze.ndjson")

    runs = discover_task_runs(ses)
    assert len(runs) == 1
    t1 = runs[0]
    assert len(t1.audio) == 4
    assert len(t1.video) == 7
    assert t1.events_tsv is not None
    assert t1.task_windows_tsv is not None
    assert t1.participant_map_tsv is not None
    assert len(t1.physio_tsvs) == 1 and t1.physio_tsvs[0].suffixes == [".tsv", ".gz"]
    assert len(t1.gaze_tsvs) == 1 and t1.gaze_tsvs[0].name == "P1_task-T1_gaze.ndjson"


def test_discover_task_runs_includes_tobii_videos_by_default(tmp_path: Path) -> None:
    ses = tmp_path / "data" / "sub-01" / "ses-20260318_grp-13_run01"
    prefix = "sub-01_ses-20260318_grp-13_run01"
    for mic in ("mic9", "mic10", "mic11", "mic12"):
        _touch(ses / "audio" / f"{prefix}_task-T1_run-01_acq-dpa_{mic}_aud.wav")
    for cam in ("cam1", "cam2", "cam3", "cam4", "cam5", "cam6"):
        _touch(
            ses
            / "video"
            / f"{prefix}_task-T1_run-01_acq-av-jabra-panacast-20-{cam}-vid-jabra-panacast-20-{cam}-vid_video.mp4"
        )
    _touch(ses / "et" / f"{prefix}_task-T1_run-01_acq-P1_tobii.mp4")

    runs = discover_task_runs(ses)
    assert len(runs) == 1
    assert len(runs[0].video) == 7


def test_discover_task_runs_falls_back_to_session_root_participant_map(tmp_path: Path) -> None:
    ses = tmp_path / "data" / "sub-01" / "ses-x"
    (ses / "audio").mkdir(parents=True)
    _touch(ses / "audio" / "sub-01_ses-x_task-T1_run-01_acq-a_aud.wav")
    _touch(ses / "participant_map.tsv")
    runs = discover_task_runs(ses)
    assert runs[0].participant_map_tsv is not None
    assert runs[0].participant_map_tsv.name == "participant_map.tsv"


def test_infer_sync_offsets_from_lsl_sync_estimates_relative_offsets(tmp_path: Path) -> None:
    sync_tsv = tmp_path / "task_sync.tsv"
    sync_tsv.write_text(
        "\n".join(
            [
                "lsl_time\tstream_name\tstream_type\tvalue_0",
                "10.0\tffmpeg_progress_jabra_panacast_20_cam1_vid\tffmpeg_progress\t100.20",
                "10.0\tffmpeg_progress_jabra_panacast_20_cam2_vid\tffmpeg_progress\t100.00",
                "10.0\tffmpeg_progress_dpa_mic9_aud\tffmpeg_progress\t99.90",
            ]
        ),
        encoding="utf-8",
    )
    inferred = infer_sync_offsets_from_lsl_sync(
        sync_tsv,
        video_labels=[
            "av-jabra-panacast-20-cam1-vid-jabra-panacast-20-cam1-vid",
            "av-jabra-panacast-20-cam2-vid-jabra-panacast-20-cam2-vid",
        ],
        audio_labels=["dpa_mic9_aud"],
    )
    assert inferred.video == {
        "av-jabra-panacast-20-cam1-vid-jabra-panacast-20-cam1-vid": -0.2,
        "av-jabra-panacast-20-cam2-vid-jabra-panacast-20-cam2-vid": 0.0,
    }
    assert inferred.audio == {"dpa_mic9_aud": 0.1}


def test_load_task_start_lsl_matches_task_and_run(tmp_path: Path) -> None:
    path = tmp_path / "task_run_windows.tsv"
    path.write_text(
        "\n".join(
            [
                "task\trun\tstart_lsl\tend_lsl",
                "T0\t01\t100.0\t110.0",
                "T1\t01\t200.5\t210.0",
            ]
        ),
        encoding="utf-8",
    )
    assert load_task_start_lsl(path, "T1", "1") == 200.5
    assert load_task_start_lsl(path, "t1", "01") == 200.5
    assert load_task_start_lsl(path, "T2", "01") is None


def test_infer_tobii_video_offsets_from_et_uses_participant_lsl_start(tmp_path: Path) -> None:
    et = tmp_path / "et"
    et.mkdir()
    (et / "sub-01_ses-x_task-T1_run-01_acq-P1_tobii.tsv").write_text(
        "lsl_time\tstream_name\tstream_type\tvalue_0\n"
        "100.2\tTobii_P1_stream\tEyeTracking\t0.4\n",
        encoding="utf-8",
    )
    (et / "sub-01_ses-x_task-T1_run-01_acq-P2_tobii.tsv").write_text(
        "lsl_time\tstream_name\tstream_type\tvalue_0\n"
        "100.0\tTobii_P2_stream\tEyeTracking\t0.5\n",
        encoding="utf-8",
    )
    offsets = infer_tobii_video_offsets_from_et(
        et,
        task="T1",
        run="01",
        video_labels=["P1_tobii", "P2_tobii", "av-jabra-panacast-20-cam1-vid"],
        lsl_ref_time=100.0,
    )
    assert offsets["P1_tobii"] == 0.1
    assert offsets["P2_tobii"] == -0.1
    assert "av-jabra-panacast-20-cam1-vid" not in offsets


def test_infer_sync_offsets_from_source_ffmpeg_lsl(tmp_path: Path) -> None:
    source = tmp_path / "sub-01" / "ses-x" / "sourcedata" / "grp-x" / "lsl"
    source.mkdir(parents=True)
    (source / "ffmpeg_progress_jabra_panacast_20_cam1_vid.jsonl").write_text(
        '{"stream_time": 10.0, "values": [100.2]}\n',
        encoding="utf-8",
    )
    (source / "ffmpeg_progress_jabra_panacast_20_cam2_vid.jsonl").write_text(
        '{"stream_time": 10.0, "values": [100.0]}\n',
        encoding="utf-8",
    )
    (source / "ffmpeg_progress_dpa_mic9_aud.jsonl").write_text(
        '{"stream_time": 10.0, "values": [99.9]}\n',
        encoding="utf-8",
    )
    inferred = infer_sync_offsets_from_source_ffmpeg_lsl(
        tmp_path / "sub-01" / "ses-x",
        video_labels=[
            "av-jabra-panacast-20-cam1-vid-jabra-panacast-20-cam1-vid",
            "av-jabra-panacast-20-cam2-vid-jabra-panacast-20-cam2-vid",
        ],
        audio_labels=["dpa_mic9_aud"],
    )
    assert inferred.video["av-jabra-panacast-20-cam1-vid-jabra-panacast-20-cam1-vid"] == -0.2
    assert inferred.video["av-jabra-panacast-20-cam2-vid-jabra-panacast-20-cam2-vid"] == 0.0
    assert inferred.audio["dpa_mic9_aud"] == 0.1


def test_annotation_gui_flips_cam1_through_cam4_labels() -> None:
    assert _should_flip_180("cam1")
    assert _should_flip_180("cam_2")
    assert _should_flip_180("camera3")
    assert _should_flip_180("av-jabra-panacast-20-cam4-vid")
    assert not _should_flip_180("cam5")
    assert not _should_flip_180("overview")


def test_annotation_gui_detects_cam5_label_for_default_selection() -> None:
    assert _is_cam5_feed("cam5")
    assert _is_cam5_feed("camera-5")
    assert _is_cam5_feed("av-jabra-panacast-20-cam5-vid")
    assert not _is_cam5_feed("cam4")
    assert not _is_cam5_feed("overview")


def test_annotation_gui_detects_tobii_video_labels() -> None:
    assert _is_tobii_feed("P1_tobii")
    assert _is_tobii_feed("sub-01_task-T1_acq-P2_tobii")
    assert not _is_tobii_feed("av-jabra-panacast-20-cam5-vid")


def test_annotation_gui_detects_p1_to_p4_participant_feeds() -> None:
    assert _is_p1_p4_feed("P1_tobii")
    assert _is_p1_p4_feed("sub-01_task-T1_run-01_acq-P4_tobii")
    assert not _is_p1_p4_feed("P5_tobii")
    assert not _is_p1_p4_feed("av-jabra-panacast-20-cam1-vid")


def test_flip_frame_180_rotates_array() -> None:
    frame = np.array(
        [
            [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[4, 0, 0], [5, 0, 0], [6, 0, 0]],
        ],
        dtype=np.uint8,
    )

    flipped = _flip_frame_180(frame)

    assert flipped.tolist() == [
        [[6, 0, 0], [5, 0, 0], [4, 0, 0]],
        [[3, 0, 0], [2, 0, 0], [1, 0, 0]],
    ]


def test_face_box_iou_is_positive_for_overlapping_boxes() -> None:
    iou = _box_iou((10, 10, 20, 20), (15, 15, 20, 20))
    assert 0.0 < iou < 1.0


def test_stabilize_face_boxes_smooths_matching_boxes() -> None:
    prev = [(100, 100, 50, 50)]
    cur = [(110, 110, 52, 52)]
    stabilized = _stabilize_face_boxes(prev, cur, smooth_alpha=0.5, min_iou=0.05)
    assert len(stabilized) == 1
    sx, sy, sw, sh = stabilized[0]
    assert sx == 105
    assert sy == 105
    assert sw == 51
    assert sh == 51


def test_stabilize_face_boxes_keeps_unmatched_new_box() -> None:
    prev = [(10, 10, 20, 20)]
    cur = [(200, 200, 30, 30)]
    stabilized = _stabilize_face_boxes(prev, cur, smooth_alpha=0.8, min_iou=0.2)
    assert stabilized == [(200, 200, 30, 30)]


def test_decode_res10_dnn_detections_filters_by_confidence() -> None:
    detections = np.zeros((1, 1, 2, 7), dtype=np.float32)
    detections[0, 0, 0] = [0.0, 1.0, 0.92, 0.1, 0.2, 0.5, 0.7]
    detections[0, 0, 1] = [0.0, 1.0, 0.25, 0.2, 0.2, 0.6, 0.6]

    boxes = _decode_res10_dnn_detections(detections, width=1000, height=500, confidence_threshold=0.5)
    assert boxes == [(100, 100, 400, 249)]


def test_decode_res10_dnn_detections_rejects_invalid_shape() -> None:
    detections = np.zeros((1, 2, 3), dtype=np.float32)
    boxes = _decode_res10_dnn_detections(detections, width=640, height=480, confidence_threshold=0.5)
    assert boxes == []


def test_select_face_detection_roi_uses_top_half_when_restricted() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    roi, ox, oy = _select_face_detection_roi(frame, restrict_to_upper_quadrants=True)
    assert roi.shape == (100, 300, 3)
    assert (ox, oy) == (0, 0)


def test_select_face_detection_roi_uses_manual_roi_when_present() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    roi, ox, oy = _select_face_detection_roi(
        frame,
        restrict_to_upper_quadrants=True,
        manual_roi=(20, 30, 80, 70),
    )
    assert roi.shape == (70, 80, 3)
    assert (ox, oy) == (20, 30)


def test_offset_boxes_applies_offset() -> None:
    boxes = [(10, 20, 30, 40), (5, 6, 7, 8)]
    shifted = _offset_boxes(boxes, offset_x=3, offset_y=11)
    assert shifted == [(13, 31, 30, 40), (8, 17, 7, 8)]


def test_discover_task_runs_missing_dirs(tmp_path: Path) -> None:
    ses = tmp_path / "bids" / "sub-01" / "ses-empty"
    ses.mkdir(parents=True)
    assert discover_task_runs(ses) == []


def test_default_annotations_path(tmp_path: Path) -> None:
    ses = tmp_path / "bids" / "sub-01" / "ses-x"
    (ses / "audio").mkdir(parents=True)
    _touch(ses / "audio" / "sub-01_ses-x_task-T1_run-01_acq-a_audio.wav")
    runs = discover_task_runs(ses)
    assert runs[0].default_annotations_path() == (
        ses / "annot" / "sub-01_ses-x_task-T1_run-01_annotations.json"
    )


def test_clock_play_pause_position_advances() -> None:
    c = MasterClock(duration=10.0)
    t = [0.0]
    c._now = lambda: t[0]  # type: ignore[assignment]

    assert c.position() == 0.0
    c.play()
    t[0] += 1.5
    assert abs(c.position() - 1.5) < 1e-9

    c.pause()
    t[0] += 5.0
    assert abs(c.position() - 1.5) < 1e-9  # paused, no advance


def test_clock_seek_and_rate() -> None:
    c = MasterClock(duration=10.0)
    t = [0.0]
    c._now = lambda: t[0]  # type: ignore[assignment]

    c.seek(4.0)
    assert c.position() == 4.0

    c.set_rate(2.0)
    c.play()
    t[0] += 1.0
    assert abs(c.position() - 6.0) < 1e-9  # 4 + 2.0*1s


def test_clock_clamps_to_duration() -> None:
    c = MasterClock(duration=5.0)
    t = [0.0]
    c._now = lambda: t[0]  # type: ignore[assignment]
    c.play()
    t[0] += 10.0
    assert c.position() == 5.0
    assert not c.is_playing()


def test_clock_seek_negative_and_beyond_are_clamped() -> None:
    c = MasterClock(duration=3.0)
    c.seek(-5.0)
    assert c.position() == 0.0
    c.seek(99.0)
    assert c.position() == 3.0


def test_annotations_empty_load_when_missing(tmp_path: Path) -> None:
    doc = anns.load(tmp_path / "nope.json")
    assert doc.tiers == []
    assert doc.entries == []
    assert doc.schema_version == anns.SCHEMA_VERSION


def test_annotations_add_remove_tier_cascades_entries() -> None:
    doc = anns.AnnotationDoc()
    doc.add_tier(anns.Tier(id="P1.backchannel", participant="P1"))
    doc.add_tier(anns.Tier(id="P1.nod", participant="P1", kind="point"))
    doc.add_entry(anns.Entry(tier="P1.backchannel", start=1.0, end=2.0, label="mhm"))
    doc.add_entry(anns.Entry(tier="P1.nod", start=3.0, end=3.0))
    assert len(doc.entries_in_tier("P1.backchannel")) == 1
    doc.remove_tier("P1.backchannel")
    # Entries on removed tier are dropped too.
    assert [e.tier for e in doc.entries] == ["P1.nod"]


def test_annotations_add_entry_rejects_unknown_tier() -> None:
    doc = anns.AnnotationDoc()
    try:
        doc.add_entry(anns.Entry(tier="ghost", start=0.0, end=1.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown tier")


def test_annotations_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "annotations.json"
    doc = anns.AnnotationDoc(
        sub="sub-01", ses="ses-x", task="T1", run="01",
        participants={"P1": "sub-01", "P2": "sub-02"},
    )
    doc.add_tier(anns.Tier(id="P1.backchannel", participant="P1"))
    doc.add_entry(anns.Entry(tier="P1.backchannel", start=12.34, end=12.91, label="mhm"))
    anns.save(path, doc)

    loaded = anns.load(path)
    assert loaded.sub == "sub-01"
    assert loaded.participants == {"P1": "sub-01", "P2": "sub-02"}
    assert len(loaded.tiers) == 1 and loaded.tiers[0].id == "P1.backchannel"
    assert len(loaded.entries) == 1
    e = loaded.entries[0]
    assert (e.start, e.end, e.label) == (12.34, 12.91, "mhm")


def test_annotations_tolerate_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    # Minimal file — only tiers + entries, no header fields.
    path.write_text(
        '{"tiers":[{"id":"t"}],"entries":[{"tier":"t","start":1.0}]}',
        encoding="utf-8",
    )
    doc = anns.load(path)
    assert doc.tiers[0].kind == "span"  # default
    assert doc.entries[0].end == 1.0    # mirrored from start when missing
