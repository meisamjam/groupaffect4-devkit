import csv
import gzip
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(path: str, name: str):
    spec = spec_from_file_location(name, Path(path))
    mod = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_compute_task_windows_basic():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs")
    rows = [
        {"task": "T0", "wall_clock": "1000", "lsl_clock": "10"},
        {"task": "T0", "wall_clock": "1010", "lsl_clock": "20"},
        {"task": "T1", "wall_clock": "1100", "lsl_clock": "110"},
        {"task": "T2", "wall_clock": "1200", "lsl_clock": "210"},
        {"task": "T3", "wall_clock": "1300", "lsl_clock": "310"},
        {"task": "T4", "wall_clock": "1400", "lsl_clock": "410"},
        {"task": "T4", "wall_clock": "1500", "lsl_clock": "510"},
    ]

    windows, offset = mod._compute_task_windows(rows)

    assert offset == 990.0
    assert [w["task"] for w in windows] == ["T0", "T1", "T2", "T3", "T4"]
    assert windows[0]["start_wall_clock"] == 1000.0
    assert windows[0]["end_wall_clock"] == 1100.0
    assert windows[-1]["start_wall_clock"] == 1400.0
    assert windows[-1]["end_wall_clock"] == 1500.0
    assert windows[1]["start_lsl"] == 110.0
    assert windows[1]["end_lsl"] == 210.0


def test_compute_task_windows_phase_boundaries():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_phase")
    rows = [
        {"task": "T0", "event_type": "push_content", "phase": "study_introduction", "wall_clock": "1000", "lsl_clock": "10"},
        {"task": "T0", "event_type": "push_content", "phase": "finish", "wall_clock": "1090", "lsl_clock": "100"},
        {"task": "T1", "event_type": "push_content", "phase": "tobii_calibration", "wall_clock": "1100", "lsl_clock": "110"},
        {"task": "T1", "event_type": "push_content", "phase": "finish", "wall_clock": "1190", "lsl_clock": "200"},
        {"task": "T2", "event_type": "tobii_calibration", "phase": "tobii_calibration", "wall_clock": "1200", "lsl_clock": "210"},
        {"task": "T2", "event_type": "task_end", "phase": "finish", "wall_clock": "1290", "lsl_clock": "300"},
        {"task": "T3", "event_type": "push_content", "phase": "tobii_calibration", "wall_clock": "1300", "lsl_clock": "310"},
        {"task": "T3", "event_type": "push_content", "phase": "finish", "wall_clock": "1390", "lsl_clock": "400"},
        {"task": "T4", "event_type": "push_content", "phase": "tobii_calibration", "wall_clock": "1400", "lsl_clock": "410"},
        {"task": "T4", "event_type": "push_content", "phase": "finish", "wall_clock": "1490", "lsl_clock": "500"},
    ]

    windows, _ = mod._compute_task_windows(rows)
    bounds = {w["task"]: (w["start_wall_clock"], w["end_wall_clock"]) for w in windows}

    assert bounds["T0"] == (1000.0, 1090.0)
    assert bounds["T1"] == (1100.0, 1190.0)
    assert bounds["T2"] == (1200.0, 1290.0)
    assert bounds["T3"] == (1300.0, 1390.0)
    assert bounds["T4"] == (1400.0, 1490.0)


def test_compute_break_windows_pre_between_post():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_breaks")
    rows = [
        {"task": "T0", "event_type": "push_content", "phase": "study_introduction", "wall_clock": "900", "lsl_clock": "10"},
        {"task": "T0", "event_type": "task_end", "phase": "finish", "wall_clock": "1000", "lsl_clock": "110"},
        {"task": "T1", "event_type": "push_content", "phase": "tobii_calibration", "wall_clock": "1100", "lsl_clock": "210"},
        {"task": "T1", "event_type": "task_end", "phase": "finish", "wall_clock": "1200", "lsl_clock": "310"},
        {"task": "T2", "event_type": "push_content", "phase": "tobii_calibration", "wall_clock": "1300", "lsl_clock": "410"},
        {"task": "T2", "event_type": "task_end", "phase": "finish", "wall_clock": "1400", "lsl_clock": "510"},
        {"task": "T4", "event_type": "task_end", "phase": "finish", "wall_clock": "1700", "lsl_clock": "810"},
    ]

    task_windows, offset = mod._compute_task_windows(rows)
    break_windows = mod._compute_break_windows(rows, task_windows, offset)

    labels = [w["task"] for w in break_windows]
    assert labels[0] == "BREAK_T0_T1"
    assert "BREAK_T1_T2" in labels
    assert "POST" in labels


def test_split_lsl_table_by_windows(tmp_path):
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_2")

    in_file = tmp_path / "sub-01_ses-demo_task-T0T1T2T3T4_acq-lsl_tobii.tsv.gz"
    with gzip.open(in_file, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["lsl_time", "stream_name", "value_0"])
        writer.writerow(["10.0", "Tobii_P1_stream", "a"])
        writer.writerow(["15.0", "Tobii_P1_stream", "b"])
        writer.writerow(["25.0", "Tobii_P1_stream", "c"])

    windows = [
        {
            "task": "T0",
            "run": "01",
            "start_lsl": 9.0,
            "end_lsl": 20.0,
        },
        {
            "task": "T1",
            "run": "01",
            "start_lsl": 20.0,
            "end_lsl": 30.0,
        },
    ]

    out_files = mod._split_lsl_table_by_windows(
        input_file=in_file,
        output_dir=tmp_path,
        sub_label="01",
        ses_label="demo",
        suffix="acq-lsl_tobii.tsv.gz",
        windows=windows,
    )

    assert len(out_files) == 2
    assert out_files[0].name == "sub-01_ses-demo_task-T0_run-01_acq-lsl_tobii.tsv.gz"
    assert out_files[1].name == "sub-01_ses-demo_task-T1_run-01_acq-lsl_tobii.tsv.gz"

    with gzip.open(out_files[0], "rt", encoding="utf-8", newline="") as f:
        rows0 = list(csv.reader(f, delimiter="\t"))
    with gzip.open(out_files[1], "rt", encoding="utf-8", newline="") as f:
        rows1 = list(csv.reader(f, delimiter="\t"))

    assert len(rows0) == 3
    assert len(rows1) == 2


def test_load_tobii_scene_start_from_recording_g3(tmp_path):
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_3")
    tobii_dir = tmp_path / "deviceA"
    tobii_dir.mkdir(parents=True)
    scene = tobii_dir / "scenevideo.mp4"
    scene.write_bytes(b"x")
    (tobii_dir / "recording.g3").write_text(
        '{"created":"2026-03-19T12:06:14.462877Z","duration":500.0}',
        encoding="utf-8",
    )

    ts, source, device = mod._load_tobii_scene_start(scene)
    assert ts is not None
    assert source == "recording.g3"
    assert device == "deviceA"


def test_load_progress_start(tmp_path):
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_4")
    run_dir = tmp_path / "run"
    sync_dir = run_dir / "sourcedata" / "sync"
    sync_dir.mkdir(parents=True)
    tsv = sync_dir / "dpa_an1_aud_ffmpeg_progress.tsv"
    tsv.write_text(
        "host_time_sec\tout_time_sec\tframe\tdrop_frames\tdup_frames\n"
        "100.10\t0.10\t3\t0\t0\n"
        "100.20\t0.20\t6\t0\t0\n"
        "100.30\t0.30\t9\t0\t0\n",
        encoding="utf-8",
    )

    start = mod._load_progress_start(run_dir, "dpa_an1_aud")
    assert start is not None
    assert abs(start - 100.0) < 1e-9


def test_iter_stimuli_answer_rows_shapes():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_5")

    vad = {
        "type": "vad",
        "valence": 5,
        "arousal": 3,
        "dominance": None,
    }
    postblock = {
        "type": "postblock",
        "responses": {"overall_valence": 6, "team_coordination": 4},
    }
    generic = {
        "task": "T2",
        "final_topic": "topic_a",
        "final_format": "format_b",
    }

    vad_rows = mod._iter_stimuli_answer_rows(vad)
    post_rows = mod._iter_stimuli_answer_rows(postblock)
    generic_rows = mod._iter_stimuli_answer_rows(generic)

    assert ("valence", 5) in vad_rows
    assert ("arousal", 3) in vad_rows
    assert all(k != "dominance" for k, _ in vad_rows)
    assert ("overall_valence", 6) in post_rows
    assert ("team_coordination", 4) in post_rows
    assert ("final_topic", "topic_a") in generic_rows
    assert ("final_format", "format_b") in generic_rows


def test_write_stimuli_answers_table_and_participant_signal_map(tmp_path):
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_6")

    session_dir = tmp_path / "sub-01" / "ses-demo"
    stimuli_dir = session_dir / "sourcedata" / "stimuli" / "stimuli_run"
    stimuli_dir.mkdir(parents=True)

    responses = stimuli_dir / "responses_demo.jsonl"
    responses.write_text(
        "\n".join(
            [
                '{"device_id":"tablet1","participant":1,"task":"T0","type":"vad","valence":4,"arousal":3,"dominance":2,"received_at":100.1,"server_received_lsl":10.1}',
                '{"device_id":"tablet2","participant":2,"task":"T1","type":"postblock","responses":{"overall_valence":6,"engagement":5},"received_at":120.0,"server_received_lsl":30.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (session_dir / "et").mkdir(parents=True, exist_ok=True)
    with gzip.open(session_dir / "et" / "sub-01_ses-demo_task-T0T1T2T3T4_acq-lsl_tobii.tsv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["lsl_time", "stream_name", "stream_type", "value_0"])
        writer.writerow(["10.0", "Tobii_P1_stream", "EyeTracking", "x"])

    (session_dir / "physio").mkdir(parents=True, exist_ok=True)
    with gzip.open(session_dir / "physio" / "sub-01_ses-demo_task-T0T1T2T3T4_acq-lsl_emotibit.tsv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["lsl_time", "stream_name", "stream_type", "value_0"])
        writer.writerow(["10.0", "Emotibit_P2_stream", "Emotibit", "y"])

    answers_tsv, summary = mod._write_stimuli_answers_table(
        session_dir=session_dir,
        stimuli_dir=stimuli_dir,
        sub_label="01",
        ses_label="demo",
    )
    assert answers_tsv is not None
    assert answers_tsv.exists()
    assert summary["rows"] == 5
    assert summary["participants"]["P1"] == 1
    assert summary["participants"]["P2"] == 1

    participant_map = mod._write_participant_signal_map(
        session_dir=session_dir,
        sub_label="01",
        ses_label="demo",
        answers_tsv=answers_tsv,
    )
    assert participant_map.exists()
    rows = list(csv.DictReader(participant_map.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    signals = {r["signal"]: r for r in rows}
    assert "Tobii_P1_stream" in signals
    assert signals["Tobii_P1_stream"]["participant"] == "P1"
    assert "Emotibit_P2_stream" in signals
    assert signals["Emotibit_P2_stream"]["participant"] == "P2"


def test_dpa_audio_label_detection():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_7")

    assert mod._is_dpa_audio_label("dpa_mic1_aud")
    assert mod._is_dpa_audio_label("dpa_mic2_aud")
    assert mod._is_dpa_audio_label("dpa_mic3_aud")
    assert mod._is_dpa_audio_label("dpa_mic4_aud")
    assert mod._is_dpa_audio_label("dpa_an1_aud")
    assert not mod._is_dpa_audio_label("room_mic")
    assert not mod._is_dpa_audio_label("usb_mic")


def test_windowed_audio_label_detection():
    mod = _load_module("tools/multisource_to_bids_runs.py", "multisource_to_bids_runs_8")

    assert mod._is_windowed_audio_label("dpa_mic1_aud")
    assert mod._is_windowed_audio_label("jabra_panacast_20_cam2_vid_audio")
    assert mod._is_windowed_audio_label("jabra_panacast_50_vid_audio")
    assert not mod._is_windowed_audio_label("room_mic")
    assert not mod._is_windowed_audio_label("usb_mic")
