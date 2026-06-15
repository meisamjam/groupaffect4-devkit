from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.features import extract_physio_features as physio


def _physio_frame(duration_s: float = 80.0, fs: float = 25.0) -> pd.DataFrame:
    n = int(duration_s * fs)
    t = np.arange(n, dtype=float) / fs
    return pd.DataFrame(
        {
            "lsl_time": 1000.0 + t,
            "stream_name": "Emotibit_P1_stream",
            "stream_type": "EmotiBit",
            "value_0": 4000.0 + 100.0 * np.sin(2.0 * np.pi * 1.2 * t),
            "value_1": 70000.0,
            "value_2": 140000.0,
            "value_3": 0.1 + 0.001 * t,
            "value_6": 72.0 + np.sin(2.0 * np.pi * 0.1 * t),
            "value_10": 35.0 + 0.01 * t,
            "value_13": 0.0,
            "value_14": 0.0,
            "value_15": 1.0,
        }
    )


def _write_physio(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def test_compute_physio_features_uses_device_hr_and_basic_qc() -> None:
    df = _physio_frame()
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)

    assert out["hr_source"] == "device_hr"
    assert 71.0 < float(out["hr_mean_bpm"]) < 73.0
    assert float(out["coverage_pct"]) == 100.0
    assert out["physio_available"] is True
    assert "eda_low_coverage" not in str(out["qc_flag"])


def test_compute_physio_features_flags_low_eda_coverage() -> None:
    df = _physio_frame()
    df.loc[df.index[:1800], "value_3"] = np.nan

    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)

    assert "eda_low_coverage" in str(out["qc_flag"])
    assert float(out["eda_coverage_pct"]) < 80.0


def test_main_writes_canonical_qc_and_legacy_tables(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data" / "sub-01" / "ses-demo"
    physio_dir = root / "physio"
    base = "sub-01_ses-demo"
    t0 = _physio_frame()
    t1 = _physio_frame()
    t1["value_6"] = t1["value_6"] + 5.0
    _write_physio(physio_dir / f"{base}_task-T0_run-01_acq-P1_emotibit.tsv.gz", t0)
    _write_physio(physio_dir / f"{base}_task-T1_run-01_acq-P1_emotibit.tsv.gz", t1)
    out_dir = tmp_path / "features"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_physio_features.py",
            "--data-root",
            str(tmp_path / "data"),
            "--out-dir",
            str(out_dir),
            "--window-s",
            "30",
            "--step-s",
            "30",
        ],
    )

    assert physio.main() == 0
    task = pd.read_csv(out_dir / "physio_participant_task.tsv", sep="\t")
    qc = pd.read_csv(out_dir / "physio_qc_summary.tsv", sep="\t")

    assert (out_dir / "features_physio_participant_task.tsv").exists()
    assert (out_dir / "physio_feature_definitions.tsv").exists()
    assert list(task["task_id"]) == ["T0", "T1"]
    assert task.loc[task["task_id"] == "T1", "hr_mean_bpm_delta_t0"].iloc[0] > 4.0
    assert len(qc) == 2
    assert set(qc["participant_id"]) == {"P1"}


def test_build_qc_summary_can_include_missing_expected_rows() -> None:
    row = {
        "session_id": "ses-demo",
        "participant_id": "P1",
        "task_id": "T0",
        "physio_available": True,
        "ppg_coverage_pct": 100.0,
        "eda_coverage_pct": 100.0,
        "temp_coverage_pct": 100.0,
        "imu_available": True,
        "duration_s": 80.0,
        "coverage_pct": 100.0,
        "qc_flag": "ok",
        "qc_notes": "ok",
    }

    qc = physio.build_qc_summary(
        pd.DataFrame([row]),
        expected_sessions=["ses-demo"],
        expected_tasks=["T0", "T1"],
        expected_participants=["P1", "P2"],
    )

    assert len(qc) == 4
    missing = qc[(qc["task_id"] == "T1") & (qc["participant_id"] == "P2")].iloc[0]
    assert bool(missing["physio_available"]) is False
    assert missing["qc_flag"] == "missing_physio"


def test_channel_map_unconfirmed_flag_set_by_default() -> None:
    df = _physio_frame()
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)
    assert "channel_map_unconfirmed" in str(out["qc_flag"])


def test_channel_map_confirmed_clears_flag() -> None:
    df = _physio_frame()
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10, channel_map_confirmed=True)
    assert "channel_map_unconfirmed" not in str(out["qc_flag"])


def test_qc_notes_is_human_readable_and_differs_from_qc_flag() -> None:
    df = _physio_frame()
    df.loc[df.index[:1800], "value_3"] = np.nan
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)
    assert "eda" in str(out["qc_notes"]).lower()
    assert out["qc_notes"] != out["qc_flag"]


def test_eda_phasic_detection_limited_flag_when_low_eda_coverage() -> None:
    df = _physio_frame()
    df.loc[df.index[:1800], "value_3"] = np.nan
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)
    assert "eda_phasic_detection_limited" in str(out["qc_flag"])


@pytest.mark.skipif(physio.find_peaks is None, reason="scipy not available; eda_phasic_detection_limited always set")
def test_eda_phasic_detection_not_flagged_when_coverage_ok() -> None:
    df = _physio_frame()
    out = physio.compute_physio_features(df, ppg_idx=0, eda_idx=3, temp_idx=10)
    assert "eda_phasic_detection_limited" not in str(out["qc_flag"])
