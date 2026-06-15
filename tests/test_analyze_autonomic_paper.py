from __future__ import annotations

import numpy as np
import pandas as pd

from tools.features import analyze_autonomic_paper as auto


def test_prepare_pupil_task_adds_t0_deltas_and_usability() -> None:
    pupil = pd.DataFrame(
        [
            {
                "session_id": "ses-demo",
                "task": "T0",
                "participant_id": "P1",
                "pupil_mean": 3.0,
                "pupil_std": 0.2,
                "pupil_slope_per_s": 0.01,
                "pupil_missing_frac": 0.05,
                "gaze_valid_frac": 0.95,
            },
            {
                "session_id": "ses-demo",
                "task": "T1",
                "participant_id": "P1",
                "pupil_mean": 3.4,
                "pupil_std": 0.3,
                "pupil_slope_per_s": 0.02,
                "pupil_missing_frac": 0.10,
                "gaze_valid_frac": 0.90,
            },
        ]
    )

    out = auto.prepare_pupil_task(pupil, min_valid_frac=0.7, max_missing_frac=0.3)
    t1 = out[out["task_id"] == "T1"].iloc[0]

    assert np.isclose(t1["pupil_mean_delta_t0"], 0.4)
    assert bool(t1["pupil_usable"]) is True


def test_composite_scores_keep_pupil_separate_from_physio_index() -> None:
    rows = []
    for task_id, pupil_delta in [("T1", -0.2), ("T2", 0.2)]:
        for participant_id in ["P1", "P2", "P3"]:
            rows.append(
                {
                    "session_id": "ses-demo",
                    "task_id": task_id,
                    "participant_id": participant_id,
                    "hr_mean_bpm_delta_t0": 1.0 if task_id == "T2" else 0.0,
                    "hrv_rmssd_ms_delta_t0": -1.0 if task_id == "T2" else 0.0,
                    "eda_tonic_mean_delta_t0": 0.1 if task_id == "T2" else 0.0,
                    "eda_phasic_rate_hz_delta_t0": 0.01 if task_id == "T2" else 0.0,
                    "pupil_mean_delta_t0": pupil_delta,
                    "accel_dynamic_mean_delta_t0": 0.0,
                }
            )

    out = auto.build_composite_scores(pd.DataFrame(rows))

    assert "physio_arousal_index" in out.columns
    assert "pupil_diameter_index" in out.columns
    assert "autonomic_arousal_index" not in out.columns
    assert out.loc[out["task_id"] == "T2", "physio_arousal_index"].mean() > 0
    assert out.loc[out["task_id"] == "T2", "pupil_diameter_index"].mean() > 0


def test_modality_coverage_counts_both_usable_rows() -> None:
    qc = pd.DataFrame(
        [
            {
                "session_id": "ses-demo",
                "task_id": "T1",
                "participant_id": "P1",
                "physio_available": True,
                "ppg_usable": True,
            },
            {
                "session_id": "ses-demo",
                "task_id": "T1",
                "participant_id": "P2",
                "physio_available": True,
                "ppg_usable": False,
            },
        ]
    )
    physio = qc[["session_id", "task_id", "participant_id"]].copy()
    pupil = pd.DataFrame(
        [
            {
                "session_id": "ses-demo",
                "task_id": "T1",
                "participant_id": "P1",
                "pupil_usable": True,
            },
            {
                "session_id": "ses-demo",
                "task_id": "T1",
                "participant_id": "P2",
                "pupil_usable": True,
            },
        ]
    )

    out = auto.build_modality_coverage(qc, physio, pupil)
    row = out.iloc[0]

    assert row["expected_participant_rows"] == 2
    assert row["both_usable_count"] == 1
    assert row["both_usable_pct"] == 50.0
