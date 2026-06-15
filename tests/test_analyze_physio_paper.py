from __future__ import annotations

import numpy as np
import pandas as pd

from tools.features import analyze_physio_paper as paper


def _task_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session_id": "ses-demo",
                "participant_id": "P1",
                "task_id": "T0",
                "hr_mean_bpm_delta_t0": 0.0,
                "eda_tonic_mean_delta_t0": 0.0,
            },
            {
                "session_id": "ses-demo",
                "participant_id": "P1",
                "task_id": "T1",
                "hr_mean_bpm_delta_t0": 5.0,
                "eda_tonic_mean_delta_t0": 0.2,
            },
            {
                "session_id": "ses-demo",
                "participant_id": "P2",
                "task_id": "T1",
                "hr_mean_bpm_delta_t0": 7.0,
                "eda_tonic_mean_delta_t0": np.nan,
            },
        ]
    )


def test_feature_usability_keeps_expected_qc_denominator() -> None:
    task = _task_table()
    qc = pd.DataFrame(
        [
            {
                "session_id": "ses-demo",
                "participant_id": "P1",
                "task_id": "T0",
                "ppg_usable": True,
                "eda_usable": True,
                "temp_usable": True,
                "imu_usable": True,
            },
            {
                "session_id": "ses-demo",
                "participant_id": "P2",
                "task_id": "T1",
                "ppg_usable": False,
                "eda_usable": False,
                "temp_usable": True,
                "imu_usable": True,
            },
            {
                "session_id": "ses-demo",
                "participant_id": "P3",
                "task_id": "T1",
                "ppg_usable": False,
                "eda_usable": False,
                "temp_usable": False,
                "imu_usable": False,
            },
        ]
    )

    out = paper.build_feature_usability(task, qc)
    hr = out[out["feature"] == "hr_mean_bpm_delta_t0"].iloc[0]
    ppg = out[out["feature"] == "ppg_usable"].iloc[0]

    assert hr["nonnull_rows"] == 3
    assert hr["actual_feature_rows"] == 3
    assert hr["expected_qc_rows"] == 3
    assert hr["nonnull_pct_of_expected"] == 100.0
    assert ppg["nonnull_rows"] == 1
    assert np.isclose(ppg["nonnull_pct_of_expected"], 100.0 / 3.0)


def test_task_delta_stats_uses_active_tasks_only() -> None:
    out = paper.build_task_delta_stats(_task_table())
    hr = out[(out["feature"] == "hr_mean_bpm_delta_t0") & (out["task_id"] == "T1")].iloc[0]

    assert hr["n"] == 2
    assert hr["mean"] == 6.0
    assert "T0" not in set(out["task_id"])


def test_temporal_profile_bins_windows_by_relative_task_progress() -> None:
    window = pd.DataFrame(
        {
            "session_id": ["ses-demo"] * 3,
            "task_id": ["T1"] * 3,
            "participant_id": ["P1"] * 3,
            "window_start_lsl": [10.0, 40.0, 70.0],
            "window_end_lsl": [20.0, 50.0, 80.0],
            "hr_mean_bpm": [70.0, 80.0, 90.0],
        }
    )

    out = paper.build_temporal_profile(window)
    hr = out[out["feature"] == "hr_mean_bpm"].set_index("task_segment")

    assert hr.loc["early", "mean"] == 70.0
    assert hr.loc["middle", "mean"] == 80.0
    assert hr.loc["late", "mean"] == 90.0
