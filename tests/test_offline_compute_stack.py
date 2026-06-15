from __future__ import annotations

import csv
from pathlib import Path

from tools.offline_compute_stack import StackPaths, build_stage_commands, upsert_queue_row


def _make_stack(tmp_path: Path) -> StackPaths:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "stage" / "data"
    config_dir = tmp_path / "stage" / "configs"
    work_root = tmp_path / "work"
    processed_root = tmp_path / "stage" / "processed"
    queue_tsv = tmp_path / "stage" / "work_queue.tsv"

    for path in [repo_root, data_root, config_dir, work_root, processed_root]:
        path.mkdir(parents=True, exist_ok=True)

    return StackPaths(
        repo_root=repo_root,
        data_root=data_root,
        config_dir=config_dir,
        work_root=work_root,
        processed_root=processed_root,
        queue_tsv=queue_tsv,
    )


def test_queue_upsert_add_and_update(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.tsv"

    first = upsert_queue_row(
        queue_tsv=queue_path,
        session_id="20260311_grp-06_run01",
        status="staged",
        assigned_to="storage-worker",
        notes="initial",
    )
    assert first["session_id"] == "ses-20260311_grp-06_run01"
    assert first["status"] == "staged"

    second = upsert_queue_row(
        queue_tsv=queue_path,
        session_id="ses-20260311_grp-06_run01",
        status="pose-3d",
        assigned_to="gpu-main",
        notes="processing",
    )
    assert second["status"] == "pose-3d"
    assert second["assigned_to"] == "gpu-main"

    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 1
    assert rows[0]["session_id"] == "ses-20260311_grp-06_run01"
    assert rows[0]["status"] == "pose-3d"


def test_build_stage_commands_includes_expected_tools(tmp_path: Path) -> None:
    stack = _make_stack(tmp_path)

    commands = build_stage_commands(
        stack=stack,
        session_id="20260311_grp-06_run01",
        split_media=True,
        max_workers=2,
        sub_id="sub-01",
    )

    assert "bids" in commands
    assert "pose_json" in commands
    assert "pose3d_dryrun" in commands
    assert "qc_sync" in commands

    bids = commands["bids"]
    assert any("multisource_to_bids_runs.py" in token for token in bids)
    assert "--split-media" in bids

    pose3d_dryrun = commands["pose3d_dryrun"]
    assert any("video_only_3d_pipeline.py" in token for token in pose3d_dryrun)
    assert "--dry-run" in pose3d_dryrun
