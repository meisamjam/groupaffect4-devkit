# AffectAI Data Processing — Navigation Hub

> Data collection is complete. This hub navigates the **post-collection processing** pipelines.

## I want to...

### Understand the system
- **End-to-end dataset-to-model workflow** → [docs/END_TO_END_DATASET_TO_MODELS.md](docs/END_TO_END_DATASET_TO_MODELS.md)
- **Architecture overview** → [docs/architecture.md](docs/architecture.md)
- **Full data flow** → [docs/data_flow.md](docs/data_flow.md)
- **Recent changes** → [CHANGES.md](CHANGES.md)

### Run the processing pipelines

#### Offline compute stack
- **Two-workstation processing layout** -> [docs/offline_compute_stack.md](docs/offline_compute_stack.md)
- **Recommended split** -> RTX 5080 node for heavy 3D/video jobs; Phoenix workstation for staging, BIDS packaging, QC, and archive coordination

#### Pipeline 1 — Sync & BIDS packaging
- **Merge sources + derive task runs** → `tools/multisource_to_bids_runs.py`
- **Raw → BIDS layout** → `tools/raw_to_bids.py`
- **Upload raw data** → `tools/upload_raw_data.py`
- **Ingest Tobii downloads** → `tools/ingest_tobii_downloads.py`

#### Pipeline 2 — 3D pose, gaze & gesture
- **Single-command pipeline** → `tools/video_only_3d_pipeline.py`
- **Camera calibration** → `tools/calibrate_charuco.py` → `tools/recenter_calibration.py`
- **Calibration QC** → `tools/validate_calibration_robust.py`, `tools/visualize_calibration.py`
- **Multicam 3D body** → `tools/multicam_pose3d.py`
- **Skeleton refinement** → `tools/refine_skeleton_3d.py`
- **Face & hand mesh** → `tools/face_hand_pipeline.py`
- **Tobii world gaze** → `tools/tobii_multicam_glasses_tracker.py`
- **Alternative body (FreeMoCap)** → `tools/reconstruct_3d.py`, `tools/process_freemocap.py`
- **Alternative body (OpenPose)** → `tools/run_openpose.py`, `tools/triangulate_openpose.py`

#### Pipeline 3 — Analysis & QC
- **Sync QC report** → `tools/qc/qc_sync_report.py`
- **Gaze QC** → `tools/qc/qc_tobii_world_gaze.py`
- **Frame sync analysis** → `tools/analyze_frame_sync.py`, `tools/compare_lsl_frame_logs.py`
- **LSL offset analysis** → `tools/analyze_sync.py`
- **Frame table + sync map** → `tools/sync/build_frames_and_map.py`

### Visualize results
- **Animated 3D skeleton** → `tools/render_skeleton_3d.py`
- **Synchronized grid layout** → `tools/layout_video_3d.py`
- **OpenPose 2D overlay** → `tools/visualize_skeleton.py`
- **Calibration rig 3D** → `tools/visualize_calibration.py`

### Understand BIDS output structure
- **BIDS layout** → [docs/architecture.md](docs/architecture.md)
- **Raw data upload** → [docs/raw_data_upload_and_bids_conversion.md](docs/raw_data_upload_and_bids_conversion.md)
- **Personality scores (BFI-44)** → [docs/bfi44_personality_scores.md](docs/bfi44_personality_scores.md)

### Troubleshoot
- **Known issues** → [docs/known_issues.md](docs/known_issues.md)
- **Calibration enhancements** → [docs/calibration_enhancements.md](docs/calibration_enhancements.md)
- **Calibration usage** → [docs/calibration_usage.md](docs/calibration_usage.md)

### Develop
- **Know the rules** → [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **LLM context** → [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)
- **Architecture decisions** → [docs/decisions.md](docs/decisions.md)

---

## Essential Commands

```bash
make check                                          # ruff lint + pytest

# Pipeline 1 — BIDS packaging
python tools/multisource_to_bids_runs.py --help

# Pipeline 2 — 3D pose (dry-run to check prerequisites)
python tools/video_only_3d_pipeline.py --dry-run ...

# Pipeline 3 — QC
python tools/qc/qc_sync_report.py --help
python tools/qc/qc_tobii_world_gaze.py --help
```
