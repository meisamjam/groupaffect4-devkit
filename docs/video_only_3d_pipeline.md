# Video-Only 3D Pipeline (Multicam + Tobii + Gestures)

This pipeline is for post-hoc processing from recorded video-only inputs, with optional audio left untouched.

It provides:
- Tobii glasses pose estimation from fixed multicam + glasses markers
- Gaze transformation into the shared world coordinate system (desk-centered if calibration/marker map is desk-centered)
- Multi-person 3D skeleton reconstruction
- Rule-based gesture extraction per participant

Implementation entrypoint:
- `tools/video_only_3d_pipeline.py`

Recommended upstream feature pass:
- `tools/extract_video_features.py`

Run `extract_video_features.py --dry-run` before expensive detector inference. The feature pass
writes reusable frame sync, ArUco marker, body, face, and hand artifacts under `features_video/`
and supports MediaPipe plus optional RTMPose/RTMW-via-MMPose body extraction. The current 3D
pipeline still requires OpenPose-compatible pose JSON via `--pose-root`; feature-native 3D
consumers should use `features_video/feature_manifest.json` and the per-camera `.npz`/JSONL
files as their input contract.

## Inputs

Required:
- `--calibration`: multicam calibration TOML (if missing, pipeline can auto-generate one from six P20 videos)
- `--videos-dir`: fixed multicamera session videos for glasses tracking
- `--tracker-config`: tracker YAML for glasses markers + gaze files
- `--pose-root`: directory containing `*_json` folders with 2D pose detections
- `--output-dir`: output directory for pipeline artifacts

Optional synchronization inputs:
- `--events-jsonl`
- `--frame-log-dir`
- `--lsl-dir`

Optional person-consistency inputs:
- `--camera-zones` for stable person indices by seating layout
- `--flip-cameras` for upside-down cameras when calibration was also flip-corrected
- `--face-cameras` for close-up cameras mapped to known participants

Optional auto-calibration controls:
- `--auto-calibrate-missing` / `--no-auto-calibrate-missing`
- `--board-config` (default: `configs/desk_markers_large.yaml`)
- `--camera-specs` (default: `configs/camera_specs.json`)

## Example

Feature preflight/extraction:

```powershell
python tools/extract_video_features.py `
  --videos-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/video `
  --output-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/features_video `
  --marker-config configs/desk_markers_large.yaml `
  --dry-run
python tools/extract_video_features.py `
  --videos-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/video `
  --output-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/features_video `
  --marker-config configs/desk_markers_large.yaml `
  --body --hands --faces --markers `
  --body-backbone mediapipe-pose
```

3D pipeline:

```powershell
python tools/video_only_3d_pipeline.py `
  --calibration sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/video/video_camera_calibration.toml `
  --videos-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/video `
  --tracker-config configs/tobii_multicam_glasses_tracker.example.yaml `
  --pose-root sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/mediapipe `
  --output-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01_phase/video_only_3d `
  --camera-zones cam1+cam4:0,1 cam2+cam3:2,3 `
  --refine-skeleton
```

## Outputs

- `tobii_world/`
  - `{glasses_id}_pose.ndjson`
  - `{glasses_id}_gaze_world.ndjson`
  - `summary.json`
- `skeleton_3d.npy`
- `skeleton_3d_refined.npy` (if `--refine-skeleton`)
- `gestures_events.ndjson`
- `gestures_summary.json`
- `pipeline_summary.json`

## Gesture labels (current heuristic set)

Per participant (`P1..P4`, based on person index in skeleton output):
- `left_hand_to_head`
- `right_hand_to_head`
- `left_hand_to_chest`
- `right_hand_to_chest`
- `left_arm_extended`
- `right_arm_extended`
- `both_hands_to_head`

These are deterministic geometric heuristics from BODY_25 keypoints. Thresholds are configurable via CLI.

## Notes

- Shared world coordinates are inherited from calibration + marker-map conventions used by existing tools.
- If the specified calibration file does not exist and auto fallback is enabled, the pipeline auto-runs ChArUco calibration from six detected `panacast-20-cam1..6` videos under `--videos-dir` and uses `auto_calibration_charuco.toml`.
- For robust P1..P4 consistency, use `--camera-zones` to enforce seat-based identity stabilization.
- This pipeline does not modify raw vendor files; all outputs are derived artifacts.
