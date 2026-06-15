# Video Feature Extraction Workflow

This workflow runs compact, reusable feature extraction directly from session camera videos.

Tool entrypoint:
- `tools/extract_video_features.py`

Outputs:
- `frame_sync.jsonl` per camera
- `marker_detections_2d.jsonl` per camera (ArUco)
- `body_2d.npz` per camera (optional)
- `face_2d.npz` per camera (optional)
- `hands_2d.npz` per camera (optional)
- `feature_manifest.json` (full run) or `feature_extraction_dry_run.json` (preflight)

## 1) Preflight (recommended)

Run dry-run first to validate video discovery and frame-log matching without heavy inference:

```powershell
python tools/extract_video_features.py `
  --videos-dir <session>/video `
  --output-dir <session>/features_video `
  --frame-log-dir <session>/sourcedata/av/frame_logs `
  --dry-run
```

Review:
- `<session>/features_video/feature_extraction_dry_run.json`

For already split BIDS task clips, the extractor now auto-reads `annot/*_task_run_windows.tsv`
when present and records clip start `unix_time_s` / `lsl_time` metadata even without frame logs.

## 2) Run extraction (MediaPipe baseline)

```powershell
python tools/extract_video_features.py `
  --videos-dir <session>/video `
  --output-dir <session>/features_video `
  --frame-log-dir <session>/sourcedata/av/frame_logs `
  --marker-config configs/desk_markers_large.yaml `
  --body --hands --faces --markers `
  --body-backbone mediapipe-pose `
  --aruco-dicts DICT_4X4_50,DICT_4X4_250
```

## 3) Optional RTMPose/RTMW body extraction

Use MMPose backbone for body while keeping marker/face/hand extraction enabled or disabled as needed:

```powershell
python tools/extract_video_features.py `
  --videos-dir <session>/video `
  --output-dir <session>/features_video_rtmpose `
  --frame-log-dir <session>/sourcedata/av/frame_logs `
  --body --no-hands --no-faces --markers `
  --body-backbone rtmpose-mmpose `
  --rtmpose-model rtmw-l `
  --device cuda:0
```

## 4) Performance controls

- `--body-stride`, `--face-stride`, `--hand-stride`, `--marker-stride`: sample every Nth frame
- `--resize-width`: lower inference resolution for faster processing
- `--max-frames`: cap total processed frames per video (0 means full video)
- `--float-dtype float16|float32`: dense-array storage precision

## Notes

- Heavy dependencies (OpenCV/MediaPipe/MMPose) are imported lazily so `--help` and `--dry-run` stay fast.
- For raw unsplit videos, per-frame absolute timing comes from frame logs when available.
- For BIDS split task clips under `session/video/`, per-frame absolute timing falls back to
  `annot/*_task_run_windows.tsv` plus clip-local video PTS, so timestamps remain usable in
  downstream analysis even when raw capture logs are not copied with the split dataset.
- This workflow only writes derived artifacts and does not modify raw vendor files.
