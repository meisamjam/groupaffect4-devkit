# Data Flow Architecture

This document describes the complete data pipeline for the AffectAI meeting experiment
(4 participants, 4 tasks, multimodal recording). **Data collection is complete.**
This repo focuses on the post-processing pipelines below.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RAW DATA (already collected)                                           │
│                                                                         │
│  Recording PC:  XDF (LabRecorder), Tobii NDJSON, EmotiBit JSONL,       │
│                 Vicon NDJSON, stimuli JSONL, events.tsv                 │
│  AV PC:         camera MKV + frame logs, DPA WAV, progress TSV         │
│  Tobii device:  manually downloaded scene video + gaze recordings       │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│  PIPELINE 1 — Sync & BIDS packaging                                       │
│  tools/multisource_to_bids_runs.py                                        │
│    → merge sources, derive task windows (T0–T4), write BIDS tree          │
│    → run tools/raw_to_bids.py for modality layout                         │
│    → optional --split-media: per-task video/audio clips via ffmpeg        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ↓  BIDS dataset
┌───────────────────────────────────────────────────────────────────────────┐
│  PIPELINE 2 — 3D pose, gaze & gesture                                     │
│  tools/video_only_3d_pipeline.py                                          │
│    Stage 1: calibrate_charuco.py → recenter_calibration.py               │
│    Stage 2: tobii_multicam_glasses_tracker.py → 6-DoF pose + world gaze  │
│    Stage 3: multicam_pose3d.py → triangulated body skeleton               │
│    Stage 4: refine_skeleton_3d.py → smoothed skeleton                    │
│    Stage 5: face_hand_pipeline.py → face mesh + blendshapes + hands      │
│    Stage 6: gesture extraction → gestures_events.ndjson                  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ↓  mocap/ outputs
┌───────────────────────────────────────────────────────────────────────────┐
│  PIPELINE 3 — Analysis & QC                                               │
│  tools/qc/qc_sync_report.py       → sync drift CSV                       │
│  tools/qc/qc_tobii_world_gaze.py  → gaze scatter + time-series PNG       │
│  tools/analyze_sync.py            → LSL inter-device offset report       │
│  tools/analyze_frame_sync.py      → per-camera frame start spread        │
│  tools/compare_lsl_frame_logs.py  → LSL vs PTS-based sync accuracy       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Collection Layer

### Input Sources

**1. Tobii Pro Glasses 3 (4 units, one per participant)**
- **Signals:** gaze (2D + 3D), pupil diameter, egocentric scene video, IMU
- **Data path A:** Tobii SDK → `tools/tobii_glasses_lsl_bridge.py` → LSL streams per device
- **Data path B:** Vicon SDK embeds Tobii eye-tracker data in Vicon frames → `tools/vicon_nexus_lsl_bridge.py` → LSL
- **LSL streams:** `Tobii_{id}_Gaze`, `Tobii_{id}_Pupil`, optionally `_Gaze3D`, `_Imu`, `_Event`, `_SyncPort`
- **Archival:** NDJSON per device in `data/streams/tobii_glasses/{id}.ndjson`
- **Bridge:** [tools/tobii_glasses_lsl_bridge.py](../tools/tobii_glasses_lsl_bridge.py)

**2. EmotiBit (4 units, one per participant)**
- **Signals:** PPG (green/red/IR), GSR/EDA, temperature, accelerometer, gyroscope, magnetometer
- **Raw input:** UDP packets on port 12346 (via Oscilloscope app)
- **Data path:** UDP → `emotibit.py` → LSL streams per device + channel
- **LSL streams:** `EmotiBit_{device}_{channel}` (PG, EA, T0, AX, HR, etc.)
- **Archival:** JSONL per channel in `sourcedata/physio/emotibit/`
- **Module:** [src/affectai_capture/devices/emotibit.py](../src/affectai_capture/devices/emotibit.py)

**3. Cameras — Jabra PanaCast 20/50 (5 main + 2 optional extra)**
- **Signals:** video (1080p/720p @ 30fps) + embedded microphone audio
- **Recorded on:** AV PC via `ffmpeg_multicap.py`
- **Sync to LSL:** frame logs (per-frame PTS + Unix time) → LSL progress streams → Recording PC
- **Archival:** MKV video files in `video/`, frame logs in `sourcedata/sync/`
- **Tool:** [tools/ffmpeg_multicap.py](../tools/ffmpeg_multicap.py)

**4. DPA 4060 Microphones (5 units: 4 close-talk + 1 room/spare)**
- **Signals:** audio (48 kHz, 16-bit)
- **Recorded on:** AV PC via RME Fireface 802 interface
- **Sync to LSL:** audio interface clock → LSL progress streams → Recording PC
- **Archival:** WAV files in `audio/`
- **Tool:** [tools/dpa_recorder.py](../tools/dpa_recorder.py)

**5. Vicon Optical System (6 cameras)**
- **Signals:** 3D marker trajectories, segment translations/rotations
- **Data path:** Vicon DataStream SDK → `vicon_nexus_lsl_bridge.py` → LSL
- **LSL streams:** `ViconDataStreamClock`, `ViconDataStreamFrame`, per-segment structured outlets
- **Archival:** NDJSON per session
- **Bridge:** [tools/vicon_nexus_lsl_bridge.py](../tools/vicon_nexus_lsl_bridge.py)

**6. Stimulus Markers & Tablet Responses**
- **Stimulus events:** `display_server.py` pushes task phases to tablets/big-screen via SSE
- **Marker stream:** `AffectAI_Markers` LSL outlet + per-device outlets (`AffectAI_Tablet1–4`, `_Moderator`, `_BigScreen`)
- **Tablet responses:** HTTP POST from browser → JSONL per participant per session
- **Module:** [stimuli/display_server.py](../stimuli/display_server.py), [src/affectai_capture/stimulus.py](../src/affectai_capture/stimulus.py)

### Dual Output Strategy

Each device adapter produces **two outputs simultaneously**:

#### Output A: LSL Network Streams
- **Purpose:** Real-time synchronization, multi-device coordination
- **Protocol:** Lab Streaming Layer (LSL)
- **Clock:** LSL unified clock (automatic synchronization)
- **Consumers:** Central recorder (LabRecorder → XDF), monitoring tools

#### Output B: Archival Files
- **Purpose:** Device-native backup, debugging, offline analysis
- **Format:** JSONL (newline-delimited JSON) or native format (MKV/WAV)
- **Clock:** Original device timestamp (preserved)
- **Location:** `sub-{id}/ses-{id}/sourcedata/{device}/`

## LSL Network Layer

### Complete Stream Inventory

**Tobii Glasses (per device, ×4):**
- `Tobii_{id}_Gaze` (2ch: x, y) — 2D gaze position
- `Tobii_{id}_Pupil` (2ch: left, right) — pupil diameter mm
- `Tobii_{id}_Gaze3D` (3ch: x, y, z) — optional 3D gaze vector
- `Tobii_{id}_Imu` (9ch: accel xyz, gyro xyz, mag xyz) — optional IMU
- `Tobii_{id}_Event` (2ch: tag, obj) — optional device events
- `Tobii_{id}_SyncPort` (2ch: direction, value) — optional sync port

**EmotiBit (per device, ×4):**
- `EmotiBit_{device}_ppg_green` — PPG photoplethysmography
- `EmotiBit_{device}_eda` — electrodermal activity (GSR)
- `EmotiBit_{device}_temperature_0` — skin temperature
- `EmotiBit_{device}_accel_{x,y,z}` — accelerometer
- `EmotiBit_{device}_heart_rate` — derived HR (bpm)
- (See `CHANNEL_NAMES` in [emotibit.py](../src/affectai_capture/devices/emotibit.py) for full list)

**Vicon:**
- `ViconDataStreamClock` — clock heartbeat (always)
- `ViconDataStreamFrame` — JSON payload per frame
- `ViconSegmentTranslation` (3ch: x,y,z) — optional structured numeric
- `TobiiEyeTracker` (6ch: pos xyz, gaze xyz) — Tobii via Vicon SDK
- Per-stream: `Vicon_{subject}_{segment}_{Translation|Rotation}` — optional

**Stimulus/Markers:**
- `AffectAI_Markers` — unified experiment event markers
- `AffectAI_Tablet{1-4}` — per-participant tablet events
- `AffectAI_Moderator` — moderator actions
- `AffectAI_BigScreen` — shared display events

**AV PC Clocks:**
- Camera frame-log progress streams (5ch: out_time_sec, media_time_us, frame, drop, dup)
- DPA audio progress streams

### Timestamp Synchronization

**Challenge:** Multiple independent clocks across two PCs and many devices
1. **Device clocks:** EmotiBit (µs since boot), Tobii (SDK ticks), Vicon (frame number)
2. **System clocks:** Python `time.time()` on each PC
3. **LSL clock:** `pylsl.local_clock()` — unified across network

**Solution:** LSL as common timebase
- Each adapter timestamps samples with `local_clock()` at ingestion
- LSL synchronizes clocks across PCs via NTP-like protocol
- Original device timestamps preserved in archival files
- 4-tier camera sync: frame logs → LSL progress → progress TSV → events JSONL

**Accuracy:** ~1ms for LSL streams on same PC; ±33ms (1 frame) for camera sync

## Central Recording

All LSL streams converge on the **Recording PC** where LabRecorder saves to XDF.

**Recommended mode: Hybrid**
- LSL → XDF for synchronized multi-stream recording
- Device-native files (JSONL, MKV, WAV) for archival backup
- `events.tsv` as authoritative session timeline

## Post-Processing Layer

### Pipeline 1 — Sync & BIDS packaging

**Step 1: Merge sources and derive task runs**
```bash
python tools/multisource_to_bids_runs.py \
  --av-dir    data/raw/av_pc/ \
  --rec-dir   data/raw/rec_pc/ \
  --stim-dir  data/raw/stimuli/ \
  --tobii-dir data/raw/tobii/ \
  --out       data/bids/sub-001/ses-001
```

**Step 2: Per-task media clips (optional)**
```bash
python tools/multisource_to_bids_runs.py ... --split-media
```

**Expected BIDS output per participant per task run:**
```
sub-01/ses-01/
  annot/sub-01_ses-01_task_run_windows.tsv
  annot/sub-01_ses-01_participant_signal_map.tsv
  beh/sub-01_ses-01_task-T1_run-01_events.tsv
  beh/sub-01_ses-01_task-T0T1T2T3T4_stimuli_answers.tsv
  et/sub-01_ses-01_task-T1_run-01_acq-tobii_gaze.tsv
  physio/sub-01_ses-01_task-T1_run-01_acq-emotibit_physio.tsv
  audio/sub-01_ses-01_task-T1_run-01_acq-dpa-close-talk_audio.wav
  video/sub-01_ses-01_task-T1_run-01_acq-jabra-panacast-20-cam1_video.mkv
  events.tsv
```

### Pipeline 2 — 3D pose, gaze & gesture

**Step 1: Feature-first video surrogate extraction**
```bash
python tools/extract_video_features.py \
  --videos-dir data/bids/sub-001/ses-001/video/ \
  --output-dir data/bids/sub-001/ses-001/features_video/ \
  --frame-log-dir data/bids/sub-001/ses-001/sourcedata/av/frame_logs \
  --marker-config configs/desk_markers_large.yaml \
  --dry-run
python tools/extract_video_features.py \
  --videos-dir data/bids/sub-001/ses-001/video/ \
  --output-dir data/bids/sub-001/ses-001/features_video/ \
  --frame-log-dir data/bids/sub-001/ses-001/sourcedata/av/frame_logs \
  --marker-config configs/desk_markers_large.yaml \
  --body --hands --faces --markers \
  --body-backbone mediapipe-pose \
  --aruco-dicts DICT_4X4_50,DICT_4X4_250
```

Use `--body-backbone rtmpose-mmpose --rtmpose-model rtmw-l --device cuda:0` on the
GPU workstation when running the RTMPose/RTMW body backbone. Feature outputs are derived
artifacts under `features_video/` and can be reused for calibration QC, marker audit,
body/face/hand analysis, and future feature-native 3D stages.

**Step 2: Camera calibration**
```bash
python tools/calibrate_charuco.py calibrate \
  --video-dir data/raw/av_pc/video/ --out calibration_charuco.toml
python tools/recenter_calibration.py \
  --in calibration_charuco.toml --out calibration_recentered.toml
python tools/validate_calibration_robust.py calibration_recentered.toml
```

**Step 3: Full 3D pipeline**
```bash
python tools/video_only_3d_pipeline.py \
  --videos-dir  data/bids/sub-001/ses-001/video/ \
  --calibration calibration_recentered.toml \
  --events      data/bids/sub-001/ses-001/events.tsv \
  --out         data/bids/sub-001/ses-001/mocap/
```

**Outputs:** `skeleton_3d.npy`, `skeleton_3d_refined.npy`, `{glasses_id}_pose.ndjson`,
`{glasses_id}_gaze_world.ndjson`, `gestures_events.ndjson`, `gestures_summary.json`

Current note: `video_only_3d_pipeline.py` still consumes OpenPose-compatible pose JSON through
`--pose-root`. `tools/extract_video_features.py` is the preferred feature-surrogate layer and
the place to run MediaPipe/RTMPose/marker extraction once per video; downstream feature-native
3D adapters should consume `features_video/feature_manifest.json` and the per-camera `.npz` /
JSONL files directly.

### Pipeline 3 — Analysis & QC

```bash
# Sync drift and frame alignment
python tools/qc/qc_sync_report.py   --session data/bids/sub-001/ses-001/
python tools/analyze_frame_sync.py  --log-dir data/bids/sub-001/ses-001/sourcedata/av/
python tools/compare_lsl_frame_logs.py \
  --progress-tsv data/.../ffmpeg_progress_cam1.tsv \
  --frame-log    data/.../panacast-20-cam1_frames.jsonl

# Gaze QC
python tools/qc/qc_tobii_world_gaze.py \
  --gaze-dir data/bids/sub-001/ses-001/mocap/ \
  --out      data/bids/sub-001/ses-001/annot/gaze_qc/
```

### BIDS Metadata

**physio.json (sidecar):**
```json
{
  "SamplingFrequency": 25,
  "StartTime": 0.0,
  "Columns": ["PPG_green", "EDA", "temperature", "heart_rate"],
  "Manufacturer": "EmotiBit",
  "RecordingDuration": 5400.0
}
```

**events.tsv (timeline spine):**
```
onset	duration	trial_type	value	description
0.0	0.0	session_init	n/a	Session folders created
120.5	0.0	T1_start	n/a	Hidden-Profile Decision start
125.3	0.0	annot_screen_start	va_grid	VAD prompt displayed
```

## Storage Estimates

### Per-Session Data Volumes (90 min, 4 participants)

| Modality | Format | Estimated Size |
|---|---|---|
| Video (5 cameras) | MKV | ~15–25 GB |
| Audio (5 DPA + room) | WAV | ~2–3 GB |
| Tobii scene video (×4) | MP4 | ~4–8 GB |
| LSL XDF (all streams) | XDF | ~200–500 MB |
| EmotiBit JSONL (×4) | JSONL | ~800 MB–1.2 GB |
| Vicon NDJSON | NDJSON | ~500 MB–1 GB |
| Tobii NDJSON (×4) | NDJSON | ~200–400 MB |
| BIDS TSV (converted) | TSV/TSV.gz | ~100–200 MB |
| Markers + events | TSV/JSONL | < 5 MB |
| **Total per session** | | **~25–40 GB** |

## Tools & Utilities

**Sync & BIDS packaging:**
- [tools/multisource_to_bids_runs.py](../tools/multisource_to_bids_runs.py) — master post-processing (merge sources → BIDS task runs)
- [tools/raw_to_bids.py](../tools/raw_to_bids.py) — raw → BIDS modality layout
- [tools/ingest_tobii_downloads.py](../tools/ingest_tobii_downloads.py) — copy Tobii device recordings into sourcedata
- [tools/upload_raw_data.py](../tools/upload_raw_data.py) — upload raw session to Azure Blob

**3D pose & gaze:**
- [tools/extract_video_features.py](../tools/extract_video_features.py) — feature-first video surrogate extraction (sync, ArUco, body, face, hands)
- [tools/video_only_3d_pipeline.py](../tools/video_only_3d_pipeline.py) — single-command 3D pipeline
- [tools/calibrate_charuco.py](../tools/calibrate_charuco.py) — ChArUco calibration workflow
- [tools/recenter_calibration.py](../tools/recenter_calibration.py) — re-origin camera extrinsics
- [tools/validate_calibration_robust.py](../tools/validate_calibration_robust.py) — calibration QC
- [tools/multicam_pose3d.py](../tools/multicam_pose3d.py) — epipolar triangulation engine
- [tools/refine_skeleton_3d.py](../tools/refine_skeleton_3d.py) — skeleton refinement (filter + interpolate + smooth)
- [tools/face_hand_pipeline.py](../tools/face_hand_pipeline.py) — MediaPipe face mesh + blendshapes + hands
- [tools/tobii_multicam_glasses_tracker.py](../tools/tobii_multicam_glasses_tracker.py) — Tobii 6-DoF + world gaze

**Sync & QC:**
- [tools/analyze_sync.py](../tools/analyze_sync.py) — LSL inter-device offset analysis
- [tools/analyze_frame_sync.py](../tools/analyze_frame_sync.py) — per-camera frame start spread
- [tools/compare_lsl_frame_logs.py](../tools/compare_lsl_frame_logs.py) — LSL vs PTS-based sync accuracy
- [tools/sync/build_frames_and_map.py](../tools/sync/build_frames_and_map.py) — frame tables + piecewise LSL sync maps
- [tools/qc/qc_sync_report.py](../tools/qc/qc_sync_report.py) — sync QC report (CSV)
- [tools/qc/qc_tobii_world_gaze.py](../tools/qc/qc_tobii_world_gaze.py) — gaze QC (scatter + time-series PNG)

**Visualization:**
- [tools/render_skeleton_3d.py](../tools/render_skeleton_3d.py) — animated MP4 of 3D skeleton
- [tools/layout_video_3d.py](../tools/layout_video_3d.py) — synchronized 3×2 grid (cameras + 3D panel)
- [tools/visualize_calibration.py](../tools/visualize_calibration.py) — interactive 3D camera rig plot
- [tools/visualize_skeleton.py](../tools/visualize_skeleton.py) — 3D plot + 2D keypoint overlay

**Tests:**
- [tests/test_calibrate_charuco.py](../tests/test_calibrate_charuco.py) — calibration CLI + camera spec matching
- [tests/test_multisource_to_bids_runs.py](../tests/test_multisource_to_bids_runs.py) — task window derivation + stimuli answers
- [tests/test_tobii_multicam_glasses_tracker.py](../tests/test_tobii_multicam_glasses_tracker.py) — DLT triangulation + gaze-to-world
- [tests/test_extract_video_features.py](../tests/test_extract_video_features.py) — feature extractor preflight, timing, and backbone helpers
- [tests/test_video_only_3d_pipeline.py](../tests/test_video_only_3d_pipeline.py) — gesture extraction + dry-run
- [tests/test_stream_bridges.py](../tests/test_stream_bridges.py) — stream name/config validation

## References

- **LSL Protocol:** [Lab Streaming Layer Documentation](https://labstreaminglayer.readthedocs.io/)
- **BIDS Specification:** [Brain Imaging Data Structure](https://bids-specification.readthedocs.io/)
- **Architecture Overview:** [docs/architecture.md](architecture.md)
