# Architecture

This repository provides Python tooling for **post-collection processing** of AffectAI multimodal
data. Data has been collected from a 4-participant meeting experiment; this repo handles
synchronisation, BIDS packaging, 3D reconstruction, and initial analysis.

## Experiment overview
- **4 participants** (P1–P4) per session, **4 tasks** (T1–T4)
- T1 Hidden-Profile Decision, T2 Mini-Negotiation, T3 Idea Generation (NGT), T4 Public-Goods Micro-Game
- All streams were recorded across two PCs with LSL-based central synchronisation

## Processing pipelines

### Offline compute stack

Offline processing can run on a two-workstation stack: an RTX 5080 machine as the primary
GPU worker, and the Phoenix RTX A2000 workstation as staging storage, secondary worker, and QC
node. The 14 TB USB drive should be treated as archive/backup storage rather than a live video
processing disk. See [offline_compute_stack.md](offline_compute_stack.md) for the network,
storage, and command workflow.

### Pipeline 1 — Synchronisation & BIDS packaging

```
raw sources (AV-PC + Rec-PC + Stimuli + Tobii)
        ↓  tools/multisource_to_bids_runs.py
BIDS session tree (sub-*/ses-*/)
  ├── annot/   task_run_windows.tsv, participant_signal_map.tsv
  ├── beh/     per-task events + stimuli answers
  ├── et/      Tobii gaze + pupil TSV
  ├── physio/  EmotiBit PPG, EDA, temperature
  ├── audio/   DPA WAV clips (per task)
  ├── video/   camera MKV clips (per task)
  └── events.tsv  ← authoritative timeline spine
```

- Phase-aware task window derivation (T0–T4) from experiment-control markers
- 4-tier camera sync: frame logs → LSL → progress TSV → events JSONL
- Optional `--split-media` for per-task video/audio clips via ffmpeg

### Pipeline 2 — 3D pose, gaze & gesture

```
calibration_charuco.toml  +  session videos
        ↓  tools/extract_video_features.py
Feature layer: frame sync + ArUco + body/face/hand 2D arrays + manifest
        ↓  tools/video_only_3d_pipeline.py
Stage 1: tobii_multicam_glasses_tracker.py  →  6-DoF pose + world-frame gaze NDJSON
Stage 2: multicam_pose3d.py                →  triangulated body skeleton (epipolar + DLT)
Stage 3: refine_skeleton_3d.py             →  velocity-filtered, interpolated, Butterworth
Stage 4: face_hand_pipeline.py             →  468-point face mesh + blendshapes + hands
Stage 5: gesture extraction                →  gestures_events.ndjson, gestures_summary.json
```

The feature layer is the preferred first heavy video pass. It supports MediaPipe baseline
extraction plus optional RTMPose/RTMW-via-MMPose for body keypoints, writes a dry-run preflight
summary, and preserves split-task clip timing when frame logs are absent.

### Pipeline 3 — Analysis & QC

```
BIDS dataset
  ↓ tools/qc/qc_sync_report.py          → sync drift CSV
  ↓ tools/qc/qc_tobii_world_gaze.py     → gaze scatter + time-series PNG
  ↓ tools/analyze_sync.py               → inter-device LSL offset report
  ↓ tools/analyze_frame_sync.py         → per-camera frame start spread
  ↓ tools/compare_lsl_frame_logs.py     → LSL vs PTS-based sync accuracy
  ↓ tools/sync/build_frames_and_map.py  → piecewise LSL sync maps
```

## Recorded data inventory

These are the device streams recorded during collection (reference only):

| Device | Count | Signals |
|---|---|---|
| Tobii Pro Glasses 3 | 4 (one/participant) | gaze, pupil diameter, egocentric video, IMU |
| EmotiBit | 4 (one/participant) | PPG, GSR/EDA, temperature, IMU |
| Cameras (Jabra PanaCast 20/50) | 5 main + 2 optional | video + embedded audio |
| DPA 4060 microphones | 5 (4 close-talk + 1 room) | audio |
| Vicon optical system | 6 cameras | 3D marker trajectories |
| Tablets | 4 (one/participant) | self-report responses (V-A, probes) |
| Big Screen | 1 | shared stimulus display |

## BIDS output structure

```
study_root/
  participants.tsv              # anonymised roster (sub_id, group, age_band, sex)
  participants.json
  dataset_description.json
  sub-{participant}/
    ses-{session}/
      et/                       # Tobii gaze + pupil TSV
      physio/                   # EmotiBit PPG, EDA, temp
      audio/                    # DPA close-talk + room audio clips
      video/                    # camera recordings + Tobii scene video clips
      mocap/                    # skeleton 3D, pose NDJSON, gestures
      beh/                      # tablet responses, task events, stimuli answers
      annot/                    # task windows, participant signal map, annotations
      events.tsv                # ← one authoritative timeline spine
      sourcedata/
        av/                     # AV-PC raw MKV/WAV + frame logs + progress TSV
        tobii_lsl/              # Tobii LSL bridge NDJSON dumps
        tobii_device/           # manually downloaded Tobii recordings
        vicon_lsl/              # Vicon datastream NDJSON
        sync/                   # upload manifests, QC artifacts
```

## Repository map

```
src/affectai_capture/          package (FreeMoCapProcessor, registration, devices)
tools/                         all processing + QC scripts
  multisource_to_bids_runs.py  master BIDS packaging
  raw_to_bids.py               raw → BIDS layout
  extract_video_features.py    feature-first video surrogate extraction
  video_only_3d_pipeline.py    single-command 3D pipeline
  multicam_pose3d.py           epipolar triangulation engine
  refine_skeleton_3d.py        skeleton refinement
  face_hand_pipeline.py        MediaPipe face + hand
  calibrate_charuco.py         ChArUco calibration workflow
  recenter_calibration.py      extrinsic re-origin
  validate_calibration_robust.py  calibration QC
  tobii_multicam_glasses_tracker.py  6-DoF gaze tracking
  analyze_sync.py / analyze_frame_sync.py / compare_lsl_frame_logs.py
  qc/                          sync + gaze QC reports
  sync/                        frame table + LSL sync map builders
configs/                       camera specs, ArUco geometry, participant maps
tests/                         pytest suite
docs/                          documentation
```

## Key design choices

1. **LSL as universal timebase** — all streams timestamped with `pylsl.local_clock()`; NTP-like sync across PCs
2. **BIDS-first output** — all derived files use `sub-`, `ses-`, `task-`, `run-`, `acq-` naming
3. **Dual archival** — each device wrote to both LSL (real-time) and NDJSON/JSONL (device-native backup)
4. **Lazy optional imports** — heavy deps (`cv2`, `freemocap`, `pylsl`) imported only when needed; tests run without full stack
5. **Tobii redundancy** — gaze captured on two independent paths (Tobii SDK + Vicon SDK embedding)

## Participant registration
- `src/affectai_capture/registration.py` — maps real names → participant IDs (P1–P4)
- Demographics and personality linkage (BFI-45) per session
- Outputs BIDS-compliant `participants.tsv` (anonymised) + per-session `participants.json`
