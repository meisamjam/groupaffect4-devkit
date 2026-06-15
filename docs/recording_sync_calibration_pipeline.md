# Recording, Synchronisation & Calibration Pipeline

End-to-end reference for the AffectAI multicam capture system: hardware setup,
recording workflow, temporal synchronisation, spatial calibration, 3D pose
reconstruction, and post-processing.

---

## 1  Hardware

### 1.1  Cameras

| Label | Model | Resolution | FPS | Bitrate | Notes |
|-------|-------|------------|-----|---------|-------|
| cam1 – cam4 | Jabra PanaCast 20 | 1920 × 1080 | 30 | 4 000 kbps | Mounted **upside-down** (flip in post) |
| cam5 | Jabra PanaCast 20 | 1920 × 1080 | 30 | 4 000 kbps | Top-front-center overview, mounted **upright** |
| cam6 | Jabra PanaCast 20 | 1920 × 1080 | 30 | 4 000 kbps | Back-middle rear overview, mounted **upright** |
| P50 | Jabra PanaCast 50 | 1280 × 720 | 30 | 2 500 kbps | Wide-angle room view, mounted **upright** |

- All cameras require **USB 3.0 SuperSpeed**; separate root hubs recommended.
- When multiple P20s share the same Windows display name, each is identified via
  a unique DirectShow PnP path (`video_alt_name` in the config JSON).
- Intelligent zoom is disabled at capture start (`--keep-zoom` to override).

### 1.2  Audio

| Label | Device | Channels | Rate | Interface |
|-------|--------|----------|------|-----------|
| `dpa_an1_aud` | DPA d:fine CORE 4066 | 1 | 48 kHz | RME Fireface 802 (Analog 1+2) |
| `dpa_mic9_aud` – `dpa_mic12_aud` | DPA d:fine CORE 4066 | 1 each | 48 kHz | RME Fireface 802 (Analog 9–12) |

Camera-embedded microphone audio is captured **separately** (`mux_audio: false`)
to avoid USB bandwidth contention and preserve video sync stability.

### 1.3  Desk Layout

```
            cam2          cam3
              \            /
               +----------+
               |  desk    |  ← persons 2+3 face cam2/cam3
               |          |
               +----------+
              /      |      \
            cam1   cam6   cam4     ← persons 0+1 face cam1/cam4
              cam5 (front-center)   P50 (overhead / shared view)
```

Each P20 sees four people: two from the front (its zone targets) and two from
behind (the opposite zone).  The front-facing filter rejects back-of-head
detections before matching.

---

## 2  Recording

### 2.1  Config

[configs/ffmpeg_multicap.json](../configs/ffmpeg_multicap.json) — one entry per device:

```jsonc
{
  "session_dir": "./data/sub-<id>/ses-<date>",
  "devices": [
    {
      "label": "jabra_panacast_20_cam1_vid",
      "width": 1920, "height": 1080, "fps": 30,
      "video_bitrate": 4000,
      "format": "mkv", "subdir": "video",
      "force_wallclock_timestamps": true,
      "mux_audio": false,
      "input_video_codec": null,
      "video_alt_name": "@device_pnp_\\\\?\\usb#vid_0b0e&pid_3020&…",
      "audio_alt_name": "@device_cm_…\\wave_{…}"
    }
    // … cam2, cam3, cam4, P50, DPA mics
  ]
}
```

Key settings:
- `force_wallclock_timestamps: true` — system clock, not device clock.
- `mux_audio: false` — video-only MKV; audio captured as separate WAV.
- `input_video_codec: null` — auto-negotiate (cameras expose YUY2/NV12).

### 2.2  Command

```bash
python tools/ffmpeg_multicap.py \
  --config configs/ffmpeg_multicap.json \
  --frame-log --record-lsl --stabilization-delay 2.0
```

| Flag | Purpose |
|------|---------|
| `--frame-log` | Per-frame PTS + Unix time via ffmpeg `showinfo` filter → `frame_logs/` |
| `--record-lsl` | LSL progress streams → `lsl/` (10 Hz per camera) |
| `--stabilization-delay 2.0` | Wait 2 s after ffmpeg starts to let USB handshake settle |

### 2.3  Output Structure

```
<session>/
├── video/
│   ├── jabra_panacast_20_cam1_vid_video.mkv
│   ├── jabra_panacast_20_cam2_vid_video.mkv
│   ├── jabra_panacast_20_cam3_vid_video.mkv
│   ├── jabra_panacast_20_cam4_vid_video.mkv
│   ├── jabra_panacast_50_vid_video.mkv
│   └── ffmpeg_multicap_events.jsonl        ← lifecycle events
├── audio/
│   ├── dpa_an1_aud.wav
│   ├── dpa_mic9_aud.wav … dpa_mic12_aud.wav
├── frame_logs/
│   └── {label}_frames.jsonl                ← per-frame anchors (Tier 1)
├── lsl/
│   ├── ffmpeg_progress_{label}.jsonl       ← 10 Hz progress (Tier 2)
│   └── ffmpeg_clock.jsonl                  ← unified LSL clock
└── sourcedata/sync/
    └── {label}_ffmpeg_progress.tsv         ← progress TSV (Tier 3)
```

---

## 3  Synchronisation

### 3.1  Sync Tiers (best → worst)

The 3D reconstruction pipeline (`multicam_pose3d.py`) auto-selects the
highest-priority tier that covers all cameras.

| Tier | Source | File pattern | Method | Accuracy |
|------|--------|--------------|--------|----------|
| 1 | Frame logs | `frame_logs/{label}_frames.jsonl` | median(`unix_time_s − pts_time`) over 30 samples | ~0.5 ms MAD |
| 2 | LSL progress | `lsl/ffmpeg_progress_{label}.jsonl` | median(`stream_time − out_time_sec`) @ ~10 Hz | ~1 ms |
| 3 | Progress TSV | `sourcedata/sync/{label}_ffmpeg_progress.tsv` | median(`host_time_sec − out_time_sec`) | ~1 ms |
| 4 | Events JSONL | `video/ffmpeg_multicap_events.jsonl` | single `capture_started` unix timestamp | ~100 ms |

Each record contains: `unix_time_s`, `unix_time_ns`, `lsl_time`, `monotonic_ns`, ISO timestamp.

### 3.2  Real-World Offsets (`ses-20260202_test`, Tier 1)

```
cam_1 (cam2):  reference (latest start)
cam_0 (cam1):  skip  5 frames  (0.167 s)
cam_3 (cam4):  skip  7 frames  (0.233 s)
cam_2 (cam3):  skip  8 frames  (0.267 s)
cam_4 (P50):   skip  9 frames  (0.300 s)
```

Frame-log MAD: 0.50–0.74 ms across all cameras.  After alignment, visual sync
is ±1 frame (imperceptible at 30 fps).

### 3.3  DPA Audio Synchronisation

#### 3.3.1  Recording Desync (Root Cause)

Each DPA mic pair (e.g. `dpa_mic9_aud` + `dpa_mic10_aud`) shares the same
RME Fireface 802 DirectShow device (e.g. `Analog (9+10)`).  However,
`ffmpeg_multicap.py` launches a **separate ffmpeg process** per mic — each
process opens its own DirectShow handle and negotiates the audio buffer
independently.  This causes:

- **50–250 ms start-time jitter** between mics (measured via progress-TSV
  anchor offsets across all sessions).
- **0.5 s length difference** in ~50 % of sessions (one DirectShow buffer
  boundary = 24 000 samples at 48 kHz).
- **grp-13**: 27 s desync due to a mid-session ffmpeg restart.
- **Audio-clock drift** of ~0.04 ms/s relative to the LSL/XDF clock —
  a single median anchor introduces 50–100 ms error for tasks at the start
  or end of a 64-minute recording.

**Future fix for `ffmpeg_multicap.py`:** Capture each stereo pair with one
ffmpeg process writing a 2-channel WAV, then split to mono in post-processing.
This would eliminate the DirectShow contention entirely.

#### 3.3.2  Post-Hoc Alignment (`xdf_sync_pipeline.py`)

`split_av_audio_by_windows()` corrects all of the above at processing time.

**Anchor computation — three methods (best → fallback):**

| Method | Source | When used | Clock |
|--------|--------|-----------|-------|
| **XDF streams** | `ffmpeg_progress_dpa_mic*_aud` streams inside the EEG `.xdf` | AV XDF intact, streams present | Unified LSL (per-sample clock correction by pyxdf) |
| **JSONL logs** | `<capture>/lsl/ffmpeg_progress_dpa_*.jsonl` `received_time` field | AV XDF corrupted/missing | Wall clock of LSL recorder converted via `wall_epoch − wall_minus_xdf_lsl` |
| **Progress TSV** | `sourcedata/sync/{label}_ffmpeg_progress.tsv` + `ffmpeg_clock` bridge | No XDF streams and no JSONL | AV host clock bridged via `ffmpeg_clock` median offset |

Each method fits a **linear regression** `anchor(t) = slope·t + intercept`
across the full recording to capture audio-clock drift. RMSE of residuals is
computed for each mic; the method with lower RMSE wins per mic when multiple
methods are available.

**Empirical drift (grp-12 `ses-20260318`, XDF method):**

| Mic | Slope (ms/s) | RMSE |
|-----|-------------|------|
| mic9 | 0.043 | < 1 ms |
| mic10 | 0.043 | < 1 ms |
| mic11 | 0.041 | < 1 ms |
| mic12 | 0.042 | < 1 ms |

**ffmpeg seeking — sample-accurate clips:**

All splits use `-ss` **after** `-i` (decode-then-seek) and re-encode to
`pcm_s16le`. This is lossless (PCM→PCM) and eliminates the ~35 ms packet
boundary error introduced by `-ss before -i -c copy`.

**Alignment rules:**

1. **Per-mic `-ss`** — each mic's `anchor(t)` is evaluated at the task's XDF
   start time, since each mic genuinely started at a different real-world moment.
2. **Common `-t`** — all mics for a task share the same duration (equal length
   clips).
3. **Negative start handling** — when a task begins before any mic started
   recording, all mics are trimmed to the shortest usable common duration.

**Output quality (verified):** `pcm_s16le, 48000 Hz, 16-bit, mono` — identical
codec to originals. RMS levels: −42 to −34 dBFS (all within normal speech range).

**Cross-correlation validation (grp-12):**

| Pair | Device | Drift over 64 min |
|------|--------|------------------|
| mic9 vs mic10 | Same DirectShow device (Analog 9+10) | 0.4 ms |
| mic11 vs mic12 | Same DirectShow device (Analog 11+12) | 3.8 ms |
| mic9/10 vs mic11/12 | Different devices | 10–20 ms |

Constant offsets between mics (60–236 ms) are corrected by the per-mic anchors.

#### 3.3.3  JSONL Fallback (Corrupted AV XDF)

For sessions where the AV XDF is too corrupted for pyxdf to recover streams
(e.g. grp-15 `ses-20260319`), the pipeline falls back to the raw JSONL files
written by the LSL Lab Recorder.

Each JSONL line contains `received_time` (ISO wall-clock timestamp of the LSL
recorder machine) and `values[0]` (ffmpeg `out_time_sec`).  The conversion to
unified XDF time is:

```
xdf_t = datetime.fromisoformat(received_time).timestamp() − wall_minus_xdf_lsl
anchor_diff = xdf_t − out_time_sec
```

`wall_minus_xdf_lsl` is the `stimuli_wall_clock − xdf_lsl` offset computed from
the events TSV.  This aligns the JSONL data to the same clock as the EEG XDF.

Note: `received_time` reflects **network arrival jitter** (~500 ms stdev) rather
than the true sample time. The regression RMSE is correspondingly ~0.5 s, but
the **slope** (drift rate) remains accurate, so sync errors are bounded by the
residual jitter spread over the full recording length — typically < 5 ms.

#### 3.3.4  Session Processing Results

| Session | Method | Mics | Audio clips |
|---------|--------|------|-------------|
| grp-06 run01 | XDF streams | 4 | partial (16 files) |
| grp-07 | XDF streams | 4 | 20 WAVs (5 tasks) |
| grp-08 (Mar 13) | XDF streams | 4 | 20 WAVs |
| grp-08 (Mar 16) | XDF streams | 4 | 20 WAVs |
| grp-09 | XDF streams | 4 | 16 WAVs (T0 skipped — recording started after task onset) |
| grp-10 | XDF streams | 4 | 20 WAVs |
| grp-11 | XDF streams | 4 | partial (14 files) |
| grp-12 | XDF streams | 4 | 20 WAVs |
| grp-13 | XDF streams | 4 | 20 WAVs |
| grp-14 | XDF streams | 4 | 20 WAVs |
| grp-15 | JSONL fallback (AV XDF corrupted) | 4 | 20 WAVs |

See `_compute_dpa_anchors_from_xdf()`, `_compute_dpa_anchors_from_jsonl()`,
`_compute_dpa_anchors()`, `_select_best_dpa_anchors()`, and
`split_av_audio_by_windows()` in `tools/xdf_sync_pipeline.py`.

### 3.4  Verify Sync

```bash
python tools/create_sync_test_video.py \
  --input <session>/video \
  --output <session>/video/sync_grid.mp4 \
  --layout 3x2 --source frame --pad-tail --cfr 30 \
  --dpa-audio dpa_mic9_aud
```

Produces a 3 × 2 grid (5 camera tiles + blank) with aligned audio, codec
`mpeg4 + yuv420p`.

---

## 4  Spatial Calibration

### 4.1  ChArUco Board

| Parameter | Value |
|-----------|-------|
| Board type | 5 × 3 (5 squares wide, 3 high) |
| Square size | 52 mm (must measure your print) |
| ArUco marker | 80 % of square side |
| Dictionary | 4 × 4_50 (OpenCV default) |

### 4.2  Five-Step Workflow

```bash
# 1. Print board
python tools/calibrate_charuco.py print-board --output charuco_board.png

# 2. Record ~60 s of board waving in view of all cameras
python tools/calibrate_charuco.py record \
  --config configs/ffmpeg_multicap.json --duration 60

# 3. Verify CharUco detection in sample frames
python tools/calibrate_charuco.py detect \
  --videos-dir <calibration>/video --board-type 5x3 --frames 20

# 4. Run anipose calibration
python tools/calibrate_charuco.py calibrate \
  --videos-dir <calibration>/video --square-size 52

# 5. Validate calibration quality
python tools/calibrate_charuco.py validate \
  --toml calibration_charuco.toml
```

Dependencies: `freemocap ≥ 1.3.0`, `opencv-python ≥ 4.8`, conda env
`affectai-freemocap`.

### 4.3  Output: TOML

Each camera section contains:

| Field | Description |
|-------|-------------|
| `name` | Camera label (matches video filename stem) |
| `size` | `[width, height]` in pixels |
| `matrix` | 3 × 3 intrinsic camera matrix (focal length, principal point) |
| `distortions` | 5 radial/tangential coefficients (critical for Jabra wide-angle, \|dist\| up to 1.19) |
| `rotation` | Rodrigues 3-vector (extrinsic) |
| `translation` | `[x, y, z]` in mm (extrinsic) |
| `world_orientation` | 3 × 3 rotation matrix |
| `world_position` | `[x, y, z]` in mm (world frame) |

Example (cam1 = reference camera):

```toml
[cam_0]
name = "jabra_panacast_20_cam1_vid_video"
size   = [1920, 1080]
matrix = [[919.05, 0, 959.5], [0, 919.05, 539.5], [0, 0, 1]]
distortions = [-0.647, 0, 0, 0, 0]
rotation    = [0, 0, 0]
translation = [0, 0, 0]

[metadata]
charuco_square_size = 52.0
date_time_calibrated = "2026-02-25T15:54:04"
```

---

## 5  2D Pose Estimation

Tool: `tools/test_mediapipe_pose.py`

```bash
python tools/test_mediapipe_pose.py \
  --session-dir <session> --max-frames 0 \
  --write-json <session>/mediapipe
```

- Uses **MediaPipe Tasks API** (v0.10.32) with `num_poses=4`.
- Outputs **OpenPose-compatible BODY_25 JSON** per frame per camera.
- Detection rate: cam1 62 %, cam2 94 %, cam3 95 %, cam4 86 %, P50 90 %
  (overall 85.4 %).

---

## 6  3D Reconstruction

Tool: `tools/multicam_pose3d.py`

### 6.1  Pipeline Steps

1. **Auto-map** pose-JSON directories → calibration cameras via TOML `name`.
2. **Align frames** using multi-tier sync offsets (§ 3.1).
3. **Undistort** 2D keypoints with calibration distortion coefficients.
4. **Filter back-facing** detections (Nose + Eye confidence ≥ 0.3).
5. **Zone-aware person matching** via epipolar geometry; shared cameras
   (P50) auto-assigned per frame.
6. **DLT triangulate** with reprojection-error filtering.
7. **Emit per-frame QC** (reprojection error, camera count, confidence).

### 6.2  Command

```bash
python tools/multicam_pose3d.py reconstruct \
  --calibration <session>/video/video_camera_calibration.toml \
  --pose-root <session>/mediapipe \
  --camera-zones "cam1+cam4:0,1" "cam2+cam3:2,3" \
  --frame-log-dir <session>/frame_logs \
  --lsl-dir <session>/lsl \
  --session-dir <session> \
  --events-jsonl <session>/video/ffmpeg_multicap_events.jsonl \
  --output <session>/skeleton_3d_synced.npy --fps 30
```

### 6.3  Key CLI Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--calibration` | — | Path to `.toml` |
| `--pose-root` | — | Root with `*_json/` sub-dirs |
| `--session-dir` | — | Auto-discover calibration + sync artifacts |
| `--camera-zones` | — | Zone-aware matching (e.g. `cam1+cam4:0,1 cam2+cam3:2,3`) |
| `--frame-log-dir` | — | Tier 1 sync source |
| `--lsl-dir` | — | Tier 2 sync source |
| `--events-jsonl` | — | Tier 4 sync source |
| `--max-epipolar-px` | 40 | Epipolar distance threshold (px) |
| `--max-reproj-px` | 30 | Reprojection error threshold (px) |
| `--no-front-facing-filter` | off | Disable back-of-head rejection |
| `--min-face-conf` | 0.3 | Face keypoint confidence for front-facing check |
| `--fps` | 30 | Target frame rate |

### 6.4  Output Format

```
skeleton_3d.npy — shape (F, P, 25, 7)
```

| Dim | Meaning |
|-----|---------|
| F | Number of synchronised frames |
| P | Number of people (fixed by zone config, e.g. 4) |
| 25 | BODY_25 keypoints (OpenPose convention) |
| 7 | `[x, y, z, confidence, reproj_error, n_cameras, group_id]` |

Companion `skeleton_3d.json` contains QC summary (reproj stats, valid joint %,
camera counts).

---

## 7  Post-Processing

### 7.1  Skeleton Refinement

Tool: `tools/refine_skeleton_3d.py` — four-stage pipeline:

| Stage | What it does | Key parameter |
|-------|-------------|---------------|
| 1. Quality gate | NaN-out low-confidence, high-reproj, few-camera joints | `--min-confidence 0.3`, `--max-reproj 20`, `--min-cameras 2` |
| 2. Velocity filter | NaN-out impossibly fast joint motions | `--max-velocity 300` mm/frame |
| 3. Gap interpolation | Cubic spline (≥ 4 anchors) or linear fill | `--max-gap 10` frames |
| 4. Temporal smoothing | 2nd-order Butterworth low-pass | `--smooth-cutoff 6.0` Hz |

```bash
python tools/refine_skeleton_3d.py \
  --input <session>/skeleton_3d_synced.npy \
  --upper-body --fps 30
```

Output: `*_refined.npy` + `*_refined.json` (per-stage QC stats).

### 7.2  Layout Video

Tool: `tools/layout_video_3d.py` — composites 5 camera tiles + 3D skeleton in
a 3 × 2 grid.

```bash
python tools/layout_video_3d.py \
  --session <session> --flip-p20 --upper-body \
  --output <session>/layout_3d.mp4
```

| Flag | Purpose |
|------|---------|
| `--flip-p20` | Rotate P20 feeds 180° (cameras are upside-down) |
| `--upper-body` | Render only head + spine + arms (no legs) |

Multi-person colours: person 0 = green, 1 = red, 2 = blue, 3 = yellow.

---

## 8  End-to-End Pipeline Summary

```
┌──────────────────────────────────────────────────────────────┐
│ 1. RECORD                                                    │
│    ffmpeg_multicap.py --frame-log --record-lsl               │
│    → .mkv + .wav + frame_logs/ + lsl/ + events.jsonl         │
├──────────────────────────────────────────────────────────────┤
│ 2. VERIFY SYNC                                               │
│    create_sync_test_video.py --source frame --layout 3x2     │
│    → sync_grid.mp4 (visual check)                            │
├──────────────────────────────────────────────────────────────┤
│ 3. CALIBRATE  (once per camera setup change)                 │
│    calibrate_charuco.py  print-board → record → detect →     │
│                          calibrate → validate                │
│    → calibration.toml (intrinsics + extrinsics per camera)   │
├──────────────────────────────────────────────────────────────┤
│ 4. 2D POSE ESTIMATION  (per session)                         │
│    test_mediapipe_pose.py --session-dir … --write-json …     │
│    → *_json/ directories (BODY_25 keypoints per frame)       │
├──────────────────────────────────────────────────────────────┤
│ 5. 3D RECONSTRUCTION                                         │
│    multicam_pose3d.py --camera-zones … --frame-log-dir …     │
│    → skeleton_3d.npy  (F × P × 25 × 7)                      │
├──────────────────────────────────────────────────────────────┤
│ 6. REFINEMENT                                                │
│    refine_skeleton_3d.py --upper-body                        │
│    → skeleton_3d_refined.npy                                 │
├──────────────────────────────────────────────────────────────┤
│ 7. VISUALISATION                                             │
│    layout_video_3d.py --flip-p20 --upper-body                │
│    → layout_3d.mp4  (5 tiles + 3D skeleton)                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 9  QC Results (`ses-20260202_test`, 4-person, synced)

| Metric | Value |
|--------|-------|
| Synchronised frames | 1 566 |
| Sync tier | Frame-log (Tier 1), MAD 0.5–0.7 ms |
| Camera offsets | 0 – 9 frames (0 – 0.3 s) |
| Reproj (accepted) mean | 13.2 px |
| Reproj 95th percentile | 18.8 px |
| Valid joints | 16 % |
| Cameras per joint | 2.3 mean |
| People detected | 4.0 mean |
