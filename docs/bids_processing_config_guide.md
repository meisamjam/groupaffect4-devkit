# BIDS Processing Pipeline — Configuration & Stimuli Annotation Guide

**Document**: Guide for using configuration files in the BIDS processing pipeline  
**Updated**: 2026-03-27  
**Relevant tools**: `bids_processing_pipeline.py`, `multisource_to_bids_runs.py`, `raw_to_bids.py`

---

## Overview: Session Matching & Name Binding

The pipeline locates and processes multimodal data from multiple source directories using **inventory-driven name matching**. Each session is identified by a unique combination of:

- `session_id`: e.g., `ses-20260311_grp-06_run01`
- `group_id`: e.g., `grp-06`
- `phase`: e.g., `final`, `pilot`, `test`
- `participants`: e.g., `["sub-01", "sub-02", "sub-03", "sub-04"]`

---

## 1. Inventory File Structure

**File**: `data/high_level_session_inventory.csv`

### Key Columns

```csv
session                | group_id | phase_tags        | recording_session_names    | av_session_names    | participants_ids                    | raw_modalities
ses-20260311_grp-06... | grp-06   | final;test        | ses-20260311_grp-06_run01 | (empty)             | sub-01;sub-02;sub-03;sub-04       | lsl;tobii_lsl
ses-20260309_grp-04... | grp-04   | pilot             | ses-20260309_grp-04_run01 | ses-20260309_grp-04 | sub-01;sub-02;sub-03;sub-04       | av;tobii_lsl
```

### Parsing Rules

- **session**: Direct session identifier
- **group_id**: Used to match AV-PC and Stimuli sources
- **phase_tags**: Semicolon-separated; first is primary phase for directory traversal
- **participants_ids**: Semicolon-separated BIDS subject IDs; defaults to `sub-01;sub-02;sub-03;sub-04`
- **raw_modalities**: Available data types; used for validation

---

## 2. Session Source Location Strategy

### Matching Algorithm

For each session from inventory:

#### **Recording-PC Sources**
```
Search path: data/affectai-capture-recording/sessions/{primary_phase}/sub-01/
Pattern:    ses-{session_id}*
Example:    data/affectai-capture-recording/sessions/final/sub-01/ses-20260311_grp-06_run01/
```

**Contents**:
- `*.xdf` — LabRecorder output (all LSL streams merged)
- `sourcedata/tobii_lsl/*.ndjson` — Tobii LSL bridge backups
- `sourcedata/emotibit_lsl/` — EmotiBit LSL bridge backups (if available)
- `events.tsv` — Authoritative experiment timeline with markers
- `sourcedata/sync/` — Frame logs, progress TSV, LSL clock data

**Sources detected**:
- LSL streams (XDF format)
- Tobii glasses data (via LSL)
- EmotiBit physiological data (via LSL)
- Experiment markers (via LSL or events.tsv)

---

#### **AV-PC Sources**
```
Search path: data/AV/{primary_phase}/
Pattern:    *{group_id}*
Example:    data/AV/final/ses-20260311_grp-06_run01/ or similar directory
```

**Contents**:
- `*.mkv` — Camera video files (PanaCast 20/50, numbered cam1–cam6 or P50)
- `*.wav` — DPA microphone audio (numbered mic1–mic5)
- `frame_logs/*.jsonl` — Per-frame timing anchors (unix_time, pts_time)
- `sourcedata/sync/*.tsv` — Progress streams (camera frame sync)
- `lsl/*.jsonl` — LSL progress streams (ffmpeg_progress_*.jsonl)

**Sync tier info**:
- Tier 1: Frame logs (~0.5 ms accuracy)
- Tier 2: LSL progress streams (~1 ms accuracy)
- Tier 3: Progress TSV (~1 ms accuracy)
- Tier 4: Events JSONL (~100 ms accuracy)

---

#### **Tobii Sources (Manual Downloads)**
```
Search path: data/Tobii/
Pattern:    *{session_id}*
Example:    data/Tobii/20260311_grp-06_run01/ or /Tobii_P1_20260311_125935/
```

**Contents**:
- Scene video files (`.mp4`, `.mov`)
- Gaze recordings (`.g3`, raw Tobii format, or `.csv` exports)
- Timestamp mappings

**Device mapping**:
- Loaded from `configs/emotibit_participants_by_source.json` (for participant assignment)
- Maps IP addresses or device IDs to participant roles (P1, P2, P3, P4)

---

#### **Stimuli & Task Markers**
```
Search path: data/stimuli/{primary_phase}/
Pattern:    *{group_id}*
Example:    data/stimuli/final/20260311_grp-06_run01_20260311_125450/
```

**Contents**:
- Task marker logs (JSONL or TSV format)
- Task window definitions (T0: baseline, T1: task1, T2: task2, T3: task3, T4: task4)
- Tablet response logs (participant selections, ratings)
- Big-screen display events

**Task windows**:
- **T0**: Baseline/calibration period (30–60 seconds)
- **T1**: First experiment task (e.g., Hidden-Profile Decision)
- **T2**: Second task (e.g., Mini-Negotiation)
- **T3**: Third task (e.g., Idea Generation)
- **T4**: Fourth task (e.g., Public-Goods Game)

---

## 3. Configuration Files

### 3.1 EmotiBit Participant Mapping

**File**: `configs/emotibit_participants_by_source.json`  
**Purpose**: Map device/IP to participant role (P1–P4)

```json
{
  "participants": {
    "P1": "MD-V7-0001141",  // Device serial number
    "P2": "MD-V7-0001160",
    "P3": "MD-V7-0001409",
    "P4": "MD-V7-0000837"
  },
  "by_source": {
    "192.168.10.201": "P1",  // IP address mapping
    "192.168.10.202": "P2",
    "192.168.10.203": "P3",
    "192.168.10.204": "P4"
  }
}
```

**Usage in pipeline**:
1. Loaded automatically by `bids_processing_pipeline.py`
2. Passed to `raw_to_bids.py` via `--participant-map` argument
3. Used to assign EmotiBit LSL streams to correct participant

**Fallback**:
- If not found, uses default mapping P1→sub-01, P2→sub-02, etc.

---

### 3.2 Camera Specifications

**File**: `configs/camera_specs.json`  
**Purpose**: Camera model, resolution, FPS, mounting details

```json
{
  "cam1": {
    "model": "Jabra PanaCast 20",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "mounting": "upside-down",
    "zone": ["P1", "P2"]  // Participant viewing zones
  },
  "cam5": {
    "model": "Jabra PanaCast 50",
    "width": 1280,
    "height": 720,
    "mounting": "upright",
    "view": "overview"
  }
}
```

**Usage**:
- Referenced by 3D pose pipeline (calibration, camera matrices)
- Defines orientation corrections (flip/rotate) for inverted cameras
- Specifies participant "zones" for person identity stabilization

---

### 3.3 Desk Zones & Marker Maps

**File**: `configs/desk_zones.json`  
**Purpose**: Seating layout and participant positions relative to cameras

```json
{
  "zones": {
    "zone_1": {
      "cameras": ["cam1", "cam4"],
      "participants": ["P1", "P2"],
      "position": "front_left_right"
    },
    "zone_2": {
      "cameras": ["cam2", "cam3"],
      "participants": ["P3", "P4"],
      "position": "front_top_bottom"
    }
  }
}
```

**File**: `configs/desk_markers_large.yaml`  
**Purpose**: ChArUco calibration marker locations

---

### 3.4 Tobii Tracker Configuration

**File**: `configs/tobii_multicam_glasses_tracker.example.yaml`  
**Purpose**: Tobii glasses marker templates, gaze mapping parameters

---

## 4. Stimuli Annotations in BIDS Output

### 4.1 Task Window Annotations

**Output file**: `annot/sub-01_ses-{session_id}_task-T0T1T2T3T4_task_run_windows.tsv`

```tsv
task	run	start_time	end_time	start_frame	end_frame	description
T0	01	0.0	60.5	0	1830	Baseline/calibration
T1	01	61.2	450.8	1832	13524	Hidden-Profile Decision task
T2	01	451.5	720.3	13600	21609	Mini-Negotiation task
T3	01	721.0	1050.2	21630	31507	Idea Generation (NGT) task
T4	01	1051.0	1350.5	31630	40515	Public-Goods Micro-Game
```

**Columns**:
- `task`: T0–T4 identifier
- `run`: Run number (01 for single run)
- `start_time`, `end_time`: Absolute timestamps (seconds from session start)
- `start_frame`, `end_frame`: Corresponding frame numbers in primary sync tier
- `description`: Human-readable task name

**Source**: Derived from events.tsv markers (experiment control logs)

---

### 4.2 Stimuli Responses & Tablet Logs

**Output file**: `beh/sub-01_ses-{session_id}_stimuli_answers.tsv`

```tsv
participant	task	question_id	response	timestamp	device
sub-01	T1	decision_confidence	7	123.45	tablet_1
sub-01	T2	negotiation_score	45	201.23	tablet_1
sub-01	T3	idea_count	12	320.15	tablet_1
sub-01	T4	public_goods_choice	0.8	405.10	tablet_1
```

**Content**:
- All tablet responses (ratings, rankings, multiple-choice)
- VAD (Valence-Arousal-Dominance) probe responses
- System-detected events (e.g., big-screen display changes)

**Source**: Parsed from stimuli logs (JSONL or TSV from display_server.py)

---

### 4.3 Participant Signal Map

**Output file**: `annot/sub-01_ses-{session_id}_participant_signal_map.tsv`

```tsv
participant	et_signal	physio_signal	audio_signal	role
sub-01	Tobii_P1_Gaze	EmotiBit_MD-V7-0001141	dpa_mic1	P1
sub-02	Tobii_P2_Gaze	EmotiBit_MD-V7-0001160	dpa_mic2	P2
sub-03	Tobii_P3_Gaze	EmotiBit_MD-V7-0001409	dpa_mic3	P3
sub-04	Tobii_P4_Gaze	EmotiBit_MD-V7-0000837	dpa_mic4	P4
```

**Purpose**: Explicit mapping of raw stream names to participant identity

**Generated by**:
- `emotibit_participants_by_source.json` (physio)
- Device configuration and participant seating (et, audio)
- `desk_zones.json` (camera zones to participant assignment)

---

### 4.4 Synchronization Metadata

**Output file**: `annot/sub-01_ses-{session_id}_sync_metadata.json`

```json
{
  "session_id": "ses-20260311_grp-06_run01",
  "group_id": "grp-06",
  "participants": ["sub-01", "sub-02", "sub-03", "sub-04"],
  "modalities": ["lsl", "tobii_lsl"],
  "phase": "final",
  "sync_sources": {
    "recording_pc_xdf": true,
    "av_pc_video": false,
    "tobii_gaze": false,
    "stimuli_markers": true
  },
  "sync_tiers": [
    "1_frame_logs (best, ~0.5ms)",
    "2_lsl_progress (10Hz, ~1ms)",
    "3_progress_tsv (~1ms)",
    "4_events_jsonl (worst, ~100ms)"
  ],
  "tasks": ["T0", "T1", "T2", "T3", "T4"],
  "output_modalities": [
    "annot/ (task windows, sync maps, participant signal map)",
    "beh/ (events.tsv, per-task events, stimuli responses)",
    "et/ (Tobii gaze + pupil TSV)",
    "physio/ (EmotiBit PPG, EDA, temperature)",
    "audio/ (DPA close-talk + room, per-task clips)",
    "video/ (camera MKV, per-task clips if --split-media)"
  ],
  "processing_timestamp": "2026-03-27T15:30:42.123456",
  "pipeline_version": "2.0_multimodal_sync"
}
```

---

## 5. Pipeline Execution with Config Files

### CLI Usage

```bash
python bids_processing_pipeline.py \
  --data-root "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed" \
  --output-root "E:\processed_data" \
  --inventory "data/high_level_session_inventory.csv" \
  --config-dir "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\configs" \
  --max-workers 4 \
  --split-media \
  --link-files
```

### Arguments

- `--data-root`: Root containing source directories (affectai-capture-recording/, AV/, Tobii/, stimuli/)
- `--output-root`: BIDS output location
- `--inventory`: CSV file path
- `--config-dir`: Directory with `emotibit_participants_by_source.json`, camera specs, etc. (defaults to `{data-root}/configs`)
- `--max-workers`: Parallel worker pool size
- `--split-media`: Create per-task video/audio clips
- `--link-files`: Use hard links instead of copies

### Internal Flow

1. **Load inventory** → Parse CSV, extract session metadata
2. **Load configs** → Read `emotibit_participants_by_source.json`, log configuration
3. **Create pool** → Spawn N workers for parallel processing
4. **Per-session worker**:
   - **Find sources**: Match session against data directories (Recording-PC, AV-PC, Tobii, Stimuli)
   - **Stage 1**: Call `multisource_to_bids_runs.py` (merge, derive task windows, write sourcedata/)
   - **Stage 2**: Call `raw_to_bids.py` (canonicalize to BIDS, extract LSL streams)
   - **Stage 3**: Validate sync, generate `sync_metadata.json`
   - **Stage 4**: Cleanup sourcedata/, retain only processed outputs
5. **Aggregate results** → Count successes/failures, report timings and modalities

---

## 6. Name Matching Validation

### Successful Match Indicators

✅ **All sources found**:
```
Sources found: Recording=True, AV=True, Tobii=True, Stimuli=True
```

✅ **Inventory matches AV directory layout**:
- CSV `group_id`: `grp-06`
- Directory found: `data/AV/final/*grp-06*`

✅ **Stimulus annotations populated**:
```
✓ Sync metadata validated and saved
  → annot/sub-01_ses-20260311_grp-06_run01_sync_metadata.json
```

---

### Debugging Mismatches

**Symptom**: Sources not found for a session

```bash
# Check inventory CSV for correct group_id and phase
grep "ses-20260311_grp-06_run01" data/high_level_session_inventory.csv

# Verify directory structure
ls -R data/affectai-capture-recording/sessions/final/sub-01/ | grep "20260311_grp-06"
ls -R data/AV/final/ | grep "grp-06"
ls -R data/stimuli/final/ | grep "grp-06"
```

**Common issues**:
1. Phase mismatch: CSV says `final`, but directory is `pilot`
2. Group ID format: CSV has `grp-06`, but directory has `group-06` or `06`
3. Missing delimiter: Session not found because glob pattern doesn't match

---

## 7. Output Validation

### Check Stimuli Annotations

```bash
# Verify task windows were derived
head -20 processed_data/sub-01/ses-*/annot/*_task_run_windows.tsv

# Check participant signal mapping
cat processed_data/sub-01/ses-*/annot/*_participant_signal_map.tsv

# Inspect sync metadata (formatted JSON)
jq . processed_data/sub-01/ses-*/annot/*_sync_metadata.json | head -30
```

### Verify Participant-Modality Alignment

```bash
# List all processed files to confirm modality coverage
ls -la processed_data/sub-01/ses-*/et/
ls -la processed_data/sub-01/ses-*/physio/
ls -la processed_data/sub-01/ses-*/audio/
ls -la processed_data/sub-01/ses-*/beh/
```

### Confirm BIDS Compliance

```bash
bids-validator processed_data/
```

---

## 8. Troubleshooting

### "No Recording or AV sources found"

**Cause**: Neither Recording-PC (XDF) nor AV-PC (MKV/WAV) data located  
**Fix**: Check:
1. Inventory group_id matches directory name
2. Phase directory exists (final/pilot/test)
3. Session subdirectory follows `ses-YYYYMMDD_grp-XX_runYY` pattern

---

### "multisource_to_bids_runs failed"

**Cause**: Tool unable to merge sources or derive task windows  
**Fix**: Check:
1. events.tsv present in Recording-PC data (needed for task markers)
2. Frame logs exist for AV-PC data (needed for video sync)
3. EmotiBit config correctly maps participant roles

---

### "Stimuli annotations missing or sparse"

**Cause**: Stimuli directory not found or lacks marker logs  
**Fix**: Verify:
1. `data/stimuli/{phase}/{group_id}` exists
2. Contains JSONL or TSV files with task events
3. Event timestamps align with events.tsv timeline

---

## Summary

| Component | Config File | Loaded By | Purpose |
|-----------|------------|-----------|---------|
| **Participant mapping** | `emotibit_participants_by_source.json` | Pipeline main() | Assign EmotiBit/Tobii to sub-01–04 |
| **Camera info** | `camera_specs.json` | raw_to_bids.py, video_only_3d_pipeline.py | Calibration, orientation, zones |
| **Desk layout** | `desk_zones.json` | (optional) downstream 3D pipeline | Person identity stabilization |
| **Task windows** | events.tsv (+ stimuli logs) | multisource_to_bids_runs.py | Derive T0–T4 boundaries |
| **Stimuli responses** | stimuli/{phase}/{group_id}/*.jsonl | raw_to_bids.py | Populate beh/stimuli_answers.tsv |
| **Sync hierarchy** | (generated) | Pipeline | Create annot/sync_metadata.json |

All configuration files are optional but recommended for robust multimodal processing.
