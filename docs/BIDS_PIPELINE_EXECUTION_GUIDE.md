# BIDS Processing Pipeline — Complete Execution Guide

**Pipeline**: Multimodal BIDS Processing with Configuration Support  
**Version**: 2.0  
**Date**: 2026-03-27  
**Status**: Ready for deployment

---

## What This Pipeline Does

Processes raw multimodal data (XDF, video, Tobii, physiological, stimuli) into a **clean BIDS dataset** with:

1. ✅ **Multi-source merging** — Combines Recording-PC (XDF/LSL), AV-PC (video/audio), Tobii, Stimuli
2. ✅ **Task windowing** — Derives task boundaries (T0–T4) from experiment markers
3. ✅ **Synchronization** — Validates 4-tier sync hierarchy, generates sync metadata
4. ✅ **BIDS canonicalization** — Organizes into modality folders (et/, physio/, audio/, video/, beh/, annot/)
5. ✅ **Stimuli annotations** — Task windows, participant responses, signal mapping
6. ✅ **Cleanup** — Removes raw sourcedata, retains only processed outputs
7. ✅ **Parallel processing** — Uses configurable worker pool for fast execution

---

## Files Created

### Pipeline Script
- **`tools/bids_processing_pipeline.py`** (700+ lines)
  - Main orchestrator with inventory-driven session discovery
  - Configuration file loading (EmotiBit mapping, camera specs)
  - Parallel multiprocessing with worker pool
  - Comprehensive error handling and progress logging
  - Detailed BIDS compliance reporting

### Documentation
- **`docs/bids_processing_config_guide.md`** (500+ lines)
  - Complete source location strategy
  - Configuration file formats and usage
  - Stimuli annotation details
  - Name matching rules and validation
  - Troubleshooting guide

- **`docs/BIDS_PIPELINE_QUICKREF.md`** (200 lines)
  - One-page quick reference
  - Session matching rules
  - CLI examples
  - Troubleshooting checklist

---

## System Requirements

**Python Environment**: `affectai` conda environment
- Python 3.10.19
- NumPy, Pandas, SciPy
- PyYAML, jsonschema (for config validation)
- subprocess (stdlib) for tool orchestration

**Disk Space**: 200–500 GB for 33 sessions (depending on `--split-media`)

**Time**: ~2–4 hours for all 33 sessions (4 workers, typical hardware)

---

## Step 1: Verify Data Structure

Before running the pipeline, ensure your data is organized correctly:

```powershell
# Check Recording-PC structure
ls "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\data\affectai-capture-recording\sessions\"

# Expected:
# final/sub-01/ses-*
# pilot/sub-01/ses-*
# test/sub-01/ses-*

# Check AV-PC structure
ls "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\data\AV\"

# Expected:
# final/ses-* or *grp-*
# pilot/ses-* or *grp-*
# test/ses-* or *grp-*

# Check inventory CSV
head "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\data\high_level_session_inventory.csv"

# Expected columns: session, group_id, phase_tags, participants_ids, raw_modalities
```

---

## Step 2: Verify Configuration Files

```powershell
# Check EmotiBit participant mapping
cat "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\configs\emotibit_participants_by_source.json" | ConvertFrom-Json | ConvertTo-Json

# Expected:
# {
#   "participants": {"P1": "MD-V7-...", ...},
#   "by_source": {"192.168.10.201": "P1", ...}
# }

# Verify all required configs present
ls "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed\configs\*.json"
```

---

## Step 3: Run the Pipeline

### Activate Environment

```powershell
conda activate affectai
```

### Execute Pipeline

```powershell
cd "D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed"

python tools/bids_processing_pipeline.py `
  --data-root . `
  --output-root "E:\processed_data" `
  --inventory "data/high_level_session_inventory.csv" `
  --config-dir "configs" `
  --max-workers 4 `
  --split-media `
  --link-files
```

### Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--data-root` | `.` (or full path) | Root containing data sources |
| `--output-root` | `E:\processed_data` | BIDS output location |
| `--inventory` | `data/high_level_session_inventory.csv` | Inventory CSV file |
| `--config-dir` | `configs` | Configuration directory |
| `--max-workers` | `4` | Parallel worker count (adjust to CPU count) |
| `--split-media` | flag | Create per-task video/audio clips |
| `--link-files` | flag | Use hard links to save disk space |

### Expected Output

```
2026-03-27 15:30:20 [INFO] Loading sessions from data/high_level_session_inventory.csv
2026-03-27 15:30:21 [INFO] Loaded 33 sessions
2026-03-27 15:30:21 [INFO] 
================================================================================
PIPELINE CONFIGURATION
================================================================================
Data root:        D:\AffecAI Data\affectai-data-processing\affectai-data-processing-seed
Output root:      E:\processed_data
Config dir:       configs
Inventory:        data/high_level_session_inventory.csv
Workers:          4
Split media:      True
Link files:       True
Sessions to process: 33
================================================================================

2026-03-27 15:30:22 [INFO] Starting processing with 4 workers...
2026-03-27 15:30:22 [INFO] Pipeline: Merge → Sync → Split Tasks → BIDS → Clean
2026-03-27 15:30:22 [INFO]
2026-03-27 15:30:22 [INFO] Session[ses-20260311_grp-06_run01] Starting: ses-20260311_grp-06_run01 (grp-06, phase=final)
...
[Processing in parallel across 4 workers...]
...
2026-03-27 17:45:30 [INFO]
================================================================================
BIDS PROCESSING PIPELINE - FINAL REPORT
================================================================================

📊 PROCESSING STATISTICS
  Sessions processed:  33
  ✓ Successful:        33
  ✗ Failed:            0
  Success rate:        100.0%

📁 OUTPUT STATISTICS
  Processed files:     12450
  Raw files removed:   5230
  Duration:            2.35 h (141 min)
  Avg per session:     256 s

📍 DATA LOCATIONS
  Input sources:       D:\...\affectai-data-processing-seed
  BIDS output:         E:\processed_data
  Config directory:    configs

... [full report with modality details, stimuli annotations, etc.] ...

================================================================================

✨ Next steps:
  1. Validate BIDS compliance: bids-validator E:\processed_data
  2. Review sync_metadata.json for timing information
  3. Check events.tsv for complete task timeline
  4. Optional: Run downstream pipelines (3D pose, face/hand, etc.)
================================================================================
```

---

## Step 4: Validate Output

### Check BIDS Compliance

```powershell
# If bids-validator installed
bids-validator "E:\processed_data"

# Expected: 0 errors (or only warnings about optional fields)
```

### Inspect Sync Metadata

```powershell
# View first session sync metadata
Get-Content "E:\processed_data\sub-01\ses-20260311_grp-06_run01\annot\sub-01_ses-20260311_grp-06_run01_sync_metadata.json" | ConvertFrom-Json

# Expected fields:
# {
#   "session_id": "ses-20260311_grp-06_run01",
#   "group_id": "grp-06",
#   "participants": ["sub-01", "sub-02", "sub-03", "sub-04"],
#   "sync_sources": { "recording_pc_xdf": true, "av_pc_video": false, ... },
#   "tasks": ["T0", "T1", "T2", "T3", "T4"],
#   "sync_tiers": [ ... ]
# }
```

### Check Task Windows

```powershell
# View task window annotations
Get-Content "E:\processed_data\sub-01\ses-20260311_grp-06_run01\annot\sub-01_ses-20260311_grp-06_run01_task-T0T1T2T3T4_task_run_windows.tsv" | Select-Object -First 10

# Expected:
# task    run    start_time    end_time    description
# T0      01     0.0           60.5        Baseline/calibration
# T1      01     61.2          450.8       Hidden-Profile Decision
# ...
```

### Verify Participant Signal Map

```powershell
# View participant-modality mapping
Get-Content "E:\processed_data\sub-01\ses-20260311_grp-06_run01\annot\sub-01_ses-20260311_grp-06_run01_participant_signal_map.tsv"

# Expected:
# participant    et_signal           physio_signal                audio_signal
# sub-01         Tobii_P1_Gaze       EmotiBit_MD-V7-0001141      dpa_mic1
# sub-02         Tobii_P2_Gaze       EmotiBit_MD-V7-0001160      dpa_mic2
# ...
```

### List Generated Modalities

```powershell
# Count files per modality
(ls "E:\processed_data\sub-01\ses-20260311_grp-06_run01\et\" -Recurse | Measure-Object).Count
(ls "E:\processed_data\sub-01\ses-20260311_grp-06_run01\physio\" -Recurse | Measure-Object).Count
(ls "E:\processed_data\sub-01\ses-20260311_grp-06_run01\audio\" -Recurse | Measure-Object).Count
(ls "E:\processed_data\sub-01\ses-20260311_grp-06_run01\video\" -Recurse | Measure-Object).Count

# Expected: non-zero counts for available modalities
```

---

## Step 5: Handle Errors (if any)

### Common Issues & Solutions

#### "No Recording or AV sources found"

**Check**:
```powershell
# Verify session exists in inventory
grep "ses-20260311_grp-06" "data/high_level_session_inventory.csv"

# Check directory structure
ls -Path "data/affectai-capture-recording/sessions/" -Include "ses-*" -Recurse | Select -First 5

# Verify phase matches
# (if CSV says "final" but directory is "pilot", that's the issue)
```

**Fix**: Update CSV phase_tags or move data to correct phase directory

---

#### "multisource_to_bids_runs failed"

**Check**:
```powershell
# Verify events.tsv present (needed for task markers)
ls "data/affectai-capture-recording/sessions/*/sub-01/ses-*/events.tsv"

# Verify frame logs for AV data
ls "data/AV/*/frame_logs/*"
```

**Fix**: Ensure events.tsv is present in Recording-PC data

---

#### "Stimuli annotations missing"

**Check**:
```powershell
# Verify stimuli directory exists
ls "data/stimuli/"

# Check group_id matches stimuli directory name
ls "data/stimuli/final/*grp-06*"
```

**Fix**: Create stimuli directory if missing, or match group_id in inventory

---

## Optional: Run Downstream Pipelines

After BIDS processing, optional pipelines are available:

### 3D Skeleton Reconstruction

```powershell
python tools/video_only_3d_pipeline.py `
  --videos-dir "E:\processed_data\sub-01\ses-20260311_grp-06_run01\video" `
  --calibration "E:\processed_data\sub-01\ses-20260311_grp-06_run01\video\video_camera_calibration.toml" `
  --output-dir "E:\processed_data\sub-01\ses-20260311_grp-06_run01\mocap" `
  --refine-skeleton
```

### 3D Pose QC Report

```powershell
python tools/qc/qc_sync_report.py `
  --session-dir "E:\processed_data\sub-01\ses-20260311_grp-06_run01" `
  --output-dir "E:\processed_data\sub-01\ses-20260311_grp-06_run01\qc"
```

---

## Summary

| Stage | Tool | Input | Output | Time |
|-------|------|-------|--------|------|
| **Merge sources** | `multisource_to_bids_runs.py` | Raw XDF/MKV/Tobii | Merged BIDS tree | ~1 min |
| **Canonicalize** | `raw_to_bids.py` | Merged sourcedata/ | BIDS modality folders | ~2 min |
| **Sync validation** | (internal) | LSL + video sync data | sync_metadata.json | <10 sec |
| **Cleanup** | (internal) | sourcedata/ | (deleted) | <1 sec |
| **Per session total** | — | — | — | ~4 min |
| **33 sessions (4 workers)** | — | — | — | ~2.5 h |

---

## Reference

- **Full config guide**: [docs/bids_processing_config_guide.md](bids_processing_config_guide.md)
- **Quick reference**: [docs/BIDS_PIPELINE_QUICKREF.md](BIDS_PIPELINE_QUICKREF.md)
- **Architecture**: [docs/architecture.md](architecture.md)
- **Data flow**: [docs/data_flow.md](data_flow.md)

---

**Status**: ✅ Ready to execute. All configurations validated, sourcing strategy tested.
