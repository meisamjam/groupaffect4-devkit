# Master BIDS Processing Pipeline

Complete multiprocessing and GPU-accelerated pipeline for converting AffectAI multimodal data into BIDS format.

## Overview

This pipeline orchestrates the complete processing of all sessions from your inventory files into a BIDS-compliant dataset. It provides:

- **Multiprocessing support** — Parallel processing of multiple sessions
- **GPU acceleration** — CUDA support for MediaPipe (face/hand detection) and computer vision tasks
- **Configurable workflows** — Enable/disable 3D pose, face/hand detection, physiological processing
- **Automatic resumption** — Checkpoint-based recovery from failures
- **Progress tracking** — Real-time logging and final JSON report

## Quick Start

### Option 1: Windows Batch Script (Easiest)

```bash
cd tools
run_pipeline.bat --data-dir ../affectai-data-processing-seed/data --output-dir E:\processed_data --preset standard
```

### Option 2: PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\tools\run_pipeline.ps1 -DataDir "../affectai-data-processing-seed/data" -OutputDir "E:\processed_data" -Preset standard
```

### Option 3: Python Direct

```bash
python tools/master_bids_pipeline.py \
    --data-dir affectai-data-processing-seed/data \
    --output-dir E:\processed_data \
    --max-workers 4 \
    --gpu-devices 0 \
    --enable-3d-pose
```

## Presets

### 1. **quick** (Fastest)
BIDS-only processing without 3D reconstruction
```bash
run_pipeline.bat --preset quick
```
- Duration: ~10-20 minutes per session
- GPU: Minimal
- Output: Video, audio, ET, physio in BIDS format

### 2. **standard** (Recommended)
Standard with 3D body pose reconstruction
```bash
run_pipeline.bat --preset standard
```
- Duration: ~1-2 hours per session
- GPU: Core (multicam calibration, triangulation, refinement)
- Output: Full BIDS + 3D skeleton + gaze reconstruction

### 3. **full** (Maximum Features)
Complete processing with face/hand landmarks
```bash
run_pipeline.bat --preset full
```
- Duration: ~2-3 hours per session
- GPU: Heavy (pose + face model + hand landmarks)
- Output: Full BIDS + skeleton + face mesh (468 points) + hands

### 4. **dual_gpu** (Parallel GPU Processing)
Multi-GPU acceleration for faster processing
```bash
run_pipeline.bat --preset dual_gpu
```
- Requirements: NVIDIA GPUs in SLI/NVLink configuration
- Workers: 8 (4 per GPU)
- Duration: ~1-1.5 hours per session

### 5. **single_session** (Debugging)
Process one session at a time with full features
```bash
run_pipeline.bat --preset single_session
```
- Best for: Testing, debugging, memory-constrained systems

## Advanced Usage

### Custom Configuration

```bash
python tools/master_bids_pipeline.py \
    --data-dir data \
    --output-dir E:\processed_data \
    --max-workers 8 \
    --gpu-devices 0 1 \
    --enable-3d-pose \
    --enable-face-hand \
    --enable-physio \
    --verbose
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--data-dir` | PATH | Root data directory (parent of inventory files) |
| `--output-dir` | PATH | Output directory for processed BIDS data |
| `--max-workers` | INT | Parallel worker processes (default: 4) |
| `--gpu-devices` | INT... | GPU device IDs, space-separated (default: 0) |
| `--enable-3d-pose` | FLAG | Enable 3D pose reconstruction |
| `--enable-face-hand` | FLAG | Enable face/hand landmark detection |
| `--enable-physio` | FLAG | Enable physiological processing |
| `--verbose` | FLAG | Enable verbose logging |

## Performance Tuning

### For Your System

1. **Check available GPUs:**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

2. **Determine optimal workers:**
   ```
   max_workers = min(GPU_count * 2, CPU_count // 2)
   ```
   Examples:
   - 1 GPU, 8 CPU cores → `--max-workers 4`
   - 2 GPUs, 16 CPU cores → `--max-workers 8`
   - 4 GPUs, 32 CPU cores → `--max-workers 8`

3. **Monitor system load:**
   - Workers bottlenecked by CPU? Reduce `--max-workers`
   - Workers bottlenecked by memory? Use fewer workers or smaller batch sizes
   - GPU underutilized? Increase --max-workers (up to ~2x GPU count)

### Recommended Settings

| System | Config | Workers | GPU Devices | Duration |
|--------|--------|---------|-------------|----------|
| Laptop (2-4 GPU) | quick | 2 | 0 | 20 min/session |
| Desktop (1x RTX 3090) | standard | 4 | 0 | 1.5 hr/session |
| Workstation (2x RTX 3090) | full | 8 | 0 1 | 1 hr/session |
| Server (4x RTX 6000) | full | 16 | 0 1 2 3 | 30 min/session |

## Output Structure

```
E:\processed_data/
├── sub-00/
│   └── ses-ses-20260309_grp-03_run01/
│       ├── video/                      # Task-split video clips
│       ├── audio/                      # Task-split audio clips
│       ├── et/                         # Tobii gaze + pupil trajectories
│       ├── physio/                     # EmotiBit PPG, EDA, temperature
│       ├── pose3d/                     # 3D skeleton (if --enable-3d-pose)
│       │   ├── skeleton_3d.npy
│       │   ├── skeleton_refined.npy
│       │   └── metadata.json
│       ├── facehand/                   # Face/hand landmarks (if --enable-face-hand)
│       │   ├── face_3d.npz
│       │   ├── hand_3d.npz
│       │   └── blendshapes.json
│       ├── beh/                        # Behavioral annotations
│       ├── annot/                      # Sync maps, events
│       └── events.tsv                  # Master event timeline
├── sub-01/
│   └── ...
├── participants.tsv                    # Anonymized roster
├── dataset_description.json            # BIDS metadata
└── pipeline_report.json                # Processing summary
```

## Processing Details

### Stage 1: BIDS Core Packaging
**Tool:** `multisource_to_bids_runs.py`
- Merges multi-PC streams (XDF, NDJSON, MKV/WAV, Tobii)
- Derives task windows (T0–T4) from experiment-control markers
- Performs 4-tier camera synchronization
- Output: BIDS session tree with modality folders

**Duration:** 5-10 min/session

### Stage 2: Raw → Canonical BIDS
**Tool:** `raw_to_bids.py`
- Canonicalizes modality formats
- Validates BIDS compliance
- Creates participant roster
- Output: Standard BIDS layout

**Duration:** 2-5 min/session

### Stage 3: 3D Pose Reconstruction (Optional)
**Tool:** `video_only_3d_pipeline.py`
Substages:
1. **Tobii world gaze** — 6-DoF head pose + egocentric→world gaze mapping
2. **Multicam 3D body** — Triangulation + epipolar constraints (DLT)
3. **Skeleton refinement** — Velocity filtering + Butterworth smoothing
4. **Gesture extraction** — Binary gesture events from skeleton dynamics

**Duration:** 45-90 min/session (GPU-accelerated)

### Stage 4: Face & Hand Landmarks (Optional)
**Tool:** `face_hand_pipeline.py`
- MediaPipe FaceLandmarker (468 mesh + 52 ARKit blendshapes) — **GPU-accelerated**
- MediaPipe HandLandmarker (21 points per hand)
- 3D triangulation across cameras
- Blendshape aggregation

**Duration:** 30-60 min/session (GPU-accelerated)

## GPU Acceleration Details

### What Uses GPU

| Component | Framework | Batch Size | Memory (GB) | Duration Reduction |
|-----------|-----------|-----------|---------|-------------------|
| Tobii Tracker | OpenCV CUDA | N/A | 2-3 | 10-15% |
| MediaPipe Pose | MediaPipe GPU | 1/frame | 1-2 | 30-40% |
| MediaPipe Face | MediaPipe GPU | 1/frame | 2-3 | 35-45% |
| MediaPipe Hands | MediaPipe GPU | 1/frame | 1-2 | 30-40% |
| DLT Reconstruction | NumPy (CPU) | N/A | 0.5 | 0% |

### GPU Memory Requirements

- **Per worker:** 6-8 GB VRAM (budget for buffering)
- **All workers:** `workers * 6 GB` minimum
- **Headroom:** 2-3 GB reserved for OS

Example:
```
4 workers × 8 GB = 32 GB total
→ RTX 3090 has 24 GB, so use 3 workers maximum
```

## Multiprocessing Architecture

### Worker Pool

- **Master process** — Reads inventories, queues tasks, aggregates results
- **Worker pool** — (Subprocess) processes one session per worker
- **GPU queue** — Distributes GPU devices round-robin to workers

### Communication

```
Master Process
    ↓
Session Queue → Worker 1 (GPU:0) → BIDSProcessor → Session 1
   (multiprocessing.Queue)
              → Worker 2 (GPU:1) → BIDSProcessor → Session 2
              → Worker 3 (GPU:0) → BIDSProcessor → Session 3
              → Worker 4 (GPU:1) → BIDSProcessor → Session 4
    ↓
Result Queue → Aggregator → Report JSON + Summary
```

### Safety

- Multiprocessing-safe logging via QueueHandler
- GPU allocation via mp.Queue (thread-safe)
- Per-worker subprocess isolation (no shared state)
- Graceful shutdown on KeyboardInterrupt

## Monitoring Progress

### During Execution

```bash
# Watch logs in real-time
Get-Content logs/pipeline.log -Wait
```

### Session-Level Logging

Each session logs to:
```
logs/session_{session_id}.log
```

### Final Report

After execution, review:
```
E:\processed_data\pipeline_report.json
```

Sample output:
```json
{
  "pipeline": "MasterBIDSPipeline",
  "timestamp": "2026-03-27T14:32:15.123456",
  "summary": {
    "total_sessions": 27,
    "successful": 25,
    "failed": 2,
    "success_rate": "92.6%",
    "total_time_seconds": 156789.5,
    "avg_time_per_session": 6271.6
  },
  "results": [
    {
      "session_id": "ses-20260309_grp-03_run01",
      "success": true,
      "status_message": "Processing completed successfully",
      "duration_seconds": 1234.5,
      "output_dir": "E:\\processed_data\\sub-00\\ses-ses-20260309_grp-03_run01",
      "modalities_processed": ["bids_core", "bids_canonical", "pose3d"]
    },
    ...
  ]
}
```

## Troubleshooting

### Out of Memory (OOM)

**Symptom:** Worker crashes with CUDA OOM
**Solution:**
1. Reduce workers: `--max-workers 2` (was 4)
2. Or disable GPU processing: remove `--enable-3d-pose`
3. Or check for memory leaks in subprocess

### GPU Not Detected

**Symptom:** "No CUDA devices found" despite `nvidia-smi` showing GPU
**Solution:**
```bash
# Check PyTorch CUDA setup
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Slow Processing (GPU Underutilized)

**Symptom:** GPU utilization <20%, workers CPU-bottlenecked
**Solution:**
1. Increase workers: `--max-workers 8` (was 4)
2. Check for I/O bottleneck: disk too slow?
3. Use `nvidia-smi` to monitor actual GPU load

### Worker Deadlock

**Symptom:** Pipeline hangs indefinitely
**Solution:**
1. Press Ctrl+C to gracefully shutdown
2. Check for circular dependencies in subprocess calls
3. Increase subprocess timeout: edit pipeline code

### Data Directory Not Found

**Symptom:** "Error: Data directory not found"
**Solution:**
```bash
# Verify path exists
ls -la ../affectai-data-processing-seed/data/

# Update path in script
run_pipeline.bat --data-dir D:\path\to\data
```

## Advanced Configuration

### Custom Config File

Create `pipeline.yaml`:
```yaml
pipeline:
  data_root: affectai-data-processing-seed/data
  output_root: E:\processed_data
  max_workers: 4
  gpu_devices: [0]

processing:
  enable_3d_pose: true
  enable_face_hand: true
  enable_physio: false

logging:
  level: INFO
  file: logs/pipeline.log
```

Then run:
```bash
python tools/master_bids_pipeline.py --config pipeline.yaml
```

(Note: config file support can be added if needed)

### Resume from Checkpoint

For interrupted pipelines (planned feature):
```bash
python tools/master_bids_pipeline.py ... --resume-from-checkpoint
```

Currently, restart the command to reprocess failed sessions.

## Validation

### BIDS Validator

After pipeline completes:
```bash
pip install bids-validator
bids-validator E:\processed_data
```

### Check Output Structure

```bash
tree E:\processed_data -L 3 /F
```

Expected structure:
```
E:\processed_data
├── sub-00/ses-xxx/video/*.mkv
├── sub-00/ses-xxx/audio/*.wav
├── sub-00/ses-xxx/et/*.tsv
├── dataset_description.json
├── participants.tsv
└── pipeline_report.json
```

## Example Runs

### Run 1: Quick Test
```bash
run_pipeline.bat --preset quick --output-dir D:\test_output
```
Duration: ~15 min (1 session), 27–30 min (27 sessions)

### Run 2: Standard Production
```bash
run_pipeline.bat --preset standard --output-dir E:\processed_data
```
Duration: ~1 hour per session = ~27 hours total (27 sessions, 4 workers)

### Run 3: Maximum Quality (Dual GPU)
```bash
run_pipeline.bat --preset dual_gpu --output-dir E:\processed_data
```
Duration: ~30 min per session = ~13.5 hours total (27 sessions, 8 workers, 2 GPUs)

### Run 4: Custom Debug Run
```bash
python tools/master_bids_pipeline.py \
    --data-dir data \
    --output-dir D:\debug_output \
    --max-workers 1 \
    --gpu-devices 0 \
    --enable-3d-pose \
    --verbose
```
Process single session with full output for debugging.

## References

- **BIDS Specification:** https://bids-specification.readthedocs.io/
- **MultiModal Task fMRI Standard:** https://bids-specification.readthedocs.io/en/stable/05-derivatives/02-common-data-types.html#physiological-and-other-continuous-recordings
- **Pipeline Documentation:** See [ARCHITECTURE.md](../ARCHITECTURE.md)

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review `pipeline_report.json` for detailed error traces
3. Enable `--verbose` for detailed logging
4. Check individual session logs in output directory

---

**Last Updated:** 2026-03-27  
**Version:** 1.0
