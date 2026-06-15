# Pipeline Architecture & Design

## Overview

The Master BIDS Pipeline is designed as a **modular, scalable orchestrator** that:

1. **Reads inventory metadata** from JSON/CSV inventory files
2. **Plans session processing** with optimal resource allocation
3. **Launches workers** in parallel to process sessions
4. **Distributes GPU devices** fairly among workers
5. **Aggregates results** and generates reports

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load Inventories                                         │
│     ├── high_level_data_inventory.json                      │
│     ├── high_level_group_inventory.csv                      │
│     └── high_level_session_inventory.csv                    │
│                                                               │
│  2. Plan Sessions                                            │
│     └── SessionConfig[] ← (session_id, modalities, GPU)    │
│                                                               │
│  3. Multiprocessing Pool (4-8 workers)                       │
│     ├── Worker 1 ─→ Session 1 ─→ BIDSProcessor ─→ Result  │
│     ├── Worker 2 ─→ Session 2 ─→ BIDSProcessor ─→ Result  │
│     ├── Worker 3 ─→ Session 3 ─→ BIDSProcessor ─→ Result  │
│     └── Worker 4 ─→ Session 4 ─→ BIDSProcessor ─→ Result  │
│                                                               │
│  4. Aggregate & Report                                       │
│     └── pipeline_report.json ← (success/fail, timing, etc)  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Processing Per Session

```
BIDSProcessor (per session)
│
├─ setup_directories()
│  └─ Create output directory structure
│
├─ run_multisource_to_bids()
│  │
│  └─ Subprocess: tools/multisource_to_bids_runs.py
│     ├── Read raw sources (XDF, MKV, Tobii, etc.)
│     ├── Merge multi-PC streams
│     ├── Derive task windows (T0–T4)
│     └── Write BIDS session tree
│
├─ run_raw_to_bids()
│  │
│  └─ Subprocess: tools/raw_to_bids.py
│     ├── Canonicalize modality formats
│     ├── Validate BIDS compliance
│     └── Write standard BIDS layout
│
├─ run_video_only_3d_pipeline() [optional]
│  │
│  └─ Subprocess: tools/video_only_3d_pipeline.py
│     ├── Tobii world gaze tracking
│     ├── Multicam 3D pose reconstruction
│     ├── Skeleton refinement
│     └── Gesture extraction
│
└─ run_face_hand_pipeline() [optional, GPU-intensive]
   │
   └─ Subprocess: tools/face_hand_pipeline.py
      ├── MediaPipe FaceLandmarker (468 points + blendshapes)
      ├── MediaPipe HandLandmarker (21 points/hand)
      ├── 3D triangulation
      └── Blendshape aggregation
```

## Data Flow

### Input Files
```
affectai-data-processing-seed/data/
├── high_level_data_inventory.json
│  └── {"recording_root": "...", "sessions": [...], ...}
├── high_level_group_inventory.csv
│  └── group_id, session_count, participants, modalities, ...
├── high_level_session_inventory.csv
│  └── session, group_id, participants, raw_modalities, ...
│
└── affectai-capture-recording/
   └── sessions/
      ├── ses-20260309_grp-03_run01/
      │  ├── *.xdf              ← Recording PC streams (LSL)
      │  ├── sourcedata/av/     ← AV-PC media
      │  ├── sourcedata/tobii_lsl/ ← Tobii gaze stream
      │  └── sourcedata/tobii_device/ ← Tobii recordings
      └── ...
```

### Output Structure
```
E:\processed_data/
├── sub-00/ses-20260309_grp-03_run01/
│  ├── video/                   ← Task-split video clips
│  │  ├── sub-00_ses-20260309_grp-03_run01_task-T0_video.mkv
│  │  ├── sub-00_ses-20260309_grp-03_run01_task-T1_video.mkv
│  │  └── ...
│  │
│  ├── audio/                   ← Task-split audio clips
│  │  ├── sub-00_ses-20260309_grp-03_run01_task-T0_audio.wav
│  │  └── ...
│  │
│  ├── et/                      ← Eyetracking (Tobii)
│  │  └── sub-00_ses-20260309_grp-03_run01_et.tsv
│  │
│  ├── physio/                  ← Physiological
│  │  ├── sub-00_ses-20260309_grp-03_run01_ecg.tsv
│  │  ├── sub-00_ses-20260309_grp-03_run01_ppg.tsv
│  │  └── sub-00_ses-20260309_grp-03_run01_eda.tsv
│  │
│  ├── pose3d/                  ← 3D Pose (if --enable-3d-pose)
│  │  ├── skeleton_3d.npy       ← [frames × joints × xyz] 
│  │  ├── skeleton_refined.npy  ← Butterworth-filtered
│  │  ├── calibration.toml      ← Camera intrinsics/extrinsics
│  │  └── metadata.json         ← Processing log
│  │
│  ├── facehand/                ← Face/Hand (if --enable-face-hand)
│  │  ├── face_3d.npz           ← Face landmarks [frames × 468 × 3]
│  │  ├── hand_3d.npz           ← Hand landmarks per person
│  │  ├── blendshapes.json      ← ARKit blendshape coefficients
│  │  └── confidence.json       ← Detection confidence per frame
│  │
│  ├── beh/                     ← Behavioral
│  │  └── sub-00_ses-20260309_grp-03_run01_beh.json
│  │
│  ├── annot/                   ← Annotations
│  │  ├── task_run_windows.tsv
│  │  ├── participant_signal_map.tsv
│  │  └── events.jsonl
│  │
│  └── events.tsv               ← Master timeline
│
├── sub-01/ses-.../...
├── ...
├── dataset_description.json    ← BIDS metadata
├── participants.tsv            ← Subject roster
├── participants.json
├── pipeline_report.json        ← Execution summary
└── logs/                       ← Session-level logs [optional]
```

## Multiprocessing Architecture

### Worker Pool Pattern

```python
with mp.Pool(processes=max_workers) as pool:
    tasks = [(config1), (config2), ..., (configN)]
    results = pool.starmap(process_session_worker, tasks)
```

**Benefits:**
- Each worker is an independent process (no GIL)
- Workers don't share state (no race conditions)
- Graceful shutdown on KeyboardInterrupt
- Resource cleanup automatic

### GPU Device Distribution

```python
gpu_devices = [0, 1]  # Available GPUs
device_queue = mp.Queue()  # Thread-safe device allocation

for device_id in gpu_devices:
    device_queue.put(device_id)

# In worker:
device_id = device_queue.get()      # Acquire GPU
process_with_gpu(device_id)         # Use GPU
device_queue.put(device_id)         # Release GPU
```

**Benefits:**
- Lock-free synchronization (Queue is thread-safe)
- Fair device distribution
- Automatic timeout protection

### Memory Isolation

Each worker process:
- **Runs in isolated memory space** (no parent dependencies)
- **Imports modules independently** (fresh imports)
- **Uses subprocess for tools** (clean separation)

This prevents:
- Memory leaks accumulating across sessions
- CUDA context conflicts
- File handle exhaustion

## Logging Architecture

```
Master Process
│
├─ main logger → console + pipeline.log
│
└─ Worker 1 → Session 1 logger → console + session_1.log
   Worker 2 → Session 2 logger → console + session_2.log
   Worker 3 → Session 3 logger → console + session_3.log
   ...
```

Each logger:
- **Prefixes messages** with session ID
- **Writes to file** independently
- **Queues to master** for aggregation

## Configuration Management

### Inventory Loading
```python
# JSON
data_inventory = json.load(fp)
# Sessions: list[dict], Metadata: dict

# CSV with parsing
group_inventory = [
    {
        "group_id": "grp-01",
        "sessions": ["ses-20260318_grp-01_run01"],  # Parsed from semicolon-sep
        "participants_ids": ["sub-01", "sub-02"],   # "
        "raw_modalities": ["lsl", "tobii_lsl"],     # "
        ...
    }
]
```

### Session Planning
```python
SessionConfig(
    session_id="ses-20260309_grp-03_run01",
    group_id="grp-03",
    participants=["sub-01", "sub-02", "sub-03", "sub-04"],
    raw_modalities=["tobii_lsl"],
    phase_tags=["pilot"],
    input_root=Path("..."),
    output_root=Path("E:\\processed_data\\sub-00\\ses-20260309_grp-03_run01"),
    enable_3d_pose=True,
    enable_face_hand=True,
    gpu_device_id=0,  # Assigned round-robin
)
```

## Resource Allocation

### CPU
```
Total cores = 8
Workers = 4
Cores per worker = 8 / 4 = 2
```

### GPU
```
GPUs = 2 (IDs: 0, 1)
Workers = 8
Distribution: Round-robin assignment
  Worker 0 → GPU 0
  Worker 1 → GPU 1
  Worker 2 → GPU 0
  Worker 3 → GPU 1
  ...
```

### Memory
```
System RAM = 32 GB
Per worker = ~6-8 GB (video buffer + processing)
Max workers = 32 / 8 = 4
Headroom = 2 GB reserved for OS
```

## Error Handling

### Per-Worker Error Recovery

```python
try:
    processor = BIDSProcessor(config, raw_to_bids_module)
    result = processor.process()  # Run all stages
return ProcessingResult(
    success=True/False,
    status_message="...",
    error_details="Traceback..." if failed,
)
```

**Key benefits:**
- One session failure doesn't crash others
- Errors grouped in final report
- Can resume from checkpoint (future)

### Master Process Error Handling

```python
try:
    pipeline.run()  # Main flow
except KeyboardInterrupt:
    logger.info("User interrupted - graceful shutdown")
    # Workers complete current session, then exit
except Exception as e:
    logger.error(traceback.format_exc())
    return False
```

## Performance Characteristics

### Throughput
```
Sequential (1 worker):
  27 sessions × 90 min/session = 40.5 hours

Parallel (4 workers):
  27 sessions / 4 ≈ 6 batches × 90 min = 9 hours

Speedup = 40.5 / 9 ≈ 4.5x (excellent linear scaling)
```

### Latency
```
First result: ~90 min (time for first session)
All results: ~9 hours (total wall-clock time)
```

### Resource Utilization
```
CPU:   70-80% (multiprocessing + subprocess work)
GPU:   50-70% (video I/O bottleneck)
Memory: 70-80% (session buffering)
Disk:  Constant streaming writes (~10-50 MB/s)
```

## Extensibility Points

Future additions:
1. **Config file support** — TOML/YAML configuration
2. **Checkpoint/resume** — Save intermediate results
3. **Progress tracking** — Real-time dashboard
4. **Distributed processing** — Multi-machine via Ray
5. **Custom hooks** — User-defined preprocessing/postprocessing
6. **Dynamic batching** — Adjust workers based on load

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Multiprocessing over threading** | Bypasses Python GIL for true parallelism |
| **Subprocess for tools** | Tool isolation, independent Python versions/packages |
| **Round-robin GPU assignment** | Fair distribution, no coordinator bottleneck |
| **Per-worker logging** | Scalable, non-blocking I/O |
| **Manager process for state** | Proper cleanup, exception handling |
| **JSON report output** | Machine-readable, version control friendly |
| **Preset configurations** | Lower friction for common use cases |

---

**Design Version:** 1.0  
**Last Updated:** 2026-03-27
