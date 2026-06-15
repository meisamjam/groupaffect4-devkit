# FreeMoCap Integration Guide

FreeMoCap is an open-source, markerless motion capture system. This guide describes how to use it for **post-hoc analysis** of video recordings captured during AffectAI sessions (e.g., from Jabra PanaCast cameras).

## Overview

This integration supports:
- **Post-hoc processing:** Extract 3D skeleton data from saved video files
- **BIDS-compatible output:** Results stored in `/mocap/` directory following BIDS conventions
- **Multi-task processing:** Process multiple task videos from a single session in one command
- **Flexible input:** Works with any video format (MP4, MOV, AVI, etc.)

## Installation

### Option 1: Install with pip (Recommended)

```bash
cd affectai-capture  # Your repository root
pip install -e ".[freemocap]"
```

This installs FreeMoCap (≥1.3.0) as an optional dependency.

### Option 2: Manual installation

```bash
pip install freemocap>=1.3.0
```

### Verify installation

```bash
python -c "import freemocap; print(freemocap.__version__)"
```

## Quick Start

### Basic usage: Single video file

Process a single video file and extract skeleton data:

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --video-file data/sub-01/ses-01/video/sub-01_ses-01_task-T1_video.mp4 \
    --task T1
```

### Auto-discover mode: Process all task videos

Let the tool find all video files in a session:

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --auto-discover
```

### Process specific tasks only

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --auto-discover \
    --task T1 \
    --task T2
```

### Filter by video source/label

If you have multiple video sources (e.g., Jabra camera + RGB camera), filter which to process:

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --auto-discover \
    --video-label 'jabra_panacast'
```

### Advanced: Custom confidence threshold

Control the minimum confidence level for keypoint detection (higher = stricter):

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --auto-discover \
    --confidence-threshold 0.7
```

### Verbose logging

Enable detailed logging for debugging:

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --video-file data/sub-01/ses-01/video/....mp4 \
    --task T1 \
    --verbose
```

## Output files

After processing, you'll find BIDS-compatible outputs in the `mocap/` directory:

```
data/sub-01/ses-01/mocap/
├── sub-01_ses-01_task-T1_run-01_acq-freemocap_skeletons.tsv    # 3D skeleton data
├── sub-01_ses-01_task-T1_run-01_acq-freemocap_skeletons.json   # Metadata
├── sub-01_ses-01_task-T2_run-01_acq-freemocap_skeletons.tsv
├── sub-01_ses-01_task-T2_run-01_acq-freemocap_skeletons.json
└── freemocap_processing_log.json                               # Processing summary
```

### Output file formats

#### TSV skeleton data (`*_skeletons.tsv`)

Tab-separated values containing 3D joint positions:

```
frame   joint    x       y       z       confidence
0       nose     0.123   0.456   0.789   0.95
0       neck     0.125   0.450   0.790   0.93
...
```

Columns:
- `frame`: Frame number in video
- `joint`: Joint name (e.g., 'nose', 'neck', 'shoulder_left', etc.)
- `x`, `y`, `z`: 3D coordinates in meters
- `confidence`: Detection confidence (0.0–1.0)

#### Metadata JSON (`*_skeletons.json`)

Processing metadata following BIDS conventions:

```json
{
  "Description": "Markerless 3D skeleton tracking",
  "Source": "FreeMoCap",
  "AcquisitionLabel": "freemocap",
  "SkeletonModel": "mediapipe_holistic",
  "FrameRate": 30,
  "NumberOfFrames": 900,
  "ConfidenceThreshold": 0.5
}
```

#### Processing log (`freemocap_processing_log.json`)

Summary of the processing session:

```json
{
  "session_dir": "data/sub-01/ses-01",
  "output_label": "freemocap",
  "confidence_threshold": 0.5,
  "processed_videos": 2,
  "failed_videos": 0,
  "results": {
    "T1_run01": {
      "status": "completed",
      "outputs": {
        "skeleton_tsv": "data/sub-01/ses-01/mocap/..._skeletons.tsv",
        "metadata": "data/sub-01/ses-01/mocap/..._skeletons.json"
      }
    }
  }
}
```

## Integration with session workflow

### Typical workflow

1. **Capture phase:** Record video with Jabra PanaCast cameras during task block (automatic)

   ```bash
   python -m src.affectai_capture.session_manager start sub-001 ses-001
   # ... run experiment ...
   ```

2. **Post-hoc processing:** Extract skeleton data after session completes

   ```bash
   python tools/process_freemocap.py \
       --session-dir data/sub-001/ses-001 \
       --auto-discover
   ```

3. **Quality check:** Review outputs in `mocap/` directory

   ```bash
   ls -la data/sub-001/ses-001/mocap/
   ```

4. **Analysis:** Use skeleton data in downstream analysis pipelines

## Troubleshooting

### Issue: `ImportError: No module named 'freemocap'`

**Solution:** Install FreeMoCap:

```bash
pip install -e ".[freemocap]"
# or
pip install freemocap>=1.3.0
```

### Issue: `FileNotFoundError: Video file not found`

**Solution:** Check that:
1. Session directory exists: `data/sub-XX/ses-YY/`
2. Video files exist: `data/sub-XX/ses-YY/video/*.mp4`
3. File paths are correct (Linux/Mac use `/`, Windows uses `\`)

**Debug:**

```bash
python tools/process_freemocap.py \
    --session-dir data/sub-01/ses-01 \
    --auto-discover \
    --verbose
```

### Issue: Processing hangs or is very slow

**Possible causes:**
- Large video files (>1GB) take significant time (~1 min per 30s video)
- Insufficient GPU/CPU resources
- FreeMoCap configuration issues

**Options:**
- Process one task at a time instead of `--auto-discover`
- Reduce video resolution or frame rate before processing
- Check system resources: `top` (Linux/Mac) or Task Manager (Windows)

### Issue: Low confidence scores (<0.5) in output

**Possible causes:**
- Video quality issues (motion blur, poor lighting)
- Subject partially out of frame
- Occlusions or unusual poses

**Solutions:**
- Review source video quality
- Lower `--confidence-threshold` if detection is otherwise good
- Adjust camera position for better subject visibility

## Technical details

### Skeleton model

FreeMoCap uses [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) for pose estimation, which detects:

- **Body:** 33 keypoints (pose landmarks)
- **Hands:** 21 keypoints per hand
- **Face:** 468 keypoints (optional)

### Coordinate system

Output coordinates are in:
- **Units:** Meters
- **Reference frame:** Camera-relative (not absolute world coordinates)
- **Origin:** Typically at camera center

### Processing pipeline

For each video:

1. Load video and extract frames
2. Run MediaPipe Holistic inference on each frame
3. Filter detections by confidence threshold
4. Convert to BIDS-compatible format
5. Save skeleton TSV + metadata JSON

## API Usage (Python)

You can also use FreeMoCap processing directly in your own Python scripts:

```python
from pathlib import Path
from src.affectai_capture.devices.freemocap_processor import FreeMoCapProcessor

# Initialize processor
session_dir = Path('data/sub-01/ses-01')
processor = FreeMoCapProcessor(session_dir)

# Process single video
outputs = processor.process_video(
    video_path=Path('data/sub-01/ses-01/video/task-T1.mp4'),
    task_name='T1',
    run=1,
    confidence_threshold=0.6,
)

print(f"✓ Saved: {outputs['skeleton_tsv']}")
print(f"✓ Saved: {outputs['metadata']}")
```

### Batch processing

```python
from src.affectai_capture.devices.freemocap_processor import process_session_videos

video_files = {
    'T1_run1': Path('data/sub-01/ses-01/video/task-T1.mp4'),
    'T2_run1': Path('data/sub-01/ses-01/video/task-T2.mp4'),
}

results = process_session_videos(
    session_dir=Path('data/sub-01/ses-01'),
    video_files=video_files,
)

for task_id, outputs in results.items():
    if 'error' in outputs:
        print(f"✗ {task_id}: {outputs['error']}")
    else:
        print(f"✓ {task_id}: {outputs}")
```

## References

- **FreeMoCap documentation:** https://freemocap.org/
- **MediaPipe Holistic:** https://google.github.io/mediapipe/solutions/holistic.html
- **BIDS motion capture specification:** https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html
- **AffectAI architecture:** `docs/architecture.md`

## Future enhancements

Possible extensions to this integration:

- [ ] Real-time skeleton streaming via LSL for live monitoring
- [ ] Integration with Vicon markers for hybrid tracking
- [ ] Hand gesture recognition from skeleton data
- [ ] Automated video quality assessment before processing
- [ ] Multi-view reconstruction (multiple camera angles)
- [ ] Direct integration with stimulus pipeline for event-contingent analysis

## Questions or issues?

See `docs/known_issues.md` or check the main documentation hub: `ARCHITECTURE.md`
