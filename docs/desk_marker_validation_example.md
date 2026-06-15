# Desk Marker Validation Example

This guide shows how to use the new desk marker validation feature to validate camera calibration accuracy.

## Prerequisites

1. **Calibration TOML file** - Generated from `calibrate_charuco.py` or `online_calibration.py`
2. **Marker map YAML** - Table marker positions, generated with `online_calibration.py --export-table-marker-map`
3. **Video files** - Recorded session videos with visible ArUco desk markers

## Step-by-Step Example

### 1. Generate Marker Map (if not already done)

During online calibration, export the marker map:

```bash
python tools/online_calibration.py \
  --config configs/ffmpeg_multicap.json \
  --duration 60 \
  --export-calibration \
  --export-session-dir data/sub-test/ses-20260308_test/video \
  --export-table-marker-map \
  --table-marker-ids "0,1,2,3,4" \
  --table-width-cm 180 \
  --table-depth-cm 120 \
  --table-marker-size-mm 50
```

This creates `data/sub-test/ses-20260308_test/video/table_marker_map.yaml`.

### 2. Run Basic Validation (without markers)

```bash
python tools/validate_calibration_robust.py \
  --toml data/sub-test/ses-20260308_test/video/video_camera_calibration.toml \
  --output validation_basic.txt
```

Output:
```
======================================================================
OVERALL QUALITY
======================================================================
Score: 85.0/100
Grade: Good
Recommendation: Accept calibration

✓ No significant issues detected
```

### 3. Run Enhanced Validation (with desk markers)

```bash
python tools/validate_calibration_robust.py \
  --toml data/sub-test/ses-20260308_test/video/video_camera_calibration.toml \
  --marker-map data/sub-test/ses-20260308_test/video/table_marker_map.yaml \
  --videos-dir data/sub-test/ses-20260308_test/video \
  --output validation_with_markers.txt \
  --json validation_with_markers.json

# Optional override if camera orientation rules differ:
# --camera-config configs/ffmpeg_multicap.json
# --flipped-camera-patterns "custom_cam_a,custom_cam_b"
# Use --flipped-camera-patterns "" to disable 180deg correction.
```

Output with marker validation:
```
======================================================================
OVERALL QUALITY
======================================================================
Score: 88.0/100
Grade: Good
Recommendation: Accept calibration

✓ No significant issues detected

======================================================================
DESK MARKER VALIDATION
======================================================================
Total marker detections: 1250
Mean reprojection error: 3.42 pixels
Max reprojection error: 8.15 pixels
Acceptance threshold: < 5.0 pixels
Status: ✓ PASSED

Per-camera results:
  jabra_panacast_20_cam1_vid_video: 312 detections, error=3.21px
  jabra_panacast_20_cam2_vid_video: 298 detections, error=3.55px
  jabra_panacast_20_cam3_vid_video: 325 detections, error=3.48px
  jabra_panacast_20_cam4_vid_video: 315 detections, error=3.44px
```

### 4. Interpret Results

**Quality Score Components:**
- Base score: 100
- Deductions for:
  - Focal length mismatches
  - Distortion issues
  - Poor camera geometry
  - High marker reprojection error

**Marker Validation Status:**
- ✓ **PASSED** (mean < 5px): Calibration is accurate, accept for use
- ⚠ **FAIR** (mean 5-10px): Calibration is acceptable but not optimal
- ✗ **FAILED** (mean > 10px): Re-calibrate recommended

**Recommendations:**
- Score ≥75 + marker validation passed → **Accept calibration**
- Score 60-75 → **Accept with caution** (verify in test recordings)
- Score <60 or marker validation failed → **Reject, re-calibrate**

## Advanced Options

### Adjust Sampling

For faster validation on long videos:

```bash
python tools/validate_calibration_robust.py \
  --toml <calibration.toml> \
  --marker-map <marker_map.yaml> \
  --videos-dir <video_dir> \
  --max-frames 50 \
  --sample-stride 20
```

- `--max-frames`: Analyze at most N frames per camera (default: 100)
- `--sample-stride`: Skip frames (e.g., 20 = every 20th frame, default: 10)
- `--camera-config`: Camera config JSON used to read per-device `rotate_180` metadata
- `--flipped-camera-patterns`: Optional comma-separated override. If omitted, `rotate_180=true` cameras from config are corrected.

### JSON Output for Automation

```bash
python tools/validate_calibration_robust.py \
  --toml <calibration.toml> \
  --marker-map <marker_map.yaml> \
  --videos-dir <video_dir> \
  --json validation.json
```

Parse JSON in scripts:

```python
import json

with open("validation.json") as f:
    data = json.load(f)

score = data["quality_score"]["score"]
recommendation = data["quality_score"]["recommendation"]
marker_error = data["marker_validation"]["mean_reprojection_error_px"]

if score >= 75 and marker_error < 5.0:
    print("✓ Calibration accepted")
else:
    print("✗ Calibration rejected, re-calibrate")
```

## Troubleshooting

### No marker detections

**Problem:** `Status: No marker detections found`

**Solutions:**
1. Check marker IDs match between video and marker map
2. Ensure markers are visible and in focus in videos
3. Verify ArUco dictionary matches (default: DICT_4X4_50)
4. Try different frame sampling (`--sample-stride 5`)

### High reprojection error

**Problem:** `Mean reprojection error: 12.5 pixels (> 10px threshold)`

**Solutions:**
1. Re-run calibration with more frames and better board coverage
2. Use `--init-focal` during calibration to seed focal length from specs
3. Ensure calibration videos and validation videos use same camera settings
4. Check that marker map dimensions match physical setup

### Video file not found

**Problem:** `No video found for camera X in <directory>`

**Solution:**
The tool looks for videos matching camera names from TOML:
- `<camera_name>.mkv`
- `<camera_name>.mp4`
- `<camera_name>.avi`

Ensure video filenames match calibration camera names exactly.

## Integration with Workflow

Recommended calibration + validation workflow:

1. **Online calibration** with export:
   ```bash
   python tools/online_calibration.py \
     --export-calibration \
     --export-session-dir <session>/video \
     --export-table-marker-map \
     --table-marker-ids "0,1,2,3,4" \
     --table-width-cm 180 \
     --table-depth-cm 120
   ```

2. **Basic validation** (quick check):
   ```bash
   python tools/validate_calibration_robust.py \
     --toml <session>/video/video_camera_calibration.toml
   ```

3. **Marker validation** (if basic passed):
   ```bash
   python tools/validate_calibration_robust.py \
     --toml <session>/video/video_camera_calibration.toml \
     --marker-map <session>/video/table_marker_map.yaml \
     --videos-dir <session>/video \
     --json <session>/video/validation_report.json
   ```

4. **Decision:**
   - Score ≥75 + marker error <5px → Use calibration
   - Otherwise → Re-calibrate

## See Also

- [Calibration Usage Guide](calibration_usage.md)
- [Online Calibration Docs](online_calibration.md)
- Video tutorial: `data/calibration/README.md`
