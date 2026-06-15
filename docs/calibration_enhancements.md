# Video Calibration Enhancements Summary

## Overview

Your video calibration system has been enhanced with comprehensive validation, quality scoring, and visualization tools that work independently of FreeMoCap.

## New Tools

### 1. Robust Calibration Validator (`validate_calibration_robust.py`)

**Purpose:** Comprehensive calibration quality assessment without FreeMoCap dependency

**Features:**
- ✅ Intrinsics summary (focal length, principal point, resolution)
- ✅ Focal length validation against camera specs (with regex pattern matching)
- ✅ Inter-camera geometry analysis (distances between all camera pairs)
- ✅ Distortion coefficient analysis with non-monotonic detection
- ✅ Automated quality scoring (0-100 with letter grade)
- ✅ Severity-graded warnings (ERROR/WARNING)
- ✅ Actionable recommendations for improvement

**Usage:**
```bash
python tools/validate_calibration_robust.py \
  --toml <session>/video/video_camera_calibration.toml \
  --output validation_report.txt \
  --json validation_report.json
```

**Quality Grades:**
- 90-100: Excellent
- 75-89: Good
- 60-74: Fair (usable but check warnings)
- 40-59: Poor (consider re-calibrating)
- 0-39: Failed (must re-calibrate)

**Exit Codes:**
- 0: Quality ≥40 (acceptable)
- 1: Quality <40 (failed)

---

### 2. Calibration Geometry Visualizer (`visualize_calibration.py`)

**Purpose:** 3D visualization of camera rig layout

**Features:**
- 📍 Camera positions (red spheres)
- 🔷 Orientation arrows (blue, along camera Z-axis)
- 📏 Inter-camera distances (color-coded by quality)
  - Red: <0.5m (too close for good triangulation)
  - Gray: 0.5-5m (good separation)
  - Orange: >5m (very far apart)
- 🎯 Coordinate frame at origin (RGB for XYZ)
- 📊 Focal length summary with expected ratios

**Usage:**
```bash
# Interactive plot
python tools/visualize_calibration.py \
  --toml <session>/video/video_camera_calibration.toml

# Save to file
python tools/visualize_calibration.py \
  --toml <session>/video/video_camera_calibration.toml \
  --output calibration_geometry.png
```

---

### 3. Enhanced Online Calibration

**Improvement:** `online_calibration.py` now uses robust validation as fallback

**Behavior:**
1. First attempts FreeMoCap validation
2. If import errors detected, automatically falls back to robust validator
3. Validation report always contains meaningful metrics (no error traces)
4. Metadata includes which validation method was used

**No changes to usage** — existing commands work as before:
```bash
python tools/online_calibration.py \
  --config configs/ffmpeg_multicap.json \
  --duration 60 \
  --square-size-mm 69 \
  --board-mode fixed --board-width 5 --board-height 3 \
  --export-calibration \
  --export-session-dir <session>
```

---

## Camera Specs Configuration

The validator uses `configs/camera_specs.json` to validate focal lengths. This file includes:

- Expected focal lengths for Jabra PanaCast 20 and 50
- Expected distortion ranges
- Regex patterns for camera name matching

**Camera Models Supported:**
- `jabra_panacast_20`: 120° HFOV, fx≈554px @ 1080p
- `jabra_panacast_50`: 133° HFOV, fx≈418px @ 1080p

---

## Your Current Calibration Status

Based on validation of `ses-20260202_test`:

**Quality Score:** 0/100 (Failed)

**Critical Issues:**
1. **Focal length mismatches** (all cameras):
   - cam2: 2.28× expected (1265px vs 554px)
   - cam3: 2.45× expected (1357px vs 554px)
   - cam4: 1.56× expected (865px vs 554px)
   - cam5: 1.45× expected (605px vs 418px)

2. **Non-monotonic distortion** (2 cameras):
   - cam1: k1=-1.19 (r_crit=0.53)
   - cam5: k1=-0.57 (r_crit=0.77)
   - These cameras will fail `cv2.undistortPoints()` near image edges

3. **Cameras too close**:
   - cam2 and cam3: only 0.25m apart
   - Poor triangulation baseline

**Recommendations:**
1. **Re-calibrate with `--init-focal`** to seed expected focal lengths:
   ```bash
   python tools/calibrate_charuco.py calibrate \
     --videos-dir <videos> \
     --square-size 69 \
     --init-focal \
     --output <session>/video/video_camera_calibration.toml
   ```

2. **Use `multicam_pose3d.py`** for 3D reconstruction (has built-in distortion handling)

3. **Verify camera placement** — cam2 and cam3 should be further apart

4. **Ensure good ChArUco visibility** during calibration:
   - Board visible in ≥3 cameras simultaneously
   - Good lighting, minimal motion blur
   - Cover full field of view (board at various depths and angles)

---

## Integration Points

### Modified Files
- ✅ `tools/online_calibration.py` — fallback validation logic
- ✅ `docs/calibration_usage.md` — usage examples
- ✅ `docs/llm/context_snapshot.md` — tool routing table
- ✅ `CHANGES.md` — changelog entries

### New Files
- ✅ `tools/validate_calibration_robust.py`
- ✅ `tools/visualize_calibration.py`

### No Breaking Changes
All existing commands and workflows continue to work unchanged.

---

## Testing

```bash
# Test robust validator
python tools/validate_calibration_robust.py \
  --toml data/sub-meisam/ses-20260202_test/video/video_camera_calibration.toml

# Test visualizer
python tools/visualize_calibration.py \
  --toml data/sub-meisam/ses-20260202_test/video/video_camera_calibration.toml \
  --output test_geometry.png

# Test online calibration with validation
python tools/online_calibration.py \
  --config configs/ffmpeg_multicap.json \
  --duration 10 \
  --square-size-mm 69 \
  --board-mode fixed --board-width 5 --board-height 3 \
  --export-calibration \
  --export-session-dir data/test_calibration
```

---

## Next Steps

1. **Run a new calibration session** with `--init-focal` flag
2. **Validate the new calibration** using the robust validator
3. **Visualize camera geometry** to verify spatial layout
4. **Check quality score** — aim for ≥75 for production use

---

## Questions / Future Enhancements

Potential additions (not implemented yet):
- [ ] Per-camera reprojection error heatmaps
- [ ] Automatic focal length initialization from camera specs in `online_calibration.py`
- [ ] Coverage analysis (what % of image space sees ChArUco markers)
- [ ] Time-series plots of calibration stability over multiple sessions
- [ ] Integration with session orchestrator GUI (show validation in calibration panel)

**Need help?** Review:
- `docs/calibration_usage.md` — usage guide
- `docs/online_calibration.md` — live calibration workflow
- `CHANGES.md` — detailed changelog
