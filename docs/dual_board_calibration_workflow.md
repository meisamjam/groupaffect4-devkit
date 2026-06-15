# Dual-Board Calibration Workflow

## Overview

Custom calibration strategy using:
1. **Dynamic 5x3 ChArUco** - moved around workspace during calibration
2. **Fixed 7x5 ChArUco** - placed at desk back-center as world origin reference
3. **Large 100mm corner markers** - DICT_4X4_50 for eye tracker + desk tracking

## Board Setup

### Dynamic Board (5x3 ChArUco)
- **Size**: 345mm × 207mm (5 squares × 3 squares, 69mm each)
- **Dictionary**: DICT_4X4_250
- **Marker IDs**: 0-6
- **Usage**: Move around during initial calibration recording
- **Print command**:
  ```powershell
  C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py generate-board `
    --board-type 5x3 `
    --square-size 69 `
    --output calibration_board_5x3_69mm.png
  ```

### Fixed Board (7x5 ChArUco)
- **Size**: 483mm × 345mm (7 squares × 5 squares, 69mm each)
- **Dictionary**: DICT_4X4_250
- **Marker IDs**: 0-23 (overlaps with 5x3, but duplicate filtering handles this)
- **Position**: Back-center of desk (X=887mm from left edge, Y=650mm from front)
- **Visible cameras**: PanaCast 20 (cam1-6) ✓, PanaCast 50 ❌
- **Usage**: Continuous validation, world origin reference, bundle adjustment
- **Print command**:
  ```powershell
  C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py generate-board `
    --board-type 7x5 `
    --square-size 69 `
    --output calibration_board_7x5_69mm.png
  ```

### Desk Corner Markers (ArUco only)
- **Size**: 100mm × 100mm each
- **Dictionary**: DICT_4X4_50 (different from ChArUco to avoid ID collision!)
- **Marker IDs**: 10, 11, 12, 13 (corners)
- **Position**: 1cm inset from desk edges
- **Generate**: Use https://chev.me/arucogen/
  - Select "4x4 (50 markers)" 
  - Generate IDs: 10, 11, 12, 13
  - Print at 100mm × 100mm

## Calibration Workflow

### Option A: Single Calibration (All 7 Cameras)

Use dynamic 5x3 board for all cameras:

```powershell
# 1. Record calibration video (move 5x3 board around, fixed 7x5 visible in background)
C:/Users/meisa/.conda/envs/affectai/python.exe tools/ffmpeg_multicap.py `
  --config configs/ffmpeg_multicap.json

# 2. Calibrate (duplicate-ID filtering handles both boards)
C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py calibrate `
  --videos-dir data/sub-XXX/ses-YYY/video `
  --square-size 69 `
  --board-type 5x3 `
  --init-focal `
  --camera-specs configs/camera_specs.json

# 3. Validate with desk markers
C:/Users/meisa/.conda/envs/affectai/python.exe tools/validate_calibration_robust.py `
  --toml calibration_charuco.toml `
  --marker-map configs/desk_markers_large.yaml `
  --videos-dir data/sub-XXX/ses-YYY/video `
  --camera-config configs/ffmpeg_multicap.json
```

**Pros**: Simple, single calibration step  
**Cons**: PanaCast 50 might have poor coverage (fixed 7x5 not visible)

---

### Option B: Two-Stage Calibration (Recommended)

Stage 1: Initial calibration (all 7 cameras, dynamic 5x3 board only)  
Stage 2: Refine P20 cameras using fixed 7x5 board

```powershell
# STAGE 1: Initial calibration with dynamic 5x3 board
C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py calibrate `
  --videos-dir data/sub-XXX/ses-YYY/video `
  --square-size 69 `
  --board-type 5x3 `
  --init-focal `
  --camera-specs configs/camera_specs.json `
  --output calibration_stage1_all_cameras.toml

# STAGE 2: Refine P20 cameras (cam1-6) using fixed 7x5 board
# (Manual step: create subset video directory with only cam1-6)
New-Item -ItemType Directory -Path data/sub-XXX/ses-YYY/video_p20_only
Copy-Item data/sub-XXX/ses-YYY/video/*cam[1-6]*.mkv data/sub-XXX/ses-YYY/video_p20_only/

C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py calibrate `
  --videos-dir data/sub-XXX/ses-YYY/video_p20_only `
  --square-size 69 `
  --board-type 7x5 `
  --init-focal `
  --camera-specs configs/camera_specs.json `
  --output calibration_stage2_p20_refined.toml

# STAGE 3: Merge calibrations (manual - copy P50 from stage1, P20 from stage2)
# Or just use stage2 for P20 cameras and stage1 for P50 separately
```

**Pros**: P20 cameras get refined calibration from fixed board  
**Cons**: More complex, need to manage two calibration files

---

### Option C: PanaCast 50 Separate Calibration

If P50 cannot see fixed 7x5 board at all:

```powershell
# Calibrate P20 cameras (cam1-6) with fixed 7x5 board
C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py calibrate `
  --videos-dir data/sub-XXX/ses-YYY/video_p20_only `
  --square-size 69 `
  --board-type 7x5 `
  --init-focal `
  --camera-specs configs/camera_specs.json `
  --output calibration_p20.toml

# Calibrate P50 camera separately with dynamic 5x3 board
# (Record during separate session where P50 can see dynamic board clearly)
C:/Users/meisa/.conda/envs/affectai/python.exe tools/calibrate_charuco.py calibrate `
  --videos-dir data/sub-XXX/ses-YYY/video_p50_only `
  --square-size 69 `
  --board-type 5x3 `
  --init-focal `
  --camera-specs configs/camera_specs.json `
  --output calibration_p50.toml

# Merge manually (copy P50 from calibration_p50.toml into calibration_p20.toml)
```

**Pros**: Each camera group optimized independently  
**Cons**: Most complex, requires manual TOML merging

## Physical Setup

### Desk Layout
```
                    1774mm
    ┌──────────────────────────────────┐
    │ [10]                       [11]  │
    │  ↖                           ↗   │
    │                                  │ 778mm
    │         FIXED 7x5 BOARD          │
    │         (483 x 345mm)            │
    │       @ back-center              │
    │                                  │
    │  ↙                           ↘   │
    │ [13]                       [12]  │
    └──────────────────────────────────┘

[10,11,12,13] = 100mm ArUco markers (DICT_4X4_50)
Fixed board center: (887mm, 650mm, 0mm) from front-left corner
```

### Camera Views
- **PanaCast 20 (cam1-6)**: Can see fixed 7x5 board ✓
- **PanaCast 50**: Cannot see fixed board properly (too wide FOV/mounted high) ❌

## Bundle Adjustment (Future Enhancement)

After initial calibration, use fixed 7x5 board + desk markers for continuous refinement:

```powershell
# Detect fixed board + desk markers in normal recording
C:/Users/meisa/.conda/envs/affectai/python.exe tools/online_multicam_feed.py `
  --config configs/ffmpeg_multicap.json `
  --calibration calibration_charuco.toml `
  --marker-map configs/desk_markers_large.yaml

# Run bundle adjustment to refine extrinsics using both:
# - Fixed 7x5 ChArUco corners (high accuracy)
# - Desk corner markers (world frame anchors)
# (Tool not yet implemented - future work)
```

## Troubleshooting

### "Duplicate marker IDs detected"
✓ Expected! Your tool has duplicate-ID filtering specifically for this case.  
Both 5x3 and 7x5 boards share IDs 0-6. The filtering picks the geometrically consistent cluster.

### Fixed board not detected
- Check P20 camera views - is board visible and flat?
- Ensure board is well-lit
- Board might be too close/far for some cameras
- Try adjusting board position slightly forward (~600mm Y instead of 650mm)

### PanaCast 50 poor calibration
- Option B or C recommended (separate calibration)
- P50 has 133° FOV - fixed board might be too small/close
- Consider moving P50 calibration to use dynamic board exclusively

### Desk markers not detected (validation step)
- Ensure dictionary is DICT_4X4_50 (not DICT_4X4_250!)
- Check marker size is truly 100mm (measure with ruler)
- Markers should be flat and well-lit

## Files Generated

- `calibration_board_5x3_69mm.png` - Dynamic board (print on A3/A4)
- `calibration_board_7x5_69mm.png` - Fixed board (print on A2 or 2×A3)
- `configs/desk_markers_large.yaml` - Marker map with 100mm corners
- `calibration_charuco.toml` - Initial calibration result
- Optional: `calibration_p20.toml` and `calibration_p50.toml` for separate workflows

## Verification

After calibration, check focal lengths:

```powershell
# Should see fx ≈ 554px for P20 cameras, fx ≈ 418px for P50
Get-Content calibration_charuco.toml | Select-String -Pattern "matrix.*\[" -Context 1,0
```

Good calibration:
- fx within ±10% of expected (P20: 499-609px, P50: 376-460px)
- All cameras have similar fx (±5% variation)
- Distortion k1 within expected range (P20: -0.45 to -0.15)

## References

- Calibration tool: `tools/calibrate_charuco.py`
- Validation tool: `tools/validate_calibration_robust.py`
- Camera specs: `configs/camera_specs.json`
- Desk dimensions: 1774mm × 778mm
