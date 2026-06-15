# 3D Capture Reference (6× PanaCast 20 + 1× PanaCast 50, ChArUco, Desk + Glasses Markers)

## Scope

This document consolidates the current repository information for the 3D capture setup that uses:
- 6 × Jabra PanaCast 20
- 1 × Jabra PanaCast 50
- ChArUco boards (dynamic + fixed workflows)
- Desk ArUco markers
- Tobii glasses ArUco markers

Primary sources:
- `configs/ffmpeg_multicap.json`
- `configs/camera_specs.json`
- `configs/desk_markers_large.yaml`
- `configs/tobii_multicam_glasses_tracker.example.yaml`
- `configs/table_marker_map.yaml`
- `docs/llm/context_snapshot.md`
- `docs/jabra_recording_checklist.md`
- `docs/online_calibration.md`
- `docs/desk_marker_validation_example.md`
- `docs/dual_board_calibration_workflow.md`

---

## 1) Active camera rig and capture settings

### Camera count and roles

Current documented lab rig:
- `cam1`–`cam4`: PanaCast 20, ceiling-mounted, upside-down (require 180° orientation correction)
- `cam5`: PanaCast 20, front-center overview, upright
- `cam6`: PanaCast 20, rear/back overview, upright
- `cam7` (`jabra_panacast_50_vid`): PanaCast 50 wide room view, upright

### Per-device capture baseline (`configs/ffmpeg_multicap.json`)

For all 7 video devices:
- Resolution: `1920x1080`
- FPS: `30`
- Container: `mkv`
- `force_wallclock_timestamps: true`
- `show_camera_dialog: false`
- `input_video_codec: null` (auto-negotiate)

Orientation metadata:
- `rotate_180: true` on cam1–cam4 (P20 face cams)
- No 180° rotate flag on cam5, cam6, cam7

P50 quality override:
- `pixel_format: yuyv422` is explicitly set on P50 to avoid low-bitrate `nv12` default behavior.

### Camera model priors (`configs/camera_specs.json`)

- **PanaCast 20**
  - HFOV: `120°`
  - expected_fx @ 1080p: `~554.3 px`
  - expected k1 range: `[-0.45, -0.15]`
- **PanaCast 50**
  - HFOV: `133°`
  - expected_fx @ 1080p: `~418.2 px`
  - expected k1 range: `[-0.55, -0.20]`

---

## 2) ChArUco board setup (what exists in repo)

### Workflow support

Both offline and online workflows support ChArUco calibration:
- `tools/calibrate_charuco.py`
- `tools/online_calibration.py`

Board modes:
- `auto` (tries `5x3` and `7x5`)
- `fixed` (`--board-width`, `--board-height`)

### Board definitions seen in repo

1) In `configs/ffmpeg_multicap.json` (calibration metadata):
- Board A: `5x3`, square size `69 mm`
- Board B: `7x5`, square size `47 mm`

2) In dual-board/glasses-tracker docs/config examples:
- Dynamic board: often `5x3`, `69 mm`
- Fixed board: often `7x5`, `69 mm`

> Note: There is a **size inconsistency for 7x5** across files (`47 mm` vs `69 mm`). Use the actually printed board size when calibrating (`--square-size-mm`) and keep board config aligned with physical print.

### Practical calibration expectations

From current docs and tools:
- Capture with good spatial coverage (center + edges/corners)
- Keep board visible in multiple cameras at the same time
- Use sync artifacts (`frame-log`, `record-lsl`, progress/events) for robust multi-camera alignment
- Validate reprojection and camera health before downstream 3D reconstruction

---

## 3) Desk marker systems (two schemas currently present)

## A) Lab dual-board marker map (`configs/desk_markers_large.yaml`)

Coordinate frame:
- Origin: fixed 7x5 board center (`fixed_7x5_board_center`)
- Axes: `x right`, `y back`, `z up`
- Marker dictionary: `DICT_4X4_50`

Desk geometry:
- Width: `1.800 m`
- Depth: `0.800 m`
- Height: `0.750 m`

Desk markers:
- Marker size: `0.050 m` (50 mm)
- IDs: `0..5`
- Placement: 4 corners + left/right edge centers
  - `front_left: 0`
  - `front_right: 1`
  - `back_right: 2`
  - `back_left: 3`
  - `left_center: 4`
  - `right_center: 5`

Also includes:
- `fixed_charuco_board` block (7x5 board as world origin)
- Camera placement hints (`cam1..cam7` positions/focus)
- Participant seat mapping (`P1..P4`)

## B) Exported table marker map format (`configs/table_marker_map.yaml`)

This is the lightweight validation/export schema used by calibration export flows:
- Dictionary: `DICT_4X4_50`
- Marker IDs usually `0..4`
- Typical mapping:
  - `front_left: 0`
  - `front_right: 1`
  - `back_right: 2`
  - `back_left: 3`
  - `center: 4`

In current file:
- `width_m: 1.7740`
- `depth_m: 0.7780`
- `marker_size_m: 0.052`

> Interpretation: repo supports both `world.marker_map` and `table_markers` style maps in tools, plus this compact `table_marker_map.yaml` export style for validation.

---

## 4) Glasses marker setup (Tobii multi-glasses tracking)

From `configs/tobii_multicam_glasses_tracker.example.yaml` and `configs/desk_markers_large.yaml`:

Marker dictionary:
- `DICT_4X4_50`

Per-glasses pair definitions:
- `tobii_p1`: marker IDs `10` (left), `11` (right)
- `tobii_p2`: marker IDs `12`, `13`
- `tobii_p3`: marker IDs `14`, `15`
- `tobii_p4`: marker IDs `16`, `17`

Geometry assumptions:
- Marker size: `0.025 m` (25 mm)
- Marker center distance: `140 mm`
- Camera offset from front edge line: `60 mm`
- Left/right marker offsets: `±72.5 mm` on x-axis

Used by:
- `tools/tobii_multicam_glasses_tracker.py`
- `tools/tobii_multi_glasses_world_align.py` (downstream world alignment workflows)

---

## 5) End-to-end 3D calibration and validation flow

## A) Online calibration + export

Typical command shape:

```bash
python tools/online_calibration.py \
  --config configs/ffmpeg_multicap.json \
  --duration 60 \
  --board-mode auto \
  --square-size-mm <printed_size_mm> \
  --show-feed \
  --export-calibration \
  --export-session-dir <session_dir>
```

Optional desk marker map export:

```bash
python tools/online_calibration.py \
  --config configs/ffmpeg_multicap.json \
  --duration 60 \
  --export-calibration \
  --export-session-dir <session_dir> \
  --export-table-marker-map \
  --table-marker-ids "0,1,2,3,4" \
  --table-width-cm 180 \
  --table-depth-cm 120 \
  --table-marker-size-mm 50
```

## B) Robust validation with desk markers

```bash
python tools/validate_calibration_robust.py \
  --toml <session>/video/video_camera_calibration.toml \
  --marker-map <session>/video/table_marker_map.yaml \
  --videos-dir <session>/video \
  --camera-config configs/ffmpeg_multicap.json
```

Validation heuristics in docs:
- Mean marker reprojection error `< 5 px`: good
- `5–10 px`: fair
- `> 10 px`: poor (recalibrate)

---

## 6) Camera orientation and downstream 3D notes

- Cam1–Cam4 are physically upside-down in the described rig.
- Orientation correction should be applied consistently using either:
  - `rotate_180` metadata from `configs/ffmpeg_multicap.json`, or
  - explicit flip-camera patterns in downstream tools.

This matters for:
- ChArUco detection consistency
- Desk marker reprojection validation
- 3D triangulation and pose reconstruction correctness

---

## 7) Key implementation files

Capture + calibration:
- `tools/ffmpeg_multicap.py`
- `tools/calibrate_charuco.py`
- `tools/online_calibration.py`
- `tools/validate_calibration_robust.py`

3D + marker workflows:
- `tools/multicam_pose3d.py`
- `tools/online_multicam_feed.py`
- `tools/tobii_multicam_glasses_tracker.py`
- `tools/tobii_multi_glasses_world_align.py`
- `tools/generate_aruco_marker_sheet.py`

Configs:
- `configs/ffmpeg_multicap.json`
- `configs/camera_specs.json`
- `configs/desk_markers_large.yaml`
- `configs/tobii_multicam_glasses_tracker.example.yaml`
- `configs/table_marker_map.yaml`

---

## 8) Suggested single source of truth (operational recommendation)

To avoid drift across files, choose one active profile per study run and lock:
- board types + exact printed square sizes
- desk marker IDs and physical dimensions
- glasses marker IDs and geometry
- camera orientation flags and naming

Then keep those values synchronized across:
- `configs/ffmpeg_multicap.json`
- marker map YAML(s)
- glasses tracker config
- calibration run commands
