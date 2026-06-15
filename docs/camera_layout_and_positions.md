# Camera Layout and Positions — AffectAI Lab Setup

> **Source of truth:** [`configs/desk_markers_large.yaml`](../configs/desk_markers_large.yaml),
> [`docs/recording_sync_calibration_pipeline.md`](recording_sync_calibration_pipeline.md),
> [`configs/ffmpeg_multicap.json`](../configs/ffmpeg_multicap.json)

---

## 1. Room Overview

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║    [WALL]──────────────────────────────────────────[WALL]         ║
║                                                                   ║
║                    ┌───────────────┐                              ║
║                    │  BIG SCREEN   │  ← HDMI → Recording PC      ║
║                    │  (rear wall)  │    /bigscreen endpoint        ║
║                    └───────┬───────┘                              ║
║                            │ (facing participants)                ║
║                                                                   ║
║   cam7/P50 ──────────────────────────────── cam6                  ║
║   (front-center wall,          (back-center,                       ║
║    upright, wide-angle)         upright, rear overview)            ║
║                                                                   ║
║         cam1 ┐           ┌ cam3                                   ║
║              │  ┌──────┐ │ ← upside-down, mounted at ~88 cm      ║
║              │  │      │ │                                         ║
║              │  │ DESK │ │   1.80 m × 0.80 m × 0.75 m tall       ║
║              │  │      │ │                                         ║
║              │  └──────┘ │                                         ║
║         cam4 ┘           └ cam2                                   ║
║              ↑ upside-down, mounted at ~88 cm                      ║
║                                                                   ║
║   cam5 (front-left low, table level, upright)                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

**Coordinate frame:** Origin = fixed ChArUco 7×5 board center (desk centre).
`x` = right, `y` = back (toward Big Screen wall), `z` = up.

---

## 2. Participant Seating (P1–P4)

```
                    ┌─────────────────────┐
                    │     BIG SCREEN      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
         P4 ──┼── back-left        back-right ──┼── P1
              │                                 │
              │        [D E S K]                │
              │       1.80 m × 0.80 m           │
              │      ChArUco board (centre)      │
              │      DPA room mic (centre)       │
              │                                 │
         P3 ──┼── front-left      front-right ──┼── P2
              │                                 │
              └─────────────────────────────────┘
```

| Seat | Position  | Tobii       | EmotiBit | DPA Mic        | Tablet   |
|------|-----------|-------------|----------|----------------|----------|
| P1   | back-right  | tg01      | #1       | Mic 9 (RME An9–10 L)  | Tablet 1 |
| P2   | front-right | tg02      | #2       | Mic 10 (RME An9–10 R) | Tablet 2 |
| P3   | front-left  | tg03      | #3       | Mic 11 (RME An11–12 L)| Tablet 3 |
| P4   | back-left   | tg04      | #4       | Mic 12 (RME An11–12 R)| Tablet 4 |

**Room microphone (DPA #5):** fixed at centre of table, Input 1 on RME Fireface 802.

**Tobii Glasses marker pairs** (ArUco ID, 25 mm, DICT_4×4_50):

| Participant | Left marker ID | Right marker ID |
|-------------|----------------|-----------------|
| P1          | 10             | 11              |
| P2          | 12             | 13              |
| P3          | 14             | 15              |
| P4          | 16             | 17              |

---

## 3. Camera Inventory

| Label  | Model         | Resolution   | FPS | Bitrate   | Orientation    | Mount position              |
|--------|---------------|-------------|-----|-----------|----------------|-----------------------------|
| cam1   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | **Upside-down** | Left-front side of desk, ~88 cm high |
| cam2   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | **Upside-down** | Right-front side of desk, ~88 cm high |
| cam3   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | **Upside-down** | Right-back side of desk, ~88 cm high |
| cam4   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | **Upside-down** | Left-back side of desk, ~88 cm high |
| cam5   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | Upright         | Front-left, table-level (0 cm) — overview |
| cam6   | PanaCast 20   | 1920 × 1080 | 30  | 4 000 kbps | Upright         | Back-center, ~88 cm — rear overview |
| cam7 (P50) | PanaCast 50 | 1920 × 1080 | 30 | 4 000 kbps | Upright       | Front-center wall/shelf, ~90 cm — wide-angle room view |

> **P50 note:** Maximum stable capture resolution is 1280×720 (not 1080p). Requires USB 3.0 SuperSpeed.
> In practice, config specifies 1920×1080 as the requested resolution; actual delivered may be capped by the device.
> Use `pixel_format: yuyv422` for P50.

---

## 4. Camera Positions in World Coordinates

Origin = ChArUco board centre (desk centre). Units: metres.

| Camera | Label              | x (m)   | y (m)   | z (m)  | Primary focus |
|--------|--------------------|---------|---------|--------|---------------|
| cam1   | left_front_middle  | −0.9000 | −0.2000 | 0.8800 | P1, P2        |
| cam2   | right_front_middle | +0.9000 | −0.2000 | 0.8800 | P3, P4        |
| cam3   | right_back_middle  | +0.9000 | +0.2000 | 0.8800 | P3, P4        |
| cam4   | left_back_middle   | −0.9000 | +0.2000 | 0.8800 | P1, P2        |
| cam5   | front_left_low     | −0.9000 | −0.4000 | 0.0000 | table overview |
| cam6   | back_center        | 0.0000  | +0.4000 | 0.8800 | all           |
| cam7   | front_center_p50   | 0.0000  | −0.4000 | 0.9000 | all           |

---

## 5. Camera–Participant Zone Assignment

Cameras are grouped into zones for 3D pose reconstruction:

```
Zone A (left side):   cam1 + cam4  →  P1 (back-right) + P2 (front-right)
Zone B (right side):  cam2 + cam3  →  P3 (front-left) + P4 (back-left)
Shared (all):         cam5, cam6, cam7/P50 (auto-assigned per frame)
```

```
         cam2 (right-front)       cam3 (right-back)
               ↗                       ↖
          ┌──────────────────────────────────┐
          │  P4 (back-left)   P1 (back-right)│
          │                                  │
          │          [ desk ]                │
          │                                  │
          │  P3 (front-left)  P2 (front-right│
          └──────────────────────────────────┘
               ↖                       ↗
         cam4 (left-back)        cam1 (left-front)

    cam5 (front-left, table level)   cam6 (back-center)
    cam7/P50 (front-center, wide)
```

**CLI argument for `multicam_pose3d.py`:**
```bash
--camera-zones "cam1+cam4:0,1" "cam2+cam3:2,3"
```

Person IDs in pipeline: 0=P1, 1=P2, 2=P3, 3=P4.

---

## 6. Camera Configuration Settings

All cameras share these `ffmpeg_multicap.json` flags:

| Setting                      | Value   | Reason                                                       |
|------------------------------|---------|--------------------------------------------------------------|
| `force_wallclock_timestamps` | `true`  | System time is more stable than device timestamps across USB hubs |
| `mux_audio`                  | `false` | Audio captured separately; prevents USB bandwidth contention |
| `input_video_codec`          | `mjpeg` | Forces MJPEG from camera (prevents raw YUY2 = multi-GB files) |
| `format`                     | `mkv`   | Container with embedded timestamp metadata                   |
| `rotate_180`                 | `true`  | cam1–cam4 only — hardware-mounted upside-down               |
| `video_alt_name`             | PnP path | DirectShow unique device path (required when multiple cameras share same display name) |

Cameras cam1–cam4 additionally require:
- `"rotate_180": true` in config → ffmpeg applies `vf=transpose=2,transpose=2` filter at capture time
- `--flip-cameras cam_0 cam_1 cam_2 cam_3` in post-processing (calibration-space rotation)

---

## 7. Desk Physical Dimensions & ArUco Marker Map

**Desk:** 1.80 m wide × 0.80 m deep × 0.75 m tall

Desk-edge markers for spatial calibration (50 mm, DICT_4×4_50):

| ID | Label              | x (m)   | y (m)   | Position        |
|----|--------------------|---------|---------|-----------------|
| 0  | front_left_corner  | −0.9250 | −0.3750 | Front-left edge |
| 1  | front_right_corner | +0.8750 | −0.3750 | Front-right edge|
| 2  | back_right_corner  | +0.8750 | +0.4250 | Back-right edge |
| 3  | back_left_corner   | −0.9250 | +0.4250 | Back-left edge  |
| 4  | left_center        | −0.9250 | 0.0000  | Left middle edge|
| 5  | right_center       | +0.8750 | 0.0000  | Right middle edge|

**Fixed ChArUco calibration board** (at desk centre, is world origin):
- Type: 7×5, square size 69 mm, dictionary DICT_4×4_250
- Centre position: (0.0, 0.0, 0.0) — world coordinate origin

---

## 8. Big Screen

| Property        | Value                                          |
|-----------------|------------------------------------------------|
| Connection      | HDMI → Recording PC                           |
| Physical position | Rear wall, facing all four participants       |
| Software endpoint | `http://<recording-pc>:8080/bigscreen`       |
| LSL stream      | `AffectAI_BigScreen`                          |
| Content         | Shared task stimuli, group outcomes, timers   |
| Role in seating | Behind P1/P4 (back-row side), visible to P2/P3 |

---

## 9. ChArUco Spatial Calibration Summary

| Parameter        | Value                       |
|------------------|-----------------------------|
| Board type       | ChArUco 5×3 (capture) / 7×5 (desk origin) |
| Square size      | 52 mm (capture board) / 69 mm (desk board) |
| Dictionary       | DICT_4×4_50 / DICT_4×4_250 |
| Required library | freemocap ≥ 1.3.0, opencv ≥ 4.8 |
| Output           | per-camera TOML: name, matrix (3×3), distortions (5), rotation (Rodrigues), translation, world_orientation/position |
| Detection rates  | cam1: 62%, cam2: 94%, cam3: 95%, cam4: 86%, P50: 90% |

---

## 10. Hardware Requirements

- **USB 3.0 (SuperSpeed) ports** required for all cameras — especially P50
- **P50** must use USB 3.0 cable (USB 2.0–compatible cable insufficient)
- **Separate USB root hubs** recommended for camera clusters (max 2 cameras per hub)
- **USB 3.0 cable** for P50 (USB 2.0 compatible cable will not reliably deliver video)

---

## 11. Cross-References

| Topic | File |
|-------|------|
| Full sync pipeline & calibration workflow | [`docs/recording_sync_calibration_pipeline.md`](recording_sync_calibration_pipeline.md) |
| Sync best practices & real-world measurements | [`docs/SYNC_BEST_PRACTICES.md`](SYNC_BEST_PRACTICES.md) |
| ffmpeg capture config (all devices) | [`configs/ffmpeg_multicap.json`](../configs/ffmpeg_multicap.json) |
| Desk marker definitions (YAML) | [`configs/desk_markers_large.yaml`](../configs/desk_markers_large.yaml) |
| Participant device setup checklist | [`docs/data_collection_guide.md`](data_collection_guide.md) |
| Dual-board calibration workflow | [`docs/dual_board_calibration_workflow.md`](dual_board_calibration_workflow.md) |
| 3D pose reconstruction CLI | [`tools/multicam_pose3d.py`](../tools/multicam_pose3d.py) |
| Camera specs (intrinsics reference) | [`configs/camera_specs.json`](../configs/camera_specs.json) |
