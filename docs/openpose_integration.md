# Multi-Person 3D Skeleton Tracking with OpenPose

5-camera Jabra PanaCast setup → OpenPose 2D pose detection → calibration-based 3D triangulation.

## Setup

- **Cameras:** 5× Jabra PanaCast (cam1–cam5)
- **Calibration:** `video_camera_calibration.toml` (from ChArUco)
- **OpenPose:** Local in `tools/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/`

Verify installation:
```bash
python scripts/run_openpose.py --help
```

### Installing OpenPose (if not already present)

Download pre-built release: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases

Or build from source (Visual Studio + CMake):
```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Step 1: Convert MKV → MP4

```batch
@echo off
for %%i in (1 2 3 4 5) do (
  echo Converting cam%%i...
  ffmpeg -i jabra_panacast_20_cam%%i_vid_video.mkv ^
    -c:v libx264 -preset fast -crf 18 ^
    jabra_panacast_20_cam%%i_vid_video.mp4 -y
)
```

## Step 2: Run OpenPose on All Cameras

**Easy method** — batch-run all 5 in parallel:
```bash
scripts\process_all_cameras_openpose.bat
```

**Manual** — one camera at a time:
```bash
python scripts/run_openpose.py --video "data/sub-meisam/ses-20260202_test/video/jabra_panacast_20_cam1_vid_video.mp4" ^
  --output openpose_output/output_cam1_json
```

Repeat for cam2–cam5. JSON poses appear in `openpose_output/output_cam*_json/`.

### OpenPose JSON format

Each frame produces a file like `000000000000_keypoints.json`:
```json
{
  "version": 1.8,
  "people": [
    {"person_id": [0], "pose_keypoints_2d": [x0, y0, conf0, x1, y1, conf1, ...]},
    {"person_id": [1], "pose_keypoints_2d": [...]}
  ]
}
```

### Useful OpenPose flags

| Flag | Effect |
|---|---|
| `--display 0` | Disable GUI (5× faster) |
| `--number_people_max 10` | Cap detections per frame |
| `--write_video <out>` | Save annotated video for QC |
| `--render_pose 1` | Overlay skeleton on video |

## Step 3: Triangulate to 3D

```bash
python tools/triangulate_openpose.py triangulate ^
  --calibration "data/sub-meisam/ses-20260202_test/video/video_camera_calibration.toml" ^
  --pose-dirs ^
    "openpose_output/output_cam1_json" ^
    "openpose_output/output_cam2_json" ^
    "openpose_output/output_cam3_json" ^
    "openpose_output/output_cam4_json" ^
    "openpose_output/output_cam5_json" ^
  --output "results/skeleton_3d_multi_person.npy"
```

**Output:** `skeleton_3d_multi_person.npy`
- Shape: `(n_frames, n_people, 25, 4)` where 4 = `[x, y, z, confidence]` in mm

## Step 4: Validate & Analyse

```bash
python tools/triangulate_openpose.py validate --file results/skeleton_3d_multi_person.npy
```

### Load in Python

```python
import numpy as np, json

skeleton_3d = np.load('results/skeleton_3d_multi_person.npy')
with open('results/skeleton_3d_multi_person.json') as f:
    metadata = json.load(f)

# Person 0, right wrist, all frames
person_0_rwrist = skeleton_3d[:, 0, 10, :3]  # (n_frames, 3)

# Frame 100, person 2, all joints
frame_100_p2 = skeleton_3d[100, 2, :, :3]    # (25, 3)
```

### Plot trajectory

```python
import matplotlib.pyplot as plt

head = skeleton_3d[:, 0, 0, :3]  # Nose
valid = ~np.isnan(head).any(axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(head[valid, 0], head[valid, 1], head[valid, 2], 'b-')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
ax.set_title('Person 0 Head Trajectory')
plt.show()
```

## Keypoint indices (BODY_25)

```
 0  Nose           9  Left wrist     17  Left ankle
 1  Neck          10  Right wrist    18  Right ankle
 2  Right shoulder 11 Left hip       19  Left big toe
 3  Right elbow   12  Right hip      20  Left small toe
 4  Right wrist   13  Left knee      21  Left heel
 5  Left shoulder 14  Right knee     22  Right big toe
 6  Left elbow    15  Left eye       23  Right small toe
 7  Left wrist    16  Right eye      24  Right heel
 8  Mid hip       
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| No people detected | Poor lighting / resolution | Check video quality; try `--model_type BODY_25B` |
| "list index out of range" | Missing/unordered JSONs | Verify sequential numbering in output dirs |
| 3D positions are NaN | Joint occluded in <2 cameras | Expected for partly hidden joints |
| Large reprojection errors | Calibration inaccuracy | Re-run charuco; `python scripts/validate_calibration.py` |
| Jittery 3D trajectories | Frame sync / low confidence | Use `tools/refine_skeleton_3d.py` post-processing |

## Performance tips

- Run OpenPose on all 5 cameras in parallel (separate terminals)
- Test on a short clip first: `ffmpeg -i video.mp4 -t 1 video_short.mp4`
- Always use `--display 0` for speed
- Monitor progress: `dir output_cam1_json/ /s | find /c "json"`

## Complete copy-paste workflow

```batch
REM 1. Verify
python scripts/run_openpose.py --help

REM 2. Convert videos
for %%i in (1 2 3 4 5) do ffmpeg -i jabra_panacast_20_cam%%i_vid_video.mkv -c:v libx264 -preset fast -crf 18 jabra_panacast_20_cam%%i_vid_video.mp4 -y

REM 3. Run OpenPose
scripts\process_all_cameras_openpose.bat

REM 4. Triangulate
python tools/triangulate_openpose.py triangulate ^
  --calibration "data/sub-meisam/ses-20260202_test/video/video_camera_calibration.toml" ^
  --pose-dirs "openpose_output/output_cam1_json" "openpose_output/output_cam2_json" "openpose_output/output_cam3_json" "openpose_output/output_cam4_json" "openpose_output/output_cam5_json" ^
  --output "results/skeleton_3d_multi_person.npy"

REM 5. Validate
python tools/triangulate_openpose.py validate --file results/skeleton_3d_multi_person.npy
```
