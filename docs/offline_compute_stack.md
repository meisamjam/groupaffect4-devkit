# Offline Compute Stack

This guide describes how to use two Windows workstations plus one 14 TB USB 3 drive as an
offline processing stack for AffectAI video, pose, gaze, and QC workloads.

The stack is for post-collection processing. It does not replace the original AV PC /
Recording PC capture roles; it is the recommended layout for running expensive work after the
raw sessions have already been collected.

## Hardware Roles

| Node | Hardware | Recommended role |
|------|----------|------------------|
| `gpu-main` | Lenovo Legion T7, RTX 5080, Core Ultra 9, 64 GB RAM, 2 TB SSD | Primary GPU worker for 3D/video inference and the heaviest batch jobs |
| `storage-worker` | Phoenix AMD workstation, Ryzen 7 7800X3D, RTX A2000 12 GB, 64 GB RAM, 4 TB + 4 TB SSD | Fast staging store, secondary GPU worker, BIDS/QC jobs |
| `archive-usb` | 14 TB USB 3 external drive | Archive and backup for raw and finished outputs |

Use the internal SSDs for active computation. Use the USB disk as an archive target, not as the
live working directory for GPU inference.

## Network Topology

Recommended direct cable:

```text
gpu-main  <---- dedicated Ethernet cable ---->  storage-worker
```

Prefer 10 GbE if available. 1 GbE works for control and small transfers, but it is slow for
large multicamera sessions. If the machines only have 1 GbE today, use the workflow below
anyway: copy one job to local SSD, process locally, then sync outputs back.

Example static IPs for the dedicated cable:

| Machine | Dedicated NIC IPv4 | Subnet mask | Gateway |
|---------|--------------------|-------------|---------|
| `gpu-main` | `10.10.10.1` | `255.255.255.0` | blank |
| `storage-worker` | `10.10.10.2` | `255.255.255.0` | blank |

Keep normal internet/lab networking on the other NIC or Wi-Fi. Do not set a gateway on the
dedicated direct-cable NIC unless it is also your internet route.

### Windows Setup Commands

Run PowerShell as Administrator on `gpu-main`:

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 10.10.10.1 -PrefixLength 24
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ResetServerAddresses
```

Run PowerShell as Administrator on `storage-worker`:

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 10.10.10.2 -PrefixLength 24
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ResetServerAddresses
```

Confirm connectivity from `gpu-main`:

```powershell
ping 10.10.10.2
```

If the interface is not named `Ethernet`, list names first:

```powershell
Get-NetAdapter
```

## Storage Layout

Use clear drive roles. Example:

```text
storage-worker
  D:\affectai_stage\          fast SSD staging area
  D:\affectai_stage\processed\ processed BIDS / mocap / QC outputs
  H:\affectai_archive\        14 TB USB archive, if attached here

gpu-main
  D:\affectai_work\           local scratch for active GPU jobs
  Z:\                         mapped share to \\10.10.10.2\affectai_stage
```

Recommended share on `storage-worker`:

```powershell
New-Item -ItemType Directory -Force D:\affectai_stage
New-SmbShare -Name affectai_stage -Path D:\affectai_stage -ChangeAccess "$env:USERNAME"
```

Map the share on `gpu-main`:

```powershell
New-PSDrive -Name Z -PSProvider FileSystem -Root \\10.10.10.2\affectai_stage -Persist
```

Do not connect the same USB drive directly to both machines. Attach it to one machine only, then
share folders over the network if the other machine needs access.

## Data Placement Rules

Use this pattern for every heavy video job:

1. Keep raw or finished archive copies on `archive-usb`.
2. Stage the next sessions on `storage-worker` internal SSD.
3. Copy the assigned session to the worker machine local SSD before GPU processing.
4. Write intermediate outputs locally.
5. Sync final outputs back to `storage-worker` and then archive to the 14 TB disk.

Example copy from `storage-worker` share to `gpu-main` local SSD:

```powershell
robocopy Z:\sessions\Final\grp-06 D:\affectai_work\grp-06 /MIR /MT:16 /R:2 /W:5
```

Example sync finished outputs back to the stage share:

```powershell
robocopy D:\affectai_work\grp-06\outputs Z:\outputs\grp-06 /MIR /MT:16 /R:2 /W:5
```

Use `/MIR` only when the destination is a disposable mirror folder for that session. For archive
folders, prefer `/E` until you are confident the source/destination pairing is correct.

## Processing Strategy

Split work by session, not by having both machines write into the same output directory.

Recommended assignment:

| Workload | Preferred node | Notes |
|----------|----------------|-------|
| BIDS merging and task splitting | `storage-worker` | SSD capacity is useful; use a small worker count first |
| Video feature extraction (`extract_video_features.py`) | `gpu-main`, then `storage-worker` for extra sessions | Keep videos local while extracting sync/marker/body/face/hand features |
| Legacy MediaPipe/OpenPose pose JSON | `gpu-main` | Still needed by `video_only_3d_pipeline.py --pose-root` until feature-native 3D adapters replace it |
| `video_only_3d_pipeline.py` | `gpu-main` | RTX 5080 should be the primary heavy worker |
| QC reports | `storage-worker` | Mostly CPU and disk I/O; can run while GPU jobs run elsewhere |
| Archive copy | `storage-worker` | Keep the USB disk on one machine for predictable ownership |

Do not run two writers against the same BIDS session directory. A good rule is:

```text
one session + one output directory + one machine at a time
```

## Environment Setup

On both machines, use the same repository revision and environment name:

```powershell
cd C:\Codes\affectai-processing\affectai-data-processing
conda activate affectai
python --version
git rev-parse --short HEAD
```

Check the command help before a full run:

```powershell
python tools\multisource_to_bids_runs.py --help
python tools\bids_processing_pipeline.py --help
python tools\extract_video_features.py --help
python tools\video_only_3d_pipeline.py --help
python tools\qc\qc_sync_report.py --help
```

## Example Workflow

### 1. Stage Raw Data

On `storage-worker`, copy raw sources from the USB archive to SSD staging:

```powershell
robocopy H:\affectai_archive\data D:\affectai_stage\data /E /MT:16 /R:2 /W:5
robocopy H:\affectai_archive\configs D:\affectai_stage\configs /E /MT:16 /R:2 /W:5
```

Expected staged structure for the wrapper pipeline:

```text
D:\affectai_stage\
  data\
    affectai-capture-recording\
    AV\
    Tobii\
    stimuli\
    high_level_session_inventory.csv
  configs\
```

### 2. Run BIDS Packaging on `storage-worker`

Start conservatively with two workers; increase only if disk I/O is not saturated.

```powershell
cd C:\Codes\affectai-processing\affectai-data-processing
conda activate affectai

python tools\bids_processing_pipeline.py `
  --data-root D:\affectai_stage\data `
  --output-root D:\affectai_stage\processed\bids `
  --inventory D:\affectai_stage\data\high_level_session_inventory.csv `
  --config-dir D:\affectai_stage\configs `
  --max-workers 2 `
  --split-media
```

If you want a single session or need more manual control, run the lower-level tool directly:

```powershell
python tools\multisource_to_bids_runs.py `
  --av-session-dir D:\affectai_stage\data\AV\final\ses-20260311_grp-06_run01 `
  --recording-session-dir D:\affectai_stage\data\affectai-capture-recording\sessions\final\sub-01\ses-20260311_grp-06_run01 `
  --stimuli-dir D:\affectai_stage\data\stimuli\final\ses-20260311_grp-06_run01 `
  --output-session-dir D:\affectai_stage\processed\bids\sub-01\ses-20260311_grp-06_run01 `
  --split-media
```

Adjust paths to the actual session directory names. Keep participant references anonymized as
`P1` to `P4` or BIDS `sub-*` IDs in logs and output folders.

### 3. Extract Video Features on `gpu-main`

Copy one packaged session to the local GPU scratch disk:

```powershell
robocopy Z:\processed\bids\sub-01\ses-20260311_grp-06_run01 D:\affectai_work\ses-20260311_grp-06_run01 /E /MT:16 /R:2 /W:5
```

Run a feature preflight first:

```powershell
cd C:\Codes\affectai-processing\affectai-data-processing
conda activate affectai

python tools\extract_video_features.py `
  --videos-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --output-dir D:\affectai_work\ses-20260311_grp-06_run01\features_video `
  --marker-config configs\desk_markers_large.yaml `
  --dry-run
```

Then run the baseline MediaPipe feature extraction:

```powershell
python tools\extract_video_features.py `
  --videos-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --output-dir D:\affectai_work\ses-20260311_grp-06_run01\features_video `
  --marker-config configs\desk_markers_large.yaml `
  --body --hands --faces --markers `
  --body-backbone mediapipe-pose `
  --aruco-dicts DICT_4X4_50,DICT_4X4_250
```

For RTMPose/RTMW body features on the RTX worker, use a dedicated MMPose-capable environment:

```powershell
python tools\extract_video_features.py `
  --videos-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --output-dir D:\affectai_work\ses-20260311_grp-06_run01\features_video_rtmpose `
  --marker-config configs\desk_markers_large.yaml `
  --body --no-hands --no-faces --markers `
  --body-backbone rtmpose-mmpose `
  --rtmpose-model rtmw-l `
  --device cuda:0
```

Current bridge step: generate OpenPose-compatible MediaPipe JSON for the session videos when
running `video_only_3d_pipeline.py`, because that pipeline still consumes `--pose-root`:

```powershell
python tools\test_mediapipe_pose.py `
  --session-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --write-json D:\affectai_work\ses-20260311_grp-06_run01\mediapipe `
  --max-frames 0 `
  --model-complexity 2
```

### 4. Dry-Run the 3D Pipeline

Always validate prerequisites before a long GPU run:

```powershell
python tools\video_only_3d_pipeline.py `
  --calibration D:\affectai_work\ses-20260311_grp-06_run01\video\video_camera_calibration.toml `
  --videos-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --tracker-config configs\tobii_multicam_glasses_tracker.example.yaml `
  --pose-root D:\affectai_work\ses-20260311_grp-06_run01\mediapipe `
  --output-dir D:\affectai_work\ses-20260311_grp-06_run01\video_only_3d `
  --camera-zones cam1+cam4:0,1 cam2+cam3:2,3 `
  --flip-cameras cam_0 cam_1 cam_2 cam_3 `
  --refine-skeleton `
  --dry-run
```

Inspect:

```powershell
Get-Content D:\affectai_work\ses-20260311_grp-06_run01\video_only_3d\pipeline_dry_run.json
```

### 5. Run the 3D Pipeline

Remove `--dry-run` when the prerequisite report is ready:

```powershell
python tools\video_only_3d_pipeline.py `
  --calibration D:\affectai_work\ses-20260311_grp-06_run01\video\video_camera_calibration.toml `
  --videos-dir D:\affectai_work\ses-20260311_grp-06_run01\video `
  --tracker-config configs\tobii_multicam_glasses_tracker.example.yaml `
  --pose-root D:\affectai_work\ses-20260311_grp-06_run01\mediapipe `
  --output-dir D:\affectai_work\ses-20260311_grp-06_run01\video_only_3d `
  --camera-zones cam1+cam4:0,1 cam2+cam3:2,3 `
  --flip-cameras cam_0 cam_1 cam_2 cam_3 `
  --refine-skeleton
```

Sync outputs back:

```powershell
robocopy D:\affectai_work\ses-20260311_grp-06_run01\video_only_3d Z:\processed\mocap\ses-20260311_grp-06_run01 /E /MT:16 /R:2 /W:5
```

### 6. Run QC on `storage-worker`

`qc_sync_report.py` expects a session directory that still contains `sourcedata\sync`. If the
wrapper pipeline has already cleaned raw `sourcedata`, run this against the preserved staged or
merged session copy instead.

```powershell
python tools\qc\qc_sync_report.py `
  --session-dir D:\affectai_stage\processed\bids\sub-01\ses-20260311_grp-06_run01
```

For Tobii world-gaze QC after 3D processing:

```powershell
python tools\qc\qc_tobii_world_gaze.py `
  --input-dir D:\affectai_stage\processed\mocap\ses-20260311_grp-06_run01\tobii_world `
  --output-dir D:\affectai_stage\processed\qc\ses-20260311_grp-06_run01\tobii_world
```

Run each QC tool with `--help` first if the command fails; some options depend on the exact
artifact layout produced by the upstream run.

## Batch Scheduling

Keep a small manual queue file outside raw data, for example:

```text
D:\affectai_stage\work_queue.tsv
session_id	status	assigned_to	notes
ses-20260311_grp-06_run01	bids	storage-worker	initial package
ses-20260312_grp-07_run01	pose	gpu-main	video_only_3d
```

Suggested status values:

| Status | Meaning |
|--------|---------|
| `staged` | Raw data copied from archive to SSD |
| `bids` | BIDS packaging in progress |
| `features` | Feature-first video surrogate extraction in progress |
| `pose-json` | 2D pose JSON generation in progress |
| `pose-3d` | `video_only_3d_pipeline.py` in progress |
| `qc` | QC reports in progress |
| `archived` | Final outputs copied back to USB/archive |

This is intentionally simple. Do not use a shared queue that launches jobs on both machines until
the manual path is stable.

## Health Checks

Network throughput quick check using a disposable file:

```powershell
fsutil file createnew D:\affectai_work\net_test.bin 10737418240
Measure-Command { robocopy D:\affectai_work Z:\_net_test net_test.bin /R:0 /W:0 }
Remove-Item D:\affectai_work\net_test.bin
Remove-Item Z:\_net_test\net_test.bin
```

GPU visibility:

```powershell
nvidia-smi
```

Disk free space:

```powershell
Get-PSDrive -PSProvider FileSystem
```

Python environment:

```powershell
python -c "import sys; print(sys.version)"
python -m pytest tests\test_video_only_3d_pipeline.py
```

## Failure Modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| GPU job is slow but GPU use is low | Reading videos over 1 GbE or USB archive | Copy session to local SSD first |
| BIDS output is inconsistent | Two workers wrote the same session directory | Assign one session output to one machine |
| Network copy stalls | USB archive or 1 GbE bottleneck | Stage on internal SSD; reduce parallel copies |
| `pipeline_dry_run.json` reports missing pose folders | MediaPipe/OpenPose JSON was not generated for every camera | Re-run pose JSON extraction and check camera labels |
| `feature_extraction_dry_run.json` reports missing timing | Frame logs were not staged and task windows were not found | Copy `sourcedata/av/frame_logs` or run from a BIDS session with `annot/*_task_run_windows.tsv` |
| Calibration file missing | Session did not include exported calibration TOML | Use the auto-calibration fallback or locate the session calibration artifact |
| Share path fails | Dedicated NIC IP changed or Windows sharing permissions missing | Re-check `Get-NetIPAddress`, `ping`, and SMB share permissions |

## Safety Rules

- Never store real participant names in commands, logs, queue files, or documentation.
- Never modify raw vendor files in place; write derived outputs to a separate folder.
- Do not process directly from the 14 TB USB disk when a job reads many video frames.
- Do not attach the USB disk to both machines at the same time.
- Run dry-runs before long video or BIDS jobs.
- Use one output directory per session per worker.
