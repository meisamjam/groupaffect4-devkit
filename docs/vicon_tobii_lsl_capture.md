# Vicon + Tobii to LSL capture section

This section uses the **Vicon DataStream SDK** and the **Tobii Glasses 2/3 SDK** directly (no UDP JSON bridge).

## 1) Vicon DataStream -> LSL bridge

Script: `tools/vicon_nexus_lsl_bridge.py`

### What it does
- Connects to a Vicon DataStream server over the SDK.
- Captures segment data (and optional device + eye tracker data).
- Records every frame to NDJSON.
- Publishes to LSL:
  - `ViconDataStreamClock` (always)
  - `ViconDataStreamFrame` (JSON payload per frame)
  - `ViconSegmentTranslation` (x,y,z) when `--structured-lsl` is enabled
  - `TobiiEyeTracker` (pos xyz + gaze xyz) when `--tobii-lsl` is enabled
  - Per-stream numeric outlets with Nexus names when `--per-stream-lsl` is enabled

### SDK install note
This repo includes the Vicon DataStream SDK under `tools/DataStream SDK/`.
By default, the bridge uses the bundled **Python** SDK (no pythonnet required).
If you installed the SDK elsewhere, pass `--sdk-python-path` to the SDK Python root.
If you want the .NET SDK, pass `--sdk-backend dotnet` and set `--sdk-dll`.

### Commands

```bash
# Full forwarding to LSL + local raw recording (default: Python SDK)
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801

# Heavy-stream fallback: record locally, publish only clock heartbeat to LSL
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --clock-only

# Force .NET SDK backend (requires pythonnet)
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --sdk-backend dotnet \
  --sdk-dll "tools\DataStream SDK\Win64\dotNET\ViconDataStreamSDK_DotNET.dll"

# Multicast mode (use when Nexus is configured for multicast)
python tools/vicon_nexus_lsl_bridge.py --multicast --local-ip 10.0.0.5 \
  --multicast-ip 224.0.0.1:44801

# Live plot of a segment translation (defaults to first subject/segment)
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --plot

# Live plot of a specific subject/segment
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --plot --plot-subject Subject01 --plot-segment Head

# Structured numeric LSL (x,y,z) for one segment
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --structured-lsl \
  --structured-subject Subject01 --structured-segment Head

# Tobii numeric LSL from Nexus (pos xyz + gaze xyz)
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --tobii-lsl

# Tobii numeric LSL for a specific eye tracker index
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --tobii-lsl --tobii-index 0

# Per-stream LSL streams for all subjects/segments/devices/eyes
python tools/vicon_nexus_lsl_bridge.py --server 10.0.0.2:801 --per-stream-lsl
```

## 2) 4x Tobii Glasses -> LSL bridge

Script: `tools/tobii_glasses_lsl_bridge.py`

Config: `configs/tobii_glasses_streams.yaml`

### What it does
- Uses the Tobii Glasses 2/3 SDK to connect to devices.
- Supports explicit IP/serial and discovery modes.
- Publishes one regular sampled stream per glasses:
  - `Tobii_P1_stream` ... `Tobii_P4_stream` (type `EyeTracking`)
  - Default channels: gaze + pupil (`gaze_x`, `gaze_y`, `pupil_left`, `pupil_right`, `gaze_valid`)
  - `--with-3d` appends 3D gaze/origin/direction channels
  - `--with-imu` appends IMU accel/gyro/mag channels
  - `--nominal-srate` controls nominal LSL rate metadata for `Tobii_P*_stream` (default `50.0`)
- Publishes one irregular global Tobii event stream when enabled:
  - `evetns_tobii` (type `Event`) for irregular packets from `--with-events` and/or `--with-sync-port`
  - Payload columns: `device_id,event_type,timestamp_ticks,key_a,value_a,key_b,value_b`
- Legacy mode:
  - `--split-streams` restores separate outlets (`Tobii_<id>_Event`, `Tobii_<id>_Imu`, `Tobii_<id>_SyncPort`)
- Records raw samples to `data/streams/tobii_glasses/<id>.ndjson`.

### SDK install note
Build or install the Tobii Glasses SDK and point `--sdk-dll` to the compiled `G3SDK.dll`.

### Command

```bash
python tools/tobii_glasses_lsl_bridge.py --config configs/tobii_glasses_streams.yaml --sdk-dll "C:\Program Files\Tobii\Glasses3 SDK\G3SDK.dll"

# Include optional regular + irregular channels
python tools/tobii_glasses_lsl_bridge.py --config configs/tobii_glasses_streams.yaml --sdk-dll "C:\Program Files\Tobii\Glasses3 SDK\G3SDK.dll" \
  --with-3d --with-events --with-imu --with-sync-port

# Override nominal LSL sample-rate metadata for Tobii_P*_stream
python tools/tobii_glasses_lsl_bridge.py --config configs/tobii_glasses_streams.yaml --sdk-dll "C:\Program Files\Tobii\Glasses3 SDK\G3SDK.dll" \
  --with-3d --with-imu --nominal-srate 100

# Legacy split streams (separate Event/IMU/SyncPort outlets)
python tools/tobii_glasses_lsl_bridge.py --config configs/tobii_glasses_streams.yaml --sdk-dll "C:\Program Files\Tobii\Glasses3 SDK\G3SDK.dll" \
  --with-3d --with-events --with-imu --with-sync-port --split-streams
```

## 3) Offline multi-glasses world alignment (marker-based)

Script: `tools/tobii_multi_glasses_world_align.py`

Config template: `configs/tobii_offline_world_align.example.yaml`

### What it does
- Reads recorded scene video + recorded Tobii gaze NDJSON for each glasses unit.
- Detects ArUco markers in each scene frame and estimates per-frame camera pose with `solvePnP`.
- Uses the solved pose to project gaze into a shared world frame (board plane `z=0`).
- Writes derived outputs only:
  - `<output>/<device>_frame_pose.ndjson` (pose per frame)
  - `<output>/<device>_gaze_world.ndjson` (world gaze with debug payload)
  - `<output>/<device>_gaze_world.csv` (tabular world gaze)
  - `<output>/summary.json` (run-level alignment summary)

### Command

```bash
python tools/tobii_multi_glasses_world_align.py \
  --config configs/tobii_offline_world_align.example.yaml \
  --output-dir data/derived/tobii_world

# Prefer Tobii 3D eye vectors/origins when available
python tools/tobii_multi_glasses_world_align.py \
  --config configs/tobii_offline_world_align.example.yaml \
  --output-dir data/derived/tobii_world \
  --ray-source gaze3d
```

Notes:
- Define `world.marker_map` with fixed marker corner coordinates in meters.
- Keep at least one mapped marker visible in each glasses scene camera whenever possible.
- If your Tobii `timestamp_ticks` use a non-.NET timebase, set `--ticks-per-second` accordingly.
- `--ray-source gaze3d` uses `left/right_eye.gaze_origin` + `gaze_direction` (or `gaze3d` fallback) from NDJSON. Ensure the bridge was run with `--with-3d` during recording.

### QC plots (4-user coordinated gaze)

Script: `tools/qc/qc_tobii_world_gaze.py`

```bash
python tools/qc/qc_tobii_world_gaze.py \
  --input-dir data/derived/tobii_world

# Optional: overlay marker polygons + board bounds from alignment config
python tools/qc/qc_tobii_world_gaze.py \
  --input-dir data/derived/tobii_world \
  --align-config configs/tobii_offline_world_align.example.yaml
```

Outputs under `<input-dir>/qc/`:
- `tobii_world_gaze_summary.json`
- `tobii_world_gaze_summary.csv`
- `tobii_world_gaze_scatter.png`
- `tobii_world_gaze_timeseries.png`

## Dependencies
- `pylsl` for LSL publishing
- `pythonnet` for SDK interop

## Nexus streaming notes
- DataStream must be enabled in Nexus and set to Unicast on port 801 (or valid multicast group in 224-239 range).
- Tobii data only appears if the Tobii device is integrated and actively streaming in Nexus.
- If `Eye Tracker` values are all zeros, Nexus is not publishing live gaze data yet (device not streaming/calibrated).
