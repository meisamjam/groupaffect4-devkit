# LSL Architecture Alignment Analysis

## Current Architecture vs. Target Architecture

### Target Architecture (Your Description)

```
┌─────────────────────────────────────────────────────────────┐
│                    Central Recording PC                       │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           LSL Recorder (receives all streams)         │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ▲                                   │
│                           │ All LSL streams                   │
│  ┌────────────────────────┴────────────────────────────┐  │
│  │        LSL Network (timestamp synchronization)        │  │
│  └────────────────────────┬────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│   Device 1   │    │   Device 2    │    │    Device 3      │
│              │    │               │    │                  │
│ Tobii + Vicon│    │  EmotiBit ×4  │    │ GStreamer AV     │
│  (via Nexus) │    │               │    │ (markers only)   │
│              │    │               │    │                  │
│  → LSL data  │    │  → LSL data   │    │  → LSL markers   │
└──────────────┘    └───────────────┘    └──────────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │  GStreamer Pipeline     │
                                    │  - Jabra P50 video bar  │
                                    │  - 4× ceiling cameras   │
                                    │  - 4× DPA microphones   │
                                    │                         │
                                    │  Actual data stored     │
                                    │  on disk separately     │
                                    └─────────────────────────┘

        ┌────────────────┐
        │   Device 4     │
        │                │
        │ Tablet Stimuli │
        │                │
        │ → LSL markers  │
        │   (flags +     │
        │    responses)  │
        └────────────────┘
```

### Current Implementation Status

| Component | Implemented | LSL Integration | Notes |
|-----------|------------|-----------------|-------|
| **LSL Marker Stream** | ✅ Yes | ✅ Core | `devices/lsl_markers.py` - works |
| **Stimulus Markers** | ✅ Yes | ✅ Integrated | `stimulus.py` - sends to LSL + events.tsv |
| **Tablet Server** | ✅ Yes | ✅ **DONE** | `stimuli/display_server.py` — 7 LSL streams: `AffectAI_Participant_1–4`, `_Moderator`, `_BigScreen`, `_Experiment` |
| **EmotiBit Integration** | ✅ Yes | ✅ **DONE** (2026-01-12) | LSL streams + JSONL per device+channel; merged `Emotibit_P#_stream` via `tools/emotibit_lsl_merger.py` |
| **Tobii/Vicon (Nexus)** | ✅ Yes | ✅ **DONE** | `tools/tobii_glasses_lsl_bridge.py` (4× Tobii → LSL) + `tools/vicon_nexus_lsl_bridge.py` (Vicon DataStream → LSL) |
| **AV frame sync (ffmpeg)** | ✅ Yes | ✅ **DONE** | `tools/ffmpeg_multicap.py` + `tools/dpa_recorder.py` — LSL clock/frame-log streams; GStreamer replaced by ffmpeg+DirectShow |
| **LSL Recorder** | ✅ Yes | ✅ **DONE** | `tools/lsl_xdf_recorder.py` (pure-Python, single-writer-thread XDF) + LabRecorder.exe fallback |
| **BIDS Post-Processor** | ✅ Yes | ✅ **DONE** | `tools/raw_to_bids.py` (AV/Tobii/LSL → BIDS; optional XDF extraction via `pyxdf`) |

---

## Gap Analysis

### Gap 1: EmotiBit LSL Streaming ✅ RESOLVED

**Status:** Implemented 2026-01-12

**Solution Deployed:**
- LSL StreamOutlet created per device+channel (e.g., EmotiBit_10_49_228_101_ppg_green)
- Real-time streaming: PPG, EDA, temperature, IMU streams
- Stream types: PPG, EDA, Temperature, IMU, Physio with proper metadata
- Timestamps: `pylsl.local_clock()` for synchronization across modalities
- Multi-device: Separate outlets per source IP (supports 4× devices)
- Backward compatible: JSONL files still written for archival/debugging
- CLI: `--lsl` enabled by default; `--no-lsl` to disable

**See:**
- Code: `src/affectai_capture/devices/emotibit.py` (method `_get_lsl_outlet` at line 380)
- Demo: `examples/emotibit_lsl_demo.md`
- Verification: `python tools/lsl_stream_viewer.py`
- Change history: `CHANGES.md` (2026-01-12 entry)

---

### Gap 2: Tablet Server Not Sending LSL Markers ✅ RESOLVED

**Resolution (2026):** `stimuli/display_server.py` now publishes 7 LSL outlets —
`AffectAI_Participant_1–4`, `AffectAI_Moderator`, `AffectAI_BigScreen`, `AffectAI_Experiment`.
All stimulus pushes, VAD probes, task events, and participant responses are dual-written to
both the local `events.tsv` (timeline spine) and the corresponding LSL streams.  JSONL
backups are retained.

**Reference:** `stimuli/display_server.py`, `stimuli/event_logger.py`, `docs/llm/context_snapshot.md`

---

### Gap 3: No Tobii/Vicon (Nexus) LSL Integration ✅ RESOLVED

**Resolution (2026):**
- `tools/tobii_glasses_lsl_bridge.py` — 4× Tobii Pro Glasses 3 → LSL (gaze, pupil, 3D gaze, IMU, events). One unified `Tobii_P#_stream` per participant at nominal 50 Hz; irregular packets on `evetns_tobii`.
- `tools/vicon_nexus_lsl_bridge.py` — Vicon DataStream SDK → LSL (segments, devices, eye-trackers).

---

### Gap 4: No AV Timing Markers in LSL ✅ RESOLVED

**Resolution (2026):** GStreamer was not adopted; AV capture uses **ffmpeg + DirectShow** instead.
- `tools/ffmpeg_multicap.py` — 7 cameras (PanaCast 20/50) + LSL clock stream (`ffmpeg_clock@100 Hz`) + per-camera frame-log JSONLs (`ffmpeg_progress_<label>.jsonl`) that carry wall-clock timestamps for post-hoc sync.
- `tools/dpa_recorder.py` — 5× DPA microphones; also publishes timing anchors to LSL.
- 4-tier synchronisation: frame logs → LSL → progress TSV → events JSONL (see `docs/SYNC_BEST_PRACTICES.md`).

---

### Gap 5: No Centralized LSL Recorder ✅ RESOLVED

**Resolution (2026):**
- `tools/lsl_xdf_recorder.py` — pure-Python LSL→XDF 1.0 recorder (no LabRecorder.exe required). Single-writer-thread architecture: N puller threads push encoded chunks to per-stream `deque`s; one writer thread drains all queues every 100 ms. Key CLI options: `--output`, `--resolve-timeout`, `--prefixes`, `--boundary-interval` (default 30 s), `--late-discovery-interval` (default 5 s). Supports pylsl 1.18+, micro-batching, session-stable stream identity, start/stop manifests.
- **LabRecorder.exe** remains supported as a fallback via `tools/LabRecorder/LabRecorderCLI.exe`.
- Both write to `<session_dir>/sourcedata/lsl/<session_id>.xdf`.

---

### Gap 6: LSL→BIDS Post-Processing Pipeline ✅ RESOLVED

**Resolution (2026):** `tools/raw_to_bids.py` converts AV/Recording/Tobii raw sources to BIDS-oriented modality outputs with optional XDF stream extraction via `pyxdf`. The BIDS layout (`sub-{id}/ses-{id}/` with `eeg/`, `et/`, `physio/`, `audio/`, `video/`, `mocap/`, `beh/`, `annot/`) is populated from the raw session tree. The authoritative `events.tsv` (timeline spine) is written per-session. See `docs/raw_data_upload_and_bids_conversion.md`.

---

## Implementation Status — All Gaps Closed ✅

> All six original gaps are resolved as of 2026. See the Implementation Status table above and the gap sections for current references.
> For the live operational picture see `docs/llm/context_snapshot.md`.

---

## Implementation Status (Live Tracking)

| Gap | Status | Completion | Reference |
|-----|--------|------------|-----------|
| Gap 1: EmotiBit LSL streaming | ✅ DONE | 2026-01-12 | `CHANGES.md`, `examples/emotibit_lsl_demo.md`; merged streams via `tools/emotibit_lsl_merger.py` |
| Gap 2: Tablet/stimulus LSL markers | ✅ DONE | 2026 | `stimuli/display_server.py` (7 LSL outlets), `stimuli/event_logger.py` |
| Gap 3: Tobii + Vicon LSL | ✅ DONE | 2026 | `tools/tobii_glasses_lsl_bridge.py`, `tools/vicon_nexus_lsl_bridge.py` |
| Gap 4: AV timing markers in LSL | ✅ DONE | 2026 | `tools/ffmpeg_multicap.py` (LSL clock + frame logs), `tools/dpa_recorder.py` |
| Gap 5: Central LSL recorder | ✅ DONE | 2026 | `tools/lsl_xdf_recorder.py` (pure Python) + LabRecorder.exe fallback |
| Gap 6: XDF→BIDS converter | ✅ DONE | 2026 | `tools/raw_to_bids.py` (+ `pyxdf`) |

---

### Phase 2: Device Adapters (Week 2-3)
4. **Nexus LSL adapter** (MEDIUM) - requires Vicon SDK access
5. **GStreamer marker injection** (MEDIUM) - requires GStreamer pipeline setup

---

> **All implementation phases are complete.** For current operational details see `docs/llm/context_snapshot.md`.
> The original code sketches below are preserved for historical reference only — the actual implementations differ.

## Historical Code Sketches (reference only)

### 1. `src/affectai_capture/devices/emotibit.py`

Add LSL streaming:
```python
from pylsl import StreamInfo, StreamOutlet, local_clock

class EmotiBitStreamer:
    def __init__(self, ..., enable_lsl=True):
        ...
        self.lsl_outlets = {}  # channel_tag -> StreamOutlet
        self.enable_lsl = enable_lsl
        
    def _get_lsl_outlet(self, packet: EmotiBitPacket):
        key = f"{packet.device_id}:{packet.channel_tag}"
        if key in self.lsl_outlets:
            return self.lsl_outlets[key]
        
        # Create LSL stream per channel per device
        stream_name = f"EmotiBit_{packet.device_id}_{packet.channel_tag}"
        channel_name = CHANNEL_NAMES.get(packet.channel_tag, packet.channel_tag)
        
        info = StreamInfo(
            name=stream_name,
            type='Physio',
            channel_count=1,
            nominal_srate=25.0,  # Approximate
            channel_format='float32',
            source_id=f"emotibit_{packet.device_id}_{packet.channel_tag}"
        )
        outlet = StreamOutlet(info)
        self.lsl_outlets[key] = outlet
        return outlet
    
    def _write_packet(self, packet: EmotiBitPacket):
        ...
        # Send to LSL
        if self.enable_lsl and packet.samples:
            outlet = self._get_lsl_outlet(packet)
            lsl_ts = local_clock()
            for sample in packet.samples:
                if isinstance(sample, (int, float)):
                    outlet.push_sample([float(sample)], lsl_ts)
```

### 2. `stimuli/tablet_server.py`

Add LSL marker outlet:
```python
from affectai_capture.devices.lsl_markers import LSLMarkerOutlet

class TabletHandler(BaseHTTPRequestHandler):
    lsl_outlet = None  # Class variable
    
    def _handle_push(self):
        payload = self._read_json_body()
        ...
        # Send LSL marker
        if self.lsl_outlet:
            marker_data = {
                "event": "prompt_displayed",
                "prompt_name": payload.get("prompt_name"),
                "timestamp": time.time()
            }
            self.lsl_outlet.push(json.dumps(marker_data))
        ...
    
    def _handle_response(self):
        payload = self._read_json_body()
        ...
        # Send LSL marker
        if self.lsl_outlet:
            marker_data = {
                "event": "prompt_response",
                "prompt_name": payload.get("prompt_name"),
                "response": payload.get("response"),
                "timestamp": time.time()
            }
            self.lsl_outlet.push(json.dumps(marker_data))
        ...

# In main():
if __name__ == "__main__":
    try:
        TabletHandler.lsl_outlet = LSLMarkerOutlet()
        print("✅ LSL marker outlet initialized")
    except:
        print("⚠️  LSL not available, markers will not be sent")
        TabletHandler.lsl_outlet = None
```

### 3. New file: `src/affectai_capture/devices/nexus_lsl.py`

```python
"""Vicon Nexus LSL adapter for synchronization markers.

Sends timing markers from Vicon Nexus to LSL for cross-modal sync.
Does NOT stream full mocap data (stays in .c3d files).
"""

# TODO: Implement using Vicon SDK or Nexus LSL plugin
# Requires access to Vicon system
```

### 4. New file: `tools/gstreamer_lsl_injector.py`

```python
"""GStreamer callback to inject timing markers into LSL.

Monitors GStreamer pipeline and sends periodic markers for A/V sync.
"""

# TODO: Implement GStreamer appsink callback
# Requires GStreamer Python bindings (gi.repository.Gst)
```

### 5. New file: `tools/xdf_to_bids.py`

```python
"""Convert LSL .xdf recordings to BIDS-compliant structure.

Extracts streams from XDF and writes to appropriate BIDS modalities.
"""

import pyxdf
from pathlib import Path

def convert_xdf_to_bids(xdf_file: Path, bids_root: Path):
    data, header = pyxdf.load_xdf(str(xdf_file))
    
    # Extract marker stream → events.tsv
    # Extract EmotiBit streams → physio/*.jsonl
    # Extract tablet markers → beh/responses.tsv
    # etc.
    
    pass
```

---

## Testing Strategy

1. **Test EmotiBit LSL streaming:**
   - Run mock_emotibit with LSL enabled
   - Use LabRecorder to verify streams appear
   - Check XDF file contains EmotiBit data

2. **Test tablet LSL markers:**
   - Push prompt to tablet
   - Verify LSL marker appears in LabRecorder
   - Submit response, verify response marker

3. **Integration test:**
   - Run all devices simultaneously
   - Record with LabRecorder
   - Verify all streams synchronized in XDF

4. **BIDS conversion test:**
   - Convert XDF to BIDS
   - Verify events.tsv alignment
   - Check all modalities present

---

## Current Architecture Reference

All six gaps are closed. For the live system picture see:
- `docs/llm/context_snapshot.md` — canonical current-state reference
- `docs/architecture.md` — system overview
- `docs/data_flow.md` — end-to-end data flow
- `docs/SYNC_BEST_PRACTICES.md` — synchronisation details
- `tools/lsl_xdf_recorder.py` — XDF recorder CLI (`--help` for all options)
