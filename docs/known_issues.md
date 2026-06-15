# Known issues / sharp edges

## Vendor SDKs
Some devices require proprietary SDKs/drivers. Keep adapters isolated and optional.

## Timebase mismatches
Different devices may expose different clocks. Always store:
- marker clock (LSL) when available
- device clock (native)
- explicit anchors (clap/LED) when feasible

### SSE stimulus onset uncertainty (display_server)
`server_sent_lsl` (captured at `wfile.write()`) records when bytes were handed to the kernel TCP buffer. Actual **tablet render onset** = `server_sent_lsl + network_latency + browser_parse + layout`. On a local Wi-Fi LAN this residual gap is typically **5–20 ms**. If sub-10 ms stimulus onset precision is required, add an explicit browser ack POST with a `requestAnimationFrame` timestamp to measure the true render time.

## DirectShow exclusive access blocks ffmpeg after lock_exposure

On Windows, `cv2.VideoCapture` opened via `CAP_DSHOW` (and to a lesser extent
`CAP_MSMF`) acquires an exclusive DirectShow filter graph that is reference-counted
by the Windows COM subsystem. Even after `cap.release()` + `del cap` +
`gc.collect()`, some Jabra camera models (pid_3029, pid_3020, pid_3021, pid_3013)
keep the handle alive long enough that a subsequent `ffmpeg -f dshow -i video=…`
attempt returns `Error opening input: I/O error`. Only pid_302a cameras (cam3,
cam4) released fast enough under DSHOW to allow ffmpeg to open them.

### Mitigation (current)
- `camera_setup_script` is **disabled** in `configs/ffmpeg_multicap.json` for all
  7 video devices. ffmpeg opens cameras directly without any OpenCV pre-open,
  eliminating the handle conflict. All 7 cameras start successfully.
- Exposure/WB lock can be applied **manually before a session** via Jabra Direct
  app (Option B in `docs/jabra_recording_checklist.md`) or as a fully separate
  pre-roll step: run `tools/lock_exposure.py`, wait >30 s, then start
  `ffmpeg_multicap.py`.

### Do NOT re-enable camera_setup_script
Running lock_exposure.py via `camera_setup_script` (i.e. inside the
`ffmpeg_multicap.py` launch sequence) will reproduce this failure for at least
5 of the 7 cameras regardless of cooldown duration.

## DirectShow lists some cameras as `(none)` instead of `(video)`

On Windows, `ffmpeg -f dshow -list_devices true -i dummy` can occasionally list
some Jabra camera interfaces as `(none)` (control-only) rather than `(video)`.
When this happens, those cameras are **present in PnP** but cannot be opened for
capture, and ffmpeg returns `Error opening input: I/O error` for those devices.

### Symptoms
- Config `video_alt_name` values match connected devices, but affected cameras
  still fail immediately with input I/O errors.
- Only cameras listed as `(video)` are capturable; `(none)` entries always fail.

### Mitigation
- Run `python tools/ffmpeg_multicap.py --list-devices` and verify all target
  cameras appear as `(video)` before recording.
- If any are `(none)`, replug affected cameras and/or power-cycle the USB hub,
  then re-check device listing.
- Keep `input_video_codec` on auto (`null`) unless a specific camera requires a
  forced format.

## Data collection gaps and quality issues

Systematic audit results are documented in [`docs/data_audit.md`](data_audit.md).
Key issues affecting processing:

### grp-16: missing CurrentStudy XDF
No CurrentStudy XDF was recorded for grp-16 (Mar 20). The `sub-P001` XDFs were
investigated and all belong to earlier dates (grp-09 / test sessions). Stimuli
events data is available (`20260320_grp-16_run01_20260320_095857`). A stub
directory (`20260320_grp-01_run01_20260320_151441`) exists from a false start
with the wrong session ID — it contains only `session_start`, no task data.
Task window derivation works from stimuli events; only the CurrentStudy XDF is
missing.

### grp-10: stimuli app overnight — inflated T0/T1 durations
The stimuli application for grp-10 ran for ~19 hours (started the evening
before the session). T0 and T1 task durations are inflated to ~19 h in the raw
events data. This is a single-file data quality issue, not a pipeline bug.
Downstream processing should use the Tobii LSL or experiment control markers
to derive correct task boundaries.

### grp-06: only 3 participants
grp-06 had only 3 participants (P3 absent). P3 Tobii video is missing because
the participant was not present. No CurrentStudy XDF or AV recordings exist.

### grp-14 P1 name discrepancy
Schedule says "Tatiana Crucerescu" for P1, but the recording metadata says
"Laura Gravila". Likely a late participant substitution.

### sub-P001 directory naming
`data/CurrentStudy/sub-P001/` uses a generic LabRecorder default name instead
of the expected `sub-grp-{N}` pattern. The 294 MB XDF inside belongs to grp-09.

## Identifiable data
Video/audio can be identifiable. Do not commit raw identifiable data to this repository.

## DPA microphone audio-clock drift

The RME Fireface 802 audio clock drifts at ~0.04 ms/s relative to the XDF/LSL
clock. Over a 64-minute recording this introduces 50–100 ms error if a single
median anchor is used for all tasks. `xdf_sync_pipeline.py` mitigates this by
fitting a **linear regression** `anchor(t) = slope·t + intercept` over the full
recording for each mic, then evaluating it at each task's XDF start time.

## DPA audio sync when AV XDF is corrupted

For some sessions (e.g. grp-15) the AV-side XDF is heavily corrupted and
pyxdf recovers 0 streams. In this case the pipeline falls back to the raw JSONL
files under `<capture>/lsl/`, using the `received_time` field (LSL recorder
wall-clock) converted to unified XDF time:

```python
xdf_t = datetime.fromisoformat(received_time).timestamp() - wall_minus_xdf_lsl
```

`received_time` has ~500 ms network jitter, so regression RMSE is ~0.5 s, but
the fitted slope accurately captures the drift rate, and sync error is < 5 ms
in practice. See `_compute_dpa_anchors_from_jsonl()` in `xdf_sync_pipeline.py`.

## ffmpeg `-ss before -i` seek inaccuracy

Using `-ss` before `-i` with `-c copy` snaps to the nearest decodable packet
boundary (~35 ms error for 48 kHz audio). All DPA clips are now produced with
`-ss` **after** `-i` and re-encoded to `pcm_s16le`, giving sample-accurate
seeking at no quality cost (PCM → PCM is lossless).

## Non-monotonic lens distortion (`cv2.undistortPoints` failure)
When a camera has strong barrel distortion (k1 << 0), the radial distortion
function `r_d = r_u * (1 + k1 * r_u²)` becomes non-monotonic beyond
`r_crit = 1 / sqrt(-3 * k1)`. At those radii `cv2.undistortPoints()` **fails
silently** — it returns incorrect/diverged coordinates without any error.
Image-level undistortion (`cv2.remap`) also produces folding artefacts in the
non-monotonic region.

### Affected cameras (ses-20260202_test calibration)
| Camera | k1      | r_crit | r_max | Status          |
|--------|---------|--------|-------|-----------------|
| cam_0  | −1.186  | 0.530  | 1.670 | NON-MONOTONIC   |
| cam_4  | −0.569  | 0.766  | 1.823 | NON-MONOTONIC at edges |
| cam_1  | +0.134  | —      | —     | OK              |
| cam_2  | +0.592  | —      | —     | OK              |
| cam_3  | −0.013  | ~5.2   | 1.15  | OK (margin)     |

### Mitigation (implemented)
- **Monotonicity filter** (`_distortion_is_monotonic`): Per-observation check
  `1 + 3·k1·r² > 0`. Cameras failing this are excluded from triangulation.
- **Nonlinear triangulation**: DLT initial estimate + Levenberg-Marquardt
  refinement using the **forward** distortion model (`_project_distorted`),
  which is always well-defined regardless of monotonicity.
- Both `multicam_pose3d.py` and `face_hand_pipeline.py` implement this.
  `triangulate_openpose.py` has the monotonicity filter + undistortion.
- `calibrate_charuco.py validate` now prints distortion warnings.

### Root cause
Likely miscalibrated — cam_0's k1 = −1.186 is extremely large for a 1920×1080
sensor. Re-calibration with better charuco coverage should reduce |k1|.
