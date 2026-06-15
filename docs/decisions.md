# Decisions (ADRs)

Add new decisions as dated entries. Keep this file short.

## ADR-0001 — BIDS-first session spine
**Status:** Accepted  
**Date:** 2025-10-01  
**Decision:** Organize outputs under `/sub-{id}/ses-{id}/` with one authoritative `events.tsv` per session.  
**Rationale:** Makes cross-modality alignment explicit and durable.

## ADR-0002 — Deterministic repo maintenance scripts
**Status:** Accepted  
**Date:** 2025-12-31  
**Decision:** Use deterministic scripts (not LLMs) to keep sources indexes and progress reports consistent (`tools/`).  
**Rationale:** Avoid hallucinated repo state; keep guardrails simple.

## ADR-0003 — LSL for markers; device adapters are opt-in
**Status:** Draft  
**Date:** 2025-12-31  
**Decision:** Provide an LSL marker writer in-core; device integrations (Tobii, g.tec, Vicon) are adapters that may require vendor SDKs.  
**Rationale:** Keep core tooling runnable without proprietary dependencies.

## ADR-0004 — Jabra PanaCast 20 records at 1080p via center-crop, not downscale
**Status:** Accepted  
**Date:** 2026-04-10  
**Decision:** The authoritative `expected_fx_1080p` for Jabra PanaCast 20 in `configs/camera_specs.json` is **950 px** (empirical median with AI Zoom off), not the spec-formula value of 554 px.  
**Context:** The camera sensor is 4K (3840×2160). In 1080p mode the firmware performs a **center-crop** of the full-FoV image, not a 2× downscale. Center-crop preserves the focal length in pixels (fx_4K ≈ 1108 px → fx_1080p ≈ 950 px empirically). Multiple calibration runs confirm the range 865–1120 px with AI Zoom off. The spec-formula (fx = (960/tan(60°)) = 554 px) assumes full-FoV read-out at 1080p, which is wrong.  
**Rationale:** Using 554 px as the seeded intrinsic causes anipose calibration to diverge. Correct seed → 0.24 px reprojection error vs 1.67 px with wrong seed.  
**Consequences:** `configs/camera_specs.json` field `expected_fx_1080p` = 950.0 for `jabra_panacast_20`. The `expected_fx_1080p_nominal_range` is [700, 1400]. Do not revert to the spec formula.

## ADR-0005 — Per-camera overrides in camera_specs.json for non-default firmware states
**Status:** Accepted  
**Date:** 2026-04-10  
**Decision:** `configs/camera_specs.json` has a top-level `camera_overrides` block that lets individual cameras override model-level expected intrinsics when their firmware settings differ from the standard configuration. Both `calibrate_charuco.py` (`_match_camera_model`) and `validate_calibration_robust.py` (`match_camera_model`) merge these overrides on top of model defaults.  
**Context:** cam5 was calibrated with Jabra Intelligent Zoom (AI framing) active, giving fx=1273 px and k1=+1.84 (pincushion distortion — digital zoom crop). cam6 appears to record at full 120° FoV (fx≈590 px) rather than center-crop mode, possibly due to a different hardware/firmware revision or manual locked zoom setting. The base P20 model tolerances would flag both as "bad".  
**Consequences:** Always lock AI Zoom off via Jabra Direct before running calibration (cam5 will then return to fx~950). The override for cam5 records its AI-Zoom-active state; update the override's `expected_fx_1080p` if you re-calibrate with zoom off. cam6 override should be revisited once a dedicated calibration pass captures board frames in its FoV.

## ADR-0006 — Sequential read fallback for MKV seek failures in calibrate_charuco.py detect command
**Status:** Accepted  
**Date:** 2026-04-10  
**Decision:** `calibrate_charuco.py detect` and the board-type auto-selection sampler (`_score_video_file`) both fall back to sequential frame reading when the seek-based pass yields zero ChArUco detections.  
**Context:** MKV files use variable GOP structure. `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, N)` silently returns incorrect frames on many MKV files, producing 0 detections even when the board is clearly visible. The fix: after a seek-based pass with 0 detections, reset to frame 0 and re-read sequentially at the same sample density. This adds negligible overhead (only fires when seek genuinely failed) and eliminates false-negative "camera sees no board" diagnostics.  
**Consequences:** Detection results are now reliable for both MKV and MP4/AVI inputs. The sequential fallback emits an INFO log when it activates so operators can confirm when seek was unreliable.

## ADR-0007 — Glasses camera localization: three-tier approach
**Status:** Accepted (A + B implemented; C planned)  
**Date:** 2026-04-10  
**Decision:** The Tobii Glasses 3 scene cameras are localized per time-frame using a three-tier approach. Approach A (ArUco markers on glasses, triangulated by the fixed camera rig) is the primary source. Approach B (PnP from desk/board markers in the Tobii scene video) is the fallback. Approach C (fused Kalman-filter blend of A and B) is planned.  
**Context:** The glasses cameras move with each participant's head and cannot be represented by a static entry in the calibration TOML. A static TOML-based pose would be inaccurate by orders of magnitude for gaze-to-world projection. The fixed cameras already cover the workspace continuously; attaching small ArUco markers to each glasses frame provides sub-millimetre pose estimates whenever at least two fixed cameras see the same marker. Scene-video PnP (Approach B) provides ~5–10 mm accuracy from the glasses' own forward-facing camera seeing known desk markers, and works independently of whether the fixed cameras can see the glasses.  
**Tools:** `tools/tobii_multicam_glasses_tracker.py` (A), `tools/tobii_multi_glasses_world_align.py` (B).  
**Consequences:** Both tools require the same calibration TOML as the body-pose pipeline, ensuring all 3D outputs share the same world frame. Approach C implementation should use per-frame confidence from corner count (A) and PnP residual (B) as Kalman observation weights.
