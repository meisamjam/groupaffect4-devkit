# Changes

## 2026-06-15 - dataset-to-model end-to-end onboarding docs

- Added a new top-level `README.md` with a practical quick-start path from
  dataset download to feature extraction, preprocessing, and model training.
- Added `docs/END_TO_END_DATASET_TO_MODELS.md` as the canonical linear
  walkthrough for users reproducing analyses from a released GroupAffect-4
  archive.
- Updated `ARCHITECTURE.md` navigation to include the new end-to-end guide.
- Updated `docs/llm/context_snapshot.md` doc-routing so future sessions can
  quickly locate the onboarding workflow.

## 2026-06-09 - CSMI paper recommendation merge

- Fast-forwarded to the remote CSMI paper update and preserved its collective
  affect framing while re-applying the remaining literature-review
  recommendations.
- Clarified that the ordinal SAM path is central to the target-construction
  thesis, added a non-overclaim caveat for BFI conditioning, and tied IMU /
  Dominance interpretations to body-movement and social-signal literature.

## 2026-06-08 - ICMI literature review and citation expansion

- Added `paper/ICMI/literature_review.md` with a claim-support audit,
  comprehensive literature review draft, and paper-level recommendations.
- Expanded `paper/ICMI/icmi2026_short.tex` Related Work to connect the
  current results story to VAD/SAM target construction, physiological and
  oculomotor affect recognition, social/group corpora, semi-supervised
  augmentation, validation methodology, and personality moderation.
- Added missing bibliography entries for MAHNOB-HCI, ASCERTAIN, AMIGOS,
  K-EmoCon, SALSA/AMI, physiological-signal reviews, ordinal label learning,
  FixMatch, label smoothing, time-series augmentation, and validation-method
  literature.

## 2026-06-04 - ICMI paper story consistency pass

- Revised `paper/ICMI/icmi2026_short.tex` to distinguish the primary
  known-team temporal holdout from LOSO cross-subject transfer and LOGO
  unseen-group transfer (`0.227 +/- 0.037`), reducing ambiguity in the
  generalisation story.
- Tightened the three-class narrative: balanced labels make the comparison
  valid, aligned GP augmentation helps task-CV (`0.309 +/- 0.105` vs.
  `0.287 +/- 0.080`) but reduces canonical T3 macro-F1, and BFI conditioning
  remains task-context-sensitive rather than a canonical gain.
- Corrected the ordinal refinement story so the T3 task-CV MAE `1.325`
  result is attributed to `sigma=1 + FixMatch, no GP`, not to pure label
  smoothing alone; updated `ORDINAL_RESULTS.md` with the June 4 refinement
  files and interpretation.

## 2026-06-02 - ICMI ordinal original-label affect modelling

- Added a standalone ordinal VAD path in `tools/mumt/model_ordinal.py` and
  `tools/mumt/train_ordinal.py`, leaving the original 3-class binned
  `train_simple.py` / `model_simple.py` pipeline untouched.
- Added `tools/mumt/ORDINAL_CLASSIFICATION.md` documenting the ordinal process,
  commands, GP handling, and result interpretation.
- Ran canonical T3 ordinal training on `data/mumt/dataset_15s.pkl`:
  `results/ordinal_t3.csv` reports MAE 1.474, Spearman 0.324, QWK 0.205,
  and 1-9 macro-F1 0.051.
- Ran rotating task-CV ordinal training:
  `results/ordinal_taskcv.csv` reports mean MAE 1.505 +/- 0.103,
  Spearman 0.215 +/- 0.169, QWK 0.150 +/- 0.113, and 1-9 macro-F1
  0.071 +/- 0.003.
- Verified the GP ordinal process with a smoke run
  (`results/ordinal_gp_smoke.csv`): GP targets are derived from stored
  `(mu, sigma)` cumulative probabilities and stored 3-class GP labels are
  ignored. For task splits, GP rows are filtered to training tasks.
- Updated `paper/ICMI/icmi2026_short.tex` with the completed ordinal method and
  results, and kept audio/speech removed from the paper and bibliography.
- Updated `docs/llm/context_snapshot.md` with the current MuMT/ICMI modelling
  state so future assistant sessions can find the ordinal path and results.

## 2026-04-28 - Physio extraction and paper analysis workflow

- Added a physio dataset-paper analysis layer in `tools/features/analyze_physio_paper.py`,
  producing feature usability, task-delta effect sizes, session/task summaries, QC flag counts,
  cross-feature correlations, temporal profiles, and paper-review PNG figures.
- Added `tools/features/analyze_autonomic_paper.py` to combine EmotiBit features with Tobii
  pupil size features for dataset-paper task fingerprints, modality coverage, composite physio
  response summaries, pupil-physio links, and temporal-profile figures.
- Extended EmotiBit feature extraction with deeper PPG/HRV, EDA, temperature, and IMU summaries
  plus QC flags for missingness, coverage, HRV reliability, PPG/HR agreement, and motion.
- Added physio visualization tooling and focused regression tests for extraction and paper-analysis
  summaries; documented the new commands in `tools/features/README.md`.

## 2026-04-27 - feature-first processing workflow docs

- Updated `docs/data_flow.md`, `docs/architecture.md`, `docs/video_only_3d_pipeline.md`, and
  `docs/offline_compute_stack.md` so Pipeline 2 now starts with
  `tools/extract_video_features.py --dry-run` followed by feature extraction before calibration,
  3D gaze, skeleton, and gesture work.
- Documented the current bridge state clearly: `video_only_3d_pipeline.py` still consumes
  OpenPose-compatible `--pose-root` JSON, while `features_video/feature_manifest.json` and the
  compact `.npz`/JSONL files are the intended feature-native input contract for downstream
  adapters.
- Updated offline stack scheduling/status language to include a `features` stage and RTMPose/RTMW
  extraction examples on the GPU worker.

## 2026-04-24 - video feature extraction follow-on (dry-run preflight + runbook)

- **Extended `tools/extract_video_features.py` with `--dry-run`:** adds a non-decoding preflight mode that validates selected videos, camera labels, output target, and frame-log matching before heavy inference starts. Dry-run writes `feature_extraction_dry_run.json` to the output directory.
- **Added helper summary builder:** `_build_dry_run_summary()` records per-camera source path, existence, file size, inferred camera ID, and matched frame-log path.
- **Added tests in `tests/test_extract_video_features.py`:**
  - dry-run summary includes camera listing and frame-log mapping
  - parser recognizes `--dry-run`
- **Added `docs/video_feature_extraction.md`:** documented step-by-step workflow for preflight, MediaPipe extraction, optional RTMPose/RTMW body extraction, and throughput tuning flags.
- **Updated `docs/llm/context_snapshot.md`:** video feature extraction doc-routing entry now points to the new runbook and notes `--dry-run` support.
- **Split-task timestamp support in `tools/extract_video_features.py`:** when processing BIDS task clips from `session/video/`, the extractor now auto-reads `annot/*_task_run_windows.tsv` and carries usable session-level `unix_time_s` / `lsl_time` into `frame_sync.jsonl`, dry-run summaries, and feature metadata even when raw `frame_logs/` are unavailable.
- **Added regression coverage:** `tests/test_extract_video_features.py` now checks split-clip timing resolution from task windows and verifies `frame_sync.jsonl` falls back to `task_run_windows+video_pts` timestamps.
- **Improved environment failure message in `tools/extract_video_features.py`:** OpenCV imports now go through a shared helper that detects the common NumPy 2.x / OpenCV binary mismatch and raises a short actionable error instead of a long `_ARRAY_API` / `numpy.core.multiarray` traceback.

## 2026-04-10 — World-gaze interest-area & shared-attention visualizer

- **New tool `tools/visualize_gaze_attention.py`**: multi-panel figure for world-gaze output with
  interest-area zones and shared-attention analysis.
  - Panel 1: desk top-view fixation scatter (dot size ∝ fixation duration), colour-coded by
    participant, with zone patches overlaid (P1–P4 circles, Screen, Moderator, Desk).
  - Panel 2: shared-attention heatmap (zones × 5 s time bins; cell value = Σ participant
    gaze-fraction; ≥ 2.0 = multiple participants attending same zone simultaneously).
  - Panels 3–6: per-participant zone-dwell heatmaps (fraction per 5 s bin).
  - Panel 7: overall zone-dwell bar chart (fraction of fixations per zone per participant).
  - Panel 8: interest-area colour legend.
- **IVT fixation detection** built-in (`detect_fixations()`): velocity-threshold grouping in
  world XY (default 0.40 m/s); produces centroid + duration per fixation event.
  Use `--raw-gaze` to bypass and plot all gaze samples instead.
- **Correct physical zone layout** (verified against `tobii_multicam_glasses_tracker.example.yaml`):
  - x-axis = desk DEPTH (0.8 m): Moderator side (x < −0.4) ↔ Screen side (x > +0.4)
  - y-axis = desk WIDTH (1.8 m): Left-side P3/P4 (y < −0.4) ↔ Right-side P1/P2 (y > +0.4)
  - Clockwise tour (top-down): BACK/Moderator → LEFT P4/P3 → FRONT/Screen → RIGHT P1/P2
  - Screen and Moderator span only the short end width (~0.6/0.3 m), not the full desk width.
- **CLI flags**: `--raw-gaze`, `--fixation-velocity M_S`, `--fixation-min-samples N`,
  `--bin-sec`, `--zone-config YAML`, `--session-label`, `--t-start`, `--duration`,
  `--min-markers`, `--max-reproj-px`.
- Tested on grp-13 T1 (all 4 participants, 600 s window):
  `gaze_world_T1/attention_zones_T1_v6.png`.

## 2026-04-10 — grp-13 T1 world-gaze alignment

- Converted Tobii gaze TSV.gz → NDJSON for all 4 participants (P1–P4) using
  `tools/bids_tobii_tsv_to_ndjson.py` (39 k–40 k samples each).
- Auto-generated `configs/tobii_offline_world_align_grp13_T1.yaml` via
  `tools/generate_world_align_config.py`.
- Ran `tobii_multi_glasses_world_align.py`: 23 k–28 k samples aligned per participant
  (12 k–14 k pose frames with PnP solution); outputs in
  `F:\processed_data\sub-01\ses-20260318_grp-13_run01\gaze_world_T1\`.
- MediaPipe pose extraction for grp-13 T1 (`test_mediapipe_pose.py --session-dir`) started
  (running asynchronously on 6 P20 cam videos).

## 2026-04-08 — Pre-session questionnaire: Diana Taune form added

- **Added Diana Taune's questionnaire response** to `metadata/GN Hearing Research – Pre-Session Questionnaire(Sheet1) (1).xlsx` (appended as final row).
- Reformatted incoming file (2) (CSV-in-single-column xlsx) to match the canonical 59-column structure of file (1).
- **Regenerated output files** via `python tools/extract_bfi44_participants.py`:
  - `metadata/participants.tsv` (58 rows, 50 with BFI-44 scores)
  - `metadata/participants.json` (BIDS sidecar)
  - `metadata/name_matching_review.tsv` (match audit)
- Match statistics: 49 → **50 matched**; session participants without questionnaire: 9 → **8**; unmatched questionnaire responses: 7 (unchanged).
- `sub-025` (Diana Taune): age 32, female, right-handed, Fluent English, Master's degree; BFI-44 E=3.125 A=4.444 C=3.444 N=3.0 O=4.4.

## 2026-04-07 - offline compute stack CLI helper

- Added `tools/offline_compute_stack.py` to operationalize the two-workstation offline compute workflow from `docs/offline_compute_stack.md`.
- New subcommands:
  - `init` for stack directory + queue TSV initialization
  - `queue-upsert` for session status tracking (`work_queue.tsv`)
  - `plan` for per-session JSON command plans (BIDS, pose JSON, 3D dry-run/full, QC)
  - `run` for stage-specific command execution with explicit `--execute` safety
- Added tests in `tests/test_offline_compute_stack.py` covering queue upsert behavior and stage command generation.
- Validation: `pytest tests/test_offline_compute_stack.py -q` passed; CLI `--help` check passed.

## 2026-04-07 - offline compute stack guide

- Added `docs/offline_compute_stack.md` for using the Lenovo RTX 5080 workstation, Phoenix RTX A2000 workstation, and 14 TB USB drive as a post-collection processing stack.
- Documented direct-cable networking, Windows static IP/share setup, data staging rules, BIDS packaging commands, MediaPipe pose JSON generation, `video_only_3d_pipeline.py` dry-run/full-run commands, QC commands, and failure modes.
- Updated `ARCHITECTURE.md`, `docs/architecture.md`, and `docs/llm/context_snapshot.md` so future processing sessions can find the guide.

## 2026-04-01 — participants.tsv: exact age column (replaces age_band)

- **Renamed column** `age_band` → `age` in `metadata/participants.tsv` and `metadata/participants.json`.
- **Exact integer age** stored for 51 of 56 respondents who entered a numeric age (col 7 of questionnaire).
- **Band midpoint** stored for 3 matched respondents who used the dropdown band (col 8): `25-34` → 29, `35-44` → 39.
- Updated `docs/bfi44_personality_scores.md` to reflect `age` column and description.

## 2026-04-01 — Pre-session questionnaire updated; participants.tsv regenerated

- **Updated questionnaire source:** `metadata/GN Hearing Research – Pre-Session Questionnaire(Sheet1) (1).xlsx`
  has been updated (same 56 respondents, same column structure; response values may have changed).
- **Regenerated output files** via `python tools/extract_bfi44_participants.py`:
  - `metadata/participants.tsv` (58 rows, 49 with BFI-44 scores)
  - `metadata/participants.json` (BIDS sidecar)
  - `metadata/name_matching_review.tsv` (match audit)
- Match statistics unchanged: 49 matched, 9 session participants without questionnaire, 7 questionnaire responses unmatched to a session.

## 2026-03-31 — Post-collection data audit and metadata report

- **Added `tools/generate_session_metadata_report.py`:** generates a comprehensive per-session
  metadata report TSV (26 rows, 50+ columns) covering all data sources, per-participant stream
  attribution, task durations, and gaps. Supports `--probe-xdf` to count streams from XDF files.
- **Added `docs/data_audit.md`:** documents the full audit of all 26 sessions — completeness
  tiers, device-to-participant mappings, data fixes applied, stimuli multi-directory handling,
  schedule cross-check results, orphan data classification, and recommended processing order.
- **Updated `docs/known_issues.md`:** added data collection gaps (grp-16 missing CurrentStudy,
  grp-10 inflated T0/T1, grp-06 only 3 participants, grp-14 name discrepancy, sub-P001 naming).
- **Updated `data/high_level_data_inventory.json`:**
  - Fixed `session_count` 27 → 26 (duplicate grp-01 entry was removed earlier).
  - Added `currentstudy_additional_xdf` note to grp-09 (sub-P001 294 MB XDF belongs here).
  - Added `currentstudy_confirmed_missing` and `notes` to grp-16.
  - Added Tobii candidates for grp-07 (`20260312T121359Z` = P1) and grp-08 (`20260313T090918Z` = P4).
- **Fixed `tools/xdf_sync_pipeline.py`:** numpy truth-value ambiguity in `extract_xdf_streams()`,
  per-XDF try/except for corrupt file isolation, stimuli multi-directory merging (search inside
  candidate dirs, compute per-TSV durations independently, keep longest clean run per task).
- **Data fixes applied:** renamed 12 misnamed stimuli files (grp-01 → grp-11), renamed misnamed
  recording session XDF/JSON (grp-01 → grp-11), identified sub-P001 XDF as grp-09, classified
  AffectAI/test XDF as Vicon MoCap test (not session data).
- **Added `configs/session_schedule.tsv`** cross-check: schedule date/time/name comparison with
  mojibake-safe name normalization (latin-1 → utf-8 re-encoding).
- **Why:** establishes a verified data inventory and audit trail before downstream pipeline
  processing begins.

## 2026-03-26 — Repo pivot: post-collection processing focus

- Data collection is complete. The repository focus has shifted to **post-collection processing**,
  feature extraction, and analysis across three pipelines:
  1. **Sync & BIDS packaging** — `multisource_to_bids_runs.py` + `raw_to_bids.py`
  2. **3D pose, gaze & gesture** — `video_only_3d_pipeline.py` and supporting tools
  3. **Analysis & QC** — `qc/qc_sync_report.py`, `qc/qc_tobii_world_gaze.py`, `analyze_sync.py`
- Updated `README.md` to lead with processing pipelines and remove capture-only commands.
- Updated `ARCHITECTURE.md` to serve as a processing-focused navigation hub.
- Updated `docs/architecture.md` to describe the three processing pipelines and BIDS output
  structure; moved device inventory to reference-only section.
- Updated `docs/data_flow.md` overview and post-processing section to reflect actual pipeline
  commands and correct output paths; updated Tools section to list processing tools only.



- Added `tools/video_only_3d_pipeline.py` as a single-command offline pipeline for multicam video-only post-processing.
- Pipeline stages: Tobii glasses pose/world-gaze (`tobii_multicam_glasses_tracker.py`) -> multicam 3D reconstruction (`multicam_pose3d.py`) -> optional refinement (`refine_skeleton_3d.py`) -> gesture extraction per participant.
- Added deterministic gesture event outputs (`gestures_events.ndjson`, `gestures_summary.json`) with configurable thresholds and minimum-duration run collapsing.
- Added `--dry-run` mode to `tools/video_only_3d_pipeline.py` to validate prerequisites up front and emit `pipeline_dry_run.json` (including missing inputs like absent pose `*_json` folders) before heavy processing.
- Added missing-calibration fallback in `tools/video_only_3d_pipeline.py`: when `--calibration` is absent, the pipeline now auto-runs `tools/calibrate_charuco.py calibrate` using six detected P20 videos (`panacast-20-cam1..6`) from `--videos-dir` and uses the generated `auto_calibration_charuco.toml`.
- Added `docs/video_only_3d_pipeline.md` with required inputs, example command, outputs, and gesture labels.
- Added tests in `tests/test_video_only_3d_pipeline.py` for gesture extraction behavior and dry-run prerequisite checks.
- Fixed `tools/tobii_multicam_glasses_tracker.py` video-to-calibration camera matching to be separator-insensitive (hyphen/underscore/prefix differences), so long BIDS-style filenames map correctly to calibration camera names.
- Fixed `tools/test_mediapipe_pose.py` JSON output robustness on Windows by creating parent output directories before writing per-frame keypoint JSON files.
- Why: provide one practical pipeline for desk-centered shared-coordinate Tobii gaze fusion and gesture extraction from multi-camera video sessions.

## 2026-03-22 — Multisource post-processing: participant-linked stimuli answers + duplicate-capture-safe media naming

- **Extended `tools/multisource_to_bids_runs.py`:** now exports a normalized long-form stimuli answer table at `beh/sub-*_ses-*_task-T0T1T2T3T4_stimuli_answers.tsv` by parsing stimuli `responses_*.jsonl` payloads into readable rows (`response_type`, `participant`, `item_key`, `item_value`, task/phase, clocks).
- **Phase-aware run boundaries:** task windows are now derived from experiment-control phase markers when present: `T0` starts at intro push (`study_introduction`/intro phase) and ends at task finish; `T1`..`T4` start at Tobii calibration push (`tobii_calibration`) and end at task finish. Falls back to legacy first-task-event/next-task-start logic only if required markers are missing.
- **Participant signal mapping:** added `annot/sub-*_ses-*_participant_signal_map.tsv` to relate participant-linked signals across Tobii streams, EmotiBit streams, and stimuli response devices to `P1`..`P4` when inferable.
- **EmotiBit mapping context:** participant map export includes static config mappings from `configs/emotibit_participants_by_source.json` (participant↔hardware-id and participant↔source endpoint) when available.
- **Duplicate-capture handling for media split:** when multiple captures exist for the same logical AV/Tobii source, clip naming now adds a deterministic capture token (`cap-<run>`) in `acq-*` to prevent task-clip overwrite collisions.
- **Added tests:** `tests/test_multisource_to_bids_runs.py` now validates stimuli answer normalization and participant signal-map generation behavior.
- **Why:** addresses real multi-PC/session edge cases where source names differ, duplicate captures coexist, and downstream analysis needs participant-resolved stimuli answers and sensor mapping without manual reconstruction.

## 2026-03-22 — Multisource post-processing: add per-task media splitting (video/audio clips)

- **Extended `tools/multisource_to_bids_runs.py`:** added `--split-media` flag to generate per-task clipped video and audio files (T0–T4, run-01) from source media.
- **Media sources:** handles AV cameras (7× MKV), DPA audio (5× WAV), and Tobii scene videos (per-device MP4). Detects media start times from frame logs, progress TSV, LSL anchors, or capture events.
- **Sync-aware trimming:** maps task window wall-clock boundaries to media-relative times and uses ffmpeg `-ss` (seek-start) + `-t` (duration) for deterministic trimming. Audio delays are handled transparently.
- **BIDS output naming:** `sub-{id}_ses-{id}_task-T*_run-01_acq-{source}-{device}_{modality}.{ext}` (e.g., `task-T0_run-01_acq-av-jabra-panacast-20-cam1_video.mkv`).
- **Integration:** `merge_and_convert()` automatically runs media splitting when `--split-media` flag is passed; task windows already computed during run chunking are reused.
- **Lazy-load cv2:** sync-video module (which imports cv2) is only loaded when media splitting is enabled, unblocking tests that don't require video processing.
- **Added validation:** tested on real `sessions/Final` data; all 40 AV video clips (8 cameras × 5 tasks) generated successfully; audio clips skipped when start anchors unavailable (graceful fallback).
- **Why:** decoupled per-task media files enable downstream analysis and review tools without re-editing or frame-counting; sync-aware trimming preserves temporal alignment with event markers.

## 2026-03-21 — Multisource post-processing: merge AV/Recording/Stimuli/Tobii into BIDS task runs (T0–T4)

- **Added `tools/multisource_to_bids_runs.py`:** new end-to-end post-processing command that ingests split source folders (AV PC, Recording PC, Stimuli logs, optional Tobii downloads), merges them into one `ses-*` tree under `sourcedata/` (copy or hard-link mode), runs existing `tools/raw_to_bids.py`, and generates per-task run outputs (`task-T0`…`task-T4`, `run-01`).
- **Run chunking outputs:** writes task windows (`annot/*_task_run_windows.tsv`), per-task behavior events (`beh/sub-*_task-T*_run-01_events.tsv`), and run-sliced LSL-derived tables when available (`et/`, `physio/`, `annot/`, `beh/` with `lsl_time` filtering).
- **Authoritative events spine:** when stimuli experiment logs are provided, session-level `events.tsv` is derived from stimuli events so one timeline spine reflects actual task/phase execution.
- **Added tests:** `tests/test_multisource_to_bids_runs.py` validates deterministic task-window derivation and run-splitting of LSL TSV data.
- **Why:** sessions recorded on separate PCs/devices can now be merged and chunked into BIDS-aligned task runs without manual file surgery.

## 2026-03-19 — Session GUI: scrollable schedule list with quick top/bottom navigation

- **`tools/session_orchestrator_gui.py`:** schedule group table now has a vertical scrollbar and mouse-wheel scrolling support.
- **`tools/session_orchestrator_gui.py`:** added `Top` and `Bottom` buttons in the Schedule section to jump to first/last group row and auto-select it.
- **Why:** makes long schedules easier to navigate and select without losing access to rows at the top or bottom.

## 2026-03-18 — Recording GUI: Vicon LSL bridge default set to disabled

- **`tools/session_orchestrator_gui.py`:** Vicon bridge toggle now initializes to OFF by default (`self._vicon_enable_var = tk.BooleanVar(value=False)`).
- **`tools/session_orchestrator_gui_recording.py`:** role-locked Recording-PC entrypoint also forces `Enable Vicon bridge` to OFF at startup.
- **Why:** prevents automatic Vicon bridge attempts from any GUI startup path; operators can still enable it explicitly when needed.

## 2026-03-17 — Recording GUI: robust Tobii LSL stream detection

- **`tools/session_orchestrator_gui.py`:** `_wait_for_tobii_lsl_streams` now uses a tolerant stream matcher instead of a strict `name.startswith("Tobii_") and type == "EyeTracking"` filter.
- Detection now accepts regular participant Tobii streams by name/type/source-id variants and excludes known irregular Tobii outlets (`_Event`, `_Imu`, `_SyncPort`, `evetns_tobii`).
- **Why:** avoids false negatives in GUI readiness when Tobii streams are present but metadata differs from the strict legacy expectation.

## 2026-03-17 — Vicon bridge: lower-overhead defaults for JSON streams and file I/O

- **`tools/vicon_nexus_lsl_bridge.py`:** reduced runtime overhead in high-frequency sessions by adding configurable output throttles and buffered file flushing.
- **Added JSON LSL rate limits** (enabled by default):
  - `--frame-json-lsl-max-hz` (default `50.0`)
  - `--markers-json-lsl-max-hz` (default `50.0`)
  - use `<=0` to disable throttling.
- **Added NDJSON flush control:** `--output-flush-interval-s` (default `1.0`), replacing per-frame flushes.
- **Added loop pacing control:** `--loop-sleep-ms` (default `1.0`) to avoid max-spin CPU usage.
- **Why:** keeps raw capture fidelity while reducing CPU, serialization pressure, and I/O churn from redundant full-frame JSON emissions.

## 2026-03-17 — Recording GUI: expose/pass Vicon performance controls

- **`tools/session_orchestrator_gui.py`:** Recording-PC **Vicon Bridge Settings** now include explicit numeric fields for:
  - `Frame JSON Hz` (default `50`)
  - `Marker JSON Hz` (default `50`)
  - `Flush (s)` (default `1.0`)
  - `Loop sleep (ms)` (default `1.0`)
- **Launch wiring:** `_step_vicon_bridge` now validates these values and always passes them to `tools/vicon_nexus_lsl_bridge.py` as `--frame-json-lsl-max-hz`, `--markers-json-lsl-max-hz`, `--output-flush-interval-s`, and `--loop-sleep-ms`.
- **Why:** makes runtime load tuning explicit in GUI recording flow instead of relying on script defaults only.

## 2026-03-16 — Recording GUI: add Vicon LSL bridge step

- Added a new Recording-PC session step in [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py): **Vicon LSL Bridge**.
- Added Recording-PC configuration field `Vicon server` (host:port, default `localhost:801`).
- The new step launches [tools/vicon_nexus_lsl_bridge.py](tools/vicon_nexus_lsl_bridge.py) with marker-focused flags:
  - `--per-stream-lsl --markers-lsl --structured-lsl --no-devices --no-eye-trackers`
  - output path is session-scoped: `sourcedata/vicon_lsl/vicon_datastream_raw.ndjson`
- Added readiness wait similar to Tobii flow: GUI now waits for any `Vicon*` LSL stream before continuing.
- Added capture tracking artifact section `vicon_lsl_bridge` for session manifests.
- Why: enables Vicon marker/subject streams to be captured by Recording-PC GUI pipeline just like Tobii bridge streams.

## 2026-03-16 — Recording GUI: Vicon preflight + recording settings

- Expanded Recording-PC settings in [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py) with a **Vicon Bridge Settings** block:
  - enable/disable Vicon bridge step,
  - toggle `per-stream`, `markers JSON`, `structured segment` outputs,
  - toggle inclusion of Vicon device and eye-tracker outputs.
- Vicon step launch command now honors these GUI toggles instead of hardcoded flags.
- Preflight now checks Vicon DataStream endpoint reachability from the configured `Vicon server` (`host:port`) and reports status in the diagnostics panel.
- LSL monitor remains Vicon-aware with `Vicon:*` summary counts and stream listing.
- Updated Recording-PC default `Vicon server` value to `10.145.48.221:801` to match current lab setup.

## 2026-03-16 — Vicon bridge: raw marker trajectories for moved balls

- **Added marker capture** to [tools/vicon_nexus_lsl_bridge.py](tools/vicon_nexus_lsl_bridge.py) for both SDK backends (Python and .NET):
  - subject-attached markers (`GetMarker*`),
  - labeled markers (`GetLabeledMarker*`),
  - unlabeled markers (`GetUnlabeledMarker*`).
- **Added marker controls:**
  - `--no-markers` to disable marker collection,
  - `--markers-lsl` to publish one JSON marker-frame stream (`ViconMarkerFrame`) per frame.
- **Extended per-stream forwarding:** with `--per-stream-lsl`, marker positions are now published as numeric LSL streams:
  - `ViconMarker_<subject>_<marker>`
  - `ViconLabeledMarker_<id>`
  - `ViconUnlabeledMarker_<id>`
- **Why:** supports raw trajectory capture of moved balls in addition to segment transforms.
## 2026-03-16 — GUI Azure defaults: tools/azure_strorage credentials + SAS-safe destination

- **`tools/session_orchestrator_gui.py`:** Azure destination now defaults to `https://affectai.blob.core.windows.net/raw` and Blob credential path auto-defaults to `tools/azure_strorage/azure_strorage/.env` when present (falls back to existing JSON locations).
- **`tools/session_orchestrator_gui.py`:** Blob credential loader now accepts both JSON and `.env` formats, including `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY`, and `AZURE_CONTAINER_NAME` from the Azure helper tool folder.
- **`tools/session_orchestrator_gui.py`:** Raw upload now prefers account-key mode when credentials are available, so operators can upload without separate `azcopy login`.
- **`tools/upload_raw_data.py`:** fixed `--destination-url` handling with SAS query params by appending `/{session_id}/{role}` to the URL path (not after the query string), preventing malformed AzCopy targets.
- **`tools/azure_strorage/azure_strorage/.env.example`:** prefilled `AZURE_STORAGE_ACCOUNT_NAME=affectai` and `AZURE_CONTAINER_NAME=raw` to match the lab storage account and default upload container.

## 2026-03-16 — GUI Azure upload: support URL and account-key credential modes

- **`tools/upload_raw_data.py`:** added dual upload backends for raw session upload: existing `--destination-url` (`azcopy`) path and a new SDK fallback using `--account-name` + `--container-name` + key from env (`--account-key-env`, default `AFFECTAI_AZURE_ACCOUNT_KEY`).
- **`tools/upload_raw_data.py`:** SDK-mode `--dry-run` is now fully offline (no Azure SDK import/network call), so operators can preflight command paths even when Azure packages/network are unavailable.
- **`tools/session_orchestrator_gui.py`:** Blob credential loader now accepts both URL-style JSON (`destination_url`/`container_url`/etc.) and account-based JSON (`account_name`, `account_key`, `container_name`). GUI upload falls back to SDK mode when URL is not provided.
- **Security hardening:** account keys are now passed to subprocesses via environment variable only (not CLI args), so they are not echoed in Data Ops command logs.
- **`configs/azure_blob_credentials.example.json`:** expanded template to show both supported credential shapes.

## 2026-03-13 — docs: sync lsl_architecture_alignment + context_snapshot with current state

- **`docs/lsl_architecture_alignment.md`:** marked all six LSL architecture gaps (Gaps 2–6) as resolved; updated the implementation status table; replaced stale TODO code templates with historical-reference notes and a pointer to `docs/llm/context_snapshot.md`. All gaps were already implemented; the document was misleadingly showing MISSING/TODO entries.
- **`docs/llm/context_snapshot.md`:** added explicit CLI option names `--late-discovery-interval` (default 5 s) and `--boundary-interval` (default 30 s) to the XDF recorder entry, replacing the vague "configurable via CLI" note.

## 2026-03-12 — configs: add grp-08 to session_schedule.tsv

- Added `grp-08` (Thu 12 March 13:00–14:10) sourced from `configs/GN Hearing Research Session – Availability.xlsx`.
- Participants: harshavardhan reddy, Jana Rösner Gjoni, Olena Mubako, Daniela Stancu.
- **Files:** [configs/session_schedule.tsv](configs/session_schedule.tsv)

## 2026-03-12 — LSL XDF recorder: per-stream micro-batching for file efficiency

- **Problem:** GUI recorder produced many tiny sample chunks (often ~1 sample/chunk), inflating XDF size and write overhead versus LabRecorder.
- **Fix:** `tools/lsl_xdf_recorder.py` now micro-batches per stream in the puller loop (target 32 samples, max 100 ms latency) before encoding/enqueuing a chunk.
- **Reliability guard:** pending buffers are flushed before reconnect and on stop/final-drain, so batching does not drop samples.
- **Outcome:** significantly higher samples-per-chunk and closer file efficiency to LabRecorder while preserving capture completeness.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py)

## 2026-03-12 — LSL XDF recorder: pylsl 1.18 compatibility fix (critical)

- **Problem:** GUI recorder XDF files were much smaller than LabRecorder because most puller threads crashed immediately.
- **Root cause:** `tools/lsl_xdf_recorder.py` used `StreamInlet.pull_numeric_chunk` and `pylsl.LostError`, but pylsl 1.18.1 provides neither API.
- **Observed impact:** only one surviving stream (`evetns_tobii`) contributed samples, while other streams produced headers but no sample chunks.
- **Fix:** switched all pull paths to `StreamInlet.pull_chunk` (works for numeric + string streams) and replaced `pylsl.LostError` handling with `RuntimeError` stream-loss handling + reconnect.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py)

## 2026-03-12 — LSL XDF recorder: restart-safe stream continuity

- **Problem:** using volatile outlet UID as the de-dup/reconnect key can miss stream continuity after an outlet restart (new UID), and a 30 s late-discovery cadence delays attachment of newly visible streams.
- **Fix:** recorder now tracks logical streams with a session-stable key `(name, type, source_id, hostname, channel_count, channel_format)` and reconnects/late-joins against that key.
- **Improved coverage:** late-stream discovery default reduced from 30 s to 5 s, with new CLI option `--late-discovery-interval`.
- **Extra resolve support:** `extra_predicates` are now honored via `resolve_bypred` in addition to normal discovery.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py)

## 2026-03-12 — LSL XDF recorder: start/stop stream manifest logging

- **Added:** deterministic stream manifest logging at recorder start and stop in [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py).
- **Start log:** prints one line per resolved stream (`name`, `type`, `source_id`, `uid`, `hostname`, channels, nominal rate) in stable sorted order.
- **Stop log:** prints one line per recorded stream with the same identity fields plus per-stream `samples` totals and XDF stream id.
- **Why:** gives operators a direct parity checklist vs LabRecorder stream inventory and a quick way to spot missing/under-recorded streams.

## 2026-03-12 — LSL XDF recorder: stream identity + reconnect correctness

- **Problem:** Python recorder sessions could miss streams (and produce smaller XDFs than LabRecorder) when multiple outlets shared weak identifiers or when reconnect matched by stream name only.
- **Fix:** `tools/lsl_xdf_recorder.py` moved away from name-only matching to multi-field identity matching; it now uses robust identity keys and restart-safe session continuity matching for reconnect/late-join behavior.
- **Throughput hardening:** reduced pull timeout (`0.02 s`) and increased per-pull `max_samples` to `4096` to better drain high-rate bursts and reduce inlet overrun risk under heavy multi-stream load.
- **Why:** Improves parity with LabRecorder stream coverage and final XDF completeness in high-concurrency lab sessions.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py)

## 2026-03-11 — LSL XDF recorder: single-writer-thread architecture

- **Root cause of tiny XDF files:** With ~79 LSL streams each running a
  puller thread, all 79 threads competed for a single `_write_lock` to write
  to the file.  Python's GIL + lock convoy starvation meant ≤1% of available
  samples were actually written (336 samples in 25 min; expected ~2 M).
- **Fix:** Rewrote `lsl_xdf_recorder.py` to a **single-writer-thread**
  architecture.  Puller threads now push encoded chunks to per-stream
  `collections.deque` queues (GIL-atomic append).  A single `_writer_loop`
  thread drains all queues every 100 ms, writes sample + clock-offset chunks,
  and handles boundary chunks + flush — eliminating all cross-thread lock
  contention.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py)

## 2026-03-11 — LSL XDF recorder: guarantee all streams in one XDF

- **Root cause of missing Tobii in XDF:** `_step_tobii_bridge` returned after only 2 s (checking the process was alive), but the Tobii bridge takes 30-120 s to connect to hardware and publish LSL outlets.  The recorder then started and resolved streams before any `Tobii_P*_stream` was visible.
- **Fix:** Added `_wait_for_tobii_lsl_streams()` to `session_orchestrator_gui.py`. After spawning the bridge process, the step now polls LSL every 5 s (using a lazy `from pylsl import resolve_streams`) until all expected `Tobii_P*_stream` / `EyeTracking` streams are visible, up to 150 s.  The GUI status bar shows live progress (e.g. `Tobii LSL Bridge: 2/4 stream(s) ready`).  On timeout it warns and continues — the XDF recorder's late-discovery loop will add any still-missing streams.
- **ffmpeg / AV-PC streams:** Handled by the late-discovery loop (added in previous commit); the loop rescans every 30 s and dynamically adds new streams mid-recording.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py)

## 2026-03-11 - Calibration TOML Auto-Resolve + Live Validation Marker Robustness

- **Changed:** AV GUI feed start and calibration step-3 live validation now auto-resolve the newest calibration TOML when the selected path is empty or stale. Discovery scans session and out-root scopes and prefers `video_camera_calibration.toml` (with fallback names).
- **Changed:** `Find Latest` calibration now uses the same broader newest-file resolver instead of only scanning one output directory.
- **Enhanced:** `validate_calibration_robust.py` now supports multi-dictionary ArUco detection via `--aruco-dicts`; marker-map dictionary is always included and additional dictionaries are merged by marker id for better detection recall.
- **Changed:** GUI live validation now passes `--aruco-dicts DICT_4X4_50,DICT_4X4_250` to align marker detection robustness with Show Feed behavior.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [tools/validate_calibration_robust.py](tools/validate_calibration_robust.py)

## 2026-03-11 - Feed: Camera-Aware Zone Resolution + Robust Glasses Marker Detection

- **Fixed:** Feed now applies desk-zone camera overrides using camera aliases from capture settings (`cam1`, `camera1`, `cam_0`) plus calibration camera keys, so zone organization files are honored even when camera labels differ from zone JSON keys.
- **Fixed:** Glasses marker detection now runs on full-resolution frames and supports multiple ArUco dictionaries (default: `DICT_4X4_50,DICT_4X4_250`), then scales marker overlays/centers to preview view. This improves marker recall in fast/downscaled feed mode.
- **Changed:** Added `--aruco-dicts` option to `online_multicam_feed.py` for explicit marker dictionary control.
- **Files:** [tools/online_multicam_feed.py](tools/online_multicam_feed.py)

## 2026-03-11 - Feed Overlay: Correct Skeleton Alignment + Camera Mapping

- **Fixed:** `Show Feed` skeleton overlays now align with configured desk zones when fast mode preview downscaling is enabled. Zone rectangles are scaled with the preview frame before pose crop/detection.
- **Fixed:** 3D triangulation now stores per-person keypoints by camera label (not person text label), so camera-to-calibration mapping remains valid during live feed.
- **Changed:** 2D feed overlay now draws both landmark points and skeleton connections for clearer operator visualization.
- **Files:** [tools/online_multicam_feed.py](tools/online_multicam_feed.py)

## 2026-03-11 - Calibration: Auto-Exclude Cameras with Insufficient Charuco Detections

- **Fixed:** Calibration no longer fails with "Could not build calibration graph" when one or more cameras have zero (or near-zero) charuco detections. Cameras with fewer than `--min-charuco-frames` detections (default: 15) are automatically excluded before the graph is built; a clear `EXCLUDED <camera>: N frames` warning is printed for each dropped camera, and the calibration proceeds with the remaining cameras.
- **Added:** `--min-charuco-frames` CLI argument to `calibrate_charuco.py calibrate` (default: 15). Lower it if you have very short clips; raise it to require better coverage per camera.
- **Root cause in this session:** `jabra_panacast_20_cam6` had 0/2220 charuco frames and `jabra_panacast_50` had only 10/2220, both producing isolated graph nodes. The other 5 PanaCast 20 cameras had excellent co-visibility (800–1822 frame pairs) and calibrate successfully once the two sparse cameras are excluded.
- **Files:** [tools/calibrate_charuco.py](tools/calibrate_charuco.py)

## 2026-03-11 - Calibration Discovery: Broad Scan + video/-Subfolder Handling

- **Fixed:** "2) Calibrate" and "Find Latest" now work even when no session is actively loaded in the GUI (empty Session ID, stale auto-incremented ID, or participants not yet populated). Discovery first checks the form-derived session dir, then falls back to an `rglob("video")` scan across all under `{out_root}/{recording_type}`, picking the newest run dir that contains at least one actual video file.  This ensures previously recorded `.mkv` captures (e.g. `20260311_grp-A_run01_20260311_091843/video/*.mkv`) are always found regardless of naming convention.
- **Fixed:** If the operator browses to or pastes the `video/` subfolder path directly into *Capture dir*, the GUI now silently goes up one level to return the correct run dir instead of failing with "No video files found".
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py)

## 2026-03-11 - Calibration Folder Auto-Creation + Cleaner Stop Logs

- **Changed:** Calibration capture now auto-creates/resolves the session directory from the GUI Session ID (`{out_root}/{recording_type}/sub-XX/ses-{session_id}`) instead of requiring manual folder selection when no session is running.
- **Changed:** Calibration recordings and live-validation captures now write under `<session_dir>/sourcedata/av/calibration/` (timestamped `calibration-video*` / `calibration-validation-live*`), with ffmpeg run labels tied to the session name.
- **Changed:** Calibration panel now shows a live `Output dir` preview so operators can verify the exact session-scoped calibration folder before starting step 1.
- **Fixed:** Calibration capture auto-discovery now supports session-stamped run folders under `<session_dir>/sourcedata/av/calibration/` (for example `calibration-video_<session_id>_<timestamp>`), so step 2 no longer falls back to the session root when locating videos.
- **Changed:** Calibration `Capture dir` field now includes a folder browser button so operators can reuse any previous capture folder directly.
- **Fixed:** LSL sidecar recording in `ffmpeg_multicap.py` now disables inlet auto-recovery and is stopped before capture teardown, preventing repeated `Stream transmission broke off ... re-connecting` shutdown spam in calibration logs.
- **Fixed:** Calibration log monitor suppresses that specific benign liblsl reconnect line so operators only see actionable calibration output.
- **Fixed:** Calibration subprocesses are launched unbuffered (`PYTHONUNBUFFERED=1`) so startup/countdown logs are streamed in order instead of appearing late at shutdown.
- **Fixed:** LSL recorder worker shutdown now closes inlet streams before thread join, reducing noisy `Stream transmission broke off` reconnect spam during stop.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [tools/ffmpeg_multicap.py](tools/ffmpeg_multicap.py), [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)

## 2026-03-11 — LSL XDF recorder: fix write-lock contention + late-stream discovery

- **Bug (write throughput):** `_encode_and_write` called `self._fh.flush()` while holding the shared `_write_lock`.  With 87+ concurrent stream threads each doing this, the lock was contended >90% of the time, reducing throughput from the expected 25–40 KB/s to ~3 KB/s.  Fixed by removing `flush()` from `_encode_and_write` and instead flushing once per `boundary_interval` (30 s default) inside `_boundary_loop`.
- **Bug (missing streams):** Streams that come online after the 5-second resolve window (e.g. Tobii LSL Bridge finishing device initialisation ~2 min into a session) were never recorded.  Added `_late_stream_discovery_loop` — a daemon thread that rescans for new LSL streams every 30 s and dynamically adds new `StreamHeader` + inlet threads to the live XDF file.
- **GUI default:** `_py_recorder_resolve_var` changed from `"5"` to `"30"` seconds so the initial discovery window is long enough to catch most services.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py), [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py)

## 2026-03-10 — mDNS tablet discovery + Android companion app

- **Problem:** DHCP IP changes on the lab router broke hardcoded tablet URLs; operators had to manually update every tablet browser.
- **Fix (server-side):** `stimuli/display_server.py` now optionally registers the server as `affectai-display._http._tcp.local.` via mDNS (requires `pip install zeroconf`).  Tablets can permanently bookmark `http://affectai-display.local:8080/tablet/N` — the hostname resolves to whatever the current server IP is.  Falls back gracefully when `zeroconf` is not installed (prints a hint, IP-based URLs still show as before).
- **Moderator panel:** Network panel now shows the stable `.local` URLs in green when mDNS is active, with IP-based fallback URLs below.  Hint text updated to mention the `zeroconf` option.
- **`requirements.txt`:** Added `zeroconf>=0.131` as an optional dependency.
- **New `stimuli/android_tablet_app/`:** Complete Android/Kotlin companion app.  Uses `NsdManager` to auto-discover the `affectai-display` mDNS service, resolves host+port, and opens the correct `/tablet/N` URL in a full-screen WebView.  Falls back to the last-known IP, then to a manual URL entry form.  Tablet number (1-4) is set once via a settings panel and persisted in SharedPreferences.
- **Files:** `stimuli/display_server.py`, `requirements.txt`, `stimuli/android_tablet_app/` (new).
## 2026-03-10 - Stop Session: Full Process Teardown

- **Fixed:** `_force_stop_all_processes` now clears `_calibration_job_name` and `_calibration_done_callback` when it kills a running calibration job, preventing stale callbacks from chaining into the next calibration step after a forced stop.
- **Fixed:** `_monitor_calibration_output` no longer shows an error dialog when the calibration process was intentionally terminated by Stop Session; instead it logs a quiet info message ("stopped externally").
- **Result:** Pressing Stop Session cleanly kills all running processes (FFmpeg multicap via supervisor, surveillance feed, calibration step 1/2/3, LabRecorder/XDF recorder, Tobii) and resets calibration UI status to "Stopped" with no spurious dialogs.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py)

## 2026-03-10 - AV GUI Live Validation Capture (Auto)

### Calibration Step 3 Behavior
- **Changed:** `Live Validation` now performs a short automatic FFmpeg capture first (default: 20s), then runs desk-marker validation on that fresh capture.
- **Added:** `Live val (s)` GUI field to set short capture duration for step 3.
- **Storage:** step-3 capture writes under session folder as `calibration-validation-live*`, with validation reports in that run's `video/` directory.
- **Motivation:** provide true live validation workflow instead of validating only older calibration clips.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)

## 2026-03-10 - AV GUI Calibration Workflow (3-Step)

### Session Orchestrator AV Calibration Panel
- **Replaced calibration controls** in GUI with explicit 3-step flow:
  1. **Record Calibration Video** (75s by default) using `ffmpeg_multicap.py`
  2. **Calibrate** using `tools/calibrate_charuco.py calibrate`
  3. **Live Validation** using desk-marker checks via `tools/validate_calibration_robust.py`
- **Storage behavior:** calibration recordings now run into the selected session folder under `calibration-video*` directories, keeping artifacts alongside the main session data tree.
- **Calibration outputs:** generated TOML saved as `video/video_camera_calibration.toml` in the selected calibration capture folder and auto-populated into the GUI calibration path field.
- **Validation outputs:** writes text + JSON reports into the same calibration capture `video/` directory (`video_camera_calibration_live_validation.*`).
- **GUI additions:** explicit status line for calibration jobs, board-config path input, marker-map input, capture-dir selector (`Find Latest`), and calibration TOML path field.
- **Motivation:** match operator workflow request for deterministic capture → solve → validate sequence in AV GUI.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)

## 2026-03-10 - AV GUI Feed Control Fix

### Session Orchestrator AV Feed Behavior
- **Fixed:** AV feed control was incorrectly gated on `recording_active`, which is only true on Recording PC; this kept surveillance feed disabled on AV role.
- **Changed:** Feed readiness is now role-aware:
  - Recording PC surveillance feed requires active recording marker state.
  - AV PC surveillance feed requires active `ffmpeg_multicap` process and initialized session dir.
- **Added:** Automatic surveillance feed startup hook after FFmpeg step starts successfully (when "Enable feed" is checked).
- **Fixed:** After stopping feed on AV role, Start Feed button now re-enables correctly when prerequisites are still met.
- **Motivation:** Make AV feed controls functional and consistent with AV-only workflow.
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)
## 2026-03-10 (feature) - Pure-Python LSL→XDF Recorder Replaces LabRecorder.exe

- **New `tools/lsl_xdf_recorder.py`:** pure-Python LSL→XDF 1.0 recorder using only `pylsl` (no LabRecorder.exe needed). Resolves all visible LSL streams (or a prefix-filtered subset), records them to a single `.xdf` file with clock-offset chunks and periodic boundary markers. CLI: `python tools/lsl_xdf_recorder.py --output <path>.xdf [--prefixes Emotibit_ ffmpeg_clock ...] [--resolve-timeout 5]`.
- **XDF stored in `sourcedata/lsl/`:** both the Python recorder and LabRecorder.exe path now write to `<session_dir>/sourcedata/lsl/<session_id>.xdf`, keeping XDF alongside other source data.
- **GUI path display:** a "Saving to:" label below the size counter shows the exact XDF file path once recording starts.
- **File size updates in real time:** each sample chunk is immediately flushed to disk so the OS file size reflects live data.
- **Stream reconnection:** if a stream disappears (network hiccup / device restart), `_StreamRecorder` waits for it to reappear by name and resumes writing into the same XDF file without restarting the recording. Back-off is 2–30 s. The clock-poller also uses the fresh inlet automatically.
- **Files:** [tools/lsl_xdf_recorder.py](tools/lsl_xdf_recorder.py), [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py)

## 2026-03-10 - AV GUI Uses P50 Preset by Default

### Session Orchestrator AV Start Behavior
- **Added:** AV GUI preflight now auto-applies P50 color controls before starting `ffmpeg_multicap`
  - Uses `tools/lock_exposure.py` with: `brightness=160`, `contrast=150`, `saturation=138`, `sharpness=170`
  - P50 camera is resolved from the active FFmpeg config (`video_alt_name` preferred, `video_index` fallback)
  - Startup fails loudly if the preset cannot be applied, preventing accidental untuned recordings
- **Changed:** AV role now suppresses `--show-camera-dialog` even if toggled in UI, so recording starts without camera property popups
- **Changed:** AV entrypoint explicitly defaults camera-dialog option to `false`
- **Motivation:** Ensure the finalized P50 color-tuned settings are consistently used for GUI-driven recording sessions and avoid interruptive dialogs
- **Files:** [tools/session_orchestrator_gui.py](tools/session_orchestrator_gui.py), [tools/session_orchestrator_gui_av.py](tools/session_orchestrator_gui_av.py), [docs/llm/context_snapshot.md](docs/llm/context_snapshot.md)

## 2026-03-10 - P50 Color Balance Final Tuning (Saturation 138)

### Camera Control Preset Finalized
- **Determined optimal P50 image control settings:** brightness=160, contrast=150, saturation=138, sharpness=170
  - **Color balance:** R/B ratio 1.017 (neutral, with slight warm cast to match natural lighting)
  - **Bitrate:** ~49 Mbps maintained (no quality loss)
  - **Visual quality:** Eliminates orange/yellow cast from previous settings while keeping image warm
- **Default workflow before each session:**
  ```powershell
  python tools/lock_exposure.py 4 --brightness 160 --contrast 150 --saturation 138 --sharpness 170
  python tools/ffmpeg_multicap.py --config configs/ffmpeg_multicap.json --group-id <session_id> ...
  ```
- **Note:** Settings are non-persistent; must be reapplied before each recording session
- **Files:** [README.md](README.md), [tools/lock_exposure.py](tools/lock_exposure.py)

### Configuration Cleanup
- **Removed:** `show_camera_dialog: true` from camera device blocks (cam4, cam6, P50)
  - No longer interrupts capture startup with manual property dialogs
  - Simplifies unattended recording workflow
- **Files:** [configs/ffmpeg_multicap.json](configs/ffmpeg_multicap.json)

## 2026-03-10 - Multi-Camera Recording Improvements

### Recording Duration Control
- **Fixed:** Removed hardcoded 1-hour (3600s) recording limits from all FFmpeg capture commands
- **Added:** `--max-duration` command-line argument (default: 7200s = 2 hours)
  - Use `--max-duration 0` for unlimited recording
  - Use `--max-duration 10800` for 3-hour sessions
- **Motivation:** User reported unexpected 1-hour stops; needed safe default + configurable override
- **Files:** [tools/ffmpeg_multicap.py](tools/ffmpeg_multicap.py), [README.md](README.md)

### Camera Quality Fix - PanaCast 50 Pixel Format
**Problem:** Three cameras recording at significantly lower bitrates:
- PanaCast 50: 32 Mbps (55% below normal)
- Cam 4 (P20): 50 Mbps (35% below normal)  
- Cam 6 (P20): 52 Mbps (33% below normal)
- Expected: 65-85 Mbps for MJPEG at 1920x1080@30fps

**Root Cause Analysis:**
- Used `ffmpeg -list_options` to detect native camera output formats
- **PanaCast 50**: Outputs `nv12` (4:2:0 chroma subsampling, 1.5 bytes/pixel)
- **PanaCast 20**: Outputs `yuyv422` (4:2:2 chroma subsampling, 2.0 bytes/pixel)
- 33% less raw data → 55% lower bitrate after MJPEG compression
- USB topology check confirmed bandwidth was not the bottleneck

**Solution Implemented:**
1. Added `pixel_format` field to `DeviceConfig` class
2. Updated DirectShow command builder to pass `-pixel_format <format>` to ffmpeg
3. Configured PanaCast 50 to force `yuyv422` format in config
4. Improved MJPEG quality: `-q:v 3` → `-q:v 2` (lower = better quality)
5. Enabled camera property dialogs for cam4, cam6, P50 for manual exposure inspection

**Expected Results:** 
- PanaCast 50 bitrate should increase from 32 Mbps to 60-80 Mbps
- Cam4/Cam6 may need exposure/scene adjustments via camera dialogs

**Files:** [tools/ffmpeg_multicap.py](tools/ffmpeg_multicap.py), [configs/ffmpeg_multicap.json](configs/ffmpeg_multicap.json), [docs/camera_quality_diagnostics.md](docs/camera_quality_diagnostics.md)

### Device Configuration Updates
- **Fixed:** PanaCast 50 device path corrections
  - PID: 3013 → 3015
  - Video index: 4 → 6  
  - Audio index: 5 → 6
  - Updated device alternative names to match detected hardware
- **Motivation:** "Configured cameras are missing" errors after reconnection/firmware change
- **Files:** [configs/ffmpeg_multicap.json](configs/ffmpeg_multicap.json)

### Testing & Verification
Run next recording session with:
```powershell
python tools/ffmpeg_multicap.py --config configs/ffmpeg_multicap.json --group-id test_quality --frame-log --record-lsl --stabilization-delay 2.0
```

Check resulting bitrates:
```powershell
$ff='c:\Users\meisa\.conda\envs\affectai\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'
& $ff -i "data\...\jabra_panacast_50_vid_video.mkv" 2>&1 | Select-String "bitrate:"
```

Target: PanaCast 50 bitrate >60000 kb/s (was 32434, should be ~70000)

2026-03-09 (fix)
- **session_orchestrator_gui: calibration videos now save to session video/calibration subfolder during active sessions** — When a session is running, clicking "Run Calibration" in the GUI now saves the 75-second calibration video to `<session_dir>/video/calibration/` instead of the standalone output directory, keeping calibration recordings alongside main session videos for easier organization and post-processing.
- **session_orchestrator_gui: AV start fallback when shared lock is not visible** — `tools/session_orchestrator_gui.py` now explains that both PCs must use the same shared `out_root` for lock-file visibility and, in `tools/session_orchestrator_gui_av.py`, allows an explicit operator-confirmed manual-sync override when Recording-PC lock cannot be detected (for split-filesystem setups).
- **session_orchestrator_gui: LSL monitor enabled on AV role for diagnostics** — AV role now shows the LSL stream panel (`LSL Streams (Diagnostic)`), supports manual/auto scans like Recording role, increases LSL resolve timeout to improve cross-host discovery, and broadens Stimuli stream classification to include legacy stream names (`participant_*`, `moderator`, `bigscreen`, `experiment`).

2026-03-09 (session orchestrator)
- **Removed EmotiBit UDP preflight** — Removed `Check EmotiBit UDP` button and supporting methods (`_run_emotibit_udp_preflight`, `_probe_emotibit_udp_activity`, `_load_expected_emotibit_sources`, `_emotibit_udp_preflight_bg`) from `tools/session_orchestrator_gui.py`. EmotiBit LSL Merger now relies on static participant mapping config rather than runtime UDP probing.
- **Azure blob configuration moved to end-of-session** — Moved Azure Blob URL and credentials fields from Configuration section to End of Session/Data Ops section in GUI, keeping them out of view during session setup but accessible for upload configuration at session conclusion.
- **Configuration panel show/hide** — Added `_toggle_config_visibility()` method to collapse Configuration frame during active session and re-expand it when session ends, reducing on-screen clutter during live recording.

2026-03-09 (feature)
- **display_server: Tobii post-processing calibration step before each task brief** — Added a "0. Tobii Calibration" button before the brief for tasks T1–T4 in the moderator console. Clicking it sends a full-screen aiming-point (SVG crosshair / red dot) to the big screen and all four tablets and emits a `tobii_calibration` LSL marker to every stream (moderator, P1–P4, bigscreen) for precise Tobii post-processing alignment. The moderator instructions for T1–T4 now include step 0 for this procedure; the calibration phase is marked untimed so the elapsed timer is unaffected.

2026-03-09 (feature)
- **emotibit_lsl_merger: subscribe Oscilloscope LSL → merged participant outlets** — New tool `tools/emotibit_lsl_merger.py` resolves the 76 per-signal LSL streams published by EmotiBit Oscilloscope (each carrying `source_id = MD-V7-XXXX`), groups them by hardware ID, maps to P1–P4 via `configs/emotibit_participants.json`, and re-publishes 4 merged 26-channel `Emotibit_P#_stream` outlets at 25 Hz. This replaces the UDP-relay path and makes participant identity unambiguous without requiring per-device IP routing. Run: `python tools/emotibit_lsl_merger.py --participant-map configs/emotibit_participants.json`.

2026-03-09 (fix)
- **emotibit startup diagnostics: explicit idle-UDP watchdog logs** — `src/affectai_capture/devices/emotibit.py` now logs a clear warning when the UDP listener is bound but no packets have arrived yet (first warning at ~10s, then every ~15s). When `by_source` mappings are configured, expected sender sources are printed at startup and repeated in idle warnings to make source/network mismatches obvious during session bring-up.
- **emotibit optional LSL fallback for unmapped sources** — Added `--allow-unmapped-lsl` to `python -m affectai_capture.devices.emotibit` (and passthrough in `tools/emotibit_one_command.py`). When enabled, EmotiBit packets are still published to LSL using source-keyed stream names (e.g., `Emotibit_src_127_0_0_1_stream`) even if participant mapping fails. This is intended for diagnostics/bridge mode; participant identity may be ambiguous.

2026-03-09 (feature)
- **emotibit UDP->LSL: source-based participant mapping for unified participant streams** — `src/affectai_capture/devices/emotibit.py` now supports participant mapping by source endpoint (IP/name) in addition to hardware-ID mapping. `--participant-map` accepts either the legacy flat schema (`{"P1": "MD-V7-..."}`) or a new schema with `participants` + optional `by_source` (e.g., `"192.168.10.201": "P1"`). This makes it easier to force stable `Emotibit_P1_stream`...`Emotibit_P4_stream` assignment when Oscilloscope/app-side naming is ambiguous. Added example config: `configs/emotibit_participants_by_source.example.json`.
- **session_orchestrator_gui (AV PC): one-click EmotiBit UDP activity preflight** — Added `Check EmotiBit UDP` button to pre-flight controls in `tools/session_orchestrator_gui.py` (AV role). It listens on UDP `:12346` and reports active sender count/packet activity (and expected/missing senders when `by_source` is provided in participant map). Standard `Run Pre-flight` on AV now also includes an EmotiBit UDP activity row.
- **emotibit one-command launcher for fixed setup** — Added `tools/emotibit_one_command.py` to run the practical workflow in a single command per session: validate fixed participant map (`participants` + optional `by_source`) and launch `python -m affectai_capture.devices.emotibit ...`. This formalizes the "configure Oscilloscope once, then run one Python command" approach without requiring 4 concurrent Oscilloscope windows.
2026-03-09 (session orchestrator)
- **Recording type field + hierarchical session folders** — GUI now includes recording_type dropdown (test/pilot/final) positioned at top of left panel that creates top-level folders; session structure is now `{out_root}/{recording_type}/sub-XX/ses-{session_id}/`
- **Session naming from schedule** — Session IDs now auto-generate from schedule as `{group_id}_{date}_{start_time}` (e.g., `grp-01_20260305_1300`)
- **Schedule TSV expanded** — `session_schedule.tsv` now includes `date`, `start_time`, `end_time` columns; created `tools/transform_raw_to_schedule.py` to generate schedule from `configs/session_raw_data.tsv` (includes all participants regardless of confirmation checkmark)
- **Enhanced seat assignment display** — Seat assignment panel now shows "Group: {id} | Date: {date} | Time: {start}-{end}" above the participant table
- **Session history tracking** — When LSL recording starts, session metadata (timestamp, recording_type, group_id, date, time, session_id, P1-P4 names, role, session_dir) is logged to `sessions/session_history.tsv` for tracking data collection progress and seat assignments
- **Cross-modality capture manifest + XDF start guard** — `session_orchestrator_gui.py` now writes `ses-<session_id>_capture_tracking.json` checkpoints (session init, ffmpeg start, Tobii SD start/stop, LabRecorder/XDF start, recording start/stop, session end) with machine-readable file snapshots for XDF + AV media + sync logs, and `Start LSL Recording` now waits for a live LabRecorder process before marking recording active.
- **Role-specific workflow hardening (AV vs Recording)** — AV start is now blocked until an existing lock confirms `recording-pc` is already active for the same group/session (prevents misaligned launches). Session steps were split so AV handles `ffmpeg_multicap` (and optional Tobii on-device only when explicitly enabled), while Recording handles `emotibit_lsl_merger`, `display_server`, `tobii_glasses_lsl_bridge`, and `LabRecorder`.
- **Role-focused UI simplification** — AV view now emphasizes calibration/feed/capture + AV upload/Tobii ingest and hides raw→BIDS and Recording-specific EmotiBit UDP checks; Recording view emphasizes LSL stream monitoring/recording and shows a tablet Wi-Fi URL banner (`http://<local-ip>:<port>/`) for stimulus access.

2026-03-08 (calibration)
- **AV GUI now enforces sync-safe calibration recording defaults** — `tools/session_orchestrator_gui_av.py` now uses an AV-specific subclass that forces `frame-log`, `record-lsl`, marker stream, `ffmpeg_clock@100Hz`, `ffmpeg_progress_` prefix, `stabilization-delay 2.0`, and `sequential-start-delay 0.3` at startup.
- **Calibration sync alignment + guidance update** — `calibrate_charuco.py calibrate` now attempts temporal pre-alignment from `ffmpeg_multicap_events.jsonl` (`capture_started` offsets) before MKV->MP4 trimming, and `docs/calibration_usage.md` now explicitly requires sync artifacts (`--frame-log`, `--record-lsl`) when recording calibration with raw `ffmpeg_multicap.py`.
- **`calibrate_charuco.py` recording protocol now mirrors production sync capture** — `record` now launches `ffmpeg_multicap.py` with `--group-id`, `--frame-log`, `--record-lsl`, `--stabilization-delay 2.0`, `--sequential-start-delay 0.3`, `--lsl-stream-name ffmpeg_clock`, `--lsl-rate 100`, `--lsl-prefixes ffmpeg_progress_`, and `--enable-markers`.
- **75s/15s dynamic-board cue workflow** — calibration recording defaults updated to 75 seconds with periodic beep cues every 15 seconds to move the dynamic board through new poses/regions.
- **Two-board support via auto board selection** — `detect`, `calibrate`, and `ground-plane` now accept `--board-type auto` and select between `5x3` and `7x5` based on observed ChArUco detections.
- **Config-driven board sizes** — calibration commands can now read board square sizes from JSON/YAML config (`--board-config`, defaults to `--config` for `record`), including `calibration.boards` and `fixed_charuco_board` schemas.
- **Windows camera shutdown hardening** — calibration recorders now stop ffmpeg capture trees via process-group `CTRL_BREAK_EVENT` with `taskkill /T /F` fallback to prevent cameras remaining active after script exit.
- **`calibrate_cameras.py` aligned with the same sync-safe recording command** — removed obsolete ffmpeg flags, added 75s + 15s cue-beep protocol, and made frame-log timestamp parsing robust to `unix_time_s`/`unix_time`/`host_time` variants.

2026-03-08 (sync+av)
- **Audio sync bug fix in `create_sync_test_video.py`**: Fixed mono audio delay syntax (was incorrectly using stereo `adelay=105|105`, now correctly uses mono `adelay=105` for DPA microphones)
- **Session orchestrator GUI: missing `--group-id` parameter**: Now properly passes group ID to ffmpeg_multicap for correct session directory naming (e.g., `group_A_20260308_151626` instead of timestamped-only folders)
- **Session orchestrator GUI: optimized ffmpeg_multicap defaults for multi-camera sync**:
  - LSL prefix: `ffmpeg_` → `ffmpeg_progress_` (matches actual LSL progress stream file naming: `ffmpeg_progress_<label>.jsonl`)
  - Stabilization delay: `3.0s` → `2.0s` (optimal for camera initialization, matches production usage)
  - Sequential start delay: `0s` → `0.3s` (improves USB bandwidth stability with 7+ simultaneous cameras)
- **Automatic sync diagnostics in `create_sync_test_video.py`**: Now writes `.sync_diagnostics.json` alongside output video with all timing data, video shifts, audio alignment, and complete ffmpeg command for retrospective debugging and validation
- **Enhanced audio sync logging**: Human-readable explanations of trim/pad operations (e.g., "Audio started 0.105s AFTER video ref, adding 0.105s silence before audio")
- **Documentation**: Added inline comments in session_orchestrator_gui.py explaining optimal settings and updated README.md with recommended ffmpeg_multicap command patterns

2026-03-08 (feature)
- **display_server: first-name-only participant display + personalized post-task participant labels** — `stimuli/display_server.py` now normalizes participant names to first-name only (ledger auto-load + moderator manual name entry) for tablet/bigscreen display. Post-task questionnaire payloads are now built per send so participant-referenced items (`*_p1`..`*_p4`) render as `P# (FirstName)` on tablets when names are available, while response keys/log markers remain anonymized and unchanged.
- **display_server: T1 familiarity now per participant (P1-P4)** — Replaced the single T1 `group_familiarity` item with four T1-only familiarity items (`familiarity_p1`..`familiarity_p4`). When names are available, each familiarity prompt asks directly using the participant first name on tablet display.
2026-03-08
- **desk marker size correction: 50mm in shared lab configs** — Updated `configs/desk_markers_large.yaml` desk marker geometry from 100mm to 50mm while keeping the same 6-marker edge layout, participant seating map, glasses settings (25mm markers), and 7-camera position metadata. Updated world `marker_map` corner coordinates and `table_marker_layout.marker_size_m` accordingly.
- **generate_aruco_marker_sheet: lab profile desk markers now 50mm** — Updated `tools/generate_aruco_marker_sheet.py` `lab_dual_board` preset to generate 50mm desk markers and retain 25mm glasses markers, matching `configs/desk_markers_large.yaml`.
- **dual-board config alignment: 6 edge markers + center-origin board + 7-camera metadata** — Updated `configs/desk_markers_large.yaml` to the current lab geometry: desk `1.80 x 0.80 x 0.75 m`, 50mm desk markers (IDs `0-5`) at 4 corners plus left/right edge centers, fixed 7x5 ChArUco board at `[0,0,0]` as world origin, glasses marker geometry (25mm markers, 140mm center spacing, 60mm camera-front-edge offset), participant seat mapping, and camera metadata including `cam7` (`PanaCast 50`) at front-center with `z=0.90 m`.
- **marker schema compatibility across validator/tracker tools** — `tools/validate_calibration_robust.py` and `tools/tobii_multicam_glasses_tracker.py` now accept both `world.marker_map` and `table_markers` marker schemas so one shared config can drive desk reprojection validation and glasses tracking. Added regression coverage in `tests/test_tobii_multicam_glasses_tracker.py` for `world.marker_map` parsing.
- **generate_aruco_marker_sheet: lab dual-board profile + geometry export** — Extended `tools/generate_aruco_marker_sheet.py` with `--profile lab_dual_board` to generate a custom marker set for the 180cm x 80cm desk + Tobii glasses workflow: 6 desk-edge markers at 50mm (4 corners + left/right center), 4 glasses pairs at 25mm, and exported machine-readable geometry (`lab_dual_board_layout.json`) including desk dimensions, fixed 7x5 board origin at table center, camera placements/heights, and participant seat mapping. Added tracker-ready config export (`tobii_multicam_glasses_tracker_lab.json`) compatible with `tools/tobii_multicam_glasses_tracker.py`. Added `--paper-size a3` support for this profile to emit print-ready A3 outputs (`table_markers_a3_page1.png`, `table_markers_a3_page2.png`, `glasses_markers_a3.png`) while preserving true marker dimensions.
- **validate_calibration_robust: desk marker validation step** — Added optional ArUco marker-based validation to `tools/validate_calibration_robust.py`. New `--marker-map` and `--videos-dir` arguments enable validation against known desk marker positions by detecting markers in recorded videos, computing reprojection error (detected vs. projected positions), and providing quantitative accuracy metrics. Validates using `table_marker_map.yaml` format (compatible with `online_calibration.py --export-table-marker-map`). Computes mean/max reprojection error per camera and overall, integrates into quality score (penalties for >2.5px, >5px, >10px error), and provides clear accept/reject recommendations (accept if <5px, reject if >10px). Options: `--max-frames` (default 100) and `--sample-stride` (default 10) control sampling. Output includes new "DESK MARKER VALIDATION" section in text report with per-camera detection counts and errors. Updated quality score to include "recommendation" field (Accept/Accept with caution/Reject). Requires PyYAML for marker map parsing.
- **validate_calibration_robust: Cam1-4 180deg correction in marker validation** — Desk-marker validation now applies 180deg frame correction by default for camera names matching `cam1,cam2,cam3,cam4` before ArUco detection, reflecting the current upside-down physical mounts. Added CLI override `--flipped-camera-patterns` (empty string disables) and report annotations (`[rot180]`) so corrected cameras are explicit in validation output.
- **camera orientation now centralized in config (`rotate_180`) for camera tools** — Added `"rotate_180": true` for Cam1-4 in `configs/ffmpeg_multicap.json` so upside-down mount metadata is persisted in config. `tools/online_calibration.py` and `tools/online_multicam_feed.py` now read and apply 180deg rotation from this field before detection/preview. `tools/validate_calibration_robust.py` now reads flipped-camera patterns from `--camera-config` (default `configs/ffmpeg_multicap.json`) and uses `--flipped-camera-patterns` only as an optional manual override.

2026-03-08 (ops)
- **ffmpeg multicap USB rebalance + recording-PC port map snapshot** — Updated `configs/ffmpeg_multicap.json` DirectShow `video_index` assignments to match current post-upgrade enumeration after adding USB 3 cards (stable `video_alt_name`/`audio_alt_name` identities unchanged). Added `docs/usb_port_map_recording_pc.md` with live controller-path mapping for all cameras and RME Fireface, plus load check summary (`2+2+2+1` cameras across controller paths, max 2 per host controller).
- **USB distribution one-command preflight helper** — Added `tools/check_usb_distribution.ps1` to map configured cameras to USB host controller keys on Windows, report per-controller camera counts, and fail with a nonzero exit code when cameras are missing or any controller exceeds the configured threshold (`-MaxCamerasPerController`, default 2).

2026-03-07 (feature)
- **session_orchestrator_gui: control-center stop hardening + cross-PC session-name lock + recording/status UX updates** — `tools/session_orchestrator_gui.py` now enforces a shared two-PC session lock file (`<out_root>/_session_locks/<date>_<group>.json`) so AV and Recording PCs must use the same session/group/seat mapping before launch. Stop/close now performs an immediate best-effort shutdown of all managed processes from the GUI control center (orchestrator children, surveillance feed, online calibration, Tobii on-device recording, and running Data Ops subprocesses). Recording panel now shows live LSL timer and current XDF size while recording. LSL stream monitor tree now has vertical/horizontal scrollbars. Azure upload destination can now be loaded from JSON credentials in GUI (`Blob cred JSON` field + Load button), with fallback use during upload if manual Azure URL is empty. Calibration panel adds `Import Saved Files` to copy saved calibration materials into session/output calibration folders and auto-select imported TOML when present.
- **session_orchestrator_gui: visible session-lock status line in control center** — Added a lock-status line under Session ID showing lock state, lock-file path, and active roles (`av-pc` / `recording-pc`), including explicit mismatch/release/failure feedback.
- **session_orchestrator_gui: compact lock text + full hover details + live refresh** — Lock status now displays a shortened lock-path string for readability, keeps full path/details in a hover tooltip, and auto-refreshes from disk every 5 s so role changes made on the other PC are reflected without restarting the GUI.
- **session_orchestrator_gui: lock freshness age indicator** — Lock status now includes `updated X ago` derived from the lock file timestamp and uses staleness-aware colouring so operators can quickly detect stale lock state.
- **session_orchestrator_gui: per-step Stop controls (with Redo) for managed processes** — Session Steps panel now includes `Stop` buttons alongside `Redo` for process-backed steps (e.g., FFmpeg/Stimuli/Tobii bridge/LabRecorder), allowing operators to close a specific running step without ending the full session.
- **session_orchestrator_gui: confirmation dialog for critical per-step Stop actions** — stopping critical steps (currently FFmpeg Multicap and Display Server) now requires explicit confirmation to reduce accidental capture/stimulus interruptions.
- **session_orchestrator_gui: Stop Session now explicitly stops recording before full shutdown** — pressing `Stop Session` now calls recording-stop logic first (including recording stop event write) and then shuts down stimuli and all remaining managed processes.
- **configs: add Azure blob credential JSON template** — Added `configs/azure_blob_credentials.example.json` for GUI-based blob destination loading without hardcoding credentials in code.

2026-03-07 (fix)
- **online_calibration: align camera labels with ffmpeg_multicap device identity mapping** — Online calibration now resolves OpenCV `video_index` values on Windows from the current DirectShow device catalog using `video_alt_name` (stable unique path) and `video_name` fallback, matching ffmpeg_multicap’s device-selection strategy. This prevents mislabeled recordings where camera labels drift from physical devices when DirectShow/OpenCV index ordering changes between runs. Startup logs now print per-camera resolved index mapping (`configured -> resolved`, resolution reason) and include resolved camera name in open logs.

2026-03-07 (fix)
- **online_calibration: fast-recording now uses post-hoc validation semantics, seeded export by default, and better recording-time metadata** — `--fast-recording` now skips live Charuco validation checks (which are intentionally bypassed in this mode) and reports `validation_skipped=fast_recording` instead of emitting false "0 detections" camera failures. TOML export now defaults to seeded intrinsics (`--export-init-focal` enabled by default; override with `--no-export-init-focal`) so calibration uses the robust duplicate-ID-filtered path that succeeds on Jabra sessions where non-seeded mode often reports no board points. Recording timing logs now write `pts_time` from wall-clock deltas instead of nominal frame index/FPS, and fast-recording video writer FPS is configurable via `--fast-recording-fps` (default 10.0) to reduce severe fast-playback artifacts when hardware drops below nominal camera FPS.

2026-03-07 (fix)
- **online_calibration: enforce configured camera list and recording-name consistency** — Online calibration now validates configured-vs-opened camera labels before capture and configured-vs-recorded video labels before calibration export. This prevents silent partial runs where one camera is missing (e.g., cam4) and avoids naming mismatches between config labels and recorded filenames. Label comparison normalizes `_video` and `_vid` variants. Default behavior now requires all configured cameras to be present; use `--allow-missing-cameras` only when intentionally running partial setups.

2026-03-07 (fix)
- **online_calibration: clarify "board visible but low detection" cases with marker-visibility diagnostics** — Added `marker_visible_frames` and `marker_visibility_rate` per camera in `online_calibration.json`, and emits a targeted validation hint when markers are frequently visible but strict Charuco interpolation remains low. This helps distinguish true detection failure from interpolation/coverage issues.

2026-03-07 (fix)
- **online_calibration: prevent Windows cp1252 decode crashes during export subprocess capture** — Added explicit `encoding="utf-8"` and `errors="replace"` to all `subprocess.run(..., text=True)` calls used by sync export, `calibrate_charuco calibrate`, and validation steps. This avoids `UnicodeDecodeError` crashes in background reader threads on Windows terminals while still preserving logs.

2026-03-07 (fix)
- **online_calibration: success beep now follows visible marker capture (not only strict Charuco interpolation)** — Added marker-based beep gating so operator feedback aligns with what is visible in the live preview. New CLI option `--success-beep-min-markers` (default `4`) marks a camera beep-eligible when enough filtered markers are visible, even if strict corner interpolation for validation is not yet successful. Success event payload now records both beep camera set and strict-detected camera set for debugging.

2026-03-07 (fix)
- **online_calibration: fix empty frame logs that broke sync export** — Fixed per-frame sync logging in `tools/online_calibration.py` so `<session>/frame_logs/*_frames.jsonl` is populated during recording. Root cause was using non-existent `args.fps` in the logging path and coupling log writes too tightly to video-write exception handling. Now frame logs use per-camera fps (`runtime.cfg.fps`), include both `unix_time_s` and `unix_time` anchors, use a dedicated `recorded_frame_number`, and emit explicit errors if logging fails. This restores synchronization-start estimation for `--export-use-sync` runs.

2026-03-07 (fix)
- **online_calibration: louder and more visible success beep feedback** — Increased audible feedback when 2+ cameras detect the ChArUco board. Enhanced `Beeper.success_capture()` with longer tone durations (120ms instead of 80ms), added third tone for more distinctive pattern, and increased duration between tones. Added logging that shows which cameras triggered the beep ("Beep trigger: N cameras detected (cam1, cam2...)") and debugging output for audio backend failures (winsound vs sounddevice). System bell fallback (\a) now prints every tone attempt if hardware audio fails. Users can now clearly hear when multi-camera capture is successful.

2026-03-07 (feature)
- **online_calibration: frame logging for video sync** — Added frame timing data capture during video recording for synchronization with the existing sync pipeline. When `--export-calibration` and `--export-session-dir` are used, frame logs (`<session>/frame_logs/{camera}_frames.jsonl`) are now written alongside videos, containing frame number, presentation timestamp, and multiple timing references (unix_time_s, unix_time_ns, monotonic_ns) matching the format used by ffmpeg_multicap.py. Frame log files are automatically read by the sync preprocessing step (_prepare_synced_videos_for_export) to align multi-camera videos before calibration. Ensures synchronization data is captured during online calibration just like full recording sessions, enabling reliable multi-camera TOML generation.

2026-03-07 (fix)
- **online_calibration: skip sync gracefully when no sync data sources exist** — Online calibration records videos but has no frame logs/LSL/events data (unlike full capture sessions). When `--export-use-sync` is enabled (default), code previously tried to sync non-existent data and crashed with "Could not compute synchronization starts" error. Now: (1) check if any sync data sources exist (frame_logs/, lsl/, ffmpeg_multicap_events.jsonl), (2) if none exist, skip sync preprocessing gracefully and use raw recorded videos directly for calibration, (3) mark sync as skipped with reason `no_sync_data`. Online calibration now completes end-to-end without requiring external sync data.

2026-03-07 (fix)
- **online_calibration: fix video file detection for batch calibration** — Fixed glob patterns in `_prepare_synced_videos_for_export()` that prevent batch calibration from finding recorded videos. Code was looking for `*_video.mkv` but online_calibration records files as `*_vid.mkv` (matching camera config labels). Now checks both patterns: `*_vid{ext}` first (online recording), then `*_video{ext}` (legacy batch naming) across all video extensions (.mkv, .mp4, .avi, .mov). This enables the complete video recording + batch calibration workflow: live detection → video recording → sync export → TOML generation.

2026-03-07 (feature)
- **online_calibration: video recording + batch calibration** — Added video recording capability to `tools/online_calibration.py`. When `--export-calibration` and `--export-session-dir` are combined, videos are automatically recorded from all cameras during the live ChArUco detection phase and saved to `<session>/video/`. These recorded videos are then used by `calibrate_charuco.py` to estimate camera intrinsics/extrinsics (TOML file), replacing the previous behavior of skipping calibration. Updated `CameraRuntime` dataclass to track video writer and output path. New `_init_video_writer()` function creates MP4/MKV writers for each camera with fallback codec selection (MJPG preferred for compatibility). Frames are written to video during real-time detection with minimal performance impact. This enables operators to: (1) run live ChArUco detection for real-time feedback, (2) simultaneously record calibration video, (3) generate TOML calibration from the recorded video all in one workflow.

2026-03-07 (fix)
- **online_calibration: skip sync/calibration export gracefully when no videos exist** — Online calibration captures from live cameras (no video recording), so `--export-calibration` would fail trying to sync or calibrate non-existent videos. Now: (1) `--export-session-dir` and `<session>/video/` are created automatically if missing, (2) sync preprocessing is skipped if no video files found, (3) `calibrate_charuco` step is skipped if no videos, (4) online calibration completes successfully with detection metrics in artifacts JSON (`calibrate_skipped: true`, `sync_skipped: true` tags). Fixes "No video files found" and "Videos directory does not exist" errors that occurred when using `--export-calibration` with live capture mode.

2026-03-07 (feature)
- **Robust calibration validation without FreeMoCap dependency** — Added `tools/validate_calibration_robust.py`, a standalone validator that generates comprehensive calibration quality reports without requiring FreeMoCap. Provides: intrinsics/extrinsics summary, focal length validation against camera specs (with regex pattern matching from `configs/camera_specs.json`), inter-camera geometry analysis (distances, warnings for cameras too close <0.5m or too far >5m), distortion coefficient analysis with non-monotonic detection, automated quality scoring (0-100), severity-graded warnings (error/warning), and actionable recommendations. `online_calibration.py` now uses this as fallback when FreeMoCap validation fails (import errors, missing dependencies), preventing validation reports from being polluted with stack traces. Returns proper exit codes based on quality score (1 if <40, 0 otherwise).

2026-03-07 (feature)
- **3D calibration geometry visualizer** — Added `tools/visualize_calibration.py` to create interactive or saved 3D plots of camera rig geometry. Shows: camera positions (red spheres), orientation arrows (blue, along camera Z-axis), inter-camera distances (color-coded: red <0.5m, gray 0.5-5m, orange >5m), coordinate frame at origin (RGB for XYZ), and focal length summary with ratios vs expected values. Useful for spotting geometric issues (cameras too close, bad triangulation baselines) and understanding extrinsic calibration layout. Usage: `python tools/visualize_calibration.py --toml <path> [--output <image.png>]`.

2026-03-07 (feature)
- **tobii_glasses_lsl_bridge: regular channels per glasses + irregular packets in `evetns_tobii`** — `tools/tobii_glasses_lsl_bridge.py` now keeps regular sampled channels in each participant stream (`Tobii_P1_stream`..`Tobii_P4_stream`): gaze/pupil (+ optional 3D and optional IMU). Irregular samples (`event`, `sync_port`) are routed to a separate global stream `evetns_tobii` (string fields with device_id/type/key-value payload). Added `--split-streams` to preserve legacy separate outlets (`Tobii_*_Event`, `Tobii_*_Imu`, `Tobii_*_SyncPort`) when needed for backward compatibility.

2026-03-07 (fix)
- **tobii_glasses_lsl_bridge: set nominal sample rate for regular Tobii streams** — `Tobii_P*_stream` now publishes LSL metadata with configurable nominal rate (`--nominal-srate`, default `50.0`) instead of `0 Hz`, so stream monitors no longer report these regular channels as "irregular". Irregular stream `evetns_tobii` remains `0 Hz` by design.

2026-03-07 (feature)
- **session_orchestrator_gui: EmotiBit participant map path in config + pre-flight** — Added an explicit `EmotiBit map` path field to GUI configuration (default `configs/emotibit_participants.json`) so participant mapping can be changed from GUI without editing code. Pre-flight checks now validate presence of the configured EmotiBit map file.

2026-03-07 (feature)
- **configurable EmotiBit participant mapping file** — `src/affectai_capture/devices/emotibit.py` now supports `--participant-map` for participant→hardware-ID mapping, and loads `configs/emotibit_participants.json` by default when present (fallback to built-in defaults if the default file is missing). This makes `Emotibit_P1_stream`…`Emotibit_P4_stream` assignment editable without code changes.

2026-03-07 (feature)
- **participant-level LSL stream unification for Tobii + EmotiBit (BIDS-aligned)** — Updated `tools/tobii_glasses_lsl_bridge.py` to publish one unified eye-tracking stream per participant (`Tobii_P1_stream`..`Tobii_P4_stream`) with gaze+pupil channels in a single outlet (plus optional 3D channels when `--with-3d` is enabled). Updated `src/affectai_capture/devices/emotibit.py` to publish one unified stream per participant (`Emotibit_P1_stream`..`Emotibit_P4_stream`) with all EmotiBit channels represented in one vector stream. Added default participant hardware-ID map for EmotiBit (`MD-V7-*` IDs), updated default Tobii config IDs/serial mapping in `configs/tobii_glasses_streams.yaml`, and expanded downstream compatibility in `tools/raw_to_bids.py`, `tools/session_orchestrator_gui.py`, and `tools/lsl_stream_viewer.py` so both legacy and new stream names are recognized during monitoring and raw→BIDS extraction.

2026-03-07 (feature)
- **session_orchestrator_gui: cross-role raw data ops (upload/ingest/convert)** — Added a new **Data Ops** panel visible on both AV PC and Recording PC with three actions: **Upload Raw (Azure)**, **Ingest Tobii Download**, and **Raw → BIDS**. Upload runs `tools/upload_raw_data.py` (via `azcopy`) and writes a sync-artifact manifest under `sourcedata/sync/` so timing anchors (`events.tsv`, XDF, ffmpeg LSL/frame/progress logs, Tobii LSL NDJSON) are tracked with each upload. Ingest runs `tools/ingest_tobii_downloads.py` to place manually downloaded Tobii on-device files into `sourcedata/tobii_device/<device>/` with an index JSON. Conversion runs `tools/raw_to_bids.py` to generate BIDS-oriented outputs from AV + Recording + Tobii raw sources and optional XDF extraction (if `pyxdf` is available).

2026-03-07 (fix)
- **stimuli: VAD timing improvements + T4 display sizing fixes + group familiarity question** — Updated T4 VAD schedule to only send 2 prompts: one simultaneous at outcome reveal (90s) and one towards end of discussion (270s), eliminating mid-discussion prompt. Reduced font sizes in T4 shared instructions table (font-size: 0.92em, padding: 7px 10px) and outcome reveal screens (bigscreen title: 34px, main text: 22px; tablet: 20px title, reduced padding) to prevent text cutoff. Added "How well do you know the other participants?" question (group_familiarity) as first item in T1 post-task questionnaire to assess group familiarity (1=strangers to 7=close friends/colleagues).

2026-03-07 (feature)
- **display_server: auto-load participant names from registration ledger** — When `--out-root` is provided, display_server now automatically loads participant names from `.private/registration_ledger.jsonl` for the current session. Names are displayed on tablets ("P1 — Alice") and bigscreen ("P1: Alice • P2: Bob • P3: Charlie • P4: Diana") but are NEVER sent to LSL markers or event logs (only anonymous P1-P4 IDs are logged). Session orchestrator now passes `--out-root` to display_server, enabling automatic name loading when sessions are started via the orchestrator GUI or CLI. Names can still be manually set/updated via the moderator console.

2026-03-07 (feature)
- **display_server: participant names for display only (never logged to LSL/events)** — Added participant name input fields to moderator console and display of names on tablets and bigscreen. Names are stored in server memory and shown as "P1 — Alice", "P2 — Bob", etc. on device headers/screens. These names are NEVER sent to LSL markers or events.tsv (only P1-P4 anonymous IDs are logged). New endpoints: `GET /get_participant_names`, `POST /set_participant_names`. Global state: `PARTICIPANT_NAMES` dict protected by `PARTICIPANT_NAMES_LOCK`. Event log only records `set_participant_names` action with count, not actual names.

2026-03-06 (fix)
- **online_calibration: force UTF-8 in export subprocesses on Windows** — `--export-calibration` now launches `calibrate_charuco.py`/`validate` with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` to avoid `UnicodeEncodeError` crashes from non-ASCII FreeMoCap log output on cp1252 terminals.

2026-03-06 (feature)
- **online_calibration: export table marker map for coordinate framing** — Added optional `--export-table-marker-map` step to write `table_marker_map.yaml` using table dimensions and marker IDs, so table ArUco corners + center can define the world frame immediately after `--export-calibration`.

2026-03-06 (feature)
- **online_multicam_feed: live ArUco + zone-based pose + 3D preview** — Added `tools/online_multicam_feed.py` to show per-camera feeds with ArUco overlays, run MediaPipe pose per desk zone for multi-person tracking, and triangulate a live 3D preview from calibration TOML. Supports two modes: `--mode live` (read from DirectShow cameras, default) and `--mode surveillance` (read from growing MKV files written by `ffmpeg_multicap.py` without blocking camera access). Fast/low-quality surveillance is enabled via `--fast` (skip frames, downscale preview, use lighter MediaPipe model). Example zone config: `configs/desk_zones.json`. GUI integration: session orchestrator GUI now has optional surveillance feed control with enable checkbox, status indicator, and auto-start when recording begins.

2026-03-06 (feature)
- **session_orchestrator_gui: AV/Recording variants + calibration & feed controls** — Added role-locked GUI entrypoints (`tools/session_orchestrator_gui_av.py`, `tools/session_orchestrator_gui_recording.py`), calibration controls (load latest + run online calibration), and feed options (surveillance vs test, full CLI args exposed).

2026-03-06 (feature)
- **session_orchestrator_gui: FFmpeg options + recording PC action buttons** — Added FFmpeg CLI options panel (LSL name/rate/prefixes, frame log, markers, delays, zoom/mux/dialog toggles) and Recording PC quick actions to start stimuli, Tobii LSL, and LSL recording.

2026-03-06 (fix)
- **online_calibration: fast sync-export clipping to avoid long re-encode stalls** — Sync preprocessing for `--export-calibration` now trims each synchronized camera video to a short clip by default (`--export-sync-clip-seconds` defaults to `--duration`) instead of re-encoding full session-length files. This prevents long waits/interrupts on 30+ minute sessions while preserving alignment for calibration windows; use `--export-sync-clip-seconds 0` to keep full videos.

2026-03-06 (fix)
- **online_calibration: robust ffmpeg encoder fallback for sync-trim export** — Fixed `--export-calibration` failures on FFmpeg builds without `libx264` (common on Windows/conda-forge). Sync preprocessing in `tools/online_calibration.py` now detects available encoders and uses `libx264` when present, otherwise falls back to `h264_mf` and then `mjpeg`, preventing `Unknown encoder 'libx264'` export errors.

2026-03-06 (fix)
- **online_calibration: sync-aware TOML export for non-hardware-synced cameras** — `--export-calibration` now synchronizes calibration videos before estimating intrinsics/extrinsics (default enabled). Export uses existing sync signals with fallback order frame logs -> LSL progress -> ffmpeg events, trims aligned clips into `<session>/video/_synced_for_calibration/`, and runs `calibrate_charuco` on the synchronized set. Added flags: `--no-export-sync`, `--export-sync-source`, `--export-sync-align-to`, `--export-sync-lsl-prefix`, `--export-sync-frame-log-samples`.

2026-03-06 (fix)
- **online_calibration: board mismatch recovery + live detection preview** — Updated `tools/online_calibration.py` to support `--board-mode auto` (tries both `5x3` and `7x5` ChArUco board layouts and locks to the detected one), plus `--show-feed` live camera grid overlays with marker/corner counts and board status per camera. This resolves zero-detection runs caused by board-size mismatch and gives operators direct visual feedback during online calibration.
- **online_calibration: fast default validation to prevent apparent hangs** — Reprojection validation is now optional (`--enable-reprojection-validation`) because per-camera `calibrateCameraCharuco` over many cameras can add substantial post-capture runtime and look like the script is stuck. Added phase timing logs and timing metrics in `online_calibration.json` (`open_seconds`, `capture_seconds`, `validation_seconds`, `total_seconds`).
- **online_calibration: export session TOML + validation report** — Added optional `--export-calibration` step that invokes `tools/calibrate_charuco.py` after the online run and writes session-style artifacts under `<session>/video/`, including `video_camera_calibration.toml`, `video_camera_calibration_validation_report.txt`, `video_camera_calibration_calibrate.log`, and `video_camera_calibration_report.json`.

2026-03-06 (feature)
- **Online camera calibration with real-time ChArUco detection** — Added `tools/online_calibration.py`, a new live calibration tool that reads camera layout from `configs/ffmpeg_multicap.json`, tracks a printed ChArUco board across multiple cameras, and plays operator feedback beeps: success beep when board is detected in N cameras, distinct completion beep at end, and error beep on failed validation. Outputs `online_calibration.json` (per-camera metrics + validation errors + reprojection error) and `online_detections.jsonl` (frame-level detections). Returns exit code 1 if validation fails.

2026-03-06 (fix)
- **ffmpeg_multicap: fail-fast preflight for DirectShow `(none)` cameras** — Added a Windows preflight check that maps each configured `video_alt_name` to DirectShow role state (`video|audio|none`) and aborts early if any configured camera is `(none)` or missing. This replaces long partial runs that only fail after launch with `Error opening input: I/O error`.
- **ffmpeg_multicap + ffmpeg_multicap.json: stop forcing MJPEG by default** — Changed `DeviceConfig.input_video_codec` default to `None` and set all video entries in `configs/ffmpeg_multicap.json` to `"input_video_codec": null`. DirectShow now negotiates the correct per-camera format (e.g., YUYV vs MJPEG) instead of failing on a hardcoded codec.

2026-03-05 (fix)
- **lock_exposure: revert MSMF-first back to DSHOW-first** — MSMF backend blocked ALL 7 cameras (vs DSHOW which allowed pid_302a cameras cam3+cam4 to work). Reverted `_open_capture()` to try `[CAP_DSHOW, CAP_MSMF, CAP_ANY]`. Moot for the primary pipeline (`camera_setup_script` is disabled in `configs/ffmpeg_multicap.json`) but correct for standalone use.
- **confirmed fix: removing camera_setup_script resolves all I/O errors** — Session `ses-20260202_test_165506` confirms all 7 video ffmpeg processes reached `capture_started` with zero I/O errors after `camera_setup_script` was removed from the config. Root cause confirmed as Windows COM DirectShow exclusive handle held by OpenCV even after `cap.release()`. `docs/known_issues.md`, `docs/jabra_recording_checklist.md`, and `docs/llm/context_snapshot.md` updated accordingly.

2026-03-05 (fix)
- **ffmpeg_multicap.json: disable camera_setup_script (exposure lock)** — Despite switching to `cv2.CAP_MSMF` and adding `del cap; gc.collect()`, all 7 video captures still failed with `Error opening input: I/O error` immediately after lock_exposure released. Root cause is Windows driver-level exclusive access that outlasts any COM teardown technique. Removed `camera_setup_script` and `camera_setup_args` from all 7 video entries in `configs/ffmpeg_multicap.json`; cameras now open directly by ffmpeg with default driver settings. Exposure/white-balance locking can be re-evaluated via a separate non-overlapping pre-roll step if needed.

2026-03-05 (fix)
- **lock_exposure: use MSMF backend to avoid blocking ffmpeg dshow** — `cv2.CAP_DSHOW` acquires an exclusive DirectShow filter graph that Windows COM keeps alive for several seconds after `cap.release()`, preventing ffmpeg from opening the same device. Changed `_open_capture()` to prefer `cv2.CAP_MSMF` (Windows Media Foundation), which uses a separate handle path and releases without blocking dshow. Also added explicit `del cap; gc.collect()` after `cap.release()` to force COM object teardown.
- **ffmpeg_multicap: fix UnicodeEncodeError crash on Windows** — emoji characters in `print()` calls (`🔍`, `✅`, `⏳`, etc.) raised `UnicodeEncodeError: charmap codec can't encode` on Windows CP1252 consoles. Replaced all 9 emoji with plain ASCII equivalents and added `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` guard at startup.

2026-03-05 (fix)
- **ffmpeg_multicap: sequential camera setup eliminates DirectShow race** — Extracted `run_camera_setup()` from `start()` and moved it to a sequential pre-pass in `start_all()`. All cameras are set up one at a time (lock_exposure → OpenCV release → 0.5 s cooldown) before any ffmpeg process launches, preventing DirectShow handle contention that caused I/O errors.
- **ffmpeg_multicap.json: merged config with current PNP paths** — Ran `--update` to detect 7 currently connected cameras (3 new PNP paths replaced 3 stale ones), then merged back custom settings: `camera_setup_script`, `input_video_codec: "mjpeg"`, `mux_audio: false`, `force_wallclock_timestamps`, `show_camera_dialog: false`, and DPA `audio_pan` stereo→mono splitting (dpa_mic9–12).

2026-03-05 (fix)
- **lock_exposure: resolve PNP device paths to integer indices** — OpenCV `VideoCapture` cannot open cameras by DirectShow `@device_pnp_…` alt-name paths. Added `_resolve_pnp_to_index()` which runs `ffmpeg -list_devices` to enumerate video devices, matches the PNP alt-name, and returns the corresponding integer index so OpenCV can open the camera and set VideoProcAmp properties.

2026-03-05 (fix)
- **ffmpeg_multicap: fix camera setup script on Windows + disable blocking dialog** — `camera_setup_script` (e.g. `tools/lock_exposure.py`) was spawned without the Python interpreter, causing `[WinError 193]` on Windows; now prepends `sys.executable` when the script ends with `.py`. Also set `show_camera_dialog: false` for all 7 cameras in `configs/ffmpeg_multicap.json` — the blocking DirectShow dialog cannot work when cameras launch in parallel; `lock_exposure.py` sets properties programmatically instead.

2026-03-05
- **Session orchestrator GUI: step-based launch with status, redo, and recording control** — Replaced monolithic Start/Stop Session with a per-step panel. Each step (Init BIDS, FFmpeg, Tobii, Display Server, LabRecorder — role-dependent) shows a live status dot (pending/running/ok/error) and a Redo button to restart that step individually without tearing down the whole session. New "Start Recording (LSL)" button enables only after all steps are green; pressing it writes a `recording_start` event to `events.tsv`. "Stop Recording" writes `recording_stop`. Process health is monitored per-step, and a failed process automatically turns its dot red.
- **ffmpeg_multicap: timestamped session directories** — session directory name now gets an `_HHMMSS` suffix automatically (e.g. `ses-20260202_test_143022`) so re-runs never overwrite previous recordings.
2026-03-05 (cleanup)
- **Stimuli(`task_content.py` + `display_server.py`): removed all display-dead objects and fields** — purged content that is defined but never rendered on tablets or bigscreen:
  - `T1_SHARED_BRIEF` removed: no moderator button triggers phase `shared_brief` for T1
  - `T1_DISCUSSION_SELECTION` removed: pushed then immediately overridden by custom inline HTML; original dict never reaches a display surface
  - `T2_SHARED_BRIEF["title"]` + `["instructions"]` removed: T2 `shared_brief` push is overridden by the topics/formats reference board; only `topics` and `formats` keys are ever read
  - `T2_NEGOTIATION` removed: no moderator button triggers phase `negotiation` for T2
  - `T3_SHARED_INSTRUCTIONS["phases"]` removed: no JS renderer reads the `phases` key
  - `T3_RANKING_FORM` removed: phase `ranking_form` was removed from T3 flow (see entry 2026-03-05 ranking-form removal)
  - `T4_EXAMPLE_BRIEF` removed: no moderator button triggers phase `example_brief` for T4
  - `T4_OUTCOME_TEMPLATE` removed: T4 outcome is handled by `/t4_outcome` endpoint; phase `outcome_reveal` is never triggered via the normal push flow
  - Corresponding dead imports and routing branches removed from `display_server.py`; `outcome_reveal`/`example_brief`/`negotiation` removed from push-handler phase dispatch list
  - No behaviour change; all live display paths are unaffected.

2026-03-05 (fix)
- **Tobii LSL bridge: `os.add_dll_directory` requires absolute path:** `_load_tobii_sdk` in `tools/tobii_glasses_lsl_bridge.py` now calls `dll_path.resolve()` before deriving `sdk_abs_dir` so `os.add_dll_directory`, `sys.path`, `clr.AddReference`, and the `System.Reactive.dll` candidate path all use the resolved absolute path. Previously, passing a relative path (e.g. `tools\vendor\tobii_sdk\net472`) caused `OSError: [WinError 87] The parameter is incorrect` on Windows, preventing the bridge from loading the SDK when launched from the repo root with a relative `--sdk-dll` argument.

2026-03-05 (fix)
- **Stimuli(T1/T2/T3/T4): instruction clarity pass + T4 example in brief:**
  - T1: untimed brief rewritten as moderator-read intro; adds 1:15 silent-reading clarification, no-memorisation note, no-showing-tablets rule, and rephrase rule. Discussion and candidate-selection bigscreen text updated to match.
  - T2: shared brief rewritten to explain two-issue negotiation (topic + format); role card described as having priority/preferences; advocacy requirement and no-showing-tablets rule added. Bigscreen reference board updated accordingly.
  - T3: all instructions now specify "concrete social-event idea" rather than generic labels. "Theme + Activity" phrasing removed from four locations (shared instructions, idea-generation sheet, ranking form, group-selection form) and from the moderator script. T3 bigscreen waiting screens (`idea_generation`, `group_selection`) also updated to remove stale "theme + signature activity" wording.
  - T4: equal-contribution worked-example table moved into the untimed brief (`T4_SHARED_INSTRUCTIONS`) replacing the formula block. Separate "Examples (Untimed)" moderator button removed; remaining steps renumbered 1→4.

2026-03-05 (fix)
- **Stimuli(T3 ideas board): layout polish** — title 28 px, idea text 14 px, yellow brief banner removed from bigscreen; tablets now receive a brief reminder during the discussion phase ("You are selecting the best idea for a GN-wide social event").
- **Stimuli(T3 ideas board): font sizes reduced slightly** to fit the 7-minute discussion screen without overflow: title 20 px, participant headings 13 px, idea text 12 px, banner 13 px (was 24 / 15 / 13 / 14 px — slightly too large on 1080p display).

2026-03-05
- **Stimuli(timing): T3/T4 phase durations + VAD re-alignment:** T4 outcome reveal shortened 120 s → 60 s (bigscreen + tablet payloads); T4 discussion extended 120 s → 180 s (HTML text updated to "3 minutes") — button labels updated to "1 min" and "3 min". T3 idea-generation shortened 180 s → 150 s (instructions text updated to "2 minutes 30 seconds", button label updated to "2 min 30 sec"); T3 ranking/discussion extended 240 s → 420 s. VAD schedule re-aligned: T3 probes `[240,390,540]` → `[210,360,510]` (halfpoint 360 = 150+210; idea-gen still skipped); T4 probes `[75*,195,270]` → `[90*,165,240]` (outcome-reveal centre 90 s, two probes in 180-s discussion). Moderator console step 2 for T3 updated to reference "2 min 30 sec".

- **Session orchestrator GUI (`tools/session_orchestrator_gui.py`):** Tkinter dark-mode GUI for the session orchestrator. Load schedule TSV, browse groups, view randomised P1–P4 seat assignments, configure all paths (ffmpeg, Tobii, LabRecorder, output root, display host/port), select PC role, and start/stop sessions — all from one window. Live process health indicators (green/red dots per child PID), real-time scrolling log with colour-coded levels, elapsed-time counter, and background-thread execution so the UI never freezes. **Role-specific config fields:** switching between Recording PC and AV PC shows only the relevant options (AV PC → FFmpeg config; Recording PC → LabRecorder, display host/port). **Live LSL stream monitor:** auto-scans the network every 5 s (or manual Scan Now), displays all discovered LSL streams in a treeview (name, type, channels, rate, hostname) with green status dots and device-type summary (Tobii:N | EmotiBit:N | Vicon:N | ffmpeg:N | Stimuli:N) — works before and during sessions. **Auto-run numbering:** session ID auto-generates as `YYYYMMDD_{group}_run{NN}` with auto-incrementing run number to prevent overwriting previous recordings. **Overwrite guard:** warns before starting if session directory already exists. **Pre-flight device check:** dedicated panel probes Tobii glasses reachability (TCP :80 per IP), enumerates DirectShow cameras (AV PC), verifies LabRecorder/Tobii SDK/FFmpeg config existence, checks display port availability (Recording PC). Prominent log banner on start shows SESSION ID / GROUP / ROLE / OUTPUT DIR with reminder to use same session ID on other PC. Default paths pre-filled: Tobii SDK → `tools/vendor/tobii_sdk/net472/G3SDK.dll`, LabRecorder → `tools/LabRecorder/LabRecorderCLI.exe`. Uses only stdlib tkinter — no new dependencies.
- **Session orchestrator (`tools/session_orchestrator.py`):** New automation tool that orchestrates full data-collection sessions across two PCs. Reads a participant schedule TSV (`configs/session_schedule.example.tsv`), randomly assigns names to P1–P4 seats (deterministic seed from group_id), creates BIDS session folders + registration, and launches role-specific process trees. AV PC role starts `ffmpeg_multicap.py` (video+audio+LSL) and Tobii on-device recording via G3SDK `Recorder.Start()`. Recording PC role starts `display_server.py` (stimuli+LSL), `tobii_glasses_lsl_bridge.py` (gaze→LSL for sync), and LabRecorder (XDF). Process supervisor monitors child health and handles graceful Ctrl-C shutdown + session summary. No new dependencies.

2026-03-04 (fix)
- **Stimuli(VAD schedule): phase-aligned probes, ≥3 per task:** Replaced periodic random-interval VAD timer with `_VAD_TASK_SCHEDULE` — each task has a fixed list of probe times (seconds from timer start) placed at interactive-phase centres: T0 [60,150,240], T1 [60,210,360] (discussion phase), T2 [60,240,400] (negotiation), T3 [240,390,540] (skips 3-min silent idea generation), T4 [75*,195,270] (outcome reveal simultaneous + 2 staggered in discussion). Per-tablet stagger (0–20 s random offset) is preserved; T4 first probe is simultaneous for all tablets. `_VAD_MIN_INTERVAL_S` lowered from 150 s → 60 s to match tighter schedules. ±15 s jitter per probe.
- **Tobii direct SDK -> LSL bridge boot reliability on Windows:** `tools/tobii_glasses_lsl_bridge.py` now registers an `AppDomain.AssemblyResolve` handler and prepends the SDK folder to `sys.path` before loading `G3SDK.dll`, so transitive .NET assemblies are resolved deterministically from the same SDK directory. This removes startup failures when using local SDK bundles instead of a global Tobii installation and keeps Tobii→LSL launch reproducible from repo-local paths.
- **Tobii bridge dropout hardening (offline devices no longer kill bridge):** `tools/tobii_glasses_lsl_bridge.py` now probes configured IP devices on `:80` before SDK subscription and skips unreachable units instead of attempting to subscribe and crashing later in background SDK tasks. Device initialization is wrapped per-device so failures are isolated; the bridge continues publishing streams for reachable glasses instead of exiting all streams.

2026-03-04 (fix)
- **Stimuli(T4): formula parentheses, token=money language, outcome reveal clarity:**
  - Formula rewritten throughout with explicit parentheses: `Final tokens = (10 − your contribution) + ((1.5 × group total) / 4)`
  - "Tokens = money" framing added to shared instructions, example brief, and contribution form
  - `T4_EXAMPLE_BRIEF` reduced to a single worked example (equal-contribution case only — Case B free-rider removed); layout now 2-column
  - Bigscreen outcome reveal now explicitly states *"Your final total = tokens you kept + pool share"*
  - Tablet outcome reveal rows renamed for clarity: "Tokens you kept (10 − X)" and pool calculation shown step-by-step; counterfactual now uses correct sign variable
  - Discussion phase individual rows also use named `kept_p` variable

2026-03-05
- **Tobii LSL bridge: fix pythonnet Subscribe error:** Replaced all 4 `IObservableExtensions.Subscribe(observable, action)` calls in `tools/tobii_glasses_lsl_bridge.py` with a new `_subscribe_signal()` helper that uses `System.Reactive.Observer.Create[T](action)` to build a concrete `IObserver<T>` and passes it directly to `IObservable<T>.Subscribe()`. Pythonnet cannot resolve C# generic extension methods; this bypasses the issue. Also loads `System.Reactive.dll` (transitive G3SDK dependency) during SDK init.
2026-03-03 (fix)
- **Stimuli: remove non-VAD notifications; extend VAD cadence to 3–4 min:** Removed `showNotification` calls from `showPostblockPanel()` and form submit success/fail — only VAD-related notifications remain. `_VAD_MIN_INTERVAL_S` raised from 60 → 150 s; VAD loop sleep changed from 120 s (T4) / uniform(120,180) (other tasks) to 180 s / uniform(180,240) respectively.
- **Stimuli(VAD timing/finish flow):** Periodic VAD no longer fires immediately when the timer starts; it now waits an initial buffer (`_VAD_INITIAL_DELAY_S=90s`) before the first prompt so participants can settle into the task. `stop_vad_timer()` now clears timer state and pending staggered sends abort if stop is triggered. Server-side safeguards now stop VAD on task `finish` and also when `/send_postblock` is called, preventing new timed VAD prompts after the post-task questionnaire. Manual “all tablets” VAD sends now use staggered dispatch (non-simultaneous delivery).

2026-03-03 (feat)
- **Stimuli: revised task timings + T1 candidate-selection form:**
  - T1 silent reading 60 s → 75 s; instructions text updated to "1 minute 15 seconds"
  - T1 discussion 600 s → 420 s (7 min); instructions updated to mention 7-min window and last-minute form
  - New `T1_CANDIDATE_SELECTION_FORM` (60 s, select A/B/C + rationale textarea) pushed to all tablets + bigscreen at the `candidate_selection` phase; `get_task_content` wired; moderator phase button added
  - T3 idea generation 120 s → 180 s (3 min)
  - T3 discussion bigscreen timer 240 s → 420 s (7 min)
  - T3 group selection 120 s → 60 s
  - T4 contribution form 75 s → 60 s; bigscreen timer updated to match
  - Moderator phase button labels updated throughout to reflect new durations

 Three improvements to keep key information accessible during active work phases. (1) T1 `discussion_selection`: tablets now receive their own evidence card again (instead of a generic "look at the screen" transition) so each participant can refer to their private card throughout the entire discussion. (2) T2 `role_card` and `settlement_form`: bigscreen now shows a permanent two-column reference board listing all 4 available workshop topics and 4 formats, with a timer during negotiation and a "complete Settlement Form" reminder during settlement — replaces the plain text T2_NEGOTIATION content. (3) T3 all bigscreen phases (`idea_generation`, `show_ideas_discussion`, `group_selection`): a yellow brief banner is shown on the bigscreen throughout — "GN-wide social event — a clear theme + one signature activity" — so participants always know what they are generating ideas for.

- **Stimuli: remove standalone Post-Task Questionnaire button from moderator:** The "Send Post-Task Questionnaire" button in the moderator console has been removed since the questionnaire is already sent automatically when Finish Task is clicked. An informational note replaces it.

- **Stimuli: VAD timer auto-start/stop:** VAD timer now automatically starts when the designated phase is pushed (T0: Free Talk; T1: Discussion+Selection; T2: Role Cards; T3: Idea Generation; T4: Contribution) and automatically stops when Finish Task is clicked.

- **Stimuli: T3 ranking form removed; Group Selection gets 2-min timer:** During the "Ideas Board + Discussion" phase, tablets no longer receive a ranking form — participants discuss all ideas freely. The server-side `show_ideas_discussion` handler no longer auto-pushes `T3_RANKING_FORM` to tablets. `T3_GROUP_SELECTION_FORM` now carries `timer_seconds: 120`; the bigscreen group-selection broadcast also passes the 2-min timer; and `group_selection` is removed from `untimedPhases` in the tablet JS so the countdown renders. T3 moderator reminder updated to reflect free discussion and the timed group-selection step.

2026-03-03 (fix)
- **Stimuli: T2 settlement decision auto-logged from tablet (no moderator action needed):** When any participant submits the Settlement Form (`final_topic` + `final_format` fields present in a T2 `/response` payload), `_handle_response` now automatically calls `push_lsl_final_decision_marker("T2", ...)` and pushes the group-decision summary to the bigscreen — identical output to what the old manual "Log Topic & Format Decision" button produced. The manual T2 decision block in the moderator console has been replaced with a green informational panel confirming auto-logging. T2 instructions panel updated to remove step 5 manual-log instruction. `final-decisions-note` text updated to mention T2 alongside T4.

- **Stimuli: renderForm submit-button bugs fixed (T3/T4):** Two bugs in the tablet `renderForm()` JS function (embedded in `stimuli/display_server.py`): (1) `number` inputs had no `value` attribute — if a participant did not touch the T4 contribution slider, `FormData` sent an empty string which caused `int("")` → `ValueError` on the server → HTTP 400 → tablet showed "✗ Submit failed". Fix: added `value="<field.min>"` as default and `required` attribute. (2) `select` fields (e.g. T3 group_selection `idea_author`) had no placeholder option — the first real option (`P1 — Idea 1`) was silently pre-selected. Fix: added `<option value="">-- Select --</option>` as the first option. Server-side validation for T4 contributions was already robust (`int()` in a `try/except`); the failure was purely client-side.

2026-03-03
- **Combined tests for calibration + Tobii glasses tracker:** Added `tests/test_calibrate_charuco.py` (38 tests) covering board definitions, camera-spec loading/matching, intrinsic matrix construction, video discovery/exclusion, H.264 encoder detection, CLI dispatch, and repo-level spec validation. Added `tests/test_tobii_multicam_glasses_tracker.py` (47 tests) covering CameraCalibration (projection, undistortion, world position), config parsing (YAML → TrackerConfig, GlassesMarkerConfig, TableMarkerConfig), DLT triangulation (stereo + 3-camera synthetic geometry), marker corner triangulation, glasses 6-DoF pose estimation (0/1/2 markers, centre, quaternion unitarity), gaze-to-world transformation (ray-plane intersection, confidence, edge cases), gaze NDJSON loading, video discovery + camera mapping, TOML loading, and repo example config validation (marker IDs, table corners on z=0). All 85 tests pass; ruff clean.
- **7th camera (6th PanaCast 20) — back-middle position:** Added `jabra_panacast_20_cam6_vid` to `configs/ffmpeg_multicap.json` — total setup is now **6× P20 + 1× P50 = 7 cameras**. cam6 is mounted upright at back-middle for rear overview coverage (PID 3021, PNP `#9&17d42a77`). Orientation: cam1–cam4 are upside-down (flip in post), cam5/cam6/P50 are upright. Updated: `configs/lab_small.yaml` (count 6→7, notes), `src/affectai_capture/manifest_small.py` (P20 count 5→6, description 7 main), `docs/jabra_recording_checklist.md` (inventory table with orientation column), `docs/recording_sync_calibration_pipeline.md` (camera table + desk layout + orientation), `docs/llm/context_snapshot.md` (camera counts + layout + flip flags).
- **Multicam glasses tracker (fixed cameras + ArUco markers):** Added `tools/tobii_multicam_glasses_tracker.py` — tracks 4× Tobii Glasses 6-DoF pose using small ArUco markers attached to glasses frames (left + right temple), detected by the 6 fixed lab cameras. Workflow: (1) load multicam calibration TOML, (2) detect ArUco markers in all camera views, (3) triangulate marker corners to 3D per frame, (4) compute glasses pose from left/right marker positions, (5) transform Tobii gaze rays to world coordinates. Outputs per-glasses: `*_pose.ndjson` (position + quaternion), `*_gaze_world.ndjson` (world gaze + confidence). Complements the scene-camera-based alignment tool when table markers are not visible to Tobii cameras. Added `configs/tobii_multicam_glasses_tracker.example.yaml` with marker layout: table markers (IDs 0–4) at corners + centre, glasses markers (IDs 10–17) for P1–P4 left/right. Includes offset measurement guidance for marker→eye-centre transform.

- **ArUco marker sheet generator:** Added `tools/generate_aruco_marker_sheet.py` — generates printable PNG sheets with all required ArUco markers for the multicam glasses tracking setup. Creates: (1) combined A4 sheet with all markers, (2) separate table marker sheet (50mm), (3) separate glasses marker sheet (15mm). CLI: `--table-only`, `--glasses-only`, `--table-size`, `--glasses-size`. Uses DICT_4X4_50 dictionary. Print at 100% scale on matte paper.

- **Offline multi-Tobii world alignment tool (marker-based):** Added `tools/tobii_multi_glasses_world_align.py` to align up to 4 recorded Tobii Glasses streams into one shared coordinate frame offline. The tool reads per-device scene video + gaze NDJSON, detects ArUco markers, estimates per-frame camera pose via `solvePnP`, and projects gaze rays to the board world plane (`z=0`). Supports `--ray-source gaze2d` (image gaze + intrinsics) and `--ray-source gaze3d` (eye origin/direction or `gaze3d` fallback from Tobii 3D payloads). Outputs are derived artifacts only: per-device pose NDJSON, per-device world-gaze NDJSON/CSV, and run summary JSON. Added template config `configs/tobii_offline_world_align.example.yaml` and usage docs in `docs/vicon_tobii_lsl_capture.md`.

- **Tobii world-gaze QC plotting tool:** Added `tools/qc/qc_tobii_world_gaze.py` for post-alignment quality checks across all glasses devices. The tool reads `*_gaze_world.ndjson`, writes summary metrics (`tobii_world_gaze_summary.json` + `.csv`), and renders two quick-look plots (`tobii_world_gaze_scatter.png`, `tobii_world_gaze_timeseries.png`) to inspect cross-device coordination and drift over time.

- **Tobii QC board-overlay option:** `tools/qc/qc_tobii_world_gaze.py` now supports `--align-config` to load marker polygons from the same alignment YAML and overlay marker outlines plus global board bounds on `tobii_world_gaze_scatter.png`. This makes it easy to spot out-of-board gaze projections at a glance.

2026-03-04
- **6th camera (5th PanaCast 20) + calibration improvements:** Added `jabra_panacast_20_cam5_vid` (top-front-center overview camera, PID 3021) to `configs/ffmpeg_multicap.json` — total setup is now **5x P20 + 1x P50 = 6 cameras**. All video devices now have `show_camera_dialog: true` and `camera_setup_script: "tools/lock_exposure.py"` for automatic exposure/WB lock. (1) **`tools/lock_exposure.py` (new):** Automated exposure, white-balance, and gain lock via OpenCV UVC controls. Workflow: open camera -> let auto-exposure settle (configurable, default 2s) -> read settled values -> switch to manual mode and lock. Supports: single camera by index/name/alt-name, `--list` diagnostic mode, explicit `--exposure`/`--wb`/`--gain` overrides, `--auto-settle` duration. Works as standalone tool or as `camera_setup_script` called by `ffmpeg_multicap.py` before each capture. (2) **`calibrate_charuco.py ground-plane` sub-command (new):** Dedicated ground-plane calibration step. Records a short clip of the ChArUco board lying flat on the table, detects the board in each camera view via solvePnP, picks the best view, computes the table-surface coordinate frame (X/Y on table, Z pointing up, origin at board centre), and saves `calibration_charuco_groundplane.json` alongside the TOML. Also injects `groundplane_reference_camera`, `groundplane_reproj_px`, and `groundplane_file` into the TOML metadata section. CLI: `--videos-dir`, `--toml`, `--square-size`, `--board-type`, `--frames`. (3) **Unicode fix in `ffmpeg_multicap.py --list-devices`:** Replaced emoji characters (camera/mic/info icons) with ASCII equivalents to fix `UnicodeEncodeError` on Windows cp1252 terminals. (4) **Manifest & config updated:** `manifest_small.py` P20 count 4->5, video description updated to 6 main cameras. `lab_small.yaml` camera count 5->6, notes updated to describe 5 P20 + 1 P50 layout with exposure lock. `docs/jabra_recording_checklist.md` updated with camera inventory table, automated lock_exposure.py instructions as Option A (recommended), and dedicated ground-plane calibration step. `docs/llm/context_snapshot.md` updated camera count and added `lock_exposure.py` reference. Files changed: `configs/ffmpeg_multicap.json`, `tools/lock_exposure.py` (new), `tools/calibrate_charuco.py`, `tools/ffmpeg_multicap.py`, `src/affectai_capture/manifest_small.py`, `configs/lab_small.yaml`, `docs/jabra_recording_checklist.md`, `docs/llm/context_snapshot.md`.

- **Factory-spec focal length seeding for calibration (`--init-focal`):** Added `configs/camera_specs.json` with known FOV and expected focal lengths for Jabra PanaCast 20 (120-deg HFOV, fx=554px at 1080p) and PanaCast 50 (133-deg HFOV, fx=418px at 1080p). New `--init-focal` flag on `calibrate_charuco.py calibrate` seeds anipose's bundle-adjustment optimiser with the factory-expected intrinsic matrix instead of relying on `cv2.initCameraMatrix2D` (which produces bad estimates when ChArUco board coverage is insufficient). Camera names are matched to model specs via regex patterns in `camera_specs.json`. When all cameras match known models, calibration runs with `init_intrinsics=False`; mixed setups fall back to anipose defaults for unmatched cameras. The `validate` sub-command now compares calibrated focal lengths against expected specs and flags BAD/SUSPECT ratios. Files changed: `configs/camera_specs.json` (new), `tools/calibrate_charuco.py`.

- **Stimuli: dual-write event logger with per-source streams & group support:** New `stimuli/event_logger.py` module — `ExperimentEventLogger` class that **always writes local TSV** (`events_<session>_<stream>.tsv`) and **optionally pushes to LSL** when `--lsl-markers` is passed. **7 streams**: `participant_1`–`participant_4`, `moderator`, `bigscreen`, and `experiment` (cumulative: every event from every source). Each TSV row carries `wall_clock`, `lsl_clock`, `session_id`, `group_id`, `stream`, `event_type`, `task`, `phase`, `participant`, `device_id`, and a JSON `detail` column. LSL stream XML `<desc>` embeds `session_id` and `group_id`. New `--group-id` CLI argument (embedded in every event record and LSL metadata). All existing `push_lsl_marker` / `push_lsl_event_marker` / `push_lsl_response_marker` / `push_lsl_final_decision_marker` calls now delegate to `EVENT_LOG`, gaining local-file persistence without changing any call site. New event types added: `session_start`, `session_end`, `response_idea_submission` (T3), `moderator_clear_cache`. Files changed: `stimuli/event_logger.py` (new), `stimuli/display_server.py`.

- **Stimuli: T4 individual payoffs, finish questionnaire note, T3 bigscreen timer fix, WRAPUP dropdown:** (1) **T4 individual payoffs in outcome:** `_build_t4_outcome_payload` now accepts `contributions` dict and renders a per-participant table showing tokens contributed, tokens kept, public share, and total payoff (kept + share). Both manual and auto-reveal paths pass contributions through to bigscreen + tablets. `T4_LAST_OUTCOME` and the discussion phase also include individual breakdowns. (2) **Finish phase questionnaire instruction:** `TASK_FINISH_CONTENT` (T1–T4) now tells participants "Please fill in the questionnaire on your tablet now" before the short-break message. T0 unchanged (no questionnaire). (3) **T3 bigscreen timer fix:** Bigscreen `startTimer()` silently returned when `#timer` div didn't exist — this happened for `show_ideas_discussion` because its content was set via `html` (not `formatBigScreenContent` which creates the timer div). Fix: `startTimer` now dynamically creates the timer div if missing. (4) **WRAPUP dropdown re-added:** `<option value="WRAPUP">` was missing from the moderator `<select>` (previous fix didn't persist). Files changed: `stimuli/display_server.py`, `stimuli/task_content.py`.

- **Stimuli: T3 bigscreen styling fix (ideas board + group selection):** Ideas-grid card backgrounds were `rgba(255,255,255,0.08)` — invisible on the white `.content` div. Fixed to `#e3f2fd` (light blue) with `border: 1px solid #90caf9`. Participant name colour changed from `#64b5f6` → `#1565c0` for readability. "No ideas yet" placeholder given explicit `color:#666`. Heading uses `class="task-title"` for CSS consistency. Inline `<div id="timer">` added to `show_ideas_discussion` HTML so the countdown renders without relying on `formatBigScreenContent`. Same card styling applied to `group_selection` block. File changed: `stimuli/display_server.py`.

- **Stimuli: 8 moderator/display improvements & bug fixes:** (1) **Final decision → bigscreen:** `logFinalDecision` now broadcasts a summary card to the big screen after the moderator submits it (T1: candidate, T2: topic + format, T3: idea + author). (2) **Per-task finish content:** `TASK_FINISH_CONTENT` is now a dict keyed T0–T4 with task-specific titles and "short break" messaging (was a single generic dict). (3) **T0 finish button:** T0 (Free Talk) gains a "✔ Finish T0" phase button in the moderator console. (4) **T0 VAD timer:** T0 instructions now mention starting the VAD timer during free talk. (5) **T1 merge evidence + silent reading:** "Send Evidence Cards" and "Silent Reading" collapsed into a single "Evidence Cards + Reading (1 min)" button; pushing it sends evidence cards to tablets AND silent-reading content + 60 s timer to bigscreen simultaneously. (6) **T3 idea-board bug fix:** Tablet form sends `idea_1/2/3` as top-level JSON keys via `Object.fromEntries()`, but the server was checking a non-existent `responses` sub-object — ideas were never stored. Fixed: handler now checks both levels. (7) **T4 outcome button fix:** "Reveal Outcome" button was calling `pushQuick('T4','outcome')` which routed nowhere; now calls `showT4Outcome()` (hits `/t4_outcome`). (8) **Wrap-up stage:** New `WRAPUP` pseudo-task with three phases — intro screen, "Send Final VAD" button, and thanks/session-complete screen — plus matching `WRAPUP_CONTENT`/`WRAPUP_THANKS` in task_content.py. Both `renderTaskButtons` functions updated to handle finish confirm, T4 outcome, and WRAPUP VAD consistently. Files changed: `stimuli/task_content.py`, `stimuli/display_server.py`.

- **Stimuli: T4 force-reveal for moderator:** `_handle_t4_outcome` no longer requires the 2-minute timer to elapse for a manual moderator reveal. If fewer than 4 contributions exist, the endpoint returns `{"status": "needs_confirm"}` and the JS shows a confirmation dialog; if the moderator confirms, a retry with `{"force": true}` computes the outcome directly (bypassing `_emit_t4_outcome_if_ready`) and broadcasts to all devices. With all 4 contributions the outcome is revealed immediately, no timer wait. The automatic timer-based reveal path (`_emit_t4_outcome_if_ready` / `_schedule_t4_reveal_if_needed`) keeps its guards unchanged. File changed: `stimuli/display_server.py`.

- **Stimuli: T3 merge + T3 selection display + T4 outcome alert + WRAPUP dropdown:** (1) **T3 merge Show Ideas Board + Discussion & Ranking:** Collapsed into single "Ideas Board + Discussion (4 min)" button (`show_ideas_discussion` phase) — pushes ideas grid + 4-min timer to bigscreen and ranking form to tablets in one step. (2) **T3 group selection → bigscreen ideas board:** "Group Selection" now shows the full ideas board on bigscreen (with "Which idea wins?" prompt) so everyone can see the options during final selection; after `logFinalDecision`, the winning idea + author is pushed to bigscreen. (3) **T4 Reveal Outcome alert:** `showT4Outcome()` now shows an `alert()` popup to the moderator when the outcome cannot be revealed (e.g. "Only 0/4 contributions received" or "wait Ns"), instead of failing silently. Fixed `showT4Outcome` / `pollT4Status` function corruption from prior edit. (4) **WRAPUP dropdown missing:** Added `<option value="WRAPUP">Wrap-Up & Thanks</option>` to the moderator task-select dropdown (backend + `taskPhases` already existed but the dropdown entry was absent). Files changed: `stimuli/display_server.py`.

2026-03-03
- **Role-specific LaTeX session guides (`docs/sources/moderator_guide.tex`, `docs/sources/data_collector_guide.tex`, new):** Two printable 0-to-100 session guides split by role. **Moderator guide:** verbatim scripts for all tasks (T0–T4) in colour-coded script boxes, task-by-task phase instructions with dashboard actions and timing cues, seating map (TikZ), in-session checklist (consent → debrief), session notes page. **Data collector guide:** full terminal layout for two-PC architecture, copy-paste CLI commands for every bridge and recorder (Vicon, Tobii, EmotiBit, display server, ffmpeg_multicap, dpa_recorder), LSL stream health-check table (30+ streams with checkboxes), camera verification table, BIDS session init & registration commands, recording start/stop sequence, sync clap + baseline protocol, data verification & archival (spot checks, modality validation, manifest, NAS transfer, privacy check), post-session checklist, troubleshooting quick-reference tables for all devices.

2026-03-02
- **Bigscreen viewport-fit layout:** Reworked bigscreen CSS and rendering so all content fits within a single viewport (no scrolling). Body uses `height: 100vh; overflow: hidden` with flex layout. Reduced padding, fonts, and margins throughout. Added JS `fitToViewport()` auto-scale fallback that shrinks content via CSS transform if it still overflows. T3 idea board and T4 outcome/discussion HTML made more compact. Timer reduced from 72 px to 56 px. File changed: `stimuli/display_server.py`.

- **Moderator finish workflow, bigscreen timers, T3 idea board, T4 phased timers:** Added "Finish" button to every task (T1–T4) in the moderator console; clicking it pushes a "Task Complete" screen to all devices and automatically sends the post-task questionnaire. **T2:** Sending role cards now automatically pushes an 8-min negotiation timer to the bigscreen. Settlement form phase pushes to all tablets + bigscreen. **T3:** Separate bigscreen timers — Idea Generation (2 min), Discussion & Ranking (4 min). New "Show Ideas Board" phase aggregates all submitted ideas and displays on bigscreen as `P1-Idea 1`, `P2-Idea 2`, etc. Server-side T3 idea collection (`T3_LOCK`/`T3_IDEAS`). Group Selection and idea author dropdown updated to `P1 — Idea 1` through `P4 — Idea 3` format. **T4:** Contribution form pushes 75-sec timer to bigscreen. Outcome reveal includes 2-min timer. Discussion phase preserves outcome on screen with its own 2-min timer (`T4_LAST_OUTCOME`). New server globals: `T3_LOCK`, `T3_IDEAS`, `T4_LAST_OUTCOME`. Files changed: `stimuli/task_content.py` (T2_NEGOTIATION, timer_seconds fields, TASK_FINISH_CONTENT, P-id×Idea# selection), `stimuli/display_server.py` (imports, routing, idea collection, finish workflow, bigscreen timers, JS updates).

- **Data collection guide (`docs/data_collection_guide.md`, new):** Complete end-to-end session protocol covering all 6 phases: lab setup (software startup sequence for both PCs), participant arrival & registration, instrumentation & calibration (DPA, EmotiBit, Tobii, tablets, Vicon), system verification & sync (LSL health check, sync clap, baseline), task execution (T0–T4 with moderator steps, probe scheduling, timing), session close & debrief, data verification & archival (spot checks, modality validation, NAS transfer, checksums). Includes printable checklists (pre/in/post-session), troubleshooting tables for all devices, desk layout diagram, BIDS output structure appendix, complete LSL stream inventory (~30+ streams), and moderator script reference. Also added `.private/` to `.gitignore` for registration ledger privacy.

- **Comprehensive audit & alignment – documentation, code, configs:** Systematic alignment of repository documentation and code with the full experiment specification (4 participants × 4 tasks, two-PC architecture, all devices).
  - **`docs/architecture.md` rewritten:** Added full device inventory table (Tobii ×4, EmotiBit ×4, Jabra PanaCast 5+2, DPA ×5, Vicon ×6, Tablets ×4, Big Screen ×1), ASCII two-PC architecture diagram (Recording PC + AV PC), participant registration section, updated repository map, expanded BIDS data spine with directory tree.
  - **`docs/data_flow.md` rewritten:** Fixed 3 broken references to non-existent files (`stimulus.py`, `tablet_server.py`, `sync.py`). Added all 6 input sources, complete LSL stream inventory (~30+ streams), two-PC architecture diagram, accurate storage estimates (~25–40 GB/session).
  - **`docs/llm/context_snapshot.md` updated:** Added device inventory table, two-PC architecture section, participant registration reference, experiment design section.
  - **`src/affectai_capture/registration.py` (new):** Participant name-to-ID mapping module. `SessionRoster` writes anonymised `participants.tsv` (BIDS), per-session `participants.json`, `.private/registration_ledger.jsonl` (git-ignored, real names). BFI-45 personality linkage via `link_personality()`.
  - **`tests/test_registration.py` (new):** 10 tests across 3 classes covering participant validation, BIDS output (no name leaks), private ledger, personality linkage, error cases.
  - **`src/affectai_capture/manifest_small.py` fixed:** DPA count 4→5, added `physio` modality (EmotiBit PPG/EDA/temp), renamed cameras `ceiling_rgba`→`jabra_panacast_20/50`, added 2 optional extra cameras, `motion/`→`mocap/` directory, task labels `T1T2`→`T1T2T3T4`, updated BIDS_STRUCTURE_SMALL string.
  - **`configs/lab_small.yaml` fixed:** DPA count 4→5 (4 participant + 1 room spare), added EmotiBit device entry (4 units, PPG/EDA/temp via Oscilloscope→UDP→LSL), replaced `rgb_ceiling_camera`+`face_camera` with `jabra_panacast_cameras` (5 primary + 2 optional), `motion/`→`mocap/` Vicon output, tasks T1-T2→T1-T4.

2026-03-01
- **Stimuli feedback implementation (all phases):** Applied 14 design-feedback items across 5 files. **Phase 1 (Task content):** D3 — T1 rephrase instruction added to shared brief & discussion selection. E1 — T2 role cards now carry `preference_area` (format/topic × collaborative/independent/technical/people). F1 — T3 idea generation limited to exactly 3 ideas. F2 — T3 ranking phase renamed "Discussion & Ranking" (no longer silent); LSL markers `T3_ranking` → `T3_discussion_ranking`; moderator script updated. F3 — new `T3_GROUP_SELECTION_FORM` records winning idea + author + rationale. F4 — moderator T3 final-decision panel gains "Whose idea?" dropdown (`t3-author`); `logFinalDecision` sends `idea_author`. **Phase 2 (Moderator UX):** A1 — numbered phase buttons with new steps (evidence cards, settlement form, group selection, reveal outcome). A2 — untimed phase list; timer no longer starts on brief/intro phases. A3 — step-by-step numbered moderator instructions with timed/untimed distinction. C2 — `/clear_cache` endpoint; server-side content cache cleared on task switch. **Phase 3 (Probes):** B1 — emoji anchors on VAD valence (😞↔😊) and arousal (😴↔⚡); `emoji` field added to `Probe` dataclass. B2 — dominance reworded from "in control or overpowered" → "How much influence do you feel you have"; post-block `perceived_control` → `perceived_influence` (T1, T2). B3 — probe push spacing randomised (0.4–1.2 s jitter). C1 — VAD intro (T0) now shows emoji anchors. Schema version bumped to `2026-03-01.1`. Files changed: `stimuli/task_content.py`, `stimuli/display_server.py`, `stimuli/probe_definitions.py`, `stimuli/tasks/t3_idea_generation.py`, `stimuli/tasks/push_helpers.py`.

- **Root .md files reorganised:** Moved `ADMIN_SETUP.md` → `docs/admin_setup.md`, `CALIBRATION_USAGE.md` → `docs/calibration_usage.md`, `TEAM.md` → `docs/team.md`. Merged near-duplicate `OPENPOSE_INTEGRATION.md` + `QUICKSTART_OPENPOSE.md` into single `docs/openpose_integration.md` (265 + 242 → 160 lines). Root now has only standard GitHub convention files (README, CHANGES, CODE_OF_CONDUCT, CONTRIBUTING, SECURITY, SUPPORT) plus ARCHITECTURE.md as navigation hub. Updated cross-references in `docs/calibration_usage.md` and `docs/llm/context_snapshot.md`.

2026-02-28
- **Root directory reorganisation:** Moved 7 loose scripts/batch files from repo root into `scripts/` (run_openpose.py, process_all_cameras_openpose.bat, validate_calibration.py, test_triangulation.py, test_reconstruction.py, analyze_mismatch.py, setup_recording.bat). Moved 4 output artifacts (skeleton_3d.json/.npy, calibration_charuco.toml, charuco_board.png) into `new_data/`. Removed nested `affectai-capture/` clone. Untracked `new_data/` and `openpose_output/` from git (~17K files). Updated `.gitignore`, internal path references, and all affected documentation.
- **Knowledge system restructuring for Copilot efficiency:** 3-layer reading architecture to prevent context window waste. (1) Archived 8 stale/unapplied proposal files (2,700+ lines total) from `docs/llm/` to `docs/llm/archive/` — KNOWLEDGE_ASSESSMENT.md, STRUCTURE_OPTIMIZATION.md, ASSESSMENT_SUMMARY.md, WHAT_WAS_DONE.md, OPTIMIZATION_SUMMARY.md, CONTRADICTIONS.md, progress_report.md, enhance_lsl_with_frame_logs.md. (2) Rewrote `docs/llm/context_snapshot.md` (185→58 lines): removed 100 lines of dated changelog entries duplicating CHANGES.md, kept only current state + doc routing table. (3) Rewrote `docs/llm/README.md` (292→30 lines): replaced bloated assessment index with compact file table. (4) Merged `memory/constitution.md` invariants into `.github/copilot-instructions.md` (added § Reading order, § Invariants, § archive exclusion). (5) Fixed `ARCHITECTURE.md` (240→48 lines): removed duplicate link sections, phantom directory tree, stale references. (6) Updated `docs/llm/DEVOPS_PROMPT_LIBRARY.md` to point contradiction tracking to `docs/known_issues.md`. Net reduction: ~3,400 lines of Copilot-facing docs eliminated; each fact now lives in exactly one place.
- **Camera settings & calibration improvements (configs, tools, docs):** Systematic enhancements based on wide-angle Jabra PanaCast calibration lessons. (1) **P50 resolution fix:** `configs/ffmpeg_multicap.json` PanaCast 50 changed from 1280×720/2500kbps to 1920×1080/4000kbps — must match calibration TOML intrinsics (was a silent resolution mismatch causing bad undistortion). (2) **Default board 7×5:** `calibrate_charuco.py` default changed from 5×3 (8 corners) to 7×5 (24 corners) — more corners give better coverage of wide-angle lens distortion. (3) **Ground-plane ON by default:** `--groundplane` now defaults to True; added `--no-groundplane` to opt out. (4) **Spatial coverage analysis:** `calibrate_charuco.py detect` now reports 3×3 grid region coverage with empty-zone warnings and tips for wide-angle periphery coverage. (5) **Camera exposure/WB lock support:** New `DeviceConfig` fields: `show_camera_dialog` (opens native DirectShow property dialog), `dshow_extra_args` (arbitrary input args), `camera_setup_script` (pre-capture script for UVC control). New `--show-camera-dialog` CLI flag applies to all video devices. Pre-capture hook runs `camera_setup_script` before ffmpeg launch. (6) **Jabra recording checklist:** New `docs/jabra_recording_checklist.md` covering resolution consistency, exposure/WB lock, lighting, framing, edge avoidance, frame rate stability, charuco calibration procedure, and what NOT to manually edit in calibration output.
- **Face + hand 3D pipeline (`tools/face_hand_pipeline.py`):** New `detect` + `reconstruct` pipeline for multi-camera 3D face landmarks (478 pts), hand landmarks (21 pts × 2 hands), and ARKit blendshapes (52 coefficients) using MediaPipe FaceLandmarker + HandLandmarker. **Detection:** body-pose-guided crops (OpenPose BODY_25 head keypoints) for reliable face detection on wide-angle Jabra cameras; IMAGE mode with per-camera crop ROIs. Handles upside-down P20 cameras (undistort-first, then flip; un-flip landmarks back to raw camera space for triangulation). Uses `presence` attribute (not `visibility`, which is always 0 in MediaPipe 0.10.14). Detection rates: face 93%/68%/100% on 3 cameras, hands 57–96% on all 5. **Reconstruction:** nonlinear triangulation — DLT initial estimate + Levenberg-Marquardt refinement with forward distortion model (`scipy.optimize.least_squares`). Monotonicity-based camera filter (`1 + 3·k1·r² > 0`) excludes cameras where `cv2.undistortPoints()` is unreliable (cam_0 k1=−1.186, P50 k1=−0.569 at edges). Face-centroid-proximity hand association (wrist-to-face distance, 40% diagonal threshold). Results on `ses-20260202_test` (300 frames): face 99,901 landmarks median 6.98 px reproj, person 0 in 209/300 frames; hands 5,713 landmarks median 2.86 px, right 200/300, left 53/300; blendshapes 300/300 (100%). Output: `.npz` with `face_3d (F,P,478,4)`, `hand_3d (F,P,2,21,4)`, `blendshapes (F,P,52)`.
- **Distortion-robust triangulation backported to all pipelines:** Lessons from face/hand pipeline applied across codebase. (1) `multicam_pose3d.py`: `triangulate_point()` now has monotonicity filter (`_distortion_is_monotonic`) + LM refinement with forward distortion model (`_project_distorted`), replacing DLT-only; skips LM when all cameras have negligible distortion. (2) `triangulate_openpose.py`: Now loads distortion coefficients from calibration TOML, undistorts 2D points via `cv2.undistortPoints()` before DLT (was using raw pixels), and filters non-monotonic cameras. (3) `calibrate_charuco.py validate`: New distortion coefficient analysis table (k1, k2, |dist|, monotonic?) with warnings for non-monotonic cameras and remediation advice. (4) `docs/known_issues.md`: Documents the `cv2.undistortPoints` silent failure mode, affected cameras, and mitigation.
- **multicam_pose3d: face/upper-body close-up cameras (`--face-cameras`):** New `--face-cameras cam5:0 cam6:1` CLI flag for dedicated close-up cameras that are pre-assigned to a specific person. These cameras bypass zone/epipolar matching and contribute only head+upper-body keypoints (BODY_25 indices {0–8, 15–18}). Best detection per face camera is picked by mean confidence. Camera count is flexible — any number of face cameras can be added. New `FaceCameraAssignment` dataclass + `parse_face_cameras()` function; `CameraCalibration` gains `role` ("scene"/"face") and `kp_mask` attributes. Observations from face cameras are injected directly into the per-person observation list.
- **layout_video_3d: generic `--flip-cameras` list:** New `--flip-cameras cam5 cam6` CLI flag to flip any camera feed 180° by substring match (replaces hardcoded P20 detection). Unified with existing `--flip-p20` flag — both contribute to a shared `_flip_substrings` list. Grid layout already dynamic (`cols = ceil(sqrt(n_panels))`), so additional cameras are handled automatically.
- **recenter_calibration (`tools/recenter_calibration.py`):** New tool to re-express multi-camera calibration TOML with a chosen reference camera as world origin. Transforms all extrinsic parameters (rotation, translation, world_orientation, world_position) while preserving intrinsics and pair-wise distances (verified). Supports optional `--flip-cameras` to apply 180° image-flip correction for upside-down-mounted cameras (modifies extrinsics: `R' = diag(-1,-1,1) @ R`). Used to make Jabra PanaCast 50 (cam_4) the coordinate centre instead of cam_0 (an upside-down P20). Paired with `multicam_pose3d.py --flip-cameras` for 2D keypoint consistency.
- **multicam_pose3d: --flip-cameras + P50 calibration:** Added `--flip-cameras cam_0 cam_1 cam_2 cam_3` CLI flag to flip 2D keypoints 180° (`(x,y) → (W-1-x, H-1-y)`) for upside-down P20 cameras before undistortion. Must be paired with a flip-corrected calibration TOML (from `recenter_calibration.py --flip-cameras`). Also generated `video_camera_calibration_p50.toml` with P50 as world origin — reconstruction QC unchanged (13.2px mean reproj, 16% valid joints) since re-centering is a rigid transform.
- **Pipeline reference document (`docs/recording_sync_calibration_pipeline.md`):** New end-to-end reference covering hardware (5 cameras + 5 DPA mics), recording (`ffmpeg_multicap.py`), 4-tier synchronisation (frame logs → LSL → TSV → events), spatial calibration (ChArUco 5×3, anipose), 2D pose estimation (MediaPipe BODY_25), 3D reconstruction with zone-aware matching and front-facing filter, skeleton refinement (4-stage pipeline), and layout video generation. Includes config examples, CLI flags, output formats, desk layout diagram, and QC results from `ses-20260202_test`.
- **multicam_pose3d: multi-tier frame synchronisation:** Replaced single-source events JSONL sync with a 4-tier approach inspired by `create_sync_test_video.py`. Tiers (best first): (1) frame logs (`frame_logs/{label}_frames.jsonl`) — per-frame `unix_time - pts_time` median, sub-ms MAD; (2) LSL progress JSONL (`lsl/ffmpeg_progress_{label}.jsonl`) — `stream_time - out_time_sec` median at ~10 Hz; (3) progress TSV (`sourcedata/sync/{label}_ffmpeg_progress.tsv`) — `host_time_sec - out_time_sec` median; (4) events JSONL `capture_started` timestamps (least accurate). Auto-selects best tier with full camera coverage. New CLI: `--frame-log-dir`, `--lsl-dir`; `--session-dir` auto-discovers `frame_logs/` and `lsl/` subdirs. On `ses-20260202_test`: frame-log tier selected (5/5 cams, MAD 0.5-0.7ms), revealed up to 9-frame offsets between cameras. Reproj improved 13.4→13.2px mean, 19.9→18.8px 95th, valid joints 14→16%.
- **multicam_pose3d: front-facing filter for back-of-head rejection:** Added `is_front_facing()` filter to `tools/multicam_pose3d.py` that rejects back-facing detections in zone cameras. Each P20 camera sees 2 people from the front (zone targets) + 2 from behind (opposite zone); the filter requires Nose + at least one Eye keypoint to have confidence ≥ `--min-face-conf` (default 0.3). Enabled by default when `--camera-zones` is used; disable with `--no-front-facing-filter`. Implementation: `_filter_front_facing()` preserves original detection indices via `_orig_idx` attribute, `match_persons_zonewise()` applies filter and remaps indices. CLI flags: `--no-front-facing-filter`, `--min-face-conf 0.3`.
- **multicam_pose3d: zone-aware 4-person reconstruction:** Added `--camera-zones` to `tools/multicam_pose3d.py` for multi-person seated setups where each camera pair covers a known subset of participants. Format: `--camera-zones cam1+cam4:0,1 cam2+cam3:2,3`. Within each zone, standard epipolar matching (max 2 people, highly reliable). Cameras not in any zone (e.g. P50) are auto-detected as shared and assigned per-frame to the closest zone person via epipolar distance. Person ordering: by horizontal centroid (left→right) for stable assignment with seated participants. Tested on `ses-20260202_test`: reproj 10.4px mean (improved from 14.9px), output shape (1568, 4, 25, 7). Works with existing `refine_skeleton_3d.py` and `layout_video_3d.py` (multi-person colors).
- **refine_skeleton_3d: multi-view confidence filtering + temporal boosting (`tools/refine_skeleton_3d.py`):** New 4-stage post-processing pipeline for 3D skeleton quality refinement. (1) Quality gate: NaN-out joints below `--min-confidence`, above `--max-reproj`, or seen by fewer than `--min-cameras`; optional `--upper-body` to discard lower-body keypoints. (2) Velocity outlier rejection: NaN-out joints with frame-to-frame displacement > `--max-velocity` mm. (3) Gap interpolation: cubic-spline (≥4 anchor points) or linear fill for gaps ≤ `--max-gap` frames. (4) Temporal smoothing: Butterworth low-pass filter (`--smooth-cutoff` Hz). On `ses-20260202_test`: 14392 valid → gate 8937 (removed 170 low-conf, 3324 high-reproj, 2013 lower-body) → velocity 8931 → interp 14609 (recovered 5678 gap-frames) → final 14609 (9.3%). Outputs `*_refined.npy` + companion JSON metadata with per-stage stats.
- **layout_video_3d: P20 camera flip + upper-body skeleton:** Added `--flip-p20` flag to rotate all Panacast 20 camera feeds 180° (cameras are mounted upside-down). Added `--upper-body` flag to render only head+spine+arms in the 3D skeleton panel (excludes leg keypoints, better suited for seated/desk setups). Both flags used together: `python tools/layout_video_3d.py --session <dir> --flip-p20 --upper-body`.

2026-02-27
- **Calibration-aware multicam 3D pose reconstruction (`tools/multicam_pose3d.py`):** New pipeline that fully exploits synchronised multi-camera recording + spatial calibration for 3D skeleton reconstruction. Features: (1) auto-maps pose-JSON directories → calibration cameras via TOML `name` field, (2) frame-level temporal alignment using per-camera start offsets from `ffmpeg_multicap_events.jsonl`, (3) 2D keypoint undistortion with calibration distortion coefficients (critical for Jabra wide-angle lenses with |dist| up to 1.19), (4) cross-camera person matching via epipolar geometry (fundamental matrix + median joint distance), (5) DLT triangulation with reprojection-error filtering (default 30px threshold), (6) outputs per-frame QC with accepted/rejected reproj stats, camera count, and valid-joint percentage. Initial run on `ses-20260202_test` with MediaPipe 2D detections: 1568 synchronised frames, 5 cameras, accepted reproj 15.7px mean / 15.5px median, 26% valid joints (limited by lower-body occlusion in desk setup). Output: `(frames, people, 25_keypoints, 7=[x,y,z,conf,reproj,n_cams,group])` numpy array + companion JSON metadata.
- **MediaPipe pose detection (`tools/test_mediapipe_pose.py`):** Full-session 2D pose detection across all 5 cameras using MediaPipe Tasks API (v0.10.32). Outputs OpenPose-compatible BODY_25 JSON per frame. Overall detection rate: 85.4% (cam1: 62%, cam2: 94%, cam3: 95%, cam4: 86%, p50: 90%). Replaces non-functional OpenPose workflow (binary never installed; all JSONs contained empty `people:[]`).
- **ffmpeg_multicap: fix MJPEG input codec failure:** Conda-forge ffmpeg 6.1.1 cannot negotiate MJPEG on PanaCast 20/50 DirectShow pins (cameras only expose YUY2/NV12). The `DeviceConfig.input_video_codec` default of `"mjpeg"` caused `Could not set video options` → I/O error on all cameras. Fixed by adding `"input_video_codec": null` to all video devices in `configs/ffmpeg_multicap.json`. Cameras now capture native YUY2; ffmpeg re-encodes to MJPEG for output.
- **ffmpeg_multicap: updated cam4 video_alt_name:** cam4 device changed from PID 302a to PID 3021 (`9&99a8671`).
- **ffmpeg_multicap: stable audio device paths for camera mics:** Added `audio_alt_name` (DirectShow device path) to cam1, cam2, cam3, P50 entries in `configs/ffmpeg_multicap.json`. Paths determined via USB container-ID pairing. Fixes `ValueError: No audio device resolved for DirectShow input` caused by fragile `audio_index` shifting between boots.
- **ffmpeg_multicap: resilient startup for non-essential captures:** Camera-mic audio captures (`mux_audio: false` separate audio) now marked `essential=False`. A missing Jabra mic logs a warning but no longer aborts the entire recording session. Video and DPA microphone failures remain fatal. Thread-level exception handling added to `_start_capture`.
- **ffmpeg_multicap: audio capture trigger broadened:** Separate audio capture for video devices now created when `audio_alt_name` or `audio_name` is present, not only when `audio_index` is set.

2026-02-26
- **display_server.py: fix SyntaxError on startup (Windows):** Duplicate `html = """<!DOCTYPE html>` line inside `_serve_moderator_page()` closed the string literal prematurely, causing a `SyntaxError` on every `python display_server.py serve` invocation. Removed the stray duplicate line.
- **display_server.py: fix UnicodeEncodeError on Windows cp1252:** `serve()` printed `→` (U+2192) in the session-ID status line. Windows terminals with cp1252 encoding raised `UnicodeEncodeError`. Replaced with ASCII `->` arrow.

2026-02-25
- **Multi-person 3D tracking with OpenPose:** Added `tools/triangulate_openpose.py` for triangulating multi-person 2D skeletal poses (from OpenPose JSON output) to 3D using camera spatial calibration. Handles partial camera coverage, multiple people per frame, confidence weighting. Outputs (n_frames, n_people, 25_keypoints, 4=[x,y,z,conf]) numpy array. Includes `validate` sub-command for output inspection. Created `OPENPOSE_INTEGRATION.md` with complete workflow: installation, running on multi-camera videos, triangulation, validation, and Python API examples.
- **Charuco spatial calibration tool:** Added `tools/calibrate_charuco.py` — 5-step workflow (print-board, record, detect, calibrate, validate) for multi-camera spatial calibration using FreeMoCap's anipose backend. Produces `.toml` intrinsic/extrinsic calibration files needed for 3D skeleton triangulation from pre-recorded Jabra PanaCast videos. Also added `environment-freemocap.yml` conda env spec (Python 3.10 + freemocap>=1.3.0). Auto-converts .mkv to .mp4, auto-trims videos to equal frame counts (anipose requirement), includes `detect` sub-command for charuco visibility check.
- **DPA audio sync fix: handle audio starting after video:** `tools/create_sync_test_video.py` `compute_dpa_shift()` now returns a signed shift (positive=trim, negative=pad) instead of clamping to zero. Previously, when DPA microphones started later than the video reference (~0.6s in lab setup), the shift was clamped to 0 and no alignment was applied, causing audible delay. Now uses `adelay` filter to pad silence when audio starts late, `atrim` to trim when audio starts early. Affects all three tiers (LSL JSONL, progress TSV, events).
- **ffmpeg_multicap config: added 4th PanaCast 20 + updated audio indices:** `configs/ffmpeg_multicap.json` updated to match current device enumeration — added `jabra_panacast_20_cam2_vid` and corrected audio indices for cam3 (→4), cam4 (→3), and P50 (→1).

2026-02-24
2026-02-26 — Task 1 evidence cards and instructions updated in `stimuli/task_content.py` to match Overleaf `main.tex` and protocol. All evidence cards and shared brief now synchronized for protocol alignment.
2026-01-02 — aligned stimulus design with protocol task definitions
- **VAD haptic notification:** Tablet VAD prompt now triggers device vibration (Vibration API) when the VAD panel appears, with graceful no-op fallback on unsupported browsers/devices.
- **T4 VAD cadence fixed:** VAD timer now runs at a fixed 2-minute interval for Task 4 (still immediate first prompt), while other tasks remain randomized at 2–3 minutes.
- **T1 moderator button cleanup:** Removed `Show Instructions` from Task 1 quick actions in the moderator console.
- **T2 settlement form removed:** Removed `settlement_form` from Task 2 moderator quick actions and task-content push routing.
- **T3 ranking form quick action:** Added `Show Ranking Form` to Task 3 moderator quick actions.
- **T3 exercise brief and timer:** Updated `T3_SHARED_INSTRUCTIONS` text to reflect "Exercise Brief" framing for the whole task; added `timer_seconds: 600` (10 minutes) for big-screen countdown display.
- **T4 moderator button order:** Reordered T4 quick actions so Discussion appears after Show Contribution Form instead of before it.

2026-02-23
- **LSL progress stream improvements (ffmpeg_multicap + sync):**
  - `tools/ffmpeg_multicap.py`: added `-stats_period 0.1` to all four ffmpeg command builders (avfoundation audio, avfoundation video, dshow audio, dshow video) so ffmpeg emits progress blocks at 10 Hz instead of the default 2 Hz. Removed the 0.1s push throttle — ffmpeg rate now limits naturally. Expanded LSL progress sample from 4 to 5 channels: `[out_time_sec, media_time_us, frame, drop_frames, dup_frames]` — adds `media_time_us` (raw microseconds) as `values[1]` so readers can compute `stream_time - media_time_us/1e6` without a unit-conversion ambiguity that existed with the old 4-channel format (old `values[1]` was frame count, not µs).
  - `tools/create_sync_test_video.py`: `load_first_lsl_timestamp` updated with explicit format detection for 5-channel (current), 4-channel (old), and 3-channel (legacy) LSL records. Added `load_lsl_anchor_candidates()` — reads all records from an LSL JSONL and returns a list of `stream_time - out_time_sec` candidates, analogous to `load_progress_anchor_candidates()` for TSV. `compute_dpa_shift()` tiers reordered: LSL JSONL (multi-sample median) is now Tier 1, progress TSV is Tier 2 (fallback when `--record-lsl` was not used at capture time).
- **DPA audio offset correction knob:** `tools/create_sync_test_video.py` gains `--dpa-audio-offset <seconds>` (default 0.0). Additive correction applied to the auto-computed DPA shift; use a negative value (e.g. `-0.1`) when audio sounds late in the output, which can happen because video camera device-init latency biases the progress-TSV start estimate later than the true capture start, inflating the shift.
- **DPA audio sync fix: use shared progress-TSV LSL clock:** `tools/create_sync_test_video.py`: the original DPA shift was using the first video's video-trim offset (meaningless for audio). Replaced with `compute_dpa_shift()` which computes the DPA audio shift relative to the video reference using three tiers all compared within a single shared clock: (1) progress TSV `median(host_time_sec - out_time_sec)` — `host_time_sec` is `local_clock()` (pylsl LSL clock), same timebase for all ffmpeg processes on the same host; (2) LSL JSONL `stream_time - media_time_us/1e6`; (3) `capture_started` events unix timestamps. `--dpa-audio` argument changed from index (`int`) to device label (`str`, e.g. `dpa_an1_aud`).
- **DPA microphone audio support in sync test video generation:** `tools/create_sync_test_video.py` now accepts `--dpa-audio <label>` to include one DPA microphone synchronized with the video grid. Audio is aligned using the same synchronization offsets as video streams. Uses `--dpa-audio-dir` to specify custom audio directory (defaults to `<input>/../audio`). Audio is resampled to 48 kHz mono, AAC encoded at 128 kbps.

2026-02-20
- **Frame-log based sync validated and documented (5-camera test):** Tested complete sync workflow with 5 Jabra cameras (4× P20 + 1× P50) using frame logs for alignment. Achieved **sub-frame accuracy** (±33 ms spread = ±1 frame @ 30 fps; imperceptible in grid video). Config: `mux_audio: false`, `force_wallclock_timestamps: true`, `--frame-log --record-lsl --stabilization-delay 2` during capture. Grid creation: `--source frame --pad-tail --cfr 30`. Output converted from mjpeg to `mpeg4 + yuv420p` for universal MP4 playability (17–20 MB file). Documented best practices in `docs/SYNC_BEST_PRACTICES.md`.
- **P50 configured for 720p (not 1080p):** PanaCast 50 USB 3.0 interface doesn't support 1080p capture. Set resolution to 1280×720 (max stable) and bitrate to 2500 kbps. Verified with ffprobe that actual captured resolution now matches config.
- **Config parser hardened for UTF-8 BOM:** Added `encoding="utf-8-sig"` to both `load_config()` calls in `tools/ffmpeg_multicap.py`. PowerShell `Set-Content -Encoding UTF8` always writes a UTF-8 BOM; Python's `json.loads()` was failing on BOM prefix. Now silently strips BOM on read.
- **Audio-only devices: removed unsupported `audio_channel_select` field:** `DeviceConfig` dataclass does not support per-channel selection via config. Removed 4 unsupported `audio_channel_select` fields from DPA mic entries in `configs/ffmpeg_multicap.json`. Channel splitting is handled via separate RME Fireface alt-name paths per device.
- **Sync test video output: switch from mjpeg to mpeg4 codec:** Changed `tools/create_sync_test_video.py` output codec from `mjpeg` to `mpeg4` with `yuv420p` pixel format. MJPEG in MP4 container is not supported by Windows Movies, QuickTime, or most web players. MPEG-4 Part 2 is universally playable and produces much smaller files (17 MB vs 131 MB for 13s recording).

2026-02-20 (earlier)
- **Revert camera configs to video-only (mux_audio: false):** Removed embedded audio from all 5 camera entries in `configs/ffmpeg_multicap.json`. The validated sync workflow (2026-02-04) used video-only MKV files; embedded AAC audio caused A/V drift under USB load. DPA mics via RME Fireface 802 are the authoritative audio source — camera mic audio is redundant and was the source of frame disruptions and desync.
- **Fix A/V desync in muxed camera captures:** Removed `-fflags +genpts` (it regenerated video PTS synthetically from USB-jittered arrival order while audio kept its wallclock PTS, causing drift). Added `-af aresample=async=1` to the AAC encode step instead — this resamples audio in real time to track the video timeline, compensating for any residual wallclock jitter. `+genpts` was only needed to suppress muxer DTS errors; those are now prevented by MJPEG frames being small and predictable.
- **Force MJPEG input from cameras:** Added `input_video_codec` field to `DeviceConfig` (default `"mjpeg"`). Emits `-vcodec mjpeg` before the dshow input to request MJPEG capture pin instead of raw YUY2/NV12. Without this, some cameras (cam2, P50) defaulted to raw uncompressed video — files were multi-GB and unplayable in most players despite `-c:v copy`.
- **Fix muxer packet errors in ffmpeg_multicap:** Added `-max_muxing_queue_size 1024` (output) to `_build_dshow_command()` in `tools/ffmpeg_multicap.py` and confirmed mutually exclusive timestamp flags (`-use_wallclock_as_timestamps 1` vs `-use_video_device_timestamps 0`) to avoid unstable mux behavior under DirectShow wallclock capture.
- **Disable Jabra intelligent zoom (default):** `tools/ffmpeg_multicap.py` now automatically scans for Jabra PanaCast devices on local network and disables intelligent zoom, auto-framing, and speaker tracking before recording. Use `--keep-zoom` to preserve auto-framing if needed. Ensures fixed camera framing for consistent video backgrounds.
- **WDM-KS audio support in ffmpeg_multicap:** `tools/ffmpeg_multicap.py` now supports WDM-KS backend for audio-only devices via `audio_backend: "wdmks"`. This bypasses TotalMix routing on Windows by capturing directly from RME Fireface hardware inputs using `sounddevice`. Config uses `wdmks_device_id` and `wdmks_channel` (0=left, 1=right) to specify the kernel streaming device. New `--list-wdmks` flag lists available WDM-KS input devices.
- **Muxed video+audio capture:** Added `mux_audio: true` (default) to `DeviceConfig`. When enabled, Jabra camera video captures include the camera's audio stream in the same MKV file (AAC encoded). Set `mux_audio: false` to capture video-only and audio separately as before.
- **Progress logs are now run-scoped:** `tools/ffmpeg_multicap.py` now writes `sourcedata/sync/*_ffmpeg_progress.tsv` in overwrite mode per run, preventing mixed-run `out_time_sec` resets in a single file and incorrect sync anchors.
- **Capture lifecycle hardening:** `FFmpegCapture.stop()` now handles already-exited processes safely, avoids duplicate completion events, and suppresses false non-zero-exit errors on intentional stop. Monitor threads now reconcile running state from process exit.
- **Startup failure cleanup:** `FFmpegMulticap.start_all()` now tears down already-started captures, LSL clock, and marker outlets if any capture fails to start, preventing partial orphaned recording sessions.
- **Sync tooling updated for current log formats:** `tools/create_sync_test_video.py`, `tools/compare_lsl_frame_logs.py`, and `tools/sync/build_frames_and_map.py` now read current TSV progress logs from `sourcedata/sync`, keep legacy `progress_logs/*.jsonl` fallback, and select only the latest run segment when `out_time_sec` resets are detected.
- **LSL naming/documentation alignment:** compare/sync tooling and docs now default to `ffmpeg_progress_` stream names, with fallback to legacy names; README and context snapshot paths were updated accordingly.
- **SSE reconnect safety (display server):** `stimuli/display_server.py` now guards device unregister by checking the active `DeviceClient` instance before removal. This prevents stale/disconnected SSE handlers from unregistering a newer reconnection of the same device.

- **Added RME Fireface 802 inputs to capture config:** `configs/ffmpeg_all_cameras.json` now includes three audio-only devices: `rme_12` (Analog 1+2), `rme_34` (Analog 3+4), `rme_56` (Analog 5+6), each with their unique DirectShow alt-name paths. Output as 256 kbps WAV files under `audio/` subdir.

2026-02-18
- **Alt-name disambiguation for duplicate device names:** `tools/ffmpeg_multicap.py` now supports `video_alt_name` / `audio_alt_name` fields in `DeviceConfig` and JSON configs. `_build_dshow_command` prefers these unique DirectShow device paths (e.g. `@device_pnp_\\?\usb#...`) over display names, which is required when multiple cameras share the same display name (e.g. 4× Jabra PanaCast 20). `--list-devices` now shows alt names under each device. `--update` populates alt names automatically. Updated `configs/ffmpeg_all_cameras.json` with all 5 camera alt names for the current lab setup.
- **Codec fix for conda ffmpeg (no GPL/libx264):** The conda-installed ffmpeg is built with `--disable-gpl` and lacks `libx264`. `_build_dshow_command` now uses `-c:v copy` (stream passthrough of native camera MJPEG) by default, and falls back to `-c:v mjpeg -q:v 3` when `--frame-log` is active (since `showinfo` filter requires a decode pipeline). `tools/create_sync_test_video.py` grid encoder also updated from `libx264` to `mjpeg -q:v 3`. This avoids the silent `Unrecognized option 'preset'` failure that produced empty MKV files. Verified: 5-camera 8s test captured successfully and grid sync video created at 59 MB.

2026-02-18
- **Display server session safety + LSL linkage:** `stimuli/display_server.py` now supports `--session-id` and `--lsl-markers`. Responses are saved with `session_id` in a per-session master file (`responses_<session_id>.jsonl`) and per-participant files (`responses_<session_id>_p1..p4.jsonl`) to prevent cross-group mixing. LSL markers now include both tablet response events and moderator control events (phase pushes, VAD timer start/stop, probe/postblock sends).
- **Moderator UX simplification:** Replaced phase-dropdown workflow with task-specific action buttons (`Show Brief`, `Show Role Cards`, etc.), added a moderator phase timer (elapsed from task start, resets only on task change, manual start/pause/reset + phase presets auto-starting on push), and removed post-task questionnaire preview controls from the moderator page.
- **Tablet panel lifecycle clarity:** New task content now clears lingering probe/postblock panels; after submit, probe/questionnaire panels confirm submission and hide until new content is pushed.
- **Documentation update:** Updated `stimuli/STRUCTURE.md` and `stimuli/README.md` to reflect that **PsychoPy is not required** for the display_server workflow (primary participant interface). PsychoPy task runner (`task_runner.py`) remains available for legacy single-screen mode. Added detailed session management and LSL marker documentation to both files.
- **Audio quality test:** Validated 4× DPA d:fine CORE 4066 headset microphones with RME Fireface 802 interface + REAPER. Recorded individual mic baselines and 2-mic/2-speaker overlap for SNR analysis. Added `docs/audio_quality_test.md` and updated `docs/inventory.md` with audio equipment.
2026-02-17
- **FreeMoCap integration:** Added markerless motion capture via FreeMoCap. New optional dependency `freemocap>=1.3.0` (install with `pip install -e ".[freemocap]"`). Includes post-hoc processor `tools/process_freemocap.py` for extracting 3D skeleton data from video recordings (e.g., Jabra PanaCast). Device adapter `devices/freemocap_processor.py` handles BIDS-compliant output formatting to `mocap/` directory. Quickstart guide: `docs/freemocap_quickstart.md`. Auto-discover task videos, configurable confidence thresholds, batch processing support.
- **Multi-outlet LSL streams:** Display server now creates 6 separate LSL marker streams (`AffectAI_Moderator`, `AffectAI_Participant_1` through `_4`, `AffectAI_BigScreen`) instead of a single combined stream. Markers auto-route to device-specific streams based on `participant`, `device_id`, or `target` kwargs. Enables per-device analysis without post-hoc filtering.
- **Tablet wake lock:** Tablet pages now request Screen Wake Lock API to prevent auto-sleep during tasks. Participants can discuss for extended periods without the tablet screen turning off.
- **Connection indicator:** Tablets and bigscreen show 🟢/🔴 visual connection status. Auto-reconnect triggers after 5s of no heartbeat (reduced from 8s) or when connection state is not OPEN.
- **Moderator task panels:** Redesigned moderator console with task-specific control panels replacing generic dropdowns. Each task shows only its relevant buttons: T1 (shared brief, evidence cards + verbal cues), T2 (shared brief, role cards + verbal cues), T3 (shared brief only + verbal cues), T4 (instructions, contribution form, show outcome). T4 panel includes live contribution status (X/4 received) and manual outcome trigger.
- **T4 outcome endpoints:** Added `/t4_status` (GET) for polling contribution count and `/t4_outcome` (POST) to manually compute and push group outcome to all devices.
- **Probes simplified:** Removed rotating probe system. During tasks, only VAD (Valence–Arousal–Dominance, Likert 1–9) is used; Dominance hidden for T4. All other questions (engagement, confusion, fairness, trust, mental demand) are handled by the post-task questionnaire. This reduces interruption during discussions.
- **Post-block LSL markers:** Sending the post-block questionnaire now emits a `push_postblock` LSL marker (including `questionnaire_id` when available), and raw `/push` treats `postblock_questionnaire` as a postblock push.

2026-02-12
- **Task-specific probes:** Added fairness and confusion probes. Each task now has its own probe set: T1 (engagement, confusion), T2 (confidence, trust, confusion), T3 (fairness, confidence), T4 (fairness, trust). VAD (valence, arousal, dominance) continues via timer/moderator for all tasks (dominance hidden for T4).
- **Probe spacing fix:** `push_phase_probes()` now sends probes sequentially with 0.5s spacing to prevent tablet panel collision. Previously sent all 4 probes simultaneously, causing 3 to be dropped.
- **VAD timer fix:** Immediate first VAD prompt on timer start (no 2-3 min delay). Replaced busy-wait with `threading.Event` for clean stop.
- **Panel collision guard:** Added `activePanelType` tracking. VAD/probe prompts are silently skipped if another panel is active. Post-block always takes priority.
- **VAD non-blocking:** Added Skip button to VAD panel; partial/empty submissions allowed.
- **Probe validation:** Rotating probe submit now requires a selection before sending.
- **Reconnect debounce:** Visibility and focus handlers share a 2s debounce to prevent double reconnects on tablet wake.
- **Probe notifications:** When a new probe (VAD, rotating probe, or post-block questionnaire) arrives on a tablet, participants see: (1) an orange notification banner sliding down from the top with a message like "New question — please respond", (2) a pulsing glow animation on the probe panel, (3) a short audio chime (Web Audio API, no files needed), and (4) smooth auto-scroll to the panel. The banner auto-dismisses after 6s (10s for post-block) and is also dismissed on submit.
- **Probe system redesign:** Replaced single feelings panel with three separate probe panels on tablet pages: (1) VAD panel (Valence + Arousal 1–9 Likert, Dominance 1–9 hidden for T4), submitted together via periodic 2-3 min timer or on-demand; (2) Rotating probe panel (1–7 Likert with labeled anchors; mental_demand, engagement, decision_confidence, trust_cooperation, effort_load); (3) Post-block questionnaire panel (V/A 1–9, mental demand/engagement 1–7, task integrity 0–100 slider) with task-specific items from protocol. All panels hidden by default, shown when triggered by moderator. Moderator console now has VAD timer start/stop, probe send, and post-block questionnaire controls. Server-side VAD timer thread sends periodic prompts at 120-180s intervals. All responses include `client_timestamp` and typed `type` field.

2026-02-11
- **PsychoPy HUD:** Added an optional capture-PC HUD window showing task/phase and a simple timer (`stimuli/display_hud.py`), and wired it into `stimuli/task_runner.py` and T1–T4 tasks.

2026-02-10
- **T4 outcome broadcast:** `stimuli/display_server.py` now broadcasts the T4 outcome to both the bigscreen and all four tablets (was bigscreen only). Ensures all participants see the outcome reveal simultaneously.

2026-02-05
- **Vicon structured LSL:** Added `--structured-lsl` to publish numeric x/y/z translation for a chosen segment.
- **Vicon multicast support:** `tools/vicon_nexus_lsl_bridge.py` now supports multicast connections via `--multicast` and `--local-ip`.
- **Vicon SDK path handling:** Improved Python SDK path resolution and DLL loading for the bundled DataStream SDK.
- **Vicon SDK backend:** `tools/vicon_nexus_lsl_bridge.py` now uses the bundled Python SDK by default, with an optional .NET fallback via `--sdk-backend dotnet`.
- **Vicon live plot:** Added `--plot` options to `tools/vicon_nexus_lsl_bridge.py` to visualize a segment translation in real time.
- **SSE heartbeat ping + moderator note:** Server now emits `ping` SSE events for client watchdogs, and the moderator console notes the wired big screen on the capture PC.
- **SSE auto-reconnect:** Fixed bigscreen and tablet pages to auto-reconnect on SSE failures with exponential backoff (1s→30s). Fixed missing `timerInterval` declaration in bigscreen that broke countdown timers.
- **Multi-device display system:** Added `stimuli/display_server.py` to push task content to Tablets 1-4 (participant-specific) and Big Screen (shared) via SSE. Includes moderator console for device status monitoring and content push control.
- **Sticky display state:** Reconnects now receive the last pushed payload to avoid blank screens after disconnects.
- **T4 shared contribution flow:** Added server-side aggregation for T4 contributions; once all 4 tablets submit, the big screen shows computed outcome values.
- **Big screen countdowns:** Added live countdown timers for T1/T2 shared briefs using `timer_seconds`.
- **Task content module:** Created `stimuli/task_content.py` with all T1-T4 materials extracted from `main.tex`: T1 shared brief + 4 evidence cards, T2 shared brief + 4 role cards + settlement form, T3 instructions + idea/ranking forms, T4 instructions + contribution form + outcome template.
- **T1 full implementation:** Updated `stimuli/tasks/t1_hidden_profile.py` from placeholder to full hidden-profile decision task with all phases (Introduction, Open Discussion, Midpoint Cue, Decision) and proper timing from main.tex.
- **Overleaf sync:** Added `overleaf-stimuli/` git clone of Overleaf project. `stimuli/main.tex` now synced from shared Overleaf document.
- **Stimuli structure documentation:** Added `stimuli/STRUCTURE.md` documenting device architecture, task phases, content distribution, and data flow.

2026-02-04
- **Sync QC fix:** Corrected gap calculations in `tools/qc/qc_sync_report.py` to prevent unpacking errors and enable QC report generation.
- **Jabra-only sync workflow:** Added `configs/jabra_only.json` using DirectShow alternative device paths to disambiguate duplicate PanaCast 20 names on Windows.
- **Documentation:** Recorded the validated Jabra-only capture + synchronized grid workflow in `docs/sync_sources_summary.md` and updated `docs/llm/context_snapshot.md`.
- **Jabra auto-capture:** Added `tools/record_jabra_and_sync.py` to auto-detect connected Jabra cameras, record with frame logs + LSL recording, build sync maps, and create a grid video. DirectShow parsing now captures alternative device paths.
- **Fix:** `tools/record_jabra_and_sync.py` now amends `sys.path` so it can be run as a script without `ModuleNotFoundError`.
- **Fix:** Jabra auto-capture now tolerates Ctrl+C during LSL shutdown without crashing.

2026-02-03
- **Task stimuli implementation:** Created PsychoPy task code for all 4 tasks (T1-T4) from `docs/sources/stimuli/main.tex`. Implemented `stimuli/tasks/` package with base class (`base_task.py`) providing LSL markers, event logging, and timing management. T1 (Hidden Profile) is placeholder; T2 (Negotiation) includes role cards A-D with priority weights; T3 (Idea Generation) implements 5-phase NGT protocol; T4 (Public Goods) implements contribution/payoff calculation with 1.5x multiplier. All tasks include precise timing phases and LSL event markers. Main entry point: `stimuli/task_runner.py`.
- **PsychoPy PyQt5 fix:** Removed broken PyQt6 package, confirmed PyQt5 works with `$env:PSYCHOPY_QT_LIB="PyQt5"`. Resolved QtCore DLL import error when running `stimuli/tablet_questionnaire.py`.

2026-02-02
- **Stabilization delay:** New `--stabilization-delay` flag for `tools/ffmpeg_multicap.py` adds a countdown before recording starts, ensuring all camera streams are fully initialized. Use 2-3 seconds for best sync.
- **Best sync source:** `--source lsl` provides the most accurate alignment for sync videos (uses media_time anchors captured at exact frame arrival). Recommended over `--source frame` for multi-camera sync.
- **Drift-only calibration fix:** Calibration now only applies drift correction, not initial delay (which is already handled by frame/LSL alignment), fixing double-correction bug.
- **Camera calibration system:** New `tools/calibrate_cameras.py` measures per-camera drift and delay relative to a reference camera. Calibration data (initial_delay_ms, drift_rate_ms_per_min, jitter_ms) saved to JSON for use in post-processing.
- **Calibrated sync video creation:** `tools/create_sync_test_video.py` now accepts `--calibration` flag to apply drift compensation. Corrections are applied at recording midpoint for best average accuracy.
- **Calibrated recording:** New `tools/record_calibrated.py` wraps ffmpeg_multicap with calibration context, saving calibration metadata with each recording session for traceability.
- **Calibration analysis results:** PanaCast 20 vs 50 showed 312ms initial delay and 142ms/min drift. For a 5-minute recording, uncorrected drift would exceed 700ms. Calibration enables sub-frame (<33ms) sync accuracy.
- **Multi-camera sync recording verified:** Successfully recorded 3 Jabra cameras (PanaCast 20, 50, 50 Content) with frame logs and progress tracking. Parallel startup achieved 3ms synchronization. Frame-log based alignment detected 310ms offset and created perfectly synchronized 2x1 grid video.
- **Windows DirectShow capture fix:** Conda-installed ffmpeg crashes with DirectShow (error code 3221225785). Solution: Use native Windows ffmpeg build from gyan.dev instead. Added `scripts/setup_recording.bat` to configure PATH correctly.
- **Environment setup:** Created conda environment `affectai` with Python 3.10 and all dependencies from requirements.txt. Recording infrastructure verified working on Windows.

2026-01-30
- tools/create_sync_test_video.py: Add tail padding (zero-mask shorter inputs) and remove `-shortest` to ensure streams align on a common timeline; add `--pad-tail` flag. Improves sync accuracy and prevents early termination.
- Windows: Add `--overwrite-output` option and safe replace logic for CFR re-encode to avoid WinError 5; by default, CFR output is saved alongside original.
- tools/compare_lsl_frame_logs.py: Add `--emit-json` to output recommended per-device offset (`recommended_offset_s`) from anchor-based delay and quality metrics (MAD, drift). Enables automated compensation.
- tools/create_sync_test_video.py: Add `--source best` to auto-select alignment source (prefer LSL anchors, else frame logs with low jitter, else events), and `--use-compare-offsets` to apply offsets computed from progress/anchor logs automatically.
- 2026-01-30 — **Media-time anchors + alignment fixes + merge-safe events**
  - Replaced per-device per-frame LSL streams with **progress-based** anchor handling
  - Added progress logs under `<session>/progress_logs/` with merge-safe timestamps and ffmpeg fields
  - Optional LSL anchor streams named `ffmpeg_<label>_anchor` with values `[lsl_time, media_time_us, frame]`
  - DirectShow now supports `use_video_device_timestamps` and a config flag to force wallclock timestamps
  - Sync test video uses trim/pad alignment inside filter_complex and supports `--align-to earliest|latest`
  - Grid composition preserves aspect ratio using scale+pad
  - Event logs include `unix_time_s`, `unix_time_ns`, `lsl_time`, `monotonic_ns` with ISO UTC for readability
  - Added unit tests for progress parsing, showinfo parsing, and alignment shift logic

- 2026-01-30 — **Sync test video: median frame-log anchoring**
  - **Issue:** Single-sample frame-log anchors can introduce ~100ms jitter in aligned output
  - **Fix:** Use median of multiple `unix_time - pts_time` samples for more stable offsets
  - **New option:** `--frame-log-samples` to control sample count (default: 30)
  - **Expected result:** tighter alignment in `tools/create_sync_test_video.py` outputs

- 2026-01-27 — **Frame log regex fix: now correctly parsing showinfo output with variable spacing**
  - **Issue:** Frame logs were created but remained empty (0 bytes) even when devices captured frames
  - **Root cause:** Regex pattern didn't account for variable spacing in showinfo output format
  - **Fix:** Updated regex from `r"n:(?P<n>\d+)\s+.*?pts_time:..."` to `r"n:\s*(\d+).*?pts_time:\s*([0-9.-]+)"` to handle ffmpeg's showinfo format: `[Parsed_showinfo_0 @ ...] n:   0 pts_time:0.0000001 ...`
  - **Validation:** Frame logs now capture successfully (607 frame records for 20-second capture); comparison tool detects ~3.5s offset between progress-parse LSL timestamps and actual frame capture time
  - **Key finding:** LSL progress parsing (`frame=N` extraction) happens 3-4s AFTER actual frame capture, explaining video synchronization drift; frame logs provide more accurate timing

- 2026-01-27 — FFmpeg multicap parallel device startup + anchor-based sync
  - **Root cause identified:** Sequential device startup (1s apart) and LSL timestamps reflecting progress-parse time (not actual frame capture time) caused persistent sync mismatch
  - **Fix 1:** `FFmpegMulticap.start_all()` now launches all ffmpeg processes in parallel threads for near-simultaneous startup
  - **Fix 2:** `compute_frame_log_offsets()` in `tools/create_sync_test_video.py` now uses `unix_time` as a wall-clock anchor to establish global time reference across devices
  - **Fix 3:** `tools/analyze_sync.py` now filters to the latest run (gaps > 5 minutes create a new run) to avoid mixing multiple sessions and reporting huge offsets
  - Frame logs now enable accurate cross-device alignment: `video_start_unix = unix_time - pts_time` per device
  - Verify: re-record with `--frame-log`, rebuild sync video, compare device start offsets (should be <100ms)

- 2026-01-26 — FFmpeg multicap per-frame PTS logging (showinfo)
  - Added `--frame-log` to `tools/ffmpeg_multicap.py` to emit per-frame logs for video devices
  - Injects `-vf showinfo` and parses stderr to JSONL at `<session>/frame_logs/{label}_frames.jsonl`
  - Records `frame`, `pts_time`, `lsl_clock` (pylsl.local_clock), and `unix_time` for precise alignment
  - On Windows/DirectShow, enables `-use_wallclock_as_timestamps 1` when frame logging is active
  - Intended for downstream sync tooling (e.g., aligning by captured PTS rather than coarse progress)
  - Verify: run a short capture with `--enable-device-streams --frame-log`, then inspect `<session>/frame_logs/*.jsonl`

- 2026-01-26 — Sync test video prefers frame logs when available
  - `tools/create_sync_test_video.py` now reads `<session>/frame_logs/*_frames.jsonl` (from `--frame-log`) for alignment
  - Falls back to LSL JSONL if no frame logs are present; still supports `--no-lsl-sync`
  - Alignment offsets are reported with their source (frame logs or LSL)
  - Optional `--use-events` fallback aligns via capture start times from `video/ffmpeg_multicap_events.jsonl`

- 2026-01-26 — FFmpeg multicap LSL recorder fix
  - Restored `LSLClockPublisher.stop/_run` and removed stray duplicate methods in `tools/ffmpeg_multicap.py`
  - Ruff F811 redefinition resolved; inline LSL recording path ready for lint/tests

- 2026-01-26 — Sync test video alignment with LSL
  - `tools/create_sync_test_video.py` now aligns inputs using first LSL timestamps from `<session>/lsl/*.jsonl`
  - Offsets are computed per stream (default prefix `ffmpeg_`); gaps are padded so timelines match the earliest start
  - Falls back to unsynchronized stacking if no LSL files are found or `--no-lsl-sync` is set
  - Fixed xstack layout to use absolute pixel positions (e.g., `0_0|960_0|1920_0`) instead of broken relative references
  - Fixed synchronization using `-itsoffset` input option (proper timestamp delay) instead of `tpad` filter
  - **Critical fix:** Extrapolate back to frame 0 using frame numbers from LSL streams (LSL recorder starts after capture, so first LSL record is NOT frame 0)

- 2026-01-26 — LSL stream recording for sync
  - Added `tools/lsl_record.py` to save ffmpeg_* LSL streams to JSONL (for offline alignment)
  - Records one file per stream under `<session>/lsl/` with stream_time + received_time
  - Defaults to streams prefixed `ffmpeg_`; configurable via `--prefix`
  - Supports fixed duration (`--duration`) or Ctrl+C to stop

- 2026-01-26 — FFmpeg multicap synchronization testing tools
  - Added `tools/create_sync_test_video.py` for visual sync verification (creates grid/side-by-side videos)
  - Added `tools/analyze_sync.py` for capture timing analysis from events.jsonl
  - Supports multiple layouts (2x2, 2x1, 1x2, 3x1) for multi-device comparison
  - Note: Process start times vary (1-5 seconds stagger), but LSL timestamps provide μs-precision frame sync

- 2026-01-26 — FFmpeg multicap Windows DirectShow capture fix (dual-input + stderr monitoring)
  - Fixed DirectShow command builder to use separate `-f dshow` inputs for video and audio
  - Windows DirectShow requires independent input declarations (unlike macOS AVFoundation)
  - Removed excess quoting from device names (subprocess handles argument boundaries)
  - Added stderr monitoring thread to prevent buffer deadlock and log ffmpeg errors
  - Video and audio recording now works correctly on Windows with DirectShow backend
  - Known issue: Devices with special Unicode characters (e.g., ® in Intel® microphones) may fail

- 2026-01-26 — FFmpeg multicap Windows DirectShow device detection fix
  - Fixed `_parse_dshow_devices()` to correctly parse modern ffmpeg DirectShow output
  - Previous parser looked for non-existent "DirectShow video/audio devices" headers
  - New parser extracts devices from `[dshow @ ...] "Device Name" (video)|(audio)` format
  - Skips "Alternative name" lines that contain device paths
  - Device listing now works correctly on Windows with `--list-devices` flag

- 2026-01-26 — FFmpeg multicap Windows support
  - Added platform-aware device listing and config update (DirectShow on Windows)
  - Configs now include device names (`video_name`/`audio_name`) for Windows captures
  - Capture builder switches between avfoundation (macOS) and DirectShow (Windows)

- 2026-01-20 — FFmpeg multicap auto-update devices
  - Added `--update` flag to automatically detect and update config file with available devices before running
  - Scans all AVFoundation video and audio devices, excludes screen capture devices
  - Matches video devices with their corresponding audio devices where possible
  - Creates both video+audio and audio-only device entries
  - Displays summary of detected devices and updates config file
  - Usage: `python tools/ffmpeg_multicap.py --config configs/ffmpeg_multicap.json --update --enable-device-streams`

- 2026-01-20 — FFmpeg multicap device listing
  - Added `--list-devices` flag to `tools/ffmpeg_multicap.py` to show all available AVFoundation audio/video devices
  - Displays device indices and names before starting capture for easier configuration
  - Usage: `python tools/ffmpeg_multicap.py --list-devices`

- 2026-01-20 — FFmpeg multicap separate audio/video outputs
  - Modified `tools/ffmpeg_multicap.py` to output separate audio and video files for video+audio devices (P20/P50)
  - Video: `{label}_video.mkv` in `video/` subdirectory (H.264, CRF 28)
  - Audio: `{label}_audio.wav` in `audio/` subdirectory (PCM 16-bit, 48 kHz, stereo)
  - Uses single ffmpeg command with `-map 0:v` and `-map 0:a` for efficient concurrent capture
  - Audio-only devices (DPA) continue to output WAV in `audio/` subdirectory

- 2026-01-20 — FFmpeg multicap per-device LSL streams with frame/sample numbers
  - Extended `tools/ffmpeg_multicap.py` to publish per-device LSL streams at nominal rates
  - Video devices: stream at fps (30 Hz) with `[timestamp, frame_number]` parsed from ffmpeg progress output
  - Audio devices: stream at sample rate (48 kHz) with `[timestamp, sample_number]` from software counter
  - Enabled via `--enable-device-streams` flag; each device gets its own LSL outlet (e.g., `ffmpeg_p20_a` at 30 Hz)
  - Shared LSL clock remains available for backward compatibility; device streams allow precise per-device alignment

- 2026-01-20 — FFmpeg multicap tool with LSL clock for multi-device capture
  - Added `tools/ffmpeg_multicap.py` to orchestrate parallel ffmpeg captures from multiple Panacast devices (P20/P50) and DPA audio-only inputs
  - Config-driven via JSON; each device spawns its own ffmpeg subprocess with AVFoundation inputs (avfoundation on macOS)
  - Publishes shared LSL clock for physiological data alignment (configurable stream name/rate)
  - H.264 video (MKV) or WAV audio-only; graceful shutdown via Ctrl+C
  - Event logging to JSONL (capture_started, capture_completed, errors)
  - Requires ffmpeg and pylsl; avoids complex GStreamer plugin setup
  - Config template: `configs/ffmpeg_multicap.json` (4× P20, 2× P50, 4× DPA example)

- 2026-01-20 — GStreamer multicap tool with LSL clock
  - Added `tools/gst_multicap.py` to run multiple avfoundation pipelines (P20/P50 video+audio, DPA audio) under a shared Gst clock
  - Config-driven via JSON; writes MKV (or WAV for audio-only) and emits shared clock over LSL for physiological alignment
  - Requires GStreamer (x264enc, matroskamux, avenc_aac or faac, wavenc), PyGObject, and pylsl installed in the environment

- 2026-01-20 — Panacast USB device selection (indexes or names)
  - `collect --mode usb` accepts `--usb-video-index/--usb-audio-index` for explicit AVFoundation selection
  - Optional `--usb-device-name` and `--usb-target {p20,p50}` to align capture with the intended device when multiple Jabra units are connected
  - Metrics now log the chosen input mode (index vs name)

- 2026-01-20 — Panacast USB zoom control detection (network-based)
  - Added `_try_disable_usb_device_zoom()` to detect and disable zoom on USB devices if network-accessible
    - Searches common subnets (192.168.x.x, 10.0.0.x, 172.16.x.x)
    - Attempts to match device by serial number
    - If found on network: Uses REST API to disable zoom automatically
    - If not found: Provides instructions for manual web UI configuration
  - USB collection now attempts zoom control at startup (default: disabled)
  - Added `--keep-zoom` support for USB mode (to keep zoom enabled)
  - Created [docs/panacast_usb_disable_zoom.md](docs/panacast_usb_disable_zoom.md) with detailed instructions
  - Intelligent framing/zoom is device-level feature, applies to all outputs
  - All tests passing (22/22)

- 2026-01-20 — Jabra Panacast zoom control (network mode)
  - Added `disable_zoom()` and `enable_zoom()` methods to `PanacastAPIClient` class
    - Attempts multiple REST API endpoint variants for device compatibility
    - Disables automatic zoom/framing/intelligent framing features
    - Falls back gracefully if endpoints unavailable (older firmware)
  - Added `--keep-zoom` CLI flag to `collect` command
    - **Default (no flag):** Automatic zoom disabled → fixed video framing
    - **With `--keep-zoom`:** Automatic zoom enabled → auto-tracking
    - Applied at collection start for all registered network devices
  - Updated `PanacastCollector.start_collection()` to accept `disable_zoom` parameter
  - Default zoom setting chosen for research scenarios (consistent backgrounds)
  - Documentation updated with zoom control examples and use cases
  - All tests passing (22/22)
  - Code quality checks passing (ruff, pytest)

- 2026-01-20 — Jabra Panacast USB video + audio capture (fixes and device discovery)
  - Fixed `src/affectai_capture/devices/jabra_panacast.py` USB capture implementation
    - **Issue discovered:** Panacast 20 over USB supports max 1920x1080@30fps (not 4K)
    - **Fix:** Separate video and audio input handling with proper AVFoundation syntax
      - Video input: `ffmpeg -f avfoundation -framerate 30 -i "Jabra PanaCast 20" ...`
      - Audio input: `ffmpeg -f avfoundation -i ":Jabra PanaCast 20" ...`
      - Critical: `-framerate 30` must come BEFORE device name (not after)
    - **Output:** Single MP4 with H.264 video (1920x1080@30fps, CRF 28) + AAC audio (128kbps stereo)
    - **Result:** Working end-to-end video+audio capture from USB device
  - Updated documentation with actual device capabilities (1920x1080, not 4K)
  - Updated all examples and troubleshooting guides
  - Created [examples/panacast_usb_video_demo.md](examples/panacast_usb_video_demo.md) with working commands

- 2026-01-19 — Jabra Panacast (20 & 50) device adapter + multi-device collection + USB support
  - Created `src/affectai_capture/devices/jabra_panacast.py` (850+ lines)
    - **Network mode:** Multi-device orchestration via REST API (device discovery, status polling)
      - RTMP/RTSP/MJPEG stream capture using ffmpeg subprocess
      - Real-time metrics logging (audio/video) to JSONL
      - Graceful shutdown with signal handling
      - Optional LSL marker integration (for task synchronization)
    - **USB mode:** Direct USB audio capture for locally-connected devices (macOS/Linux)
      - Auto-detect USB Panacast devices via system_profiler/lsusb
      - Audio device enumeration
      - WAV audio capture via ffmpeg (48kHz, 16-bit, mono)
      - Device metadata and event logging to JSONL
    - Optional LSL marker integration (for task synchronization)
    - Supports both Panacast 20 (8-mic, 4K) and Panacast 50 (13-mic, 4K dual NDI)
  - Created `tests/test_jabra_panacast.py` (239 lines, 100% coverage)
    - Unit tests for API client, stream capture, metrics, orchestration
    - Mock-based testing for device endpoints
  - Created `docs/jabra_panacast_guide.md` (comprehensive deployment guide)
    - Hardware setup (network, power, positioning)
    - Software prerequisites and installation
    - 5 usage patterns (discovery, single/multi-device, auto-discovery, LSL integration)
    - BIDS output structure and metrics format
    - Python API examples (basic, advanced, metrics monitoring)
    - Troubleshooting guide (network, streams, audio, ffmpeg)
    - REST API endpoint reference + stream protocol specs
    - Integration with AffectAI protocol (timing, synchronization, metadata)
    - Performance specifications (bitrate, network, storage)
    - Maintenance procedures (firmware, health checks)
  - Updated `docs/llm/context_snapshot.md` (added Jabra module to core tooling)
  - All checks passing: ruff ✅ + pytest ✅ (22 tests)

- 2026-01-13 — repository structure optimization proposal + navigation hub
  - Created `ARCHITECTURE.md` (root-level navigation hub)
    - Quick-decision tree for all users ("I want to...")
    - Directory map with purpose annotations
    - Essential commands reference
    - Status snapshot
    - Quick links to all key documents
  - Created `docs/llm/STRUCTURE_OPTIMIZATION.md` (comprehensive refactoring analysis)
    - Audit of documentation sprawl (~48 files, 37% redundancy)
    - Consolidation targets (EmotiBit: 5→2, Stimulus: 2→1, Architecture: 4→3)
    - Unified Copilot discipline proposal (3 files → 1)
    - Progress tracking automation (lightweight + Makefile integration)
    - Phased migration plan (5 phases, ~5 days, zero breaking changes)
  - Created `docs/llm/OPTIMIZATION_SUMMARY.md` (2-page executive summary)
    - Quick summary of issues identified
    - Expected outcomes (32-40% improvements)
    - Two implementation paths (incremental vs. full)
    - Recommendations for adoption
  - Updated `docs/llm/README.md` (expanded with navigation guidance)
    - Added quick-start section (15 min to productivity)
    - Added Copilot workflow section
    - Added status snapshot
    - Integrated optimization materials
  - Updated `docs/llm/context_snapshot.md` (reference optimization hub)
  - All checks passing; analysis-only, non-breaking

- 2026-01-13 — DevOps prompt library (standard templates for Copilot collaboration)
  - Created `docs/llm/DEVOPS_PROMPT_LIBRARY.md` (1,000+ lines)
    - 6 golden rules for Copilot prompting (bootstrap context, ADRs, protocol as source of truth, etc.)
    - 6 task-specific prompt templates (adding features, fixing contradictions, managing drift, etc.)
    - 3 DevOps-specific patterns (IaC, dependency updates, monitoring)
    - Meta guidance: Do's, Don'ts, quality checklist
    - File purpose matrix, lessons learned, escalation rules
  - Updated `docs/llm/context_snapshot.md` to reference DevOps prompt library

- 2026-01-12 — knowledge management improvements (Phase 1 + Phase 2)
  - **Documentation alignment:** Fixed code/doc contradictions identified in assessment
    - Updated `docs/llm/context_snapshot.md`: EmotiBit now shows LSL + JSONL output
    - Updated `docs/lsl_architecture_alignment.md`: Gap 1 marked ✅ RESOLVED (2026-01-12)
    - Added Implementation Status tracking table to lsl_architecture_alignment.md
  - **Knowledge graph expansion:** Added 4 new nodes, 6 new edges
    - New nodes: lsl_architecture_alignment, emotibit_lsl_demo, lsl_stream_viewer, lsl_streams output
    - New edges: emotibit → LSL, stimulus_markers → LSL streams, data flow paths
  - **Prompt playbook expansion:** Added 4 new playbook entries
    - Handling contradictory documentation (escalation workflow)
    - Verifying LSL streams (troubleshooting guide)
    - Adding new device adapters (8-step checklist)
    - Updating knowledge graph (validation workflow)
  - **Assessment deliverables:** Created comprehensive LLM readiness analysis
    - `docs/llm/KNOWLEDGE_ASSESSMENT.md` (745 lines, full analysis)
    - `docs/llm/CONTRADICTIONS.md` (205 lines, quick-fix reference)
    - `docs/llm/ASSESSMENT_SUMMARY.md` (291 lines, stakeholder brief)
    - `docs/llm/README.md` (266 lines, navigation hub)
  - Validated with `tools/build_kg.py` (no dangling edges)
  - All checks passing (`make check`)

- 2026-01-09 — added lightweight knowledge graph for LLM navigation
  - Added `docs/llm/knowledge_graph.json` with nodes/edges for core docs, modules, and outputs
  - Added `tools/build_kg.py` to validate and normalize the graph
  - Updated `docs/llm/context_snapshot.md` to reference the knowledge graph

- 2026-01-12 — added LSL streaming to EmotiBit integration
  - Enhanced `src/affectai_capture/devices/emotibit.py` with real-time LSL streaming (enabled by default)
  - Creates separate LSL outlet per device+channel (e.g., EmotiBit_10_49_228_101_ppg_green)
  - Stream types: PPG, EDA, Temperature, IMU, Physio with proper channel metadata
  - Uses `pylsl.local_clock()` for synchronized timestamps across all modalities
  - Supports multi-device (4× EmotiBit devices identified by source IP)
  - CLI: `--lsl` (default), `--no-lsl` to disable LSL streaming
  - Maintains backward compatibility: still writes JSONL files for archival/debugging
  - Resolves Gap 1 from `docs/lsl_architecture_alignment.md`
  - Added `docs/lsl_architecture_alignment.md` documenting LSL integration gaps and roadmap

- 2026-01-08 — updated EmotiBit integration with accurate packet format from ofxEmotiBit v1.12.2 source analysis
  - Reviewed ofxEmotiBit-1.12.2 source code (EmotiBitOscilloscope, EmotiBitDataParser, EmotiBitTestingHelper)
  - Updated packet parsing documentation: 6-field header (timestamp, packet_count, data_length, type_tag, protocol_version, data_reliability)
  - Expanded CHANNEL_NAMES dictionary with all EmotiBit channels: physiological (PPG, EDA electrodes), temperature (T0/TH/T1), IMU (accel/gyro/mag), derived metrics (HR, IBI, SCR), system (battery, errors)
  - Updated docs/emotibit_integration.md with complete channel reference table organized by category
  - Source: ofxEmotiBit-1.12.2 (C++/openFrameworks codebase included in src/affectai_capture/devices/)

- 2026-01-08 — added live visualization to EmotiBit integration (HR + EDA real-time plotting)
  - Enhanced `src/affectai_capture/devices/emotibit.py` with `--visualize` flag
  - Added `HeartRateEstimator` class (simple peak detection from PPG Green, 10-sec rolling window)
  - Added `LiveVisualizer` class (matplotlib-based dual plots: HR + EDA, 30-sec sliding window)
  - Optional dependency: matplotlib (gracefully degrades to headless if unavailable)
  - Updated `docs/emotibit_integration.md` with visualization guide and requirements

- 2026-01-08 — added EmotiBit UDP streaming integration for physiological data capture
  - Added `src/affectai_capture/devices/emotibit.py` with UDP listener (port 12346), CSV packet parser, JSONL writer per channel
  - Channels: PPG (Green/Red/Infrared), EDA, Temperature, Accelerometer, Gyroscope, Magnetometer
  - Output: `<SESSION_DIR>/sourcedata/physio/emotibit/*.jsonl` (one file per channel)
  - CLI: `python -m affectai_capture.devices.emotibit --host 0.0.0.0 --port 12346 --session <PATH>`
  - Added `docs/emotibit_integration.md` with protocol details, sync strategy, troubleshooting
  - Standard library only; graceful shutdown via SIGINT/SIGTERM; all ruff/pytest checks passing

- 2026-01-04 — added methodology guide and scaffolds for assets/templates
  - Added `docs/METHODOLOGY.md` documenting protocol-driven, spec-first workflow
  - Added `media/` and `templates/` skeletons; refreshed `docs/STRUCTURE_ALIGNMENT.md` and updated alignment score

- 2026-01-04 — added prompt scheduling scaffolding and corrected small-variant timing
  - Added `src/affectai_capture/prompts.py` with prompt metadata (V–A grid, rotating probes, BFI-45 placeholders) and scheduling helpers
  - Added `tests/test_prompts.py` covering spacing, caps, and default/periodic plans
  - Small variant session duration confirmed 60–75 min (T1: 10, T2: 8, T3: 10, T4: 5) with fixed ~3 min probes (flexible delivery)

- 2026-01-04 — added methodology guide, templates, and media scaffolds
  - Added `docs/METHODOLOGY.md` documenting protocol-driven, spec-first workflow
  - Added `templates/feature-spec.md`, `templates/device-adapter.py`, `templates/experiment-protocol.md`
  - Added `media/diagrams/README.md` and `media/logo/README.md` for assets; updated README and structure alignment

- 2026-01-04 — expanded small-variant protocol to include all T1–T4 tasks
  - Updated `docs/stimulus_design_small.md` with T3 (idea generation) and T4 (public-goods game) task definitions
  - Initial session duration estimate 90–110 min (T1: 15, T2: 15, T3: 15, T4: 10) — superseded by 60–75 min correction above
  - Synchronized prompting across all four tasks (V–A grid, cognitive/fairness probes, ≥2 min spacing)
  - Updated `configs/lab_small.yaml` and `src/affectai_capture/manifest_small.py` to reflect full T1–T4 scope
  - Small variant now contains complete AffectAI task battery (decision-making, negotiation, creativity, cooperation)
- 2026-01-04 — implemented AffectAI small-variant protocol (T1–T2 MSc pilot)
  - Updated `docs/sources/aux/Affect_AI_data_collection_small.source_card.md` with concrete specs: 4 Tobii + 4 DPA + Jabra + 6 Vicon + tablet BFI-45
  - Created `configs/lab_small.yaml` with full hardware, synchronization, quality-control checklist aligned with main protocol
  - Created `docs/stimulus_design_small.md` defining T1–T2 task flow, in-situ prompts, events.tsv markers (preserves synchronization discipline)
  - Created `src/affectai_capture/manifest_small.py` with BIDS structure, modality checklist, file-validation functions
  - Updated `docs/llm/context_snapshot.md` to reflect small variant + Overleaf GitHub Sync integration
  - All checks passing; small variant ready for MSc-level pilot execution
- 2026-01-04 — integrated Overleaf LaTeX protocol documents via GitHub Sync
  - Updated `.gitignore` to ignore LaTeX build artifacts (`.pdf`, `.aux`, `.log`, `.synctex.gz`, etc.)
  - Removed PDFs from git tracking via `git rm --cached`; committed transition
  - Documented Overleaf GitHub Sync approach (preferred over direct git clone)
  - Protocol source now version-controlled as `.tex` (text-based, LLM-readable)
- 2026-01-03 — added team collaboration and change tracking infrastructure
  - Created `TEAM.md` defining roles, workflows, branch protection, and onboarding
  - Created `.github/CODEOWNERS` for automatic review assignment by code area
  - Created issue templates: bug_report, feature_request, protocol_question
  - Documented GitHub repository settings for access control and branch protection
  - Prepared for multi-user collaborative development with role-based access
- 2026-01-03 — aligned repository structure with GitHub Spec-Kit standards
  - Added critical legal/governance files: LICENSE (MIT), SECURITY.md, CODE_OF_CONDUCT.md
  - Added SUPPORT.md for user/developer guidance
  - Created `.devcontainer/devcontainer.json` and `post-create.sh` for reproducible development
  - Created `docs/STRUCTURE_ALIGNMENT.md` comparing with Spec-Kit and documenting recommendations
  - Now 78% aligned with Spec-Kit best practices; exceeds in protocol-driven design and LLM integration
- 2026-01-02 — aligned stimulus design with protocol task definitions
  - Updated `docs/stimulus_design.md` to match T1–T4 task blocks from `tasks_description.md`
  - Revised `stimuli/tablet_questionnaire.py` to implement labels per `labels_codebook.md`
  - Changed post-block scales: replaced "stress" with "mental_demand"; added task-specific integrity checks
  - Comprehensive event marker taxonomy aligned with task phases and prompt timing rules
- 2026-01-02 — added stimulus design and tablet questionnaire
  - Created `docs/stimulus_design.md` defining task blocks, event markers, and session timeline
  - Implemented `stimuli/tablet_questionnaire.py` (PsychoPy) for post-block self-annotations
  - Added `src/affectai_capture/stimulus.py` for standardized event marker helpers
  - Documented tablet deployment (PsychoJS) and data integration in `stimuli/README.md`
- 2026-01-02 — fixed code compatibility issues
  - Migrated ruff config from deprecated top-level to `lint` section in pyproject.toml
  - Fixed mutable default argument in LSLMarkerOutlet (B008)
  - Auto-fixed import sorting across all Python files
- 2025-12-31 — repo scaffold created
  - Added Python capture scaffold, sources sync, and progress reporting skeleton.
