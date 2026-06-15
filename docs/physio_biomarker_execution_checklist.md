# Physio Biomarker Execution Checklist

## Objective
Operational checklist to run feature and semantic biomarker extraction across the current seed dataset with task awareness and data-availability awareness.

## A) Preflight (Per Session)
1. Confirm required files exist:
- `events.tsv`
- `annot/*_task_run_windows.tsv`
- `annot/*_participant_signal_map.tsv`
- at least one `physio/*_acq-P*_emotibit.tsv.gz`
- at least one `et/*_acq-P*_tobii.tsv.gz`

2. Confirm anonymized analysis IDs:
- only `P1`-`P4` in outputs.
- do not export real names from `participant_map.tsv`.

3. Record data coverage:
- participants with EmotiBit data
- participants with Tobii data
- overlap count (both modalities)

## B) Task Segmentation Rules
- Use task windows from `annot/*_task_run_windows.tsv`.
- Treat:
  - `T0` as baseline reference
  - `T1`-`T4` as target tasks for model features
- Optional: derive event-locked windows from `events.tsv` for:
  - `tobii_calibration`
  - `finish`
  - decision/selection related events

## C) Feature Extraction Order
1. EmotiBit features (PPG/EDA/temp + QC)
2. Tobii pupil features (validity-aware)
3. Merge by `session, task, participant, lsl_time/window`
4. Build participant-task and rolling-window aggregates
5. Compute dyad/group synchrony metrics
6. Compute semantic biomarker composites

## D) Quality Gates
- Drop/flag windows under minimum coverage threshold.
- Flag physiologically implausible jumps.
- Flag excessive NaN stretches.
- Carry `quality_flag` and `coverage_pct` in all downstream tables.

## E) Session Tiering for Analysis
Recommended primary sessions (full P1-P4 EmotiBit + Tobii in seed data):
- `ses-20260312_grp-07_run01`
- `ses-20260313_grp-08_run01`
- `ses-20260316_grp-08_run01`
- `ses-20260318_grp-13_run01`
- `ses-20260319_grp-14_run01`
- `ses-20260319_grp-15_run01`

Secondary sessions (partial overlap, still useful with missingness handling):
- `ses-20260317_grp-09_run01`
- `ses-20260318_grp-12_run01`

## F) Planned Output Artifacts
- `features_participant_task.tsv`
- `features_participant_window_30s.tsv`
- `features_group_dynamics_window_30s.tsv`
- `semantic_biomarkers_participant_task.tsv`
- `semantic_biomarkers_window_30s.tsv`
- `qc_feature_coverage_report.tsv`

## G) Suggested CLI Layout (to implement)
- `python tools/features/extract_physio_features.py --data-root ... --sessions ...`
- `python tools/features/extract_pupil_features.py --data-root ... --sessions ...`
- `python tools/features/compute_group_dynamics.py --features-dir ...`
- `python tools/features/build_semantic_biomarkers.py --features-dir ... --config docs/semantic_biomarkers_catalog.md`

## H) Reproducibility Requirements
- Log software version + timestamp in each run.
- Log included/excluded sessions and reasons.
- Preserve deterministic sort/order for all exported tables.
