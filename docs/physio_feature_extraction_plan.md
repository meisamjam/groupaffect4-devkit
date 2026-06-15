# Physiological Feature Extraction Plan (Task-Aware, Data-Structure-Aware)

## Goal
Build a reproducible feature extraction pipeline for cognitive load and conversation dynamics using available modalities:
- EmotiBit: PPG, EDA/GSR, temperature, IMU (from `physio/*.tsv.gz`)
- Tobii: pupil + gaze validity (from `et/*.tsv.gz`)
- Task/events timing spine (from `events.tsv`, `annot/*task_run_windows.tsv`)

## Scope of This Plan
- Dataset root (seed): `affectai-data-processing-seed/data/sub-01/ses-*/`
- Main analyzed tasks: `T1`-`T4`
- Baseline/regulation reference: `T0`
- Analysis unit levels:
  - Participant-task summary
  - Participant 30s rolling windows
  - Event-centered windows (before/after key task events)

## Data Structures to Use
- Task windows: `annot/sub-01_*_task-T0T1T2T3T4_task_run_windows.tsv`
- Participant-signal mapping: `annot/sub-01_*_participant_signal_map.tsv`
- EmotiBit pooled: `physio/sub-01_*_task-*_acq-lsl_emotibit.tsv.gz`
- EmotiBit participant split: `physio/sub-01_*_task-*_acq-P{1..4}_emotibit.tsv.gz`
- Tobii pooled: `et/sub-01_*_task-*_acq-lsl_tobii.tsv.gz`
- Tobii participant split: `et/sub-01_*_task-*_acq-P{1..4}_tobii.tsv.gz`
- Timeline spine: `events.tsv`

## Task-Aware Feature Strategy
- `T0`: baseline estimation per participant for normalization.
- `T1` (hidden-profile): attention + decision pressure markers.
- `T2` (negotiation): stress/reactivity + social synchrony.
- `T3` (idea generation/discussion): cognitive effort + group convergence/divergence.
- `T4` (public-goods): tension, fairness pressure, recovery after reveal.

## Features to Extract
1. PPG/HR/HRV
- Heart-rate proxy
- RMSSD, SDNN, pNN50 (when beat-quality is sufficient)
- Pulse amplitude variability

2. EDA/GSR
- Tonic SCL (mean, slope)
- SCR count/rate
- SCR amplitude distribution
- Recovery slope after peaks

3. Temperature
- Mean per task
- Delta vs T0 baseline
- Within-task slope

4. Pupil (Tobii)
- Mean pupil size
- Pupil dilation variability
- Dilation velocity
- Missing/blink fraction

5. Cross-person dynamics (conversation)
- Dyadic synchrony (corr + lagged corr) for EDA/PPG/pupil
- Group coupling (dispersion/convergence over time)
- Event-locked co-activation around task transitions/decisions

## QC and Inclusion Rules
- Enforce participant IDs only (`P1`-`P4`) in analysis outputs.
- Minimum coverage threshold per participant-task window.
- Artifact flags:
  - impossible jumps
  - excessive NaN segments
  - high-motion contamination windows (IMU-informed)
- Keep confidence columns with each derived feature.

## Output Tables
- `features_participant_task.tsv`
  - one row per `session x task x participant`
- `features_participant_window_30s.tsv`
  - one row per rolling window
- `features_group_dynamics_window_30s.tsv`
  - dyad/group synchrony features
- `features_event_locked.tsv`
  - event-centered physiology features

## Implementation Milestones
1. Channel dictionary + schema lock
- map `value_0..N` to semantic channel names per stream type.

2. Core extractors
- `tools/features/extract_physio_features.py`
- `tools/features/extract_pupil_features.py`

3. Dynamics layer
- `tools/features/compute_group_dynamics.py`

4. Dataset assembly
- `tools/features/build_analysis_tables.py`

5. Validation
- add tests for window slicing, channel mapping, missingness/QC behavior.

## Recommended Session Tiers
- Tier A (full P1-P4 EmotiBit + Tobii): primary modeling set.
- Tier B (partial modality coverage): secondary/robustness analyses.

## Privacy/Anonymization Requirement
- Do not propagate real names from `participant_map.tsv` into derived outputs.
- Use only anonymized IDs (`P1`-`P4`, `sub-*`) in features and reports.
