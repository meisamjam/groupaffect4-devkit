# Semantic Biomarkers Catalog (AffectAI Tasks T0-T4)

## Purpose
Define interpretable, task-aware biomarkers built from multimodal physiological features (EmotiBit + Tobii), suitable for cognitive and conversation-dynamics analyses.

## Construction Principles
- Use latent indices (multi-feature composites), not single-signal claims.
- Normalize within participant using `T0` baseline when available.
- Compute at both task-level and rolling-window level.
- Store uncertainty/confidence for missing or low-quality segments.

## Biomarker Definitions

## 1) Cognitive Load Index
- Signals: pupil dilation mean/slope, HRV suppression (RMSSD/SDNN), tonic EDA.
- Interpretation: higher index = higher mental effort.
- Primary tasks: `T1`, `T3`.

## 2) Arousal / Stress Reactivity
- Signals: SCR rate/amplitude, HR acceleration proxy, pupil volatility.
- Interpretation: transient sympathetic activation.
- Primary tasks: `T2`, `T4`.

## 3) Sustained Attention
- Signals: pupil stability, gaze validity continuity, reduced random fluctuation.
- Interpretation: stable attentional engagement over interval.
- Primary tasks: `T1`, `T3`.

## 4) Decision Pressure
- Signals: pre-decision rise in EDA + HR + pupil in final pre-choice window.
- Interpretation: escalating pressure during commitment periods.
- Primary tasks: `T1` candidate selection, `T2` settlement, `T3` group selection, `T4` contribution/reveal.

## 5) Social Synchrony
- Signals: dyadic/group coupling metrics (EDA/PPG/pupil correlations + lag structure).
- Interpretation: physiological alignment (co-regulation) in interaction.
- Primary tasks: `T2`, `T3`, `T4`.

## 6) Conversation Dominance Strain
- Signals: repeated asymmetric arousal during speaking/turn windows.
- Interpretation: one participant carries disproportionate physiological load.
- Primary tasks: `T2`, `T4`.

## 7) Conflict / Tension Episodes
- Signals: simultaneous multi-person EDA bursts + unstable coupling.
- Interpretation: acute interpersonal tension moments.
- Primary tasks: `T2`, `T4`.

## 8) Engagement / Involvement
- Signals: moderate sustained arousal + stable attention profile.
- Interpretation: active but regulated participation.
- Primary tasks: `T1`-`T4`.

## 9) Recovery / Regulation Capacity
- Signals: post-peak return rates (EDA down-slope, HR normalization, pupil reset).
- Interpretation: resilience after challenge peaks.
- Primary tasks: all; especially `T4` after reveal.

## 10) Fatigue / Cognitive Depletion
- Signals: decline in pupil baseline/reactivity, reduced physiological responsiveness, drift trends.
- Interpretation: accumulated load across session.
- Primary tasks: longitudinal from `T1` to `T4`.

## Intuitive Labels for Reporting
- `Calm Focus`
- `Effortful Thinking`
- `Escalating Tension`
- `Socially Aligned`
- `Socially Fragmented`
- `Recovered / Regulated`
- `Decision Crunch`

## Data and Output Requirements
- Inputs:
  - `physio/*_acq-P*_emotibit.tsv.gz`
  - `et/*_acq-P*_tobii.tsv.gz`
  - `annot/*_task_run_windows.tsv`
  - `events.tsv`
- Output columns should include:
  - `session_id, task, participant_id, window_start_lsl, window_end_lsl`
  - biomarker value columns
  - `quality_flag, coverage_pct, missing_modalities`

## Validation Plan
- Internal consistency: check expected directions (e.g., decision-pressure windows > neutral windows).
- Task contrast sanity checks:
  - `T0` lower arousal than intense segments in `T2`/`T4`
  - higher social synchrony during collaborative segments than fragmented segments.
- Missingness stress test:
  - compare full-coverage vs partial-coverage estimates.
