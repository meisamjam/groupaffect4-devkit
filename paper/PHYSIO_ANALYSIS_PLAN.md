# AffectAI NeurIPS 2026 - Physio Analysis Plan

_Working draft: 2026-04-27_  
_Branch: `feature/physio-data-extraction`_

## Purpose

This plan turns the paper-level Project D into an executable physiological
analysis workstream. It follows the current paper strategy:

> dataset characterisation + one conservative worked example + release
> documentation.

For this submission, physiology should be simple, robust, and transparent. The
goal is not a large biomarker library. The goal is to produce defensible
participant-task summaries, coverage/QC statistics, and a small set of features
that can support the worked example if signal quality permits.

## Scope

Included:

- EmotiBit per-participant streams only.
- Tobii pupil size can be added as a companion oculophysiology signal for
  dataset-characterisation figures, but should be interpreted separately from
  wearable physiology because pupil size is sensitive to luminance and display
  context.
- Task-level summaries for `T0` to `T4`.
- `T0` baseline-normalised summaries for `T1` to `T4`.
- Simple rolling-window output only if it helps QC or the worked example.
- QC flags and missingness summaries for every participant-task row.

Deferred:

- Complex semantic biomarkers.
- Strong synchrony or dominance claims.
- Event-locked models around fine-grained conversation events.
- Clinical interpretation of physiology.
- Any analysis requiring real participant names.

## Canonical Outputs

The paper plan expects these canonical files:

| Output | Purpose |
|---|---|
| `features/physio_participant_task.tsv` | Main participant-task feature table for paper/worked example. |
| `features/physio_qc_summary.tsv` | Session, participant, task, and feature-level coverage/QC summary. |
| `features/physio_window_30s.tsv` | Optional rolling-window table for diagnostics or simple dynamics. |
| `paper/tables/physio_feature_definitions.tsv` | Short feature dictionary for paper/release documentation. |

Companion pupil/autonomic outputs:

| Output | Purpose |
|---|---|
| `features/features_pupil_participant_task.tsv` | Tobii pupil task summaries. |
| `features/features_pupil_window_30s.tsv` | Tobii pupil rolling-window summaries. |
| `results/autonomic/autonomic_paper_key_findings.tsv` | Ranked paper-facing findings across EmotiBit + pupil. |
| `figures/autonomic/autonomic_task_fingerprint_heatmap.png` | Compact task-by-signal fingerprint. |
| `figures/autonomic/autonomic_modality_coverage.png` | Session/task coverage for usable physio + pupil rows. |

Existing tooling currently writes:

- `features_physio_participant_task.tsv`
- `features_physio_window_30s.tsv`

For the paper, either update the extractor to write the canonical names above
or write a thin compatibility export step. Avoid keeping two competing tables.

## Inputs

Required per session:

- `physio/*_task-T*_run-01_acq-P*_emotibit.tsv.gz`
- `annot/*_task_run_windows.tsv`
- `annot/*_participant_signal_map.tsv`
- `events.tsv`
- `metadata/analysable_sessions.tsv` once Project A is available

Useful cross-checks:

- `metadata/session_metadata_report.tsv`
- `metadata/modality_coverage.tsv`
- EmotiBit participant maps in `configs/`

## Feature Set

### Primary features for paper use

Keep these as the default paper feature set:

| Family | Feature | Notes |
|---|---|---|
| Coverage | `physio_available`, `duration_s`, `coverage_pct` | Required for all rows. |
| PPG/HR | `hr_mean_bpm`, `hr_sd_bpm` | Prefer derived/validated HR if present; otherwise mark proxy clearly. |
| HRV | `hrv_rmssd_ms` | Include only when beat detection passes QC. |
| EDA | `eda_tonic_mean`, `eda_tonic_slope`, `eda_phasic_rate` | Phasic rate must be QC-flagged if peak detection is crude. |
| Temperature | `temp_mean`, `temp_slope` | Useful as slow trend/context feature. |
| Motion | `motion_mean`, `motion_high_fraction`, `motion_flag` | Use IMU where available to interpret noisy PPG/EDA. |
| Baseline | `*_delta_t0`, `*_z_t0` | Compute for `T1`-`T4` when `T0` exists for the same participant. |

### Features to avoid for this deadline

- Large frequency-domain HRV feature sets.
- Many EDA decomposition variants with unclear parameter choices.
- Person-to-person synchrony metrics as a headline result.
- Any feature that needs extensive manual tuning per session.

## QC Rules

Every participant-task row should have a `qc_flag` and a `qc_notes` field.
Use missing values plus flags instead of filling unreliable features.

Suggested minimum rules:

| Rule | Flag |
|---|---|
| No participant-task physio file | `missing_physio` |
| Duration less than 70% of task window | `short_duration` |
| Usable finite samples below 80% for a signal | `low_coverage` |
| Median sample rate outside expected device range | `sample_rate_unusual` |
| HR outside plausible adult range after QC | `hr_implausible` |
| Too few clean PPG peaks for HRV | `hrv_unreliable` |
| Temperature outside plausible skin range | `temp_implausible` |
| High motion during window/task | `motion_contaminated` |
| Channel mapping not confirmed | `channel_map_unconfirmed` |

Recommended feature inclusion:

- Use HR/EDA/temperature means when coverage is acceptable.
- Use HRV only if peak intervals are physiologically plausible and enough
  clean beats are available.
- Use EDA phasic rate only as descriptive unless peak extraction has been
  spot-checked.
- Keep rows with partial data, but expose which feature families are usable.

## Implementation Plan

### 1. Lock the session set

Use Project A outputs when ready:

- `metadata/analysable_sessions.tsv`
- `metadata/modality_coverage.tsv`

Before that exists, use only clearly complete final-phase sessions and mark the
list as provisional. Do not silently expand the session set after the worked
example is specified.

### 2. Confirm EmotiBit schema

For a small sample of task-split files:

- inspect columns and `value_*` order,
- confirm PPG, EDA, temperature, and IMU channel positions,
- check whether any streams include derived HR,
- write the confirmed mapping into a small feature-definition note.

This is the most important technical guardrail because the existing extractor
uses configurable `value_*` indices.

### 3. Harden `extract_physio_features.py`

Current tool: `tools/features/extract_physio_features.py`.

Needed for the paper:

- canonical output names,
- `task_id` column rather than only `task`,
- explicit `ppg_available`, `eda_available`, `temp_available`, `imu_available`,
- `qc_flag` and `qc_notes`,
- baseline-normalised `T1`-`T4` deltas against `T0`,
- `features/physio_qc_summary.tsv`,
- deterministic row order,
- tests for missing channels, low coverage, and baseline normalisation.

Keep the old output names only if another existing script depends on them, and
document the alias clearly.

### 4. Generate the first feature pass

Run on the agreed session set:

```bash
python tools/features/extract_physio_features.py \
  --data-root <bids_or_processed_data_root> \
  --out-dir features \
  --sessions <session_id ...> \
  --window-s 30 \
  --step-s 15
```

Then immediately inspect:

- row counts: expected `sessions x tasks x participants`,
- missing participant-task rows,
- per-feature missingness,
- sample-rate distribution,
- outlier HR/EDA/temp values,
- whether `T0` baseline exists per participant.

### 5. Produce QC summary

`features/physio_qc_summary.tsv` should support the paper table and limitations.
Suggested rows can be one of:

- session-level summary,
- participant-task summary,
- feature-family summary.

Minimum columns:

```text
session_id
participant_id
task_id
physio_available
ppg_usable
eda_usable
temp_usable
imu_usable
duration_s
coverage_pct
qc_flag
qc_notes
```

### 6. Choose worked-example variables

Preferred worked-example framing:

> How do simple physiology summaries change across structured group tasks,
> relative to each participant's baseline, and how do those summaries align
> descriptively with available task outcomes or valence/arousal probes?

Primary variables:

- `hr_mean_bpm_delta_t0`
- `eda_tonic_mean_delta_t0`
- `eda_phasic_rate_delta_t0`
- `temp_slope`
- `motion_flag`

Recommended output:

- one compact task-by-feature heatmap or dot plot,
- one short table of coverage and usable rows,
- one optional pupil-size panel or cross-modal scatter if Tobii validity is
  high enough,
- optional descriptive association with task outcome/VAD if Project F is ready.

Statistical posture:

- descriptive summaries first,
- within-participant baseline normalisation,
- show uncertainty where simple,
- treat pupil diameter as oculophysiology/display-context sensitive unless a
  luminance correction is added,
- avoid strong significance or prediction claims.

### 7. Paper integration

Physio should feed four paper locations:

- `paper/tables/modalities.tex`: confirmed channel/rate wording.
- `paper/tables/dataset_stats.tex`: EmotiBit coverage percentage.
- `paper/main.tex` Section 4: acquisition and signal description.
- `paper/main.tex` Sections 6-7: worked example and QC/missingness.

Final text should state clearly:

- which sessions/tasks had usable EmotiBit data,
- which feature families were used,
- which were excluded or caveated,
- that wearable physiology is sensitive to motion/contact quality.

## Deliverable Checklist

By the physio handoff:

- [ ] Session list fixed or marked provisional.
- [ ] EmotiBit channel map confirmed.
- [ ] `features/physio_participant_task.tsv` created.
- [ ] `features/physio_qc_summary.tsv` created.
- [ ] Optional `features/physio_window_30s.tsv` created only if needed.
- [ ] Feature-definition note/table created.
- [ ] QC caveats written in plain language.
- [ ] Paper numbers signed off by the physio owner.

## Fallback Plan

If PPG/HRV quality is weak:

- keep EDA tonic, temperature, and coverage/QC summaries;
- drop HRV from the worked example;
- describe PPG/HRV as released but not used analytically.

If EDA peak detection is weak:

- keep tonic EDA mean/slope;
- drop or caveat phasic-event rate.

If all physio features are too noisy for the worked example:

- use physiology only in dataset-characterisation and QC tables;
- run the worked example using task outcomes/self-report/audio if available;
- explicitly state that physio feature validation is included as release
  infrastructure but not used for a substantive result in this submission.
