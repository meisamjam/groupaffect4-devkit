# AffectAI NeurIPS 2026 — Analysis & QC Plan

_Working draft: 2026-04-27_  
_Internal freeze target: Sunday, May 3 EOD_  
_Hard deadline: Wednesday, May 6 AoE_

## Purpose

This document describes the analysis and quality-control work needed for the
NeurIPS 2026 submission.

It is aligned with the current submission strategy:

> **dataset-characterisation paper + one conservative worked example + release documentation**

This replaces the earlier benchmark-heavy plan. The goal is to produce honest,
reproducible, and defensible statistics that support the dataset paper.

The document should be read as a working plan for team discussion. Owners,
deadlines, and fallbacks can be adjusted if needed, but the overall principle is
to avoid late expansion and avoid overclaiming.

---

## 1. Analysis framing

The paper should be framed as a dataset-characterisation paper rather than a
leaderboard or large-scale benchmark paper.

The main analytical contribution is to document:

1. what data were collected,
2. which sessions and modalities are analysable,
3. how the modalities are synchronised,
4. what the main missingness patterns are,
5. what task outcomes and self-report variables are available,
6. what release artefacts are available,
7. and one example of how the dataset can be used.

The worked example should be modest. It should illustrate the dataset structure
and analytical possibilities, not claim a definitive model of group affect or
performance.

Dominance analysis, video/3D pose, world-frame gaze, and multiple baselines are
not part of this submission. They are reserved for follow-up work.

---

## 2. Roles and deliverables

| Person | Suggested analysis/QC role | Main deliverables |
|---|---|---|
| **Meisam** | Sync/coverage, worked-example framing, release/paper integration | analysable-session list, modality coverage, C specification, release metadata, final paper |
| **Anna** | Physio and audio/prosodic features | physio feature table, audio/prosodic feature table, QC notes |
| **Alice** | Task audit and worked-example execution | task outcome table, descriptive task statistics, C results/figures |
| **Shan** | Transcription and optional content labels | transcripts, transcript QC, optional labels + agreement |

These are suggested responsibilities. If one area slips, the first response
should be to simplify scope rather than push unstable results into the paper.

---

## 3. Critical path

The core dependency chain is:

```text
A: analysable-session list
    ↓
D/E/F: feature tables + task outcomes
    ↓
C: worked example
    ↓
I: paper integration
```

Transcripts are useful but should not endanger the core submission:

```text
B: transcripts
    ↘ supports E/F/C if stable
    ↘ supports G only if B finishes early and quality is acceptable
```

Release infrastructure runs in parallel:

```text
H: release infrastructure
    ↘ depends on A/B/D/E/F artefact status
    ↘ should describe only what is actually ready
```

The worked example should be robust to missing components. If one modality is
not ready, the worked example should either run on the available subset or be
reframed descriptively.

---

## 4. Core artefacts and canonical paths

To avoid conflicting versions, each output should have one canonical file path.

| Artefact | Suggested path | Owner |
|---|---|---|
| Analysable-session list | `metadata/analysable_sessions.tsv` | Meisam |
| Modality coverage table | `metadata/modality_coverage.tsv` | Meisam |
| Session-level QC notes | `metadata/session_qc_notes.tsv` | Meisam |
| Physio participant-task features | `features/physio_participant_task.tsv` | Anna |
| Physio QC summary | `features/physio_qc_summary.tsv` | Anna |
| Audio/prosody participant-task features | `features/audio_participant_task.tsv` | Anna |
| Audio QC summary | `features/audio_qc_summary.tsv` | Anna |
| Task outcomes | `features/task_outcomes.tsv` | Alice |
| Task descriptives | `features/task_descriptives.tsv` | Alice |
| Worked-example results | `results/worked_example_results.tsv` | Alice + Meisam |
| Worked-example figure | `figures/worked_example.*` | Alice + Meisam |
| Transcripts | `transcripts/<session_id>_transcript.tsv` | Shan |
| Transcript QC | `metadata/transcript_qc_summary.tsv` | Shan |
| Optional content labels | `annotations/content_labels.tsv` | Shan |
| Dataset stats table source | `tables/dataset_stats_source.tsv` | Meisam |
| Final paper numbers checklist | `metadata/final_number_signoff.tsv` | Meisam + all |

The exact paths can be adjusted to match the repository, but each artefact
should have one single source of truth.

---

## 5. A — Sync and coverage characterisation

**Suggested owner:** Meisam

### Goal

Produce the session-level inclusion/exclusion logic and per-modality coverage
statistics used in the paper, release documentation, and limitations section.

This is the foundation for every later analysis. The paper should never imply
that a modality is complete if it is only available for a subset of sessions or
participants.

### Inputs

- raw session trees,
- existing audit documentation,
- BIDS/release conversion outputs,
- QC scripts and metadata reports,
- any manual notes on missing or problematic recordings.

### Minimum outputs

- `metadata/analysable_sessions.tsv`,
- `metadata/modality_coverage.tsv`,
- `metadata/session_qc_notes.tsv`,
- numbers for `tables/dataset_stats.tex`,
- limitations text for missingness.

### Suggested fields for `analysable_sessions.tsv`

```text
session_id
group_id
tier
included_in_headline
included_in_worked_example
reason_if_excluded
has_physio
has_gaze_2d
has_close_talk_audio
has_room_audio
has_vad_probes
has_bfi44
has_task_outcomes
has_transcript
notes
```

### Suggested tier definitions

| Tier | Meaning | Use |
|---|---|---|
| Tier 1 | Core modalities present and no major documented anomaly | Main dataset statistics and worked example |
| Tier 2 | Minor documented gaps but still usable for selected analyses | Included where required modalities are available |
| Tier 3 | Significant gaps | Report for transparency; exclude from headline worked example unless justified |

### Minimum paper outputs

- total number of recorded sessions,
- number of analysable sessions,
- number of unique participants,
- per-modality coverage,
- number of sessions in each tier,
- explicit statement of which sessions were used in the worked example.

### Deadline

- Working list by Tuesday EOD.
- Final numbers by Saturday EOD.

---

## 6. B — Transcription pipeline

**Suggested owner:** Shan  
**Support:** Anna for audio-processing issues; Alice for transcript-quality checks.

### Goal

Produce transcript artefacts that can be described as part of the release. If
quality is good enough, transcripts may support audio/content analyses. If not,
they remain a documented release artefact with clear caveats.

The transcript pipeline should not delay the core dataset-characterisation paper.

### Minimum outputs

- one full session transcribed end-to-end by Tuesday EOD,
- transcripts for the remaining available sessions by Wednesday/Thursday if feasible,
- transcript schema,
- transcript-quality notes,
- recommendation: core artefact / use with caveats / simplify.

### Proposed transcript schema

```text
session_id
onset
duration
speaker
text
confidence
source_channel
qc_flag
```

Optional additional fields:

```text
task_id
task_phase
word_count
asr_model
redaction_status
```

### Quality checks

- Does each speaker correspond reasonably to the intended close-talk channel?
- Does cross-talk suppression remove obvious bleed without deleting real speech?
- Are timestamps aligned well enough for descriptive analyses?
- Are names and identifying mentions redacted?
- Are low-confidence segments marked rather than silently treated as reliable?

### Recommended transcript status categories

| Status | Meaning | Paper use |
|---|---|---|
| Validated | Manual spot-checks support use in analysis | Can support selected analyses |
| Usable with caveats | Automatic transcript appears reasonable but not fully validated | Release artefact; limited descriptive use |
| Unstable | Speaker labels/timestamps/text are unreliable | Do not use analytically; describe limitation |
| Not available | Pipeline failed or not completed | Omit from release statistics except as missingness |

### Fallback

If diarisation/merge is unstable:

- keep per-channel transcripts if they are useful,
- drop WER-style claims,
- drop optional content labelling G,
- describe transcript limitations clearly,
- do not use transcripts as validated ground truth.

---

## 7. D — Physiological feature extraction

**Suggested owner:** Anna  
**Consulted:** Meisam if signals are unusual.

### Goal

Produce per-participant physiological summaries suitable for dataset
characterisation and, if ready, the worked example.

The emphasis should be on simple, robust, explainable features rather than a
large feature bank.

### Minimum outputs

- `features/physio_participant_task.tsv`,
- `features/physio_qc_summary.tsv`,
- short feature-definition note,
- list of excluded or unreliable participant-task windows.

Optional:

- `features/physio_window_30s.tsv` if useful and feasible.

### Suggested schema for `physio_participant_task.tsv`

```text
session_id
participant_id
task_id
window_start
window_end
ppg_available
eda_available
temp_available
imu_available
hr_mean
hr_sd
hrv_rmssd
eda_tonic_mean
eda_phasic_rate
temp_mean
temp_slope
motion_flag
qc_flag
```

Features that are unreliable should be omitted or marked with `qc_flag`, not
filled with misleading values.

### Candidate features

Keep the feature set simple and defensible:

- PPG-derived heart-rate summary,
- HRV summary only if reliable,
- EDA tonic level,
- EDA phasic-event count/rate only if reliable,
- temperature mean/slope,
- motion flags from IMU to interpret noisy physiology.

### QC questions

- Which participants/tasks have usable PPG?
- Which participants/tasks have usable EDA?
- Are motion artefacts too large in some task windows?
- Which features should be excluded or marked unreliable?
- Are missing values encoded consistently?

### Deadline

- Initial table by Wednesday/Thursday.
- Final sign-off by Saturday EOD.

---

## 8. E — Audio/prosodic feature extraction

**Suggested owner:** Anna  
**Dependency:** B only if transcript-aligned features are used.

### Goal

Produce simple audio/prosodic features for dataset characterisation and, if
ready, the worked example.

The safe baseline is audio-only/prosodic features. Transcript-derived features
should be included only if B is stable enough.

### Minimum outputs

- `features/audio_participant_task.tsv`,
- `features/audio_qc_summary.tsv`,
- short feature-definition note,
- list of excluded or unreliable participant-task windows.

Optional:

- `features/audio_window_30s.tsv` if useful and feasible.

### Suggested schema for `audio_participant_task.tsv`

```text
session_id
participant_id
task_id
window_start
window_end
audio_available
speaking_time_s
speaking_fraction
energy_mean
energy_sd
pause_count
turn_count
overlap_flag
speech_rate_proxy
pitch_mean
pitch_sd
qc_flag
```

Transcript-derived fields such as `speech_rate_proxy` or `turn_count` should be
included only if transcript timing is sufficiently stable.

### Candidate features

Keep the feature set robust:

- speaking-time estimate,
- energy mean/variance,
- pause count if reliable,
- turn count if reliable,
- overlap/cross-talk flag if reliable,
- speech-rate proxy if transcripts are stable,
- pitch/prosodic summaries only where audio quality supports them.

### Fallback

If transcripts are delayed or unstable:

- compute audio/prosodic features directly from the audio where possible,
- omit transcript-derived features,
- state the limitation clearly in the worked example.

### Deadline

- Core outputs by Thursday/Friday.
- Final sign-off by Saturday EOD.

---

## 9. F — Task-outcome audit

**Suggested owner:** Alice  
**Support:** Shan if needed for interpreting `stimuli_answers.tsv`.

### Goal

Produce the task-level descriptive and outcome variables that support the paper
and the worked example.

Only reliably available outcomes should be used. Ambiguous or incomplete fields
should be documented rather than forced into the analysis.

### Minimum outputs

- `features/task_outcomes.tsv`,
- `features/task_descriptives.tsv`,
- notes on missing or inconsistent task responses,
- mapping from task windows to outcome variables.

### Suggested schema for `task_outcomes.tsv`

```text
session_id
group_id
task_id
task_name
outcome_available
primary_outcome
secondary_outcome
group_level_score
participant_level_score
vad_response_rate
notes
```

### Candidate outcomes

Use only what is reliably available:

- T1: hidden-profile decision correctness / decision outcome,
- T2: negotiation settlement / agreement indicator,
- T3: idea-generation/ranking descriptors,
- T4: contribution amounts and group-level public-good outcome,
- valence/arousal probe summaries by task.

Dominance-related summaries may be reported as released variables if available,
but dominance analysis is not part of the core paper.

### QC questions

- Are task windows aligned with the correct responses?
- Are participant IDs/seat IDs consistent?
- Are all task outcomes interpretable?
- Are missing responses marked explicitly?
- Are VAD response rates sufficient for task-level summaries?

### Deadline

- Initial table by Thursday EOD.
- Final sign-off by Saturday EOD.

---

## 10. C — Worked example

**Suggested owners:** Meisam + Alice  
**Inputs:** A + at least one of D/E/F; ideally D + E + F.

### Goal

Provide one modest worked example showing how AffectAI can support multimodal
analysis.

This should be framed as an illustration of dataset use, not as a large
benchmark or definitive scientific result.

### Candidate framing

A conservative framing is preferred:

> How do task-level outcomes or valence/arousal changes relate to simple
> physiological and audio/prosodic summaries across structured group tasks?

The exact question should be written down before running the final analysis.

### Required outputs

- written analysis specification by Thursday EOD,
- one table or figure by Friday/Saturday,
- short interpretation with limitations,
- no post-hoc retuning to force a positive result.

### Suggested analysis specification

The written specification should include:

```text
research_question
included_sessions
included_tasks
included_modalities
primary_features
primary_outcomes
exclusion_rules
statistical_summary
figure_or_table_plan
known_limitations
```

### Recommended statistical posture

Given the small number of groups:

- prefer descriptive summaries,
- report effect sizes where meaningful,
- show uncertainty where possible,
- avoid strong significance claims,
- avoid training complex predictive models,
- do not tune the analysis after seeing weak results.

### Decision rules

| Available inputs | Worked-example action |
|---|---|
| D + E + F | Run full planned descriptive/multimodal example. |
| Any two of D/E/F | Run reduced example and state missing modality as limitation. |
| Only one of D/E/F | Run a simpler descriptive example. |
| No defensible result | Reframe §6 as supported task specifications and possible analyses. |

### Possible outputs

- task-level feature/outcome heatmap,
- compact table of task-level summaries,
- descriptive plot of feature differences across task phases,
- correlation/effect-size table linking task outcomes to feature summaries.

The final output should be easy to explain in the paper.

---

## 11. G — Optional LLM content labelling

**Suggested owner:** Shan

### Goal

Explore whether transcript-derived content labels are stable enough to include.

This is optional and must not delay the core submission. It should be attempted
only if B is stable early enough.

### Minimum outputs if attempted

- label schema,
- labelled subset or full set,
- agreement/kappa estimate,
- recommendation: release / appendix-only / drop.

### Possible label types

Keep labels simple if attempted:

- task-relevant statement,
- agreement/disagreement,
- question,
- proposal,
- decision/commitment,
- off-task/social talk.

Avoid complex affective or dominance labels in this submission.

### Decision rule

| Quality | Action |
|---|---|
| κ > 0.6 | Include as release artefact, if time allows. |
| κ = 0.4--0.6 | Appendix-only or exploratory note. |
| κ < 0.4, unfinished, or hard to explain | Drop. |

If G is dropped, it should not affect the core paper.

---

## 12. Dataset statistics and paper tables

The paper should include a small number of stable, high-value tables.

### Required tables

| Table | Source | Owner |
|---|---|---|
| Dataset overview / statistics | `metadata/analysable_sessions.tsv`, `modality_coverage.tsv` | Meisam |
| Task overview | protocol/task documentation | Meisam + Alice |
| Modality overview | acquisition documentation | Meisam |
| Modality coverage | `modality_coverage.tsv` | Meisam |
| Worked-example result | C outputs | Alice + Meisam |

### Optional tables

| Table | Include only if stable |
|---|---|
| Transcript QC summary | if manual checks or convincing automatic QC exist |
| Audio feature summary | if E is ready |
| Physio feature QC table | if D produces clear per-feature coverage |
| Content-label agreement | only if G is attempted and interpretable |

### Number sign-off

By Saturday EOD, each domain owner should sign off the numbers in their area:

| Domain | Sign-off owner |
|---|---|
| Session and modality coverage | Meisam |
| Physio features/QC | Anna |
| Audio/prosody features/QC | Anna |
| Task outcomes | Alice |
| Worked example | Alice + Meisam |
| Transcripts/content labels | Shan |

---

## 13. Cross-cutting rules

### 13.1 Single source of truth

Each artefact should have one canonical path. Avoid multiple competing versions
of the same table.

### 13.2 Written handoffs

Every handoff should include:

- file path,
- version/date,
- short description,
- known caveats,
- whether the file is ready for paper use.

### 13.3 Conservative claims

Small-N results should be described cautiously.

Use:

- descriptive statistics,
- effect sizes,
- uncertainty intervals where useful,
- direct language about missingness.

Avoid:

- strong predictive claims,
- clinical claims,
- claims about general population behaviour,
- claims that require unvalidated transcripts or gaze alignment.

### 13.4 Privacy

No released TSV should contain real names or direct identifiers. Transcript
outputs should include a redaction step before release.

### 13.5 No late expansion

After Saturday EOD, no new analysis should be added unless it fixes an error or
is explicitly agreed by the team.

### 13.6 Document null results

A weak or null worked example is acceptable if it is clearly described. It is
better to report an honest result than to tune the analysis late.

---

## 14. Analysis risks and contingencies

| Risk | Mitigation |
|---|---|
| Transcript pipeline unstable | Use per-channel transcripts if useful, drop G, describe limitation. |
| Speaker labels/cross-talk unreliable | Do not use transcripts analytically; keep only as caveated artefact. |
| Physio features noisy | Report coverage/QC honestly; exclude unreliable features. |
| Audio features delayed | Run C without audio or use audio only descriptively. |
| Task outcomes incomplete | Use task descriptives and response-rate summaries. |
| Worked example weak/null | Report honestly or reframe as supported task specification. |
| Too many numbers disagree across files | Saturday sign-off by domain owners; one canonical table. |
| Release metadata not validated in time | Describe only the release status that is true at submission. |
| Audio access status unresolved | Default to conservative/gated language. |

---

## 15. Meeting questions

For the team meeting, the most important questions are:

1. Does everyone agree that the paper is dataset-characterisation first?
2. Are A, D, E, F, and H realistic by the internal freeze?
3. What is the simplest defensible worked example?
4. Which transcript outputs are realistic before the deadline?
5. Which artefacts should be treated as core release files versus optional?
6. Are there any known blockers in feature extraction or task-outcome audit?
7. What should we explicitly cut now to protect the submission?

---

## 16. Closing principle

The guiding principle is:

> Submit the strongest honest version of the paper that we can defend by May 6.

That means prioritising clarity, reproducibility, and transparent limitations
over adding unstable analyses late.
