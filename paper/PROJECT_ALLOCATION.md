# AffectAI NeurIPS 2026 — Project Allocation

_Working draft: 2026-04-27_  
_Internal freeze target: Sunday May 3 EOD_  
_Hard deadline: Wednesday May 6 AoE_

## Purpose

This file defines the project-level allocation for the submission sprint. It
should be read together with `TEAM_ALLOCATION.md` and `ANALYSIS_PLAN.md`.

The allocation is intentionally flexible. Each project has a suggested owner,
but the team can adjust responsibilities in the meeting if a different split is
more realistic.

The current submission strategy is:

> dataset-characterisation paper + one worked example + release infrastructure.

This replaces the earlier, more ambitious benchmark-heavy framing. The new plan
prioritises defensible scope, transparent missingness, and a coherent paper by
the May 6 deadline.

---

## 1. Project summary

| Project | Name | Suggested lead | Collaborators | Priority | Output |
|---|---|---|---|---|---|
| **A** | Sync and coverage | Meisam | — | Non-negotiable | analysable-session list, coverage table, QC summary |
| **B** | Transcripts | Shan | Anna, Alice | Important | transcript artefacts, QC notes |
| **C** | Worked example | Meisam + Alice | Anna/Shan as needed | Important but flexible | analysis spec, result table/figure, §6 text |
| **D** | Physio features | Anna | Meisam consulted | Important | physio feature table, QC notes |
| **E** | Audio/prosodic features | Anna | Shan | Important | audio/prosodic feature table |
| **F** | Task-outcome audit | Alice | Shan if needed | Non-negotiable | task performance/descriptive table |
| **G** | LLM content labels | Shan | Alice if needed | Optional | labels + kappa, appendix-only note, or drop |
| **H** | Release infrastructure | Meisam | all provide artefacts | Non-negotiable | HF structure, Croissant, Datasheet |
| **I** | Paper integration | Meisam | all sign off numbers | Non-negotiable | submission-ready PDF |

---

## 2. Priority classes

### Non-negotiable

These must exist for a defensible dataset paper:

- A: sync and coverage,
- F: task-outcome/task-description audit,
- H: release infrastructure,
- I: paper integration.

### Important but adjustable

These strengthen the paper but can be simplified:

- B: transcripts,
- C: worked example,
- D: physio features,
- E: audio/prosodic features.

### Optional

This should only stay if quality is good and it does not threaten the core:

- G: LLM content labelling.

---

## 3. Allocation rationale

### Project A — Sync and coverage

Suggested owner: **Meisam**

Rationale: This is central to the dataset-characterisation claim and needs to
be consistent with the paper narrative, tables, and release documentation.

Minimum output:

- final or working analysable-session list,
- per-modality coverage table,
- inclusion/exclusion logic,
- limitations notes for missing sessions/modalities.

### Project B — Transcripts

Suggested owner: **Shan**

Rationale: Shan owns the transcription pipeline as a release artefact. Anna can
support audio-processing questions; Alice can support transcript-quality checks
because transcripts may feed into the task audit.

Minimum output:

- one session end-to-end by Tuesday EOD,
- all available transcripts by Wednesday EOD if feasible,
- short QC note on reliability,
- fallback recommendation if diarisation/cross-talk handling is unstable.

### Project C — Worked example

Suggested owners: **Meisam + Alice**

Rationale: The worked example should be framed conservatively and tied to the
paper’s main claim. Meisam owns the narrative and specification; Alice runs the
analysis if the required inputs are available.

Minimum output:

- written analysis specification by Thursday EOD,
- one result table or figure by Friday/Saturday,
- honest interpretation, including null/weak results if that is what we find.

Fallback:

- If the result is not defensible, reframe §6 as supported task specifications
  and possible analyses rather than a strong empirical result.

### Project D — Physio features

Suggested owner: **Anna**

Rationale: Anna should have full ownership of the feature extraction and QC,
with Meisam available for interpretation if signals look unusual.

Minimum output:

- participant-task feature table,
- feature definitions,
- QC/missingness notes.

### Project E — Audio/prosodic features

Suggested owner: **Anna**

Rationale: This naturally belongs with Anna’s audio work. It depends partly on
B if transcript-aligned features are used, but basic prosodic/acoustic features
can still be produced without full content labels.

Minimum output:

- participant/task or participant/window audio feature table,
- feature definitions,
- note on whether transcript-dependent features were included or skipped.

### Project F — Task-outcome audit

Suggested owner: **Alice**

Rationale: Task outcomes and task-characterisation are essential for the paper,
even if the worked example becomes weaker. This is part of the dataset’s value.

Minimum output:

- task performance table,
- task-level descriptive statistics,
- notes on missing/inconsistent task responses,
- mapping from task windows to outcome variables.

### Project G — LLM content labelling

Suggested owner: **Shan**

Rationale: This is useful but optional. It should not threaten B or the paper
core.

Minimum output if kept:

- label schema,
- labels,
- agreement/kappa estimate,
- recommendation: release / appendix-only / drop.

Drop rule:

- if agreement is weak, if B is delayed, or if labels require too much
  explanation, G should be dropped or moved to appendix-only.

### Project H — Release infrastructure

Suggested owner: **Meisam**

Rationale: This needs a single owner because it touches paper claims,
Croissant, responsible-use language, and the dataset access plan.

Minimum output:

- release folder structure,
- Hugging Face or equivalent dataset structure,
- Croissant metadata draft,
- Datasheet draft,
- access-tier wording.

### Project I — Paper integration

Suggested owner: **Meisam**

Rationale: The paper needs one voice and one person checking that numbers,
claims, tables, and limitations are consistent.

Minimum output:

- coherent full draft by Saturday/Sunday,
- final PDF for team read,
- submission-ready package.

---

## 4. Calendar with dependencies

| Date | Critical dependency | Decision/expected output |
|---|---|---|
| Tue Apr 28 EOD | A and first B test | analysable list available or temporary list used; one transcript tested |
| Wed Apr 29 EOD | B to D/E/F | transcripts available, or fallback transcript plan chosen |
| Thu Apr 30 EOD | D/E/F to C | decide full/reduced worked-example feature set |
| Fri May 1 EOD | C and G | worked-example results available; G include/appendix/drop decision |
| Sat May 2 EOD | sign-off | each owner signs off numbers in their area |
| Sun May 3 EOD | freeze | submission-ready PDF and release description |
| May 4--6 | buffer | final fixes and submission only |

---

## 5. Recommended reductions already built into the plan

Compared with the earlier plan, the current version reduces risk by:

- removing dominance analysis from the submission,
- moving video/3D pose/world-frame gaze outside the paper,
- not promising a full leaderboard,
- treating LLM content labels as optional,
- using one worked example rather than several primary baselines,
- reporting small-N results descriptively and conservatively.

---

## 6. Open points for the team meeting

1. Are the suggested owners reasonable?
2. What is the simplest defensible worked example?
3. Which feature tables can realistically be ready by Thursday/Friday?
4. Is B likely to support transcript-derived features, or should we keep B mainly as a release artefact?
5. Should G be attempted at all, or only if B finishes early?
6. What should be the exact internal folder structure for handoffs?
7. Who signs off which numbers before the Sunday freeze?

