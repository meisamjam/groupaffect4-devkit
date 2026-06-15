# AffectAI NeurIPS 2026 — Working Team Allocation & Timeline

_Working draft: 2026-04-27_  
_Internal freeze target: Sunday May 3 EOD_  
_Hard deadline: Wednesday May 6 AoE_

## Purpose

This is a working coordination plan for the NeurIPS 2026 submission sprint.
It is intended as a shared starting point for the team meeting, not as a rigid
instruction list. We should adjust ownership, timing, and scope together based
on availability, blockers, and what is realistic.

Analytical details live in `ANALYSIS_PLAN.md`. Paper structure lives in
`main.tex` and `README.md`. This file answers:

- who is suggested to lead each workstream,
- which artefacts are needed,
- when dependencies should be handed over,
- and what we simplify if something does not converge.

The current scope assumes that the paper is primarily a
**dataset-characterisation paper with one carefully framed worked example**.
Dominance analysis, video/3D pose, and more complex baselines are left for
follow-up work.

---

## 1. Proposed ownership model

Each workstream should have one clear owner, but the plan should remain
flexible. Ownership means making sure the output exists, not doing every detail
alone.

| Code | Workstream | Suggested owner | Support / consultation | Main output |
|---|---|---|---|---|
| **A** | Sync and coverage characterisation | Meisam | all as needed | analysable-session list, modality coverage, inclusion/exclusion logic |
| **B** | Transcription pipeline | Shan | Anna for audio issues, Alice for QC | transcript artefacts and transcript-quality notes |
| **C** | Worked example | Meisam + Alice | Anna/Shan depending on features used | analysis framing, results, figures, §6 text |
| **D** | Physiological features | Anna | Meisam consulted if signals are unclear | physio feature table and QC notes |
| **E** | Audio/prosodic features | Anna | Shan for transcript/audio dependencies | audio/prosodic feature table |
| **F** | Task-outcome audit | Alice | Shan for `stimuli_answers.tsv` reading if needed | task performance/descriptive table |
| **G** | LLM content labelling | Shan | Alice for spot-check/QC if needed | labels + agreement, appendix-only note, or drop decision |
| **H** | Release infrastructure | Meisam | all provide artefacts | Hugging Face structure, Croissant, Datasheet |
| **I** | Paper writing and integration | Meisam | all sign off their numbers | submission-ready PDF |

---

## 2. Per-person working brief

### Meisam

Suggested role: submission lead, release owner, and paper integrator.

Main responsibilities:

- Own **A**: sync and coverage characterisation.
- Co-own **C**: frame the worked example and write the paper section.
- Own **H**: Hugging Face, Croissant, Datasheet, release structure.
- Own **I**: paper integration, final consistency, submission PDF.
- Stay consulted on **D** if Anna encounters unclear physiological signals.

Expected outputs:

- analysable-session list,
- modality coverage table,
- release folder structure,
- Croissant/Datasheet status,
- integrated paper draft,
- final submission PDF.

### Anna

Suggested role: physiological and audio/prosodic feature owner.

Main responsibilities:

- Own **D**: physiological feature extraction and QC.
- Own **E**: audio/prosodic feature extraction.
- Support **B** if Shan needs help with audio-processing or cross-talk issues.
- Sign off physiological and audio-related numbers before the internal freeze.

Expected outputs:

- physio feature table,
- audio/prosodic feature table,
- feature definitions/short notes,
- QC notes for unreliable or excluded signals,
- sign-off on relevant paper numbers.

### Alice

Suggested role: task audit and worked-example execution owner.

Main responsibilities:

- Own **F**: task-outcome audit and descriptive task statistics.
- Co-own **C**: run the worked-example analysis under the agreed framing.
- Support **B** with transcript-quality review if needed.
- Sign off task-performance and worked-example numbers.

Expected outputs:

- task-performance table,
- descriptive task statistics,
- worked-example results and figures,
- notes on task-level limitations,
- sign-off on relevant paper numbers.

### Shan

Suggested role: transcription and optional content-labelling owner.

Main responsibilities:

- Own **B**: transcription pipeline as a release artefact.
- Own **G** only if B is stable early enough.
- Support **F** with `stimuli_answers.tsv` reading if needed.
- Recommend dropping **G** if agreement or quality is not sufficient.

Expected outputs:

- one full end-to-end transcribed session as an early test,
- all available transcripts,
- transcript quality notes,
- LLM labels and agreement statistics if stable,
- or a clear recommendation to drop/appendix-only G.

---

## 3. Proposed daily timeline

This schedule is intentionally front-loaded so that we can identify problems
early. The May 4--6 window should be treated as submission buffer, not normal
analysis time.

| Date | Meisam | Anna | Alice | Shan |
|---|---|---|---|---|
| **Mon Apr 27** | Align plan, prepare A/H/I | Prepare D/E | Prepare F/C | Prepare B/G |
| **Tue Apr 28** | Finish or nearly finish A; start H setup | Start D | Start F | B day 1: one full session end-to-end |
| **Wed Apr 29** | C framing; continue H | Continue D; assist B if needed | Continue F; transcript QC if available | B day 2: remaining sessions + checks |
| **Thu Apr 30** | Written C spec; H upload/check | Finalise D/E core outputs | Finish F; prepare/run C | Start G only if B is stable |
| **Fri May 1** | Paper writing/integration; H finalise | E wrap/QC | C results + write-up | G wrap, appendix-only, or drop decision |
| **Sat May 2** | Full paper integration | Sign off D/E numbers | Sign off F/C numbers | Sign off B/G numbers |
| **Sun May 3** | Internal freeze and submission-ready PDF | On call | On call | On call |
| **Mon--Wed May 4--6** | Buffer, final fixes, submission | Buffer | Buffer | Buffer |

The buffer is for final edits, broken references, metadata issues, release
packaging, table/figure consistency, and submission-portal problems. It should
not be used to add new analysis unless the team explicitly decides that it is
safe.

---

## 4. Handoffs

Handoffs should be written rather than only verbal. The artefact should be put
in the shared folder and announced with a short message, for example:

> “The physio feature table is in `/shared/features/physio_features.csv`. Please
> use this version for C. Ping me if anything looks inconsistent.”

| Target time | From → To | Artefact |
|---|---|---|
| Tue EOD | Meisam → all | Draft/final analysable-session list |
| Tue EOD | Shan → Anna + Alice | One full-session transcript for review |
| Wed EOD | Shan → Anna | All available transcripts for audio/prosodic processing |
| Wed EOD | Shan → Alice | All available transcripts for task audit/QC |
| Wed EOD | Anna → Alice | Initial physio feature table, if ready |
| Wed EOD | Alice → Shan | Initial transcript-quality feedback, if transcripts are ready |
| Thu EOD | Anna → Alice + Meisam | Audio/prosodic feature table |
| Thu EOD | Alice → Meisam | Task performance/descriptive table |
| Thu EOD | Meisam → Alice | Written C analysis specification |
| Fri EOD | Alice → Meisam | Worked-example results and figures |
| Fri EOD | Shan → Meisam | G status: include, appendix-only, or drop |
| Sat EOD | Anna + Alice + Shan → Meisam | Sign-off on numbers in their areas |
| Sun EOD | Meisam → all | Submission-ready PDF for final read |

These times can be adjusted in the meeting. The important point is to keep the
dependencies visible.

---

## 5. Checkpoints

The checkpoints are intended to protect the deadline, not to add pressure. Each
checkpoint should result in a simple decision: continue, simplify, or drop.

### Checkpoint 1 — Tuesday EOD

Questions:

1. Did B produce one full end-to-end transcribed session?
2. Is A ready enough for the others to use?

| Situation | Suggested response |
|---|---|
| B works on one session | Continue as planned. |
| B is close but needs help | Anna helps Shan on Wednesday; D may slip slightly. |
| B does not work reliably on real audio | Drop WER-style reporting and G; use simpler transcript release description. |
| A is ready | Everyone uses Meisam’s analysable-session list. |
| A is not ready | Use Tier-1 + Tier-2 list temporarily while Meisam finishes A. |

### Checkpoint 2 — Thursday EOD

Questions:

1. Are D, E, and F usable?
2. Can the worked example run on the available feature set?
3. Is G stable enough to keep?

| Situation | Suggested response |
|---|---|
| D + E + F are ready | Run C on the planned full feature set. |
| Two of D/E/F are ready | Run C on the available subset and mark the missing part as a limitation. |
| Only one of D/E/F is ready | Simplify C around the available output. |
| C is not defensible | Reframe §6 as supported task specifications and descriptive examples. |
| G has κ > 0.6 | Include labels in the release. |
| G has κ = 0.4--0.6 | Appendix-only; not a core release artefact. |
| G has κ < 0.4 or is unfinished | Drop G from the paper/release. |

### Checkpoint 3 — Saturday EOD

Questions:

1. Does the paper read coherently end-to-end?
2. Are all numbers signed off by the relevant owner?
3. Is the release package in an acceptable state?
4. Do we need to simplify before the Sunday freeze?

| Situation | Suggested response |
|---|---|
| Paper reads well | Sunday is final read and submission preparation. |
| Paper has isolated TODOs | Meisam closes TODOs Sunday; team remains on call. |
| Worked example is weak | Reframe it more descriptively and avoid overclaiming. |
| Release metadata is incomplete | State only what is true and ready; do not overstate. |

---

## 6. Risk register

| Risk | Likelihood | Impact | Suggested mitigation |
|---|---:|---:|---|
| Transcription/diarisation fails on real audio | Medium | High | Test one full session early; fall back to per-channel transcripts if needed. |
| Cross-talk merge gives wrong speaker labels | Medium | High | Manual check early; do not rely on merged diarisation if unstable. |
| Worked example does not show a clean signal | Medium | High | Pre-specify C; report weak/null results honestly or reframe. |
| Small N limits statistical claims | High | Medium | Keep the paper descriptive and avoid overclaiming. |
| Physio or audio features are delayed | Medium | Medium | Run C on available subset and mark missing modality as limitation. |
| Audio consent/release status is ambiguous | Low--Medium | High | Default to gated raw audio; do not wait for slow clarification before submission. |
| Croissant validation fails late | Low | Medium | Build metadata early; if validation fails, include JSON and describe status accurately. |
| Too many sections remain disconnected | Medium | High | Meisam integrates continuously from Friday onward. |
| A blocker is not communicated early | Medium | High | Use short daily updates and checkpoint escalation. |

---

## 7. What is deliberately not in this sprint

- **Dominance analysis** as a main paper analysis. This moves to the follow-up.
- **Per-camera video, 3D pose, world-frame gaze, and Vicon/mocap-style work.** These are outside this submission.
- **A leaderboard.** The paper is dataset-characterisation focused, not a competitive benchmark paper.
- **Multiple primary baselines.** One worked example is safer than several weak baselines.
- **Authorship order.** This should be handled separately.
- **Camera-ready work.** This belongs after the notification stage.

---

## 8. One-line summary per person

- **Meisam:** finish sync/coverage, manage release infrastructure, frame the worked example, and integrate the paper.
- **Anna:** own physiological and audio/prosodic features, support audio/transcription issues if needed, and sign off feature numbers.
- **Alice:** own task-outcome audit, run the worked example with Meisam, and sign off task/result numbers.
- **Shan:** own transcription, test one full session early, run content labelling only if stable, and recommend dropping it if quality is weak.

---

## 9. Communication

- Use one dedicated sprint channel.
- Pin this plan, shared-folder link, current analysable-session list, latest PDF, and checkpoint times.
- Each person posts a short async update by 09:00:

```text
Yesterday:
Today:
Blockers:
```

- Synchronous calls only at the three checkpoints, maximum 30 minutes.
- If anyone is more than half a day behind on a dependency, they should say it early. This is not a blame rule; it is a deadline-protection rule.

---

## 10. Working principle

We should submit the strongest honest version of the paper that we can defend
by May 6. A focused, transparent dataset paper is better than a broader paper
with unstable analyses or claims we cannot fully support.
