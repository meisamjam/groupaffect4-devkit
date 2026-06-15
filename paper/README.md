# AffectAI → NeurIPS 2026 Working Submission Folder

_Working draft: 2026-04-27_

This folder contains the working material for the AffectAI NeurIPS 2026
submission.

The current strategy is:

> dataset-characterisation paper + one carefully framed worked example + release
> infrastructure.

The paper should be focused, transparent, and defensible. It should not promise
a full leaderboard, multiple major baselines, dominance analysis, or video/3D
pose analysis for this submission.

---

## Scope of this version

### Included

- dataset motivation and design,
- structured group-task protocol,
- per-participant physiology,
- 2D eye tracking,
- close-talk and room audio,
- tablet self-reports and task responses,
- BFI-44 personality scores,
- synchronisation and modality coverage,
- transcript artefacts if quality is sufficient,
- one worked example if results are defensible,
- Hugging Face/Croissant/Datasheet release structure,
- ethics, access, and responsible-use language.

### Deferred

- dominance analysis,
- per-camera video analysis,
- 3D pose,
- world-frame gaze,
- Vicon/mocap-style analysis,
- leaderboard-style benchmarking,
- multiple primary baselines,
- camera-ready material.

---

## File map

| File | Purpose |
|---|---|
| `TEAM_ALLOCATION.md` | flexible team-facing sprint plan: ownership, timeline, handoffs, checkpoints |
| `PROJECT_ALLOCATION.md` | project-level allocation A--I with priority classes and dependencies |
| `ANALYSIS_PLAN.md` | analysis/QC plan aligned with the reduced scope |
| `PHYSIO_ANALYSIS_PLAN.md` | executable Project D plan for EmotiBit features, QC, and paper handoff |
| `neurips2026_plan.md` | high-level submission strategy and paper framing |
| `main.tex` | NeurIPS paper skeleton |
| `dataset_stats.tex` | dataset statistics and quality table |
| `modalities.tex` | release-modality table |
| `task_overview.tex` | task overview table |
| `references.bib` | bibliography |

---

## Compile

Place the official NeurIPS 2026 style file next to `main.tex`, then run:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## Status of `main.tex`

| Section | Status |
|---|---|
| Abstract | draft; needs final numbers |
| Introduction | draft; aligned with dataset-characterisation framing |
| Related Work | TODO |
| Dataset Design | partial draft |
| Modalities and Acquisition | partial draft |
| Processing and Release | draft; release-tier decision still needed |
| Worked Example and Supported Analyses | placeholder; depends on C |
| Dataset Statistics and Quality | placeholder; depends on A/D/E/F/B |
| Limitations | draft/partial |
| Ethics, Access, Responsible Use | draft/partial |
| Appendix | placeholders |

---

## Decisions still pending

1. Final worked-example question.
2. Final analysable-session list.
3. Raw-audio access tier.
4. Whether transcripts are analytical input or release artefact only.
5. Whether LLM content labels are included, appendix-only, or dropped.
6. Final numbers for dataset statistics and modality coverage.

---

## Working rule

After Saturday EOD, only fixes and integration should happen. New analysis after
that point should be avoided unless it corrects an error or the team explicitly
agrees it is safe.

