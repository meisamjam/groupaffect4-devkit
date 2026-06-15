# AffectAI → NeurIPS 2026 Evaluations & Datasets Track — Submission Plan

_Working draft: 2026-04-27_  
_Internal freeze target: Sunday May 3 EOD_  
_Hard deadline: Wednesday May 6 AoE_

## 1. Current strategy

The current strategy is to submit AffectAI as a focused
**dataset-characterisation paper** with one carefully framed worked example.

The paper should not be positioned as a full benchmark/leaderboard paper. The
stronger and more defensible claim is that AffectAI contributes a structured,
well-documented, multimodal group-interaction dataset with:

- per-participant physiology,
- 2D eye tracking,
- close-talk audio,
- behavioural self-reports,
- BFI-44 personality scores,
- task outcomes,
- transcript artefacts if quality is sufficient,
- synchronisation/coverage QC,
- release metadata and responsible-use documentation.

The worked example should illustrate what the dataset enables. It should not be
oversold as a definitive result.

---

## 2. Central framing

Suggested central framing:

> AffectAI is a multimodal dataset of co-located small-group interaction,
> designed to support transparent study of affective, cognitive, and
> behavioural dynamics during structured collaborative tasks. The paper
> contributes the dataset design, acquisition setup, synchronisation strategy,
> modality coverage, release infrastructure, and one worked example of
> multimodal analysis.

This framing gives us enough ambition while keeping the claims honest.

---

## 3. What we keep from the existing draft/material

Reusable material:

- dataset motivation,
- structured task design,
- modality table,
- synchronisation story,
- BIDS/release language,
- BFI-44 description,
- data audit and completeness-tier logic,
- limitations around small N, single site, single language, missingness, and
  voice-identification risk.

Material to reduce or defer:

- dominance analysis,
- video-heavy claims,
- 3D pose/world-frame gaze claims,
- Vicon/mocap language,
- multiple benchmark/baseline promises,
- leaderboard-style framing.

---

## 4. Paper structure

Target structure:

1. **Introduction**  
   Motivation, gap, contributions, and the dataset-characterisation framing.

2. **Related work**  
   Group multimodal corpora, affect/physiology corpora, conversational corpora,
   and the specific gap addressed by AffectAI.

3. **Dataset design**  
   Participants, ethics, session flow, structured tasks, self-report probes,
   BFI-44.

4. **Modalities and acquisition**  
   Physiology, Tobii 2D gaze, close-talk audio, room audio, tablets, personality,
   transcript artefacts. Video/3D-related streams should be mentioned only as
   outside the current release if needed.

5. **Processing and release**  
   BIDS layout, sync, transcript pipeline, anonymisation, release tiers,
   Croissant, Datasheet, hosting.

6. **Worked example and supported analyses**  
   One worked example if results are defensible; otherwise task specifications
   and possible analyses.

7. **Dataset statistics and quality**  
   Session tiers, per-modality coverage, VAD response rate, sync quality,
   transcript/audio quality where available.

8. **Limitations**  
   Small N, convenience sample, single lab/site, missingness, sparse labels,
   voice risk, limited generalisability.

9. **Ethics, access, and responsible use**  
   Consent, anonymisation, gated access if needed, intended use, out-of-scope
   use, maintenance.

10. **Conclusion**

Appendix:

- BFI-44 scoring,
- sync pipeline detail,
- per-session quality table,
- transcript pipeline parameters,
- Datasheet for Datasets.

---

## 5. Minimum viable submission

If several optional pieces fail, the paper should still contain:

1. dataset motivation,
2. study design,
3. participant/session/task description,
4. modality and acquisition overview,
5. synchronisation strategy,
6. modality coverage and missingness,
7. release structure,
8. supported task specifications,
9. limitations,
10. ethics/access/responsible-use section.

This is the minimum defensible version.

The stronger version additionally includes:

- clean transcript artefacts,
- physio features,
- audio/prosodic features,
- task outcome tables,
- one worked example,
- optional content labels.

---

## 6. Current open decisions

These should be resolved in the team meeting or at the first checkpoint:

1. What is the simplest defensible worked example?
2. Which feature tables can realistically be ready by Thursday/Friday?
3. Should transcripts be used analytically, or mainly presented as a release artefact?
4. Should G be attempted, or only kept as a stretch goal?
5. What access tier is safest for raw audio?
6. Which numbers are definitely available for the paper by Saturday?

---

## 7. Near-term actions

| Action | Owner | Target |
|---|---|---|
| Confirm working scope with team | Meisam + all | meeting |
| Produce analysable-session list | Meisam | Tue EOD |
| Run one full transcript session | Shan | Tue EOD |
| Start physio/audio feature extraction | Anna | Tue/Wed |
| Start task-outcome audit | Alice | Tue/Wed |
| Write worked-example specification | Meisam | Thu EOD |
| Decide G include/appendix/drop | Shan + Meisam | Fri EOD |
| Sign off numbers | all | Sat EOD |
| Freeze PDF | Meisam | Sun EOD |

