---
applyTo: "**"
---

# BIDS Conventions — AffectAI Data Processing

These rules apply to all code, scripts, and configuration that generate or read BIDS-structured output.

## Directory hierarchy

```
<bids_root>/
  participants.tsv          ← cross-session anonymised roster
  sub-{id}/
    ses-{id}/
      eeg/
      et/                   ← eye-tracking (Tobii)
      physio/               ← EmotiBit (PPG, EDA, temp, IMU)
      audio/                ← DPA microphone recordings
      video/                ← Jabra P20/P50 camera recordings
      mocap/                ← Vicon/3D pose outputs
      beh/                  ← behavioural (stimuli answers, responses)
      annot/                ← annotation and signal maps
```

## File naming

Pattern: `sub-{id}_ses-{id}_task-{task}_run-{run}_<suffix>.<ext>`

| Entity | Values | Example |
|--------|--------|---------|
| `sub-` | P1, P2, P3, P4 | `sub-P1` |
| `ses-` | BIDS session ID (e.g., `grp15run01`) | `ses-grp15run01` |
| `task-` | T0, T1, T2, T3, T4 | `task-T2` |
| `run-` | `01` (zero-padded) | `run-01` |

- Use **underscores** between key-value entities, hyphens within values: `sub-P1_ses-grp15run01_task-T1_run-01_events.tsv`
- Do not invent new entity names; use the standard BIDS entities above

## Task labels

| Label | Session phase |
|-------|--------------|
| `T0` | Baseline / intro |
| `T1` | Hidden-Profile Decision task |
| `T2` | Mini-Negotiation task |
| `T3` | Idea Generation (NGT) task |
| `T4` | Public-Goods Micro-Game task |

## events.tsv rules

- One **authoritative** `events.tsv` per session — the timeline spine
- Never duplicate events across multiple files
- Required columns: `onset`, `duration`, `trial_type`, `task`, `phase`, `participant`, `stream`
- `onset` is in seconds relative to session start (LSL clock-aligned)
- Participant ID column: `P1`–`P4` only — never real names

## participants.tsv rules

- Stored at BIDS root (study root), not inside `sub-*/ses-*/`
- Anonymised IDs only: `participant_id` column values = `P1`, `P2`, `P3`, `P4`
- Required columns: `participant_id`, `group_id`
- No PHI (personal health information) or real names

## Output files (Pipeline 1)

| File | Location | Purpose |
|------|----------|---------|
| `*_events.tsv` | `ses-*/` | Session timeline spine |
| `*_stimuli_answers.tsv` | `ses-*/beh/` | Normalised stimuli responses |
| `*_participant_signal_map.tsv` | `ses-*/annot/` | Device-to-participant mapping |

## Common mistakes to avoid

- Do not put `events.tsv` inside a modality subfolder (`video/`, `eeg/`) — it belongs at `ses-*/`
- Do not use `run-1` (not zero-padded) — always `run-01`
- Do not create per-task `participants.tsv` — only one at the study root
- Do not add `acq-*` entity unless multiple acquisitions of the same modality exist in one session
