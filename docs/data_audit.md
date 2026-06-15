# Data Audit — AffectAI Post-Collection Inventory

_Last updated: 2026-03-31_

This document records the results of a systematic audit of all collected data,
covering completeness, gaps, fixes applied, and per-session quality notes.

---

## 1  Collection Overview

| Item | Value |
|------|-------|
| Scheduled groups | 16 (grp-01 … grp-16) |
| Pilot groups | grp-03, grp-04 |
| Test groups | grp-05, grp-A, grp-B, grp-C |
| Final sessions (analysable) | 13 sessions across 11 groups |
| Participants per group | 4 (except grp-06: 3) |
| Recording dates | 2026-03-09 — 2026-03-20 |
| Total sessions in inventory | 26 |

### Recording Pathways

| Pathway | Location | Contents |
|---------|----------|----------|
| **Recording PC** | `data/affectai-capture-recording/sessions/{phase}/sub-01/{session}/` | XDF (LabRecorder: Tobii ET, EmotiBit physio, Vicon markers, experiment markers), Tobii NDJSON, stimuli events |
| **AV PC** | `data/AV/{phase}/sub-01/{session}/` | MKV video (7 cameras: 6×P20, 1×P50), WAV audio (DPA mics), frame logs |
| **CurrentStudy PC** | `data/CurrentStudy/sub-grp-{N}/ses-S001/eeg/` | Additional XDF archive recorded on a second computer |
| **Tobii Pro raw** | `data/Tobii/{timestamp}/` | Per-device scene video (MP4), gaze data (gz), IMU, metadata (incl. RuSerial for device ID) |
| **Central mic** | `data/Central mic/` | 16 WAV files (shared across all sessions) |
| **Stimuli events** | `data/affectai-capture-recording/stimuli/data/{candidate}/` | `events_*_experiment.tsv` with task onset/offset times (T0–T4) |

### Device-to-Participant Mappings

**Tobii Pro Glasses 3** (from `configs/tobii_glasses_streams.yaml`):

| Participant | Serial |
|-------------|--------|
| P1 | TG03B-080201022801 |
| P2 | TG03B-080201129491 |
| P3 | TG03B-080201020821 |
| P4 | TG03B-080201139301 |

**EmotiBit** (from `configs/emotibit_participants_by_source.json`):

| Participant | Serial | IP |
|-------------|--------|----|
| P1 | MD-V7-0001141 | 192.168.10.201 |
| P2 | MD-V7-0001160 | 192.168.10.202 |
| P3 | MD-V7-0001409 | 192.168.10.203 |
| P4 | MD-V7-0000837 | 192.168.10.204 |

---

## 2  Session Completeness

### Tier 1 — Fully Complete

All sources present, all tasks (T0–T4), schedule date/names match.

| Session | Group | Date | Participants |
|---------|-------|------|-------------|
| ses-20260317_grp-09_run01 | grp-09 | Mar 17 | Diana Taune, Ervin Bahtijar, RJ Martz, Tonio Michel Ermakoff |
| ses-20260318_grp-12_run01 | grp-12 | Mar 18 | Dags Olsteins, Luyao Wang, Pernille Zeuner, Tobias Piechowiak |
| ses-20260318_grp-13_run01 | grp-13 | Mar 18 | Giorgio Alvazzi, Joe Jensen, Mathias Povelsen, Pernille Schiolten |
| ses-20260319_grp-14_run01 | grp-14 | Mar 19 | Brice Modeste, Javier Alejandro Volpe, Laura Gravila, Nikita Ratchinsky |
| ses-20260319_grp-15_run01 | grp-15 | Mar 19 | Caroline Sigsgaard, Diego Caviedes Nozal, Oliver Olesen, Stine Holm |

### Tier 2 — Nearly Complete (minor gaps)

| Session | Group | Gap | Notes |
|---------|-------|-----|-------|
| ses-20260312_grp-07_run01 | grp-07 | Tobii P1 video was missing | **Resolved**: folder `20260312T121359Z` added (confirmed P1 via RuSerial) |
| ses-20260313_grp-08_run01 | grp-08 | Tobii P4 video was missing | **Resolved**: folder `20260313T090918Z` added (confirmed P4 via RuSerial). Group also has a re-recording session on Mar 16 |
| ses-20260318_grp-11_run01 | grp-11 | T0 task data missing | Stimuli events start at T1; T0 (study introduction) not captured in events TSV |
| ses-20260318_grp-01_run01 | grp-01 | No AV recordings | Recording PC data and CurrentStudy XDF present; AV folder empty |

### Tier 3 — Significant Gaps

| Session | Group | Gaps |
|---------|-------|------|
| ses-20260311_grp-06_run01 | grp-06 | Only 3 participants (P3 absent). No CurrentStudy XDF. No AV recordings. P3 Tobii video absent (genuine — participant not present) |
| ses-20260317_grp-10_run01 | grp-10 | No Tobii LSL streams in XDF. T0/T1 durations inflated (~19 h) — stimuli app ran overnight before session. CurrentStudy XDF present |
| ses-20260320_grp-16_run01 | grp-16 | **No CurrentStudy XDF** (confirmed missing — sub-P001 XDFs all belong to earlier dates). Stimuli events data now available. Recording PC XDF and AV recordings present. A stub stimuli dir (`20260320_grp-01_run01_*`) is a false start with wrong session ID |

### Pilot / Test Sessions (limited modalities, not for analysis)

| Session | Group | Phase | Available |
|---------|-------|-------|-----------|
| ses-20260309_grp-03_run01/02 | grp-03 | pilot | Tobii LSL only, no AV |
| ses-20260309_grp-04_run01 | grp-04 | pilot | AV + Tobii LSL |
| ses-20260312_grp-05_run01 | grp-05 | test | LSL + Tobii LSL, CurrentStudy XDF |
| ses-20260310_grp-A_run01 … ses-20260317_grp-A_run01 | grp-A | test | 7 sessions, LSL + Tobii LSL, some AV |
| ses-20260312_grp-B_run01 | grp-B | test | LSL + Tobii LSL |
| ses-20260317_grp-C_run01 | grp-C | test | Tobii LSL only |

---

## 3  Data Fixes Applied

### 3.1  grp-01 → grp-11 Stimuli File Rename

12 stimuli files in `data/affectai-capture-recording/stimuli/data/` were named
with a `grp-01` prefix but actually belonged to grp-11 (identified by date
2026-03-18 and schedule cross-check). Files renamed:
- `20260318_grp-01_run01_*` → `20260318_grp-11_run01_*`

### 3.2  grp-01 → grp-11 Recording Session Rename

An XDF and JSON pair in `sessions/Final/sub-01/ses-20260318_grp-01_run01/` was
misnamed. Content inspection (datetime in XDF header, EmotiBit/Tobii streams)
confirmed it was a grp-11 recording. Renamed to `ses-20260318_grp-11_run01` and
merged into the existing grp-11 session directory.

### 3.3  Duplicate grp-01 Inventory Entry Removed

After the rename, the spurious second grp-01 session entry was removed from the
inventory (27 → 26 sessions).

### 3.4  grp-07 Tobii P1 Video Gap Filled

Tobii folder `20260312T121359Z` was added to `data/Tobii/`. Device serial
confirmed as P1 (TG03B-080201022801) via `meta/RuSerial`. Scene video: 2368 MB.

### 3.5  grp-08 Tobii P4 Video Gap Filled

Tobii folder `20260313T090918Z` was added to `data/Tobii/`. Device serial
confirmed as P4 (TG03B-080201139301) via `meta/RuSerial`. Scene video: 2579 MB.

### 3.6  sub-P001 XDF Identified as grp-09

The 294 MB XDF in `data/CurrentStudy/sub-P001/ses-S001/eeg/` has datetime
`2026-03-17T13:14` — this is grp-09's session. It started ~20 min before the
official `sub-grp-09` XDF (96 MB, started at 13:35), providing a longer
Recording-PC capture of the same session. The `sub-P001` directory name was a
generic LabRecorder default; the data is genuine grp-09 content.

Other XDFs in `sub-P001/.old/` are small (3 MB) test files from Mar 11 and
Mar 12 — not session data.

### 3.7  AffectAI/test XDF Classified

The 456 MB XDF in `data/AffectAI/test/sub-P001/ses-S001/eeg/` (datetime
2026-03-16T16:58) contains only Vicon marker streams and Tobii glasses
rotation/translation streams (36 streams total). This is a **Vicon MoCap test
recording**, not a session. It can be ignored for analysis.

---

## 4  Stimuli Multi-Directory Handling

Several sessions have multiple stimuli directories due to restarts of the
stimuli application:

| Group | Stimuli directories | Handling |
|-------|-------------------|----------|
| grp-04 | 3 | Keep longest clean run per task |
| grp-06 | 3 (2 for run01, 1 for run02) | Per-TSV independent duration |
| grp-07 | 2 | Keep longest clean run per task |
| grp-09 | 3 | Keep longest clean run per task |

The pipeline (`generate_session_metadata_report.py`, `xdf_sync_pipeline.py`)
searches inside each candidate directory, computes task durations per-TSV
independently, and retains the longest clean run per task. This avoids inflated
wall-clock spans that occur when durations are computed across restart
boundaries.

---

## 5  Schedule Cross-Check

Schedule source: `configs/session_schedule.tsv` (16 groups).

| Check | Result |
|-------|--------|
| Date match | All sessions match scheduled dates. grp-08 has a second session on Mar 16 (re-recording) |
| Name match | All groups match **except grp-14 P1**: schedule says "Tatiana Crucerescu", recording says "Laura Gravila" — likely a late participant substitution |

---

## 6  Orphan / Unattributed Data

| Path | Contents | Status |
|------|----------|--------|
| `data/CurrentStudy/sub-P001/` | grp-09 XDF (294 MB) + old test files (3 MB each) | **Attributed** — belongs to grp-09 |
| `data/CurrentStudy/sub-P001/.old/` | 2 small XDF files (Mar 11, Mar 12) | Test recordings, not session data |
| `data/AffectAI/test/` | 1 Vicon MoCap test XDF (456 MB, Mar 16) | Test recording, not session data |
| `data/Central mic/` | 16 WAV files | Shared resource, not per-session |

---

## 7  Priority Processing Order

Recommended order for downstream pipeline processing, based on data completeness:

1. **grp-15** — fully complete; AV XDF corrupted but audio synced via JSONL fallback ✅
2. **grp-12, grp-13** — fully complete ✅
3. **grp-14** — fully complete (note P1 name discrepancy) ✅
4. **grp-09** — fully complete, has additional 294 MB CurrentStudy XDF ✅
5. **grp-07, grp-08** — Tobii video gaps now resolved ✅
6. **grp-01, grp-11** — minor gaps (no AV / no T0)
7. **grp-10** — needs T0/T1 duration correction (stimuli app overnight issue)
8. **grp-06** — 3 participants only, significant gaps
9. **grp-16** — missing CurrentStudy XDF and stimuli events

All sessions with XDF + AV data have been processed by `xdf_sync_pipeline.py`
and have split DPA audio clips at:
`D:\data_witout-video\processed_audio\sub-01\ses-<date>_<group>_run01\audio\`
(20 WAVs per session: 4 mics × 5 tasks T0–T4, except where noted above).

---

## 8  Automated Report

The machine-readable metadata report is generated by:

```bash
python tools/generate_session_metadata_report.py --probe-xdf \
    --output metadata/session_metadata_report.tsv
```

This produces a 26-row TSV with 50+ columns covering all sources, per-participant
stream attribution, task durations, and schedule cross-check fields. See the
report TSV for exact per-session values.
