# BIDS Processing Pipeline — Quick Reference

## Session Matching Rules

**From inventory CSV:**
- `session` → session_id (e.g., `ses-20260311_grp-06_run01`)
- `group_id` → used for AV-PC, Stimuli matching (e.g., `grp-06`)
- `phase_tags` → primary directory (first tag, e.g., `final`, `pilot`, `test`)
- `participants_ids` → BIDS participant list (e.g., `sub-01;sub-02;sub-03;sub-04`)

## Source Location Pattern

| Source | Search Path | Search Pattern | Example |
|--------|-------------|-----------------|---------|
| **Recording-PC** | `data/affectai-capture-recording/sessions/{phase}/sub-01/` | `ses-{session_id}*` | `ses-20260311_grp-06_run01` |
| **AV-PC** | `data/AV/{phase}/` | `*{group_id}*` | `ses-20260311_grp-04_run01/` |
| **Tobii** | `data/Tobii/` | `*{session_id}*` | `20260311_grp-06_run01/` |
| **Stimuli** | `data/stimuli/{phase}/` | `*{group_id}*` | `20260311_grp-06_run01_20260311_125450/` |

## Configuration Files to Load

| Config | Path | Purpose |
|--------|------|---------|
| **EmotiBit participants** | `configs/emotibit_participants_by_source.json` | Map IP/device → P1–P4 roles |
| **Camera specs** | `configs/camera_specs.json` | Camera model, resolution, mounting |
| **Desk zones** | `configs/desk_zones.json` | Participant seating layout |

## BIDS Output Structure (per session)

```
sub-01/ses-20260311_grp-06_run01/
├── annot/
│   ├── sub-01_ses-20260311_grp-06_run01_sync_metadata.json
│   ├── sub-01_ses-20260311_grp-06_run01_task-T0T1T2T3T4_task_run_windows.tsv
│   └── sub-01_ses-20260311_grp-06_run01_participant_signal_map.tsv
├── beh/
│   ├── sub-01_ses-20260311_grp-06_run01_events.tsv
│   ├── sub-01_ses-20260311_grp-06_run01_task-T1_run-01_events.tsv
│   └── sub-01_ses-20260311_grp-06_run01_stimuli_answers.tsv
├── et/     (Tobii gaze + pupil)
├── physio/ (EmotiBit PPG, EDA, temperature)
├── audio/  (DPA microphone clips per task)
└── video/  (Camera clips per task, if --split-media)
```

## Task Window Naming (T0–T4)

- **T0**: Baseline/calibration (preparation, ~1 min)
- **T1**: Hidden-Profile Decision task (~6 min)
- **T2**: Mini-Negotiation task (~5 min)
- **T3**: Idea Generation (NGT) task (~6 min)
- **T4**: Public-Goods Micro-Game (~5 min)

**File naming pattern for task-specific data**:
```
sub-01_ses-20260311_grp-06_run01_task-T1_run-01_*.*
```

## Stimuli Annotations

### Task Windows TSV
```
task    run    start_time    end_time    description
T1      01     61.2          450.8       Hidden-Profile Decision
T2      01     451.5         720.3       Mini-Negotiation
```

### Participant Signal Map
```
participant    et_signal            physio_signal                audio_signal
sub-01         Tobii_P1_Gaze        EmotiBit_MD-V7-0001141      dpa_mic1
sub-02         Tobii_P2_Gaze        EmotiBit_MD-V7-0001160      dpa_mic2
```

### Stimuli Responses
```
participant    task    question_id                 response    timestamp
sub-01         T1      decision_confidence         7           123.45
sub-01         T2      negotiation_score           45          201.23
```

## Synchronization Tiers (Best to Worst)

1. **Frame logs** (~0.5 ms MAD) — `frame_logs/{camera}_frames.jsonl`
2. **LSL progress** (~1 ms) — `lsl/ffmpeg_progress_*.jsonl`
3. **Progress TSV** (~1 ms) — `sourcedata/sync/*_progress.tsv`
4. **Events JSONL** (~100 ms) — `video/ffmpeg_multicap_events.jsonl`

Auto-selected by 3D pose pipeline; best available tier is used.

## Participant Mapping JSON

```json
{
  "participants": {
    "P1": "MD-V7-0001141",  // EmotiBit device serial
    "P2": "MD-V7-0001160",
    "P3": "MD-V7-0001409",
    "P4": "MD-V7-0000837"
  },
  "by_source": {
    "192.168.10.201": "P1",  // IP → role
    "192.168.10.202": "P2",
    "192.168.10.203": "P3",
    "192.168.10.204": "P4"
  }
}
```

## CLI Execution

```bash
python bids_processing_pipeline.py \
  --data-root "D:\...\affectai-data-processing-seed" \
  --output-root "E:\processed_data" \
  --inventory "data/high_level_session_inventory.csv" \
  --max-workers 4 \
  --split-media \
  --link-files
```

## Troubleshooting Checklist

- [ ] Inventory CSV has correct `session`, `group_id`, `phase_tags` columns
- [ ] Recording-PC path: `data/affectai-capture-recording/sessions/{phase}/sub-01/ses-*`
- [ ] AV-PC path: `data/AV/{phase}/*{group_id}*` (if AV data present)
- [ ] Stimuli path: `data/stimuli/{phase}/*{group_id}*` (if stimuli data present)
- [ ] `events.tsv` exists in Recording-PC data (needed for task markers)
- [ ] `configs/emotibit_participants_by_source.json` loaded and validated
- [ ] Output BIDS passes `bids-validator`
- [ ] Sync metadata JSON exists in `annot/` folder
- [ ] Task windows TSV populated with T0–T4 boundaries
- [ ] Participant signal map TSV shows correct modality-to-participant mapping

## Next Steps After Processing

```bash
# 1. Validate BIDS compliance
bids-validator E:\processed_data

# 2. Check sync metadata
jq . E:\processed_data\sub-01\ses-*/annot/*_sync_metadata.json | head -40

# 3. Verify task windows
head -10 E:\processed_data\sub-01\ses-*/annot/*_task_run_windows.tsv

# 4. List output modalities
ls -R E:\processed_data\sub-01\ses-*/et E:\processed_data\sub-01\ses-*/physio

# 5. (Optional) Run downstream 3D pose pipeline
python tools/video_only_3d_pipeline.py \
  --videos-dir E:\processed_data\sub-01\ses-XX\video \
  --calibration E:\processed_data\sub-01\ses-XX\video\video_camera_calibration.toml \
  --output-dir E:\processed_data\sub-01\ses-XX\mocap
```
