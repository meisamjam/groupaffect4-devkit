# Raw Data Upload + Tobii Ingest + Raw→BIDS

This runbook covers post-session handling for both AV PC and Recording PC.

## 1) Upload raw session data to Azure Blob

From GUI (`tools/session_orchestrator_gui.py`):
- Open **Data Ops** panel
- Set **Azure Blob** container URL, or load Blob JSON with `account_name` + `account_key` + `container_name`
- Click **Upload Raw (Azure)**

CLI equivalent:

```bash
python tools/upload_raw_data.py \
  --session-dir sessions/sub-01/ses-20260307_grpA_run01 \
  --destination-url "https://<account>.blob.core.windows.net/<container>?<SAS>" \
  --role recording-pc
```

Alternative CLI mode (Azure SDK; no `azcopy` required):

```bash
set AFFECTAI_AZURE_ACCOUNT_KEY=<storage-account-key>
python tools/upload_raw_data.py \
  --session-dir sessions/sub-01/ses-20260307_grpA_run01 \
  --account-name <storage-account> \
  --container-name <container> \
  --role recording-pc
```

What is uploaded:
- Entire `ses-*` directory (raw + sourcedata + modality folders)
- Sync manifest written locally before upload:
  - `sourcedata/sync/raw_upload_manifest_<role>.json`

Sync artifacts tracked in manifest:
- `events.tsv`
- `*.xdf`
- ffmpeg LSL JSONL, frame logs, progress logs
- Tobii LSL NDJSON

## 2) Tobii on-device video workaround (manual downloads)

Because Tobii scene videos are recorded on glasses and downloaded later, ingest them into the same session:

From GUI (`Data Ops`):
- Click **Ingest Tobii Download**
- Pick downloaded Tobii folder
- Enter device ID (e.g., `p1`, `p2`, `tobii-01`)

CLI equivalent:

```bash
python tools/ingest_tobii_downloads.py \
  --session-dir sessions/sub-01/ses-20260307_grpA_run01 \
  --download-root D:/TobiiExports/P1_20260307 \
  --device-id p1
```

Output:
- Files copied under `sourcedata/tobii_device/<device_id>/`
- Index updated:
  - `sourcedata/tobii_device/tobii_download_index.json`

## 3) Convert raw AV + Recording + Tobii sources to BIDS-oriented outputs

From GUI (`Data Ops`):
- Click **Raw → BIDS**

CLI equivalent:

```bash
python tools/raw_to_bids.py \
  --session-dir sessions/sub-01/ses-20260307_grpA_run01
```

Optional hard-link mode (saves space where possible):

```bash
python tools/raw_to_bids.py \
  --session-dir sessions/sub-01/ses-20260307_grpA_run01 \
  --link
```

Raw sources used:
- AV PC raw: `sourcedata/av/**`
- Recording PC raw: `*.xdf` and `sourcedata/tobii_lsl/*.ndjson`
- Tobii downloaded raw: `sourcedata/tobii_device/**`

Generated outputs:
- Canonicalized files in `video/`, `audio/`, `et/`, `physio/`, `beh/`, `annot/`
- Summary report:
  - `annot/sub-*_ses-*_task-T0T1T2T3T4_raw_to_bids_summary.json`

Notes:
- Raw vendor files are never overwritten.
- XDF extraction runs only if `pyxdf` is installed.
- LSL XDF extraction supports both legacy and participant-level unified stream names for Tobii/EmotiBit (`Tobii_*`/`TobiiGlasses*`, `EmotiBit_*`/`Emotibit_*`).
- EmotiBit participant assignment for unified streams can be configured in `configs/emotibit_participants.json` (or via `--participant-map`).

## 4) Merge split source folders and chunk into task runs (T0–T4)

Use this when data is stored in separate source roots (for example `AV/`, `Recording/`, `Stimuli/`, `Tobii/`) and you want one merged BIDS session plus per-task run files.

```bash
python tools/multisource_to_bids_runs.py \
  --av-session-dir sessions/Final/AV/ses-20260319_grp-15_run01 \
  --recording-session-dir sessions/Final/Recording/ses-20260319_grp-15_run01 \
  --stimuli-dir sessions/Final/Stimuli/20260319_grp-15_run01_20260319_125450 \
  --tobii-dir sessions/Final/Tobii/20260319T120614Z \
  --output-session-dir sessions/Final/merged/sub-99/ses-20260319_grp-15_run01 \
  --link
```

Outputs include:
- merged raw under `sourcedata/` in the output session
- canonical BIDS-oriented modality files via `tools/raw_to_bids.py`
- task windows: `annot/sub-*_task-T0T1T2T3T4_task_run_windows.tsv`
- per-task event files: `beh/sub-*_task-T*_run-01_events.tsv`
- run-sliced LSL tables (when present):
  - `et/sub-*_task-T*_run-01_acq-lsl_tobii.tsv.gz`
  - `physio/sub-*_task-T*_run-01_acq-lsl_emotibit.tsv.gz`
  - `annot/sub-*_task-T*_run-01_acq-lsl_sync.tsv`
  - `beh/sub-*_task-T*_run-01_recording-lsl_events.tsv`

Notes:
- `--link` prefers hard-links to avoid duplicating large files.
- Add `--no-write-session-events` to keep an existing `events.tsv` unchanged.
- Raw vendor files are still preserved as ingested inputs; only derived outputs are generated in modality folders.
