# AffectAI Data Processing — Copilot Instructions

> **Always-active context for GitHub Copilot.** Read this file at the start of every session before editing any code.

---

## Session startup checklist

1. Read `docs/llm/context_snapshot.md` — current project state, pipeline status, known issues
2. Read `docs/decisions.md` — architectural decisions (consult before any design change)
3. Check `CHANGES.md` for the most recent modifications

---

## Project identity

**What this is:** Post-collection multimodal data processing for a group social-affect study. Data collection is complete. The repo contains three offline processing pipelines (BIDS packaging, 3D pose/gaze, QC) and supporting tooling.

**Language:** Python 3.10+  
**Package manager:** conda (see `environment-freemocap.yml`) + pip (`pyproject.toml`)  
**Linter/formatter:** ruff (`line-length = 100`, `target-version = "py310"`)  
**Tests:** pytest — `make check` runs both ruff and pytest  

---

## Repository layout (quick reference)

```
tools/                  ← primary workspace (CLI scripts)
  qc/                   ← QC scripts (sync, gaze)
  sync/                 ← sync/frame alignment helpers
src/affectai_capture/   ← importable package
tests/                  ← pytest suite (must pass)
configs/                ← YAML/JSON configs (camera specs, zones, device maps)
docs/                   ← architecture, decisions, guides
docs/llm/               ← AI context files (context_snapshot, prompt_playbook)
metadata/               ← participants.tsv, session inventory
.github/agents/         ← Copilot CLI custom agents
.github/instructions/   ← topic-specific instruction files
```

---

## Code style rules

- **Type hints:** All public functions and methods must have type-annotated signatures
- **Docstrings:** `tools/` scripts: module-level docstring describing purpose + CLI args. Functions: single-line or Google-style for non-trivial logic
- **Imports:** stdlib → third-party → local, separated by blank lines. No star imports
- **Logging:** Use `logging` module (not `print`) for diagnostic output. Use `--verbose` / `--quiet` flags to control level
- **CLI:** Use `argparse` with `--help` descriptions on all arguments. All tools must be runnable as `python tools/<script>.py --help`
- **Pathlib:** Use `pathlib.Path` for all file/directory operations (not `os.path`)
- **No bare except:** Always catch specific exceptions, e.g., `except (FileNotFoundError, ValueError) as e:`
- **Config over hardcode:** Read session paths, group IDs, and device parameters from CLI args or config files — never hardcode

---

## Anonymisation rules (critical — participant privacy)

- Participant identifiers in code, logs, events.tsv, LSL markers, and BIDS outputs: **`P1`–`P4` only**
- Real participant names live exclusively in `.private/registration_ledger.jsonl` (gitignored) and in-memory during a session
- Display-only normalisation (first name for tablet/bigscreen display) is permitted but must never reach LSL or log files
- Any tool that handles real names must have an explicit note in its docstring: `# Privacy: real names are never written to disk or LSL`

---

## BIDS conventions

- Directory structure: `sub-{id}/ses-{id}/` with modality subdirs: `eeg/`, `et/`, `physio/`, `audio/`, `video/`, `mocap/`, `beh/`, `annot/`
- Filenames: `sub-{id}_ses-{id}_task-{T0..T4}_run-01_<suffix>.<ext>`
- Task labels: `T0` (baseline/intro), `T1`–`T4` (study tasks)
- One authoritative `events.tsv` per session (timeline spine); never duplicate events across files
- `participants.tsv` at study root for cross-session roster (anonymised IDs only)

---

## Two-PC architecture rules

| PC | Role | Key tools |
|----|------|-----------|
| **AV PC** | Cameras (7×Jabra P20/P50) + Mics (5×DPA) + LSL clock streams | `ffmpeg_multicap.py`, `dpa_recorder.py`, `online_multicam_feed.py` |
| **Recording PC** | LSL central recorder + Stimuli server + Tobii bridge + EmotiBit + Vicon | `session_orchestrator*.py`, `tobii_glasses_lsl_bridge.py`, `lsl_xdf_recorder.py` |

- All tools must be dual-PC-aware or clearly scoped to one role in their `--help`
- Session ID must match across both PCs (`YYYYMMDD_{group}_run{NN}`)
- The lock file in `<out_root>/_session_locks/` is the synchronisation contract between PCs

---

## Pipeline-specific guidance

### Pipeline 1 — Sync & BIDS packaging
- Entry: `tools/multisource_to_bids_runs.py`
- Phase-aware task windows: `T0` intro→finish, `T1`–`T4` tobii_calibration→finish
- Always run with `--help` first to check required input structure
- Outputs go to `sub-{id}/ses-{id}/` under the configured `--out-root`

### Pipeline 2 — 3D pose, gaze & gesture
- Entry: `tools/video_only_3d_pipeline.py` (use `--dry-run` to validate prerequisites)
- Camera flip flags: `--flip-cameras cam_0 cam_1 cam_2 cam_3` for ceiling-mounted P20s
- Calibration TOML is required; if missing, the pipeline can auto-calibrate from six P20 videos
- Coordinate system: all 3D outputs share the calibrated world frame

### Pipeline 3 — QC
- `tools/qc/qc_sync_report.py` — frame/LSL sync alignment report
- `tools/qc/qc_tobii_world_gaze.py` — gaze QC scatter/timeseries plots
- QC tools are read-only; they must not modify source data

---

## Testing guidelines

- Test file for each major tool: `tests/test_<tool_name>.py`
- Use `pytest.mark.parametrize` for data-driven tests
- Never modify ground-truth fixtures in `tests/` without adding a regression test that captures the old behaviour
- `make check` must pass before any commit

---

## Commit hygiene

- Run `make check` before committing
- Commit message format: `<type>: <short description>` where type is `fix`, `feat`, `refactor`, `docs`, `test`, or `chore`
- Keep commits atomic: one logical change per commit
