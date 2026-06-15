# AGENTS.md

AffectAI Data Processing — post-collection multimodal processing toolkit for a group social-affect research study. Data collection is complete; this repo contains the pipelines that turn raw recordings into BIDS-structured, 3D-annotated, QC-verified outputs.

> **Cross-platform standard** — this file is read automatically by GitHub Copilot, Copilot CLI, and compatible AI coding assistants. All project rules live here or are linked from here.

---

## Structure

| Path | Purpose |
|------|---------|
| `tools/` | **Primary workspace** — 30+ CLI scripts across three pipelines (BIDS, 3D, QC) |
| `src/affectai_capture/` | Importable Python package (registration, stream bridges, helpers) |
| `tests/` | pytest test suite — must pass before every commit |
| `configs/` | YAML/JSON configuration files (camera specs, zone maps, device maps) |
| `docs/` | Architecture, decisions, calibration guides, LLM prompt playbooks |
| `docs/llm/` | AI-facing context: `context_snapshot.md` (current state), `prompt_playbook.md` |
| `docs/decisions.md` | Authoritative record of architectural decisions — read before changing design |
| `metadata/` | BIDS-compliant metadata (participants.tsv, session inventory) |
| `.github/agents/` | Custom Copilot CLI agent files (`.agent.md`) |
| `.github/instructions/` | Topic-specific auto-loaded instruction files |
| `.github/copilot-instructions.md` | GitHub Copilot specific rules (always-active) |

---

## Pipelines at a glance

| # | Pipeline | Entry point |
|---|----------|------------|
| 1 | **Sync & BIDS packaging** | `tools/multisource_to_bids_runs.py` |
| 2 | **3D pose, gaze & gesture** | `tools/video_only_3d_pipeline.py` |
| 3 | **Analysis & QC** | `tools/qc/qc_sync_report.py`, `tools/qc/qc_tobii_world_gaze.py` |

---

## Do

- Read `docs/llm/context_snapshot.md` at the start of every session — it contains the authoritative project state
- Read `.github/copilot-instructions.md` before making any code or tool change
- Read `docs/decisions.md` before proposing or changing architectural decisions
- Keep all participant data anonymised: use `P1`–`P4` IDs in logs, events.tsv, LSL markers — never real names
- Follow BIDS naming conventions: `sub-{id}/ses-{id}/` hierarchy, underscore-separated key-value entities, recognised modality folders (`eeg/`, `et/`, `physio/`, `video/`, `beh/`, `annot/`)
- Run `make check` (ruff + pytest) before suggesting a commit; all tests must pass
- Respect the two-PC architecture: AV PC handles cameras/audio, Recording PC handles LSL/XDF/stimuli
- Use `--dry-run` flags on pipeline scripts to validate prerequisites before proposing a full run
- Python 3.10+ features are welcome: type hints, `match`, `dataclasses`, `pathlib`
- Dependencies go in `pyproject.toml`; use optional extras for heavy deps (`[video]`, `[freemocap]`)

## Don't

- Do not write real participant names anywhere in code, comments, logs, or generated files
- Do not modify `tests/` fixtures or ground-truth data without updating the corresponding test
- Do not change BIDS output structure without updating `docs/architecture.md` and `docs/data_flow.md`
- Do not add `tools/` scripts that duplicate existing pipeline steps — extend existing scripts via flags
- Do not assume a single-PC setup — all tools must be dual-PC-aware or clearly scoped to one role
- Do not use bare `except:` clauses; catch specific exceptions and log with `logging`
- Do not hardcode session paths or group IDs — read from config, CLI args, or schedule TSV
- Do not add heavy imports at module level in `tools/` scripts (keep startup fast)

---

## Build & verify

```bash
# Lint + test (required before every commit)
make check

# Run individual pipelines (dry-run first)
python tools/multisource_to_bids_runs.py --help
python tools/video_only_3d_pipeline.py --dry-run --session <session_dir> --calibration <toml>
python tools/qc/qc_sync_report.py --help
python tools/qc/qc_tobii_world_gaze.py --help

# Calibration
python tools/calibrate_charuco.py record --help
python tools/calibrate_charuco.py calibrate --help
```

---

## Available agents (Copilot CLI)

| Agent | Invoke with | Specialty |
|-------|------------|-----------|
| `bids-pipeline` | `/agent bids-pipeline` | BIDS packaging, multisource merge, run chunking |
| `pose3d-pipeline` | `/agent pose3d-pipeline` | 3D pose, multicam calibration, gaze-to-world |
| `calibration` | `/agent calibration` | ChArUco calibration, distortion, ground-plane |
| `qc-analyst` | `/agent qc-analyst` | Sync QC, frame alignment, gaze QC, audit reports |
| `knowledge-manager` | `/agent knowledge-manager` | Audit and maintain docs, context_snapshot.md, decisions log |
| `paper-writer` | `/agent paper-writer` | Draft dataset paper sections targeting ICMI 2026 (ACM long paper, 8 pp) |

Agent files are in `.github/agents/`. To list and switch: `> /agent` inside `copilot` interactive mode.
