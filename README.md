# GroupAffect-4 Devkit

**GroupAffect-4 Devkit: Target Construction, Augmentation, Preprocessing, and Temporal Modeling for Physiological Affect Sensing**

> Meisam Jamshidi Seikavandi · CogniSense Lab, GN Hearing · [paper](paper/arxiv/) · [dataset](https://zenodo.org/records/XXXXXXX) · Python 3.10+

---

## Overview

GroupAffect-4 is a multimodal physiological dataset collected from groups of four participants in naturalistic social-interaction tasks.
This devkit provides the full offline processing stack — from raw BIDS-formatted recordings to trained affect models — along with reproducibility scripts and the companion arXiv paper source.

**What the devkit covers:**

| Stage | Entry point |
|---|---|
| BIDS packaging & sync | `tools/multisource_to_bids_runs.py` |
| 3-D pose, gaze & gesture | `tools/video_only_3d_pipeline.py` |
| QC (sync + gaze reports) | `tools/qc/` |
| Physiological feature extraction | `tools/features/` |
| Label construction & GP augmentation | `tools/mumt/label_augmentation.py` |
| SVM / MLP baselines | `tools/mumt/train_simple.py` |
| Temporal deep models (Conv1D / GRU + SimCLR) | `tools/mumt/train_temporal.py` |
| Ordinal regression | `tools/mumt/train_ordinal.py` |

---

## Requirements

```bash
conda env create -f environment-freemocap.yml
conda activate affectai-gpu
pip install -e .          # installs the src/affectai_capture package
```

Python 3.10+, CUDA 11.8+ recommended for temporal model training.
Full dependency list: [`requirements.txt`](requirements.txt) / [`pyproject.toml`](pyproject.toml).

---

## Quick Start — Dataset to Results in 5 Steps

### 1. Download the dataset

```powershell
$recordId = "<ZENODO_RECORD_ID>"   # replace with published record
Invoke-WebRequest "https://zenodo.org/records/$recordId/files/groupaffect4.zip?download=1" `
    -OutFile "data\groupaffect4.zip"
Expand-Archive "data\groupaffect4.zip" "data\zenodo"
```

### 2. Build the labeled feature dataset

```bash
python tools/mumt/pickle_generation_affectai.py \
    --dataset-path data/zenodo \
    --window-sec 15 \
    --output data/mumt/dataset_15s.pkl
```

### 3. Build unlabeled pretraining windows & GP-augmented pool

```bash
python tools/mumt/pickle_generation_pretrain.py \
    --dataset-path data/zenodo --window-sec 30 --step-sec 15 \
    --output data/mumt/pretrain_dataset.pkl

python tools/mumt/label_augmentation.py \
    --dataset  data/mumt/dataset_15s.pkl \
    --pretrain data/mumt/pretrain_dataset.pkl \
    --output   data/mumt/augmented_pool_slow.pkl
```

### 4. Train & evaluate

```bash
# SVM baseline
python tools/mumt/train_simple.py \
    --dataset data/mumt/dataset_15s.pkl \
    --split-mode task --test-task T3 --seed 42

# Temporal model (Conv1D + SimCLR + time-warp augmentation)
python tools/mumt/train_temporal.py \
    --dataset data/mumt/dataset_15s.pkl \
    --pool    data/mumt/augmented_pool_slow.pkl \
    --encoder conv1d --aug simclr --out results/temporal.csv
```

### 5. Run QC reports

```bash
python tools/qc/qc_sync_report.py     --session-dir <bids_session_dir>
python tools/qc/qc_tobii_world_gaze.py --session-dir <bids_session_dir>
```

See [`docs/END_TO_END_DATASET_TO_MODELS.md`](docs/END_TO_END_DATASET_TO_MODELS.md) for the complete walkthrough.

---

## Repository Structure

```
paper/                  ← arXiv paper source (LaTeX + figures)
tools/
  multisource_to_bids_runs.py   ← Pipeline 1: BIDS packaging
  video_only_3d_pipeline.py     ← Pipeline 2: 3-D pose/gaze
  qc/                           ← Pipeline 3: QC scripts
  features/                     ← Physiological feature extraction
  mumt/                         ← Label augmentation & model training
src/affectai_capture/   ← Importable Python package
tests/                  ← pytest suite (run with: make check)
configs/                ← Camera specs, zone maps, device configs
docs/                   ← Architecture, decisions, execution guides
metadata/               ← participants.tsv, session inventory
```

---

## BIDS Conventions

Sessions are organised as `sub-{P1..P4}/ses-{YYYYMMDD_grpNN_runNN}/` with modality subdirs `eeg/`, `et/`, `physio/`, `audio/`, `video/`, `mocap/`, `beh/`, `annot/`.
Filenames follow the pattern `sub-{id}_ses-{id}_task-{T0..T4}_run-01_{suffix}.{ext}`.
Task labels: `T0` baseline/intro, `T1`–`T4` study tasks.

---

## Development

```bash
make check   # ruff lint + pytest
```

Code style: ruff with `line-length = 100`, `target-version = "py310"`.
All public functions must have type-annotated signatures.

---

## Citation

If you use this dataset or devkit in your research, please cite:

```bibtex
@misc{groupaffect4devkit2026,
  title   = {{GroupAffect-4 Devkit}: Target Construction, Augmentation,
             Preprocessing, and Temporal Modeling for Physiological Affect Sensing},
  author  = {Jamshidi Seikavandi, Meisam},
  year    = {2026},
  url     = {https://github.com/meisamjam/groupaffect4-devkit},
  note    = {arXiv preprint}
}
```

Dataset DOI: `10.5281/zenodo.XXXXXXX` (replace with published DOI).

---

## License

Code: MIT.  
Dataset: Creative Commons Attribution 4.0 (CC BY 4.0).  
See [`LICENSE`](LICENSE) for details.
