# End-to-End Guide: Dataset Download to Analysis and Models

This guide provides a single reproducible path from a downloaded GroupAffect-4 dataset to:
- feature extraction
- preprocessing experiments
- model training (SVM, neural temporal, ordinal)
- analysis outputs and reports

All commands are repo-root relative.

## 0) Choose Your Entry Path

Use one of these starts:

1. Public BIDS release already downloaded or about to be downloaded.
- Continue with Sections 1 to 8.

2. Raw split sources (AV + Recording + Stimuli + Tobii) that still need conversion.
- First run [docs/raw_data_upload_and_bids_conversion.md](docs/raw_data_upload_and_bids_conversion.md), then continue from Section 3.

## 1) Environment Setup

```powershell
conda env create -f environment-freemocap.yml
conda activate affectai-gpu
```

Optional development install:

```powershell
pip install -e .
```

## 2) Download GroupAffect-4 and Place It Correctly

The processing scripts expect a BIDS root containing `sub-*/ses-*`.

```powershell
# Replace with the published Zenodo record ID from the dataset release page.
$recordId = "<ZENODO_RECORD_ID>"
Invoke-WebRequest -Uri "https://zenodo.org/records/$recordId/files/groupaffect4.zip?download=1" -OutFile "data\groupaffect4.zip"
Expand-Archive -Path "data\groupaffect4.zip" -DestinationPath "data\zenodo"
```

Verify structure (adjust path if the archive adds an extra folder layer):

```powershell
Get-ChildItem data\zenodo
```

Expected at minimum:
- `participants.tsv`
- `sub-*/ses-*/beh`
- `sub-*/ses-*/et`
- `sub-*/ses-*/physio`
- `sub-*/ses-*/annot`

## 3) Build Model-Ready Pickles from BIDS

Run a coverage check first:

```powershell
python tools/mumt/pickle_generation_affectai.py --dataset-path data/zenodo --window-sec 15 --validate-coverage
```

Generate labeled windows:

```powershell
python tools/mumt/pickle_generation_affectai.py --dataset-path data/zenodo --window-sec 15 --output data/mumt/dataset_15s.pkl
python tools/mumt/pickle_generation_affectai.py --dataset-path data/zenodo --window-sec 30 --output data/mumt/dataset_30s.pkl
```

Generate unlabeled pretraining windows:

```powershell
python tools/mumt/pickle_generation_pretrain.py --dataset-path data/zenodo --window-sec 30 --step-sec 15 --output data/mumt/pretrain_dataset.pkl
```

## 4) Build the Label-Augmented Pool

```powershell
python tools/mumt/label_augmentation.py --dataset data/mumt/dataset_15s.pkl --pretrain data/mumt/pretrain_dataset.pkl --output data/mumt/augmented_pool_slow.pkl
```

Optional cross-physiology augmentation signals:

```powershell
python tools/mumt/label_augmentation.py --dataset data/mumt/dataset_15s.pkl --pretrain data/mumt/pretrain_dataset.pkl --use-cross-physio --output data/mumt/augmented_pool_crossphys.pkl
```

## 5) Feature Extraction and Analysis

### 5.1 Physio + Pupil + Group Dynamics + Semantic Biomarkers

```powershell
python tools/features/run_feature_pipeline.py --data-root data/zenodo --out-dir data/derived_features --window-s 30 --step-s 15
```

### 5.2 Paper-Facing Physio/Autonomic Analyses

```powershell
python tools/features/analyze_physio_paper.py --features-dir data/derived_features --results-dir results/physio --figures-dir figures/physio
python tools/features/analyze_autonomic_paper.py --features-dir data/derived_features --results-dir results/autonomic --figures-dir figures/autonomic
```

### 5.3 Video Modality Features

Use [docs/video_feature_extraction.md](docs/video_feature_extraction.md) for full options.

```powershell
python tools/extract_video_features.py --videos-dir <session_video_dir> --output-dir <session_features_video_dir> --dry-run
python tools/extract_video_features.py --videos-dir <session_video_dir> --output-dir <session_features_video_dir> --body --hands --faces --markers --body-backbone mediapipe-pose
```

### 5.4 3D Pose + World Gaze + Gesture Pipeline

Use [docs/video_only_3d_pipeline.md](docs/video_only_3d_pipeline.md) for calibration and inputs.

```powershell
python tools/video_only_3d_pipeline.py --dry-run --calibration <calibration.toml> --videos-dir <session_video_dir> --tracker-config <tracker.yaml> --pose-root <pose_json_root> --output-dir <output_dir>
```

## 6) Preprocessing Experiments (Including Speech-Enhanced Branch)

If your dataset includes populated `speech_features`, run:

```powershell
python tools/mumt/enrich_speech_features.py --input data/mumt/dataset_15s_speech.pkl --output data/mumt/dataset_15s_speech_enriched.pkl
python tools/mumt/improved_preprocessing.py --input data/mumt/dataset_15s_speech_enriched.pkl --output data/mumt/dataset_15s_v2.pkl
```

Then compare v2 variants:

```powershell
python tools/mumt/run_v2_experiments.py --v2-dataset data/mumt/dataset_15s_v2.pkl --orig-dataset data/mumt/dataset_15s.pkl --pool data/mumt/augmented_pool_slow.pkl --out results/v2_experiment_comparison.csv
python tools/mumt/per_dim_optimized_svm.py --v2-dataset data/mumt/dataset_15s_v2.pkl --orig-dataset data/mumt/dataset_15s.pkl --pool data/mumt/augmented_pool_slow.pkl --out results/per_dim_optimized.csv
```

## 7) Model Training and Benchmarking

### 7.1 Baseline MuMT MLP/Pool (No Augmentation)

`NOPOOL` is an intentional non-existent path that disables pool loading.

```powershell
python tools/mumt/train_simple.py --dataset data/mumt/dataset_15s.pkl --split-mode task --test-task T3 --augmented-pool NOPOOL --seed 42 --epochs 200 --eval-every 5 --patience 60
```

### 7.2 Augmented MuMT Baseline

```powershell
python tools/mumt/train_simple.py --dataset data/mumt/dataset_15s.pkl --split-mode task --test-task T3 --augmented-pool data/mumt/augmented_pool_slow.pkl --aug-frac 0.3 --dim-aug-scale v=1.0,a=0.2,d=0.6 --seed 42
```

### 7.3 SVM Augmentation Comparison (A0 to AP2)

```powershell
python tools/mumt/svm_aug_comparison.py --dataset data/mumt/dataset_15s.pkl --pool data/mumt/augmented_pool_slow.pkl --pool-gsr data/mumt/augmented_pool_gsr.pkl --out results/svm_aug_comparison.csv
```

### 7.4 Temporal Models (MLP / Conv1D / GRU)

```powershell
python tools/mumt/train_temporal.py --dataset data/mumt/dataset_15s.pkl --pool data/mumt/augmented_pool_slow.pkl --encoder all --aug all --test-task T3 --out results/temporal_comparison.csv
```

### 7.5 Literature-Style Experiment Suite

```powershell
python tools/mumt/literature_experiments.py --encoder conv1d --seeds 3 --out results/literature_experiments_conv1d.csv
python tools/mumt/literature_experiments.py --encoder gru --seeds 3 --out results/literature_experiments_gru.csv
```

### 7.6 Ordinal Original-Label Path

See [tools/mumt/ORDINAL_CLASSIFICATION.md](tools/mumt/ORDINAL_CLASSIFICATION.md).

```powershell
python tools/mumt/train_ordinal.py --dataset data/mumt/dataset_15s.pkl --split-mode task --test-task T3 --output-csv results/ordinal_t3.csv
python tools/mumt/train_ordinal.py --dataset data/mumt/dataset_15s.pkl --task-cv --output-csv results/ordinal_taskcv.csv
```

## 8) Optional Paper Table Reproduction Wrapper

You can use `ICMI_rep` wrappers:

```powershell
python ICMI_rep/reproduce.py --list
python ICMI_rep/reproduce.py --table all
```

When wrapper arguments drift from core tools, prefer direct `tools/mumt/*.py` commands from this guide as the canonical interface.

## 9) Expected Output Locations

- Pickles: `data/mumt/*.pkl`
- Feature tables: `data/derived_features/*.tsv`
- Physio/autonomic analysis: `results/physio/`, `results/autonomic/`
- Model outputs: `results/*.csv`
- Figures: `figures/physio/`, `figures/autonomic/`

## 10) Final Validation

```powershell
make check
```

If you process many sessions, also run QC reports per session:

```powershell
python tools/qc/qc_sync_report.py --help
python tools/qc/qc_tobii_world_gaze.py --help
```
