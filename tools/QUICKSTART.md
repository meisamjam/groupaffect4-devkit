# Quick Reference Guide — Master BIDS Pipeline

## 5-Minute Setup

### Prerequisites
```bash
# Python 3.10+
python --version

# NVIDIA GPU (optional but recommended)
nvidia-smi

# Verify environment
pip list | grep torch opencv pandas pydantic
```

### Installation (One-Time)
```bash
cd affectai-data-processing-seed
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev,lsl,video,audio]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python
```

## Common Commands

### ⚡ Fast (10-20 min)
```bash
cd tools
python run_pipeline.py \
    --data-dir ../affectai-data-processing-seed/data \
    --output-dir E:\processed_data \
    --preset quick
```

### ⭐ Recommended (1-2 hours)
```bash
python run_pipeline.py \
    --data-dir ../affectai-data-processing-seed/data \
    --output-dir E:\processed_data \
    --preset standard
```

### 🚀 Maximum (2-3 hours)
```bash
python run_pipeline.py \
    --data-dir ../affectai-data-processing-seed/data \
    --output-dir E:\processed_data \
    --preset full
```

### 💻 Dual GPU (1-1.5 hours)
```bash
python run_pipeline.py \
    --data-dir ../affectai-data-processing-seed/data \
    --output-dir E:\processed_data \
    --preset dual_gpu
```

## Output Check

After completion:
```bash
# View summary
cat E:\processed_data\pipeline_report.json | python -m json.tool

# Validate BIDS
pip install bids-validator
bids-validator E:\processed_data

# Check file counts
dir /s E:\processed_data\*.tsv /b | find /c /v ""  # Windows
find E:\processed_data -name "*.tsv" | wc -l        # PowerShell
```

## Key Paths
- **Input inventory:** `affectai-data-processing-seed/data/high_level_*_inventory.*`
- **Output root:** `E:\processed_data`
- **Pipeline script:** `affectai-data-processing-seed/tools/master_bids_pipeline.py`
- **Launcher:** `affectai-data-processing-seed/tools/run_pipeline.py`
- **Batch script:** `affectai-data-processing-seed/tools/run_pipeline.bat`
- **Reports:** `E:\processed_data\pipeline_report.json`

## Troubleshooting Checklist

- [ ] Data directory exists: `ls affectai-data-processing-seed/data/high_level_*`
- [ ] Output directory writable: Create test file in `E:\processed_data`
- [ ] Python 3.10+: `python --version`
- [ ] GPU detected (if using 3D pose): `nvidia-smi`
- [ ] MediaPipe installed: `python -c "import mediapipe; print(mediapipe.__version__)"`
- [ ] Enough disk space: `27 sessions × 50-200 GB = 1.4-5.4 TB` (adjust for your options)

## Performance Cheat Sheet

| Hardware | Preset | Duration | Memory |
|----------|--------|----------|--------|
| CPU-only | quick | 30 min | 8 GB |
| 1x RTX 3060 | quick | 10 min | 12 GB |
| 1x RTX 3060 | standard | 2 hr | 16 GB |
| 1x RTX 3090 | standard | 1 hr | 20 GB |
| 2x RTX 3090 | full | 45 min | 32 GB |

## File Structure After Completion

```
E:\processed_data/
├── sub-00/ses-.../video/      ← Split task videos
├── sub-00/ses-.../audio/      ← Split task audio
├── sub-00/ses-.../et/         ← Eye tracking
├── sub-00/ses-.../pose3d/     ← 3D skeleton (if --enable-3d-pose)
├── sub-00/ses-.../facehand/   ← Face/hands (if --enable-face-hand)
├── dataset_description.json    ← BIDS metadata
├── participants.tsv            ← Subject roster
└── pipeline_report.json        ← Execution summary
```

## Next Steps

1. **Run pipeline** — Choose preset and execute
2. **Monitor progress** — Watch logs or check task manager
3. **Validate output** — Run BIDS validator
4. **Analyze results** — Use output in downstream analysis

---

**Questions?** See [PIPELINE_README.md](PIPELINE_README.md) for detailed documentation.
