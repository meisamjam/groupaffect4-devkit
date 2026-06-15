# Ordinal Original-Label Classification

Standalone ordinal VAD training for GroupAffect-4.  This path is separate from
the original 3-class binned experiments, so `train_simple.py`, `model_simple.py`,
and the existing macro-F1 tables remain reproducible.

## Scope

- Uses original SAM labels `1..9` for valence, arousal, and dominance.
- Masks missing dominance labels, notably in T4.
- Uses only physiology and eye-tracking feature groups: gaze, pupil, EDA, PPG,
  and IMU.  Audio/speech features are not part of this path.
- Predicts eight cumulative logits per VAD dimension:

```text
logit_k ~= P(y > k), k = 1..8
score = 1 + sum(sigmoid(logit_k))
```

Primary metrics are MAE, Spearman correlation, and quadratic-weighted kappa
(QWK).  The 1-9 macro-F1 is sparse at this data size and should be treated only
as a secondary diagnostic.

## Files

- `tools/mumt/model_ordinal.py` - ordinal MLP/Pool models with cumulative heads.
- `tools/mumt/train_ordinal.py` - dataset wrapper, ordinal loss, metrics, splits,
  optional GP-ordinal augmentation, and CSV export.

## Commands Run

Canonical T3 ordinal run:

```powershell
conda run -n affectai-gpu python tools\mumt\train_ordinal.py `
  --dataset data\mumt\dataset_15s.pkl `
  --split-mode task --test-task T3 `
  --epochs 200 --eval-every 5 --patience 60 `
  --batch 16 `
  --output-csv results\ordinal_t3.csv `
  --ckpt-dir data\mumt\ordinal_checkpoints `
  --device auto
```

Rotating task-CV ordinal run:

```powershell
conda run -n affectai-gpu python tools\mumt\train_ordinal.py `
  --dataset data\mumt\dataset_15s.pkl `
  --task-cv `
  --epochs 200 --eval-every 5 --patience 60 `
  --batch 16 `
  --output-csv results\ordinal_taskcv.csv `
  --ckpt-dir data\mumt\ordinal_checkpoints `
  --device auto
```

GP-ordinal smoke test:

```powershell
conda run -n affectai-gpu python tools\mumt\train_ordinal.py `
  --dataset data\mumt\dataset_15s.pkl `
  --split-mode task --test-task T3 `
  --epochs 1 --eval-every 1 --patience 0 `
  --batch 64 `
  --augmented-pool data\mumt\augmented_pool.pkl `
  --aug-frac 0.30 `
  --dim-aug-scale "v=1.0,a=0.2,d=0.6" `
  --ckpt-dir= `
  --output-csv results\ordinal_gp_smoke.csv `
  --device auto
```

For task splits, GP augmentation is filtered to training tasks rather than by
session.  Session filtering would remove every augmented row because every
session contributes to T3/T4 task windows.  GP-ordinal targets are derived from
stored posterior moments `(mu, sigma)`:

```text
q_k = P(Y > k) = 1 - Phi(k + 0.5; mu, sigma)
```

Stored 3-class GP probabilities are ignored.

## Results

Canonical T3 (`results/ordinal_t3.csv`):

| Run | MAE | Spearman | QWK | 1-9 macro-F1 |
|---|---:|---:|---:|---:|
| T3 | 1.474 | 0.324 | 0.205 | 0.051 |

Task-CV (`results/ordinal_taskcv.csv`):

| Fold | MAE | Spearman | QWK | 1-9 macro-F1 |
|---|---:|---:|---:|---:|
| T2 | 1.424 | -0.001 | 0.008 | 0.073 |
| T3 | 1.441 | 0.412 | 0.283 | 0.072 |
| T4 | 1.650 | 0.235 | 0.158 | 0.067 |
| Mean | 1.505 | 0.215 | 0.150 | 0.071 |
| Std | 0.103 | 0.169 | 0.113 | 0.003 |

Interpretation:

- Ordinal training removes fold-specific threshold choices and reports errors in
  original SAM units.
- T3 transfers best under the ordinal target.
- T2 has near-zero rank correlation because the strict temporal split trains on
  T0 only, which is too little task diversity for ordered affect prediction.
- T4 means exclude dominance because dominance labels are missing for T4.
