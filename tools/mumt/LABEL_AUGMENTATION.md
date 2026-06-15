# Label Augmentation and Preprocessing Documentation

> **MuMT-Affect / GroupAffect-4 — Continuous VAD Label Generation**
> Version 1.0 — May 2026

---

## 0. Scope

This document specifies the complete pipeline for augmenting the 305 labeled windows
(219 train / 53 val / 33 test) with additional weakly-labeled windows drawn from the
8,978-window pretraining pool. It defines:

1. **What label sources are permitted** and why (leakage analysis)
2. **How to generate labels** per source (algorithm-level)
3. **How to encode uncertainty** into soft training targets
4. **How to integrate augmented labels** into the existing training objective
5. **Preprocessing requirements** tied to each source

---

## 1. The Leakage Constraint

### 1.1 Definition

A modality is **leaked** for window `w` when:

```
Label(w, dim) is derived (even partially) from Modality M
AND
Modality M is also an input feature for window w during supervised training
```

This causes the model to implicitly learn the labeling function (e.g., "large EDA phasic
peak → predict arousal=High") rather than the underlying affect relationship. The learned
association is not generalisable — it is a tautology.

### 1.2 Modality Inventory

The five input modalities used by `MuMTAffectGroupAffect`:

| ID | Variable in code | Columns (from `dataset_affectai.py`) | Primary physiological signal |
|----|-----------------|--------------------------------------|------------------------------|
| M1 | `gaze_seq` | `GAZE_SEQ_COLS` — gaze XY, direction 3D×2, validity | Oculomotor / visual attention |
| M2 | `pupil_seq` | `PUPIL_SEQ_COLS` — pupil L/R, validity | Locus coeruleus / cognitive load |
| M3 | `eda_seq` | `EDA_SEQ_COLS` — raw EDA, EDA_Phasic, EDA_Tonic, **HR proxy**, temperature | Sympathetic ANS arousal |
| M4 | `ppg_seq` | `PPG_SEQ_COLS` — IR/Red/Green raw PPG | Cardiac activity / parasympathetic ANS |
| M5 | `imu_seq` | `IMU_SEQ_COLS` — accel XYZ, gyro XYZ | Motor activity / movement |

> **Note:** M3 (`eda_seq`) contains the HR proxy (`value_6`) and skin temperature (`value_10`)
> alongside EDA channels. M4 (`ppg_seq`) contains raw PPG from which HR, RMSSD, and HRV
> are derived. These two modalities carry the primary physiological evidence for
> arousal and valence direction respectively.

### 1.3 Leakage Risk by Label Source

| Label source | Modalities used for label generation | Leakage risk for model inputs | Decision |
|---|---|---|---|
| **Self-report temporal interpolation** | None (OU/GP on Likert scores only) | ✅ Zero | **Approved, all modalities usable as input** |
| **Cross-person self-report propagation** | None (other persons' Likert scores) | ✅ Zero | **Approved, all modalities usable as input** |
| **LOGO-CV OOF model predictions** | All 5 modalities of *other groups* | ✅ Zero (independent model) | **Approved, all modalities usable as input** |
| **Cross-person EDA → target person label** | Person A's EDA → Person B's label | ✅ Zero (A ≠ B) | **Approved with attribution constraint** |
| **Cross-person PPG/HR → target person label** | Person A's PPG → Person B's label | ✅ Zero (A ≠ B) | **Approved with attribution constraint** |
| **Own EDA → own arousal label** | M3 of same window → label of same window | ❌ Direct leakage | **Prohibited during supervised phases** |
| **Own PPG/HR → own valence label** | M4 of same window → label of same window | ❌ Direct leakage | **Prohibited during supervised phases** |
| **Own pupil → own arousal label** | M2 of same window → label of same window | ❌ Direct leakage | **Prohibited during supervised phases** |

**The cross-person rule is the core of the leakage-free design:**

> Person k's physiological signals may inform Person l's (l ≠ k) augmented labels.
> Person k's own physiological signals may NEVER be used to generate Person k's own
> supervised label in the same window.

This means physiological evidence for label generation is always sourced from a
*different group member* in the same session — exploiting emotional contagion and
physiological synchrony — while the target person's own signals remain exclusively
used as model inputs.

---

## 2. Label Source Specifications

### 2.1 Source S1 — Self-Report Temporal Interpolation (OU/GP)

**What it provides:** Continuous VAD trajectories between sparse self-reports for each
individual, using only their own Likert-scale self-reports as observations.

**Theoretical basis:** Ornstein-Uhlenbeck (OU) stochastic process (Kuppens et al., 2010;
Meng et al., 2026). Emotion reverts to a personal baseline with dimension-specific
half-lives:

```
θ_arousal  = log(2) / 1.2   ≈ 0.578 s⁻¹   (half-life 1.2 s)
θ_valence  = log(2) / 12.3  ≈ 0.056 s⁻¹   (half-life 12.3 s)
θ_dominance = log(2) / 6.0  ≈ 0.116 s⁻¹   (half-life ~6 s, estimated)
```

**Algorithm:**

```
For each (subject, session, task):
  1. Collect all self-reports: [(t_i, v_i, a_i, d_i), ...]
  2. For each VAD dimension d:
     a. OU covariance kernel: k(τ) = σ²_d * exp(-θ_d * |τ|)
        σ²_d = empirical variance of self-reports for dimension d
     b. Build GP with OU kernel + observation noise σ_obs = 0.8
     c. Query posterior at each unlabeled window centre t*:
        μ*(t*), σ²*(t*) = GP.predict(t*)
     d. Compute temporal decay weight:
        w_d(t*) = exp(-θ_d * min_i(|t* - t_i|))
  3. Retain windows where w_d(t*) ≥ w_min:
     w_min_arousal  = 0.05  (within ~5 s of a self-report)
     w_min_valence  = 0.10  (within ~22 s of a self-report)
     w_min_dominance = 0.07
```

**Leakage check:** GP observations are Likert scores only. No physiological modality
is involved. All 5 model input modalities (M1–M5) remain clean for this window.

**Output per window:** `(μ*_V, σ²*_V), (μ*_A, σ²*_A), (μ*_D, σ²*_D), w_d`

---

### 2.2 Source S2 — Cross-Person Self-Report Propagation (Group OU/LMC)

**What it provides:** For any unlabeled window of person k at time t, uses self-reports
from other group members (l ≠ k) at nearby times as additional GP observations.

**Theoretical basis:** Emotional contagion (Hatfield et al., 1993; Barsade, 2002).
Group members in collaborative tasks show convergent affect trajectories with
cross-person correlation r ≈ 0.25–0.45 for arousal, r ≈ 0.15–0.35 for valence
(Mou et al., 2019).

**Algorithm:**

```
For each (session, task):
  1. Pool all self-reports from all 4 group members as a multi-output GP:
     Observations: {(t_i, k_i, v_i)} where k_i ∈ {1,2,3,4} is the person index
  2. Inter-person covariance (LMC, single shared latent process):
     k_cross(t, t', k, l) = ρ_{kl} * k_OU(t, t')
     ρ_{kl} estimated per session from within-session pairwise Pearson correlation
     If insufficient data: use population prior ρ_prior = 0.30 (V), 0.35 (A), 0.25 (D)
  3. Cross-person observation noise for person k's report informing person l's label:
     σ_social(k→l) = σ_obs / sqrt(ρ_{kl})   [≈ 1.0–2.5 Likert units]
  4. Query posterior for person l at time t*:
     Use all same-task self-reports from persons k ≠ l as observations with σ_social
     Combine with person l's own S1 posterior (information fusion):
     Fuse: precision-weighted combination of S1 and S2 posteriors
  5. Final weight:
     w_cross(t*, l) = w_S1(t*, l) * (1 + ρ_mean * N_nearby_cross_reports)
     where N_nearby_cross_reports = number of cross-person reports within 1 OU half-life
```

**Key constraint (leakage prevention):** The GP observations used for person l's label
are ONLY the Likert scores from persons k ≠ l. Under no circumstances is person l's
own EDA, PPG, or pupil data used to generate person l's label.

**Output per window:** Tighter `(μ*_d, σ²*_d)` than S1 alone, especially in mid-task
gaps where at least one group member has reported nearby in time.

---

### 2.3 Source S3 — LOGO-CV Out-of-Fold Model Predictions

**What it provides:** Pseudo-labels for all windows of group g generated by a MuMT
model trained on groups 1–9 (never having seen group g). These predictions use all
5 input modalities — but leakage is absent because the predictor is fully independent
of the target group's data.

**Theoretical basis:** Self-training / pseudo-labeling (Scudder, 1965; Lee, 2013).
Out-of-fold predictions avoid the circularity of using a model's own training data
to generate pseudo-labels for itself (Parthasarathy & Busso, 2017).

**Algorithm:**

```
Precompute once (expensive):
  For fold g in {session_1, ..., session_10}:
    1. Train MuMT (v10+perdim, full Phase 0–3) on all groups except g
    2. Run inference on ALL windows of group g (labeled + unlabeled)
    3. Store:
       logits_d(w) for d in {V, A, D}
       confidence_d(w) = max(softmax(logits_d(w)))   [0.33–1.0]
       σ_pred_d(w) = uncertainty from MC-Dropout (20 forward passes, dropout=0.3)

At augmentation time:
  For each unlabeled window w in group g:
    μ_pred_d(w) = argmax(softmax(logits_d(w)))  [0, 1, 2 class index]
    p_soft_d(w) = softmax(logits_d(w) / T_cal)  [temperature-calibrated soft label]
    where T_cal is estimated by Platt scaling on the labeled validation windows of g
    w_oof(w) = confidence_d(w) * (1 - σ_pred_d(w) / σ_max)
```

**Leakage check:** The model predicting group g's labels was trained on groups 1–9.
It has never processed group g's windows. The pseudo-label for window w is a function
of group g's M1–M5 inputs passed through a model that contains no information from
group g. This is fully clean for all modalities.

**Output per window:** `p_soft_d(w)` (3-class soft label vector), `w_oof(w)` (scalar weight)

---

### 2.4 Source S4 — Cross-Person EDA → Target Person Arousal Label

**What it provides:** High-confidence arousal event markers at specific time points,
using EDA from group member k to generate arousal evidence for group member l (l ≠ k),
based on physiological synchrony.

**Theoretical basis:** Inter-person physiological synchrony (Feldman, 2007;
Levenson & Gottman, 1983). Cross-person EDA cross-correlation provides a calibration
coefficient for how reliably one person's SCR event predicts another's arousal elevation.

**Algorithm:**

```
Preprocessing (per session, per task):
  1. Extract EDA_Phasic for all 4 group members
  2. Compute pairwise cross-correlation at lag 0 and ±5s:
     r_EDA(k,l) = max_lag corr(EDA_Phasic_k, EDA_Phasic_l) in lag ∈ [-5s, +5s]
  3. Detect SCR events for each person k:
     scr_events_k = [(t_j, amplitude_j), ...]
     using neurokit2: nk.eda_process(eda_k)['SCR_Peaks'] with amplitude > threshold

At label generation for person l, window w:
  4. For each SCR event from person k (k ≠ l) at time t_j:
     IF t_j falls within window w's time range [t_w - 30s, t_w]:
       arousal_observation = (t_j - 2.5s, A_high=7.0)    [SCR latency correction]
       σ_scr_cross = σ_scr_own / r_EDA(k,l)              [scaled by cross-correlation]
       Add to S1/S2 GP as additional observation with σ = σ_scr_cross
       Typical σ_scr_cross ∈ [1.0, 2.5] depending on r_EDA

  5. Final arousal label for person l at window w:
     Updated GP posterior incorporating S1 + S2 + S4 observations
```

**Leakage prevention:** Person l's own EDA (M3) is NEVER used as a GP observation
for person l's label. Only persons k ≠ l contribute EDA-derived arousal observations.
Person l's M3 is available exclusively as a model input.

**Cross-person attribution table:**

```
Person   Label generation uses EDA from   Model input uses EDA from
------   --------------------------------  --------------------------
P1       P2, P3, P4 only                  P1 only (M3 of P1)
P2       P1, P3, P4 only                  P2 only (M3 of P2)
P3       P1, P2, P4 only                  P3 only (M3 of P3)
P4       P1, P2, P3 only                  P4 only (M3 of P4)
```

---

### 2.5 Source S5 — Cross-Person PPG/HR → Target Person Valence Direction

**What it provides:** Valence direction evidence (positive/negative shift) using
cardiac signals from group member k to generate valence observations for person l (l ≠ k).

**Theoretical basis:** Cacioppo & Berntson (1994) autonomic space model; Thayer & Lane
(2000) neurovisceral integration framework. HR deceleration = parasympathetic dominance =
positive valence (orienting response); HR acceleration = sympathetic dominance = negative
valence (defensive response). Cross-person cardiac synchrony provides inter-individual
transfer (r_HR ≈ 0.20–0.35 in collaborative tasks; Levenson & Gottman, 1983).

**Algorithm:**

```
Preprocessing (per person k, per window):
  1. Compute HR features from PPG (M4) using neurokit2:
     peaks = nk.ppg_findpeaks(ppg_k['value_0'])  [IR channel, most reliable]
     rri   = np.diff(peaks) / sampling_rate       [RR intervals in seconds]
     hr    = 60 / rri.mean()
     rmssd = np.sqrt(np.mean(np.diff(rri)**2))

  2. Compute cardiac direction signal for 5-second non-overlapping bins:
     For bin b at time t_b:
       delta_hr(b)    = hr(b) - hr(b-1)           [positive = acceleration]
       delta_rmssd(b) = rmssd(b) - rmssd(b-1)     [positive = more parasympathetic]
       cardiac_signal(b) = -delta_hr(b) + delta_rmssd(b) * 0.5  [combined, positive = appetitive]

At label generation for person l, window w:
  3. For each cardiac signal from person k (k ≠ l):
     IF bin b overlaps with window w:
       valence_shift = clip(cardiac_signal_k(b) * scale_factor, -1.5, 1.5)
       valence_observation = (μ_valence + valence_shift)
       σ_hr_cross = σ_hr_own / r_HR(k,l)          [calibrated by pairwise HR correlation]
       Typical σ_hr_cross ∈ [2.0, 3.5]
       Add to GP as very weak observation

  4. Scale factor calibration:
     scale_factor = estimated from within-session regression of cardiac_signal → valence reports
     (on labeled windows only, using LOGO-CV to avoid leakage)
```

**Leakage prevention:** Same cross-person attribution as S4. Person l's PPG (M4) is
used only as a model input. Valence evidence comes only from k ≠ l persons' PPG.

---

## 3. Combined Confidence Weights

Each augmented window receives a per-dimension weight `w_d(w)` combining all source
contributions:

```
w_d(w) = w_S1_d(w)                  [OU temporal decay, own reports only]
         * w_S2_d(w)                 [cross-person self-report boost factor ≥ 1.0]
         * [1 + γ * w_S4_d(w)]       [additive SCR cross-person boost, arousal only]
         * [1 + δ * w_S5_d(w)]       [additive HR cross-person boost, valence only]
         * w_oof(w)                  [OOF confidence, if S3 available]

where:
  γ = 0.3  (arousal SCR boost coefficient)
  δ = 0.2  (valence HR boost coefficient)

Final instance weight (loss weighting):
  instance_weight(w) = clip(w_d(w), w_min=0.05, w_max=0.95)
```

The weight is bounded below at 0.05 (never zero — allows gradient flow) and above at
0.95 (never equal to full self-report confidence, which is 1.0).

---

## 4. Soft Label Generation

### 4.1 GP Posterior → Class Probability Vector

The GP posterior at window centre t* is N(μ*, σ²*) on the continuous 1–9 Likert scale.
Convert to 3-class soft label using CDF integration over the same bins as the hard labels:

```python
from scipy.stats import norm

def gp_to_soft_label(mu_star, sigma_star, thresholds=(3.0, 6.0), T_scale=1.0):
    """
    Convert GP posterior N(mu_star, sigma_star²) to soft 3-class label.
    T_scale > 1 softens distribution (more uncertainty).
    T_scale < 1 sharpens distribution (more confidence).
    
    thresholds: bin boundaries on 1-9 Likert scale
                Low:  x ≤ 3
                Mid:  3 < x ≤ 6
                High: x > 6
    """
    sigma_eff = sigma_star * T_scale
    p_low  = norm.cdf(thresholds[0], loc=mu_star, scale=sigma_eff)
    p_high = 1.0 - norm.cdf(thresholds[1], loc=mu_star, scale=sigma_eff)
    p_mid  = 1.0 - p_low - p_high
    return np.array([p_low, p_mid, p_high], dtype=np.float32)
```

### 4.2 Temperature Scaling

The temperature T_scale encodes uncertainty relative to self-report noise:

```
T_scale(w) = 1.0 + (σ*(w) - σ_sr) / σ_max

where:
  σ_sr  = 0.8   [self-report noise level, reference]
  σ_max = max σ*(w) across all augmented windows in split
  T_scale ∈ [1.0, 2.0] for typical augmented windows
```

Windows with σ* = σ_sr (very close to a self-report) get T_scale = 1.0 (same sharpness
as ground truth). Windows far from any self-report get T_scale ≈ 2.0 (near-uniform
soft label).

### 4.3 S3 Soft Labels (OOF predictions)

LOGO-CV OOF predictions are already soft (softmax outputs). Apply Platt scaling:

```python
def calibrate_oof_soft_label(logits, T_platt):
    """T_platt estimated per fold by minimising NLL on labeled val windows."""
    return torch.softmax(torch.tensor(logits) / T_platt, dim=-1).numpy()
```

Merge with GP-derived soft labels via precision weighting:

```
p_merged = (w_GP * p_GP + w_OOF * p_OOF) / (w_GP + w_OOF)
w_GP  = 1.0 / σ²*(w)          [GP precision]
w_OOF = confidence_oof(w)      [OOF model confidence]
```

---

## 5. Training Objective with Augmented Labels

### 5.1 Loss Function

```python
def augmented_loss(pred_logits, targets, instance_weights, is_hard_label):
    """
    pred_logits:      (B, 3) logits for one VAD dimension
    targets:          (B, 3) — one-hot if is_hard_label, soft vector if augmented
    instance_weights: (B,)   — 1.0 for self-report labels, w_d(w) for augmented
    is_hard_label:    (B,)   bool — True for original self-reports
    """
    # Soft cross-entropy works for both hard (one-hot) and soft targets
    log_probs = F.log_softmax(pred_logits, dim=-1)          # (B, 3)
    ce_per_sample = -(targets * log_probs).sum(dim=-1)       # (B,)
    weighted_loss = (instance_weights * ce_per_sample).mean()
    return weighted_loss
```

### 5.2 Batch Composition

Recommended batch composition for supervised phases (Phase 2, Phase 3):

```
Batch of B=64:
  - 32 samples: original self-report windows (w=1.0, hard one-hot labels)
  - 16 samples: S1/S2 interpolated windows (w=0.2–0.8, soft labels)
  - 12 samples: S3 OOF pseudo-labeled windows (w=0.15–0.6, soft labels)
  - 4 samples:  S4/S5 cross-person physio-guided (w=0.05–0.3, soft labels)
```

Original self-report samples always constitute ≥ 50% of each batch. Augmented
samples are up-sampled from the pool proportionally to their weight.

### 5.3 Phase Assignment

| Phase | Uses augmented labels? | Which sources? | Notes |
|-------|----------------------|----------------|-------|
| Phase 0 (SSL pretrain) | No | — | Unlabeled data only; SSL objectives unchanged |
| Phase 1 (personality) | No | — | Personality regression uses original BFI-44 scores only |
| Phase 2 (joint frozen) | Yes | S1, S2, S3 | Start with conservative w_max=0.5; transformers frozen |
| Phase 3 (fine-tune) | Yes | S1, S2, S3, (S4, S5 optional) | Full w_max=0.95; reduce if val F1 drops |

S4 and S5 are marked optional for Phase 3 because cross-person physiological evidence
has the highest σ and risks confusing the fine-tuned heads. Include only if LOGO-CV
shows improvement on validation F1.

---

## 6. Preprocessing Pipeline

### 6.1 Step-by-Step Build Order

```
Step 1 — Run existing preprocessing
  python tools/mumt/pickle_generation_affectai.py
  → data/mumt/dataset.pkl          (305 labeled windows)

  python tools/mumt/pickle_generation_pretrain.py
  → data/mumt/pretrain.pkl         (8,978 unlabeled windows)

Step 2 — Compute session-level OU parameters (once, reuse)
  script: tools/mumt/compute_ou_params.py  [to be created]
  Input:  data/mumt/dataset.pkl  (self-reports only)
  Output: data/mumt/ou_params.json
  Content: {
    "valence":   {"theta": 0.056, "sigma2": 2.1, "mu": 5.3},
    "arousal":   {"theta": 0.578, "sigma2": 1.8, "mu": 4.9},
    "dominance": {"theta": 0.116, "sigma2": 1.9, "mu": 5.1}
  }
  Note: If estimated locally, validate against Meng et al. (2026) values.
        Use Meng values as prior if local estimate differs by > 30%.

Step 3 — Compute cross-person EDA/HR correlation matrix (S4, S5)
  script: tools/mumt/compute_cross_person_sync.py  [to be created]
  Input:  data/mumt/pretrain.pkl + dataset.pkl
  Output: data/mumt/cross_sync.pkl
  Content: per (session, task):
    r_EDA[4×4] pairwise EDA cross-correlation
    r_HR[4×4]  pairwise HR cross-correlation

Step 4 — Generate GP labels (S1, S2)
  script: tools/mumt/generate_gp_labels.py  [to be created]
  Input:  data/mumt/pretrain.pkl, dataset.pkl, ou_params.json, cross_sync.pkl
  Output: data/mumt/augmented_labels_s1s2.pkl
  Content: per unlabeled window w:
    mu_V, sigma_V, w_V, soft_label_V[3]
    mu_A, sigma_A, w_A, soft_label_A[3]
    mu_D, sigma_D, w_D, soft_label_D[3]
    sources: list of contributing observations (for debugging)

Step 5 — Generate OOF pseudo-labels (S3)
  script: tools/mumt/generate_oof_labels.py  [to be created]
  Input:  data/mumt/dataset.pkl, pretrain.pkl
  Method: Run LOGO-CV (k=10 folds), inference on held-out group after Phase 3
  Output: data/mumt/augmented_labels_s3.pkl
  Content: per window w (labeled + unlabeled):
    logits_V[3], logits_A[3], logits_D[3]
    confidence_V, confidence_A, confidence_D
    T_platt_V, T_platt_A, T_platt_D  (per fold)
  Note: This step is slow (~10× training cost). Run once and cache.

Step 6 — Merge and filter augmented label pool
  script: tools/mumt/build_augmented_dataset.py  [to be created]
  Input:  all above + filter thresholds
  Output: data/mumt/augmented_pool.pkl
  Filter: retain windows where at least one dimension meets w_min threshold:
    w_V(w) ≥ 0.10 OR w_A(w) ≥ 0.05 OR w_D(w) ≥ 0.07
  Expected yield: 1,500–3,500 windows from 8,978 pool
```

### 6.2 Leakage Verification Checklist

Before any training run using augmented labels, verify:

```
□ augmented_labels_s1s2.pkl was built using ONLY Likert scores (dataset.pkl column: valence/arousal/dominance)
□ S4 EDA observations for person l used only EDA columns from persons k ≠ l
□ S5 HR observations for person l used only PPG columns from persons k ≠ l
□ S3 OOF predictor for group g was trained on groups {1..10} \ {g}  (check fold log)
□ augmented_pool.pkl does NOT contain any window that also appears in test split
□ VAD binning thresholds (ou_params.json or compute_vad_thresholds()) were fit on TRAINING split only
□ Cross-sync correlation matrices (cross_sync.pkl) were computed on ALL sessions (no leakage — they calibrate σ, not labels)
```

---

## 7. Dataset Split Rules

Augmented labels must respect the train/val/test split defined in `train_affectai.py`:

```
Test set:  FIXED — original self-report labels only. NO augmented labels ever
           enter the test set, regardless of source or confidence.

Val set:   FIXED — original self-report labels only. No augmented labels.
           Val set is used to select T_platt and verify augmentation benefit.

Train set: Original labels + augmented labels from S1, S2, S3, S4, S5.
           Subject-level split is preserved: augmented labels belong to the
           same split as the subject's original self-report labels.
```

The session (group) is the unit of splitting (LOGO-CV). All windows belonging to
session g (labeled + augmented) move to the test fold together. This prevents
session-level leakage where S2/S4/S5 cross-person labels from training sessions
contaminate a test session via shared group context.

---

## 8. Expected Impact by Dimension

Based on CTSEM timescales and source-specific σ values:

| Dimension | Primary beneficial source | Expected yield | Notes |
|-----------|--------------------------|----------------|-------|
| **Valence** | S1 (GP interpolation) + S2 (cross-person reports) | High — valence changes slowly (12.3s half-life), can interpolate over 60s gaps | Cross-person self-reports are the most reliable valence evidence |
| **Arousal** | S4 (cross-person SCR) + S3 (OOF predictions) | Moderate — arousal changes fast (1.2s half-life), only SCR events provide sub-window precision | GP temporal interpolation is unreliable beyond 5s for arousal |
| **Dominance** | S1 + S2 (intermediate half-life ~6s) + S3 | Moderate — dominance benefits from cross-person group context; difficult to infer from physiology | S3 OOF predictions are the primary dominance source since physio guidance is weakest |

---

## 9. Key Implementation Notes

### 9.1 EDA Preprocessing Warning

`EDA_SEQ_COLS` in `dataset_affectai.py` includes `value_6` (HR proxy) and `value_10`
(temperature) alongside EDA channels. When applying S4 (cross-person EDA → arousal),
use ONLY the EDA-specific columns (`value_3`, `EDA_Phasic`, `EDA_Tonic`) — not HR proxy.
The HR proxy in M3 overlaps informationally with M4 (PPG-derived HR), so mixing the two
channels would complicate leakage tracking.

Recommended: when implementing S4, access raw EDA from `pickle_generation_affectai.py`
output (`value_3` only) rather than from the model-input `eda_seq` tensor.

### 9.2 OU Parameter Sensitivity

The arousal half-life (1.2s) is very short relative to the 30-second window size.
This means S1 arousal interpolation will almost always yield very small weights
(w_A ≈ exp(-0.578 * 15) ≈ 0.0002 at window centre). The practical implication:

> **S1 arousal labels are useful only for windows whose trailing edge (t_w) is within
> ~5 seconds of a self-report.** All other arousal augmentation must come from S3 or S4.

This is not a bug — it correctly encodes the physics of arousal dynamics. Forcing
arousal interpolation at longer time scales would produce misleading pseudo-labels.

### 9.3 Cross-Person Synchrony Calibration Fallback

If fewer than 3 within-task paired self-reports exist for a session (needed to compute
r_EDA, r_HR empirically), use task-type priors:

```python
SYNC_PRIORS = {
    "T0": {"r_EDA": 0.20, "r_HR": 0.15},   # baseline, low synchrony expected
    "T1": {"r_EDA": 0.35, "r_HR": 0.25},   # collaborative task 1
    "T2": {"r_EDA": 0.40, "r_HR": 0.30},   # collaborative task 2
    "T3": {"r_EDA": 0.30, "r_HR": 0.22},
    "T4": {"r_EDA": 0.30, "r_HR": 0.22},
}
```

Priors are conservative (lower than literature means) to avoid over-propagating
cross-person labels when synchrony is uncertain.

### 9.4 GRL Lambda Schedule (existing bug fix, required for augmented training)

The GRL lambda in `train_affectai.py` is hardcoded at `grl_lambda=1.0` from epoch 1.
Before adding augmented labels, fix this to the standard Ganin et al. (2016) schedule:

```python
def grl_lambda_schedule(current_epoch, total_epochs):
    p = current_epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
```

This is a prerequisite for stable augmented training — a fixed λ=1.0 during early
Phase 2 epochs destabilises the disentanglement objective, which is amplified when
the soft augmented labels (lower SNR than self-reports) are also present.

---

## 10. References

- Barsade, S. G. (2002). The ripple effect: Emotional contagion and its influence on group behavior. *Administrative Science Quarterly*, 47(4), 644–675.
- Cacioppo, J. T., & Berntson, G. G. (1994). Relationship between attitudes and evaluative space. *Psychological Bulletin*, 115(3), 401.
- Feldman, R. (2007). Parent–infant synchrony and the construction of shared timing. *Social Cognitive and Affective Neuroscience*, 2(2), 123–131.
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(59), 1–35.
- Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1993). Emotional contagion. *Current Directions in Psychological Science*, 2(3), 96–100.
- Kuppens, P., et al. (2010). Emotional inertia and psychological maladjustment. *Psychological Science*, 21(7), 984–991.
- Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshops*.
- Levenson, R. W., & Gottman, J. M. (1983). Marital interaction: Physiological linkage and affective exchange. *Journal of Personality and Social Psychology*, 45(3), 587.
- Meng, X., et al. (2026). Moderating roles of the Big Five in valence–arousal dynamics: A TFace-Bi-GRU-SE and CTSEM study. *MDPI Information*, 17(4), 334.
- Mou, L., et al. (2019). Group-level emotion recognition using a unimodal face-based prediction. *IEEE TAFFC*.
- Parthasarathy, S., & Busso, C. (2017). Semi-supervised speech emotion recognition with ladder networks. *IEEE TASLP*, 28, 196–205.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Thayer, J. F., & Lane, R. D. (2000). A model of neurovisceral integration in emotion regulation. *Journal of Affective Disorders*, 61(3), 201–216.
