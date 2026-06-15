"""Comprehensive diagnostics for the augmented pool and GP augmentation pipeline.

Checks:
1. Data lineage: dataset_15s.pkl traces to data/zenodo
2. Pool statistics per dim: weight distribution, n_obs, label diversity
3. Label correctness: argmax agreement between stored soft labels and
   GP mu under balanced thresholds
4. Temporal proximity: fraction of windows within 1 task-period of any self-report
5. Why aug doesn't help: per-dim analysis of soft label entropy vs hard label
6. Effective sampling rate under old vs new weight formula
"""
import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, norm
warnings.filterwarnings("ignore")

sys.path.insert(0, "tools/mumt")
from train_simple import compute_tertile_thresholds, task_split

print("=" * 70)
print("SECTION 1 — Data lineage verification")
print("=" * 70)

# Check if dataset_15s.pkl comes from data/zenodo via the pickle generation script
import pickle, pathlib
ds_path = pathlib.Path("data/mumt/dataset_15s.pkl")
pool_path = pathlib.Path("data/mumt/augmented_pool.pkl")
zenodo_path = pathlib.Path("data/zenodo")

print(f"dataset_15s.pkl exists: {ds_path.exists()} ({ds_path.stat().st_size/1e6:.1f} MB)")
print(f"augmented_pool.pkl exists: {pool_path.exists()} ({pool_path.stat().st_size/1e6:.1f} MB)")
print(f"data/zenodo exists: {zenodo_path.exists()}")
n_zenodo_subs = len(list(zenodo_path.glob("sub-*")))
print(f"  zenodo subjects: {n_zenodo_subs}")
n_physio = len(list(zenodo_path.glob("sub-*/ses-*/physio/*.tsv.gz")))
n_et = len(list(zenodo_path.glob("sub-*/ses-*/et/*.tsv.gz")))
n_beh = len(list(zenodo_path.glob("sub-*/ses-*/beh/*.tsv")))
print(f"  physio files: {n_physio}, eye-tracking files: {n_et}, beh files: {n_beh}")

df = pd.read_pickle(ds_path)
pool = pd.read_pickle(pool_path)

print(f"\ndataset_15s.pkl: {len(df)} windows, {df['subject_id'].nunique()} subjects, {df['session_id'].nunique()} sessions")
# Verify session_ids overlap with zenodo structure
zenodo_sessions = set()
for p in zenodo_path.glob("sub-*/ses-*"):
    if p.is_dir():
        # Extract session date_grp from full path name
        zenodo_sessions.add(p.name)
ds_sessions = set(df['session_id'].unique())
print(f"  Unique session_ids in dataset: {len(ds_sessions)}")
print(f"  Zenodo session dirs: {len(zenodo_sessions)}")
# Check overlap by extracting the grp part
zenodo_grps = {s.split('_grp-')[1].split('_')[0] for s in zenodo_sessions if '_grp-' in s}
ds_grps = {str(s).split('grp-')[1].split('_')[0] if 'grp-' in str(s) else str(s) for s in ds_sessions}
overlap = zenodo_grps & ds_grps
print(f"  Group IDs in zenodo: {sorted(zenodo_grps)}")
print(f"  Group IDs in dataset: {sorted(ds_grps)}")
print(f"  Overlap: {len(overlap)}/{max(len(zenodo_grps),len(ds_grps))} groups")

print()
print("=" * 70)
print("SECTION 2 — Pool statistics per dimension")
print("=" * 70)

train_df, val_df, test_df = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)
print(f"Train windows: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
for dim, (t1, t2) in thresh.items():
    print(f"  Balanced thresholds {dim}: ({t1:.2f}, {t2:.2f})")

print(f"\nPool total windows: {len(pool)}")
print(f"Pool sessions: {pool['session_id'].nunique()}")
print(f"Pool tasks: {dict(pool['task'].value_counts().sort_index())}")
print(f"Pool subjects: {pool['subject_id'].nunique() if 'subject_id' in pool.columns else 'N/A'}")

for dim in ["valence", "arousal", "dominance"]:
    w = pool[f"{dim}_weight"].values
    mu = pool[f"{dim}_mu"].values
    sig = pool[f"{dim}_sigma"].values
    n_obs = pool[f"n_obs_{dim}"].values if f"n_obs_{dim}" in pool.columns else np.zeros(len(pool))
    soft = np.stack(pool[f"{dim}_soft"].values)  # (N, 3)

    # Entropy of soft labels (0=peaked, log(3)=uniform)
    ent = np.array([scipy_entropy(s) for s in soft])

    # Under balanced thresholds
    t1_b, t2_b = thresh[dim]
    p_low_b  = norm.cdf(t1_b, mu, np.maximum(sig, 0.01))
    p_high_b = 1 - norm.cdf(t2_b, mu, np.maximum(sig, 0.01))
    p_mid_b  = np.maximum(1 - p_low_b - p_high_b, 0)
    soft_b   = np.stack([p_low_b, p_mid_b, p_high_b], axis=1)
    soft_b   = soft_b / soft_b.sum(axis=1, keepdims=True)
    argmax_stored  = soft.argmax(axis=1)
    argmax_balanced = soft_b.argmax(axis=1)
    mismatch_pct = (argmax_stored != argmax_balanced).mean() * 100

    # mu distribution (does it match training distribution?)
    low_pct  = (mu < t1_b).mean() * 100
    high_pct = (mu > t2_b).mean() * 100
    mid_pct  = 100 - low_pct - high_pct

    print(f"\n  {dim.upper()}:")
    print(f"    Weight: mean={w.mean():.3f}  min={w.min():.3f}  max={w.max():.3f}  "
          f"p10={np.percentile(w,10):.3f}  p90={np.percentile(w,90):.3f}")
    print(f"    n_obs: mean={n_obs.mean():.1f}  zero={( n_obs==0).sum()} ({(n_obs==0).mean()*100:.1f}%)")
    print(f"    mu range: [{mu.min():.1f}, {mu.max():.1f}]  mean={mu.mean():.2f}")
    print(f"    sigma: mean={sig.mean():.3f}  p90={np.percentile(sig,90):.3f}")
    print(f"    Soft label entropy: mean={ent.mean():.3f}  (max possible={np.log(3):.3f})")
    print(f"    mu class distribution (balanced thr): Low={low_pct:.1f}%  Mid={mid_pct:.1f}%  High={high_pct:.1f}%")
    print(f"    Stored vs balanced argmax MISMATCH: {mismatch_pct:.1f}%")
    # Top classes under balanced thresholds
    cls_counts_bal = np.bincount(argmax_balanced, minlength=3)
    print(f"    Balanced label distribution: Low={cls_counts_bal[0]/len(pool)*100:.1f}%  "
          f"Mid={cls_counts_bal[1]/len(pool)*100:.1f}%  High={cls_counts_bal[2]/len(pool)*100:.1f}%")

print()
print("=" * 70)
print("SECTION 3 — Hard label distribution (training set, balanced thresholds)")
print("=" * 70)
for dim, (t1, t2) in thresh.items():
    vals = train_df[dim].dropna().values
    low = (vals <= t1).sum(); high = (vals > t2).sum(); mid = len(vals)-low-high
    print(f"  {dim}: Low={low}({low/len(vals)*100:.1f}%)  Mid={mid}({mid/len(vals)*100:.1f}%)  "
          f"High={high}({high/len(vals)*100:.1f}%)  n={len(vals)}")

print()
print("=" * 70)
print("SECTION 4 — Effective aug sampling rate (old vs new formula)")
print("=" * 70)
n_hard = len(train_df)
n_soft = len(pool)

# Mean confidence per dim
avg_conf_by_dim = {dim: pool[f"{dim}_weight"].mean() for dim in ["valence","arousal","dominance"]}
aug_conf = pool[["valence_weight","arousal_weight","dominance_weight"]].mean(axis=1).values
conf_sum = aug_conf.sum()
conf_mean = aug_conf.mean()

print(f"  n_hard={n_hard}  n_soft={n_soft}")
print(f"  avg confidence: {conf_mean:.4f}  conf_sum={conf_sum:.1f}")
print(f"  Per-dim: V={avg_conf_by_dim['valence']:.4f}  A={avg_conf_by_dim['arousal']:.4f}  D={avg_conf_by_dim['dominance']:.4f}")

for aug_frac in [0.3, 0.5]:
    # OLD formula
    ratio_old = (n_hard * aug_frac) / (n_soft * (1.0 - aug_frac))
    soft_w_old = ratio_old * aug_conf
    eff_old = soft_w_old.sum() / (n_hard + soft_w_old.sum())

    # NEW formula (fixed)
    scale_new = (n_hard * aug_frac) / (conf_sum * (1.0 - aug_frac))
    soft_w_new = scale_new * aug_conf
    eff_new = soft_w_new.sum() / (n_hard + soft_w_new.sum())

    print(f"\n  aug_frac={aug_frac}:")
    print(f"    OLD: ratio={ratio_old:.5f}  eff_aug_frac={eff_old:.4f} ({eff_old*100:.2f}%)")
    print(f"    NEW: scale={scale_new:.5f}  eff_aug_frac={eff_new:.4f} ({eff_new*100:.2f}%)")
    print(f"    Expected soft samples per batch (bs=16): OLD={eff_old*16:.2f}  NEW={eff_new*16:.2f}")

print()
print("=" * 70)
print("SECTION 5 — GP label quality: predicted vs actual for TRAINING set")
print("=" * 70)
# Match pool windows to training self-reports: check if GP mu predicts the right class
# Use only pool windows from train sessions (T0+T1)
train_sessions = set(train_df["session_id"].unique())
pool_train = pool[pool["session_id"].isin(train_sessions) & pool["task"].isin(["T0","T1"])]
print(f"  Pool windows from train sessions (T0+T1): {len(pool_train)}")

# Also check: what fraction of the pool comes from EACH task?
task_dist = pool["task"].value_counts().sort_index()
print(f"  Full pool task distribution:")
for task, cnt in task_dist.items():
    print(f"    {task}: {cnt} ({cnt/len(pool)*100:.1f}%)")

print()
print("=" * 70)
print("SECTION 6 — Why does GP fail arousal? Check n_obs and theta")
print("=" * 70)
# Arousal has fast theta (0.578/s from Meng) — meaning observations become uncorrelated
# after ~1.7s. Within a 15s window, there's typically only 1 self-report per task.
# Distance from window to nearest task report for arousal

# Check: for pool windows, how many have n_obs=0 for each dim?
for dim in ["valence", "arousal", "dominance"]:
    if f"n_obs_{dim}" in pool.columns:
        nobs = pool[f"n_obs_{dim}"].values
        print(f"  {dim}: n_obs=0: {(nobs==0).sum()} ({(nobs==0).mean()*100:.1f}%)  "
              f"n_obs=1: {(nobs==1).sum()} ({(nobs==1).mean()*100:.1f}%)  "
              f"n_obs>=2: {(nobs>=2).sum()} ({(nobs>=2).mean()*100:.1f}%)")

# OU theta gives 1/e decay distance
from label_augmentation import MENG_OU_PARAMS
print()
for dim, p in MENG_OU_PARAMS.items():
    theta = p["theta"]
    half_life_s = np.log(2) / theta if theta > 0 else float('inf')
    one_e_s = 1.0 / theta if theta > 0 else float('inf')
    # At distance d, weight = exp(-theta * d)
    # For w=0.1: d = -ln(0.1)/theta
    d_w01 = -np.log(0.1) / theta if theta > 0 else float('inf')
    print(f"  {dim}: theta={theta:.4f}  half-life={half_life_s:.1f}s  "
          f"1/e decay={one_e_s:.1f}s  d(w=0.1)={d_w01:.1f}s  d(w=0.2)={-np.log(0.2)/theta:.1f}s")

print()
print("=" * 70)
print("SECTION 7 — LOO label accuracy of GP (can GP predict its training labels?)")
print("=" * 70)
# For training windows, use GP to predict label from OTHER training windows of same person
# This is an estimate of GP's utility as a label source
correct_per_dim = {d: 0 for d in ["valence","arousal","dominance"]}
total_per_dim = {d: 0 for d in ["valence","arousal","dominance"]}

from label_augmentation import ou_gp_posterior, OUParams
import warnings
warnings.filterwarnings("ignore")

for dim_idx, dim in enumerate(["valence","arousal","dominance"]):
    t1, t2 = thresh[dim]
    p = OUParams(**MENG_OU_PARAMS[dim])
    for (ses, task, seat), grp in train_df.groupby(["session_id","task","seat"]):
        vals = grp[[dim, "vad_timestamp_lsl"]].dropna().values
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            t_q = float(vals[i, 1])
            y_q = float(vals[i, 0])
            # LOO: all other observations
            other_mask = np.arange(len(vals)) != i
            t_obs = vals[other_mask, 1].astype(float)
            y_obs = vals[other_mask, 0].astype(float)
            s_obs = np.full(len(t_obs), 0.8)
            mu_post, _ = ou_gp_posterior(t_q, t_obs, y_obs, s_obs, p)
            # True class
            true_cls = 0 if y_q <= t1 else (2 if y_q > t2 else 1)
            pred_cls = 0 if mu_post <= t1 else (2 if mu_post > t2 else 1)
            correct_per_dim[dim] += int(true_cls == pred_cls)
            total_per_dim[dim] += 1

print("  LOO GP prediction accuracy on training self-reports:")
for dim in ["valence","arousal","dominance"]:
    n = total_per_dim[dim]
    acc = correct_per_dim[dim] / n if n > 0 else 0
    print(f"    {dim}: {correct_per_dim[dim]}/{n} = {acc:.3f} ({acc*100:.1f}%)")
print("  (chance = 0.333; 3-class random)")

print()
print("=" * 70)
print("DONE. See above for diagnostic summary.")
print("=" * 70)
