"""Compute correct OU parameters from GroupAffect-4 training data.

This script:
1. Runs estimate_ou_params() from label_augmentation.py on the training set
2. Shows what the corrected GP posterior distribution looks like
3. Shows the expected soft label distribution under balanced thresholds
4. Estimates LOO accuracy improvement

Run this BEFORE regenerating the augmented pool.
"""
import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")

sys.path.insert(0, "tools/mumt")
from train_simple import compute_tertile_thresholds, task_split
from label_augmentation import (
    estimate_ou_params, MENG_OU_PARAMS, DIMS,
    ou_gp_posterior, OUParams, soft_label_from_posterior,
)

df = pd.read_pickle("data/mumt/dataset_15s.pkl")
train_df, val_df, test_df = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)

print("=" * 65)
print("Training data statistics")
print("=" * 65)
for dim in DIMS:
    vals = train_df[dim].dropna().values
    print(f"  {dim}: n={len(vals)}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"min={vals.min():.1f}  max={vals.max():.1f}")

print()
print("=" * 65)
print("OU parameter estimation from TRAINING data")
print("=" * 65)
ou_est = estimate_ou_params(train_df)
for dim in DIMS:
    p = ou_est[dim]
    p_meng = OUParams(**MENG_OU_PARAMS[dim])
    print(f"  {dim}:")
    print(f"    MENG default: theta={p_meng.theta:.4f}  sigma2={p_meng.sigma2:.3f}  mu={p_meng.mu:.3f}")
    print(f"    ESTIMATED:    theta={p.theta:.4f}  sigma2={p.sigma2:.3f}  mu={p.mu:.3f}")

print()
print("=" * 65)
print("Expected soft label distribution under balanced thresholds")
print("=" * 65)
print("Using PRIOR (no observations) for pool window far from any report:")
for dim in DIMS:
    t1, t2 = thresh[dim]
    for label, p in [("MENG prior", OUParams(**MENG_OU_PARAMS[dim])),
                      ("ESTIMATED prior", ou_est[dim])]:
        mu, sigma = p.mu, float(np.sqrt(p.sigma2))
        pl = norm.cdf(t1, mu, sigma)
        ph = 1 - norm.cdf(t2, mu, sigma)
        pm = max(1 - pl - ph, 0)
        total = pl + pm + ph
        print(f"  {dim} {label}: mu={mu:.2f}  "
              f"Low={pl/total*100:.1f}%  Mid={pm/total*100:.1f}%  High={ph/total*100:.1f}%  "
              f"(thresholds: {t1}, {t2})")

print()
print("=" * 65)
print("LOO GP accuracy comparison: MENG vs ESTIMATED params")
print("=" * 65)
for label, params_dict in [("MENG", {d: OUParams(**MENG_OU_PARAMS[d]) for d in DIMS}),
                             ("ESTIMATED", ou_est)]:
    print(f"\n  {label} params:")
    for dim in DIMS:
        t1, t2 = thresh[dim]
        p = params_dict[dim]
        correct = 0; total = 0
        for (ses, task, seat), grp in train_df.groupby(["session_id","task","seat"]):
            vals = grp[[dim, "vad_timestamp_lsl"]].dropna().values
            if len(vals) < 2: continue
            for i in range(len(vals)):
                t_q = float(vals[i, 1]); y_q = float(vals[i, 0])
                other = np.arange(len(vals)) != i
                t_obs = vals[other, 1].astype(float)
                y_obs = vals[other, 0].astype(float)
                s_obs = np.full(len(t_obs), 0.8)
                mu_post, _ = ou_gp_posterior(t_q, t_obs, y_obs, s_obs, p)
                true_cls = 0 if y_q <= t1 else (2 if y_q > t2 else 1)
                pred_cls = 0 if mu_post <= t1 else (2 if mu_post > t2 else 1)
                correct += int(true_cls == pred_cls); total += 1
        print(f"    {dim}: {correct}/{total} = {correct/total*100:.1f}% "
              f"({'ABOVE' if correct/total > 1/3 else 'BELOW'} chance=33.3%)")

print()
print("=" * 65)
print("Task-level mean VAD from TRAINING data (for reference)")
print("=" * 65)
for task in ["T0","T1","T2","T3","T4"]:
    task_df = train_df[train_df["task"] == task]
    if len(task_df) == 0: continue
    row = {d: f"{task_df[d].mean():.2f}" for d in DIMS}
    print(f"  {task} (n={len(task_df)}): V={row['valence']}  A={row['arousal']}  D={row['dominance']}")

print()
print("Correct TASK_PRIOR_ADJUSTMENT values (= task_mean - overall_train_mean):")
overall = {d: train_df[d].mean() for d in DIMS}
for task in ["T0","T1","T2","T3","T4"]:
    task_df = train_df[train_df["task"] == task]
    if len(task_df) == 0: continue
    adj = {d: task_df[d].mean() - overall[d] for d in DIMS}
    print(f"  {task}: V={adj['valence']:+.2f}  A={adj['arousal']:+.2f}  D={adj['dominance']:+.2f}")

print()
print("Done. Use ESTIMATED params to regenerate the augmented pool.")
