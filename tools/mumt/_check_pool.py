"""Verify threshold mismatch between stored soft labels and balanced thresholds."""
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, "tools/mumt")
from train_simple import compute_tertile_thresholds, task_split

pool = pd.read_pickle("data/mumt/augmented_pool.pkl")
df   = pd.read_pickle("data/mumt/dataset_15s.pkl")
train_df, _, _ = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)

print("=== Threshold comparison ===")
print(f"Old (hardcoded): (3.0, 6.0) for all dims")
for dim, (t1, t2) in thresh.items():
    print(f"  Balanced {dim}: t=({t1:.2f}, {t2:.2f})")

print()
print("=== Stored vs recomputed soft labels (first 5 windows, valence) ===")
t1_bal, t2_bal = thresh["valence"]
for i in range(5):
    row   = pool.iloc[i]
    mu    = row["valence_mu"]
    sigma = row["valence_sigma"]
    stored = row["valence_soft"]
    # Recompute with balanced thresholds
    p_low  = norm.cdf(t1_bal, mu, sigma)
    p_high = 1 - norm.cdf(t2_bal, mu, sigma)
    p_mid  = 1 - p_low - p_high
    recomp = np.array([p_low, p_mid, p_high], dtype=np.float32)
    recomp /= recomp.sum()
    argmax_stored = ["Low","Mid","High"][np.argmax(stored)]
    argmax_recomp = ["Low","Mid","High"][np.argmax(recomp)]
    same = "OK" if argmax_stored == argmax_recomp else "MISMATCH"
    print(f"  mu={mu:.2f} sig={sigma:.2f}  stored={np.round(stored,2)} ({argmax_stored})  "
          f"balanced={np.round(recomp,2)} ({argmax_recomp})  {same}")

print()
print("=== Global mismatch rate ===")
mismatches = {"valence": 0, "arousal": 0, "dominance": 0}
total = len(pool)
for dim, (t1, t2) in thresh.items():
    for _, row in pool.iterrows():
        mu = row[f"{dim}_mu"]
        sig = row[f"{dim}_sigma"]
        stored = row[f"{dim}_soft"]
        p_low  = norm.cdf(t1, mu, sig)
        p_high = 1 - norm.cdf(t2, mu, sig)
        p_mid  = max(1 - p_low - p_high, 0)
        recomp = np.array([p_low, p_mid, p_high])
        if np.argmax(stored) != np.argmax(recomp):
            mismatches[dim] += 1
    print(f"  {dim}: {mismatches[dim]}/{total} ({100*mismatches[dim]/total:.1f}%) "
          f"argmax mismatches between stored and balanced-threshold soft labels")
