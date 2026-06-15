"""Quick verification of corrected pool label distribution."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np; import pandas as pd; from scipy.stats import norm
sys.path.insert(0, "tools/mumt")
from train_simple import compute_tertile_thresholds, task_split

df = pd.read_pickle("data/mumt/dataset_15s.pkl")
train_df, _, _ = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)

for pool_name, pool_path in [
    ("ORIGINAL (meng mu)",  "data/mumt/augmented_pool.pkl"),
    ("CORRECTED (train mu)", "data/mumt/augmented_pool_corrected.pkl"),
]:
    pool = pd.read_pickle(pool_path)
    # Filter to T0+T1 (what gets used in canonical T3 experiment)
    pool_train = pool[pool["task"].isin(["T0","T1"])]
    print(f"\n=== {pool_name} [T0+T1 windows: {len(pool_train)}] ===")
    for dim in ["valence","arousal","dominance"]:
        t1, t2 = thresh[dim]
        mu = pool_train[f"{dim}_mu"].values
        sig = np.maximum(pool_train[f"{dim}_sigma"].values, 0.01)
        w   = pool_train[f"{dim}_weight"].values
        # Recompute soft labels under balanced thresholds from (mu, sigma)
        p_low  = norm.cdf(t1, mu, sig)
        p_high = 1 - norm.cdf(t2, mu, sig)
        p_mid  = np.maximum(1 - p_low - p_high, 0)
        argmax = np.argmax(np.stack([p_low, p_mid, p_high], axis=1), axis=1)
        low_pct = (argmax==0).mean()*100; mid_pct = (argmax==1).mean()*100; high_pct = (argmax==2).mean()*100
        print(f"  {dim}: mu={mu.mean():.2f}  Low={low_pct:.1f}%  Mid={mid_pct:.1f}%  High={high_pct:.1f}%  "
              f"w_mean={w.mean():.3f}  w_median={np.median(w):.3f}")
    print(f"  Training hard: V=29%/35%/36%  A=33%/19%/48%  D=15%/39%/47%")

# Check LOO accuracy on corrected pool
from label_augmentation import ou_gp_posterior, OUParams, MENG_OU_PARAMS
import json
with open("data/mumt/correct_ou_params.json") as f:
    corrected = json.load(f)

print("\n=== LOO GP accuracy: corrected vs original ===")
for label, params_d in [("ORIGINAL (Meng)", MENG_OU_PARAMS), ("CORRECTED (train mu)", corrected)]:
    print(f"\n  {label}:")
    for dim in ["valence","arousal","dominance"]:
        t1, t2 = thresh[dim]
        p = OUParams(**params_d[dim])
        correct=0; total=0
        for (ses,task,seat), grp in train_df.groupby(["session_id","task","seat"]):
            vals = grp[[dim,"vad_timestamp_lsl"]].dropna().values
            if len(vals)<2: continue
            for i in range(len(vals)):
                t_q=float(vals[i,1]); y_q=float(vals[i,0])
                other=np.arange(len(vals))!=i
                t_obs=vals[other,1].astype(float); y_obs=vals[other,0].astype(float)
                s_obs=np.full(len(t_obs),0.8)
                mu_post,_=ou_gp_posterior(t_q,t_obs,y_obs,s_obs,p)
                true_cls=0 if y_q<=t1 else (2 if y_q>t2 else 1)
                pred_cls=0 if mu_post<=t1 else (2 if mu_post>t2 else 1)
                correct+=int(true_cls==pred_cls); total+=1
        print(f"    {dim}: {correct}/{total} = {correct/total*100:.1f}%  (chance=33.3%)")
