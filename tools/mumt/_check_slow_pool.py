"""Compare all three pools on label distributions and LOO accuracy."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np; import pandas as pd; from scipy.stats import norm
sys.path.insert(0, "tools/mumt")
from train_simple import compute_tertile_thresholds, task_split
from label_augmentation import ou_gp_posterior, OUParams
import json

df = pd.read_pickle("data/mumt/dataset_15s.pkl")
train_df, _, _ = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)

print("=== POOL LABEL DISTRIBUTION COMPARISON (T0+T1 windows only) ===")
print("Training hard labels: V=29/35/36%  A=33/19/48%  D=15/39/47%  (Low/Mid/High)")
print()
for name, path in [
    ("ORIGINAL   (Meng   mu=5.3/4.9/5.1, theta=Meng)", "data/mumt/augmented_pool.pkl"),
    ("CORRECTED  (Train  mu=7.1/6.0/6.2, theta=Meng)", "data/mumt/augmented_pool_corrected.pkl"),
    ("SLOW       (Train  mu=7.1/6.0/6.2, theta=0.01)", "data/mumt/augmented_pool_slow.pkl"),
]:
    pool = pd.read_pickle(path)
    pt = pool[pool["task"].isin(["T0","T1"])]
    print(f"{name} [{len(pt)} T0+T1 windows]")
    for dim in ["valence","arousal","dominance"]:
        t1, t2 = thresh[dim]
        mu = pt[f"{dim}_mu"].values
        sig = np.maximum(pt[f"{dim}_sigma"].values, 0.01)
        w   = pt[f"{dim}_weight"].values
        p_low  = norm.cdf(t1, mu, sig)
        p_high = 1 - norm.cdf(t2, mu, sig)
        p_mid  = np.maximum(1 - p_low - p_high, 0)
        argmax = np.argmax(np.stack([p_low, p_mid, p_high], axis=1), axis=1)
        lp, mp, hp = (argmax==0).mean()*100, (argmax==1).mean()*100, (argmax==2).mean()*100
        # Label entropy (how peaked vs uniform)
        soft = np.stack([p_low,p_mid,p_high],axis=1)
        soft = soft / soft.sum(axis=1,keepdims=True)
        ent = -(soft * np.log(soft.clip(1e-10))).sum(axis=1).mean()
        print(f"  {dim}: mu_mean={mu.mean():.2f}  Low={lp:.1f}%  Mid={mp:.1f}%  High={hp:.1f}%  "
              f"entropy={ent:.3f}  w_mean={w.mean():.3f}")
    print()

print("=== LOO GP ACCURACY (training data only) ===")
print("Chance = 33.3%. Values above chance = GP has useful signal.")
print()
for name, params_file in [
    ("MENG default", None),
    ("CORRECTED mu (Meng theta)", "data/mumt/correct_ou_params.json"),
    ("SLOW theta=0.01", "data/mumt/correct_ou_params_slow.json"),
]:
    if params_file:
        with open(params_file) as f:
            pd_dict = json.load(f)
    else:
        from label_augmentation import MENG_OU_PARAMS as pd_dict

    print(f"  {name}:")
    for dim in ["valence","arousal","dominance"]:
        t1, t2 = thresh[dim]
        p = OUParams(**pd_dict[dim])
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
        flag = "ABOVE" if correct/total > 1/3 else "below"
        print(f"    {dim}: {correct}/{total} = {correct/total*100:.1f}%  ({flag} chance)")
    print()
