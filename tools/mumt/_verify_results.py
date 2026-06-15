"""Quick sanity-check script for paper results."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tools/mumt")

from train_simple import compute_tertile_thresholds, task_split
from dataset_affectai import build_summary_key_order
from baselines import extract_summary_matrix, get_hard_labels, run_svm

df = pd.read_pickle("data/mumt/dataset_15s.pkl")
train_df, val_df, test_df = task_split(df, test_task="T3")
thresh = compute_tertile_thresholds(train_df)

print("=== Tertile thresholds + class distribution ===")
for dim, (t1, t2) in thresh.items():
    tr_vals = train_df[dim].dropna().values
    te_vals = test_df[dim].dropna().values
    def counts(vals, t1, t2):
        low  = int((vals <= t1).sum())
        mid  = int(((vals > t1) & (vals <= t2)).sum())
        high = int((vals > t2).sum())
        return low, mid, high
    tl, tm, th = counts(tr_vals, t1, t2)
    el, em, eh = counts(te_vals, t1, t2)
    print(f"  {dim}: t=({t1:.2f},{t2:.2f})")
    print(f"    TRAIN  Low={tl}({tl/len(tr_vals)*100:.0f}%) Mid={tm}({tm/len(tr_vals)*100:.0f}%) High={th}({th/len(tr_vals)*100:.0f}%) n={len(tr_vals)}")
    print(f"    T3TEST Low={el}({el/len(te_vals)*100:.0f}%) Mid={em}({em/len(te_vals)*100:.0f}%) High={eh}({eh/len(te_vals)*100:.0f}%) n={len(te_vals)}")

print()
print("=== Feature modality key counts ===")
row = train_df.iloc[0]
for col in ["gaze_features","pupil_features","eda_features","ppg_features","imu_features","audio_features","speech_features"]:
    fd = row.get(col, {})
    n = len(fd) if isinstance(fd, dict) else "not-dict"
    print(f"  {col}: {n} keys")

key_order = build_summary_key_order(df)
train_X = np.nan_to_num(extract_summary_matrix(train_df, key_order))
test_X  = np.nan_to_num(extract_summary_matrix(test_df,  key_order))
nz = int((train_X.std(0) > 1e-6).sum())
print(f"\n  Summary matrix: train={train_X.shape}, test={test_X.shape}")
print(f"  Active (non-constant) feature cols: {nz} / {train_X.shape[1]}")

print()
print("=== SVM baseline ===")
train_lbl = get_hard_labels(train_df, thresh)
test_lbl  = get_hard_labels(test_df,  thresh)
r = run_svm(train_X, train_lbl, test_X, test_lbl)
print(f"  V={r['valence']:.3f}  A={r['arousal']:.3f}  D={r['dominance']:.3f}  Mean={r['mean']:.3f}")

print()
print("=== Val set T2 label distribution ===")
for dim, (t1, t2) in thresh.items():
    ve_vals = val_df[dim].dropna().values
    vl = int((ve_vals <= t1).sum())
    vm = int(((ve_vals > t1) & (ve_vals <= t2)).sum())
    vh = int((ve_vals > t2).sum())
    print(f"  VAL T2 {dim}: Low={vl} Mid={vm} High={vh} n={len(ve_vals)}")
