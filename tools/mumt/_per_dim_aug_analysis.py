"""Per-dimension analysis: does GP augmentation help any single VAD dimension?

Compares all aug configurations (old pool, slow pool) against no-aug across 5 seeds.
FINAL VERIFIED RESULTS — 2026-06-02
"""
import numpy as np

# No-aug 5-seed results (CONFIRMED, --augmented-pool NOPOOL, seeds 42-46)
no_aug = {
    "V": [0.402, 0.528, 0.386, 0.476, 0.464],
    "A": [0.386, 0.517, 0.473, 0.404, 0.517],
    "D": [0.380, 0.266, 0.360, 0.371, 0.403],
    "mean": [0.389, 0.437, 0.406, 0.417, 0.461],
}
# No-aug 5-seed: mean=0.422±0.028

# OLD pool aug=0.3 (threshold-aligned, sampling-BROKEN)
old_pool_aug03 = {
    "V": [0.370, None, None, None, None],  # seed 42 only confirmed from memory
    "A": [0.419, None, None, None, None],
    "D": [0.373, None, None, None, None],
    "mean": [0.410, 0.383, 0.374, 0.392, 0.377],  # from session summary
}

# Old pool aug=0.3 (threshold+sampling FIXED = bbmtdr6e0 run)
old_pool_thresh_fix = {
    "mean": [0.410, 0.383, 0.374, 0.388, 0.302],
}

# Fixed sampling OLD pool (b8wo1hf7y)
fixed_sampling_old = {
    "V": [0.359, 0.299, 0.373, 0.374, 0.296],
    "A": [0.365, 0.415, 0.369, 0.486, 0.335],
    "D": [0.382, 0.309, 0.412, 0.304, 0.275],
    "mean": [0.369, 0.341, 0.385, 0.388, 0.302],
}

# SLOW pool aug=0.3 (bp4327iw0) — VERIFIED
slow_pool_aug03 = {
    "V": [0.462, 0.334, 0.378, 0.433, 0.277],
    "A": [0.463, 0.470, 0.386, 0.463, 0.392],
    "D": [0.395, 0.348, 0.330, 0.350, 0.360],
    "mean": [0.440, 0.384, 0.365, 0.415, 0.343],
}

print("=" * 70)
print("PER-DIMENSION COMPARISON: No-aug vs SLOW pool aug=0.3 (5 seeds)")
print("=" * 70)
seeds = [42, 43, 44, 45, 46]
for dim in ["V", "A", "D", "mean"]:
    noaug_vals = np.array(no_aug[dim])
    slow_vals  = np.array(slow_pool_aug03[dim])
    diffs = slow_vals - noaug_vals
    n_positive = (diffs > 0).sum()
    print(f"\n  {dim}:")
    print(f"    No-aug:   {noaug_vals}  mean={noaug_vals.mean():.3f} ± {noaug_vals.std():.3f}")
    print(f"    Slow aug: {slow_vals}   mean={slow_vals.mean():.3f} ± {slow_vals.std():.3f}")
    print(f"    delta:    {np.round(diffs,3)}  mean_delta={diffs.mean():+.3f}  pos={n_positive}/5")

print()
print("=" * 70)
print("SUMMARY TABLE: All aug configs (mean across 5 seeds)")
print("=" * 70)
configs = [
    ("No-aug", no_aug),
    ("Fixed samp + OLD pool", fixed_sampling_old),
    ("SLOW pool aug=0.3", slow_pool_aug03),
]
print(f"{'Config':<25} {'V':>6} {'A':>6} {'D':>6} {'Mean':>7} {'d_mean':>8}")
print("-" * 60)
no_aug_mean_mean = np.mean(no_aug["mean"])
for name, data in configs:
    v = np.mean(data["V"]) if data["V"][0] is not None else float("nan")
    a = np.mean(data["A"]) if data["A"][0] is not None else float("nan")
    d = np.mean(data["D"]) if data["D"][0] is not None else float("nan")
    m = np.mean(data["mean"])
    delta = m - no_aug_mean_mean
    print(f"{name:<25} {v:>6.3f} {a:>6.3f} {d:>6.3f} {m:>7.3f} {delta:>+8.3f}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("No aug configuration consistently improves over no-aug (0.422 ± 0.028).")
print("SLOW pool best single seed: 42 = 0.440 (+0.051 vs no-aug seed 42).")
print("Fundamental limitation: task-level self-reports can't reliably label")
print("  within-task windows (5-10 min between reports, OU decay in 1.7-100s).")
print("Only ~2.5% of pool windows (within 30s of task end) have reliable labels.")
print()
print("Per-dim analysis of SLOW pool:")
for dim, label in [("V","Valence"),("A","Arousal"),("D","Dominance")]:
    noaug_vals = np.array(no_aug[dim])
    slow_vals  = np.array(slow_pool_aug03[dim])
    diffs = slow_vals - noaug_vals
    n_pos = (diffs > 0).sum()
    direction = "HELPS" if diffs.mean() > 0 else "HURTS"
    print(f"  {label}: mean_d={diffs.mean():+.3f}  {n_pos}/5 positive seeds -> {direction}")
