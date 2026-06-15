import numpy as np

# Simulate valence training data from T0+T1 (103 windows)
# Distribution: {3:1, 5:10, 6:19, 7:36, 8:23, 9:14}
vals_raw = [3]*1 + [5]*10 + [6]*19 + [7]*36 + [8]*23 + [9]*14
vals = np.sort(np.array(vals_raw, dtype=float))
n = len(vals)
print("n =", n)

def midpoint_threshold(vals, idx):
    boundary_val = vals[idx]
    below = vals[vals < boundary_val]
    if len(below) > 0:
        return (below[-1] + boundary_val) / 2.0
    return boundary_val - 0.5

idx1 = n // 3
idx2 = (2 * n) // 3
t1 = midpoint_threshold(vals, idx1)
t2 = midpoint_threshold(vals, idx2)

low  = int(np.sum(vals < t1))
mid  = int(np.sum((vals >= t1) & (vals < t2)))
high = int(np.sum(vals >= t2))
print("idx1=%d (val=%.0f), idx2=%d (val=%.0f)" % (idx1, vals[idx1], idx2, vals[idx2]))
print("t1=%.2f, t2=%.2f" % (t1, t2))
print("Low=%d (%.1f%%), Mid=%d (%.1f%%), High=%d (%.1f%%)" % (
    low, 100*low/n, mid, 100*mid/n, high, 100*high/n))
