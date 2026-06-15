"""Quick schema check on pretrain_dataset.pkl."""
import pandas as pd
pool = pd.read_pickle("data/mumt/pretrain_dataset.pkl")
print(f"Shape: {pool.shape}")
print(f"Columns: {list(pool.columns[:30])}")
print(f"Has window_t_center: {'window_t_center' in pool.columns}")
print(f"Has session_id: {'session_id' in pool.columns}")
print(f"Has task: {'task' in pool.columns}")
print(f"Tasks: {dict(pool['task'].value_counts().sort_index())}")
print(f"n_sessions: {pool['session_id'].nunique()}")
if 'window_t_center' in pool.columns:
    valid_ts = pool['window_t_center'].notna().sum()
    print(f"Windows with valid timestamps: {valid_ts}/{len(pool)} ({valid_ts/len(pool)*100:.1f}%)")
