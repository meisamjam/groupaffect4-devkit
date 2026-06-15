import sys, torch
sys.path.insert(0, 'tools/mumt')
from model_simple import build_simple_model
import pandas as pd
from dataset_affectai import build_summary_key_order

df = pd.read_pickle('data/mumt/dataset.pkl')
sko = build_summary_key_order(df)
sdim = len(sko)

for arch in ['mlp', 'pool', 'conv']:
    m = build_simple_model(arch, summary_dim=sdim)
    n = sum(p.numel() for p in m.parameters())
    B, T = 4, 400
    out = m(
        torch.randn(B,T,9), torch.randn(B,T,3), torch.randn(B,T,5),
        torch.randn(B,T,3), torch.randn(B,T,6), torch.randn(B,sdim)
    )
    vshape = tuple(out["valence_logits"].shape)
    print(f"{arch:<12}  params={n:>8,}  valence_logits={vshape}")
print("OK")
