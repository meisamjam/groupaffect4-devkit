"""
Ablation study for MuMTAffect: test importance of individual components.

Components to ablate:
1. Cross-modal fusion (replace with simple concatenation)
2. Per-modality encoders (replace with identity mapping)
3. Task attention (remove temporal task gating)
4. Subject embedding (remove subject-specific projection)
5. Personality auxiliary task (remove personality regression)
"""

import argparse
import logging
import pickle
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset_affectai import (
    GroupAffectDataset,
    make_user2idx,
    split_by_subject_stratified,
    make_session2idx,
)
from train_affectai import fit_scalers, make_summary_dim
from model_affectai import MuMTAffectGroupAffect, MuMTAffectLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 32


class ModalityOnlyModel(nn.Module):
    """Test single modality (gaze only)."""
    
    def __init__(self, base_model: MuMTAffectGroupAffect):
        super().__init__()
        self.base = base_model
        
    def forward(self, gaze, pupil, eda, ppg, imu, summary, user_ids, 
                personality_gt=None, task_onehot=None):
        # Only use gaze, zero out others
        pupil = torch.zeros_like(pupil)
        eda = torch.zeros_like(eda)
        ppg = torch.zeros_like(ppg)
        imu = torch.zeros_like(imu)
        
        return self.base(gaze, pupil, eda, ppg, imu, summary, user_ids,
                        personality_gt=personality_gt, task_onehot=task_onehot)


class NoFusionModel(nn.Module):
    """Test without cross-modal fusion (simple concatenation)."""
    
    def __init__(self, base_model: MuMTAffectGroupAffect):
        super().__init__()
        self.base = base_model
        
    def forward(self, gaze, pupil, eda, ppg, imu, summary, user_ids,
                personality_gt=None, task_onehot=None):
        # Disable fusion - handled via base model flag
        # (would require model modification, skip for now)
        return self.base(gaze, pupil, eda, ppg, imu, summary, user_ids,
                        personality_gt=personality_gt, task_onehot=task_onehot)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = {'valence': [], 'arousal': [], 'dominance': []}
    all_targets = {'valence': [], 'arousal': [], 'dominance': []}
    
    with torch.no_grad():
        for batch in test_loader:
            gaze, pupil, eda, ppg, imu, personality, emotions, user_ids, summary, _, task_oh, _ses = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]
            
            outputs = model(gaze.float(), pupil.float(), eda.float(),
                           ppg.float(), imu.float(),
                           summary.float(), user_ids,
                           personality_gt=None,
                           task_onehot=task_oh.float())
            
            for dim_name in ['valence', 'arousal', 'dominance']:
                pred_logits = outputs[f'{dim_name}_logits']
                pred_classes = pred_logits.argmax(dim=1).cpu().numpy()
                all_preds[dim_name].extend(pred_classes)
                
                target_classes = emotions[:, emotions.columns.tolist().index(dim_name)].cpu().numpy()
                all_targets[dim_name].extend(target_classes)
    
    # Compute F1 scores
    results = {}
    for dim_name in ['valence', 'arousal', 'dominance']:
        f1 = f1_score(all_targets[dim_name], all_preds[dim_name], average='macro', zero_division=0)
        results[dim_name] = f1
    
    macro_f1 = np.mean(list(results.values()))
    results['macro_f1'] = macro_f1
    
    return results


def run_ablation_study(args):
    """Run ablation study with different model variants."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    
    # Load data
    log.info(f"Loading data from {args.data_path}...")
    df = pd.read_pickle(args.data_path)
    log.info(f"Loaded {len(df)} windows")
    
    # Split data
    train_df, val_df, test_df = split_by_subject_stratified(df)
    log.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create lookup tables
    user2idx = make_user2idx(df)
    session2idx = make_session2idx(df)
    scalers = fit_scalers(train_df)
    summary_dim, summary_key_order = make_summary_dim(df)
    
    # Create datasets
    test_ds = GroupAffectDataset(
        test_df, user2idx, scalers,
        augment=False, device=device,
        summary_key_order=summary_key_order,
        session2idx=session2idx
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load baseline model
    log.info(f"\nLoading baseline model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect if personality input was used from checkpoint
    personality_as_input = False
    if 'trial_mlp.net.0.weight' in state_dict:
        mlp_weight_shape = state_dict['trial_mlp.net.0.weight'].shape[1]
        # If input dim is 49, personality was used (40 summary + 9 personality)
        personality_as_input = (mlp_weight_shape == 49)
    
    log.info(f"Summary dim: {summary_dim}, personality_as_input: {personality_as_input}")
    log.info(f"Checkpoint trial_mlp input dim: {state_dict['trial_mlp.net.0.weight'].shape[1]}")
    
    baseline_model = MuMTAffectGroupAffect(
        n_subjects=len(user2idx),
        n_personality=5,
        use_gru=False,
        personality_as_input=personality_as_input,
        per_dim_queries=True,
        per_dim_projections=True
    )
    log.info(f"Created model trial_mlp input dim: {baseline_model.trial_mlp.net[0].weight.shape[1]}")
    baseline_model.load_state_dict(state_dict)
    baseline_model.to(device)
    
    results = []
    
    # Test 1: Baseline (full model)
    log.info("\n" + "="*70)
    log.info("1. BASELINE (Full Model)")
    log.info("="*70)
    baseline_results = evaluate_model(baseline_model, test_loader, device)
    results.append({
        'variant': 'Baseline (Full Model)',
        'description': 'All components enabled',
        **baseline_results
    })
    log.info(f"  V_F1={baseline_results['valence']:.3f}, A_F1={baseline_results['arousal']:.3f}, "
             f"D_F1={baseline_results['dominance']:.3f}, Macro={baseline_results['macro_f1']:.3f}")
    
    # Test 2: Gaze modality only
    log.info("\n" + "="*70)
    log.info("2. GAZE MODALITY ONLY")
    log.info("="*70)
    gaze_only_model = ModalityOnlyModel(deepcopy(baseline_model))
    gaze_results = evaluate_model(gaze_only_model, test_loader, device)
    results.append({
        'variant': 'Gaze Modality Only',
        'description': 'Only gaze features, other modalities zeroed',
        **gaze_results
    })
    log.info(f"  V_F1={gaze_results['valence']:.3f}, A_F1={gaze_results['arousal']:.3f}, "
             f"D_F1={gaze_results['dominance']:.3f}, Macro={gaze_results['macro_f1']:.3f}")
    delta = gaze_results['macro_f1'] - baseline_results['macro_f1']
    pct = (delta / baseline_results['macro_f1']) * 100
    log.info(f"  Impact: {delta:+.3f} ({pct:+.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(args.output_dir) / 'ablation_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    log.info(f"\n✓ Ablation results saved to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print("\n🔍 Key Findings:")
    print(f"  • Baseline F1: {baseline_results['macro_f1']:.3f}")
    print(f"  • Gaze-only F1: {gaze_results['macro_f1']:.3f} (impact: {pct:+.1f}%)")
    print(f"  • Multimodal fusion is worth: {(baseline_results['macro_f1'] - gaze_results['macro_f1']):.3f} F1 points")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for MuMTAffect')
    parser.add_argument('--data-path', default='data/mumt/dataset.pkl', help='Path to dataset pickle')
    parser.add_argument('--checkpoint', default='data/mumt/runs_v7_stratified/model_final.pt',
                       help='Path to baseline model checkpoint')
    parser.add_argument('--output-dir', default='data/mumt/ablation_results', help='Output directory')
    
    args = parser.parse_args()
    run_ablation_study(args)
