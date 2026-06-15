"""
Quantization and optimization for MuMTAffect models.

Converts checkpoints to FP16 and INT8 formats for edge deployment,
benchmarking latency gains vs. accuracy degradation.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset_affectai import GroupAffectDataset, make_user2idx, make_session2idx
from model_affectai import MuMTAffectGroupAffect
from train_affectai import fit_scalers, make_summary_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 32


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(state_dict: dict) -> float:
    """Calculate model size in MB (FP32)."""
    param_size = 0
    for tensor in state_dict.values():
        param_size += tensor.nelement() * 4  # 4 bytes per FP32
    return param_size / (1024 * 1024)


def quantize_model_fp16(model: nn.Module) -> nn.Module:
    """Convert model to FP16 (half precision)."""
    for name, param in model.named_parameters():
        param.data = param.data.half()
    return model.half()


def quantize_model_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized


def benchmark_inference(model: nn.Module, test_loader: DataLoader, device: str, n_runs: int = 10) -> dict:
    """Benchmark inference latency."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 2:
                break
            inputs, _, _, _ = batch
            _ = model(inputs.to(device))
    
    # Timing
    torch.cuda.synchronize(device) if device == 'cuda' else None
    start = time.time()
    n_samples = 0
    
    with torch.no_grad():
        for run in range(n_runs):
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 10:  # Limit batches per run
                    break
                inputs, _, _, _ = batch
                batch_size = inputs.size(0)
                _ = model(inputs.to(device))
                n_samples += batch_size
    
    torch.cuda.synchronize(device) if device == 'cuda' else None
    elapsed = time.time() - start
    
    return {
        "latency_per_sample_ms": (elapsed / n_samples) * 1000,
        "throughput_samples_per_sec": n_samples / elapsed,
        "total_time_sec": elapsed,
        "n_samples": n_samples,
    }


def evaluate_quantized_model(model: nn.Module, test_loader: DataLoader, device: str) -> dict:
    """Evaluate quantized model performance."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, y_vad, _, _ = batch
            outputs = model(inputs.to(device))
            
            # Predictions
            preds = outputs['vad_pred'].argmax(dim=-1)  # (B, 3) -> (B,)
            
            all_preds.append(preds.cpu())
            all_targets.append(y_vad.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return {"f1_score": float(f1)}


def run_quantization(args):
    """Main quantization workflow."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Device: {device}")
    
    # Load data
    log.info(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        df = pickle.load(f)
    
    # Prepare for evaluation
    log.info(f"Preparing data...")
    user2idx = make_user2idx(df)
    session2idx = make_session2idx(df)
    scalers = fit_scalers(df)  # Fit on full data for consistency
    summary_dim, summary_key_order = make_summary_dim(df)
    
    # Hold out 10% for test
    test_fraction = 0.1
    test_indices = np.random.RandomState(42).choice(
        len(df), size=int(len(df) * test_fraction), replace=False
    )
    test_df = df.iloc[test_indices]
    
    test_ds = GroupAffectDataset(
        test_df, user2idx, scalers,
        augment=False, device=device,
        summary_key_order=summary_key_order,
        session2idx=session2idx
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load baseline model
    log.info(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try multiple configurations to find the right one
    # Detect personality_as_input from checkpoint by examining mlp weight shape
    mlp_input_dim = state_dict.get('trial_mlp.net.0.weight', None)
    if mlp_input_dim is not None:
        mlp_input_dim = mlp_input_dim.shape[1]
        log.info(f"Checkpoint mlp input dim: {mlp_input_dim}, summary_dim: {summary_dim}")
        personality_as_input = mlp_input_dim > summary_dim
    else:
        personality_as_input = False
    
    baseline = MuMTAffectGroupAffect(
        n_subjects=len(user2idx),
        n_personality=5,
        use_gru=False,
        personality_as_input=personality_as_input,
        per_dim_queries=True,
        per_dim_projections=True
    )
    log.info(f"Loading with personality_as_input={personality_as_input}")
    baseline.load_state_dict(state_dict, strict=False)
    baseline.to(device)
    
    results = []
    
    # ==== FP32 Baseline ====
    log.info("\n=== FP32 Baseline ===")
    fp32_model = baseline
    fp32_params = count_parameters(fp32_model)
    fp32_size = get_model_size(state_dict)
    fp32_perf = evaluate_quantized_model(fp32_model, test_loader, device)
    fp32_bench = benchmark_inference(fp32_model, test_loader, device, n_runs=5)
    
    log.info(f"Parameters: {fp32_params:,}")
    log.info(f"Model size: {fp32_size:.2f} MB")
    log.info(f"F1 score: {fp32_perf['f1_score']:.4f}")
    log.info(f"Latency: {fp32_bench['latency_per_sample_ms']:.3f} ms/sample")
    log.info(f"Throughput: {fp32_bench['throughput_samples_per_sec']:.1f} samples/sec")
    
    results.append({
        "format": "FP32 (baseline)",
        "parameters": fp32_params,
        "model_size_mb": fp32_size,
        "f1_score": fp32_perf['f1_score'],
        "latency_ms": fp32_bench['latency_per_sample_ms'],
        "throughput_samples_sec": fp32_bench['throughput_samples_per_sec'],
        "speedup_vs_fp32": 1.0,
        "accuracy_delta": 0.0,
    })
    
    # ==== FP16 Quantization ====
    log.info("\n=== FP16 Quantization ===")
    try:
        fp16_model = torch.load(args.checkpoint, map_location=device)
        if isinstance(fp16_model, dict):
            if 'model_state_dict' in fp16_model:
                fp16_model = fp16_model
            else:
                fp16_model = {'model_state_dict': fp16_model}
        
        fp16_baseline = MuMTAffectGroupAffect(
            n_subjects=len(user2idx),
            n_personality=5,
            use_gru=False,
            personality_as_input=personality_as_input,
            per_dim_queries=True,
            per_dim_projections=True
        )
        fp16_state = fp16_model if isinstance(fp16_model, dict) else fp16_model
        if isinstance(fp16_state, dict) and 'model_state_dict' in fp16_state:
            fp16_baseline.load_state_dict(fp16_state['model_state_dict'])
        else:
            fp16_baseline.load_state_dict(fp16_state)
        
        fp16_baseline = quantize_model_fp16(fp16_baseline)
        fp16_baseline.to(device)
        
        fp16_perf = evaluate_quantized_model(fp16_baseline, test_loader, device)
        fp16_bench = benchmark_inference(fp16_baseline, test_loader, device, n_runs=5)
        
        fp16_size = get_model_size(state_dict) / 2  # Approximate
        
        log.info(f"Model size: {fp16_size:.2f} MB (50% reduction)")
        log.info(f"F1 score: {fp16_perf['f1_score']:.4f}")
        log.info(f"Latency: {fp16_bench['latency_per_sample_ms']:.3f} ms/sample")
        log.info(f"Speedup: {fp32_bench['latency_per_sample_ms'] / fp16_bench['latency_per_sample_ms']:.2f}x")
        
        speedup = fp32_bench['latency_per_sample_ms'] / fp16_bench['latency_per_sample_ms']
        accuracy_delta = fp16_perf['f1_score'] - fp32_perf['f1_score']
        
        results.append({
            "format": "FP16 (half precision)",
            "parameters": fp32_params,
            "model_size_mb": fp16_size,
            "f1_score": fp16_perf['f1_score'],
            "latency_ms": fp16_bench['latency_per_sample_ms'],
            "throughput_samples_sec": fp16_bench['throughput_samples_per_sec'],
            "speedup_vs_fp32": speedup,
            "accuracy_delta": accuracy_delta,
        })
        
        # Save FP16 checkpoint
        fp16_ckpt_path = Path(args.output_dir) / "model_fp16.pt"
        torch.save(fp16_baseline.state_dict(), fp16_ckpt_path)
        log.info(f"Saved FP16 model to {fp16_ckpt_path}")
        
    except Exception as e:
        log.warning(f"FP16 quantization failed: {e}")
    
    # ==== INT8 Dynamic Quantization ====
    log.info("\n=== INT8 Dynamic Quantization ===")
    try:
        int8_baseline = MuMTAffectGroupAffect(
            n_subjects=len(user2idx),
            n_personality=5,
            use_gru=False,
            personality_as_input=personality_as_input,
            per_dim_queries=True,
            per_dim_projections=True
        )
        int8_baseline.load_state_dict(state_dict)
        int8_baseline.to(device)
        int8_baseline = int8_baseline.cpu()  # INT8 quantization on CPU
        
        int8_model = quantize_model_int8_dynamic(int8_baseline)
        int8_model.to(device)
        
        int8_perf = evaluate_quantized_model(int8_model, test_loader, device)
        int8_bench = benchmark_inference(int8_model, test_loader, device, n_runs=5)
        
        int8_size = get_model_size(state_dict) * 0.25  # Approximate
        
        log.info(f"Model size: {int8_size:.2f} MB (75% reduction)")
        log.info(f"F1 score: {int8_perf['f1_score']:.4f}")
        log.info(f"Latency: {int8_bench['latency_per_sample_ms']:.3f} ms/sample")
        log.info(f"Speedup: {fp32_bench['latency_per_sample_ms'] / int8_bench['latency_per_sample_ms']:.2f}x")
        
        speedup = fp32_bench['latency_per_sample_ms'] / int8_bench['latency_per_sample_ms']
        accuracy_delta = int8_perf['f1_score'] - fp32_perf['f1_score']
        
        results.append({
            "format": "INT8 (dynamic)",
            "parameters": fp32_params,
            "model_size_mb": int8_size,
            "f1_score": int8_perf['f1_score'],
            "latency_ms": int8_bench['latency_per_sample_ms'],
            "throughput_samples_sec": int8_bench['throughput_samples_per_sec'],
            "speedup_vs_fp32": speedup,
            "accuracy_delta": accuracy_delta,
        })
        
        # Save INT8 checkpoint
        int8_ckpt_path = Path(args.output_dir) / "model_int8.pt"
        torch.save(int8_model, int8_ckpt_path)
        log.info(f"Saved INT8 model to {int8_ckpt_path}")
        
    except Exception as e:
        log.warning(f"INT8 quantization failed: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / "quantization_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nResults saved to {results_path}")
    
    # Print summary
    log.info("\n=== Quantization Summary ===")
    print(results_df.to_string(index=False))
    
    # Save as JSON for reference
    results_json = results_df.to_dict(orient='records')
    json_path = Path(args.output_dir) / "quantization_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    log.info(f"JSON results saved to {json_path}")
    
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Quantize and benchmark MuMTAffect models for edge deployment"
    )
    parser.add_argument(
        '--checkpoint',
        default='data/mumt/production_model/model_transformer_baseline_stratified.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--dataset',
        default='data/mumt/dataset.pkl',
        help='Path to dataset pickle'
    )
    parser.add_argument(
        '--output-dir',
        default='data/mumt/quantization_results',
        help='Output directory for quantized models and results'
    )
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_quantization(args)
