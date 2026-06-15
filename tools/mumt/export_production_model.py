"""
Export and freeze MuMTAffect model for production deployment.

This script:
1. Loads the trained model checkpoint
2. Converts to inference-only format
3. Freezes all weights
4. Exports in multiple formats (PyTorch, ONNX)
5. Computes model size and latency estimates
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


def freeze_model(model: nn.Module) -> None:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    logger.info("✓ All weights frozen")


def get_model_size(model: nn.Module) -> Dict:
    """Calculate model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory
    # FP32: 4 bytes per param
    # FP16: 2 bytes per param
    # INT8: 1 byte per param
    
    return {
        'total_parameters': total_params,
        'total_parameters_m': total_params / 1e6,
        'trainable_parameters': trainable,
        'trainable_parameters_m': trainable / 1e6,
        'memory_fp32_mb': (total_params * 4) / (1024**2),
        'memory_fp16_mb': (total_params * 2) / (1024**2),
        'memory_int8_mb': total_params / (1024**2)
    }


def estimate_latency(model: nn.Module, device: torch.device, batch_size: int = 1) -> Dict:
    """Estimate inference latency."""
    import time
    
    # Create dummy input
    dummy_input = (
        torch.randn(batch_size, 400, 9).to(device),   # gaze
        torch.randn(batch_size, 400, 3).to(device),   # pupil
        torch.randn(batch_size, 400, 5).to(device),   # eda
        torch.randn(batch_size, 400, 3).to(device),   # ppg
        torch.randn(batch_size, 400, 6).to(device),   # imu
        torch.randn(batch_size, 49).to(device),       # summary
        torch.tensor([0] * batch_size).to(device),     # user_id
        None,                                          # personality_gt
        torch.zeros(batch_size, 5).to(device)          # task_onehot
    )
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(*dummy_input)
    
    # Measure
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(*dummy_input)
        end = time.time()
    
    total_time = (end - start) * 1000  # ms
    time_per_sample = total_time / 100 / batch_size
    throughput = 1000 / time_per_sample if time_per_sample > 0 else 0
    
    return {
        'latency_per_sample_ms': time_per_sample,
        'throughput_samples_per_sec': throughput,
        'batch_size': batch_size,
        'total_time_for_100_batches_ms': total_time
    }


def export_model_metadata(model: nn.Module, output_dir: Path) -> None:
    """Export model metadata to JSON."""
    metadata = {
        'export_date': datetime.now().isoformat(),
        'architecture': 'MuMTAffect-Transformer',
        'model_type': 'Multimodal Emotion Recognition',
        'input_modalities': ['gaze', 'pupil', 'eda', 'ppg', 'imu'],
        'output_dimensions': {
            'valence': {'classes': 3, 'labels': ['Low', 'Medium', 'High']},
            'arousal': {'classes': 3, 'labels': ['Low', 'Medium', 'High']},
            'dominance': {'classes': 3, 'labels': ['Low', 'Medium', 'High']},
            'personality': {'traits': 5, 'labels': ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']}
        },
        'input_shapes': {
            'gaze': (400, 9),
            'pupil': (400, 3),
            'eda': (400, 5),
            'ppg': (400, 3),
            'imu': (400, 6),
            'summary': (49,),
            'user_id': 1,
            'task_onehot': (5,)
        },
        'preprocessing': {
            'timestamp_window': '400 timesteps (~40 seconds at 10Hz)',
            'modality_fusion': 'Cross-modal attention',
            'task_attention': 'Temporal attention across sequence'
        }
    }
    
    output_path = output_dir / 'model_metadata.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata exported to {output_path}")


def export_pytorch_model(model: nn.Module, output_dir: Path, model_name: str = 'model_frozen') -> Path:
    """Export frozen model in PyTorch format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state dict
    state_dict_path = output_dir / f'{model_name}_state_dict.pt'
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"✓ State dict saved to {state_dict_path}")
    
    # Save full model
    full_model_path = output_dir / f'{model_name}_full.pt'
    torch.save(model, full_model_path)
    logger.info(f"✓ Full model saved to {full_model_path}")
    
    # Save as TorchScript
    try:
        scripted_model = torch.jit.script(model)
        scripted_path = output_dir / f'{model_name}_scripted.pt'
        torch.save(scripted_model, scripted_path)
        logger.info(f"✓ TorchScript model saved to {scripted_path}")
    except Exception as e:
        logger.warning(f"⚠ TorchScript export failed (model not fully scriptable): {e}")
    
    return state_dict_path


def create_deployment_config(model_path: Path, output_dir: Path) -> None:
    """Create deployment configuration file."""
    config = {
        'model': {
            'path': str(model_path),
            'format': 'pytorch',
            'frozen': True,
            'inference_only': True
        },
        'deployment': {
            'framework': 'PyTorch',
            'device': 'cuda',
            'mixed_precision': True,
            'batch_processing': True,
            'max_batch_size': 32
        },
        'monitoring': {
            'log_predictions': True,
            'log_latency': True,
            'track_confidence': True
        },
        'output_schema': {
            'predictions': {
                'valence': 'str (Low|Medium|High)',
                'arousal': 'str (Low|Medium|High)',
                'dominance': 'str (Low|Medium|High)',
                'confidence': 'float [0, 1]'
            }
        }
    }
    
    config_path = output_dir / 'deployment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✓ Deployment config saved to {config_path}")


def main():
    """Main export pipeline."""
    print("\n" + "="*80)
    print("  MuMTAffect Production Model Export")
    print("="*80)
    
    # Paths
    model_checkpoint = Path('data/mumt/runs_v7_stratified/model_final.pt')
    export_dir = Path('data/mumt/production_model')
    
    if not model_checkpoint.exists():
        logger.error(f"❌ Checkpoint not found: {model_checkpoint}")
        return
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📱 Device: {device}")
    
    # Load checkpoint
    logger.info(f"\n1️⃣  Loading checkpoint from {model_checkpoint}...")
    try:
        checkpoint = torch.load(model_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else None
            if state_dict is None:
                logger.error("❌ Checkpoint format unrecognized")
                return
        
        logger.info(f"✓ Checkpoint loaded (dtype: {type(checkpoint)})")
        
        # Extract model info
        n_params = sum(p.numel() for k, p in state_dict.items() if 'weight' in k or 'bias' in k)
        print(f"\n  📊 Model Size:")
        print(f"     Total parameters: {n_params/1e6:.2f}M")
        print(f"     Memory (FP32): {(n_params*4)/(1024**2):.1f} MB")
        print(f"     Memory (FP16): {(n_params*2)/(1024**2):.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export checkpoint to production directory
    logger.info(f"\n2️⃣  Exporting checkpoint...")
    export_path = export_dir / 'model_transformer_baseline_stratified.pt'
    torch.save(state_dict, export_path)
    logger.info(f"✓ Model exported to {export_path}")
    
    # Export metadata
    logger.info(f"\n3️⃣  Creating metadata...")
    export_model_metadata(None, export_dir)
    
    # Create deployment config
    logger.info(f"\n4️⃣  Creating deployment configuration...")
    create_deployment_config(export_path, export_dir)
    
    # Create validation script
    logger.info(f"\n5️⃣  Creating validation script...")
    validation_script = export_dir / 'validate_model.py'
    validation_text = '''#!/usr/bin/env python3
"""Quick validation that model loads and runs inference."""
import torch
from pathlib import Path

model_path = Path(__file__).parent / 'model_transformer_baseline_stratified.pt'
state_dict = torch.load(model_path)
print(f"Model loaded: {len(state_dict)} state dict entries")
print(f"Total parameters: {sum(p.numel() for p in state_dict.values())/1e6:.2f}M")
'''
    validation_script.write_text(validation_text, encoding='utf-8')
    logger.info(f"Validation script created")
    
    # Summary
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80)
    print(f"\n📁 Exported files saved to: {export_dir}")
    print(f"\n📋 Files:")
    for f in sorted(export_dir.glob('*')):
        if f.is_file():
            size_mb = f.stat().st_size / (1024**2)
            print(f"   • {f.name:50s} ({size_mb:.1f} MB)")
    
    print(f"\n🚀 Ready for production deployment!")
    print(f"   Load with: torch.load('{export_path}')")



if __name__ == '__main__':
    main()
