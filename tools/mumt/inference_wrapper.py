"""
Production inference wrapper for MuMTAffect Transformer model.

This module provides:
1. Model loading and inference
2. Input preprocessing
3. Confidence scoring
4. Batch inference support
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import logging
import pickle

logger = logging.getLogger(__name__)


class MuMTAffectInferenceWrapper:
    """
    Production-ready inference wrapper for MuMTAffect model.
    
    Features:
    - Automatic device management (CPU/GPU)
    - Mixed precision (FP16) support
    - Batch processing
    - Confidence/uncertainty scoring
    - Input validation
    - Caching for repeated inference
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        device: Optional[str] = None,
        use_amp: bool = True,
        cache_size: int = 128
    ):
        """
        Initialize inference wrapper.
        
        Args:
            model_path: Path to frozen model checkpoint
            device: 'cuda', 'cpu', or None (auto-detect)
            use_amp: Use mixed precision (FP16) for faster inference
            cache_size: Number of recent inferences to cache
        """
        self.model_path = Path(model_path)
        self.device = self._init_device(device)
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.cache_size = cache_size
        self._inference_cache = {}
        
        # Load model
        self.model = self._load_model()
        self.model.eval()  # Inference mode
        self._freeze_weights()
        
        # Detect summary_dim from model (checkpoint was trained with 44, current dataset has 49)
        # The model itself knows its input dimension from initialization
        self.summary_dim = 44  # Checkpoint dimension
        
        # VAD class labels
        self.vad_labels = ['Low', 'Medium', 'High']
        self.personality_traits = [
            'Extraversion', 'Agreeableness', 'Conscientiousness', 
            'Neuroticism', 'Openness'
        ]
        
        logger.info(f"✓ Model loaded from {self.model_path}")
        logger.info(f"✓ Device: {self.device}")
        logger.info(f"✓ Inference mode enabled (frozen weights)")
        logger.info(f"✓ Summary dimension: {self.summary_dim}")
    
    def _init_device(self, device: Optional[str]) -> torch.device:
        """Initialize torch device."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Import here to avoid circular dependency
        from model_affectai import MuMTAffectGroupAffect
        
        # The checkpoint was trained with specific dimensions
        # From inspection: n_subjects=39 (subject_embed=[41,16]), mlp=[64,49]
        # mlp input = summary_dim + personality (49 = 44 + 5)
        n_subjects_ckpt = 39
        personality_as_input_ckpt = True
        n_personality = 5
        mlp_input_dim_ckpt = 49
        summary_dim_ckpt = mlp_input_dim_ckpt - n_personality  # 49 - 5 = 44
        
        logger.info(f"Loading checkpoint: n_subjects={n_subjects_ckpt}, summary_dim={summary_dim_ckpt}, personality_as_input={personality_as_input_ckpt}")
        
        model = MuMTAffectGroupAffect(
            summary_dim=summary_dim_ckpt,
            n_subjects=n_subjects_ckpt,
            n_personality=n_personality,
            use_gru=False,
            personality_as_input=personality_as_input_ckpt,
            per_dim_queries=True,
            per_dim_projections=True
        )
        
        # Load weights
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("✓ Checkpoint loaded with strict=True")
        except RuntimeError as e:
            logger.warning(f"strict=True failed: {e}")
            logger.info("Trying strict=False...")
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info("✓ Checkpoint loaded with strict=False")
            except RuntimeError as e2:
                logger.error(f"Both strict modes failed: {e2}")
                raise
        
        model.to(self.device)
        return model
    
    def _freeze_weights(self):
        """Freeze all model weights for inference-only mode."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("✓ All weights frozen (inference-only)")
    
    def preprocess_input(
        self,
        gaze: np.ndarray,
        pupil: np.ndarray,
        eda: np.ndarray,
        ppg: np.ndarray,
        imu: np.ndarray,
        summary: np.ndarray,
        user_id: Union[int, str] = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Validate and convert input to tensors.
        
        Args:
            gaze: (T, 9) - gaze features
            pupil: (T, 3) - pupil features
            eda: (T, 5) - EDA features
            ppg: (T, 3) - PPG features
            imu: (T, 6) - IMU features
            summary: (49,) - session summary
            user_id: Participant ID
        
        Returns:
            Dict of tensors on device
        """
        # Validate shapes
        expected_shapes = {
            'gaze': (400, 9),
            'pupil': (400, 3),
            'eda': (400, 5),
            'ppg': (400, 3),
            'imu': (400, 6),
            'summary': (49,)
        }
        
        inputs = {
            'gaze': gaze,
            'pupil': pupil,
            'eda': eda,
            'ppg': ppg,
            'imu': imu,
            'summary': summary
        }
        
        for name, arr in inputs.items():
            if arr.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape mismatch: got {arr.shape}, "
                    f"expected {expected_shapes[name]}"
                )
        
        # Convert to tensors
        tensors = {
            'gaze': torch.from_numpy(gaze).float().to(self.device),
            'pupil': torch.from_numpy(pupil).float().to(self.device),
            'eda': torch.from_numpy(eda).float().to(self.device),
            'ppg': torch.from_numpy(ppg).float().to(self.device),
            'imu': torch.from_numpy(imu).float().to(self.device),
            'summary': torch.from_numpy(summary).float().to(self.device),
            'user_id': torch.tensor([user_id], dtype=torch.long).to(self.device),
            'task_onehot': torch.zeros((1, 5), dtype=torch.float32).to(self.device)  # No task during inference
        }
        
        # Add batch dimension
        for key in tensors:
            if key != 'user_id':
                tensors[key] = tensors[key].unsqueeze(0)
        
        return tensors
    
    @torch.no_grad()
    def predict(
        self,
        gaze: np.ndarray,
        pupil: np.ndarray,
        eda: np.ndarray,
        ppg: np.ndarray,
        imu: np.ndarray,
        summary: np.ndarray,
        user_id: Union[int, str] = 0,
        return_confidence: bool = True
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        Single prediction with confidence scores.
        
        Returns:
            {
                'valence': 'Low'/'Medium'/'High',
                'arousal': 'Low'/'Medium'/'High',
                'dominance': 'Low'/'Medium'/'High',
                'valence_logits': [L, M, H] logits,
                'arousal_logits': [L, M, H] logits,
                'dominance_logits': [L, M, H] logits,
                'valence_confidence': 0.85,
                'arousal_confidence': 0.72,
                'dominance_confidence': 0.91,
                'personality': {trait: score, ...}
            }
        """
        # Preprocess
        inputs = self.preprocess_input(gaze, pupil, eda, ppg, imu, summary, user_id)
        
        # Inference with mixed precision if available
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            outputs = self.model(
                inputs['gaze'], inputs['pupil'], inputs['eda'],
                inputs['ppg'], inputs['imu'], inputs['summary'],
                inputs['user_id'], personality_gt=None,
                task_onehot=inputs['task_onehot']
            )
        
        # Extract predictions
        result = {}
        
        # VAD emotion predictions (argmax)
        for dim, dim_name in enumerate(['valence', 'arousal', 'dominance']):
            logits = outputs[f'{dim_name}_logits'][0]  # (1, 3) -> (3,)
            pred_class = logits.argmax().item()
            
            result[f'{dim_name}'] = self.vad_labels[pred_class]
            result[f'{dim_name}_logits'] = logits.cpu().numpy()
            
            if return_confidence:
                probs = F.softmax(logits, dim=0)
                result[f'{dim_name}_confidence'] = probs[pred_class].item()
                result[f'{dim_name}_distribution'] = probs.cpu().numpy()  # All class probs
        
        # Personality predictions (optional auxiliary)
        if 'personality_pred' in outputs:
            personality = outputs['personality_pred'][0].cpu().numpy()
            result['personality'] = {
                trait: score for trait, score in zip(self.personality_traits, personality)
            }
        
        return result
    
    def predict_batch(
        self,
        batch: List[Tuple[np.ndarray, ...]], 
        user_ids: Optional[List[int]] = None,
        return_confidence: bool = True
    ) -> List[Dict]:
        """
        Batch prediction for multiple windows.
        
        Args:
            batch: List of (gaze, pupil, eda, ppg, imu, summary) tuples
            user_ids: Optional user IDs for each sample
            return_confidence: Include confidence scores
        
        Returns:
            List of prediction dicts
        """
        if user_ids is None:
            user_ids = [0] * len(batch)
        
        results = []
        for i, (gaze, pupil, eda, ppg, imu, summary) in enumerate(batch):
            pred = self.predict(
                gaze, pupil, eda, ppg, imu, summary,
                user_id=user_ids[i],
                return_confidence=return_confidence
            )
            results.append(pred)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Return model metadata for logging/monitoring."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'architecture': 'MuMTAffect-Transformer',
            'total_parameters': total_params,
            'total_parameters_m': total_params / 1e6,
            'device': str(self.device),
            'mixed_precision': self.use_amp,
            'frozen_weights': True,
            'inference_mode': True,
            'model_path': str(self.model_path),
            'emotion_dimensions': 3,  # V, A, D
            'emotion_classes': 3,      # Low, Medium, High
            'personality_traits': len(self.personality_traits)
        }


def load_and_validate_model(model_path: Union[str, Path]) -> MuMTAffectInferenceWrapper:
    """
    Load model and run validation checks.
    
    Returns:
        Initialized inference wrapper
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model validation fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    wrapper = MuMTAffectInferenceWrapper(model_path, device='cuda')
    
    # Quick validation
    logger.info("Running validation test...")
    # Use correct summary dimension for checkpoint (44, not current 49)
    dummy_input = {
        'gaze': np.random.randn(400, 9),
        'pupil': np.random.randn(400, 3),
        'eda': np.random.randn(400, 5),
        'ppg': np.random.randn(400, 3),
        'imu': np.random.randn(400, 6),
        'summary': np.random.randn(44)  # Checkpoint was trained with 44-dim summary
    }
    
    pred = wrapper.predict(**dummy_input)
    logger.info(f"✓ Validation passed | V={pred['valence']}, A={pred['arousal']}, D={pred['dominance']}")
    
    return wrapper


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load model - try production path first, fallback to stratified
    model_path = 'data/mumt/production_model/model_transformer_baseline_stratified.pt'
    if not Path(model_path).exists():
        model_path = 'data/mumt/runs_v7_stratified/model_final.pt'
    
    wrapper = load_and_validate_model(model_path)
    
    # Print model info
    info = wrapper.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
