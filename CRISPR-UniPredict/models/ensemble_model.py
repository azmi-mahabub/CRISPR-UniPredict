"""
Ensemble Model for CRISPR-UniPredict
Combines predictions from multiple model variants for improved robustness
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from dataclasses import dataclass

from models.crispr_unipredict import CRISPRUniPredict

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model"""
    n_models: int = 5
    ensemble_method: str = 'weighted_average'  # simple_average, weighted_average, stacking, voting
    device: str = 'cuda'
    use_meta_model: bool = True
    meta_model_hidden_dim: int = 128
    meta_model_dropout: float = 0.3


class MetaModel(nn.Module):
    """Meta-model for stacking ensemble"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.3,
                 output_dim: int = 2):
        """
        Initialize meta-model
        
        Args:
            input_dim: Input dimension (n_models * 2 for on-target and off-target)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            output_dim: Output dimension (2 for on-target and off-target)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"MetaModel initialized: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class EnsembleModel:
    """Ensemble model combining multiple CRISPR-UniPredict variants"""
    
    def __init__(self,
                 model_checkpoints: List[str],
                 config: Optional[EnsembleConfig] = None,
                 meta_model_checkpoint: Optional[str] = None):
        """
        Initialize ensemble model
        
        Args:
            model_checkpoints: List of paths to model checkpoints
            config: Ensemble configuration
            meta_model_checkpoint: Path to meta-model checkpoint (for stacking)
        """
        self.model_checkpoints = model_checkpoints
        self.config = config or EnsembleConfig()
        self.device = self.config.device
        
        # Load models
        self.models = []
        for checkpoint in model_checkpoints:
            model = self._load_model(checkpoint)
            self.models.append(model)
        
        logger.info(f"Loaded {len(self.models)} models for ensemble")
        
        # Initialize weights
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Initialize meta-model if using stacking
        self.meta_model = None
        if self.config.use_meta_model and self.config.ensemble_method == 'stacking':
            self.meta_model = self._init_meta_model()
            if meta_model_checkpoint:
                self._load_meta_model(meta_model_checkpoint)
    
    def _load_model(self, checkpoint_path: str) -> CRISPRUniPredict:
        """
        Load a single model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Loaded model
        """
        model = CRISPRUniPredict(device=self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return model
    
    def _init_meta_model(self) -> MetaModel:
        """
        Initialize meta-model for stacking
        
        Returns:
            Meta-model instance
        """
        # Input dimension: n_models * 2 (on-target + off-target predictions)
        input_dim = len(self.models) * 2
        
        meta_model = MetaModel(
            input_dim=input_dim,
            hidden_dim=self.config.meta_model_hidden_dim,
            dropout=self.config.meta_model_dropout,
            output_dim=2
        )
        
        meta_model.to(self.device)
        return meta_model
    
    def _load_meta_model(self, checkpoint_path: str) -> None:
        """
        Load meta-model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.meta_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.meta_model.load_state_dict(checkpoint)
        
        self.meta_model.eval()
        logger.info(f"Loaded meta-model from {checkpoint_path}")
    
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set ensemble weights
        
        Args:
            weights: Array of weights (must sum to 1)
        """
        weights = np.array(weights)
        assert len(weights) == len(self.models), "Number of weights must match number of models"
        assert np.isclose(np.sum(weights), 1.0), "Weights must sum to 1"
        
        self.weights = weights
        logger.info(f"Set ensemble weights: {weights}")
    
    def compute_weights(self,
                       val_scc_scores: List[float],
                       val_auroc_scores: List[float]) -> np.ndarray:
        """
        Compute weights based on validation performance
        
        Args:
            val_scc_scores: SCC scores for each model
            val_auroc_scores: AUROC scores for each model
        
        Returns:
            Normalized weights
        """
        val_scc_scores = np.array(val_scc_scores)
        val_auroc_scores = np.array(val_auroc_scores)
        
        # Combine scores (average of SCC and AUROC)
        combined_scores = (val_scc_scores + val_auroc_scores) / 2
        
        # Normalize to [0, 1]
        min_score = np.min(combined_scores)
        max_score = np.max(combined_scores)
        if max_score > min_score:
            normalized_scores = (combined_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(combined_scores)
        
        # Convert to weights (sum to 1)
        weights = normalized_scores / np.sum(normalized_scores)
        
        self.set_weights(weights)
        logger.info(f"Computed weights from validation scores: {weights}")
        
        return weights
    
    def predict(self,
                sgrna_onehot: torch.Tensor,
                sgrna_label: torch.Tensor,
                method: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using ensemble
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
            method: Ensemble method (uses config default if None)
        
        Returns:
            Tuple of (on_target_pred, off_target_pred)
        """
        method = method or self.config.ensemble_method
        
        if method == 'simple_average':
            return self.predict_simple_average(sgrna_onehot, sgrna_label)
        elif method == 'weighted_average':
            return self.predict_weighted_average(sgrna_onehot, sgrna_label)
        elif method == 'stacking':
            return self.predict_stacking(sgrna_onehot, sgrna_label)
        elif method == 'voting':
            return self.predict_voting(sgrna_onehot, sgrna_label)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def predict_simple_average(self,
                               sgrna_onehot: torch.Tensor,
                               sgrna_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple averaging of predictions from all models
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
        
        Returns:
            Tuple of (on_target_pred, off_target_pred)
        """
        all_on_target = []
        all_off_target = []
        
        with torch.no_grad():
            for model in self.models:
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                all_on_target.append(on_target)
                all_off_target.append(off_target)
        
        # Average predictions
        on_target_pred = torch.mean(torch.stack(all_on_target), dim=0)
        off_target_pred = torch.mean(torch.stack(all_off_target), dim=0)
        
        return on_target_pred, off_target_pred
    
    def predict_weighted_average(self,
                                 sgrna_onehot: torch.Tensor,
                                 sgrna_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted averaging based on model performance
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
        
        Returns:
            Tuple of (on_target_pred, off_target_pred)
        """
        all_on_target = []
        all_off_target = []
        
        with torch.no_grad():
            for model in self.models:
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                all_on_target.append(on_target)
                all_off_target.append(off_target)
        
        # Stack predictions
        on_target_stack = torch.stack(all_on_target)  # (n_models, batch_size, 1)
        off_target_stack = torch.stack(all_off_target)  # (n_models, batch_size, 1)
        
        # Convert weights to tensor
        weights = torch.tensor(self.weights, dtype=torch.float32, device=self.device)
        weights = weights.view(-1, 1, 1)  # (n_models, 1, 1)
        
        # Weighted average
        on_target_pred = torch.sum(on_target_stack * weights, dim=0)
        off_target_pred = torch.sum(off_target_stack * weights, dim=0)
        
        return on_target_pred, off_target_pred
    
    def predict_stacking(self,
                        sgrna_onehot: torch.Tensor,
                        sgrna_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use meta-model for final prediction (stacking)
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
        
        Returns:
            Tuple of (on_target_pred, off_target_pred)
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized. Use weighted_average instead.")
        
        # Get predictions from all models
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                # Concatenate on-target and off-target predictions
                predictions = torch.cat([on_target, off_target], dim=1)  # (batch_size, 2)
                all_predictions.append(predictions)
        
        # Stack predictions: (batch_size, n_models * 2)
        meta_input = torch.cat(all_predictions, dim=1)
        
        # Get meta-model predictions
        with torch.no_grad():
            meta_output = self.meta_model(meta_input)  # (batch_size, 2)
        
        # Split into on-target and off-target
        on_target_pred = meta_output[:, 0:1]  # (batch_size, 1)
        off_target_pred = meta_output[:, 1:2]  # (batch_size, 1)
        
        return on_target_pred, off_target_pred
    
    def predict_voting(self,
                      sgrna_onehot: torch.Tensor,
                      sgrna_label: torch.Tensor,
                      threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Majority voting for classification
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
            threshold: Classification threshold
        
        Returns:
            Tuple of (on_target_pred, off_target_pred)
        """
        all_on_target = []
        all_off_target = []
        
        with torch.no_grad():
            for model in self.models:
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                all_on_target.append(on_target)
                all_off_target.append(off_target)
        
        # Stack predictions
        on_target_stack = torch.stack(all_on_target)  # (n_models, batch_size, 1)
        off_target_stack = torch.stack(all_off_target)  # (n_models, batch_size, 1)
        
        # Majority voting for off-target (binary classification)
        off_target_votes = (off_target_stack > threshold).float()  # (n_models, batch_size, 1)
        off_target_pred = torch.mean(off_target_votes, dim=0)  # (batch_size, 1)
        
        # Average for on-target (regression)
        on_target_pred = torch.mean(on_target_stack, dim=0)  # (batch_size, 1)
        
        return on_target_pred, off_target_pred
    
    def get_model_predictions(self,
                             sgrna_onehot: torch.Tensor,
                             sgrna_label: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get predictions from each individual model
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
        
        Returns:
            Dictionary mapping model index to (on_target_pred, off_target_pred)
        """
        predictions = {}
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                predictions[i] = (on_target, off_target)
        
        return predictions
    
    def get_prediction_variance(self,
                               sgrna_onehot: torch.Tensor,
                               sgrna_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get variance of predictions across models (uncertainty estimate)
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA sequences
            sgrna_label: Label encoded sgRNA sequences
        
        Returns:
            Tuple of (on_target_var, off_target_var)
        """
        all_on_target = []
        all_off_target = []
        
        with torch.no_grad():
            for model in self.models:
                on_target, off_target = model(sgrna_onehot, sgrna_label, task_type='both')
                all_on_target.append(on_target)
                all_off_target.append(off_target)
        
        # Stack predictions
        on_target_stack = torch.stack(all_on_target)  # (n_models, batch_size, 1)
        off_target_stack = torch.stack(all_off_target)  # (n_models, batch_size, 1)
        
        # Compute variance
        on_target_var = torch.var(on_target_stack, dim=0)
        off_target_var = torch.var(off_target_stack, dim=0)
        
        return on_target_var, off_target_var
    
    def save_config(self, save_path: str) -> None:
        """
        Save ensemble configuration
        
        Args:
            save_path: Path to save configuration
        """
        config_dict = {
            'n_models': len(self.models),
            'ensemble_method': self.config.ensemble_method,
            'weights': self.weights.tolist(),
            'model_checkpoints': self.model_checkpoints,
            'device': self.device
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved ensemble config to {save_path}")
    
    @classmethod
    def load_config(cls, config_path: str) -> 'EnsembleModel':
        """
        Load ensemble from configuration file
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Loaded ensemble model
        """
        with open(config_path) as f:
            config_dict = json.load(f)
        
        ensemble = cls(
            model_checkpoints=config_dict['model_checkpoints'],
            config=EnsembleConfig(
                n_models=config_dict['n_models'],
                ensemble_method=config_dict['ensemble_method'],
                device=config_dict.get('device', 'cuda')
            )
        )
        
        ensemble.set_weights(np.array(config_dict['weights']))
        
        logger.info(f"Loaded ensemble from {config_path}")
        
        return ensemble
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"EnsembleModel(\n"
            f"  n_models={len(self.models)},\n"
            f"  ensemble_method={self.config.ensemble_method},\n"
            f"  weights={self.weights},\n"
            f"  device={self.device}\n"
            f")"
        )
