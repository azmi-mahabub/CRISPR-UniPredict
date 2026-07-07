"""
Multi-Task Loss Functions for CRISPR-UniPredict
Handles on-target (regression) and off-target (classification) tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import logging
import warnings

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for CRISPR-UniPredict
    
    Combines on-target (regression) and off-target (classification) losses
    with proper masking for mixed batches.
    
    Architecture:
    - On-target loss: MAE (Mean Absolute Error) for regression
    - Off-target loss: BCE (Binary Cross Entropy) for classification
    - Combined loss: α * L_on_target + β * L_off_target
    
    Features:
    - Handles mixed batches with both task types
    - Proper masking for inapplicable labels
    - Gradient stability checks
    - Learnable or fixed loss weights
    - Detailed loss logging
    
    Example:
        >>> criterion = MultiTaskLoss(
        ...     on_target_weight=1.0,
        ...     off_target_weight=0.5,
        ...     learnable_weights=False
        ... )
        >>> 
        >>> on_target_pred = model.predict_on_target(x, y)
        >>> off_target_pred = model.predict_off_target(x, y)
        >>> 
        >>> loss, loss_dict = criterion(
        ...     on_target_pred=on_target_pred,
        ...     off_target_pred=off_target_pred,
        ...     on_target_target=on_target_label,
        ...     off_target_target=off_target_label,
        ...     on_target_mask=on_target_mask,
        ...     off_target_mask=off_target_mask
        ... )
    """
    
    def __init__(self,
                 on_target_weight: float = 1.0,
                 off_target_weight: float = 0.5,
                 learnable_weights: bool = False,
                 on_target_loss_fn: str = 'mae',
                 off_target_loss_fn: str = 'bce',
                 reduction: str = 'mean',
                 epsilon: float = 1e-7,
                 huber_delta: float = 0.5,
                 pos_weight: float = 93.0):
        """
        Initialize Multi-Task Loss
        
        Args:
            on_target_weight: Weight for on-target loss (default: 1.0)
            off_target_weight: Weight for off-target loss (default: 0.5)
            learnable_weights: Whether weights are learnable (default: False)
            on_target_loss_fn: Loss function for on-target ('mae', 'mse', or 'huber')
            off_target_loss_fn: Loss function for off-target ('bce' or 'focal')
            reduction: Reduction method ('mean', 'sum', 'none')
            epsilon: Small value for numerical stability
            huber_delta: Delta parameter for Huber loss
        """
        super(MultiTaskLoss, self).__init__()
        
        self.on_target_loss_fn_name = on_target_loss_fn
        self.off_target_loss_fn_name = off_target_loss_fn
        self.reduction = reduction
        self.epsilon = epsilon
        self.huber_delta = huber_delta
        
        # Initialize loss weights
        if learnable_weights:
            self.on_target_weight = nn.Parameter(torch.tensor(on_target_weight))
            self.off_target_weight = nn.Parameter(torch.tensor(off_target_weight))
        else:
            self.register_buffer('on_target_weight', torch.tensor(on_target_weight))
            self.register_buffer('off_target_weight', torch.tensor(off_target_weight))
        
        # Initialize loss functions
        if on_target_loss_fn == 'mae':
            self.on_target_loss = nn.L1Loss(reduction=reduction)
        elif on_target_loss_fn == 'mse':
            self.on_target_loss = nn.MSELoss(reduction=reduction)
        elif on_target_loss_fn == 'huber':
            self.on_target_loss = nn.HuberLoss(delta=huber_delta, reduction=reduction)
        else:
            raise ValueError(f"Unknown on-target loss: {on_target_loss_fn}")
        
        if off_target_loss_fn == 'bce':
            self.off_target_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]), reduction=reduction
            )
        elif off_target_loss_fn == 'focal':
            self.off_target_loss = FocalLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown off-target loss: {off_target_loss_fn}")
        
        logger.info(
            f"MultiTaskLoss initialized:\n"
            f"  On-target loss: {on_target_loss_fn} (weight: {on_target_weight})\n"
            f"  Off-target loss: {off_target_loss_fn} (weight: {off_target_weight})\n"
            f"  Learnable weights: {learnable_weights}\n"
            f"  Reduction: {reduction}"
        )
    
    def forward(self,
                on_target_pred: Optional[torch.Tensor] = None,
                off_target_pred: Optional[torch.Tensor] = None,
                on_target_target: Optional[torch.Tensor] = None,
                off_target_target: Optional[torch.Tensor] = None,
                on_target_mask: Optional[torch.Tensor] = None,
                off_target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss
        
        Args:
            on_target_pred: On-target predictions (batch, 1) or (batch,)
            off_target_pred: Off-target predictions (batch, 1) or (batch,)
            on_target_target: On-target targets (batch, 1) or (batch,)
            off_target_target: Off-target targets (batch, 1) or (batch,)
            on_target_mask: Mask for valid on-target samples (batch,)
                           True where label is valid, False where missing
            off_target_mask: Mask for valid off-target samples (batch,)
                            True where label is valid, False where missing
        
        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Weighted sum of task losses
            - loss_dict: Dictionary with individual losses and metrics
        
        Raises:
            ValueError: If no predictions provided or shapes mismatch
        """
        # Validate inputs
        if on_target_pred is None and off_target_pred is None:
            raise ValueError("At least one prediction must be provided")
        
        loss_dict = {}
        total_loss = 0.0
        
        # Compute on-target loss
        if on_target_pred is not None and on_target_target is not None:
            on_target_loss, on_target_metrics = self._compute_on_target_loss(
                on_target_pred,
                on_target_target,
                on_target_mask
            )
            
            loss_dict['on_target_loss'] = on_target_loss
            loss_dict.update(on_target_metrics)
            
            # Add weighted loss to total
            total_loss = total_loss + self.on_target_weight * on_target_loss
        
        # Compute off-target loss
        if off_target_pred is not None and off_target_target is not None:
            off_target_loss, off_target_metrics = self._compute_off_target_loss(
                off_target_pred,
                off_target_target,
                off_target_mask
            )
            
            loss_dict['off_target_loss'] = off_target_loss
            loss_dict.update(off_target_metrics)
            
            # Add weighted loss to total
            total_loss = total_loss + self.off_target_weight * off_target_loss
        
        # Add weights to loss dict
        loss_dict['on_target_weight'] = self.on_target_weight
        loss_dict['off_target_weight'] = self.off_target_weight
        loss_dict['total_loss'] = total_loss
        
        # Check for gradient stability
        self._check_gradient_stability(total_loss)
        
        return total_loss, loss_dict
    
    def _compute_on_target_loss(self,
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute on-target (regression) loss
        
        Args:
            predictions: Predictions (batch, 1) or (batch,)
            targets: Targets (batch, 1) or (batch,)
            mask: Valid sample mask (batch,)
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Squeeze if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Apply mask
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Check if any valid samples
        if predictions.numel() == 0:
            logger.warning("No valid on-target samples in batch")
            return torch.tensor(0.0, device=predictions.device), {
                'on_target_valid_count': 0,
                'on_target_pred_mean': torch.tensor(0.0),
                'on_target_pred_std': torch.tensor(0.0),
                'on_target_target_mean': torch.tensor(0.0),
                'on_target_target_std': torch.tensor(0.0)
            }
        
        # Compute loss
        loss = self.on_target_loss(predictions, targets)
        
        # Compute metrics
        metrics = {
            'on_target_valid_count': predictions.numel(),
            'on_target_pred_mean': predictions.mean(),
            'on_target_pred_std': predictions.std(),
            'on_target_target_mean': targets.mean(),
            'on_target_target_std': targets.std(),
            'on_target_mae': F.l1_loss(predictions, targets),
            'on_target_mse': F.mse_loss(predictions, targets)
        }
        
        return loss, metrics
    
    def _compute_off_target_loss(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute off-target (classification) loss
        
        Args:
            predictions: Predictions (batch, 1) or (batch,)
            targets: Targets (batch, 1) or (batch,) with values 0 or 1
            mask: Valid sample mask (batch,)
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Squeeze if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Convert targets to float
        targets = targets.float()
        
        # Apply mask
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Check if any valid samples
        if predictions.numel() == 0:
            logger.warning("No valid off-target samples in batch")
            return torch.tensor(0.0, device=predictions.device), {
                'off_target_valid_count': 0,
                'off_target_pred_mean': torch.tensor(0.0),
                'off_target_pred_std': torch.tensor(0.0),
                'off_target_target_positive': 0,
                'off_target_target_negative': 0
            }
        
        # Compute loss
        loss = self.off_target_loss(predictions, targets)
        
        # Compute metrics
        positive_count = (targets == 1).sum().item()
        negative_count = (targets == 0).sum().item()
        
        metrics = {
            'off_target_valid_count': predictions.numel(),
            'off_target_pred_mean': predictions.mean(),
            'off_target_pred_std': predictions.std(),
            'off_target_target_positive': positive_count,
            'off_target_target_negative': negative_count,
            'off_target_accuracy': (
                ((predictions > 0.5).float() == targets).float().mean()
            ),
            'off_target_auc': self._compute_auc(predictions, targets)
        }
        
        return loss, metrics
    
    @staticmethod
    def _compute_auc(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute AUC (Area Under Curve) for binary classification
        
        Args:
            predictions: Predictions (batch,)
            targets: Targets (batch,) with values 0 or 1
        
        Returns:
            AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            
            # Check if both classes present
            if len(set(target_np)) < 2:
                return torch.tensor(0.0)
            
            auc = roc_auc_score(target_np, pred_np)
            return torch.tensor(auc)
        except Exception as e:
            logger.debug(f"Failed to compute AUC: {e}")
            return torch.tensor(0.0)
    
    def _check_gradient_stability(self, loss: torch.Tensor) -> None:
        """
        Check for gradient stability issues
        
        Args:
            loss: Loss tensor
        """
        if torch.isnan(loss):
            warnings.warn("Loss is NaN - gradient instability detected")
        elif torch.isinf(loss):
            warnings.warn("Loss is Inf - gradient instability detected")
        elif loss > 1e6:
            warnings.warn(f"Loss is very large ({loss.item():.2e}) - possible gradient explosion")
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current loss weights
        
        Returns:
            Dictionary with loss weights
        """
        return {
            'on_target_weight': self.on_target_weight.item() if isinstance(self.on_target_weight, nn.Parameter) else self.on_target_weight.item(),
            'off_target_weight': self.off_target_weight.item() if isinstance(self.off_target_weight, nn.Parameter) else self.off_target_weight.item()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Addresses class imbalance by down-weighting easy examples and
    focusing on hard examples.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for class 1 (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Predictions (batch,) with values in [0, 1]
            targets: Targets (batch,) with values 0 or 1
        
        Returns:
            Focal loss
        """
        # Compute binary cross entropy
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Compute focal term
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha term
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        loss = alpha_t * focal_term * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss with dynamic weight adjustment
    
    Automatically adjusts loss weights based on task performance
    to balance training between tasks.
    """
    
    def __init__(self,
                 initial_on_target_weight: float = 1.0,
                 initial_off_target_weight: float = 0.5,
                 temperature: float = 1.0):
        """
        Initialize Weighted Multi-Task Loss
        
        Args:
            initial_on_target_weight: Initial on-target weight
            initial_off_target_weight: Initial off-target weight
            temperature: Temperature for softmax weighting
        """
        super(WeightedMultiTaskLoss, self).__init__()
        
        self.base_loss = MultiTaskLoss(
            on_target_weight=initial_on_target_weight,
            off_target_weight=initial_off_target_weight,
            learnable_weights=True
        )
        self.temperature = temperature
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        return self.base_loss(*args, **kwargs)
    
    def update_weights(self, on_target_loss: float, off_target_loss: float) -> None:
        """
        Update loss weights based on task losses
        
        Args:
            on_target_loss: Current on-target loss
            off_target_loss: Current off-target loss
        """
        # Compute normalized weights using softmax
        losses = torch.tensor([on_target_loss, off_target_loss])
        weights = F.softmax(losses / self.temperature, dim=0)
        
        # Update weights
        with torch.no_grad():
            self.base_loss.on_target_weight.copy_(weights[0])
            self.base_loss.off_target_weight.copy_(weights[1])


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-TASK LOSS TESTING")
    print("=" * 80)
    
    # Test 1: Initialize loss
    print("\n1. INITIALIZE MULTI-TASK LOSS")
    print("-" * 80)
    
    criterion = MultiTaskLoss(
        on_target_weight=1.0,
        off_target_weight=0.5,
        learnable_weights=False
    )
    
    print(f"[OK] MultiTaskLoss initialized")
    print(f"  Weights: {criterion.get_loss_weights()}")
    
    # Test 2: Compute loss with both tasks
    print("\n2. COMPUTE LOSS WITH BOTH TASKS")
    print("-" * 80)
    
    batch_size = 8
    
    # Create dummy predictions and targets
    on_target_pred = torch.rand(batch_size, 1)
    on_target_target = torch.rand(batch_size, 1)
    on_target_mask = torch.ones(batch_size, dtype=torch.bool)
    
    off_target_pred = torch.rand(batch_size, 1)
    off_target_target = torch.randint(0, 2, (batch_size, 1)).float()
    off_target_mask = torch.ones(batch_size, dtype=torch.bool)
    
    loss, loss_dict = criterion(
        on_target_pred=on_target_pred,
        off_target_pred=off_target_pred,
        on_target_target=on_target_target,
        off_target_target=off_target_target,
        on_target_mask=on_target_mask,
        off_target_mask=off_target_mask
    )
    
    print(f"[OK] Loss computed")
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  On-target loss: {loss_dict['on_target_loss'].item():.6f}")
    print(f"  Off-target loss: {loss_dict['off_target_loss'].item():.6f}")
    
    # Test 3: Compute loss with masking
    print("\n3. COMPUTE LOSS WITH MASKING")
    print("-" * 80)
    
    # Create masks with some invalid samples
    on_target_mask = torch.tensor([True, True, False, True, False, True, True, True])
    off_target_mask = torch.tensor([True, False, True, False, True, True, True, False])
    
    loss, loss_dict = criterion(
        on_target_pred=on_target_pred,
        off_target_pred=off_target_pred,
        on_target_target=on_target_target,
        off_target_target=off_target_target,
        on_target_mask=on_target_mask,
        off_target_mask=off_target_mask
    )
    
    print(f"[OK] Loss computed with masking")
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  On-target valid samples: {loss_dict['on_target_valid_count']}")
    print(f"  Off-target valid samples: {loss_dict['off_target_valid_count']}")
    
    # Test 4: Compute loss with only on-target
    print("\n4. COMPUTE LOSS WITH ONLY ON-TARGET")
    print("-" * 80)
    
    loss, loss_dict = criterion(
        on_target_pred=on_target_pred,
        on_target_target=on_target_target,
        on_target_mask=on_target_mask
    )
    
    print(f"[OK] Loss computed (on-target only)")
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  On-target loss: {loss_dict['on_target_loss'].item():.6f}")
    print(f"  Off-target loss: {loss_dict.get('off_target_loss', 'N/A')}")
    
    # Test 5: Compute loss with only off-target
    print("\n5. COMPUTE LOSS WITH ONLY OFF-TARGET")
    print("-" * 80)
    
    loss, loss_dict = criterion(
        off_target_pred=off_target_pred,
        off_target_target=off_target_target,
        off_target_mask=off_target_mask
    )
    
    print(f"[OK] Loss computed (off-target only)")
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  On-target loss: {loss_dict.get('on_target_loss', 'N/A')}")
    print(f"  Off-target loss: {loss_dict['off_target_loss'].item():.6f}")
    
    # Test 6: Learnable weights
    print("\n6. LEARNABLE WEIGHTS")
    print("-" * 80)
    
    criterion_learnable = MultiTaskLoss(
        on_target_weight=1.0,
        off_target_weight=0.5,
        learnable_weights=True
    )
    
    print(f"[OK] MultiTaskLoss with learnable weights")
    print(f"  Weights: {criterion_learnable.get_loss_weights()}")
    
    # Test 7: Focal loss
    print("\n7. FOCAL LOSS")
    print("-" * 80)
    
    criterion_focal = MultiTaskLoss(
        off_target_loss_fn='focal'
    )
    
    loss, loss_dict = criterion_focal(
        off_target_pred=off_target_pred,
        off_target_target=off_target_target,
        off_target_mask=off_target_mask
    )
    
    print(f"[OK] Focal loss computed")
    print(f"  Off-target loss: {loss_dict['off_target_loss'].item():.6f}")
    
    # Test 8: Gradient flow
    print("\n8. GRADIENT FLOW TEST")
    print("-" * 80)
    
    on_target_pred = torch.rand(batch_size, 1, requires_grad=True)
    off_target_pred = torch.rand(batch_size, 1, requires_grad=True)
    
    loss, loss_dict = criterion(
        on_target_pred=on_target_pred,
        off_target_pred=off_target_pred,
        on_target_target=on_target_target,
        off_target_target=off_target_target,
        on_target_mask=on_target_mask,
        off_target_mask=off_target_mask
    )
    
    loss.backward()
    
    print(f"[OK] Gradients computed")
    print(f"  On-target pred grad: {on_target_pred.grad.mean().item():.6f}")
    print(f"  Off-target pred grad: {off_target_pred.grad.mean().item():.6f}")
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
