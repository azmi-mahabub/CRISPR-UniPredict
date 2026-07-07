# Multi-Task Loss Functions Guide

## Overview

The `losses.py` module provides multi-task loss functions for training CRISPR-UniPredict with both on-target (regression) and off-target (classification) tasks.

---

## Key Components

### 1. MultiTaskLoss

**Purpose**: Combine on-target and off-target losses with proper masking.

**Architecture**:
- **On-target loss**: MAE (Mean Absolute Error) for regression
- **Off-target loss**: BCE (Binary Cross Entropy) for classification
- **Combined loss**: α * L_on_target + β * L_off_target

**Features**:
- ✅ Handles mixed batches with both task types
- ✅ Proper masking for inapplicable labels
- ✅ Gradient stability checks
- ✅ Learnable or fixed loss weights
- ✅ Detailed loss logging

### 2. FocalLoss

**Purpose**: Handle class imbalance in off-target classification.

**Features**:
- ✅ Down-weights easy examples
- ✅ Focuses on hard examples
- ✅ Configurable alpha and gamma parameters

### 3. WeightedMultiTaskLoss

**Purpose**: Dynamically adjust loss weights based on task performance.

**Features**:
- ✅ Automatic weight adjustment
- ✅ Balances training between tasks
- ✅ Temperature-controlled softmax weighting

---

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from utils.losses import MultiTaskLoss

# Create loss function
criterion = MultiTaskLoss(
    on_target_weight=1.0,
    off_target_weight=0.5,
    learnable_weights=False
)

# Forward pass
on_target_pred = model.predict_on_target(x, y)
off_target_pred = model.predict_off_target(x, y)

# Compute loss
loss, loss_dict = criterion(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_label,
    off_target_target=off_target_label,
    on_target_mask=on_target_mask,
    off_target_mask=off_target_mask
)

# Backward pass
loss.backward()
optimizer.step()
```

### With Learnable Weights

```python
# Create loss with learnable weights
criterion = MultiTaskLoss(
    on_target_weight=1.0,
    off_target_weight=0.5,
    learnable_weights=True
)

# Weights are now parameters and will be optimized
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-3
)
```

### With Focal Loss

```python
# Use focal loss for off-target classification
criterion = MultiTaskLoss(
    off_target_loss_fn='focal',
    on_target_weight=1.0,
    off_target_weight=0.5
)
```

---

## MultiTaskLoss Class

### Initialization

```python
criterion = MultiTaskLoss(
    on_target_weight=1.0,           # Weight for on-target loss
    off_target_weight=0.5,          # Weight for off-target loss
    learnable_weights=False,        # Whether weights are learnable
    on_target_loss_fn='mae',        # 'mae' or 'mse'
    off_target_loss_fn='bce',       # 'bce' or 'focal'
    reduction='mean',               # 'mean', 'sum', 'none'
    epsilon=1e-7                    # Numerical stability
)
```

### Forward Pass

```python
loss, loss_dict = criterion(
    on_target_pred=on_target_pred,      # (batch, 1) or (batch,)
    off_target_pred=off_target_pred,    # (batch, 1) or (batch,)
    on_target_target=on_target_target,  # (batch, 1) or (batch,)
    off_target_target=off_target_target,# (batch, 1) or (batch,)
    on_target_mask=on_target_mask,      # (batch,) - True where valid
    off_target_mask=off_target_mask     # (batch,) - True where valid
)
```

**Returns**:
- `loss`: Total weighted loss (scalar)
- `loss_dict`: Dictionary with individual losses and metrics

### Loss Dictionary

```python
{
    'on_target_loss': float,              # On-target MAE/MSE
    'on_target_valid_count': int,         # Number of valid samples
    'on_target_pred_mean': float,         # Mean prediction
    'on_target_pred_std': float,          # Std of predictions
    'on_target_target_mean': float,       # Mean target
    'on_target_target_std': float,        # Std of targets
    'on_target_mae': float,               # MAE metric
    'on_target_mse': float,               # MSE metric
    
    'off_target_loss': float,             # Off-target BCE/Focal
    'off_target_valid_count': int,        # Number of valid samples
    'off_target_pred_mean': float,        # Mean prediction
    'off_target_pred_std': float,         # Std of predictions
    'off_target_target_positive': int,    # Number of positive samples
    'off_target_target_negative': int,    # Number of negative samples
    'off_target_accuracy': float,         # Classification accuracy
    'off_target_auc': float,              # AUC score
    
    'on_target_weight': float,            # Current on-target weight
    'off_target_weight': float,           # Current off-target weight
    'total_loss': float                   # Total weighted loss
}
```

### Methods

#### `get_loss_weights()`

Get current loss weights.

```python
weights = criterion.get_loss_weights()
print(weights)
# {'on_target_weight': 1.0, 'off_target_weight': 0.5}
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.crispr_unipredict import CRISPRUniPredict
from utils.losses import MultiTaskLoss
from utils.preprocessing.dataloader_factory import create_dataloaders
from configs.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create dataloaders
dataloaders = create_dataloaders(config)

# Initialize model
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)

# Initialize loss function
criterion = MultiTaskLoss(
    on_target_weight=config.training.loss.loss_weights['on_target'],
    off_target_weight=config.training.loss.loss_weights['off_target'],
    learnable_weights=False,
    on_target_loss_fn=config.training.loss.on_target_loss,
    off_target_loss_fn=config.training.loss.off_target_loss
)

# Initialize optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate_heads,
    weight_decay=config.training.weight_decay
)

# Training loop
for epoch in range(config.training.epochs):
    model.train()
    
    for batch in dataloaders['train']:
        # Move to device
        sgrna_onehot = batch['sgrna_onehot'].to(device)
        sgrna_label = batch['sgrna_label'].to(device)
        on_target_score = batch['on_target_score'].to(device)
        off_target_label = batch['off_target_label'].to(device)
        on_target_mask = batch['on_target_mask'].to(device)
        off_target_mask = batch['off_target_mask'].to(device)
        
        # Forward pass
        on_target_pred, off_target_pred = model(
            sgrna_onehot, sgrna_label, task_type='both'
        )
        
        # Compute loss
        loss, loss_dict = criterion(
            on_target_pred=on_target_pred,
            off_target_pred=off_target_pred,
            on_target_target=on_target_score,
            off_target_target=off_target_label,
            on_target_mask=on_target_mask,
            off_target_mask=off_target_mask
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.gradient_clip
        )
        optimizer.step()
        
        # Log metrics
        print(f"Loss: {loss.item():.6f}")
        print(f"  On-target: {loss_dict['on_target_loss'].item():.6f}")
        print(f"  Off-target: {loss_dict['off_target_loss'].item():.6f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in dataloaders['val']:
            sgrna_onehot = batch['sgrna_onehot'].to(device)
            sgrna_label = batch['sgrna_label'].to(device)
            on_target_score = batch['on_target_score'].to(device)
            off_target_label = batch['off_target_label'].to(device)
            on_target_mask = batch['on_target_mask'].to(device)
            off_target_mask = batch['off_target_mask'].to(device)
            
            on_target_pred, off_target_pred = model(
                sgrna_onehot, sgrna_label, task_type='both'
            )
            
            loss, loss_dict = criterion(
                on_target_pred=on_target_pred,
                off_target_pred=off_target_pred,
                on_target_target=on_target_score,
                off_target_target=off_target_label,
                on_target_mask=on_target_mask,
                off_target_mask=off_target_mask
            )
```

---

## Loss Functions

### On-Target Loss (Regression)

**MAE (Mean Absolute Error)**:
```
L_on_target = (1/n) * Σ |y_pred - y_true|
```

**MSE (Mean Squared Error)**:
```
L_on_target = (1/n) * Σ (y_pred - y_true)²
```

**When to use**:
- MAE: More robust to outliers
- MSE: Penalizes large errors more

### Off-Target Loss (Classification)

**BCE (Binary Cross Entropy)**:
```
L_off_target = -(1/n) * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
```

**Focal Loss**:
```
L_focal = -(1/n) * Σ α_t * (1 - p_t)^γ * BCE
```

**When to use**:
- BCE: Balanced classes
- Focal Loss: Imbalanced classes (off-target rare)

---

## Masking

### Understanding Masks

**On-target mask**: Indicates which samples have valid on-target labels
```python
on_target_mask = torch.tensor([True, False, True, True])
# Sample 1 has no on-target label
```

**Off-target mask**: Indicates which samples have valid off-target labels
```python
off_target_mask = torch.tensor([True, True, False, True])
# Sample 2 has no off-target label
```

### Proper Masking Usage

```python
# Compute loss only on valid samples
loss, loss_dict = criterion(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_score,
    off_target_target=off_target_label,
    on_target_mask=on_target_mask,      # Only samples with True are used
    off_target_mask=off_target_mask     # Only samples with True are used
)

# Check how many samples were used
print(f"On-target samples: {loss_dict['on_target_valid_count']}")
print(f"Off-target samples: {loss_dict['off_target_valid_count']}")
```

---

## Handling Mixed Batches

### Scenario: Batch with Both Task Types

```python
batch_size = 8

# Some samples have on-target labels, some have off-target
on_target_mask = torch.tensor([True, True, False, True, False, True, True, True])
off_target_mask = torch.tensor([True, False, True, False, True, True, True, False])

# Loss is computed separately for each task
loss, loss_dict = criterion(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_score,
    off_target_target=off_target_label,
    on_target_mask=on_target_mask,
    off_target_mask=off_target_mask
)

# Results
print(f"On-target loss computed on {loss_dict['on_target_valid_count']} samples")
print(f"Off-target loss computed on {loss_dict['off_target_valid_count']} samples")
print(f"Total loss: {loss.item():.6f}")
```

---

## Learnable Weights

### Motivation

Instead of fixed weights, learn optimal weights during training.

### Usage

```python
# Create loss with learnable weights
criterion = MultiTaskLoss(
    on_target_weight=1.0,
    off_target_weight=0.5,
    learnable_weights=True
)

# Add criterion parameters to optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-3
)

# Weights will be optimized during training
for batch in dataloader:
    loss, loss_dict = criterion(...)
    loss.backward()
    optimizer.step()
    
    # Check updated weights
    weights = criterion.get_loss_weights()
    print(f"Weights: {weights}")
```

---

## Focal Loss for Class Imbalance

### Problem

Off-target samples are rare (10-20% of dataset), causing:
- Model biased towards on-target
- Poor off-target performance

### Solution: Focal Loss

```python
criterion = MultiTaskLoss(
    off_target_loss_fn='focal',
    on_target_weight=1.0,
    off_target_weight=0.5
)
```

**How it works**:
1. Down-weights easy examples (high confidence)
2. Focuses on hard examples (low confidence)
3. Better handles class imbalance

**Parameters**:
- `alpha`: Weighting factor (default: 0.25)
- `gamma`: Focusing parameter (default: 2.0)

---

## Gradient Stability

### Automatic Checks

The loss function automatically checks for:
- NaN values
- Inf values
- Exploding gradients (loss > 1e6)

### Manual Checks

```python
loss, loss_dict = criterion(...)

# Check for issues
if torch.isnan(loss):
    print("WARNING: Loss is NaN")
elif torch.isinf(loss):
    print("WARNING: Loss is Inf")
elif loss > 1e6:
    print("WARNING: Loss is very large (gradient explosion)")
```

### Solutions

```python
# 1. Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Use batch normalization
# Already included in model

# 4. Reduce batch size
batch_size = 16  # Instead of 32
```

---

## Troubleshooting

### Issue: "Loss is NaN"

**Cause**: Numerical instability

**Solution**:
```python
# 1. Check for invalid targets
assert not torch.isnan(on_target_target).any()
assert not torch.isnan(off_target_target).any()

# 2. Clamp predictions
on_target_pred = torch.clamp(on_target_pred, 0.0, 1.0)
off_target_pred = torch.clamp(off_target_pred, 1e-7, 1.0 - 1e-7)

# 3. Use smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

### Issue: "Unbalanced loss contributions"

**Cause**: One task dominates training

**Solution**:
```python
# Use learnable weights
criterion = MultiTaskLoss(learnable_weights=True)

# Or adjust fixed weights
criterion = MultiTaskLoss(
    on_target_weight=0.5,
    off_target_weight=1.0  # Increase off-target weight
)
```

### Issue: "Poor off-target performance"

**Cause**: Class imbalance

**Solution**:
```python
# Use focal loss
criterion = MultiTaskLoss(off_target_loss_fn='focal')

# Or use BootstrapSampler for balanced batches
from utils.preprocessing.sampler import BootstrapSampler
sampler = BootstrapSampler(dataset, batch_size=32, on_target_ratio=0.5)
```

---

## API Reference

### MultiTaskLoss

```python
class MultiTaskLoss(nn.Module):
    def __init__(on_target_weight, off_target_weight, learnable_weights, 
                 on_target_loss_fn, off_target_loss_fn, reduction, epsilon)
    
    def forward(on_target_pred, off_target_pred, on_target_target, 
                off_target_target, on_target_mask, off_target_mask) 
        -> (loss, loss_dict)
    
    def get_loss_weights() -> Dict[str, float]
```

### FocalLoss

```python
class FocalLoss(nn.Module):
    def __init__(alpha, gamma, reduction)
    def forward(predictions, targets) -> loss
```

### WeightedMultiTaskLoss

```python
class WeightedMultiTaskLoss(nn.Module):
    def __init__(initial_on_target_weight, initial_off_target_weight, temperature)
    def forward(...) -> (loss, loss_dict)
    def update_weights(on_target_loss, off_target_loss)
```

---

## Summary

The losses module provides:
- ✅ **MultiTaskLoss**: Combined on-target and off-target losses
- ✅ **FocalLoss**: Handles class imbalance
- ✅ **WeightedMultiTaskLoss**: Dynamic weight adjustment
- ✅ **Proper masking**: Handles mixed batches
- ✅ **Gradient stability**: Automatic checks
- ✅ **Detailed logging**: Comprehensive metrics
- ✅ **Production-ready**: Tested and optimized

Perfect for training CRISPR-UniPredict models!
