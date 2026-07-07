# Optimization Configuration Guide

## Overview

The `optimization.py` module provides advanced optimizer and scheduler configuration with:

- **Differential learning rates** for different model components
- **Linear warmup** for stable training
- **ReduceLROnPlateau** scheduler for adaptive learning rate adjustment
- **Multiple scheduler types** (Cosine, Linear, Exponential)
- **Parameter group management** for fine-grained control

---

## Quick Start

### Basic Setup

```python
from training.optimization import setup_optimizer, setup_scheduler
from models.crispr_unipredict import CRISPRUniPredict
from configs.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader('configs/model_config.yaml').config

# Initialize model
model = CRISPRUniPredict(device='cuda')

# Setup optimizer with differential learning rates
optimizer = setup_optimizer(model, config)

# Setup scheduler with warmup
scheduler, warmup_scheduler = setup_scheduler(optimizer, config)
```

### Convenience Function

```python
from training.optimization import create_optimizer_and_scheduler

# Create both optimizer and scheduler
optimizer, scheduler = create_optimizer_and_scheduler(model, config)
```

---

## Differential Learning Rates

### Strategy (from CCLMoff)

Different learning rates for different components:

- **RNA-FM encoder**: 5×10⁻⁴ (pretrained, smaller LR)
- **Feature extraction** (MSC, BiGRU, MHSA): 1×10⁻³ (training from scratch)
- **Task heads**: 1×10⁻³ (task-specific)

### Why Different Learning Rates?

1. **Pretrained components** (RNA-FM):
   - Already well-trained
   - Need smaller LR to avoid catastrophic forgetting
   - Fine-tune carefully

2. **From-scratch components** (MSC, BiGRU, MHSA):
   - No prior knowledge
   - Need larger LR for faster learning
   - More aggressive updates

3. **Task-specific components** (heads):
   - Trained on task-specific data
   - Need moderate LR
   - Balance between stability and learning

### Configuration

```yaml
training:
  learning_rate_encoder: 5.0e-4    # RNA-FM and other pretrained
  learning_rate_heads: 1.0e-3      # Feature extraction and heads
  weight_decay: 0.01               # L2 regularization
  optimizer: AdamW                 # Optimizer type
```

---

## Optimizer Setup

### `setup_optimizer(model, config) -> Optimizer`

Creates optimizer with differential learning rates.

**Parameters**:
- `model`: CRISPRUniPredict instance
- `config`: Configuration object

**Returns**: Configured optimizer with parameter groups

**Example**:
```python
optimizer = setup_optimizer(model, config)

# View parameter groups
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Group {i}: {param_group['name']}, LR={param_group['lr']:.2e}")
```

### Parameter Groups

```python
# Automatically created:
# 1. RNA-FM encoder: 5e-4
# 2. Feature extraction: 1e-3
# 3. Task heads: 1e-3
```

### Supported Optimizers

```yaml
training:
  optimizer: AdamW  # or 'Adam', 'SGD'
```

- **AdamW**: Recommended (decoupled weight decay)
- **Adam**: Standard adaptive optimizer
- **SGD**: Stochastic gradient descent with momentum

---

## Scheduler Setup

### `setup_scheduler(optimizer, config) -> (Scheduler, WarmupScheduler)`

Creates scheduler with linear warmup.

**Parameters**:
- `optimizer`: Configured optimizer
- `config`: Configuration object
- `total_steps`: Optional total training steps

**Returns**: Tuple of (main_scheduler, warmup_scheduler)

### Warmup Phase

Linear warmup for first N epochs:

```python
# Warmup schedule
epoch 0: LR = 0.0 × base_lr
epoch 1: LR = 0.2 × base_lr
epoch 2: LR = 0.4 × base_lr
epoch 3: LR = 0.6 × base_lr
epoch 4: LR = 0.8 × base_lr
epoch 5: LR = 1.0 × base_lr (warmup complete)
```

**Configuration**:
```yaml
training:
  warmup_epochs: 5
```

### Main Scheduler

#### ReduceLROnPlateau (Recommended)

Reduces LR when validation metric plateaus:

```yaml
training:
  scheduler:
    type: reduce_on_plateau
    patience: 3              # Wait 3 epochs before reducing
    factor: 0.5              # Multiply LR by 0.5
    min_lr: 1.0e-6           # Minimum learning rate
```

**How it works**:
```
Epoch 1: Val loss = 0.50
Epoch 2: Val loss = 0.45 (improved)
Epoch 3: Val loss = 0.44 (improved)
Epoch 4: Val loss = 0.44 (no improvement, patience=1)
Epoch 5: Val loss = 0.44 (no improvement, patience=2)
Epoch 6: Val loss = 0.44 (no improvement, patience=3)
Epoch 7: LR reduced by factor 0.5 (patience reset)
```

#### Cosine Annealing

Gradually reduces LR following cosine curve:

```yaml
training:
  scheduler:
    type: cosine
```

#### Linear Decay

Linearly decreases LR over training:

```yaml
training:
  scheduler:
    type: linear
```

#### Exponential Decay

Exponentially decreases LR:

```yaml
training:
  scheduler:
    type: exponential
```

---

## WarmupScheduler Class

Wrapper for warmup + main scheduler.

### Usage

```python
from training.optimization import WarmupScheduler

scheduler = WarmupScheduler(optimizer, config)

for epoch in range(num_epochs):
    train()
    val_loss = validate()
    
    # Step scheduler (handles warmup automatically)
    scheduler.step(val_loss)
    
    # Get current learning rates
    lrs = scheduler.get_component_lrs()
    print(f"LRs: {lrs}")
```

### Methods

- `step(val_loss)`: Step the scheduler
- `get_last_lr()`: Get last learning rate
- `get_component_lrs()`: Get LR for each component

---

## DifferentialLRScheduler Class

Advanced scheduler with per-component learning rate schedules.

### Usage

```python
from training.optimization import DifferentialLRScheduler

scheduler = DifferentialLRScheduler(optimizer, config)

for epoch in range(num_epochs):
    train()
    scheduler.step()
    
    # Get per-component learning rates
    component_lrs = scheduler.get_component_lrs()
    print(f"RNA-FM LR: {component_lrs['rna_fm']:.2e}")
    print(f"Feature extraction LR: {component_lrs['feature_extraction']:.2e}")
    print(f"Task heads LR: {component_lrs['task_heads']:.2e}")
```

---

## Complete Training Example

```python
import torch
from configs.config_loader import ConfigLoader
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataloader_factory import create_dataloaders
from training.optimization import create_optimizer_and_scheduler
from training.trainer import Trainer

# Setup
config = ConfigLoader('configs/model_config.yaml').config
dataloaders = create_dataloaders(config)

# Initialize model
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)

# Create optimizer and scheduler
optimizer, scheduler = create_optimizer_and_scheduler(model, config)

# Training loop
for epoch in range(config.training.epochs):
    # Train
    train_loss = train_epoch(model, dataloaders['train'], optimizer)
    
    # Validate
    val_loss = validate(model, dataloaders['val'])
    
    # Step scheduler
    scheduler.step(val_loss)
    
    # Log
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Get current learning rates
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('name', f'group_{i}')
        lr = param_group['lr']
        print(f"  {name} LR: {lr:.2e}")
```

---

## Parameter Group Information

### `get_parameter_groups_info(model) -> Dict`

Get information about parameter groups.

```python
from training.optimization import get_parameter_groups_info

info = get_parameter_groups_info(model)

print(f"RNA-FM parameters: {info['rna_fm']:,}")
print(f"Feature extraction parameters: {info['feature_extraction']:,}")
print(f"Task head parameters: {info['task_heads']:,}")
print(f"Total parameters: {sum(info.values()):,}")
```

**Output**:
```
RNA-FM parameters: 640,000
Feature extraction parameters: 1,200,000
Task head parameters: 50,000
Total parameters: 1,890,000
```

---

## Learning Rate Strategies

### Strategy 1: Conservative (Recommended for Fine-tuning)

```yaml
training:
  learning_rate_encoder: 1.0e-4    # Very small for pretrained
  learning_rate_heads: 5.0e-4      # Small for task heads
  warmup_epochs: 5
  scheduler:
    type: reduce_on_plateau
    patience: 5
    factor: 0.5
```

### Strategy 2: Aggressive (For Training from Scratch)

```yaml
training:
  learning_rate_encoder: 1.0e-3    # Larger for new components
  learning_rate_heads: 1.0e-3      # Larger for task heads
  warmup_epochs: 3
  scheduler:
    type: cosine
```

### Strategy 3: Balanced (Default)

```yaml
training:
  learning_rate_encoder: 5.0e-4    # Moderate for pretrained
  learning_rate_heads: 1.0e-3      # Moderate for task heads
  warmup_epochs: 5
  scheduler:
    type: reduce_on_plateau
    patience: 3
    factor: 0.5
```

---

## Troubleshooting

### Issue: "Training diverges (loss becomes NaN)"

**Cause**: Learning rate too high

**Solution**:
```yaml
training:
  learning_rate_encoder: 1.0e-4  # Reduce
  learning_rate_heads: 5.0e-4    # Reduce
```

### Issue: "Training too slow"

**Cause**: Learning rate too low

**Solution**:
```yaml
training:
  learning_rate_encoder: 1.0e-3  # Increase
  learning_rate_heads: 2.0e-3    # Increase
```

### Issue: "Validation loss plateaus"

**Cause**: Scheduler not reducing LR

**Solution**:
```yaml
training:
  scheduler:
    patience: 2        # Reduce patience
    factor: 0.1        # Larger reduction
```

### Issue: "Model overfits"

**Cause**: Learning rate too high or weight decay too low

**Solution**:
```yaml
training:
  learning_rate_heads: 5.0e-4    # Reduce
  weight_decay: 0.05              # Increase
```

---

## Monitoring Learning Rates

### Log Learning Rates

```python
for epoch in range(num_epochs):
    train()
    val_loss = validate()
    scheduler.step(val_loss)
    
    # Log learning rates
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('name', f'group_{i}')
        lr = param_group['lr']
        logger.info(f"Epoch {epoch+1} - {name} LR: {lr:.2e}")
```

### TensorBoard Logging

```python
writer.add_scalar('learning_rate/encoder', encoder_lr, epoch)
writer.add_scalar('learning_rate/heads', heads_lr, epoch)
```

### WandB Logging

```python
wandb.log({
    'learning_rate/encoder': encoder_lr,
    'learning_rate/heads': heads_lr,
    'learning_rate/task_heads': task_heads_lr
})
```

---

## Summary

The optimization module provides:
- ✅ **Differential learning rates** for model components
- ✅ **Linear warmup** for stable training
- ✅ **Multiple scheduler types** (ReduceLROnPlateau, Cosine, Linear, Exponential)
- ✅ **Parameter group management**
- ✅ **Easy integration** with training pipeline
- ✅ **Production-ready** implementation

Perfect for training CRISPR-UniPredict models with optimal convergence!
