# Trainer Guide

## Overview

The `Trainer` class handles the complete training pipeline for CRISPR-UniPredict:

- Training and validation loops
- Checkpoint saving/loading
- Logging (TensorBoard, WandB)
- Mixed precision training
- Gradient clipping
- Early stopping
- Learning rate scheduling

---

## Quick Start

### Basic Training

```python
from configs.config_loader import ConfigLoader
from utils.preprocessing.dataloader_factory import create_dataloaders
from models.crispr_unipredict import CRISPRUniPredict
from training.trainer import Trainer

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create dataloaders
dataloaders = create_dataloaders(config)

# Initialize model
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)

# Create trainer
trainer = Trainer(model, dataloaders, config)

# Train
history = trainer.train()
```

### Resume Training

```python
# Resume from checkpoint
trainer = Trainer(
    model, 
    dataloaders, 
    config,
    resume_from='models/checkpoints/latest.pt'
)

history = trainer.train()
```

---

## Trainer Class

### Initialization

```python
trainer = Trainer(
    model=model,
    dataloaders=dataloaders,
    config=config,
    resume_from=None  # Optional checkpoint path
)
```

**Parameters**:
- `model`: CRISPRUniPredict instance
- `dataloaders`: Dict with 'train', 'val', 'test' dataloaders
- `config`: Configuration object
- `resume_from`: Path to checkpoint to resume from (optional)

### Core Methods

#### `train_epoch() -> Dict[str, float]`

Train for one epoch.

**Returns**: Dictionary with epoch metrics

```python
metrics = trainer.train_epoch()
# {
#     'total_loss': 0.45,
#     'on_target_loss': 0.30,
#     'off_target_loss': 0.15,
#     'batch_count': 100
# }
```

#### `validate() -> Dict[str, float]`

Validate on validation set.

**Returns**: Dictionary with validation metrics

```python
val_metrics = trainer.validate()
# {
#     'total_loss': 0.42,
#     'on_target_loss': 0.28,
#     'off_target_loss': 0.14,
#     'on_target_spearman_r': 0.92,
#     'off_target_auroc': 0.88,
#     ...
# }
```

#### `train() -> Dict`

Main training loop for all epochs.

**Returns**: Complete training history

```python
history = trainer.train()
# {
#     'train': [
#         {'total_loss': 0.50, 'on_target_loss': 0.35, ...},
#         {'total_loss': 0.45, 'on_target_loss': 0.30, ...},
#         ...
#     ],
#     'val': [
#         {'total_loss': 0.48, 'on_target_loss': 0.33, ...},
#         {'total_loss': 0.42, 'on_target_loss': 0.28, ...},
#         ...
#     ]
# }
```

#### `save_checkpoint(epoch, is_best=False)`

Save checkpoint.

```python
trainer.save_checkpoint(epoch=10, is_best=True)
```

**Saves**:
- `models/checkpoints/latest.pt` - Latest checkpoint
- `models/checkpoints/best.pt` - Best model (if is_best=True)
- `models/checkpoints/checkpoint_epoch_10.pt` - Periodic checkpoint

#### `load_checkpoint(checkpoint_path)`

Load checkpoint to resume training.

```python
trainer.load_checkpoint('models/checkpoints/best.pt')
```

#### `get_training_history() -> Dict`

Get complete training history.

```python
history = trainer.get_training_history()
```

#### `save_training_history(path)`

Save training history to JSON.

```python
trainer.save_training_history('results/training_history.json')
```

---

## Features

### Mixed Precision Training

Automatically enabled if configured:

```yaml
device:
  mixed_precision: true
```

**Benefits**:
- ✅ Faster training
- ✅ Reduced memory usage
- ✅ Maintained accuracy

**How it works**:
- Forward pass in float16
- Loss computation in float32
- Backward pass with gradient scaling

### Gradient Clipping

Prevents gradient explosion:

```yaml
training:
  gradient_clip: 1.0
```

**Applied to**: All model parameters

### Early Stopping

Stops training if validation metric doesn't improve:

```yaml
validation:
  early_stopping_patience: 10
  early_stopping_metric: val_loss
```

**How it works**:
- Monitor validation loss
- Stop if no improvement for N epochs
- Save best model

### Learning Rate Scheduling

Adjusts learning rate during training:

```yaml
training:
  scheduler:
    type: reduce_on_plateau
    patience: 3
    factor: 0.5
    min_lr: 1.0e-6
```

**Supported types**:
- `reduce_on_plateau`: Reduce LR when metric plateaus
- `cosine`: Cosine annealing
- `linear`: Linear decay
- `exponential`: Exponential decay

### Logging

#### TensorBoard

Automatically enabled:

```bash
tensorboard --logdir=logs
```

**Logged metrics**:
- Training loss per batch
- Validation loss per epoch
- On-target and off-target losses
- Learning rate

#### Weights & Biases (WandB)

Enable in config:

```yaml
logging:
  use_wandb: true
  wandb_project: CRISPR-UniPredict
  wandb_entity: your_entity
```

**Logged metrics**:
- All TensorBoard metrics
- Model architecture
- Configuration
- Training history

---

## Complete Training Example

```python
import torch
from configs.config_loader import ConfigLoader
from utils.preprocessing.dataloader_factory import create_dataloaders
from models.crispr_unipredict import CRISPRUniPredict
from training.trainer import Trainer

# Setup
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create dataloaders
dataloaders = create_dataloaders(config)

# Initialize model
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)

# Create trainer
trainer = Trainer(model, dataloaders, config)

# Train
print("Starting training...")
history = trainer.train()

# Save history
trainer.save_training_history('results/training_history.json')

print("Training completed!")
print(f"Best validation loss: {trainer.best_val_loss:.6f}")
print(f"Best model saved at: {trainer.best_model_path}")
```

---

## Checkpoint Structure

### Checkpoint File

```python
{
    'epoch': 10,
    'model_state_dict': {...},
    'optimizer_encoder_state_dict': {...},
    'optimizer_heads_state_dict': {...},
    'training_history': {...},
    'best_val_loss': 0.42,
    'config': {...}
}
```

### Checkpoint Files

- `latest.pt`: Latest checkpoint (always updated)
- `best.pt`: Best model so far
- `checkpoint_epoch_N.pt`: Periodic checkpoints

### Loading Checkpoint

```python
# Resume training
trainer = Trainer(model, dataloaders, config, resume_from='models/checkpoints/best.pt')
history = trainer.train()

# Or load manually
checkpoint = torch.load('models/checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Training Configuration

### Key Settings

```yaml
training:
  batch_size: 32              # Batch size
  epochs: 100                 # Number of epochs
  learning_rate_encoder: 5.0e-4  # Encoder LR
  learning_rate_heads: 1.0e-3    # Heads LR
  optimizer: AdamW            # Optimizer
  warmup_epochs: 5            # Warmup epochs
  weight_decay: 0.01          # L2 regularization
  gradient_clip: 1.0          # Gradient clipping

validation:
  val_frequency: 1            # Validate every N epochs
  early_stopping_patience: 10 # Early stopping patience
  early_stopping_metric: val_loss

logging:
  log_dir: logs               # Log directory
  checkpoint_dir: models/checkpoints  # Checkpoint directory
  use_wandb: true             # Use WandB
  save_frequency: 1           # Save checkpoint every N epochs
  print_frequency: 100        # Print metrics every N batches
```

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Open browser at http://localhost:6006
```

**View**:
- Training and validation loss
- Per-batch metrics
- Learning rate changes

### WandB

```bash
# View at https://wandb.ai/your_entity/CRISPR-UniPredict
```

**View**:
- All metrics
- Model comparison
- Hyperparameter sweep results

### Console Output

```
Epoch 1/100 - Train Loss: 0.523456, Val Loss: 0.512345
Epoch 2/100 - Train Loss: 0.456789, Val Loss: 0.445678
...
Best model saved at epoch 15
```

---

## Troubleshooting

### Issue: "Out of memory"

**Cause**: Batch size too large

**Solution**:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

Or disable mixed precision:
```yaml
device:
  mixed_precision: false
```

### Issue: "NaN loss"

**Cause**: Gradient explosion or invalid values

**Solution**:
```yaml
training:
  gradient_clip: 0.5  # Reduce clipping threshold
  learning_rate_heads: 5.0e-4  # Reduce learning rate
```

### Issue: "Training too slow"

**Cause**: Mixed precision disabled or too many workers

**Solution**:
```yaml
device:
  mixed_precision: true

data:
  num_workers: 8  # Increase workers
```

### Issue: "Model not improving"

**Cause**: Learning rate too high or too low

**Solution**:
```yaml
training:
  learning_rate_heads: 1.0e-3  # Adjust learning rate
  
training:
  scheduler:
    type: reduce_on_plateau
    patience: 5  # Reduce patience
```

---

## Advanced Usage

### Custom Optimizer

```python
# Modify _init_optimizers() in Trainer class
if self.config.training.optimizer == 'SGD':
    self.optimizer_heads = optim.SGD(
        head_params,
        lr=self.config.training.learning_rate_heads,
        momentum=0.9
    )
```

### Custom Scheduler

```python
# Modify _init_scheduler() in Trainer class
if scheduler_type == 'warmup_cosine':
    self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer_heads,
        lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0)
    )
```

### Custom Logging

```python
# Add to train_epoch() or validate()
if self.writer:
    self.writer.add_histogram('gradients', model_grad, epoch)
    self.writer.add_image('attention_weights', attention_map, epoch)
```

---

## Summary

The Trainer class provides:
- ✅ Complete training pipeline
- ✅ Mixed precision training
- ✅ Gradient clipping
- ✅ Early stopping
- ✅ Checkpoint management
- ✅ TensorBoard logging
- ✅ WandB integration
- ✅ Learning rate scheduling
- ✅ Production-ready

Perfect for training CRISPR-UniPredict models!
