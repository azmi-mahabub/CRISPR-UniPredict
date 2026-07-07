# CRISPR-UniPredict Configuration Guide

## Overview

The configuration system for CRISPR-UniPredict uses YAML files for easy management of all hyperparameters and settings. The `ConfigLoader` class provides a robust interface for loading, validating, and managing configurations.

---

## Configuration Files

### Primary Configuration File
**Location**: `configs/model_config.yaml`

Contains all hyperparameters organized into logical sections:
- **model**: Architecture and component configurations
- **training**: Training hyperparameters and optimization settings
- **validation**: Validation and early stopping settings
- **data**: Data paths and loading configurations
- **device**: Hardware and device settings
- **logging**: Logging and monitoring configurations
- **inference**: Inference-specific settings
- **model_selection**: Model selection strategy
- **ensemble**: Ensemble learning settings

---

## Configuration Structure

### 1. Model Configuration

#### Encoding
```yaml
encoding:
  max_sequence_length: 23      # sgRNA length in base pairs
  embedding_dim: 128           # Embedding dimension for label encoding
  vocab_size: 6                # Vocabulary size (A, C, G, T, start, padding)
  use_one_hot: true            # Use one-hot encoding for Branch A
  use_label_encoding: true     # Use label encoding for Branches B, C
  use_rna_fm: true             # Use RNA-FM encoder for Branch C
```

#### Multi-Scale Convolution (MSC)
```yaml
msc:
  in_channels: 4               # Input channels (one-hot encoded)
  out_channels: 64             # Output channels per branch
  kernel_sizes: [1, 3, 5, 7]   # Kernel sizes for parallel branches
  dropout: 0.35                # Dropout rate
```

#### Bidirectional GRU
```yaml
bigru:
  hidden_dim: 128              # Hidden dimension
  num_layers: 1                # Number of GRU layers
  dropout: 0.35                # Dropout rate
  bidirectional: true          # Use bidirectional processing
```

#### Multi-Head Self-Attention (MHSA)
```yaml
mhsa:
  embed_dim: 256               # Embedding dimension
  num_heads: 4                 # Number of attention heads
  dropout: 0.35                # Dropout rate
```

#### RNA-FM Encoder
```yaml
rna_fm:
  model_path: models/pretrained/rna_fm_t12.pt  # Path to pretrained model
  freeze_layers: true          # Freeze initial layers
  fine_tune_last_n: 2          # Number of final layers to unfreeze
  embedding_dim: 640           # RNA-FM embedding dimension
```

#### Feature Fusion
```yaml
fusion:
  hidden_dim: 256              # Fusion output dimension
  fusion_method: attention     # 'attention' or 'concat'
  num_branches: 3              # Number of branches to fuse
  dropout: 0.35                # Dropout rate
```

#### Task Heads
```yaml
task_heads:
  shared_dim: 256              # Shared representation dimension
  on_target_hidden: [80, 20]   # Hidden layers for on-target head
  off_target_hidden: [80, 20]  # Hidden layers for off-target head
  dropout: 0.35                # Dropout rate
```

### 2. Training Configuration

#### Basic Settings
```yaml
training:
  batch_size: 32               # Batch size for training
  epochs: 100                  # Number of training epochs
  learning_rate_encoder: 5.0e-4  # Learning rate for encoder branches
  learning_rate_heads: 1.0e-3    # Learning rate for task heads
  optimizer: AdamW             # Optimizer type
  warmup_epochs: 5             # Number of warmup epochs
  weight_decay: 0.01           # L2 regularization
  gradient_clip: 1.0           # Gradient clipping threshold
```

#### Loss Configuration
```yaml
loss:
  on_target_loss: mse          # 'mse' or 'mae'
  off_target_loss: bce         # Binary cross entropy
  loss_weights:
    on_target: 1.0             # Weight for on-target loss
    off_target: 0.5            # Weight for off-target loss
```

#### Learning Rate Scheduler
```yaml
scheduler:
  type: reduce_on_plateau      # 'reduce_on_plateau', 'cosine', 'linear', 'exponential'
  patience: 3                  # Patience for reduce_on_plateau
  factor: 0.5                  # Reduction factor
  min_lr: 1.0e-6               # Minimum learning rate
  warmup_type: linear          # 'linear' or 'constant'
```

### 3. Validation Configuration

```yaml
validation:
  val_frequency: 1             # Validation frequency (epochs)
  early_stopping_patience: 10  # Patience for early stopping
  early_stopping_metric: val_loss  # Metric to monitor
  early_stopping_mode: min     # 'min' or 'max'
```

### 4. Data Configuration

#### Paths
```yaml
data:
  train_path: data/processed/combined/train.csv
  val_path: data/processed/combined/val.csv
  test_path: data/processed/combined/test.csv
  num_workers: 4               # Number of data loading workers
  pin_memory: true             # Pin memory for faster GPU transfer
  prefetch_factor: 2           # Prefetch factor for data loader
```

#### Sampling Strategy
```yaml
sampling:
  strategy: balanced           # 'balanced', 'bootstrap', 'weighted'
  on_target_ratio: 0.5         # Ratio of on-target samples
  off_target_ratio: 0.5        # Ratio of off-target samples
```

#### Data Augmentation
```yaml
augmentation:
  use_augmentation: false      # Enable data augmentation
  augmentation_types: []       # Types of augmentation to apply
```

### 5. Device Configuration

```yaml
device:
  use_cuda: true               # Use GPU if available
  gpu_ids: [0]                 # GPU IDs to use
  mixed_precision: true        # Use automatic mixed precision
  benchmark: true              # Enable cudnn auto-tuner
```

### 6. Logging Configuration

```yaml
logging:
  log_dir: logs                # Directory for logs
  checkpoint_dir: models/checkpoints  # Directory for checkpoints
  use_wandb: true              # Use Weights & Biases
  wandb_project: CRISPR-UniPredict  # W&B project name
  wandb_entity: null           # W&B entity/team
  save_frequency: 1            # Save frequency (epochs)
  print_frequency: 100         # Print frequency (batches)
  log_level: INFO              # Logging level
  metrics_to_log:              # Metrics to log
    - loss
    - on_target_loss
    - off_target_loss
    - on_target_rmse
    - on_target_r2
    - off_target_auc
    - off_target_accuracy
    - learning_rate
```

### 7. Inference Configuration

```yaml
inference:
  batch_size: 64               # Batch size for inference
  return_attention_weights: false  # Return attention weights
  return_branch_outputs: false     # Return branch outputs
  device: cuda                 # Device for inference
```

### 8. Model Selection

```yaml
model_selection:
  strategy: best_val_loss      # 'best_val_loss', 'best_on_target', 'best_off_target'
  save_top_k: 3                # Save top K models
```

### 9. Ensemble Configuration

```yaml
ensemble:
  use_ensemble: false          # Use ensemble learning
  num_models: 5                # Number of models in ensemble
  ensemble_method: average     # 'average', 'voting', 'stacking'
```

---

## Using ConfigLoader

### Basic Usage

```python
from configs.config_loader import ConfigLoader

# Load configuration from YAML file
loader = ConfigLoader('configs/model_config.yaml')
config = loader.config

# Access specific configurations
model_cfg = loader.get_model_config()
training_cfg = loader.get_training_config()
data_cfg = loader.get_data_config()
```

### Accessing Nested Configurations

```python
# Access model architecture settings
print(config.model.msc.out_channels)      # 64
print(config.model.bigru.hidden_dim)      # 128
print(config.model.mhsa.num_heads)        # 4

# Access training settings
print(config.training.batch_size)         # 32
print(config.training.epochs)             # 100
print(config.training.optimizer)          # AdamW

# Access loss configuration
print(config.training.loss.on_target_loss)  # mse
print(config.training.loss.loss_weights)    # {'on_target': 1.0, 'off_target': 0.5}
```

### Updating Configuration

```python
# Update single values
loader.update({
    'training': {
        'batch_size': 64,
        'epochs': 200
    }
})

# Update nested values
loader.update({
    'model': {
        'msc': {
            'out_channels': 128
        }
    }
})
```

### Saving Configuration

```python
# Save as YAML
loader.save('configs/my_config.yaml', format='yaml')

# Save as JSON
loader.save('configs/my_config.json', format='json')
```

### Printing Configuration

```python
# Print entire configuration
loader.print_config()

# Print specific section
loader.print_config('model')
loader.print_config('training')
loader.print_config('data')
```

---

## Configuration Presets

### Preset 1: Quick Training (Small Dataset)

```yaml
training:
  batch_size: 16
  epochs: 20
  learning_rate_encoder: 1.0e-3
  learning_rate_heads: 2.0e-3
  warmup_epochs: 2

validation:
  early_stopping_patience: 5
```

### Preset 2: Standard Training (Medium Dataset)

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate_encoder: 5.0e-4
  learning_rate_heads: 1.0e-3
  warmup_epochs: 5

validation:
  early_stopping_patience: 10
```

### Preset 3: Large-Scale Training (Large Dataset)

```yaml
training:
  batch_size: 128
  epochs: 200
  learning_rate_encoder: 1.0e-4
  learning_rate_heads: 5.0e-4
  warmup_epochs: 10

validation:
  early_stopping_patience: 15
```

### Preset 4: Fine-Tuning (Transfer Learning)

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate_encoder: 1.0e-5  # Very low for pretrained
  learning_rate_heads: 1.0e-3    # Higher for task heads
  warmup_epochs: 2

model:
  rna_fm:
    freeze_layers: true
    fine_tune_last_n: 2
```

---

## Configuration Validation

The `ConfigLoader` automatically validates:
- ✅ Required fields are present
- ✅ Data types are correct
- ✅ Numeric values are in valid ranges
- ✅ File paths are accessible
- ✅ Enum values are valid

### Example Validation

```python
from configs.config_loader import ConfigLoader

try:
    loader = ConfigLoader('configs/model_config.yaml')
    print("Configuration is valid!")
except Exception as e:
    print(f"Configuration error: {e}")
```

---

## Creating Custom Configurations

### Method 1: Modify YAML File

```yaml
# configs/custom_config.yaml
model:
  name: CRISPR-UniPredict-Custom
  version: 1.1
  
  msc:
    out_channels: 128  # Increased from 64
  
  bigru:
    hidden_dim: 256    # Increased from 128

training:
  batch_size: 64       # Increased from 32
  epochs: 150          # Increased from 100
  learning_rate_encoder: 1.0e-4  # Decreased
```

### Method 2: Programmatic Updates

```python
from configs.config_loader import ConfigLoader

# Load base configuration
loader = ConfigLoader('configs/model_config.yaml')

# Make custom updates
loader.update({
    'model': {
        'msc': {'out_channels': 128},
        'bigru': {'hidden_dim': 256}
    },
    'training': {
        'batch_size': 64,
        'epochs': 150
    }
})

# Save custom configuration
loader.save('configs/custom_config.yaml')
```

---

## Integration with Training Script

### Example Training Script

```python
import torch
from configs.config_loader import ConfigLoader
from models.crispr_unipredict import CRISPRUniPredict

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Initialize model with configuration
model = CRISPRUniPredict(
    seq_len=config.model.encoding.max_sequence_length,
    msc_out_channels=config.model.msc.out_channels,
    mhsa_embed_dim=config.model.mhsa.embed_dim,
    bigru_hidden_dim=config.model.bigru.hidden_dim,
    embedding_dim=config.model.encoding.embedding_dim,
    hidden_dim=config.model.fusion.hidden_dim,
    dropout=config.model.msc.dropout,
    device='cuda' if config.device.use_cuda else 'cpu'
)

# Initialize optimizer with configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate_heads,
    weight_decay=config.training.weight_decay
)

# Initialize scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config.training.scheduler.factor,
    patience=config.training.scheduler.patience,
    min_lr=config.training.scheduler.min_lr
)

# Training loop
for epoch in range(config.training.epochs):
    # Training code here
    pass
```

---

## Best Practices

### 1. Version Control
- Keep `configs/model_config.yaml` in version control
- Create separate configs for different experiments
- Document changes in commit messages

### 2. Experiment Tracking
```python
# Save configuration with experiment results
loader.save(f'results/experiment_{experiment_id}/config.yaml')
```

### 3. Hyperparameter Tuning
```python
# Test different configurations
configs_to_test = [
    {'training': {'batch_size': 16}},
    {'training': {'batch_size': 32}},
    {'training': {'batch_size': 64}},
]

for config_update in configs_to_test:
    loader = ConfigLoader('configs/model_config.yaml')
    loader.update(config_update)
    # Train and evaluate
```

### 4. Documentation
```python
# Add comments to custom configurations
"""
Custom configuration for fine-tuning on small dataset
- Reduced batch size: 16
- Reduced learning rate: 1e-4
- Frozen pretrained layers
"""
```

---

## Troubleshooting

### Issue: Configuration file not found
```python
from pathlib import Path
config_path = Path(__file__).parent / 'configs' / 'model_config.yaml'
loader = ConfigLoader(config_path)
```

### Issue: Invalid YAML syntax
- Use proper YAML indentation (2 spaces)
- Quote string values with special characters
- Use `true`/`false` for booleans (not `True`/`False`)

### Issue: Type mismatch
```python
# Ensure types match configuration
config.training.batch_size = 32  # int
config.training.learning_rate_encoder = 5.0e-4  # float
config.device.gpu_ids = [0, 1]  # list
```

---

## Summary

The configuration system provides:
- ✅ Centralized hyperparameter management
- ✅ Easy experimentation and comparison
- ✅ Type-safe configuration objects
- ✅ YAML and JSON support
- ✅ Programmatic updates
- ✅ Configuration validation
- ✅ Integration with training scripts

For more details, see the `ConfigLoader` class in `configs/config_loader.py`.
