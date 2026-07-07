# DataLoader Factory Guide

## Overview

The `DataLoaderFactory` provides a centralized way to create configured DataLoaders with all preprocessing components integrated:

- **SequenceEncoder**: Encodes sequences (one-hot and label)
- **CRISPRDataset**: Loads, caches, and manages datasets
- **Custom Samplers**: Handles class imbalance (BootstrapSampler, WeightedRandomSampler)
- **Custom Collate Functions**: Batches with padding and masking

---

## Quick Start

### Basic Usage

```python
from configs.config_loader import ConfigLoader
from utils.preprocessing.dataloader_factory import create_dataloaders

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create dataloaders
dataloaders = create_dataloaders(config)

# Access dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

# Iterate batches
for batch in train_loader:
    sgrna_onehot = batch['sgrna_onehot']
    sgrna_label = batch['sgrna_label']
    on_target_score = batch['on_target_score']
    off_target_label = batch['off_target_label']
    
    # Forward pass
    on_target_pred = model(sgrna_onehot, sgrna_label)
```

### Using DataLoaderFactory Class

```python
from configs.config_loader import ConfigLoader
from utils.preprocessing.dataloader_factory import DataLoaderFactory

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create factory
factory = DataLoaderFactory(config)

# Create dataloaders
dataloaders = factory.create_dataloaders()

# Get statistics
stats = factory.get_dataset_stats()
print(f"Train samples: {stats['train']['total_samples']}")
print(f"Val samples: {stats['val']['total_samples']}")
print(f"Test samples: {stats['test']['total_samples']}")
```

---

## Configuration

The factory reads from configuration sections:

### Data Configuration
```yaml
data:
  train_path: data/processed/combined/train.csv
  val_path: data/processed/combined/val.csv
  test_path: data/processed/combined/test.csv
  num_workers: 4
  pin_memory: true
  
  sampling:
    strategy: balanced  # 'balanced', 'weighted', 'bootstrap'
    on_target_ratio: 0.5
    off_target_ratio: 0.5
  
  augmentation:
    use_augmentation: false
    augmentation_types: []
```

### Training Configuration
```yaml
training:
  batch_size: 32
```

### Inference Configuration
```yaml
inference:
  batch_size: 64
```

### Device Configuration
```yaml
device:
  use_cuda: true
  gpu_ids: [0]
```

---

## DataLoaderFactory Class

### Initialization

```python
from utils.preprocessing.dataloader_factory import DataLoaderFactory

factory = DataLoaderFactory(config)
```

**Validates**:
- Configuration has required fields
- Data files exist
- Device is available

### Methods

#### `create_dataloaders()`

Creates train, val, and test dataloaders.

```python
dataloaders = factory.create_dataloaders()

# Returns:
# {
#     'train': DataLoader,
#     'val': DataLoader,
#     'test': DataLoader
# }
```

**Features**:
- ✅ Loads datasets from config paths
- ✅ Applies BootstrapSampler to train set
- ✅ Uses custom collate function
- ✅ Sets num_workers and pin_memory from config
- ✅ Handles data augmentation for training

#### `get_dataloaders()`

Returns previously created dataloaders.

```python
dataloaders = factory.get_dataloaders()
```

#### `get_dataset_stats()`

Returns statistics for all datasets.

```python
stats = factory.get_dataset_stats()

# Returns:
# {
#     'train': {
#         'total_samples': 1000,
#         'on_target_samples': 800,
#         'off_target_samples': 200,
#         'on_target_stats': {...},
#         'off_target_stats': {...},
#         'dataset_sources': {...}
#     },
#     'val': {...},
#     'test': {...}
# }
```

---

## Convenience Functions

### `create_dataloaders(config)`

Simple function to create dataloaders.

```python
from utils.preprocessing.dataloader_factory import create_dataloaders

dataloaders = create_dataloaders(config)
```

### `create_dataloaders_with_stats(config)`

Create dataloaders and get statistics in one call.

```python
from utils.preprocessing.dataloader_factory import create_dataloaders_with_stats

dataloaders, stats = create_dataloaders_with_stats(config)

print(f"Train samples: {stats['train']['total_samples']}")
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from configs.config_loader import ConfigLoader
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataloader_factory import create_dataloaders

# Load configuration
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

# Create dataloaders
dataloaders = create_dataloaders(config)

# Initialize model
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)

# Initialize optimizer and loss
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate_heads,
    weight_decay=config.training.weight_decay
)

criterion_on = nn.MSELoss()
criterion_off = nn.BCELoss()

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
        
        # Compute losses
        if on_target_mask.any():
            loss_on = criterion_on(
                on_target_pred[on_target_mask],
                on_target_score[on_target_mask].unsqueeze(-1)
            )
        else:
            loss_on = 0
        
        if off_target_mask.any():
            loss_off = criterion_off(
                off_target_pred[off_target_mask],
                off_target_label[off_target_mask].unsqueeze(-1).float()
            )
        else:
            loss_off = 0
        
        loss = (config.training.loss.loss_weights['on_target'] * loss_on +
                config.training.loss.loss_weights['off_target'] * loss_off)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.gradient_clip
        )
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in dataloaders['val']:
            sgrna_onehot = batch['sgrna_onehot'].to(device)
            sgrna_label = batch['sgrna_label'].to(device)
            
            on_target_pred, off_target_pred = model(
                sgrna_onehot, sgrna_label, task_type='both'
            )
            
            # Validation metrics
            pass
```

---

## Batch Structure

Each batch from the dataloaders contains:

```python
{
    'sgrna_onehot': torch.Tensor,           # (batch, max_len, 4)
    'sgrna_label': torch.Tensor,            # (batch, max_len)
    'target_onehot': torch.Tensor,          # (batch, max_len, 4)
    'target_label': torch.Tensor,           # (batch, max_len)
    'on_target_score': torch.Tensor,        # (batch,)
    'off_target_label': torch.Tensor,       # (batch,)
    'on_target_mask': torch.Tensor,         # (batch,) - True where valid
    'off_target_mask': torch.Tensor,        # (batch,) - True where valid
    'attention_mask': torch.Tensor,         # (batch, max_len) - False for padded
    'sequence_lengths': torch.Tensor,       # (batch,) - original lengths
    'sgrna_sequences': List[str],           # Original sequences
    'target_sequences': List[str],          # Original sequences
    'metadata': List[dict]                  # Metadata for each sample
}
```

---

## Sampling Strategies

### Balanced (BootstrapSampler)

Ensures equal representation of on-target and off-target samples.

```yaml
data:
  sampling:
    strategy: balanced
    on_target_ratio: 0.5
    off_target_ratio: 0.5
```

**Best for**: Severe class imbalance (>80/20 split)

### Weighted (WeightedRandomSampler)

Assigns weights based on class frequency.

```yaml
data:
  sampling:
    strategy: weighted
```

**Best for**: Moderate class imbalance (60/40 - 80/20)

### Bootstrap

Alternative balanced sampling strategy.

```yaml
data:
  sampling:
    strategy: bootstrap
```

---

## Data Augmentation

Enable data augmentation for training:

```yaml
data:
  augmentation:
    use_augmentation: true
    augmentation_types:
      - reverse_complement
      - noise
```

**Available augmentations**:
- `reverse_complement`: Reverse complement sequences
- `noise`: Add small noise to one-hot encoding

---

## Performance Optimization

### 1. Adjust Number of Workers

```yaml
data:
  num_workers: 8  # Increase for faster loading
```

**Recommendation**:
- CPU cores / 2 = optimal num_workers
- Start with 4, increase if data loading is bottleneck

### 2. Pin Memory

```yaml
data:
  pin_memory: true  # Enable for GPU training
```

**Faster GPU transfer** but uses more CPU memory.

### 3. Batch Size

```yaml
training:
  batch_size: 64  # Larger batches = faster training
```

**Trade-off**: Larger batches = faster but less frequent updates

### 4. Cache Sequences

Sequences are automatically cached in `.cache/` directory.

First load: Encodes all sequences (5-10s for 10k samples)
Subsequent loads: Loads from cache (<100ms)

---

## Troubleshooting

### Issue: "Data file not found"

**Cause**: CSV paths in config don't exist

**Solution**:
```python
# Check paths
print(config.data.train_path)
print(config.data.val_path)
print(config.data.test_path)

# Update config
config.data.train_path = 'correct/path/train.csv'
```

### Issue: "Missing required config field"

**Cause**: Configuration missing required fields

**Solution**:
```python
# Check configuration
from configs.config_loader import ConfigLoader

loader = ConfigLoader('configs/model_config.yaml')
config = loader.config

# Verify required fields
print(config.data.train_path)
print(config.training.batch_size)
print(config.device.use_cuda)
```

### Issue: "Torch not compiled with CUDA enabled"

**Cause**: CUDA not available but config requests GPU

**Solution**:
```python
# Force CPU
config.device.use_cuda = False

# Or install CPU-only PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Out of memory"

**Cause**: Batch size too large

**Solution**:
```yaml
training:
  batch_size: 16  # Reduce from 32

data:
  num_workers: 2  # Reduce from 4
```

### Issue: "Slow data loading"

**Cause**: Not using cache or too few workers

**Solution**:
```python
# First run: Encodes sequences and creates cache
dataloaders = create_dataloaders(config)

# Subsequent runs: Uses cache (much faster)
dataloaders = create_dataloaders(config)

# Or increase workers
config.data.num_workers = 8
```

---

## API Reference

### DataLoaderFactory

```python
class DataLoaderFactory:
    def __init__(config)
    def create_dataloaders() -> Dict[str, DataLoader]
    def get_dataloaders() -> Dict[str, DataLoader]
    def get_dataset_stats() -> Dict[str, Dict]
```

### Convenience Functions

```python
def create_dataloaders(config) -> Dict[str, DataLoader]
def create_dataloaders_with_stats(config) -> Tuple[Dict[str, DataLoader], Dict[str, Dict]]
```

---

## Summary

The DataLoader Factory provides:
- ✅ **Centralized configuration**: All settings in one place
- ✅ **Integrated preprocessing**: Encoding, caching, sampling, collation
- ✅ **Flexible sampling**: Multiple strategies for class imbalance
- ✅ **Data augmentation**: Optional augmentation for training
- ✅ **Statistics**: Dataset statistics for monitoring
- ✅ **Error handling**: Validation and clear error messages
- ✅ **Production-ready**: Tested and optimized

Perfect for training CRISPR-UniPredict models with minimal setup!
