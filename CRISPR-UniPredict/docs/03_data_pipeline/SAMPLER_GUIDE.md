# Custom Sampler Guide

## Overview

The `sampler.py` module provides custom samplers for handling class imbalance and task-aware sampling in CRISPR datasets. This is particularly important for datasets like CCLMoff where off-target samples are much rarer than on-target samples.

---

## Problem: Class Imbalance

### The Challenge

CRISPR datasets often have severe class imbalance:
- **On-target samples**: 80-90% of dataset
- **Off-target samples**: 10-20% of dataset

This causes problems:
- Model biased towards majority class
- Minority class underrepresented in training
- Poor performance on minority class

### The Solution

Use task-aware samplers to ensure balanced representation in each batch.

---

## Samplers

### 1. BootstrapSampler

**Purpose**: Ensure equal representation of on-target and off-target samples in each batch.

**Strategy**:
1. Separate dataset indices by task type
2. For each batch:
   - Sample specified ratio from on-target indices
   - Sample remaining from off-target indices
   - Use replacement if one task has fewer samples
   - Shuffle combined samples
3. Total batches determined by smaller task

**Usage**:
```python
from torch.utils.data import DataLoader
from utils.preprocessing.dataset import CRISPRDataset
from utils.preprocessing.sampler import BootstrapSampler

# Load dataset
dataset = CRISPRDataset('data/train.csv', encoder)

# Create sampler
sampler = BootstrapSampler(
    dataset,
    batch_size=32,
    on_target_ratio=0.5,      # 50% on-target, 50% off-target
    drop_last=False,
    shuffle=True,
    seed=42
)

# Create dataloader with batch sampler
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4
)

# Iterate batches
for batch in dataloader:
    # Each batch has ~50% on-target and ~50% off-target samples
    on_target_count = batch['on_target_mask'].sum().item()
    off_target_count = batch['off_target_mask'].sum().item()
    print(f"On-target: {on_target_count}, Off-target: {off_target_count}")
```

**Parameters**:
- `dataset`: CRISPRDataset instance
- `batch_size`: Batch size (default: 32)
- `on_target_ratio`: Ratio of on-target samples (default: 0.5)
- `off_target_ratio`: Ratio of off-target samples (auto-calculated)
- `drop_last`: Drop last incomplete batch (default: False)
- `shuffle`: Shuffle indices (default: True)
- `seed`: Random seed (default: 42)

**Advantages**:
- ✅ Ensures balanced representation
- ✅ Handles severe class imbalance
- ✅ Sampling with replacement for minority class
- ✅ Reproducible with seed
- ✅ Works with mixed samples (both labels)

**When to Use**:
- Severe class imbalance (>80/20 split)
- Want equal representation of both tasks
- Training multi-task models

---

### 2. WeightedRandomSampler

**Purpose**: Handle class imbalance by assigning weights to samples.

**Strategy**:
1. Calculate class weights (inverse of class frequency)
2. Assign weight to each sample based on its class
3. Sample with replacement using these weights

**Usage**:
```python
from torch.utils.data import DataLoader
from utils.preprocessing.dataset import CRISPRDataset
from utils.preprocessing.sampler import WeightedRandomSampler

# Load dataset
dataset = CRISPRDataset('data/train.csv', encoder)

# Create sampler
sampler = WeightedRandomSampler(
    dataset,
    num_samples=len(dataset) * 2,  # Oversample
    replacement=True,
    seed=42
)

# Create dataloader with sampler
dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=32,
    num_workers=4
)

# Iterate batches
for batch in dataloader:
    # Minority class samples appear more frequently
    pass
```

**Parameters**:
- `dataset`: CRISPRDataset instance
- `num_samples`: Number of samples to draw (default: len(dataset))
- `replacement`: Sample with replacement (default: True)
- `seed`: Random seed (default: 42)

**Advantages**:
- ✅ Smooth weighting of classes
- ✅ Flexible number of samples
- ✅ Works with any number of classes
- ✅ Standard PyTorch approach

**When to Use**:
- Moderate class imbalance
- Want smooth weighting instead of strict ratios
- Need flexibility in number of samples

---

### 3. StratifiedSampler

**Purpose**: Ensure balanced sampling across strata (e.g., dataset sources).

**Strategy**:
1. Separate indices by strata (metadata key)
2. For each batch:
   - Sample from each stratum
   - Shuffle combined samples
3. Ensures representation from all strata

**Usage**:
```python
from torch.utils.data import DataLoader
from utils.preprocessing.dataset import CRISPRDataset
from utils.preprocessing.sampler import StratifiedSampler

# Load dataset
dataset = CRISPRDataset('data/train.csv', encoder)

# Create sampler
sampler = StratifiedSampler(
    dataset,
    batch_size=32,
    strata_key='dataset_source',  # Stratify by dataset source
    drop_last=False,
    shuffle=True,
    seed=42
)

# Create dataloader with batch sampler
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4
)

# Iterate batches
for batch in dataloader:
    # Each batch has samples from all dataset sources
    sources = [sample['metadata']['dataset_source'] for sample in batch]
    print(f"Sources in batch: {set(sources)}")
```

**Parameters**:
- `dataset`: CRISPRDataset instance
- `batch_size`: Batch size
- `strata_key`: Metadata key to stratify by (default: 'dataset_source')
- `drop_last`: Drop last incomplete batch (default: False)
- `shuffle`: Shuffle indices (default: True)
- `seed`: Random seed (default: 42)

**Advantages**:
- ✅ Ensures representation from all strata
- ✅ Handles multiple data sources
- ✅ Reduces dataset bias
- ✅ Improves generalization

**When to Use**:
- Multiple dataset sources
- Want balanced representation across sources
- Reduce dataset-specific bias

---

## Comparison

| Feature | Bootstrap | Weighted | Stratified |
|---------|-----------|----------|-----------|
| **Class Balance** | Strict 50/50 | Smooth weights | By strata |
| **Replacement** | Yes (minority) | Yes | No |
| **Flexibility** | Fixed ratio | Variable samples | By strata |
| **Complexity** | Medium | Low | Medium |
| **Best For** | Severe imbalance | Moderate imbalance | Multi-source data |

---

## Integration with Training

### Complete Training Example with BootstrapSampler

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.encoding import SequenceEncoder
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader
from utils.preprocessing.sampler import BootstrapSampler
from utils.preprocessing.collate import CRISPRCollator

# Setup
device = 'cuda'
encoder = SequenceEncoder(device=device)

# Load datasets
train_dataset = CRISPRDataset('data/train.csv', encoder)
val_dataset = CRISPRDataset('data/val.csv', encoder)

# Create samplers
train_sampler = BootstrapSampler(
    train_dataset,
    batch_size=32,
    on_target_ratio=0.5,
    shuffle=True,
    seed=42
)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=CRISPRCollator(device=device),
    num_workers=4
)

val_loader = CRISPRDataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    device=device
)

# Initialize model
model = CRISPRUniPredict(device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion_on = nn.MSELoss()
criterion_off = nn.BCELoss()

# Training loop
for epoch in range(100):
    model.train()
    
    for batch in train_loader:
        sgrna_onehot = batch['sgrna_onehot']
        sgrna_label = batch['sgrna_label']
        on_target_score = batch['on_target_score']
        off_target_label = batch['off_target_label']
        on_target_mask = batch['on_target_mask']
        off_target_mask = batch['off_target_mask']
        
        # Forward pass
        on_target_pred, off_target_pred = model(
            sgrna_onehot, sgrna_label, task_type='both'
        )
        
        # Compute losses with masking
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
        
        loss = loss_on + 0.5 * loss_off
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Validation code
            pass
```

---

## Handling Different Ratios

### 70/30 Split (More On-Target)

```python
sampler = BootstrapSampler(
    dataset,
    batch_size=32,
    on_target_ratio=0.7,      # 70% on-target
    off_target_ratio=0.3      # 30% off-target
)
```

### 30/70 Split (More Off-Target)

```python
sampler = BootstrapSampler(
    dataset,
    batch_size=32,
    on_target_ratio=0.3,      # 30% on-target
    off_target_ratio=0.7      # 70% off-target
)
```

### Custom Batch Composition

```python
batch_size = 32
on_target_per_batch = 24  # 75%
off_target_per_batch = 8   # 25%

# Adjust batch size to match desired ratio
sampler = BootstrapSampler(
    dataset,
    batch_size=32,
    on_target_ratio=24/32,
    off_target_ratio=8/32
)
```

---

## Performance Tips

### 1. Use Appropriate Sampler

```python
# For severe imbalance (>80/20)
sampler = BootstrapSampler(dataset, batch_size=32)

# For moderate imbalance (60/40 - 80/20)
sampler = WeightedRandomSampler(dataset, num_samples=len(dataset)*2)

# For multi-source data
sampler = StratifiedSampler(dataset, batch_size=32)
```

### 2. Combine with Collate Function

```python
from utils.preprocessing.collate import CRISPRCollator

collator = CRISPRCollator(device='cuda')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collator,
    num_workers=4
)
```

### 3. Monitor Class Distribution

```python
# Check actual distribution in batches
on_target_counts = []
off_target_counts = []

for batch in dataloader:
    on_target_counts.append(batch['on_target_mask'].sum().item())
    off_target_counts.append(batch['off_target_mask'].sum().item())

print(f"Mean on-target per batch: {np.mean(on_target_counts):.2f}")
print(f"Mean off-target per batch: {np.mean(off_target_counts):.2f}")
```

---

## Troubleshooting

### Issue: "Dataset must have both on-target and off-target samples"

**Cause**: Dataset doesn't have samples for both tasks

**Solution**:
```python
# Check dataset composition
stats = dataset.get_statistics()
print(f"On-target: {stats['on_target_samples']}")
print(f"Off-target: {stats['off_target_samples']}")

# If missing, filter dataset
on_target_dataset = dataset.filter_by_task('on_target')
off_target_dataset = dataset.filter_by_task('off_target')
```

### Issue: Unbalanced batches despite using BootstrapSampler

**Cause**: Batch size not divisible by number of tasks

**Solution**:
```python
# Use batch size divisible by 2
batch_size = 32  # 16 on-target, 16 off-target
# or
batch_size = 64  # 32 on-target, 32 off-target
```

### Issue: Memory usage with WeightedRandomSampler

**Cause**: Oversampling creates many duplicates

**Solution**:
```python
# Reduce num_samples
sampler = WeightedRandomSampler(
    dataset,
    num_samples=len(dataset),  # Don't oversample
    replacement=True
)
```

---

## Summary

The sampler module provides:
- ✅ **BootstrapSampler**: Strict balanced sampling (50/50 or custom ratio)
- ✅ **WeightedRandomSampler**: Smooth weighted sampling
- ✅ **StratifiedSampler**: Balanced sampling by strata
- ✅ **Handles severe class imbalance**: Sampling with replacement
- ✅ **Reproducible**: Seed-based randomization
- ✅ **Production-ready**: Tested and optimized

Perfect for training CRISPR models on imbalanced datasets!
