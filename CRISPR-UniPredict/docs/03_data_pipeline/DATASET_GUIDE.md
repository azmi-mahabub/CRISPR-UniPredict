# CRISPR Dataset Guide

## Overview

The `CRISPRDataset` class provides efficient PyTorch Dataset functionality for loading and processing CRISPR sequences and labels. It includes:

- **Flexible CSV loading** with support for multiple label types
- **Efficient caching** of encoded sequences for fast loading
- **Task filtering** for on-target and off-target prediction
- **Data augmentation** support (reverse complement, noise)
- **Batch collation** with automatic padding for variable-length sequences
- **Metadata handling** for dataset source, cell line, and other information

---

## CSV Format

### Required Columns
- **sgrna**: sgRNA sequence (string, e.g., "GCTAGCTAGCTAGCTAGCTAGCT")
- **target**: Target sequence (string)

### Optional Columns
- **on_target_score**: On-target efficiency score (float, 0-1)
- **off_target_label**: Off-target label (int, 0 or 1)
- **dataset_source**: Source dataset name (string)
- **cell_line**: Cell line (string)
- Any other metadata columns

### Example CSV
```csv
sgrna,target,on_target_score,off_target_label,dataset_source,cell_line
GCTAGCTAGCTAGCTAGCTAGCT,GCTAGCTAGCTAGCTAGCTAGCT,0.85,0,WT,HEK293T
ATGCATGCATGCATGCATGCATG,ATGCATGCATGCATGCATGCATG,0.72,1,ESP,HEK293T
CCGGCCGGCCGGCCGGCCGGCCG,CCGGCCGGCCGGCCGGCCGGCCG,0.91,,HF,HeLa
```

---

## Basic Usage

### 1. Initialize Dataset

```python
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset

# Create encoder
encoder = SequenceEncoder(device='cpu')

# Load dataset
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    cache_dir='.cache',
    use_cache=True,
    verbose=True
)

print(f"Loaded {len(dataset)} samples")
```

### 2. Get Single Sample

```python
sample = dataset[0]

print(f"sgRNA: {sample['sgrna_sequence']}")
print(f"One-hot shape: {sample['sgrna_onehot'].shape}")  # (23, 4)
print(f"Label shape: {sample['sgrna_label'].shape}")      # (23,)
print(f"On-target score: {sample['on_target_score']}")
print(f"Off-target label: {sample['off_target_label']}")
print(f"Metadata: {sample['metadata']}")
```

### 3. Create DataLoader

```python
from utils.preprocessing.dataset import CRISPRDataLoader

dataloader = CRISPRDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    device='cuda'
)

# Iterate batches
for batch in dataloader:
    sgrna_onehot = batch['sgrna_onehot']      # (batch, seq_len, 4)
    sgrna_label = batch['sgrna_label']        # (batch, seq_len)
    on_target = batch['on_target_score']      # (batch,)
    off_target = batch['off_target_label']    # (batch,)
    
    # Training code here
    pass
```

---

## Advanced Features

### Task Filtering

Filter dataset to specific task type:

```python
# Get indices for each task
task_indices = dataset.get_task_indices()
print(f"On-target samples: {len(task_indices['on_target'])}")
print(f"Off-target samples: {len(task_indices['off_target'])}")
print(f"Both labels: {len(task_indices['both'])}")

# Create filtered dataset
on_target_dataset = dataset.filter_by_task('on_target')
off_target_dataset = dataset.filter_by_task('off_target')
both_dataset = dataset.filter_by_task('both')

print(f"On-target dataset: {len(on_target_dataset)} samples")
```

### Dataset Statistics

```python
stats = dataset.get_statistics()

print(f"Total samples: {stats['total_samples']}")
print(f"On-target samples: {stats['on_target_samples']}")
print(f"Off-target samples: {stats['off_target_samples']}")

# On-target statistics
if 'on_target_stats' in stats:
    print(f"On-target mean: {stats['on_target_stats']['mean']:.3f}")
    print(f"On-target std: {stats['on_target_stats']['std']:.3f}")

# Off-target statistics
if 'off_target_stats' in stats:
    print(f"Off-target positive: {stats['off_target_stats']['positive']}")
    print(f"Off-target negative: {stats['off_target_stats']['negative']}")
    print(f"Positive ratio: {stats['off_target_stats']['positive_ratio']:.3f}")

# Dataset sources
if 'dataset_sources' in stats:
    print(f"Dataset sources: {stats['dataset_sources']}")
```

### Data Augmentation

Enable data augmentation for training:

```python
# Create dataset with augmentation
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    augmentation=True,
    augmentation_types=['reverse_complement', 'noise']
)

# Set to training mode (enables augmentation)
dataset.train()

# Set to evaluation mode (disables augmentation)
dataset.eval()

# Get augmented sample
sample = dataset[0]  # Will be augmented if in training mode
```

### Caching

Caching significantly speeds up data loading:

```python
# Enable caching (default)
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    use_cache=True,
    cache_dir='.cache'
)

# First load: encodes all sequences and saves cache
# Subsequent loads: loads from cache (much faster)

# Disable caching (slower but uses less disk space)
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    use_cache=False
)
```

---

## Sample Structure

Each sample returned by `__getitem__` is a dictionary:

```python
{
    'sgrna_sequence': str,              # Original sgRNA sequence
    'target_sequence': str,             # Original target sequence
    'sgrna_onehot': torch.Tensor,       # (seq_len, 4) one-hot encoded
    'sgrna_label': torch.Tensor,        # (seq_len,) label encoded
    'target_onehot': torch.Tensor,      # (seq_len, 4) one-hot encoded
    'target_label': torch.Tensor,       # (seq_len,) label encoded
    'on_target_score': float or None,   # On-target efficiency score
    'off_target_label': int or None,    # Off-target label (0 or 1)
    'metadata': dict                    # Additional metadata
}
```

---

## Batch Structure

Each batch returned by `CRISPRDataLoader` is a dictionary:

```python
{
    'sgrna_onehot': torch.Tensor,       # (batch, max_seq_len, 4)
    'sgrna_label': torch.Tensor,        # (batch, max_seq_len)
    'target_onehot': torch.Tensor,      # (batch, max_seq_len, 4)
    'target_label': torch.Tensor,       # (batch, max_seq_len)
    'on_target_score': torch.Tensor or None,  # (batch,)
    'off_target_label': torch.Tensor or None, # (batch,)
    'sgrna_sequences': list,            # Original sgRNA sequences
    'target_sequences': list,           # Original target sequences
    'metadata': list                    # List of metadata dicts
}
```

---

## Handling Variable-Length Sequences

The dataset automatically pads sequences to the maximum length in each batch:

```python
# Sequences of different lengths are padded
batch = next(iter(dataloader))

# All sequences in batch have same length (max in batch)
print(batch['sgrna_onehot'].shape)  # (batch_size, max_len, 4)
print(batch['sgrna_label'].shape)   # (batch_size, max_len)

# Padded positions have value 0
# Can be masked out during training if needed
```

---

## Handling Missing Labels

The dataset gracefully handles missing labels:

```python
sample = dataset[0]

# May be None if not in CSV
on_target = sample['on_target_score']  # float or None
off_target = sample['off_target_label']  # int or None

# In batches, None values are converted to 0
batch = next(iter(dataloader))
on_target_batch = batch['on_target_score']  # (batch,), no None values
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.encoding import SequenceEncoder
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = SequenceEncoder(device=device)

# Load datasets
train_dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    augmentation=True,
    augmentation_types=['reverse_complement']
)

val_dataset = CRISPRDataset(
    csv_path='data/val.csv',
    encoder=encoder,
    augmentation=False
)

# Create dataloaders
train_loader = CRISPRDataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    device=device
)

val_loader = CRISPRDataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    device=device
)

# Initialize model
model = CRISPRUniPredict(device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion_on_target = nn.MSELoss()
criterion_off_target = nn.BCELoss()

# Training loop
for epoch in range(100):
    model.train()
    train_dataset.train()
    
    for batch in train_loader:
        # Move to device
        sgrna_onehot = batch['sgrna_onehot'].to(device)
        sgrna_label = batch['sgrna_label'].to(device)
        on_target_label = batch['on_target_score'].to(device)
        off_target_label = batch['off_target_label'].to(device)
        
        # Forward pass
        on_target_pred, off_target_pred = model(
            sgrna_onehot, sgrna_label, task_type='both'
        )
        
        # Compute loss
        loss_on_target = criterion_on_target(on_target_pred, on_target_label.unsqueeze(-1))
        loss_off_target = criterion_off_target(off_target_pred, off_target_label.unsqueeze(-1).float())
        loss = loss_on_target + 0.5 * loss_off_target
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_dataset.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            sgrna_onehot = batch['sgrna_onehot'].to(device)
            sgrna_label = batch['sgrna_label'].to(device)
            
            on_target_pred, off_target_pred = model(
                sgrna_onehot, sgrna_label, task_type='both'
            )
```

---

## Performance Tips

### 1. Use Caching
```python
# Caching makes subsequent loads much faster
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    use_cache=True  # Default: True
)
```

### 2. Adjust Number of Workers
```python
# More workers = faster loading (if CPU allows)
dataloader = CRISPRDataLoader(
    dataset,
    num_workers=4  # Adjust based on CPU cores
)
```

### 3. Pin Memory for GPU
```python
# Faster GPU transfer
dataloader = CRISPRDataLoader(
    dataset,
    pin_memory=True  # Default: True
)
```

### 4. Batch Size
```python
# Larger batches = faster training (if memory allows)
dataloader = CRISPRDataLoader(
    dataset,
    batch_size=128  # Adjust based on GPU memory
)
```

---

## Troubleshooting

### Issue: "CSV file not found"
```python
from pathlib import Path
csv_path = Path('data/train.csv')
assert csv_path.exists(), f"File not found: {csv_path}"
```

### Issue: "Missing required columns"
```python
import pandas as pd
df = pd.read_csv('data/train.csv')
print(df.columns)  # Check column names
```

### Issue: "Out of memory"
```python
# Reduce batch size
dataloader = CRISPRDataLoader(dataset, batch_size=16)

# Disable caching
dataset = CRISPRDataset(csv_path, encoder, use_cache=False)

# Reduce number of workers
dataloader = CRISPRDataLoader(dataset, num_workers=0)
```

### Issue: "Slow data loading"
```python
# Enable caching
dataset = CRISPRDataset(csv_path, encoder, use_cache=True)

# Increase number of workers
dataloader = CRISPRDataLoader(dataset, num_workers=8)

# Pin memory
dataloader = CRISPRDataLoader(dataset, pin_memory=True)
```

---

## API Reference

### CRISPRDataset

```python
class CRISPRDataset(Dataset):
    def __init__(self, csv_path, encoder, cache_dir=None, use_cache=True,
                 augmentation=False, augmentation_types=None, task_type=None,
                 verbose=True)
    
    def __len__() -> int
    def __getitem__(idx) -> Dict
    def get_task_indices() -> Dict[str, List[int]]
    def filter_by_task(task_type) -> CRISPRDataset
    def get_statistics() -> Dict
    def train()
    def eval()
```

### CRISPRDataLoader

```python
class CRISPRDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False,
                 num_workers=0, pin_memory=True, device='cpu')
    
    def __iter__()
    def __len__() -> int
```

---

## Summary

The `CRISPRDataset` provides:
- ✅ Efficient CSV loading and encoding
- ✅ Automatic caching for fast access
- ✅ Task filtering and statistics
- ✅ Data augmentation support
- ✅ Flexible batch collation with padding
- ✅ Metadata handling
- ✅ Production-ready performance

Perfect for training CRISPR prediction models!
