# Custom Collate Functions Guide

## Overview

The `collate.py` module provides custom collate functions for handling variable-length CRISPR sequences with proper padding, masking, and task-aware batching.

---

## Key Functions

### 1. `custom_collate_fn(batch)`

Main collate function for standard batching.

**Purpose**: Combine variable-length sequences into batched tensors with proper padding and masking.

**Input**: List of samples from CRISPRDataset
```python
batch = [
    {
        'sgrna_onehot': (seq_len, 4),
        'sgrna_label': (seq_len,),
        'target_onehot': (seq_len, 4),
        'target_label': (seq_len,),
        'on_target_score': float or None,
        'off_target_label': int or None,
        'sgrna_sequence': str,
        'target_sequence': str,
        'metadata': dict
    },
    ...
]
```

**Output**: Dictionary with batched tensors
```python
{
    'sgrna_onehot': (batch, max_len, 4),
    'sgrna_label': (batch, max_len),
    'target_onehot': (batch, max_len, 4),
    'target_label': (batch, max_len),
    'on_target_score': (batch,),
    'off_target_label': (batch,),
    'on_target_mask': (batch,),           # True where label is valid
    'off_target_mask': (batch,),          # True where label is valid
    'attention_mask': (batch, max_len),   # False for padded positions
    'sequence_lengths': (batch,),         # Original lengths
    'sgrna_sequences': List[str],
    'target_sequences': List[str],
    'metadata': List[dict]
}
```

**Example**:
```python
from utils.preprocessing.collate import custom_collate_fn

batch = [sample1, sample2, sample3]
collated = custom_collate_fn(batch)

print(collated['sgrna_onehot'].shape)      # (3, 23, 4)
print(collated['attention_mask'].shape)    # (3, 23)
print(collated['on_target_mask'])          # [True, False, True]
```

---

### 2. `task_aware_collate_fn(batch)`

Task-aware collation that separates samples by task type.

**Purpose**: Group samples by whether they have on-target, off-target, or both labels.

**Output**: Dictionary with task-separated batches
```python
{
    'on_target_batch': {...},        # Batch with only on-target samples
    'off_target_batch': {...},       # Batch with only off-target samples
    'mixed_batch': {...},            # Batch with both labels
    'task_indices': {
        'on_target': [0, 2, ...],    # Original indices
        'off_target': [1, 3, ...],
        'mixed': [4, 5, ...]
    }
}
```

**Example**:
```python
from utils.preprocessing.collate import task_aware_collate_fn

task_batches = task_aware_collate_fn(batch)

# Train on-target model
if task_batches['on_target_batch'] is not None:
    on_target_batch = task_batches['on_target_batch']
    on_target_pred = model.predict_on_target(
        on_target_batch['sgrna_onehot'],
        on_target_batch['sgrna_label']
    )

# Train off-target model
if task_batches['off_target_batch'] is not None:
    off_target_batch = task_batches['off_target_batch']
    off_target_pred = model.predict_off_target(
        off_target_batch['sgrna_onehot'],
        off_target_batch['sgrna_label']
    )
```

---

### 3. `CRISPRCollator` Class

Stateful collator with configurable behavior.

**Purpose**: Provide a reusable collator with configuration options.

**Parameters**:
- `pad_value_onehot`: Padding value for one-hot sequences (default: 0.0)
- `pad_value_label`: Padding value for label sequences (default: 0)
- `create_causal_mask`: Whether to create causal mask (default: False)
- `task_aware`: Whether to use task-aware collation (default: False)
- `device`: Device to place tensors on (default: 'cpu')

**Example**:
```python
from utils.preprocessing.collate import CRISPRCollator
from utils.preprocessing.dataset import CRISPRDataLoader

# Create collator
collator = CRISPRCollator(
    pad_value_onehot=0.0,
    pad_value_label=0,
    create_causal_mask=False,
    task_aware=False,
    device='cuda'
)

# Use with DataLoader
dataloader = CRISPRDataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator
)

# Iterate batches
for batch in dataloader:
    sgrna_onehot = batch['sgrna_onehot']
    attention_mask = batch['attention_mask']
    on_target_mask = batch['on_target_mask']
```

---

## Helper Functions

### `_pad_onehot_sequences(sequences, max_len, pad_value=0.0)`

Pad one-hot encoded sequences to maximum length.

```python
from utils.preprocessing.collate import _pad_onehot_sequences

sequences = [
    torch.randn(20, 4),
    torch.randn(23, 4),
    torch.randn(21, 4)
]

padded = _pad_onehot_sequences(sequences, max_len=23)
print(padded.shape)  # (3, 23, 4)
```

---

### `_pad_label_sequences(sequences, max_len, pad_value=0)`

Pad label encoded sequences to maximum length.

```python
from utils.preprocessing.collate import _pad_label_sequences

sequences = [
    torch.tensor([2, 3, 4, 5]),
    torch.tensor([2, 3, 4, 5, 2, 3]),
]

padded = _pad_label_sequences(sequences, max_len=6)
print(padded.shape)  # (2, 6)
```

---

### `_create_attention_mask(sequence_lengths, max_len)`

Create attention mask for padded sequences.

```python
from utils.preprocessing.collate import _create_attention_mask

lengths = torch.tensor([20, 23, 21])
mask = _create_attention_mask(lengths, max_len=23)

print(mask.shape)  # (3, 23)
print(mask[0].sum())  # 20 (valid positions)
print(mask[1].sum())  # 23 (valid positions)
```

---

### `create_causal_mask(seq_len, device='cpu')`

Create causal attention mask for autoregressive models.

```python
from utils.preprocessing.collate import create_causal_mask

causal_mask = create_causal_mask(seq_len=5)
print(causal_mask)
# tensor([[True, False, False, False, False],
#         [True, True, False, False, False],
#         [True, True, True, False, False],
#         [True, True, True, True, False],
#         [True, True, True, True, True]])
```

---

### `create_padding_mask(attention_mask)`

Create padding mask ready for broadcasting in attention computation.

```python
from utils.preprocessing.collate import create_padding_mask

attention_mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, True]
])

padding_mask = create_padding_mask(attention_mask)
print(padding_mask.shape)  # (2, 1, 1, 5) - ready for broadcasting
```

---

### `mask_invalid_labels(predictions, mask, fill_value=0.0)`

Mask invalid predictions where labels are missing.

```python
from utils.preprocessing.collate import mask_invalid_labels

predictions = torch.tensor([0.85, 0.72, 0.91])
mask = torch.tensor([True, False, True])

masked = mask_invalid_labels(predictions, mask, fill_value=-1.0)
print(masked)  # tensor([0.85, -1.0, 0.91])
```

---

## Integration with Training

### Using with DataLoader

```python
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader
from utils.preprocessing.collate import CRISPRCollator
from models.encoding import SequenceEncoder

# Setup
encoder = SequenceEncoder(device='cuda')
dataset = CRISPRDataset('data/train.csv', encoder)

# Create collator
collator = CRISPRCollator(device='cuda')

# Create dataloader with custom collate function
dataloader = CRISPRDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    device='cuda'
)

# Override collate function
dataloader.loader.collate_fn = collator

# Iterate batches
for batch in dataloader:
    sgrna_onehot = batch['sgrna_onehot']
    sgrna_label = batch['sgrna_label']
    attention_mask = batch['attention_mask']
    on_target_mask = batch['on_target_mask']
    on_target_score = batch['on_target_score']
    
    # Use attention mask in model
    on_target_pred = model(sgrna_onehot, sgrna_label, attention_mask=attention_mask)
    
    # Only compute loss for valid labels
    valid_pred = on_target_pred[on_target_mask]
    valid_label = on_target_score[on_target_mask]
    loss = criterion(valid_pred, valid_label.unsqueeze(-1))
```

---

### Using Attention Masks in Model

```python
# In your model forward pass
def forward(self, sgrna_onehot, sgrna_label, attention_mask=None):
    # ... process through branches ...
    
    # Use attention mask in self-attention
    if attention_mask is not None:
        # Convert to proper format for attention
        padding_mask = create_padding_mask(attention_mask)
        attended = self.mhsa(branch_a, mask=padding_mask)
    else:
        attended = self.mhsa(branch_a)
    
    # ... rest of forward pass ...
```

---

### Using Task-Aware Batching

```python
from utils.preprocessing.collate import task_aware_collate_fn

# In training loop
for batch in dataloader:
    # Use task-aware collation
    task_batches = task_aware_collate_fn(batch)
    
    # Train on-target task
    if task_batches['on_target_batch'] is not None:
        on_target_batch = task_batches['on_target_batch']
        on_target_pred = model.predict_on_target(
            on_target_batch['sgrna_onehot'],
            on_target_batch['sgrna_label']
        )
        loss_on_target = criterion_on_target(
            on_target_pred,
            on_target_batch['on_target_score'].unsqueeze(-1)
        )
        loss_on_target.backward()
    
    # Train off-target task
    if task_batches['off_target_batch'] is not None:
        off_target_batch = task_batches['off_target_batch']
        off_target_pred = model.predict_off_target(
            off_target_batch['sgrna_onehot'],
            off_target_batch['sgrna_label']
        )
        loss_off_target = criterion_off_target(
            off_target_pred,
            off_target_batch['off_target_label'].unsqueeze(-1).float()
        )
        loss_off_target.backward()
```

---

## Understanding Masks

### Attention Mask

**Purpose**: Indicates which positions are valid (not padded)

**Shape**: `(batch, seq_len)`

**Values**:
- `True`: Valid position (original sequence)
- `False`: Padded position

**Usage**:
```python
attention_mask = batch['attention_mask']  # (batch, seq_len)

# Count valid positions per sample
valid_counts = attention_mask.sum(dim=1)  # (batch,)

# Create padding mask for attention computation
padding_mask = create_padding_mask(attention_mask)  # (batch, 1, 1, seq_len)
```

---

### Label Masks

**Purpose**: Indicates which samples have valid labels for each task

**Shape**: `(batch,)`

**Values**:
- `True`: Sample has valid label
- `False`: Sample has missing label

**Usage**:
```python
on_target_mask = batch['on_target_mask']  # (batch,)
off_target_mask = batch['off_target_mask']  # (batch,)

# Filter predictions to valid samples
valid_on_target_pred = on_target_pred[on_target_mask]
valid_on_target_label = on_target_score[on_target_mask]

# Compute loss only on valid samples
loss = criterion(valid_on_target_pred, valid_on_target_label.unsqueeze(-1))
```

---

## Performance Considerations

### Memory Usage
- Padding overhead: ~5-10% (minimal impact)
- Attention masks: ~1KB per batch
- Label masks: ~100 bytes per batch

### Computation
- Padding: O(batch_size × max_len)
- Masking: O(batch_size × max_len)
- Negligible overhead compared to model computation

### Optimization Tips
1. **Use task-aware batching** for imbalanced datasets
2. **Pre-compute masks** if using same batch multiple times
3. **Use efficient padding** with `torch.nn.utils.rnn.pad_sequence`
4. **Pin memory** for GPU transfer

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.encoding import SequenceEncoder
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader
from utils.preprocessing.collate import CRISPRCollator

# Setup
device = 'cuda'
encoder = SequenceEncoder(device=device)

# Load data
train_dataset = CRISPRDataset('data/train.csv', encoder)
val_dataset = CRISPRDataset('data/val.csv', encoder)

# Create collator
collator = CRISPRCollator(device=device)

# Create dataloaders
train_loader = CRISPRDataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = CRISPRDataLoader(val_dataset, batch_size=64, shuffle=False)

# Override collate functions
train_loader.loader.collate_fn = collator
val_loader.loader.collate_fn = collator

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
        
        # Compute losses only for valid samples
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
```

---

## Troubleshooting

### Issue: Shape mismatch after collation
```python
# Check batch shapes
print(f"sgrna_onehot: {batch['sgrna_onehot'].shape}")
print(f"attention_mask: {batch['attention_mask'].shape}")

# Ensure mask is applied correctly
if batch['attention_mask'].shape[1] != batch['sgrna_onehot'].shape[1]:
    print("ERROR: Mask and sequence length mismatch!")
```

### Issue: Loss computation with missing labels
```python
# Always check mask before computing loss
if mask.any():
    loss = criterion(pred[mask], label[mask])
else:
    print("WARNING: No valid labels in batch!")
    loss = 0
```

### Issue: Memory usage with large batches
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    # ... compute loss ...
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Summary

The collate module provides:
- ✅ Efficient padding for variable-length sequences
- ✅ Proper masking for padded and missing labels
- ✅ Task-aware batching for imbalanced data
- ✅ Causal masking for autoregressive models
- ✅ Production-ready implementation
- ✅ Comprehensive testing

Perfect for training CRISPR-UniPredict models with complex data!
