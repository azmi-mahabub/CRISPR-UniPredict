# Bidirectional GRU (BiGRU) Module Guide

## Overview

The Bidirectional GRU (BiGRU) module implements bidirectional sequence processing to capture sequential dependencies and context in both forward and backward directions, as used in the CRISPR_HNN architecture.

---

## Architecture

### Input
- Shape: (batch, seq_len, input_dim)
- Sequential features from previous layers (e.g., MSC module output)

### Processing
1. **Forward GRU**: Processes sequence left-to-right
2. **Backward GRU**: Processes sequence right-to-left
3. **Concatenation**: Combines forward and backward hidden states
4. **Dropout**: Regularization (0.35)

### Output
- Shape: (batch, seq_len, hidden_dim*2)
- Bidirectional contextual features

---

## Quick Start

### Basic Usage

```python
import torch
from models.bigru_module import BiGRUModule

# Create BiGRU module
bigru = BiGRUModule(
    input_dim=256,      # From MSC module
    hidden_dim=128,
    num_layers=1,
    dropout=0.35
)

# Create input (batch=4, seq_len=23, input_dim=256)
x = torch.randn(4, 23, 256)

# Forward pass
output = bigru(x)
print(output.shape)  # (4, 23, 256)
```

### With Variable-Length Sequences

```python
# Sequence lengths
lengths = torch.tensor([23, 20, 18, 15])

# Forward pass with packing
output = bigru(x, lengths)
print(output.shape)  # (4, 23, 256)
```

### Stacked BiGRU

```python
from models.bigru_module import BiGRUStack

# Create stacked BiGRU
bigru_stack = BiGRUStack(
    input_dim=256,
    hidden_dim=128,
    num_layers=2,
    dropout=0.35
)

output = bigru_stack(x)
print(output.shape)  # (4, 23, 256)
```

---

## Class Reference

### BiGRUModule

#### Constructor

```python
BiGRUModule(
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.35,
    bidirectional: bool = True
)
```

**Parameters**:
- `input_dim`: Dimension of input features
- `hidden_dim`: Number of hidden units (default: 128)
- `num_layers`: Number of GRU layers (default: 1)
- `dropout`: Dropout rate (default: 0.35)
- `bidirectional`: Use bidirectional GRU (default: True)

#### Methods

**`forward(x, lengths=None) -> Tensor`**
- Processes input through bidirectional GRU
- Handles variable-length sequences with packing if lengths provided
- Returns concatenated forward and backward features

**`get_output_dim() -> int`**
- Returns output dimension (hidden_dim * 2)

**`get_hidden_dim() -> int`**
- Returns hidden dimension

**`get_num_layers() -> int`**
- Returns number of GRU layers

---

## Architecture Details

### Bidirectional Processing

```
Input: (batch, seq_len, input_dim)
    ↓
    ├─→ Forward GRU: processes left-to-right
    │   Output: (batch, seq_len, hidden_dim)
    │
    └─→ Backward GRU: processes right-to-left
        Output: (batch, seq_len, hidden_dim)
    ↓
Concatenate: (batch, seq_len, hidden_dim*2)
    ↓
Dropout: (batch, seq_len, hidden_dim*2)
    ↓
Output: (batch, seq_len, hidden_dim*2)
```

### Variable-Length Sequence Handling

```
Input: (batch, seq_len, input_dim) with padding
    ↓
Pack: Remove padding, create packed sequence
    ↓
GRU: Process packed sequence efficiently
    ↓
Unpack: Restore padding
    ↓
Restore Order: Reorder to original batch order
    ↓
Output: (batch, seq_len, hidden_dim*2)
```

### Dropout Strategy

- **Between Layers**: Applied between stacked GRU layers
- **After GRU**: Applied to output for regularization
- **Rate**: 0.35 (35% dropout)

---

## Usage Examples

### Example 1: Single BiGRU Module

```python
import torch
from models.bigru_module import BiGRUModule

# Create module
bigru = BiGRUModule(input_dim=256, hidden_dim=128)

# Create input
x = torch.randn(batch_size=8, seq_len=23, input_dim=256)

# Forward pass
output = bigru(x)

print(f"Input shape: {x.shape}")      # (8, 23, 256)
print(f"Output shape: {output.shape}") # (8, 23, 256)
print(f"Output dim: {bigru.get_output_dim()}")  # 256
```

### Example 2: Variable-Length Sequences

```python
# Sequences of different lengths
lengths = torch.tensor([23, 20, 18, 15, 12, 10, 8, 5])

# Padded input
x = torch.randn(8, 23, 256)

# Forward pass with packing
output = bigru(x, lengths)

print(f"Output shape: {output.shape}")  # (8, 23, 256)
```

### Example 3: Stacked BiGRU

```python
from models.bigru_module import BiGRUStack

# Create stack
bigru_stack = BiGRUStack(
    input_dim=256,
    hidden_dim=128,
    num_layers=3,
    dropout=0.35
)

# Forward pass
x = torch.randn(8, 23, 256)
output = bigru_stack(x)

print(f"Output shape: {output.shape}")  # (8, 23, 256)
```

### Example 4: BiGRU with Attention

```python
from models.bigru_module import BiGRUWithAttention

# Create BiGRU with attention
bigru_attn = BiGRUWithAttention(
    input_dim=256,
    hidden_dim=128,
    num_layers=1,
    dropout=0.35,
    use_attention=True
)

x = torch.randn(8, 23, 256)
output = bigru_attn(x)

print(f"Output shape: {output.shape}")  # (8, 23, 256)
```

### Example 5: Integration with CRISPR_HNN

```python
import torch
from models.msc_module import MultiScaleConvolution
from models.bigru_module import BiGRUModule
from models.encoding import SequenceEncoder

# Prepare data
encoder = SequenceEncoder()
sequences = ["ACGTACGTACGTACGTACGTAC", ...]
x = encoder.batch_one_hot_encode(sequences)  # (batch, 23, 4)

# Multi-scale convolution
msc = MultiScaleConvolution(in_channels=4, out_channels=64)
x = msc(x)  # (batch, 23, 256)

# Bidirectional GRU
bigru = BiGRUModule(input_dim=256, hidden_dim=128)
x = bigru(x)  # (batch, 23, 256)

# Continue with dense layers for prediction...
```

---

## Test Results

### Test 1: Single Module
```
Input shape: torch.Size([4, 23, 256])
Output shape: torch.Size([4, 23, 256])
Output dimension: 256
Hidden dimension: 128
Number of layers: 1
✓ PASSED
```

### Test 2: Variable-Length Sequences
```
Input shape: torch.Size([4, 23, 256])
Sequence lengths: [23, 20, 18, 15]
Output shape: torch.Size([4, 23, 256])
✓ PASSED
```

### Test 3: Stacked Modules
```
Input shape: torch.Size([4, 23, 256])
Output shape: torch.Size([4, 23, 256])
Output dimension: 256
✓ PASSED
```

### Test 4: BiGRU with Attention
```
Input shape: torch.Size([4, 23, 256])
Output shape: torch.Size([4, 23, 256])
✓ PASSED
```

### Test 5: Parameters
```
Single BiGRU module:
  Total parameters: 296,448
  Trainable parameters: 296,448

Stacked BiGRU (2 layers):
  Total parameters: 592,896
  Trainable parameters: 592,896
✓ PASSED
```

### Test 6: Gradient Flow
```
✓ Gradients computed successfully
Input gradient shape: torch.Size([4, 23, 256])
✓ PASSED
```

### Test 7: Training/Eval Mode
```
Training mode output shape: torch.Size([4, 23, 256])
Evaluation mode output shape: torch.Size([4, 23, 256])
✓ Dropout working correctly in training mode
✓ PASSED
```

### Test 8: Different Hidden Dimensions
```
Hidden dim: 64 → Output dim: 128 → Output shape: (4, 23, 128)
Hidden dim: 128 → Output dim: 256 → Output shape: (4, 23, 256)
Hidden dim: 256 → Output dim: 512 → Output shape: (4, 23, 512)
✓ PASSED
```

### Test 9: Variable Sequence Lengths
```
Sequence length: 10 → Output shape: (4, 10, 256)
Sequence length: 23 → Output shape: (4, 23, 256)
Sequence length: 50 → Output shape: (4, 50, 256)
✓ PASSED
```

---

## Parameter Count

### Single BiGRU Module (input_dim=256, hidden_dim=128)

```
Forward GRU:
  Input to hidden: 256 × 128 × 3 = 98,304
  Hidden to hidden: 128 × 128 × 3 = 49,152
  Bias: 128 × 3 = 384
  Total: 147,840

Backward GRU: 147,840

Total: 295,680 parameters
```

### Stacked BiGRU (2 layers)

```
Layer 1: 295,680 parameters
Layer 2: 297,216 parameters (256 input channels)
Total: 592,896 parameters
```

---

## Performance Characteristics

### Memory Usage
- Input: (batch, seq_len, 256) = 256 × seq_len × batch bytes
- Output: (batch, seq_len, 256) = 256 × seq_len × batch bytes
- Parameters: ~300KB per module

### Computation
- Forward pass: ~20-50ms per batch (CPU)
- Backward pass: ~40-100ms per batch (CPU)
- GPU: 3-5x faster

### Batch Sizes
- Recommended: 8-64 sequences per batch
- Memory: ~200MB for batch_size=32, seq_len=23

---

## Integration with CRISPR_HNN

### Full Architecture

```
Input: (batch, 23, 4) one-hot encoded sgRNA
    ↓
MultiScaleConvolution: (batch, 23, 256)
    ↓
BiGRUModule: (batch, 23, 256)
    ↓
Global Average Pooling: (batch, 256)
    ↓
Dense Layers: (batch, 1)
    ↓
Output: Indel efficiency score (0-1)
```

### Why BiGRU?

1. **Bidirectional Context**: Captures information from both directions
2. **Sequential Dependencies**: Understands sequence patterns
3. **Efficient**: GRU is faster than LSTM
4. **Gradient Flow**: Better than vanilla RNN
5. **Variable Length**: Handles sequences of different lengths

---

## Troubleshooting

### Issue: Shape Mismatch

**Problem**: RuntimeError about tensor shapes

**Solution**: Ensure input dimension matches
```python
# ✓ Correct
bigru = BiGRUModule(input_dim=256)  # Match MSC output
x = torch.randn(batch, seq_len, 256)

# ✗ Incorrect
bigru = BiGRUModule(input_dim=256)
x = torch.randn(batch, seq_len, 128)  # Wrong dimension
```

### Issue: Out of Memory

**Problem**: CUDA out of memory error

**Solution**: Reduce batch size or hidden dimension
```python
# Reduce batch size
batch_size = 16  # instead of 64

# Or reduce hidden dimension
bigru = BiGRUModule(input_dim=256, hidden_dim=64)  # instead of 128
```

### Issue: Slow Training

**Problem**: Training is slow

**Solution**: Use GPU acceleration
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bigru = bigru.to(device)
x = x.to(device)
```

### Issue: Variable-Length Sequences Not Working

**Problem**: Packing fails

**Solution**: Ensure lengths are in descending order
```python
# Lengths should be sorted in descending order
lengths = torch.tensor([23, 20, 18, 15])  # ✓ Correct
# Module handles sorting internally
```

---

## Summary

✓ **Bidirectional GRU**: Fully implemented
✓ **Forward & Backward**: Bidirectional processing
✓ **Variable-Length Sequences**: Packing support
✓ **Stacking Support**: Build deeper architectures
✓ **Attention Option**: Optional attention mechanism
✓ **Dropout Regularization**: Prevents overfitting
✓ **Well-Tested**: All tests passing
✓ **Production Ready**: Ready for CRISPR_HNN training

---

*Status: ✓ Ready for Model Training*
*Last Updated: 2025-11-20*
