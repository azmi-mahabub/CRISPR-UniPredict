# Sequence Encoding Guide

## Overview

The `SequenceEncoder` class provides efficient encoding of DNA/RNA sequences for model training, implementing two encoding schemes from the CRISPR_HNN paper.

---

## Quick Start

### Basic Usage

```python
from models.encoding import SequenceEncoder

# Create encoder
encoder = SequenceEncoder(device='cpu')

# One-hot encode a sequence
sequence = "ACGTACGTACGTACGTACGTAC"
one_hot = encoder.one_hot_encode(sequence)
print(f"Shape: {one_hot.shape}")  # (23, 4)

# Label encode a sequence
label_encoded = encoder.label_encode(sequence, add_start_token=True)
print(f"Shape: {label_encoded.shape}")  # (24,)
```

---

## Encoding Methods

### 1. One-Hot Encoding

Converts each nucleotide to a 4-dimensional binary vector.

**Encoding Scheme**:
- A = [1, 0, 0, 0]
- C = [0, 1, 0, 0]
- G = [0, 0, 1, 0]
- T = [0, 0, 0, 1]

**Output Shape**: (sequence_length, 4)

**Example**:
```python
encoder = SequenceEncoder()

# Single sequence
sequence = "ACGT"
encoded = encoder.one_hot_encode(sequence)
print(encoded.shape)  # (4, 4)
print(encoded)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])

# Decode back
decoded = encoder.decode_one_hot(encoded)
print(decoded)  # "ACGT"
```

### 2. Label Encoding

Converts each nucleotide to an integer token.

**Encoding Scheme**:
- Start Token = 1
- A = 2
- C = 3
- G = 4
- T = 5
- Padding = 0

**Output Shape**: (sequence_length,) or (sequence_length + 1,) with start token

**Example**:
```python
encoder = SequenceEncoder()

# Single sequence with start token
sequence = "ACGT"
encoded = encoder.label_encode(sequence, add_start_token=True)
print(encoded.shape)  # (5,)
print(encoded)  # tensor([1, 2, 3, 4, 5])

# Decode back
decoded = encoder.decode_label_encoded(encoded)
print(decoded)  # "ACGT"

# Without start token
encoded = encoder.label_encode(sequence, add_start_token=False)
print(encoded)  # tensor([2, 3, 4, 5])
```

---

## Batch Processing

### Batch One-Hot Encoding

Encodes multiple sequences with automatic padding to max length.

**Output Shape**: (batch_size, max_length, 4)

**Example**:
```python
encoder = SequenceEncoder()

sequences = [
    "ACGTACGTACGTACGTACGTAC",  # 23 bp
    "TGCATGCA",                 # 8 bp
    "AAAAAAAAAAAAAAAAAAAAAA"    # 23 bp
]

# Batch encode
batch_encoded = encoder.batch_one_hot_encode(sequences)
print(batch_encoded.shape)  # (3, 23, 4)

# All sequences padded to max length (23)
print(batch_encoded[1].shape)  # (23, 4)
```

### Batch Label Encoding

Encodes multiple sequences with automatic padding to max length.

**Output Shape**: (batch_size, max_length) or (batch_size, max_length + 1) with start token

**Example**:
```python
encoder = SequenceEncoder()

sequences = [
    "ACGTACGTACGTACGTACGTAC",  # 23 bp
    "TGCATGCA",                 # 8 bp
    "AAAAAAAAAAAAAAAAAAAAAA"    # 23 bp
]

# Batch encode with start token
batch_encoded = encoder.batch_label_encode(sequences, add_start_token=True)
print(batch_encoded.shape)  # (3, 24)

# Batch encode without start token
batch_encoded = encoder.batch_label_encode(sequences, add_start_token=False)
print(batch_encoded.shape)  # (3, 23)

# Decode first sequence
decoded = encoder.decode_label_encoded(batch_encoded[0])
print(decoded)  # "ACGTACGTACGTACGTACGTAC"
```

---

## Sequence Validation

### Validate Sequence

```python
encoder = SequenceEncoder()

# Valid sequences
is_valid, error = encoder.validate_sequence("ACGTACGT")
print(is_valid)  # True
print(error)     # None

# Invalid sequences
is_valid, error = encoder.validate_sequence("INVALID")
print(is_valid)  # False
print(error)     # "Invalid nucleotides: {'I', 'N', 'V', 'L', 'D'}"

# Empty sequence
is_valid, error = encoder.validate_sequence("")
print(is_valid)  # False
print(error)     # "Sequence cannot be empty"
```

### Normalize Sequence

```python
encoder = SequenceEncoder()

# Normalize (uppercase, U->T)
normalized = encoder.normalize_sequence("acgtu")
print(normalized)  # "ACGTT"
```

---

## Helper Functions

### GC Content

Calculate GC content (fraction of G and C nucleotides).

```python
encoder = SequenceEncoder()

sequence = "ACGTACGTACGTACGTACGTAC"
gc_content = encoder.get_gc_content(sequence)
print(f"GC content: {gc_content:.2%}")  # GC content: 50.00%
```

### Sequence Length

```python
encoder = SequenceEncoder()

sequence = "ACGTACGTACGTACGTACGTAC"
length = encoder.get_sequence_length(sequence)
print(f"Length: {length}")  # Length: 23
```

---

## Device Support

### CPU Processing

```python
encoder = SequenceEncoder(device='cpu')
encoded = encoder.one_hot_encode("ACGT")
print(encoded.device)  # cpu
```

### GPU Processing

```python
encoder = SequenceEncoder(device='cuda')
encoded = encoder.one_hot_encode("ACGT")
print(encoded.device)  # cuda:0
```

---

## Integration with Training

### Feature Extraction for CRISPR_HNN

```python
import torch
from models.encoding import SequenceEncoder

# Load training data
train_df = pd.read_csv('data/processed/combined/train.csv')

# Create encoder
encoder = SequenceEncoder(device='cuda')

# Encode sgRNA sequences
sgrna_sequences = train_df['sgrna_sequence'].tolist()
sgrna_encoded = encoder.batch_one_hot_encode(sgrna_sequences)

print(f"Encoded shape: {sgrna_encoded.shape}")  # (N, 23, 4)

# Use for model training
# model = CRISPRHNNModel()
# output = model(sgrna_encoded)
```

### Feature Extraction for CCLMoff

```python
from models.encoding import SequenceEncoder

# Create encoder
encoder = SequenceEncoder(device='cuda')

# Encode sgRNA and target sequences
sgrna_sequences = train_df['sgrna_sequence'].tolist()
target_sequences = train_df['target_sequence'].tolist()

# Label encode (for language model)
sgrna_encoded = encoder.batch_label_encode(sgrna_sequences)
target_encoded = encoder.batch_label_encode(target_sequences)

print(f"sgRNA shape: {sgrna_encoded.shape}")
print(f"Target shape: {target_encoded.shape}")

# Concatenate for model input
# combined = torch.cat([sgrna_encoded, target_encoded], dim=1)
```

---

## Nucleotide Mappings

### One-Hot Encoding

| Nucleotide | Encoding |
|-----------|----------|
| A | [1, 0, 0, 0] |
| C | [0, 1, 0, 0] |
| G | [0, 0, 1, 0] |
| T | [0, 0, 0, 1] |
| U | [0, 0, 0, 1] |

### Label Encoding

| Token | Value |
|-------|-------|
| Start | 1 |
| A | 2 |
| C | 3 |
| G | 4 |
| T | 5 |
| U | 5 |
| Padding | 0 |

---

## API Reference

### SequenceEncoder Class

#### Methods

**`one_hot_encode(sequence: str) -> Tensor`**
- One-hot encode a single sequence
- Returns: Tensor of shape (seq_len, 4)

**`label_encode(sequence: str, add_start_token: bool = True) -> Tensor`**
- Label encode a single sequence
- Returns: Tensor of shape (seq_len,) or (seq_len+1,)

**`batch_one_hot_encode(sequences: List[str], max_length: Optional[int] = None, pad_value: float = 0.0) -> Tensor`**
- One-hot encode multiple sequences with padding
- Returns: Tensor of shape (batch_size, max_len, 4)

**`batch_label_encode(sequences: List[str], max_length: Optional[int] = None, add_start_token: bool = True, pad_value: int = 0) -> Tensor`**
- Label encode multiple sequences with padding
- Returns: Tensor of shape (batch_size, max_len) or (batch_size, max_len+1)

**`decode_label_encoded(encoded: Tensor, skip_start_token: bool = True) -> str`**
- Decode label-encoded sequence back to string
- Returns: Decoded sequence string

**`decode_one_hot(encoded: Tensor) -> str`**
- Decode one-hot encoded sequence back to string
- Returns: Decoded sequence string

**`validate_sequence(sequence: str) -> Tuple[bool, Optional[str]]`**
- Validate DNA/RNA sequence
- Returns: (is_valid, error_message)

**`normalize_sequence(sequence: str) -> str`**
- Normalize sequence (uppercase, U->T)
- Returns: Normalized sequence

**`get_gc_content(sequence: str) -> float`**
- Calculate GC content
- Returns: GC content as fraction (0-1)

**`get_sequence_length(sequence: str) -> int`**
- Get sequence length
- Returns: Length of sequence

---

## Performance Considerations

### Memory Usage

- **One-hot encoding**: ~4 bytes per nucleotide per sequence
- **Label encoding**: ~8 bytes per nucleotide per sequence (with start token)

### Batch Processing

For optimal performance:
- Use batch processing instead of individual sequences
- Batch size: 32-256 depending on GPU memory
- Sequences are automatically padded to max length in batch

### GPU Acceleration

```python
# GPU processing is faster for large batches
encoder = SequenceEncoder(device='cuda')

# Batch of 1000 sequences
sequences = [random_sequence() for _ in range(1000)]
encoded = encoder.batch_one_hot_encode(sequences)  # Fast on GPU
```

---

## Examples

### Example 1: Prepare Data for Training

```python
import pandas as pd
from models.encoding import SequenceEncoder

# Load data
train_df = pd.read_csv('data/processed/combined/train.csv')

# Create encoder
encoder = SequenceEncoder(device='cuda')

# Encode sgRNA sequences
sgrna_sequences = train_df['sgrna_sequence'].tolist()
sgrna_encoded = encoder.batch_one_hot_encode(sgrna_sequences)

# Encode target sequences
target_sequences = train_df['target_sequence'].tolist()
target_encoded = encoder.batch_label_encode(target_sequences)

print(f"sgRNA encoded: {sgrna_encoded.shape}")
print(f"Target encoded: {target_encoded.shape}")

# Save for training
torch.save(sgrna_encoded, 'data/processed/sgrna_encoded.pt')
torch.save(target_encoded, 'data/processed/target_encoded.pt')
```

### Example 2: Data Augmentation

```python
from models.encoding import SequenceEncoder

encoder = SequenceEncoder()

# Original sequence
original = "ACGTACGTACGTACGTACGTAC"

# Encode and decode
encoded = encoder.one_hot_encode(original)
decoded = encoder.decode_one_hot(encoded)

print(f"Original: {original}")
print(f"Decoded:  {decoded}")
assert original == decoded, "Encoding/decoding mismatch!"
```

### Example 3: Batch Processing with Variable Lengths

```python
from models.encoding import SequenceEncoder

encoder = SequenceEncoder()

# Sequences of different lengths
sequences = [
    "ACGTACGTACGTACGTACGTAC",  # 23 bp
    "TGCA",                     # 4 bp
    "ACGTACGTACGTACGTACGTACGTACGTAC",  # 31 bp
]

# Batch encode (automatically pads to max length)
batch_encoded = encoder.batch_one_hot_encode(sequences)
print(f"Batch shape: {batch_encoded.shape}")  # (3, 31, 4)

# All sequences now have same length (31)
for i, encoded in enumerate(batch_encoded):
    print(f"Sequence {i}: {encoded.shape}")  # (31, 4)
```

---

## Troubleshooting

### Issue: "Invalid nucleotides"

**Solution**: Ensure sequences contain only ACGTU
```python
# Invalid
encoder.one_hot_encode("ACGTX")  # Error: Invalid nucleotides: {'X'}

# Valid
encoder.one_hot_encode("ACGTU")  # OK (U treated as T)
```

### Issue: Empty sequence

**Solution**: Ensure sequences are not empty
```python
# Invalid
encoder.one_hot_encode("")  # Error: Sequence cannot be empty

# Valid
encoder.one_hot_encode("A")  # OK
```

### Issue: Out of memory

**Solution**: Use smaller batch sizes
```python
# Process in smaller batches
batch_size = 32
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    encoded = encoder.batch_one_hot_encode(batch)
```

---

## Summary

✓ **One-Hot Encoding**: Convert sequences to (seq_len, 4) tensors
✓ **Label Encoding**: Convert sequences to (seq_len,) integer tensors
✓ **Batch Processing**: Efficient encoding of multiple sequences
✓ **Validation**: Automatic sequence validation and normalization
✓ **GPU Support**: Accelerated processing on CUDA devices
✓ **Decoding**: Reverse encoding back to sequences

---

*Status: ✓ Ready for Model Training*
*Last Updated: 2025-11-20*
