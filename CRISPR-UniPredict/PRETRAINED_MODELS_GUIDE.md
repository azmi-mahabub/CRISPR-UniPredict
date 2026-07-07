# Pretrained Models Download Guide

## Overview

This guide explains how to download and use pretrained models for CRISPR-UniPredict.

---

## Available Models

### RNA-FM (t12)

| Property | Value |
|----------|-------|
| **Name** | RNA Foundation Model (12 layers) |
| **Size** | ~700 MB |
| **Type** | RNA language model |
| **Use Case** | Off-target prediction (CCLMoff) |
| **Source** | Facebook AI Research |

---

## Quick Start

### Automatic Download

```bash
cd CRISPR-UniPredict
python scripts/download_pretrained_models.py
```

**Expected Output**:
```
✓ Model downloaded and verified successfully!
✓ ALL MODELS DOWNLOADED SUCCESSFULLY
```

### What Gets Downloaded

- **Location**: `models/pretrained/`
- **Files**:
  - `rna_fm_t12.pt` (~700 MB)

---

## Download Methods

### Method A: Direct Download (Default)

**Pros**:
- Fast and reliable
- No additional dependencies
- Direct from Facebook AI Research

**Cons**:
- Requires internet connection
- May be slow on poor connections

**URL**: `https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt`

### Method B: fair-esm Python API

**Pros**:
- Automatic caching
- Integrated with fair-esm
- Handles dependencies

**Cons**:
- Requires fair-esm installation
- Slower initial setup

**How it works**:
1. Installs fair-esm if needed
2. Uses fair-esm API to load model
3. Saves model checkpoint

---

## Installation

### Prerequisites

```bash
# Install PyTorch (if not already installed)
pip install torch

# Install fair-esm (optional, for Method B)
pip install fair-esm
```

### Automatic Installation

The script will automatically:
1. Check if fair-esm is installed
2. Install it if missing
3. Download models

---

## Usage

### Run Download Script

```bash
python scripts/download_pretrained_models.py
```

### Load Downloaded Model

```python
import torch

# Load RNA-FM model
model_path = 'models/pretrained/rna_fm_t12.pt'
checkpoint = torch.load(model_path, map_location='cpu')

print(f"Model loaded successfully!")
print(f"Keys: {list(checkpoint.keys())}")
```

### Use with fair-esm

```python
import fair_esm

# Load model with fair-esm
model, alphabet = fair_esm.load_model_and_alphabet('rna_fm_t12')

# Use model for inference
print(f"Model: {model}")
print(f"Alphabet: {alphabet}")
```

---

## Verification

### Check Downloaded Files

```bash
ls -lh models/pretrained/
```

**Expected Output**:
```
-rw-r--r-- 1 user group 700M Nov 20 03:45 rna_fm_t12.pt
```

### Verify Checkpoint

```bash
python -c "import torch; checkpoint = torch.load('models/pretrained/rna_fm_t12.pt'); print('✓ Checkpoint valid')"
```

### Check Model Size

```bash
python -c "import os; size = os.path.getsize('models/pretrained/rna_fm_t12.pt'); print(f'Size: {size/(1024**3):.2f} GB')"
```

---

## Troubleshooting

### Issue: Download Fails

**Solution 1**: Try Method B (fair-esm API)
```bash
# Edit script to use Method B first
# Change: use_method_b_first=True
```

**Solution 2**: Manual download
```bash
# Download manually
wget -O models/pretrained/rna_fm_t12.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt

# Or with curl
curl -L -o models/pretrained/rna_fm_t12.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt
```

### Issue: "fair-esm not installed"

**Solution**: Install manually
```bash
pip install fair-esm
```

### Issue: "Checkpoint verification failed"

**Solution**: Re-download the model
```bash
# Remove corrupted file
rm models/pretrained/rna_fm_t12.pt

# Re-download
python scripts/download_pretrained_models.py
```

### Issue: Slow Download

**Solution**: Use alternative source
```bash
# Download from HuggingFace instead
wget -O models/pretrained/rna_fm_t12.pt \
  https://huggingface.co/facebook/rna-fm/resolve/main/rna_fm_t12.pt
```

---

## Manual Download

If automatic download fails, follow these steps:

### Step 1: Download File

**Option A - wget**:
```bash
wget -O models/pretrained/rna_fm_t12.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt
```

**Option B - curl**:
```bash
curl -L -o models/pretrained/rna_fm_t12.pt \
  https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt
```

**Option C - Browser**:
1. Visit: https://dl.fbaipublicfiles.com/fair-esm/models/rna_fm_t12.pt
2. Save to: `models/pretrained/rna_fm_t12.pt`

### Step 2: Verify File

```bash
# Check size (should be ~700 MB)
ls -lh models/pretrained/rna_fm_t12.pt

# Verify checkpoint
python -c "import torch; torch.load('models/pretrained/rna_fm_t12.pt'); print('✓ OK')"
```

---

## Model Details

### RNA-FM Architecture

- **Type**: Transformer-based RNA language model
- **Layers**: 12
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Parameters**: ~100M
- **Training Data**: Large-scale RNA sequences

### Use Cases

1. **Off-Target Prediction**: CCLMoff model
2. **RNA Representation**: Feature extraction
3. **Sequence Understanding**: RNA analysis

---

## Integration with Training

### For CCLMoff Model

```python
import torch
from fair_esm import load_model_and_alphabet

# Load pretrained model
model, alphabet = load_model_and_alphabet('rna_fm_t12')

# Use for feature extraction
def extract_features(sequence):
    tokens = alphabet.encode(sequence)
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    return results['representations'][12]

# Extract features for training
features = extract_features('ACGTACGTACGT')
```

### For On-Target Model

```python
# On-target model uses different architecture
# See scripts/train_on_target.py for details
```

---

## Storage

### Directory Structure

```
models/
├── pretrained/
│   ├── rna_fm_t12.pt          (~700 MB)
│   └── ...
├── on_target/
│   └── model_checkpoint.pt
└── off_target/
    └── model_checkpoint.pt
```

### Disk Space Required

| Component | Size |
|-----------|------|
| RNA-FM | 700 MB |
| Other models | 500 MB |
| **Total** | **1.2 GB** |

---

## Advanced Usage

### Load Model with Custom Settings

```python
import torch
import fair_esm

# Load model
model, alphabet = fair_esm.load_model_and_alphabet('rna_fm_t12')

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Set to eval mode
model.eval()

print(f"Model on device: {device}")
```

### Batch Processing

```python
import torch
from fair_esm import load_model_and_alphabet

model, alphabet = load_model_and_alphabet('rna_fm_t12')

sequences = [
    'ACGTACGTACGT',
    'TGCATGCATGCA',
    'AAAAAAAAAA'
]

# Batch encode
batch_tokens = [alphabet.encode(seq) for seq in sequences]

# Extract features
with torch.no_grad():
    for tokens in batch_tokens:
        results = model(tokens, repr_layers=[12])
        features = results['representations'][12]
        print(f"Features shape: {features.shape}")
```

---

## References

- **Facebook AI Research ESM**: https://github.com/facebookresearch/fair-esm
- **RNA-FM Paper**: https://arxiv.org/abs/2209.11455
- **HuggingFace Model**: https://huggingface.co/facebook/rna-fm

---

## Support

### Common Questions

**Q: Can I use CPU-only?**
A: Yes, the script works with CPU. GPU is optional for faster inference.

**Q: How long does download take?**
A: Depends on internet speed. Typically 5-15 minutes for 700 MB.

**Q: Can I use a different model?**
A: Yes, edit the MODELS dictionary in the script.

**Q: Where are models cached?**
A: In `models/pretrained/` directory.

### Getting Help

1. Check troubleshooting section above
2. Review script output for error messages
3. Check internet connection
4. Try manual download
5. Check disk space

---

## Summary

✓ **Automatic Download**: Run `python scripts/download_pretrained_models.py`
✓ **Two Methods**: Direct download or fair-esm API
✓ **Verification**: Automatic checkpoint verification
✓ **Fallback**: Manual download instructions provided
✓ **Ready for Training**: Models ready for use immediately

---

*Last Updated: 2025-11-20*
*Status: ✓ Ready*
