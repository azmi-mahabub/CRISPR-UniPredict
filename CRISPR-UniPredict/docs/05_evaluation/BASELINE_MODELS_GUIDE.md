# Baseline Models Setup Guide

## Overview

The `setup_baseline_models.py` script downloads and configures baseline models for comparison with CRISPR-UniPredict:

1. **CRISPR_HNN** - Hybrid Neural Network for on-target prediction
2. **CCLMoff** - Language Model for off-target prediction
3. **CRISPR-Net** - Deep learning for on-target and off-target
4. **DeepCRISPR** - Deep learning for specificity prediction
5. **CNN_std** - Standard CNN baseline

---

## Quick Start

### Setup All Models

```bash
python scripts/setup_baseline_models.py
```

### Setup Specific Models

```bash
python scripts/setup_baseline_models.py --models crispr_hnn cnn_std
```

### Custom Directory

```bash
python scripts/setup_baseline_models.py --baselines_dir /path/to/baselines
```

---

## Folder Structure

After setup, the following structure is created:

```
baselines/
├── crispr_hnn/
│   ├── (cloned repository)
│   └── wrapper.py
├── cclmoff/
│   ├── (cloned repository)
│   └── wrapper.py
├── crispr_net/
│   ├── (cloned repository)
│   └── wrapper.py
├── deep_crispr/
│   ├── (cloned repository)
│   └── wrapper.py
├── cnn_std/
│   ├── model.py
│   └── wrapper.py
├── baseline_interface.py
└── config.json
```

---

## Unified Interface

All baseline models are accessible through a unified interface:

```python
from baselines.baseline_interface import get_baseline_models

# Initialize
baselines = get_baseline_models(device='cuda')

# Get available models
print(baselines.available_models())

# Make predictions
on_target = baselines.predict_on_target('crispr_hnn', sgrna, target)
off_target = baselines.predict_off_target('cclmoff', sgrna, target)
```

---

## Individual Model Usage

### CRISPR_HNN

```python
from baselines.crispr_hnn.wrapper import predict_on_target, predict_off_target

# On-target prediction
score = predict_on_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT')

# Off-target (not supported)
# score = predict_off_target(...)  # Returns 0.5
```

**Capabilities:**
- ✅ On-target efficiency prediction
- ❌ Off-target prediction (not supported)

### CCLMoff

```python
from baselines.cclmoff.wrapper import predict_on_target, predict_off_target

# Off-target prediction
score = predict_off_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT', 
                          target='ATGCATGCATGCATGCATGCATG')

# On-target (not supported)
# score = predict_on_target(...)  # Returns 0.5
```

**Capabilities:**
- ❌ On-target prediction (not supported)
- ✅ Off-target risk prediction

### CRISPR-Net

```python
from baselines.crispr_net.wrapper import predict_on_target, predict_off_target

# On-target prediction
on_target = predict_on_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT')

# Off-target prediction
off_target = predict_off_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT',
                               target='ATGCATGCATGCATGCATGCATG')
```

**Capabilities:**
- ✅ On-target efficiency prediction
- ✅ Off-target risk prediction

### DeepCRISPR

```python
from baselines.deep_crispr.wrapper import predict_on_target, predict_off_target

# On-target prediction
on_target = predict_on_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT')

# Off-target prediction
off_target = predict_off_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT',
                               target='ATGCATGCATGCATGCATGCATG')
```

**Capabilities:**
- ✅ On-target efficiency prediction
- ✅ Off-target risk prediction

### CNN_std

```python
from baselines.cnn_std.wrapper import predict_on_target, predict_off_target

# On-target prediction
score = predict_on_target(sgrna='GCTAGCTAGCTAGCTAGCTAGCT')

# Off-target (not supported)
# score = predict_off_target(...)  # Returns 0.5
```

**Capabilities:**
- ✅ On-target efficiency prediction (standard CNN)
- ❌ Off-target prediction (not supported)

---

## Comparison Script

Create a comparison script to evaluate all baselines:

```python
import numpy as np
from pathlib import Path
from baselines.baseline_interface import get_baseline_models

# Load baselines
baselines = get_baseline_models(device='cuda')

# Example sgRNAs
sgrnas = [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG",
]

# Get predictions from all models
results = {}

for sgrna in sgrnas:
    results[sgrna] = {}
    
    # On-target predictions
    for model_key in ['crispr_hnn', 'crispr_net', 'deep_crispr', 'cnn_std']:
        try:
            score = baselines.predict_on_target(model_key, sgrna)
            results[sgrna][f'{model_key}_on_target'] = score
        except:
            results[sgrna][f'{model_key}_on_target'] = None
    
    # Off-target predictions
    for model_key in ['cclmoff', 'crispr_net', 'deep_crispr']:
        try:
            score = baselines.predict_off_target(model_key, sgrna)
            results[sgrna][f'{model_key}_off_target'] = score
        except:
            results[sgrna][f'{model_key}_off_target'] = None

# Print results
for sgrna, predictions in results.items():
    print(f"\n{sgrna}:")
    for model, score in predictions.items():
        if score is not None:
            print(f"  {model}: {score:.4f}")
```

---

## Model Information

### CRISPR_HNN

- **Paper**: Prediction of CRISPR-Cas9 On-Target Activity using Hybrid Neural Network
- **Repository**: https://github.com/MyungjunKim/CRISPR_HNN.git
- **Framework**: TensorFlow 2.5.0
- **Input**: One-hot encoded sgRNA (23bp)
- **Output**: On-target efficiency (0-1)
- **Datasets**: 9 public CRISPR datasets

### CCLMoff

- **Paper**: CRISPR/Cas Language Model for Off-Target Prediction
- **Repository**: https://github.com/MyungjunKim/CCLMoff.git
- **Framework**: PyTorch + Transformers
- **Input**: sgRNA + target sequences
- **Output**: Off-target probability (0-1)
- **Dataset**: 09212024_CCLMoff_dataset.csv

### CRISPR-Net

- **Paper**: CRISPR-Net: A Recurrent Convolutional Network-based Model for Predicting sgRNA Efficacy in Different Cell-types
- **Repository**: https://github.com/JasonLinjc/CRISPR-Net.git
- **Framework**: PyTorch
- **Input**: sgRNA sequence + cell type
- **Output**: On-target efficiency + off-target risk
- **Datasets**: Multiple cell types

### DeepCRISPR

- **Paper**: DeepCRISPR: Optimized CRISPR Guide RNA Design by Deep Learning
- **Repository**: https://github.com/bm2-lab/DeepCRISPR.git
- **Framework**: TensorFlow/Keras
- **Input**: sgRNA sequence
- **Output**: On-target efficiency + off-target specificity
- **Datasets**: Public CRISPR datasets

### CNN_std

- **Description**: Standard CNN baseline from literature
- **Framework**: PyTorch
- **Architecture**: 3 Conv1D layers + FC layers
- **Input**: One-hot encoded sgRNA (4, 23)
- **Output**: On-target efficiency (0-1)
- **Status**: Implemented from scratch

---

## Troubleshooting

### Issue: "git: command not found"

**Solution**: Install Git
```bash
# Windows
choco install git

# macOS
brew install git

# Linux
sudo apt-get install git
```

### Issue: "Failed to clone repository"

**Cause**: Network issues or repository unavailable

**Solution**:
```bash
# Try manual clone
git clone https://github.com/MyungjunKim/CRISPR_HNN.git baselines/crispr_hnn

# Then run setup again
python scripts/setup_baseline_models.py
```

### Issue: "Module not found" when importing

**Solution**:
```python
import sys
from pathlib import Path

# Add baselines to path
sys.path.insert(0, str(Path(__file__).parent / 'baselines'))

from baseline_interface import get_baseline_models
```

### Issue: "Model predictions are all 0.5"

**Cause**: Model not properly loaded or weights missing

**Solution**:
1. Check if model directory exists
2. Verify wrapper.py is present
3. Check model logs for errors
4. Download pretrained weights if needed

---

## Complete Comparison Example

```python
import numpy as np
import pandas as pd
from pathlib import Path
from baselines.baseline_interface import get_baseline_models

# Load test data
test_data = pd.read_csv('data/processed/combined/test.csv')
sgrnas = test_data['sgrna'].values[:100]  # First 100 for testing

# Initialize baselines
baselines = get_baseline_models(device='cuda')

# Collect predictions
predictions = {
    'sgrna': [],
    'crispr_hnn_on': [],
    'cclmoff_off': [],
    'crispr_net_on': [],
    'crispr_net_off': [],
    'deep_crispr_on': [],
    'deep_crispr_off': [],
    'cnn_std_on': []
}

for sgrna in sgrnas:
    predictions['sgrna'].append(sgrna)
    
    # CRISPR_HNN
    try:
        predictions['crispr_hnn_on'].append(
            baselines.predict_on_target('crispr_hnn', sgrna)
        )
    except:
        predictions['crispr_hnn_on'].append(np.nan)
    
    # CCLMoff
    try:
        predictions['cclmoff_off'].append(
            baselines.predict_off_target('cclmoff', sgrna)
        )
    except:
        predictions['cclmoff_off'].append(np.nan)
    
    # CRISPR-Net
    try:
        predictions['crispr_net_on'].append(
            baselines.predict_on_target('crispr_net', sgrna)
        )
        predictions['crispr_net_off'].append(
            baselines.predict_off_target('crispr_net', sgrna)
        )
    except:
        predictions['crispr_net_on'].append(np.nan)
        predictions['crispr_net_off'].append(np.nan)
    
    # DeepCRISPR
    try:
        predictions['deep_crispr_on'].append(
            baselines.predict_on_target('deep_crispr', sgrna)
        )
        predictions['deep_crispr_off'].append(
            baselines.predict_off_target('deep_crispr', sgrna)
        )
    except:
        predictions['deep_crispr_on'].append(np.nan)
        predictions['deep_crispr_off'].append(np.nan)
    
    # CNN_std
    try:
        predictions['cnn_std_on'].append(
            baselines.predict_on_target('cnn_std', sgrna)
        )
    except:
        predictions['cnn_std_on'].append(np.nan)

# Save results
df = pd.DataFrame(predictions)
df.to_csv('results/baseline_predictions.csv', index=False)

print("Baseline predictions saved to results/baseline_predictions.csv")
print(df.head())
```

---

## Summary

The baseline models setup provides:
- ✅ **Automatic download** of 5 baseline models
- ✅ **Unified interface** for consistent API
- ✅ **Wrapper scripts** for each model
- ✅ **CNN_std implementation** from scratch
- ✅ **Easy comparison** with CRISPR-UniPredict
- ✅ **Comprehensive documentation**

Perfect for model benchmarking and comparison!
