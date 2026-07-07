# Evaluation Metrics Guide

## Overview

The `metrics.py` module provides comprehensive evaluation metrics for CRISPR-UniPredict:

- **On-target prediction** (regression): Spearman, Pearson, MAE, RMSE
- **Off-target prediction** (classification): Balanced Accuracy, F1, AUROC, AUPRC

---

## Quick Start

### On-Target Metrics

```python
from utils.evaluation.metrics import MetricsCalculator
import numpy as np

calculator = MetricsCalculator()

# Predictions and targets
predictions = np.array([0.8, 0.6, 0.9, 0.7])
targets = np.array([0.85, 0.65, 0.88, 0.72])

# Compute metrics
metrics = calculator.compute_on_target_metrics(predictions, targets)

print(f"Spearman R: {metrics['spearman_r']:.4f}")
print(f"Pearson R: {metrics['pearson_r']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Off-Target Metrics

```python
# Predictions (probabilities) and targets (0 or 1)
predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
targets = np.array([0, 1, 0, 1, 0, 1])

# Compute metrics
metrics = calculator.compute_off_target_metrics(predictions, targets)

print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"F1-score: {metrics['f1_score']:.4f}")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"AUPRC: {metrics['auprc']:.4f}")
```

### All Metrics

```python
# Compute all metrics at once
all_metrics = calculator.compute_all_metrics(
    on_target_pred=on_target_predictions,
    off_target_pred=off_target_predictions,
    on_target_target=on_target_targets,
    off_target_target=off_target_targets,
    on_target_mask=on_target_mask,
    off_target_mask=off_target_mask
)

print(all_metrics['on_target'])
print(all_metrics['off_target'])
```

---

## MetricsCalculator Class

### Initialization

```python
from utils.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator(epsilon=1e-7)
```

**Parameters**:
- `epsilon`: Small value for numerical stability (default: 1e-7)

### Methods

#### `compute_on_target_metrics(predictions, targets)`

Compute regression metrics for on-target prediction.

**Parameters**:
- `predictions`: Predicted values (numpy array or torch tensor)
- `targets`: Target values (numpy array or torch tensor)

**Returns**: Dictionary with metrics

**Metrics**:
- `spearman_r`: Spearman correlation coefficient
- `spearman_pval`: P-value for Spearman correlation
- `pearson_r`: Pearson correlation coefficient
- `pearson_pval`: P-value for Pearson correlation
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Square Error
- `mse`: Mean Squared Error
- `pred_mean`: Mean of predictions
- `pred_std`: Standard deviation of predictions
- `target_mean`: Mean of targets
- `target_std`: Standard deviation of targets

#### `compute_off_target_metrics(predictions, targets, thresholds=None)`

Compute classification metrics for off-target prediction.

**Parameters**:
- `predictions`: Predicted probabilities (0-1)
- `targets`: Target labels (0 or 1)
- `thresholds`: List of thresholds to evaluate (default: [0.3, 0.5, 0.7])

**Returns**: Dictionary with metrics

**Metrics**:
- `balanced_accuracy`: Balanced accuracy
- `f1_score`: F1-score
- `auroc`: Area Under ROC Curve
- `auprc`: Area Under Precision-Recall Curve
- `threshold_metrics`: Metrics at different thresholds
  - `precision`: Precision at threshold
  - `recall`: Recall at threshold
  - `accuracy`: Accuracy at threshold
  - `f1`: F1-score at threshold
- `tn`: True negatives
- `fp`: False positives
- `fn`: False negatives
- `tp`: True positives
- `pred_mean`: Mean of predictions
- `pred_std`: Standard deviation of predictions
- `positive_ratio`: Ratio of positive samples
- `negative_ratio`: Ratio of negative samples

#### `compute_all_metrics(...)`

Compute all metrics for both tasks.

**Parameters**:
- `on_target_pred`: On-target predictions
- `off_target_pred`: Off-target predictions
- `on_target_target`: On-target targets
- `off_target_target`: Off-target targets
- `on_target_mask`: Mask for valid on-target samples (optional)
- `off_target_mask`: Mask for valid off-target samples (optional)

**Returns**: Dictionary with both on-target and off-target metrics

---

## On-Target Metrics (Regression)

### Spearman Correlation Coefficient (SCC)

**What it measures**: Rank-based correlation between predictions and targets

**Range**: -1 to 1 (higher is better)

**When to use**: Robust to outliers, non-linear relationships

```python
scc = metrics['spearman_r']  # 0.93 = excellent correlation
```

### Pearson Correlation Coefficient (PCC)

**What it measures**: Linear correlation between predictions and targets

**Range**: -1 to 1 (higher is better)

**When to use**: Assumes linear relationship

```python
pcc = metrics['pearson_r']  # 0.92 = strong linear correlation
```

### Mean Absolute Error (MAE)

**What it measures**: Average absolute difference between predictions and targets

**Range**: 0 to ∞ (lower is better)

**Interpretation**: On average, predictions are off by MAE

```python
mae = metrics['mae']  # 0.05 = predictions off by 0.05 on average
```

### Root Mean Square Error (RMSE)

**What it measures**: Square root of average squared differences

**Range**: 0 to ∞ (lower is better)

**Interpretation**: Penalizes large errors more than MAE

```python
rmse = metrics['rmse']  # 0.08 = RMS error of 0.08
```

---

## Off-Target Metrics (Classification)

### Balanced Accuracy

**What it measures**: Average of recall for each class

**Range**: 0 to 1 (higher is better)

**When to use**: Imbalanced datasets

```python
balanced_acc = metrics['balanced_accuracy']  # 0.85 = good balance
```

### F1-Score

**What it measures**: Harmonic mean of precision and recall

**Range**: 0 to 1 (higher is better)

**When to use**: Need balance between precision and recall

```python
f1 = metrics['f1_score']  # 0.80 = good balance
```

### AUROC (Area Under ROC Curve)

**What it measures**: Probability that model ranks random positive higher than random negative

**Range**: 0 to 1 (higher is better)

**When to use**: Threshold-independent performance

```python
auroc = metrics['auroc']  # 0.90 = excellent discrimination
```

### AUPRC (Area Under Precision-Recall Curve)

**What it measures**: Area under precision-recall curve

**Range**: 0 to 1 (higher is better)

**When to use**: Imbalanced datasets, focus on positive class

```python
auprc = metrics['auprc']  # 0.85 = good precision-recall trade-off
```

### Threshold-Specific Metrics

Metrics computed at different thresholds (0.3, 0.5, 0.7):

```python
threshold_metrics = metrics['threshold_metrics']

# At threshold 0.5
t50 = threshold_metrics['threshold_0.5']
print(f"Precision: {t50['precision']:.3f}")
print(f"Recall: {t50['recall']:.3f}")
print(f"Accuracy: {t50['accuracy']:.3f}")
print(f"F1: {t50['f1']:.3f}")
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from models.crispr_unipredict import CRISPRUniPredict
from utils.evaluation.metrics import MetricsCalculator
from utils.preprocessing.dataloader_factory import create_dataloaders
from configs.config_loader import ConfigLoader

# Setup
config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config
dataloaders = create_dataloaders(config)

# Initialize model and metrics
device = 'cuda' if config.device.use_cuda else 'cpu'
model = CRISPRUniPredict(device=device)
calculator = MetricsCalculator()

# Evaluation loop
model.eval()
all_on_target_pred = []
all_on_target_target = []
all_off_target_pred = []
all_off_target_target = []

with torch.no_grad():
    for batch in dataloaders['val']:
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
        
        # Collect predictions
        all_on_target_pred.append(on_target_pred[on_target_mask].cpu())
        all_on_target_target.append(on_target_score[on_target_mask].cpu())
        all_off_target_pred.append(off_target_pred[off_target_mask].cpu())
        all_off_target_target.append(off_target_label[off_target_mask].cpu())

# Concatenate all predictions
on_target_pred = torch.cat(all_on_target_pred)
on_target_target = torch.cat(all_on_target_target)
off_target_pred = torch.cat(all_off_target_pred)
off_target_target = torch.cat(all_off_target_target)

# Compute metrics
metrics = calculator.compute_all_metrics(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_target,
    off_target_target=off_target_target
)

# Print results
print("ON-TARGET METRICS:")
for key, value in metrics['on_target'].items():
    if not isinstance(value, dict):
        print(f"  {key}: {value:.4f}")

print("\nOFF-TARGET METRICS:")
for key, value in metrics['off_target'].items():
    if not isinstance(value, dict):
        print(f"  {key}: {value:.4f}")
```

---

## Convenience Functions

### `compute_on_target_metrics(predictions, targets)`

Quick function to compute on-target metrics.

```python
from utils.evaluation.metrics import compute_on_target_metrics

metrics = compute_on_target_metrics(predictions, targets)
```

### `compute_off_target_metrics(predictions, targets)`

Quick function to compute off-target metrics.

```python
from utils.evaluation.metrics import compute_off_target_metrics

metrics = compute_off_target_metrics(predictions, targets)
```

---

## Handling Masks

### With Masking

```python
# Only compute metrics on valid samples
metrics = calculator.compute_all_metrics(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_target,
    off_target_target=off_target_target,
    on_target_mask=on_target_mask,  # Only True samples used
    off_target_mask=off_target_mask  # Only True samples used
)
```

### Without Masking

```python
# Use all samples
metrics = calculator.compute_all_metrics(
    on_target_pred=on_target_pred,
    off_target_pred=off_target_pred,
    on_target_target=on_target_target,
    off_target_target=off_target_target
)
```

---

## Torch Tensor Support

The calculator supports both numpy arrays and torch tensors:

```python
import torch
import numpy as np

# Numpy arrays
pred_np = np.array([0.8, 0.6, 0.9])
target_np = np.array([0.85, 0.65, 0.88])
metrics = calculator.compute_on_target_metrics(pred_np, target_np)

# Torch tensors
pred_torch = torch.tensor([0.8, 0.6, 0.9])
target_torch = torch.tensor([0.85, 0.65, 0.88])
metrics = calculator.compute_on_target_metrics(pred_torch, target_torch)

# Mixed
metrics = calculator.compute_on_target_metrics(pred_torch, target_np)
```

---

## Interpreting Results

### Excellent Performance
```python
{
    'spearman_r': 0.95,      # Very strong rank correlation
    'pearson_r': 0.94,       # Very strong linear correlation
    'mae': 0.02,             # Small average error
    'rmse': 0.03,            # Small RMS error
    'balanced_accuracy': 0.95,  # Excellent balance
    'f1_score': 0.93,        # Excellent F1
    'auroc': 0.98,           # Excellent discrimination
    'auprc': 0.96            # Excellent precision-recall
}
```

### Good Performance
```python
{
    'spearman_r': 0.85,      # Good rank correlation
    'pearson_r': 0.83,       # Good linear correlation
    'mae': 0.05,             # Moderate average error
    'rmse': 0.07,            # Moderate RMS error
    'balanced_accuracy': 0.85,  # Good balance
    'f1_score': 0.80,        # Good F1
    'auroc': 0.90,           # Good discrimination
    'auprc': 0.85            # Good precision-recall
}
```

### Poor Performance
```python
{
    'spearman_r': 0.60,      # Weak rank correlation
    'pearson_r': 0.55,       # Weak linear correlation
    'mae': 0.15,             # Large average error
    'rmse': 0.20,            # Large RMS error
    'balanced_accuracy': 0.65,  # Poor balance
    'f1_score': 0.60,        # Poor F1
    'auroc': 0.70,           # Poor discrimination
    'auprc': 0.65            # Poor precision-recall
}
```

---

## Troubleshooting

### Issue: "Only one class present in targets"

**Cause**: All targets are 0 or all are 1

**Solution**:
```python
# Check class distribution
unique_targets = np.unique(targets)
print(f"Classes: {unique_targets}")

# Ensure both classes present
if len(unique_targets) < 2:
    print("WARNING: Only one class in targets")
```

### Issue: "NaN in metrics"

**Cause**: Division by zero or invalid values

**Solution**:
```python
# Check for NaN in predictions
if np.isnan(predictions).any():
    print("WARNING: NaN in predictions")

# Check for invalid values
if np.isinf(predictions).any():
    print("WARNING: Inf in predictions")
```

### Issue: "Length mismatch"

**Cause**: Predictions and targets have different lengths

**Solution**:
```python
# Verify lengths match
assert len(predictions) == len(targets), \
    f"Length mismatch: {len(predictions)} vs {len(targets)}"
```

---

## Summary

The metrics module provides:
- ✅ **On-target metrics**: Spearman, Pearson, MAE, RMSE
- ✅ **Off-target metrics**: Balanced Accuracy, F1, AUROC, AUPRC
- ✅ **Threshold analysis**: Metrics at different thresholds
- ✅ **Masking support**: Handle mixed batches
- ✅ **Torch support**: Works with tensors and arrays
- ✅ **Error handling**: Graceful handling of edge cases
- ✅ **Production-ready**: Comprehensive and robust

Perfect for evaluating CRISPR-UniPredict models!
