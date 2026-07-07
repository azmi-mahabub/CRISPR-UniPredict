# Evaluation Script Guide

## Overview

The `scripts/evaluate.py` provides comprehensive model evaluation with:

- Model checkpoint loading
- Test set predictions
- Comprehensive metrics computation
- Visualization generation
- Results saving and reporting

---

## Quick Start

### Basic Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv \
  --output_dir results/exp_001
```

### Custom Batch Size

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv \
  --output_dir results/exp_001 \
  --batch_size 128
```

### Use CPU

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv \
  --output_dir results/exp_001 \
  --device cpu
```

---

## Command-Line Arguments

### `--checkpoint` (required)

Path to trained model checkpoint.

```bash
python scripts/evaluate.py --checkpoint models/checkpoints/best.pt
```

### `--test_data` (required)

Path to test data CSV file.

```bash
python scripts/evaluate.py --test_data data/processed/combined/test.csv
```

### `--output_dir`

Output directory for results.

```bash
python scripts/evaluate.py --output_dir results/exp_001
```

**Default**: `results/evaluation`

### `--batch_size`

Batch size for inference.

```bash
python scripts/evaluate.py --batch_size 128
```

**Default**: `64`

### `--device`

Device to use (cuda or cpu).

```bash
python scripts/evaluate.py --device cuda
```

**Default**: Auto-detect (cuda if available, else cpu)

---

## Output Structure

### Directory Structure

```
results/exp_001/
├── evaluation.log                    # Evaluation log
├── metrics/
│   └── metrics.json                 # Computed metrics
├── predictions/
│   └── predictions.csv              # Model predictions
└── plots/
    ├── roc_curve.png                # ROC curve
    ├── precision_recall_curve.png   # PR curve
    ├── scatter_on_target.png        # Scatter plot
    └── correlation_heatmap.png      # Correlation heatmap
```

---

## Output Files

### `evaluation.log`

Complete evaluation log with timestamps.

```
2025-01-22 12:00:00 - CRISPR-Evaluate - INFO - Loading checkpoint...
2025-01-22 12:00:01 - CRISPR-Evaluate - INFO - Model loaded from models/checkpoints/best.pt
2025-01-22 12:00:02 - CRISPR-Evaluate - INFO - Test dataset loaded: 1000 samples
...
```

### `metrics.json`

Computed metrics for all tasks.

```json
{
  "on_target": {
    "spearman_r": 0.92,
    "pearson_r": 0.91,
    "mae": 0.05,
    "rmse": 0.08,
    ...
  },
  "off_target": {
    "balanced_accuracy": 0.85,
    "f1_score": 0.80,
    "auroc": 0.90,
    "auprc": 0.85,
    ...
  },
  "combined": {...}
}
```

### `predictions.csv`

Model predictions and targets.

```csv
on_target_pred,on_target_target,on_target_mask,off_target_pred,off_target_target,off_target_mask
0.85,0.82,True,0.1,0,True
0.72,0.75,True,0.9,1,True
...
```

### Visualization Files

#### `roc_curve.png`

ROC curve for off-target classification showing:
- True Positive Rate vs False Positive Rate
- AUROC score
- Random classifier baseline

#### `precision_recall_curve.png`

Precision-Recall curve for off-target classification showing:
- Precision vs Recall
- AUPRC score

#### `scatter_on_target.png`

Scatter plot for on-target regression showing:
- Predicted vs Actual scores
- Perfect prediction diagonal
- Distribution of predictions

#### `correlation_heatmap.png`

Correlation matrix heatmap showing:
- Correlations between predictions and targets
- Cross-task correlations

---

## Metrics Explained

### On-Target Metrics (Regression)

**Spearman Correlation (r)**
- Range: -1 to 1 (higher is better)
- Measures rank-based correlation
- Robust to outliers

**Pearson Correlation (r)**
- Range: -1 to 1 (higher is better)
- Measures linear correlation
- Sensitive to outliers

**Mean Absolute Error (MAE)**
- Range: 0 to ∞ (lower is better)
- Average absolute prediction error

**Root Mean Square Error (RMSE)**
- Range: 0 to ∞ (lower is better)
- Penalizes large errors more than MAE

### Off-Target Metrics (Classification)

**Balanced Accuracy**
- Range: 0 to 1 (higher is better)
- Average recall for each class
- Good for imbalanced data

**F1-Score**
- Range: 0 to 1 (higher is better)
- Harmonic mean of precision and recall

**AUROC (Area Under ROC Curve)**
- Range: 0 to 1 (higher is better)
- Probability model ranks positive higher than negative
- Threshold-independent

**AUPRC (Area Under Precision-Recall Curve)**
- Range: 0 to 1 (higher is better)
- Good for imbalanced datasets
- Focus on positive class

---

## Complete Example

```bash
# Evaluate best model
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv \
  --output_dir results/exp_001 \
  --batch_size 128 \
  --device cuda

# View results
cat results/exp_001/evaluation.log
cat results/exp_001/metrics/metrics.json
head results/exp_001/predictions/predictions.csv

# View plots
open results/exp_001/plots/roc_curve.png
open results/exp_001/plots/scatter_on_target.png
```

---

## Interpreting Results

### Excellent Performance

```json
{
  "on_target": {
    "spearman_r": 0.95,
    "pearson_r": 0.94,
    "mae": 0.02,
    "rmse": 0.03
  },
  "off_target": {
    "balanced_accuracy": 0.95,
    "f1_score": 0.93,
    "auroc": 0.98,
    "auprc": 0.96
  }
}
```

### Good Performance

```json
{
  "on_target": {
    "spearman_r": 0.85,
    "pearson_r": 0.83,
    "mae": 0.05,
    "rmse": 0.07
  },
  "off_target": {
    "balanced_accuracy": 0.85,
    "f1_score": 0.80,
    "auroc": 0.90,
    "auprc": 0.85
  }
}
```

### Poor Performance

```json
{
  "on_target": {
    "spearman_r": 0.60,
    "pearson_r": 0.55,
    "mae": 0.15,
    "rmse": 0.20
  },
  "off_target": {
    "balanced_accuracy": 0.65,
    "f1_score": 0.60,
    "auroc": 0.70,
    "auprc": 0.65
  }
}
```

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Verify checkpoint path

```bash
# List available checkpoints
ls models/checkpoints/

# Use correct path
python scripts/evaluate.py --checkpoint models/checkpoints/best.pt
```

### Issue: "Test data not found"

**Solution**: Verify test data path

```bash
# Check file exists
ls data/processed/combined/test.csv

# Use correct path
python scripts/evaluate.py --test_data data/processed/combined/test.csv
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size

```bash
python scripts/evaluate.py --batch_size 32
```

Or use CPU:

```bash
python scripts/evaluate.py --device cpu
```

### Issue: "Plots not generated"

**Solution**: Check dependencies

```bash
pip install matplotlib seaborn
```

---

## Advanced Usage

### Evaluate Multiple Checkpoints

```bash
for checkpoint in models/checkpoints/checkpoint_epoch_*.pt; do
  python scripts/evaluate.py \
    --checkpoint "$checkpoint" \
    --test_data data/processed/combined/test.csv \
    --output_dir "results/$(basename $checkpoint .pt)"
done
```

### Compare Models

```bash
# Evaluate model 1
python scripts/evaluate.py \
  --checkpoint models/exp_001/best.pt \
  --output_dir results/exp_001

# Evaluate model 2
python scripts/evaluate.py \
  --checkpoint models/exp_002/best.pt \
  --output_dir results/exp_002

# Compare metrics
diff results/exp_001/metrics/metrics.json results/exp_002/metrics/metrics.json
```

### Generate Report

```bash
# Evaluate
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv \
  --output_dir results/final

# Create report
cat > results/final/REPORT.md << EOF
# Model Evaluation Report

## Metrics
$(cat results/final/metrics/metrics.json)

## Predictions
$(head -20 results/final/predictions/predictions.csv)

## Visualizations
- ROC Curve: plots/roc_curve.png
- PR Curve: plots/precision_recall_curve.png
- Scatter: plots/scatter_on_target.png
- Correlation: plots/correlation_heatmap.png
EOF
```

---

## Performance Tips

### 1. Use GPU for Faster Inference

```bash
python scripts/evaluate.py --device cuda
```

### 2. Increase Batch Size

```bash
python scripts/evaluate.py --batch_size 256
```

### 3. Use Multiple Workers

The script automatically uses 4 workers for data loading.

---

## Summary

The evaluation script provides:
- ✅ **Easy checkpoint loading**
- ✅ **Comprehensive metrics computation**
- ✅ **Professional visualizations**
- ✅ **CSV export of predictions**
- ✅ **JSON export of metrics**
- ✅ **Detailed logging**
- ✅ **Error handling**

Perfect for evaluating CRISPR-UniPredict models!
