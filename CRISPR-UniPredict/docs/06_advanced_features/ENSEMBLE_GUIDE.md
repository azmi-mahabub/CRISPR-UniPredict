# Ensemble Model Guide

## Overview

The ensemble system combines predictions from multiple CRISPR-UniPredict model variants to improve robustness and performance. Four ensemble methods are implemented: simple averaging, weighted averaging, stacking, and voting.

## Installation

```bash
# No additional dependencies needed
# Uses existing PyTorch and model infrastructure
```

## Quick Start

### Train Ensemble (5 models, 3-5 days)

```bash
python scripts/train_ensemble.py \
  --n_models 5 \
  --base_config configs/model_config.yaml \
  --output_dir models/ensemble \
  --ensemble_method weighted_average
```

### Use Trained Ensemble

```python
from models.ensemble_model import EnsembleModel

# Load ensemble
ensemble = EnsembleModel.load_config('models/ensemble/ensemble_config.json')

# Make predictions
on_target_pred, off_target_pred = ensemble.predict(
    sgrna_onehot, sgrna_label,
    method='weighted_average'
)
```

## Model Variants

The ensemble trains 5 different model variants:

### Variant 0: Full Model
- Uses base configuration
- Standard training setup
- Baseline variant

### Variant 1: Different Random Seed
- Seed: 123 (vs default 42)
- Captures randomness effects
- Different initialization

### Variant 2: Different Train/Val Split
- Val split: 0.15 (vs default 0.2)
- Tests generalization
- Different data distribution

### Variant 3: On-Target Focus
- Loss weights: on-target=0.7, off-target=0.3
- Optimizes for on-target task
- Higher SCC/PCC

### Variant 4: Off-Target Focus
- Loss weights: on-target=0.3, off-target=0.7
- Optimizes for off-target task
- Higher AUROC/AUPRC

## Ensemble Methods

### 1. Simple Averaging

**Formula:**
```
pred = mean([model_1(x), model_2(x), ..., model_n(x)])
```

**Pros:**
- Simple and fast
- No training required
- Baseline method

**Cons:**
- Treats all models equally
- Ignores model performance differences

**Usage:**
```python
on_target, off_target = ensemble.predict_simple_average(sgrna_onehot, sgrna_label)
```

### 2. Weighted Averaging

**Formula:**
```
pred = sum(w_i * model_i(x)) for all i
where w_i = normalized validation performance
```

**Pros:**
- Accounts for model quality
- Better than simple averaging
- No training required

**Cons:**
- Weights based on validation set
- May overfit to validation set

**Usage:**
```python
on_target, off_target = ensemble.predict_weighted_average(sgrna_onehot, sgrna_label)
```

**Weight Computation:**
```python
ensemble.compute_weights(
    val_scc_scores=[0.75, 0.78, 0.72, 0.76, 0.74],
    val_auroc_scores=[0.82, 0.84, 0.81, 0.83, 0.80]
)
```

### 3. Stacking

**Architecture:**
```
Base Models (5)
    ↓
Predictions (10 values: 5×on-target + 5×off-target)
    ↓
Meta-Model (Neural Network)
    ↓
Final Predictions (2 values: on-target + off-target)
```

**Meta-Model Architecture:**
- Input: 10 (5 models × 2 tasks)
- Hidden: 128 → 64
- Output: 2
- Dropout: 0.3

**Pros:**
- Learns complex combinations
- Best performance potential
- Captures model interactions

**Cons:**
- Requires meta-model training
- Risk of overfitting
- More computation

**Usage:**
```python
on_target, off_target = ensemble.predict_stacking(sgrna_onehot, sgrna_label)
```

**Training Meta-Model:**
```bash
python scripts/train_ensemble.py \
  --n_models 5 \
  --ensemble_method stacking \
  --train_meta_model
```

### 4. Voting

**Formula:**
```
For off-target (binary):
  pred = mean([model_i(x) > 0.5 for all i])
For on-target (regression):
  pred = mean([model_i(x) for all i])
```

**Pros:**
- Simple for classification
- Robust to outliers
- Interpretable

**Cons:**
- Only for binary classification
- Loses probability information
- Less flexible

**Usage:**
```python
on_target, off_target = ensemble.predict_voting(sgrna_onehot, sgrna_label, threshold=0.5)
```

## API Reference

### EnsembleModel Class

#### Initialization

```python
from models.ensemble_model import EnsembleModel, EnsembleConfig

# Create ensemble
ensemble = EnsembleModel(
    model_checkpoints=[
        'models/ensemble/model_0_best.pt',
        'models/ensemble/model_1_best.pt',
        # ... more models
    ],
    config=EnsembleConfig(
        n_models=5,
        ensemble_method='weighted_average',
        device='cuda'
    )
)
```

#### Methods

**predict(sgrna_onehot, sgrna_label, method=None)**
- Make predictions using specified method
- Returns: (on_target_pred, off_target_pred)

**predict_simple_average(sgrna_onehot, sgrna_label)**
- Simple averaging of predictions
- Returns: (on_target_pred, off_target_pred)

**predict_weighted_average(sgrna_onehot, sgrna_label)**
- Weighted averaging based on model performance
- Returns: (on_target_pred, off_target_pred)

**predict_stacking(sgrna_onehot, sgrna_label)**
- Use meta-model for final prediction
- Returns: (on_target_pred, off_target_pred)

**predict_voting(sgrna_onehot, sgrna_label, threshold=0.5)**
- Majority voting for classification
- Returns: (on_target_pred, off_target_pred)

**set_weights(weights)**
- Set ensemble weights manually
- Args: weights (array summing to 1)

**compute_weights(val_scc_scores, val_auroc_scores)**
- Compute weights from validation performance
- Returns: normalized weights

**get_model_predictions(sgrna_onehot, sgrna_label)**
- Get predictions from each individual model
- Returns: dict mapping model index to (on_target, off_target)

**get_prediction_variance(sgrna_onehot, sgrna_label)**
- Get variance of predictions (uncertainty estimate)
- Returns: (on_target_var, off_target_var)

**save_config(save_path)**
- Save ensemble configuration to JSON

**load_config(config_path)** (classmethod)
- Load ensemble from configuration file

## Command-Line Interface

### Training Ensemble

```bash
python scripts/train_ensemble.py [OPTIONS]

Options:
  --base_config PATH          Base configuration file (default: configs/model_config.yaml)
  --n_models N               Number of models to train (default: 5)
  --output_dir PATH          Output directory (default: models/ensemble)
  --ensemble_method METHOD   Ensemble method (default: weighted_average)
  --train_meta_model         Train meta-model for stacking
```

### Examples

**Quick test (2 models):**
```bash
python scripts/train_ensemble.py --n_models 2 --output_dir models/ensemble_test
```

**Standard ensemble (5 models):**
```bash
python scripts/train_ensemble.py --n_models 5 --ensemble_method weighted_average
```

**With stacking:**
```bash
python scripts/train_ensemble.py \
  --n_models 5 \
  --ensemble_method stacking \
  --train_meta_model
```

**Custom output:**
```bash
python scripts/train_ensemble.py \
  --n_models 5 \
  --output_dir results/my_ensemble \
  --base_config configs/custom_config.yaml
```

## Output Files

### Directory Structure

```
models/ensemble/
├── model_0_best.pt          # Variant 0 checkpoint
├── model_1_best.pt          # Variant 1 checkpoint
├── model_2_best.pt          # Variant 2 checkpoint
├── model_3_best.pt          # Variant 3 checkpoint
├── model_4_best.pt          # Variant 4 checkpoint
├── meta_model_best.pt       # Meta-model (if stacking)
├── ensemble_config.json     # Ensemble configuration
├── ensemble_summary.json    # Summary statistics
└── ensemble_training.log    # Training log
```

### ensemble_config.json

```json
{
  "n_models": 5,
  "ensemble_method": "weighted_average",
  "model_checkpoints": [
    "models/ensemble/model_0_best.pt",
    "models/ensemble/model_1_best.pt",
    ...
  ],
  "weights": [0.20, 0.22, 0.18, 0.21, 0.19],
  "model_metrics": [
    {
      "variant_id": 0,
      "on_target_scc": 0.75,
      "off_target_auroc": 0.82,
      ...
    },
    ...
  ],
  "meta_model_path": "models/ensemble/meta_model_best.pt",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Performance

### Training Time

| N Models | Method | Total Time |
|----------|--------|-----------|
| 2 | simple_average | 1-2 days |
| 5 | weighted_average | 3-5 days |
| 5 | stacking | 4-6 days |

### Expected Improvement

- **Simple averaging:** 1-2% over single model
- **Weighted averaging:** 1.5-2.5% over single model
- **Stacking:** 2-3% over single model
- **Voting:** 1-2% over single model (classification only)

### Memory Usage

- Per model: 4-8 GB GPU memory
- Meta-model: <1 GB
- Total for 5 models: ~20-40 GB

### Inference Speed

- Simple averaging: 5x slower than single model
- Weighted averaging: 5x slower than single model
- Stacking: 5.5x slower than single model
- Voting: 5x slower than single model

## Usage Examples

### Example 1: Train and Use Ensemble

```python
from models.ensemble_model import EnsembleModel

# Load trained ensemble
ensemble = EnsembleModel.load_config('models/ensemble/ensemble_config.json')

# Make predictions
on_target_pred, off_target_pred = ensemble.predict(
    sgrna_onehot, sgrna_label,
    method='weighted_average'
)

# Get uncertainty estimates
on_target_var, off_target_var = ensemble.get_prediction_variance(
    sgrna_onehot, sgrna_label
)
```

### Example 2: Compare Ensemble Methods

```python
# Simple averaging
on_target_1, off_target_1 = ensemble.predict_simple_average(sgrna_onehot, sgrna_label)

# Weighted averaging
on_target_2, off_target_2 = ensemble.predict_weighted_average(sgrna_onehot, sgrna_label)

# Stacking
on_target_3, off_target_3 = ensemble.predict_stacking(sgrna_onehot, sgrna_label)

# Voting
on_target_4, off_target_4 = ensemble.predict_voting(sgrna_onehot, sgrna_label)
```

### Example 3: Get Individual Predictions

```python
# Get predictions from each model
predictions = ensemble.get_model_predictions(sgrna_onehot, sgrna_label)

for model_id, (on_target, off_target) in predictions.items():
    print(f"Model {model_id}: on_target={on_target.mean():.4f}, off_target={off_target.mean():.4f}")
```

### Example 4: Custom Weights

```python
# Set custom weights
custom_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
ensemble.set_weights(custom_weights)

# Make predictions with custom weights
on_target, off_target = ensemble.predict_weighted_average(sgrna_onehot, sgrna_label)
```

## Best Practices

### 1. Choose Ensemble Method

- **Simple averaging:** Quick baseline, no training
- **Weighted averaging:** Good balance, recommended
- **Stacking:** Best performance, requires training
- **Voting:** For binary classification only

### 2. Model Diversity

Ensure model variants are diverse:
- Different random seeds
- Different data splits
- Different loss weights
- Different architectures (optional)

### 3. Validation

Always validate ensemble on held-out test set:
```bash
python scripts/evaluate.py \
  --checkpoint models/ensemble/ensemble_config.json \
  --test_data data/processed/combined/test.csv
```

### 4. Monitor Performance

Track individual model performance:
```python
for metrics in ensemble_metrics['model_metrics']:
    print(f"Model {metrics['variant_id']}: SCC={metrics['on_target_scc']:.4f}")
```

### 5. Uncertainty Estimation

Use variance for uncertainty:
```python
on_target_var, off_target_var = ensemble.get_prediction_variance(sgrna_onehot, sgrna_label)

# High variance = uncertain prediction
uncertain_mask = on_target_var > threshold
```

## Troubleshooting

### "CUDA out of memory"

**Cause:** 5 models too large for GPU

**Solution:**
- Use fewer models: `--n_models 3`
- Use smaller batch size
- Use CPU: `device='cpu'`

### "Meta-model not initialized"

**Cause:** Using stacking without training meta-model

**Solution:**
```bash
python scripts/train_ensemble.py --ensemble_method stacking --train_meta_model
```

### "Weights don't sum to 1"

**Cause:** Manual weight setting error

**Solution:**
```python
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
assert np.isclose(np.sum(weights), 1.0)
ensemble.set_weights(weights)
```

## See Also

- `TRAINER_GUIDE.md` - Training orchestration
- `EVALUATION_GUIDE.md` - Evaluation metrics
- `OPTIMIZATION_GUIDE.md` - Learning rate scheduling
- `MONITOR_TRAINING_GUIDE.md` - Real-time monitoring

## References

- Ensemble Learning: https://en.wikipedia.org/wiki/Ensemble_learning
- Stacking: https://en.wikipedia.org/wiki/Stacking_(machine_learning)
- Weighted Averaging: https://en.wikipedia.org/wiki/Weighted_average
