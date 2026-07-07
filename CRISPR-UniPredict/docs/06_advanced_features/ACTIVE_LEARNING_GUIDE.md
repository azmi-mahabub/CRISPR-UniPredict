# Active Learning Guide

## Overview

Active learning identifies the most informative unlabeled samples for additional labeling/validation. This guide explains how to use the active learning system to efficiently improve model performance.

## Installation

```bash
# No additional dependencies needed
# Uses existing scikit-learn, PyTorch, and model infrastructure
```

## Quick Start

### Run Active Learning (5 iterations)

```bash
python scripts/active_learning.py \
  --initial_model models/checkpoints/best.pt \
  --unlabeled_data data/unlabeled_sequences.csv \
  --n_iterations 5 \
  --samples_per_iteration 1000
```

### With Ensemble

```bash
python scripts/active_learning.py \
  --initial_model models/checkpoints/best.pt \
  --unlabeled_data data/unlabeled_sequences.csv \
  --ensemble_config models/ensemble/ensemble_config.json \
  --n_iterations 5 \
  --samples_per_iteration 1000
```

## Sampling Strategies

### 1. Uncertainty Sampling

**Concept:** Select samples with highest prediction uncertainty

**For Regression (On-Target):**
- Measure variance across ensemble members
- High variance = uncertain prediction

**For Classification (Off-Target):**
- Probability near 0.5 = uncertain
- Distance from decision boundary

**Pros:**
- Identifies ambiguous cases
- Focuses on model weaknesses
- Simple to implement

**Cons:**
- May select outliers
- Doesn't consider data distribution

**Usage:**
```python
learner = ActiveLearner(model, unlabeled_data)
uncertain_indices = learner.uncertainty_sampling(n_samples=100)
```

### 2. Diversity Sampling

**Concept:** Select diverse samples from different regions of feature space

**Method:**
1. Get embeddings from RNA-FM encoder
2. Apply K-means clustering
3. Select representative from each cluster

**Pros:**
- Covers feature space well
- Reduces redundancy
- Explores new regions

**Cons:**
- Computationally expensive
- May select easy samples
- Requires embedding model

**Usage:**
```python
diverse_indices = learner.diversity_sampling(n_samples=100, n_clusters=50)
```

### 3. Disagreement Sampling

**Concept:** Select samples where ensemble members disagree

**Method:**
- Compare predictions between models
- High variance = high disagreement
- Likely challenging cases

**Pros:**
- Identifies hard examples
- Requires ensemble
- Focuses on model disagreement

**Cons:**
- Needs ensemble model
- May select outliers
- Computationally expensive

**Usage:**
```python
disagreement_indices = learner.disagreement_sampling(n_samples=100)
```

### 4. Combined Strategy

**Concept:** Combine all three strategies with weights

**Formula:**
```
selected = weighted_combination(
    uncertainty_samples,
    diversity_samples,
    disagreement_samples
)
```

**Default Weights:**
- Uncertainty: 0.4 (40%)
- Diversity: 0.4 (40%)
- Disagreement: 0.2 (20%)

**Customizable:**
```python
selected_indices, selected_data = learner.query_next_batch(
    n_samples=1000,
    uncertainty_weight=0.5,
    diversity_weight=0.3,
    disagreement_weight=0.2
)
```

## Workflow

### Step 1: Train Initial Model

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --experiment_name initial_model
```

### Step 2: Prepare Unlabeled Data

Create CSV with columns:
- `sgrna`: sgRNA sequence
- `target`: Target sequence (optional)
- Other metadata

```csv
sgrna,target,source
GCCGCGCGCGCGCGCGCGCGCGC,TGCCGCGCGCGCGCGCGCGCGCG,dataset1
ATCGATCGATCGATCGATCGATC,TATCGATCGATCGATCGATCGAT,dataset2
```

### Step 3: Run Active Learning

```bash
python scripts/active_learning.py \
  --initial_model models/checkpoints/best.pt \
  --unlabeled_data data/unlabeled_sequences.csv \
  --n_iterations 5 \
  --samples_per_iteration 1000
```

### Step 4: Get Experimental Labels

Selected samples saved to:
```
results/active_learning/selected_samples_iter_1.csv
results/active_learning/selected_samples_iter_2.csv
...
```

Get experimental validation for these samples.

### Step 5: Retrain Model

Combine original and newly labeled data:

```bash
# Combine datasets
python scripts/combine_datasets.py \
  --original data/training_data.csv \
  --new results/active_learning/selected_samples_iter_1.csv \
  --output data/training_data_v2.csv

# Retrain
python scripts/train.py \
  --config configs/model_config.yaml \
  --experiment_name model_v2
```

### Step 6: Evaluate Improvement

```bash
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/test_set.csv
```

### Step 7: Repeat

Continue iterations until performance plateaus.

## Command-Line Interface

```bash
python scripts/active_learning.py [OPTIONS]

Options:
  --initial_model PATH          Initial model checkpoint (required)
  --unlabeled_data PATH         Unlabeled sequences CSV (required)
  --ensemble_config PATH        Ensemble configuration (optional)
  --n_iterations N             Number of iterations (default: 5)
  --samples_per_iteration N    Samples per iteration (default: 1000)
  --output_dir PATH            Output directory (default: results/active_learning)
  --uncertainty_weight FLOAT   Uncertainty weight (default: 0.4)
  --diversity_weight FLOAT     Diversity weight (default: 0.4)
  --disagreement_weight FLOAT  Disagreement weight (default: 0.2)
```

## Output Files

### Directory Structure

```
results/active_learning/
├── selected_samples_iter_1.csv    # Iteration 1 selected samples
├── selected_samples_iter_2.csv    # Iteration 2 selected samples
├── selected_samples_iter_3.csv    # Iteration 3 selected samples
├── selected_samples_iter_4.csv    # Iteration 4 selected samples
├── selected_samples_iter_5.csv    # Iteration 5 selected samples
├── query_history.json             # Query history
├── active_learning_report.png     # Visualization report
└── active_learning.log            # Detailed log
```

### selected_samples_iter_N.csv

Contains:
- All columns from original unlabeled data
- `selected_index`: Index in original dataset
- Ready for experimental validation

### query_history.json

```json
[
  {
    "iteration": 1,
    "timestamp": "2024-01-01T12:00:00",
    "n_samples": 1000,
    "indices": [0, 5, 12, 23, ...]
  },
  {
    "iteration": 2,
    "timestamp": "2024-01-02T12:00:00",
    "n_samples": 1000,
    "indices": [100, 105, 112, 123, ...]
  }
]
```

## API Reference

### ActiveLearner Class

#### Initialization

```python
from scripts.active_learning import ActiveLearner

learner = ActiveLearner(
    model_checkpoint='models/checkpoints/best.pt',
    unlabeled_data_path='data/unlabeled_sequences.csv',
    ensemble_checkpoint=None,  # Optional
    output_dir='results/active_learning',
    device='cuda'
)
```

#### Methods

**uncertainty_sampling(n_samples=100)**
- Select most uncertain predictions
- Returns: array of indices

**diversity_sampling(n_samples=100, n_clusters=None)**
- Select diverse samples
- Returns: array of indices

**disagreement_sampling(n_samples=100)**
- Select samples with ensemble disagreement
- Returns: array of indices

**query_next_batch(n_samples=100, uncertainty_weight=0.4, diversity_weight=0.4, disagreement_weight=0.2)**
- Combined strategy
- Returns: (indices, selected_data)

**save_query_results(selected_indices, selected_data)**
- Save selected samples to CSV

**generate_report()**
- Generate visualization report

**print_summary()**
- Print active learning summary

## Usage Examples

### Example 1: Basic Active Learning

```python
from scripts.active_learning import ActiveLearner

# Create learner
learner = ActiveLearner(
    model_checkpoint='models/checkpoints/best.pt',
    unlabeled_data_path='data/unlabeled_sequences.csv'
)

# Query samples
for iteration in range(5):
    indices, data = learner.query_next_batch(n_samples=1000)
    learner.save_query_results(indices, data)
    
    # Get labels for these samples
    # Retrain model
    # Update learner
```

### Example 2: With Ensemble

```python
learner = ActiveLearner(
    model_checkpoint='models/checkpoints/best.pt',
    unlabeled_data_path='data/unlabeled_sequences.csv',
    ensemble_checkpoint='models/ensemble/ensemble_config.json'
)

# Ensemble enables disagreement sampling
indices, data = learner.query_next_batch(
    n_samples=1000,
    uncertainty_weight=0.3,
    diversity_weight=0.3,
    disagreement_weight=0.4  # Higher weight with ensemble
)
```

### Example 3: Custom Weights

```python
# Focus on uncertainty
indices, data = learner.query_next_batch(
    n_samples=1000,
    uncertainty_weight=0.8,
    diversity_weight=0.1,
    disagreement_weight=0.1
)

# Focus on diversity
indices, data = learner.query_next_batch(
    n_samples=1000,
    uncertainty_weight=0.2,
    diversity_weight=0.7,
    disagreement_weight=0.1
)
```

## Performance Considerations

### Computational Cost

| Strategy | Time | Memory | Notes |
|----------|------|--------|-------|
| Uncertainty | Fast | Low | Per-sample inference |
| Diversity | Slow | Medium | Embedding + clustering |
| Disagreement | Slow | Medium | Ensemble inference |
| Combined | Slow | Medium | All strategies |

### Scalability

- **1K samples:** <1 minute
- **10K samples:** 5-10 minutes
- **100K samples:** 1-2 hours
- **1M samples:** 10-20 hours

### Memory Usage

- Per sample: ~1 MB (with embeddings)
- 10K samples: ~10 GB
- 100K samples: ~100 GB

## Best Practices

### 1. Start with Combined Strategy

Use all three strategies with default weights:
```bash
python scripts/active_learning.py \
  --initial_model models/checkpoints/best.pt \
  --unlabeled_data data/unlabeled_sequences.csv \
  --n_iterations 5 \
  --samples_per_iteration 1000
```

### 2. Use Ensemble for Better Results

Ensemble enables disagreement sampling:
```bash
python scripts/active_learning.py \
  --initial_model models/checkpoints/best.pt \
  --unlabeled_data data/unlabeled_sequences.csv \
  --ensemble_config models/ensemble/ensemble_config.json \
  --disagreement_weight 0.4
```

### 3. Adjust Weights Based on Goal

- **Maximize coverage:** Increase diversity_weight
- **Focus on hard cases:** Increase disagreement_weight
- **Quick improvement:** Increase uncertainty_weight

### 4. Monitor Progress

Track metrics across iterations:
```python
# Compare model performance
# Iteration 1: SCC=0.75, AUROC=0.82
# Iteration 2: SCC=0.76, AUROC=0.83
# Iteration 3: SCC=0.77, AUROC=0.84
```

### 5. Stop When Plateauing

Continue until performance improvement slows:
```python
improvements = [0.01, 0.01, 0.005, 0.002, 0.001]
# Stop when improvement < 0.005
```

## Troubleshooting

### "CUDA out of memory"

**Cause:** Processing too many samples at once

**Solution:**
- Reduce samples_per_iteration
- Use CPU: `device='cpu'`
- Process in batches

### "RNA-FM embeddings not available"

**Cause:** RNA-FM model not installed

**Solution:**
- Install RNA-FM: See INSTALL_RNA_FM.md
- Use diversity_weight=0 to skip diversity sampling
- Use uncertainty + disagreement only

### "Ensemble not available"

**Cause:** Ensemble config not provided

**Solution:**
- Train ensemble first: `python scripts/train_ensemble.py`
- Provide ensemble_config: `--ensemble_config models/ensemble/ensemble_config.json`
- Or use uncertainty + diversity without disagreement

### "No samples selected"

**Cause:** All samples already selected

**Solution:**
- Increase unlabeled dataset size
- Reduce samples_per_iteration
- Check query_history.json for coverage

## Advanced Usage

### Custom Sampling Strategy

```python
class CustomActiveLearner(ActiveLearner):
    def custom_sampling(self, n_samples):
        """Implement custom sampling"""
        # Your logic here
        return indices
    
    def query_next_batch(self, n_samples=100):
        """Override with custom strategy"""
        return self.custom_sampling(n_samples)
```

### Integration with Hyperparameter Tuning

```bash
# 1. Tune hyperparameters
python scripts/hyperparameter_tuning.py --n_trials 50

# 2. Use best hyperparameters
python scripts/train.py --config results/hyperparameter_tuning/best_hyperparameters.yaml

# 3. Run active learning
python scripts/active_learning.py --initial_model models/checkpoints/best.pt ...
```

## See Also

- `TRAINER_GUIDE.md` - Training orchestration
- `ENSEMBLE_GUIDE.md` - Ensemble models
- `EVALUATION_GUIDE.md` - Evaluation metrics
- `HYPERPARAMETER_TUNING_GUIDE.md` - Hyperparameter optimization

## References

- Active Learning: https://en.wikipedia.org/wiki/Active_learning_(machine_learning)
- Uncertainty Sampling: https://en.wikipedia.org/wiki/Uncertainty_sampling
- Query by Committee: https://en.wikipedia.org/wiki/Query_by_committee
- Diversity Sampling: https://en.wikipedia.org/wiki/Diversity_(statistics)
