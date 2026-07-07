# Hyperparameter Tuning Guide

## Overview

The `scripts/hyperparameter_tuning.py` script automatically finds optimal hyperparameters for CRISPR-UniPredict using Optuna's Bayesian optimization (Tree-structured Parzen Estimator).

## Installation

```bash
# Install Optuna
pip install optuna

# Or with all dependencies
pip install optuna matplotlib seaborn pyyaml
```

## Quick Start

### Basic Usage

```bash
# Run 50 trials with default settings
python scripts/hyperparameter_tuning.py --n_trials 50

# Run with custom study name
python scripts/hyperparameter_tuning.py --n_trials 100 --study_name my_optimization

# Resume previous study
python scripts/hyperparameter_tuning.py --resume --n_trials 50
```

### With Persistent Storage

```bash
# Save study to SQLite database
python scripts/hyperparameter_tuning.py \
  --n_trials 50 \
  --storage sqlite:///optuna_study.db \
  --study_name crispr_optimization
```

## Hyperparameters Tuned

### Learning Rates
- **rna_fm_lr**: [1e-5, 1e-3] (log scale)
  - Learning rate for RNA-FM encoder
  - Default: 1e-4

- **feature_extraction_lr**: [5e-4, 5e-3] (log scale)
  - Learning rate for feature extraction branch
  - Default: 1e-3

- **task_heads_lr**: [5e-4, 5e-3] (log scale)
  - Learning rate for task-specific heads
  - Default: 1e-3

### Architecture Parameters
- **msc_out_channels**: [32, 64, 128]
  - Output channels for Multi-Scale Convolution
  - Default: 64

- **bigru_hidden_dim**: [64, 128, 256]
  - Hidden dimension for Bidirectional GRU
  - Default: 128

- **mhsa_num_heads**: [2, 4, 8]
  - Number of attention heads
  - Default: 4

- **dropout_rate**: [0.2, 0.5] (step: 0.05)
  - Dropout probability
  - Default: 0.35

### Training Parameters
- **batch_size**: [64, 128, 256]
  - Training batch size
  - Default: 32

- **warmup_epochs**: [3, 5, 10]
  - Number of warmup epochs
  - Default: 5

- **loss_weight_on_target**: [0.3, 0.7] (step: 0.1)
  - Weight for on-target loss
  - Default: 0.5

- **loss_weight_off_target**: [0.3, 0.7] (step: 0.1)
  - Weight for off-target loss
  - Default: 0.5

### Optimizer Parameters
- **weight_decay**: [1e-5, 1e-3] (log scale)
  - L2 regularization coefficient
  - Default: 1e-4

- **gradient_clip**: [0.5, 2.0] (step: 0.1)
  - Gradient clipping threshold
  - Default: 1.0

### Scheduler Parameters
- **scheduler_patience**: [3, 5, 7]
  - Patience for learning rate scheduler
  - Default: 5

- **scheduler_factor**: [0.1, 0.5] (step: 0.1)
  - Factor to multiply learning rate
  - Default: 0.1

## Optimization Strategy

### Objective Function

The optimization maximizes a combined metric:

```
Combined Score = (SCC + AUROC) / 2
```

Where:
- **SCC** = Spearman Correlation Coefficient (on-target task)
- **AUROC** = Area Under ROC Curve (off-target task)

This balances performance on both tasks.

### Search Algorithm

**TPE Sampler (Tree-structured Parzen Estimator)**
- Bayesian optimization method
- Efficient exploration-exploitation trade-off
- Good for high-dimensional spaces

**Median Pruner**
- Stops unpromising trials early
- Saves computation time
- Parameters:
  - `n_startup_trials=5`: Use first 5 trials to establish baseline
  - `n_warmup_steps=5`: Warmup period before pruning

### Training Configuration

Each trial:
1. Trains for maximum 20 epochs (early stopping)
2. Evaluates on validation set
3. Returns combined score
4. Can be pruned if performing poorly

## Command-Line Arguments

```
--config PATH           Base configuration file (default: configs/model_config.yaml)
--n_trials N           Number of trials to run (default: 50)
--study_name NAME      Optuna study name (default: crispr_optimization)
--storage URL          Storage URL for persistence (default: None)
--output_dir PATH      Output directory (default: results/hyperparameter_tuning)
--resume               Resume previous study (default: False)
```

## Output Files

### Results Directory: `results/hyperparameter_tuning/`

1. **best_hyperparameters.yaml**
   - Best hyperparameters found
   - Ready to use in training
   - Format: YAML

2. **all_trials.json**
   - Complete results for all trials
   - Includes hyperparameters and metrics
   - Format: JSON

3. **optuna_study.pkl**
   - Optuna study object
   - For resuming optimization
   - Format: Pickle

4. **summary.json**
   - Quick summary of optimization
   - Best score and trial ID
   - Timestamp

5. **optimization_report.png**
   - Comprehensive visualization
   - 8 subplots showing:
     - Optimization history
     - Parameter importance
     - Score distribution
     - SCC vs AUROC scatter
     - Batch size effect
     - Learning rate effect
     - Dropout effect
     - Score over time

6. **tuning.log**
   - Detailed log of all trials
   - Hyperparameters and metrics
   - Errors and warnings

## Usage Examples

### Example 1: Quick Optimization (10 trials)

```bash
python scripts/hyperparameter_tuning.py --n_trials 10
```

**Expected time:** 2-3 hours
**Use case:** Quick testing, parameter ranges

### Example 2: Standard Optimization (50 trials)

```bash
python scripts/hyperparameter_tuning.py \
  --n_trials 50 \
  --study_name standard_optimization
```

**Expected time:** 3-5 days
**Use case:** Production optimization

### Example 3: Extended Optimization (100 trials)

```bash
python scripts/hyperparameter_tuning.py \
  --n_trials 100 \
  --storage sqlite:///optuna_study.db \
  --study_name extended_optimization
```

**Expected time:** 1-2 weeks
**Use case:** Thorough exploration

### Example 4: Resume Previous Study

```bash
# Original run
python scripts/hyperparameter_tuning.py \
  --n_trials 50 \
  --storage sqlite:///optuna_study.db

# Resume with 50 more trials
python scripts/hyperparameter_tuning.py \
  --n_trials 50 \
  --storage sqlite:///optuna_study.db \
  --resume
```

### Example 5: Custom Configuration

```bash
python scripts/hyperparameter_tuning.py \
  --config configs/custom_config.yaml \
  --n_trials 50 \
  --output_dir results/custom_tuning \
  --study_name custom_study
```

## Interpreting Results

### Optimization History

The optimization history plot shows:
- X-axis: Trial number
- Y-axis: Combined score
- Red line: Best score found

**Good signs:**
- Steady improvement over trials
- Best score reached mid-optimization
- Some plateau at the end

**Warning signs:**
- No improvement after many trials
- High variance in scores
- Best score at first trial

### Parameter Importance

Shows which hyperparameters have the most impact:
- Top parameters should be tuned carefully
- Low-importance parameters can use defaults
- Use for sensitivity analysis

### Score Distribution

Histogram of all trial scores:
- Narrow distribution: Consistent performance
- Wide distribution: High variance
- Skewed distribution: Some parameters critical

### SCC vs AUROC Scatter

Relationship between on-target and off-target metrics:
- Positive correlation: Balanced improvement
- Negative correlation: Trade-off between tasks
- Outliers: Interesting hyperparameter combinations

## Best Practices

### 1. Start with Baseline

```bash
# First, establish baseline with default hyperparameters
python scripts/train.py --config configs/model_config.yaml

# Then optimize
python scripts/hyperparameter_tuning.py --n_trials 50
```

### 2. Use Persistent Storage

```bash
# Save to database for long-running optimizations
python scripts/hyperparameter_tuning.py \
  --storage sqlite:///optuna_study.db \
  --n_trials 100
```

### 3. Monitor Progress

```bash
# Check results while running
tail -f results/hyperparameter_tuning/tuning.log

# View current best
cat results/hyperparameter_tuning/summary.json
```

### 4. Adjust Search Space

If optimization is stuck:
- Expand search ranges
- Add more hyperparameters
- Increase number of trials

If optimization is too slow:
- Reduce search ranges
- Decrease number of trials
- Use fewer epochs per trial

### 5. Validate Results

After optimization:

```bash
# Train with best hyperparameters
python scripts/train.py \
  --config results/hyperparameter_tuning/best_hyperparameters.yaml

# Compare with baseline
# Check if improvement is significant
```

## Performance Considerations

### Time Estimates

| Trials | Epochs/Trial | Total Time |
|--------|-------------|-----------|
| 10 | 20 | 2-3 hours |
| 50 | 20 | 3-5 days |
| 100 | 20 | 1-2 weeks |

**Factors affecting time:**
- Dataset size
- Model complexity
- GPU availability
- Early stopping effectiveness

### Memory Usage

- Per trial: ~4-8 GB (depending on batch size)
- Study storage: ~100 MB (for 50 trials)
- Results directory: ~500 MB

### GPU Utilization

- Single GPU: 100% during training
- Multiple GPUs: Not supported (single-process)
- CPU fallback: Supported but slow

## Troubleshooting

### "Trial failed: CUDA out of memory"

**Cause:** Batch size too large

**Solution:**
```python
# Edit hyperparameter_tuning.py
# Reduce batch size range
batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
```

### "No improvement after many trials"

**Cause:** Search space too narrow or plateau reached

**Solution:**
1. Expand search ranges
2. Increase number of trials
3. Check if baseline is already optimal

### "Study not found"

**Cause:** Wrong storage URL or database deleted

**Solution:**
```bash
# Start new study
python scripts/hyperparameter_tuning.py \
  --storage sqlite:///optuna_study_new.db
```

### "Trials are very slow"

**Cause:** Too many epochs per trial

**Solution:**
```python
# Edit objective function
max_epochs = min(10, trial_config.get('training', {}).get('epochs', 50))
```

## Advanced Usage

### Custom Objective Function

Modify the `objective` method to:
- Use different metrics
- Add constraints
- Implement multi-objective optimization

### Hyperparameter Importance

```python
# After optimization
importance = optuna.importance.get_param_importances(study)
for param, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {value:.4f}")
```

### Visualization

```python
# Create custom visualizations
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_slice(study).show()
```

## Integration with Training

### Use Best Hyperparameters

```bash
# 1. Run optimization
python scripts/hyperparameter_tuning.py --n_trials 50

# 2. Train with best hyperparameters
python scripts/train.py \
  --config results/hyperparameter_tuning/best_hyperparameters.yaml \
  --experiment_name final_model

# 3. Evaluate
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv
```

### Automated Pipeline

Create `run_optimization.sh`:

```bash
#!/bin/bash

echo "Starting hyperparameter optimization..."
python scripts/hyperparameter_tuning.py \
  --n_trials 50 \
  --storage sqlite:///optuna_study.db

echo "Training with best hyperparameters..."
python scripts/train.py \
  --config results/hyperparameter_tuning/best_hyperparameters.yaml \
  --experiment_name optimized_model

echo "Evaluating..."
python scripts/evaluate.py \
  --checkpoint models/checkpoints/best.pt \
  --test_data data/processed/combined/test.csv

echo "Done!"
```

## See Also

- `TRAINER_GUIDE.md` - Training orchestration
- `TRAINING_SCRIPT_GUIDE.md` - Training script details
- `OPTIMIZATION_GUIDE.md` - Learning rate scheduling
- `MONITOR_TRAINING_GUIDE.md` - Real-time monitoring

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html)
- [Pruning](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
- [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)

## Support

For issues:
1. Check `results/hyperparameter_tuning/tuning.log`
2. Review trial results in `all_trials.json`
3. Check Optuna documentation
4. Verify GPU availability and memory
