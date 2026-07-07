# Training Script Guide

## Overview

The `scripts/train.py` is the main entry point for training CRISPR-UniPredict. It handles:

- Configuration loading
- Device setup (CPU/GPU)
- DataLoader creation
- Model initialization
- Optimizer and scheduler setup
- Training orchestration
- Logging and visualization
- Results saving

---

## Quick Start

### Basic Training

```bash
python scripts/train.py --config configs/model_config.yaml --experiment_name exp_001
```

### Resume Training

```bash
python scripts/train.py --config configs/model_config.yaml --resume models/checkpoints/best.pt
```

### Debug Mode (Quick Testing)

```bash
python scripts/train.py --config configs/model_config.yaml --debug
```

---

## Command-Line Arguments

### `--config` (required)

Path to configuration YAML file.

```bash
python scripts/train.py --config configs/model_config.yaml
```

**Default**: `configs/model_config.yaml`

### `--experiment_name`

Name for this training experiment.

```bash
python scripts/train.py --experiment_name exp_001
```

**Default**: `default_experiment`

**Output directory**: `logs/exp_001_YYYYMMDD_HHMMSS/`

### `--resume`

Path to checkpoint to resume training from.

```bash
python scripts/train.py --resume models/checkpoints/best.pt
```

**Default**: `None` (start fresh)

**When to use**:
- Resume interrupted training
- Fine-tune from pretrained model
- Continue from best checkpoint

### `--debug`

Use small subset for quick testing.

```bash
python scripts/train.py --debug
```

**What it does**:
- Limits train/val/test to 2 batches each
- Useful for testing pipeline
- Fast iteration during development

### `--seed`

Random seed for reproducibility.

```bash
python scripts/train.py --seed 42
```

**Default**: `42`

---

## Output Structure

### Experiment Directory

```
logs/
└── exp_001_20250122_120000/
    ├── training.log              # Training logs
    ├── config.json               # Configuration used
    ├── args.json                 # Command-line arguments
    ├── summary.json              # Training summary
    ├── training_history.json     # Detailed history
    └── training_history.png      # Loss plots
```

### Checkpoints

```
models/
└── checkpoints/
    ├── latest.pt                 # Latest checkpoint
    ├── best.pt                   # Best model
    └── checkpoint_epoch_10.pt    # Periodic checkpoints
```

---

## Output Files

### `training.log`

Complete training log with timestamps.

```
2025-01-22 12:00:00 - CRISPR-UniPredict - INFO - Experiment: exp_001
2025-01-22 12:00:01 - CRISPR-UniPredict - INFO - Loading configuration from configs/model_config.yaml
2025-01-22 12:00:02 - CRISPR-UniPredict - INFO - Using GPU: NVIDIA A100
...
```

### `config.json`

Configuration used for training.

```json
{
  "model": {...},
  "training": {...},
  "data": {...},
  "device": {...}
}
```

### `args.json`

Command-line arguments used.

```json
{
  "config": "configs/model_config.yaml",
  "experiment_name": "exp_001",
  "resume": null,
  "debug": false,
  "seed": 42
}
```

### `summary.json`

Training summary statistics.

```json
{
  "total_epochs": 50,
  "best_val_loss": 0.42,
  "final_train_loss": 0.35,
  "final_val_loss": 0.40,
  "train_losses": [...],
  "val_losses": [...]
}
```

### `training_history.json`

Detailed training history per epoch.

```json
{
  "train": [
    {"total_loss": 0.50, "on_target_loss": 0.35, ...},
    {"total_loss": 0.45, "on_target_loss": 0.30, ...}
  ],
  "val": [...]
}
```

### `training_history.png`

4-panel plot showing:
1. Total loss (train vs val)
2. On-target loss
3. Off-target loss
4. Loss ratio

---

## Complete Examples

### Example 1: Full Training

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --experiment_name full_training_v1 \
  --seed 42
```

**Output**:
```
logs/full_training_v1_20250122_120000/
├── training.log
├── config.json
├── summary.json
├── training_history.json
└── training_history.png
```

### Example 2: Resume Training

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --resume models/checkpoints/best.pt \
  --experiment_name resumed_training
```

**What happens**:
- Loads best model from checkpoint
- Resumes from saved epoch
- Continues training
- Saves new checkpoints

### Example 3: Debug Mode

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --debug
```

**What happens**:
- Uses only 2 batches per split
- Completes in seconds
- Tests full pipeline
- Useful for development

### Example 4: Custom Seed

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --experiment_name exp_seed_123 \
  --seed 123
```

**What happens**:
- Sets all random seeds to 123
- Ensures reproducibility
- Different results than seed 42

---

## Training Process

### Step 1: Setup

```
1. Parse arguments
2. Create experiment directory
3. Setup logging
4. Set random seeds
5. Load configuration
6. Save config and args
```

### Step 2: Initialization

```
7. Setup device (CPU/GPU)
8. Create dataloaders
9. Initialize model
10. Setup optimizer and scheduler
11. Initialize trainer
```

### Step 3: Training

```
12. For each epoch:
    - Train one epoch
    - Validate
    - Log metrics
    - Save checkpoint
    - Step scheduler
13. Save best model
```

### Step 4: Finalization

```
14. Plot training history
15. Save summary statistics
16. Print results
```

---

## Logging

### Console Output

Real-time training progress:

```
Epoch 1/100 - Train Loss: 0.523456, Val Loss: 0.512345
Epoch 2/100 - Train Loss: 0.456789, Val Loss: 0.445678
...
Best model saved at epoch 15
```

### File Logging

Detailed logs saved to `training.log`:

```
2025-01-22 12:00:00 - CRISPR-UniPredict - INFO - Experiment: exp_001
2025-01-22 12:00:01 - CRISPR-UniPredict - INFO - Using GPU: NVIDIA A100
2025-01-22 12:00:02 - CRISPR-UniPredict - INFO - Model initialized
2025-01-22 12:00:03 - CRISPR-UniPredict - INFO - Starting training...
```

### TensorBoard

View real-time metrics:

```bash
tensorboard --logdir=logs
```

### WandB

View on wandb.ai (if enabled in config):

```bash
# View at https://wandb.ai/your_entity/CRISPR-UniPredict
```

---

## Reproducibility

### Setting Seeds

```bash
python scripts/train.py --seed 42
```

**What gets seeded**:
- NumPy random
- PyTorch random
- CUDA random
- cuDNN deterministic mode

### Reproducible Results

Same seed = same results:

```bash
# Run 1
python scripts/train.py --seed 42 --experiment_name run1

# Run 2
python scripts/train.py --seed 42 --experiment_name run2

# run1 and run2 will have identical results
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size

```yaml
training:
  batch_size: 16  # Reduce from 32
```

**Solution 2**: Use CPU

```bash
python scripts/train.py --config configs/model_config.yaml
# (Set use_cuda: false in config)
```

### Issue: "Config file not found"

**Solution**: Check path

```bash
# Verify file exists
ls configs/model_config.yaml

# Use absolute path
python scripts/train.py --config /absolute/path/to/config.yaml
```

### Issue: "Checkpoint not found"

**Solution**: Verify checkpoint path

```bash
# List available checkpoints
ls models/checkpoints/

# Use correct path
python scripts/train.py --resume models/checkpoints/best.pt
```

### Issue: "Training diverges (NaN loss)"

**Solution**: Reduce learning rate

```yaml
training:
  learning_rate_encoder: 1.0e-4
  learning_rate_heads: 5.0e-4
```

### Issue: "Training too slow"

**Solution 1**: Increase workers

```yaml
data:
  num_workers: 8
```

**Solution 2**: Enable mixed precision

```yaml
device:
  mixed_precision: true
```

---

## Advanced Usage

### Custom Configuration

Create custom config file:

```yaml
# configs/custom_config.yaml
training:
  batch_size: 64
  epochs: 200
  learning_rate_heads: 5.0e-4
```

Use it:

```bash
python scripts/train.py --config configs/custom_config.yaml
```

### Multiple Experiments

Run multiple experiments:

```bash
for seed in 42 123 456; do
  python scripts/train.py \
    --experiment_name multi_seed_$seed \
    --seed $seed
done
```

### Hyperparameter Sweep

Test different learning rates:

```bash
for lr in 1e-4 5e-4 1e-3; do
  # Modify config or use command-line override
  python scripts/train.py \
    --experiment_name lr_$lr
done
```

---

## Performance Tips

### 1. Use GPU

```bash
# Ensure GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU in config
device:
  use_cuda: true
```

### 2. Enable Mixed Precision

```yaml
device:
  mixed_precision: true
```

**Benefits**:
- 2x faster training
- 50% less memory
- Same accuracy

### 3. Increase Workers

```yaml
data:
  num_workers: 8  # Match CPU cores
```

### 4. Pin Memory

```yaml
data:
  pin_memory: true  # For GPU training
```

### 5. Larger Batch Size

```yaml
training:
  batch_size: 64  # Larger batches = faster
```

---

## Summary

The training script provides:
- ✅ **Easy entry point** for training
- ✅ **Configuration management** from YAML
- ✅ **Reproducibility** with seed control
- ✅ **Checkpoint management** for resuming
- ✅ **Comprehensive logging** to file and console
- ✅ **Visualization** of training history
- ✅ **Error handling** and validation
- ✅ **Debug mode** for quick testing

Perfect for training CRISPR-UniPredict models!
