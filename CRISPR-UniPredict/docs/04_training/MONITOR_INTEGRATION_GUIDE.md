# Training Monitor Integration Guide

## Overview

The `monitor_training.py` script integrates seamlessly with the existing training pipeline. No modifications to the training script are needed—the monitor reads from the standard training output files.

## Architecture

```
Training Script (train.py)
        ↓
    Creates logs/
        ├── training.log
        ├── config.json
        └── training_history.json
        ↓
Monitor Script (monitor_training.py)
        ↓
    Reads metrics and displays:
        ├── Console output
        ├── Matplotlib dashboard
        └── Web dashboard (Dash)
```

## Quick Integration

### Step 1: Start Training

```bash
python scripts/train.py \
  --config configs/model_config.yaml \
  --experiment_name my_experiment
```

This creates a log directory like: `logs/my_experiment_20240101_120000/`

### Step 2: Start Monitoring (in another terminal)

```bash
# Get the exact log directory from the training output
python scripts/monitor_training.py \
  --log_dir logs/my_experiment_20240101_120000 \
  --mode console
```

That's it! The monitor will automatically read and display training metrics.

## Detailed Integration

### File Structure

After training starts, the following files are created:

```
logs/
└── my_experiment_20240101_120000/
    ├── training.log              # Training log (read by monitor)
    ├── config.json               # Configuration (read by monitor)
    ├── training_history.json     # Metrics (read by monitor) ← KEY FILE
    ├── checkpoints/
    │   ├── latest.pt
    │   └── best.pt
    └── dashboard_snapshots/      # Created if --save_snapshots
        └── dashboard_*.png
```

### Key File: training_history.json

The monitor reads this file, which is automatically created by the trainer:

```json
{
  "train": [
    {
      "total_loss": 0.5234,
      "on_target_loss": 0.3123,
      "off_target_loss": 0.2111,
      "batch_count": 100
    },
    {
      "total_loss": 0.4856,
      "on_target_loss": 0.2945,
      "off_target_loss": 0.1911,
      "batch_count": 100
    }
  ],
  "val": [
    {
      "total_loss": 0.4856,
      "on_target_loss": 0.2945,
      "off_target_loss": 0.1911,
      "on_target_spearman": 0.7234,
      "on_target_pearson": 0.7456,
      "off_target_auroc": 0.8234,
      "off_target_auprc": 0.7856,
      "batch_count": 50
    }
  ]
}
```

## Usage Patterns

### Pattern 1: Local Development

**Terminal 1 (Training):**
```bash
python scripts/train.py --config configs/model_config.yaml --experiment_name exp_001
```

**Terminal 2 (Monitoring):**
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode matplotlib
```

**Benefits:**
- Interactive dashboard
- Real-time visualization
- Easy to see trends

### Pattern 2: Remote Server (SSH)

**SSH Terminal (Training):**
```bash
python scripts/train.py --config configs/model_config.yaml --experiment_name exp_001
```

**Local Terminal (Monitoring via SSH):**
```bash
ssh user@server "python scripts/monitor_training.py --log_dir logs/exp_001_* --mode console"
```

**Benefits:**
- Minimal bandwidth
- Works over slow connections
- No GUI needed

### Pattern 3: Web-Based Monitoring

**Server Terminal (Training + Monitoring):**
```bash
# Terminal 1: Start training
python scripts/train.py --config configs/model_config.yaml --experiment_name exp_001

# Terminal 2: Start web dashboard
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash --port 8050
```

**Local Browser:**
```
http://server_ip:8050
```

**Benefits:**
- Access from any machine
- No SSH needed
- Team collaboration

### Pattern 4: Continuous Monitoring

**Automated Script (`monitor_continuous.sh`):**
```bash
#!/bin/bash
LOG_DIR=$1
REFRESH_RATE=${2:-10}

while true; do
    python scripts/monitor_training.py \
        --log_dir "$LOG_DIR" \
        --refresh_rate "$REFRESH_RATE" \
        --mode console
    
    sleep 60  # Restart if monitor crashes
done
```

**Usage:**
```bash
./monitor_continuous.sh logs/exp_001_* 10
```

## Monitoring Different Stages

### During Training

Monitor real-time metrics:
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode console --refresh_rate 5
```

**What to watch:**
- Training loss decreasing
- Validation loss decreasing
- No NaN values
- GPU/memory stable

### After Each Epoch

Review epoch-level metrics:
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode matplotlib --save_snapshots
```

**What to check:**
- Best model checkpoint
- Improvement trend
- Validation metrics
- Resource usage

### Post-Training Analysis

Analyze complete training:
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode matplotlib
```

**What to analyze:**
- Learning curves
- Convergence speed
- Final metrics
- Training stability

## Advanced Integration

### Custom Alert Thresholds

Edit `monitor_training.py` to customize alerts:

```python
# In TrainingMonitor.check_for_issues()

# Change early stopping patience
self.early_stopping_patience = 15  # Default: 10

# Change memory warning threshold
if system_metrics['memory_percent'] > 85:  # Default: 90
    alerts.append("⚠️ WARNING: High memory usage")

# Change loss plateau detection
if train_std < 1e-5:  # Default: 1e-6
    alerts.append("⚠️ WARNING: Training loss plateau")
```

### Integration with Experiment Tracking

Combine with experiment tracking tools:

```bash
# With MLflow
mlflow run . -P epochs=50

# Monitor in parallel
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash

# View in MLflow UI
mlflow ui
```

### Automated Report Generation

Create a report script (`generate_report.py`):

```python
from scripts.monitor_training import TrainingMonitor
import json

monitor = TrainingMonitor('logs/exp_001_*')
monitor.load_metrics()
summary = monitor.get_summary()

report = {
    'experiment': 'exp_001',
    'total_epochs': summary['total_epochs'],
    'best_epoch': summary['best_epoch'],
    'best_loss': summary['best_val_loss'],
    'final_loss': summary['current_val_loss'],
    'improvement': summary['epochs_since_improvement']
}

with open('report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

## Troubleshooting Integration

### Monitor Can't Find Metrics

**Problem:** "Waiting for training data..."

**Cause:** Training hasn't created `training_history.json` yet

**Solution:**
1. Wait for first epoch to complete
2. Verify log directory path is correct
3. Check that training script is running

```bash
# Verify files exist
ls -la logs/exp_001_*/training_history.json
```

### Metrics Not Updating

**Problem:** Monitor shows old data

**Cause:** Training paused or crashed

**Solution:**
1. Check training process: `ps aux | grep train.py`
2. Check training log: `tail -f logs/exp_001_*/training.log`
3. Verify GPU/memory: `nvidia-smi` or `htop`

### Dashboard Crashes

**Problem:** Matplotlib/Dash window closes unexpectedly

**Cause:** Corrupted metrics file or memory issue

**Solution:**
```bash
# Verify metrics file is valid JSON
python -m json.tool logs/exp_001_*/training_history.json

# Increase refresh rate to reduce memory
python scripts/monitor_training.py --log_dir logs/exp_001_* --refresh_rate 30
```

## Performance Optimization

### For Large Experiments

If monitoring large experiments with many epochs:

```bash
# Increase refresh rate to reduce CPU
python scripts/monitor_training.py \
    --log_dir logs/exp_001_* \
    --mode console \
    --refresh_rate 30  # Update every 30 seconds instead of 10
```

### For Remote Monitoring

If monitoring over SSH with limited bandwidth:

```bash
# Use console mode (minimal data transfer)
ssh user@server "python scripts/monitor_training.py \
    --log_dir logs/exp_001_* \
    --mode console \
    --refresh_rate 30"
```

### For Multiple Experiments

Monitor multiple experiments simultaneously:

```bash
# Terminal 1
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash --port 8050

# Terminal 2
python scripts/monitor_training.py --log_dir logs/exp_002_* --mode dash --port 8051

# Terminal 3
python scripts/monitor_training.py --log_dir logs/exp_003_* --mode dash --port 8052
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Monitor Training

on: [workflow_dispatch]

jobs:
  train-and-monitor:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Start training
        run: |
          python scripts/train.py \
            --config configs/model_config.yaml \
            --experiment_name ci_exp &
          
          # Give training time to start
          sleep 10
          
          # Monitor training
          python scripts/monitor_training.py \
            --log_dir logs/ci_exp_* \
            --mode console \
            --refresh_rate 30
```

### GitLab CI

```yaml
train_and_monitor:
  script:
    - python scripts/train.py --config configs/model_config.yaml --experiment_name ci_exp &
    - sleep 10
    - python scripts/monitor_training.py --log_dir logs/ci_exp_* --mode console
```

## Best Practices

### 1. Always Monitor Training

```bash
# Start monitoring immediately after training
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode console
```

### 2. Save Snapshots for Documentation

```bash
# Save dashboard snapshots for reports
python scripts/monitor_training.py \
    --log_dir logs/exp_001_* \
    --mode matplotlib \
    --save_snapshots
```

### 3. Use Appropriate Mode for Environment

| Environment | Recommended Mode | Reason |
|-------------|-----------------|--------|
| Local development | matplotlib | Interactive, visual |
| SSH/Remote | console | Minimal overhead |
| Team collaboration | dash | Web-accessible |
| Automated pipelines | console | No GUI needed |

### 4. Monitor System Resources

Always check GPU/memory to prevent OOM:

```bash
# In another terminal
watch -n 1 nvidia-smi  # GPU
watch -n 1 free -h     # Memory
```

### 5. Set Up Alerts

Configure alert thresholds for your use case:

```python
# Edit monitor_training.py
# Adjust thresholds based on your hardware
```

## See Also

- `MONITOR_TRAINING_GUIDE.md` - Complete monitoring guide
- `MONITOR_QUICK_START.txt` - Quick reference
- `TRAINER_GUIDE.md` - Training orchestration
- `TRAINING_SCRIPT_GUIDE.md` - Training script details

## Support

For integration issues:

1. **Verify training is running:**
   ```bash
   ps aux | grep train.py
   ```

2. **Check log directory exists:**
   ```bash
   ls -la logs/exp_001_*/
   ```

3. **Verify metrics file:**
   ```bash
   python -m json.tool logs/exp_001_*/training_history.json
   ```

4. **Check system resources:**
   ```bash
   nvidia-smi  # GPU
   htop        # CPU/Memory
   ```

5. **Review training log:**
   ```bash
   tail -f logs/exp_001_*/training.log
   ```
