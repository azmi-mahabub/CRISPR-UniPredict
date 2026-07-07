# Real-Time Training Monitor Guide

## Overview

The `scripts/monitor_training.py` script provides real-time monitoring of CRISPR-UniPredict training with three different interfaces:

1. **Console Mode** - Terminal-based monitoring (lightweight, no dependencies)
2. **Matplotlib Mode** - Interactive dashboard with live plots
3. **Dash Mode** - Web-based dashboard accessible via browser

## Installation

### Basic Requirements (Console Mode)
```bash
pip install psutil torch numpy
```

### Matplotlib Mode
```bash
pip install matplotlib seaborn
```

### Dash Mode (Web Dashboard)
```bash
pip install dash plotly
```

### All Features
```bash
pip install psutil torch numpy matplotlib seaborn dash plotly
```

## Usage

### Console Mode (Recommended for SSH/Remote)

Monitor training progress in the terminal with real-time updates:

```bash
python scripts/monitor_training.py \
  --log_dir logs/exp_001_20240101_120000 \
  --refresh_rate 10 \
  --mode console
```

**Output:**
```
================================================================================
CRISPR-UniPredict Training Monitor - 2024-01-01 12:30:45
================================================================================

📊 Training Progress:
  Epochs completed: 15
  Best epoch: 12 (Loss: 0.234567)
  Current loss: 0.245678
  Epochs since improvement: 3

💻 System Resources:
  CPU: 45.2%
  RAM: 62.3% (31.5 GB)
  GPU: 87.5%
  VRAM: 8.45 GB / 12.00 GB

✅ No issues detected
================================================================================
```

### Matplotlib Mode (Interactive Dashboard)

Launch an interactive matplotlib dashboard with live-updating plots:

```bash
python scripts/monitor_training.py \
  --log_dir logs/exp_001_20240101_120000 \
  --refresh_rate 10 \
  --mode matplotlib \
  --save_snapshots
```

**Features:**
- 2x3 grid of real-time plots
- Auto-refreshing every N seconds
- Save snapshots to `logs/exp_001_20240101_120000/dashboard_snapshots/`
- Close with Ctrl+C

**Dashboard Layout:**
```
┌─────────────────┬──────────────────┬──────────────────┐
│  Training Loss  │  Val Metrics     │  Learning Rate   │
├─────────────────┼──────────────────┼──────────────────┤
│  GPU Util       │  Memory Usage    │  Gradient Norm   │
├─────────────────┼──────────────────┼──────────────────┤
│ On-Target Corr  │ Off-Target Metrics│  Alerts/Status   │
└─────────────────┴──────────────────┴──────────────────┘
```

### Dash Mode (Web Dashboard)

Launch a web-based dashboard accessible via browser:

```bash
python scripts/monitor_training.py \
  --log_dir logs/exp_001_20240101_120000 \
  --refresh_rate 10 \
  --mode dash \
  --port 8050
```

Then open your browser to: **http://localhost:8050**

**Features:**
- Real-time updates every 10 seconds
- Interactive plots (zoom, pan, hover for values)
- Responsive design
- Accessible from any machine on the network

## Monitored Metrics

### Training Metrics
- **Train Loss** - Total training loss per epoch
- **Validation Loss** - Total validation loss per epoch
- **On-Target Loss** - Regression loss for on-target task
- **Off-Target Loss** - Classification loss for off-target task

### Validation Metrics
- **SCC** - Spearman's Correlation Coefficient (on-target)
- **PCC** - Pearson's Correlation Coefficient (on-target)
- **AUROC** - Area Under ROC Curve (off-target)
- **AUPRC** - Area Under Precision-Recall Curve (off-target)

### System Resources
- **CPU Usage** - Percentage of CPU utilization
- **RAM Usage** - Percentage and absolute GB of memory
- **GPU Utilization** - Percentage of GPU usage
- **VRAM Usage** - Absolute GB of GPU memory used
- **Gradient Norm** - L2 norm of gradients (inferred)

## Alert System

The monitor automatically detects and alerts on:

### Critical Alerts 🚨
- **Gradient Explosion** - Loss becomes NaN
- **High Memory Usage** - >90% RAM or VRAM (OOM risk)

### Warnings ⚠️
- **Training Stalled** - No improvement for N epochs (early stopping patience)
- **Validation Degradation** - Loss increasing over recent epochs
- **Gradient Vanishing** - Training loss plateau detected
- **Performance Degradation** - Validation metrics declining

### Info ℹ️
- **Improvement Tracking** - Best model checkpoint and loss
- **Patience Counter** - Epochs since last improvement

## Command-Line Arguments

```
--log_dir PATH          Training log directory (required)
--refresh_rate SECONDS  Update frequency (default: 10)
--mode MODE            console|matplotlib|dash (default: console)
--save_snapshots       Save matplotlib snapshots to disk
--port PORT            Port for Dash app (default: 8050)
```

## Integration with Training Script

The monitor reads from the standard training output:

```
logs/
└── exp_001_20240101_120000/
    ├── training.log           # Training log file
    ├── config.json            # Configuration
    ├── training_history.json  # Metrics (read by monitor)
    └── dashboard_snapshots/   # Saved plots (if --save_snapshots)
```

The `training_history.json` file is automatically created by the trainer and contains:

```json
{
  "train": [
    {
      "total_loss": 0.5234,
      "on_target_loss": 0.3123,
      "off_target_loss": 0.2111,
      "batch_count": 100
    },
    ...
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
    },
    ...
  ]
}
```

## Usage Examples

### Example 1: Monitor Training in Terminal

Start training in one terminal:
```bash
python scripts/train.py --config configs/model_config.yaml --experiment_name exp_001
```

Monitor in another terminal:
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode console
```

### Example 2: Web Dashboard During Training

```bash
# Terminal 1: Start training
python scripts/train.py --config configs/model_config.yaml

# Terminal 2: Start web dashboard
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash --port 8050

# Then open browser: http://localhost:8050
```

### Example 3: Save Dashboard Snapshots

```bash
python scripts/monitor_training.py \
  --log_dir logs/exp_001_* \
  --mode matplotlib \
  --save_snapshots \
  --refresh_rate 30
```

Snapshots saved to: `logs/exp_001_*/dashboard_snapshots/dashboard_YYYYMMDD_HHMMSS.png`

### Example 4: Fast Updates (5 seconds)

```bash
python scripts/monitor_training.py \
  --log_dir logs/exp_001_* \
  --refresh_rate 5 \
  --mode console
```

## Performance Considerations

### Console Mode
- **CPU Usage**: <1%
- **Memory**: ~50 MB
- **Best for**: SSH sessions, remote servers, minimal overhead

### Matplotlib Mode
- **CPU Usage**: 2-5%
- **Memory**: ~200 MB
- **Best for**: Local monitoring, interactive analysis

### Dash Mode
- **CPU Usage**: 3-8%
- **Memory**: ~300 MB
- **Best for**: Web access, team monitoring, presentations

## Troubleshooting

### "No metrics found" or "Waiting for training data..."

**Cause**: The `training_history.json` file hasn't been created yet.

**Solution**: 
- Ensure training script is running
- Check that log directory path is correct
- Wait a few seconds for first epoch to complete

### Dash app won't start

**Cause**: Dash not installed or port already in use

**Solution**:
```bash
# Install Dash
pip install dash plotly

# Use different port
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash --port 8051
```

### Matplotlib window not updating

**Cause**: Interactive mode not enabled

**Solution**: Ensure matplotlib backend supports interactive mode:
```bash
# On Windows
python -c "import matplotlib; print(matplotlib.get_backend())"

# Should show: TkAgg, Qt5Agg, or similar (not Agg)
```

### High CPU usage in matplotlib mode

**Cause**: Refresh rate too fast

**Solution**: Increase refresh rate:
```bash
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode matplotlib --refresh_rate 30
```

### GPU metrics not showing

**Cause**: CUDA not available or GPU not in use

**Solution**:
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure training uses GPU: Check `config.device.use_cuda`

## Advanced Usage

### Monitor Multiple Experiments

```bash
# Terminal 1
python scripts/monitor_training.py --log_dir logs/exp_001_* --mode dash --port 8050

# Terminal 2
python scripts/monitor_training.py --log_dir logs/exp_002_* --mode dash --port 8051

# Open browsers:
# http://localhost:8050 (exp_001)
# http://localhost:8051 (exp_002)
```

### Automated Monitoring Script

Create `monitor_all.sh`:
```bash
#!/bin/bash
LOG_DIR=$1
REFRESH_RATE=${2:-10}
MODE=${3:-console}

python scripts/monitor_training.py \
  --log_dir "$LOG_DIR" \
  --refresh_rate "$REFRESH_RATE" \
  --mode "$MODE" \
  --save_snapshots
```

Usage:
```bash
./monitor_all.sh logs/exp_001_* 10 matplotlib
```

### Custom Alert Thresholds

Edit the `check_for_issues()` method in `monitor_training.py` to customize:
- Early stopping patience threshold
- Memory warning threshold (default: 90%)
- Loss plateau detection sensitivity
- Gradient vanishing threshold

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Monitor Training
on: [workflow_dispatch]

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Start monitoring
        run: |
          python scripts/monitor_training.py \
            --log_dir logs/exp_001_* \
            --mode console \
            --refresh_rate 30
```

## Performance Tips

1. **For Remote Monitoring**: Use console mode (lowest overhead)
2. **For Real-time Analysis**: Use Dash mode (best interactivity)
3. **For Presentations**: Use matplotlib with snapshots
4. **Increase Refresh Rate**: If CPU usage is high (trade-off with responsiveness)
5. **Disable GPU Monitoring**: If not using GPU (saves ~5% CPU)

## See Also

- `TRAINER_GUIDE.md` - Training orchestration details
- `TRAINING_SCRIPT_GUIDE.md` - Training script documentation
- `EVALUATION_GUIDE.md` - Evaluation metrics explanation
- `OPTIMIZATION_GUIDE.md` - Learning rate scheduling

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review training logs: `logs/exp_*/training.log`
3. Verify metrics file exists: `logs/exp_*/training_history.json`
4. Check system resources: `htop` (Linux) or Task Manager (Windows)
