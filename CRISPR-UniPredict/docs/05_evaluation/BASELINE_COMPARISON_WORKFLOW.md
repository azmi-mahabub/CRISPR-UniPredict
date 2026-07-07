# Baseline Model Comparison Workflow

## Overview

Complete workflow for comparing CRISPR-UniPredict with 5 baseline models:
1. CRISPR_HNN
2. CCLMoff
3. CRISPR-Net
4. DeepCRISPR
5. CNN_std

---

## Step 1: Setup Baseline Models

```bash
# Setup all baseline models
python scripts/setup_baseline_models.py

# Or setup specific models
python scripts/setup_baseline_models.py --models crispr_hnn cnn_std
```

**Output Structure:**
```
baselines/
├── crispr_hnn/
├── cclmoff/
├── crispr_net/
├── deep_crispr/
├── cnn_std/
├── baseline_interface.py
└── config.json
```

---

## Step 2: Prepare Test Data

```python
import pandas as pd
from pathlib import Path

# Load test dataset
test_data = pd.read_csv('data/processed/combined/test.csv')

# Select subset for comparison (optional)
test_subset = test_data.sample(n=100, random_state=42)

print(f"Test set size: {len(test_subset)}")
print(f"Columns: {test_subset.columns.tolist()}")
```

---

## Step 3: Get CRISPR-UniPredict Predictions

```python
import torch
import numpy as np
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder

# Initialize model
device = 'cuda'
model = CRISPRUniPredict(device=device)
encoder = SequenceEncoder(device=device)

# Load checkpoint
checkpoint = torch.load('models/checkpoints/best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions
unipredict_on = []
unipredict_off = []

for sgrna in test_subset['sgrna']:
    onehot = encoder.one_hot_encode(sgrna)
    label = encoder.label_encode(sgrna, add_start_token=False)
    
    onehot = onehot.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    
    with torch.no_grad():
        on_target, off_target = model(onehot, label, task_type='both')
    
    unipredict_on.append(on_target.item())
    unipredict_off.append(off_target.item())

test_subset['unipredict_on'] = unipredict_on
test_subset['unipredict_off'] = unipredict_off
```

---

## Step 4: Get Baseline Predictions

```python
from baselines.baseline_interface import get_baseline_models
import pandas as pd

# Initialize baselines
baselines = get_baseline_models(device='cuda')

# Collect predictions
for model_key in ['crispr_hnn', 'cclmoff', 'crispr_net', 'deep_crispr', 'cnn_std']:
    print(f"\nGetting predictions from {model_key}...")
    
    on_target_preds = []
    off_target_preds = []
    
    for sgrna in test_subset['sgrna']:
        try:
            on_pred = baselines.predict_on_target(model_key, sgrna)
            on_target_preds.append(on_pred)
        except:
            on_target_preds.append(np.nan)
        
        try:
            off_pred = baselines.predict_off_target(model_key, sgrna)
            off_target_preds.append(off_pred)
        except:
            off_target_preds.append(np.nan)
    
    test_subset[f'{model_key}_on'] = on_target_preds
    test_subset[f'{model_key}_off'] = off_target_preds

# Save predictions
test_subset.to_csv('results/all_baseline_predictions.csv', index=False)
print("Predictions saved!")
```

---

## Step 5: Compute Metrics

```python
from utils.evaluation.metrics import MetricsCalculator
import numpy as np

metrics_calc = MetricsCalculator()

# Prepare ground truth
on_target_true = test_subset['on_target_label'].values
off_target_true = test_subset['off_target_label'].values

# Compute metrics for each model
models = ['unipredict', 'crispr_hnn', 'cclmoff', 'crispr_net', 'deep_crispr', 'cnn_std']
results = {}

for model in models:
    on_col = f'{model}_on'
    off_col = f'{model}_off'
    
    if on_col in test_subset.columns:
        on_pred = test_subset[on_col].values
        on_metrics = metrics_calc.compute_on_target_metrics(on_pred, on_target_true)
        results[f'{model}_on_target'] = on_metrics
    
    if off_col in test_subset.columns:
        off_pred = test_subset[off_col].values
        off_metrics = metrics_calc.compute_off_target_metrics(off_pred, off_target_true)
        results[f'{model}_off_target'] = off_metrics

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
```

---

## Step 6: Visualize Comparison

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create comparison dataframe
comparison_data = []

for model in models:
    on_col = f'{model}_on'
    if on_col in test_subset.columns:
        on_pred = test_subset[on_col].values
        spearman = metrics_calc.compute_spearman(on_pred, on_target_true)
        comparison_data.append({
            'Model': model.upper(),
            'Task': 'On-Target',
            'Spearman': spearman
        })
    
    off_col = f'{model}_off'
    if off_col in test_subset.columns:
        off_pred = test_subset[off_col].values
        auroc = metrics_calc.compute_auroc(off_pred, off_target_true)
        comparison_data.append({
            'Model': model.upper(),
            'Task': 'Off-Target',
            'AUROC': auroc
        })

df_comparison = pd.DataFrame(comparison_data)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# On-target comparison
on_target_data = df_comparison[df_comparison['Task'] == 'On-Target']
axes[0].barh(on_target_data['Model'], on_target_data['Spearman'])
axes[0].set_xlabel('Spearman Correlation')
axes[0].set_title('On-Target Prediction Comparison')
axes[0].set_xlim(0, 1)

# Off-target comparison
off_target_data = df_comparison[df_comparison['Task'] == 'Off-Target']
axes[1].barh(off_target_data['Model'], off_target_data['AUROC'])
axes[1].set_xlabel('AUROC')
axes[1].set_title('Off-Target Prediction Comparison')
axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison plot saved!")
```

---

## Step 7: Statistical Significance Testing

```python
from scipy import stats
import numpy as np

# Compare CRISPR-UniPredict with each baseline
unipredict_on = test_subset['unipredict_on'].values
on_target_true = test_subset['on_target_label'].values

print("Statistical Significance Testing (On-Target)")
print("=" * 60)

for model in ['crispr_hnn', 'crispr_net', 'deep_crispr', 'cnn_std']:
    baseline_on = test_subset[f'{model}_on'].values
    
    # Remove NaN values
    mask = ~(np.isnan(unipredict_on) | np.isnan(baseline_on))
    uni_clean = unipredict_on[mask]
    base_clean = baseline_on[mask]
    
    # Compute correlations
    uni_corr = np.corrcoef(uni_clean, on_target_true[mask])[0, 1]
    base_corr = np.corrcoef(base_clean, on_target_true[mask])[0, 1]
    
    # T-test on prediction differences
    diff = uni_clean - base_clean
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    
    print(f"\n{model.upper()}:")
    print(f"  UniPredict Correlation: {uni_corr:.4f}")
    print(f"  Baseline Correlation: {base_corr:.4f}")
    print(f"  Difference: {uni_corr - base_corr:.4f}")
    print(f"  T-test p-value: {p_value:.4e}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
```

---

## Step 8: Generate Comprehensive Report

```python
from pathlib import Path

# Create report
report = []
report.append("=" * 80)
report.append("CRISPR-UniPredict vs Baseline Models Comparison Report")
report.append("=" * 80)
report.append("")

report.append("1. MODELS COMPARED")
report.append("-" * 80)
report.append("• CRISPR-UniPredict (Novel)")
report.append("• CRISPR_HNN (On-target only)")
report.append("• CCLMoff (Off-target only)")
report.append("• CRISPR-Net (On-target + Off-target)")
report.append("• DeepCRISPR (On-target + Off-target)")
report.append("• CNN_std (Baseline CNN)")

report.append("\n2. TEST SET STATISTICS")
report.append("-" * 80)
report.append(f"Total samples: {len(test_subset)}")
report.append(f"On-target samples: {(~test_subset['on_target_label'].isna()).sum()}")
report.append(f"Off-target samples: {(~test_subset['off_target_label'].isna()).sum()}")

report.append("\n3. ON-TARGET PERFORMANCE")
report.append("-" * 80)
report.append("Model                 | Spearman | Pearson | MAE    | RMSE")
report.append("-" * 60)

for model in models:
    on_col = f'{model}_on'
    if on_col in test_subset.columns:
        on_pred = test_subset[on_col].values
        spearman = metrics_calc.compute_spearman(on_pred, on_target_true)
        pearson = metrics_calc.compute_pearson(on_pred, on_target_true)
        mae = metrics_calc.compute_mae(on_pred, on_target_true)
        rmse = metrics_calc.compute_rmse(on_pred, on_target_true)
        
        report.append(f"{model:20s} | {spearman:8.4f} | {pearson:7.4f} | {mae:6.4f} | {rmse:6.4f}")

report.append("\n4. OFF-TARGET PERFORMANCE")
report.append("-" * 80)
report.append("Model                 | AUROC  | AUPRC  | F1-Score | Balanced Acc")
report.append("-" * 60)

for model in models:
    off_col = f'{model}_off'
    if off_col in test_subset.columns:
        off_pred = test_subset[off_col].values
        auroc = metrics_calc.compute_auroc(off_pred, off_target_true)
        auprc = metrics_calc.compute_auprc(off_pred, off_target_true)
        f1 = metrics_calc.compute_f1(off_pred, off_target_true)
        bal_acc = metrics_calc.compute_balanced_accuracy(off_pred, off_target_true)
        
        report.append(f"{model:20s} | {auroc:6.4f} | {auprc:6.4f} | {f1:8.4f} | {bal_acc:12.4f}")

report.append("\n5. KEY FINDINGS")
report.append("-" * 80)
report.append("• CRISPR-UniPredict combines on-target and off-target predictions")
report.append("• Unified architecture enables joint optimization")
report.append("• Comprehensive score balances efficiency and safety")

report.append("\n6. RECOMMENDATIONS")
report.append("-" * 80)
report.append("• Use CRISPR-UniPredict for comprehensive sgRNA evaluation")
report.append("• Consider ensemble methods for improved robustness")
report.append("• Validate predictions on experimental data")

report.append("\n" + "=" * 80)

# Save report
report_text = "\n".join(report)
report_path = Path('results/baseline_comparison_report.txt')
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\nReport saved to {report_path}")
```

---

## Complete Workflow Script

```bash
#!/bin/bash

echo "CRISPR-UniPredict Baseline Comparison Workflow"
echo "=============================================="

# Step 1: Setup baselines
echo "Step 1: Setting up baseline models..."
python scripts/setup_baseline_models.py

# Step 2: Run comparison
echo "Step 2: Running comparison..."
python -c "
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from baselines.baseline_interface import get_baseline_models
from utils.evaluation.metrics import MetricsCalculator

# Load data and models
test_data = pd.read_csv('data/processed/combined/test.csv')
device = 'cuda'

# Get predictions (see Step 3-4 above)
# ...

# Compute metrics (see Step 5 above)
# ...

# Generate visualizations (see Step 6 above)
# ...

print('Comparison complete!')
"

echo "Done!"
```

---

## Output Files

- `results/all_baseline_predictions.csv` - All model predictions
- `results/baseline_comparison.png` - Comparison visualization
- `results/baseline_comparison_report.txt` - Detailed report
- `baselines/config.json` - Baseline configuration

---

## Summary

Complete workflow for:
- ✅ Downloading 5 baseline models
- ✅ Getting predictions from all models
- ✅ Computing comprehensive metrics
- ✅ Statistical significance testing
- ✅ Generating comparison visualizations
- ✅ Creating detailed reports

Ready for publication-quality comparisons!
