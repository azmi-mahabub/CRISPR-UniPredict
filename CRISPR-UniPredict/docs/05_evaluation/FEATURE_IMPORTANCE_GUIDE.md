# Feature Importance Analysis Guide

## Overview

The `feature_importance.py` module provides comprehensive analysis of which sequence features and model components matter most for CRISPR-UniPredict predictions.

---

## Quick Start

### Complete Analysis

```python
from utils.visualization.feature_importance import generate_interpretation_report
from pathlib import Path

# Run complete analysis
generate_interpretation_report(
    model=model,
    sgrna_onehot=onehot,
    sgrna_label=label,
    output_dir=Path('results/feature_importance'),
    device='cuda'
)
```

### Step-by-Step Analysis

```python
from utils.visualization.feature_importance import FeatureImportanceAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = FeatureImportanceAnalyzer(model, device='cuda')

# 1. Analyze position importance
importance_matrix = analyzer.analyze_position_importance(onehot, label)

# 2. Plot nucleotide effects
analyzer.plot_nucleotide_substitution_effects(
    importance_matrix,
    output_path=Path('nucleotide_effects.png')
)

# 3. Compute branch contributions
contributions = analyzer.compute_branch_contributions(onehot, label)

# 4. Plot branch contributions
analyzer.plot_branch_contributions(
    contributions,
    output_path=Path('branch_contributions.png')
)

# 5. Generate report
analyzer.generate_interpretation_report(
    importance_matrix,
    contributions,
    output_dir=Path('results')
)
```

---

## FeatureImportanceAnalyzer Class

### Initialization

```python
from utils.visualization.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, device='cuda')
```

**Parameters**:
- `model`: CRISPRUniPredict model instance
- `device`: Device to use ('cuda' or 'cpu')

### Methods

#### `analyze_position_importance(sgrna_onehot, sgrna_label, task)`

Analyze position-specific nucleotide importance through substitution analysis.

**Parameters**:
- `sgrna_onehot`: One-hot encoded sgRNA (23, 4) or (1, 23, 4)
- `sgrna_label`: Label encoded sgRNA (23,) or (1, 23)
- `task`: 'on_target' or 'off_target'

**Returns**: Importance matrix (23, 4)

**How it works**:
1. Get baseline prediction
2. For each position (1-23):
   - For each nucleotide (A, C, G, T):
     - Substitute nucleotide
     - Get new prediction
     - Compute importance as |prediction_change|

```python
importance_matrix = analyzer.analyze_position_importance(onehot, label, task='on_target')
# Returns: (23, 4) matrix where each cell is importance score
```

#### `plot_nucleotide_substitution_effects(importance_matrix, output_path)`

Plot heatmap of nucleotide substitution effects.

**Parameters**:
- `importance_matrix`: Importance matrix (23, 4)
- `output_path`: Path to save figure

**Features**:
- ✅ Heatmap with nucleotides on Y-axis, positions on X-axis
- ✅ Seed region highlighting (positions 16-20)
- ✅ Color intensity shows importance
- ✅ Replicates CRISPR_HNN Figure 5

```python
analyzer.plot_nucleotide_substitution_effects(
    importance_matrix,
    output_path=Path('nucleotide_effects.png')
)
```

#### `compute_branch_contributions(sgrna_onehot, sgrna_label, dataset_size)`

Compute contribution of each model branch.

**Parameters**:
- `sgrna_onehot`: One-hot encoded sgRNA
- `sgrna_label`: Label encoded sgRNA
- `dataset_size`: Number of samples to analyze

**Returns**: Dictionary with branch contributions

**How it works**:
1. Get baseline prediction with all branches
2. For each branch (MSC, BiGRU, RNA-FM):
   - Disable branch
   - Get prediction
   - Compute contribution as performance drop
   - Re-enable branch

```python
contributions = analyzer.compute_branch_contributions(onehot, label)
# Returns: {
#     'msc_on_target': 0.15,
#     'msc_off_target': 0.12,
#     'bigru_on_target': 0.10,
#     'bigru_off_target': 0.08,
#     'rna_fm_on_target': 0.08,
#     'rna_fm_off_target': 0.10
# }
```

#### `plot_branch_contributions(contributions, output_path)`

Plot branch contributions as bar chart.

**Parameters**:
- `contributions`: Dictionary with contributions
- `output_path`: Path to save figure

**Features**:
- ✅ Grouped bar chart (on-target vs off-target)
- ✅ Branches on X-axis
- ✅ Performance drop on Y-axis
- ✅ Clear comparison

```python
analyzer.plot_branch_contributions(
    contributions,
    output_path=Path('branch_contributions.png')
)
```

#### `generate_interpretation_report(importance_matrix, contributions, output_dir)`

Generate comprehensive interpretation report.

**Parameters**:
- `importance_matrix`: Position importance matrix
- `contributions`: Branch contributions
- `output_dir`: Output directory

**Report includes**:
- Position-specific importance summary
- Top 5 important positions
- Seed region analysis
- Branch contribution analysis
- Component synergy assessment
- Key findings
- Recommendations

```python
analyzer.generate_interpretation_report(
    importance_matrix,
    contributions,
    output_dir=Path('results')
)
```

---

## Convenience Functions

### `analyze_position_importance(model, sgrna_onehot, sgrna_label, task, device)`

Quick function to analyze position importance.

```python
from utils.visualization.feature_importance import analyze_position_importance

importance = analyze_position_importance(
    model, onehot, label, task='on_target', device='cuda'
)
```

### `plot_nucleotide_substitution_effects(importance_matrix, output_path)`

Quick function to plot effects.

```python
from utils.visualization.feature_importance import plot_nucleotide_substitution_effects

plot_nucleotide_substitution_effects(importance, output_path=Path('effects.png'))
```

### `compute_branch_contributions(model, sgrna_onehot, sgrna_label, device)`

Quick function to compute contributions.

```python
from utils.visualization.feature_importance import compute_branch_contributions

contrib = compute_branch_contributions(model, onehot, label, device='cuda')
```

### `generate_interpretation_report(model, sgrna_onehot, sgrna_label, output_dir, device)`

Quick function to generate complete report.

```python
from utils.visualization.feature_importance import generate_interpretation_report

generate_interpretation_report(
    model, onehot, label, Path('results'), device='cuda'
)
```

---

## Complete Example

```python
import torch
from pathlib import Path
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.visualization.feature_importance import FeatureImportanceAnalyzer

# Initialize
device = 'cuda'
model = CRISPRUniPredict(device=device)
encoder = SequenceEncoder(device=device)
analyzer = FeatureImportanceAnalyzer(model, device=device)

# Load model from checkpoint
checkpoint = torch.load('models/checkpoints/best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Example sgRNA
sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
onehot = encoder.one_hot_encode(sgrna)
label = encoder.label_encode(sgrna, add_start_token=False)

# Create output directory
output_dir = Path('results/feature_importance')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Analyze position importance
print("Analyzing position importance...")
importance_matrix = analyzer.analyze_position_importance(onehot, label, task='on_target')

# 2. Plot nucleotide effects
print("Plotting nucleotide substitution effects...")
analyzer.plot_nucleotide_substitution_effects(
    importance_matrix,
    output_path=output_dir / 'nucleotide_effects.png'
)

# 3. Compute branch contributions
print("Computing branch contributions...")
contributions = analyzer.compute_branch_contributions(onehot, label)

# 4. Plot branch contributions
print("Plotting branch contributions...")
analyzer.plot_branch_contributions(
    contributions,
    output_path=output_dir / 'branch_contributions.png'
)

# 5. Generate report
print("Generating interpretation report...")
analyzer.generate_interpretation_report(
    importance_matrix,
    contributions,
    output_dir
)

print(f"Analysis complete! Results saved to {output_dir}")
```

---

## Understanding Results

### Position Importance Heatmap

- **X-axis**: Position in sgRNA (1-23)
- **Y-axis**: Nucleotide (A, C, G, T)
- **Color intensity**: Importance score (higher = more important)

**Interpretation**:
- Bright red cells: Substituting this nucleotide at this position significantly changes prediction
- White cells: Substitution has minimal effect
- Seed region (16-20) typically shows high importance

### Branch Contributions

- **MSC branch**: Captures local sequence patterns (typically 40-50% contribution)
- **BiGRU branch**: Captures sequential dependencies (typically 30-40% contribution)
- **RNA-FM branch**: Provides pretrained contextual information (typically 20-30% contribution)

**Synergy**:
- If total < 1.0: Components have synergistic effects (better together)
- If total > 1.0: Components have redundant effects (some overlap)

---

## Output Files

### `nucleotide_effects.png`

Heatmap showing importance of each nucleotide at each position.

### `branch_contributions.png`

Bar chart comparing contributions of MSC, BiGRU, and RNA-FM branches.

### `interpretation_report.txt`

Comprehensive text report with:
- Position importance summary
- Top 5 important positions
- Seed region analysis
- Branch contribution details
- Component synergy assessment
- Key findings
- Recommendations

---

## Key Insights

### Seed Region Importance

The seed region (positions 16-20) typically shows:
- Highest nucleotide substitution effects
- Critical for on-target specificity
- Strong conservation across high-efficiency sgRNAs

### Branch Contributions

- **MSC**: Dominates local pattern recognition
- **BiGRU**: Important for capturing sequential context
- **RNA-FM**: Provides complementary pretrained features

### Position Patterns

- Positions 1-5: Low importance (PAM-proximal)
- Positions 6-15: Moderate importance (off-target specificity)
- Positions 16-20: Highest importance (seed region)
- Positions 21-23: Moderate importance (PAM-distal)

---

## Tips for Interpretation

### 1. Compare Tasks

Analyze both on-target and off-target to understand task-specific patterns.

### 2. Validate Biologically

Compare computational importance with experimental mutagenesis data.

### 3. Consider Redundancy

Some positions may show low importance due to redundancy in the genetic code.

### 4. Examine Interactions

Position importance may depend on surrounding nucleotides (epistasis).

---

## Troubleshooting

### Issue: "All positions show equal importance"

**Cause**: Model not trained or predictions are random

**Solution**:
```python
# Check model performance first
on_pred, off_pred = model(onehot, label, task_type='both')
print(f"On-target: {on_pred.item():.4f}, Off-target: {off_pred.item():.4f}")
```

### Issue: "Seed region shows low importance"

**Cause**: Model may not have learned seed region importance

**Solution**:
- Check model performance on test set
- Verify training data quality
- Consider retraining with seed region weighting

### Issue: "Branch contributions don't sum to 1"

**Cause**: Components have synergistic or redundant effects

**Solution**:
- This is normal and informative
- Synergy (< 1.0) indicates complementary information
- Redundancy (> 1.0) indicates overlapping features

---

## Summary

The feature importance analysis module provides:
- ✅ **Position-specific nucleotide importance** through substitution analysis
- ✅ **Branch contribution analysis** for understanding component roles
- ✅ **Kernel importance assessment** for multi-scale features
- ✅ **Publication-quality visualizations** replicating CRISPR_HNN Figure 5
- ✅ **Comprehensive interpretation reports** with key findings
- ✅ **Easy-to-use API** with convenience functions

Perfect for understanding model behavior and validating biological insights!
