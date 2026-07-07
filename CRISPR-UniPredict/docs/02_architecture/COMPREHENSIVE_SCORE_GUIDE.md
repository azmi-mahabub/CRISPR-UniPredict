# Comprehensive sgRNA Score Guide

## Overview

The `comprehensive_score.py` module visualizes the novel comprehensive sgRNA score that combines on-target efficiency with off-target safety.

**Comprehensive Score Formula:**
```
Score = On_target_efficiency × (1 - Off_target_risk)
```

---

## Quick Start

### Complete Analysis

```python
from utils.visualization.comprehensive_score import generate_comprehensive_report
from pathlib import Path

# Run complete analysis
generate_comprehensive_report(
    on_target_pred=on_target_predictions,
    off_target_pred=off_target_predictions,
    sgrnas=sgrna_sequences,
    output_dir=Path('results/comprehensive_score')
)
```

### Step-by-Step

```python
from utils.visualization.comprehensive_score import ComprehensiveScoreVisualizer
from pathlib import Path

# Initialize visualizer
visualizer = ComprehensiveScoreVisualizer()

# 1. Compute scores
scores = visualizer.compute_comprehensive_score(on_target_pred, off_target_pred)

# 2. Plot 2D scatter
visualizer.plot_2d_scatter(on_target_pred, off_target_pred, sgrnas,
                          output_path=Path('2d_scatter.png'))

# 3. Plot distribution
visualizer.plot_score_distribution(on_target_pred, off_target_pred,
                                  output_path=Path('distribution.png'))

# 4. Compare rankings
visualizer.plot_ranking_comparison(on_target_pred, off_target_pred, sgrnas,
                                  output_path=Path('ranking_comparison.png'))

# 5. Interactive plot
visualizer.plot_interactive_scatter(on_target_pred, off_target_pred, sgrnas,
                                   output_path=Path('interactive.html'))

# 6. Generate report
visualizer.generate_score_report(on_target_pred, off_target_pred, sgrnas,
                                output_dir=Path('results'))
```

---

## ComprehensiveScoreVisualizer Class

### Initialization

```python
from utils.visualization.comprehensive_score import ComprehensiveScoreVisualizer

visualizer = ComprehensiveScoreVisualizer(device='cuda')
```

**Parameters**:
- `device`: Device to use ('cuda' or 'cpu')

### Methods

#### `compute_comprehensive_score(on_target_pred, off_target_pred)`

Compute comprehensive score.

**Formula**: `Score = On_target × (1 - Off_target)`

**Parameters**:
- `on_target_pred`: On-target efficiency (0-1)
- `off_target_pred`: Off-target probability (0-1)

**Returns**: Comprehensive scores (0-1)

```python
scores = visualizer.compute_comprehensive_score(on_target_pred, off_target_pred)
# Returns array of scores between 0 and 1
```

#### `plot_2d_scatter(on_target_pred, off_target_pred, sgrnas, output_path)`

Plot 2D scatter with quadrant analysis.

**Features**:
- ✅ X-axis: On-target efficiency
- ✅ Y-axis: Off-target safety (1 - risk)
- ✅ Color: Comprehensive score (red-yellow-green)
- ✅ Quadrant annotations
- ✅ Identifies optimal sgRNAs (top-right)

**Quadrants**:
- **Top-Right (Optimal)**: High efficiency, high safety
- **Top-Left (Safe but Inefficient)**: Low efficiency, high safety
- **Bottom-Right (Efficient but Risky)**: High efficiency, low safety
- **Bottom-Left (Poor)**: Low efficiency, low safety

```python
visualizer.plot_2d_scatter(
    on_target_pred,
    off_target_pred,
    sgrnas=sgrna_sequences,
    output_path=Path('2d_scatter.png')
)
```

#### `plot_score_distribution(on_target_pred, off_target_pred, output_path)`

Plot histogram of comprehensive scores.

**Features**:
- ✅ Histogram with color-coded thresholds
- ✅ Poor (<0.3): Red
- ✅ Medium (0.3-0.6): Yellow
- ✅ Good (≥0.6): Green
- ✅ Mean and median lines
- ✅ Statistics annotations

```python
visualizer.plot_score_distribution(
    on_target_pred,
    off_target_pred,
    output_path=Path('distribution.png')
)
```

#### `plot_ranking_comparison(on_target_pred, off_target_pred, sgrnas, top_n, output_path)`

Compare rankings by different metrics.

**Features**:
- ✅ 3-panel comparison
- ✅ On-target only rankings
- ✅ Off-target only rankings
- ✅ Comprehensive score rankings
- ✅ Shows how rankings differ

```python
visualizer.plot_ranking_comparison(
    on_target_pred,
    off_target_pred,
    sgrnas=sgrna_sequences,
    top_n=10,
    output_path=Path('ranking_comparison.png')
)
```

#### `plot_interactive_scatter(on_target_pred, off_target_pred, sgrnas, output_path)`

Plot interactive Plotly scatter.

**Features**:
- ✅ Hover to see sgRNA sequence
- ✅ Hover to see exact predictions
- ✅ Click to zoom/pan
- ✅ Color scale for comprehensive score
- ✅ Quadrant lines

```python
visualizer.plot_interactive_scatter(
    on_target_pred,
    off_target_pred,
    sgrnas=sgrna_sequences,
    output_path=Path('interactive.html')
)
```

#### `generate_score_report(on_target_pred, off_target_pred, sgrnas, output_dir)`

Generate comprehensive text report.

**Report includes**:
- Score statistics (mean, median, std, min, max)
- Score distribution (poor/medium/good counts)
- Top 10 sgRNAs with detailed scores
- Score formula explanation
- Interpretation guidelines
- Recommendations

```python
visualizer.generate_score_report(
    on_target_pred,
    off_target_pred,
    sgrnas=sgrna_sequences,
    output_dir=Path('results')
)
```

---

## Score Interpretation

### Score Thresholds

- **Score ≥ 0.6**: Excellent sgRNA
  - High on-target efficiency
  - High off-target safety
  - Recommended for experiments

- **Score 0.3-0.6**: Acceptable sgRNA
  - Trade-off between efficiency and safety
  - Use if better options unavailable

- **Score < 0.3**: Poor sgRNA
  - Low efficiency or high off-target risk
  - Not recommended

### Score Components

**On-Target Efficiency** (X-axis):
- Measures how well sgRNA cuts at target site
- Range: 0-1 (higher is better)
- Predicted by model trained on efficiency data

**Off-Target Safety** (Y-axis):
- Measures how safe sgRNA is (1 - off-target risk)
- Range: 0-1 (higher is better)
- Predicted by model trained on off-target data

---

## Complete Example

```python
import torch
import numpy as np
from pathlib import Path
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.visualization.comprehensive_score import generate_comprehensive_report

# Initialize
device = 'cuda'
model = CRISPRUniPredict(device=device)
encoder = SequenceEncoder(device=device)

# Load model
checkpoint = torch.load('models/checkpoints/best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Example sgRNAs
sgrnas = [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG",
    # ... more sgRNAs
]

# Get predictions
on_target_preds = []
off_target_preds = []

for sgrna in sgrnas:
    onehot = encoder.one_hot_encode(sgrna)
    label = encoder.label_encode(sgrna, add_start_token=False)
    
    onehot = onehot.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    
    with torch.no_grad():
        on_target, off_target = model(onehot, label, task_type='both')
    
    on_target_preds.append(on_target.item())
    off_target_preds.append(off_target.item())

# Generate comprehensive report
output_dir = Path('results/comprehensive_score')
generate_comprehensive_report(
    on_target_pred=np.array(on_target_preds),
    off_target_pred=np.array(off_target_preds),
    sgrnas=sgrnas,
    output_dir=output_dir
)

print(f"Analysis complete! Results saved to {output_dir}")
```

---

## Output Files

### `2d_scatter.png`

2D scatter plot showing:
- X-axis: On-target efficiency
- Y-axis: Off-target safety
- Color: Comprehensive score
- Quadrant annotations

### `score_distribution.png`

Histogram showing:
- Distribution of comprehensive scores
- Color-coded thresholds (red/yellow/green)
- Mean and median lines
- Statistics annotations

### `ranking_comparison.png`

3-panel comparison showing:
- Top 10 by on-target efficiency
- Top 10 by off-target safety
- Top 10 by comprehensive score
- How rankings differ

### `interactive_scatter.html`

Interactive Plotly plot with:
- Hover to see sgRNA sequences
- Hover to see exact predictions
- Zoom and pan capabilities
- Color scale for scores

### `comprehensive_score_report.txt`

Text report with:
- Score statistics
- Distribution analysis
- Top 10 sgRNAs
- Score formula explanation
- Interpretation guidelines
- Recommendations

---

## Key Insights

### Why Comprehensive Score?

1. **On-target alone**: Ignores off-target risk
2. **Off-target alone**: Ignores efficiency
3. **Comprehensive**: Balances both factors

### Practical Applications

- **sgRNA design**: Select sgRNAs with high comprehensive scores
- **Experimental planning**: Prioritize top-ranked sgRNAs
- **Quality control**: Filter out poor sgRNAs (score < 0.3)
- **Trade-off analysis**: Understand efficiency-safety trade-offs

### Score Properties

- **Multiplicative formula**: Both components matter
- **Balanced weighting**: Neither component dominates
- **Interpretable**: Easy to explain to collaborators
- **Actionable**: Clear thresholds for decision-making

---

## Troubleshooting

### Issue: "All scores are very low"

**Cause**: Model predictions may be poor or data quality issues

**Solution**:
```python
# Check individual predictions
print(f"On-target range: {on_target_pred.min():.3f} - {on_target_pred.max():.3f}")
print(f"Off-target range: {off_target_pred.min():.3f} - {off_target_pred.max():.3f}")
```

### Issue: "Interactive plot not showing"

**Cause**: Plotly not installed

**Solution**:
```bash
pip install plotly
```

### Issue: "Quadrants not labeled clearly"

**Cause**: Data concentrated in one quadrant

**Solution**: Adjust axis limits or use different data range

---

## Summary

The comprehensive score visualization module provides:
- ✅ **Novel scoring formula** combining efficiency and safety
- ✅ **2D scatter analysis** with quadrant interpretation
- ✅ **Distribution analysis** with thresholds
- ✅ **Ranking comparison** across metrics
- ✅ **Interactive visualizations** with Plotly
- ✅ **Comprehensive reports** with recommendations
- ✅ **Easy-to-use API** with convenience functions

Perfect for sgRNA design and selection!
