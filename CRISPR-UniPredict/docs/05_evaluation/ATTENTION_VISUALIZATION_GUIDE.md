# Attention Visualization Guide

## Overview

The `attention_viz.py` module provides tools to visualize and understand what the CRISPR-UniPredict model focuses on in sequences.

---

## Quick Start

### Basic Attention Visualization

```python
from utils.visualization.attention_viz import AttentionVisualizer
from models.encoding import SequenceEncoder
from pathlib import Path

# Initialize visualizer
visualizer = AttentionVisualizer(model, device='cuda')

# Encode sequence
encoder = SequenceEncoder(device='cuda')
sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
onehot = encoder.one_hot_encode(sgrna)
label = encoder.label_encode(sgrna, add_start_token=False)

# Extract attention weights
weights = visualizer.extract_attention_weights(onehot, label)

# Plot heatmap
visualizer.plot_attention_heatmap(weights['mhsa_layer_0'], sgrna, 
                                 output_path=Path('attention_heatmap.png'))
```

---

## AttentionVisualizer Class

### Initialization

```python
from utils.visualization.attention_viz import AttentionVisualizer

visualizer = AttentionVisualizer(model, device='cuda')
```

**Parameters**:
- `model`: CRISPRUniPredict model instance
- `device`: Device to use ('cuda' or 'cpu')

### Methods

#### `extract_attention_weights(sgrna_onehot, sgrna_label)`

Extract attention weights from model.

**Parameters**:
- `sgrna_onehot`: One-hot encoded sgRNA (23, 4) or (1, 23, 4)
- `sgrna_label`: Label encoded sgRNA (23,) or (1, 23)

**Returns**: Dictionary with attention weights per layer

```python
weights = visualizer.extract_attention_weights(onehot, label)
# Returns: {'mhsa_layer_0': tensor(...), 'mhsa_layer_1': tensor(...), ...}
```

#### `plot_attention_heatmap(attention_weights, sequence, output_path, title)`

Plot attention heatmap with nucleotide annotations.

**Parameters**:
- `attention_weights`: Attention weights (seq_len, seq_len) or (heads, seq_len, seq_len)
- `sequence`: DNA sequence string
- `output_path`: Path to save figure (optional)
- `title`: Figure title

**Features**:
- ✅ Nucleotide annotations on axes
- ✅ Seed region highlighting (positions 16-20)
- ✅ Multi-head averaging
- ✅ Publication-quality output (300 DPI)

```python
visualizer.plot_attention_heatmap(
    weights['mhsa_layer_0'],
    "GCTAGCTAGCTAGCTAGCTAGCT",
    output_path=Path('attention_heatmap.png'),
    title='MHSA Layer 0 Attention'
)
```

#### `plot_position_importance(attention_weights, sequence, output_path)`

Plot position-wise importance bar chart.

**Parameters**:
- `attention_weights`: Attention weights
- `sequence`: DNA sequence
- `output_path`: Path to save figure

**Features**:
- ✅ Bar chart of position importance
- ✅ Seed region highlighting (coral color)
- ✅ Normalized importance scores

```python
visualizer.plot_position_importance(
    weights['mhsa_layer_0'],
    "GCTAGCTAGCTAGCTAGCTAGCT",
    output_path=Path('position_importance.png')
)
```

#### `compare_attention_patterns(good_sgrnas, poor_sgrnas, output_dir)`

Compare attention patterns between high and low efficiency sgRNAs.

**Parameters**:
- `good_sgrnas`: List of (sequence, label) tuples for high efficiency
- `poor_sgrnas`: List of (sequence, label) tuples for low efficiency
- `output_dir`: Directory to save figures

**Output**: 3-panel comparison plot
1. High efficiency attention patterns
2. Low efficiency attention patterns
3. Difference heatmap

```python
good_sgrnas = [
    ("GCTAGCTAGCTAGCTAGCTAGCT", 0.9),
    ("ATGCATGCATGCATGCATGCATG", 0.85),
]

poor_sgrnas = [
    ("CCGGCCGGCCGGCCGGCCGGCCG", 0.3),
    ("GGCCGGCCGGCCGGCCGGCCGGCC", 0.25),
]

visualizer.compare_attention_patterns(
    good_sgrnas,
    poor_sgrnas,
    output_dir=Path('attention_comparison')
)
```

#### `visualize_seed_region_importance(sgrnas, output_path)`

Visualize seed region importance (positions 16-20).

**Parameters**:
- `sgrnas`: List of sgRNA sequences
- `output_path`: Path to save figure

**Features**:
- ✅ Position-wise importance bar chart
- ✅ Seed region highlighted in coral
- ✅ Averaged across multiple sgRNAs
- ✅ Replicates Figure 5 from CRISPR_HNN paper

```python
sgrnas = [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG",
]

visualizer.visualize_seed_region_importance(
    sgrnas,
    output_path=Path('seed_region_importance.png')
)
```

---

## Convenience Functions

### `extract_attention_weights(model, sgrna_onehot, sgrna_label, device)`

Quick function to extract attention weights.

```python
from utils.visualization.attention_viz import extract_attention_weights

weights = extract_attention_weights(model, onehot, label, device='cuda')
```

### `plot_attention_heatmap(attention_weights, sequence, output_path, title)`

Quick function to plot attention heatmap.

```python
from utils.visualization.attention_viz import plot_attention_heatmap

plot_attention_heatmap(weights, "GCTAGCTAGCTAGCTAGCTAGCT", 
                      output_path=Path('heatmap.png'))
```

### `compare_attention_patterns(model, good_sgrnas, poor_sgrnas, output_dir, device)`

Quick function to compare patterns.

```python
from utils.visualization.attention_viz import compare_attention_patterns

compare_attention_patterns(model, good_sgrnas, poor_sgrnas, 
                          output_dir=Path('comparison'), device='cuda')
```

### `visualize_seed_region_importance(model, sgrnas, output_path, device)`

Quick function to visualize seed region.

```python
from utils.visualization.attention_viz import visualize_seed_region_importance

visualize_seed_region_importance(model, sgrnas, 
                                output_path=Path('seed_importance.png'), 
                                device='cuda')
```

---

## Complete Example

```python
import torch
from pathlib import Path
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder
from utils.visualization.attention_viz import AttentionVisualizer

# Initialize
device = 'cuda'
model = CRISPRUniPredict(device=device)
encoder = SequenceEncoder(device=device)
visualizer = AttentionVisualizer(model, device=device)

# Load model from checkpoint
checkpoint = torch.load('models/checkpoints/best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Example sgRNAs
high_efficiency_sgrnas = [
    ("GCTAGCTAGCTAGCTAGCTAGCT", 0.9),
    ("ATGCATGCATGCATGCATGCATG", 0.85),
]

low_efficiency_sgrnas = [
    ("CCGGCCGGCCGGCCGGCCGGCCG", 0.3),
    ("GGCCGGCCGGCCGGCCGGCCGGCC", 0.25),
]

# Create output directory
output_dir = Path('results/attention_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Visualize individual sgRNA
sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
onehot = encoder.one_hot_encode(sgrna)
label = encoder.label_encode(sgrna, add_start_token=False)

weights = visualizer.extract_attention_weights(onehot, label)

for layer_name, layer_weights in weights.items():
    visualizer.plot_attention_heatmap(
        layer_weights,
        sgrna,
        output_path=output_dir / f'{layer_name}_heatmap.png',
        title=f'{layer_name} Attention'
    )
    
    visualizer.plot_position_importance(
        layer_weights,
        sgrna,
        output_path=output_dir / f'{layer_name}_position_importance.png'
    )

# 2. Compare high vs low efficiency
visualizer.compare_attention_patterns(
    high_efficiency_sgrnas,
    low_efficiency_sgrnas,
    output_dir=output_dir / 'comparison'
)

# 3. Visualize seed region importance
all_sgrnas = [s[0] for s in high_efficiency_sgrnas + low_efficiency_sgrnas]
visualizer.visualize_seed_region_importance(
    all_sgrnas,
    output_path=output_dir / 'seed_region_importance.png'
)

print(f"Attention visualizations saved to {output_dir}")
```

---

## Understanding Attention Patterns

### What Do Attention Heatmaps Show?

- **X-axis**: Target position (which nucleotide is being attended to)
- **Y-axis**: Query position (which nucleotide is doing the attending)
- **Color intensity**: Attention weight (higher = more attention)

### Interpreting Patterns

**Strong diagonal**: Model focuses on local context
**Seed region focus**: Model attends to seed region (positions 16-20)
**Off-diagonal patterns**: Model captures long-range dependencies

### Seed Region (Positions 16-20)

- Critical for CRISPR-Cas9 specificity
- Usually shows high attention in efficient sgRNAs
- Highlighted in blue boxes on heatmaps

---

## Output Files

### Attention Heatmap (`attention_heatmap.png`)

- 2D heatmap showing attention weights
- Nucleotides labeled on axes
- Seed region highlighted with blue dashed box
- Color scale from white (low) to dark red (high)

### Position Importance (`position_importance.png`)

- Bar chart of position-wise importance
- Seed region bars in coral color
- Other positions in steel blue
- Normalized to 0-1 scale

### Attention Comparison (`attention_comparison.png`)

- 3-panel figure:
  1. High efficiency sgRNA attention
  2. Low efficiency sgRNA attention
  3. Difference heatmap (red-blue diverging)

### Seed Region Importance (`seed_region_importance.png`)

- Bar chart showing importance per position
- Positions 16-20 highlighted in coral
- Averaged across multiple sgRNAs
- Replicates Figure 5 from CRISPR_HNN paper

---

## Tips for Interpretation

### 1. Look for Seed Region Focus

High-performing sgRNAs typically show strong attention to positions 16-20.

### 2. Compare Patterns

Compare attention patterns between high and low efficiency sgRNAs to identify discriminative features.

### 3. Check Multiple Layers

Different layers may focus on different aspects:
- Early layers: Local patterns
- Middle layers: Seed region
- Late layers: Overall structure

### 4. Aggregate Across Samples

Average attention patterns across multiple sgRNAs for more robust insights.

---

## Troubleshooting

### Issue: "No attention weights captured"

**Cause**: Model doesn't have MHSA modules or hooks not registered

**Solution**:
```python
# Check model structure
for name, module in model.named_modules():
    if 'mhsa' in name.lower():
        print(f"Found MHSA: {name}")
```

### Issue: "Attention weights are all zeros"

**Cause**: Model in training mode or incorrect forward pass

**Solution**:
```python
model.eval()  # Set to evaluation mode
with torch.no_grad():
    weights = visualizer.extract_attention_weights(onehot, label)
```

### Issue: "Seed region not highlighted"

**Cause**: Sequence length mismatch

**Solution**:
```python
# Ensure sequence is 23 bp
assert len(sequence) == 23, f"Expected 23 bp, got {len(sequence)}"
```

---

## Summary

The attention visualization module provides:
- ✅ **Attention weight extraction** from MHSA modules
- ✅ **Publication-quality heatmaps** with nucleotide annotations
- ✅ **Position importance analysis** with seed region highlighting
- ✅ **Pattern comparison** between high/low efficiency sgRNAs
- ✅ **Seed region importance** visualization
- ✅ **Easy-to-use API** with convenience functions

Perfect for understanding model behavior and validating biological insights!
