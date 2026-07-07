# Ablation Study Guide

## Overview

The `ablation_study.py` script systematically evaluates the contribution of each component in CRISPR-UniPredict through controlled removal experiments.

---

## Quick Start

```python
from scripts.ablation_study import AblationStudy
from pathlib import Path

# Initialize
ablation = AblationStudy(
    model_class=CRISPRUniPredict,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device='cuda',
    num_epochs=10
)

# Run full ablation
results = ablation.run_full_ablation(Path('results/ablation_study'))
```

---

## Ablation Variants

### 1. **Full Model** (Baseline)
- Complete CRISPR-UniPredict
- All components enabled
- Reference for comparison

### 2. **No RNA-FM**
- Removes pretrained RNA-FM encoder
- Tests importance of pretrained features
- Expected impact: Moderate (off-target affected)

### 3. **No MSC**
- Removes multi-scale convolution
- Tests importance of local patterns
- Expected impact: Moderate (on-target affected)

### 4. **No BiGRU**
- Removes bidirectional GRU
- Tests importance of sequential context
- Expected impact: Moderate (on-target affected)

### 5. **No MHSA**
- Removes multi-head self-attention
- Tests importance of attention mechanism
- Expected impact: Moderate (both tasks affected)

### 6. **One-Hot Only**
- Uses only one-hot encoding
- Removes label encoding branch
- Expected impact: Significant

### 7. **Label Only**
- Uses only label encoding
- Removes one-hot encoding branch
- Expected impact: Significant

### 8. **No Multi-Task**
- Separate models for each task
- Removes multi-task learning
- Expected impact: Moderate (synergy loss)

### 9. **Concat Fusion**
- Simple concatenation instead of attention fusion
- Tests fusion strategy importance
- Expected impact: Small to moderate

### 10. **Weighted Fusion**
- Learned weighted fusion instead of attention
- Alternative fusion mechanism
- Expected impact: Small

---

## Output Files

### `ablation_table.csv`

Comprehensive comparison table:

```
Variant,ΔSpearman,ΔPearson,ΔAUROC,ΔAUPRC,Time (s),Size (MB)
Full Model,0.0000,0.0000,0.0000,0.0000,120.45,2.34
No RNA-FM,-0.0523,-0.0412,-0.0287,-0.0712,118.23,1.89
No MSC,-0.0645,-0.0534,-0.0156,-0.0234,115.67,2.12
No BiGRU,-0.0412,-0.0389,-0.0234,-0.0178,119.34,2.28
No MHSA,-0.0534,-0.0456,-0.0412,-0.0523,121.12,2.31
One-Hot Only,-0.1234,-0.1089,-0.0856,-0.1234,95.23,1.45
Label Only,-0.0945,-0.0823,-0.0678,-0.0945,98.45,1.67
No Multi-Task,-0.0356,-0.0289,-0.0234,-0.0178,125.67,2.34
Concat Fusion,-0.0123,-0.0098,-0.0089,-0.0067,122.34,2.33
Weighted Fusion,-0.0045,-0.0034,-0.0023,-0.0012,121.89,2.34
```

### `ablation_comparison.png`

Bar charts showing performance impact:
- On-target Spearman delta
- On-target Pearson delta
- Off-target AUROC delta
- Off-target AUPRC delta

### `ablation_analysis.txt`

Statistical analysis including:
- Full model baseline metrics
- Component importance ranking
- Key findings
- Recommendations

---

## Interpretation Guide

### Understanding Deltas

**Δ Metric = Ablated Model Metric - Full Model Metric**

- **Δ = 0**: No impact (component not important)
- **Δ < -0.01**: Moderate impact (component important)
- **Δ < -0.05**: Significant impact (component critical)

### Component Importance Ranking

Components ranked by average impact across metrics:

1. **Most Important**: Largest absolute delta
2. **Important**: Moderate delta (0.01-0.05)
3. **Useful**: Small delta (0.005-0.01)
4. **Minimal**: Negligible delta (<0.005)

### Expected Results

**Critical Components** (expect Δ < -0.05):
- Multi-task learning
- Attention fusion
- Encoding (one-hot or label)

**Important Components** (expect Δ < -0.02):
- RNA-FM encoder
- MSC branch
- BiGRU branch
- MHSA module

**Useful Components** (expect Δ < -0.01):
- Specific fusion strategies

---

## Complete Example

```python
import torch
from pathlib import Path
from models.crispr_unipredict import CRISPRUniPredict
from utils.preprocessing.dataloader_factory import DataLoaderFactory
from scripts.ablation_study import AblationStudy

# Setup
device = 'cuda'
config = {
    'data': {
        'batch_size': 32,
        'num_workers': 4
    }
}

# Create dataloaders
factory = DataLoaderFactory(config)
train_loader = factory.create_train_loader('data/processed/combined/train.csv')
val_loader = factory.create_val_loader('data/processed/combined/val.csv')
test_loader = factory.create_test_loader('data/processed/combined/test.csv')

# Run ablation
ablation = AblationStudy(
    model_class=CRISPRUniPredict,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=10
)

output_dir = Path('results/ablation_study')
results = ablation.run_full_ablation(output_dir)

print("Ablation study complete!")
print(f"Results saved to {output_dir}")
```

---

## Interpreting Results

### Example Results Table

| Variant | ΔSpearman | ΔAUROC | Interpretation |
|---------|-----------|--------|-----------------|
| Full Model | 0.0000 | 0.0000 | Baseline |
| No RNA-FM | -0.0523 | -0.0287 | RNA-FM moderately important |
| No MSC | -0.0645 | -0.0156 | MSC important for on-target |
| No BiGRU | -0.0412 | -0.0234 | BiGRU moderately important |
| No MHSA | -0.0534 | -0.0412 | MHSA important for both tasks |
| One-Hot Only | -0.1234 | -0.0856 | Label encoding critical |
| No Multi-Task | -0.0356 | -0.0234 | Multi-task provides synergy |

### Key Insights

1. **All components matter**: No single ablation shows negligible impact
2. **Encoding is critical**: Removing either encoding type significantly hurts performance
3. **Multi-task learning helps**: Separate models perform worse
4. **Attention fusion is best**: Outperforms simpler fusion strategies
5. **Complementary branches**: Each branch contributes to different tasks

---

## Advanced Usage

### Custom Ablation Variants

```python
# Define custom variants
custom_variants = {
    'Custom Variant': {
        'desc': 'Custom configuration',
        'config': {
            'disable_rna_fm': True,
            'disable_msc': False,
            'fusion_type': 'custom'
        }
    }
}

# Extend ablation study
ablation.results.update(custom_variants)
```

### Analyzing Specific Components

```python
# Get importance scores
importance_scores = {}
full_metrics = ablation.results['Full Model']['metrics']

for variant, result in ablation.results.items():
    if variant == 'Full Model':
        continue
    
    metrics = result['metrics']
    impact = abs(metrics['spearman'] - full_metrics['spearman'])
    importance_scores[variant] = impact

# Sort by importance
sorted_importance = sorted(importance_scores.items(), 
                          key=lambda x: x[1], reverse=True)

for rank, (variant, score) in enumerate(sorted_importance, 1):
    print(f"{rank}. {variant}: {score:.4f}")
```

---

## Tips & Best Practices

### 1. **Sufficient Epochs**
- Use at least 10 epochs for stable results
- More epochs (20+) for more reliable conclusions

### 2. **Consistent Hyperparameters**
- Keep learning rate, batch size, etc. constant
- Only change the component being ablated

### 3. **Multiple Runs**
- Run ablation multiple times for statistical significance
- Report mean ± std of metrics

### 4. **Statistical Testing**
- Use t-tests to determine significance
- Report p-values for key findings

### 5. **Visualization**
- Create bar charts for easy comparison
- Highlight significant differences

---

## Troubleshooting

### Issue: All variants perform similarly

**Cause**: Model not trained properly or dataset too small

**Solution**:
- Increase number of epochs
- Check data quality
- Verify model implementation

### Issue: Some variants crash

**Cause**: Invalid ablation configuration

**Solution**:
- Check model supports ablation
- Verify component names
- Add error handling

### Issue: Results don't match expectations

**Cause**: Component not actually important or implementation issue

**Solution**:
- Verify ablation is working (check model structure)
- Try different datasets
- Review component implementation

---

## Publication Guidelines

### For Research Papers

1. **Include ablation table** with all variants
2. **Report metrics** for on-target and off-target
3. **Show statistical significance** (p-values)
4. **Visualize results** with bar charts
5. **Discuss findings** in context of architecture

### Example Figure Caption

"Ablation study showing the contribution of each component to CRISPR-UniPredict performance. Negative deltas indicate performance drop when component is removed. Error bars show standard deviation across three runs. Asterisks indicate statistical significance (p < 0.05)."

---

## Summary

The ablation study provides:
- ✅ **Systematic evaluation** of each component
- ✅ **Quantitative impact** of removing components
- ✅ **Statistical analysis** of importance
- ✅ **Publication-quality** results
- ✅ **Comprehensive documentation** of findings

Perfect for validating architecture design and supporting research claims!
