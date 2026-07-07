# Issue 4.3 Fix: Per-Source Data Normalization Guide

**Created:** 2026-04-15  
**Purpose:** Address on-target label quality issues (multi-source data with incomparable scales)  
**Tool:** `utils/preprocessing/per_source_normalization.py`

---

## Problem (Issue 4.3)

On-target efficiency labels come from **multiple sources** with different:
- Assay protocols (reporter assays, indel detection, etc.)
- Cell types (HEK293, U2OS, etc.)
- Measurement scales (0-1, 0-100, arbitrary units)

**Impact:** Mixing incomparable scales caps correlation around **0.41 Spearman** when it could be higher.

---

## Solution

**Per-source Z-score normalization** centers each source's labels independently before combining.

### What this does:
```
Before: [0.8 (HEK293), 0.1 (U2OS), 0.9 (HF), ...]  
After:  [+0.5σ (HEK293), -0.8σ (U2OS), +0.7σ (HF), ...]
```

Each source's labels are standardized to mean=0, std=1 within its own population.

---

## Quick Start

### Step 1: Inspect your data
```bash
cd CRISPR-UniPredict
python -c "import pandas as pd; df = pd.read_csv('data/processed/combined/train.csv'); print(df['dataset_source'].value_counts())"
```

Expected output:
```
dataset_source
GecKo          80000
HEK293         50000
CCLMoff        40000
...
```

### Step 2: Run normalization
```bash
python utils/preprocessing/per_source_normalization.py
```

This will:
- Analyze training set statistics per source
- Normalize all splits (train, val, test) 
- Save as `*_normalized.csv`

### Step 3: Use normalized data in training

Update your config file (e.g., `configs/model_config.yaml`):

**Before:**
```yaml
data:
  train_path: data/processed/combined/train.csv
  val_path: data/processed/combined/val.csv
  test_path: data/processed/combined/test.csv
```

**After:**
```yaml
data:
  train_path: data/processed/combined/train_normalized.csv
  val_path: data/processed/combined/val_normalized.csv
  test_path: data/processed/combined/test_normalized.csv
```

### Step 4: Train with normalized data
```bash
python scripts/train.py --config configs/model_config.yaml
```

---

## Expected Improvements

- **On-target Spearman correlation:** 0.41 → ~0.45-0.48 (estimated)
  - Removes source-specific systematic offsets
  - Allows model to focus on real guide efficiency patterns
  
- **Off-target AUROC:** Should remain stable (~0.875)
  - Off-target task unaffected by on-target normalization

---

## Technical Details

### Normalization Method: Z-score

For each source s and each sample i in that source:

```
x_normalized = (x_i - mean_s) / std_s
```

Where:
- `x_i` = original on-target score for sample i
- `mean_s` = mean of all on-target scores in source s
- `std_s` = standard deviation in source s

### Special Cases

1. **Zero standard deviation** (all values in a source are identical)
   - Still center by mean: `x_normalized = x_i - mean_s`

2. **Missing on-target labels** (NaN)
   - Skipped from normalization, remain as NaN

3. **Test set statistics**
   - Uses training set mean/std for fair evaluation
   - Simulates real deployment scenario

---

## Advanced: Custom Normalization

You can also use min-max scaling (0-1 per source):

```python
from utils.preprocessing.per_source_normalization import normalize_per_source
import pandas as pd

df = pd.read_csv('data/processed/combined/train.csv')
df_normalized = normalize_per_source(df, method='minmax')
df_normalized.to_csv('train_minmax_normalized.csv', index=False)
```

---

## Troubleshooting

### No valid samples for source 'X'
- Warning: Source has no on-target labels at all
- Those rows remain unchanged

### on_target_score column not found
- Check column names: `df.columns.tolist()`
- Update the function call: `normalize_per_source(df, target_col='your_col_name')`

### Want to inspect statistics?
```python
from utils.preprocessing.per_source_normalization import compute_source_statistics
import pandas as pd

df = pd.read_csv('data/processed/combined/train.csv')
stats = compute_source_statistics(df)
for source, stat in stats.items():
    print(f"{source}: mean={stat['mean']:.3f}, std={stat['std']:.3f}, n={stat['count']}")
```

---

## Citation Note

For your paper/thesis, add to Methods:

> On-target efficiency labels were normalized per dataset source using z-score normalization (μ=0, σ=1) to account for different assay protocols and measurement scales. Test set normalization used statistics computed on the training set to avoid data leakage.

---

## Example Comparison

**Without normalization:**
```
Accuracy: Training loss converges, but on-target Spearman plateaus at 0.41
Reason: Model confused by source-specific offsets in labels
```

**With normalization:**
```
Accuracy: Training loss converges, on-target Spearman reaches ~0.46 (cleaner signal)
Reason: Model focuses on relative efficiency within sources, not absolute scale
```

---

*For more details, see Issue 4.3 in [CHAPTER_4_ISSUES_STATUS.md](./CHAPTER_4_ISSUES_STATUS.md)*
