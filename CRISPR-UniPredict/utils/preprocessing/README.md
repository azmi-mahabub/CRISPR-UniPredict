# Preprocessing Utilities

## Overview

This directory contains utilities for preprocessing and formatting CRISPR datasets.

---

## Modules

### 1. `data_formatter.py`

**Purpose**: Unified data formatting for CRISPR datasets

**Class**: `UnifiedDataFormatter`

**Features**:
- ✓ Automatic separator detection
- ✓ Column name auto-detection
- ✓ Sequence validation (DNA/RNA)
- ✓ Score normalization (min-max scaling)
- ✓ Unified schema formatting
- ✓ Individual and combined dataset generation
- ✓ Statistics tracking

**Usage**:

```python
from data_formatter import UnifiedDataFormatter

# Create formatter
formatter = UnifiedDataFormatter()

# Process all datasets
stats = formatter.process_all_datasets()

# Access results
print(f"Valid sequences: {stats['total_valid']:,}")
print(f"Invalid sequences: {stats['total_invalid']:,}")
```

**Command Line**:

```bash
python data_formatter.py
```

### 2. `sequence_utils.py`

**Purpose**: Sequence encoding and manipulation utilities

**Functions**:
- `encode_sequence()` - One-hot encoding
- `validate_sequence()` - Sequence validation
- `reverse_complement()` - DNA reverse complement
- `extract_kmers()` - K-mer extraction
- `calculate_gc_content()` - GC content calculation
- `calculate_tm()` - Melting temperature

### 3. `crispr_hnn_preprocessor.py`

**Purpose**: CRISPR_HNN specific preprocessing

**Functions**:
- `preprocess_crispr_hnn()` - Dataset preprocessing
- `load_crispr_hnn_data()` - Data loading
- `augment_sequences()` - Data augmentation

### 4. `cclmoff_preprocessor.py`

**Purpose**: CCLMoff specific preprocessing

**Functions**:
- `preprocess_cclmoff()` - Dataset preprocessing
- `load_cclmoff_data()` - Data loading

---

## Unified Data Schema

### Columns

| Column | Type | Description |
|--------|------|-------------|
| sgrna_sequence | string | 20-23 bp sgRNA sequence |
| target_sequence | string | DNA target site |
| pam_sequence | string | PAM motif |
| on_target_score | float | Normalized 0-1 |
| off_target_label | int | Binary 0/1 |
| dataset_source | string | Dataset name |
| cell_line | string | Cell line |
| detection_method | string | Method used |
| task_type | string | 'on_target' or 'off_target' |

---

## Processing Pipeline

### Step 1: Format Data

```bash
cd CRISPR-UniPredict
python utils/preprocessing/data_formatter.py
```

**Output**:
- Individual formatted datasets in `data/processed/on_target/` and `data/processed/off_target/`
- Combined datasets: `combined_on_target.csv`, `combined_off_target.csv`
- Unified dataset: `unified_dataset.csv`
- Statistics: `formatting_statistics.pkl`

### Step 2: Validate Data

```bash
python scripts/inspect_data.py
```

**Output**:
- Inspection report: `data_inspection_report.txt`

### Step 3: Train Models

```bash
python scripts/train_on_target.py
python scripts/train_off_target.py
```

---

## Data Statistics

### CRISPR_HNN (On-Target)

- **Datasets**: 9
- **Total Sequences**: 291,639
- **Validity Rate**: 100%
- **Indel Range**: 0.0-1.0 (normalized)

### CCLMoff (Off-Target)

- **Datasets**: 1
- **Total Sequences**: 3,110,438
- **Validity Rate**: 48.65%
- **Label Distribution**: Binary (0/1)

### Combined

- **Total Sequences**: 3,402,077
- **On-Target**: 291,639
- **Off-Target**: 3,110,438

---

## File Locations

### Input Data

```
data/raw/
├── crispr_hnn/
│   ├── ESP（n=58616）.csv
│   ├── HCT116（n=4239）.csv
│   ├── HELA（n=8101）.csv
│   ├── HF（n=56887）.csv
│   ├── HL60（n=2076）.csv
│   ├── Sniper-Cas9（n=37794）.csv
│   ├── SpCas9-NG（n=30585）.csv
│   ├── WT（n=55603）.csv
│   └── xCas（n=37738）.csv
└── cclmoff/
    └── 09212024_CCLMoff_dataset.csv
```

### Output Data

```
data/processed/
├── on_target/
│   ├── ESP_formatted.csv
│   ├── HCT116_formatted.csv
│   ├── HELA_formatted.csv
│   ├── HF_formatted.csv
│   ├── HL60_formatted.csv
│   ├── Sniper-Cas9_formatted.csv
│   ├── SpCas9-NG_formatted.csv
│   ├── WT_formatted.csv
│   └── xCas_formatted.csv
├── off_target/
│   └── CCLMoff_formatted.csv
└── combined/
    ├── combined_on_target.csv
    ├── combined_off_target.csv
    ├── unified_dataset.csv
    └── formatting_statistics.pkl
```

---

## Usage Examples

### Load Unified Dataset

```python
import pandas as pd

# Load unified dataset
df = pd.read_csv('data/processed/combined/unified_dataset.csv')

# Filter by task type
on_target = df[df['task_type'] == 'on_target']
off_target = df[df['task_type'] == 'off_target']

# Filter by dataset source
esp = df[df['dataset_source'] == 'ESP']
cclmoff = df[df['dataset_source'] == 'CCLMoff']
```

### Validate Sequences

```python
from data_formatter import UnifiedDataFormatter

formatter = UnifiedDataFormatter()

# Validate a sequence
is_valid = formatter.validate_sequence('ACGTACGTACGTACGTACGTACG')
print(f"Valid: {is_valid}")  # True

# Invalid sequence
is_valid = formatter.validate_sequence('ACGTXYZ')
print(f"Valid: {is_valid}")  # False
```

### Normalize Scores

```python
from data_formatter import UnifiedDataFormatter

formatter = UnifiedDataFormatter()

# Normalize a score
score = formatter.normalize_indel_frequency(0.5, min_val=0.0, max_val=1.0)
print(f"Normalized: {score}")  # 0.5
```

### Access Statistics

```python
import pickle

with open('data/processed/combined/formatting_statistics.pkl', 'rb') as f:
    stats = pickle.load(f)

print(f"Total valid: {stats['total_valid']:,}")
print(f"Total invalid: {stats['total_invalid']:,}")

for dataset, info in stats['datasets'].items():
    print(f"{dataset}: {info['valid']:,} sequences")
```

---

## Key Features

### Sequence Validation

✓ Only ACGTU nucleotides allowed
✓ Case-insensitive
✓ U → T replacement
✓ Whitespace trimming

### Score Normalization

✓ Min-max scaling to 0-1
✓ Edge case handling
✓ Range clipping

### Column Detection

✓ Automatic column name matching
✓ Keyword-based detection
✓ Graceful fallback

### Error Handling

✓ Continues on errors
✓ Detailed error reporting
✓ Progress indicators
✓ Statistics tracking

---

## Performance

| Metric | Value |
|--------|-------|
| Processing Time | ~2-3 minutes |
| Throughput | ~35,000 seq/sec |
| Memory Usage | ~2-3 GB peak |
| Largest File | 681.58 MB |

---

## Troubleshooting

### Issue: "Could not detect sequence column"

**Solution**: Check column names in raw data. Ensure they contain keywords like 'seq', 'sequence', 'guide', 'sgrna'.

### Issue: "Could not detect indel column"

**Solution**: For CRISPR_HNN data, ensure indel column exists. Check for keywords: 'indel', 'efficiency', 'activity', 'score'.

### Issue: "Could not detect label column"

**Solution**: For CCLMoff data, ensure label column exists. Check for keywords: 'label', 'off_target', 'offtarget', 'pred'.

### Issue: Memory error with large files

**Solution**: Process files in batches. Use `low_memory=False` in pandas.read_csv().

---

## Next Steps

1. **Format Data**
   ```bash
   python data_formatter.py
   ```

2. **Inspect Results**
   ```bash
   python ../scripts/inspect_data.py
   ```

3. **Train Models**
   ```bash
   python ../scripts/train_on_target.py
   python ../scripts/train_off_target.py
   ```

4. **Evaluate Results**
   ```bash
   python ../scripts/evaluate_models.py
   ```

---

## Related Files

- `../scripts/inspect_data.py` - Data inspection tool
- `../scripts/train_on_target.py` - On-target model training
- `../scripts/train_off_target.py` - Off-target model training
- `../DATA_FORMATTING_REPORT.md` - Detailed formatting report
- `../DATA_INSPECTION_SUMMARY.md` - Inspection summary

---

## Support

For questions or issues:
1. Check the DATA_FORMATTING_REPORT.md
2. Review script docstrings
3. Check project README.md
4. Review SETUP_GUIDE.md

---

*Last Updated: 2024*
*Status: ✓ Ready for Use*
