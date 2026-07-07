# CRISPR-UniPredict - Project Index

## 📋 Documentation Files

### Getting Started
1. **[README.md](README.md)** - Main project documentation
   - Project overview
   - Installation instructions
   - Quick start guide
   - Usage examples
   - Model descriptions

2. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions
   - Directory structure explanation
   - Installation steps
   - Data preparation
   - Preprocessing workflow
   - Configuration guide
   - Troubleshooting

3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project overview
   - What's included
   - Project statistics
   - Key features
   - File organization
   - Development workflow

## 🗂️ Directory Structure

### Data Management
```
data/
├── raw/
│   ├── crispr_hnn/          # On-target datasets
│   └── cclmoff/             # Off-target datasets
└── processed/
    ├── on_target/           # Processed on-target data
    ├── off_target/          # Processed off-target data
    └── combined/            # Combined datasets
```

### Model Management
```
models/
├── checkpoints/             # Training checkpoints
└── pretrained/              # Final trained models
```

### Results & Outputs
```
results/
├── plots/                   # Visualization plots
├── metrics/                 # Evaluation metrics
└── predictions/             # Model predictions
```

### Utilities
```
utils/
├── preprocessing/           # Data preprocessing
│   ├── sequence_utils.py
│   ├── crispr_hnn_preprocessor.py
│   └── cclmoff_preprocessor.py
├── evaluation/              # Model evaluation
│   └── metrics.py
└── visualization/           # Result visualization
    └── plots.py
```

### Development
```
scripts/                     # Training/inference scripts
configs/                     # Configuration files
logs/                        # Training logs
notebooks/                   # Jupyter notebooks
```

## 🔧 Utility Modules Reference

### Preprocessing (`utils/preprocessing/`)

#### `sequence_utils.py`
```python
from utils.preprocessing.sequence_utils import *

validate_sequence(sequence, seq_type='dna')
encode_sequence(sequence, encoding_type='onehot')
reverse_complement(sequence)
extract_kmers(sequence, k=20)
calculate_gc_content(sequence)
calculate_tm(sequence)
```

#### `crispr_hnn_preprocessor.py`
```python
from utils.preprocessing import preprocess_crispr_hnn, load_crispr_hnn_data

preprocess_crispr_hnn(input_dir, output_dir)
X_train, X_test, y_train, y_test = load_crispr_hnn_data(data_file)
```

#### `cclmoff_preprocessor.py`
```python
from utils.preprocessing import preprocess_cclmoff, load_cclmoff_data

preprocess_cclmoff(input_dir, output_dir)
X_train, X_test, y_train, y_test = load_cclmoff_data(data_file)
```

### Evaluation (`utils/evaluation/`)

#### `metrics.py`
```python
from utils.evaluation.metrics import *

# Classification
classification_metrics(y_true, y_pred, y_pred_proba)

# Regression
regression_metrics(y_true, y_pred)

# ROC and confusion
compute_roc_curve(y_true, y_pred_proba)
compute_confusion_matrix(y_true, y_pred)

# Per-dataset
compute_per_dataset_metrics(datasets, predictions)
```

### Visualization (`utils/visualization/`)

#### `plots.py`
```python
from utils.visualization.plots import *

plot_metrics(metrics, title, save_path)
plot_roc_curve(fpr, tpr, auc, title, save_path)
plot_confusion_matrix(cm, labels, title, save_path)
plot_predictions(y_true, y_pred, title, save_path)
```

## 📊 Models

### CCLMoff (Off-Target Prediction)
- **Task**: Binary Classification
- **Input**: sgRNA + target sequences
- **Output**: Off-target probability (0-1)
- **Data**: `data/raw/cclmoff/`
- **Processed**: `data/processed/off_target/`

### CRISPR_HNN (On-Target Activity)
- **Task**: Regression
- **Input**: 23bp sgRNA sequence
- **Output**: Indel efficiency (0-1)
- **Data**: `data/raw/crispr_hnn/`
- **Processed**: `data/processed/on_target/`

## 🚀 Quick Start

### 1. Setup
```bash
cd CRISPR-UniPredict
pip install -r requirements.txt
python verify_structure.py
```

### 2. Prepare Data
```bash
# Place datasets in:
# - data/raw/crispr_hnn/
# - data/raw/cclmoff/
```

### 3. Preprocess
```python
from utils.preprocessing import preprocess_crispr_hnn, preprocess_cclmoff

preprocess_crispr_hnn('data/raw/crispr_hnn/', 'data/processed/on_target/')
preprocess_cclmoff('data/raw/cclmoff/', 'data/processed/off_target/')
```

### 4. Train Models
```bash
# Create scripts in scripts/ directory
python scripts/train_on_target.py
python scripts/train_off_target.py
```

### 5. Evaluate & Visualize
```python
from utils.evaluation.metrics import regression_metrics
from utils.visualization.plots import plot_metrics

metrics = regression_metrics(y_test, y_pred)
plot_metrics(metrics, save_path='results/plots/metrics.png')
```

## 📁 File Locations

| Purpose | Location |
|---------|----------|
| Raw Data | `data/raw/` |
| Processed Data | `data/processed/` |
| Model Checkpoints | `models/checkpoints/` |
| Trained Models | `models/pretrained/` |
| Plots | `results/plots/` |
| Metrics | `results/metrics/` |
| Predictions | `results/predictions/` |
| Training Scripts | `scripts/` |
| Configuration | `configs/` |
| Logs | `logs/` |
| Notebooks | `notebooks/` |

## 🔗 Related Files

- **Configuration**: `configs/config_template.yaml`
- **Dependencies**: `requirements.txt`
- **Git Config**: `.gitignore`
- **Verification**: `verify_structure.py`
- **Package Init**: `__init__.py`

## 📚 Key Concepts

### Data Flow
```
Raw Data → Preprocess → Processed Data → Train → Model → Evaluate → Results
```

### Preprocessing Steps
1. Load raw CSV files
2. Validate sequences
3. Encode sequences (one-hot)
4. Split train/test
5. Save as NPZ files

### Training Workflow
1. Load preprocessed data
2. Build model architecture
3. Configure training parameters
4. Train with validation
5. Save checkpoints
6. Evaluate on test set

### Evaluation Metrics

**Classification (CCLMoff)**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Confusion Matrix

**Regression (CRISPR_HNN)**
- MSE, RMSE, MAE
- Pearson Correlation (PCC)
- Spearman Correlation (SCC)

## 🎯 Development Checklist

- [ ] Read README.md
- [ ] Review SETUP_GUIDE.md
- [ ] Run verify_structure.py
- [ ] Install dependencies
- [ ] Place data in data/raw/
- [ ] Run preprocessing
- [ ] Create training scripts
- [ ] Train models
- [ ] Evaluate performance
- [ ] Generate visualizations
- [ ] Document results
- [ ] Commit to Git

## 💡 Tips & Best Practices

1. **Data Organization**
   - Keep raw data in `data/raw/`
   - Always preprocess before training
   - Save processed data as NPZ files

2. **Configuration**
   - Use YAML config files
   - Document hyperparameters
   - Version control configurations

3. **Results**
   - Save plots with high DPI (300)
   - Export metrics as CSV/JSON
   - Keep predictions for analysis

4. **Development**
   - Use Jupyter notebooks for exploration
   - Create reusable scripts
   - Document code with docstrings
   - Use version control (Git)

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing dependencies | `pip install -r requirements.txt --upgrade` |
| Data format errors | Check CSV column names and sequence lengths |
| Memory issues | Reduce batch size or process in chunks |
| Import errors | Ensure you're in project root directory |

## 📞 Support

For help:
1. Check README.md
2. Review SETUP_GUIDE.md
3. Check utility docstrings
4. Review example notebooks
5. Check configuration template

## 📝 Project Metadata

- **Name**: CRISPR-UniPredict
- **Version**: 1.0.0
- **Status**: Ready for Development ✓
- **Python**: 3.7+
- **License**: MIT

---

**Last Updated**: 2024
**Project Status**: ✓ Complete and Ready for Development
