# CRISPR-UniPredict Setup Guide

## Project Created Successfully! ✓

The CRISPR-UniPredict project has been created with a complete directory structure and all necessary files.

## Directory Structure Overview

```
CRISPR-UniPredict/
├── data/                          # Data management
│   ├── raw/                       # Raw datasets
│   │   ├── crispr_hnn/           # CRISPR_HNN datasets (on-target)
│   │   └── cclmoff/              # CCLMoff datasets (off-target)
│   └── processed/                # Processed datasets
│       ├── on_target/            # Processed on-target data
│       ├── off_target/           # Processed off-target data
│       └── combined/             # Combined datasets
│
├── models/                        # Model management
│   ├── checkpoints/              # Training checkpoints
│   └── pretrained/               # Pretrained model weights
│
├── results/                       # Results and outputs
│   ├── plots/                    # Visualization plots
│   ├── metrics/                  # Evaluation metrics (CSV/JSON)
│   └── predictions/              # Model predictions
│
├── utils/                         # Utility modules
│   ├── preprocessing/            # Data preprocessing
│   │   ├── sequence_utils.py     # Sequence encoding/validation
│   │   ├── crispr_hnn_preprocessor.py
│   │   └── cclmoff_preprocessor.py
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py
│   └── visualization/            # Plotting utilities
│       └── plots.py
│
├── scripts/                       # Standalone scripts
├── configs/                       # Configuration files
│   └── config_template.yaml      # Configuration template
├── logs/                          # Training logs
├── notebooks/                     # Jupyter notebooks
│
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── __init__.py                   # Package initialization
└── verify_structure.py           # Structure verification script
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Project Structure

```bash
python verify_structure.py
```

This will verify all directories and files are in place.

### 3. Prepare Your Data

Place your datasets in the appropriate directories:

- **CRISPR_HNN data**: `data/raw/crispr_hnn/`
  - Expected format: CSV with columns `sgRNA` and `indel`
  - Example: `ESP（n=58616）.csv`, `WT.csv`, etc.

- **CCLMoff data**: `data/raw/cclmoff/`
  - Expected format: CSV with columns `sgRNA_seq`, `off_seq`, `label`
  - Example: `09212024_CCLMoff_dataset.csv`

### 4. Preprocess Data

```python
from utils.preprocessing import preprocess_crispr_hnn, preprocess_cclmoff

# Process CRISPR_HNN data
stats = preprocess_crispr_hnn(
    'data/raw/crispr_hnn/',
    'data/processed/on_target/'
)

# Process CCLMoff data
stats = preprocess_cclmoff(
    'data/raw/cclmoff/',
    'data/processed/off_target/'
)
```

### 5. Train Models

Create training scripts in `scripts/` directory:

```python
# scripts/train_on_target.py
from utils.preprocessing import load_crispr_hnn_data
from utils.evaluation import regression_metrics

# Load data
X_train, X_test, y_train, y_test = load_crispr_hnn_data(
    'data/processed/on_target/ESP_processed.npz'
)

# Train model (your implementation)
# ...

# Evaluate
metrics = regression_metrics(y_test, y_pred)
print(metrics)
```

### 6. Make Predictions

```python
from utils.preprocessing import encode_sequence

# Encode sgRNA
sgRNA = "AAAAAAAAACTCCAAAACCCTGG"
X = encode_sequence(sgRNA).reshape(1, 23, 4)

# Load model and predict
# model = load_model('models/pretrained/model.h5')
# prediction = model.predict(X)
```

## Utility Modules

### Preprocessing (`utils/preprocessing/`)

#### `sequence_utils.py`
- `validate_sequence()`: Validate DNA/RNA sequences
- `encode_sequence()`: One-hot or integer encoding
- `reverse_complement()`: Get reverse complement
- `extract_kmers()`: Extract k-mers
- `calculate_gc_content()`: Calculate GC%
- `calculate_tm()`: Calculate melting temperature

#### `crispr_hnn_preprocessor.py`
- `preprocess_crispr_hnn()`: Process on-target datasets
- `load_crispr_hnn_data()`: Load preprocessed data
- `augment_sequences()`: Data augmentation

#### `cclmoff_preprocessor.py`
- `preprocess_cclmoff()`: Process off-target datasets
- `load_cclmoff_data()`: Load preprocessed data

### Evaluation (`utils/evaluation/`)

#### `metrics.py`
- `classification_metrics()`: Accuracy, precision, recall, F1, AUC
- `regression_metrics()`: MSE, MAE, Pearson, Spearman
- `compute_roc_curve()`: ROC curve computation
- `compute_confusion_matrix()`: Confusion matrix
- `compute_per_dataset_metrics()`: Per-dataset metrics

### Visualization (`utils/visualization/`)

#### `plots.py`
- `plot_metrics()`: Bar chart of metrics
- `plot_roc_curve()`: ROC curve visualization
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_predictions()`: Predictions vs true values

## Configuration

Edit `configs/config_template.yaml` to customize:

```yaml
# Model architecture
crispr_hnn:
  architecture:
    conv_filters: [10, 10, 10, 10]
    attention_heads: 4
    gru_units: 128

# Training parameters
training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.0001

# Data settings
data:
  train_split: 0.8
  validation_split: 0.1
```

## File Organization

### Data Files
- Raw data: `data/raw/*/`
- Processed data: `data/processed/*/` (NPZ format)

### Model Files
- Checkpoints: `models/checkpoints/` (during training)
- Final models: `models/pretrained/` (trained models)

### Results
- Plots: `results/plots/` (PNG/PDF)
- Metrics: `results/metrics/` (CSV/JSON)
- Predictions: `results/predictions/` (NPZ/CSV)

### Logs
- Training logs: `logs/training.log`
- Experiment logs: `logs/experiment_*.log`

## Example Workflow

```python
# 1. Import utilities
from utils.preprocessing import preprocess_crispr_hnn, load_crispr_hnn_data
from utils.evaluation import regression_metrics
from utils.visualization import plot_metrics

# 2. Preprocess data
preprocess_crispr_hnn('data/raw/crispr_hnn/', 'data/processed/on_target/')

# 3. Load data
X_train, X_test, y_train, y_test = load_crispr_hnn_data(
    'data/processed/on_target/ESP_processed.npz'
)

# 4. Train model
# model = build_model()
# model.fit(X_train, y_train, ...)

# 5. Evaluate
y_pred = model.predict(X_test)
metrics = regression_metrics(y_test, y_pred)

# 6. Visualize
plot_metrics(metrics, save_path='results/plots/metrics.png')
```

## Git Workflow

The `.gitignore` file is configured to:
- Ignore Python cache files (`__pycache__/`, `*.pyc`)
- Ignore virtual environments (`venv/`, `env/`)
- Ignore large data files (`*.csv`, `*.pkl`, `*.h5`)
- Ignore model checkpoints
- Ignore IDE files (`.vscode/`, `.idea/`)

To initialize git:

```bash
git init
git add .
git commit -m "Initial commit: CRISPR-UniPredict project structure"
```

## Next Steps

1. **Add Data**: Place your datasets in `data/raw/`
2. **Preprocess**: Run preprocessing scripts
3. **Develop Models**: Create training scripts in `scripts/`
4. **Train**: Execute training with configurations
5. **Evaluate**: Use evaluation utilities
6. **Visualize**: Generate plots and reports
7. **Document**: Add notebooks and documentation

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Data Format Issues
- Ensure CSV files have correct column names
- Check sequence lengths (23 bp for sgRNA)
- Validate sequences contain only ACGT

### Memory Issues
- Reduce batch size in config
- Process data in chunks
- Use data generators for large datasets

## Support

For questions or issues:
1. Check the README.md
2. Review example notebooks
3. Check utility docstrings
4. Consult configuration template

## Project Status

✓ Project structure created
✓ All directories initialized
✓ Utility modules implemented
✓ Configuration template ready
✓ Ready for development

**Start by placing your data in `data/raw/` and running preprocessing!**
