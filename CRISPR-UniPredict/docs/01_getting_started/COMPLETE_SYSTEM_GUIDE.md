# CRISPR-UniPredict: Complete System Guide

## 🎯 Project Overview

**CRISPR-UniPredict** is a production-ready unified hybrid neural network for CRISPR prediction with:

- ✅ **Hybrid Architecture**: Three complementary branches (CNN, RNN, Pretrained)
- ✅ **Configuration System**: YAML-based hyperparameter management
- ✅ **Dataset System**: Efficient PyTorch Dataset with caching
- ✅ **Multi-Task Learning**: On-target and off-target prediction
- ✅ **Complete Documentation**: Guides for every component

---

## 📚 Documentation Structure

### Level 1: Quick Start (5-10 minutes)
Start here for immediate usage:
- **[QUICKSTART.md](QUICKSTART.md)** - Basic usage examples
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration quick reference
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** - Dataset quick reference

### Level 2: Implementation Overview (15-20 minutes)
Understand what was built:
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Model implementation
- **[CONFIGURATION_AND_DATASET_SUMMARY.md](CONFIGURATION_AND_DATASET_SUMMARY.md)** - Config & dataset implementation

### Level 3: Detailed Architecture (30-45 minutes)
Deep dive into technical details:
- **[CRISPR_UNIPREDICT_ARCHITECTURE.md](CRISPR_UNIPREDICT_ARCHITECTURE.md)** - Model architecture details
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Full configuration reference
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** - Full dataset reference

### Level 4: Source Code (as needed)
Explore implementation:
- **[models/crispr_unipredict.py](models/crispr_unipredict.py)** - Main model
- **[configs/config_loader.py](configs/config_loader.py)** - Configuration loader
- **[utils/preprocessing/dataset.py](utils/preprocessing/dataset.py)** - Dataset implementation

---

## 🚀 Getting Started in 5 Minutes

### 1. Load Configuration
```python
from configs.config_loader import ConfigLoader

config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config

print(f"Model: {config.model.name}")
print(f"Batch size: {config.training.batch_size}")
```

### 2. Prepare Data
```python
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader

encoder = SequenceEncoder(device='cuda')

dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    use_cache=True
)

dataloader = CRISPRDataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=4
)
```

### 3. Initialize Model
```python
from models.crispr_unipredict import CRISPRUniPredict

model = CRISPRUniPredict(
    seq_len=config.model.encoding.max_sequence_length,
    msc_out_channels=config.model.msc.out_channels,
    mhsa_embed_dim=config.model.mhsa.embed_dim,
    bigru_hidden_dim=config.model.bigru.hidden_dim,
    embedding_dim=config.model.encoding.embedding_dim,
    hidden_dim=config.model.fusion.hidden_dim,
    dropout=config.model.msc.dropout,
    device='cuda'
)
```

### 4. Train Model
```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate_heads,
    weight_decay=config.training.weight_decay
)

for epoch in range(config.training.epochs):
    model.train()
    
    for batch in dataloader:
        sgrna_onehot = batch['sgrna_onehot'].to('cuda')
        sgrna_label = batch['sgrna_label'].to('cuda')
        on_target_label = batch['on_target_score'].to('cuda')
        
        on_target_pred = model.predict_on_target(sgrna_onehot, sgrna_label)
        loss = criterion(on_target_pred, on_target_label.unsqueeze(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📁 Project Structure

```
CRISPR-UniPredict/
├── configs/
│   ├── model_config.yaml              # Main configuration
│   └── config_loader.py               # Configuration loader
│
├── models/
│   ├── crispr_unipredict.py           # Main model
│   ├── msc_module.py                  # Multi-Scale Convolution
│   ├── mhsa_module.py                 # Multi-Head Self-Attention
│   ├── bigru_module.py                # Bidirectional GRU
│   ├── rna_fm_encoder.py              # RNA-FM encoder
│   └── encoding.py                    # Sequence encoding
│
├── utils/
│   └── preprocessing/
│       └── dataset.py                 # PyTorch Dataset
│
├── data/
│   └── processed/
│       └── combined/
│           ├── train.csv              # Training data
│           ├── val.csv                # Validation data
│           └── test.csv               # Test data
│
├── logs/                              # Training logs
├── results/                           # Results and checkpoints
│
├── QUICKSTART.md                      # Quick start guide
├── CRISPR_UNIPREDICT_ARCHITECTURE.md  # Architecture details
├── CRISPR_UNIPREDICT_README.md        # Model overview
├── CONFIG_GUIDE.md                    # Configuration guide
├── DATASET_GUIDE.md                   # Dataset guide
├── IMPLEMENTATION_SUMMARY.md          # Implementation details
├── CONFIGURATION_AND_DATASET_SUMMARY.md # Config & dataset details
└── COMPLETE_SYSTEM_GUIDE.md           # This file
```

---

## 🔧 Configuration System

### Key Features
- **YAML-based**: Easy to read and modify
- **Type-safe**: Dataclass-based configuration
- **Hierarchical**: Organized into logical sections
- **Flexible**: Load, update, and save configurations
- **Validated**: Automatic validation with sensible defaults

### Configuration Sections
1. **model**: Architecture settings (MSC, MHSA, BiGRU, RNA-FM, Fusion)
2. **training**: Training hyperparameters (batch size, learning rates, optimizer)
3. **validation**: Validation settings (early stopping, metrics)
4. **data**: Data paths and loading settings
5. **device**: GPU and device settings
6. **logging**: Logging and monitoring (W&B, checkpoints)
7. **inference**: Inference-specific settings
8. **model_selection**: Model selection strategy
9. **ensemble**: Ensemble learning settings

### Quick Configuration Update
```python
# Load configuration
loader = ConfigLoader('configs/model_config.yaml')

# Update values
loader.update({
    'training': {
        'batch_size': 64,
        'epochs': 200
    }
})

# Save updated configuration
loader.save('configs/custom_config.yaml')
```

---

## 📊 Dataset System

### Key Features
- **Efficient caching**: Encode sequences once, use many times
- **Task filtering**: Filter by on-target or off-target
- **Data augmentation**: Reverse complement, noise
- **Statistics**: Get dataset statistics
- **Batch collation**: Automatic padding for variable-length sequences
- **Metadata handling**: Preserve dataset source and other info

### Quick Dataset Usage
```python
# Load dataset
dataset = CRISPRDataset(
    csv_path='data/train.csv',
    encoder=encoder,
    use_cache=True,
    augmentation=True
)

# Get statistics
stats = dataset.get_statistics()
print(f"Total samples: {stats['total_samples']}")
print(f"On-target samples: {stats['on_target_samples']}")

# Filter by task
on_target_dataset = dataset.filter_by_task('on_target')

# Create dataloader
dataloader = CRISPRDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate batches
for batch in dataloader:
    sgrna_onehot = batch['sgrna_onehot']
    on_target = batch['on_target_score']
    # Training code
```

---

## 🏗️ Model Architecture

### Three-Branch Design

```
Input: sgRNA (23bp)
│
├─ Branch A: CNN-based
│  └─ One-hot → MSC (256) → MHSA (4 heads) → Pool
│
├─ Branch B: RNN-based
│  └─ Label → Embedding (128) → BiGRU (256) → Pool
│
└─ Branch C: Pretrained
   └─ Label → RNA-FM (640) → Pool

Fusion: Attention-based combination
│
Shared Representation (256)
│
├─ On-Target Head: Dense(256→80→20→1) + Sigmoid
└─ Off-Target Head: Dense(256→80→20→1) + Sigmoid
```

### Total Parameters
- **Total**: 1,992,565 parameters
- **Model size**: ~8 MB
- **Trainable**: All parameters trainable by default

---

## 📈 Training Workflow

### Step 1: Prepare Configuration
```python
from configs.config_loader import ConfigLoader

config_loader = ConfigLoader('configs/model_config.yaml')
config = config_loader.config
```

### Step 2: Load Data
```python
from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader

encoder = SequenceEncoder(device='cuda')

train_dataset = CRISPRDataset(config.data.train_path, encoder)
val_dataset = CRISPRDataset(config.data.val_path, encoder)

train_loader = CRISPRDataLoader(train_dataset, batch_size=config.training.batch_size)
val_loader = CRISPRDataLoader(val_dataset, batch_size=config.inference.batch_size)
```

### Step 3: Initialize Model
```python
from models.crispr_unipredict import CRISPRUniPredict

model = CRISPRUniPredict(
    seq_len=config.model.encoding.max_sequence_length,
    msc_out_channels=config.model.msc.out_channels,
    mhsa_embed_dim=config.model.mhsa.embed_dim,
    bigru_hidden_dim=config.model.bigru.hidden_dim,
    embedding_dim=config.model.encoding.embedding_dim,
    hidden_dim=config.model.fusion.hidden_dim,
    dropout=config.model.msc.dropout,
    device='cuda'
)
```

### Step 4: Setup Training
```python
import torch.optim as optim
import torch.nn as nn

optimizer = optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate_heads,
    weight_decay=config.training.weight_decay
)

criterion_on_target = nn.MSELoss()
criterion_off_target = nn.BCELoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config.training.scheduler.factor,
    patience=config.training.scheduler.patience
)
```

### Step 5: Train
```python
for epoch in range(config.training.epochs):
    # Training
    model.train()
    train_dataset.train()
    
    for batch in train_loader:
        sgrna_onehot = batch['sgrna_onehot'].to('cuda')
        sgrna_label = batch['sgrna_label'].to('cuda')
        on_target_label = batch['on_target_score'].to('cuda')
        off_target_label = batch['off_target_label'].to('cuda')
        
        on_target_pred, off_target_pred = model(
            sgrna_onehot, sgrna_label, task_type='both'
        )
        
        loss_on = criterion_on_target(on_target_pred, on_target_label.unsqueeze(-1))
        loss_off = criterion_off_target(off_target_pred, off_target_label.unsqueeze(-1).float())
        loss = (config.training.loss.loss_weights['on_target'] * loss_on +
                config.training.loss.loss_weights['off_target'] * loss_off)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
    
    # Validation
    model.eval()
    val_dataset.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            # Validation code
            pass
```

---

## 🎓 Learning Path

### For Beginners
1. Read [QUICKSTART.md](QUICKSTART.md) (5 min)
2. Run basic example (5 min)
3. Read [CONFIG_GUIDE.md](CONFIG_GUIDE.md) (10 min)
4. Read [DATASET_GUIDE.md](DATASET_GUIDE.md) (10 min)

### For Intermediate Users
1. Read [CRISPR_UNIPREDICT_ARCHITECTURE.md](CRISPR_UNIPREDICT_ARCHITECTURE.md) (20 min)
2. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) (15 min)
3. Explore source code (30 min)
4. Run training example (30 min)

### For Advanced Users
1. Review all architecture details (30 min)
2. Study source code thoroughly (60 min)
3. Modify configuration for your use case (30 min)
4. Implement custom training loop (60 min)

---

## 🔍 Key Components

### 1. Model Architecture
- **File**: `models/crispr_unipredict.py`
- **Class**: `CRISPRUniPredict`
- **Parameters**: 1,992,565
- **Features**: Multi-branch, multi-task, selective training

### 2. Configuration System
- **File**: `configs/config_loader.py`
- **Class**: `ConfigLoader`
- **Features**: YAML loading, validation, updates, saving

### 3. Dataset System
- **File**: `utils/preprocessing/dataset.py`
- **Classes**: `CRISPRDataset`, `CRISPRDataLoader`
- **Features**: Caching, filtering, augmentation, batch collation

### 4. Sequence Encoding
- **File**: `models/encoding.py`
- **Class**: `SequenceEncoder`
- **Features**: One-hot and label encoding

---

## 💡 Tips and Tricks

### Configuration
- Use presets for quick setup (quick, standard, large-scale, fine-tuning)
- Update configuration programmatically for experiments
- Save configuration with results for reproducibility

### Dataset
- Enable caching for faster subsequent loads
- Use task filtering to focus on specific predictions
- Enable augmentation for better generalization

### Training
- Start with smaller batch size and increase gradually
- Use learning rate scheduling for better convergence
- Monitor both on-target and off-target metrics

### Inference
- Set model to eval mode before inference
- Disable augmentation for consistent predictions
- Use batch processing for efficiency

---

## 🐛 Troubleshooting

### Configuration Issues
- **File not found**: Check path is correct and file exists
- **Invalid YAML**: Check indentation (2 spaces) and syntax
- **Type mismatch**: Ensure values match expected types

### Dataset Issues
- **Missing columns**: Check CSV has required columns (sgrna, target)
- **Out of memory**: Reduce batch size or disable caching
- **Slow loading**: Enable caching and increase num_workers

### Training Issues
- **Poor convergence**: Try lower learning rate or adjust loss weights
- **Out of memory**: Reduce batch size or use gradient accumulation
- **Slow training**: Increase num_workers or use mixed precision

---

## 📞 Support Resources

### Documentation
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Configuration**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **Dataset**: [DATASET_GUIDE.md](DATASET_GUIDE.md)
- **Architecture**: [CRISPR_UNIPREDICT_ARCHITECTURE.md](CRISPR_UNIPREDICT_ARCHITECTURE.md)

### Implementation Details
- **Model**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Config & Dataset**: [CONFIGURATION_AND_DATASET_SUMMARY.md](CONFIGURATION_AND_DATASET_SUMMARY.md)
- **Completion**: [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md)

### Source Code
- **Model**: [models/crispr_unipredict.py](models/crispr_unipredict.py)
- **Config**: [configs/config_loader.py](configs/config_loader.py)
- **Dataset**: [utils/preprocessing/dataset.py](utils/preprocessing/dataset.py)

---

## ✅ Verification Checklist

- [x] Configuration system implemented and tested
- [x] Dataset system implemented and tested
- [x] Model architecture implemented and tested
- [x] All documentation complete
- [x] All tests passing
- [x] Production-ready code quality
- [x] Ready for training and inference

---

## 🎉 Summary

CRISPR-UniPredict provides a complete, production-ready system for CRISPR prediction with:

✅ **Hybrid Architecture**: Three complementary branches  
✅ **Configuration Management**: YAML-based hyperparameter control  
✅ **Efficient Dataset**: Caching, filtering, augmentation  
✅ **Multi-Task Learning**: On-target and off-target prediction  
✅ **Comprehensive Documentation**: Guides for all levels  
✅ **Production Quality**: Tested, optimized, ready to use  

**Start with [QUICKSTART.md](QUICKSTART.md) and explore the system!**

---

**Last Updated**: 2024  
**Status**: Production Ready ✓  
**Tests**: All Passing ✓  
**Documentation**: Complete ✓
