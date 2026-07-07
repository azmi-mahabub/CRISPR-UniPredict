# Installation Methods Comparison: Conda vs Pip

## Overview

CRISPR-UniPredict provides two installation methods:
1. **Conda** (via `environment.yml`)
2. **Pip** (via `requirements.txt` or `requirements_complete.txt`)

This document helps you choose the best method for your setup.

---

## Quick Comparison Table

| Feature | Conda | Pip |
|---------|-------|-----|
| **Installation Speed** | Slower (5-15 min) | Faster (10-20 min) |
| **Dependency Resolution** | Excellent | Good |
| **CUDA Support** | Native | Via index-url |
| **Virtual Environments** | Built-in | Via venv |
| **Package Selection** | Medium | Large |
| **Ease of Use** | Easy | Very Easy |
| **GPU Support** | Excellent | Good |
| **Reproducibility** | Excellent | Good |
| **Cross-platform** | Excellent | Excellent |
| **Recommended For** | GPU/Production | CPU/Development |

---

## Conda Installation

### Advantages
✓ Native CUDA support without special flags
✓ Better dependency resolution
✓ Easier GPU setup
✓ Better for production environments
✓ Integrated virtual environment management
✓ More stable for complex dependencies

### Disadvantages
✗ Slower installation
✗ Larger download size
✗ Requires conda installation
✗ More disk space

### When to Use Conda
- Working with GPUs
- Production environments
- Complex dependency chains
- Team collaboration
- Long-term projects

### Installation Command
```bash
conda env create -f environment.yml
conda activate crispr_unipredict
```

### Files
- `environment.yml` - Main environment file
- `CONDA_SETUP.md` - Detailed guide
- `QUICK_SETUP.txt` - Quick reference

---

## Pip Installation

### Advantages
✓ Faster installation
✓ Simpler setup
✓ No conda required
✓ Larger package selection
✓ Lighter weight
✓ Easier for beginners

### Disadvantages
✗ Requires special flags for CUDA
✗ Slightly weaker dependency resolution
✗ Manual virtual environment setup
✗ Less stable for complex dependencies

### When to Use Pip
- CPU-only development
- Quick prototyping
- Lightweight environments
- Beginners
- Simple projects

### Installation Command
```bash
python -m venv crispr_env
crispr_env\Scripts\activate
pip install -r requirements_complete.txt
```

### Files
- `requirements.txt` - Basic packages
- `requirements_complete.txt` - All packages (recommended)
- `PIP_INSTALLATION_GUIDE.md` - Detailed guide
- `PIP_QUICK_START.txt` - Quick reference

---

## Detailed Comparison

### Installation Process

#### Conda
```bash
# 1. Navigate to project
cd "c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict"

# 2. Create environment
conda env create -f environment.yml

# 3. Activate
conda activate crispr_unipredict
```

#### Pip
```bash
# 1. Navigate to project
cd "c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict"

# 2. Create virtual environment
python -m venv crispr_env

# 3. Activate
crispr_env\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install requirements
pip install -r requirements_complete.txt
```

### Package Management

#### Conda
```bash
# Install new package
conda install package_name

# Update package
conda update package_name

# List packages
conda list

# Export environment
conda env export > environment_exported.yml
```

#### Pip
```bash
# Install new package
pip install package_name

# Update package
pip install --upgrade package_name

# List packages
pip list

# Export requirements
pip freeze > requirements_frozen.txt
```

### Virtual Environment Management

#### Conda
```bash
# List environments
conda env list

# Deactivate
conda deactivate

# Remove environment
conda remove --name crispr_unipredict --all
```

#### Pip
```bash
# List environments (manual tracking)
# No built-in command

# Deactivate
deactivate

# Remove environment
rmdir /s crispr_env  # Windows
rm -rf crispr_env    # macOS/Linux
```

### CUDA Support

#### Conda
```bash
# CUDA support is automatic
conda env create -f environment.yml
# PyTorch 2.0.0 with CUDA 11.8 installed automatically
```

#### Pip
```bash
# Requires special index-url for CUDA
pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Then install other packages
pip install -r requirements_complete.txt
```

---

## Decision Tree

```
Do you have a GPU?
├─ YES
│  ├─ Do you want easiest setup?
│  │  ├─ YES → Use CONDA
│  │  └─ NO → Use PIP (with CUDA index-url)
│  └─ Do you need production stability?
│     ├─ YES → Use CONDA
│     └─ NO → Use PIP
│
└─ NO (CPU only)
   ├─ Do you want fastest installation?
   │  ├─ YES → Use PIP
   │  └─ NO → Use CONDA
   └─ Do you need complex dependencies?
      ├─ YES → Use CONDA
      └─ NO → Use PIP
```

---

## System Requirements

### For Conda
- Miniconda or Anaconda installed
- 2-5 GB additional disk space
- 4+ GB RAM

### For Pip
- Python 3.9+ installed
- 1-3 GB additional disk space
- 4+ GB RAM

---

## Performance Comparison

### Installation Time
- **Conda**: 5-15 minutes
- **Pip**: 10-20 minutes

### Disk Space
- **Conda**: 3-5 GB
- **Pip**: 2-4 GB

### Runtime Performance
- **Conda**: Identical to Pip
- **Pip**: Identical to Conda

### GPU Performance
- **Conda**: Excellent
- **Pip**: Excellent (with correct CUDA)

---

## Troubleshooting

### Conda Issues

| Issue | Solution |
|-------|----------|
| "conda: command not found" | Install Miniconda/Anaconda |
| CUDA not available | Reinstall PyTorch from conda-forge |
| Slow installation | Use mamba solver |
| Package conflicts | Clear cache: `conda clean --all` |

### Pip Issues

| Issue | Solution |
|-------|----------|
| "pip: command not found" | Use `python -m pip` |
| CUDA not available | Use correct index-url for PyTorch |
| Permission denied | Use virtual environment or `--user` |
| Conflicting packages | Clear cache: `pip cache purge` |

---

## Migration Between Methods

### From Conda to Pip
```bash
# Export conda environment
conda env export > environment_exported.yml

# Create pip requirements from conda
pip freeze > requirements_migrated.txt

# Install with pip
pip install -r requirements_migrated.txt
```

### From Pip to Conda
```bash
# Create conda environment from pip requirements
conda create --name crispr_unipredict --file requirements.txt

# Or manually create environment.yml from requirements.txt
```

---

## Recommendations by Use Case

### Development (Recommended: Pip)
```bash
python -m venv crispr_env
crispr_env\Scripts\activate
pip install -r requirements_complete.txt
```
- Faster setup
- Easy to modify
- Good for experimentation

### Production (Recommended: Conda)
```bash
conda env create -f environment.yml
conda activate crispr_unipredict
```
- Better stability
- Easier reproducibility
- Better for deployment

### GPU Development (Recommended: Conda)
```bash
conda env create -f environment.yml
conda activate crispr_unipredict
```
- Automatic CUDA setup
- Better GPU support
- Easier troubleshooting

### CPU-Only (Recommended: Pip)
```bash
python -m venv crispr_env
crispr_env\Scripts\activate
pip install -r requirements_complete.txt
```
- Faster installation
- Lighter weight
- Simpler setup

### Team Project (Recommended: Conda)
```bash
conda env create -f environment.yml
conda activate crispr_unipredict
```
- Better reproducibility
- Easier sharing
- Consistent across team

---

## Installation Checklist

### For Conda
- [ ] Conda installed
- [ ] environment.yml present
- [ ] Run: `conda env create -f environment.yml`
- [ ] Run: `conda activate crispr_unipredict`
- [ ] Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### For Pip
- [ ] Python 3.9+ installed
- [ ] requirements_complete.txt present
- [ ] Create venv: `python -m venv crispr_env`
- [ ] Activate venv: `crispr_env\Scripts\activate`
- [ ] Install: `pip install -r requirements_complete.txt`
- [ ] Verify: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Quick Start

### Choose Your Method

**Option 1: Conda (Recommended for GPU)**
```bash
conda env create -f environment.yml
conda activate crispr_unipredict
```

**Option 2: Pip (Recommended for CPU)**
```bash
python -m venv crispr_env
crispr_env\Scripts\activate
pip install -r requirements_complete.txt
```

---

## Support

### Conda Documentation
- https://docs.conda.io/

### Pip Documentation
- https://pip.pypa.io/

### Project Documentation
- README.md
- SETUP_GUIDE.md
- CONDA_SETUP.md
- PIP_INSTALLATION_GUIDE.md

---

## Summary

| Aspect | Conda | Pip |
|--------|-------|-----|
| **Ease** | Easy | Very Easy |
| **Speed** | Slower | Faster |
| **GPU** | Better | Good |
| **Stability** | Better | Good |
| **Recommended** | Production/GPU | Development/CPU |

**Choose based on your needs:**
- **GPU or Production?** → Use Conda
- **CPU or Development?** → Use Pip

Both methods will give you a fully functional CRISPR-UniPredict environment!

---

*Last Updated: 2024*
*Status: Ready for Use ✓*
