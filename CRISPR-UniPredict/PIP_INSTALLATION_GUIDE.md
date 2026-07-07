# Pip Installation Guide for CRISPR-UniPredict

## Overview

This guide provides instructions for installing CRISPR-UniPredict using pip instead of conda. Both `requirements.txt` and `requirements_complete.txt` are provided for different installation scenarios.

---

## Files Available

### 1. `requirements.txt` (Original)
- Contains the basic packages
- Lighter weight
- Good for quick setup

### 2. `requirements_complete.txt` (Recommended)
- Contains all packages including development tools
- Complete setup with testing and documentation tools
- Recommended for full development

---

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **pip**: Latest version (upgrade with `python -m pip install --upgrade pip`)
- **Virtual Environment**: Recommended (venv or virtualenv)

### Check Python Version
```bash
python --version
```

Should show: `Python 3.9.x` or higher

### Upgrade pip
```bash
python -m pip install --upgrade pip
```

---

## Installation Methods

### Method 1: Using Virtual Environment (Recommended)

#### Step 1: Create Virtual Environment
```bash
python -m venv crispr_env
```

#### Step 2: Activate Virtual Environment

**On Windows (Command Prompt):**
```bash
crispr_env\Scripts\activate
```

**On Windows (PowerShell):**
```bash
crispr_env\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
source crispr_env/bin/activate
```

#### Step 3: Upgrade pip in Virtual Environment
```bash
python -m pip install --upgrade pip
```

#### Step 4: Install Requirements
```bash
pip install -r requirements_complete.txt
```

#### Step 5: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### Method 2: Direct Installation (System-wide)

**Not recommended** as it may conflict with system packages.

```bash
pip install -r requirements_complete.txt
```

---

### Method 3: PyTorch with GPU Support

If you need CUDA 11.8 support specifically:

#### Step 1: Install PyTorch with CUDA
```bash
pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 2: Install Other Requirements
```bash
pip install -r requirements_complete.txt
```

---

## Quick Start Commands

### Complete Setup in One Block

```bash
# Navigate to project
cd "c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict"

# Create virtual environment
python -m venv crispr_env

# Activate (Windows Command Prompt)
crispr_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements_complete.txt

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Package Categories

### Core Scientific Computing (4 packages)
```
numpy==1.23.5
pandas==1.5.2
scipy==1.9.3
scikit-learn==1.1.3
```

### PyTorch & Deep Learning (3 packages)
```
torch==2.0.0
torchvision==0.15.2
torchaudio==2.0.1
```

### Transformers & Language Models (3 packages)
```
transformers==4.30.0
fair-esm==2.0.0
pytorch-lightning==2.0.0
```

### Visualization (3 packages)
```
matplotlib==3.6.2
seaborn==0.12.1
plotly==5.14.0
```

### Experiment Tracking (3 packages)
```
wandb==0.15.0
tensorboard==2.13.0
hydra-core==1.3.0
```

### Bioinformatics (1 package)
```
biopython==1.81
```

### Utilities (4 packages)
```
pyyaml==6.0
tqdm==4.65.0
python-dotenv==1.0.0
requests==2.31.0
```

### Jupyter & Development (4 packages)
```
jupyter==1.0.0
jupyterlab==3.6.3
ipython==8.12.0
ipykernel==6.22.0
```

### Testing & Code Quality (5 packages)
```
pytest==7.3.1
pytest-cov==4.1.0
black==23.3.0
flake8==6.0.0
isort==5.12.0
```

### Documentation (2 packages)
```
sphinx==6.3.0
sphinx-rtd-theme==1.2.0
```

---

## Verification Steps

### 1. Check Python Version
```bash
python --version
```
Expected: `Python 3.9.x`

### 2. Check PyTorch Installation
```bash
python -c "import torch; print(torch.__version__)"
```
Expected: `2.0.0`

### 3. Check CUDA Availability
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected: `True` (if GPU available)

### 4. Check Transformers
```bash
python -c "import transformers; print(transformers.__version__)"
```
Expected: `4.30.0`

### 5. Check BioPython
```bash
python -c "import Bio; print(Bio.__version__)"
```
Expected: `1.81`

### 6. Check All Key Packages
```bash
python -c "import torch, transformers, biopython, wandb, tensorboard, hydra; print('✓ All packages installed')"
```

---

## Common Issues & Solutions

### Issue: "pip: command not found"
**Solution**: Use `python -m pip` instead
```bash
python -m pip install -r requirements_complete.txt
```

### Issue: "No module named torch"
**Solution**: Reinstall PyTorch with CUDA support
```bash
pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA not available (shows False)
**Solution**: 
1. Check NVIDIA drivers: `nvidia-smi`
2. Install NVIDIA GPU drivers if needed
3. Reinstall PyTorch with CUDA support

### Issue: Permission denied
**Solution**: Use virtual environment or add `--user` flag
```bash
pip install --user -r requirements_complete.txt
```

### Issue: Conflicting dependencies
**Solution**: Clear pip cache and reinstall
```bash
pip cache purge
pip install -r requirements_complete.txt --force-reinstall
```

### Issue: Slow installation
**Solution**: Use faster pip resolver
```bash
pip install -r requirements_complete.txt --use-deprecated=legacy-resolver
```

---

## Virtual Environment Management

### Deactivate Environment
```bash
deactivate
```

### Remove Virtual Environment
```bash
# On Windows
rmdir /s crispr_env

# On macOS/Linux
rm -rf crispr_env
```

### Create Requirements from Current Environment
```bash
pip freeze > requirements_frozen.txt
```

### Install from Frozen Requirements
```bash
pip install -r requirements_frozen.txt
```

---

## Upgrading Packages

### Upgrade All Packages
```bash
pip install --upgrade -r requirements_complete.txt
```

### Upgrade Specific Package
```bash
pip install --upgrade torch
```

### Check Outdated Packages
```bash
pip list --outdated
```

---

## Comparison: Conda vs Pip

| Feature | Conda | Pip |
|---------|-------|-----|
| Speed | Slower | Faster |
| Dependency Resolution | Better | Good |
| CUDA Support | Native | Via index-url |
| Virtual Environments | Built-in | Via venv |
| Package Selection | Smaller | Larger |
| Recommended | For GPU | For CPU/Mixed |

---

## After Installation

### Start Jupyter Lab
```bash
jupyter lab
```

### Run Python Scripts
```bash
python scripts/train_on_target.py
```

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black scripts/
```

### Lint Code
```bash
flake8 scripts/
```

---

## Troubleshooting Checklist

- [ ] Python 3.9+ installed
- [ ] pip upgraded to latest version
- [ ] Virtual environment created and activated
- [ ] PyTorch installed with correct CUDA version
- [ ] All packages installed without errors
- [ ] Verification commands pass
- [ ] Jupyter Lab starts successfully
- [ ] Python scripts run without import errors

---

## System Requirements

### Minimum
- 4 GB RAM
- 10 GB disk space
- Python 3.9+

### Recommended
- 8+ GB RAM
- 20 GB disk space
- NVIDIA GPU with CUDA 11.8 support
- 50+ GB for datasets

---

## Next Steps

1. **Create virtual environment**
   ```bash
   python -m venv crispr_env
   ```

2. **Activate environment**
   ```bash
   crispr_env\Scripts\activate  # Windows
   source crispr_env/bin/activate  # macOS/Linux
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements_complete.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

5. **Start development**
   ```bash
   jupyter lab
   ```

---

## Support & Resources

### Official Documentation
- **pip**: https://pip.pypa.io/
- **PyTorch**: https://pytorch.org/docs/
- **Transformers**: https://huggingface.co/docs/transformers/
- **BioPython**: https://biopython.org/

### Project Documentation
- README.md
- SETUP_GUIDE.md
- CONDA_SETUP.md
- ENVIRONMENT_SUMMARY.md

---

## Summary

✓ Two requirements files provided
✓ Complete installation instructions
✓ Virtual environment setup
✓ Verification steps included
✓ Troubleshooting guide available

**Ready to install? Start with:**
```bash
python -m venv crispr_env
crispr_env\Scripts\activate
pip install -r requirements_complete.txt
```

---

*Last Updated: 2024*
*Status: Ready for Use ✓*
