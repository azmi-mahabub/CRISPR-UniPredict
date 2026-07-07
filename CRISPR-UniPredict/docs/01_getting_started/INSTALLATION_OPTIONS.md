# CRISPR-UniPredict Installation Options

## Overview

We provide **4 different ways** to install all dependencies including **ptflops**:

1. **Automatic Python Script** (Recommended)
2. **Manual pip install**
3. **Windows Batch Script**
4. **Linux/Mac Shell Script**

All methods install the complete set of dependencies including:
- ✓ PyTorch 2.0.0
- ✓ **ptflops** (model complexity analysis)
- ✓ **hydra-core** (configuration management)
- ✓ **tensorboard** (visualization)
- ✓ All other optional packages

---

## Option 1: Automatic Python Script (Recommended)

### For All Platforms (Windows, Linux, Mac)

```bash
python install_all_dependencies.py
```

**What it does:**
- Automatically detects your system
- Installs all core dependencies
- Installs all optional dependencies (including ptflops)
- Verifies installation
- Shows next steps

**Advantages:**
- ✓ Works on all platforms
- ✓ Automatic error handling
- ✓ Verification included
- ✓ Clear progress messages
- ✓ Easiest method

**Time:** 10-30 minutes

**Example output:**
```
================================================================================
CRISPR-UniPredict DEPENDENCY INSTALLER
================================================================================

This script will install all dependencies for CRISPR-UniPredict
This includes:
  - Core dependencies (required)
  - Optional dependencies (ptflops, hydra, tensorboard, etc.)

Estimated time: 10-30 minutes (depends on internet speed)

Continue? (yes/no): yes

================================================================================
INSTALLING CORE DEPENDENCIES
================================================================================

CORE DEPENDENCIES (Required)
  Installing numpy==1.23.5... ✓
  Installing pandas==1.5.2... ✓
  Installing scipy==1.9.3... ✓
  ...
  ✓ All 23 packages installed successfully

================================================================================
INSTALLING OPTIONAL DEPENDENCIES
================================================================================

OPTIONAL DEPENDENCIES (RNA-FM, Advanced Features)
  Installing ptflops==0.6.6... ✓
  Installing pytorch-ignite==0.4.10... ✓
  Installing hydra-core==1.1.1... ✓
  ...
  ✓ All 8 packages installed successfully

VERIFYING INSTALLATION
  ✓ PyTorch
  ✓ NumPy
  ✓ Pandas
  ✓ scikit-learn
  ✓ BioPython
  ✓ Transformers
  ✓ ptflops (optional)
  ✓ Hydra (optional)
  ✓ TensorBoard (optional)

================================================================================
INSTALLATION SUMMARY
================================================================================

✓ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!

Next steps:
  1. Add RNA-FM to PYTHONPATH:
     export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main
  2. Verify RNA-FM encoder:
     python verify_rna_fm.py
  3. Run module tests:
     python models/bigru_module.py
     python models/mhsa_module.py
```

---

## Option 2: Manual pip Install

### Simple One-Command Installation

```bash
pip install -r requirements.txt
```

**What it does:**
- Installs all packages listed in requirements.txt
- Includes core and optional dependencies
- Takes time but straightforward

**Advantages:**
- ✓ Simple one-liner
- ✓ Standard pip method
- ✓ Works on all platforms

**Time:** 10-30 minutes

**If pip doesn't work:**
```bash
python -m pip install -r requirements.txt
```

**Install specific packages:**
```bash
# Just ptflops
pip install ptflops==0.6.6

# Just hydra
pip install hydra-core==1.1.1 omegaconf==2.1.1

# Just tensorboard
pip install tensorboard==2.10.0
```

---

## Option 3: Windows Batch Script

### For Windows Users

```bash
install_dependencies.bat
```

**What it does:**
- Automated installation for Windows
- Installs all dependencies
- Handles errors gracefully
- Shows progress

**Advantages:**
- ✓ Windows-specific optimizations
- ✓ Interactive prompts
- ✓ Error handling
- ✓ Verification included

**Time:** 10-30 minutes

**Manual execution:**
```bash
# Run from Command Prompt
cd C:\path\to\CRISPR-UniPredict
install_dependencies.bat

# Or from PowerShell
.\install_dependencies.bat
```

**What gets installed:**
- PyTorch 2.0.0
- NumPy, Pandas, SciPy
- scikit-learn
- Transformers, BioPython
- Matplotlib, Seaborn, Plotly
- Jupyter, JupyterLab
- **ptflops** ✓
- **hydra-core** ✓
- **tensorboard** ✓
- Development tools (pytest, black, flake8)

---

## Option 4: Linux/Mac Shell Script

### For Linux and Mac Users

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**What it does:**
- Automated installation for Unix-like systems
- Installs all dependencies
- Handles errors gracefully
- Shows progress

**Advantages:**
- ✓ Unix-specific optimizations
- ✓ Interactive prompts
- ✓ Error handling
- ✓ Verification included

**Time:** 10-30 minutes

**Alternative:**
```bash
bash install_dependencies.sh
```

**What gets installed:**
- PyTorch 2.0.0
- NumPy, Pandas, SciPy
- scikit-learn
- Transformers, BioPython
- Matplotlib, Seaborn, Plotly
- Jupyter, JupyterLab
- **ptflops** ✓
- **hydra-core** ✓
- **tensorboard** ✓
- Development tools (pytest, black, flake8)

---

## Comparison Table

| Method | Platform | Ease | Time | Verification |
|--------|----------|------|------|--------------|
| Python Script | All | ⭐⭐⭐⭐⭐ | 10-30 min | ✓ Automatic |
| pip install | All | ⭐⭐⭐⭐ | 10-30 min | Manual |
| Batch Script | Windows | ⭐⭐⭐⭐⭐ | 10-30 min | ✓ Automatic |
| Shell Script | Linux/Mac | ⭐⭐⭐⭐⭐ | 10-30 min | ✓ Automatic |

---

## Recommended Installation Path

### Step 1: Choose Your Method

**Windows:**
```bash
python install_all_dependencies.py
# OR
install_dependencies.bat
```

**Linux/Mac:**
```bash
python install_all_dependencies.py
# OR
./install_dependencies.sh
```

**Any Platform:**
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check core packages
python -c "import torch, numpy, pandas, sklearn; print('✓ Core packages OK')"

# Check optional packages
python -c "import ptflops, hydra, tensorboard; print('✓ Optional packages OK')"
```

### Step 3: Setup RNA-FM (Optional)

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main
python verify_rna_fm.py
```

### Step 4: Run Module Tests

```bash
python models/bigru_module.py
python models/mhsa_module.py
```

---

## What Gets Installed

### Core Packages (Required)
```
torch==2.0.0
numpy==1.23.5
pandas==1.5.2
scipy==1.9.3
scikit-learn==1.1.3
transformers==4.30.0
biopython==1.81
```

### Optional Packages (Recommended)
```
ptflops==0.6.6              ← Model complexity analysis
pytorch-ignite==0.4.10      ← Training utilities
hydra-core==1.1.1           ← Configuration management
omegaconf==2.1.1            ← Config handling
tensorboard==2.10.0         ← Visualization
google-auth==2.10.0         ← Cloud authentication
```

### Development Tools
```
jupyter==1.0.0
jupyterlab==3.6.3
pytest==7.3.1
black==23.3.0
flake8==6.0.0
sphinx==6.3.0
```

---

## Troubleshooting

### "Installation Failed"

**Try Option 1: Python Script**
```bash
python install_all_dependencies.py
```

**Try Option 2: Manual Installation**
```bash
pip install -r requirements.txt
```

**Try Option 3: Install Core Only**
```bash
pip install torch numpy pandas scipy scikit-learn transformers biopython
```

### "ptflops not found"

**Install it manually:**
```bash
pip install ptflops==0.6.6
```

**Verify:**
```bash
python -c "import ptflops; print('ptflops OK')"
```

### "Permission Denied"

**Use --user flag:**
```bash
pip install --user -r requirements.txt
```

### "No space left on device"

PyTorch is ~2GB. Ensure you have enough disk space:
```bash
# Linux/Mac
df -h

# Windows
wmic logicaldisk get name,size,freespace
```

### "Slow Installation"

Check internet connection. You can also install packages one at a time:
```bash
pip install torch
pip install numpy
pip install pandas
# ... etc
```

---

## After Installation

### Verify Everything Works

```bash
# Test imports
python -c "import torch, ptflops, hydra, tensorboard; print('✓ All OK')"

# Run module tests
python models/bigru_module.py
python models/mhsa_module.py

# Test RNA-FM (if available)
python verify_rna_fm.py
```

### Start Using CRISPR-UniPredict

```python
from models.rna_fm_encoder import RNAFMEncoder
from models.bigru_module import BiGRUModule
from models.mhsa_module import MultiHeadSelfAttention

# Your code here
```

---

## Summary

✓ **4 installation methods** - choose what works for you
✓ **All include ptflops** - no missing dependencies
✓ **Automatic verification** - know if it worked
✓ **Clear error messages** - easy troubleshooting
✓ **10-30 minutes** - typical installation time

**Recommended:** `python install_all_dependencies.py`

This will install everything including ptflops automatically! 🚀
