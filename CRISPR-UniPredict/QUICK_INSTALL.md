# Quick Installation Guide

## Option 1: Automatic Installation (Recommended)

Run the automatic installer script:

```bash
python install_all_dependencies.py
```

This will:
1. ✓ Install all core dependencies
2. ✓ Install all optional dependencies (including ptflops)
3. ✓ Verify installation
4. ✓ Show next steps

**Time:** 10-30 minutes depending on internet speed

## Option 2: Manual Installation with pip

Install everything at once:

```bash
pip install -r requirements.txt
```

This installs:
- ✓ PyTorch 2.0.0
- ✓ NumPy, Pandas, SciPy, scikit-learn
- ✓ Transformers, BioPython
- ✓ **ptflops** (model complexity analysis)
- ✓ **hydra-core** (configuration management)
- ✓ **tensorboard** (visualization)
- ✓ All other dependencies

**Time:** 10-30 minutes

## Option 3: Install Only Core (Minimal)

If you want to install only essential packages:

```bash
pip install torch==2.0.0 numpy pandas scipy scikit-learn biopython transformers
```

Then add optional packages as needed:

```bash
# Add ptflops
pip install ptflops==0.6.6

# Add hydra
pip install hydra-core==1.1.1 omegaconf==2.1.1

# Add tensorboard
pip install tensorboard==2.10.0
```

## Option 4: Install from Script (Advanced)

Create a custom installation script:

```bash
# Create installation script
cat > install.sh << 'EOF'
#!/bin/bash
echo "Installing CRISPR-UniPredict dependencies..."
pip install torch==2.0.0
pip install numpy==1.23.5
pip install pandas==1.5.2
pip install scipy==1.9.3
pip install scikit-learn==1.1.3
pip install transformers==4.30.0
pip install biopython==1.81
pip install ptflops==0.6.6
pip install hydra-core==1.1.1
pip install omegaconf==2.1.1
pip install tensorboard==2.10.0
echo "Installation complete!"
EOF

# Run installation
chmod +x install.sh
./install.sh
```

## Verify Installation

After installation, verify everything works:

```bash
# Check Python packages
python -c "import torch, numpy, pandas, ptflops; print('✓ All packages installed')"

# Run module tests
python models/bigru_module.py
python models/mhsa_module.py

# Verify RNA-FM encoder (if RNA-FM is available)
python verify_rna_fm.py
```

## Troubleshooting

### "pip: command not found"

Use python -m pip instead:

```bash
python -m pip install -r requirements.txt
```

### "Permission denied"

Add `--user` flag:

```bash
pip install --user -r requirements.txt
```

### "CUDA not available"

Install CPU-only PyTorch:

```bash
pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1
```

### "ModuleNotFoundError: No module named 'pip'"

Upgrade pip:

```bash
python -m ensurepip --upgrade
```

### Installation takes too long

Check internet connection and try again. You can also install packages one at a time:

```bash
pip install torch
pip install numpy
pip install pandas
# ... etc
```

### "No space left on device"

PyTorch is large (~2GB). Ensure you have enough disk space:

```bash
# Check disk space
df -h

# Or on Windows
wmic logicaldisk get name,size,freespace
```

## What Gets Installed

### Core Dependencies (Required)
- **PyTorch 2.0.0** - Deep learning framework
- **NumPy 1.23.5** - Numerical computing
- **Pandas 1.5.2** - Data manipulation
- **SciPy 1.9.3** - Scientific computing
- **scikit-learn 1.1.3** - Machine learning
- **Transformers 4.30.0** - NLP models
- **BioPython 1.81** - Bioinformatics

### Optional Dependencies (Recommended)
- **ptflops 0.6.6** - Model complexity analysis
- **hydra-core 1.1.1** - Configuration management
- **omegaconf 2.1.1** - Config handling
- **tensorboard 2.10.0** - Visualization
- **pytorch-ignite 0.4.10** - Training utilities

### Development Tools
- **Jupyter, JupyterLab** - Interactive notebooks
- **pytest** - Testing framework
- **black, flake8, isort** - Code formatting
- **sphinx** - Documentation

## Next Steps

After installation:

1. **Setup RNA-FM** (optional but recommended):
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main
   python verify_rna_fm.py
   ```

2. **Run module tests**:
   ```bash
   python models/bigru_module.py
   python models/mhsa_module.py
   ```

3. **Start using CRISPR-UniPredict**:
   ```python
   from models.rna_fm_encoder import RNAFMEncoder
   from models.bigru_module import BiGRUModule
   from models.mhsa_module import MultiHeadSelfAttention
   ```

## Installation Comparison

| Method | Time | Ease | Completeness |
|--------|------|------|--------------|
| Automatic (Option 1) | 10-30 min | ⭐⭐⭐⭐⭐ | ✓ Complete |
| pip install (Option 2) | 10-30 min | ⭐⭐⭐⭐ | ✓ Complete |
| Core only (Option 3) | 5-10 min | ⭐⭐⭐ | ⚠ Minimal |
| Custom script (Option 4) | 10-30 min | ⭐⭐ | ✓ Complete |

## Getting Help

If installation fails:

1. Check error message carefully
2. Try installing one package at a time
3. Check Python version: `python --version` (should be 3.8+)
4. Check pip version: `pip --version`
5. Update pip: `pip install --upgrade pip`
6. Check internet connection
7. Try a different installation method

## Summary

**Recommended:** Run `python install_all_dependencies.py`

This will:
- ✓ Install everything automatically
- ✓ Handle errors gracefully
- ✓ Verify installation
- ✓ Show next steps

**Time:** ~20 minutes
**Difficulty:** Easy
**Result:** Fully functional CRISPR-UniPredict with all features
