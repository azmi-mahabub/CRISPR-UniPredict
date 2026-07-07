# Conda Environment Setup Guide for CRISPR-UniPredict

## Environment Specifications

The `environment.yml` file has been created with the following specifications:

### Environment Details
- **Name**: `crispr_unipredict`
- **Python Version**: 3.9
- **PyTorch Version**: 2.0.0
- **CUDA Version**: 11.8

### Included Packages

#### Core Scientific Libraries
- numpy
- pandas
- scipy
- scikit-learn

#### Visualization
- matplotlib
- seaborn
- plotly

#### Deep Learning & Transformers
- PyTorch 2.0.0 with CUDA 11.8
- transformers (≥4.30.0)
- fair-esm (≥2.0.0)
- pytorch-lightning (≥2.0.0)

#### Utilities & Monitoring
- wandb (Weights & Biases)
- tensorboard
- hydra-core (configuration management)
- tqdm (progress bars)

#### Bioinformatics
- biopython (≥1.81)

#### Development Tools
- jupyter & jupyterlab
- ipython
- pytest
- black (code formatter)
- flake8 (linter)

---

## Setup Instructions

### Step 1: Verify Conda Installation

First, make sure you have Conda installed. Check the version:

```bash
conda --version
```

If Conda is not installed, download and install:
- **Miniconda**: https://docs.conda.io/projects/miniconda/en/latest/
- **Anaconda**: https://www.anaconda.com/download

### Step 2: Create the Environment

Navigate to the CRISPR-UniPredict project directory:

```bash
cd "c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict"
```

Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

**Expected output:**
```
Collecting package metadata (repodata.json): done
Solving environment: done
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate crispr_unipredict
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### Step 3: Activate the Environment

Once the environment is created, activate it:

```bash
conda activate crispr_unipredict
```

**On Windows (PowerShell):**
```powershell
conda activate crispr_unipredict
```

**On Windows (Command Prompt):**
```cmd
conda activate crispr_unipredict
```

**On macOS/Linux:**
```bash
conda activate crispr_unipredict
```

You should see the environment name in your terminal prompt:
```
(crispr_unipredict) C:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict>
```

### Step 4: Verify Installation

Verify that all packages are installed correctly:

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check other key packages
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import biopython; print(f'BioPython version: {biopython.__version__}')"
```

---

## Common Commands

### List All Environments

```bash
conda env list
```

### Deactivate Current Environment

```bash
conda deactivate
```

### Remove Environment

```bash
conda remove --name crispr_unipredict --all
```

### Update Environment

If you modify `environment.yml`, update the environment:

```bash
conda env update -f environment.yml --prune
```

### Export Current Environment

To save the current environment state:

```bash
conda env export > environment_exported.yml
```

### Install Additional Packages

While the environment is active:

```bash
conda install package_name
# or
pip install package_name
```

---

## Troubleshooting

### Issue: CUDA Not Available

If `torch.cuda.is_available()` returns `False`:

1. Check NVIDIA GPU drivers:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA toolkit installation:
   ```bash
   nvcc --version
   ```

3. Reinstall PyTorch with CUDA support:
   ```bash
   conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c conda-forge
   ```

### Issue: Package Conflicts

If you encounter package conflicts during environment creation:

1. Clear conda cache:
   ```bash
   conda clean --all
   ```

2. Try creating with strict channel priority:
   ```bash
   conda env create -f environment.yml --strict-channel-priority
   ```

### Issue: Slow Environment Creation

If environment creation is slow:

1. Use mamba (faster solver):
   ```bash
   conda install -c conda-forge mamba
   mamba env create -f environment.yml
   ```

2. Or use libmamba solver:
   ```bash
   conda config --set solver libmamba
   conda env create -f environment.yml
   ```

---

## Quick Start After Activation

Once the environment is activated, you can:

### 1. Start Jupyter Lab

```bash
jupyter lab
```

### 2. Run Python Scripts

```bash
python scripts/train_on_target.py
```

### 3. Use Interactive Python

```bash
python
```

### 4. Run Pytest

```bash
pytest tests/
```

---

## Environment File Location

The `environment.yml` file is located at:
```
c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict\environment.yml
```

---

## Next Steps

1. **Activate the environment**: `conda activate crispr_unipredict`
2. **Verify installation**: Run the verification commands above
3. **Prepare data**: Place datasets in `data/raw/`
4. **Start development**: Use the utilities and scripts in the project

---

## Additional Resources

- **Conda Documentation**: https://docs.conda.io/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Transformers Documentation**: https://huggingface.co/docs/transformers/
- **BioPython Documentation**: https://biopython.org/

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the official documentation links
3. Check the project README.md
4. Review utility module docstrings

---

**Environment created successfully!** 🎉

Start by running: `conda activate crispr_unipredict`
