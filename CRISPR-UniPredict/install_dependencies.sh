#!/bin/bash

# Shell script to install all CRISPR-UniPredict dependencies on Linux/Mac

echo "================================================================================"
echo "CRISPR-UniPredict Dependency Installer (Linux/Mac)"
echo "================================================================================"
echo ""
echo "This script will install all dependencies for CRISPR-UniPredict"
echo "including ptflops, hydra-core, tensorboard, and other optional packages."
echo ""
echo "Estimated time: 10-30 minutes (depends on internet speed)"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Ask for confirmation
read -p "Continue with installation? (yes/no): " confirm
if [[ ! "$confirm" =~ ^[Yy][Ee]?[Ss]?$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "INSTALLING CORE DEPENDENCIES"
echo "================================================================================"
echo ""

# Function to install package
install_package() {
    local package=$1
    echo "Installing $package..."
    python3 -m pip install "$package"
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to install $package"
        return 1
    fi
    return 0
}

# Core dependencies
echo "Installing PyTorch..."
python3 -m pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1 || echo "WARNING: PyTorch installation may have issues"

echo "Installing NumPy, Pandas, SciPy..."
python3 -m pip install numpy==1.23.5 pandas==1.5.2 scipy==1.9.3 || echo "WARNING: Some packages failed"

echo "Installing scikit-learn..."
python3 -m pip install scikit-learn==1.1.3 || echo "WARNING: scikit-learn installation failed"

echo "Installing Transformers and BioPython..."
python3 -m pip install transformers==4.30.0 biopython==1.81 || echo "WARNING: Some packages failed"

echo "Installing visualization tools..."
python3 -m pip install matplotlib==3.6.2 seaborn==0.12.1 plotly==5.14.0 || echo "WARNING: Some packages failed"

echo "Installing Jupyter..."
python3 -m pip install jupyter==1.0.0 jupyterlab==3.6.3 ipython==8.12.0 || echo "WARNING: Some packages failed"

echo "Installing utilities..."
python3 -m pip install tqdm==4.64.1 pyyaml==6.0 python-dotenv==1.0.0 || echo "WARNING: Some packages failed"

echo "Installing development tools..."
python3 -m pip install pytest==7.3.1 black==23.3.0 flake8==6.0.0 isort==5.12.0 || echo "WARNING: Some packages failed"

echo "Installing documentation tools..."
python3 -m pip install sphinx==6.3.0 sphinx-rtd-theme==1.2.0 || echo "WARNING: Some packages failed"

echo ""
echo "================================================================================"
echo "INSTALLING OPTIONAL DEPENDENCIES"
echo "================================================================================"
echo ""

echo "Installing ptflops (model complexity analysis)..."
python3 -m pip install ptflops==0.6.6 || echo "WARNING: ptflops installation failed (optional)"

echo "Installing pytorch-ignite (training utilities)..."
python3 -m pip install pytorch-ignite==0.4.10 || echo "WARNING: pytorch-ignite installation failed (optional)"

echo "Installing Hydra (configuration management)..."
python3 -m pip install hydra-core==1.1.1 omegaconf==2.1.1 || echo "WARNING: Hydra installation failed (optional)"

echo "Installing TensorBoard (visualization)..."
python3 -m pip install tensorboard==2.10.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1 || echo "WARNING: TensorBoard installation failed (optional)"

echo "Installing additional utilities..."
python3 -m pip install regex==2021.11.10 tabulate==0.8.9 bitarray==2.5.1 || echo "WARNING: Some utilities installation failed (optional)"

echo "Installing Google authentication..."
python3 -m pip install google-auth==2.10.0 google-auth-oauthlib==0.4.6 google-auth-httplib2==0.1.0 cachetools==5.2.0 rsa==4.8 pyasn1==0.4.8 pyasn1-modules==0.2.8 || echo "WARNING: Google auth installation failed (optional)"

echo ""
echo "================================================================================"
echo "VERIFYING INSTALLATION"
echo "================================================================================"
echo ""

python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import pandas; print('Pandas:', pandas.__version__)"
python3 -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python3 -c "import transformers; print('Transformers:', transformers.__version__)"

echo ""
echo "Checking optional packages..."
python3 -c "import ptflops; print('ptflops: OK')" 2>/dev/null || echo "ptflops: NOT INSTALLED (optional)"
python3 -c "import hydra; print('Hydra: OK')" 2>/dev/null || echo "Hydra: NOT INSTALLED (optional)"
python3 -c "import tensorboard; print('TensorBoard: OK')" 2>/dev/null || echo "TensorBoard: NOT INSTALLED (optional)"

echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Setup RNA-FM (optional but recommended):"
echo "   - Add RNA-FM to PYTHONPATH"
echo "   - Run: python3 verify_rna_fm.py"
echo ""
echo "2. Run module tests:"
echo "   - python3 models/bigru_module.py"
echo "   - python3 models/mhsa_module.py"
echo ""
echo "3. Start using CRISPR-UniPredict!"
echo ""
echo "For more information, see:"
echo "  - QUICK_INSTALL.md"
echo "  - COMPLETE_SETUP.md"
echo "  - INSTALL_RNA_FM.md"
echo ""
