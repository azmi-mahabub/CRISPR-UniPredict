@echo off
REM Batch script to install all CRISPR-UniPredict dependencies on Windows

echo ================================================================================
echo CRISPR-UniPredict Dependency Installer (Windows)
echo ================================================================================
echo.
echo This script will install all dependencies for CRISPR-UniPredict
echo including ptflops, hydra-core, tensorboard, and other optional packages.
echo.
echo Estimated time: 10-30 minutes (depends on internet speed)
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Ask for confirmation
set /p confirm="Continue with installation? (yes/no): "
if /i not "%confirm%"=="yes" (
    if /i not "%confirm%"=="y" (
        echo Installation cancelled.
        pause
        exit /b 0
    )
)

echo.
echo ================================================================================
echo INSTALLING CORE DEPENDENCIES
echo ================================================================================
echo.

echo Installing PyTorch...
python -m pip install torch==2.0.0 torchvision==0.15.2 torchaudio==2.0.1
if errorlevel 1 goto error

echo Installing NumPy, Pandas, SciPy...
python -m pip install numpy==1.23.5 pandas==1.5.2 scipy==1.9.3
if errorlevel 1 goto error

echo Installing scikit-learn...
python -m pip install scikit-learn==1.1.3
if errorlevel 1 goto error

echo Installing Transformers and BioPython...
python -m pip install transformers==4.30.0 biopython==1.81
if errorlevel 1 goto error

echo Installing visualization tools...
python -m pip install matplotlib==3.6.2 seaborn==0.12.1 plotly==5.14.0
if errorlevel 1 goto error

echo Installing Jupyter...
python -m pip install jupyter==1.0.0 jupyterlab==3.6.3 ipython==8.12.0
if errorlevel 1 goto error

echo Installing utilities...
python -m pip install tqdm==4.64.1 pyyaml==6.0 python-dotenv==1.0.0
if errorlevel 1 goto error

echo Installing development tools...
python -m pip install pytest==7.3.1 black==23.3.0 flake8==6.0.0 isort==5.12.0
if errorlevel 1 goto error

echo Installing documentation tools...
python -m pip install sphinx==6.3.0 sphinx-rtd-theme==1.2.0
if errorlevel 1 goto error

echo.
echo ================================================================================
echo INSTALLING OPTIONAL DEPENDENCIES
echo ================================================================================
echo.

echo Installing ptflops (model complexity analysis)...
python -m pip install ptflops==0.6.6
if errorlevel 1 (
    echo WARNING: ptflops installation failed (optional)
)

echo Installing pytorch-ignite (training utilities)...
python -m pip install pytorch-ignite==0.4.10
if errorlevel 1 (
    echo WARNING: pytorch-ignite installation failed (optional)
)

echo Installing Hydra (configuration management)...
python -m pip install hydra-core==1.1.1 omegaconf==2.1.1
if errorlevel 1 (
    echo WARNING: Hydra installation failed (optional)
)

echo Installing TensorBoard (visualization)...
python -m pip install tensorboard==2.10.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1
if errorlevel 1 (
    echo WARNING: TensorBoard installation failed (optional)
)

echo Installing additional utilities...
python -m pip install regex==2021.11.10 tabulate==0.8.9 bitarray==2.5.1
if errorlevel 1 (
    echo WARNING: Some utilities installation failed (optional)
)

echo Installing Google authentication...
python -m pip install google-auth==2.10.0 google-auth-oauthlib==0.4.6 google-auth-httplib2==0.1.0 cachetools==5.2.0 rsa==4.8 pyasn1==0.4.8 pyasn1-modules==0.2.8
if errorlevel 1 (
    echo WARNING: Google auth installation failed (optional)
)

echo.
echo ================================================================================
echo VERIFYING INSTALLATION
echo ================================================================================
echo.

python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

echo.
echo Checking optional packages...
python -c "import ptflops; print('ptflops: OK')" 2>nul || echo ptflops: NOT INSTALLED (optional)
python -c "import hydra; print('Hydra: OK')" 2>nul || echo Hydra: NOT INSTALLED (optional)
python -c "import tensorboard; print('TensorBoard: OK')" 2>nul || echo TensorBoard: NOT INSTALLED (optional)

echo.
echo ================================================================================
echo INSTALLATION COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo.
echo 1. Setup RNA-FM (optional but recommended):
echo    - Add RNA-FM to PYTHONPATH
echo    - Run: python verify_rna_fm.py
echo.
echo 2. Run module tests:
echo    - python models/bigru_module.py
echo    - python models/mhsa_module.py
echo.
echo 3. Start using CRISPR-UniPredict!
echo.
echo For more information, see:
echo   - QUICK_INSTALL.md
echo   - COMPLETE_SETUP.md
echo   - INSTALL_RNA_FM.md
echo.
pause
exit /b 0

:error
echo.
echo ================================================================================
echo ERROR: Installation failed!
echo ================================================================================
echo.
echo Please check the error message above and try again.
echo If the problem persists, try installing packages individually:
echo   python -m pip install package_name
echo.
pause
exit /b 1
