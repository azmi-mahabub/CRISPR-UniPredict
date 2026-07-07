#!/usr/bin/env python
"""
Comprehensive dependency installer for CRISPR-UniPredict
Downloads and installs all required and optional dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

print("=" * 80)
print("CRISPR-UniPredict DEPENDENCY INSTALLER")
print("=" * 80)

# Core dependencies (required)
CORE_DEPENDENCIES = [
    "numpy==1.23.5",
    "pandas==1.5.2",
    "scipy==1.9.3",
    "scikit-learn==1.1.3",
    "torch==2.0.0",
    "torchvision==0.15.2",
    "torchaudio==2.0.1",
    "transformers==4.30.0",
    "biopython==1.81",
    "matplotlib==3.6.2",
    "seaborn==0.12.1",
    "plotly==5.14.0",
    "tqdm==4.64.1",
    "pyyaml==6.0",
    "python-dotenv==1.0.0",
    "jupyter==1.0.0",
    "jupyterlab==3.6.3",
    "ipython==8.12.0",
    "pytest==7.3.1",
    "black==23.3.0",
    "flake8==6.0.0",
    "isort==5.12.0",
    "sphinx==6.3.0",
    "sphinx-rtd-theme==1.2.0",
]

# Optional dependencies (for RNA-FM and advanced features)
OPTIONAL_DEPENDENCIES = [
    "ptflops==0.6.6",
    "pytorch-ignite==0.4.10",
    "hydra-core==1.1.1",
    "omegaconf==2.1.1",
    "regex==2021.11.10",
    "tabulate==0.8.9",
    "tensorboard==2.10.0",
    "tensorboard-data-server==0.6.1",
    "tensorboard-plugin-wit==1.8.1",
    "bitarray==2.5.1",
    "google-auth==2.10.0",
    "google-auth-oauthlib==0.4.6",
    "google-auth-httplib2==0.1.0",
    "cachetools==5.2.0",
    "rsa==4.8",
    "pyasn1==0.4.8",
    "pyasn1-modules==0.2.8",
]

def install_package(package):
    """Install a single package using pip"""
    try:
        print(f"  Installing {package}...", end=" ", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓")
        return True
    except subprocess.CalledProcessError:
        print("✗")
        return False

def install_packages(packages, category):
    """Install a list of packages"""
    print(f"\n{category}")
    print("-" * 80)
    
    failed = []
    for package in packages:
        if not install_package(package):
            failed.append(package)
    
    if failed:
        print(f"\n⚠ {len(failed)} package(s) failed to install:")
        for pkg in failed:
            print(f"  - {pkg}")
        return False
    else:
        print(f"\n✓ All {len(packages)} packages installed successfully")
        return True

def verify_installation():
    """Verify that key packages are installed"""
    print("\nVERIFYING INSTALLATION")
    print("-" * 80)
    
    packages_to_check = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "scikit-learn",
        "Bio": "BioPython",
        "transformers": "Transformers",
        "ptflops": "ptflops (optional)",
        "hydra": "Hydra (optional)",
        "tensorboard": "TensorBoard (optional)",
    }
    
    all_ok = True
    for module, name in packages_to_check.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            is_optional = "(optional)" in name
            status = "⚠" if is_optional else "✗"
            print(f"  {status} {name}")
            if not is_optional:
                all_ok = False
    
    return all_ok

def main():
    """Main installation routine"""
    
    print("\nThis script will install all dependencies for CRISPR-UniPredict")
    print("This includes:")
    print("  - Core dependencies (required)")
    print("  - Optional dependencies (ptflops, hydra, tensorboard, etc.)")
    print("\nEstimated time: 10-30 minutes (depends on internet speed)")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Installation cancelled.")
        sys.exit(0)
    
    # Install core dependencies
    print("\n" + "=" * 80)
    print("INSTALLING CORE DEPENDENCIES")
    print("=" * 80)
    
    core_ok = install_packages(CORE_DEPENDENCIES, "CORE DEPENDENCIES (Required)")
    
    # Install optional dependencies
    print("\n" + "=" * 80)
    print("INSTALLING OPTIONAL DEPENDENCIES")
    print("=" * 80)
    
    optional_ok = install_packages(OPTIONAL_DEPENDENCIES, "OPTIONAL DEPENDENCIES (RNA-FM, Advanced Features)")
    
    # Verify installation
    print("\n" + "=" * 80)
    verify_ok = verify_installation()
    
    # Summary
    print("\n" + "=" * 80)
    print("INSTALLATION SUMMARY")
    print("=" * 80)
    
    if core_ok and optional_ok and verify_ok:
        print("\n✓ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Add RNA-FM to PYTHONPATH:")
        print("     export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main")
        print("  2. Verify RNA-FM encoder:")
        print("     python verify_rna_fm.py")
        print("  3. Run module tests:")
        print("     python models/bigru_module.py")
        print("     python models/mhsa_module.py")
        return 0
    elif core_ok:
        print("\n⚠ CORE DEPENDENCIES INSTALLED")
        print("⚠ Some optional dependencies failed to install")
        print("\nYou can still use CRISPR-UniPredict, but some features may not work.")
        print("Try installing optional dependencies manually:")
        print("  pip install ptflops==0.6.6")
        print("  pip install hydra-core==1.1.1")
        print("  pip install tensorboard==2.10.0")
        return 1
    else:
        print("\n✗ INSTALLATION FAILED")
        print("Some core dependencies failed to install.")
        print("Please check your internet connection and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
