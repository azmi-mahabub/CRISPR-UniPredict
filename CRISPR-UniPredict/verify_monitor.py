"""
Verification script for monitor_training.py
Checks that all components are properly installed and working
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("✓ Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_required_packages():
    """Check required packages"""
    print("\n✓ Checking required packages...")
    
    required = {
        'psutil': 'System monitoring',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {name} ({package})")
        except ImportError:
            print(f"  ✗ {name} ({package}) - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_optional_packages():
    """Check optional packages"""
    print("\n✓ Checking optional packages...")
    
    optional = {
        'matplotlib': 'Matplotlib (for matplotlib mode)',
        'seaborn': 'Seaborn (for matplotlib mode)',
        'dash': 'Dash (for web dashboard)',
        'plotly': 'Plotly (for web dashboard)',
    }
    
    for package, name in optional.items():
        try:
            __import__(package)
            print(f"  ✓ {name} ({package})")
        except ImportError:
            print(f"  ○ {name} ({package}) - optional")

def check_files():
    """Check that all required files exist"""
    print("\n✓ Checking files...")
    
    project_root = Path(__file__).parent
    required_files = [
        'scripts/monitor_training.py',
        'scripts/test_monitor.py',
        'MONITOR_README.md',
        'MONITOR_TRAINING_GUIDE.md',
        'MONITOR_QUICK_START.txt',
        'MONITOR_INTEGRATION_GUIDE.md',
        'MONITOR_CHEATSHEET.txt',
    ]
    
    all_ok = True
    for file in required_files:
        filepath = project_root / file
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {file} ({size} bytes)")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_monitor_script():
    """Check that monitor script is valid Python"""
    print("\n✓ Checking monitor script...")
    
    monitor_file = Path(__file__).parent / 'scripts' / 'monitor_training.py'
    
    try:
        with open(monitor_file) as f:
            compile(f.read(), str(monitor_file), 'exec')
        print(f"  ✓ {monitor_file.name} - valid Python")
        return True
    except SyntaxError as e:
        print(f"  ✗ {monitor_file.name} - syntax error: {e}")
        return False

def check_test_script():
    """Check that test script is valid Python"""
    print("\n✓ Checking test script...")
    
    test_file = Path(__file__).parent / 'scripts' / 'test_monitor.py'
    
    try:
        with open(test_file) as f:
            compile(f.read(), str(test_file), 'exec')
        print(f"  ✓ {test_file.name} - valid Python")
        return True
    except SyntaxError as e:
        print(f"  ✗ {test_file.name} - syntax error: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n✓ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU available: {device_name} ({device_memory:.2f} GB)")
            return True
        else:
            print(f"  ○ GPU not available (CPU mode)")
            return False
    except Exception as e:
        print(f"  ○ Could not check GPU: {e}")
        return False

def print_summary(results):
    """Print summary of checks"""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    print("="*70)
    print(f"Result: {passed}/{total} checks passed")
    print("="*70)
    
    return passed == total

def print_next_steps():
    """Print next steps"""
    print("\n📚 NEXT STEPS:")
    print("-" * 70)
    print("\n1. Quick Start (2 minutes):")
    print("   cat MONITOR_QUICK_START.txt")
    print("\n2. Test the Monitor:")
    print("   python scripts/test_monitor.py --mode generate --epochs 20")
    print("   python scripts/monitor_training.py --log_dir logs/test_monitor --mode console")
    print("\n3. Use with Training:")
    print("   # Terminal 1:")
    print("   python scripts/train.py --config configs/model_config.yaml")
    print("   # Terminal 2:")
    print("   python scripts/monitor_training.py --log_dir logs/exp_001_* --mode console")
    print("\n4. Read Documentation:")
    print("   - MONITOR_QUICK_START.txt (quick reference)")
    print("   - MONITOR_TRAINING_GUIDE.md (complete guide)")
    print("   - MONITOR_INTEGRATION_GUIDE.md (integration patterns)")
    print("\n" + "-" * 70)

def main():
    """Main verification"""
    print("\n" + "="*70)
    print("CRISPR-UniPredict Training Monitor - Verification")
    print("="*70 + "\n")
    
    results = {}
    
    # Run checks
    results['Python Version'] = check_python_version()
    results['Required Packages'] = check_required_packages()
    check_optional_packages()  # Don't fail on optional
    results['Files Exist'] = check_files()
    results['Monitor Script'] = check_monitor_script()
    results['Test Script'] = check_test_script()
    check_gpu()  # Don't fail on GPU
    
    # Print summary
    all_ok = print_summary(results)
    
    # Print next steps
    print_next_steps()
    
    if all_ok:
        print("\n✅ All checks passed! Monitor is ready to use.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install psutil torch numpy matplotlib seaborn dash plotly")
        return 1

if __name__ == '__main__':
    sys.exit(main())
