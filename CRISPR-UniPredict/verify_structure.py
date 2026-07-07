"""
Verify CRISPR-UniPredict project structure
"""

import os
from pathlib import Path


def verify_project_structure():
    """Verify all required directories and files exist."""
    
    project_root = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        "data/raw/crispr_hnn",
        "data/raw/cclmoff",
        "data/processed/on_target",
        "data/processed/off_target",
        "data/processed/combined",
        "models/checkpoints",
        "models/pretrained",
        "results/plots",
        "results/metrics",
        "results/predictions",
        "utils/preprocessing",
        "utils/evaluation",
        "utils/visualization",
        "scripts",
        "configs",
        "logs",
        "notebooks",
    ]
    
    # Required files
    required_files = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "__init__.py",
        "utils/__init__.py",
        "utils/preprocessing/__init__.py",
        "utils/preprocessing/sequence_utils.py",
        "utils/preprocessing/crispr_hnn_preprocessor.py",
        "utils/preprocessing/cclmoff_preprocessor.py",
        "utils/evaluation/__init__.py",
        "utils/evaluation/metrics.py",
        "utils/visualization/__init__.py",
        "utils/visualization/plots.py",
        "configs/config_template.yaml",
    ]
    
    print("=" * 80)
    print("CRISPR-UniPredict Project Structure Verification")
    print("=" * 80)
    
    # Check directories
    print("\n📁 Checking directories...")
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (MISSING)")
            missing_dirs.append(dir_path)
    
    # Check files
    print("\n📄 Checking files...")
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_dirs = len(required_dirs)
    total_files = len(required_files)
    
    print(f"\nDirectories: {total_dirs - len(missing_dirs)}/{total_dirs} ✓")
    print(f"Files: {total_files - len(missing_files)}/{total_files} ✓")
    
    if not missing_dirs and not missing_files:
        print("\n✓ Project structure is complete!")
        return True
    else:
        if missing_dirs:
            print(f"\nMissing directories ({len(missing_dirs)}):")
            for d in missing_dirs:
                print(f"  - {d}")
        if missing_files:
            print(f"\nMissing files ({len(missing_files)}):")
            for f in missing_files:
                print(f"  - {f}")
        return False


def print_project_tree():
    """Print project directory tree."""
    
    project_root = Path(__file__).parent
    
    print("\n" + "=" * 80)
    print("PROJECT TREE")
    print("=" * 80 + "\n")
    
    def print_tree(path, prefix="", is_last=True):
        """Recursively print directory tree."""
        if path.name.startswith('.'):
            return
        
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{path.name}/")
        
        if path.is_dir() and path.name not in ['__pycache__', '.git']:
            contents = sorted(path.iterdir())
            dirs = [p for p in contents if p.is_dir()]
            files = [p for p in contents if p.is_file()]
            
            all_items = dirs + files
            
            for i, item in enumerate(all_items):
                is_last_item = (i == len(all_items) - 1)
                extension = "    " if is_last else "│   "
                
                if item.is_file():
                    connector = "└── " if is_last_item else "├── "
                    print(f"{prefix}{extension}{connector}{item.name}")
                else:
                    print_tree(item, prefix + extension, is_last_item)
    
    print_tree(project_root)


if __name__ == "__main__":
    # Verify structure
    success = verify_project_structure()
    
    # Print tree
    print_project_tree()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ Project is ready for development!")
    else:
        print("⚠ Please create missing directories and files.")
    print("=" * 80)
