"""
Copy RNA-FM files from RNA-FM-main to CRISPR-UniPredict
"""

import shutil
from pathlib import Path

# Define paths
rna_fm_source = Path(r"c:\Users\shahe\Desktop\both paper models\RNA-FM-main")
crispr_target = Path(r"c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict")

# Create target directories
models_dir = crispr_target / "models" / "pretrained"
utils_dir = crispr_target / "utils" / "rna_fm"

models_dir.mkdir(parents=True, exist_ok=True)
utils_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COPYING RNA-FM FILES TO CRISPR-UniPredict")
print("=" * 80)

# Copy fm module (core model code)
print("\n1. Copying RNA-FM module (fm/)...")
fm_source = rna_fm_source / "fm"
fm_target = utils_dir / "fm"

if fm_source.exists():
    if fm_target.exists():
        shutil.rmtree(fm_target)
    shutil.copytree(fm_source, fm_target)
    print(f"   ✓ Copied: {fm_source.name} -> {fm_target}")
else:
    print(f"   ✗ Source not found: {fm_source}")

# Copy setup.py
print("\n2. Copying setup.py...")
setup_source = rna_fm_source / "setup.py"
setup_target = utils_dir / "setup.py"

if setup_source.exists():
    shutil.copy2(setup_source, setup_target)
    print(f"   ✓ Copied: {setup_source.name}")
else:
    print(f"   ✗ Source not found: {setup_source}")

# Copy environment.yml
print("\n3. Copying environment.yml...")
env_source = rna_fm_source / "environment.yml"
env_target = utils_dir / "environment.yml"

if env_source.exists():
    shutil.copy2(env_source, env_target)
    print(f"   ✓ Copied: {env_source.name}")
else:
    print(f"   ✗ Source not found: {env_source}")

# Copy README
print("\n4. Copying README.md...")
readme_source = rna_fm_source / "README.md"
readme_target = utils_dir / "README.md"

if readme_source.exists():
    shutil.copy2(readme_source, readme_target)
    print(f"   ✓ Copied: {readme_source.name}")
else:
    print(f"   ✗ Source not found: {readme_source}")

# Copy LICENSE
print("\n5. Copying LICENSE...")
license_source = rna_fm_source / "LICENSE"
license_target = utils_dir / "LICENSE"

if license_source.exists():
    shutil.copy2(license_source, license_target)
    print(f"   ✓ Copied: {license_source.name}")
else:
    print(f"   ✗ Source not found: {license_source}")

# Copy tutorials (optional)
print("\n6. Copying tutorials/...")
tutorials_source = rna_fm_source / "tutorials"
tutorials_target = utils_dir / "tutorials"

if tutorials_source.exists():
    if tutorials_target.exists():
        shutil.rmtree(tutorials_target)
    shutil.copytree(tutorials_source, tutorials_target)
    print(f"   ✓ Copied: {tutorials_source.name} -> {tutorials_target}")
else:
    print(f"   ✗ Source not found: {tutorials_source}")

# Copy docs (optional)
print("\n7. Copying docs/...")
docs_source = rna_fm_source / "docs"
docs_target = utils_dir / "docs"

if docs_source.exists():
    if docs_target.exists():
        shutil.rmtree(docs_target)
    shutil.copytree(docs_source, docs_target)
    print(f"   ✓ Copied: {docs_source.name} -> {docs_target}")
else:
    print(f"   ✗ Source not found: {docs_source}")

# Print summary
print("\n" + "=" * 80)
print("COPY SUMMARY")
print("=" * 80)

print(f"\nRNA-FM files copied to:")
print(f"  {utils_dir}")

print(f"\nDirectory structure:")
for item in sorted(utils_dir.iterdir()):
    if item.is_dir():
        count = len(list(item.rglob("*")))
        print(f"  📁 {item.name}/ ({count} items)")
    else:
        size_kb = item.stat().st_size / 1024
        print(f"  📄 {item.name} ({size_kb:.1f} KB)")

print(f"\n✓ RNA-FM module successfully copied!")
print(f"\nYou can now use RNA-FM in CRISPR-UniPredict:")
print(f"  from utils.rna_fm.fm.pretrained import rna_fm_t12")
print(f"  model, alphabet = rna_fm_t12()")

print("\n" + "=" * 80)
