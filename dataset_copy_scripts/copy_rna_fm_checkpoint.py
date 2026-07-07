"""
Copy RNA-FM checkpoint to CRISPR-UniPredict
"""

import shutil
from pathlib import Path

# Define paths
source = Path(r"c:\Users\shahe\Desktop\both paper models\RNA-FM_pretrained.pth")
destination = Path(r"c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict\models\pretrained\rna_fm_t12.pt")

print("=" * 80)
print("COPYING RNA-FM CHECKPOINT")
print("=" * 80)

# Check source exists
if not source.exists():
    print(f"✗ Source file not found: {source}")
    exit(1)

print(f"\nSource: {source}")
print(f"Size: {source.stat().st_size / (1024**3):.2f} GB")

# Create destination directory
destination.parent.mkdir(parents=True, exist_ok=True)

# Copy file
print(f"\nCopying to: {destination}")
print("Please wait...")

shutil.copy2(source, destination)

print(f"✓ File copied successfully!")

# Verify
if destination.exists():
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"\n✓ Destination file exists")
    print(f"  Size: {size_mb:.1f} MB")
    
    print(f"\n✓ Checkpoint file ready!")
else:
    print(f"✗ Destination file not found after copy!")
    exit(1)

print("\n" + "=" * 80)
print("✓ RNA-FM CHECKPOINT READY")
print("=" * 80)

print(f"\nYou can now use it:")
print(f"  from utils.rna_fm.fm.pretrained import rna_fm_t12")
print(f"  model, alphabet = rna_fm_t12('models/pretrained/rna_fm_t12.pt')")
