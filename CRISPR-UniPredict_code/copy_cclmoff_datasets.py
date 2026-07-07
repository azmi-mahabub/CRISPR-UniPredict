"""
Copy CCLMoff datasets to CRISPR-UniPredict project
"""

import shutil
import os
from pathlib import Path

# Source and destination directories
source_dir = Path(r"c:\Users\shahe\Desktop\both paper models\cclmoff")
dest_dir = Path(r"c:\Users\shahe\Desktop\both paper models\CRISPR-UniPredict\data\raw\cclmoff")

# Ensure destination exists
dest_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COPYING CCLMOFF DATASETS")
print("=" * 80)
print(f"\nSource: {source_dir}")
print(f"Destination: {dest_dir}\n")

# Get CSV files (main dataset)
csv_files = list(source_dir.glob("*.csv"))

if not csv_files:
    print("No CSV files found!")
else:
    print(f"Found {len(csv_files)} CSV file(s) to copy:\n")
    
    total_size = 0
    for csv_file in sorted(csv_files):
        dest_file = dest_dir / csv_file.name
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        total_size += file_size_mb
        
        print(f"Copying: {csv_file.name}")
        print(f"  Size: {file_size_gb:.2f} GB ({file_size_mb:.2f} MB)")
        
        try:
            shutil.copy2(csv_file, dest_file)
            print(f"  ✓ Copied successfully\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print("=" * 80)
    print("COPY SUMMARY")
    print("=" * 80)
    print(f"Total files copied: {len(csv_files)}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print(f"Destination: {dest_dir}")
    
    # Verify
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    copied_files = list(dest_dir.glob("*.csv"))
    print(f"\nFiles in destination directory: {len(copied_files)}\n")
    
    for f in sorted(copied_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024
        print(f"✓ {f.name}")
        print(f"  Size: {size_gb:.2f} GB ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("✓ ALL CCLMOFF DATASETS COPIED SUCCESSFULLY!")
    print("=" * 80)
