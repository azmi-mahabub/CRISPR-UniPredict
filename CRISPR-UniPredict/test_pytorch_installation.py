"""
Test PyTorch installation
"""

import torch

print("=" * 80)
print("PYTORCH INSTALLATION TEST")
print("=" * 80)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

# Test tensor creation
print("\n" + "-" * 80)
print("TENSOR CREATION TEST")
print("-" * 80)

x = torch.randn(3, 4)
print(f"✓ Created tensor: {x.shape}")
print(f"✓ Tensor device: {x.device}")

# Test on device
x_device = x.to(device)
print(f"✓ Moved to device: {x_device.device}")

# Test basic operations
print("\n" + "-" * 80)
print("BASIC OPERATIONS TEST")
print("-" * 80)

y = torch.randn(3, 4)
z = x + y
print(f"✓ Addition: {z.shape}")

z = torch.matmul(x, torch.randn(4, 5))
print(f"✓ Matrix multiplication: {z.shape}")

# Test encoding module
print("\n" + "-" * 80)
print("SEQUENCE ENCODER TEST")
print("-" * 80)

try:
    from models.encoding import SequenceEncoder
    
    encoder = SequenceEncoder(device='cpu')
    print(f"✓ SequenceEncoder imported successfully")
    
    # Test one-hot encoding
    sequence = "ACGTACGTACGTACGTACGTAC"
    one_hot = encoder.one_hot_encode(sequence)
    print(f"✓ One-hot encoding: {one_hot.shape}")
    
    # Test label encoding
    label_encoded = encoder.label_encode(sequence)
    print(f"✓ Label encoding: {label_encoded.shape}")
    
    # Test batch encoding
    sequences = ["ACGT", "TGCA", "AAAA"]
    batch_one_hot = encoder.batch_one_hot_encode(sequences)
    print(f"✓ Batch one-hot encoding: {batch_one_hot.shape}")
    
    batch_label = encoder.batch_label_encode(sequences)
    print(f"✓ Batch label encoding: {batch_label.shape}")
    
    print(f"\n✓ All encoding tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ PYTORCH INSTALLATION SUCCESSFUL")
print("=" * 80)
