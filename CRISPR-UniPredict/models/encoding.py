"""
Sequence Encoding for CRISPR-UniPredict
Implements one-hot and label encoding for DNA/RNA sequences
Based on CRISPR_HNN paper encoding schemes
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Optional
from pathlib import Path


class SequenceEncoder:
    """Encode DNA/RNA sequences for model training"""
    
    # Nucleotide to integer mapping
    NUCLEOTIDE_TO_INT = {
        'A': 2,
        'C': 3,
        'G': 4,
        'T': 5,
        'U': 5,  # Treat U as T (RNA to DNA)
    }
    
    # Integer to nucleotide mapping
    INT_TO_NUCLEOTIDE = {
        2: 'A',
        3: 'C',
        4: 'G',
        5: 'T',
    }
    
    # One-hot encoding vectors
    ONE_HOT_VECTORS = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'U': torch.tensor([0, 0, 0, 1], dtype=torch.float32),  # Treat U as T
    }
    
    # Start token for label encoding
    START_TOKEN = 1
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize encoder
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        
        # Move one-hot vectors to device (safely handle CPU)
        try:
            self.ONE_HOT_VECTORS = {
                k: v.to(device) for k, v in self.ONE_HOT_VECTORS.items()
            }
        except (RuntimeError, AssertionError):
            # If CUDA not available, use CPU
            self.device = 'cpu'
            self.ONE_HOT_VECTORS = {
                k: v.to('cpu') for k, v in self.ONE_HOT_VECTORS.items()
            }
    
    @staticmethod
    def validate_sequence(sequence: str) -> Tuple[bool, Optional[str]]:
        """
        Validate DNA/RNA sequence
        
        Args:
            sequence: Sequence string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(sequence, str):
            return False, "Sequence must be a string"
        
        if len(sequence) == 0:
            return False, "Sequence cannot be empty"
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Check for valid nucleotides
        valid_nucleotides = set('ACGTU')
        invalid_chars = set(sequence) - valid_nucleotides
        
        if invalid_chars:
            return False, f"Invalid nucleotides: {invalid_chars}"
        
        return True, None
    
    @staticmethod
    def normalize_sequence(sequence: str) -> str:
        """
        Normalize sequence (uppercase, U->T)
        
        Args:
            sequence: Sequence string
            
        Returns:
            Normalized sequence
        """
        sequence = sequence.upper()
        sequence = sequence.replace('U', 'T')
        return sequence
    
    def one_hot_encode(self, sequence: str) -> torch.Tensor:
        """
        One-hot encode a sequence
        
        Encoding: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
        
        Args:
            sequence: DNA/RNA sequence string
            
        Returns:
            Tensor of shape (seq_len, 4)
        """
        # Validate
        is_valid, error = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {error}")
        
        # Normalize
        sequence = self.normalize_sequence(sequence)
        
        # Encode
        encoded = []
        for nucleotide in sequence:
            encoded.append(self.ONE_HOT_VECTORS[nucleotide])
        
        # Stack into tensor
        result = torch.stack(encoded, dim=0)
        
        return result.to(self.device)
    
    def label_encode(self, sequence: str, add_start_token: bool = True) -> torch.Tensor:
        """
        Label encode a sequence
        
        Encoding: Start=1, A=2, C=3, G=4, T=5
        
        Args:
            sequence: DNA/RNA sequence string
            add_start_token: Whether to add start token at beginning
            
        Returns:
            Tensor of shape (seq_len,) or (seq_len+1,) if add_start_token=True
        """
        # Validate
        is_valid, error = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {error}")
        
        # Normalize
        sequence = self.normalize_sequence(sequence)
        
        # Encode
        encoded = []
        
        if add_start_token:
            encoded.append(self.START_TOKEN)
        
        for nucleotide in sequence:
            encoded.append(self.NUCLEOTIDE_TO_INT[nucleotide])
        
        # Convert to tensor
        result = torch.tensor(encoded, dtype=torch.long, device=self.device)
        
        return result
    
    def batch_one_hot_encode(self, sequences: List[str], 
                            max_length: Optional[int] = None,
                            pad_value: float = 0.0) -> torch.Tensor:
        """
        One-hot encode a batch of sequences with padding
        
        Args:
            sequences: List of sequence strings
            max_length: Maximum sequence length (auto-detect if None)
            pad_value: Value to use for padding
            
        Returns:
            Tensor of shape (batch_size, max_len, 4)
        """
        if not sequences:
            raise ValueError("Sequences list is empty")
        
        # Validate all sequences
        for seq in sequences:
            is_valid, error = self.validate_sequence(seq)
            if not is_valid:
                raise ValueError(f"Invalid sequence: {error}")
        
        # Normalize sequences
        sequences = [self.normalize_sequence(seq) for seq in sequences]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Encode all sequences
        batch_encoded = []
        
        for sequence in sequences:
            # One-hot encode
            encoded = []
            for nucleotide in sequence:
                encoded.append(self.ONE_HOT_VECTORS[nucleotide])
            
            # Stack
            encoded_tensor = torch.stack(encoded, dim=0)
            
            # Pad to max_length
            if len(sequence) < max_length:
                padding = torch.full(
                    (max_length - len(sequence), 4),
                    pad_value,
                    dtype=torch.float32,
                    device=self.device
                )
                encoded_tensor = torch.cat([encoded_tensor, padding], dim=0)
            
            batch_encoded.append(encoded_tensor)
        
        # Stack batch
        result = torch.stack(batch_encoded, dim=0)
        
        return result.to(self.device)
    
    def batch_label_encode(self, sequences: List[str],
                          max_length: Optional[int] = None,
                          add_start_token: bool = True,
                          pad_value: int = 0) -> torch.Tensor:
        """
        Label encode a batch of sequences with padding
        
        Args:
            sequences: List of sequence strings
            max_length: Maximum sequence length (auto-detect if None)
            add_start_token: Whether to add start token
            pad_value: Value to use for padding
            
        Returns:
            Tensor of shape (batch_size, max_len+1) if add_start_token=True
                         or (batch_size, max_len) if add_start_token=False
        """
        if not sequences:
            raise ValueError("Sequences list is empty")
        
        # Validate all sequences
        for seq in sequences:
            is_valid, error = self.validate_sequence(seq)
            if not is_valid:
                raise ValueError(f"Invalid sequence: {error}")
        
        # Normalize sequences
        sequences = [self.normalize_sequence(seq) for seq in sequences]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Account for start token
        total_length = max_length + (1 if add_start_token else 0)
        
        # Encode all sequences
        batch_encoded = []
        
        for sequence in sequences:
            encoded = []
            
            # Add start token
            if add_start_token:
                encoded.append(self.START_TOKEN)
            
            # Encode nucleotides
            for nucleotide in sequence:
                encoded.append(self.NUCLEOTIDE_TO_INT[nucleotide])
            
            # Pad to total_length
            if len(encoded) < total_length:
                encoded.extend([pad_value] * (total_length - len(encoded)))
            
            # Convert to tensor
            encoded_tensor = torch.tensor(
                encoded,
                dtype=torch.long,
                device=self.device
            )
            
            batch_encoded.append(encoded_tensor)
        
        # Stack batch
        result = torch.stack(batch_encoded, dim=0)
        
        return result.to(self.device)
    
    def decode_label_encoded(self, encoded: torch.Tensor, 
                            skip_start_token: bool = True) -> str:
        """
        Decode label-encoded sequence back to string
        
        Args:
            encoded: Label-encoded tensor
            skip_start_token: Whether to skip first token (start token)
            
        Returns:
            Decoded sequence string
        """
        # Convert to numpy if needed
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        
        # Convert to list if needed
        if isinstance(encoded, np.ndarray):
            encoded = encoded.tolist()
        
        # Skip start token if needed
        if skip_start_token and len(encoded) > 0 and encoded[0] == self.START_TOKEN:
            encoded = encoded[1:]
        
        # Decode
        decoded = []
        for token in encoded:
            if token in self.INT_TO_NUCLEOTIDE:
                decoded.append(self.INT_TO_NUCLEOTIDE[token])
            elif token == 0:  # Padding
                break
        
        return ''.join(decoded)
    
    def decode_one_hot(self, encoded: torch.Tensor) -> str:
        """
        Decode one-hot encoded sequence back to string
        
        Args:
            encoded: One-hot encoded tensor of shape (seq_len, 4)
            
        Returns:
            Decoded sequence string
        """
        # Convert to numpy if needed
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()
        
        # Decode
        decoded = []
        nucleotides = ['A', 'C', 'G', 'T']
        
        for one_hot in encoded:
            # Check if all zeros (padding)
            if np.all(one_hot == 0):
                break
            
            # Get argmax
            idx = np.argmax(one_hot)
            decoded.append(nucleotides[idx])
        
        return ''.join(decoded)
    
    def get_sequence_length(self, sequence: str) -> int:
        """Get length of sequence"""
        is_valid, _ = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError("Invalid sequence")
        return len(sequence)
    
    def get_gc_content(self, sequence: str) -> float:
        """
        Calculate GC content of sequence
        
        Args:
            sequence: DNA/RNA sequence
            
        Returns:
            GC content as fraction (0-1)
        """
        is_valid, _ = self.validate_sequence(sequence)
        if not is_valid:
            raise ValueError("Invalid sequence")
        
        sequence = self.normalize_sequence(sequence)
        gc_count = sequence.count('G') + sequence.count('C')
        
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0


def create_encoder(device: str = 'cpu') -> SequenceEncoder:
    """
    Create a sequence encoder
    
    Args:
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        SequenceEncoder instance
    """
    return SequenceEncoder(device=device)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("SEQUENCE ENCODER EXAMPLES")
    print("=" * 80)
    
    # Create encoder
    encoder = SequenceEncoder(device='cpu')
    
    # Example sequences
    sequences = [
        "ACGTACGTACGTACGTACGTAC",
        "TGCATGCATGCATGCATGCATG",
        "AAAAAAAAAAAAAAAAAAAAAA"
    ]
    
    # One-hot encoding
    print("\n1. ONE-HOT ENCODING")
    print("-" * 80)
    
    for seq in sequences[:1]:
        encoded = encoder.one_hot_encode(seq)
        print(f"Sequence: {seq}")
        print(f"Shape: {encoded.shape}")
        print(f"First 3 nucleotides:")
        print(encoded[:3])
    
    # Label encoding
    print("\n2. LABEL ENCODING")
    print("-" * 80)
    
    for seq in sequences[:1]:
        encoded = encoder.label_encode(seq, add_start_token=True)
        print(f"Sequence: {seq}")
        print(f"Shape: {encoded.shape}")
        print(f"Encoded: {encoded}")
        decoded = encoder.decode_label_encoded(encoded)
        print(f"Decoded: {decoded}")
    
    # Batch one-hot encoding
    print("\n3. BATCH ONE-HOT ENCODING")
    print("-" * 80)
    
    batch_encoded = encoder.batch_one_hot_encode(sequences)
    print(f"Sequences: {len(sequences)}")
    print(f"Batch shape: {batch_encoded.shape}")
    print(f"Expected: ({len(sequences)}, {max(len(s) for s in sequences)}, 4)")
    
    # Batch label encoding
    print("\n4. BATCH LABEL ENCODING")
    print("-" * 80)
    
    batch_encoded = encoder.batch_label_encode(sequences, add_start_token=True)
    print(f"Sequences: {len(sequences)}")
    print(f"Batch shape: {batch_encoded.shape}")
    print(f"Expected: ({len(sequences)}, {max(len(s) for s in sequences) + 1})")
    
    # Decode first sequence
    decoded = encoder.decode_label_encoded(batch_encoded[0])
    print(f"First sequence decoded: {decoded}")
    
    # GC content
    print("\n5. GC CONTENT")
    print("-" * 80)
    
    for seq in sequences:
        gc = encoder.get_gc_content(seq)
        print(f"Sequence: {seq}")
        print(f"GC content: {gc:.2%}")
    
    # Validate sequences
    print("\n6. SEQUENCE VALIDATION")
    print("-" * 80)
    
    test_sequences = [
        "ACGT",
        "INVALID",
        "ACGTU",
        ""
    ]
    
    for seq in test_sequences:
        is_valid, error = encoder.validate_sequence(seq)
        status = "✓" if is_valid else "✗"
        print(f"{status} '{seq}': {error if error else 'Valid'}")
    
    print("\n" + "=" * 80)
    print("✓ ENCODER EXAMPLES COMPLETE")
    print("=" * 80)
