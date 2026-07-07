"""
Sequence encoding and validation utilities
"""

import numpy as np
from typing import Union, Tuple


def validate_sequence(sequence: str, seq_type: str = 'dna') -> bool:
    """
    Validate DNA/RNA sequence.
    
    Args:
        sequence: DNA or RNA sequence string
        seq_type: 'dna' or 'rna'
    
    Returns:
        True if valid, False otherwise
    """
    sequence = sequence.upper()
    
    if seq_type == 'dna':
        valid_bases = set('ACGT')
    elif seq_type == 'rna':
        valid_bases = set('ACGU')
    else:
        return False
    
    return all(base in valid_bases for base in sequence)


def encode_sequence(sequence: str, encoding_type: str = 'onehot') -> np.ndarray:
    """
    Encode DNA sequence.
    
    Args:
        sequence: DNA sequence string
        encoding_type: 'onehot', 'integer', or 'embedding'
    
    Returns:
        Encoded sequence as numpy array
    """
    sequence = sequence.upper()
    
    if encoding_type == 'onehot':
        return _onehot_encode(sequence)
    elif encoding_type == 'integer':
        return _integer_encode(sequence)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def _onehot_encode(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence."""
    code_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    
    encoded = []
    for base in sequence:
        encoded.append(code_dict.get(base, [0, 0, 0, 0]))
    
    return np.array(encoded, dtype=np.float32)


def _integer_encode(sequence: str) -> np.ndarray:
    """Integer encode DNA sequence."""
    code_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    
    encoded = [code_dict.get(base, 0) for base in sequence]
    return np.array(encoded, dtype=np.int32)


def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of DNA sequence.
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(sequence.upper()))


def extract_kmers(sequence: str, k: int = 20) -> list:
    """
    Extract k-mers from sequence.
    
    Args:
        sequence: DNA sequence string
        k: k-mer size
    
    Returns:
        List of k-mers
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of sequence.
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        GC content (0-1)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0


def calculate_tm(sequence: str) -> float:
    """
    Calculate melting temperature (Tm) of sequence.
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Melting temperature in Celsius
    """
    sequence = sequence.upper()
    
    if len(sequence) < 14:
        # Simple formula for short sequences
        return 4 * (sequence.count('G') + sequence.count('C')) + \
               2 * (sequence.count('A') + sequence.count('T'))
    else:
        # GC% method for longer sequences
        gc_content = calculate_gc_content(sequence)
        return 64.9 + 41 * (gc_content - 0.5) / 0.5
