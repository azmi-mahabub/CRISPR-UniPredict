"""
CRISPR_HNN data preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .sequence_utils import encode_sequence, validate_sequence


def preprocess_crispr_hnn(
    input_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Preprocess CRISPR_HNN datasets.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed data
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'datasets_processed': 0,
        'total_sequences': 0,
        'valid_sequences': 0,
        'invalid_sequences': 0,
        'files': {}
    }
    
    # Process all CSV files
    for csv_file in input_path.glob('*.csv'):
        print(f"Processing {csv_file.name}...")
        
        df = pd.read_csv(csv_file)
        dataset_name = csv_file.stem
        
        # Validate sequences
        valid_mask = df['sgRNA'].apply(lambda x: validate_sequence(str(x), 'dna'))
        
        valid_df = df[valid_mask].copy()
        invalid_count = (~valid_mask).sum()
        
        # Encode sequences
        X_onehot = np.array([
            encode_sequence(str(seq)).reshape(1, 23, 4)
            for seq in valid_df['sgRNA']
        ])
        
        y = valid_df['indel'].values.astype(np.float32)
        
        # Save processed data
        output_file = output_path / f"{dataset_name}_processed.npz"
        np.savez(output_file, X=X_onehot, y=y)
        
        # Update statistics
        stats['datasets_processed'] += 1
        stats['total_sequences'] += len(df)
        stats['valid_sequences'] += len(valid_df)
        stats['invalid_sequences'] += invalid_count
        stats['files'][dataset_name] = {
            'total': len(df),
            'valid': len(valid_df),
            'invalid': invalid_count,
            'output_file': str(output_file)
        }
        
        print(f"  ✓ Processed {len(valid_df)} sequences (skipped {invalid_count})")
    
    return stats


def load_crispr_hnn_data(
    data_file: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed CRISPR_HNN data.
    
    Args:
        data_file: Path to .npz file
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    data = np.load(data_file)
    X = data['X']
    y = data['y']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def augment_sequences(sequences: np.ndarray, augmentation_factor: int = 1) -> np.ndarray:
    """
    Data augmentation for sequences (reverse complement).
    
    Args:
        sequences: Array of one-hot encoded sequences
        augmentation_factor: Number of augmentations
    
    Returns:
        Augmented sequences
    """
    augmented = [sequences]
    
    if augmentation_factor >= 1:
        # Add reverse complement
        reverse_complement = sequences[:, ::-1, ::-1]  # Reverse and flip channels
        augmented.append(reverse_complement)
    
    return np.vstack(augmented)
