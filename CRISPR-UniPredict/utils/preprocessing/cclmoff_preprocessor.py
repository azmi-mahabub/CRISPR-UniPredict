"""
CCLMoff data preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from .sequence_utils import encode_sequence, validate_sequence


def preprocess_cclmoff(
    input_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Preprocess CCLMoff datasets.
    
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
        'total_pairs': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0,
        'label_distribution': {},
        'files': {}
    }
    
    # Process all CSV files
    for csv_file in input_path.glob('*.csv'):
        print(f"Processing {csv_file.name}...")
        
        df = pd.read_csv(csv_file, nrows=10000)  # Limit for demo
        dataset_name = csv_file.stem
        
        # Validate sequences
        valid_mask = (
            df['sgRNA_seq'].apply(lambda x: validate_sequence(str(x), 'dna')) &
            df['off_seq'].apply(lambda x: validate_sequence(str(x), 'dna'))
        )
        
        valid_df = df[valid_mask].copy()
        invalid_count = (~valid_mask).sum()
        
        # Encode sequence pairs
        X = []
        for _, row in valid_df.iterrows():
            sgrna_enc = encode_sequence(str(row['sgRNA_seq']))
            off_enc = encode_sequence(str(row['off_seq']))
            
            # Pad to same length
            max_len = max(len(sgrna_enc), len(off_enc))
            sgrna_enc = np.pad(sgrna_enc, ((0, max_len - len(sgrna_enc)), (0, 0)))
            off_enc = np.pad(off_enc, ((0, max_len - len(off_enc)), (0, 0)))
            
            combined = np.concatenate([sgrna_enc, off_enc])
            X.append(combined)
        
        X = np.array(X, dtype=np.float32)
        y = valid_df['label'].values.astype(np.int32)
        
        # Save processed data
        output_file = output_path / f"{dataset_name}_processed.npz"
        np.savez(output_file, X=X, y=y)
        
        # Update statistics
        label_dist = np.bincount(y)
        stats['datasets_processed'] += 1
        stats['total_pairs'] += len(df)
        stats['valid_pairs'] += len(valid_df)
        stats['invalid_pairs'] += invalid_count
        stats['label_distribution'][dataset_name] = {
            'negative': int(label_dist[0]) if len(label_dist) > 0 else 0,
            'positive': int(label_dist[1]) if len(label_dist) > 1 else 0
        }
        stats['files'][dataset_name] = {
            'total': len(df),
            'valid': len(valid_df),
            'invalid': invalid_count,
            'output_file': str(output_file)
        }
        
        print(f"  ✓ Processed {len(valid_df)} pairs (skipped {invalid_count})")
    
    return stats


def load_cclmoff_data(
    data_file: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed CCLMoff data.
    
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
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
