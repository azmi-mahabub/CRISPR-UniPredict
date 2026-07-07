"""
Per-Source Data Normalization for CRISPR-UniPredict
Addresses Issue 4.3: On-target labels from multiple sources with incomparable scales

This module normalizes on-target labels per dataset source to account for
systematic differences in assay protocols, cell types, and measurement scales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_source_statistics(df: pd.DataFrame, target_col: str = 'on_target_score') -> dict:
    """
    Compute mean and std for each dataset source.
    
    Args:
        df: DataFrame with 'dataset_source' column and target label column
        target_col: Name of target column to normalize (default: 'on_target_score')
    
    Returns:
        Dictionary: {source_id: {'mean': float, 'std': float, 'count': int}}
    """
    stats = {}
    
    for source in df['dataset_source'].unique():
        mask = (df['dataset_source'] == source) & (df[target_col].notna())
        if mask.sum() == 0:
            logger.warning(f"No valid {target_col} samples for source '{source}'")
            continue
        
        source_data = df.loc[mask, target_col]
        stats[source] = {
            'mean': source_data.mean(),
            'std': source_data.std(),
            'count': mask.sum(),
            'min': source_data.min(),
            'max': source_data.max(),
        }
    
    return stats


def normalize_per_source(df: pd.DataFrame, 
                         target_col: str = 'on_target_score',
                         method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize target labels per dataset source.
    
    Args:
        df: DataFrame with 'dataset_source' column
        target_col: Target column to normalize
        method: 'zscore' (standard normalization) or 'minmax' (0-1 scaling)
    
    Returns:
        DataFrame with normalized target column
    """
    df = df.copy()
    statistics = compute_source_statistics(df, target_col)
    
    logger.info(f"Per-source normalization (method={method}):")
    logger.info(f"  Found {len(statistics)} sources with valid labels")
    
    for source, stats in statistics.items():
        mask = (df['dataset_source'] == source) & (df[target_col].notna())
        
        if method == 'zscore':
            # Z-score: (x - mean) / std
            if stats['std'] > 0:
                df.loc[mask, target_col] = (
                    (df.loc[mask, target_col] - stats['mean']) / stats['std']
                )
            else:
                # If std is 0, center only
                df.loc[mask, target_col] = df.loc[mask, target_col] - stats['mean']
                
        elif method == 'minmax':
            # Min-max scaling: (x - min) / (max - min) -> [0, 1]
            val_range = stats['max'] - stats['min']
            if val_range > 0:
                df.loc[mask, target_col] = (
                    (df.loc[mask, target_col] - stats['min']) / val_range
                )
            else:
                # If all values same, map to 0.5
                df.loc[mask, target_col] = 0.5
        
        logger.info(f"  {source}: {stats['count']} samples, "
                   f"original mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    return df


def normalize_splits(data_dir: str = 'data/processed/combined',
                    method: str = 'zscore',
                    output_dir: str = None) -> dict:
    """
    Normalize all train/val/test splits and save them.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        method: Normalization method ('zscore' or 'minmax')
        output_dir: Where to save normalized CSVs (default: data_dir)
    
    Returns:
        Dictionary with statistics from training set
    """
    if output_dir is None:
        output_dir = data_dir
    
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    # Compute statistics from training set only
    train_path = Path(data_dir) / 'train.csv'
    train_df = pd.read_csv(train_path)
    train_stats = compute_source_statistics(train_df)
    all_stats['train_stats'] = train_stats
    
    # Normalize each split using training statistics
    for split in splits:
        csv_path = Path(data_dir) / f'{split}.csv'
        logger.info(f"Processing {split}.csv...")
        
        df = pd.read_csv(csv_path)
        df = normalize_per_source(df, method=method)
        
        # Save normalized version
        output_path = Path(output_dir) / f'{split}_normalized.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
    
    return all_stats


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("=" * 80)
    print("Per-Source Data Normalization Script")
    print("=" * 80)
    
    data_dir = 'data/processed/combined'
    normalize_splits(data_dir, method='zscore')
    
    print("\nTo use normalized data, update your config:")
    print('  train_path: data/processed/combined/train_normalized.csv')
    print('  val_path: data/processed/combined/val_normalized.csv')
    print('  test_path: data/processed/combined/test_normalized.csv')
