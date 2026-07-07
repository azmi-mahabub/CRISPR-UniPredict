"""
Data Splitter for CRISPR-UniPredict
Stratified train/validation/test splitting with no data leakage
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DataSplitter:
    """Stratified data splitting with no data leakage"""
    
    def __init__(self, base_dir=None, random_seed=42):
        """
        Initialize data splitter
        
        Args:
            base_dir: Base directory for data
            random_seed: Random seed for reproducibility
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "data"
        
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed" / "combined"
        self.random_seed = random_seed
        
        # Create output directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistics = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'train_on_target': 0,
            'train_off_target': 0,
            'val_on_target': 0,
            'val_off_target': 0,
            'test_on_target': 0,
            'test_off_target': 0,
            'train_pct': 0.0,
            'val_pct': 0.0,
            'test_pct': 0.0,
            'random_seed': random_seed,
            'split_date': datetime.now().isoformat(),
            'errors': []
        }
    
    def split_dataset(self, df: pd.DataFrame, 
                     test_size: float = 0.1, 
                     val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits
        
        Splits data into train (80%), validation (10%), and test (10%)
        with stratification by task_type to maintain on_target/off_target ratio.
        
        Args:
            df: Input DataFrame with 'task_type' column
            test_size: Fraction for test set (default 0.1 = 10%)
            val_size: Fraction for validation set (default 0.1 = 10%)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n{'─' * 80}")
        print(f"SPLITTING DATASET")
        print(f"{'─' * 80}")
        
        try:
            # Validate input
            if 'task_type' not in df.columns:
                raise ValueError("DataFrame must contain 'task_type' column")
            
            if len(df) == 0:
                raise ValueError("DataFrame is empty")
            
            print(f"\nInput shape: {df.shape}")
            print(f"Task type distribution:")
            print(df['task_type'].value_counts())
            
            # Step 1: Split into test (10%) and temp (90%)
            # Stratify by task_type
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=df['task_type']
            )
            
            print(f"\n✓ Step 1: Separated test set ({test_size*100:.1f}%)")
            print(f"  Train+Val: {len(train_val_df):,} samples")
            print(f"  Test: {len(test_df):,} samples")
            
            # Step 2: Split train_val into train (80% of original) and val (10% of original)
            # Adjusted test_size for second split: val_size / (1 - test_size)
            val_size_adjusted = val_size / (1 - test_size)
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=self.random_seed,
                stratify=train_val_df['task_type']
            )
            
            print(f"\n✓ Step 2: Separated validation set ({val_size*100:.1f}%)")
            print(f"  Train: {len(train_df):,} samples")
            print(f"  Val: {len(val_df):,} samples")
            
            # Verify no data leakage
            train_indices = set(train_df.index)
            val_indices = set(val_df.index)
            test_indices = set(test_df.index)
            
            overlap_train_val = train_indices & val_indices
            overlap_train_test = train_indices & test_indices
            overlap_val_test = val_indices & test_indices
            
            if overlap_train_val or overlap_train_test or overlap_val_test:
                raise ValueError("Data leakage detected between splits!")
            
            print(f"\n✓ No data leakage detected")
            
            # Print split statistics
            print(f"\n{'─' * 80}")
            print(f"SPLIT STATISTICS")
            print(f"{'─' * 80}")
            
            total = len(df)
            print(f"\nTotal samples: {total:,}")
            print(f"  Train: {len(train_df):,} ({len(train_df)/total*100:.2f}%)")
            print(f"  Val:   {len(val_df):,} ({len(val_df)/total*100:.2f}%)")
            print(f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.2f}%)")
            
            # Task type distribution
            print(f"\nOn-target samples:")
            train_on = len(train_df[train_df['task_type'] == 'on_target'])
            val_on = len(val_df[val_df['task_type'] == 'on_target'])
            test_on = len(test_df[test_df['task_type'] == 'on_target'])
            total_on = train_on + val_on + test_on
            
            print(f"  Train: {train_on:,} ({train_on/total_on*100:.2f}%)")
            print(f"  Val:   {val_on:,} ({val_on/total_on*100:.2f}%)")
            print(f"  Test:  {test_on:,} ({test_on/total_on*100:.2f}%)")
            
            print(f"\nOff-target samples:")
            train_off = len(train_df[train_df['task_type'] == 'off_target'])
            val_off = len(val_df[val_df['task_type'] == 'off_target'])
            test_off = len(test_df[test_df['task_type'] == 'off_target'])
            total_off = train_off + val_off + test_off
            
            print(f"  Train: {train_off:,} ({train_off/total_off*100:.2f}%)")
            print(f"  Val:   {val_off:,} ({val_off/total_off*100:.2f}%)")
            print(f"  Test:  {test_off:,} ({test_off/total_off*100:.2f}%)")
            
            # Update statistics
            self.statistics['total_samples'] = total
            self.statistics['train_samples'] = len(train_df)
            self.statistics['val_samples'] = len(val_df)
            self.statistics['test_samples'] = len(test_df)
            self.statistics['train_on_target'] = train_on
            self.statistics['train_off_target'] = train_off
            self.statistics['val_on_target'] = val_on
            self.statistics['val_off_target'] = val_off
            self.statistics['test_on_target'] = test_on
            self.statistics['test_off_target'] = test_off
            self.statistics['train_pct'] = len(train_df) / total * 100
            self.statistics['val_pct'] = len(val_df) / total * 100
            self.statistics['test_pct'] = len(test_df) / total * 100
            
            return train_df, val_df, test_df
            
        except Exception as e:
            error_msg = f"Error splitting dataset: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.statistics['errors'].append(error_msg)
            raise
    
    def create_splits(self, input_file: Optional[Path] = None,
                     test_size: float = 0.1,
                     val_size: float = 0.1) -> Dict:
        """
        Main function to create and save train/val/test splits
        
        Args:
            input_file: Path to unified dataset (auto-detect if None)
            test_size: Test set fraction (default 0.1 = 10%)
            val_size: Validation set fraction (default 0.1 = 10%)
            
        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("CRISPR-UniPredict DATA SPLITTING")
        print("=" * 80)
        
        try:
            # Auto-detect input file if not provided
            if input_file is None:
                input_file = self.processed_dir / "unified_dataset.csv"
            
            input_file = Path(input_file)
            
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            print(f"\nLoading unified dataset...")
            print(f"File: {input_file.name}")
            
            # Load unified dataset
            df = pd.read_csv(input_file)
            print(f"✓ Loaded: {df.shape[0]:,} samples × {df.shape[1]} columns")
            
            # Create splits
            train_df, val_df, test_df = self.split_dataset(df, test_size, val_size)
            
            # Save splits
            print(f"\n{'─' * 80}")
            print(f"SAVING SPLITS")
            print(f"{'─' * 80}\n")
            
            train_path = self.processed_dir / "train.csv"
            val_path = self.processed_dir / "val.csv"
            test_path = self.processed_dir / "test.csv"
            
            train_df.to_csv(train_path, index=False)
            print(f"✓ Saved train split: {train_path.name}")
            print(f"  Samples: {len(train_df):,}")
            print(f"  Size: {train_path.stat().st_size / (1024*1024):.2f} MB")
            
            val_df.to_csv(val_path, index=False)
            print(f"\n✓ Saved validation split: {val_path.name}")
            print(f"  Samples: {len(val_df):,}")
            print(f"  Size: {val_path.stat().st_size / (1024*1024):.2f} MB")
            
            test_df.to_csv(test_path, index=False)
            print(f"\n✓ Saved test split: {test_path.name}")
            print(f"  Samples: {len(test_df):,}")
            print(f"  Size: {test_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save statistics
            stats_path = self.processed_dir / "split_statistics.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(self.statistics, f)
            
            print(f"\n✓ Saved statistics: {stats_path.name}")
            
            # Print summary
            self._print_summary()
            
            return self.statistics
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            self.statistics['errors'].append(str(e))
            raise
    
    def _print_summary(self):
        """Print splitting summary"""
        print("\n" + "=" * 80)
        print("SPLITTING SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal samples: {self.statistics['total_samples']:,}")
        print(f"Random seed: {self.statistics['random_seed']}")
        print(f"Split date: {self.statistics['split_date']}")
        
        print(f"\n{'Split':<15} {'Samples':>12} {'Percentage':>12} {'On-Target':>12} {'Off-Target':>12}")
        print(f"{'-'*63}")
        
        print(f"{'Train':<15} {self.statistics['train_samples']:>12,} {self.statistics['train_pct']:>11.2f}% "
              f"{self.statistics['train_on_target']:>12,} {self.statistics['train_off_target']:>12,}")
        
        print(f"{'Validation':<15} {self.statistics['val_samples']:>12,} {self.statistics['val_pct']:>11.2f}% "
              f"{self.statistics['val_on_target']:>12,} {self.statistics['val_off_target']:>12,}")
        
        print(f"{'Test':<15} {self.statistics['test_samples']:>12,} {self.statistics['test_pct']:>11.2f}% "
              f"{self.statistics['test_on_target']:>12,} {self.statistics['test_off_target']:>12,}")
        
        print(f"{'-'*63}")
        print(f"{'TOTAL':<15} {self.statistics['total_samples']:>12,} {'100.00':>11}% "
              f"{self.statistics['train_on_target'] + self.statistics['val_on_target'] + self.statistics['test_on_target']:>12,} "
              f"{self.statistics['train_off_target'] + self.statistics['val_off_target'] + self.statistics['test_off_target']:>12,}")
        
        # Verify percentages
        print(f"\nOn-target distribution:")
        total_on = (self.statistics['train_on_target'] + 
                   self.statistics['val_on_target'] + 
                   self.statistics['test_on_target'])
        
        if total_on > 0:
            train_on_pct = self.statistics['train_on_target'] / total_on * 100
            val_on_pct = self.statistics['val_on_target'] / total_on * 100
            test_on_pct = self.statistics['test_on_target'] / total_on * 100
            
            print(f"  Train: {train_on_pct:.2f}%")
            print(f"  Val:   {val_on_pct:.2f}%")
            print(f"  Test:  {test_on_pct:.2f}%")
        
        print(f"\nOff-target distribution:")
        total_off = (self.statistics['train_off_target'] + 
                    self.statistics['val_off_target'] + 
                    self.statistics['test_off_target'])
        
        if total_off > 0:
            train_off_pct = self.statistics['train_off_target'] / total_off * 100
            val_off_pct = self.statistics['val_off_target'] / total_off * 100
            test_off_pct = self.statistics['test_off_target'] / total_off * 100
            
            print(f"  Train: {train_off_pct:.2f}%")
            print(f"  Val:   {val_off_pct:.2f}%")
            print(f"  Test:  {test_off_pct:.2f}%")
        
        if self.statistics['errors']:
            print(f"\nErrors ({len(self.statistics['errors'])}):")
            for error in self.statistics['errors']:
                print(f"  ⚠ {error}")
        
        print("\n" + "=" * 80)
        print("✓ SPLITTING COMPLETE")
        print("=" * 80)
    
    def load_statistics(self, stats_file: Optional[Path] = None) -> Dict:
        """
        Load splitting statistics from pickle file
        
        Args:
            stats_file: Path to statistics file (auto-detect if None)
            
        Returns:
            Statistics dictionary
        """
        if stats_file is None:
            stats_file = self.processed_dir / "split_statistics.pkl"
        
        stats_file = Path(stats_file)
        
        if not stats_file.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_file}")
        
        with open(stats_file, 'rb') as f:
            return pickle.load(f)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("CRISPR-UniPredict Data Splitter")
    print("=" * 80)
    
    # Create splitter
    splitter = DataSplitter(random_seed=42)
    
    # Create splits
    statistics = splitter.create_splits(
        test_size=0.1,
        val_size=0.1
    )
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    
    print(f"\nSplit files in {splitter.processed_dir.name}/:")
    for f in sorted(splitter.processed_dir.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        rows = len(pd.read_csv(f, nrows=1))
        print(f"  • {f.name:<30} ({size_mb:>8.2f} MB)")
    
    print(f"\nStatistics file:")
    stats_file = splitter.processed_dir / "split_statistics.pkl"
    if stats_file.exists():
        print(f"  • {stats_file.name}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
