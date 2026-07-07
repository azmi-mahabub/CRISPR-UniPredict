"""
Unified Data Formatter for CRISPR-UniPredict
Converts CRISPR_HNN and CCLMoff datasets to a unified format
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class UnifiedDataFormatter:
    """Format and unify datasets from different sources"""
    
    # Valid DNA/RNA nucleotides
    VALID_NUCLEOTIDES = set('ACGTU')
    
    # Column name variations to detect
    SEQUENCE_KEYWORDS = ['seq', 'sequence', 'guide', 'grna', 'sgrna', 'target']
    INDEL_KEYWORDS = ['indel', 'efficiency', 'activity', 'score']
    LABEL_KEYWORDS = ['label', 'off_target', 'offtarget', 'pred']
    
    def __init__(self, base_dir=None):
        """Initialize formatter"""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "data"
        
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create output directories
        (self.processed_dir / "on_target").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "off_target").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "combined").mkdir(parents=True, exist_ok=True)
        
        self.statistics = {
            'total_sequences': 0,
            'total_valid': 0,
            'total_invalid': 0,
            'datasets': {},
            'on_target_count': 0,
            'off_target_count': 0,
            'errors': []
        }
    
    def validate_sequence(self, seq: str) -> bool:
        """
        Validate if sequence contains only valid DNA/RNA nucleotides
        
        Args:
            seq: DNA/RNA sequence string
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(seq, str):
            return False
        
        if len(seq) == 0:
            return False
        
        # Convert to uppercase and check
        seq_upper = seq.upper()
        
        # Replace U with T for consistency
        seq_upper = seq_upper.replace('U', 'T')
        
        # Check if all characters are valid nucleotides
        return all(c in self.VALID_NUCLEOTIDES for c in seq_upper)
    
    def normalize_indel_frequency(self, value: float, 
                                 min_val: float = 0.0, 
                                 max_val: float = 1.0) -> Optional[float]:
        """
        Normalize indel frequency to 0-1 range using min-max scaling
        
        Args:
            value: Original indel frequency value
            min_val: Minimum value in dataset
            max_val: Maximum value in dataset
            
        Returns:
            Normalized value between 0-1, or None if invalid
        """
        try:
            value = float(value)
            
            # Handle edge cases
            if pd.isna(value):
                return None
            
            if min_val == max_val:
                return 0.5  # Default to middle if no range
            
            # Min-max normalization
            normalized = (value - min_val) / (max_val - min_val)
            
            # Clip to [0, 1]
            normalized = max(0.0, min(1.0, normalized))
            
            return normalized
        except (ValueError, TypeError):
            return None
    
    def detect_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """
        Detect column name based on keywords
        
        Args:
            df: DataFrame to search
            keywords: List of keywords to match
            
        Returns:
            Column name if found, None otherwise
        """
        col_lower = [col.lower() for col in df.columns]
        
        for keyword in keywords:
            for i, col in enumerate(col_lower):
                if keyword in col:
                    return df.columns[i]
        
        return None
    
    def format_crispr_hnn_dataset(self, dataset_name: str, 
                                 file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Format a CRISPR_HNN dataset to unified format
        
        Args:
            dataset_name: Name of dataset (e.g., 'ESP', 'WT')
            file_path: Path to CSV file (auto-detect if None)
            
        Returns:
            Formatted DataFrame
        """
        print(f"\n{'─' * 80}")
        print(f"Processing CRISPR_HNN: {dataset_name}")
        print(f"{'─' * 80}")
        
        try:
            # Auto-detect file path if not provided
            if file_path is None:
                crispr_dir = self.raw_dir / "crispr_hnn"
                matching_files = list(crispr_dir.glob(f"*{dataset_name}*.csv"))
                
                if not matching_files:
                    raise FileNotFoundError(f"No file found for {dataset_name}")
                
                file_path = matching_files[0]
            
            # Load file
            df = pd.read_csv(file_path)
            print(f"  Loaded: {file_path.name}")
            print(f"  Shape: {df.shape}")
            
            # Detect columns
            seq_col = self.detect_column(df, self.SEQUENCE_KEYWORDS)
            indel_col = self.detect_column(df, self.INDEL_KEYWORDS)
            
            if seq_col is None:
                raise ValueError("Could not detect sequence column")
            if indel_col is None:
                raise ValueError("Could not detect indel column")
            
            print(f"  Sequence column: {seq_col}")
            print(f"  Indel column: {indel_col}")
            
            # Create formatted dataframe
            formatted = pd.DataFrame()
            
            # Extract and validate sequences
            sequences = []
            indel_scores = []
            valid_count = 0
            invalid_count = 0
            
            for idx, row in df.iterrows():
                seq = str(row[seq_col]).strip()
                indel = row[indel_col]
                
                # Validate sequence
                if not self.validate_sequence(seq):
                    invalid_count += 1
                    continue
                
                # Normalize sequence (U -> T)
                seq = seq.upper().replace('U', 'T')
                
                sequences.append(seq)
                indel_scores.append(indel)
                valid_count += 1
            
            print(f"  Valid sequences: {valid_count}")
            print(f"  Invalid sequences: {invalid_count}")
            
            # Normalize indel scores
            min_indel = min(indel_scores) if indel_scores else 0
            max_indel = max(indel_scores) if indel_scores else 1
            
            normalized_scores = [
                self.normalize_indel_frequency(score, min_indel, max_indel)
                for score in indel_scores
            ]
            
            # Build formatted dataframe
            formatted['sgrna_sequence'] = sequences
            formatted['on_target_score'] = normalized_scores
            formatted['target_sequence'] = None
            formatted['pam_sequence'] = None
            formatted['off_target_label'] = None
            formatted['dataset_source'] = dataset_name
            formatted['cell_line'] = self._extract_cell_line(dataset_name)
            formatted['detection_method'] = 'CRISPR_HNN'
            formatted['task_type'] = 'on_target'
            
            # Update statistics
            self.statistics['datasets'][dataset_name] = {
                'valid': valid_count,
                'invalid': invalid_count,
                'task': 'on_target'
            }
            self.statistics['total_sequences'] += valid_count + invalid_count
            self.statistics['total_valid'] += valid_count
            self.statistics['total_invalid'] += invalid_count
            self.statistics['on_target_count'] += valid_count
            
            print(f"  ✓ Formatted: {len(formatted)} sequences")
            
            return formatted
            
        except Exception as e:
            error_msg = f"Error processing {dataset_name}: {str(e)}"
            print(f"  ✗ {error_msg}")
            self.statistics['errors'].append(error_msg)
            return pd.DataFrame()
    
    def format_cclmoff_dataset(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Format CCLMoff dataset to unified format
        
        Args:
            file_path: Path to CSV file (auto-detect if None)
            
        Returns:
            Formatted DataFrame
        """
        print(f"\n{'─' * 80}")
        print(f"Processing CCLMoff Dataset")
        print(f"{'─' * 80}")
        
        try:
            # Auto-detect file path if not provided
            if file_path is None:
                cclmoff_dir = self.raw_dir / "cclmoff"
                matching_files = list(cclmoff_dir.glob("*.csv"))
                
                if not matching_files:
                    raise FileNotFoundError("No CCLMoff file found")
                
                file_path = matching_files[0]
            
            # Load file (with low_memory to handle mixed types)
            print(f"  Loading: {file_path.name}")
            df = pd.read_csv(file_path, low_memory=False)
            print(f"  Shape: {df.shape}")
            
            # Detect columns
            sgrna_col = self.detect_column(df, ['sgrna', 'guide'])
            off_col = self.detect_column(df, ['off_seq', 'off_target'])
            label_col = self.detect_column(df, self.LABEL_KEYWORDS)
            
            if sgrna_col is None:
                raise ValueError("Could not detect sgRNA column")
            if label_col is None:
                raise ValueError("Could not detect label column")
            
            print(f"  sgRNA column: {sgrna_col}")
            print(f"  Off-target column: {off_col}")
            print(f"  Label column: {label_col}")
            
            # Create formatted dataframe
            formatted = pd.DataFrame()
            
            # Extract and validate sequences
            sgrna_seqs = []
            off_seqs = []
            labels = []
            valid_count = 0
            invalid_count = 0
            
            for idx, row in df.iterrows():
                sgrna = str(row[sgrna_col]).strip()
                off_seq = str(row[off_col]).strip() if off_col and pd.notna(row[off_col]) else None
                label = int(row[label_col])
                
                # Validate sgRNA sequence
                if not self.validate_sequence(sgrna):
                    invalid_count += 1
                    continue
                
                # Normalize sequences (U -> T)
                sgrna = sgrna.upper().replace('U', 'T')
                if off_seq:
                    off_seq = off_seq.upper().replace('U', 'T')
                
                sgrna_seqs.append(sgrna)
                off_seqs.append(off_seq)
                labels.append(label)
                valid_count += 1
                
                # Progress indicator
                if valid_count % 100000 == 0:
                    print(f"    Processed: {valid_count:,} sequences")
            
            print(f"  Valid sequences: {valid_count:,}")
            print(f"  Invalid sequences: {invalid_count:,}")
            
            # Build formatted dataframe
            formatted['sgrna_sequence'] = sgrna_seqs
            formatted['target_sequence'] = off_seqs
            formatted['pam_sequence'] = None
            formatted['on_target_score'] = None
            formatted['off_target_label'] = labels
            formatted['dataset_source'] = 'CCLMoff'
            formatted['cell_line'] = None
            formatted['detection_method'] = 'CCLMoff'
            formatted['task_type'] = 'off_target'
            
            # Update statistics
            self.statistics['datasets']['CCLMoff'] = {
                'valid': valid_count,
                'invalid': invalid_count,
                'task': 'off_target'
            }
            self.statistics['total_sequences'] += valid_count + invalid_count
            self.statistics['total_valid'] += valid_count
            self.statistics['total_invalid'] += invalid_count
            self.statistics['off_target_count'] += valid_count
            
            print(f"  ✓ Formatted: {len(formatted):,} sequences")
            
            return formatted
            
        except Exception as e:
            error_msg = f"Error processing CCLMoff: {str(e)}"
            print(f"  ✗ {error_msg}")
            self.statistics['errors'].append(error_msg)
            return pd.DataFrame()
    
    def _extract_cell_line(self, dataset_name: str) -> Optional[str]:
        """Extract cell line from dataset name"""
        cell_lines = {
            'ESP': 'ESP',
            'HCT116': 'HCT116',
            'HELA': 'HeLa',
            'HF': 'HumanFibroblast',
            'HL60': 'HL60',
            'WT': 'WildType',
            'Sniper': 'Sniper',
            'SpCas9': 'SpCas9',
            'xCas': 'xCas'
        }
        
        for key, value in cell_lines.items():
            if key in dataset_name:
                return value
        
        return None
    
    def process_all_datasets(self) -> Dict:
        """
        Process all datasets and save to unified format
        
        Returns:
            Statistics dictionary
        """
        print("\n" + "=" * 80)
        print("UNIFIED DATA FORMATTING")
        print("=" * 80)
        
        on_target_dfs = []
        off_target_dfs = []
        
        # Process CRISPR_HNN datasets
        print("\n" + "=" * 80)
        print("PROCESSING CRISPR_HNN DATASETS")
        print("=" * 80)
        
        crispr_dir = self.raw_dir / "crispr_hnn"
        if crispr_dir.exists():
            csv_files = sorted(crispr_dir.glob("*.csv"))
            
            for file_path in csv_files:
                # Extract dataset name
                dataset_name = file_path.stem.split('（')[0].strip()
                
                if dataset_name == 'test_ont':
                    continue  # Skip test file
                
                # Format dataset
                formatted_df = self.format_crispr_hnn_dataset(dataset_name, file_path)
                
                if not formatted_df.empty:
                    # Save individual file
                    output_path = self.processed_dir / "on_target" / f"{dataset_name}_formatted.csv"
                    formatted_df.to_csv(output_path, index=False)
                    print(f"  Saved: {output_path.name}")
                    
                    on_target_dfs.append(formatted_df)
        
        # Process CCLMoff dataset
        print("\n" + "=" * 80)
        print("PROCESSING CCLMOFF DATASET")
        print("=" * 80)
        
        formatted_df = self.format_cclmoff_dataset()
        
        if not formatted_df.empty:
            # Save individual file
            output_path = self.processed_dir / "off_target" / "CCLMoff_formatted.csv"
            formatted_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path.name}")
            
            off_target_dfs.append(formatted_df)
        
        # Combine on-target datasets
        print("\n" + "=" * 80)
        print("COMBINING DATASETS")
        print("=" * 80)
        
        if on_target_dfs:
            combined_on_target = pd.concat(on_target_dfs, ignore_index=True)
            output_path = self.processed_dir / "combined" / "combined_on_target.csv"
            combined_on_target.to_csv(output_path, index=False)
            print(f"\n✓ Combined on-target: {len(combined_on_target):,} sequences")
            print(f"  Saved: {output_path.name}")
        
        if off_target_dfs:
            combined_off_target = pd.concat(off_target_dfs, ignore_index=True)
            output_path = self.processed_dir / "combined" / "combined_off_target.csv"
            combined_off_target.to_csv(output_path, index=False)
            print(f"\n✓ Combined off-target: {len(combined_off_target):,} sequences")
            print(f"  Saved: {output_path.name}")
        
        # Create unified dataset
        print("\n" + "=" * 80)
        print("CREATING UNIFIED DATASET")
        print("=" * 80)
        
        all_dfs = on_target_dfs + off_target_dfs
        
        if all_dfs:
            unified_df = pd.concat(all_dfs, ignore_index=True)
            output_path = self.processed_dir / "combined" / "unified_dataset.csv"
            unified_df.to_csv(output_path, index=False)
            print(f"\n✓ Unified dataset: {len(unified_df):,} sequences")
            print(f"  Saved: {output_path.name}")
        
        # Save statistics
        print("\n" + "=" * 80)
        print("SAVING STATISTICS")
        print("=" * 80)
        
        stats_path = self.processed_dir / "combined" / "formatting_statistics.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(self.statistics, f)
        print(f"\n✓ Statistics saved: {stats_path.name}")
        
        # Print summary
        self._print_summary()
        
        return self.statistics
    
    def _print_summary(self):
        """Print formatting summary"""
        print("\n" + "=" * 80)
        print("FORMATTING SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal sequences processed: {self.statistics['total_sequences']:,}")
        print(f"Valid sequences: {self.statistics['total_valid']:,}")
        print(f"Invalid sequences: {self.statistics['total_invalid']:,}")
        print(f"On-target sequences: {self.statistics['on_target_count']:,}")
        print(f"Off-target sequences: {self.statistics['off_target_count']:,}")
        
        if self.statistics['total_sequences'] > 0:
            valid_pct = (self.statistics['total_valid'] / self.statistics['total_sequences']) * 100
            print(f"Validity rate: {valid_pct:.2f}%")
        
        print(f"\nDatasets processed:")
        for dataset_name, info in self.statistics['datasets'].items():
            print(f"  {dataset_name:20s} - Valid: {info['valid']:>10,d}, Invalid: {info['invalid']:>8,d}")
        
        if self.statistics['errors']:
            print(f"\nErrors ({len(self.statistics['errors'])}):")
            for error in self.statistics['errors']:
                print(f"  ⚠ {error}")
        
        print("\n" + "=" * 80)
        print("✓ FORMATTING COMPLETE")
        print("=" * 80)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("CRISPR-UniPredict Unified Data Formatter")
    print("=" * 80)
    
    # Create formatter
    formatter = UnifiedDataFormatter()
    
    # Process all datasets
    statistics = formatter.process_all_datasets()
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    
    processed_dir = formatter.processed_dir
    
    print(f"\nOn-target datasets: {(processed_dir / 'on_target').name}/")
    if (processed_dir / 'on_target').exists():
        for f in sorted((processed_dir / 'on_target').glob("*.csv")):
            print(f"  • {f.name}")
    
    print(f"\nOff-target datasets: {(processed_dir / 'off_target').name}/")
    if (processed_dir / 'off_target').exists():
        for f in sorted((processed_dir / 'off_target').glob("*.csv")):
            print(f"  • {f.name}")
    
    print(f"\nCombined datasets: {(processed_dir / 'combined').name}/")
    if (processed_dir / 'combined').exists():
        for f in sorted((processed_dir / 'combined').glob("*")):
            print(f"  • {f.name}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
