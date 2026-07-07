"""
PyTorch Dataset for CRISPR Prediction
Handles loading, encoding, and caching of CRISPR sequences and labels
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import logging
from torch.utils.data import Dataset
import pickle
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CRISPRDataset(Dataset):
    """
    PyTorch Dataset for CRISPR prediction tasks
    
    Loads CRISPR sequences and labels from CSV files, encodes them,
    and provides efficient access with optional caching.
    
    Features:
    - Loads data from CSV files
    - Encodes sequences (one-hot and label encoding)
    - Caches encoded sequences for faster loading
    - Handles missing labels gracefully
    - Supports task filtering and sampling
    - Optional data augmentation
    
    Input CSV format:
    - sgrna: sgRNA sequence (required)
    - target: Target sequence (required)
    - on_target_score: On-target efficiency score (optional, float)
    - off_target_label: Off-target label (optional, int: 0 or 1)
    - dataset_source: Source dataset name (optional)
    - cell_line: Cell line (optional)
    - ... other metadata columns
    """
    
    def __init__(self,
                 csv_path: Union[str, Path],
                 encoder,
                 cache_dir: Optional[Union[str, Path]] = None,
                 use_cache: bool = True,
                 augmentation: bool = False,
                 augmentation_types: Optional[List[str]] = None,
                 task_type: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize CRISPR Dataset
        
        Args:
            csv_path: Path to CSV file with sequences and labels
            encoder: SequenceEncoder instance for encoding sequences
            cache_dir: Directory to cache encoded sequences (default: .cache)
            use_cache: Whether to use cached encodings (default: True)
            augmentation: Whether to apply data augmentation (default: False)
            augmentation_types: List of augmentation types to apply
                              (e.g., ['reverse_complement', 'noise'])
            task_type: Filter dataset to specific task ('on_target', 'off_target', or None for all)
            verbose: Print progress information (default: True)
        
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns missing
        """
        self.csv_path = Path(csv_path)
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir else Path('.cache')
        self.use_cache = use_cache
        self.augmentation = augmentation
        self.augmentation_types = augmentation_types or []
        self.task_type = task_type
        self.verbose = verbose
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if self.verbose:
            logger.info(f"Loading dataset from {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Normalize sequence column names first
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Handle both column naming conventions
        # Map sgrna_sequence -> sgrna and target_sequence -> target
        if 'sgrna_sequence' in self.df.columns and 'sgrna' not in self.df.columns:
            self.df.rename(columns={'sgrna_sequence': 'sgrna'}, inplace=True)
        if 'target_sequence' in self.df.columns and 'target' not in self.df.columns:
            self.df.rename(columns={'target_sequence': 'target'}, inplace=True)
        
        # Validate required columns
        required_columns = ['sgrna', 'target']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing sequences
        mask_valid_sequences = self.df['sgrna'].notna() & self.df['target'].notna()
        
        # Validate sequences (only ACGT allowed)
        valid_nucleotides = set('ACGTU')
        def is_valid_sequence(seq):
            if not isinstance(seq, str):
                return False
            return all(c.upper() in valid_nucleotides for c in seq)
        
        mask_valid_sequences = mask_valid_sequences & (
            self.df['sgrna'].apply(is_valid_sequence) & 
            self.df['target'].apply(is_valid_sequence)
        )
        
        if (~mask_valid_sequences).sum() > 0:
            if self.verbose:
                logger.info(f"Removing {(~mask_valid_sequences).sum()} rows with missing or invalid sequences")
            self.df = self.df[mask_valid_sequences].reset_index(drop=True)
        
        # Handle task filtering
        self.original_indices = np.arange(len(self.df))
        if task_type == 'on_target':
            # Filter to samples with on_target_score
            mask = self.df['on_target_score'].notna()
            self.df = self.df[mask].reset_index(drop=True)
            self.original_indices = self.original_indices[mask.values]
        elif task_type == 'off_target':
            # Filter to samples with off_target_label
            mask = self.df['off_target_label'].notna()
            self.df = self.df[mask].reset_index(drop=True)
            self.original_indices = self.original_indices[mask.values]
        
        if self.verbose:
            logger.info(f"Loaded {len(self.df)} samples")
            if 'on_target_score' in self.df.columns:
                on_target_count = self.df['on_target_score'].notna().sum()
                logger.info(f"  On-target samples: {on_target_count}")
            if 'off_target_label' in self.df.columns:
                off_target_count = self.df['off_target_label'].notna().sum()
                logger.info(f"  Off-target samples: {off_target_count}")
        
        # Initialize encoding cache
        self._encoding_cache = {}
        self._load_or_create_cache()
    
    def _get_cache_path(self) -> Path:
        """Get path to cache file for this dataset"""
        # Create hash of CSV path and parameters
        cache_key = f"{self.csv_path.stem}_{self.task_type or 'all'}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        return self.cache_dir / f"encodings_{cache_hash}.pkl"
    
    def _load_or_create_cache(self) -> None:
        """Load cached encodings or create new cache"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            if self.verbose:
                logger.info(f"Loading cached encodings from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self._encoding_cache = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(self._encoding_cache)} cached encodings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Creating new cache.")
        
        # Create cache by encoding all sequences
        if self.verbose:
            logger.info("Creating encoding cache...")
        
        for idx in tqdm(range(len(self.df)), disable=not self.verbose, desc="Encoding sequences"):
            sgrna = self.df.iloc[idx]['sgrna']
            target = self.df.iloc[idx]['target']
            
            # Encode sequences
            sgrna_onehot = self.encoder.one_hot_encode(sgrna)
            sgrna_label = self.encoder.label_encode(sgrna, add_start_token=False)
            target_onehot = self.encoder.one_hot_encode(target)
            target_label = self.encoder.label_encode(target, add_start_token=False)
            
            self._encoding_cache[idx] = {
                'sgrna_onehot': sgrna_onehot,
                'sgrna_label': sgrna_label,
                'target_onehot': target_onehot,
                'target_label': target_label
            }
        
        # Save cache
        if self.use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self._encoding_cache, f)
                if self.verbose:
                    logger.info(f"Saved cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def _encode_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get encoded sequences for sample
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with encoded sequences
        """
        # Check cache first
        if idx in self._encoding_cache:
            return self._encoding_cache[idx]
        
        # Encode on-the-fly if not cached
        sgrna = self.df.iloc[idx]['sgrna']
        target = self.df.iloc[idx]['target']
        
        sgrna_onehot = self.encoder.one_hot_encode(sgrna)
        sgrna_label = self.encoder.label_encode(sgrna, add_start_token=False)
        target_onehot = self.encoder.one_hot_encode(target)
        target_label = self.encoder.label_encode(target, add_start_token=False)
        
        encodings = {
            'sgrna_onehot': sgrna_onehot,
            'sgrna_label': sgrna_label,
            'target_onehot': target_onehot,
            'target_label': target_label
        }
        
        # Cache for future use
        if self.use_cache:
            self._encoding_cache[idx] = encodings
        
        return encodings
    
    def _apply_augmentation(self, sgrna_onehot: torch.Tensor, 
                           sgrna_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to sequences
        
        Args:
            sgrna_onehot: One-hot encoded sequence
            sgrna_label: Label encoded sequence
        
        Returns:
            Augmented sequences
        """
        if not self.augmentation or not self.augmentation_types:
            return sgrna_onehot, sgrna_label
        
        # Apply random augmentation
        aug_type = np.random.choice(self.augmentation_types)
        
        if aug_type == 'reverse_complement':
            # Reverse complement
            sgrna_onehot = torch.flip(sgrna_onehot, dims=[0])
            sgrna_onehot = self._complement_onehot(sgrna_onehot)
            sgrna_label = torch.flip(sgrna_label, dims=[0])
            sgrna_label = self._complement_label(sgrna_label)
        
        elif aug_type == 'noise':
            # Add small noise to one-hot encoding
            noise = torch.randn_like(sgrna_onehot) * 0.01
            sgrna_onehot = torch.clamp(sgrna_onehot + noise, 0, 1)
        
        return sgrna_onehot, sgrna_label
    
    @staticmethod
    def _complement_onehot(onehot: torch.Tensor) -> torch.Tensor:
        """Get complement of one-hot encoded sequence"""
        # Complement: A<->T, C<->G
        # One-hot: [A, C, G, T] -> [T, G, C, A]
        complement = torch.zeros_like(onehot)
        complement[:, 0] = onehot[:, 3]  # A <- T
        complement[:, 1] = onehot[:, 2]  # C <- G
        complement[:, 2] = onehot[:, 1]  # G <- C
        complement[:, 3] = onehot[:, 0]  # T <- A
        return complement
    
    @staticmethod
    def _complement_label(label: torch.Tensor) -> torch.Tensor:
        """Get complement of label encoded sequence"""
        # Complement mapping: A(2)<->T(5), C(3)<->G(4)
        complement_map = {2: 5, 5: 2, 3: 4, 4: 3, 1: 1, 0: 0}
        complement = label.clone()
        for orig, comp in complement_map.items():
            complement[label == orig] = comp
        return complement
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, int, str, None]]:
        """
        Get one sample from dataset
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with:
            - sgrna_sequence: Original sgRNA sequence (str)
            - target_sequence: Original target sequence (str)
            - sgrna_onehot: One-hot encoded sgRNA (torch.Tensor)
            - sgrna_label: Label encoded sgRNA (torch.Tensor)
            - target_onehot: One-hot encoded target (torch.Tensor)
            - target_label: Label encoded target (torch.Tensor)
            - on_target_score: On-target score (float or None)
            - off_target_label: Off-target label (int or None)
            - metadata: Dictionary with additional info
        """
        row = self.df.iloc[idx]
        
        # Get encoded sequences
        encodings = self._encode_sequence(idx)
        sgrna_onehot = encodings['sgrna_onehot'].clone()
        sgrna_label = encodings['sgrna_label'].clone()
        
        # Apply augmentation if enabled
        if self.augmentation and self.training:
            sgrna_onehot, sgrna_label = self._apply_augmentation(sgrna_onehot, sgrna_label)
        
        # Get labels
        on_target_score = row.get('on_target_score', None)
        off_target_label = row.get('off_target_label', None)
        
        # Convert to float/int if not None
        if pd.notna(on_target_score):
            on_target_score = float(on_target_score)
        else:
            on_target_score = None
        
        if pd.notna(off_target_label):
            off_target_label = int(off_target_label)
        else:
            off_target_label = None
        
        # Collect metadata
        metadata_columns = [col for col in self.df.columns 
                           if col not in ['sgrna', 'target', 'on_target_score', 'off_target_label']]
        metadata = {col: row.get(col, None) for col in metadata_columns}
        
        return {
            'sgrna_sequence': row['sgrna'],
            'target_sequence': row['target'],
            'sgrna_onehot': sgrna_onehot,
            'sgrna_label': sgrna_label,
            'target_onehot': encodings['target_onehot'],
            'target_label': encodings['target_label'],
            'on_target_score': on_target_score,
            'off_target_label': off_target_label,
            'metadata': metadata
        }
    
    def get_task_indices(self) -> Dict[str, List[int]]:
        """
        Get indices for each task type
        
        Returns:
            Dictionary with:
            - 'on_target': Indices of samples with on_target_score
            - 'off_target': Indices of samples with off_target_label
            - 'both': Indices of samples with both labels
        """
        on_target_mask = self.df['on_target_score'].notna()
        off_target_mask = self.df['off_target_label'].notna()
        
        on_target_indices = np.where(on_target_mask)[0].tolist()
        off_target_indices = np.where(off_target_mask)[0].tolist()
        both_indices = np.where(on_target_mask & off_target_mask)[0].tolist()
        
        return {
            'on_target': on_target_indices,
            'off_target': off_target_indices,
            'both': both_indices
        }
    
    def filter_by_task(self, task_type: str) -> 'CRISPRDataset':
        """
        Create a filtered dataset for specific task
        
        Args:
            task_type: 'on_target', 'off_target', or 'both'
        
        Returns:
            New CRISPRDataset with filtered samples
        """
        if task_type not in ['on_target', 'off_target', 'both']:
            raise ValueError(f"Invalid task_type: {task_type}")
        
        # Create new dataset with task filtering
        new_dataset = CRISPRDataset(
            csv_path=self.csv_path,
            encoder=self.encoder,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            augmentation=self.augmentation,
            augmentation_types=self.augmentation_types,
            task_type=task_type if task_type != 'both' else None,
            verbose=False
        )
        
        # Share cache
        new_dataset._encoding_cache = self._encoding_cache
        
        return new_dataset
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(self.df),
            'on_target_samples': self.df['on_target_score'].notna().sum(),
            'off_target_samples': self.df['off_target_label'].notna().sum(),
            'both_labels': (self.df['on_target_score'].notna() & 
                           self.df['off_target_label'].notna()).sum(),
        }
        
        # On-target statistics
        if stats['on_target_samples'] > 0:
            on_target_scores = self.df['on_target_score'].dropna()
            stats['on_target_stats'] = {
                'mean': float(on_target_scores.mean()),
                'std': float(on_target_scores.std()),
                'min': float(on_target_scores.min()),
                'max': float(on_target_scores.max()),
            }
        
        # Off-target statistics
        if stats['off_target_samples'] > 0:
            off_target_labels = self.df['off_target_label'].dropna()
            stats['off_target_stats'] = {
                'positive': int((off_target_labels == 1).sum()),
                'negative': int((off_target_labels == 0).sum()),
                'positive_ratio': float((off_target_labels == 1).mean()),
            }
        
        # Dataset sources
        if 'dataset_source' in self.df.columns:
            stats['dataset_sources'] = self.df['dataset_source'].value_counts().to_dict()
        
        return stats
    
    def train(self):
        """Set dataset to training mode (enables augmentation)"""
        self.training = True
    
    def eval(self):
        """Set dataset to evaluation mode (disables augmentation)"""
        self.training = False
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"CRISPRDataset(path={self.csv_path.name}, "
                f"samples={len(self.df)}, "
                f"task_type={self.task_type})")


class CRISPRDataLoader:
    """
    Convenience wrapper for creating DataLoaders with CRISPR datasets
    
    Handles batch collation and device placement
    """
    
    def __init__(self, dataset: CRISPRDataset, 
                 batch_size: int = 32,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 device: str = 'cpu'):
        """
        Initialize data loader
        
        Args:
            dataset: CRISPRDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for GPU transfer
            device: Device to place data on
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Batched data
        """
        # Pad sequences to same length
        def pad_sequences(sequences):
            """Pad sequences to maximum length in batch"""
            max_len = max(seq.shape[0] for seq in sequences)
            padded = []
            for seq in sequences:
                if seq.shape[0] < max_len:
                    padding = torch.zeros(max_len - seq.shape[0], *seq.shape[1:], dtype=seq.dtype)
                    padded.append(torch.cat([seq, padding], dim=0))
                else:
                    padded.append(seq)
            return torch.stack(padded)
        
        # Pad one-hot sequences
        sgrna_onehot = pad_sequences([item['sgrna_onehot'] for item in batch])
        target_onehot = pad_sequences([item['target_onehot'] for item in batch])
        
        # Pad label sequences
        sgrna_label = pad_sequences([item['sgrna_label'].unsqueeze(-1) for item in batch]).squeeze(-1)
        target_label = pad_sequences([item['target_label'].unsqueeze(-1) for item in batch]).squeeze(-1)
        
        # Handle labels (may be None)
        on_target_scores = []
        off_target_labels = []
        
        for item in batch:
            on_target_scores.append(item['on_target_score'])
            off_target_labels.append(item['off_target_label'])
        
        # Convert to tensors if not all None
        on_target_tensor = None
        if any(score is not None for score in on_target_scores):
            on_target_tensor = torch.tensor(
                [score if score is not None else 0.0 for score in on_target_scores],
                dtype=torch.float32
            )
        
        off_target_tensor = None
        if any(label is not None for label in off_target_labels):
            off_target_tensor = torch.tensor(
                [label if label is not None else 0 for label in off_target_labels],
                dtype=torch.long
            )
        
        return {
            'sgrna_onehot': sgrna_onehot,
            'sgrna_label': sgrna_label,
            'target_onehot': target_onehot,
            'target_label': target_label,
            'on_target_score': on_target_tensor,
            'off_target_label': off_target_tensor,
            'sgrna_sequences': [item['sgrna_sequence'] for item in batch],
            'target_sequences': [item['target_sequence'] for item in batch],
            'metadata': [item['metadata'] for item in batch]
        }
    
    def __iter__(self):
        """Iterate over batches"""
        return iter(self.loader)
    
    def __len__(self):
        """Return number of batches"""
        return len(self.loader)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CRISPR DATASET TESTING")
    print("=" * 80)
    
    # Test 1: Create dummy CSV
    print("\n1. CREATE DUMMY DATASET")
    print("-" * 80)
    
    import tempfile
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.encoding import SequenceEncoder
    
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("sgrna,target,on_target_score,off_target_label,dataset_source,cell_line\n")
        
        # Generate dummy data
        sequences = [
            "GCTAGCTAGCTAGCTAGCTAGCT",
            "ATGCATGCATGCATGCATGCATG",
            "CCGGCCGGCCGGCCGGCCGGCCG",
            "GGCCGGCCGGCCGGCCGGCCGGCC",
            "TTAATTAATTAATTAATTAATTAA",
        ]
        
        for i, seq in enumerate(sequences):
            on_target = 0.5 + 0.1 * i if i < 3 else None
            off_target = i % 2 if i < 4 else None
            f.write(f"{seq},{seq},{on_target},{off_target},test_dataset,HEK293T\n")
    
    print(f"[OK] Created dummy dataset at {csv_path}")
    
    # Test 2: Load dataset
    print("\n2. LOAD DATASET")
    print("-" * 80)
    
    encoder = SequenceEncoder(device='cpu')
    dataset = CRISPRDataset(
        csv_path=csv_path,
        encoder=encoder,
        cache_dir='.cache',
        use_cache=True,
        verbose=True
    )
    
    print(f"[OK] Dataset loaded: {dataset}")
    
    # Test 3: Get sample
    print("\n3. GET SAMPLE")
    print("-" * 80)
    
    sample = dataset[0]
    print(f"  sgrna_sequence: {sample['sgrna_sequence']}")
    print(f"  sgrna_onehot shape: {sample['sgrna_onehot'].shape}")
    print(f"  sgrna_label shape: {sample['sgrna_label'].shape}")
    print(f"  on_target_score: {sample['on_target_score']}")
    print(f"  off_target_label: {sample['off_target_label']}")
    print(f"  metadata: {sample['metadata']}")
    
    # Test 4: Get task indices
    print("\n4. GET TASK INDICES")
    print("-" * 80)
    
    task_indices = dataset.get_task_indices()
    print(f"  On-target indices: {task_indices['on_target']}")
    print(f"  Off-target indices: {task_indices['off_target']}")
    print(f"  Both labels indices: {task_indices['both']}")
    
    # Test 5: Filter by task
    print("\n5. FILTER BY TASK")
    print("-" * 80)
    
    on_target_dataset = dataset.filter_by_task('on_target')
    print(f"[OK] On-target dataset: {on_target_dataset}")
    print(f"  Samples: {len(on_target_dataset)}")
    
    # Test 6: Get statistics
    print("\n6. DATASET STATISTICS")
    print("-" * 80)
    
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test 7: Data loader
    print("\n7. CREATE DATA LOADER")
    print("-" * 80)
    
    dataloader = CRISPRDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[OK] DataLoader created")
    print(f"  Batch size: 2")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Test 8: Iterate batches
    print("\n8. ITERATE BATCHES")
    print("-" * 80)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"  Batch {batch_idx}:")
        print(f"    sgrna_onehot shape: {batch['sgrna_onehot'].shape}")
        print(f"    sgrna_label shape: {batch['sgrna_label'].shape}")
        print(f"    on_target_score: {batch['on_target_score']}")
        print(f"    off_target_label: {batch['off_target_label']}")
        print(f"    sgrna_sequences: {batch['sgrna_sequences']}")
        if batch_idx >= 1:
            break
    
    # Cleanup
    import os
    os.remove(csv_path)
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
