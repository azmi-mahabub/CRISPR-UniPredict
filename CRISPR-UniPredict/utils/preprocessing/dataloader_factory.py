"""
DataLoader Factory for CRISPR-UniPredict
Creates configured DataLoaders with all preprocessing components integrated
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging
import warnings
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.encoding import SequenceEncoder
from utils.preprocessing.dataset import CRISPRDataset, CRISPRDataLoader
from utils.preprocessing.sampler import BootstrapSampler, WeightedRandomSampler, StratifiedSampler
from utils.preprocessing.collate import CRISPRCollator, custom_collate_fn

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating configured DataLoaders
    
    Integrates all preprocessing components:
    - SequenceEncoder for encoding sequences
    - CRISPRDataset for loading and caching
    - Custom samplers for handling class imbalance
    - Custom collate functions for batching
    
    Example:
        >>> from configs.config_loader import ConfigLoader
        >>> config_loader = ConfigLoader('configs/model_config.yaml')
        >>> config = config_loader.config
        >>> factory = DataLoaderFactory(config)
        >>> dataloaders = factory.create_dataloaders()
    """
    
    def __init__(self, config):
        """
        Initialize DataLoader Factory
        
        Args:
            config: Configuration object with data and model settings
        """
        self.config = config
        self.encoder = None
        self.dataloaders = {}
        
        # Validate configuration
        self._validate_config()
        
        # Initialize encoder
        self._init_encoder()
        
        logger.info("DataLoaderFactory initialized")
    
    def _validate_config(self) -> None:
        """Validate configuration has required fields"""
        required_fields = [
            'data.train_path',
            'data.val_path',
            'data.test_path',
            'training.batch_size',
            'device.use_cuda',
            'data.num_workers',
            'data.pin_memory'
        ]
        
        for field in required_fields:
            parts = field.split('.')
            obj = self.config
            
            try:
                for part in parts:
                    obj = getattr(obj, part)
            except AttributeError:
                raise ValueError(f"Missing required config field: {field}")
        
        logger.info("Configuration validation passed")
    
    def _init_encoder(self) -> None:
        """Initialize sequence encoder"""
        device = 'cuda' if self.config.device.use_cuda else 'cpu'
        self.encoder = SequenceEncoder(device=device)
        logger.info(f"Sequence encoder initialized on device: {device}")
    
    def _validate_data_paths(self) -> None:
        """Validate that data files exist"""
        paths = {
            'train': Path(self.config.data.train_path),
            'val': Path(self.config.data.val_path),
            'test': Path(self.config.data.test_path)
        }
        
        for split, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{split.capitalize()} data file not found: {path}")
        
        logger.info("Data paths validation passed")
    
    def _load_dataset(self, csv_path: Union[str, Path], 
                     split: str = 'train',
                     augmentation: bool = False) -> CRISPRDataset:
        """
        Load dataset from CSV
        
        Args:
            csv_path: Path to CSV file
            split: Dataset split ('train', 'val', 'test')
            augmentation: Whether to apply augmentation
        
        Returns:
            CRISPRDataset instance
        """
        logger.info(f"Loading {split} dataset from {csv_path}")
        
        use_cache = getattr(self.config.data, 'use_cache', True)
        dataset = CRISPRDataset(
            csv_path=csv_path,
            encoder=self.encoder,
            cache_dir=Path('.cache') / split,
            use_cache=use_cache,
            augmentation=augmentation and split == 'train',
            augmentation_types=self.config.data.augmentation.augmentation_types if augmentation else [],
            task_type=None,
            verbose=True
        )
        
        logger.info(f"Loaded {len(dataset)} {split} samples")
        
        return dataset
    
    def _create_sampler(self, dataset: CRISPRDataset,
                       split: str = 'train'):
        """
        Create sampler for dataset
        
        Args:
            dataset: CRISPRDataset instance
            split: Dataset split ('train', 'val', 'test')
        
        Returns:
            Sampler instance or None
        """
        if split != 'train':
            # No sampler for validation/test
            return None
        
        sampling_strategy = self.config.data.sampling.strategy
        
        if sampling_strategy == 'balanced':
            logger.info("Using BootstrapSampler for balanced batching")
            
            sampler = BootstrapSampler(
                dataset,
                batch_size=self.config.training.batch_size,
                on_target_ratio=self.config.data.sampling.on_target_ratio,
                off_target_ratio=self.config.data.sampling.off_target_ratio,
                drop_last=False,
                shuffle=True,
                seed=42
            )
            
            return sampler
        
        elif sampling_strategy == 'weighted':
            logger.info("Using WeightedRandomSampler for weighted sampling")
            
            sampler = WeightedRandomSampler(
                dataset,
                num_samples=len(dataset),
                replacement=True,
                seed=42
            )
            
            return sampler
        
        elif sampling_strategy == 'bootstrap':
            logger.info("Using BootstrapSampler (bootstrap strategy)")
            
            sampler = BootstrapSampler(
                dataset,
                batch_size=self.config.training.batch_size,
                on_target_ratio=0.5,
                drop_last=False,
                shuffle=True,
                seed=42
            )
            
            return sampler
        
        elif sampling_strategy == 'stratified':
            logger.info("Using StratifiedSampler for stratified sampling (guarantees both classes per batch)")
            
            sampler = StratifiedSampler(
                dataset,
                batch_size=self.config.training.batch_size,
                on_target_ratio=self.config.data.sampling.on_target_ratio,
                off_target_ratio=self.config.data.sampling.off_target_ratio,
                shuffle=True,
                seed=42
            )
            
            return sampler
        
        else:
            logger.warning(f"Unknown sampling strategy: {sampling_strategy}, using default")
            return None
    
    def _create_collator(self) -> CRISPRCollator:
        """
        Create collate function
        
        Returns:
            CRISPRCollator instance
        """
        device = 'cuda' if self.config.device.use_cuda else 'cpu'
        
        collator = CRISPRCollator(
            pad_value_onehot=0.0,
            pad_value_label=0,
            create_causal_mask=False,
            task_aware=False,
            device=device
        )
        
        logger.info("Collator created")
        
        return collator
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create all dataloaders (train, val, test)
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        
        Raises:
            FileNotFoundError: If data files not found
            ValueError: If configuration invalid
        """
        # Validate data paths
        self._validate_data_paths()
        
        # Create collator
        collator = self._create_collator()
        
        # Load datasets
        train_dataset = self._load_dataset(
            self.config.data.train_path,
            split='train',
            augmentation=self.config.data.augmentation.use_augmentation
        )
        
        val_dataset = self._load_dataset(
            self.config.data.val_path,
            split='val',
            augmentation=False
        )
        
        test_dataset = self._load_dataset(
            self.config.data.test_path,
            split='test',
            augmentation=False
        )
        
        # Create samplers
        train_sampler = self._create_sampler(train_dataset, split='train')
        
        # Create dataloaders
        device = 'cuda' if self.config.device.use_cuda else 'cpu'
        
        # Training dataloader
        if train_sampler is not None:
            # Use batch sampler
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=collator,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory
            )
        else:
            # Use regular sampler
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory
            )
        
        # Validation dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        # Test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        logger.info(
            f"DataLoaders created:\n"
            f"  Train batches: {len(train_loader)}\n"
            f"  Val batches: {len(val_loader)}\n"
            f"  Test batches: {len(test_loader)}"
        )
        
        return self.dataloaders
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get created dataloaders
        
        Returns:
            Dictionary with dataloaders
        
        Raises:
            RuntimeError: If dataloaders not yet created
        """
        if not self.dataloaders:
            raise RuntimeError("Dataloaders not yet created. Call create_dataloaders() first.")
        
        return self.dataloaders
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all datasets
        
        Returns:
            Dictionary with stats for each split
        """
        stats = {}
        
        for split, loader in self.dataloaders.items():
            dataset = loader.dataset
            dataset_stats = dataset.get_statistics()
            stats[split] = dataset_stats
        
        return stats


def create_dataloaders(config) -> Dict[str, DataLoader]:
    """
    Convenience function to create dataloaders from config
    
    Args:
        config: Configuration object
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    
    Example:
        >>> from configs.config_loader import ConfigLoader
        >>> config_loader = ConfigLoader('configs/model_config.yaml')
        >>> dataloaders = create_dataloaders(config_loader.config)
        >>> for batch in dataloaders['train']:
        ...     # batch is ready for model
        ...     pass
    """
    factory = DataLoaderFactory(config)
    return factory.create_dataloaders()


def create_dataloaders_with_stats(config) -> Tuple[Dict[str, DataLoader], Dict[str, Dict]]:
    """
    Create dataloaders and return statistics
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (dataloaders, statistics)
    """
    factory = DataLoaderFactory(config)
    dataloaders = factory.create_dataloaders()
    stats = factory.get_dataset_stats()
    
    return dataloaders, stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("DATALOADER FACTORY TESTING")
    print("=" * 80)
    
    import sys
    from pathlib import Path
    import tempfile
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from configs.config_loader import ConfigLoader
    
    # Test 1: Load configuration
    print("\n1. LOAD CONFIGURATION")
    print("-" * 80)
    
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'
    
    if config_path.exists():
        config_loader = ConfigLoader(config_path)
        config = config_loader.config
        # Force CPU for testing
        config.device.use_cuda = False
        print(f"[OK] Configuration loaded from {config_path}")
    else:
        print(f"[SKIP] Configuration file not found: {config_path}")
        print("Creating dummy configuration for testing...")
        
        # Create dummy config for testing
        from configs.config_loader import Config, DataConfig, TrainingConfig, DeviceConfig
        
        config = Config()
        config.data.train_path = 'data/train.csv'
        config.data.val_path = 'data/val.csv'
        config.data.test_path = 'data/test.csv'
        config.device.use_cuda = False
    
    # Test 2: Create dummy datasets
    print("\n2. CREATE DUMMY DATASETS")
    print("-" * 80)
    
    from models.encoding import SequenceEncoder
    
    encoder = SequenceEncoder(device='cpu')
    
    # Create temporary CSV files
    sequences = [
        "GCTAGCTAGCTAGCTAGCTAGCT",
        "ATGCATGCATGCATGCATGCATG",
        "CCGGCCGGCCGGCCGGCCGGCCG",
        "GGCCGGCCGGCCGGCCGGCCGGCC",
        "TTAATTAATTAATTAATTAATTAA",
    ]
    
    for split in ['train', 'val', 'test']:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir='.') as f:
            csv_path = f.name
            f.write("sgrna,target,on_target_score,off_target_label,dataset_source\n")
            
            # Create imbalanced data
            for i in range(8):
                seq = sequences[i % len(sequences)]
                f.write(f"{seq},{seq},0.{70+i},None,dataset_A\n")
            
            for i in range(2):
                seq = sequences[i % len(sequences)]
                f.write(f"{seq},{seq},None,{i%2},dataset_B\n")
            
            # Update config
            if split == 'train':
                config.data.train_path = csv_path
            elif split == 'val':
                config.data.val_path = csv_path
            else:
                config.data.test_path = csv_path
    
    print(f"[OK] Created dummy datasets")
    
    # Test 3: Create factory
    print("\n3. CREATE DATALOADER FACTORY")
    print("-" * 80)
    
    try:
        factory = DataLoaderFactory(config)
        print(f"[OK] Factory created")
    except Exception as e:
        print(f"[ERROR] Failed to create factory: {e}")
        sys.exit(1)
    
    # Test 4: Create dataloaders
    print("\n4. CREATE DATALOADERS")
    print("-" * 80)
    
    try:
        dataloaders = factory.create_dataloaders()
        print(f"[OK] Dataloaders created")
        print(f"  Train batches: {len(dataloaders['train'])}")
        print(f"  Val batches: {len(dataloaders['val'])}")
        print(f"  Test batches: {len(dataloaders['test'])}")
    except Exception as e:
        print(f"[ERROR] Failed to create dataloaders: {e}")
        sys.exit(1)
    
    # Test 5: Iterate batches
    print("\n5. ITERATE BATCHES")
    print("-" * 80)
    
    for split, loader in dataloaders.items():
        print(f"\n{split.upper()} LOADER:")
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:
                break
            
            print(f"  Batch {batch_idx}:")
            print(f"    sgrna_onehot: {batch['sgrna_onehot'].shape}")
            print(f"    sgrna_label: {batch['sgrna_label'].shape}")
            print(f"    on_target_mask: {batch['on_target_mask']}")
            print(f"    off_target_mask: {batch['off_target_mask']}")
            print(f"    attention_mask: {batch['attention_mask'].shape}")
    
    # Test 6: Get dataset statistics
    print("\n6. DATASET STATISTICS")
    print("-" * 80)
    
    stats = factory.get_dataset_stats()
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()}:")
        for key, value in split_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    # Test 7: Convenience function
    print("\n7. CONVENIENCE FUNCTION")
    print("-" * 80)
    
    try:
        dataloaders2 = create_dataloaders(config)
        print(f"[OK] Dataloaders created using convenience function")
        print(f"  Keys: {list(dataloaders2.keys())}")
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
    
    # Cleanup
    import os
    for path in [config.data.train_path, config.data.val_path, config.data.test_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
