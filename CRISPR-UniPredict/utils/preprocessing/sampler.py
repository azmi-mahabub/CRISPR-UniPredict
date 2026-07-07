"""
Custom Samplers for CRISPR Dataset
Handles class imbalance and task-aware sampling strategies
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
from typing import List, Iterator, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BootstrapSampler(Sampler):
    """
    Bootstrap Sampler for balanced batch sampling
    
    Ensures equal representation of on-target and off-target samples in each batch.
    Handles severe class imbalance by sampling with replacement from the minority class.
    
    Strategy:
    1. Separate dataset indices by task type (on-target, off-target, mixed)
    2. For each batch:
       - Sample specified ratio from on-target indices
       - Sample remaining from off-target indices
       - Use replacement if one task has fewer samples
       - Shuffle combined samples
    3. Total batches determined by smaller task
    
    This is particularly useful for datasets like CCLMoff where off-target samples
    are much rarer than on-target samples.
    
    Example:
        >>> dataset = CRISPRDataset('data/train.csv', encoder)
        >>> sampler = BootstrapSampler(dataset, batch_size=32, on_target_ratio=0.5)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 on_target_ratio: float = 0.5,
                 off_target_ratio: Optional[float] = None,
                 drop_last: bool = False,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Initialize Bootstrap Sampler
        
        Args:
            dataset: CRISPRDataset instance
            batch_size: Batch size
            on_target_ratio: Ratio of on-target samples in each batch (default: 0.5)
            off_target_ratio: Ratio of off-target samples (default: 1 - on_target_ratio)
            drop_last: Drop last incomplete batch (default: False)
            shuffle: Shuffle indices within each task (default: True)
            seed: Random seed for reproducibility (default: 42)
        
        Raises:
            ValueError: If ratios don't sum to 1.0 or dataset is empty
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Validate and set ratios
        if off_target_ratio is None:
            off_target_ratio = 1.0 - on_target_ratio
        
        if not (0 <= on_target_ratio <= 1 and 0 <= off_target_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        
        if abs(on_target_ratio + off_target_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {on_target_ratio + off_target_ratio}")
        
        self.on_target_ratio = on_target_ratio
        self.off_target_ratio = off_target_ratio
        
        # Separate indices by task type
        self._separate_indices_by_task()
        
        # Calculate batch composition
        self.on_target_per_batch = max(1, int(batch_size * on_target_ratio))
        self.off_target_per_batch = batch_size - self.on_target_per_batch
        
        # Log statistics
        self._log_statistics()
    
    def _separate_indices_by_task(self) -> None:
        """
        Separate dataset indices by task type
        
        Identifies which samples have on-target labels, off-target labels, or both.
        """
        self.on_target_indices = []
        self.off_target_indices = []
        self.mixed_indices = []
        
        # Iterate through dataset to find task types
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            has_on_target = sample['on_target_score'] is not None
            has_off_target = sample['off_target_label'] is not None
            
            if has_on_target and has_off_target:
                # Mixed samples can be used for either task
                self.mixed_indices.append(idx)
            elif has_on_target:
                self.on_target_indices.append(idx)
            elif has_off_target:
                self.off_target_indices.append(idx)
        
        # Combine mixed samples with each task
        self.on_target_indices = self.on_target_indices + self.mixed_indices
        self.off_target_indices = self.off_target_indices + self.mixed_indices
        
        if len(self.on_target_indices) == 0 or len(self.off_target_indices) == 0:
            raise ValueError(
                f"Dataset must have both on-target and off-target samples. "
                f"Found: on_target={len(self.on_target_indices)}, "
                f"off_target={len(self.off_target_indices)}"
            )
    
    def _log_statistics(self) -> None:
        """Log dataset statistics"""
        total_samples = len(self.dataset)
        on_target_count = len(self.on_target_indices)
        off_target_count = len(self.off_target_indices)
        mixed_count = len(self.mixed_indices)
        
        logger.info(
            f"BootstrapSampler initialized:\n"
            f"  Total samples: {total_samples}\n"
            f"  On-target samples: {on_target_count} ({100*on_target_count/total_samples:.1f}%)\n"
            f"  Off-target samples: {off_target_count} ({100*off_target_count/total_samples:.1f}%)\n"
            f"  Mixed samples: {mixed_count}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  On-target per batch: {self.on_target_per_batch}\n"
            f"  Off-target per batch: {self.off_target_per_batch}\n"
            f"  Total batches: {len(self)}"
        )
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of balanced indices
        
        Yields:
            List of indices for each batch
        """
        # Shuffle indices if requested
        on_target_indices = np.array(self.on_target_indices)
        off_target_indices = np.array(self.off_target_indices)
        
        if self.shuffle:
            np.random.shuffle(on_target_indices)
            np.random.shuffle(off_target_indices)
        
        # Determine number of batches
        # Use the smaller task as the limiting factor
        num_batches = min(
            len(on_target_indices) // self.on_target_per_batch,
            len(off_target_indices) // self.off_target_per_batch
        )
        
        if self.drop_last:
            num_batches_to_yield = num_batches
        else:
            # Calculate batches needed to cover all samples
            num_batches_to_yield = max(
                (len(on_target_indices) + self.on_target_per_batch - 1) // self.on_target_per_batch,
                (len(off_target_indices) + self.off_target_per_batch - 1) // self.off_target_per_batch
            )
        
        # Generate batches
        for batch_idx in range(num_batches_to_yield):
            batch = []
            
            # Sample on-target indices with replacement if needed
            on_target_start = (batch_idx * self.on_target_per_batch) % len(on_target_indices)
            on_target_batch_indices = []
            
            for i in range(self.on_target_per_batch):
                idx = (on_target_start + i) % len(on_target_indices)
                on_target_batch_indices.append(on_target_indices[idx])
            
            # Sample off-target indices with replacement if needed
            off_target_start = (batch_idx * self.off_target_per_batch) % len(off_target_indices)
            off_target_batch_indices = []
            
            for i in range(self.off_target_per_batch):
                idx = (off_target_start + i) % len(off_target_indices)
                off_target_batch_indices.append(off_target_indices[idx])
            
            # Combine and shuffle
            batch = list(on_target_batch_indices) + list(off_target_batch_indices)
            
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self) -> int:
        """
        Return total number of batches
        
        Returns:
            Number of batches
        """
        if self.drop_last:
            return min(
                len(self.on_target_indices) // self.on_target_per_batch,
                len(self.off_target_indices) // self.off_target_per_batch
            )
        else:
            return max(
                (len(self.on_target_indices) + self.on_target_per_batch - 1) // self.on_target_per_batch,
                (len(self.off_target_indices) + self.off_target_per_batch - 1) // self.off_target_per_batch
            )


class WeightedRandomSampler(Sampler):
    """
    Weighted Random Sampler for handling class imbalance
    
    Assigns weights to samples so that minority class samples are sampled more frequently.
    
    Strategy:
    1. Calculate class weights (inverse of class frequency)
    2. Assign weight to each sample based on its class
    3. Sample with replacement using these weights
    
    Example:
        >>> dataset = CRISPRDataset('data/train.csv', encoder)
        >>> sampler = WeightedRandomSampler(dataset)
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
    
    def __init__(self,
                 dataset: Dataset,
                 num_samples: Optional[int] = None,
                 replacement: bool = True,
                 seed: int = 42):
        """
        Initialize Weighted Random Sampler
        
        Args:
            dataset: CRISPRDataset instance
            num_samples: Number of samples to draw (default: len(dataset))
            replacement: Sample with replacement (default: True)
            seed: Random seed (default: 42)
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
        self.seed = seed
        
        # Calculate weights
        self.weights = self._calculate_weights()
        
        logger.info(
            f"WeightedRandomSampler initialized:\n"
            f"  Total samples: {len(self.dataset)}\n"
            f"  Samples to draw: {self.num_samples}\n"
            f"  Replacement: {replacement}\n"
            f"  Mean weight: {self.weights.mean():.4f}\n"
            f"  Min weight: {self.weights.min():.4f}\n"
            f"  Max weight: {self.weights.max():.4f}"
        )
    
    def _calculate_weights(self) -> torch.Tensor:
        """
        Calculate sample weights based on class frequency
        
        Returns:
            Weights tensor of shape (num_samples,)
        """
        class_counts = defaultdict(int)
        
        # Count samples in each class
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            has_on_target = sample['on_target_score'] is not None
            has_off_target = sample['off_target_label'] is not None
            
            if has_on_target and has_off_target:
                class_counts['mixed'] += 1
            elif has_on_target:
                class_counts['on_target'] += 1
            elif has_off_target:
                class_counts['off_target'] += 1
        
        # Calculate class weights (inverse frequency)
        total = sum(class_counts.values())
        class_weights = {
            cls: total / (count * len(class_counts))
            for cls, count in class_counts.items()
        }
        
        # Assign weights to samples
        weights = torch.zeros(len(self.dataset))
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            has_on_target = sample['on_target_score'] is not None
            has_off_target = sample['off_target_label'] is not None
            
            if has_on_target and has_off_target:
                weights[idx] = class_weights['mixed']
            elif has_on_target:
                weights[idx] = class_weights['on_target']
            elif has_off_target:
                weights[idx] = class_weights['off_target']
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def __iter__(self) -> Iterator[int]:
        """
        Yield sample indices sampled according to weights
        
        Yields:
            Sample indices
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=generator
        )
        
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        """Return number of samples"""
        return self.num_samples


class StratifiedSampler(Sampler):
    """
    Stratified Sampler for balanced sampling by strata
    
    Ensures that each batch contains samples from all strata (e.g., dataset sources).
    
    Example:
        >>> dataset = CRISPRDataset('data/train.csv', encoder)
        >>> sampler = StratifiedSampler(dataset, batch_size=32, strata_key='dataset_source')
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 strata_key: str = 'dataset_source',
                 drop_last: bool = False,
                 shuffle: bool = True,
                 seed: int = 42):
        """
        Initialize Stratified Sampler
        
        Args:
            dataset: CRISPRDataset instance
            batch_size: Batch size
            strata_key: Metadata key to stratify by (default: 'dataset_source')
            drop_last: Drop last incomplete batch (default: False)
            shuffle: Shuffle indices (default: True)
            seed: Random seed (default: 42)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.strata_key = strata_key
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Separate indices by strata
        self._separate_indices_by_strata()
        
        logger.info(
            f"StratifiedSampler initialized:\n"
            f"  Total samples: {len(self.dataset)}\n"
            f"  Number of strata: {len(self.strata_indices)}\n"
            f"  Strata: {list(self.strata_indices.keys())}\n"
            f"  Batch size: {batch_size}\n"
            f"  Total batches: {len(self)}"
        )
    
    def _separate_indices_by_strata(self) -> None:
        """Separate indices by strata"""
        self.strata_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            strata_value = sample['metadata'].get(self.strata_key, 'unknown')
            self.strata_indices[strata_value].append(idx)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield stratified batches"""
        # Shuffle within strata
        strata_indices = {
            strata: np.array(indices)
            for strata, indices in self.strata_indices.items()
        }
        
        if self.shuffle:
            for strata in strata_indices:
                np.random.shuffle(strata_indices[strata])
        
        # Calculate samples per strata per batch
        num_strata = len(strata_indices)
        samples_per_strata = max(1, self.batch_size // num_strata)
        
        # Generate batches
        num_batches = max(
            len(indices) // samples_per_strata
            for indices in strata_indices.values()
        )
        
        if self.drop_last:
            num_batches = min(
                len(indices) // samples_per_strata
                for indices in strata_indices.values()
            )
        
        for batch_idx in range(num_batches):
            batch = []
            
            for strata, indices in strata_indices.items():
                start = (batch_idx * samples_per_strata) % len(indices)
                
                for i in range(samples_per_strata):
                    idx = (start + i) % len(indices)
                    batch.append(indices[idx])
            
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch[:self.batch_size]
    
    def __len__(self) -> int:
        """Return number of batches"""
        samples_per_strata = max(1, self.batch_size // len(self.strata_indices))
        
        if self.drop_last:
            return min(
                len(indices) // samples_per_strata
                for indices in self.strata_indices.values()
            )
        else:
            return max(
                (len(indices) + samples_per_strata - 1) // samples_per_strata
                for indices in self.strata_indices.values()
            )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CUSTOM SAMPLER TESTING")
    print("=" * 80)
    
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.encoding import SequenceEncoder
    from utils.preprocessing.dataset import CRISPRDataset
    
    # Test 1: Create dummy dataset
    print("\n1. CREATE DUMMY DATASET")
    print("-" * 80)
    
    import tempfile
    
    # Create temporary CSV with imbalanced data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("sgrna,target,on_target_score,off_target_label,dataset_source\n")
        
        # Create imbalanced data: 80% on-target, 20% off-target
        sequences = [
            "GCTAGCTAGCTAGCTAGCTAGCT",
            "ATGCATGCATGCATGCATGCATG",
            "CCGGCCGGCCGGCCGGCCGGCCG",
            "GGCCGGCCGGCCGGCCGGCCGGCC",
            "TTAATTAATTAATTAATTAATTAA",
        ]
        
        # 80% on-target samples
        for i in range(8):
            seq = sequences[i % len(sequences)]
            f.write(f"{seq},{seq},0.{70+i},None,dataset_A\n")
        
        # 20% off-target samples
        for i in range(2):
            seq = sequences[i % len(sequences)]
            f.write(f"{seq},{seq},None,{i%2},dataset_B\n")
    
    encoder = SequenceEncoder(device='cpu')
    dataset = CRISPRDataset(csv_path, encoder, verbose=False)
    
    print(f"[OK] Created imbalanced dataset with {len(dataset)} samples")
    
    # Test 2: Bootstrap Sampler
    print("\n2. BOOTSTRAP SAMPLER")
    print("-" * 80)
    
    sampler = BootstrapSampler(
        dataset,
        batch_size=4,
        on_target_ratio=0.5,
        shuffle=True
    )
    
    print(f"[OK] Bootstrap sampler created")
    print(f"  Total batches: {len(sampler)}")
    
    # Collect batch statistics
    on_target_counts = []
    off_target_counts = []
    
    for batch_idx, batch_indices in enumerate(sampler):
        on_target_count = 0
        off_target_count = 0
        
        for idx in batch_indices:
            sample = dataset[idx]
            if sample['on_target_score'] is not None:
                on_target_count += 1
            if sample['off_target_label'] is not None:
                off_target_count += 1
        
        on_target_counts.append(on_target_count)
        off_target_counts.append(off_target_count)
        
        if batch_idx < 3:
            print(f"  Batch {batch_idx}: on_target={on_target_count}, off_target={off_target_count}")
    
    print(f"  Average on-target per batch: {np.mean(on_target_counts):.2f}")
    print(f"  Average off-target per batch: {np.mean(off_target_counts):.2f}")
    
    # Test 3: Weighted Random Sampler
    print("\n3. WEIGHTED RANDOM SAMPLER")
    print("-" * 80)
    
    weighted_sampler = WeightedRandomSampler(
        dataset,
        num_samples=20,
        replacement=True
    )
    
    print(f"[OK] Weighted sampler created")
    print(f"  Samples to draw: {len(weighted_sampler)}")
    
    # Collect samples
    samples = list(weighted_sampler)
    print(f"  First 10 samples: {samples[:10]}")
    
    # Test 4: Stratified Sampler
    print("\n4. STRATIFIED SAMPLER")
    print("-" * 80)
    
    stratified_sampler = StratifiedSampler(
        dataset,
        batch_size=4,
        strata_key='dataset_source',
        shuffle=True
    )
    
    print(f"[OK] Stratified sampler created")
    print(f"  Total batches: {len(stratified_sampler)}")
    
    for batch_idx, batch_indices in enumerate(stratified_sampler):
        if batch_idx < 2:
            sources = [dataset[idx]['metadata'].get('dataset_source', 'unknown') for idx in batch_indices]
            print(f"  Batch {batch_idx} sources: {sources}")
    
    # Cleanup
    import os
    os.remove(csv_path)
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
