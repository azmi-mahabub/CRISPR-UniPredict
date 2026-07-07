"""
Fast DataLoader for CRISPR-UniPredict - Optimized for Windows
Loads data directly without slow caching mechanism
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

from models.encoding import SequenceEncoder

class FastCRISPRDataset(Dataset):
    """Fast dataset that encodes on-the-fly without caching"""
    
    def __init__(self, csv_path: str, encoder, max_samples=None, verbose=True):
        self.csv_path = Path(csv_path)
        self.encoder = encoder
        self.verbose = verbose
        
        if verbose:
            logger.info(f"Loading {csv_path}")
        
        self.df = pd.read_csv(self.csv_path, nrows=max_samples)
        
        # Normalize column names
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Map column names
        if 'sgrna_sequence' in self.df.columns and 'sgrna' not in self.df.columns:
            self.df.rename(columns={'sgrna_sequence': 'sgrna'}, inplace=True)
        if 'target_sequence' in self.df.columns and 'target' not in self.df.columns:
            self.df.rename(columns={'target_sequence': 'target'}, inplace=True)

        # On-target-only rows often have no protospacer target in the CSV; duplicate sgRNA so rows are not dropped.
        if 'sgrna' in self.df.columns and 'target' in self.df.columns:
            miss = self.df['target'].isna() & self.df['sgrna'].notna()
            if miss.any():
                self.df.loc[miss, 'target'] = self.df.loc[miss, 'sgrna']

        # Filter valid sequences - must be 22-24 bp with valid nucleotides
        valid_nucleotides = set('ACGTU')
        def is_valid(seq):
            if not isinstance(seq, str):
                return False
            # Must be 22-24 bp and contain only valid nucleotides
            return 22 <= len(seq) <= 24 and all(c.upper() in valid_nucleotides for c in seq)
        
        mask = self.df['sgrna'].apply(is_valid) & self.df['target'].apply(is_valid)
        self.df = self.df[mask].reset_index(drop=True)
        
        if verbose:
            logger.info(f"Loaded {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        sgrna = row['sgrna']
        target = row['target']
        
        # Pad sequences to 23 bp (right-pad with 'A')
        max_len = 23
        sgrna = sgrna.ljust(max_len, 'A')[:max_len]
        target = target.ljust(max_len, 'A')[:max_len]
        
        # Encode on-the-fly
        sgrna_onehot = self.encoder.one_hot_encode(sgrna)
        sgrna_label = self.encoder.label_encode(sgrna, add_start_token=False)
        target_onehot = self.encoder.one_hot_encode(target)
        target_label = self.encoder.label_encode(target, add_start_token=False)
        
        # Get labels
        on_target = row.get('on_target_score', None)
        off_target = row.get('off_target_label', None)
        
        # Check if label is valid (not NaN)
        on_target_valid = pd.notna(on_target)
        off_target_valid = pd.notna(off_target)
        
        # 0.0 is a placeholder — always gated by on_target_mask / off_target_mask in loss
        on_target = float(on_target) if on_target_valid else 0.0
        off_target = float(off_target) if off_target_valid else 0.0
        
        # Create masks for valid labels (bool type) - mask is True if label is valid
        on_target_mask = on_target_valid
        off_target_mask = off_target_valid
        
        return {
            'sgrna_onehot': sgrna_onehot,
            'sgrna_label': sgrna_label,
            'target_onehot': target_onehot,
            'target_label': target_label,
            'sgrna_str': sgrna,
            'target_str': target,
            'on_target_score': torch.tensor(on_target, dtype=torch.float32),
            'off_target_label': torch.tensor(off_target, dtype=torch.float32),
            'on_target_mask': torch.tensor(on_target_mask, dtype=torch.bool),
            'off_target_mask': torch.tensor(off_target_mask, dtype=torch.bool),
        }


def build_stratified_debug_indices(
    dataset: "FastCRISPRDataset",
    max_samples: int = 256,
    seed: int = 42,
) -> List[int]:
    """
    Pick row indices for --debug runs so batches include both on-target and off-target
    labeled rows when the CSV contains them. Avoids the first-N-rows trap (often all off-target).
    """
    df = dataset.df
    n = len(df)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    half = max(1, max_samples // 2)

    on_col = "on_target_score" if "on_target_score" in df.columns else None
    off_col = "off_target_label" if "off_target_label" in df.columns else None

    picks: List[int] = []

    if on_col is not None:
        on_idx = df.index[df[on_col].notna()].tolist()
        take = min(half, len(on_idx))
        if take > 0:
            if len(on_idx) == take:
                chosen = on_idx
            else:
                pos = rng.choice(len(on_idx), size=take, replace=False)
                chosen = [on_idx[i] for i in pos]
            picks.extend(int(i) for i in chosen)

    if off_col is not None:
        off_idx = df.index[df[off_col].notna()].tolist()
        take = min(half, len(off_idx))
        if take > 0:
            if len(off_idx) == take:
                chosen = off_idx
            else:
                pos = rng.choice(len(off_idx), size=take, replace=False)
                chosen = [off_idx[i] for i in pos]
            picks.extend(int(i) for i in chosen)

    seen = set()
    ordered: List[int] = []
    for i in picks:
        if i not in seen:
            seen.add(i)
            ordered.append(i)

    for i in range(n):
        if len(ordered) >= max_samples:
            break
        if i not in seen:
            ordered.append(i)
            seen.add(i)

    return ordered[:max_samples]


def collate_fn(batch):
    """Collate function for batching"""
    sgrna_onehot = torch.stack([item['sgrna_onehot'] for item in batch])
    sgrna_label = torch.stack([item['sgrna_label'] for item in batch])
    on_target = torch.stack([item['on_target_score'] for item in batch])
    off_target = torch.stack([item['off_target_label'] for item in batch])
    on_target_mask = torch.stack([item['on_target_mask'] for item in batch])
    off_target_mask = torch.stack([item['off_target_mask'] for item in batch])
    sgrna_strs = [item['sgrna_str'] for item in batch]
    target_strs = [item['target_str'] for item in batch]
    
    return {
        'sgrna_onehot': sgrna_onehot,
        'sgrna_label': sgrna_label,
        'sgrna_strs': sgrna_strs,
        'target_strs': target_strs,
        'on_target_score': on_target,
        'off_target_label': off_target,
        'on_target_mask': on_target_mask,
        'off_target_mask': off_target_mask,
    }

def create_fast_dataloaders(config, device='cuda') -> Dict[str, DataLoader]:
    """Create fast dataloaders without caching"""
    
    encoder = SequenceEncoder(device=device)
    
    # Load datasets
    train_dataset = FastCRISPRDataset(
        config.data.train_path,
        encoder,
        verbose=True
    )
    
    val_dataset = FastCRISPRDataset(
        config.data.val_path,
        encoder,
        verbose=True
    )
    
    test_dataset = FastCRISPRDataset(
        config.data.test_path,
        encoder,
        verbose=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
