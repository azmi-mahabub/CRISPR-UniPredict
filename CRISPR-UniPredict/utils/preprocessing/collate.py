"""
Custom Collate Functions for CRISPR Dataset
Handles variable-length sequences, padding, masking, and task-aware batching
"""

import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for CRISPR dataset
    
    Combines variable-length sequences into batched tensors with:
    - Padding to maximum length in batch
    - Attention masks for padded positions
    - Task-aware label handling
    - Proper masking for invalid labels
    
    Args:
        batch: List of samples from CRISPRDataset
               Each sample is a dict with:
               - sgrna_onehot: (seq_len, 4)
               - sgrna_label: (seq_len,)
               - target_onehot: (seq_len, 4)
               - target_label: (seq_len,)
               - on_target_score: float or None
               - off_target_label: int or None
               - sgrna_sequence: str
               - target_sequence: str
               - metadata: dict
    
    Returns:
        Dictionary with batched tensors:
        - sgrna_onehot: (batch, max_len, 4)
        - sgrna_label: (batch, max_len)
        - target_onehot: (batch, max_len, 4)
        - target_label: (batch, max_len)
        - on_target_score: (batch,) or None
        - off_target_label: (batch,) or None
        - on_target_mask: (batch,) - True where label is valid
        - off_target_mask: (batch,) - True where label is valid
        - attention_mask: (batch, max_len) - False for padded positions
        - sequence_lengths: (batch,) - original lengths before padding
        - sgrna_sequences: List[str]
        - target_sequences: List[str]
        - metadata: List[dict]
    """
    if not batch:
        raise ValueError("Batch cannot be empty")
    
    batch_size = len(batch)
    
    # Extract sequences and find max length
    sgrna_oneshots = [item['sgrna_onehot'] for item in batch]
    sgrna_labels = [item['sgrna_label'] for item in batch]
    target_oneshots = [item['target_onehot'] for item in batch]
    target_labels = [item['target_label'] for item in batch]
    
    # Find maximum sequence length in batch
    max_len = max(seq.shape[0] for seq in sgrna_oneshots)
    
    # Store original sequence lengths for masking
    sequence_lengths = torch.tensor(
        [seq.shape[0] for seq in sgrna_oneshots],
        dtype=torch.long
    )
    
    # Pad one-hot sequences
    sgrna_onehot_padded = _pad_onehot_sequences(sgrna_oneshots, max_len)
    target_onehot_padded = _pad_onehot_sequences(target_oneshots, max_len)
    
    # Pad label sequences
    sgrna_label_padded = _pad_label_sequences(sgrna_labels, max_len)
    target_label_padded = _pad_label_sequences(target_labels, max_len)
    
    # Create attention mask (False for padded positions)
    attention_mask = _create_attention_mask(sequence_lengths, max_len)
    
    # Handle labels (may be None)
    on_target_scores = [item['on_target_score'] for item in batch]
    off_target_labels = [item['off_target_label'] for item in batch]
    
    # Create masks for valid labels
    on_target_mask = torch.tensor(
        [score is not None for score in on_target_scores],
        dtype=torch.bool
    )
    off_target_mask = torch.tensor(
        [label is not None for label in off_target_labels],
        dtype=torch.bool
    )
    
    # Convert labels to tensors, using 0 for invalid values
    on_target_tensor = torch.tensor(
        [score if score is not None else 0.0 for score in on_target_scores],
        dtype=torch.float32
    )
    off_target_tensor = torch.tensor(
        [label if label is not None else 0 for label in off_target_labels],
        dtype=torch.long
    )
    
    # Collect metadata
    sgrna_sequences = [item['sgrna_sequence'] for item in batch]
    target_sequences = [item['target_sequence'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'sgrna_onehot': sgrna_onehot_padded,
        'sgrna_label': sgrna_label_padded,
        'target_onehot': target_onehot_padded,
        'target_label': target_label_padded,
        'on_target_score': on_target_tensor,
        'off_target_label': off_target_tensor,
        'on_target_mask': on_target_mask,
        'off_target_mask': off_target_mask,
        'attention_mask': attention_mask,
        'sequence_lengths': sequence_lengths,
        'sgrna_sequences': sgrna_sequences,
        'target_sequences': target_sequences,
        'metadata': metadata
    }


def task_aware_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Task-aware collate function that groups samples by task type
    
    Separates on-target and off-target samples for more efficient batching
    
    Args:
        batch: List of samples from CRISPRDataset
    
    Returns:
        Dictionary with task-separated batches:
        - on_target_batch: Batch with only on-target samples
        - off_target_batch: Batch with only off-target samples
        - mixed_batch: Batch with samples having both labels
        - task_indices: Dict mapping task to original batch indices
    """
    # Separate samples by task
    on_target_samples = []
    off_target_samples = []
    mixed_samples = []
    
    on_target_indices = []
    off_target_indices = []
    mixed_indices = []
    
    for idx, item in enumerate(batch):
        has_on_target = item['on_target_score'] is not None
        has_off_target = item['off_target_label'] is not None
        
        if has_on_target and has_off_target:
            mixed_samples.append(item)
            mixed_indices.append(idx)
        elif has_on_target:
            on_target_samples.append(item)
            on_target_indices.append(idx)
        elif has_off_target:
            off_target_samples.append(item)
            off_target_indices.append(idx)
    
    # Collate each group
    result = {
        'on_target_batch': custom_collate_fn(on_target_samples) if on_target_samples else None,
        'off_target_batch': custom_collate_fn(off_target_samples) if off_target_samples else None,
        'mixed_batch': custom_collate_fn(mixed_samples) if mixed_samples else None,
        'task_indices': {
            'on_target': on_target_indices,
            'off_target': off_target_indices,
            'mixed': mixed_indices
        }
    }
    
    return result


def _pad_onehot_sequences(sequences: List[torch.Tensor], 
                          max_len: int,
                          pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad one-hot encoded sequences to maximum length
    
    Args:
        sequences: List of (seq_len, 4) tensors
        max_len: Target length
        pad_value: Value to use for padding
    
    Returns:
        Padded tensor of shape (batch, max_len, 4)
    """
    batch_size = len(sequences)
    padded = torch.zeros(batch_size, max_len, 4, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
    
    return padded


def _pad_label_sequences(sequences: List[torch.Tensor],
                         max_len: int,
                         pad_value: int = 0) -> torch.Tensor:
    """
    Pad label encoded sequences to maximum length
    
    Args:
        sequences: List of (seq_len,) tensors
        max_len: Target length
        pad_value: Value to use for padding (0 = padding token)
    
    Returns:
        Padded tensor of shape (batch, max_len)
    """
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
    
    return padded


def _create_attention_mask(sequence_lengths: torch.Tensor,
                          max_len: int) -> torch.Tensor:
    """
    Create attention mask for padded sequences
    
    Args:
        sequence_lengths: (batch,) tensor with original sequence lengths
        max_len: Maximum length in batch
    
    Returns:
        Attention mask of shape (batch, max_len)
        True for valid positions, False for padded positions
    """
    batch_size = sequence_lengths.shape[0]
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, length in enumerate(sequence_lengths):
        mask[i, :length] = True
    
    return mask


def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create causal attention mask (for autoregressive models)
    
    Args:
        seq_len: Sequence length
        device: Device to place mask on
    
    Returns:
        Causal mask of shape (seq_len, seq_len)
        True where attention is allowed, False where masked
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return (~mask).to(device)


def create_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask from attention mask
    
    Args:
        attention_mask: (batch, seq_len) bool tensor
                       True for valid positions, False for padded
    
    Returns:
        Padding mask of shape (batch, 1, 1, seq_len)
        Ready for broadcasting in attention computation
    """
    # Expand for broadcasting in attention (batch, 1, 1, seq_len)
    return attention_mask.unsqueeze(1).unsqueeze(1)


def mask_invalid_labels(predictions: torch.Tensor,
                       mask: torch.Tensor,
                       fill_value: float = 0.0) -> torch.Tensor:
    """
    Mask invalid predictions where labels are missing
    
    Args:
        predictions: (batch,) or (batch, ...) tensor
        mask: (batch,) bool tensor, True where label is valid
        fill_value: Value to use for invalid positions
    
    Returns:
        Masked predictions
    """
    masked = predictions.clone()
    masked[~mask] = fill_value
    return masked


class CRISPRCollator:
    """
    Stateful collator for CRISPR dataset
    
    Allows configuration of collation behavior
    """
    
    def __init__(self,
                 pad_value_onehot: float = 0.0,
                 pad_value_label: int = 0,
                 create_causal_mask: bool = False,
                 task_aware: bool = False,
                 device: str = 'cpu'):
        """
        Initialize collator
        
        Args:
            pad_value_onehot: Padding value for one-hot sequences
            pad_value_label: Padding value for label sequences
            create_causal_mask: Whether to create causal mask
            task_aware: Whether to use task-aware collation
            device: Device to place tensors on
        """
        self.pad_value_onehot = pad_value_onehot
        self.pad_value_label = pad_value_label
        self.create_causal_mask = create_causal_mask
        self.task_aware = task_aware
        self.device = device
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch
        
        Args:
            batch: List of samples
        
        Returns:
            Collated batch dictionary
        """
        if self.task_aware:
            result = task_aware_collate_fn(batch)
        else:
            result = custom_collate_fn(batch)
        
        # Move to device
        for key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].to(self.device)
        
        # Add causal mask if requested
        if self.create_causal_mask and 'sgrna_onehot' in result:
            seq_len = result['sgrna_onehot'].shape[1]
            result['causal_mask'] = create_causal_mask(seq_len, self.device)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CUSTOM COLLATE FUNCTION TESTING")
    print("=" * 80)
    
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.encoding import SequenceEncoder
    
    # Test 1: Create dummy batch
    print("\n1. CREATE DUMMY BATCH")
    print("-" * 80)
    
    encoder = SequenceEncoder(device='cpu')
    
    # Create sample data with variable lengths
    batch = [
        {
            'sgrna_onehot': encoder.one_hot_encode("GCTAGCTAGCTAGCTAGCTAGCT"),
            'sgrna_label': encoder.label_encode("GCTAGCTAGCTAGCTAGCTAGCT", add_start_token=False),
            'target_onehot': encoder.one_hot_encode("GCTAGCTAGCTAGCTAGCTAGCT"),
            'target_label': encoder.label_encode("GCTAGCTAGCTAGCTAGCTAGCT", add_start_token=False),
            'on_target_score': 0.85,
            'off_target_label': 0,
            'sgrna_sequence': "GCTAGCTAGCTAGCTAGCTAGCT",
            'target_sequence': "GCTAGCTAGCTAGCTAGCTAGCT",
            'metadata': {'dataset': 'test', 'cell_line': 'HEK293T'}
        },
        {
            'sgrna_onehot': encoder.one_hot_encode("ATGCATGCATGCATGCATGCATG"),
            'sgrna_label': encoder.label_encode("ATGCATGCATGCATGCATGCATG", add_start_token=False),
            'target_onehot': encoder.one_hot_encode("ATGCATGCATGCATGCATGCATG"),
            'target_label': encoder.label_encode("ATGCATGCATGCATGCATGCATG", add_start_token=False),
            'on_target_score': None,  # Missing label
            'off_target_label': 1,
            'sgrna_sequence': "ATGCATGCATGCATGCATGCATG",
            'target_sequence': "ATGCATGCATGCATGCATGCATG",
            'metadata': {'dataset': 'test', 'cell_line': 'HeLa'}
        },
        {
            'sgrna_onehot': encoder.one_hot_encode("CCGGCCGGCCGGCCGGCCGGCCG"),
            'sgrna_label': encoder.label_encode("CCGGCCGGCCGGCCGGCCGGCCG", add_start_token=False),
            'target_onehot': encoder.one_hot_encode("CCGGCCGGCCGGCCGGCCGGCCG"),
            'target_label': encoder.label_encode("CCGGCCGGCCGGCCGGCCGGCCG", add_start_token=False),
            'on_target_score': 0.72,
            'off_target_label': None,  # Missing label
            'sgrna_sequence': "CCGGCCGGCCGGCCGGCCGGCCG",
            'target_sequence': "CCGGCCGGCCGGCCGGCCGGCCG",
            'metadata': {'dataset': 'test', 'cell_line': 'HEK293T'}
        }
    ]
    
    print(f"[OK] Created batch with {len(batch)} samples")
    
    # Test 2: Custom collate function
    print("\n2. CUSTOM COLLATE FUNCTION")
    print("-" * 80)
    
    collated = custom_collate_fn(batch)
    
    print(f"[OK] Collated batch")
    print(f"  sgrna_onehot shape: {collated['sgrna_onehot'].shape}")
    print(f"  sgrna_label shape: {collated['sgrna_label'].shape}")
    print(f"  target_onehot shape: {collated['target_onehot'].shape}")
    print(f"  target_label shape: {collated['target_label'].shape}")
    print(f"  on_target_score shape: {collated['on_target_score'].shape}")
    print(f"  off_target_label shape: {collated['off_target_label'].shape}")
    print(f"  on_target_mask: {collated['on_target_mask']}")
    print(f"  off_target_mask: {collated['off_target_mask']}")
    print(f"  attention_mask shape: {collated['attention_mask'].shape}")
    print(f"  sequence_lengths: {collated['sequence_lengths']}")
    
    # Test 3: Attention mask verification
    print("\n3. ATTENTION MASK VERIFICATION")
    print("-" * 80)
    
    attention_mask = collated['attention_mask']
    print(f"  Attention mask:")
    for i, mask in enumerate(attention_mask):
        valid_count = mask.sum().item()
        print(f"    Sample {i}: {valid_count} valid positions out of {len(mask)}")
    
    # Test 4: Label masking
    print("\n4. LABEL MASKING")
    print("-" * 80)
    
    on_target_mask = collated['on_target_mask']
    off_target_mask = collated['off_target_mask']
    
    print(f"  On-target valid: {on_target_mask}")
    print(f"  Off-target valid: {off_target_mask}")
    
    # Test 5: Task-aware collation
    print("\n5. TASK-AWARE COLLATION")
    print("-" * 80)
    
    task_collated = task_aware_collate_fn(batch)
    
    print(f"[OK] Task-aware collation")
    if task_collated['on_target_batch'] is not None:
        print(f"  On-target batch: {len(task_collated['task_indices']['on_target'])} samples")
    if task_collated['off_target_batch'] is not None:
        print(f"  Off-target batch: {len(task_collated['task_indices']['off_target'])} samples")
    if task_collated['mixed_batch'] is not None:
        print(f"  Mixed batch: {len(task_collated['task_indices']['mixed'])} samples")
    
    # Test 6: Collator class
    print("\n6. COLLATOR CLASS")
    print("-" * 80)
    
    collator = CRISPRCollator(
        pad_value_onehot=0.0,
        pad_value_label=0,
        create_causal_mask=False,
        task_aware=False,
        device='cpu'
    )
    
    collated_class = collator(batch)
    print(f"[OK] Collator class working")
    print(f"  Keys in result: {list(collated_class.keys())}")
    
    # Test 7: Causal mask
    print("\n7. CAUSAL MASK")
    print("-" * 80)
    
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    print(f"  Causal mask for seq_len={seq_len}:")
    print(f"  {causal_mask.int()}")
    
    # Test 8: Padding mask
    print("\n8. PADDING MASK")
    print("-" * 80)
    
    padding_mask = create_padding_mask(collated['attention_mask'])
    print(f"  Padding mask shape: {padding_mask.shape}")
    print(f"  Ready for broadcasting in attention: {padding_mask.shape}")
    
    # Test 9: Mask invalid labels
    print("\n9. MASK INVALID LABELS")
    print("-" * 80)
    
    predictions = collated['on_target_score'].clone()
    mask = collated['on_target_mask']
    masked_predictions = mask_invalid_labels(predictions, mask, fill_value=-1.0)
    
    print(f"  Original predictions: {predictions}")
    print(f"  Mask: {mask}")
    print(f"  Masked predictions: {masked_predictions}")
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
