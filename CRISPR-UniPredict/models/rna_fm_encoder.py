import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from pathlib import Path
import warnings
import sys

# Add RNA-FM to path if available
try:
    import fm
    from fm.pretrained import rna_fm_t12
    RNA_FM_AVAILABLE = True
except ImportError:
    RNA_FM_AVAILABLE = False
    warnings.warn("RNA-FM not available in Python path. Install from RNA-FM-main directory.")


class RNAFMEncoder(nn.Module):
    """
    RNA-FM Encoder for sequence representation learning
    
    Wraps the pretrained RNA-FM model (12 transformer blocks) to generate
    contextual embeddings for RNA/DNA sequences. Supports fine-tuning of
    the last few layers while keeping earlier layers frozen.
    
    Architecture:
    - Pretrained RNA-FM with 12 transformer blocks
    - Token embedding dimension: 640
    - Optional layer freezing for efficient fine-tuning
    - Special tokens: [CLS] for classification, [SEP] for separation
    - Handles variable-length sequences with padding
    
    Input format:
    - Single sequence: [CLS] sequence [SEP]
    - Pair sequences: [CLS] sgRNA [SEP] target [SEP]
    
    Output:
    - [CLS] token embedding for downstream classification tasks
    - Full sequence embeddings for detailed analysis
    """
    
    def __init__(self, model_path: str = 'models/pretrained/rna_fm_t12.pt',
                 freeze_layers: bool = True, num_unfreeze_layers: int = 2,
                 device: str = 'cpu'):
        """
        Initialize RNA-FM Encoder
        
        Args:
            model_path: Path to pretrained RNA-FM checkpoint
            freeze_layers: Whether to freeze initial layers (default: True)
            num_unfreeze_layers: Number of final layers to unfreeze for fine-tuning (default: 2)
            device: Device to use ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If model checkpoint not found
            ImportError: If RNA-FM not available
        """
        super(RNAFMEncoder, self).__init__()
        
        if not RNA_FM_AVAILABLE:
            raise ImportError(
                "RNA-FM not available. Please install from RNA-FM-main directory or "
                "add it to PYTHONPATH"
            )
        
        self.model_path = Path(model_path)
        self.device = device
        self.freeze_layers = freeze_layers
        self.num_unfreeze_layers = num_unfreeze_layers
        
        # Load pretrained model and alphabet
        self.model, self.alphabet = self._load_pretrained_model()
        self.model = self.model.to(device)
        
        # Get model configuration
        self.embed_dim = self.model.args.embed_dim
        self.num_layers = self.model.args.layers
        
        # Freeze layers if specified
        if freeze_layers:
            self._freeze_layers(num_unfreeze_layers)
        
        # Get special token indices
        self.cls_idx = self.alphabet.cls_idx
        self.sep_idx = self.alphabet.get_idx("<sep>")
        self.pad_idx = self.alphabet.padding_idx
        self.unk_idx = self.alphabet.unk_idx
    
    def _load_pretrained_model(self) -> Tuple:
        """
        Load pretrained RNA-FM model and alphabet
        
        Returns:
            Tuple of (model, alphabet)
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        from fm.pretrained import rna_fm_t12

        try:
            if self.model_path.exists():
                try:
                    model, alphabet = rna_fm_t12(str(self.model_path))
                except (OSError, RuntimeError, FileNotFoundError):
                    # Incomplete local copy (e.g. missing *-contact-regression.pt) — use hub
                    model, alphabet = rna_fm_t12(None)
            else:
                model, alphabet = rna_fm_t12(None)
            return model, alphabet
        except Exception as e:
            raise RuntimeError(f"Failed to load RNA-FM model: {str(e)}")
    
    def _freeze_layers(self, num_unfreeze: int = 2):
        """
        Freeze initial layers and unfreeze final layers for fine-tuning
        
        Args:
            num_unfreeze: Number of final layers to keep unfrozen
        """
        # Freeze embedding layer
        for param in self.model.embed_tokens.parameters():
            param.requires_grad = False
        
        for param in self.model.embed_positions.parameters():
            param.requires_grad = False
        
        # Freeze initial transformer layers
        num_freeze = self.num_layers - num_unfreeze
        for i in range(num_freeze):
            for param in self.model.layers[i].parameters():
                param.requires_grad = False
        
        # Keep final layers unfrozen for fine-tuning
        for i in range(num_freeze, self.num_layers):
            for param in self.model.layers[i].parameters():
                param.requires_grad = True
    
    def tokenize(self, sequence: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Tokenize RNA/DNA sequence
        
        Converts sequence string to token indices. Optionally adds [CLS] at start
        and [SEP] at end.
        
        Args:
            sequence: RNA/DNA sequence string (A, C, G, U/T)
            add_special_tokens: Whether to add [CLS] and [SEP] tokens (default: True)
            
        Returns:
            Token indices as 1D tensor
            
        Raises:
            ValueError: If sequence contains invalid nucleotides
        """
        # Normalize sequence
        sequence = sequence.upper()
        sequence = sequence.replace('T', 'U')  # DNA→RNA: T→U for RNA-FM alphabet

        # Validate sequence
        valid_nucleotides = set('ACGU')
        invalid_chars = set(sequence) - valid_nucleotides
        if invalid_chars:
            raise ValueError(f"Invalid nucleotides in sequence: {invalid_chars}")
        
        # Tokenize
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.cls_idx)
        
        # Convert each nucleotide to token index
        for nuc in sequence:
            token_idx = self.alphabet.get_idx(nuc)
            tokens.append(token_idx)
        
        if add_special_tokens:
            tokens.append(self.sep_idx)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def encode_pair(self, sgrna_seq: str, target_seq: str,
                    max_length: Optional[int] = None,
                    return_full_embeddings: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Encode sgRNA and target sequence pair
        
        Formats sequences as: [CLS] sgRNA [SEP] target [SEP]
        Returns [CLS] token embedding for classification by default.
        
        Args:
            sgrna_seq: sgRNA sequence string
            target_seq: Target sequence string
            max_length: Maximum sequence length (default: None, no truncation)
            return_full_embeddings: Whether to return full embeddings (default: False)
                                   If False, returns only [CLS] token embedding
                                   
        Returns:
            If return_full_embeddings=False:
                Tensor of shape (embed_dim,) - [CLS] token embedding
            If return_full_embeddings=True:
                Tuple of (cls_embedding, full_embeddings)
                - cls_embedding: (embed_dim,)
                - full_embeddings: (seq_len, embed_dim)
                
        Raises:
            ValueError: If sequences contain invalid nucleotides
        """
        # Tokenize sequences
        sgrna_tokens = self.tokenize(sgrna_seq, add_special_tokens=False)
        target_tokens = self.tokenize(target_seq, add_special_tokens=False)
        
        # Combine: [CLS] sgRNA [SEP] target [SEP]
        tokens = torch.cat([
            torch.tensor([self.cls_idx], dtype=torch.long),
            sgrna_tokens,
            torch.tensor([self.sep_idx], dtype=torch.long),
            target_tokens,
            torch.tensor([self.sep_idx], dtype=torch.long)
        ])
        
        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        dev = next(self.model.parameters()).device
        tokens = tokens.unsqueeze(0).to(dev)
        
        with torch.set_grad_enabled(self.training):
            result = self.model(tokens, repr_layers=[self.num_layers])
        
        # Extract embeddings from final layer
        embeddings = result['representations'][self.num_layers]  # (batch, seq_len, embed_dim)
        
        # Get [CLS] token embedding (first token)
        cls_embedding = embeddings[0, 0, :]  # (embed_dim,)
        
        if return_full_embeddings:
            full_embeddings = embeddings[0]  # (seq_len, embed_dim)
            return cls_embedding, full_embeddings
        else:
            return cls_embedding
    
    def forward(self, sgrna: str, target: str,
                max_length: Optional[int] = None,
                return_full_embeddings: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through RNA-FM encoder
        
        Encodes sgRNA and target sequence pair and returns contextual embeddings.
        
        Args:
            sgrna: sgRNA sequence string
            target: Target sequence string
            max_length: Maximum sequence length
            return_full_embeddings: Whether to return full sequence embeddings
            
        Returns:
            [CLS] token embedding or tuple of (cls_embedding, full_embeddings)
        """
        return self.encode_pair(sgrna, target, max_length, return_full_embeddings)
    
    def encode_single(self, sequence: str,
                      max_length: Optional[int] = None,
                      return_full_embeddings: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Encode single sequence
        
        Formats sequence as: [CLS] sequence [SEP]
        
        Args:
            sequence: RNA/DNA sequence string
            max_length: Maximum sequence length
            return_full_embeddings: Whether to return full embeddings
            
        Returns:
            [CLS] token embedding or tuple of (cls_embedding, full_embeddings)
        """
        # Tokenize
        tokens = self.tokenize(sequence, add_special_tokens=True)
        
        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        dev = next(self.model.parameters()).device
        tokens = tokens.unsqueeze(0).to(dev)
        
        with torch.set_grad_enabled(self.training):
            result = self.model(tokens, repr_layers=[self.num_layers])
        
        # Extract embeddings from final layer
        embeddings = result['representations'][self.num_layers]  # (batch, seq_len, embed_dim)
        
        # Get [CLS] token embedding
        cls_embedding = embeddings[0, 0, :]  # (embed_dim,)
        
        if return_full_embeddings:
            full_embeddings = embeddings[0]  # (seq_len, embed_dim)
            return cls_embedding, full_embeddings
        else:
            return cls_embedding
    
    def encode_batch(self, sequences: list, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode batch of sequences
        
        Args:
            sequences: List of sequence strings
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape (batch_size, embed_dim) - [CLS] embeddings
        """
        batch_embeddings = []
        
        for seq in sequences:
            embedding = self.encode_single(seq, max_length, return_full_embeddings=False)
            batch_embeddings.append(embedding)
        
        return torch.stack(batch_embeddings)
    
    def encode_batch_pairs(self, sgrna_sequences: list, target_sequences: list,
                          max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode batch of sgRNA-target pairs efficiently using batched tokenization
        
        This is the optimized version that processes all pairs in a single GPU forward pass
        instead of looping through pairs sequentially.
        
        Format for each pair: [CLS] sgRNA [SEP] target [SEP]
        
        Args:
            sgrna_sequences: List of sgRNA sequence strings
            target_sequences: List of target sequence strings
            max_length: Maximum sequence length (default: None, no truncation)
            
        Returns:
            Tensor of shape (batch_size, embed_dim) - [CLS] embeddings for all pairs
            
        Raises:
            ValueError: If sgrna_sequences and target_sequences have different lengths
        """
        if len(sgrna_sequences) != len(target_sequences):
            raise ValueError(
                f"sgrna_sequences and target_sequences must have same length. "
                f"Got {len(sgrna_sequences)} and {len(target_sequences)}"
            )
        
        # Tokenize all sequences at once (CPU operation)
        tokenized_pairs = []
        for sgrna_seq, target_seq in zip(sgrna_sequences, target_sequences):
            # Normalize and validate
            sgrna_seq = sgrna_seq.upper().replace('T', 'U')   # DNA→RNA for RNA-FM
            target_seq = target_seq.upper().replace('T', 'U')

            valid_nucleotides = set('ACGU')
            if set(sgrna_seq) - valid_nucleotides or set(target_seq) - valid_nucleotides:
                raise ValueError(f"Invalid nucleotides in sequences")
            
            # Tokenize: [CLS] sgRNA [SEP] target [SEP]
            sgrna_tokens = torch.tensor(
                [self.alphabet.get_idx(nuc) for nuc in sgrna_seq],
                dtype=torch.long
            )
            target_tokens = torch.tensor(
                [self.alphabet.get_idx(nuc) for nuc in target_seq],
                dtype=torch.long
            )
            
            pair_tokens = torch.cat([
                torch.tensor([self.cls_idx], dtype=torch.long),
                sgrna_tokens,
                torch.tensor([self.sep_idx], dtype=torch.long),
                target_tokens,
                torch.tensor([self.sep_idx], dtype=torch.long)
            ])
            
            # Truncate if needed
            if max_length is not None and len(pair_tokens) > max_length:
                pair_tokens = pair_tokens[:max_length]
            
            tokenized_pairs.append(pair_tokens)
        
        # Pad all sequences to same length for batching
        max_seq_len = max(len(tokens) for tokens in tokenized_pairs)
        
        batch_tokens = []
        for tokens in tokenized_pairs:
            # Pad with padding token
            padding_len = max_seq_len - len(tokens)
            if padding_len > 0:
                padded = torch.cat([
                    tokens,
                    torch.full((padding_len,), self.pad_idx, dtype=torch.long)
                ])
            else:
                padded = tokens
            batch_tokens.append(padded)
        
        # Stack into batch tensor: (batch_size, max_seq_len)
        batch_tensor = torch.stack(batch_tokens)
        
        # Move to model device
        dev = next(self.model.parameters()).device
        batch_tensor = batch_tensor.to(dev)
        
        # Single GPU forward pass for entire batch
        with torch.set_grad_enabled(self.training):
            result = self.model(batch_tensor, repr_layers=[self.num_layers])
        
        # Extract [CLS] embeddings from all sequences in batch
        embeddings = result['representations'][self.num_layers]  # (batch_size, max_seq_len, embed_dim)
        cls_embeddings = embeddings[:, 0, :]  # (batch_size, embed_dim)
        
        return cls_embeddings
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'model_path': str(self.model_path),
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.model.args.attention_heads,
            'ffn_embed_dim': self.model.args.ffn_embed_dim,
            'vocab_size': len(self.alphabet),
            'cls_idx': self.cls_idx,
            'sep_idx': self.sep_idx,
            'pad_idx': self.pad_idx,
            'frozen': self.freeze_layers,
            'num_unfreeze_layers': self.num_unfreeze_layers if self.freeze_layers else 0,
            'device': self.device
        }
    
    def get_trainable_params(self) -> int:
        """
        Get number of trainable parameters
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """
        Get total number of parameters
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())


# Utility functions

def create_rna_fm_encoder(model_path: str = 'models/pretrained/rna_fm_t12.pt',
                          freeze_layers: bool = True,
                          num_unfreeze_layers: int = 2,
                          device: str = 'cpu') -> RNAFMEncoder:
    """
    Create RNA-FM encoder with default settings
    
    Args:
        model_path: Path to pretrained checkpoint
        freeze_layers: Whether to freeze initial layers
        num_unfreeze_layers: Number of final layers to unfreeze
        device: Device to use
        
    Returns:
        RNAFMEncoder instance
    """
    return RNAFMEncoder(
        model_path=model_path,
        freeze_layers=freeze_layers,
        num_unfreeze_layers=num_unfreeze_layers,
        device=device
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("RNA-FM ENCODER MODULE EXAMPLES")
    print("=" * 80)
    
    # Check if RNA-FM is available
    if not RNA_FM_AVAILABLE:
        print("\n⚠ RNA-FM not available in Python path")
        print("Please add RNA-FM-main to PYTHONPATH:")
        print("  export PYTHONPATH=$PYTHONPATH:/path/to/RNA-FM-main")
        sys.exit(1)
    
    # Test parameters
    model_path = 'models/pretrained/rna_fm_t12.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Initialize encoder
    print("\n1. INITIALIZE RNA-FM ENCODER")
    print("-" * 80)
    
    try:
        encoder = RNAFMEncoder(
            model_path=model_path,
            freeze_layers=True,
            num_unfreeze_layers=2,
            device=device
        )
        print(f"✓ Encoder initialized successfully on {device}")
    except FileNotFoundError as e:
        print(f"✗ Model not found: {e}")
        print("  Make sure rna_fm_t12.pt exists in models/pretrained/")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error initializing encoder: {e}")
        sys.exit(1)
    
    # Test 2: Model information
    print("\n2. MODEL INFORMATION")
    print("-" * 80)
    
    info = encoder.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 3: Parameter count
    print("\n3. PARAMETER COUNT")
    print("-" * 80)
    
    total_params = encoder.get_total_params()
    trainable_params = encoder.get_trainable_params()
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Test 4: Tokenization
    print("\n4. TOKENIZATION")
    print("-" * 80)
    
    test_seq = "ACGTACGT"
    tokens = encoder.tokenize(test_seq, add_special_tokens=True)
    print(f"  Sequence: {test_seq}")
    print(f"  Tokens: {tokens.tolist()}")
    print(f"  Token count: {len(tokens)}")
    
    # Test 5: Single sequence encoding
    print("\n5. SINGLE SEQUENCE ENCODING")
    print("-" * 80)
    
    seq = "ACGTACGTACGTACGT"
    cls_embedding = encoder.encode_single(seq)
    print(f"  Sequence: {seq}")
    print(f"  [CLS] embedding shape: {cls_embedding.shape}")
    print(f"  Embedding dimension: {encoder.embed_dim}")
    print(f"  Embedding mean: {cls_embedding.mean().item():.6f}")
    print(f"  Embedding std: {cls_embedding.std().item():.6f}")
    
    # Test 6: Pair sequence encoding
    print("\n6. PAIR SEQUENCE ENCODING")
    print("-" * 80)
    
    sgrna = "ACGTACGTACGTACGT"
    target = "TGCATGCATGCATGCA"
    
    cls_embedding = encoder.encode_pair(sgrna, target)
    print(f"  sgRNA: {sgrna}")
    print(f"  Target: {target}")
    print(f"  [CLS] embedding shape: {cls_embedding.shape}")
    
    # Test 7: Full embeddings
    print("\n7. FULL SEQUENCE EMBEDDINGS")
    print("-" * 80)
    
    cls_emb, full_emb = encoder.encode_pair(
        sgrna, target, return_full_embeddings=True
    )
    print(f"  [CLS] embedding shape: {cls_emb.shape}")
    print(f"  Full embeddings shape: {full_emb.shape}")
    print(f"  Sequence length: {full_emb.shape[0]}")
    
    # Test 8: Batch encoding
    print("\n8. BATCH ENCODING")
    print("-" * 80)
    
    sequences = [
        "ACGTACGTACGTACGT",
        "TGCATGCATGCATGCA",
        "AAAAAAAAAAAAAAAA",
        "GGGGGGGGGGGGGGGG"
    ]
    
    batch_embeddings = encoder.encode_batch(sequences)
    print(f"  Number of sequences: {len(sequences)}")
    print(f"  Batch embeddings shape: {batch_embeddings.shape}")
    print(f"  Expected shape: ({len(sequences)}, {encoder.embed_dim})")
    
    # Test 9: Variable-length sequences
    print("\n9. VARIABLE-LENGTH SEQUENCES")
    print("-" * 80)
    
    for seq_len in [10, 20, 50]:
        seq = "A" * seq_len
        embedding = encoder.encode_single(seq)
        print(f"  Sequence length: {seq_len} → Embedding shape: {embedding.shape}")
    
    # Test 10: Invalid sequence handling
    print("\n10. INVALID SEQUENCE HANDLING")
    print("-" * 80)
    
    try:
        invalid_seq = "ACGTXYZ"
        encoder.encode_single(invalid_seq)
        print("  ✗ Should have raised error for invalid sequence")
    except ValueError as e:
        print(f"  ✓ Correctly caught invalid sequence: {e}")
    
    # Test 11: Gradient computation
    print("\n11. GRADIENT COMPUTATION")
    print("-" * 80)
    
    # Create encoder without frozen layers for gradient test
    encoder_trainable = RNAFMEncoder(
        model_path=model_path,
        freeze_layers=False,
        device=device
    )
    
    # Get embedding with gradient tracking
    seq = "ACGTACGTACGTACGT"
    tokens = encoder_trainable.tokenize(seq, add_special_tokens=True)
    tokens = tokens.unsqueeze(0).to(device)
    
    result = encoder_trainable.model(tokens, repr_layers=[encoder_trainable.num_layers])
    embeddings = result['representations'][encoder_trainable.num_layers]
    cls_embedding = embeddings[0, 0, :]
    
    # Compute loss and backward
    loss = cls_embedding.sum()
    loss.backward()
    
    print(f"  ✓ Gradients computed successfully")
    print(f"  Loss: {loss.item():.6f}")
    
    # Test 12: Device handling
    print("\n12. DEVICE HANDLING")
    print("-" * 80)
    
    print(f"  Current device: {device}")
    print(f"  Encoder device: {encoder.device}")
    print(f"  Model device: {next(encoder.model.parameters()).device}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 80)
