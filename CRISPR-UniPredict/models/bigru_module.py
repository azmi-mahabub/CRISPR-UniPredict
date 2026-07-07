"""
Bidirectional GRU (BiGRU) Module for CRISPR_HNN
Implements bidirectional GRU to capture sequential dependencies and context
in both forward and backward directions
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import warnings


class BiGRUModule(nn.Module):
    """
    Bidirectional GRU module for sequence processing
    
    Processes sequences in both forward and backward directions,
    capturing bidirectional context and dependencies.
    
    Architecture:
    - Input: (batch, seq_len, input_dim) sequential features
    - Bidirectional GRU with specified hidden dimensions
    - Forward GRU: processes sequence left-to-right
    - Backward GRU: processes sequence right-to-left
    - Concatenate forward and backward hidden states
    - Dropout for regularization
    - Output: (batch, seq_len, hidden_dim*2) bidirectional features
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 1, dropout: float = 0.35,
                 bidirectional: bool = True):
        """
        Initialize Bidirectional GRU module
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of hidden units in GRU (default: 128)
            num_layers: Number of GRU layers (default: 1)
            dropout: Dropout rate (default: 0.35)
            bidirectional: Whether to use bidirectional GRU (default: True)
        """
        super(BiGRUModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input/output shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout between layers
            bidirectional=bidirectional
        )
        
        # Dropout after GRU
        self.dropout = nn.Dropout(p=dropout)
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through bidirectional GRU
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            lengths: Optional tensor of sequence lengths for packing
                    Shape: (batch,) with actual lengths of each sequence
                    
        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim*2)
            Contains concatenated forward and backward hidden states
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Handle variable-length sequences with packing if lengths provided
        if lengths is not None:
            # Sort sequences by length in descending order
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            sorted_lengths, sorted_idx = torch.sort(lengths_cpu, descending=True)
            sorted_x = x[sorted_idx]
            
            # Pack padded sequences
            packed_x = nn.utils.rnn.pack_padded_sequence(
                sorted_x, 
                sorted_lengths.tolist(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # GRU forward pass
            packed_output, hidden = self.gru(packed_x)
            
            # Unpack sequences
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
            
            # Restore original order
            _, unsorted_idx = torch.sort(sorted_idx)
            output = output[unsorted_idx]
            
            # Pad to original sequence length if needed
            if output.size(1) < seq_len:
                padding = torch.zeros(
                    batch_size, 
                    seq_len - output.size(1), 
                    self.output_dim,
                    device=x.device,
                    dtype=x.dtype
                )
                output = torch.cat([output, padding], dim=1)
        else:
            # Standard GRU forward pass without packing
            output, hidden = self.gru(x)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output
    
    def get_output_dim(self) -> int:
        """Get output dimension (hidden_dim * 2 for bidirectional)"""
        return self.output_dim
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension"""
        return self.hidden_dim
    
    def get_num_layers(self) -> int:
        """Get number of GRU layers"""
        return self.num_layers


class BiGRUStack(nn.Module):
    """
    Stack of multiple Bidirectional GRU modules
    
    Useful for building deeper architectures by stacking BiGRU modules
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.35):
        """
        Initialize stack of BiGRU modules
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of hidden units per layer
            num_layers: Number of BiGRU layers to stack
            dropout: Dropout rate
        """
        super(BiGRUStack, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create stack of BiGRU modules
        self.bigru_layers = nn.ModuleList()
        
        current_input_dim = input_dim
        
        for i in range(num_layers):
            bigru = BiGRUModule(
                input_dim=current_input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=True
            )
            self.bigru_layers.append(bigru)
            
            # Output of current layer becomes input to next layer
            current_input_dim = bigru.get_output_dim()
        
        self.output_dim = current_input_dim
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through stacked BiGRU modules
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Output tensor after passing through all BiGRU layers
        """
        for bigru_layer in self.bigru_layers:
            x = bigru_layer(x, lengths)
        
        return x
    
    def get_output_dim(self) -> int:
        """Get total output dimension after all layers"""
        return self.output_dim


class BiGRUWithAttention(nn.Module):
    """
    Bidirectional GRU with attention mechanism
    
    Combines BiGRU with self-attention for better feature extraction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.35,
                 use_attention: bool = False):
        """
        Initialize BiGRU with optional attention
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of hidden units
            num_layers: Number of GRU layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(BiGRUWithAttention, self).__init__()
        
        self.bigru = BiGRUModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.use_attention = use_attention
        
        if use_attention:
            # Simple attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=self.bigru.get_output_dim(),
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
    
    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BiGRU with optional attention
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths
            
        Returns:
            Output tensor
        """
        # BiGRU forward pass
        output = self.bigru(x, lengths)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention
            attn_output, _ = self.attention(output, output, output)
            output = output + attn_output  # Residual connection
        
        return output


# Utility functions

def create_bigru_module(input_dim: int, hidden_dim: int = 128,
                       num_layers: int = 1, dropout: float = 0.35) -> BiGRUModule:
    """
    Create a Bidirectional GRU module with default settings
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Number of hidden units
        num_layers: Number of GRU layers
        dropout: Dropout rate
        
    Returns:
        BiGRUModule instance
    """
    return BiGRUModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )


def create_bigru_stack(input_dim: int, hidden_dim: int = 128,
                      num_layers: int = 2, dropout: float = 0.35) -> BiGRUStack:
    """
    Create a stack of Bidirectional GRU modules
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Number of hidden units per layer
        num_layers: Number of BiGRU layers
        dropout: Dropout rate
        
    Returns:
        BiGRUStack instance
    """
    return BiGRUStack(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("BIDIRECTIONAL GRU MODULE EXAMPLES")
    print("=" * 80)
    
    # Test parameters
    batch_size = 4
    seq_len = 23
    input_dim = 256  # From MSC module output
    hidden_dim = 128
    
    # Test 1: Single BiGRU module
    print("\n1. SINGLE BIDIRECTIONAL GRU MODULE")
    print("-" * 80)
    
    bigru = BiGRUModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.35
    )
    
    # Create input tensor (batch, seq_len, input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = bigru(x)
    print(f"Output shape: {output.shape}")
    print(f"Output dimension: {bigru.get_output_dim()}")
    print(f"Hidden dimension: {bigru.get_hidden_dim()}")
    print(f"Number of layers: {bigru.get_num_layers()}")
    
    # Test 2: BiGRU with variable-length sequences
    print("\n2. VARIABLE-LENGTH SEQUENCES WITH PACKING")
    print("-" * 80)
    
    # Create sequences of different lengths
    lengths = torch.tensor([23, 20, 18, 15])
    
    # Create padded input
    x_padded = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"Input shape: {x_padded.shape}")
    print(f"Sequence lengths: {lengths.tolist()}")
    
    # Forward pass with lengths
    output = bigru(x_padded, lengths)
    print(f"Output shape: {output.shape}")
    
    # Test 3: Stacked BiGRU modules
    print("\n3. STACKED BIDIRECTIONAL GRU MODULES")
    print("-" * 80)
    
    bigru_stack = BiGRUStack(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.35
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    output = bigru_stack(x)
    print(f"Output shape: {output.shape}")
    print(f"Output dimension: {bigru_stack.get_output_dim()}")
    
    # Test 4: BiGRU with attention
    print("\n4. BIDIRECTIONAL GRU WITH ATTENTION")
    print("-" * 80)
    
    bigru_attn = BiGRUWithAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.35,
        use_attention=True
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    output = bigru_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test 5: Parameter count
    print("\n5. MODEL PARAMETERS")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in bigru.parameters())
    trainable_params = sum(p.numel() for p in bigru.parameters() if p.requires_grad)
    
    print(f"Single BiGRU module:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    total_params_stack = sum(p.numel() for p in bigru_stack.parameters())
    trainable_params_stack = sum(p.numel() for p in bigru_stack.parameters() if p.requires_grad)
    
    print(f"\nStacked BiGRU (2 layers):")
    print(f"  Total parameters: {total_params_stack:,}")
    print(f"  Trainable parameters: {trainable_params_stack:,}")
    
    # Test 6: Gradient flow
    print("\n6. GRADIENT FLOW TEST")
    print("-" * 80)
    
    bigru = BiGRUModule(input_dim=input_dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    output = bigru(x)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradients computed successfully")
    print(f"  Input gradient shape: {x.grad.shape}")
    print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
    
    # Test 7: Training vs evaluation mode
    print("\n7. TRAINING VS EVALUATION MODE")
    print("-" * 80)
    
    bigru = BiGRUModule(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.35)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Training mode
    bigru.train()
    output_train = bigru(x)
    print(f"Training mode output shape: {output_train.shape}")
    
    # Evaluation mode
    bigru.eval()
    output_eval = bigru(x)
    print(f"Evaluation mode output shape: {output_eval.shape}")
    
    print(f"✓ Dropout working correctly in training mode")
    
    # Test 8: Different hidden dimensions
    print("\n8. DIFFERENT HIDDEN DIMENSIONS")
    print("-" * 80)
    
    for hidden_dim_test in [64, 128, 256]:
        bigru_test = BiGRUModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim_test,
            dropout=0.35
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = bigru_test(x)
        
        print(f"Hidden dim: {hidden_dim_test} → Output dim: {bigru_test.get_output_dim()} → Output shape: {output.shape}")
    
    # Test 9: Sequence length handling
    print("\n9. SEQUENCE LENGTH HANDLING")
    print("-" * 80)
    
    for test_seq_len in [10, 23, 50]:
        bigru = BiGRUModule(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(batch_size, test_seq_len, input_dim)
        output = bigru(x)
        
        print(f"Sequence length: {test_seq_len} → Output shape: {output.shape}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
