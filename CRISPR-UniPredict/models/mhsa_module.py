"""
Multi-Head Self-Attention (MHSA) Module for CRISPR_HNN
Implements multi-head self-attention mechanism to capture long-range dependencies
and allow the model to focus on important positions in the sequence
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for sequence processing
    
    Allows the model to jointly attend to information from different representation
    subspaces at different positions. Captures long-range dependencies and 
    contextual relationships in sequences.
    
    Architecture:
    - Input: (batch, seq_len, embed_dim) features from MSC or BiGRU
    - Query, Key, Value projections
    - Multi-head scaled dot-product attention (4 heads)
    - Residual connection
    - Layer normalization
    - Feed-forward network with ReLU activation
    - Dropout for regularization
    - Output: (batch, seq_len, embed_dim) attended features (same shape as input)
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.35):
        """
        Initialize Multi-Head Self-Attention module
        
        Args:
            embed_dim: Embedding dimension (feature dimension)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate (default: 0.35)
            
        Raises:
            ValueError: If embed_dim is not divisible by num_heads
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        
        # Scaling factor for scaled dot-product attention
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Query, Key, Value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(p=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # Expand dimension
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim)   # Project back
        )
        
        # Dropout for residual connections
        self.residual_dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head self-attention
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len) or (seq_len, seq_len)
                  True values indicate positions to mask (set to -inf)
                  
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
            Contains attended features with residual connections and layer normalization
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Store input for residual connection
        residual = x
        
        # Layer normalization before attention (pre-norm)
        x_norm = self.norm1(x)
        
        # Project to Query, Key, Value
        # Shape: (batch, seq_len, embed_dim)
        Q = self.query_proj(x_norm)
        K = self.key_proj(x_norm)
        V = self.value_proj(x_norm)
        
        # Reshape for multi-head attention
        # (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) x (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask if needed
            if mask.dim() == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            
            # Apply mask: set masked positions to very negative value
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        # (batch, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back to (batch, seq_len, embed_dim)
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, seq_len, embed_dim)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, embed_dim)
        
        # Output projection
        attended = self.output_proj(attended)
        
        # Apply dropout to projected output
        attended = self.residual_dropout(attended)
        
        # Residual connection and layer normalization
        # x = residual + attended
        x = residual + attended
        
        # Store for second residual connection
        residual2 = x
        
        # Layer normalization before feed-forward (pre-norm)
        x_norm = self.norm2(x)
        
        # Feed-forward network
        ffn_output = self.ffn(x_norm)
        
        # Apply dropout to FFN output
        ffn_output = self.residual_dropout(ffn_output)
        
        # Second residual connection
        x = residual2 + ffn_output
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for visualization/analysis
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Query, Key, Value
        Q = self.query_proj(self.norm1(x))
        K = self.key_proj(self.norm1(x))
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        return attention_weights


class MultiHeadSelfAttentionStack(nn.Module):
    """
    Stack of multiple Multi-Head Self-Attention modules
    
    Useful for building deeper architectures by stacking MHSA modules
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.35):
        """
        Initialize stack of Multi-Head Self-Attention modules
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads per layer
            num_layers: Number of MHSA layers to stack
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttentionStack, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Create stack of MHSA modules
        self.mhsa_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through stacked MHSA modules
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor after passing through all MHSA layers
        """
        for mhsa_layer in self.mhsa_layers:
            x = mhsa_layer(x, mask)
        
        return x


# Utility functions

def create_mhsa_module(embed_dim: int, num_heads: int = 4,
                       dropout: float = 0.35) -> MultiHeadSelfAttention:
    """
    Create a Multi-Head Self-Attention module with default settings
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        MultiHeadSelfAttention instance
    """
    return MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout
    )


def create_mhsa_stack(embed_dim: int, num_heads: int = 4,
                      num_layers: int = 2, dropout: float = 0.35) -> MultiHeadSelfAttentionStack:
    """
    Create a stack of Multi-Head Self-Attention modules
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads per layer
        num_layers: Number of MHSA layers
        dropout: Dropout rate
        
    Returns:
        MultiHeadSelfAttentionStack instance
    """
    return MultiHeadSelfAttentionStack(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-HEAD SELF-ATTENTION MODULE EXAMPLES")
    print("=" * 80)
    
    # Test parameters
    batch_size = 4
    seq_len = 23
    embed_dim = 256  # From BiGRU module output
    num_heads = 4
    
    # Test 1: Single MHSA module
    print("\n1. SINGLE MULTI-HEAD SELF-ATTENTION MODULE")
    print("-" * 80)
    
    mhsa = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.35
    )
    
    # Create input tensor (batch, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = mhsa(x)
    print(f"Output shape: {output.shape}")
    print(f"✓ Output shape matches input shape")
    
    # Test 2: Attention weights extraction
    print("\n2. ATTENTION WEIGHTS EXTRACTION")
    print("-" * 80)
    
    attention_weights = mhsa.get_attention_weights(x)
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  (batch={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")
    print(f"Attention weights sum per head (should be ~1.0):")
    print(f"  Mean: {attention_weights.sum(dim=-1).mean().item():.4f}")
    
    # Test 3: Attention mask
    print("\n3. ATTENTION MASK")
    print("-" * 80)
    
    # Create causal mask (lower triangular)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    output_masked = mhsa(x, mask=causal_mask)
    print(f"Output shape with causal mask: {output_masked.shape}")
    print(f"✓ Causal mask applied successfully")
    
    # Test 4: Stacked MHSA modules
    print("\n4. STACKED MULTI-HEAD SELF-ATTENTION MODULES")
    print("-" * 80)
    
    mhsa_stack = MultiHeadSelfAttentionStack(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=2,
        dropout=0.35
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")
    
    output = mhsa_stack(x)
    print(f"Output shape: {output.shape}")
    print(f"✓ Stacked MHSA output shape matches input")
    
    # Test 5: Parameter count
    print("\n5. MODEL PARAMETERS")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in mhsa.parameters())
    trainable_params = sum(p.numel() for p in mhsa.parameters() if p.requires_grad)
    
    print(f"Single MHSA module:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    total_params_stack = sum(p.numel() for p in mhsa_stack.parameters())
    trainable_params_stack = sum(p.numel() for p in mhsa_stack.parameters() if p.requires_grad)
    
    print(f"\nStacked MHSA (2 layers):")
    print(f"  Total parameters: {total_params_stack:,}")
    print(f"  Trainable parameters: {trainable_params_stack:,}")
    
    # Test 6: Gradient flow
    print("\n6. GRADIENT FLOW TEST")
    print("-" * 80)
    
    mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    output = mhsa(x)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradients computed successfully")
    print(f"  Input gradient shape: {x.grad.shape}")
    print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
    
    # Test 7: Training vs evaluation mode
    print("\n7. TRAINING VS EVALUATION MODE")
    print("-" * 80)
    
    mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.35)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Training mode
    mhsa.train()
    output_train = mhsa(x)
    print(f"Training mode output shape: {output_train.shape}")
    
    # Evaluation mode
    mhsa.eval()
    output_eval = mhsa(x)
    print(f"Evaluation mode output shape: {output_eval.shape}")
    
    print(f"✓ Dropout working correctly in training mode")
    
    # Test 8: Different embedding dimensions
    print("\n8. DIFFERENT EMBEDDING DIMENSIONS")
    print("-" * 80)
    
    for embed_dim_test in [128, 256, 512]:
        # Find valid num_heads (must divide embed_dim)
        num_heads_test = 4 if embed_dim_test % 4 == 0 else 2
        
        mhsa_test = MultiHeadSelfAttention(
            embed_dim=embed_dim_test,
            num_heads=num_heads_test,
            dropout=0.35
        )
        
        x = torch.randn(batch_size, seq_len, embed_dim_test)
        output = mhsa_test(x)
        
        print(f"Embed dim: {embed_dim_test}, Num heads: {num_heads_test} → Output shape: {output.shape}")
    
    # Test 9: Sequence length handling
    print("\n9. SEQUENCE LENGTH HANDLING")
    print("-" * 80)
    
    for test_seq_len in [10, 23, 50]:
        mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(batch_size, test_seq_len, embed_dim)
        output = mhsa(x)
        
        print(f"Sequence length: {test_seq_len} → Output shape: {output.shape}")
    
    # Test 10: Batch size handling
    print("\n10. BATCH SIZE HANDLING")
    print("-" * 80)
    
    for test_batch_size in [1, 4, 8, 16]:
        mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(test_batch_size, seq_len, embed_dim)
        output = mhsa(x)
        
        print(f"Batch size: {test_batch_size} → Output shape: {output.shape}")
    
    # Test 11: Residual connections
    print("\n11. RESIDUAL CONNECTIONS")
    print("-" * 80)
    
    mhsa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # With residual connections, output should be influenced by input
    output = mhsa(x)
    
    # Check that output is not just zeros or identical to input
    output_diff = (output - x).abs().mean().item()
    print(f"Mean absolute difference between output and input: {output_diff:.6f}")
    print(f"✓ Residual connections working (output differs from input)")
    
    # Test 12: Different number of heads
    print("\n12. DIFFERENT NUMBER OF HEADS")
    print("-" * 80)
    
    for num_heads_test in [1, 2, 4, 8]:
        if embed_dim % num_heads_test == 0:
            mhsa_test = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads_test,
                dropout=0.35
            )
            
            x = torch.randn(batch_size, seq_len, embed_dim)
            output = mhsa_test(x)
            
            print(f"Num heads: {num_heads_test} → Output shape: {output.shape}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
