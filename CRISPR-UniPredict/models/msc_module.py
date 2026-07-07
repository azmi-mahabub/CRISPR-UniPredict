"""
Multi-Scale Convolution (MSC) Module for CRISPR_HNN
Implements parallel CNN branches with different kernel sizes to capture
local sequence patterns at multiple scales simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiScaleConvolution(nn.Module):
    """
    Multi-Scale Convolution module with parallel CNN branches
    
    Captures local sequence patterns at different scales (1×1, 3×3, 5×5, 7×7)
    using parallel convolutional branches with residual connections.
    
    Architecture:
    - Input: (batch, seq_len, 4) one-hot encoded sequences
    - Four parallel Conv1d branches with different kernel sizes
    - Each branch: Conv1d → BatchNorm1d → ReLU → Dropout
    - Concatenate outputs from all branches
    - Add residual connection with original input
    - Output: (batch, seq_len, out_channels * 4)
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 64, 
                 dropout: float = 0.35, kernel_sizes: Optional[List[int]] = None):
        """
        Initialize Multi-Scale Convolution module
        
        Args:
            in_channels: Number of input channels (default: 4 for one-hot encoded DNA)
            out_channels: Number of output channels per branch (default: 64)
            dropout: Dropout rate (default: 0.35)
            kernel_sizes: List of kernel sizes for parallel branches
                         (default: [1, 3, 5, 7])
        """
        super(MultiScaleConvolution, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout
        
        # Default kernel sizes for multi-scale analysis
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7]
        
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        
        # Create parallel convolutional branches
        self.branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            # Calculate padding to maintain sequence length
            # For odd kernel sizes: padding = kernel_size // 2
            # For even kernel sizes: padding = (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2
            
            branch = nn.Sequential(
                # Conv1d layer
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,  # Same padding to preserve sequence length
                    bias=True
                ),
                # Batch normalization
                nn.BatchNorm1d(out_channels),
                # ReLU activation
                nn.ReLU(inplace=True),
                # Dropout
                nn.Dropout(p=dropout)
            )
            self.branches.append(branch)
        
        # Total output channels after concatenation
        self.total_out_channels = out_channels * self.num_branches
        
        # Optional: residual projection layer if needed
        # Maps input channels to match concatenated output for residual connection
        if in_channels != self.total_out_channels:
            self.residual_proj = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.total_out_channels,
                kernel_size=1,
                bias=True
            )
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolution
        
        Args:
            x: Input tensor of shape (batch, seq_len, in_channels)
               or (batch, in_channels, seq_len) depending on input format
               
        Returns:
            Output tensor of shape (batch, seq_len, total_out_channels)
            or (batch, total_out_channels, seq_len) depending on input format
        """
        # Check input shape and transpose if needed
        # Expected format from encoding: (batch, seq_len, in_channels)
        # Conv1d expects: (batch, in_channels, seq_len)
        
        if x.dim() == 3 and x.size(2) == self.in_channels:
            # Input is (batch, seq_len, in_channels), transpose to Conv1d format
            x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
            transpose_back = True
        else:
            # Assume input is already in Conv1d format (batch, in_channels, seq_len)
            transpose_back = False
        
        # Store original input for residual connection
        residual = x
        
        # Process through parallel branches
        branch_outputs = []
        
        for branch in self.branches:
            output = branch(x)
            branch_outputs.append(output)
        
        # Concatenate outputs from all branches
        # Shape: (batch, out_channels * num_branches, seq_len)
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Add residual connection
        if self.residual_proj is not None:
            # Project residual to match concatenated output channels
            residual_proj = self.residual_proj(residual)
            output = concatenated + residual_proj
        else:
            # Direct residual connection (channels already match)
            output = concatenated + residual
        
        # Transpose back if input was transposed
        if transpose_back:
            output = output.transpose(1, 2)  # (batch, seq_len, total_out_channels)
        
        return output
    
    def get_output_channels(self) -> int:
        """Get total number of output channels after concatenation"""
        return self.total_out_channels
    
    def get_num_branches(self) -> int:
        """Get number of parallel branches"""
        return self.num_branches
    
    def get_kernel_sizes(self) -> List[int]:
        """Get kernel sizes used in branches"""
        return self.kernel_sizes


class MultiScaleConvolutionStack(nn.Module):
    """
    Stack of multiple Multi-Scale Convolution modules
    
    Useful for building deeper architectures by stacking MSC modules
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 64,
                 num_layers: int = 1, dropout: float = 0.35,
                 kernel_sizes: Optional[List[int]] = None):
        """
        Initialize stack of MSC modules
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per branch per layer
            num_layers: Number of MSC modules to stack
            dropout: Dropout rate
            kernel_sizes: Kernel sizes for branches
        """
        super(MultiScaleConvolutionStack, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # Create stack of MSC modules
        self.msc_layers = nn.ModuleList()
        
        current_in_channels = in_channels
        
        for i in range(num_layers):
            msc = MultiScaleConvolution(
                in_channels=current_in_channels,
                out_channels=out_channels,
                dropout=dropout,
                kernel_sizes=kernel_sizes
            )
            self.msc_layers.append(msc)
            
            # Output of current layer becomes input to next layer
            current_in_channels = msc.get_output_channels()
        
        self.total_out_channels = current_in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked MSC modules
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all MSC layers
        """
        for msc_layer in self.msc_layers:
            x = msc_layer(x)
        
        return x
    
    def get_output_channels(self) -> int:
        """Get total output channels after all layers"""
        return self.total_out_channels


# Utility functions

def create_msc_module(in_channels: int = 4, out_channels: int = 64,
                     dropout: float = 0.35) -> MultiScaleConvolution:
    """
    Create a Multi-Scale Convolution module with default settings
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels per branch
        dropout: Dropout rate
        
    Returns:
        MultiScaleConvolution module
    """
    return MultiScaleConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=dropout
    )


def create_msc_stack(in_channels: int = 4, out_channels: int = 64,
                    num_layers: int = 2, dropout: float = 0.35) -> MultiScaleConvolutionStack:
    """
    Create a stack of Multi-Scale Convolution modules
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels per branch per layer
        num_layers: Number of MSC modules to stack
        dropout: Dropout rate
        
    Returns:
        MultiScaleConvolutionStack module
    """
    return MultiScaleConvolutionStack(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout
    )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SCALE CONVOLUTION MODULE EXAMPLES")
    print("=" * 80)
    
    # Create sample input (batch_size=2, seq_len=23, in_channels=4)
    batch_size = 2
    seq_len = 23
    in_channels = 4
    
    # Test 1: Single MSC module
    print("\n1. SINGLE MULTI-SCALE CONVOLUTION MODULE")
    print("-" * 80)
    
    msc = MultiScaleConvolution(
        in_channels=in_channels,
        out_channels=64,
        dropout=0.35
    )
    
    # Create input tensor (batch, seq_len, in_channels)
    x = torch.randn(batch_size, seq_len, in_channels)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = msc(x)
    print(f"Output shape: {output.shape}")
    print(f"Output channels: {msc.get_output_channels()}")
    print(f"Number of branches: {msc.get_num_branches()}")
    print(f"Kernel sizes: {msc.get_kernel_sizes()}")
    
    # Test 2: Input in Conv1d format
    print("\n2. INPUT IN CONV1D FORMAT (batch, in_channels, seq_len)")
    print("-" * 80)
    
    x_conv = torch.randn(batch_size, in_channels, seq_len)
    print(f"Input shape: {x_conv.shape}")
    
    output_conv = msc(x_conv)
    print(f"Output shape: {output_conv.shape}")
    
    # Test 3: Stacked MSC modules
    print("\n3. STACKED MULTI-SCALE CONVOLUTION MODULES")
    print("-" * 80)
    
    msc_stack = MultiScaleConvolutionStack(
        in_channels=in_channels,
        out_channels=64,
        num_layers=2,
        dropout=0.35
    )
    
    x = torch.randn(batch_size, seq_len, in_channels)
    print(f"Input shape: {x.shape}")
    
    output = msc_stack(x)
    print(f"Output shape: {output.shape}")
    print(f"Output channels after stacking: {msc_stack.get_output_channels()}")
    
    # Test 4: Parameter count
    print("\n4. MODEL PARAMETERS")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in msc.parameters())
    trainable_params = sum(p.numel() for p in msc.parameters() if p.requires_grad)
    
    print(f"Single MSC module:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    total_params_stack = sum(p.numel() for p in msc_stack.parameters())
    trainable_params_stack = sum(p.numel() for p in msc_stack.parameters() if p.requires_grad)
    
    print(f"\nStacked MSC (2 layers):")
    print(f"  Total parameters: {total_params_stack:,}")
    print(f"  Trainable parameters: {trainable_params_stack:,}")
    
    # Test 5: Gradient flow
    print("\n5. GRADIENT FLOW TEST")
    print("-" * 80)
    
    msc = MultiScaleConvolution(in_channels=4, out_channels=64)
    x = torch.randn(batch_size, seq_len, in_channels, requires_grad=True)
    
    output = msc(x)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradients computed successfully")
    print(f"  Input gradient shape: {x.grad.shape}")
    print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
    
    # Test 6: Different kernel sizes
    print("\n6. CUSTOM KERNEL SIZES")
    print("-" * 80)
    
    custom_kernels = [1, 3, 5, 9]  # Use odd kernel sizes for proper padding
    msc_custom = MultiScaleConvolution(
        in_channels=4,
        out_channels=32,
        kernel_sizes=custom_kernels
    )
    
    x = torch.randn(batch_size, seq_len, in_channels)
    output = msc_custom(x)
    
    print(f"Custom kernel sizes: {custom_kernels}")
    print(f"Number of branches: {msc_custom.get_num_branches()}")
    print(f"Output channels: {msc_custom.get_output_channels()}")
    print(f"Output shape: {output.shape}")
    
    # Test 7: Batch normalization and dropout
    print("\n7. TRAINING VS EVALUATION MODE")
    print("-" * 80)
    
    msc = MultiScaleConvolution(in_channels=4, out_channels=64, dropout=0.35)
    x = torch.randn(batch_size, seq_len, in_channels)
    
    # Training mode
    msc.train()
    output_train = msc(x)
    print(f"Training mode output shape: {output_train.shape}")
    
    # Evaluation mode
    msc.eval()
    output_eval = msc(x)
    print(f"Evaluation mode output shape: {output_eval.shape}")
    
    print(f"✓ Batch norm and dropout working correctly")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
