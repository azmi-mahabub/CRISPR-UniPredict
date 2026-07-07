"""
Attention Visualization for CRISPR-UniPredict
Visualizes what the model focuses on in sequences
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualizes attention weights from CRISPR-UniPredict model
    
    Extracts and visualizes attention patterns to understand
    what the model focuses on in sequences
    """
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize visualizer
        
        Args:
            model: CRISPRUniPredict model instance
            device: Device to use
        """
        self.model = model
        self.device = device
        self.attention_hooks = {}
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register hooks to capture attention weights"""
        for name, module in self.model.named_modules():
            if 'mhsa' in name.lower():
                module.register_forward_hook(self._create_hook(name))
    
    def _create_hook(self, name: str):
        """Create forward hook for attention module"""
        def hook(module, input, output):
            # Store attention weights
            if hasattr(module, 'attention_weights'):
                self.attention_hooks[name] = module.attention_weights.detach().cpu()
        
        return hook
    
    def extract_attention_weights(self,
                                 sgrna_onehot: torch.Tensor,
                                 sgrna_label: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from model
        
        Args:
            sgrna_onehot: One-hot encoded sgRNA (1, 23, 4)
            sgrna_label: Label encoded sgRNA (1, 23)
        
        Returns:
            Dictionary with attention weights per layer
        """
        # Ensure batch dimension
        if sgrna_onehot.dim() == 2:
            sgrna_onehot = sgrna_onehot.unsqueeze(0)
        if sgrna_label.dim() == 1:
            sgrna_label = sgrna_label.unsqueeze(0)
        
        # Move to device
        sgrna_onehot = sgrna_onehot.to(self.device)
        sgrna_label = sgrna_label.to(self.device)
        
        # Clear previous hooks
        self.attention_hooks = {}
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(sgrna_onehot, sgrna_label, task_type='both')
        
        return self.attention_hooks
    
    def plot_attention_heatmap(self,
                              attention_weights: torch.Tensor,
                              sequence: str,
                              output_path: Optional[Path] = None,
                              title: str = 'Attention Heatmap') -> None:
        """
        Plot attention heatmap
        
        Args:
            attention_weights: Attention weights (seq_len, seq_len) or (heads, seq_len, seq_len)
            sequence: DNA sequence string
            output_path: Path to save figure
            title: Figure title
        """
        try:
            # Handle multi-head attention
            if attention_weights.dim() == 3:
                # Average across heads
                attention_weights = attention_weights.mean(dim=0)
            
            # Convert to numpy
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.numpy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot heatmap
            sns.heatmap(
                attention_weights,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax,
                xticklabels=list(sequence),
                yticklabels=list(sequence),
                vmin=0,
                vmax=1
            )
            
            # Highlight seed region (positions 16-20)
            seed_start, seed_end = 16, 20
            rect_x = plt.Rectangle(
                (seed_start, 0),
                seed_end - seed_start,
                len(sequence),
                fill=False,
                edgecolor='blue',
                linewidth=2,
                linestyle='--'
            )
            rect_y = plt.Rectangle(
                (0, seed_start),
                len(sequence),
                seed_end - seed_start,
                fill=False,
                edgecolor='blue',
                linewidth=2,
                linestyle='--'
            )
            ax.add_patch(rect_x)
            ax.add_patch(rect_y)
            
            # Labels
            ax.set_xlabel('Target Position', fontsize=12)
            ax.set_ylabel('Query Position', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add seed region label
            ax.text(
                seed_start + (seed_end - seed_start) / 2,
                -1.5,
                'Seed Region',
                ha='center',
                fontsize=10,
                color='blue',
                fontweight='bold'
            )
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Attention heatmap saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot attention heatmap: {e}")
    
    def plot_position_importance(self,
                                attention_weights: torch.Tensor,
                                sequence: str,
                                output_path: Optional[Path] = None) -> None:
        """
        Plot position-wise importance
        
        Args:
            attention_weights: Attention weights
            sequence: DNA sequence
            output_path: Path to save figure
        """
        try:
            # Handle multi-head attention
            if attention_weights.dim() == 3:
                attention_weights = attention_weights.mean(dim=0)
            
            # Convert to numpy
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.numpy()
            
            # Compute position importance (sum of attention weights)
            position_importance = attention_weights.sum(axis=0)
            
            # Normalize
            position_importance = position_importance / position_importance.max()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot bar chart
            positions = np.arange(len(sequence))
            bars = ax.bar(positions, position_importance, color='steelblue', alpha=0.7)
            
            # Highlight seed region
            seed_start, seed_end = 16, 20
            for i in range(seed_start, seed_end):
                bars[i].set_color('coral')
            
            # Labels
            ax.set_xlabel('Position in sgRNA', fontsize=12)
            ax.set_ylabel('Attention Importance', fontsize=12)
            ax.set_title('Position-wise Attention Importance', fontsize=14, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels(list(sequence))
            
            # Add seed region background
            ax.axvspan(seed_start - 0.5, seed_end - 0.5, alpha=0.2, color='red', label='Seed Region')
            ax.legend()
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Position importance plot saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot position importance: {e}")
    
    def compare_attention_patterns(self,
                                  good_sgrnas: List[Tuple[str, str]],
                                  poor_sgrnas: List[Tuple[str, str]],
                                  output_dir: Optional[Path] = None) -> None:
        """
        Compare attention patterns between high and low efficiency sgRNAs
        
        Args:
            good_sgrnas: List of (sequence, label) tuples for high efficiency
            poor_sgrnas: List of (sequence, label) tuples for low efficiency
            output_dir: Directory to save figures
        """
        try:
            from models.encoding import SequenceEncoder
            
            encoder = SequenceEncoder(device=self.device)
            
            # Extract attention for good sgRNAs
            good_attentions = []
            for sgrna, _ in good_sgrnas:
                onehot = encoder.one_hot_encode(sgrna)
                label = encoder.label_encode(sgrna, add_start_token=False)
                
                weights = self.extract_attention_weights(onehot, label)
                # Get first MHSA layer
                for key, value in weights.items():
                    if 'mhsa' in key.lower():
                        good_attentions.append(value)
                        break
            
            # Extract attention for poor sgRNAs
            poor_attentions = []
            for sgrna, _ in poor_sgrnas:
                onehot = encoder.one_hot_encode(sgrna)
                label = encoder.label_encode(sgrna, add_start_token=False)
                
                weights = self.extract_attention_weights(onehot, label)
                for key, value in weights.items():
                    if 'mhsa' in key.lower():
                        poor_attentions.append(value)
                        break
            
            # Average attention patterns
            good_avg = np.mean([a.numpy() if isinstance(a, torch.Tensor) else a 
                               for a in good_attentions], axis=0)
            poor_avg = np.mean([a.numpy() if isinstance(a, torch.Tensor) else a 
                               for a in poor_attentions], axis=0)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Good sgRNAs
            sns.heatmap(good_avg.mean(axis=0), cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Attention'})
            axes[0].set_title('High Efficiency sgRNAs', fontweight='bold')
            axes[0].set_xlabel('Position')
            axes[0].set_ylabel('Position')
            
            # Poor sgRNAs
            sns.heatmap(poor_avg.mean(axis=0), cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Attention'})
            axes[1].set_title('Low Efficiency sgRNAs', fontweight='bold')
            axes[1].set_xlabel('Position')
            axes[1].set_ylabel('Position')
            
            # Difference
            diff = good_avg.mean(axis=0) - poor_avg.mean(axis=0)
            sns.heatmap(diff, cmap='RdBu_r', center=0, ax=axes[2], cbar_kws={'label': 'Difference'})
            axes[2].set_title('Difference (High - Low)', fontweight='bold')
            axes[2].set_xlabel('Position')
            axes[2].set_ylabel('Position')
            
            plt.tight_layout()
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / 'attention_comparison.png', dpi=300, bbox_inches='tight')
                logger.info(f"Attention comparison saved to {output_dir / 'attention_comparison.png'}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to compare attention patterns: {e}")
    
    def visualize_seed_region_importance(self,
                                        sgrnas: List[str],
                                        output_path: Optional[Path] = None) -> None:
        """
        Visualize seed region importance (positions 16-20)
        
        Args:
            sgrnas: List of sgRNA sequences
            output_path: Path to save figure
        """
        try:
            from models.encoding import SequenceEncoder
            
            encoder = SequenceEncoder(device=self.device)
            
            # Extract position importances
            position_importances = []
            
            for sgrna in sgrnas:
                onehot = encoder.one_hot_encode(sgrna)
                label = encoder.label_encode(sgrna, add_start_token=False)
                
                weights = self.extract_attention_weights(onehot, label)
                
                for key, value in weights.items():
                    if 'mhsa' in key.lower():
                        # Handle multi-head
                        if value.dim() == 3:
                            value = value.mean(dim=0)
                        
                        # Compute position importance
                        pos_imp = value.numpy().sum(axis=0)
                        pos_imp = pos_imp / pos_imp.max()
                        position_importances.append(pos_imp)
                        break
            
            # Average across sgRNAs
            avg_importance = np.mean(position_importances, axis=0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot
            positions = np.arange(len(avg_importance))
            colors = ['coral' if 16 <= i < 20 else 'steelblue' for i in positions]
            
            bars = ax.bar(positions, avg_importance, color=colors, alpha=0.7, edgecolor='black')
            
            # Labels
            ax.set_xlabel('Position in sgRNA', fontsize=12, fontweight='bold')
            ax.set_ylabel('Attention Importance', fontsize=12, fontweight='bold')
            ax.set_title('Seed Region Importance in sgRNA', fontsize=14, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels([str(i) for i in positions])
            
            # Add seed region annotation
            ax.axvspan(15.5, 19.5, alpha=0.2, color='red')
            ax.text(17.5, ax.get_ylim()[1] * 0.95, 'Seed Region\n(Positions 16-20)',
                   ha='center', fontsize=11, fontweight='bold', color='red')
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Seed region importance plot saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to visualize seed region importance: {e}")


# Convenience functions

def extract_attention_weights(model,
                             sgrna_onehot: torch.Tensor,
                             sgrna_label: torch.Tensor,
                             device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Extract attention weights from model
    
    Args:
        model: CRISPRUniPredict model
        sgrna_onehot: One-hot encoded sgRNA
        sgrna_label: Label encoded sgRNA
        device: Device to use
    
    Returns:
        Dictionary with attention weights
    """
    visualizer = AttentionVisualizer(model, device)
    return visualizer.extract_attention_weights(sgrna_onehot, sgrna_label)


def plot_attention_heatmap(attention_weights: torch.Tensor,
                          sequence: str,
                          output_path: Optional[Path] = None,
                          title: str = 'Attention Heatmap') -> None:
    """
    Plot attention heatmap
    
    Args:
        attention_weights: Attention weights
        sequence: DNA sequence
        output_path: Path to save figure
        title: Figure title
    """
    visualizer = AttentionVisualizer(None)
    visualizer.plot_attention_heatmap(attention_weights, sequence, output_path, title)


def compare_attention_patterns(model,
                              good_sgrnas: List[Tuple[str, str]],
                              poor_sgrnas: List[Tuple[str, str]],
                              output_dir: Optional[Path] = None,
                              device: str = 'cuda') -> None:
    """
    Compare attention patterns
    
    Args:
        model: CRISPRUniPredict model
        good_sgrnas: High efficiency sgRNAs
        poor_sgrnas: Low efficiency sgRNAs
        output_dir: Output directory
        device: Device to use
    """
    visualizer = AttentionVisualizer(model, device)
    visualizer.compare_attention_patterns(good_sgrnas, poor_sgrnas, output_dir)


def visualize_seed_region_importance(model,
                                    sgrnas: List[str],
                                    output_path: Optional[Path] = None,
                                    device: str = 'cuda') -> None:
    """
    Visualize seed region importance
    
    Args:
        model: CRISPRUniPredict model
        sgrnas: List of sgRNA sequences
        output_path: Path to save figure
        device: Device to use
    """
    visualizer = AttentionVisualizer(model, device)
    visualizer.visualize_seed_region_importance(sgrnas, output_path)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("ATTENTION VISUALIZATION TESTING")
    print("=" * 80)
    
    print("\n[INFO] Attention visualization module created successfully")
    print("[INFO] To use the attention visualizer:")
    print("  1. Initialize: visualizer = AttentionVisualizer(model, device='cuda')")
    print("  2. Extract weights: weights = visualizer.extract_attention_weights(onehot, label)")
    print("  3. Plot heatmap: visualizer.plot_attention_heatmap(weights, sequence)")
    print("  4. Compare patterns: visualizer.compare_attention_patterns(good, poor)")
    print("  5. Seed importance: visualizer.visualize_seed_region_importance(sgrnas)")
    
    print("\n" + "=" * 80)
    print("[OK] Attention visualization module ready for use")
    print("=" * 80)
