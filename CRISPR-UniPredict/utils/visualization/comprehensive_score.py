"""
Comprehensive sgRNA Score Visualization
Visualizes the novel comprehensive score combining on-target and off-target predictions
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive plots will be skipped.")

logger = logging.getLogger(__name__)


class ComprehensiveScoreVisualizer:
    """
    Visualizes the comprehensive sgRNA score
    
    Comprehensive Score = On_target_efficiency × (1 - Off_target_risk)
    
    This combines on-target efficiency with off-target safety to provide
    a unified metric for sgRNA quality
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize visualizer
        
        Args:
            device: Device to use
        """
        self.device = device
    
    @staticmethod
    def compute_comprehensive_score(on_target_pred: np.ndarray,
                                   off_target_pred: np.ndarray) -> np.ndarray:
        """
        Compute comprehensive score
        
        Formula: Score = On_target × (1 - Off_target)
        
        Args:
            on_target_pred: On-target efficiency predictions (0-1)
            off_target_pred: Off-target probability predictions (0-1)
        
        Returns:
            Comprehensive scores (0-1)
        """
        on_target_pred = np.asarray(on_target_pred).squeeze()
        off_target_pred = np.asarray(off_target_pred).squeeze()
        
        # Ensure same length
        if len(on_target_pred) != len(off_target_pred):
            raise ValueError("Predictions must have same length")
        
        # Compute comprehensive score
        off_target_safety = 1.0 - off_target_pred
        comprehensive_score = on_target_pred * off_target_safety
        
        return comprehensive_score
    
    def plot_2d_scatter(self,
                       on_target_pred: np.ndarray,
                       off_target_pred: np.ndarray,
                       sgrnas: Optional[List[str]] = None,
                       output_path: Optional[Path] = None) -> None:
        """
        Plot 2D scatter: on-target vs off-target safety
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            sgrnas: List of sgRNA sequences (optional)
            output_path: Path to save figure
        """
        try:
            on_target_pred = np.asarray(on_target_pred).squeeze()
            off_target_pred = np.asarray(off_target_pred).squeeze()
            
            # Compute comprehensive score
            off_target_safety = 1.0 - off_target_pred
            comprehensive_score = self.compute_comprehensive_score(on_target_pred, off_target_pred)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create scatter plot
            scatter = ax.scatter(
                on_target_pred,
                off_target_safety,
                c=comprehensive_score,
                s=100,
                cmap='RdYlGn',
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5,
                vmin=0,
                vmax=1
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Comprehensive Score', fontsize=12, fontweight='bold')
            
            # Add quadrant lines
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Annotate quadrants
            ax.text(0.75, 0.75, 'Optimal\n(High efficiency\nHigh safety)',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            ax.text(0.25, 0.75, 'Safe but\nInefficient',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            ax.text(0.75, 0.25, 'Efficient but\nRisky',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            ax.text(0.25, 0.25, 'Poor\n(Low efficiency\nHigh risk)',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            # Labels
            ax.set_xlabel('On-Target Efficiency', fontsize=12, fontweight='bold')
            ax.set_ylabel('Off-Target Safety (1 - Risk)', fontsize=12, fontweight='bold')
            ax.set_title('Comprehensive sgRNA Score Analysis', fontsize=14, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"2D scatter plot saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot 2D scatter: {e}")
    
    def plot_score_distribution(self,
                               on_target_pred: np.ndarray,
                               off_target_pred: np.ndarray,
                               dataset_names: Optional[List[str]] = None,
                               output_path: Optional[Path] = None) -> None:
        """
        Plot comprehensive score distribution
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            dataset_names: Names of datasets (for comparison)
            output_path: Path to save figure
        """
        try:
            on_target_pred = np.asarray(on_target_pred).squeeze()
            off_target_pred = np.asarray(off_target_pred).squeeze()
            
            # Compute comprehensive score
            comprehensive_score = self.compute_comprehensive_score(on_target_pred, off_target_pred)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot histogram
            n, bins, patches = ax.hist(comprehensive_score, bins=50, color='steelblue', 
                                       alpha=0.7, edgecolor='black')
            
            # Color code by threshold
            for i, patch in enumerate(patches):
                if bins[i] < 0.3:
                    patch.set_facecolor('lightcoral')
                elif bins[i] < 0.6:
                    patch.set_facecolor('lightyellow')
                else:
                    patch.set_facecolor('lightgreen')
            
            # Add threshold lines
            ax.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Poor threshold (0.3)')
            ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2, label='Good threshold (0.6)')
            
            # Add statistics
            mean_score = comprehensive_score.mean()
            median_score = np.median(comprehensive_score)
            
            ax.axvline(x=mean_score, color='blue', linestyle='-', linewidth=2, label=f'Mean ({mean_score:.3f})')
            ax.axvline(x=median_score, color='purple', linestyle='-', linewidth=2, label=f'Median ({median_score:.3f})')
            
            # Labels
            ax.set_xlabel('Comprehensive Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Comprehensive sgRNA Scores', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add text annotations
            poor_count = (comprehensive_score < 0.3).sum()
            medium_count = ((comprehensive_score >= 0.3) & (comprehensive_score < 0.6)).sum()
            good_count = (comprehensive_score >= 0.6).sum()
            
            textstr = f'Poor (<0.3): {poor_count}\nMedium (0.3-0.6): {medium_count}\nGood (≥0.6): {good_count}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Score distribution plot saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot score distribution: {e}")
    
    def plot_ranking_comparison(self,
                               on_target_pred: np.ndarray,
                               off_target_pred: np.ndarray,
                               sgrnas: List[str],
                               top_n: int = 10,
                               output_path: Optional[Path] = None) -> None:
        """
        Compare rankings by different metrics
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            sgrnas: List of sgRNA sequences
            top_n: Number of top sgRNAs to show
            output_path: Path to save figure
        """
        try:
            on_target_pred = np.asarray(on_target_pred).squeeze()
            off_target_pred = np.asarray(off_target_pred).squeeze()
            
            # Compute comprehensive score
            comprehensive_score = self.compute_comprehensive_score(on_target_pred, off_target_pred)
            off_target_safety = 1.0 - off_target_pred
            
            # Get top N by each metric
            top_on_target_idx = np.argsort(on_target_pred)[-top_n:][::-1]
            top_off_target_idx = np.argsort(off_target_safety)[-top_n:][::-1]
            top_comprehensive_idx = np.argsort(comprehensive_score)[-top_n:][::-1]
            
            # Create comparison dataframe
            comparison_data = []
            
            for rank, idx in enumerate(top_comprehensive_idx, 1):
                sgrna = sgrnas[idx] if idx < len(sgrnas) else f"sgRNA_{idx}"
                on_target_rank = np.where(top_on_target_idx == idx)[0]
                on_target_rank = on_target_rank[0] + 1 if len(on_target_rank) > 0 else top_n + 1
                
                off_target_rank = np.where(top_off_target_idx == idx)[0]
                off_target_rank = off_target_rank[0] + 1 if len(off_target_rank) > 0 else top_n + 1
                
                comparison_data.append({
                    'Rank': rank,
                    'sgRNA': sgrna[:10] + '...' if len(sgrna) > 10 else sgrna,
                    'On-Target Rank': on_target_rank,
                    'Off-Target Rank': off_target_rank,
                    'Comprehensive Score': comprehensive_score[idx]
                })
            
            df = pd.DataFrame(comparison_data)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            
            # Plot 1: On-target rankings
            axes[0].barh(range(len(df)), df['On-Target Rank'], color='steelblue', alpha=0.7)
            axes[0].set_yticks(range(len(df)))
            axes[0].set_yticklabels(df['sgRNA'], fontsize=9)
            axes[0].set_xlabel('Rank by On-Target Efficiency', fontsize=11, fontweight='bold')
            axes[0].set_title('On-Target Only Rankings', fontsize=12, fontweight='bold')
            axes[0].invert_xaxis()
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Off-target rankings
            axes[1].barh(range(len(df)), df['Off-Target Rank'], color='coral', alpha=0.7)
            axes[1].set_yticks(range(len(df)))
            axes[1].set_yticklabels(df['sgRNA'], fontsize=9)
            axes[1].set_xlabel('Rank by Off-Target Safety', fontsize=11, fontweight='bold')
            axes[1].set_title('Off-Target Only Rankings', fontsize=12, fontweight='bold')
            axes[1].invert_xaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Comprehensive rankings
            colors = plt.cm.RdYlGn(df['Comprehensive Score'])
            axes[2].barh(range(len(df)), df['Rank'], color=colors, alpha=0.7)
            axes[2].set_yticks(range(len(df)))
            axes[2].set_yticklabels(df['sgRNA'], fontsize=9)
            axes[2].set_xlabel('Rank by Comprehensive Score', fontsize=11, fontweight='bold')
            axes[2].set_title('Comprehensive Score Rankings', fontsize=12, fontweight='bold')
            axes[2].invert_xaxis()
            axes[2].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Ranking comparison plot saved to {output_path}")
            
            plt.show()
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot ranking comparison: {e}")
    
    def plot_interactive_scatter(self,
                                on_target_pred: np.ndarray,
                                off_target_pred: np.ndarray,
                                sgrnas: List[str],
                                output_path: Optional[Path] = None) -> None:
        """
        Plot interactive scatter using Plotly
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            sgrnas: List of sgRNA sequences
            output_path: Path to save figure
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping interactive plot.")
            return
        
        try:
            on_target_pred = np.asarray(on_target_pred).squeeze()
            off_target_pred = np.asarray(off_target_pred).squeeze()
            
            # Compute comprehensive score
            off_target_safety = 1.0 - off_target_pred
            comprehensive_score = self.compute_comprehensive_score(on_target_pred, off_target_pred)
            
            # Create dataframe
            df = pd.DataFrame({
                'On-Target Efficiency': on_target_pred,
                'Off-Target Safety': off_target_safety,
                'Comprehensive Score': comprehensive_score,
                'sgRNA': sgrnas[:len(on_target_pred)]
            })
            
            # Create interactive scatter
            fig = px.scatter(
                df,
                x='On-Target Efficiency',
                y='Off-Target Safety',
                color='Comprehensive Score',
                hover_name='sgRNA',
                hover_data={
                    'On-Target Efficiency': ':.3f',
                    'Off-Target Safety': ':.3f',
                    'Comprehensive Score': ':.3f'
                },
                color_continuous_scale='RdYlGn',
                size_max=15,
                title='Interactive Comprehensive sgRNA Score Analysis'
            )
            
            # Add quadrant lines
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Update layout
            fig.update_layout(
                width=1000,
                height=800,
                font=dict(size=12),
                hovermode='closest'
            )
            
            # Save
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(output_path))
                logger.info(f"Interactive plot saved to {output_path}")
            
            fig.show()
        
        except Exception as e:
            logger.error(f"Failed to plot interactive scatter: {e}")
    
    def generate_score_report(self,
                             on_target_pred: np.ndarray,
                             off_target_pred: np.ndarray,
                             sgrnas: List[str],
                             output_dir: Path) -> None:
        """
        Generate comprehensive score report
        
        Args:
            on_target_pred: On-target predictions
            off_target_pred: Off-target predictions
            sgrnas: List of sgRNA sequences
            output_dir: Output directory
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            on_target_pred = np.asarray(on_target_pred).squeeze()
            off_target_pred = np.asarray(off_target_pred).squeeze()
            
            # Compute comprehensive score
            comprehensive_score = self.compute_comprehensive_score(on_target_pred, off_target_pred)
            off_target_safety = 1.0 - off_target_pred
            
            # Create report
            report = []
            report.append("=" * 80)
            report.append("COMPREHENSIVE sgRNA SCORE ANALYSIS REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Score statistics
            report.append("1. COMPREHENSIVE SCORE STATISTICS")
            report.append("-" * 80)
            report.append(f"Mean Score: {comprehensive_score.mean():.4f}")
            report.append(f"Median Score: {np.median(comprehensive_score):.4f}")
            report.append(f"Std Dev: {comprehensive_score.std():.4f}")
            report.append(f"Min Score: {comprehensive_score.min():.4f}")
            report.append(f"Max Score: {comprehensive_score.max():.4f}")
            
            # Score distribution
            report.append("\n2. SCORE DISTRIBUTION")
            report.append("-" * 80)
            
            poor_count = (comprehensive_score < 0.3).sum()
            medium_count = ((comprehensive_score >= 0.3) & (comprehensive_score < 0.6)).sum()
            good_count = (comprehensive_score >= 0.6).sum()
            
            report.append(f"Poor (<0.3): {poor_count} ({100*poor_count/len(comprehensive_score):.1f}%)")
            report.append(f"Medium (0.3-0.6): {medium_count} ({100*medium_count/len(comprehensive_score):.1f}%)")
            report.append(f"Good (≥0.6): {good_count} ({100*good_count/len(comprehensive_score):.1f}%)")
            
            # Top sgRNAs
            report.append("\n3. TOP 10 sgRNAs BY COMPREHENSIVE SCORE")
            report.append("-" * 80)
            
            top_indices = np.argsort(comprehensive_score)[-10:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                sgrna = sgrnas[idx] if idx < len(sgrnas) else f"sgRNA_{idx}"
                report.append(
                    f"{rank:2d}. {sgrna:23s} | "
                    f"On-Target: {on_target_pred[idx]:.4f} | "
                    f"Off-Target Safety: {off_target_safety[idx]:.4f} | "
                    f"Score: {comprehensive_score[idx]:.4f}"
                )
            
            # Score formula explanation
            report.append("\n4. SCORE FORMULA")
            report.append("-" * 80)
            report.append("Comprehensive Score = On-Target Efficiency × (1 - Off-Target Risk)")
            report.append("")
            report.append("This formula combines:")
            report.append("• On-Target Efficiency: How well sgRNA cuts at target site (0-1)")
            report.append("• Off-Target Safety: How safe sgRNA is (1 - off-target risk)")
            report.append("")
            report.append("Interpretation:")
            report.append("• Score > 0.6: Excellent sgRNA (high efficiency, high safety)")
            report.append("• Score 0.3-0.6: Acceptable sgRNA (trade-off between efficiency and safety)")
            report.append("• Score < 0.3: Poor sgRNA (low efficiency or high off-target risk)")
            
            # Recommendations
            report.append("\n5. RECOMMENDATIONS")
            report.append("-" * 80)
            
            if good_count > 0:
                report.append(f"✓ {good_count} high-quality sgRNAs available (score ≥ 0.6)")
                report.append("  Recommendation: Use these sgRNAs for experiments")
            else:
                report.append("✗ No high-quality sgRNAs found (score ≥ 0.6)")
                report.append("  Recommendation: Consider design alternatives or accept trade-offs")
            
            if medium_count > 0:
                report.append(f"• {medium_count} medium-quality sgRNAs available (0.3 ≤ score < 0.6)")
                report.append("  Recommendation: Use if high-quality options unavailable")
            
            report.append("\n" + "=" * 80)
            
            # Save report
            report_text = "\n".join(report)
            report_path = output_dir / 'comprehensive_score_report.txt'
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Report saved to {report_path}")
            print(report_text)
        
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


# Convenience functions

def compute_comprehensive_score(on_target_pred: np.ndarray,
                               off_target_pred: np.ndarray) -> np.ndarray:
    """Compute comprehensive score"""
    return ComprehensiveScoreVisualizer.compute_comprehensive_score(on_target_pred, off_target_pred)


def plot_2d_scatter(on_target_pred: np.ndarray,
                   off_target_pred: np.ndarray,
                   sgrnas: Optional[List[str]] = None,
                   output_path: Optional[Path] = None) -> None:
    """Plot 2D scatter"""
    visualizer = ComprehensiveScoreVisualizer()
    visualizer.plot_2d_scatter(on_target_pred, off_target_pred, sgrnas, output_path)


def plot_score_distribution(on_target_pred: np.ndarray,
                           off_target_pred: np.ndarray,
                           output_path: Optional[Path] = None) -> None:
    """Plot score distribution"""
    visualizer = ComprehensiveScoreVisualizer()
    visualizer.plot_score_distribution(on_target_pred, off_target_pred, output_path=output_path)


def plot_ranking_comparison(on_target_pred: np.ndarray,
                           off_target_pred: np.ndarray,
                           sgrnas: List[str],
                           top_n: int = 10,
                           output_path: Optional[Path] = None) -> None:
    """Plot ranking comparison"""
    visualizer = ComprehensiveScoreVisualizer()
    visualizer.plot_ranking_comparison(on_target_pred, off_target_pred, sgrnas, top_n, output_path)


def generate_comprehensive_report(on_target_pred: np.ndarray,
                                 off_target_pred: np.ndarray,
                                 sgrnas: List[str],
                                 output_dir: Path) -> None:
    """Generate comprehensive report"""
    visualizer = ComprehensiveScoreVisualizer()
    visualizer.plot_2d_scatter(on_target_pred, off_target_pred, sgrnas, 
                              output_dir / '2d_scatter.png')
    visualizer.plot_score_distribution(on_target_pred, off_target_pred,
                                      output_dir / 'score_distribution.png')
    visualizer.plot_ranking_comparison(on_target_pred, off_target_pred, sgrnas,
                                      output_path=output_dir / 'ranking_comparison.png')
    visualizer.plot_interactive_scatter(on_target_pred, off_target_pred, sgrnas,
                                       output_path=output_dir / 'interactive_scatter.html')
    visualizer.generate_score_report(on_target_pred, off_target_pred, sgrnas, output_dir)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE SCORE VISUALIZATION TESTING")
    print("=" * 80)
    
    print("\n[INFO] Comprehensive score visualization module created successfully")
    print("[INFO] To use the comprehensive score visualizer:")
    print("  1. Initialize: visualizer = ComprehensiveScoreVisualizer()")
    print("  2. Compute score: score = visualizer.compute_comprehensive_score(on_target, off_target)")
    print("  3. Plot 2D scatter: visualizer.plot_2d_scatter(on_target, off_target)")
    print("  4. Plot distribution: visualizer.plot_score_distribution(on_target, off_target)")
    print("  5. Compare rankings: visualizer.plot_ranking_comparison(on_target, off_target, sgrnas)")
    print("  6. Interactive plot: visualizer.plot_interactive_scatter(on_target, off_target, sgrnas)")
    print("  7. Generate report: visualizer.generate_score_report(on_target, off_target, sgrnas, output_dir)")
    
    print("\n" + "=" * 80)
    print("[OK] Comprehensive score visualization module ready for use")
    print("=" * 80)
