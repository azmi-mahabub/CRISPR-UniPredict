"""
CRISPR-UniPredict: Unified Framework for CRISPR-Cas9 sgRNA Design

A comprehensive framework combining:
- CCLMoff: Off-target prediction using pretrained RNA language models
- CRISPR_HNN: On-target activity prediction using hybrid neural networks
"""

__version__ = "1.0.0"
__author__ = "CRISPR-UniPredict Team"

from . import utils

__all__ = ['utils', '__version__', '__author__']
