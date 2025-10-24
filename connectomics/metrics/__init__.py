"""
Evaluation metrics for PyTorch Connectomics.

This package provides segmentation evaluation metrics:
- metrics_seg.py: Segmentation metrics (Adapted Rand, Dice, Jaccard, etc.)

Note: PyTorch Lightning handles training monitoring and logging.

Import patterns:
    from connectomics.metrics import adapted_rand, get_binary_jaccard
    from connectomics.metrics.metrics_seg import instance_matching
"""

from .metrics_seg import *

__all__ = [
    # Segmentation metrics
    'jaccard',
    'get_binary_jaccard',
    'adapted_rand',
    'instance_matching',
    'cremi_distance',
]
