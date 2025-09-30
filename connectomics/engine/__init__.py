"""
Training engine components for PyTorch Connectomics.

This package provides training/testing infrastructure:
- logger.py: Training monitoring and logging (TensorBoard, etc.)
- metrics.py: Evaluation metrics (Adapted Rand, Dice, etc.)

Note: PyTorch Lightning trainer/callbacks are in the lightning/ package.

Import patterns:
    from connectomics.engine.logger import build_monitor
    from connectomics.engine.metrics import adapted_rand, get_binary_jaccard
"""

from .logger import *
from .metrics import *

__all__ = [
    # Logger
    'build_monitor',
    'Monitor',
    
    # Metrics
    'get_binary_jaccard',
    'adapted_rand',
    'instance_matching',
]
