"""
Utility functions for PyTorch Connectomics.

This package provides lightweight helpers:
- visualizer.py: Visualization utilities for TensorBoard
- analysis.py: Analysis tools for instance segmentation
- debug_hooks.py: Debugging helpers for NaN detection
- download.py: Dataset download utilities
- errors.py: Error handling and preflight checks

Import patterns:
    from connectomics.utils.visualizer import Visualizer
    from connectomics.utils.errors import preflight_check
    from connectomics.utils.download import download_dataset

Note: Legacy system.py utilities (get_args, init_devices) have been moved to legacy/
      and replaced by Hydra config system. See connectomics.training.lit for modern alternatives.
"""

from .visualizer import *  # noqa: F403

__all__ = [  # noqa: F405
    # Visualizer
    "Visualizer",
    "LightningVisualizer",
]
