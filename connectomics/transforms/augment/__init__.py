"""
MONAI-native augmentation interface for PyTorch Connectomics.

This module provides pure MONAI transforms for connectomics-specific
data augmentation, enabling seamless integration with MONAI Compose pipelines.
"""

# MONAI-native augmentation interface
from .monai_augmentor import build_augmentor, create_inference_transforms
from .monai_transforms import (
    RandMisAlignmentd,
    RandMissingSectiond,
    RandMissingPartsd,
    RandMotionBlurd,
    RandCutNoised,
    RandCutBlurd
)

__all__ = [
    # Factory functions for building augmentation pipelines
    'build_augmentor',
    'create_inference_transforms',

    # Connectomics-specific MONAI transforms (not in standard MONAI)
    'RandMisAlignmentd',
    'RandMissingSectiond',
    'RandMissingPartsd',
    'RandMotionBlurd',
    'RandCutNoised',
    'RandCutBlurd',
]