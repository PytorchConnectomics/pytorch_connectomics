"""
MONAI-native augmentation interface for PyTorch Connectomics.

This module provides pure MONAI transforms for connectomics-specific
data augmentation, enabling seamless integration with MONAI Compose pipelines.
"""

# MONAI-native augmentation interface
from .build import (
    build_train_transforms,
    build_val_transforms,
    build_test_transforms,
)
from .transforms import (
    RandMisAlignmentd,
    RandMissingSectiond,
    RandMissingPartsd,
    RandMotionBlurd,
    RandCutNoised,
    RandCutBlurd,
    RandMixupd,
    RandCopyPasted,
)

__all__ = [
    # Factory functions for building augmentation pipelines
    "build_train_transforms",
    "build_val_transforms",
    "build_test_transforms",
    # Connectomics-specific MONAI transforms (not in standard MONAI)
    "RandMisAlignmentd",
    "RandMissingSectiond",
    "RandMissingPartsd",
    "RandMotionBlurd",
    "RandCutNoised",
    "RandCutBlurd",
    "RandMixupd",
    "RandCopyPasted",
]
