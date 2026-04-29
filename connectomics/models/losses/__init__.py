"""
MONAI-native loss functions for PyTorch Connectomics.

This module provides a clean interface for loss function creation using MONAI's
native implementations, with additional connectomics-specific losses.

Design pattern follows the same package structure used across data processing and augmentation.
"""

# Main factory functions (recommended interface)
from .build import (
    create_loss,
    list_available_losses,
)

# Connectomics-specific losses (for direct use if needed)
from .losses import (
    GANLoss,
    WeightedMAELoss,
    WeightedMSELoss,
)
from .metadata import LossMetadata, get_loss_metadata, get_loss_metadata_for_module

# Regularization losses
from .regularization import (
    BinaryRegularization,
    ContourDistanceConsistency,
    ForegroundContourConsistency,
    ForegroundDistanceConsistency,
    NonOverlapRegularization,
)

# MONAI losses can be imported directly from monai.losses if needed
# from monai.losses import DiceLoss, DiceCELoss, FocalLoss, etc.

__all__ = [
    # Factory functions (primary interface)
    "create_loss",
    # Utility
    "list_available_losses",
    "LossMetadata",
    "get_loss_metadata",
    "get_loss_metadata_for_module",
    # Custom losses
    "WeightedMSELoss",
    "WeightedMAELoss",
    "GANLoss",
    # Regularization losses
    "BinaryRegularization",
    "ForegroundDistanceConsistency",
    "ContourDistanceConsistency",
    "ForegroundContourConsistency",
    "NonOverlapRegularization",
]
