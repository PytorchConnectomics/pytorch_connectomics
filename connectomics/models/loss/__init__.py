"""
MONAI-native loss functions for PyTorch Connectomics.

This module provides a clean interface for loss function creation using MONAI's
native implementations, with additional connectomics-specific losses.

Design pattern follows transforms/augment/ for consistency.
"""

# Main factory functions (recommended interface)
from .build import (
    create_loss,
    create_combined_loss,
    create_loss_from_config,
    create_binary_segmentation_loss,
    create_multiclass_segmentation_loss,
    create_focal_loss,
    list_available_losses,
)

# Connectomics-specific losses (for direct use if needed)
from .losses import (
    WeightedMSELoss,
    WeightedMAELoss,
    GANLoss,
)

# Regularization losses
from .regularization import (
    BinaryRegularization,
    ForegroundDistanceConsistency,
    ContourDistanceConsistency,
    ForegroundContourConsistency,
    NonOverlapRegularization,
)

# MONAI losses can be imported directly from monai.losses if needed
# from monai.losses import DiceLoss, DiceCELoss, FocalLoss, etc.

__all__ = [
    # Factory functions (primary interface)
    "create_loss",
    "create_combined_loss",
    "create_loss_from_config",
    # Convenience factory functions
    "create_binary_segmentation_loss",
    "create_multiclass_segmentation_loss",
    "create_focal_loss",
    # Utility
    "list_available_losses",
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
