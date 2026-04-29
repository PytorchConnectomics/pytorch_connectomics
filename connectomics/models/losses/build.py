"""
MONAI-native loss functions for PyTorch Connectomics.

This module provides loss function composition using MONAI's native losses,
with additional connectomics-specific loss functions as needed.

Design pattern aligned with the rest of the normalized package layout.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

# Import MONAI losses
from monai.losses import (
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    FocalLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
)

# Import custom connectomics losses
from .losses import (
    CrossEntropyLossWrapper,
    GANLoss,
    PerChannelBCEWithLogitsLoss,
    SmoothL1Loss,
    WeightedBCEWithLogitsLoss,
    WeightedMAELoss,
    WeightedMSELoss,
)
from .metadata import (
    LossMetadata,
    attach_loss_metadata,
    get_loss_metadata,
    get_loss_metadata_for_module,
)

# Import regularization losses
from .regularization import (
    BinaryRegularization,
    ContourDistanceConsistency,
    ForegroundContourConsistency,
    ForegroundDistanceConsistency,
    NonOverlapRegularization,
)


def _get_loss_registry() -> Dict[str, type[nn.Module]]:
    """Return the canonical mapping of loss names to constructors."""
    return {
        # MONAI Dice variants
        "DiceLoss": DiceLoss,
        "DiceCELoss": DiceCELoss,
        "DiceFocalLoss": DiceFocalLoss,
        "GeneralizedDiceLoss": GeneralizedDiceLoss,
        # MONAI other losses
        "FocalLoss": FocalLoss,
        "TverskyLoss": TverskyLoss,
        # PyTorch standard losses (for convenience)
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "CrossEntropyLoss": CrossEntropyLossWrapper,  # Use wrapper for shape handling
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "SmoothL1Loss": SmoothL1Loss,
        # Custom connectomics losses
        "WeightedBCEWithLogitsLoss": WeightedBCEWithLogitsLoss,
        "PerChannelBCEWithLogitsLoss": PerChannelBCEWithLogitsLoss,
        "WeightedMSELoss": WeightedMSELoss,
        "WeightedMAELoss": WeightedMAELoss,
        "GANLoss": GANLoss,
        # Regularization losses
        "BinaryRegularization": BinaryRegularization,
        "ForegroundDistanceConsistency": ForegroundDistanceConsistency,
        "ContourDistanceConsistency": ContourDistanceConsistency,
        "ForegroundContourConsistency": ForegroundContourConsistency,
        "NonOverlapRegularization": NonOverlapRegularization,
    }


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Create a single loss function by name.

    Args:
        loss_name: Name of the loss function
        **kwargs: Loss-specific parameters

    Returns:
        Initialized loss function

    Examples:
        >>> loss = create_loss('DiceLoss', include_background=False)
        >>> loss = create_loss('DiceCELoss', to_onehot_y=True, softmax=True)
        >>> loss = create_loss('FocalLoss', gamma=2.0)
    """
    loss_registry = _get_loss_registry()

    if loss_name not in loss_registry:
        available = list(loss_registry.keys())
        raise ValueError(f"Unknown loss: {loss_name}. Available losses: {available}")

    loss_fn = loss_registry[loss_name](**kwargs)
    return attach_loss_metadata(loss_fn, loss_name)


def list_available_losses() -> List[str]:
    """List all available loss functions."""
    return list(_get_loss_registry().keys())


__all__ = [
    "create_loss",
    "list_available_losses",
    "LossMetadata",
    "get_loss_metadata",
    "get_loss_metadata_for_module",
]
