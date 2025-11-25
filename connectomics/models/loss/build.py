"""
MONAI-native loss functions for PyTorch Connectomics.

This module provides loss function composition using MONAI's native losses,
with additional connectomics-specific loss functions as needed.

Design pattern inspired by transforms/augment/monai_compose.py.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import torch
import torch.nn as nn

# Import MONAI losses
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    FocalLoss,
    TverskyLoss,
    GeneralizedDiceLoss,
)

# Import custom connectomics losses
from .losses import (
    CrossEntropyLossWrapper,
    WeightedBCEWithLogitsLoss,
    WeightedMSELoss,
    WeightedMAELoss,
    GANLoss,
)

# Import regularization losses
from .regularization import (
    BinaryRegularization,
    ForegroundDistanceConsistency,
    ContourDistanceConsistency,
    ForegroundContourConsistency,
    NonOverlapRegularization,
)


def create_loss(
    loss_name: str,
    **kwargs
) -> nn.Module:
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
    # Map loss names to MONAI/custom loss classes
    loss_registry = {
        # MONAI Dice variants
        'DiceLoss': DiceLoss,
        'DiceCELoss': DiceCELoss,
        'DiceFocalLoss': DiceFocalLoss,
        'GeneralizedDiceLoss': GeneralizedDiceLoss,

        # MONAI other losses
        'FocalLoss': FocalLoss,
        'TverskyLoss': TverskyLoss,

        # PyTorch standard losses (for convenience)
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
        'CrossEntropyLoss': CrossEntropyLossWrapper,  # Use wrapper for shape handling
        'MSELoss': nn.MSELoss,
        'L1Loss': nn.L1Loss,

        # Custom connectomics losses
        'WeightedBCEWithLogitsLoss': WeightedBCEWithLogitsLoss,
        'WeightedMSELoss': WeightedMSELoss,
        'WeightedMAELoss': WeightedMAELoss,
        'GANLoss': GANLoss,

        # Regularization losses
        'BinaryRegularization': BinaryRegularization,
        'ForegroundDistanceConsistency': ForegroundDistanceConsistency,
        'ContourDistanceConsistency': ContourDistanceConsistency,
        'ForegroundContourConsistency': ForegroundContourConsistency,
        'NonOverlapRegularization': NonOverlapRegularization,
    }

    if loss_name not in loss_registry:
        available = list(loss_registry.keys())
        raise ValueError(
            f"Unknown loss: {loss_name}. Available losses: {available}"
        )

    return loss_registry[loss_name](**kwargs)


def create_combined_loss(
    loss_names: List[str],
    loss_weights: Optional[List[float]] = None,
    loss_kwargs: Optional[List[Dict]] = None,
) -> nn.Module:
    """
    Create a weighted combination of multiple loss functions.

    Args:
        loss_names: List of loss function names
        loss_weights: Optional weights for each loss (default: equal weights)
        loss_kwargs: Optional list of kwargs dicts for each loss

    Returns:
        Combined loss module

    Examples:
        >>> loss = create_combined_loss(
        ...     loss_names=['DiceLoss', 'BCEWithLogitsLoss'],
        ...     loss_weights=[1.0, 1.0]
        ... )
        >>> loss = create_combined_loss(
        ...     loss_names=['DiceLoss', 'FocalLoss'],
        ...     loss_weights=[0.5, 0.5],
        ...     loss_kwargs=[
        ...         {'include_background': False},
        ...         {'gamma': 2.0, 'alpha': 0.25}
        ...     ]
        ... )
    """
    # Validate inputs
    if loss_weights is None:
        loss_weights = [1.0] * len(loss_names)

    if len(loss_names) != len(loss_weights):
        raise ValueError(
            f"Number of loss names ({len(loss_names)}) must match "
            f"number of weights ({len(loss_weights)})"
        )

    if loss_kwargs is None:
        loss_kwargs = [{}] * len(loss_names)

    if len(loss_names) != len(loss_kwargs):
        raise ValueError(
            f"Number of loss names ({len(loss_names)}) must match "
            f"number of kwargs ({len(loss_kwargs)})"
        )

    # Single loss - no need for wrapper
    if len(loss_names) == 1:
        return create_loss(loss_names[0], **loss_kwargs[0])

    # Multiple losses - create combined loss
    class CombinedLoss(nn.Module):
        """Weighted combination of multiple loss functions."""

        def __init__(self, loss_fns: List[nn.Module], weights: List[float]):
            super().__init__()
            self.loss_fns = nn.ModuleList(loss_fns)
            self.weights = weights
            self.loss_names = loss_names

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute weighted sum of losses."""
            total_loss = 0.0

            for loss_fn, weight, name in zip(self.loss_fns, self.weights, self.loss_names):
                individual_loss = loss_fn(pred, target)
                total_loss += weight * individual_loss

            return total_loss

        def __repr__(self):
            loss_str = ", ".join([
                f"{name}(weight={w:.2f})"
                for name, w in zip(self.loss_names, self.weights)
            ])
            return f"CombinedLoss({loss_str})"

    # Create individual loss functions
    loss_fns = []
    for loss_name, kwargs in zip(loss_names, loss_kwargs):
        loss_fns.append(create_loss(loss_name, **kwargs))

    return CombinedLoss(loss_fns, loss_weights)


def create_loss_from_config(cfg) -> nn.Module:
    """
    Create loss function from Hydra config.

    Args:
        cfg: Hydra Config object with model.loss_functions and model.loss_weights

    Returns:
        Initialized loss function

    Examples:
        >>> from connectomics.config import load_config
        >>> cfg = load_config('config.yaml')
        >>> loss = create_loss_from_config(cfg)
    """
    loss_names = cfg.model.loss_functions
    loss_weights = cfg.model.loss_weights

    # Check if loss_kwargs is available in config
    loss_kwargs = None
    if hasattr(cfg.model, 'loss_kwargs'):
        loss_kwargs = cfg.model.loss_kwargs

    return create_combined_loss(
        loss_names=loss_names,
        loss_weights=loss_weights,
        loss_kwargs=loss_kwargs,
    )


# Convenience factory functions for common loss configurations
def create_binary_segmentation_loss(
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    include_background: bool = True,
) -> nn.Module:
    """
    Create standard loss for binary segmentation.

    Args:
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss
        include_background: Whether to include background in Dice

    Returns:
        Combined DiceLoss + BCEWithLogitsLoss
    """
    return create_combined_loss(
        loss_names=['DiceLoss', 'BCEWithLogitsLoss'],
        loss_weights=[dice_weight, bce_weight],
        loss_kwargs=[
            {'include_background': include_background, 'sigmoid': True},
            {},
        ]
    )


def create_multiclass_segmentation_loss(
    num_classes: int,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    include_background: bool = False,
) -> nn.Module:
    """
    Create standard loss for multi-class segmentation.

    Args:
        num_classes: Number of classes
        dice_weight: Weight for Dice loss
        ce_weight: Weight for CE loss
        include_background: Whether to include background in Dice

    Returns:
        Combined DiceLoss + CrossEntropyLoss
    """
    return create_combined_loss(
        loss_names=['DiceLoss', 'CrossEntropyLoss'],
        loss_weights=[dice_weight, ce_weight],
        loss_kwargs=[
            {
                'include_background': include_background,
                'to_onehot_y': True,
                'softmax': True,
            },
            {},
        ]
    )


def create_focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> nn.Module:
    """
    Create Focal loss for handling class imbalance.

    Args:
        gamma: Focusing parameter
        alpha: Weighting factor

    Returns:
        FocalLoss
    """
    return create_loss('FocalLoss', gamma=gamma, alpha=alpha)


def list_available_losses() -> List[str]:
    """List all available loss functions."""
    return [
        # MONAI losses
        'DiceLoss', 'DiceCELoss', 'DiceFocalLoss', 'GeneralizedDiceLoss',
        'FocalLoss', 'TverskyLoss',
        # PyTorch losses
        'BCEWithLogitsLoss', 'CrossEntropyLoss', 'MSELoss', 'L1Loss',
        # Custom losses
        'WeightedMSELoss', 'WeightedMAELoss', 'GANLoss',
        # Regularization losses
        'BinaryRegularization', 'ForegroundDistanceConsistency',
        'ContourDistanceConsistency', 'ForegroundContourConsistency',
        'NonOverlapRegularization',
    ]


__all__ = [
    'create_loss',
    'create_combined_loss',
    'create_loss_from_config',
    'create_binary_segmentation_loss',
    'create_multiclass_segmentation_loss',
    'create_focal_loss',
    'list_available_losses',
]