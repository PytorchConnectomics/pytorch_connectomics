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

from .metadata import (
    LossMetadata,
    attach_loss_metadata,
    get_loss_metadata,
    get_loss_metadata_for_module,
)

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
    PerChannelBCEWithLogitsLoss,
    WeightedMSELoss,
    WeightedMAELoss,
    SmoothL1Loss,
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


class CombinedLoss(nn.Module):
    """Weighted combination of multiple loss functions."""

    def __init__(
        self,
        loss_fns: List[nn.Module],
        weights: List[float],
        loss_names: List[str],
    ):
        super().__init__()
        self.loss_fns = nn.ModuleList(loss_fns)
        self.weights = weights
        self.loss_names = loss_names

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss

    def __repr__(self):
        loss_str = ", ".join(
            [f"{name}(weight={w:.2f})" for name, w in zip(self.loss_names, self.weights)]
        )
        return f"CombinedLoss({loss_str})"


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
        loss_kwargs = [{} for _ in range(len(loss_names))]

    if len(loss_names) != len(loss_kwargs):
        raise ValueError(
            f"Number of loss names ({len(loss_names)}) must match "
            f"number of kwargs ({len(loss_kwargs)})"
        )

    # Single loss - no need for wrapper
    if len(loss_names) == 1:
        return create_loss(loss_names[0], **loss_kwargs[0])

    # Create individual loss functions
    loss_fns = []
    for loss_name, kwargs in zip(loss_names, loss_kwargs):
        loss_fns.append(create_loss(loss_name, **kwargs))

    return CombinedLoss(loss_fns, loss_weights, loss_names)


def create_loss_from_config(cfg) -> nn.Module:
    """
    Create loss function from Hydra config.

    Args:
        cfg: Hydra Config object with model.loss.losses list

    Returns:
        Initialized loss function

    Examples:
        >>> from connectomics.config import load_config
        >>> cfg = load_config('config.yaml')
        >>> loss = create_loss_from_config(cfg)
    """
    loss_cfg = getattr(cfg.model, "loss", None)
    losses = getattr(loss_cfg, "losses", None)
    if losses is None:
        losses = [
            {"function": "DiceLoss", "weight": 1.0},
            {"function": "BCEWithLogitsLoss", "weight": 1.0},
        ]

    loss_names = [entry["function"] for entry in losses]
    loss_weights = [float(entry.get("weight", 1.0)) for entry in losses]
    loss_kwargs = [dict(entry.get("kwargs", {})) for entry in losses]

    return create_combined_loss(
        loss_names=loss_names,
        loss_weights=loss_weights,
        loss_kwargs=loss_kwargs,
    )


def list_available_losses() -> List[str]:
    """List all available loss functions."""
    return list(_get_loss_registry().keys())


__all__ = [
    "create_loss",
    "create_combined_loss",
    "create_loss_from_config",
    "list_available_losses",
    "LossMetadata",
    "get_loss_metadata",
    "get_loss_metadata_for_module",
]
