"""
PyTorch Connectomics models module.

Clean, modern interface using MONAI and nnUNet models.
"""

from .build import build_model, update_state_dict

# Export loss functions
from .loss import (
    create_loss,
    create_combined_loss,
    create_loss_from_config,
    create_binary_segmentation_loss,
    create_multiclass_segmentation_loss,
)

__all__ = [
    # Model building
    "build_model",
    "update_state_dict",
    # Loss functions
    "create_loss",
    "create_combined_loss",
    "create_loss_from_config",
    "create_binary_segmentation_loss",
    "create_multiclass_segmentation_loss",
]
