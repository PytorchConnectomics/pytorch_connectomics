"""
PyTorch Connectomics models module.

Clean, modern interface using MONAI and nnUNet models.
"""

from .build import build_model

# Export canonical loss factory
from .losses import create_loss

__all__ = [
    # Model building
    "build_model",
    # Loss functions
    "create_loss",
]
