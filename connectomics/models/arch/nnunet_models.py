"""
nnUNet model integration for PyTorch Connectomics.

This module provides nnUNet model implementations to replace custom architectures
with proven, state-of-the-art models for medical image segmentation.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# nnUNet imports (conditional)
try:
    from nnunet_mednext import create_mednext_v1
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    warnings.warn("nnUNet MedNeXt not available. Install with: pip install nnunet-mednext")

# Alternative nnUNet imports
try:
    from monai.networks.nets import BasicUNet, UNet, UNETR, SwinUNETR
    MONAI_NNUNET_AVAILABLE = True
except ImportError:
    MONAI_NNUNET_AVAILABLE = False


class MedNeXtUNet(nn.Module):
    """
    MedNeXt U-Net from nnUNet framework.

    This is a modern, optimized U-Net variant that has shown excellent
    performance on medical imaging tasks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        model_id: Model size ('S', 'B', 'M', 'L')
        kernel_size: Convolution kernel size
        deep_supervision: Whether to use deep supervision
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 model_id: str = 'S',
                 kernel_size: int = 3,
                 deep_supervision: bool = False,
                 **kwargs):
        super().__init__()

        if not NNUNET_AVAILABLE:
            raise ImportError("nnUNet MedNeXt not available. Install with: pip install nnunet-mednext")

        self.deep_supervision = deep_supervision

        # Create MedNeXt model
        self.model = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass."""
        return self.model(x)


class MonaiUNet3D(nn.Module):
    """
    MONAI 3D U-Net implementation.

    High-performance 3D U-Net implementation from MONAI with optimizations
    for medical imaging tasks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: Feature map sizes for each level
        strides: Strides for downsampling
        kernel_size: Convolution kernel size
        num_res_units: Number of residual units
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 strides: Tuple[int, ...] = (1, 2, 2, 2),
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 num_res_units: int = 2,
                 **kwargs):
        super().__init__()

        if not MONAI_NNUNET_AVAILABLE:
            raise ImportError("MONAI not available. Install with: pip install monai")

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            norm='batch',
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class MonaiBasicUNet3D(nn.Module):
    """
    MONAI Basic U-Net 3D implementation.

    Lightweight U-Net implementation suitable for smaller datasets
    or resource-constrained environments.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: Feature map sizes
        dropout: Dropout probability
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: Tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        if not MONAI_NNUNET_AVAILABLE:
            raise ImportError("MONAI not available. Install with: pip install monai")

        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class MonaiUNETR(nn.Module):
    """
    MONAI UNETR (U-Net Transformer) implementation.

    Vision Transformer-based U-Net that combines the strengths of
    transformers and U-Net for medical image segmentation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Input image size
        feature_size: Feature embedding size
        hidden_size: Hidden layer size
        mlp_dim: MLP dimension
        num_heads: Number of attention heads
        norm_name: Normalization layer name
        dropout_rate: Dropout rate
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 img_size: Tuple[int, int, int] = (128, 128, 128),
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 norm_name: str = "instance",
                 dropout_rate: float = 0.0,
                 **kwargs):
        super().__init__()

        if not MONAI_NNUNET_AVAILABLE:
            raise ImportError("MONAI not available. Install with: pip install monai")

        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            norm_name=norm_name,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class MonaiSwinUNETR(nn.Module):
    """
    MONAI Swin UNETR implementation.

    Swin Transformer-based U-Net that uses hierarchical vision transformers
    for improved performance on medical image segmentation tasks.

    Args:
        img_size: Input image size
        in_channels: Number of input channels
        out_channels: Number of output channels
        feature_size: Feature size
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        dropout_path_rate: Dropout path rate
        use_checkpoint: Whether to use gradient checkpointing
    """

    def __init__(self,
                 img_size: Tuple[int, int, int] = (128, 128, 128),
                 in_channels: int = 1,
                 out_channels: int = 1,
                 feature_size: int = 48,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 **kwargs):
        super().__init__()

        if not MONAI_NNUNET_AVAILABLE:
            raise ImportError("MONAI not available. Install with: pip install monai")

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


# Model registry for easy access
NNUNET_MODEL_REGISTRY = {
    'mednext_s': lambda **kwargs: MedNeXtUNet(model_id='S', **kwargs),
    'mednext_b': lambda **kwargs: MedNeXtUNet(model_id='B', **kwargs),
    'mednext_m': lambda **kwargs: MedNeXtUNet(model_id='M', **kwargs),
    'mednext_l': lambda **kwargs: MedNeXtUNet(model_id='L', **kwargs),
    'monai_unet3d': MonaiUNet3D,
    'monai_basic_unet3d': MonaiBasicUNet3D,
    'monai_unetr': MonaiUNETR,
    'monai_swin_unetr': MonaiSwinUNETR,
}


def create_nnunet_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create nnUNet models.

    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific arguments

    Returns:
        Initialized model
    """
    if model_name not in NNUNET_MODEL_REGISTRY:
        available_models = list(NNUNET_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")

    return NNUNET_MODEL_REGISTRY[model_name](**kwargs)


def list_available_nnunet_models() -> List[str]:
    """List all available nnUNet models."""
    available = []

    for model_name in NNUNET_MODEL_REGISTRY.keys():
        try:
            # Try to create the model to check if dependencies are available
            create_nnunet_model(model_name, in_channels=1, out_channels=1)
            available.append(model_name)
        except ImportError:
            continue

    return available