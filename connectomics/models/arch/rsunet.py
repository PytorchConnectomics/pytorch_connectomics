"""
Configurable Residual Symmetric U-Net (RSUNet) for connectomics.

Inspired by pytorch-emvision (Kisuk Lee, MIT) but with modern PyTorch design:
- Single configurable class (not multiple separate functions)
- Support for 2D, 3D, and 2D/3D hybrid convolutions
- Flexible normalization (Batch, Group, Instance, None)
- Flexible activation functions (ReLU, LeakyReLU, PReLU, ELU)
- Configurable downsampling factors (anisotropic/isotropic)
- Deep supervision support
- PyTorch Connectomics interface

Key design principles from EMVision:
- Pre-activation residual blocks (BN→Act→Conv)
- Addition-based skip connections (not concatenation)
- Bilinear upsampling for smooth gradients
- Anisotropic convolutions for EM data (1,2,2) by default
"""

import math
from typing import List, Tuple, Union, Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import ConnectomicsModel
from .registry import register_architecture


class BilinearUp3d(nn.Module):
    """
    Caffe-style bilinear upsampling for 3D data.

    Learnable upsampling with fixed bilinear interpolation weights.
    More stable than transposed convolution for small feature maps.
    """

    def __init__(
        self, in_channels: int, out_channels: int, factor: Tuple[int, int, int] = (1, 2, 2)
    ):
        super().__init__()
        assert in_channels == out_channels, "BilinearUp3d requires in_channels == out_channels"
        self.groups = in_channels
        self.factor = factor
        self.kernel_size = [(2 * f) - (f % 2) for f in self.factor]
        self.padding = [int(math.ceil((f - 1) / 2.0)) for f in factor]
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose3d(
            x, self.weight, stride=self.factor, padding=self.padding, groups=self.groups
        )

    def init_weights(self):
        """Initialize bilinear interpolation weights."""
        weight = torch.Tensor(self.groups, 1, *self.kernel_size)
        width = weight.size(-1)
        height = weight.size(-2)
        assert width == height, "Bilinear weight assumes square kernel in HW"
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(height):
                weight[..., h, w] = (1 - abs(w / f - c)) * (1 - abs(h / f - c))
        self.register_buffer("weight", weight)


class NormAct(nn.Module):
    """Normalization + Activation layer."""

    def __init__(
        self,
        channels: int,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        **act_kwargs,
    ):
        super().__init__()

        # Normalization
        if norm == "batch":
            self.norm = nn.BatchNorm3d(channels)
        elif norm == "group":
            # Ensure num_groups divides channels
            num_groups = min(num_groups, channels)
            while channels % num_groups != 0:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm3d(channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization: {norm}")

        # Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            slope = act_kwargs.get("negative_slope", 0.01)
            self.act = nn.LeakyReLU(slope, inplace=True)
        elif activation == "prelu":
            init = act_kwargs.get("init", 0.25)
            self.act = nn.PReLU(init=init)
        elif activation == "elu":
            alpha = act_kwargs.get("alpha", 1.0)
            self.act = nn.ELU(alpha, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x))


class ResBlock(nn.Module):
    """
    Residual block with pre-activation design.

    Architecture: x → Norm→Act→Conv → Norm→Act→Conv → (+) → out
                  ↓_____________________________________↑
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        **act_kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.norm_act1 = NormAct(channels, norm, activation, num_groups, **act_kwargs)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)

        self.norm_act2 = NormAct(channels, norm, activation, num_groups, **act_kwargs)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.norm_act1(x))
        x = self.conv2(self.norm_act2(x))
        return x + residual


class ConvBlock(nn.Module):
    """
    Convolution block: Pre→Residual→Post

    Pattern from EMVision: provides richer features than single ResBlock.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        **act_kwargs,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        # Pre-convolution
        self.pre = nn.Sequential(
            NormAct(in_channels, norm, activation, num_groups, **act_kwargs),
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        )

        # Residual block
        self.res = ResBlock(out_channels, kernel_size, norm, activation, num_groups, **act_kwargs)

        # Post-convolution
        self.post = nn.Sequential(
            NormAct(out_channels, norm, activation, num_groups, **act_kwargs),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.res(x)
        return self.post(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool → ConvBlock"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_factor: Tuple[int, int, int] = (1, 2, 2),
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        **act_kwargs,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(down_factor)
        self.conv = ConvBlock(
            in_channels, out_channels, kernel_size, norm, activation, num_groups, **act_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Upsampling block with skip connection.

    Uses addition (not concatenation) for efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_factor: Tuple[int, int, int] = (1, 2, 2),
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        **act_kwargs,
    ):
        super().__init__()

        # Upsampling
        self.up = BilinearUp3d(in_channels, in_channels, factor=up_factor)

        # 1x1 conv to match channels
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        # Convolution after skip addition
        self.conv = ConvBlock(
            out_channels, out_channels, kernel_size, norm, activation, num_groups, **act_kwargs
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.proj(x) + skip  # Addition, not concatenation
        return self.conv(x)


class RSUNet(ConnectomicsModel):
    """
    Residual Symmetric U-Net with flexible configuration.

    Features:
    - Configurable depth and channel widths
    - Support for 2D, 3D, and 2D/3D hybrid convolutions
    - Flexible normalization (batch/group/instance/none)
    - Flexible activation (relu/leakyrelu/prelu/elu)
    - Anisotropic or isotropic downsampling
    - Deep supervision support
    - Addition-based skip connections (efficient)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        width: Channel width at each level (e.g., [16, 32, 64, 128])
        kernel_sizes: Kernel size per level (int or list of tuples)
        down_factors: Downsampling factors per level (anisotropic/isotropic)
        norm: Normalization type ('batch', 'group', 'instance', 'none')
        activation: Activation function ('relu', 'leakyrelu', 'prelu', 'elu')
        num_groups: Number of groups for GroupNorm (default: 8)
        deep_supervision: Enable multi-scale outputs (default: False)
        depth_2d: Number of shallow layers using 2D convolutions (0 = all 3D)
        kernel_2d: Kernel size for 2D layers (e.g., (1, 3, 3))
        **act_kwargs: Additional activation arguments (e.g., negative_slope for LeakyReLU)

    Example:
        # Anisotropic EM data (default)
        model = RSUNet(1, 2, width=[16, 32, 64, 128])

        # Isotropic data
        model = RSUNet(1, 2, width=[16, 32, 64], down_factors=[(2,2,2)] * 2)

        # 2D/3D hybrid
        model = RSUNet(1, 2, width=[16, 32, 64, 128], depth_2d=2, kernel_2d=(1,3,3))

        # With Group Norm and PReLU
        model = RSUNet(1, 2, width=[16, 32, 64], norm='group', activation='prelu', init=0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: List[int] = [16, 32, 64, 128, 256],
        kernel_sizes: Union[int, List[Union[int, Tuple[int, int, int]]]] = 3,
        down_factors: Optional[List[Tuple[int, int, int]]] = None,
        norm: str = "batch",
        activation: str = "relu",
        num_groups: int = 8,
        deep_supervision: bool = False,
        depth_2d: int = 0,
        kernel_2d: Tuple[int, int, int] = (1, 3, 3),
        **act_kwargs,
    ):
        super().__init__()

        assert len(width) > 1, "Need at least 2 levels"
        self.depth = len(width) - 1
        self.width = width
        self.supports_deep_supervision = deep_supervision
        self.output_scales = 5 if deep_supervision else 1

        # Default: anisotropic downsampling (1,2,2) for EM data
        if down_factors is None:
            down_factors = [(1, 2, 2)] * self.depth
        assert len(down_factors) == self.depth

        # Process kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(width)
        elif len(kernel_sizes) < len(width):
            kernel_sizes = list(kernel_sizes) + [kernel_sizes[-1]] * (
                len(width) - len(kernel_sizes)
            )

        # Handle 2D/3D hybrid
        if depth_2d > 0:
            for i in range(min(depth_2d, len(kernel_sizes))):
                kernel_sizes[i] = kernel_2d

        # Initial convolution
        self.input_conv = ConvBlock(
            in_channels, width[0], kernel_sizes[0], norm, activation, num_groups, **act_kwargs
        )

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        for d in range(self.depth):
            self.down_blocks.append(
                DownBlock(
                    width[d],
                    width[d + 1],
                    down_factors[d],
                    kernel_sizes[d + 1],
                    norm,
                    activation,
                    num_groups,
                    **act_kwargs,
                )
            )

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for d in reversed(range(self.depth)):
            self.up_blocks.append(
                UpBlock(
                    width[d + 1],
                    width[d],
                    down_factors[d],
                    kernel_sizes[d],
                    norm,
                    activation,
                    num_groups,
                    **act_kwargs,
                )
            )

        # Final normalization
        self.final_norm = NormAct(width[0], norm, activation, num_groups, **act_kwargs)

        # Output head
        self.output_head = nn.Conv3d(width[0], out_channels, kernel_size=1)

        # Deep supervision heads (from deeper to shallower)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            # Create heads for deeper levels (before upsampling to final resolution)
            # ds_0 = bottleneck level (width[-1])
            # ds_1 = one level up (width[-2])
            # etc.
            for d in range(min(4, self.depth)):
                level_idx = self.depth - d  # Go from deepest to shallower
                self.ds_heads.append(nn.Conv3d(width[level_idx], out_channels, kernel_size=1))

        # Initialize weights
        self.init_weights()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Single tensor or dict with deep supervision outputs
        """
        # Encoder
        x = self.input_conv(x)

        skips = []
        for down in self.down_blocks:
            skips.append(x)
            x = down(x)

        # Collect deep supervision features (from bottleneck to top)
        if self.supports_deep_supervision:
            ds_features = []

        # Decoder
        for i, up in enumerate(self.up_blocks):
            # For deep supervision, store features from deeper layers
            if self.supports_deep_supervision and (self.depth - i - 1) < len(self.ds_heads):
                ds_features.append(x)

            x = up(x, skips.pop())

        # Final output
        x = self.final_norm(x)
        output = self.output_head(x)

        # Return with or without deep supervision
        if not self.supports_deep_supervision:
            return output

        # Deep supervision outputs (deeper to shallower)
        result = {"output": output}
        for i, (feat, head) in enumerate(zip(ds_features, self.ds_heads)):
            result[f"ds_{i}"] = head(feat)

        return result

    def init_weights(self):
        """Initialize model weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ============================================================================
# Architecture Registry Builders
# ============================================================================


@register_architecture("rsunet")
def build_rsunet(cfg) -> RSUNet:
    """
    Build RSUNet from Hydra config.

    Config example:
        model:
          architecture: rsunet
          in_channels: 1
          out_channels: 2
          filters: [16, 32, 64, 128]
          rsunet_norm: batch           # batch/group/instance/none
          rsunet_activation: relu      # relu/leakyrelu/prelu/elu
          rsunet_num_groups: 8         # For group norm
          rsunet_down_factors: [[1,2,2], [1,2,2], [1,2,2]]  # Anisotropic
          deep_supervision: false      # Enable multi-scale outputs
          rsunet_depth_2d: 0           # Number of 2D layers (hybrid mode)
          rsunet_kernel_2d: [1, 3, 3]  # Kernel for 2D layers
    """
    width = list(cfg.model.filters) if hasattr(cfg.model, "filters") else [16, 32, 64, 128]

    # Parse down factors
    down_factors = None
    if hasattr(cfg.model, "rsunet_down_factors") and cfg.model.rsunet_down_factors is not None:
        down_factors = [tuple(f) for f in cfg.model.rsunet_down_factors]

    # Parse kernel for 2D
    kernel_2d = (1, 3, 3)
    if hasattr(cfg.model, "rsunet_kernel_2d") and cfg.model.rsunet_kernel_2d is not None:
        kernel_2d = tuple(cfg.model.rsunet_kernel_2d)

    # Activation kwargs
    act_kwargs = {}
    if hasattr(cfg.model, "rsunet_act_negative_slope"):
        act_kwargs["negative_slope"] = cfg.model.rsunet_act_negative_slope
    if hasattr(cfg.model, "rsunet_act_init"):
        act_kwargs["init"] = cfg.model.rsunet_act_init

    return RSUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        width=width,
        norm=getattr(cfg.model, "rsunet_norm", "batch"),
        activation=getattr(cfg.model, "rsunet_activation", "relu"),
        num_groups=getattr(cfg.model, "rsunet_num_groups", 8),
        deep_supervision=getattr(cfg.model, "deep_supervision", False),
        down_factors=down_factors,
        depth_2d=getattr(cfg.model, "rsunet_depth_2d", 0),
        kernel_2d=kernel_2d,
        **act_kwargs,
    )


@register_architecture("rsunet_iso")
def build_rsunet_iso(cfg) -> RSUNet:
    """
    Build RSUNet for isotropic data.

    Convenience builder that sets down_factors to (2,2,2).
    """
    width = list(cfg.model.filters) if hasattr(cfg.model, "filters") else [16, 32, 64, 128]
    depth = len(width) - 1

    return RSUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        width=width,
        down_factors=[(2, 2, 2)] * depth,  # Isotropic
        norm=getattr(cfg.model, "rsunet_norm", "batch"),
        activation=getattr(cfg.model, "rsunet_activation", "relu"),
        num_groups=getattr(cfg.model, "rsunet_num_groups", 8),
        deep_supervision=getattr(cfg.model, "deep_supervision", False),
    )


__all__ = ["RSUNet", "BilinearUp3d", "build_rsunet", "build_rsunet_iso"]
