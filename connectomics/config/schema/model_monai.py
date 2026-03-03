from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MonaiConfig:
    """MONAI UNet/BasicUNet configuration."""

    filters: Tuple[int, ...] = (32, 64, 128, 256, 512)
    dropout: float = 0.0
    norm: str = "batch"
    num_groups: int = 8
    activation: str = "relu"
    spatial_dims: int = 3
    num_res_units: int = 2
    kernel_size: int = 3
    strides: Optional[List[int]] = None
    act: str = "relu"
    upsample: str = "deconv"
    upsample_mode: str = "deconv"
    upsample_interp_mode: str = "linear"
    upsample_align_corners: bool = True


@dataclass
class TransformerConfig:
    """Transformer architecture configuration (UNETR/SwinUNETR)."""

    feature_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    pos_embed: str = "perceptron"
    norm: str = "instance"
    dropout: float = 0.0
    use_checkpoint: bool = False
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
