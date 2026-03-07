from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .model_mednext import MedNeXtConfig
from .model_monai import MonaiConfig, TransformerConfig
from .model_nnunet import NNUNetConfig
from .model_rsunet import RSUNetConfig


@dataclass
class LossBalancingConfig:
    """Configuration for adaptive loss weighting."""

    strategy: Optional[str] = None  # None, "uncertainty", or "gradnorm"
    gradnorm_alpha: float = 0.5
    gradnorm_lambda: float = 1.0
    gradnorm_parameter_strategy: str = "last"  # "first", "last", or "all"


@dataclass
class LossConfig:
    """Loss and deep supervision configuration."""

    # Loss configuration -- unified list where each entry defines a loss function,
    # its weight, kwargs, and channel routing (pred/target slices).
    # For pred_target losses, pred_slice/target_slice are optional:
    # omitted slices mean "use all channels".
    # Optional per-entry fields include:
    # - pos_weight: positive number or "auto" (for weight-aware losses).
    #   For WeightedBCEWithLogitsLoss:
    #   - number: feeds directly to BCE pos_weight.
    #   - "auto": computes neg/pos from the current batch (capped at 10.0).
    #   For other weight-aware losses:
    #   - number/"auto": builds spatial positive-class weighting maps.
    #   Defaults when omitted:
    #   - WeightedBCEWithLogitsLoss: 1.0 (no class-ratio reweighting)
    #   - other weight-aware losses: "auto"
    # - pred_slice / target_slice / pred2_slice / mask_slice:
    #   contiguous Python-style channel selectors using int or slice strings
    #   such as 0, -1, ":", "0:3", or ":-1".
    # - apply_deep_supervision: bool
    # When None, defaults to [DiceLoss + BCEWithLogitsLoss] applied to all channels.
    # Deep supervision (supported by MedNeXt, RSUNet, and some MONAI models)
    deep_supervision: bool = False
    deep_supervision_weights: Optional[List[float]] = (
        None  # None = auto: [1.0, 0.5, 0.25, 0.125, 0.0625]
    )
    deep_supervision_clamp_min: float = -20.0  # Clamp logits to prevent numerical instability
    deep_supervision_clamp_max: float = 20.0  # Especially important at coarser scales

    losses: Optional[List[Dict[str, Any]]] = None
    loss_balancing: LossBalancingConfig = field(default_factory=LossBalancingConfig)


@dataclass
class ModelArchConfig:
    """Architecture profile spec for extensible OOP arch resolution."""

    profile: Optional[str] = None
    type: str = "monai_basic_unet3d"
    variant: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Comprehensive configuration for neural network architectures, loss functions,
    and multi-task learning setups. Supports multiple architectures including
    UNet variants, MedNeXt, RSUNet, and transformer-based models.

    Key Features:
    - Multiple architecture support (UNet, MedNeXt, RSUNet, UNETR, etc.)
    - Configurable loss functions with weighting
    - Multi-task learning capabilities
    - Architecture-specific parameter tuning
    - Deep supervision support
    """

    # Architecture
    arch: ModelArchConfig = field(default_factory=ModelArchConfig)

    # I/O dimensions
    input_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    output_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 1
    out_channels: int = 1

    # Architecture-specific nested blocks
    monai: MonaiConfig = field(default_factory=MonaiConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mednext: MedNeXtConfig = field(default_factory=MedNeXtConfig)
    rsunet: RSUNetConfig = field(default_factory=RSUNetConfig)
    nnunet: NNUNetConfig = field(default_factory=NNUNetConfig)

    # Structured loss configuration (preferred for new configs)
    loss: LossConfig = field(default_factory=LossConfig)

    # External model weights loading
    # For loading pretrained weights from external checkpoints (e.g., BANIS, nnUNet)
    external_weights_path: Optional[str] = None  # Path to external checkpoint file
    external_weights_key_prefix: str = "model."  # Prefix to strip from state_dict keys
