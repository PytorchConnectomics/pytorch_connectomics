"""
MedNeXt model wrappers with deep supervision support.

MedNeXt is a ConvNeXt-based architecture for 3D medical image segmentation.
Supports 4 model sizes (S, B, M, L) and multiple kernel sizes (3, 5, 7).

Reference:
    Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets
    for Medical Image Segmentation", MICCAI 2023
    https://arxiv.org/abs/2303.09975

Installation:
    pip install -e /projects/weilab/weidf/lib/MedNeXt

See .claude/MEDNEXT.md for detailed documentation.
"""

import torch
import torch.nn as nn
from typing import Union, Dict

try:
    from nnunet_mednext import create_mednext_v1, MedNeXt as MedNeXtBase

    MEDNEXT_AVAILABLE = True
except ImportError:
    MEDNEXT_AVAILABLE = False
    create_mednext_v1 = None
    MedNeXtBase = None

from .base import ConnectomicsModel
from .registry import register_architecture


class MedNeXtWrapper(ConnectomicsModel):
    """
    Wrapper for MedNeXt models with deep supervision support.

    MedNeXt can output predictions at multiple scales when deep_supervision=True:
    - Output 0: Full resolution (main output)
    - Output 1: 1/2 resolution
    - Output 2: 1/4 resolution
    - Output 3: 1/8 resolution
    - Output 4: 1/16 resolution (bottleneck)

    This is critical for MedNeXt's performance - deep supervision is recommended.
    """

    def __init__(self, model: nn.Module, deep_supervision: bool = False):
        super().__init__()
        self.model = model
        self.supports_deep_supervision = deep_supervision
        self.output_scales = 5 if deep_supervision else 1

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional deep supervision.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            For single-scale (deep_supervision=False):
                torch.Tensor of shape (B, num_classes, D, H, W)

            For multi-scale (deep_supervision=True):
                Dict with keys:
                    - 'output': Main output (full resolution)
                    - 'ds_1': 1/2 resolution output
                    - 'ds_2': 1/4 resolution output
                    - 'ds_3': 1/8 resolution output
                    - 'ds_4': 1/16 resolution output
        """
        outputs = self.model(x)

        if self.supports_deep_supervision and isinstance(outputs, list):
            # Convert list to dict for Lightning compatibility
            return {
                "output": outputs[0],  # Main output (full resolution)
                "ds_1": outputs[1],  # 1/2 resolution
                "ds_2": outputs[2],  # 1/4 resolution
                "ds_3": outputs[3],  # 1/8 resolution
                "ds_4": outputs[4],  # 1/16 resolution (bottleneck)
            }
        else:
            return outputs


def _check_mednext_available():
    """Check if MedNeXt is installed."""
    if not MEDNEXT_AVAILABLE:
        raise ImportError(
            "MedNeXt is not installed.\n"
            "Install from: /projects/weilab/weidf/lib/MedNeXt\n"
            "Run: pip install -e /projects/weilab/weidf/lib/MedNeXt\n"
            "Or add to PYTHONPATH: export PYTHONPATH=/projects/weilab/weidf/lib/MedNeXt:"
            "$PYTHONPATH\n"
            "\nSee .claude/MEDNEXT.md for detailed setup instructions."
        )


@register_architecture("mednext")
def build_mednext(cfg) -> ConnectomicsModel:
    """
    Build MedNeXt model using predefined sizes.

    Supports 4 model sizes from MICCAI 2023 paper:
        - S (Small): 5.6M params (3x3x3) / 5.9M params (5x5x5)
        - B (Base): 10.5M params (3x3x3) / 11.0M params (5x5x5)
        - M (Medium): 17.6M params (3x3x3) / 18.3M params (5x5x5)
        - L (Large): 61.8M params (3x3x3) / 63.0M params (5x5x5)

    Config parameters:
        - model.in_channels: Number of input channels (default: 1)
        - model.out_channels: Number of output classes (required)
        - model.mednext_size: Model size 'S', 'B', 'M', or 'L' (default: 'S')
        - model.mednext_kernel_size: Kernel size 3, 5, or 7 (default: 3)
        - model.deep_supervision: Enable deep supervision (default: False, RECOMMENDED: True)

    Important notes:
        - Deep supervision is RECOMMENDED for best performance
        - MedNeXt prefers 1mm isotropic spacing (unlike nnUNet's median spacing)
        - Use AdamW optimizer with lr=1e-3 and no LR scheduler (constant LR)
        - Use kernel_size=3 first, then optionally use UpKern to initialize kernel_size=5

    Args:
        cfg: Hydra config object

    Returns:
        MedNeXtWrapper containing MedNeXt model

    Example config:
        model:
          architecture: mednext
          in_channels: 1
          out_channels: 2
          mednext_size: S
          mednext_kernel_size: 3
          deep_supervision: true

    See .claude/MEDNEXT.md for complete documentation.
    """
    _check_mednext_available()

    # Extract config from Hydra/OmegaConf
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    model_size = getattr(cfg.model, "mednext_size", "S")
    kernel_size = getattr(cfg.model, "mednext_kernel_size", 3)
    deep_supervision = getattr(cfg.model, "deep_supervision", False)

    # Validate model size
    if model_size not in ["S", "B", "M", "L"]:
        raise ValueError(
            f"MedNeXt model_size must be 'S', 'B', 'M', or 'L'. Got: {model_size}\n"
            f"Model sizes:\n"
            f"  - S (Small): 5.6M params\n"
            f"  - B (Base): 10.5M params\n"
            f"  - M (Medium): 17.6M params\n"
            f"  - L (Large): 61.8M params"
        )

    # Validate kernel size
    if kernel_size not in [3, 5, 7]:
        raise ValueError(
            f"MedNeXt kernel_size must be 3, 5, or 7. Got: {kernel_size}\n"
            f"Recommended: Start with kernel_size=3"
        )

    # Build model using factory function
    model = create_mednext_v1(
        num_input_channels=in_channels,
        num_classes=out_channels,
        model_id=model_size,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
    )

    return MedNeXtWrapper(model, deep_supervision=deep_supervision)


@register_architecture("mednext_custom")
def build_mednext_custom(cfg) -> ConnectomicsModel:
    """
    Build MedNeXt with custom architecture parameters.

    For advanced users who need full control over MedNeXt architecture.
    Most users should use 'mednext' architecture with predefined sizes.

    Config parameters:
        - model.in_channels: Number of input channels (default: 1)
        - model.out_channels: Number of output classes (required)
        - model.mednext_base_channels: Base channel count (default: 32)
        - model.mednext_exp_r: Expansion ratio, int or list (default: 4)
        - model.mednext_kernel_size: Kernel size (default: 7)
        - model.deep_supervision: Enable deep supervision (default: False)
        - model.mednext_do_res: Residual connections in blocks (default: True)
        - model.mednext_do_res_up_down: Residual in up/down blocks (default: True)
        - model.mednext_block_counts: Blocks per level, list of 9 ints
          (default: [2,2,2,2,2,2,2,2,2])
        - model.mednext_checkpoint_style: Gradient checkpointing, None or
          'outside_block' (default: None)
        - model.mednext_norm: Normalization 'group' or 'layer' (default: 'group')
        - model.mednext_dim: Dimension '2d' or '3d' (default: '3d')
        - model.mednext_grn: Global Response Normalization (default: False)

    Args:
        cfg: Hydra config object

    Returns:
        MedNeXtWrapper containing custom MedNeXt model

    Example config:
        model:
          architecture: mednext_custom
          in_channels: 1
          out_channels: 2
          mednext_base_channels: 32
          mednext_exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
          mednext_kernel_size: 7
          deep_supervision: true
          mednext_block_counts: [3, 4, 4, 4, 4, 4, 4, 4, 3]
          mednext_checkpoint_style: outside_block

    See .claude/MEDNEXT.md for complete parameter documentation.
    """
    _check_mednext_available()

    # Extract all custom parameters (Hydra only)
    params = {
        "in_channels": cfg.model.in_channels,
        "n_channels": getattr(cfg.model, "mednext_base_channels", 32),
        "n_classes": cfg.model.out_channels,
        "exp_r": getattr(cfg.model, "mednext_exp_r", 4),
        "kernel_size": getattr(cfg.model, "mednext_kernel_size", 7),
        "deep_supervision": getattr(cfg.model, "deep_supervision", False),
        "do_res": getattr(cfg.model, "mednext_do_res", True),
        "do_res_up_down": getattr(cfg.model, "mednext_do_res_up_down", True),
        "block_counts": getattr(cfg.model, "mednext_block_counts", [2, 2, 2, 2, 2, 2, 2, 2, 2]),
        "checkpoint_style": getattr(cfg.model, "mednext_checkpoint_style", None),
        "norm_type": getattr(cfg.model, "mednext_norm", "group"),
        "dim": getattr(cfg.model, "mednext_dim", "3d"),
        "grn": getattr(cfg.model, "mednext_grn", False),
    }

    # Validate parameters
    if params["dim"] not in ["2d", "3d"]:
        raise ValueError(f"mednext_dim must be '2d' or '3d', got: {params['dim']}")

    if params["norm_type"] not in ["group", "layer"]:
        raise ValueError(f"mednext_norm must be 'group' or 'layer', got: {params['norm_type']}")

    if len(params["block_counts"]) != 9:
        raise ValueError(
            f"mednext_block_counts must have exactly 9 elements (one per level), "
            f"got {len(params['block_counts'])}"
        )

    # Build custom model
    model = MedNeXtBase(**params)

    return MedNeXtWrapper(model, deep_supervision=params["deep_supervision"])


# Utility function for UpKern weight loading
def upkern_load_weights(
    target_model: MedNeXtWrapper, source_model: MedNeXtWrapper
) -> MedNeXtWrapper:
    """
    Load weights from small kernel model to large kernel model using UpKern.

    UpKern initializes large kernel weights by trilinear interpolation of small kernel weights.
    This allows training a 3x3x3 model first, then fine-tuning a 5x5x5 model.

    Args:
        target_model: MedNeXt model with large kernels (e.g., 5x5x5)
        source_model: MedNeXt model with small kernels (e.g., 3x3x3), pre-trained

    Returns:
        target_model with initialized weights

    Requirements:
        - Models must have identical architecture except kernel size
        - Source model kernel size must be smaller than target

    Example:
        # Train small kernel model
        model_3x3 = build_mednext(cfg_3x3)
        # ... train model_3x3 ...

        # Initialize large kernel model with UpKern
        model_5x5 = build_mednext(cfg_5x5)
        model_5x5 = upkern_load_weights(model_5x5, model_3x3)
        # ... fine-tune model_5x5 ...

    See MEDNEXT.md section "UpKern weight loading" for details.
    """
    try:
        from nnunet_mednext.run.load_weights import upkern_load_weights as _upkern_load
    except ImportError:
        raise ImportError(
            "UpKern utility not found in MedNeXt installation.\n"
            "Ensure MedNeXt is properly installed from /projects/weilab/weidf/lib/MedNeXt"
        )

    # Extract the actual MedNeXt models from wrappers
    target_inner = target_model.model
    source_inner = source_model.model

    # Load weights using UpKern
    target_inner = _upkern_load(target_inner, source_inner)

    # Update wrapper
    target_model.model = target_inner

    return target_model


__all__ = [
    "MedNeXtWrapper",
    "build_mednext",
    "build_mednext_custom",
    "upkern_load_weights",
]
