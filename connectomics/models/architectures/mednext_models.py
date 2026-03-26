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

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import torch
import torch.nn as nn

try:
    from nnunet_mednext import MedNeXt as MedNeXtBase
    from nnunet_mednext import MedNeXtBlock, create_mednext_v1

    MEDNEXT_AVAILABLE = True
except ImportError:
    MEDNEXT_AVAILABLE = False
    create_mednext_v1 = None
    MedNeXtBase = None
    MedNeXtBlock = None

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


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config value from either a mapping or an attribute-based object."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _infer_mednext_head_block_kwargs(model: nn.Module) -> dict[str, Any]:
    """Infer head block parameters from the final decoder block of a MedNeXt trunk."""
    if not hasattr(model, "dec_block_0") or len(model.dec_block_0) == 0:
        raise ValueError("MedNeXt trunk must expose a non-empty dec_block_0 to build task heads.")

    ref_block = model.dec_block_0[0]
    if not isinstance(ref_block, MedNeXtBlock):
        raise TypeError(
            "Expected MedNeXt dec_block_0 to contain MedNeXtBlock instances for multi-head reuse."
        )

    kernel_size = ref_block.conv1.kernel_size
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]

    if isinstance(ref_block.norm, nn.GroupNorm):
        norm_type = "group"
    else:
        norm_type = "layer"

    return {
        "exp_r": ref_block.conv2.out_channels // ref_block.conv2.in_channels,
        "kernel_size": int(kernel_size),
        "do_res": ref_block.do_res,
        "norm_type": norm_type,
        "dim": ref_block.dim,
        "grn": ref_block.grn,
    }


class MedNeXtTaskHead(nn.Module):
    """A task-specific MedNeXt head on top of the shared full-resolution feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_channels: int | None = None,
        *,
        exp_r: int,
        kernel_size: int,
        do_res: bool,
        norm_type: str,
        dim: str,
        grn: bool,
    ):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"MedNeXt task head num_blocks must be >= 0, got {num_blocks}")
        if out_channels <= 0:
            raise ValueError(f"MedNeXt task head out_channels must be positive, got {out_channels}")
        if hidden_channels is None:
            hidden_channels = in_channels
        if hidden_channels <= 0:
            raise ValueError(
                f"MedNeXt task head hidden_channels must be positive, got {hidden_channels}"
            )
        if hidden_channels > in_channels:
            raise ValueError(
                "MedNeXt task head hidden_channels must not exceed the shared feature width "
                f"({hidden_channels} > {in_channels})"
            )
        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d
        else:
            raise ValueError(f"MedNeXt task head dim must be '2d' or '3d', got {dim}")

        self.input_projection = (
            conv(in_channels, hidden_channels, kernel_size=1)
            if hidden_channels != in_channels
            else nn.Identity()
        )
        blocks = [
            MedNeXtBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                exp_r=exp_r,
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn,
            )
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.projection = conv(hidden_channels, out_channels, kernel_size=1)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.projection(x)


class MedNeXtMultiHeadWrapper(ConnectomicsModel):
    """
    MedNeXt wrapper with multiple named task heads at the final resolution.

    Output contract:
        {"output": {"head_name": tensor, ...}}

    Deep supervision is intentionally unsupported here in v1.
    """

    def __init__(
        self,
        model: nn.Module,
        heads: Mapping[str, Any],
        *,
        primary_head: str | None = None,
    ):
        super().__init__()
        if getattr(model, "do_ds", False):
            raise ValueError(
                "MedNeXtMultiHeadWrapper does not support deep supervision yet. "
                "Disable deep supervision for the trunk first."
            )
        if not hasattr(model, "forward_features"):
            raise ValueError(
                "MedNeXt trunk must expose forward_features() before using MedNeXtMultiHeadWrapper."
            )
        if not heads:
            raise ValueError("MedNeXtMultiHeadWrapper requires at least one named task head.")

        self.model = model
        self.supports_deep_supervision = False
        self.output_scales = 1
        self.feature_channels = int(self.model.stem.out_channels)
        self.head_block_kwargs = _infer_mednext_head_block_kwargs(model)

        task_heads = {}
        head_specs = {}
        for head_name, head_cfg in heads.items():
            out_channels = int(_cfg_value(head_cfg, "out_channels", head_cfg))
            num_blocks = int(_cfg_value(head_cfg, "num_blocks", 0))
            hidden_channels = _cfg_value(head_cfg, "hidden_channels", None)
            hidden_channels = int(hidden_channels) if hidden_channels is not None else None
            task_heads[head_name] = MedNeXtTaskHead(
                in_channels=self.feature_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                hidden_channels=hidden_channels,
                **self.head_block_kwargs,
            )
            head_specs[head_name] = {
                "out_channels": out_channels,
                "num_blocks": num_blocks,
                "hidden_channels": hidden_channels or self.feature_channels,
            }

        self.heads = nn.ModuleDict(task_heads)
        self.head_specs = head_specs
        resolved_primary_head = primary_head or next(iter(self.heads.keys()))
        if resolved_primary_head not in self.heads:
            raise ValueError(
                f"primary_head '{resolved_primary_head}' is not one of the configured heads: "
                f"{sorted(self.heads.keys())}"
            )
        self.primary_head = resolved_primary_head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the shared MedNeXt feature map for downstream head logic."""
        return self.model.forward_features(x)

    def forward_heads(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply all named task heads to a shared feature map."""
        return {head_name: head(features) for head_name, head in self.heads.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        features = self.forward_features(x)
        return {"output": self.forward_heads(features)}


def _get_mednext_heads_cfg(cfg) -> tuple[dict[str, Any], str | None]:
    """Extract named MedNeXt head configuration from cfg.model."""
    raw_heads = getattr(cfg.model, "heads", None)
    if not raw_heads:
        return {}, None
    return dict(raw_heads), getattr(cfg.model, "primary_head", None)


def _resolve_mednext_num_classes(cfg, head_cfg: Mapping[str, Any]) -> int:
    """Choose the vendored MedNeXt projection width."""
    if head_cfg:
        total_head_channels = 0
        for spec in head_cfg.values():
            total_head_channels += int(_cfg_value(spec, "out_channels", 0))
        return max(1, total_head_channels)
    return int(cfg.model.out_channels)


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
        - model.mednext.size: Model size 'S', 'B', 'M', or 'L' (default: 'S')
        - model.mednext.kernel_size: Kernel size 3, 5, or 7 (default: 3)
        - model.loss.deep_supervision: Enable deep supervision (default: False, RECOMMENDED: True)

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
          mednext:
            size: S
            kernel_size: 3
          loss:
            deep_supervision: true

    See .claude/MEDNEXT.md for complete documentation.
    """
    _check_mednext_available()

    # Extract config from Hydra/OmegaConf
    in_channels = cfg.model.in_channels
    model_size = getattr(cfg.model.mednext, "size", "S")
    kernel_size = getattr(cfg.model.mednext, "kernel_size", 3)
    loss_cfg = getattr(cfg.model, "loss", None)
    deep_supervision = getattr(loss_cfg, "deep_supervision", False)
    head_cfg, primary_head = _get_mednext_heads_cfg(cfg)
    out_channels = _resolve_mednext_num_classes(cfg, head_cfg)

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

    if head_cfg:
        return MedNeXtMultiHeadWrapper(model, head_cfg, primary_head=primary_head)
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
        - model.mednext.base_channels: Base channel count (default: 32)
        - model.mednext.exp_r: Expansion ratio, int or list (default: 4)
        - model.mednext.kernel_size: Kernel size (default: 7)
        - model.loss.deep_supervision: Enable deep supervision (default: False)
        - model.mednext.do_res: Residual connections in blocks (default: True)
        - model.mednext.do_res_up_down: Residual in up/down blocks (default: True)
        - model.mednext.block_counts: Blocks per level, list of 9 ints
          (default: [2,2,2,2,2,2,2,2,2])
        - model.mednext.checkpoint_style: Gradient checkpointing, None or
          'outside_block' (default: None)
        - model.mednext.norm: Normalization 'group' or 'layer' (default: 'group')
        - model.mednext.dim: Dimension '2d' or '3d' (default: '3d')
        - model.mednext.grn: Global Response Normalization (default: False)

    Args:
        cfg: Hydra config object

    Returns:
        MedNeXtWrapper containing custom MedNeXt model

    Example config:
        model:
          architecture: mednext_custom
          in_channels: 1
          out_channels: 2
          mednext:
            base_channels: 32
            exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
            kernel_size: 7
          loss:
            deep_supervision: true
            block_counts: [3, 4, 4, 4, 4, 4, 4, 4, 3]
            checkpoint_style: outside_block

    See .claude/MEDNEXT.md for complete parameter documentation.
    """
    _check_mednext_available()
    head_cfg, primary_head = _get_mednext_heads_cfg(cfg)

    # Extract all custom parameters (Hydra only)
    params = {
        "in_channels": cfg.model.in_channels,
        "n_channels": getattr(cfg.model.mednext, "base_channels", 32),
        "n_classes": _resolve_mednext_num_classes(cfg, head_cfg),
        "exp_r": getattr(cfg.model.mednext, "exp_r", 4),
        "kernel_size": getattr(cfg.model.mednext, "kernel_size", 7),
        "deep_supervision": getattr(cfg.model.loss, "deep_supervision", False),
        "do_res": getattr(cfg.model.mednext, "do_res", True),
        "do_res_up_down": getattr(cfg.model.mednext, "do_res_up_down", True),
        "block_counts": getattr(cfg.model.mednext, "block_counts", [2, 2, 2, 2, 2, 2, 2, 2, 2]),
        "checkpoint_style": getattr(cfg.model.mednext, "checkpoint_style", None),
        "norm_type": getattr(cfg.model.mednext, "norm", "group"),
        "dim": getattr(cfg.model.mednext, "dim", "3d"),
        "grn": getattr(cfg.model.mednext, "grn", False),
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

    if head_cfg:
        return MedNeXtMultiHeadWrapper(model, head_cfg, primary_head=primary_head)
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
    "MedNeXtMultiHeadWrapper",
    "MedNeXtTaskHead",
    "MedNeXtWrapper",
    "build_mednext",
    "build_mednext_custom",
    "upkern_load_weights",
]
