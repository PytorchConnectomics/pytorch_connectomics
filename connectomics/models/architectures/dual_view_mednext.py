"""Dual-view fusion variants built from the canonical MedNeXt modules.

The encoder/decoder blocks come from the same ``create_mednext_v1`` factory as
the ordinary ``mednext`` architecture.  This file changes only how two input
views share encoder weights and how their multiscale features are fused.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .base import ConnectomicsModel
from .mednext_models import MEDNEXT_AVAILABLE, create_mednext_v1
from .registry import register_architecture


class FixedAngleFiLM(nn.Module):
    """Condition one feature scale on a fixed acquisition angle."""

    def __init__(self, channels: int, angles_deg: List[float]):
        super().__init__()
        if len(angles_deg) != 2:
            raise ValueError("dual_view.view_angles_deg must contain exactly two angles")
        radians = torch.tensor(angles_deg, dtype=torch.float32) * (math.pi / 180.0)
        self.register_buffer(
            "angle_encoding",
            torch.stack((torch.sin(radians), torch.cos(radians)), dim=1),
            persistent=True,
        )
        hidden = max(8, channels // 4)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        # Begin as an identity transform so conditioning cannot destabilize the
        # unconditioned MedNeXt initialization.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, view_index: int) -> torch.Tensor:
        condition = self.angle_encoding[view_index].to(dtype=x.dtype)
        condition = condition.unsqueeze(0).expand(x.shape[0], -1)
        gamma, beta = self.mlp(condition).chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1], *([1] * (x.ndim - 2)))
        return x * (1.0 + gamma.reshape(shape)) + beta.reshape(shape)


class ReliabilityGate3d(nn.Module):
    """Equal-initialized spatial-and-channel reliability fusion."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        if reduction <= 0:
            raise ValueError("dual_view.gate_reduction must be positive")
        hidden = max(8, channels // reduction)
        evidence_channels = 4 * channels
        self.spatial_logits = nn.Sequential(
            nn.Conv3d(evidence_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 2, kernel_size=1),
        )
        self.channel_logits = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(evidence_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 2 * channels, kernel_size=1),
        )
        self.residual = nn.Conv3d(3 * channels, channels, kernel_size=1)

        # Zero logits yield exactly 0.5/0.5 softmax weights.  The zero residual
        # makes the entire block an arithmetic mean at initialization.
        nn.init.zeros_(self.spatial_logits[-1].weight)
        nn.init.zeros_(self.spatial_logits[-1].bias)
        nn.init.zeros_(self.channel_logits[-1].weight)
        nn.init.zeros_(self.channel_logits[-1].bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError(
                "Dual-view features must be registered and shape-matched; "
                f"got {tuple(left.shape)} and {tuple(right.shape)}"
            )
        difference = torch.abs(left - right)
        product = left * right
        evidence = torch.cat((left, right, difference, product), dim=1)
        spatial = self.spatial_logits(evidence).unsqueeze(2)
        channel = self.channel_logits(evidence).reshape(
            left.shape[0], 2, left.shape[1], 1, 1, 1
        )
        weights = torch.softmax(spatial + channel, dim=1)
        fused = weights[:, 0] * left + weights[:, 1] * right
        correction = self.residual(torch.cat((fused, difference, product), dim=1))
        return fused + correction


class DualViewMedNeXt(ConnectomicsModel):
    """Siamese or partially shared dual-view MedNeXt with one decoder."""

    SUPPORTED_DESIGNS = {"siamese_mean", "partial_gated"}

    def __init__(
        self,
        trunk: nn.Module,
        *,
        design: str,
        deep_supervision: bool,
        view_angles_deg: List[float],
        angle_conditioning: bool,
        gate_reduction: int,
        checkpoint_style: str | None,
    ):
        super().__init__()
        if design not in self.SUPPORTED_DESIGNS:
            raise ValueError(
                f"dual_view.design must be one of {sorted(self.SUPPORTED_DESIGNS)}, got {design!r}"
            )
        self.design = design
        self.trunk = trunk
        self.supports_deep_supervision = deep_supervision
        self.output_scales = 5 if deep_supervision else 1
        self.use_checkpointing = checkpoint_style == "outside_block"

        if design == "partial_gated":
            # The trunk owns the left branch and the single decoder.  Only the
            # optically sensitive shallow path is duplicated for the right view.
            self.right_stem = copy.deepcopy(trunk.stem)
            self.right_enc_block_0 = copy.deepcopy(trunk.enc_block_0)
            self.right_down_0 = copy.deepcopy(trunk.down_0)

        channels = [
            int(trunk.stem.out_channels),
            int(trunk.down_0.conv3.out_channels),
            int(trunk.down_1.conv3.out_channels),
            int(trunk.down_2.conv3.out_channels),
            int(trunk.down_3.conv3.out_channels),
        ]
        if design == "partial_gated":
            self.fusion = nn.ModuleList(
                [ReliabilityGate3d(c, reduction=gate_reduction) for c in channels]
            )
        else:
            self.fusion = nn.ModuleList([nn.Identity() for _ in channels])

        self.angle_conditioning = bool(angle_conditioning)
        self.angle_film = nn.ModuleList(
            [FixedAngleFiLM(c, view_angles_deg) for c in channels]
            if self.angle_conditioning
            else []
        )

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _condition(
        self, left: torch.Tensor, right: torch.Tensor, scale: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.angle_conditioning:
            return left, right
        film = self.angle_film[scale]
        return film(left, 0), film(right, 1)

    def _fuse(self, left: torch.Tensor, right: torch.Tensor, scale: int) -> torch.Tensor:
        if self.design == "siamese_mean":
            return 0.5 * (left + right)
        return self.fusion[scale](left, right)

    def _encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        if image.ndim != 5 or image.shape[1] != 2:
            raise ValueError(
                "DualViewMedNeXt expects registered [B,2,Z,Y,X] input; "
                f"got {tuple(image.shape)}"
            )
        left, right = image[:, 0:1], image[:, 1:2]

        left = self._run(self.trunk.stem, left)
        if self.design == "partial_gated":
            right = self._run(self.right_stem, right)
            right_enc_0 = self.right_enc_block_0
            right_down_0 = self.right_down_0
        else:
            right = self._run(self.trunk.stem, right)
            right_enc_0 = self.trunk.enc_block_0
            right_down_0 = self.trunk.down_0

        left = self._run(self.trunk.enc_block_0, left)
        right = self._run(right_enc_0, right)
        left, right = self._condition(left, right, 0)
        fused = [self._fuse(left, right, 0)]

        left = self._run(self.trunk.down_0, left)
        right = self._run(right_down_0, right)

        encoder_stages = (
            (self.trunk.enc_block_1, self.trunk.down_1),
            (self.trunk.enc_block_2, self.trunk.down_2),
            (self.trunk.enc_block_3, self.trunk.down_3),
        )
        for scale, (encoder, down) in enumerate(encoder_stages, start=1):
            left = self._run(encoder, left)
            right = self._run(encoder, right)
            left, right = self._condition(left, right, scale)
            fused.append(self._fuse(left, right, scale))
            left = self._run(down, left)
            right = self._run(down, right)

        left = self._run(self.trunk.bottleneck, left)
        right = self._run(self.trunk.bottleneck, right)
        left, right = self._condition(left, right, 4)
        fused.append(self._fuse(left, right, 4))
        return fused

    def forward(self, image: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x_res_0, x_res_1, x_res_2, x_res_3, x = self._encode(image)

        result: Dict[str, torch.Tensor] = {}
        if self.supports_deep_supervision:
            result["ds_4"] = self._run(self.trunk.out_4, x)

        x = self._run(self.trunk.up_3, x) + x_res_3
        x = self._run(self.trunk.dec_block_3, x)
        if self.supports_deep_supervision:
            result["ds_3"] = self._run(self.trunk.out_3, x)

        x = self._run(self.trunk.up_2, x) + x_res_2
        x = self._run(self.trunk.dec_block_2, x)
        if self.supports_deep_supervision:
            result["ds_2"] = self._run(self.trunk.out_2, x)

        x = self._run(self.trunk.up_1, x) + x_res_1
        x = self._run(self.trunk.dec_block_1, x)
        if self.supports_deep_supervision:
            result["ds_1"] = self._run(self.trunk.out_1, x)

        x = self._run(self.trunk.up_0, x) + x_res_0
        x = self._run(self.trunk.dec_block_0, x)
        output = self._run(self.trunk.out_0, x)

        if not self.supports_deep_supervision:
            return output
        return {"output": output, **result}


@register_architecture("mednext_dual_view")
def build_dual_view_mednext(cfg) -> DualViewMedNeXt:
    """Build a two-view MedNeXt from the ordinary MedNeXt preset factory."""

    if not MEDNEXT_AVAILABLE or create_mednext_v1 is None:
        raise ImportError(
            "MedNeXt is not installed. Install the repository's MedNeXt dependency with:\n"
            "    pip install git+https://github.com/PytorchConnectomics/MedNeXt.git"
        )
    if int(cfg.model.in_channels) != 2:
        raise ValueError("mednext_dual_view requires model.in_channels=2")

    size = str(cfg.model.mednext.size)
    kernel_size = int(cfg.model.mednext.kernel_size)
    if size not in {"S", "B", "M", "L"}:
        raise ValueError("model.mednext.size must be one of S, B, M, or L")
    if kernel_size not in {3, 5, 7}:
        raise ValueError("model.mednext.kernel_size must be 3, 5, or 7")

    deep_supervision = bool(cfg.model.loss.deep_supervision)
    checkpoint_style = cfg.model.mednext.checkpoint_style
    trunk = create_mednext_v1(
        num_input_channels=1,
        num_classes=int(cfg.model.out_channels),
        model_id=size,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
    )
    return DualViewMedNeXt(
        trunk,
        design=str(cfg.model.dual_view.design),
        deep_supervision=deep_supervision,
        view_angles_deg=list(cfg.model.dual_view.view_angles_deg),
        angle_conditioning=bool(cfg.model.dual_view.angle_conditioning),
        gate_reduction=int(cfg.model.dual_view.gate_reduction),
        checkpoint_style=checkpoint_style,
    )


__all__ = [
    "DualViewMedNeXt",
    "FixedAngleFiLM",
    "ReliabilityGate3d",
    "build_dual_view_mednext",
]
