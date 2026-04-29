"""
Inference utilities for PyTorch Connectomics.

This orchestrator wires together sliding-window inference and TTA. Decoding and
evaluation are separate stages.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..config import Config
from .sliding import build_sliding_inferer, is_2d_inference_mode
from .tta import TTAPredictor

logger = logging.getLogger(__name__)


class InferenceManager:
    """Manager for inference operations including sliding window and TTA."""

    def __init__(
        self,
        cfg: Config | DictConfig,
        model: nn.Module,
        forward_fn: callable,
    ):
        self.cfg = cfg
        self.model = model
        self.forward_fn = forward_fn

        if is_2d_inference_mode(cfg):
            self.sliding_inferer = None
            logger.warning(
                "Sliding-window inference disabled for 2D models with do_2d=True. "
                "Using direct inference instead."
            )
        else:
            self.sliding_inferer = build_sliding_inferer(cfg)

        self.tta = TTAPredictor(
            cfg=cfg, sliding_inferer=self.sliding_inferer, forward_fn=self.forward_fn
        )

    def predict_with_tta(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_align_to_image: bool = False,
        requested_head: Optional[str] = None,
    ) -> torch.Tensor:
        """Run prediction with optional TTA and sliding window."""
        return self.tta.predict(
            images,
            mask=mask,
            mask_align_to_image=mask_align_to_image,
            requested_head=requested_head,
        )

    def predict_named_heads_with_tta(
        self,
        images: torch.Tensor,
        heads: list[str],
        mask: Optional[torch.Tensor] = None,
        mask_align_to_image: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run prediction once per requested head and return a mapping of tensors."""
        predictions: dict[str, torch.Tensor] = {}
        for head_name in heads:
            predictions[head_name] = self.predict_with_tta(
                images,
                mask=mask,
                mask_align_to_image=mask_align_to_image,
                requested_head=head_name,
            )
        return predictions

    def is_distributed_tta_sharding_enabled(self) -> bool:
        """Return whether distributed TTA sharding is active for this process."""
        return self.tta.is_distributed_sharding_enabled()

    def is_distributed_window_sharding_enabled(self) -> bool:
        """Return whether lazy sliding-window sharding is active for this process."""
        sliding_cfg = getattr(getattr(self.cfg, "inference", None), "sliding_window", None)
        if sliding_cfg is None:
            return False
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_dist else 1
        return bool(
            getattr(sliding_cfg, "lazy_load", False)
            and getattr(sliding_cfg, "distributed_sharding", False)
            and is_dist
            and world_size > 1
        )

    def should_skip_postprocess_on_rank(self) -> bool:
        """Return True on ranks that only contributed a distributed inference shard."""
        if self.tta.should_skip_postprocess_on_rank():
            return True
        if self.is_distributed_window_sharding_enabled():
            return torch.distributed.get_rank() != 0
        return False


__all__ = ["InferenceManager"]
