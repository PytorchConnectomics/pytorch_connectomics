"""
Inference utilities for PyTorch Connectomics.

This orchestrator wires together sliding-window inference, TTA, decoding, and I/O.
Most heavy logic now lives in dedicated helper modules for clarity and reuse.
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
    ) -> torch.Tensor:
        """Run prediction with optional TTA and sliding window."""
        return self.tta.predict(
            images,
            mask=mask,
            mask_align_to_image=mask_align_to_image,
        )

    def is_distributed_tta_sharding_enabled(self) -> bool:
        """Return whether distributed TTA sharding is active for this process."""
        return self.tta.is_distributed_sharding_enabled()

    def should_skip_postprocess_on_rank(self) -> bool:
        """Return True on nonzero ranks after distributed TTA aggregation."""
        return self.tta.should_skip_postprocess_on_rank()


__all__ = ["InferenceManager"]
