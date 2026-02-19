"""
Inference utilities for PyTorch Connectomics.

This orchestrator wires together sliding-window inference, TTA, decoding, and I/O.
Most heavy logic now lives in dedicated helper modules for clarity and reuse.
"""

from __future__ import annotations

from typing import Optional
import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..config import Config
from .sliding import build_sliding_inferer
from .tta import TTAPredictor


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

        if getattr(cfg.data, "do_2d", False):
            self.sliding_inferer = None
            warnings.warn(
                "Sliding-window inference disabled for 2D models with do_2d=True. "
                "Using direct inference instead.",
                UserWarning,
            )
        else:
            self.sliding_inferer = build_sliding_inferer(cfg)

        self.tta = TTAPredictor(
            cfg=cfg, sliding_inferer=self.sliding_inferer, forward_fn=self.forward_fn
        )

    def predict_with_tta(
        self, images: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run prediction with optional TTA and sliding window."""
        return self.tta.predict(images, mask=mask)


__all__ = ["InferenceManager"]
