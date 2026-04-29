"""Inference stage helpers."""

from __future__ import annotations

from typing import Optional

import torch

from .manager import InferenceManager


def run_prediction_inference(
    manager: InferenceManager,
    images: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    mask_align_to_image: bool = False,
    requested_head: Optional[str] = None,
) -> torch.Tensor:
    """Run model prediction without decoding or evaluation."""
    return manager.predict_with_tta(
        images,
        mask=mask,
        mask_align_to_image=mask_align_to_image,
        requested_head=requested_head,
    )


__all__ = ["run_prediction_inference"]
