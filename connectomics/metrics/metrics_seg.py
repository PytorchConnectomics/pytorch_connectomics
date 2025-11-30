"""
Segmentation metrics entrypoints.

This module re-exports numpy/scipy implementations from `segmentation_numpy`
and provides lightweight torchmetrics-compatible wrappers for online evaluation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torchmetrics
import numpy as np

from .segmentation_numpy import (
    adapted_rand,
    cremi_distance,
    instance_matching,
    jaccard,
    matching_criteria,
)

__all__ = [
    "jaccard",
    "get_binary_jaccard",
    "adapted_rand",
    "instance_matching",
    "cremi_distance",
    "matching_criteria",
    "AdaptedRandError",
]


def get_binary_jaccard(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Convenience wrapper to compute Jaccard on binary masks with optional thresholding.
    """
    if pred.dtype != np.bool_ and pred.dtype != np.int64 and pred.dtype != np.uint8:
        pred = (pred > threshold).astype(np.uint8)
    if target.dtype != np.bool_ and target.dtype != np.int64 and target.dtype != np.uint8:
        target = (target > threshold).astype(np.uint8)
    return jaccard(pred, target)


class AdaptedRandError(torchmetrics.Metric):
    """
    Torchmetrics-style wrapper around the numpy-based adapted Rand implementation.

    This wrapper lets us accumulate scores during Lightning `test_step` without
    manual numpy↔torch conversions in the training loop.
    """

    full_state_update: bool = False

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Move to CPU and numpy for the underlying implementation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        score = float(adapted_rand(preds_np, target_np))
        self.total += torch.tensor(score, device=self.total.device)
        self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(0.0, device=self.total.device)
        return self.total / self.count
