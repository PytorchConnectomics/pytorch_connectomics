"""
Segmentation metrics entrypoints.

This module re-exports numpy/scipy implementations from `segmentation_numpy`
and provides lightweight torchmetrics-compatible wrappers for online evaluation.
"""

from __future__ import annotations


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

    Args:
        return_all_stats: If True, also compute and return precision and recall
        dist_sync_on_step: Whether to sync across distributed processes on each step
    """

    full_state_update: bool = False

    def __init__(self, return_all_stats: bool = False, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.return_all_stats = return_all_stats

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        if return_all_stats:
            self.add_state("total_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Move to CPU and numpy for the underlying implementation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if self.return_all_stats:
            are, precision, recall = adapted_rand(preds_np, target_np, all_stats=True)
            self.total += torch.tensor(are, device=self.total.device)
            self.total_precision += torch.tensor(precision, device=self.total_precision.device)
            self.total_recall += torch.tensor(recall, device=self.total_recall.device)
        else:
            score = float(adapted_rand(preds_np, target_np, all_stats=False))
            self.total += torch.tensor(score, device=self.total.device)

        self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            if self.return_all_stats:
                return {
                    "adapted_rand_error": torch.tensor(0.0, device=self.total.device),
                    "adapted_rand_precision": torch.tensor(0.0, device=self.total.device),
                    "adapted_rand_recall": torch.tensor(0.0, device=self.total.device),
                }
            return torch.tensor(0.0, device=self.total.device)

        if self.return_all_stats:
            return {
                "adapted_rand_error": self.total / self.count,
                "adapted_rand_precision": self.total_precision / self.count,
                "adapted_rand_recall": self.total_recall / self.count,
            }
        return self.total / self.count
