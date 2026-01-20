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
    instance_matching_simple,
    jaccard,
    matching_criteria,
    voi,
)

__all__ = [
    "jaccard",
    "get_binary_jaccard",
    "adapted_rand",
    "instance_matching",
    "instance_matching_simple",
    "cremi_distance",
    "matching_criteria",
    "AdaptedRandError",
    "VariationOfInformation",
    "InstanceAccuracy",
    "InstanceAccuracySimple",
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


class VariationOfInformation(torchmetrics.Metric):
    """
    Torchmetrics-style wrapper around the numpy-based VOI implementation.

    VOI (Variation of Information) measures the information-theoretic distance
    between two clusterings. It decomposes into:
    - VOI Split (H(X|Y)): Over-segmentation error (false splits)
    - VOI Merge (H(Y|X)): Under-segmentation error (false merges)

    Lower values are better (0 = perfect match).

    This wrapper lets us accumulate scores during Lightning `test_step` without
    manual numpy↔torch conversions in the training loop.
    """

    full_state_update: bool = False

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("split_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("merge_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Move to CPU and numpy for the underlying implementation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        split, merge = voi(preds_np, target_np)
        self.split_total += torch.tensor(split, device=self.split_total.device)
        self.merge_total += torch.tensor(merge, device=self.merge_total.device)
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Return total VOI (split + merge)."""
        if self.count == 0:
            return torch.tensor(0.0, device=self.split_total.device)
        split_avg = self.split_total / self.count
        merge_avg = self.merge_total / self.count
        return split_avg + merge_avg

    def compute_split(self) -> torch.Tensor:
        """Return VOI split (over-segmentation error)."""
        if self.count == 0:
            return torch.tensor(0.0, device=self.split_total.device)
        return self.split_total / self.count

    def compute_merge(self) -> torch.Tensor:
        """Return VOI merge (under-segmentation error)."""
        if self.count == 0:
            return torch.tensor(0.0, device=self.merge_total.device)
        return self.merge_total / self.count


class InstanceAccuracy(torchmetrics.Metric):
    """
    Torchmetrics-style wrapper around instance_matching for instance-level accuracy.

    Instance accuracy measures the fraction of correctly detected instances:
        accuracy = TP / (TP + FP + FN)

    Where:
    - TP (True Positives): Number of GT instances correctly matched to predictions
    - FP (False Positives): Number of predicted instances not matched to GT
    - FN (False Negatives): Number of GT instances not matched to predictions

    Matching is based on IoU threshold (default 0.5).

    Higher values are better (1.0 = perfect detection).

    This wrapper lets us accumulate scores during Lightning `test_step` without
    manual numpy↔torch conversions in the training loop.
    """

    full_state_update: bool = False

    def __init__(self, thresh: float = 0.5, criterion: str = 'iou', 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.thresh = thresh
        self.criterion = criterion
        self.add_state("tp_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Move to CPU and numpy for the underlying implementation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        stats = instance_matching(target_np, preds_np, thresh=self.thresh, criterion=self.criterion)
        self.tp_total += torch.tensor(stats['tp'], device=self.tp_total.device)
        self.fp_total += torch.tensor(stats['fp'], device=self.fp_total.device)
        self.fn_total += torch.tensor(stats['fn'], device=self.fn_total.device)

    def compute(self) -> torch.Tensor:
        """Return instance-level accuracy: TP / (TP + FP + FN)."""
        denom = self.tp_total + self.fp_total + self.fn_total
        if denom == 0:
            return torch.tensor(0.0, device=self.tp_total.device)
        return self.tp_total.float() / denom.float()


class InstanceAccuracySimple(torchmetrics.Metric):
    """
    Torchmetrics-style wrapper for relaxed instance-level accuracy (NO Hungarian matching).

    WARNING: This is a RELAXED metric for debugging/analysis only, NOT for benchmark ranking.
    Unlike InstanceAccuracy, this does NOT use optimal bipartite matching.
    
    Simple counting approach:
        - Count all (GT, Pred) pairs with IoU >= threshold as TP
        - fp = n_pred - tp
        - fn = n_true - tp
        - accuracy = tp / (tp + fp + fn)

    This metric is useful for:
    - Quick debugging and sanity checks
    - Understanding raw overlap statistics
    - Comparing with strict Hungarian-based metrics

    Higher values are better (1.0 = perfect detection).

    This wrapper lets us accumulate scores during Lightning `test_step` without
    manual numpy↔torch conversions in the training loop.
    """

    full_state_update: bool = False

    def __init__(self, thresh: float = 0.5, criterion: str = 'iou', 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.thresh = thresh
        self.criterion = criterion
        self.add_state("tp_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Move to CPU and numpy for the underlying implementation
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        stats = instance_matching_simple(target_np, preds_np, thresh=self.thresh, criterion=self.criterion)
        self.tp_total += torch.tensor(stats['tp'], device=self.tp_total.device)
        self.fp_total += torch.tensor(stats['fp'], device=self.fp_total.device)
        self.fn_total += torch.tensor(stats['fn'], device=self.fn_total.device)

    def compute(self) -> torch.Tensor:
        """Return relaxed instance-level accuracy: TP / (TP + FP + FN)."""
        denom = self.tp_total + self.fp_total + self.fn_total
        if denom == 0:
            return torch.tensor(0.0, device=self.tp_total.device)
        return self.tp_total.float() / denom.float()
    
    def compute_precision(self) -> torch.Tensor:
        """Return instance-level precision: TP / (TP + FP)."""
        denom = self.tp_total + self.fp_total
        if denom == 0:
            return torch.tensor(0.0, device=self.tp_total.device)
        return self.tp_total.float() / denom.float()
    
    def compute_recall(self) -> torch.Tensor:
        """Return instance-level recall: TP / (TP + FN)."""
        denom = self.tp_total + self.fn_total
        if denom == 0:
            return torch.tensor(0.0, device=self.tp_total.device)
        return self.tp_total.float() / denom.float()
    
    def compute_f1(self) -> torch.Tensor:
        """Return instance-level F1: 2*TP / (2*TP + FP + FN)."""
        denom = 2 * self.tp_total + self.fp_total + self.fn_total
        if denom == 0:
            return torch.tensor(0.0, device=self.tp_total.device)
        return (2 * self.tp_total).float() / denom.float()
