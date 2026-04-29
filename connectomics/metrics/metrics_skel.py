"""
Skeleton-based evaluation metrics for curvilinear structures.

This module provides metrics for evaluating segmentation quality of curvilinear
structures (e.g., neurons, blood vessels) based on skeleton correctness,
completeness, and quality.

Based on:
    Mosinska et al., "Beyond the Pixel-Wise Loss for Topology-Aware Delineation"
    https://arxiv.org/abs/1712.02190

Metrics:
    - Correctness: TP / (TP + FP) on skeleton
    - Completeness: TP / (TP + FN) on skeleton
    - Quality: (Completeness * Correctness) /
      (Completeness + Correctness - Completeness * Correctness)
    - Foreground IoU: Intersection over Union of foreground regions

File-backed batch evaluation lives in ``connectomics.evaluation``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from skimage.morphology import dilation, skeletonize, square


def compute_skeleton_metrics(
    skeleton_output: List[np.ndarray],
    skeleton_gt: List[np.ndarray],
    skeleton_output_dil: List[np.ndarray],
    skeleton_gt_dil: List[np.ndarray],
) -> Tuple[float, float, float]:
    """Compute skeleton-based metrics for curvilinear structure evaluation.

    Args:
        skeleton_output: List of skeletonized predictions (binarized)
        skeleton_gt: List of skeletonized ground truth
        skeleton_output_dil: List of dilated skeletonized predictions
        skeleton_gt_dil: List of dilated skeletonized ground truth

    Returns:
        Tuple of (correctness, completeness, quality) metrics
            - Correctness: True positives / (True positives + False positives)
            - Completeness: True positives / (True positives + False negatives)
            - Quality: F-measure combining correctness and completeness

    Notes:
        - All inputs should be binary arrays (0 or 1)
        - Dilation factor should be consistent (typically 5x5 square)
        - Quality is 0 if denominator is 0 (no predictions or ground truth)
    """
    tpcor = 0  # True positives for correctness
    tpcom = 0  # True positives for completeness
    fn = 0  # False negatives
    fp = 0  # False positives

    for i in range(len(skeleton_output)):
        # Correctness: predicted skeleton matches dilated GT
        tpcor += ((skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)).sum()

        # Completeness: GT skeleton matches dilated prediction
        tpcom += ((skeleton_gt[i] == skeleton_output_dil[i]) & (skeleton_gt[i] == 1)).sum()

        # False negatives: GT not covered by dilated prediction
        fn += (skeleton_gt[i] == 1).sum() - (
            (skeleton_gt[i] == skeleton_output_dil[i]) & (skeleton_gt[i] == 1)
        ).sum()

        # False positives: prediction not matching dilated GT
        fp += (skeleton_output[i] == 1).sum() - (
            (skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)
        ).sum()

    # Calculate metrics with safety checks
    correctness = tpcor / (tpcor + fp) if (tpcor + fp) > 0 else 0.0
    completeness = tpcom / (tpcom + fn) if (tpcom + fn) > 0 else 0.0

    # Quality (F-measure)
    denominator = completeness + correctness - completeness * correctness
    quality = (completeness * correctness / denominator) if denominator > 0 else 0.0

    return correctness, completeness, quality


def compute_precision_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_size: int = 5,
) -> Tuple[float, float, float]:
    """Compute precision and recall metrics for single prediction-GT pair.

    Args:
        pred: Predicted binary segmentation mask
        gt: Ground truth binary segmentation mask
        dilation_size: Size of square structuring element for dilation. Default: 5

    Returns:
        Tuple of (correctness, completeness, quality) metrics

    Notes:
        - Automatically skeletonizes input masks
        - Applies dilation with specified size (default 5x5 square)
        - Returns 0.0 for all metrics if inputs are empty
    """
    # Skeletonize both prediction and ground truth
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)

    # Dilate skeletons for tolerance in matching
    pred_dil = dilation(pred_skel, square(dilation_size))
    gt_dil = dilation(gt_skel, square(dilation_size))

    return compute_skeleton_metrics([pred_skel], [gt_skel], [pred_dil], [gt_dil])


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate foreground Intersection over Union (IoU).

    Args:
        pred: Predicted binary mask
        gt: Ground truth binary mask

    Returns:
        Foreground IoU score in range [0.0, 1.0]
            - 1.0: Perfect overlap
            - 0.0: No overlap or empty union

    Notes:
        - Both inputs should be binary (0 or 1)
        - Returns 0.0 if union is empty (both masks are empty)
    """
    inter = np.logical_and(pred, gt).astype(np.float32)
    union = np.logical_or(pred, gt).astype(np.float32)

    if union.sum() == 0:
        return 0.0

    return inter.sum() / union.sum()


def binarize_masks(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize prediction and ground truth masks.

    Args:
        pred: Prediction mask (0-255 range)
        gt: Ground truth mask (0-255 range)
        threshold: Threshold for binarizing prediction. Default: 128

    Returns:
        Tuple of (binarized_pred, binarized_gt) as uint8 arrays

    Notes:
        - Prediction: threshold at specified value (default 128)
        - Ground truth: exclude 0 (background) and 255 (ignore) pixels
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = ((gt != 0) & (gt != 255)).astype(np.uint8)
    return pred_bin, gt_bin


def evaluate_image_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: int = 128,
    dilation_size: int = 5,
) -> Tuple[float, float, float, float]:
    """Evaluate single prediction-ground truth pair.

    Args:
        pred: Prediction mask (0-255 range)
        gt: Ground truth mask (0-255 range)
        threshold: Threshold for binarizing prediction. Default: 128
        dilation_size: Dilation size for skeleton matching. Default: 5

    Returns:
        Tuple of (iou, correctness, completeness, quality) metrics
            - Returns (1.0, 1.0, 1.0, 1.0) if GT is empty
            - All values in range [0.0, 1.0]
    """
    # Binarize masks
    pred_bin, gt_bin = binarize_masks(pred, gt, threshold)

    # Handle empty ground truth
    num_gt = gt_bin.sum()
    if num_gt == 0:
        return 1.0, 1.0, 1.0, 1.0

    # Compute metrics
    iou = compute_iou(pred_bin, gt_bin)
    correctness, completeness, quality = compute_precision_recall(pred_bin, gt_bin, dilation_size)

    return iou, correctness, completeness, quality


__all__ = [
    "compute_skeleton_metrics",
    "compute_precision_recall",
    "compute_iou",
    "binarize_masks",
    "evaluate_image_pair",
]
