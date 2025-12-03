"""
Evaluation metrics for PyTorch Connectomics.

This package provides comprehensive evaluation metrics:
- metrics_seg.py: Segmentation metrics (Adapted Rand, Dice, Jaccard, etc.)
- metrics_skel.py: Skeleton-based metrics for curvilinear structures

Note: PyTorch Lightning handles training monitoring and logging.

Import patterns:
    from connectomics.metrics import adapted_rand, get_binary_jaccard
    from connectomics.metrics import compute_skeleton_metrics, evaluate_image_pair
    from connectomics.metrics.metrics_seg import instance_matching
    from connectomics.metrics.metrics_skel import evaluate_directory
"""

from .metrics_seg import *
from .metrics_skel import (
    compute_skeleton_metrics,
    compute_precision_recall,
    compute_iou,
    binarize_masks,
    evaluate_image_pair,
    evaluate_file_pair,
    evaluate_directory,
)

__all__ = [
    # Segmentation metrics
    "jaccard",
    "get_binary_jaccard",
    "adapted_rand",
    "instance_matching",
    "cremi_distance",
    # Skeleton metrics
    "compute_skeleton_metrics",
    "compute_precision_recall",
    "compute_iou",
    "binarize_masks",
    "evaluate_image_pair",
    "evaluate_file_pair",
    "evaluate_directory",
]
