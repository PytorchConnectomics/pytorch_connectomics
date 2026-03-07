"""
Evaluation metrics for PyTorch Connectomics.

This package provides comprehensive evaluation metrics:
- metrics_seg.py: Segmentation metrics (Adapted Rand, VOI, instance matching)
- metrics_skel.py: Skeleton-based metrics for curvilinear structures

Note: PyTorch Lightning handles training monitoring and logging.

Import patterns:
    from connectomics.metrics import AdaptedRandError, VariationOfInformation
    from connectomics.metrics import evaluate_image_pair, evaluate_directory
    from connectomics.metrics.segmentation_numpy import adapted_rand, instance_matching
"""

from .metrics_seg import (
    adapted_rand,
    instance_matching,
    instance_matching_simple,
    matching_criteria,
    AdaptedRandError,
    VariationOfInformation,
    InstanceAccuracy,
    InstanceAccuracySimple,
)
from .metrics_skel import (
    evaluate_image_pair,
    evaluate_file_pair,
    evaluate_directory,
)

__all__ = [
    # Segmentation metrics (numpy)
    "adapted_rand",
    "instance_matching",
    "instance_matching_simple",
    "matching_criteria",
    # Segmentation metrics (torchmetrics wrappers)
    "AdaptedRandError",
    "VariationOfInformation",
    "InstanceAccuracy",
    "InstanceAccuracySimple",
    # Skeleton metrics
    "evaluate_image_pair",
    "evaluate_file_pair",
    "evaluate_directory",
]
