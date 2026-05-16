"""
Evaluation metrics for PyTorch Connectomics.

This package provides comprehensive evaluation metrics:
- metrics_seg.py: Segmentation metrics (Adapted Rand, VOI, instance matching)
- metrics_skel.py: Skeleton-based metrics for curvilinear structures
- nerl.py: NERL scoring helpers backed by em_erl

Note: PyTorch Lightning handles training monitoring and logging.

Import patterns:
    from connectomics.metrics import AdaptedRandError, VariationOfInformation
    from connectomics.metrics import evaluate_image_pair
    from connectomics.evaluation import evaluate_directory
    from connectomics.metrics.segmentation_numpy import adapted_rand, instance_matching
"""

from .metrics_seg import (
    AdaptedRandError,
    InstanceAccuracy,
    InstanceAccuracySimple,
    VariationOfInformation,
    adapted_rand,
    instance_matching,
    instance_matching_simple,
    matching_criteria,
)
from .metrics_skel import evaluate_image_pair
from .nerl import (
    NerlGraphOptions,
    NerlScoreResult,
    compute_nerl_score,
    compute_nerl_score_details,
    import_em_erl,
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
    # NERL metrics
    "NerlGraphOptions",
    "NerlScoreResult",
    "compute_nerl_score",
    "compute_nerl_score_details",
    "import_em_erl",
]
