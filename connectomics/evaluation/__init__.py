"""Evaluation stage orchestration."""

from .curvilinear import evaluate_directory, evaluate_file_pair
from .nerl import import_em_erl
from .report import (
    compute_test_metrics,
    configured_evaluation_metrics,
    evaluation_metric_requested,
    log_test_epoch_metrics,
    save_metrics_to_file,
)
from .stage import EvaluationStageResult, run_evaluation_stage

__all__ = [
    "EvaluationStageResult",
    "compute_test_metrics",
    "configured_evaluation_metrics",
    "evaluation_metric_requested",
    "import_em_erl",
    "log_test_epoch_metrics",
    "run_evaluation_stage",
    "save_metrics_to_file",
    "evaluate_file_pair",
    "evaluate_directory",
]
