"""Evaluation stage orchestration."""

from .curvilinear import evaluate_directory, evaluate_file_pair
from .stage import EvaluationStageResult, run_evaluation_stage

__all__ = [
    "EvaluationStageResult",
    "run_evaluation_stage",
    "evaluate_file_pair",
    "evaluate_directory",
]
