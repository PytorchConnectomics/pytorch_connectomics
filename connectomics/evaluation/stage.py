"""Evaluation stage orchestration helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .report import compute_test_metrics, evaluation_metric_requested, is_test_evaluation_enabled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationStageResult:
    """Result summary for one evaluation-stage call."""

    computed: bool
    duration_s: float = 0.0
    reason: str = ""


def run_evaluation_stage(
    module,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    *,
    filenames: list[str],
    batch_idx: int,
) -> EvaluationStageResult:
    """Run evaluation over decoded predictions."""
    evaluation_enabled = is_test_evaluation_enabled(module)
    nerl_requested = evaluation_enabled and evaluation_metric_requested(module, "nerl")
    if not evaluation_enabled:
        return EvaluationStageResult(computed=False, reason="evaluation disabled")
    if labels is None and not nerl_requested:
        return EvaluationStageResult(
            computed=False,
            reason="no ground truth labels or NERL graph metric",
        )

    start = time.time()
    volume_names = filenames if filenames else [f"volume_{batch_idx}"]

    if len(volume_names) > 1:
        pred_arr = np.asarray(decoded_predictions)
        can_split_pred = pred_arr.ndim > 0 and pred_arr.shape[0] == len(volume_names)
        can_split_label = (
            labels is not None and labels.ndim > 0 and labels.shape[0] == len(volume_names)
        )
        if can_split_pred and (labels is None or can_split_label):
            for i, name in enumerate(volume_names):
                label_i = None if labels is None else labels[i]
                compute_test_metrics(module, pred_arr[i], label_i, name)
        else:
            logger.warning(
                "Could not split batched predictions/labels by volume; "
                "computing a single aggregate metric."
            )
            compute_test_metrics(module, pred_arr, labels, volume_names[0])
    else:
        compute_test_metrics(module, decoded_predictions, labels, volume_names[0])

    return EvaluationStageResult(computed=True, duration_s=time.time() - start)


__all__ = ["EvaluationStageResult", "run_evaluation_stage"]
