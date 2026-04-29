"""Evaluation stage orchestration helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationStageResult:
    """Result summary for one evaluation-stage call."""

    computed: bool
    duration_s: float = 0.0
    reason: str = ""


def run_evaluation_stage(
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    *,
    filenames: list[str],
    batch_idx: int,
    evaluation_enabled: bool,
    nerl_requested: bool,
    compute_metrics_fn: Callable[[np.ndarray, Optional[torch.Tensor], str | None], None],
) -> EvaluationStageResult:
    """Run evaluation over decoded predictions.

    The metric implementation callback is injected so this stage stays
    independent from Lightning while the existing module metrics are migrated.
    """
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
                compute_metrics_fn(pred_arr[i], label_i, name)
        else:
            logger.warning(
                "Could not split batched predictions/labels by volume; "
                "computing a single aggregate metric."
            )
            compute_metrics_fn(pred_arr, labels, volume_names[0])
    else:
        compute_metrics_fn(decoded_predictions, labels, volume_names[0])

    return EvaluationStageResult(computed=True, duration_s=time.time() - start)


__all__ = ["EvaluationStageResult", "run_evaluation_stage"]
