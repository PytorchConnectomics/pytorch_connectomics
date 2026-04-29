"""Evaluation metric helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torchmetrics

from ..metrics.metrics_seg import AdaptedRandError
from ..metrics.segmentation_numpy import instance_matching, instance_matching_simple, voi

logger = logging.getLogger(__name__)


def align_metric_tensors(
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_tensor = pred_tensor.squeeze()
    labels_tensor = labels_tensor.squeeze()

    if pred_tensor.shape == labels_tensor.shape:
        return pred_tensor, labels_tensor

    logger.warning("Shape mismatch: pred=%s, labels=%s", pred_tensor.shape, labels_tensor.shape)
    if pred_tensor.ndim == labels_tensor.ndim - 1:
        pred_tensor = pred_tensor.unsqueeze(0)
    elif labels_tensor.ndim == pred_tensor.ndim - 1:
        labels_tensor = labels_tensor.unsqueeze(0)

    if pred_tensor.shape != labels_tensor.shape:
        logger.warning(
            "Cannot compute metrics: incompatible shapes after alignment, " "pred=%s, labels=%s",
            pred_tensor.shape,
            labels_tensor.shape,
        )
        return None, None

    return pred_tensor, labels_tensor


def is_instance_segmentation(pred_tensor: torch.Tensor) -> bool:
    return pred_tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ) or (pred_tensor.dtype == torch.float32 and pred_tensor.max() > 1.0)


def compute_instance_metrics(
    module,
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    instance_iou_threshold: float,
) -> None:
    pred_instances = pred_tensor.long()
    labels_instances = labels_tensor.long()

    if hasattr(module, "test_adapted_rand") and isinstance(
        module.test_adapted_rand, torchmetrics.Metric
    ):
        per_volume_metric = AdaptedRandError(return_all_stats=True).to(module.device)
        per_volume_metric.update(pred_instances.cpu(), labels_instances.cpu())
        adapted_rand_value = per_volume_metric.compute()
        if isinstance(adapted_rand_value, dict):
            are_score = adapted_rand_value.get(
                "adapted_rand_error",
                adapted_rand_value.get("are", list(adapted_rand_value.values())[0]),
            )
            are_score = are_score.item() if hasattr(are_score, "item") else float(are_score)
        else:
            are_score = adapted_rand_value.item()
        logger.info("%sAdapted Rand Error: %.6f", volume_prefix, are_score)
        if isinstance(adapted_rand_value, dict):
            for k, v in adapted_rand_value.items():
                val = v.item() if hasattr(v, "item") else float(v)
                logger.info("%s  %s: %.6f", volume_prefix, k, val)

        metrics_dict["adapted_rand_error"] = are_score
        module.test_adapted_rand.update(pred_instances.cpu(), labels_instances.cpu())

    if hasattr(module, "test_voi") and isinstance(module.test_voi, torchmetrics.Metric):
        split, merge = voi(pred_instances.cpu().numpy(), labels_instances.cpu().numpy())
        logger.info("%sVOI Split: %.6f", volume_prefix, split)
        logger.info("%sVOI Merge: %.6f", volume_prefix, merge)
        logger.info("%sVOI Total: %.6f", volume_prefix, split + merge)

        metrics_dict["voi_split"] = split
        metrics_dict["voi_merge"] = merge
        metrics_dict["voi_total"] = split + merge

        module.test_voi.update(pred_instances.cpu(), labels_instances.cpu())

    if hasattr(module, "test_instance_accuracy") and isinstance(
        module.test_instance_accuracy, torchmetrics.Metric
    ):
        stats = instance_matching(
            labels_instances.cpu().numpy(),
            pred_instances.cpu().numpy(),
            thresh=instance_iou_threshold,
            criterion="iou",
        )
        logger.info("%sInstance Accuracy: %.6f", volume_prefix, stats["accuracy"])
        metrics_dict["instance_accuracy"] = stats["accuracy"]

        module.test_instance_accuracy.update(pred_instances.cpu(), labels_instances.cpu())

    if hasattr(module, "test_instance_accuracy_detail") and isinstance(
        module.test_instance_accuracy_detail, torchmetrics.Metric
    ):
        stats_simple = instance_matching_simple(
            labels_instances.cpu().numpy(),
            pred_instances.cpu().numpy(),
            thresh=instance_iou_threshold,
            criterion="iou",
        )
        logger.info(
            "%sInstance Accuracy (Detail): %.6f [relaxed, non-Hungarian]",
            volume_prefix,
            stats_simple["accuracy"],
        )
        logger.info("%s  Precision: %.6f", volume_prefix, stats_simple["precision"])
        logger.info("%s  Recall: %.6f", volume_prefix, stats_simple["recall"])
        logger.info("%s  F1: %.6f", volume_prefix, stats_simple["f1"])

        metrics_dict["instance_accuracy_detail"] = stats_simple["accuracy"]
        metrics_dict["instance_precision_detail"] = stats_simple["precision"]
        metrics_dict["instance_recall_detail"] = stats_simple["recall"]
        metrics_dict["instance_f1_detail"] = stats_simple["f1"]

        module.test_instance_accuracy_detail.update(
            pred_instances.cpu(),
            labels_instances.cpu(),
        )


def compute_binary_metrics(
    module,
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    prediction_threshold: float,
) -> None:
    if pred_tensor.max() <= 1.0:
        pred_binary = (pred_tensor > prediction_threshold).long()
    else:
        pred_binary = (torch.sigmoid(pred_tensor) > prediction_threshold).long()

    labels_binary = (
        (labels_tensor > prediction_threshold).long()
        if labels_tensor.max() <= 1.0
        else labels_tensor.long()
    )

    if hasattr(module, "test_jaccard") and module.test_jaccard is not None:
        jaccard_value = torchmetrics.functional.jaccard_index(
            pred_binary,
            labels_binary,
            task="binary",
        )
        logger.info("%sJaccard: %.6f", volume_prefix, jaccard_value.item())
        metrics_dict["jaccard"] = jaccard_value.item()
        module.test_jaccard.update(pred_binary, labels_binary)

    if hasattr(module, "test_dice") and module.test_dice is not None:
        dice_value = torchmetrics.functional.dice(pred_binary, labels_binary)
        logger.info("%sDice: %.6f", volume_prefix, dice_value.item())
        metrics_dict["dice"] = dice_value.item()
        module.test_dice.update(pred_binary, labels_binary)

    if hasattr(module, "test_accuracy") and module.test_accuracy is not None:
        accuracy_value = torchmetrics.functional.accuracy(
            pred_binary,
            labels_binary,
            task="binary",
        )
        logger.info("%sAccuracy: %.6f", volume_prefix, accuracy_value.item())
        metrics_dict["accuracy"] = accuracy_value.item()
        module.test_accuracy.update(pred_binary, labels_binary)


__all__ = [
    "align_metric_tensors",
    "compute_binary_metrics",
    "compute_instance_metrics",
    "is_instance_segmentation",
]
