"""Evaluation reporting and Lightning metric update helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchmetrics

from ..decoding.experiment_log import log_decode_experiment
from ..runtime.output_naming import final_prediction_output_tag
from .context import EvaluationContext
from .metrics import (
    align_metric_tensors,
    compute_binary_metrics,
    compute_instance_metrics,
    is_instance_segmentation,
)
from .nerl import compute_nerl_metrics

logger = logging.getLogger(__name__)


def configured_evaluation_metrics(context: EvaluationContext) -> set[str]:
    return context.requested_metrics


def evaluation_metric_requested(context: EvaluationContext, metric_name: str) -> bool:
    return context.metric_requested(metric_name)


def is_test_evaluation_enabled(context: EvaluationContext) -> bool:
    return context.is_enabled


def save_metrics_to_file(context: EvaluationContext, metrics_dict: Dict[str, Any]) -> None:
    """Save per-volume evaluation metrics in the configured test output directory."""
    metric_keys = [k for k in metrics_dict.keys() if k != "volume_name"]
    if not metric_keys:
        return

    output_path = context.resolved_output_path()
    if output_path is None:
        logger.warning("Cannot save metrics: output_path not found for mode=test")
        return

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_name = metrics_dict.get("volume_name", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = final_prediction_output_tag(
        context.cfg,
        checkpoint_path=context.checkpoint_path,
    )
    metrics_file = output_dir / f"evaluation_metrics_{volume_name}_{tag}.txt"

    if "nerl_per_gt_erl" in metrics_dict:
        try:
            per_gt_erl_file = (
                output_dir / f"evaluation_metrics_{volume_name}_{tag}_nerl_per_gt_erl.npz"
            )
            np.savez_compressed(
                per_gt_erl_file,
                gt_segment_id=np.asarray(metrics_dict.get("nerl_gt_segment_ids", [])),
                erl=np.asarray(metrics_dict["nerl_per_gt_erl"], dtype=np.float64),
            )
            metrics_dict["nerl_per_gt_erl_file"] = str(per_gt_erl_file)
        except Exception as exc:
            logger.warning("Failed to save NERL per-GT ERL file: %s", exc)

    try:
        with open(metrics_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Volume: {volume_name}\n")
            f.write("=" * 80 + "\n\n")

            if "adapted_rand_error" in metrics_dict:
                f.write("Instance Segmentation Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(
                    "  Adapted Rand Error:           " f"{metrics_dict['adapted_rand_error']:.6f}\n"
                )

                if "voi_split" in metrics_dict:
                    f.write(f"  VOI Split:                    {metrics_dict['voi_split']:.6f}\n")
                    f.write(f"  VOI Merge:                    {metrics_dict['voi_merge']:.6f}\n")
                    f.write(f"  VOI Total:                    {metrics_dict['voi_total']:.6f}\n")

                if "instance_accuracy" in metrics_dict:
                    f.write(
                        "  Instance Accuracy:            "
                        f"{metrics_dict['instance_accuracy']:.6f}\n"
                    )

                if "instance_accuracy_detail" in metrics_dict:
                    f.write(
                        "\n  Instance Accuracy (Detail):   "
                        f"{metrics_dict['instance_accuracy_detail']:.6f}\n"
                    )
                    f.write(
                        "    Precision:                  "
                        f"{metrics_dict['instance_precision_detail']:.6f}\n"
                    )
                    f.write(
                        "    Recall:                     "
                        f"{metrics_dict['instance_recall_detail']:.6f}\n"
                    )
                    f.write(
                        "    F1:                         "
                        f"{metrics_dict['instance_f1_detail']:.6f}\n"
                    )
                f.write("\n")

            if "jaccard" in metrics_dict or "dice" in metrics_dict:
                f.write("Binary/Semantic Segmentation Metrics:\n")
                f.write("-" * 80 + "\n")
                if "jaccard" in metrics_dict:
                    f.write(f"  Jaccard Index:                {metrics_dict['jaccard']:.6f}\n")
                if "dice" in metrics_dict:
                    f.write(f"  Dice Score:                   {metrics_dict['dice']:.6f}\n")
                if "accuracy" in metrics_dict:
                    f.write(f"  Accuracy:                     {metrics_dict['accuracy']:.6f}\n")
                f.write("\n")

            if "nerl" in metrics_dict:
                f.write("Neurite ERL Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  NERL:                         {metrics_dict['nerl']:.6f}\n")
                pred_erl = metrics_dict.get("nerl_pred_erl", metrics_dict.get("nerl_erl"))
                gt_erl = metrics_dict.get("nerl_gt_erl", metrics_dict.get("nerl_max_erl"))
                if pred_erl is not None:
                    f.write(f"  Pred ERL:                     {pred_erl:.6f}\n")
                if gt_erl is not None:
                    f.write(f"  GT ERL:                       {gt_erl:.6f}\n")
                if "nerl_num_skeletons" in metrics_dict:
                    f.write(
                        f"  Skeletons:                    {metrics_dict['nerl_num_skeletons']}\n"
                    )
                if "nerl_graph" in metrics_dict:
                    f.write(f"  Graph:                        {metrics_dict['nerl_graph']}\n")
                if "nerl_per_gt_erl_file" in metrics_dict:
                    f.write(
                        "  Per-GT ERL File:              "
                        f"{metrics_dict['nerl_per_gt_erl_file']}\n"
                    )
                f.write("\n")

            f.write("=" * 80 + "\n")

        logger.info("Metrics saved to: %s", metrics_file)
    except Exception as exc:
        logger.warning("Failed to save metrics to file: %s", exc)

    log_decode_experiment(
        cfg=context.cfg,
        output_dir=output_dir,
        volume_name=volume_name,
        timestamp=timestamp,
        metrics_dict=metrics_dict,
        checkpoint_path=context.checkpoint_path,
    )


def _persist_metrics(context: EvaluationContext, metrics_dict: Dict[str, Any]) -> None:
    if context.persist_metrics(metrics_dict):
        return
    save_metrics_to_file(context, metrics_dict)


def compute_test_metrics(
    context: EvaluationContext,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    volume_name: str | None = None,
) -> None:
    """Update configured metrics and save per-volume evaluation summaries."""
    if not is_test_evaluation_enabled(context):
        return

    volume_prefix = f"[{volume_name}] " if volume_name else ""
    metrics_dict: Dict[str, Any] = {"volume_name": volume_name if volume_name else "unknown"}
    requested_metrics = configured_evaluation_metrics(context)

    if "nerl" in requested_metrics:
        compute_nerl_metrics(
            context,
            decoded_predictions,
            volume_prefix,
            metrics_dict,
            volume_name,
        )

    if labels is None:
        _persist_metrics(context, metrics_dict)
        return

    pred_tensor = torch.from_numpy(decoded_predictions)
    labels_tensor = labels.detach().cpu() if labels.is_cuda else labels.detach()
    pred_tensor, labels_tensor = align_metric_tensors(pred_tensor, labels_tensor)
    if pred_tensor is None or labels_tensor is None:
        _persist_metrics(context, metrics_dict)
        return

    prediction_threshold = float(
        context.cfg_value(context.evaluation_cfg, "prediction_threshold", 0.5)
    )
    instance_iou_threshold = float(
        context.cfg_value(context.evaluation_cfg, "instance_iou_threshold", 0.5)
    )

    if is_instance_segmentation(pred_tensor):
        compute_instance_metrics(
            context,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            instance_iou_threshold,
        )
    else:
        compute_binary_metrics(
            context,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            prediction_threshold,
        )

    _persist_metrics(context, metrics_dict)


def log_test_epoch_metrics(context: EvaluationContext) -> None:
    """Log aggregated test metrics once after all ranks finish processing."""
    if not is_test_evaluation_enabled(context):
        return

    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_dist else 0
    if context.distributed_single_volume_sharding and rank != 0:
        return
    sync_dist = not context.distributed_single_volume_sharding

    adapted_rand_metric = context.metric("adapted_rand")
    if isinstance(adapted_rand_metric, torchmetrics.Metric):
        epoch_stats = adapted_rand_metric.compute()
        if isinstance(epoch_stats, dict):
            context.log_metric(
                "test_adapted_rand",
                epoch_stats["adapted_rand_error"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
            context.log_metric(
                "test_adapted_rand_precision",
                epoch_stats["adapted_rand_precision"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
            context.log_metric(
                "test_adapted_rand_recall",
                epoch_stats["adapted_rand_recall"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
        else:
            context.log_metric(
                "test_adapted_rand",
                epoch_stats,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )

    voi_metric = context.metric("voi")
    if isinstance(voi_metric, torchmetrics.Metric):
        context.log_metric(
            "test_voi",
            voi_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )
        context.log_metric(
            "test_voi_split",
            voi_metric.compute_split(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        context.log_metric(
            "test_voi_merge",
            voi_metric.compute_merge(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )

    instance_accuracy_metric = context.metric("instance_accuracy")
    if isinstance(instance_accuracy_metric, torchmetrics.Metric):
        context.log_metric(
            "test_instance_accuracy",
            instance_accuracy_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    instance_accuracy_detail_metric = context.metric("instance_accuracy_detail")
    if isinstance(instance_accuracy_detail_metric, torchmetrics.Metric):
        context.log_metric(
            "test_instance_accuracy_detail",
            instance_accuracy_detail_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )
        context.log_metric(
            "test_instance_precision_detail",
            instance_accuracy_detail_metric.compute_precision(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        context.log_metric(
            "test_instance_recall_detail",
            instance_accuracy_detail_metric.compute_recall(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        context.log_metric(
            "test_instance_f1_detail",
            instance_accuracy_detail_metric.compute_f1(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )

    jaccard_metric = context.metric("jaccard")
    if jaccard_metric is not None:
        context.log_metric(
            "test_jaccard",
            jaccard_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    dice_metric = context.metric("dice")
    if dice_metric is not None:
        context.log_metric(
            "test_dice",
            dice_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    accuracy_metric = context.metric("accuracy")
    if accuracy_metric is not None:
        context.log_metric(
            "test_accuracy",
            accuracy_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )


__all__ = [
    "compute_test_metrics",
    "configured_evaluation_metrics",
    "evaluation_metric_requested",
    "is_test_evaluation_enabled",
    "log_test_epoch_metrics",
    "save_metrics_to_file",
]
