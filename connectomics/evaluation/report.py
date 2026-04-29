"""Evaluation reporting and Lightning metric update helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchmetrics

from ..runtime.output_naming import final_prediction_output_tag
from .metrics import (
    align_metric_tensors,
    compute_binary_metrics,
    compute_instance_metrics,
    is_instance_segmentation,
)
from .nerl import compute_nerl_metrics, get_effective_evaluation_config, module_cfg_value

logger = logging.getLogger(__name__)


def configured_evaluation_metrics(module) -> set[str]:
    evaluation_cfg = get_effective_evaluation_config(module)
    metrics = module_cfg_value(module, evaluation_cfg, "metrics", None)
    if metrics is None:
        return set()
    if isinstance(metrics, str):
        return {metrics.lower()}
    return {str(metric).lower() for metric in metrics}


def evaluation_metric_requested(module, metric_name: str) -> bool:
    return metric_name.lower() in configured_evaluation_metrics(module)


def is_test_evaluation_enabled(module) -> bool:
    if hasattr(module, "_is_test_evaluation_enabled"):
        return bool(module._is_test_evaluation_enabled())
    evaluation_cfg = get_effective_evaluation_config(module)
    return bool(module_cfg_value(module, evaluation_cfg, "enabled", False))


def _prediction_checkpoint_path(module) -> str | Path | None:
    if hasattr(module, "_get_prediction_checkpoint_path"):
        return module._get_prediction_checkpoint_path()
    return getattr(module, "_prediction_checkpoint_path", None)


def _runtime_output_path(module) -> str | None:
    if hasattr(module, "_get_runtime_inference_config"):
        inference_cfg = module._get_runtime_inference_config()
    else:
        inference_cfg = getattr(getattr(module, "cfg", None), "inference", None)
    save_prediction_cfg = getattr(inference_cfg, "save_prediction", None)
    return getattr(save_prediction_cfg, "output_path", None)


def save_metrics_to_file(module, metrics_dict: Dict[str, Any]) -> None:
    """Save per-volume evaluation metrics in the configured test output directory."""
    metric_keys = [k for k in metrics_dict.keys() if k != "volume_name"]
    if not metric_keys:
        return

    output_path = _runtime_output_path(module)
    if output_path is None:
        logger.warning("Cannot save metrics: output_path not found for mode=test")
        return

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_name = metrics_dict.get("volume_name", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = final_prediction_output_tag(
        module.cfg,
        checkpoint_path=_prediction_checkpoint_path(module),
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

    if hasattr(module, "_log_decode_experiment"):
        module._log_decode_experiment(output_dir, volume_name, timestamp, metrics_dict)


def _persist_metrics(module, metrics_dict: Dict[str, Any]) -> None:
    if hasattr(module, "save_metrics_to_file"):
        module.save_metrics_to_file(metrics_dict)
        return
    if hasattr(module, "_save_metrics_to_file"):
        module._save_metrics_to_file(metrics_dict)
        return
    save_metrics_to_file(module, metrics_dict)


def compute_test_metrics(
    module,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    volume_name: str | None = None,
) -> None:
    """Update configured metrics and save per-volume evaluation summaries."""
    if not is_test_evaluation_enabled(module):
        return

    volume_prefix = f"[{volume_name}] " if volume_name else ""
    metrics_dict: Dict[str, Any] = {"volume_name": volume_name if volume_name else "unknown"}
    requested_metrics = configured_evaluation_metrics(module)

    if "nerl" in requested_metrics:
        compute_nerl_metrics(
            module,
            decoded_predictions,
            volume_prefix,
            metrics_dict,
            volume_name,
        )

    if labels is None:
        _persist_metrics(module, metrics_dict)
        return

    pred_tensor = torch.from_numpy(decoded_predictions).float().to(module.device)
    labels_tensor = labels.float().to(pred_tensor.device)
    pred_tensor, labels_tensor = align_metric_tensors(pred_tensor, labels_tensor)
    if pred_tensor is None or labels_tensor is None:
        _persist_metrics(module, metrics_dict)
        return

    evaluation_cfg = get_effective_evaluation_config(module)
    prediction_threshold = float(
        module_cfg_value(module, evaluation_cfg, "prediction_threshold", 0.5)
    )
    instance_iou_threshold = float(
        module_cfg_value(module, evaluation_cfg, "instance_iou_threshold", 0.5)
    )

    if is_instance_segmentation(pred_tensor):
        compute_instance_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            instance_iou_threshold,
        )
    else:
        compute_binary_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            prediction_threshold,
        )

    _persist_metrics(module, metrics_dict)


def _is_distributed_tta_sharding_active(module) -> bool:
    inference_manager = getattr(module, "inference_manager", None)
    if inference_manager is None:
        return False
    return bool(inference_manager.is_distributed_tta_sharding_enabled())


def _is_distributed_window_sharding_active(module) -> bool:
    inference_manager = getattr(module, "inference_manager", None)
    if inference_manager is None:
        return False
    if not hasattr(inference_manager, "is_distributed_window_sharding_enabled"):
        return False
    return bool(inference_manager.is_distributed_window_sharding_enabled())


def _is_distributed_single_volume_sharding_active(module) -> bool:
    return _is_distributed_tta_sharding_active(module) or _is_distributed_window_sharding_active(
        module
    )


def log_test_epoch_metrics(module) -> None:
    """Log aggregated test metrics once after all ranks finish processing."""
    if not is_test_evaluation_enabled(module):
        return

    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_dist else 0
    distributed_single_volume_sharding = _is_distributed_single_volume_sharding_active(module)
    if distributed_single_volume_sharding and rank != 0:
        return
    sync_dist = not distributed_single_volume_sharding

    if hasattr(module, "test_adapted_rand") and isinstance(
        module.test_adapted_rand, torchmetrics.Metric
    ):
        epoch_stats = module.test_adapted_rand.compute()
        if isinstance(epoch_stats, dict):
            module.log(
                "test_adapted_rand",
                epoch_stats["adapted_rand_error"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
            module.log(
                "test_adapted_rand_precision",
                epoch_stats["adapted_rand_precision"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
            module.log(
                "test_adapted_rand_recall",
                epoch_stats["adapted_rand_recall"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )
        else:
            module.log(
                "test_adapted_rand",
                epoch_stats,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
            )

    if hasattr(module, "test_voi") and isinstance(module.test_voi, torchmetrics.Metric):
        module.log(
            "test_voi",
            module.test_voi,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )
        module.log(
            "test_voi_split",
            module.test_voi.compute_split(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        module.log(
            "test_voi_merge",
            module.test_voi.compute_merge(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )

    if hasattr(module, "test_instance_accuracy") and isinstance(
        module.test_instance_accuracy, torchmetrics.Metric
    ):
        module.log(
            "test_instance_accuracy",
            module.test_instance_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    if hasattr(module, "test_instance_accuracy_detail") and isinstance(
        module.test_instance_accuracy_detail, torchmetrics.Metric
    ):
        module.log(
            "test_instance_accuracy_detail",
            module.test_instance_accuracy_detail,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )
        module.log(
            "test_instance_precision_detail",
            module.test_instance_accuracy_detail.compute_precision(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        module.log(
            "test_instance_recall_detail",
            module.test_instance_accuracy_detail.compute_recall(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )
        module.log(
            "test_instance_f1_detail",
            module.test_instance_accuracy_detail.compute_f1(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=sync_dist,
        )

    if hasattr(module, "test_jaccard") and module.test_jaccard is not None:
        module.log(
            "test_jaccard",
            module.test_jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    if hasattr(module, "test_dice") and module.test_dice is not None:
        module.log(
            "test_dice",
            module.test_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )

    if hasattr(module, "test_accuracy") and module.test_accuracy is not None:
        module.log(
            "test_accuracy",
            module.test_accuracy,
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
