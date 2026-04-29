"""Helpers for test-time inference, decoding, postprocessing, and metrics."""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ...data.processing.affinity import (
    compute_affinity_crop_pad,
    crop_spatial_by_pad,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
)
from ...data.processing.misc import get_padsize
from ...decoding import run_decoding_stage
from ...evaluation import run_evaluation_stage
from ...inference import (
    apply_prediction_transform,
    is_chunked_inference_enabled,
    run_chunked_affinity_cc_inference,
    run_chunked_prediction_inference,
    write_outputs,
)
from ...inference.lazy import (
    get_lazy_image_reference_shape,
    lazy_predict_volume,
    load_lazy_volume,
)
from ...metrics.metrics_seg import AdaptedRandError
from ...metrics.segmentation_numpy import instance_matching, instance_matching_simple, voi
from ...utils.channel_slices import resolve_channel_indices, resolve_channel_range
from ...utils.model_outputs import (
    get_model_head_names,
    resolve_output_heads,
)
from .utils import (
    final_prediction_output_tag,
    is_tta_cache_suffix,
    tta_cache_suffix,
)

logger = logging.getLogger(__name__)


@dataclass
class TestContext:
    """Explicit contract between ConnectomicsModule and test_pipeline functions.

    Bundles the resolved config values, inference manager, and device reference
    so test_pipeline functions don't need to call private methods on the module.
    The ``module`` reference is retained only for Lightning-specific operations
    (metric logging via ``module.log()``, metric attribute access).
    """

    cfg: Any
    inference_cfg: Any
    evaluation_cfg: Any
    device: torch.device
    inference_manager: Any
    evaluation_enabled: bool = False
    prediction_threshold: float = 0.5
    instance_iou_threshold: float = 0.5
    save_prediction_cfg: Any = None
    filenames: List[str] = field(default_factory=list)
    output_dir_value: Optional[str] = None
    cache_suffix: str = "_x1_prediction.h5"

    @staticmethod
    def from_module(module, batch: Dict[str, Any]) -> "TestContext":
        """Build a TestContext from a ConnectomicsModule for a given batch."""
        inference_cfg = module._get_runtime_inference_config()
        evaluation_cfg = module._get_test_evaluation_config()
        evaluation_enabled = module._is_test_evaluation_enabled()

        prediction_threshold = 0.5
        instance_iou_threshold = 0.5
        if evaluation_enabled and evaluation_cfg is not None:
            prediction_threshold = module._cfg_float(
                evaluation_cfg,
                "prediction_threshold",
                0.5,
            )
            instance_iou_threshold = module._cfg_float(
                evaluation_cfg,
                "instance_iou_threshold",
                0.5,
            )

        mode, output_dir_value, cache_suffix, filenames = module._resolve_test_output_config(batch)
        save_prediction_cfg = inference_cfg.save_prediction

        return TestContext(
            cfg=module.cfg,
            inference_cfg=inference_cfg,
            evaluation_cfg=evaluation_cfg,
            device=module.device,
            inference_manager=module.inference_manager,
            evaluation_enabled=evaluation_enabled,
            prediction_threshold=prediction_threshold,
            instance_iou_threshold=instance_iou_threshold,
            save_prediction_cfg=save_prediction_cfg,
            filenames=filenames,
            output_dir_value=output_dir_value,
            cache_suffix=cache_suffix,
        )


def _cleanup_inference_memory(module, stage: str, *, release_model: bool = False) -> None:
    """Release temporary inference memory according to inference.memory_cleanup."""
    inference_cfg = getattr(getattr(module, "cfg", None), "inference", None)
    cleanup_cfg = getattr(inference_cfg, "memory_cleanup", None)
    if cleanup_cfg is not None and not bool(getattr(cleanup_cfg, "enabled", True)):
        return

    release_model_requested = release_model and bool(
        getattr(cleanup_cfg, "release_model_after_inference", False)
        if cleanup_cfg is not None
        else False
    )
    if release_model_requested and hasattr(module, "model"):
        try:
            module.model.to("cpu")
            if hasattr(module, "inference_manager"):
                module.inference_manager.model = module.model
            logger.info("Moved model to CPU after inference to free accelerator memory.")
        except Exception as exc:
            logger.warning(f"Model CPU release failed after inference: {exc}")

    if cleanup_cfg is None or bool(getattr(cleanup_cfg, "gc_collect", True)):
        gc.collect()

    if cleanup_cfg is None or bool(getattr(cleanup_cfg, "empty_cuda_cache", True)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleared CUDA cache after {stage}.")


def _resolve_inference_crop_pad(module) -> Optional[tuple[tuple[int, int], ...]]:
    """Return configured symmetric or asymmetric prediction crop, if any."""
    inference_cfg = None
    if hasattr(module, "_get_runtime_inference_config"):
        inference_cfg = module._get_runtime_inference_config()
    elif hasattr(module, "cfg") and hasattr(module.cfg, "inference"):
        inference_cfg = module.cfg.inference
    if inference_cfg is None:
        return None

    crop_pad = getattr(inference_cfg, "crop_pad", None)
    if crop_pad is None:
        return None

    crop_pad_values = tuple(int(v) for v in crop_pad)
    if not crop_pad_values or not any(crop_pad_values):
        return None
    if any(v < 0 for v in crop_pad_values):
        raise ValueError(f"inference.crop_pad must be non-negative, got {crop_pad_values}")

    if len(crop_pad_values) in (2, 4):
        spatial_rank = 2
    elif len(crop_pad_values) in (3, 6):
        spatial_rank = 3
    else:
        raise ValueError(
            "inference.crop_pad must have length 2/3 for symmetric cropping "
            f"or 4/6 for asymmetric cropping, got {crop_pad_values}."
        )

    return tuple(get_padsize(list(crop_pad_values), ndim=spatial_rank))


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


def _should_skip_postprocess_on_rank(module) -> bool:
    inference_manager = getattr(module, "inference_manager", None)
    if inference_manager is None:
        return False
    return bool(inference_manager.should_skip_postprocess_on_rank())


def _distributed_tta_barrier(module) -> None:
    if not _is_distributed_tta_sharding_active(module):
        return
    # Rank 0 may spend a long time in CPU-side decoding/evaluation after the
    # distributed TTA reduction. Holding nonzero ranks in an NCCL barrier here
    # can trip the watchdog during long ABISS runs, so treat this as a no-op.
    return


def _is_unstacked_test_batch(batch: Dict[str, Any]) -> bool:
    """Return True when collate preserved per-sample tensors as a Python list."""
    return isinstance(batch.get("image"), (list, tuple))


def _wrap_single_sample_value(value: Any) -> Any:
    """Convert a collated list entry back into a singleton batch value."""
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return value
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    return [value]


def _extract_single_sample_batch(batch: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    """Slice a list-collated test batch into a regular singleton batch."""
    sample_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            if sample_idx >= len(value):
                continue
            sample_batch[key] = _wrap_single_sample_value(value[sample_idx])
        else:
            sample_batch[key] = value
    return sample_batch


def _coerce_singleton_batch_tensor(value: Any) -> Any:
    """Normalize singleton list wrappers around image/label/mask tensors."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        inner = _coerce_singleton_batch_tensor(value[0])
        if torch.is_tensor(inner):
            return inner if inner.ndim > 0 and inner.shape[0] == 1 else inner.unsqueeze(0)
    return value


def _is_lazy_test_sample(batch: Dict[str, Any]) -> bool:
    image = batch.get("image")
    return isinstance(image, (str, os.PathLike))


def _maybe_load_lazy_labels(
    module,
    label_value: Any,
    *,
    mode: str,
) -> Optional[torch.Tensor]:
    if label_value is None:
        return None
    if not isinstance(label_value, (str, os.PathLike)):
        return _coerce_singleton_batch_tensor(label_value)

    label_np = load_lazy_volume(module.cfg, str(label_value), kind="label", mode=mode)
    return torch.from_numpy(label_np[np.newaxis, ...])


def _resolve_mask_align_to_image(module) -> bool:
    mask_transform_cfg = getattr(module.cfg.data, "mask_transform", None) or getattr(
        module.cfg.data, "data_transform", None
    )
    if mask_transform_cfg is None:
        return False
    return bool(getattr(mask_transform_cfg, "align_to_image", False))


def _crop_spatial_border(
    data: np.ndarray | torch.Tensor,
    crop_pad: tuple[tuple[int, int], ...],
    *,
    item_name: str,
) -> np.ndarray | torch.Tensor:
    """Crop border padding from the last spatial dimensions."""
    return crop_spatial_by_pad(data, crop_pad, item_name=item_name)


def _apply_prediction_crop_pad_if_needed(
    module,
    data: np.ndarray | torch.Tensor,
    reference_image_shape: tuple[int, ...],
    *,
    item_name: str,
) -> np.ndarray | torch.Tensor:
    """Crop prediction-like tensors back to the pre-context-pad spatial shape."""
    crop_pad = _resolve_inference_crop_pad(module)
    if crop_pad is None:
        return data

    if len(reference_image_shape) < len(crop_pad):
        raise ValueError(
            "reference_image_shape rank must be >= inference.crop_pad rank. "
            f"Got reference_image_shape={reference_image_shape}, crop_pad={crop_pad}"
        )

    spatial_slice = slice(-len(crop_pad), None)
    padded_spatial_shape = tuple(int(v) for v in reference_image_shape[spatial_slice])
    expected_cropped_shape = tuple(
        padded_spatial_shape[i] - crop_pad[i][0] - crop_pad[i][1] for i in range(len(crop_pad))
    )
    if any(size <= 0 for size in expected_cropped_shape):
        raise ValueError(
            "inference.crop_pad is too large for the padded input shape. "
            f"crop_pad={crop_pad}, padded_shape={padded_spatial_shape}"
        )

    data_spatial_shape = tuple(int(v) for v in data.shape[spatial_slice])
    if data_spatial_shape == expected_cropped_shape:
        return data
    if data_spatial_shape != padded_spatial_shape:
        raise ValueError(
            "Cannot apply inference.crop_pad to "
            f"{item_name}: spatial shape {data_spatial_shape} matches neither "
            f"padded input {padded_spatial_shape} nor cropped shape "
            f"{expected_cropped_shape}."
        )

    cropped = _crop_spatial_border(data, crop_pad, item_name=item_name)
    logger.info(f"Cropped {item_name}: {tuple(data.shape)} -> {tuple(cropped.shape)}")
    return cropped


def _resolve_reference_spatial_shape_after_crop_pad(
    module,
    reference_image_shape: tuple[int, ...],
) -> tuple[int, ...]:
    crop_pad = _resolve_inference_crop_pad(module)
    spatial_rank = 3 if len(reference_image_shape) >= 3 else len(reference_image_shape)
    reference_spatial_shape = tuple(int(v) for v in reference_image_shape[-spatial_rank:])
    if crop_pad is None:
        return reference_spatial_shape

    crop_rank = len(crop_pad)
    unchanged_prefix = reference_spatial_shape[:-crop_rank]
    cropped_suffix = tuple(
        reference_spatial_shape[len(unchanged_prefix) + axis]
        - crop_pad[axis][0]
        - crop_pad[axis][1]
        for axis in range(crop_rank)
    )
    return unchanged_prefix + cropped_suffix


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _resolve_output_head_target_slice(module, output_head: Optional[str]) -> Any:
    if not output_head:
        return None
    heads = getattr(getattr(module.cfg, "model", None), "heads", None)
    head_cfg = _mapping_get(heads, output_head, None)
    return _mapping_get(head_cfg, "target_slice", None)


def _resolve_affinity_offsets_for_inference_output(
    module,
    *,
    output_head: Optional[str],
) -> list[tuple[int, int, int]]:
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        return []

    affinity_groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not affinity_groups:
        return []

    label_channels = max(end for (start, end), _offsets in affinity_groups)
    channel_offsets: list[Optional[tuple[int, int, int]]] = [None] * label_channels
    for (start, end), offsets in affinity_groups:
        for channel, offset in zip(range(start, end), offsets):
            channel_offsets[channel] = offset

    target_slice = _resolve_output_head_target_slice(module, output_head)
    if target_slice is not None:
        start_idx, end_idx = resolve_channel_range(
            target_slice,
            num_channels=label_channels,
            context=f"target_slice for output head {output_head!r}",
        )
        channel_offsets = channel_offsets[start_idx:end_idx]

    select_channel = getattr(getattr(module.cfg, "inference", None), "select_channel", None)
    if select_channel is not None:
        selected_indices = resolve_channel_indices(
            select_channel,
            num_channels=len(channel_offsets),
            context="inference.select_channel",
        )
        channel_offsets = [channel_offsets[idx] for idx in selected_indices]

    return [offset for offset in channel_offsets if offset is not None]


def _resolve_affinity_inference_crop(
    module,
    *,
    output_head: Optional[str] = None,
) -> Optional[tuple[tuple[int, int], ...]]:
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        return None
    affinity_mode = resolve_affinity_mode_from_cfg(cfg)
    if affinity_mode is None:
        return None

    offsets = _resolve_affinity_offsets_for_inference_output(module, output_head=output_head)
    if not offsets:
        return None

    crop_pad = compute_affinity_crop_pad(offsets, affinity_mode=affinity_mode)
    if not crop_pad or not any(before or after for before, after in crop_pad):
        return None
    return crop_pad


def _apply_affinity_inference_crop_if_needed(
    module,
    data: np.ndarray | torch.Tensor,
    *,
    reference_spatial_shape: tuple[int, ...],
    item_name: str,
    output_head: Optional[str] = None,
) -> np.ndarray | torch.Tensor:
    crop_pad = _resolve_affinity_inference_crop(module, output_head=output_head)
    if crop_pad is None:
        return data

    if len(reference_spatial_shape) != len(crop_pad):
        raise ValueError(
            f"Affinity crop rank mismatch for {item_name}: "
            f"reference_spatial_shape={reference_spatial_shape}, crop_pad={crop_pad}"
        )

    expected_cropped_shape = tuple(
        int(reference_spatial_shape[axis]) - crop_pad[axis][0] - crop_pad[axis][1]
        for axis in range(len(crop_pad))
    )
    if any(size <= 0 for size in expected_cropped_shape):
        raise ValueError(
            f"Affinity crop {crop_pad} is too large for {item_name} shape {reference_spatial_shape}"
        )

    spatial_slice = slice(-len(crop_pad), None)
    data_spatial_shape = tuple(int(v) for v in data.shape[spatial_slice])
    if data_spatial_shape == expected_cropped_shape:
        return data
    if data_spatial_shape != reference_spatial_shape:
        return data

    cropped = crop_spatial_by_pad(data, crop_pad, item_name=item_name)
    logger.info(f"Affinity-cropped {item_name}: {tuple(data.shape)} -> {tuple(cropped.shape)}")
    return cropped


def _apply_predecode_prediction_crops(
    module,
    data: np.ndarray | torch.Tensor,
    *,
    reference_image_shape: tuple[int, ...],
    item_name: str,
    output_head: Optional[str] = None,
) -> tuple[np.ndarray | torch.Tensor, tuple[int, ...]]:
    """Apply prediction-space crops before decoding affinities into instances."""
    original_shape = tuple(data.shape)
    data = _apply_prediction_crop_pad_if_needed(
        module,
        data,
        reference_image_shape,
        item_name=item_name,
    )
    reference_spatial_shape = _resolve_reference_spatial_shape_after_crop_pad(
        module, reference_image_shape
    )
    data = _apply_affinity_inference_crop_if_needed(
        module,
        data,
        reference_spatial_shape=reference_spatial_shape,
        item_name=item_name,
        output_head=output_head,
    )
    if isinstance(data, np.ndarray) and tuple(data.shape) != original_shape:
        data = np.array(data, copy=True, order="C")
        logger.info(
            "Compacted cropped %s storage to final shape %s (%.1f MiB)",
            item_name,
            tuple(data.shape),
            data.nbytes / (1024**2),
        )
        gc.collect()
    return data, reference_spatial_shape


def _align_metric_tensors(
    pred_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_tensor = pred_tensor.squeeze()
    labels_tensor = labels_tensor.squeeze()

    if pred_tensor.shape == labels_tensor.shape:
        return pred_tensor, labels_tensor

    logger.warning(f"Shape mismatch: pred={pred_tensor.shape}, labels={labels_tensor.shape}")
    if pred_tensor.ndim == labels_tensor.ndim - 1:
        pred_tensor = pred_tensor.unsqueeze(0)
    elif labels_tensor.ndim == pred_tensor.ndim - 1:
        labels_tensor = labels_tensor.unsqueeze(0)

    if pred_tensor.shape != labels_tensor.shape:
        logger.warning(
            "Cannot compute metrics: incompatible shapes after alignment, "
            f"pred={pred_tensor.shape}, labels={labels_tensor.shape}"
        )
        return None, None

    return pred_tensor, labels_tensor


def _is_instance_segmentation(pred_tensor: torch.Tensor) -> bool:
    return pred_tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ) or (pred_tensor.dtype == torch.float32 and pred_tensor.max() > 1.0)


def _compute_instance_metrics(
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
        logger.info(f"{volume_prefix}Adapted Rand Error: {are_score:.6f}")
        if isinstance(adapted_rand_value, dict):
            for k, v in adapted_rand_value.items():
                val = v.item() if hasattr(v, "item") else float(v)
                logger.info(f"{volume_prefix}  {k}: {val:.6f}")

        metrics_dict["adapted_rand_error"] = are_score
        module.test_adapted_rand.update(pred_instances.cpu(), labels_instances.cpu())

    if hasattr(module, "test_voi") and isinstance(module.test_voi, torchmetrics.Metric):
        split, merge = voi(pred_instances.cpu().numpy(), labels_instances.cpu().numpy())
        logger.info(f"{volume_prefix}VOI Split: {split:.6f}")
        logger.info(f"{volume_prefix}VOI Merge: {merge:.6f}")
        logger.info(f"{volume_prefix}VOI Total: {split + merge:.6f}")

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
        logger.info(f"{volume_prefix}Instance Accuracy: {stats['accuracy']:.6f}")
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
            f"{volume_prefix}Instance Accuracy (Detail): "
            f"{stats_simple['accuracy']:.6f} [relaxed, non-Hungarian]"
        )
        logger.info(f"{volume_prefix}  Precision: {stats_simple['precision']:.6f}")
        logger.info(f"{volume_prefix}  Recall: {stats_simple['recall']:.6f}")
        logger.info(f"{volume_prefix}  F1: {stats_simple['f1']:.6f}")

        metrics_dict["instance_accuracy_detail"] = stats_simple["accuracy"]
        metrics_dict["instance_precision_detail"] = stats_simple["precision"]
        metrics_dict["instance_recall_detail"] = stats_simple["recall"]
        metrics_dict["instance_f1_detail"] = stats_simple["f1"]

        module.test_instance_accuracy_detail.update(pred_instances.cpu(), labels_instances.cpu())


def _compute_binary_metrics(
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
        logger.info(f"{volume_prefix}Jaccard: {jaccard_value.item():.6f}")
        metrics_dict["jaccard"] = jaccard_value.item()
        module.test_jaccard.update(pred_binary, labels_binary)

    if hasattr(module, "test_dice") and module.test_dice is not None:
        dice_value = torchmetrics.functional.dice(pred_binary, labels_binary)
        logger.info(f"{volume_prefix}Dice: {dice_value.item():.6f}")
        metrics_dict["dice"] = dice_value.item()
        module.test_dice.update(pred_binary, labels_binary)

    if hasattr(module, "test_accuracy") and module.test_accuracy is not None:
        accuracy_value = torchmetrics.functional.accuracy(
            pred_binary,
            labels_binary,
            task="binary",
        )
        logger.info(f"{volume_prefix}Accuracy: {accuracy_value.item():.6f}")
        metrics_dict["accuracy"] = accuracy_value.item()
        module.test_accuracy.update(pred_binary, labels_binary)


def _module_cfg_value(module, cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(module, "_cfg_value"):
        return module._cfg_value(cfg, name, default)
    return getattr(cfg, name, default)


def _get_effective_evaluation_config(module) -> Any:
    evaluation_cfg = module._get_test_evaluation_config()
    if evaluation_cfg is not None:
        return evaluation_cfg
    return getattr(getattr(module, "cfg", None), "evaluation", None)


def _configured_evaluation_metrics(module) -> set[str]:
    evaluation_cfg = _get_effective_evaluation_config(module)
    metrics = _module_cfg_value(module, evaluation_cfg, "metrics", None)
    if metrics is None:
        return set()
    if isinstance(metrics, str):
        return {metrics.lower()}
    return {str(metric).lower() for metric in metrics}


def _evaluation_metric_requested(module, metric_name: str) -> bool:
    return metric_name.lower() in _configured_evaluation_metrics(module)


def _select_volume_config_value(value: Any, volume_name: str | None) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return value
    if isinstance(value, dict):
        if volume_name and volume_name in value:
            return value[volume_name]
        if "default" in value:
            return value["default"]
        if len(value) == 1:
            return next(iter(value.values()))
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else None
    return value


def _import_em_erl():
    try:
        from em_erl import ERLGraph, compute_erl_score, compute_segment_lut

        return ERLGraph, compute_erl_score, compute_segment_lut
    except ModuleNotFoundError:
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        em_erl_root = repo_root / "lib" / "em_erl"
        if em_erl_root.exists():
            sys.path.insert(0, str(em_erl_root))
        from em_erl import ERLGraph, compute_erl_score, compute_segment_lut

        return ERLGraph, compute_erl_score, compute_segment_lut


def _reorder_coordinate_axes(
    coords: np.ndarray,
    *,
    source_order: str,
    target_order: str | None,
) -> np.ndarray:
    source_order = str(source_order).lower()
    target_order = source_order if target_order is None else str(target_order).lower()
    valid_axes = {"x", "y", "z"}
    if len(source_order) != 3 or set(source_order) != valid_axes:
        raise ValueError(f"Invalid skeleton coordinate order: {source_order!r}")
    if len(target_order) != 3 or set(target_order) != valid_axes:
        raise ValueError(f"Invalid prediction coordinate order: {target_order!r}")
    axis_indices = [source_order.index(axis) for axis in target_order]
    return np.asarray(coords)[:, axis_indices]


def _networkx_skeleton_to_erl_graph(skeleton: Any, evaluation_cfg: Any, module: Any):
    ERLGraph, _, _ = _import_em_erl()

    id_attr = _module_cfg_value(module, evaluation_cfg, "nerl_skeleton_id_attribute", "id")
    pos_attr = _module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_position_attribute",
        "index_position",
    )
    edge_len_attr = _module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_edge_length_attribute",
        "edge_length",
    )
    source_order = _module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_skeleton_position_order",
        "xyz",
    )
    target_order = _module_cfg_value(
        module,
        evaluation_cfg,
        "nerl_prediction_position_order",
        None,
    )

    node_ids = list(skeleton.nodes)
    if not node_ids:
        raise ValueError("NERL skeleton has no nodes")

    raw_skeleton_ids = []
    node_coords = []
    for node_id in node_ids:
        node_data = skeleton.nodes[node_id]
        raw_skeleton_ids.append(node_data[id_attr])
        node_coords.append(node_data[pos_attr])

    skeleton_ids = list(dict.fromkeys(raw_skeleton_ids))
    skeleton_index_by_id = {skeleton_id: i for i, skeleton_id in enumerate(skeleton_ids)}
    node_index_by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    node_skeleton_index = np.asarray(
        [skeleton_index_by_id[skeleton_id] for skeleton_id in raw_skeleton_ids],
        dtype=np.uint32,
    )
    node_coords_arr = _reorder_coordinate_axes(
        np.asarray(node_coords, dtype=np.float32),
        source_order=source_order,
        target_order=target_order,
    )

    edge_buckets: list[list[tuple[int, int, float]]] = [[] for _ in skeleton_ids]
    skeleton_len = np.zeros(len(skeleton_ids), dtype=np.float64)
    for u, v, edge_data in skeleton.edges(data=True):
        if u not in node_index_by_id or v not in node_index_by_id:
            continue
        u_idx = node_index_by_id[u]
        v_idx = node_index_by_id[v]
        skel_idx = int(node_skeleton_index[u_idx])
        if skel_idx != int(node_skeleton_index[v_idx]):
            continue
        if edge_len_attr in edge_data:
            edge_len = float(edge_data[edge_len_attr])
        else:
            edge_len = float(np.linalg.norm(node_coords_arr[u_idx] - node_coords_arr[v_idx]))
        edge_buckets[skel_idx].append((u_idx, v_idx, edge_len))
        skeleton_len[skel_idx] += edge_len

    edge_ptr = [0]
    edge_u = []
    edge_v = []
    edge_len = []
    for bucket in edge_buckets:
        for u_idx, v_idx, length in bucket:
            edge_u.append(u_idx)
            edge_v.append(v_idx)
            edge_len.append(length)
        edge_ptr.append(len(edge_u))

    return ERLGraph(
        skeleton_id=np.asarray(skeleton_ids),
        skeleton_len=skeleton_len,
        node_skeleton_index=node_skeleton_index,
        node_coords_zyx=node_coords_arr,
        edge_u=np.asarray(edge_u, dtype=np.uint32),
        edge_v=np.asarray(edge_v, dtype=np.uint32),
        edge_len=np.asarray(edge_len, dtype=np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.uint64),
    )


def _load_nerl_graph(graph_source: Any, evaluation_cfg: Any, module: Any):
    ERLGraph, _, _ = _import_em_erl()
    if isinstance(graph_source, ERLGraph):
        return graph_source, False
    if hasattr(graph_source, "node_coords_zyx") and hasattr(graph_source, "edge_ptr"):
        return graph_source, False

    graph_path = Path(graph_source)
    suffix = graph_path.suffix.lower()
    if suffix == ".npz":
        return ERLGraph.from_npz(graph_path), False
    if suffix in {".pkl", ".pickle"}:
        import pickle

        with open(graph_path, "rb") as f:
            skeleton = pickle.load(f)
        return _networkx_skeleton_to_erl_graph(skeleton, evaluation_cfg, module), True
    raise ValueError(
        "evaluation.nerl_graph must be an ERLGraph .npz or "
        f"NetworkX skeleton pickle, got {graph_path}"
    )


def _nerl_node_positions(module, graph: Any, voxel_coords: bool, evaluation_cfg: Any) -> np.ndarray:
    if voxel_coords:
        return np.asarray(graph.node_coords_zyx, dtype=np.int64)

    resolution = _module_cfg_value(module, evaluation_cfg, "nerl_resolution", None)
    if resolution is None:
        data_cfg = getattr(getattr(module, "cfg", None), "data", None)
        test_cfg = getattr(data_cfg, "test", None)
        resolution = getattr(test_cfg, "resolution", None)
    return graph.get_nodes_position(resolution)


def _prepare_nerl_segmentation(decoded_predictions: np.ndarray) -> np.ndarray:
    seg = np.asarray(decoded_predictions)
    while seg.ndim > 3 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim > 3:
        singleton_axes = tuple(i for i, size in enumerate(seg.shape) if size == 1)
        if singleton_axes:
            seg = np.squeeze(seg, axis=singleton_axes)
    if seg.ndim != 3:
        raise ValueError(f"NERL expects a 3D decoded instance volume, got shape {seg.shape}")
    if not np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.uint32, copy=False)
    return seg


def _extract_nerl_score_outputs(score: Any) -> tuple[float, float, int, np.ndarray]:
    """Return aggregate and per-GT ERL values from an em_erl score object."""
    score_erl = np.asarray(score.erl)
    if score_erl.ndim > 1:
        score_erl = score_erl[0]

    pred_erl = getattr(score, "pred_erl", None)
    gt_erl = getattr(score, "gt_erl", None)
    if pred_erl is None:
        pred_erl = score_erl[0]
    if gt_erl is None:
        gt_erl = score_erl[1]
    num_skeletons = int(score_erl[2]) if score_erl.size > 2 else int(len(score.skeleton_len))

    per_gt_erl = None
    for attr_name in (
        "per_gt_erl",
        "gt_segment_erl",
        "skeleton_erl_pair",
        "skeleton_erl_pairs",
    ):
        attr_value = getattr(score, attr_name, None)
        if attr_value is not None:
            per_gt_erl = np.asarray(attr_value, dtype=np.float64)
            break

    if per_gt_erl is None:
        skeleton_pred_erl = getattr(score, "skeleton_pred_erl", None)
        if skeleton_pred_erl is None:
            skeleton_pred_erl = score.skeleton_erl
        skeleton_gt_erl = getattr(score, "skeleton_gt_erl", None)
        if skeleton_gt_erl is None:
            skeleton_gt_erl = score.skeleton_len

        skeleton_pred_erl = np.asarray(skeleton_pred_erl, dtype=np.float64)
        skeleton_gt_erl = np.asarray(skeleton_gt_erl, dtype=np.float64)
        if skeleton_pred_erl.ndim == 2 and skeleton_pred_erl.shape[1] >= 2:
            per_gt_erl = skeleton_pred_erl[:, :2]
        else:
            per_gt_erl = np.column_stack([skeleton_pred_erl, skeleton_gt_erl])

    if per_gt_erl.ndim == 1:
        per_gt_erl = per_gt_erl.reshape(0, 2) if per_gt_erl.size == 0 else per_gt_erl.reshape(1, -1)
    if per_gt_erl.ndim != 2 or per_gt_erl.shape[1] != 2:
        raise ValueError(f"NERL per-GT ERL array must have shape [N, 2], got {per_gt_erl.shape}")

    return float(pred_erl), float(gt_erl), num_skeletons, per_gt_erl


def _compute_nerl_metrics(
    module,
    decoded_predictions: np.ndarray,
    volume_prefix: str,
    metrics_dict: Dict[str, Any],
    volume_name: str | None,
) -> None:
    evaluation_cfg = _get_effective_evaluation_config(module)
    graph_value = _select_volume_config_value(
        _module_cfg_value(module, evaluation_cfg, "nerl_graph", None),
        volume_name,
    )
    if graph_value is None:
        logger.warning(
            "%sSkipping NERL: set evaluation.nerl_graph to an "
            "ERLGraph .npz or BANIS/NISB skeleton.pkl",
            volume_prefix,
        )
        return

    mask_value = _select_volume_config_value(
        _module_cfg_value(module, evaluation_cfg, "nerl_mask", None),
        volume_name,
    )
    _, compute_erl_score, compute_segment_lut = _import_em_erl()
    erl_graph, voxel_coords = _load_nerl_graph(graph_value, evaluation_cfg, module)
    node_positions = _nerl_node_positions(module, erl_graph, voxel_coords, evaluation_cfg)
    segment = _prepare_nerl_segmentation(decoded_predictions)

    merge_threshold = int(_module_cfg_value(module, evaluation_cfg, "nerl_merge_threshold", 1))
    chunk_num = int(_module_cfg_value(module, evaluation_cfg, "nerl_chunk_num", 1))
    node_segment_lut, mask_segment_id = compute_segment_lut(
        segment,
        node_positions,
        mask=mask_value,
        chunk_num=chunk_num,
        data_type=segment.dtype,
    )

    score = compute_erl_score(
        erl_graph,
        node_segment_lut,
        mask_segment_id,
        merge_threshold=merge_threshold,
    )
    score.compute_erl()

    pred_erl, gt_erl, num_skeletons, per_gt_erl = _extract_nerl_score_outputs(score)
    nerl = pred_erl / gt_erl if gt_erl > 0 else float("nan")

    logger.info(f"{volume_prefix}NERL: {nerl:.6f}")
    logger.info(f"{volume_prefix}  Pred ERL: {pred_erl:.6f}")
    logger.info(f"{volume_prefix}  GT ERL: {gt_erl:.6f}")
    logger.info(f"{volume_prefix}  # Skeletons: {num_skeletons}")

    metrics_dict["nerl"] = nerl
    metrics_dict["nerl_pred_erl"] = pred_erl
    metrics_dict["nerl_gt_erl"] = gt_erl
    metrics_dict["nerl_erl"] = pred_erl
    metrics_dict["nerl_max_erl"] = gt_erl
    metrics_dict["nerl_num_skeletons"] = num_skeletons
    metrics_dict["nerl_graph"] = str(graph_value)
    metrics_dict["nerl_gt_segment_ids"] = np.asarray(erl_graph.skeleton_id)
    metrics_dict["nerl_per_gt_erl"] = per_gt_erl

    if hasattr(module, "log"):
        try:
            module.log(
                "test_nerl",
                nerl,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as exc:  # pragma: no cover - logging should never fail evaluation
            logger.debug("Failed to log test_nerl metric: %s", exc)


def compute_test_metrics(
    module,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    volume_name: str | None = None,
) -> None:
    """Update configured metrics and save per-volume evaluation summaries."""
    if not module._is_test_evaluation_enabled():
        return

    volume_prefix = f"[{volume_name}] " if volume_name else ""
    metrics_dict: Dict[str, Any] = {"volume_name": volume_name if volume_name else "unknown"}
    requested_metrics = _configured_evaluation_metrics(module)

    if "nerl" in requested_metrics:
        _compute_nerl_metrics(
            module,
            decoded_predictions,
            volume_prefix,
            metrics_dict,
            volume_name,
        )

    if labels is None:
        module._save_metrics_to_file(metrics_dict)
        return

    pred_tensor = torch.from_numpy(decoded_predictions).float().to(module.device)
    labels_tensor = labels.float().to(pred_tensor.device)
    pred_tensor, labels_tensor = _align_metric_tensors(pred_tensor, labels_tensor)
    if pred_tensor is None or labels_tensor is None:
        module._save_metrics_to_file(metrics_dict)
        return

    evaluation_cfg = _get_effective_evaluation_config(module)
    prediction_threshold = module._cfg_float(
        evaluation_cfg,
        "prediction_threshold",
        0.5,
    )
    instance_iou_threshold = module._cfg_float(
        evaluation_cfg,
        "instance_iou_threshold",
        0.5,
    )

    if _is_instance_segmentation(pred_tensor):
        _compute_instance_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            instance_iou_threshold,
        )
    else:
        _compute_binary_metrics(
            module,
            pred_tensor,
            labels_tensor,
            volume_prefix,
            metrics_dict,
            prediction_threshold,
        )

    module._save_metrics_to_file(metrics_dict)


def log_test_epoch_metrics(module) -> None:
    """Log aggregated test metrics once after all ranks finish processing."""
    if not module._is_test_evaluation_enabled():
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


def _log_volume_header(volume_name: str, title: str) -> None:
    logger.info(f"{'=' * 70}")
    logger.info(f"{title}: {volume_name}")
    logger.info(f"{'=' * 70}")


def _process_decoding_postprocessing(
    module,
    predictions_np: np.ndarray,
    *,
    filenames: list[str],
    mode: str,
    batch_meta: Any,
    save_final_predictions: bool,
) -> np.ndarray:
    logger.info("[STAGE: Decoding Instances]")
    result = run_decoding_stage(module.cfg, predictions_np)
    decoded_predictions = result.decoded
    logger.info(f"Decoding completed ({result.duration_s:.1f}s)")

    if not result.has_decoding_config:
        logger.info("Skipping postprocessing (no decoding configuration)")
        logger.info("Skipping decoded segmentation summary (no decoding configuration)")
        logger.info("Skipping final prediction save (no decoding configuration)")
        return decoded_predictions

    postprocessed_predictions = result.postprocessed
    logger.info("Decoded Segmentation Summary:")
    logger.info(f"    Shape:      {decoded_predictions.shape}")
    logger.info(f"    Dtype:      {decoded_predictions.dtype}")
    logger.info(f"    Min:        {decoded_predictions.min()}")
    logger.info(f"    Max:        {decoded_predictions.max()}")
    logger.info(f"    Instances:  {decoded_predictions.max()} (max label)")
    max_summary_voxels = 100_000_000
    if decoded_predictions.size <= max_summary_voxels:
        logger.info(f"    Unique IDs: {len(np.unique(decoded_predictions))}")
    else:
        logger.info(
            "    Unique IDs: skipped for large volume (%d voxels > %d)",
            decoded_predictions.size,
            max_summary_voxels,
        )

    if save_final_predictions:
        logger.info("[STAGE: Saving Final Predictions]")
        save_start = time.time()
        write_outputs(
            module.cfg,
            postprocessed_predictions,
            filenames,
            suffix=final_prediction_output_tag(
                module.cfg,
                checkpoint_path=module._get_prediction_checkpoint_path(),
            ),
            mode=mode,
            batch_meta=batch_meta,
        )
        logger.info(f"Final predictions saved ({time.time() - save_start:.1f}s)")

    return decoded_predictions


def _evaluate_decoded_predictions(
    module,
    decoded_predictions: np.ndarray,
    labels: Optional[torch.Tensor],
    *,
    filenames: list[str],
    batch_idx: int,
) -> None:
    evaluation_enabled = module._is_test_evaluation_enabled()
    nerl_requested = evaluation_enabled and _evaluation_metric_requested(module, "nerl")

    def _compute_metrics(
        pred_arr: np.ndarray,
        label_tensor: Optional[torch.Tensor],
        volume_name: str | None,
    ) -> None:
        compute_test_metrics(module, pred_arr, label_tensor, volume_name=volume_name)

    if evaluation_enabled and (labels is not None or nerl_requested):
        logger.info("[STAGE: Computing Evaluation Metrics]")
        result = run_evaluation_stage(
            decoded_predictions,
            labels,
            filenames=filenames,
            batch_idx=batch_idx,
            evaluation_enabled=evaluation_enabled,
            nerl_requested=nerl_requested,
            compute_metrics_fn=_compute_metrics,
        )
        logger.info(f"Evaluation completed ({result.duration_s:.1f}s)")
        return

    if labels is None:
        logger.info("[STAGE: Evaluation] Skipped (no ground truth labels or NERL graph metric)")
    else:
        logger.info("[STAGE: Evaluation] Skipped (evaluation disabled)")


def _predict_output_head(
    module,
    *,
    lazy_sample: bool,
    images: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    image_path: Optional[str],
    mask_path: Optional[str],
    mask_align_to_image: bool,
    reference_image_shape: tuple[int, ...],
    requested_head: Optional[str] = None,
    affinity_crop_output_head: Optional[str] = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Run inference for one selected head and apply the standard prediction crops."""
    if lazy_sample:
        if image_path is None:
            raise ValueError("lazy_sample=True requires image_path to be provided.")
        predictions = lazy_predict_volume(
            module.cfg,
            module.forward,
            image_path,
            mask_path=mask_path,
            mask_align_to_image=mask_align_to_image,
            device=module.device,
            requested_head=requested_head,
        )
    else:
        if images is None:
            raise ValueError("lazy_sample=False requires images to be provided.")
        predictions = module.inference_manager.predict_with_tta(
            images,
            mask=mask,
            mask_align_to_image=mask_align_to_image,
            requested_head=requested_head,
        )

    predictions_np = predictions.detach().cpu().float().numpy()
    del predictions
    if predictions_np.size == 0:
        if _should_skip_postprocess_on_rank(module):
            logger.info(
                "Skipping prediction crop/postprocessing on this rank after distributed "
                "inference aggregation."
            )
            return predictions_np, ()
        raise RuntimeError(
            "Inference returned an empty prediction tensor unexpectedly. "
            "This is only expected on nonzero ranks during distributed single-volume "
            "inference sharding."
        )
    predictions_np, reference_spatial_shape = _apply_predecode_prediction_crops(
        module,
        predictions_np,
        reference_image_shape=reference_image_shape,
        item_name="predictions",
        output_head=affinity_crop_output_head,
    )
    return predictions_np, reference_spatial_shape


def _save_intermediate_prediction_outputs(
    module,
    predictions_np: np.ndarray,
    *,
    filenames: list[str],
    mode: str,
    batch_meta: Any,
    output_head: Optional[str] = None,
) -> None:
    """Persist one intermediate prediction tensor using the configured TTA suffix."""
    cache_suffix = tta_cache_suffix(
        module.cfg,
        checkpoint_path=module._get_prediction_checkpoint_path(),
        output_head=output_head,
    )
    write_outputs(
        module.cfg,
        predictions_np,
        filenames,
        suffix=cache_suffix.removeprefix("_").removesuffix(".h5"),
        mode=mode,
        batch_meta=batch_meta,
    )


def run_test_step(module, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
    """End-to-end test-step workflow with cache reuse and staged processing."""
    if _is_unstacked_test_batch(batch):
        images = batch.get("image") or []
        num_samples = len(images)
        if num_samples == 0:
            return torch.tensor(0.0, device=module.device)

        base_batch_idx = batch_idx * num_samples
        for sample_idx in range(num_samples):
            sample_batch = _extract_single_sample_batch(batch, sample_idx)
            run_test_step(module, sample_batch, batch_idx=base_batch_idx + sample_idx)
        return torch.tensor(0.0, device=module.device)

    # Build explicit context from module (single point of private method access)
    ctx = TestContext.from_module(module, batch)

    filenames = ctx.filenames
    output_dir_value = ctx.output_dir_value
    cache_suffix = ctx.cache_suffix
    save_prediction_cfg = ctx.save_prediction_cfg
    mode = "tune" if getattr(module, "_tune_mode", False) else "test"
    lazy_sample = _is_lazy_test_sample(batch)

    if lazy_sample:
        image_path = str(batch["image"])
        mask_path = (
            str(batch["mask"]) if isinstance(batch.get("mask"), (str, os.PathLike)) else None
        )
        labels = None
        reference_image_shape = get_lazy_image_reference_shape(module.cfg, image_path, mode=mode)
        crop_pad = _resolve_inference_crop_pad(module)
        images = None
        mask = None
    else:
        images = _coerce_singleton_batch_tensor(batch["image"])
        labels = _coerce_singleton_batch_tensor(batch.get("label"))
        mask = _coerce_singleton_batch_tensor(batch.get("mask"))
        crop_pad = _resolve_inference_crop_pad(module)
        reference_image_shape = tuple(int(v) for v in images.shape)

    predictions_np, loaded_from_file, loaded_suffix = module._load_cached_predictions(
        output_dir_value,
        filenames,
        cache_suffix,
        mode,
    )

    # Determine whether loaded predictions need decoding:
    # - saved_prediction_path → always run decoding (it's raw affinity)
    # - *_decoding*.h5 → already decoded, skip decoding
    # - *_prediction*.h5 with TTA suffix → intermediate, run decoding
    # - other → final, skip decoding
    _saved_pred = getattr(getattr(module.cfg, "inference", None), "saved_prediction_path", "")
    _from_saved_path = bool(loaded_from_file and _saved_pred)
    _is_decoding_file = loaded_from_file and "_decoding" in (loaded_suffix or "")
    loaded_final_predictions = (
        loaded_from_file
        and not _from_saved_path
        and (_is_decoding_file or not is_tta_cache_suffix(loaded_suffix))
    )
    loaded_intermediate_predictions = loaded_from_file and not loaded_final_predictions
    volume_name = filenames[0] if filenames else f"volume_{batch_idx}"
    is_global_zero = bool(getattr(getattr(module, "trainer", None), "is_global_zero", True))
    distributed_single_volume_sharding = _is_distributed_single_volume_sharding_active(module)
    configured_heads = resolve_output_heads(module.cfg, purpose="test-time inference")
    merge_heads = len(configured_heads) > 1
    selected_output_head = (
        None if merge_heads else (configured_heads[0] if configured_heads else None)
    )
    if (
        distributed_single_volume_sharding
        and not merge_heads
        and bool(getattr(save_prediction_cfg, "save_all_heads", False))
    ):
        extra_head_names = [
            head_name
            for head_name in get_model_head_names(module.cfg)
            if head_name != selected_output_head
        ]
        if extra_head_names:
            raise RuntimeError(
                "Distributed single-volume inference sharding does not support "
                "inference.save_prediction.save_all_heads=true unless all requested heads "
                "are included in inference.output_heads for merged inference."
            )

    if loaded_final_predictions:
        if distributed_single_volume_sharding and not is_global_zero:
            logger.info("Nonzero rank skipping cached final prediction postprocessing.")
            _distributed_tta_barrier(module)
            return torch.tensor(0.0, device=module.device)
        logger.info(
            "Loaded final predictions from disk, skipping inference/decoding/postprocessing"
        )
        if lazy_sample:
            labels = _maybe_load_lazy_labels(module, batch.get("label"), mode=mode)
        # Final predictions were saved after crop_pad and affinity crop were
        # already applied, so skip both spatial crops — go straight to evaluation.
        _evaluate_decoded_predictions(
            module,
            predictions_np,
            labels,
            filenames=filenames,
            batch_idx=batch_idx,
        )
        del predictions_np
        _cleanup_inference_memory(module, "cached final evaluation")
        _distributed_tta_barrier(module)
        return torch.tensor(0.0, device=module.device)

    if loaded_intermediate_predictions:
        if distributed_single_volume_sharding and not is_global_zero:
            logger.info("Nonzero rank skipping cached intermediate postprocessing.")
            _distributed_tta_barrier(module)
            return torch.tensor(0.0, device=module.device)
        logger.info("Loaded intermediate predictions from disk, skipping inference")
        if lazy_sample:
            labels = _maybe_load_lazy_labels(module, batch.get("label"), mode=mode)
        # In tune mode, the Optuna tuner handles decoding — skip it here.
        if mode == "tune":
            logger.info("Tune mode: skipping decoding (Optuna tuner will handle it)")
            del predictions_np
            _cleanup_inference_memory(module, "cached tune prediction")
            _distributed_tta_barrier(module)
            return torch.tensor(0.0, device=module.device)
        _log_volume_header(volume_name, "PROCESSING VOLUME")
        if _from_saved_path:
            logger.info("Applying pre-decode crop_pad/affinity_crop to saved_prediction_path data")
            predictions_np, reference_spatial_shape = _apply_predecode_prediction_crops(
                module,
                predictions_np,
                reference_image_shape=reference_image_shape,
                item_name="saved predictions",
                output_head=selected_output_head,
            )
            predictions_np = apply_prediction_transform(module.cfg, predictions_np)
        else:
            logger.info(
                "Skipping crop_pad/affinity_crop for cached intermediate predictions "
                "(already saved after those crops)"
            )
            reference_spatial_shape = tuple(int(v) for v in predictions_np.shape[-3:])
        decoded_predictions = _process_decoding_postprocessing(
            module,
            predictions_np,
            filenames=filenames,
            mode=mode,
            batch_meta=batch.get("image_meta_dict"),
            save_final_predictions=True,
        )
        del predictions_np
        _cleanup_inference_memory(module, "cached intermediate decoding")
        _evaluate_decoded_predictions(
            module,
            decoded_predictions,
            (
                _apply_affinity_inference_crop_if_needed(
                    module,
                    labels,
                    reference_spatial_shape=reference_spatial_shape,
                    item_name="labels",
                    output_head=selected_output_head,
                )
                if labels is not None and _from_saved_path
                else labels
            ),
            filenames=filenames,
            batch_idx=batch_idx,
        )
        _log_volume_header(volume_name, "VOLUME COMPLETE")
        del decoded_predictions
        _cleanup_inference_memory(module, "cached intermediate evaluation")
        _distributed_tta_barrier(module)
        return torch.tensor(0.0, device=module.device)

    logger.info("No cached predictions found, running inference")

    # If the model is a lightweight dummy (e.g. nn.Identity), inference would
    # produce garbage.  Error out early instead of crashing later in TTA.
    if getattr(module, "_skip_inference", False):
        raise RuntimeError(
            "Cached predictions expected but not found for this volume. "
            "Cannot run inference with a lightweight (dummy) model. "
            "Re-run with the real model checkpoint to generate predictions first."
        )

    mask_align_to_image = _resolve_mask_align_to_image(module)
    if is_chunked_inference_enabled(module.cfg):
        if not lazy_sample:
            raise RuntimeError(
                "inference.strategy=chunked requires lazy test data "
                "(set data.dataloader.profile=lazy / sliding_window.lazy_load=true)."
            )
        if mode == "tune":
            raise RuntimeError(
                "Chunked inference+decoding is not integrated with tune mode yet; "
                "run mode=test with fixed decoding parameters."
            )
        if merge_heads:
            raise RuntimeError("Chunked inference does not support merged multi-head outputs yet.")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                raise RuntimeError(
                    "Chunked inference currently runs as a single-rank streaming job. "
                    "Run test with one GPU/process for this mode."
                )
        if not output_dir_value:
            raise RuntimeError(
                "Chunked inference writes a streamed final volume and requires "
                "inference.save_prediction.output_path."
            )

        _log_volume_header(volume_name, "CHUNKED INFERENCE PLAN")
        logger.info(f"Input source:      {image_path}")
        logger.info(f"Input shape:       {reference_image_shape}")
        logger.info("Input device:      [lazy disk-backed volume]")
        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunking_cfg = module.cfg.inference.chunking
        chunk_output_mode = str(getattr(chunking_cfg, "output_mode", "decoded")).lower()
        inference_start = time.time()

        if chunk_output_mode == "raw_prediction":
            raw_suffix = (
                tta_cache_suffix(
                    module.cfg,
                    checkpoint_path=module._get_prediction_checkpoint_path(),
                    output_head=selected_output_head,
                )
                .removeprefix("_")
                .removesuffix(".h5")
            )
            output_path = output_dir / f"{filenames[0]}_{raw_suffix}.h5"
            run_chunked_prediction_inference(
                module.cfg,
                module.forward,
                image_path,
                output_path=output_path,
                device=module.device,
                mask_path=mask_path,
                mask_align_to_image=mask_align_to_image,
                requested_head=selected_output_head,
            )
            inference_duration = time.time() - inference_start
            logger.info(
                "Chunked raw prediction inference completed in %.2f minutes (%.1fs)",
                inference_duration / 60.0,
                inference_duration,
            )
            _cleanup_inference_memory(module, "chunked raw inference", release_model=True)

            decode_after = bool(getattr(module.cfg.inference, "decode_after_inference", True))
            if not decode_after:
                if module._is_test_evaluation_enabled():
                    logger.warning(
                        "Skipping evaluation because decode_after_inference=false produced "
                        "only raw predictions."
                    )
                _log_volume_header(volume_name, "VOLUME COMPLETE")
                _cleanup_inference_memory(module, "chunked raw inference", release_model=True)
                _distributed_tta_barrier(module)
                return torch.tensor(0.0, device=module.device)

            from ...data.io import read_volume

            logger.info("[STAGE: Loading Chunked Raw Prediction For Whole-Volume Decode]")
            predictions_np = read_volume(str(output_path), dataset="main")
            reference_spatial_shape = tuple(int(v) for v in predictions_np.shape[-3:])
            if lazy_sample:
                labels = _maybe_load_lazy_labels(module, batch.get("label"), mode=mode)
            decoded_predictions = _process_decoding_postprocessing(
                module,
                predictions_np,
                filenames=filenames,
                mode=mode,
                batch_meta=batch.get("image_meta_dict"),
                save_final_predictions=True,
            )
            del predictions_np
            _cleanup_inference_memory(module, "chunked raw decoding")
            _evaluate_decoded_predictions(
                module,
                decoded_predictions,
                (
                    _apply_affinity_inference_crop_if_needed(
                        module,
                        labels,
                        reference_spatial_shape=reference_spatial_shape,
                        item_name="labels",
                        output_head=selected_output_head,
                    )
                    if labels is not None
                    else None
                ),
                filenames=filenames,
                batch_idx=batch_idx,
            )
            del decoded_predictions
            _cleanup_inference_memory(module, "chunked raw evaluation")
        else:
            output_suffix = final_prediction_output_tag(
                module.cfg,
                checkpoint_path=module._get_prediction_checkpoint_path(),
                output_head=selected_output_head,
            )
            output_path = output_dir / f"{filenames[0]}_{output_suffix}.h5"
            run_chunked_affinity_cc_inference(
                module.cfg,
                module.forward,
                image_path,
                output_path=output_path,
                device=module.device,
                mask_path=mask_path,
                mask_align_to_image=mask_align_to_image,
                requested_head=selected_output_head,
            )
            inference_duration = time.time() - inference_start
            logger.info(
                "Chunked inference+decoding completed in %.2f minutes (%.1fs)",
                inference_duration / 60.0,
                inference_duration,
            )
            _cleanup_inference_memory(module, "chunked inference decoding", release_model=True)
            if module._is_test_evaluation_enabled():
                logger.warning(
                    "Skipping evaluation for chunked inference; streaming metrics are not "
                    "implemented and loading the full label defeats this mode's memory goal."
                )
        _log_volume_header(volume_name, "VOLUME COMPLETE")
        _distributed_tta_barrier(module)
        return torch.tensor(0.0, device=module.device)

    _log_volume_header(volume_name, "INFERENCE PLAN")
    if lazy_sample:
        logger.info(f"Input source:      {image_path}")
        logger.info(f"Input shape:       {reference_image_shape}")
        logger.info("Input device:      [lazy disk-backed volume]")
    else:
        logger.info(f"Input shape:       {tuple(images.shape)}")
        logger.info(f"Input device:      {images.device}")
    if crop_pad is not None:
        logger.info(f"Inference crop:    {list(crop_pad)}")

    inference_cfg = module._get_runtime_inference_config()
    sw_cfg = getattr(inference_cfg, "sliding_window", None)
    if sw_cfg is not None:
        roi_size = getattr(sw_cfg, "window_size", "N/A")
        overlap = getattr(sw_cfg, "overlap", "N/A")
        sw_batch = getattr(sw_cfg, "sw_batch_size", "N/A")
        blending = getattr(sw_cfg, "blending", "gaussian")
        logger.info(f"Sliding window ROI: {roi_size}")
        logger.info(f"Overlap:            {overlap}")
        logger.info(f"SW batch size:      {sw_batch}")
        logger.info(f"Blending mode:      {blending}")
    else:
        logger.info("Sliding window:     [Direct inference, no sliding window]")
    image_ndim = len(reference_image_shape) if lazy_sample else images.ndim
    logger.info(f"TTA:                {module._summarize_tta_plan(image_ndim)}")
    logger.info(f"{'=' * 70}")

    inference_start = time.time()
    logger.info("Starting sliding-window inference...")

    if merge_heads:
        selected_output_head = "+".join(configured_heads)
        per_head_preds: list[np.ndarray] = []
        reference_spatial_shape: tuple[int, ...] = ()
        skip_local_distributed_shard = False
        for head_name in configured_heads:
            head_pred_np, reference_spatial_shape = _predict_output_head(
                module,
                lazy_sample=lazy_sample,
                images=images,
                mask=mask,
                image_path=image_path if lazy_sample else None,
                mask_path=mask_path if lazy_sample else None,
                mask_align_to_image=mask_align_to_image,
                reference_image_shape=reference_image_shape,
                requested_head=head_name,
                affinity_crop_output_head=None,
            )
            if head_pred_np.size == 0:
                skip_local_distributed_shard = True
                continue
            per_head_preds.append(head_pred_np)
        if skip_local_distributed_shard and _should_skip_postprocess_on_rank(module):
            logger.info(
                "Completed local distributed inference shard; rank 0 will perform "
                "postprocessing."
            )
            del images, mask
            _cleanup_inference_memory(module, "distributed inference shard", release_model=True)
            _distributed_tta_barrier(module)
            return torch.tensor(0.0, device=module.device)
        if not per_head_preds:
            raise RuntimeError("Merged-head inference produced no predictions on rank 0.")
        # Predictions are (B, C, ...) — concat along channel axis preserving head order.
        predictions_np = np.concatenate(per_head_preds, axis=1)
        del per_head_preds
        _cleanup_inference_memory(module, "merged head concatenation")
        logger.info(
            f"Merged heads {configured_heads} along channel axis → shape {predictions_np.shape}"
        )
    else:
        selected_output_head = configured_heads[0] if configured_heads else None
        predictions_np, reference_spatial_shape = _predict_output_head(
            module,
            lazy_sample=lazy_sample,
            images=images,
            mask=mask,
            image_path=image_path if lazy_sample else None,
            mask_path=mask_path if lazy_sample else None,
            mask_align_to_image=mask_align_to_image,
            reference_image_shape=reference_image_shape,
            requested_head=selected_output_head,
            affinity_crop_output_head=selected_output_head,
        )
    if distributed_single_volume_sharding and _should_skip_postprocess_on_rank(module):
        logger.info(
            "Completed local distributed inference shard; rank 0 will perform postprocessing."
        )
        del predictions_np, images, mask
        _cleanup_inference_memory(module, "distributed inference shard", release_model=True)
        _distributed_tta_barrier(module)
        return torch.tensor(0.0, device=module.device)
    if lazy_sample:
        labels = _maybe_load_lazy_labels(module, batch.get("label"), mode=mode)

    inference_duration = time.time() - inference_start
    logger.info(
        f"Inference completed in {inference_duration / 60:.2f} minutes ({inference_duration:.1f}s)"
    )
    predictions_np = apply_prediction_transform(module.cfg, predictions_np)

    logger.info("Prediction Summary:")
    logger.info(f"    Shape:  {predictions_np.shape}")
    logger.info(f"    Dtype:  {predictions_np.dtype}")
    logger.info(f"    Min:    {predictions_np.min():.6f}")
    logger.info(f"    Max:    {predictions_np.max():.6f}")
    logger.info(f"    Mean:   {predictions_np.mean():.6f}")

    save_intermediate = bool(getattr(inference_cfg.save_prediction, "enabled", False))
    if save_intermediate:
        logger.info("[STAGE: Saving Intermediate Predictions]")
        save_start = time.time()
        _save_intermediate_prediction_outputs(
            module,
            predictions_np,
            filenames=filenames,
            mode=mode,
            batch_meta=batch.get("image_meta_dict"),
            output_head=selected_output_head,
        )

        save_all_heads = bool(getattr(inference_cfg.save_prediction, "save_all_heads", False))
        model_head_names = get_model_head_names(module.cfg)
        # When heads are already merged into predictions_np, the single saved file
        # covers every head — skip the redundant per-head re-prediction loop.
        extra_head_names = (
            []
            if merge_heads
            else [head_name for head_name in model_head_names if head_name != selected_output_head]
        )
        if save_all_heads and extra_head_names:
            logger.info(f"Saving additional output heads: {', '.join(extra_head_names)}")
            for head_name in extra_head_names:
                extra_predictions_np, _ = _predict_output_head(
                    module,
                    lazy_sample=lazy_sample,
                    images=images,
                    mask=mask,
                    image_path=image_path if lazy_sample else None,
                    mask_path=mask_path if lazy_sample else None,
                    mask_align_to_image=mask_align_to_image,
                    reference_image_shape=reference_image_shape,
                    requested_head=head_name,
                )
                _save_intermediate_prediction_outputs(
                    module,
                    apply_prediction_transform(module.cfg, extra_predictions_np),
                    filenames=filenames,
                    mode=mode,
                    batch_meta=batch.get("image_meta_dict"),
                    output_head=head_name,
                )
                del extra_predictions_np
                _cleanup_inference_memory(module, f"extra head {head_name} save")
        logger.info(f"Intermediate predictions saved ({time.time() - save_start:.1f}s)")

    del images, mask
    _cleanup_inference_memory(module, "model inference", release_model=True)

    # In tune mode, skip decoding — the Optuna tuner will handle it.
    if mode == "tune":
        logger.info("Tune mode: skipping decoding (Optuna tuner will handle it)")
        del predictions_np
        _cleanup_inference_memory(module, "tune prediction")
        _distributed_tta_barrier(module)
        return torch.tensor(0.0, device=module.device)

    decoded_predictions = _process_decoding_postprocessing(
        module,
        predictions_np,
        filenames=filenames,
        mode=mode,
        batch_meta=batch.get("image_meta_dict"),
        save_final_predictions=True,
    )
    del predictions_np
    _cleanup_inference_memory(module, "decoding")
    _evaluate_decoded_predictions(
        module,
        decoded_predictions,
        (
            _apply_affinity_inference_crop_if_needed(
                module,
                labels,
                reference_spatial_shape=reference_spatial_shape,
                item_name="labels",
                output_head=(None if merge_heads else selected_output_head),
            )
            if labels is not None
            else None
        ),
        filenames=filenames,
        batch_idx=batch_idx,
    )
    _log_volume_header(volume_name, "VOLUME COMPLETE")
    del decoded_predictions
    _cleanup_inference_memory(module, "evaluation")
    _distributed_tta_barrier(module)
    return torch.tensor(0.0, device=module.device)
