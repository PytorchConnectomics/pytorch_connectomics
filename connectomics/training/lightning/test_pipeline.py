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
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ...data.processing.affinity import (
    compute_affinity_crop_pad,
    crop_spatial_by_pad,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
)
from ...data.processing.misc import get_padsize
from ...decoding import run_decoding_stage
from ...decoding.streamed_chunked import run_chunked_affinity_cc_inference
from ...evaluation import evaluation_metric_requested, run_evaluation_stage
from ...inference import (
    apply_prediction_transform,
    is_chunked_inference_enabled,
    run_chunked_prediction_inference,
    write_outputs,
)
from ...inference.lazy import (
    get_lazy_image_reference_shape,
    lazy_predict_volume,
    load_lazy_volume,
)
from ...runtime.output_naming import (
    final_prediction_output_tag,
    is_tta_cache_suffix,
    tta_cache_suffix,
)
from ...utils.channel_slices import resolve_channel_indices, resolve_channel_range
from ...utils.model_outputs import (
    get_model_head_names,
    resolve_output_heads,
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
    nerl_requested = evaluation_enabled and evaluation_metric_requested(module, "nerl")

    if evaluation_enabled and (labels is not None or nerl_requested):
        logger.info("[STAGE: Computing Evaluation Metrics]")
        result = run_evaluation_stage(
            module,
            decoded_predictions,
            labels,
            filenames=filenames,
            batch_idx=batch_idx,
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
                checkpoint_path=module._get_prediction_checkpoint_path(),
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
                checkpoint_path=module._get_prediction_checkpoint_path(),
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
