"""Output writing utilities for inference and tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from omegaconf import DictConfig

from ..config import Config

logger = logging.getLogger(__name__)


def resolve_output_filenames(
    cfg: Config | DictConfig, batch: Dict[str, Any], global_step: int = 0
) -> List[str]:
    """Extract and resolve filenames from batch metadata."""
    images = batch.get("image")
    if images is not None:
        if isinstance(images, (list, tuple)):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]
    else:
        batch_size = 1

    meta = batch.get("image_meta_dict")
    filenames: List[str] = []

    if isinstance(meta, list):
        for meta_item in meta:
            if isinstance(meta_item, dict):
                filename = meta_item.get("filename_or_obj")
                if filename is not None:
                    filenames.append(filename)
        batch_size = max(batch_size, len(filenames))
    elif isinstance(meta, dict):
        meta_filenames = meta.get("filename_or_obj")
        if isinstance(meta_filenames, (list, tuple)):
            filenames = [f for f in meta_filenames if f is not None]
        elif meta_filenames is not None:
            filenames = [meta_filenames]
        if len(filenames) > 0:
            batch_size = max(batch_size, len(filenames))

    resolved_names: List[str] = []
    for idx in range(batch_size):
        if idx < len(filenames) and filenames[idx]:
            resolved_names.append(Path(str(filenames[idx])).stem)
        else:
            resolved_names.append(f"volume_{global_step}_{idx}")

    if len(resolved_names) < batch_size:
        logger.warning(
            f"resolve_output_filenames - Only {len(resolved_names)} "
            f"filenames but batch_size is {batch_size}, padding with fallback names"
        )
        while len(resolved_names) < batch_size:
            resolved_names.append(f"volume_{global_step}_{len(resolved_names)}")

    return resolved_names


def _extract_meta_for_index(batch_meta: Any, idx: int) -> Dict[str, Any]:
    if batch_meta is None:
        return {}
    if isinstance(batch_meta, list):
        if idx < len(batch_meta) and isinstance(batch_meta[idx], dict):
            return dict(batch_meta[idx])
        return {}
    if isinstance(batch_meta, dict):
        out: Dict[str, Any] = {}
        for key, value in batch_meta.items():
            if isinstance(value, (list, tuple)) and idx < len(value):
                out[key] = value[idx]
            else:
                out[key] = value
        return out
    return {}


def _infer_spatial_dims_from_array(array: np.ndarray) -> int:
    if array.ndim <= 2:
        return array.ndim
    if array.ndim == 3:
        return 3
    return array.ndim - 1


def _spatial_shape(array: np.ndarray, spatial_dims: int) -> tuple:
    if array.ndim == spatial_dims:
        return tuple(int(v) for v in array.shape)
    return tuple(int(v) for v in array.shape[-spatial_dims:])


def _resample_array_to_shape(
    array: np.ndarray,
    target_shape: Sequence[int],
    spatial_dims: int,
    order: int,
) -> np.ndarray:
    from scipy.ndimage import zoom

    target = tuple(int(v) for v in target_shape)
    if _spatial_shape(array, spatial_dims) == target:
        return array

    def _zoom_single(vol: np.ndarray) -> np.ndarray:
        factors = np.array(target, dtype=np.float32) / np.maximum(
            np.array(vol.shape, dtype=np.float32), 1.0
        )
        return zoom(
            vol.astype(np.float32, copy=False),
            zoom=factors,
            order=order,
            mode="nearest",
            prefilter=order > 1,
        )

    if array.ndim == spatial_dims + 1:
        channels = [_zoom_single(array[c])[None] for c in range(array.shape[0])]
        return np.vstack(channels).astype(array.dtype, copy=False)
    if array.ndim == spatial_dims:
        return _zoom_single(array).astype(array.dtype, copy=False)
    return array


def _fit_array_to_shape(
    array: np.ndarray, target_shape: Sequence[int], spatial_dims: int
) -> np.ndarray:
    target = tuple(int(v) for v in target_shape)
    if _spatial_shape(array, spatial_dims) == target:
        return array

    if array.ndim == spatial_dims + 1:
        out = np.zeros((array.shape[0], *target), dtype=array.dtype)
        in_shape = array.shape[1:]
        write_shape = tuple(min(int(in_shape[d]), target[d]) for d in range(spatial_dims))
        out_slices = (slice(None),) + tuple(slice(0, w) for w in write_shape)
        in_slices = (slice(None),) + tuple(slice(0, w) for w in write_shape)
        out[out_slices] = array[in_slices]
        return out

    if array.ndim == spatial_dims:
        out = np.zeros(target, dtype=array.dtype)
        in_shape = array.shape
        write_shape = tuple(min(int(in_shape[d]), target[d]) for d in range(spatial_dims))
        out_slices = tuple(slice(0, w) for w in write_shape)
        in_slices = tuple(slice(0, w) for w in write_shape)
        out[out_slices] = array[in_slices]
        return out

    return array


def _restore_prediction_to_input_space(
    sample: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    preprocess_meta = meta.get("nnunet_preprocess")
    if not isinstance(preprocess_meta, dict) or not preprocess_meta.get("enabled", False):
        return sample

    array = sample
    spatial_dims = int(
        preprocess_meta.get("spatial_dims", _infer_spatial_dims_from_array(array))
    )
    is_integer = np.issubdtype(array.dtype, np.integer)
    interp_order = 0 if is_integer else 1

    if preprocess_meta.get("applied_resample", False):
        cropped_shape = preprocess_meta.get("cropped_spatial_shape")
        if isinstance(cropped_shape, (list, tuple)) and len(cropped_shape) == spatial_dims:
            array = _resample_array_to_shape(
                array,
                target_shape=cropped_shape,
                spatial_dims=spatial_dims,
                order=interp_order,
            )

    if preprocess_meta.get("applied_crop", False):
        bbox = preprocess_meta.get("crop_bbox")
        original_shape = preprocess_meta.get("original_spatial_shape")
        if (
            isinstance(bbox, (list, tuple))
            and isinstance(original_shape, (list, tuple))
            and len(bbox) == spatial_dims
            and len(original_shape) == spatial_dims
        ):
            crop_target_shape = tuple(int(b[1]) - int(b[0]) for b in bbox)
            array = _fit_array_to_shape(array, crop_target_shape, spatial_dims=spatial_dims)

            if array.ndim == spatial_dims + 1:
                restored = np.zeros((array.shape[0], *original_shape), dtype=array.dtype)
                slices = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
                restored[(slice(None), *slices)] = array
            else:
                restored = np.zeros(
                    tuple(int(v) for v in original_shape), dtype=array.dtype
                )
                slices = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
                restored[slices] = array
            array = restored

    transpose_axes = preprocess_meta.get("transpose_axes")
    if isinstance(transpose_axes, (list, tuple)) and len(transpose_axes) == spatial_dims:
        inverse_axes = np.argsort(np.asarray(transpose_axes))
        if array.ndim == spatial_dims + 1:
            perm = [0] + [int(i) + 1 for i in inverse_axes]
            array = np.transpose(array, perm)
        elif array.ndim == spatial_dims:
            array = np.transpose(array, tuple(int(i) for i in inverse_axes))

    return array


def _should_restore_outputs(cfg: Config | DictConfig, mode: str) -> bool:
    _inference_cfg, data_cfg, _output_dir = _resolve_mode_configs(cfg, mode)
    if data_cfg is not None:
        pre = getattr(data_cfg, "nnunet_preprocessing", None)
        if pre is not None:
            return bool(
                getattr(pre, "enabled", False)
                and getattr(pre, "restore_to_input_space", False)
            )

    return False


def _resolve_mode_configs(
    cfg: Config | DictConfig,
    mode: str,
) -> tuple[Any, Any, str | None]:
    """Resolve merged runtime inference/data/output configuration."""
    inference_cfg = getattr(cfg, "inference", None)
    output_dir_value = None
    if inference_cfg is not None:
        save_pred_cfg = getattr(inference_cfg, "save_prediction", None)
        if save_pred_cfg is not None:
            output_dir_value = getattr(save_pred_cfg, "output_path", None)

    # Resolve data config: prefer stage-specific data (cfg.test.data / cfg.tune.data),
    # fall back to cfg.data (post-stage-resolution)
    data_cfg = None
    if mode == "test" and hasattr(cfg, "test") and hasattr(cfg.test, "data"):
        data_cfg = cfg.test.data
    elif mode == "tune" and hasattr(cfg, "tune") and hasattr(cfg.tune, "data"):
        data_cfg = cfg.tune.data
    if data_cfg is None:
        data_cfg = getattr(cfg, "data", None)

    # Also check stage-specific output_path if not found in inference config
    if not output_dir_value:
        stage_cfg = getattr(cfg, mode, None)
        if stage_cfg is not None:
            output_dir_value = getattr(stage_cfg, "output_path", None)

    return inference_cfg, data_cfg, output_dir_value


def apply_save_prediction_transform(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """Apply intensity scaling and dtype conversion from save_prediction config."""
    intensity_scale = -1.0
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "save_prediction"):
        save_pred_cfg = cfg.inference.save_prediction
        intensity_scale = getattr(save_pred_cfg, "intensity_scale", -1.0)

    if intensity_scale >= 0:
        data = data.astype(np.float32)
        data_min = data.min()
        data_max = data.max()

        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
            logger.info(f"Normalized predictions to [0, 1] (min={data_min:.4f}, max={data_max:.4f})")
        else:
            logger.warning(f"data_min == data_max ({data_min:.4f}), skipping normalization")

        if intensity_scale != 1.0:
            data = data * float(intensity_scale)
            logger.info(
                f"Scaled predictions by {intensity_scale} -> "
                f"range [{data.min():.4f}, {data.max():.4f}]"
            )
    else:
        logger.info(
            f"Intensity scaling disabled (scale={intensity_scale} < 0), keeping raw predictions"
        )
        return data

    target_dtype_str = None
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "save_prediction"):
        save_pred_cfg = cfg.inference.save_prediction
        target_dtype_str = getattr(save_pred_cfg, "intensity_dtype", None)

    if target_dtype_str is not None:
        dtype_map = {
            "uint8": np.uint8,
            "int8": np.int8,
            "uint16": np.uint16,
            "int16": np.int16,
            "uint32": np.uint32,
            "int32": np.int32,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
        }

        if target_dtype_str not in dtype_map:
            logger.warning(
                f"Unknown dtype '{target_dtype_str}' in save_prediction config. "
                f"Supported: {list(dtype_map.keys())}. Keeping current dtype."
            )
            return data

        target_dtype = dtype_map[target_dtype_str]
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            data = np.clip(data, info.min, info.max)
            logger.info(f"Converting to {target_dtype_str} (clipped to [{info.min}, {info.max}])")

        data = data.astype(target_dtype)

    return data


def apply_postprocessing(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """Apply inference postprocessing transforms."""
    if not hasattr(cfg, "inference") or not hasattr(cfg.inference, "postprocessing"):
        return data

    postprocessing = cfg.inference.postprocessing
    if not getattr(postprocessing, "enabled", False):
        return data

    binary_config = getattr(postprocessing, "binary", None)
    if binary_config is not None and getattr(binary_config, "enabled", False):
        from connectomics.decoding.postprocess.postprocess import apply_binary_postprocessing

        if data.ndim in (4, 5):
            batch_size = data.shape[0]
        elif data.ndim == 3:
            batch_size = 1
            data = data[np.newaxis, ...]
        elif data.ndim == 2:
            batch_size = 1
            data = data[np.newaxis, np.newaxis, ...]
        else:
            batch_size = 1

        results = []
        for batch_idx in range(batch_size):
            sample = data[batch_idx]

            if sample.ndim == 4:
                foreground_prob = sample[0]
            elif sample.ndim == 3:
                foreground_prob = sample
            elif sample.ndim == 2:
                foreground_prob = sample[np.newaxis, ...]
            else:
                foreground_prob = sample

            processed = apply_binary_postprocessing(foreground_prob, binary_config)

            if sample.ndim == 4:
                processed = processed[np.newaxis, ...]
            elif sample.ndim == 2:
                processed = processed[np.newaxis, np.newaxis, ...]

            results.append(processed)

        data = np.stack(results, axis=0)

    output_transpose = getattr(postprocessing, "output_transpose", [])
    if output_transpose and len(output_transpose) > 0:
        try:
            data = np.transpose(data, axes=output_transpose)
        except Exception as exc:
            logger.warning(
                f"Transpose failed with axes {output_transpose}: {exc}. Keeping original shape."
            )

    return data


def write_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str = "prediction",
    mode: str = "test",
    batch_meta: Any = None,
) -> None:
    """Persist predictions to disk.

    Note: output_transpose is NOT applied here. It is applied once in
    apply_postprocessing() to avoid double-transpose when both functions
    are called on the same data.
    """
    inference_cfg, _data_cfg, output_dir_value = _resolve_mode_configs(cfg, mode)
    if inference_cfg is None:
        return

    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import save_volume, write_hdf5

    if predictions.ndim >= 4:
        actual_batch_size = predictions.shape[0]
    elif predictions.ndim == 3:
        if len(filenames) > 0 and predictions.shape[0] == len(filenames):
            actual_batch_size = predictions.shape[0]
        else:
            actual_batch_size = 1
            predictions = predictions[np.newaxis, ...]
    else:
        actual_batch_size = 1
        predictions = predictions[np.newaxis, ...]

    if len(filenames) != actual_batch_size:
        logger.warning(
            f"write_outputs - filename count ({len(filenames)}) "
            f"does not match batch size ({actual_batch_size}). Using first "
            f"{min(len(filenames), actual_batch_size)} filenames."
        )

    should_restore = _should_restore_outputs(cfg, mode) and suffix == "prediction"

    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            logger.warning(
                f"write_outputs - no filename for batch index {idx}, skipping"
            )
            continue

        sample = predictions[idx]
        if should_restore:
            sample_meta = _extract_meta_for_index(batch_meta, idx)
            sample = _restore_prediction_to_input_space(sample, sample_meta)

        filename = filenames[idx]
        sample = np.squeeze(sample)

        output_formats = ["h5"]
        if hasattr(inference_cfg, "save_prediction"):
            save_pred_cfg = inference_cfg.save_prediction
            if hasattr(save_pred_cfg, "output_formats") and save_pred_cfg.output_formats:
                output_formats = save_pred_cfg.output_formats

        for fmt in output_formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "h5":
                out_path = output_dir / f"{filename}_{suffix}.h5"
                write_hdf5(
                    out_path,
                    sample.astype(np.float32)
                    if not np.issubdtype(sample.dtype, np.integer)
                    else sample,
                    dataset="main",
                )
                logger.info(f"Saved HDF5: {out_path.name}")

            elif fmt_lower in ["tif", "tiff"]:
                out_path = output_dir / f"{filename}_{suffix}.tiff"
                try:
                    save_volume(str(out_path), sample, file_format="tiff")
                    logger.info(f"Saved TIFF: {out_path.name} (shape: {sample.shape})")
                except Exception as exc:
                    logger.warning(f"TIFF export failed: {exc}")

            elif fmt_lower in ["nii", "nii.gz"]:
                out_path = output_dir / f"{filename}_{suffix}.nii.gz"
                try:
                    save_volume(str(out_path), sample, file_format="nii.gz")
                    logger.info(f"Saved NIfTI: {out_path.name}")
                except Exception as exc:
                    logger.warning(f"NIfTI export failed: {exc}")

            elif fmt_lower == "png":
                out_dir = output_dir / f"{filename}_{suffix}_png"
                try:
                    save_volume(str(out_dir), sample, file_format="png")
                    logger.info(f"Saved PNG stack: {out_dir.name}/")
                except Exception as exc:
                    logger.warning(f"PNG export failed: {exc}")

            else:
                logger.warning(
                    f"Unknown format '{fmt}' - skipping. Supported: h5, tiff, nii.gz, png"
                )


__all__ = [
    "apply_save_prediction_transform",
    "apply_postprocessing",
    "resolve_output_filenames",
    "write_outputs",
]
