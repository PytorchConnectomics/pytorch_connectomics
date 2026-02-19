"""Output writing utilities for inference and tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from omegaconf import DictConfig

from ..config import Config
from .postprocessing import analyze_h5_array


def resolve_output_filenames(
    cfg: Config | DictConfig, batch: Dict[str, Any], global_step: int = 0
) -> List[str]:
    """Extract and resolve filenames from batch metadata."""
    images = batch.get("image")
    if images is not None:
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
        print(
            f"  WARNING: resolve_output_filenames - Only {len(resolved_names)} "
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


def _fit_array_to_shape(array: np.ndarray, target_shape: Sequence[int], spatial_dims: int) -> np.ndarray:
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


def _restore_prediction_to_input_space(sample: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    preprocess_meta = meta.get("nnunet_preprocess")
    if not isinstance(preprocess_meta, dict) or not preprocess_meta.get("enabled", False):
        return sample

    array = sample
    spatial_dims = int(preprocess_meta.get("spatial_dims", _infer_spatial_dims_from_array(array)))
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
                restored = np.zeros(tuple(int(v) for v in original_shape), dtype=array.dtype)
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
    if mode == "tune":
        if hasattr(cfg, "tune") and cfg.tune and hasattr(cfg.tune, "data"):
            pre = getattr(cfg.tune.data, "nnunet_preprocessing", None)
            return bool(
                getattr(pre, "enabled", False) and getattr(pre, "restore_to_input_space", False)
            )
        return False

    if hasattr(cfg, "test") and hasattr(cfg.test, "data"):
        pre = getattr(cfg.test.data, "nnunet_preprocessing", None)
        if pre is not None:
            return bool(
                getattr(pre, "enabled", False) and getattr(pre, "restore_to_input_space", False)
            )

    if hasattr(cfg, "data"):
        pre = getattr(cfg.data, "nnunet_preprocessing", None)
        if pre is not None:
            return bool(
                getattr(pre, "enabled", False) and getattr(pre, "restore_to_input_space", False)
            )

    return False


def write_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str = "prediction",
    mode: str = "test",
    batch_meta: Any = None,
) -> None:
    """Persist predictions to disk."""
    if not hasattr(cfg, "inference"):
        return

    output_dir_value = None
    if mode == "tune":
        if hasattr(cfg, "tune") and cfg.tune and hasattr(cfg.tune, "output"):
            output_dir_value = cfg.tune.output.output_pred
    else:
        if hasattr(cfg, "test") and hasattr(cfg.test, "data") and hasattr(cfg.test.data, "output_path"):
            output_dir_value = cfg.test.data.output_path

    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import save_volume, write_hdf5

    output_transpose = []
    if hasattr(cfg.inference, "postprocessing"):
        output_transpose = getattr(cfg.inference.postprocessing, "output_transpose", [])

    save_channels = None
    if hasattr(cfg.inference, "sliding_window"):
        save_channels = getattr(cfg.inference.sliding_window, "save_channels", None)

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
        print(
            f"  WARNING: write_outputs - filename count ({len(filenames)}) "
            f"does not match batch size ({actual_batch_size}). Using first "
            f"{min(len(filenames), actual_batch_size)} filenames."
        )

    should_restore = _should_restore_outputs(cfg, mode) and suffix == "prediction"

    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            print(f"  WARNING: write_outputs - no filename for batch index {idx}, skipping")
            continue

        sample = predictions[idx]
        if should_restore:
            sample_meta = _extract_meta_for_index(batch_meta, idx)
            sample = _restore_prediction_to_input_space(sample, sample_meta)

        filename = filenames[idx]

        if save_channels is not None and sample.ndim >= 4:
            channel_indices = list(save_channels)
            num_channels = sample.shape[0]
            if num_channels > len(channel_indices):
                try:
                    sample = sample[channel_indices]
                    print(
                        f"  Selected channels {channel_indices} from "
                        f"{predictions[idx].shape[0]} channels"
                    )
                except Exception as exc:
                    print(
                        f"  WARNING: write_outputs - channel selection failed: "
                        f"{exc}, keeping all channels"
                    )

        if output_transpose and len(output_transpose) > 0:
            try:
                sample = np.transpose(sample, axes=output_transpose)
            except Exception as exc:
                print(f"  WARNING: write_outputs - transpose failed: {exc}, keeping original shape")

        sample = np.squeeze(sample)

        output_formats = ["h5"]
        analyze_h5 = False
        if hasattr(cfg, "inference") and hasattr(cfg.inference, "save_prediction"):
            save_pred_cfg = cfg.inference.save_prediction
            if hasattr(save_pred_cfg, "output_formats") and save_pred_cfg.output_formats:
                output_formats = save_pred_cfg.output_formats
            analyze_h5 = getattr(save_pred_cfg, "analyze_h5", False)

        for fmt in output_formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "h5":
                out_path = output_dir / f"{filename}_{suffix}.h5"
                write_hdf5(
                    out_path,
                    sample.astype(np.float32) if not np.issubdtype(sample.dtype, np.integer) else sample,
                    dataset="main",
                )
                print(f"  Saved HDF5: {out_path.name}")

                if analyze_h5:
                    try:
                        analyze_h5_array(sample, f"{filename}_{suffix}")
                    except Exception as exc:
                        print(f"  WARNING: HDF5 analysis failed: {exc}")

            elif fmt_lower in ["tif", "tiff"]:
                out_path = output_dir / f"{filename}_{suffix}.tiff"
                try:
                    save_volume(str(out_path), sample, file_format="tiff")
                    print(f"  Saved TIFF: {out_path.name} (shape: {sample.shape})")
                except Exception as exc:
                    print(f"  WARNING: TIFF export failed: {exc}")

            elif fmt_lower in ["nii", "nii.gz"]:
                out_path = output_dir / f"{filename}_{suffix}.nii.gz"
                try:
                    save_volume(str(out_path), sample, file_format="nii.gz")
                    print(f"  Saved NIfTI: {out_path.name}")
                except Exception as exc:
                    print(f"  WARNING: NIfTI export failed: {exc}")

            elif fmt_lower == "png":
                out_dir = output_dir / f"{filename}_{suffix}_png"
                try:
                    save_volume(str(out_dir), sample, file_format="png")
                    print(f"  Saved PNG stack: {out_dir.name}/")
                except Exception as exc:
                    print(f"  WARNING: PNG export failed: {exc}")

            else:
                print(f"  WARNING: Unknown format '{fmt}' - skipping. Supported: h5, tiff, nii.gz, png")


__all__ = ["resolve_output_filenames", "write_outputs"]
