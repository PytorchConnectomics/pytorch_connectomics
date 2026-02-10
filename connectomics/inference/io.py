"""
I/O and decoding utilities for inference.

Contains helpers for postprocessing, decode modes, filename resolution, and writing outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import warnings

import numpy as np
from omegaconf import DictConfig

from ..config import Config


# ============================================================
# PART 3: Automatic .h5 Output Analysis Function
# ============================================================
def analyze_h5_array(data: np.ndarray, name: str) -> None:
    """
    Analyze and print statistics about a saved prediction array.
    
    This provides immediate feedback on data quality without requiring
    external tools or manual scripting.
    
    Args:
        data: Numpy array to analyze (already loaded from .h5)
        name: Descriptive name for logging
    """
    print(f"\n  {'─'*66}")
    print(f"  H5 ANALYSIS: {name}")
    print(f"  {'─'*66}")
    print(f"  Shape:              {data.shape}")
    print(f"  Dtype:              {data.dtype}")
    print(f"  Min:                {data.min()}")
    print(f"  Max:                {data.max()}")
    print(f"  Mean:               {data.mean():.6f}")
    
    # Unique values (computed once)
    unique_vals = np.unique(data)
    num_unique = len(unique_vals)
    print(f"  Unique values:      {num_unique}")
    
    # First 30 unique values
    if num_unique <= 30:
        print(f"  Values:             {sorted(unique_vals.tolist())}")
    else:
        first_30 = sorted(unique_vals[:30].tolist())
        print(f"  First 30 values:    {first_30}")
    
    # Non-zero statistics
    nonzero_count = np.count_nonzero(data)
    nonzero_pct = 100.0 * nonzero_count / data.size
    print(f"  Non-zero voxels:    {nonzero_count:,} / {data.size:,}")
    print(f"  Non-zero %:         {nonzero_pct:.2f}%")
    
    # Warnings for common issues
    if data.max() == 0:
        print(f"  ⚠️  WARNING: All zeros - empty output!")
    elif data.max() == data.min():
        print(f"  ⚠️  WARNING: Constant array (all values = {data.min()})")
    
    print(f"  {'─'*66}\n")


def apply_save_prediction_transform(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """
    Apply intensity scaling and dtype conversion from save_prediction config.

    This is used when saving intermediate predictions (before decoding).

    Default behavior (no config):
    - Normalizes predictions to [0, 1] using min-max normalization
    - Keeps dtype as float32

    Config options:
    - intensity_scale: If < 0, disables normalization (raw values)
                       If > 0, normalize to [0, 1] then multiply by scale
    - intensity_dtype: Target dtype for conversion (uint8, float32, etc.)

    Args:
        cfg: Configuration object
        data: Predictions array to transform

    Returns:
        Transformed predictions with applied scaling and dtype conversion
    """
    # Default: keep raw predictions if no config
    intensity_scale = -1.0  # Default: keep raw predictions

    if hasattr(cfg, "inference") and hasattr(cfg.inference, "save_prediction"):
        save_pred_cfg = cfg.inference.save_prediction
        intensity_scale = getattr(save_pred_cfg, "intensity_scale", -1.0)

    # Apply intensity scaling (if intensity_scale >= 0, normalize to [0, 1] then scale)
    if intensity_scale >= 0:
        # Convert to float32 for normalization
        data = data.astype(np.float32)

        # Min-max normalization to [0, 1]
        data_min = data.min()
        data_max = data.max()

        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
            print(f"  Normalized predictions to [0, 1] (min={data_min:.4f}, max={data_max:.4f})")
        else:
            print(f"  Warning: data_min == data_max ({data_min:.4f}), skipping normalization")

        # Apply scaling factor
        if intensity_scale != 1.0:
            data = data * float(intensity_scale)
            print(
                f"  Scaled predictions by {intensity_scale} -> "
                f"range [{data.min():.4f}, {data.max():.4f}]"
            )
    else:
        print(
            f"  Intensity scaling disabled (scale={intensity_scale} < 0), keeping raw predictions"
        )
        # Skip dtype conversion when intensity_scale < 0 to preserve raw predictions
        return data

    # Apply dtype conversion
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
            warnings.warn(
                f"Unknown dtype '{target_dtype_str}' in save_prediction config. "
                f"Supported: {list(dtype_map.keys())}. Keeping current dtype.",
                UserWarning,
            )
            return data

        target_dtype = dtype_map[target_dtype_str]

        # Get dtype info for proper clamping
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            data = np.clip(data, info.min, info.max)
            print(f"  Converting to {target_dtype_str} (clipped to [{info.min}, {info.max}])")

        data = data.astype(target_dtype)

    return data


def apply_postprocessing(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """
    Apply postprocessing transformations to predictions.

    This method applies:
    1. Binary postprocessing (morphological operations, connected components filtering)
    2. Axis transposition (output_transpose)

    Note: Intensity scaling and dtype conversion are handled by apply_save_prediction_transform()
    """
    if not hasattr(cfg, "inference") or not hasattr(cfg.inference, "postprocessing"):
        return data

    postprocessing = cfg.inference.postprocessing

    # Check if postprocessing is enabled
    if not getattr(postprocessing, "enabled", False):
        return data

    binary_config = getattr(postprocessing, "binary", None)
    if binary_config is not None and getattr(binary_config, "enabled", False):
        from connectomics.decoding.postprocess import apply_binary_postprocessing

        if data.ndim == 4:
            batch_size = data.shape[0]
        elif data.ndim == 5:
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

            processed = apply_binary_postprocessing(
                foreground_prob,
                threshold=getattr(binary_config, "threshold", 0.5),
                min_size=getattr(binary_config, "min_size", None),
                closing_radius=getattr(binary_config, "closing_radius", None),
                opening_radius=getattr(binary_config, "opening_radius", None),
                erosion_iterations=getattr(binary_config, "erosion_iterations", 0),
                dilation_iterations=getattr(binary_config, "dilation_iterations", 0),
                skeletonize=getattr(binary_config, "skeletonize", False),
                hole_size=getattr(binary_config, "hole_size", None),
                object_size=getattr(binary_config, "object_size", None),
            )

            if sample.ndim == 4:
                processed = processed[np.newaxis, ...]
            elif sample.ndim == 2:
                processed = processed[np.newaxis, np.newaxis, ...]

            results.append(processed)

        data = np.stack(results, axis=0)

    # Apply axis transposition if configured
    output_transpose = getattr(postprocessing, "output_transpose", [])
    if output_transpose and len(output_transpose) > 0:
        try:
            data = np.transpose(data, axes=output_transpose)
        except Exception as e:
            warnings.warn(
                f"Transpose failed with axes {output_transpose}: {e}. Keeping original shape.",
                UserWarning,
            )

    return data


def apply_decode_mode(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """Apply decode mode transformations to convert probability maps to instance segmentation."""
    decode_modes = None
    if hasattr(cfg, "test") and cfg.test and hasattr(cfg.test, "decoding"):
        decode_modes = cfg.test.decoding
        print(f"  🔧 Using test.decoding: {decode_modes}")
    elif hasattr(cfg, "inference") and hasattr(cfg.inference, "decoding"):
        decode_modes = cfg.inference.decoding
        print(f"  🔧 Using inference.decoding: {decode_modes}")

    if not decode_modes:
        print("  ⚠️  No decoding configuration found (test.decoding or inference.decoding)")
        return data

    from connectomics.decoding import (
        decode_instance_binary_contour_distance,
        decode_affinity_cc,
        decode_distance_watershed,
    )

    decode_fn_map = {
        "decode_instance_binary_contour_distance": decode_instance_binary_contour_distance,
        "decode_affinity_cc": decode_affinity_cc,
        "decode_distance_watershed": decode_distance_watershed,
    }

    # Handle different input shapes:
    # - 5D: (B, C, Z, H, W) - batch of multi-channel 3D volumes
    # - 4D: (C, Z, H, W) - single multi-channel 3D volume (add batch dim)
    # - 3D: (Z, H, W) - single-channel 3D volume (add batch and channel dims)
    # - 2D: (H, W) - single 2D image (add batch, channel, and Z dims)

    original_ndim = data.ndim
    if data.ndim == 4:
        # Assume (C, Z, H, W) - add batch dimension
        data = data[np.newaxis, ...]  # Now (B=1, C, Z, H, W)
        batch_size = 1
    elif data.ndim == 5:
        batch_size = data.shape[0]
    else:
        batch_size = 1
        if data.ndim == 3:
            data = data[np.newaxis, np.newaxis, ...]  # (Z, H, W) -> (B=1, C=1, Z, H, W)
        elif data.ndim == 2:
            data = data[np.newaxis, np.newaxis, np.newaxis, ...]  # (H, W) -> (B=1, C=1, Z=1, H, W)

    results = []
    for batch_idx in range(batch_size):
        sample = data[batch_idx]  # Now sample is (C, Z, H, W)

        for decode_cfg in decode_modes:
            fn_name = decode_cfg.name if hasattr(decode_cfg, "name") else decode_cfg.get("name")
            kwargs = (
                decode_cfg.kwargs if hasattr(decode_cfg, "kwargs") else decode_cfg.get("kwargs", {})
            )

            if hasattr(kwargs, "items"):
                kwargs = dict(kwargs)
            else:
                kwargs = {}

            if fn_name not in decode_fn_map:
                raise ValueError(
                    f"Unknown decode function '{fn_name}'. "
                    f"Available functions: {list(decode_fn_map.keys())}. "
                    f"Please update your config to use one of the available functions."
                )

            decode_fn = decode_fn_map[fn_name]

            try:
                sample = decode_fn(sample, **kwargs)
                # Note: decode functions return (Z, H, W) for instance segmentation
                # Don't add extra dimensions here - let the final stacking handle it
            except Exception as e:
                raise RuntimeError(
                    f"Error applying decode function '{fn_name}': {e}. "
                    f"Please check your decode configuration and parameters."
                ) from e

        results.append(sample)

    # Stack results along batch dimension
    if len(results) > 1:
        decoded = np.stack(results, axis=0)  # Multiple batches: (B, Z, H, W) or (B, C, Z, H, W)
    else:
        decoded = results[0]  # Single batch: (Z, H, W) or (C, Z, H, W)

    return decoded


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
    """Extract per-sample metadata regardless of collate representation."""
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
        # Most decoded outputs are (D, H, W) or (C, H, W); treat as 3D spatial by default.
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
        if (
            hasattr(cfg, "test")
            and hasattr(cfg.test, "data")
            and hasattr(cfg.test.data, "output_path")
        ):
            output_dir_value = cfg.test.data.output_path

    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import write_hdf5

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
        output_path = output_dir / f"{filename}_{suffix}.h5"

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
                except Exception as e:
                    print(
                        f"  WARNING: write_outputs - channel selection failed: "
                        f"{e}, keeping all channels"
                    )

        if output_transpose and len(output_transpose) > 0:
            try:
                sample = np.transpose(sample, axes=output_transpose)
            except Exception as e:
                print(f"  WARNING: write_outputs - transpose failed: {e}, keeping original shape")

        sample = np.squeeze(sample)
        
        # ============================================================
        # Get output formats from config (default: ['h5', 'nii.gz'] for backward compatibility)
        # ============================================================
        output_formats = ["h5", "nii.gz"]  # Default
        if hasattr(cfg, "inference") and hasattr(cfg.inference, "save_prediction"):
            save_pred_cfg = cfg.inference.save_prediction
            if hasattr(save_pred_cfg, "output_formats") and save_pred_cfg.output_formats:
                output_formats = save_pred_cfg.output_formats
        
        # ============================================================
        # Save in all requested formats
        # ============================================================
        for fmt in output_formats:
            fmt_lower = fmt.lower()
            
            if fmt_lower == "h5":
                # HDF5 format
                h5_path = output_dir / f"{filename}_{suffix}.h5"
                write_hdf5(
                    h5_path,
                    sample.astype(np.float32) if not np.issubdtype(sample.dtype, np.integer) else sample,
                    dataset="main",
                )
                print(f"  ✓ Saved HDF5: {h5_path.name}")
                
                # ============================================================
                # PART 3: Automatic .h5 Output Analysis
                # ============================================================
                try:
                    analyze_h5_array(sample, f"{filename}_{suffix}")
                except Exception as e:
                    print(f"  ⚠️  HDF5 analysis failed: {e}")
            
            elif fmt_lower in ["tif", "tiff"]:
                # TIFF format
                tiff_path = output_dir / f"{filename}_{suffix}.tiff"
                try:
                    from connectomics.data.io import save_volume
                    save_volume(
                        str(tiff_path),
                        sample,
                        file_format="tiff"
                    )
                    print(f"  ✓ Saved TIFF: {tiff_path.name} (shape: {sample.shape})")
                except ImportError as e:
                    print(f"  ⚠️  TIFF export failed (missing tifffile): {e}")
                except Exception as e:
                    print(f"  ⚠️  TIFF export failed: {e}")
            
            elif fmt_lower in ["nii", "nii.gz"]:
                # NIfTI format
                nifti_path = output_dir / f"{filename}_{suffix}.nii.gz"
                try:
                    import nibabel as nib
                    # Use identity affine
                    affine = np.eye(4)
                    
                    # FIX: Convert from internal (Z, Y, X) back to NIfTI (X, Y, Z) convention
                    # This reverses the transpose from read_volume() to match original input orientation
                    if sample.ndim == 3:
                        nifti_data = sample.transpose(2, 1, 0)  # (Z,Y,X) → (X,Y,Z)
                    elif sample.ndim == 4:
                        nifti_data = sample.transpose(3, 2, 1, 0)  # (C,Z,Y,X) → (X,Y,Z,C)
                    else:
                        nifti_data = sample  # Fallback for unexpected dimensions
                        print(f"  ⚠️  Unexpected NIfTI dimension: {sample.ndim}D, saving without transpose")
                    
                    # Preserve dtype
                    nifti_img = nib.Nifti1Image(nifti_data, affine)
                    nib.save(nifti_img, str(nifti_path))
                    print(f"  ✓ Saved NIfTI: {nifti_path.name} (shape: {nifti_data.shape})")
                except ImportError:
                    print(f"  ⚠️  NIfTI export skipped (nibabel not available)")
                except Exception as e:
                    print(f"  ⚠️  NIfTI export failed: {e}")
            
            elif fmt_lower == "png":
                # PNG format (saves as slice stack in a subdirectory)
                png_dir = output_dir / f"{filename}_{suffix}_png"
                try:
                    from connectomics.data.io import save_volume
                    save_volume(
                        str(png_dir),
                        sample,
                        file_format="png"
                    )
                    print(f"  ✓ Saved PNG stack: {png_dir.name}/")
                except Exception as e:
                    print(f"  ⚠️  PNG export failed: {e}")
            
            else:
                print(f"  ⚠️  Unknown format '{fmt}' - skipping. Supported: h5, tiff, nii.gz, png")
