"""
I/O and decoding utilities for inference.

Contains helpers for postprocessing, decode modes, filename resolution, and writing outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import warnings

import numpy as np
from omegaconf import DictConfig

from ..config import Config


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
    )

    decode_fn_map = {
        "decode_instance_binary_contour_distance": decode_instance_binary_contour_distance,
        "decode_affinity_cc": decode_affinity_cc,
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


def write_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str = "prediction",
    mode: str = "test",
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

    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            print(f"  WARNING: write_outputs - no filename for batch index {idx}, skipping")
            continue

        sample = predictions[idx]
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
        write_hdf5(
            output_path,
            sample.astype(np.float32) if not np.issubdtype(sample.dtype, np.integer) else sample,
            dataset="main",
        )

        print(f"  ✓ Saved prediction: {output_path}")
