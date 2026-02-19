"""Postprocessing and save-time transforms for inference outputs."""

from __future__ import annotations

import warnings

import numpy as np
from omegaconf import DictConfig

from ..config import Config


def analyze_h5_array(data: np.ndarray, name: str) -> None:
    """Print lightweight summary stats for a saved array."""
    print("\n  " + "-" * 66)
    print(f"  H5 ANALYSIS: {name}")
    print("  " + "-" * 66)
    print(f"  Shape:              {data.shape}")
    print(f"  Dtype:              {data.dtype}")
    print(f"  Min:                {data.min()}")
    print(f"  Max:                {data.max()}")
    print(f"  Mean:               {data.mean():.6f}")

    unique_vals = np.unique(data)
    num_unique = len(unique_vals)
    print(f"  Unique values:      {num_unique}")

    if num_unique <= 30:
        print(f"  Values:             {sorted(unique_vals.tolist())}")
    else:
        first_30 = sorted(unique_vals[:30].tolist())
        print(f"  First 30 values:    {first_30}")

    nonzero_count = np.count_nonzero(data)
    nonzero_pct = 100.0 * nonzero_count / data.size
    print(f"  Non-zero voxels:    {nonzero_count:,} / {data.size:,}")
    print(f"  Non-zero %:         {nonzero_pct:.2f}%")

    if data.max() == 0:
        print("  WARNING: All zeros - empty output")
    elif data.max() == data.min():
        print(f"  WARNING: Constant array (all values = {data.min()})")

    print("  " + "-" * 66 + "\n")


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
            print(f"  Normalized predictions to [0, 1] (min={data_min:.4f}, max={data_max:.4f})")
        else:
            print(f"  Warning: data_min == data_max ({data_min:.4f}), skipping normalization")

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
            warnings.warn(
                f"Unknown dtype '{target_dtype_str}' in save_prediction config. "
                f"Supported: {list(dtype_map.keys())}. Keeping current dtype.",
                UserWarning,
            )
            return data

        target_dtype = dtype_map[target_dtype_str]
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            data = np.clip(data, info.min, info.max)
            print(f"  Converting to {target_dtype_str} (clipped to [{info.min}, {info.max}])")

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
        from connectomics.decoding.postprocess import apply_binary_postprocessing

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
            warnings.warn(
                f"Transpose failed with axes {output_transpose}: {exc}. Keeping original shape.",
                UserWarning,
            )

    return data


__all__ = [
    "analyze_h5_array",
    "apply_save_prediction_transform",
    "apply_postprocessing",
]
