"""
Sliding-window inference helpers for PyTorch Connectomics.

This module isolates ROI/overlap resolution and SlidingWindowInferer creation
so the logic can be reused by both the Lightning module and TTA routines.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import torch
from monai.inferers import SlidingWindowInferer


def resolve_inferer_roi_size(cfg) -> Optional[Tuple[int, ...]]:
    """Determine the ROI size for sliding-window inference."""
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "sliding_window"):
        window_size = getattr(cfg.inference.sliding_window, "window_size", None)
        if window_size:
            return tuple(int(v) for v in window_size)

    if hasattr(cfg, "model") and hasattr(cfg.model, "output_size"):
        output_size = getattr(cfg.model, "output_size", None)
        if output_size:
            roi_size = tuple(int(v) for v in output_size)
            # For 2D models with do_2d=True, convert to 3D ROI size
            if getattr(cfg.data, "do_2d", False) and len(roi_size) == 2:
                roi_size = (1,) + roi_size  # Add depth dimension
            return roi_size

    if hasattr(cfg, "data") and hasattr(cfg.data, "patch_size"):
        patch_size = getattr(cfg.data, "patch_size", None)
        if patch_size:
            roi_size = tuple(int(v) for v in patch_size)
            # For 2D models with do_2d=True, convert to 3D ROI size
            if getattr(cfg.data, "do_2d", False) and len(roi_size) == 2:
                roi_size = (1,) + roi_size  # Add depth dimension
            return roi_size

    return None


def resolve_inferer_overlap(cfg, roi_size: Tuple[int, ...]) -> Union[float, Tuple[float, ...]]:
    """Resolve overlap parameter using inference config."""
    if not hasattr(cfg, "inference") or not hasattr(cfg.inference, "sliding_window"):
        return 0.5

    overlap = getattr(cfg.inference.sliding_window, "overlap", None)
    if overlap is not None:
        if isinstance(overlap, (list, tuple)):
            return tuple(float(max(0.0, min(o, 0.99))) for o in overlap)
        return float(max(0.0, min(overlap, 0.99)))

    stride = getattr(cfg.inference, "stride", None)
    if stride:
        values = []
        for size, step in zip(roi_size, stride):
            if size <= 0:
                values.append(0.0)
                continue
            ratio = 1.0 - float(step) / float(size)
            values.append(float(max(0.0, min(ratio, 0.99))))
        if len(set(values)) == 1:
            return values[0]
        return tuple(values)

    return 0.5


def build_sliding_inferer(cfg):
    """
    Build a MONAI SlidingWindowInferer if configuration permits.

    Returns:
        SlidingWindowInferer or None, plus diagnostic info for logging.
    """
    roi_size = resolve_inferer_roi_size(cfg)
    if roi_size is None:
        warnings.warn(
            "Sliding-window inference disabled: unable to determine ROI size. "
            "Set inference.window_size or model.output_size in the config.",
            UserWarning,
        )
        return None

    overlap = resolve_inferer_overlap(cfg, roi_size)
    # Use system.inference.batch_size as default, fall back to
    # sliding_window.sw_batch_size if specified
    system_batch_cfg = getattr(getattr(cfg, "system", None), "inference", None)
    system_batch_value = getattr(system_batch_cfg, "batch_size", 1) if system_batch_cfg else 1
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    config_sw_batch_size = getattr(sliding_cfg, "sw_batch_size", None) if sliding_cfg else None
    sw_batch_size = max(
        1, int(config_sw_batch_size if config_sw_batch_size is not None else system_batch_value)
    )
    mode = getattr(sliding_cfg, "blending", "gaussian") if sliding_cfg else "gaussian"
    sigma_scale = float(getattr(sliding_cfg, "sigma_scale", 0.125)) if sliding_cfg else 0.125
    padding_mode = getattr(sliding_cfg, "padding_mode", "constant") if sliding_cfg else "constant"
    keep_input_on_cpu = bool(getattr(sliding_cfg, "keep_input_on_cpu", False)) if sliding_cfg else False
    sw_device = getattr(sliding_cfg, "sw_device", None) if sliding_cfg else None
    output_device = getattr(sliding_cfg, "output_device", None) if sliding_cfg else None

    if isinstance(sw_device, str) and sw_device.lower() in {"", "none", "null"}:
        sw_device = None
    if isinstance(output_device, str) and output_device.lower() in {"", "none", "null"}:
        output_device = None

    if keep_input_on_cpu:
        # MONAI defaults both devices to inputs.device; when inputs stay on CPU,
        # explicitly route windows to CUDA and stitch on CPU unless overridden.
        if sw_device is None and torch.cuda.is_available():
            sw_device = "cuda"
        if output_device is None:
            output_device = "cpu"
        if sw_device is None:
            warnings.warn(
                "inference.sliding_window.keep_input_on_cpu=True but no sw_device was set "
                "and CUDA is unavailable. Sliding-window inference will run on CPU.",
                UserWarning,
            )

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode=padding_mode,
        sw_device=sw_device,
        device=output_device,
        progress=True,
    )

    print(
        "  Sliding-window inference configured: "
        f"roi_size={roi_size}, overlap={overlap}, sw_batch={sw_batch_size}, "
        f"mode={mode}, sigma_scale={sigma_scale}, padding={padding_mode}, "
        f"keep_input_on_cpu={keep_input_on_cpu}, sw_device={sw_device}, output_device={output_device}"
    )

    return inferer
