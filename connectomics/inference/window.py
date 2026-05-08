"""
Sliding-window inference helpers for PyTorch Connectomics.

Single in-house engine for both eager (in-memory tensor) and lazy
(disk-backed zarr/h5/tiff) inputs. The lazy reader lives in
``connectomics/inference/lazy.py``; this module owns:

- ROI / overlap / runtime knob resolution
- scan-interval and importance-map kernels
- the eager engine ``EagerSlidingWindowEngine`` invoked via
  ``build_sliding_inferer`` (callable contract:
  ``engine(inputs=tensor, network=fn) -> tensor``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_DISTANCE_TRANSFORM_BLEND_MODES = {
    "distance",
    "distance_transform",
    "distance-transform",
    "distance_transform_cdt",
    "banis",
    "banis_distance",
}


def _cfg_value(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_blending_mode(mode: str) -> str:
    return str(mode).strip().lower()


def is_distance_transform_blending(mode: str) -> bool:
    """Return True for BANIS-style distance-transform window blending."""
    return _normalize_blending_mode(mode) in _DISTANCE_TRANSFORM_BLEND_MODES


def compute_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int | None = None,
    overlap: Union[float, Sequence[float]] = 0.0,
) -> tuple[int, ...]:
    """Compute sliding-window stride from ROI size and per-axis overlap.

    Mirrors the contract of MONAI's private ``_get_scan_interval`` (positional
    ``num_spatial_dims`` is accepted and ignored — it is derived from
    ``len(roi_size)``). ``overlap`` may be a scalar or a per-axis sequence;
    values are clamped to ``[0, 0.99]``. When ``image_size <= roi_size`` along
    an axis, the stride collapses to ``image_size`` (a single window covers
    the axis).
    """
    del num_spatial_dims  # derived from len(roi_size)
    spatial_dims = len(roi_size)
    if isinstance(overlap, (list, tuple)):
        overlaps = [float(overlap[i]) for i in range(spatial_dims)]
    else:
        overlaps = [float(overlap)] * spatial_dims
    overlaps = [max(0.0, min(o, 0.99)) for o in overlaps]

    intervals: list[int] = []
    for axis in range(spatial_dims):
        roi = int(roi_size[axis])
        img = int(image_size[axis])
        if img <= roi:
            intervals.append(img)
            continue
        stride = max(1, int(round(roi * (1.0 - overlaps[axis]))))
        intervals.append(stride)
    return tuple(intervals)


def dense_patch_slices(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> list[tuple[slice, ...]] | list[tuple[int, ...]]:
    """Enumerate sliding-window starts covering the image.

    Replaces ``monai.data.utils.dense_patch_slices``. The last window per
    axis is snapped so that ``start + roi_size <= image_size`` whenever
    possible; an axis with ``image_size <= roi_size`` always gets one
    window starting at 0 (window may extend past the volume; callers pad).
    """
    spatial_dims = len(roi_size)
    starts_per_axis: list[list[int]] = []
    for axis in range(spatial_dims):
        roi = int(roi_size[axis])
        img = int(image_size[axis])
        stride = max(1, int(scan_interval[axis]))
        if img <= roi:
            starts_per_axis.append([0])
            continue
        starts = list(range(0, img - roi + 1, stride))
        last = img - roi
        if not starts or starts[-1] != last:
            starts.append(last)
        starts_per_axis.append(starts)

    def _walk(axis: int, prefix: tuple[int, ...]) -> list[tuple[int, ...]]:
        if axis == spatial_dims:
            return [prefix]
        out: list[tuple[int, ...]] = []
        for s in starts_per_axis[axis]:
            out.extend(_walk(axis + 1, prefix + (s,)))
        return out

    starts = _walk(0, ())
    if not return_slice:
        return list(starts)
    return [
        tuple(slice(s, s + int(roi_size[i])) for i, s in enumerate(start))
        for start in starts
    ]


def compute_importance_map(
    roi_size: Sequence[int],
    *,
    mode: str = "constant",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-window blending weights for sliding-window accumulation.

    Supported modes:

    - ``constant`` — uniform weight ``1.0``.
    - ``bump`` — Wu's compactly-supported bump function (DeepEM
      ``bump_map_wu``): ``exp(-Σ_axes 1/(1 - u_axis²))`` where each
      ``u_axis ∈ (-1, 1)`` is the normalized coordinate along that axis.
      Peak at the ROI center, smooth, and exactly zero at every boundary
      voxel — clean overlap blending without the pad-vs-dim constraint
      that bites Gaussian-decay weights at small ROIs. The result is
      peak-normalized to ``1.0`` and floor-clamped to ``finfo.tiny`` so
      the weight accumulator never divides by zero.

    See ``lib/DeepEM/deepem/test/mask.py::bump_map_wu`` for the reference
    implementation we follow.
    """
    spatial = tuple(int(v) for v in roi_size)
    if any(v <= 0 for v in spatial):
        raise ValueError(f"roi_size must contain positive values, got {roi_size}.")
    normalized = _normalize_blending_mode(mode)
    if normalized == "constant":
        return torch.ones(spatial, device=device, dtype=dtype)
    if normalized != "bump":
        raise ValueError(
            f"compute_importance_map: unsupported mode {mode!r}; expected "
            "'constant' or 'bump' (use is_distance_transform_blending for the "
            "distance-transform path)."
        )

    # Per-axis Wu bump factor: u = (idx + 1) / (size + 1) * 2 - 1 ∈ (-1, 1).
    # Final weight is the product (equivalent to exp of summed log-factors,
    # but numerically nicer to multiply per-axis kernels directly).
    importance: Optional[torch.Tensor] = None
    for axis, size in enumerate(spatial):
        idx = torch.arange(size, device=device, dtype=dtype)
        u = (idx + 1.0) / (size + 1.0) * 2.0 - 1.0
        # exp(-1 / (1 - u²)). When the normalization above keeps u strictly
        # inside (-1, 1), 1 - u² > 0 — but compute the divisor with a tiny
        # floor for fp16 safety.
        denom = (1.0 - u * u).clamp_min(torch.finfo(dtype).tiny)
        axis_kernel = torch.exp(-1.0 / denom)
        # Peak-normalize each axis kernel so the 3D peak is exactly 1.0.
        axis_kernel = axis_kernel / axis_kernel.max().clamp_min(torch.finfo(dtype).tiny)
        view = [1] * len(spatial)
        view[axis] = size
        importance = (
            axis_kernel.view(view)
            if importance is None
            else importance * axis_kernel.view(view)
        )
    assert importance is not None
    return importance.clamp_min(torch.finfo(dtype).tiny if dtype.is_floating_point else 0)


def build_sliding_importance_map(
    roi_size: Sequence[int],
    *,
    mode: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build per-window blending weights.

    ``distance_transform`` matches ``lib/banis``:
    ``distance_transform_cdt(np.pad(np.ones(roi), 1))[1:-1]``. For a solid
    rectangular window this is exactly ``min(distance_to_each_face) + 1``.
    """
    normalized_mode = _normalize_blending_mode(mode)
    if normalized_mode not in _DISTANCE_TRANSFORM_BLEND_MODES:
        return compute_importance_map(
            tuple(int(v) for v in roi_size),
            mode=normalized_mode,
            device=device,
            dtype=dtype,
        )

    spatial_shape = tuple(int(v) for v in roi_size)
    if not spatial_shape or any(v <= 0 for v in spatial_shape):
        raise ValueError(f"roi_size must contain positive values, got {roi_size}.")

    importance_map = None
    for axis, size in enumerate(spatial_shape):
        coord = torch.arange(size, device=device, dtype=dtype)
        dist = torch.minimum(coord + 1, torch.as_tensor(size, device=device, dtype=dtype) - coord)
        view_shape = [1] * len(spatial_shape)
        view_shape[axis] = size
        dist = dist.reshape(view_shape)
        importance_map = dist if importance_map is None else torch.minimum(importance_map, dist)

    return importance_map


def build_sliding_accumulator_weight_maps(
    roi_size: Sequence[int],
    *,
    mode: str,
    device: torch.device | str,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build value and weight maps for weighted sliding-window accumulation.

    The value map follows ``inference.model.output_dtype`` to control the large
    channel accumulator. The weight map is always fp32 because it is single
    channel and small, while fp16 can underflow bump-tail weights.
    """
    value_map = build_sliding_importance_map(
        roi_size,
        mode=mode,
        device=device,
        dtype=value_dtype,
    )
    if value_dtype == torch.float32:
        weight_map = value_map
    else:
        weight_map = build_sliding_importance_map(
            roi_size,
            mode=mode,
            device=device,
            dtype=torch.float32,
        )
    return value_map, weight_map


def normalize_weighted_accumulator(
    value_accumulator: torch.Tensor, weight_accumulator: torch.Tensor
) -> torch.Tensor:
    """Divide value by weight in-place while preserving value dtype."""
    clamp_value = 1.0e-6
    if value_accumulator.dtype == torch.float16:
        clamp_value = max(clamp_value, float(torch.finfo(torch.float16).tiny))
    divisor = torch.clamp_min(weight_accumulator, clamp_value)
    if divisor.dtype != value_accumulator.dtype:
        divisor = divisor.to(value_accumulator.dtype)
    value_accumulator /= divisor
    return value_accumulator


def apply_border_mask(importance_map: torch.Tensor, border_mask: Sequence[int]) -> torch.Tensor:
    """Zero outer ``k`` voxels of each spatial axis in ``importance_map``."""
    if not border_mask or all(int(b) <= 0 for b in border_mask):
        return importance_map
    spatial_dims = len(border_mask)
    spatial_shape = importance_map.shape[-spatial_dims:]
    for axis, k in enumerate(border_mask):
        k = int(k)
        if k <= 0:
            continue
        size = int(spatial_shape[axis])
        if 2 * k >= size:
            raise ValueError(
                f"inference.sliding_window.border_mask[{axis}]={k} is too large "
                f"for window size {size} on that axis."
            )
        idx = [slice(None)] * importance_map.ndim
        trailing = -(spatial_dims - axis)
        idx[trailing] = slice(0, k)
        importance_map[tuple(idx)] = 0
        idx[trailing] = slice(size - k, size)
        importance_map[tuple(idx)] = 0
    return importance_map


_MODEL_OUTPUT_DTYPE_ALIASES: dict[str, "torch.dtype"] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_model_output_dtype(cfg) -> torch.dtype:
    """Return the configured inference model-output dtype, defaulting to float32."""
    inference_cfg = _cfg_value(cfg, "inference", None)
    inference_model_cfg = _cfg_value(inference_cfg, "model", None)
    raw = _cfg_value(inference_model_cfg, "output_dtype", None)
    if raw is None:
        return torch.float32
    name = str(raw).strip().lower().removeprefix("torch.")
    if name in _MODEL_OUTPUT_DTYPE_ALIASES:
        return _MODEL_OUTPUT_DTYPE_ALIASES[name]
    raise ValueError(
        "inference.model.output_dtype must be one of "
        f"{sorted(_MODEL_OUTPUT_DTYPE_ALIASES)}, got {raw!r}."
    )


def resolve_border_mask(cfg, spatial_dims: int) -> list[int]:
    """Return the configured per-axis border mask, normalized to length ``spatial_dims``."""
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    raw = getattr(sliding_cfg, "border_mask", None) if sliding_cfg else None
    if not raw:
        return []
    values = [int(v) for v in raw]
    if len(values) == 1:
        values = values * spatial_dims
    if len(values) != spatial_dims:
        raise ValueError(
            f"inference.sliding_window.border_mask must have length 1 or {spatial_dims}, "
            f"got {len(values)}."
        )
    return values


def is_2d_inference_mode(cfg) -> bool:
    """Return True when data config indicates 2D mode."""
    train_cfg = getattr(getattr(cfg, "data", None), "train", None)
    val_cfg = getattr(getattr(cfg, "data", None), "val", None)
    return bool(getattr(train_cfg, "do_2d", False) or getattr(val_cfg, "do_2d", False))


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
            if is_2d_inference_mode(cfg) and len(roi_size) == 2:
                roi_size = (1,) + roi_size
            return roi_size

    if hasattr(cfg, "data") and hasattr(cfg.data, "data_transform"):
        patch_size = getattr(cfg.data.data_transform, "patch_size", None)
        if patch_size:
            roi_size = tuple(int(v) for v in patch_size)
            if is_2d_inference_mode(cfg) and len(roi_size) == 2:
                roi_size = (1,) + roi_size
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

    return 0.5


def _resolve_sliding_window_runtime(cfg, roi_size: Tuple[int, ...]) -> dict:
    overlap = resolve_inferer_overlap(cfg, roi_size)
    data_cfg = getattr(cfg, "data", None)
    data_loader_cfg = getattr(data_cfg, "dataloader", None) if data_cfg else None
    data_batch_value = getattr(data_loader_cfg, "batch_size", 1) if data_loader_cfg else 1
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    config_sw_batch_size = getattr(sliding_cfg, "sw_batch_size", None) if sliding_cfg else None
    sw_batch_size = max(
        1, int(config_sw_batch_size if config_sw_batch_size is not None else data_batch_value)
    )
    mode = _normalize_blending_mode(
        getattr(sliding_cfg, "blending", "bump") if sliding_cfg else "bump"
    )
    padding_mode = getattr(sliding_cfg, "padding_mode", "constant") if sliding_cfg else "constant"
    cval = float(getattr(sliding_cfg, "cval", 0.0)) if sliding_cfg else 0.0
    keep_input_on_cpu = (
        bool(getattr(sliding_cfg, "keep_input_on_cpu", False)) if sliding_cfg else False
    )
    sw_device = getattr(sliding_cfg, "sw_device", None) if sliding_cfg else None
    output_device = getattr(sliding_cfg, "output_device", None) if sliding_cfg else None

    if isinstance(sw_device, str) and sw_device.lower() in {"", "none", "null"}:
        sw_device = None
    if isinstance(output_device, str) and output_device.lower() in {"", "none", "null"}:
        output_device = None

    if keep_input_on_cpu:
        if sw_device is None and torch.cuda.is_available():
            sw_device = "cuda"
        if output_device is None:
            output_device = "cpu"
        if sw_device is None:
            logger.warning(
                "inference.sliding_window.keep_input_on_cpu=True but no sw_device was set "
                "and CUDA is unavailable. Sliding-window inference will run on CPU."
            )

    return {
        "overlap": overlap,
        "sw_batch_size": sw_batch_size,
        "mode": mode,
        "padding_mode": padding_mode,
        "cval": cval,
        "keep_input_on_cpu": keep_input_on_cpu,
        "sw_device": sw_device,
        "output_device": output_device,
    }


def _extract_padded_patch_batch(
    tensor: torch.Tensor,
    patch_slices: Sequence[tuple[slice, ...]],
    *,
    roi_size: tuple[int, ...],
    padding_mode: str,
    cval: float,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    if tensor.shape[0] != 1:
        raise ValueError(
            "Patch-first sliding-window TTA currently expects singleton batches. "
            f"Got batch size {tensor.shape[0]}."
        )

    spatial_dims = len(roi_size)
    image_size = tuple(int(v) for v in tensor.shape[-spatial_dims:])
    patches = []
    locations: list[tuple[int, ...]] = []

    for patch_slice in patch_slices:
        location = tuple(int(s.start) for s in patch_slice)
        end = tuple(location[axis] + int(roi_size[axis]) for axis in range(spatial_dims))

        inner_start = tuple(max(0, location[axis]) for axis in range(spatial_dims))
        inner_end = tuple(min(image_size[axis], end[axis]) for axis in range(spatial_dims))
        inner_slices = [
            slice(int(inner_start[axis]), int(inner_end[axis])) for axis in range(spatial_dims)
        ]
        inner = tensor[(slice(None), slice(None), *inner_slices)]

        pad_pairs = []
        for axis in range(spatial_dims):
            pad_before = max(0, -location[axis])
            pad_after = max(0, end[axis] - image_size[axis])
            pad_pairs.append((pad_before, pad_after))

        if any(before or after for before, after in pad_pairs):
            pad = []
            for before, after in reversed(pad_pairs):
                pad.extend([before, after])
            # PyTorch's reflect / circular modes require pad < dim per axis.
            # When a per-window pad would equal or exceed the inner-tensor
            # dim along any axis (the image-much-smaller-than-ROI case),
            # fall back to constant padding for that window. The configured
            # mode still applies whenever pads are within bounds.
            inner_dims = inner.shape[-spatial_dims:]
            mode = padding_mode
            if mode in {"reflect", "circular"}:
                exceeds = any(
                    (pad_pairs[axis][0] >= int(inner_dims[axis])
                     or pad_pairs[axis][1] >= int(inner_dims[axis]))
                    for axis in range(spatial_dims)
                )
                if exceeds:
                    mode = "constant"
            if mode == "constant":
                inner = F.pad(inner, tuple(pad), mode=mode, value=cval)
            else:
                inner = F.pad(inner, tuple(pad), mode=mode)

        patches.append(inner)
        locations.append(location)

    return torch.cat(patches, dim=0), locations


class EagerSlidingWindowEngine:
    """In-house eager sliding-window inferer over an in-memory tensor.

    Callable contract:
    ``engine(inputs=tensor, network=fn) -> tensor``. ``inputs`` is
    ``(B, C, *spatial)``; ``network`` is a callable that takes a window batch
    and returns a tensor. The engine accumulates predictions into a
    weight-blended output of shape ``(B, C_out, *spatial)``.
    """

    def __init__(
        self,
        *,
        roi_size: Sequence[int],
        sw_batch_size: int,
        overlap: Union[float, Sequence[float]],
        mode: str,
        padding_mode: str,
        cval: float,
        sw_device: torch.device | str | None,
        output_device: torch.device | str | None,
        progress: bool = False,
    ) -> None:
        self.roi_size = tuple(int(v) for v in roi_size)
        self.sw_batch_size = max(1, int(sw_batch_size))
        self.overlap = overlap
        self.mode = _normalize_blending_mode(mode)
        self.padding_mode = padding_mode
        self.cval = float(cval)
        self.sw_device = sw_device
        self.output_device = output_device
        self.progress = bool(progress)

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if inputs.dim() < len(self.roi_size) + 2:
            raise ValueError(
                f"EagerSlidingWindowEngine: inputs must have shape (B, C, *spatial); "
                f"got shape {tuple(inputs.shape)} for roi_size {self.roi_size}."
            )
        if inputs.shape[0] != 1:
            raise ValueError(
                "EagerSlidingWindowEngine currently expects batch size 1; "
                f"got batch {inputs.shape[0]}."
            )
        spatial_dims = len(self.roi_size)
        original_image_size = tuple(int(v) for v in inputs.shape[-spatial_dims:])
        # Pad the input to at least ROI size on every axis so the engine never
        # produces a truncated probe / partial window. Output is cropped back
        # to ``original_image_size`` at the end.
        pad_per_axis = [
            max(0, int(self.roi_size[i]) - original_image_size[i])
            for i in range(spatial_dims)
        ]
        if any(pad_per_axis):
            # ``reflect`` / ``replicate`` require pad < image dim per axis.
            # When growing the input up to ROI size, the pad amount can equal
            # or exceed the corresponding dim, which is fine for ``constant``
            # but errors for ``reflect``. Always use constant padding for the
            # up-front grow-to-ROI step; per-window boundary padding inside
            # ``_extract_padded_patch_batch`` continues to honor
            # ``self.padding_mode`` (small pads, never overruns).
            pad_pairs = []
            for axis in range(spatial_dims):
                pad_pairs.append((0, pad_per_axis[axis]))
            f_pad = []
            for before, after in reversed(pad_pairs):
                f_pad.extend([before, after])
            inputs = F.pad(inputs, tuple(f_pad), mode="constant", value=self.cval)
        image_size = tuple(int(v) for v in inputs.shape[-spatial_dims:])
        scan_interval = compute_scan_interval(
            image_size, self.roi_size, overlap=self.overlap
        )
        patch_slices = dense_patch_slices(image_size, self.roi_size, scan_interval)

        sw_device = torch.device(self.sw_device) if self.sw_device else inputs.device
        output_device = torch.device(self.output_device) if self.output_device else inputs.device

        # Probe one window through the same padded extraction the rest of the
        # batch uses, so image-smaller-than-ROI cases still produce a probe of
        # shape (1, C_in, *roi_size) that the network and weight maps expect.
        probe_input, probe_locations = _extract_padded_patch_batch(
            inputs,
            [patch_slices[0]],
            roi_size=self.roi_size,
            padding_mode=self.padding_mode,
            cval=self.cval,
        )
        probe_input = probe_input.to(device=sw_device, non_blocking=True)
        with torch.no_grad():
            probe_output = network(probe_input)
        if not isinstance(probe_output, torch.Tensor):
            raise ValueError(
                "EagerSlidingWindowEngine: `network` must return a torch.Tensor; "
                f"got {type(probe_output).__name__}."
            )
        out_channels = int(probe_output.shape[1])
        out_dtype = probe_output.dtype

        value_map, weight_map = build_sliding_accumulator_weight_maps(
            self.roi_size,
            mode=self.mode,
            device=output_device,
            value_dtype=out_dtype,
        )
        # Reshape to broadcast over (B, C, *spatial) and (B, 1, *spatial) targets.
        value_map_b = value_map.view(1, 1, *self.roi_size)
        weight_map_b = weight_map.view(1, 1, *self.roi_size)

        accum_shape = (1, out_channels, *image_size)
        value_accum = torch.zeros(accum_shape, device=output_device, dtype=out_dtype)
        weight_accum = torch.zeros(
            (1, 1, *image_size), device=output_device, dtype=weight_map.dtype
        )

        def _accumulate(window_output: torch.Tensor, location: tuple[int, ...]) -> None:
            target = tuple(slice(location[i], location[i] + self.roi_size[i])
                           for i in range(spatial_dims))
            window_on_out = window_output.to(device=output_device, dtype=out_dtype)
            value_accum[(slice(None), slice(None)) + target] += (
                window_on_out * value_map_b
            )
            weight_accum[(slice(None), slice(None)) + target] += weight_map_b

        # Accumulate the probe (location came from _extract_padded_patch_batch).
        _accumulate(probe_output[0:1], probe_locations[0])

        # Iterate the rest in batches.
        remaining = patch_slices[1:]
        for batch_start in range(0, len(remaining), self.sw_batch_size):
            batch_slices = remaining[batch_start : batch_start + self.sw_batch_size]
            batch_tensor, locations = _extract_padded_patch_batch(
                inputs,
                batch_slices,
                roi_size=self.roi_size,
                padding_mode=self.padding_mode,
                cval=self.cval,
            )
            batch_tensor = batch_tensor.to(device=sw_device, non_blocking=True)
            with torch.no_grad():
                batch_output = network(batch_tensor)
            for i, loc in enumerate(locations):
                _accumulate(batch_output[i : i + 1], loc)

        normalized = normalize_weighted_accumulator(value_accum, weight_accum)
        if any(pad_per_axis):
            crop = (slice(None), slice(None)) + tuple(
                slice(0, original_image_size[i]) for i in range(spatial_dims)
            )
            normalized = normalized[crop].contiguous()
        return normalized


def build_sliding_inferer(cfg) -> Optional[EagerSlidingWindowEngine]:
    """
    Build the in-house eager sliding-window engine.

    BANIS-style boundary handling (``snap_to_edge``, ``target_context``) is
    only honored by the lazy sliding-window path
    (``connectomics/inference/lazy.py``); the eager engine does not snap and
    does not consume ``border_mask`` (logged as a warning if set). For lazy
    inputs use ``lazy_predict_volume`` / ``lazy_predict_region`` directly.
    """
    roi_size = resolve_inferer_roi_size(cfg)
    if roi_size is None:
        logger.warning(
            "Sliding-window inference disabled: unable to determine ROI size. "
            "Set inference.window_size or model.output_size in the config."
        )
        return None

    runtime = _resolve_sliding_window_runtime(cfg, roi_size)
    if resolve_border_mask(cfg, len(roi_size)):
        logger.warning(
            "inference.sliding_window.border_mask is set but the eager "
            "sliding-window engine ignores it; use the lazy sliding-window "
            "path (lazy reader) to apply border masking."
        )

    engine = EagerSlidingWindowEngine(
        roi_size=roi_size,
        sw_batch_size=runtime["sw_batch_size"],
        overlap=runtime["overlap"],
        mode=runtime["mode"],
        padding_mode=runtime["padding_mode"],
        cval=runtime["cval"],
        sw_device=runtime["sw_device"],
        output_device=runtime["output_device"],
        progress=False,
    )

    logger.debug(
        "Sliding-window inference configured: "
        f"roi_size={roi_size}, overlap={runtime['overlap']}, sw_batch={runtime['sw_batch_size']}, "
        f"mode={runtime['mode']}, "
        f"padding={runtime['padding_mode']}, keep_input_on_cpu={runtime['keep_input_on_cpu']}, "
        f"sw_device={runtime['sw_device']}, output_device={runtime['output_device']}"
    )

    return engine


__all__ = [
    "EagerSlidingWindowEngine",
    "apply_border_mask",
    "build_sliding_accumulator_weight_maps",
    "build_sliding_importance_map",
    "build_sliding_inferer",
    "compute_importance_map",
    "compute_scan_interval",
    "dense_patch_slices",
    "is_2d_inference_mode",
    "is_distance_transform_blending",
    "normalize_weighted_accumulator",
    "resolve_border_mask",
    "resolve_inferer_overlap",
    "resolve_inferer_roi_size",
    "resolve_model_output_dtype",
]
