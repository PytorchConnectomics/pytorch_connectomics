"""Lazy sliding-window inference utilities."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval, compute_importance_map

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ..data.augmentation.augment_ops import smart_normalize
from ..data.io.io import _detect_format, _get_tiff_volume_shape, _tiff_series_are_stackable
from ..data.processing.misc import get_padsize
from .sliding import resolve_inferer_overlap, resolve_inferer_roi_size
from .tta import TTAPredictor


def _normalize_transpose_axes(transpose_axes: Optional[Sequence[int]]) -> tuple[int, ...]:
    if transpose_axes is None:
        return ()
    axes = tuple(int(a) for a in transpose_axes)
    if not axes:
        return ()
    if len(axes) != 3 or sorted(axes) != [0, 1, 2]:
        raise ValueError(f"transpose_axes must be a permutation of [0,1,2], got {transpose_axes}")
    return axes


def _invert_transpose_axes(transpose_axes: Sequence[int]) -> tuple[int, ...]:
    if not transpose_axes:
        return ()
    inverse = [0, 0, 0]
    for out_axis, in_axis in enumerate(transpose_axes):
        inverse[in_axis] = out_axis
    return tuple(inverse)


def _scaled_length(length: int, factor: float) -> int:
    if factor <= 0:
        raise ValueError(f"scale factor must be positive, got {factor}.")
    return max(1, int(math.floor(float(length) * float(factor) + 1e-6)))


def _normalize_pad_mode(mode: str) -> str:
    pad_mode = str(mode).lower()
    if pad_mode == "replicate":
        return "edge"
    return pad_mode


def _output_indices_to_input_coords(
    start: int,
    end: int,
    *,
    input_size: int,
    output_size: int,
    mode: str,
    align_corners: Optional[bool],
) -> np.ndarray:
    indices = np.arange(int(start), int(end), dtype=np.float32)
    if indices.size == 0:
        return indices

    if input_size <= 1 or output_size <= 1:
        return np.zeros_like(indices, dtype=np.float32)

    if mode == "nearest":
        coords = np.floor(indices * float(input_size) / float(output_size))
        return np.clip(coords, 0, input_size - 1).astype(np.float32, copy=False)

    if align_corners:
        coords = indices * float(input_size - 1) / float(output_size - 1)
    else:
        coords = ((indices + 0.5) * float(input_size) / float(output_size)) - 0.5
        coords = np.clip(coords, 0.0, float(input_size - 1))
    return coords.astype(np.float32, copy=False)


def _normalize_grid_axis(coords: np.ndarray, axis_length: int) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32, copy=False)
    if axis_length <= 1:
        return np.zeros_like(coords, dtype=np.float32)
    return ((2.0 * coords) / float(axis_length - 1) - 1.0).astype(np.float32, copy=False)


def _reflect_indices(indices: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * length - 2
    mod = np.abs(indices).astype(np.int64) % period
    return np.where(mod < length, mod, period - mod)


def _pad_channel_first(
    array: np.ndarray,
    pads: Sequence[tuple[int, int]],
    *,
    mode: str,
    constant_value: float = 0.0,
) -> np.ndarray:
    if not any(before > 0 or after > 0 for before, after in pads):
        return array

    np_mode = _normalize_pad_mode(mode)
    pad_width = [(0, 0)] + [(int(before), int(after)) for before, after in pads]
    if np_mode == "constant":
        return np.pad(array, pad_width, mode=np_mode, constant_values=constant_value)

    spatial_shape = array.shape[1:]
    if np_mode == "reflect" and any(size <= 1 for size in spatial_shape):
        np_mode = "edge"

    return np.pad(array, pad_width, mode=np_mode)


def _coerce_overlap(overlap: float | Sequence[float], spatial_dims: int) -> tuple[float, ...]:
    if isinstance(overlap, (list, tuple)):
        if len(overlap) != spatial_dims:
            raise ValueError(
                f"Overlap rank mismatch: expected {spatial_dims} values, got {overlap}."
            )
        return tuple(float(v) for v in overlap)
    return tuple(float(overlap) for _ in range(spatial_dims))


def _resolve_scale_factors(cfg, *, kind: str, mode: str) -> Optional[tuple[float, ...]]:
    data_cfg = cfg.data
    resize_cfg = getattr(data_cfg.data_transform, "resize", None)
    patch_size_cfg = getattr(data_cfg.dataloader, "patch_size", None)

    if mode in {"test", "tune"} and resize_cfg:
        if (
            patch_size_cfg
            and len(patch_size_cfg) == len(resize_cfg)
            and all(float(v) > 0 for v in patch_size_cfg)
        ):
            return tuple(
                float(out_size) / float(in_size)
                for out_size, in_size in zip(resize_cfg, patch_size_cfg)
            )
        raise ValueError(
            "Lazy sliding-window inference requires data.dataloader.patch_size when "
            "data_transform.resize is configured."
        )

    if kind in {"image", "label"}:
        image_resize_factors = getattr(data_cfg.image_transform, "resize", None)
        if image_resize_factors:
            return tuple(float(v) for v in image_resize_factors)

    if kind == "mask":
        mask_cfg = getattr(data_cfg, "mask_transform", None) or data_cfg.data_transform
        mask_resize_factors = getattr(mask_cfg, "resize", None)
        if mask_resize_factors:
            return tuple(float(v) for v in mask_resize_factors)

    return None


class LazyVolumeAccessor:
    """Random-access reader that reproduces the relevant test-time transforms lazily."""

    def __init__(
        self,
        path: str,
        *,
        kind: str,
        transpose_axes: Sequence[int] = (),
        scale_factors: Optional[Sequence[float]] = None,
        context_pad: Optional[Sequence[tuple[int, int]]] = None,
        context_pad_mode: str = "constant",
        normalize_mode: str = "none",
        clip_percentile_low: float = 0.0,
        clip_percentile_high: float = 1.0,
        binarize: bool = False,
        threshold: float = 0.0,
    ):
        self.path = str(path)
        self.kind = kind
        self.fmt = _detect_format(self.path)
        self.transpose_axes = _normalize_transpose_axes(transpose_axes)
        self.inverse_transpose_axes = _invert_transpose_axes(self.transpose_axes)
        self.scale_factors = (
            tuple(float(v) for v in scale_factors) if scale_factors is not None else None
        )
        self.context_pad = tuple(context_pad or ((0, 0), (0, 0), (0, 0)))
        self.context_pad_mode = context_pad_mode
        self.normalize_mode = normalize_mode
        self.clip_percentile_low = float(clip_percentile_low)
        self.clip_percentile_high = float(clip_percentile_high)
        self.binarize = bool(binarize)
        self.threshold = float(threshold)

        self._handle = None
        self._dataset = None
        self._tiff = None
        self._open_handle()

        self.raw_shape = tuple(int(v) for v in self._get_raw_shape())
        self.layout = self._infer_layout(self.raw_shape)
        self.channel_count, self.raw_spatial_shape = self._resolve_channel_and_spatial_shape(
            self.raw_shape, self.layout
        )
        self.logical_spatial_shape = self._transpose_shape(self.raw_spatial_shape)
        self.transformed_spatial_shape = tuple(
            _scaled_length(size, factor)
            for size, factor in zip(
                self.logical_spatial_shape,
                self.scale_factors or (1.0,) * len(self.logical_spatial_shape),
            )
        )
        self.padded_spatial_shape = tuple(
            self.transformed_spatial_shape[axis]
            + int(self.context_pad[axis][0])
            + int(self.context_pad[axis][1])
            for axis in range(len(self.transformed_spatial_shape))
        )

    def close(self) -> None:
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        if self._handle is not None and hasattr(self._handle, "close"):
            self._handle.close()
            self._handle = None
        self._dataset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _open_handle(self) -> None:
        if self.fmt == "h5":
            import h5py

            self._handle = h5py.File(self.path, "r")
            dataset_name = list(self._handle.keys())[0]
            self._dataset = self._handle[dataset_name]
        elif self.fmt == "zarr":
            import zarr

            self._dataset = zarr.open(self.path, mode="r")
        elif self.fmt == "tiff":
            import tifffile

            self._tiff = tifffile.TiffFile(self.path)
        else:
            raise ValueError(
                f"Lazy sliding-window inference does not support format '{self.fmt}' for {self.path}."
            )

    def _get_raw_shape(self) -> tuple[int, ...]:
        if self.fmt in {"h5", "zarr"}:
            return tuple(int(v) for v in self._dataset.shape)
        if self.fmt == "tiff":
            return tuple(int(v) for v in _get_tiff_volume_shape(self.path))
        raise AssertionError("unreachable")

    @staticmethod
    def _infer_layout(shape: Sequence[int]) -> str:
        if len(shape) == 3:
            return "no_channel"
        if len(shape) != 4:
            raise ValueError(f"Unsupported lazy volume rank {len(shape)} for shape {shape}.")

        min_axis = int(np.argmin(shape))
        if min_axis == 0:
            return "channel_first"
        if min_axis == 3:
            return "channel_last"
        if min_axis == 1:
            return "channel_second"
        return "channel_first"

    @staticmethod
    def _resolve_channel_and_spatial_shape(
        shape: Sequence[int], layout: str
    ) -> tuple[int, tuple[int, int, int]]:
        if layout == "no_channel":
            return 1, tuple(int(v) for v in shape)
        if layout == "channel_first":
            return int(shape[0]), tuple(int(v) for v in shape[1:])
        if layout == "channel_last":
            return int(shape[-1]), tuple(int(v) for v in shape[:3])
        if layout == "channel_second":
            return int(shape[1]), (int(shape[0]), int(shape[2]), int(shape[3]))
        raise ValueError(f"Unknown layout: {layout}")

    def _transpose_shape(self, shape: Sequence[int]) -> tuple[int, int, int]:
        if not self.transpose_axes:
            return tuple(int(v) for v in shape)
        return tuple(int(shape[idx]) for idx in self.transpose_axes)

    def _transpose_spatial_array(self, array: np.ndarray) -> np.ndarray:
        if not self.transpose_axes:
            return array
        return np.transpose(array, self.transpose_axes)

    def _logical_to_raw_slices(
        self, logical_start: Sequence[int], logical_end: Sequence[int]
    ) -> tuple[slice, slice, slice]:
        if not self.transpose_axes:
            return tuple(slice(int(logical_start[idx]), int(logical_end[idx])) for idx in range(3))

        raw_slices = []
        for raw_axis in range(3):
            logical_axis = self.inverse_transpose_axes[raw_axis]
            raw_slices.append(
                slice(int(logical_start[logical_axis]), int(logical_end[logical_axis]))
            )
        return tuple(raw_slices)

    def _read_h5_or_zarr(self, raw_slices: tuple[slice, slice, slice]) -> np.ndarray:
        if self.layout == "no_channel":
            return np.asarray(self._dataset[raw_slices])
        if self.layout == "channel_first":
            return np.asarray(self._dataset[(slice(None),) + raw_slices])
        if self.layout == "channel_last":
            return np.asarray(self._dataset[raw_slices + (slice(None),)])
        if self.layout == "channel_second":
            z_slice, y_slice, x_slice = raw_slices
            return np.asarray(self._dataset[(z_slice, slice(None), y_slice, x_slice)])
        raise AssertionError("unreachable")

    def _read_tiff(self, raw_slices: tuple[slice, slice, slice]) -> np.ndarray:
        z_slice, y_slice, x_slice = raw_slices
        start = int(z_slice.start or 0)
        stop = int(z_slice.stop if z_slice.stop is not None else self.raw_spatial_shape[0])
        key = range(start, stop)

        if len(self.raw_shape) == 2:
            data = self._tiff.pages[0].asarray()[y_slice, x_slice]
            return data[np.newaxis, ...]

        if len(self._tiff.series) == 0:
            data = self._tiff.asarray(key=key)
        elif _tiff_series_are_stackable(self._tiff):
            data = self._tiff.asarray(key=key)
        else:
            data = self._tiff.series[0].asarray(key=key)

        data = np.asarray(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim >= 3:
            data = (
                data[..., y_slice, x_slice]
                if data.ndim == 4 and self.layout == "channel_first"
                else data
            )
        if self.layout == "no_channel":
            return data[:, y_slice, x_slice]
        if self.layout == "channel_last":
            return data[:, y_slice, x_slice, :]
        if self.layout == "channel_second":
            return data[:, :, y_slice, x_slice]
        if self.layout == "channel_first":
            # TIFF channel-first is uncommon; fall back to the fully loaded representation above.
            return data[:, :, y_slice, x_slice]
        raise AssertionError("unreachable")

    def _read_raw_crop(
        self, logical_start: Sequence[int], logical_end: Sequence[int]
    ) -> np.ndarray:
        raw_slices = self._logical_to_raw_slices(logical_start, logical_end)
        if self.fmt in {"h5", "zarr"}:
            data = self._read_h5_or_zarr(raw_slices)
        else:
            data = self._read_tiff(raw_slices)

        if self.layout == "no_channel":
            data = self._transpose_spatial_array(data)
            return data[np.newaxis, ...]
        if self.layout == "channel_last":
            if self.transpose_axes:
                data = np.transpose(data, [*self.transpose_axes, 3])
            return np.moveaxis(data, -1, 0)
        if self.layout == "channel_second":
            if self.transpose_axes:
                data = np.transpose(data, [1, 0, *[axis + 2 for axis in self.transpose_axes]])
            else:
                data = np.transpose(data, (1, 0, 2, 3))
            return data
        if self.transpose_axes:
            data = np.transpose(data, [0, *[axis + 1 for axis in self.transpose_axes]])
        return data

    def _read_transformed_bbox(
        self, start: Sequence[int], end: Sequence[int], *, mode: str, align_corners: Optional[bool]
    ) -> np.ndarray:
        bbox_shape = tuple(int(end[idx]) - int(start[idx]) for idx in range(3))
        if any(size <= 0 for size in bbox_shape):
            return np.zeros((self.channel_count, *bbox_shape), dtype=np.float32)

        if not self.scale_factors:
            return self._read_raw_crop(start, end).astype(np.float32, copy=False)

        output_coords = [
            _output_indices_to_input_coords(
                int(start[idx]),
                int(end[idx]),
                input_size=int(self.logical_spatial_shape[idx]),
                output_size=int(self.transformed_spatial_shape[idx]),
                mode=mode,
                align_corners=align_corners,
            )
            for idx in range(3)
        ]
        raw_start = tuple(int(math.floor(float(coords.min()))) for coords in output_coords)
        raw_end = tuple(
            min(
                int(self.logical_spatial_shape[idx]),
                int(math.ceil(float(output_coords[idx].max()))) + 1,
            )
            for idx in range(3)
        )
        raw_crop = self._read_raw_crop(raw_start, raw_end).astype(np.float32, copy=False)
        local_coords = [output_coords[idx] - float(raw_start[idx]) for idx in range(3)]

        if mode == "nearest":
            local_indices = [coords.astype(np.int64, copy=False) for coords in local_coords]
            gathered = np.take(raw_crop, local_indices[0], axis=1)
            gathered = np.take(gathered, local_indices[1], axis=2)
            gathered = np.take(gathered, local_indices[2], axis=3)
            return gathered.astype(np.float32, copy=False)

        grid_axes = [
            _normalize_grid_axis(local_coords[idx], int(raw_crop.shape[idx + 1]))
            for idx in range(3)
        ]
        zz, yy, xx = np.meshgrid(grid_axes[0], grid_axes[1], grid_axes[2], indexing="ij")
        grid = np.stack([xx, yy, zz], axis=-1)
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)
        raw_tensor = torch.from_numpy(raw_crop).unsqueeze(0)
        sampled = F.grid_sample(
            raw_tensor,
            grid_tensor,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def _read_padded_inner_region(
        self,
        inner_start: Sequence[int],
        inner_end: Sequence[int],
        *,
        mode: str,
        align_corners: Optional[bool],
    ) -> np.ndarray:
        mapped_indices = []
        valid_axes = []
        bbox_start = []
        bbox_end = []

        for axis in range(3):
            coords = np.arange(int(inner_start[axis]), int(inner_end[axis]), dtype=np.int64)
            unpadded = coords - int(self.context_pad[axis][0])
            size = int(self.transformed_spatial_shape[axis])
            pad_mode = _normalize_pad_mode(self.context_pad_mode)

            if pad_mode == "constant":
                valid = (unpadded >= 0) & (unpadded < size)
                mapped = np.clip(unpadded, 0, max(size - 1, 0))
            elif pad_mode == "reflect":
                valid = np.ones_like(unpadded, dtype=bool)
                mapped = _reflect_indices(unpadded, size)
            elif pad_mode == "edge":
                valid = np.ones_like(unpadded, dtype=bool)
                mapped = np.clip(unpadded, 0, max(size - 1, 0))
            else:
                raise ValueError(f"Unsupported context pad mode '{self.context_pad_mode}'.")

            mapped_indices.append(mapped)
            valid_axes.append(valid)
            bbox_start.append(int(mapped.min()) if mapped.size else 0)
            bbox_end.append(int(mapped.max()) + 1 if mapped.size else 0)

        region = self._read_transformed_bbox(
            bbox_start,
            bbox_end,
            mode=mode,
            align_corners=align_corners,
        )

        local_indices = [mapped_indices[axis] - bbox_start[axis] for axis in range(3)]
        gathered = np.take(region, local_indices[0], axis=1)
        gathered = np.take(gathered, local_indices[1], axis=2)
        gathered = np.take(gathered, local_indices[2], axis=3)

        if _normalize_pad_mode(self.context_pad_mode) == "constant":
            valid_mask = (
                valid_axes[0][:, None, None]
                & valid_axes[1][None, :, None]
                & valid_axes[2][None, None, :]
            )
            gathered = gathered * valid_mask[None].astype(gathered.dtype, copy=False)

        return gathered

    def read_patch(
        self,
        location: Sequence[int],
        patch_size: Sequence[int],
        *,
        outer_pad_mode: str,
        outer_pad_value: float,
    ) -> np.ndarray:
        start = tuple(int(v) for v in location)
        size = tuple(int(v) for v in patch_size)
        end = tuple(start[idx] + size[idx] for idx in range(3))

        inner_start = tuple(max(0, start[idx]) for idx in range(3))
        inner_end = tuple(min(int(self.padded_spatial_shape[idx]), end[idx]) for idx in range(3))

        if any(inner_end[idx] <= inner_start[idx] for idx in range(3)):
            inner = np.zeros((self.channel_count, 0, 0, 0), dtype=np.float32)
        else:
            interp_mode = "nearest" if self.kind in {"label", "mask"} else "bilinear"
            align_corners = None if interp_mode == "nearest" else True
            inner = self._read_padded_inner_region(
                inner_start,
                inner_end,
                mode=interp_mode,
                align_corners=align_corners,
            )

        outer_pads = []
        for axis in range(3):
            before = max(0, -start[axis])
            after = max(0, end[axis] - int(self.padded_spatial_shape[axis]))
            outer_pads.append((before, after))

        patch = _pad_channel_first(
            inner,
            outer_pads,
            mode=outer_pad_mode,
            constant_value=outer_pad_value,
        )

        if self.binarize:
            patch = (patch > self.threshold).astype(np.float32, copy=False)

        if self.kind == "image" and self.normalize_mode != "none":
            patch = smart_normalize(
                patch,
                self.normalize_mode,
                divide_value=None,
                clip_percentile_low=self.clip_percentile_low,
                clip_percentile_high=self.clip_percentile_high,
            ).astype(np.float32, copy=False)

        return patch.astype(np.float32, copy=False)

    def load_full(self) -> np.ndarray:
        interp_mode = "nearest" if self.kind in {"label", "mask"} else "bilinear"
        align_corners = None if interp_mode == "nearest" else True
        full = self._read_transformed_bbox(
            (0, 0, 0),
            self.transformed_spatial_shape,
            mode=interp_mode,
            align_corners=align_corners,
        )
        if self.binarize:
            full = (full > self.threshold).astype(np.float32, copy=False)
        return full.astype(np.float32, copy=False)


def _build_accessor(cfg, path: str, *, kind: str, mode: str) -> LazyVolumeAccessor:
    data_cfg = cfg.data
    transpose_axes = getattr(data_cfg.data_transform, "val_transpose", None) or ()
    scale_factors = _resolve_scale_factors(cfg, kind=kind, mode=mode)
    pad_size = get_padsize(getattr(data_cfg.data_transform, "pad_size", [0, 0, 0]), ndim=3)

    context_pad = pad_size if kind in {"image", "mask"} else ((0, 0), (0, 0), (0, 0))
    context_pad_mode = (
        getattr(data_cfg.data_transform, "pad_mode", "reflect") if kind == "image" else "constant"
    )

    normalize_mode = "none"
    clip_low = 0.0
    clip_high = 1.0
    if kind == "image":
        normalize_mode = getattr(data_cfg.image_transform, "normalize", "none")
        clip_low = float(getattr(data_cfg.image_transform, "clip_percentile_low", 0.0))
        clip_high = float(getattr(data_cfg.image_transform, "clip_percentile_high", 1.0))

    binarize = False
    threshold = 0.0
    if kind == "mask":
        mask_cfg = getattr(data_cfg, "mask_transform", None) or data_cfg.data_transform
        binarize = bool(getattr(mask_cfg, "binarize", False))
        threshold = float(getattr(mask_cfg, "threshold", 0.0))

    return LazyVolumeAccessor(
        path,
        kind=kind,
        transpose_axes=transpose_axes,
        scale_factors=scale_factors,
        context_pad=context_pad,
        context_pad_mode=context_pad_mode,
        normalize_mode=normalize_mode,
        clip_percentile_low=clip_low,
        clip_percentile_high=clip_high,
        binarize=binarize,
        threshold=threshold,
    )


def get_lazy_image_reference_shape(cfg, image_path: str, *, mode: str = "test") -> tuple[int, ...]:
    with _build_accessor(cfg, image_path, kind="image", mode=mode) as accessor:
        patch_size_cfg = getattr(cfg.data.dataloader, "patch_size", None)
        if patch_size_cfg and any(
            accessor.transformed_spatial_shape[idx] < int(patch_size_cfg[idx]) for idx in range(3)
        ):
            raise ValueError(
                "Lazy sliding-window inference currently requires the transformed test volume "
                "to be at least as large as data.dataloader.patch_size in every axis. "
                f"Got transformed_shape={accessor.transformed_spatial_shape}, "
                f"patch_size={tuple(int(v) for v in patch_size_cfg)}."
            )
        return (
            1,
            int(accessor.channel_count),
            *tuple(int(v) for v in accessor.padded_spatial_shape),
        )


def load_lazy_volume(cfg, path: str, *, kind: str, mode: str = "test") -> np.ndarray:
    with _build_accessor(cfg, path, kind=kind, mode=mode) as accessor:
        return accessor.load_full()


def _ensure_region_min_size(
    start: Sequence[int],
    stop: Sequence[int],
    *,
    min_shape: Sequence[int],
    bounds_shape: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Expand a region in-bounds so every axis is at least ``min_shape``."""
    expanded_start = [int(v) for v in start]
    expanded_stop = [int(v) for v in stop]

    for axis in range(3):
        bounds = int(bounds_shape[axis])
        min_size = int(min_shape[axis])
        if bounds < min_size:
            raise ValueError(
                "Lazy region inference requires the transformed test volume to be at least "
                f"as large as the ROI in every axis. Axis {axis}: bounds={bounds}, "
                f"roi={min_size}."
            )

        current = expanded_stop[axis] - expanded_start[axis]
        if current >= min_size:
            continue

        missing = min_size - current
        grow_before = min(missing // 2, expanded_start[axis])
        expanded_start[axis] -= grow_before
        missing -= grow_before
        grow_after = min(missing, bounds - expanded_stop[axis])
        expanded_stop[axis] += grow_after
        missing -= grow_after
        if missing > 0:
            grow_before = min(missing, expanded_start[axis])
            expanded_start[axis] -= grow_before
            missing -= grow_before
        if missing > 0:
            raise ValueError(
                f"Could not expand lazy inference region on axis {axis} to ROI size {min_size}."
            )

    return tuple(expanded_start), tuple(expanded_stop)


def lazy_predict_region(
    cfg,
    forward_fn,
    image_path: str,
    *,
    region_start: Sequence[int],
    region_stop: Sequence[int],
    mask_path: Optional[str] = None,
    mask_align_to_image: bool = False,
    device: torch.device | str = "cpu",
    requested_head: Optional[str] = None,
) -> torch.Tensor:
    """Run lazy sliding-window inference for one bounded region.

    ``region_start`` and ``region_stop`` are in transformed/padded ZYX
    coordinates, matching the coordinate system used by full-volume lazy
    inference. Only this region is accumulated in CPU memory.
    """
    roi_size = resolve_inferer_roi_size(cfg)
    if roi_size is None:
        raise ValueError(
            "Lazy region inference requires inference.sliding_window.window_size "
            "or model.output_size to be configured."
        )
    if len(roi_size) != 3:
        raise ValueError(f"Lazy region inference currently supports 3D only, got {roi_size}.")

    overlap = _coerce_overlap(resolve_inferer_overlap(cfg, roi_size), spatial_dims=3)
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    sw_batch_size = max(
        1,
        int(
            getattr(sliding_cfg, "sw_batch_size", None)
            or getattr(getattr(cfg, "data", None), "dataloader", None).batch_size
        ),
    )
    blend_mode = getattr(sliding_cfg, "blending", "gaussian")
    sigma_scale = float(getattr(sliding_cfg, "sigma_scale", 0.125))
    outer_pad_mode = getattr(sliding_cfg, "padding_mode", "constant")
    outer_pad_value = float(getattr(sliding_cfg, "cval", 0.0))
    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=forward_fn)
    if predictor.is_distributed_sharding_enabled():
        raise RuntimeError(
            "Lazy region inference does not support "
            "`inference.test_time_augmentation.distributed_sharding`. Disable it first."
        )

    infer_device = torch.device(device)
    accumulation_device = torch.device("cpu")

    with _build_accessor(cfg, image_path, kind="image", mode="test") as image_accessor:
        bounds_shape = tuple(int(v) for v in image_accessor.padded_spatial_shape)
        start = tuple(max(0, int(v)) for v in region_start)
        stop = tuple(min(bounds_shape[idx], int(region_stop[idx])) for idx in range(3))
        if any(stop[idx] <= start[idx] for idx in range(3)):
            raise ValueError(f"Empty lazy inference region: start={start}, stop={stop}")

        start, stop = _ensure_region_min_size(
            start,
            stop,
            min_shape=roi_size,
            bounds_shape=bounds_shape,
        )
        image_size = tuple(int(stop[idx]) - int(start[idx]) for idx in range(3))
        scan_interval = _get_scan_interval(
            image_size, roi_size, num_spatial_dims=3, overlap=overlap
        )
        patch_slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)
        importance_map = (
            compute_importance_map(
                tuple(int(v) for v in roi_size),
                mode=blend_mode,
                sigma_scale=overlap if blend_mode == "constant" else sigma_scale,
                device=accumulation_device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mask_accessor = (
            _build_accessor(cfg, mask_path, kind="mask", mode="test")
            if mask_path is not None
            else None
        )
        try:
            value_accumulator = None
            weight_accumulator = torch.zeros(
                (1, 1, *image_size), device=accumulation_device, dtype=torch.float32
            )

            for batch_start in range(0, len(patch_slices), sw_batch_size):
                current_slices = patch_slices[batch_start : batch_start + sw_batch_size]
                image_batch = []
                mask_batch = [] if mask_accessor is not None else None
                locations = []

                for patch_slice in current_slices:
                    local_location = tuple(int(s.start) for s in patch_slice)
                    global_location = tuple(start[idx] + local_location[idx] for idx in range(3))
                    image_batch.append(
                        image_accessor.read_patch(
                            global_location,
                            roi_size,
                            outer_pad_mode=outer_pad_mode,
                            outer_pad_value=outer_pad_value,
                        )
                    )
                    if mask_accessor is not None:
                        mask_batch.append(
                            mask_accessor.read_patch(
                                global_location,
                                roi_size,
                                outer_pad_mode="constant",
                                outer_pad_value=0.0,
                            )
                        )
                    locations.append(local_location)

                image_tensor = torch.from_numpy(np.stack(image_batch, axis=0)).to(
                    device=infer_device, dtype=torch.float32
                )
                mask_tensor = None
                if mask_batch is not None:
                    mask_tensor = torch.from_numpy(np.stack(mask_batch, axis=0)).to(
                        device=infer_device, dtype=torch.float32
                    )

                prediction = predictor.predict(
                    image_tensor,
                    mask=mask_tensor,
                    mask_align_to_image=mask_align_to_image,
                    requested_head=requested_head,
                )
                if prediction.shape[2:] != tuple(int(v) for v in roi_size):
                    raise RuntimeError(
                        "Lazy region inference currently requires model predictions to "
                        "have the same spatial shape as the sliding-window ROI. "
                        f"Got prediction.shape={tuple(prediction.shape)} and roi_size={roi_size}."
                    )

                prediction = prediction.detach().to(device=accumulation_device, dtype=torch.float32)
                if value_accumulator is None:
                    value_accumulator = torch.zeros(
                        (1, int(prediction.shape[1]), *image_size),
                        device=accumulation_device,
                        dtype=torch.float32,
                    )

                for patch_idx, location in enumerate(locations):
                    slices = tuple(
                        slice(location[axis], location[axis] + int(roi_size[axis]))
                        for axis in range(3)
                    )
                    value_accumulator[(slice(None), slice(None), *slices)] += (
                        prediction[patch_idx : patch_idx + 1] * importance_map
                    )
                    weight_accumulator[(slice(None), slice(None), *slices)] += importance_map

            if value_accumulator is None:
                raise RuntimeError(f"No lazy region patches were generated for {image_path}.")

            value_accumulator /= torch.clamp_min(weight_accumulator, 1.0e-6)
            del weight_accumulator
            return value_accumulator.cpu()
        finally:
            if mask_accessor is not None:
                mask_accessor.close()


def lazy_predict_volume(
    cfg,
    forward_fn,
    image_path: str,
    *,
    mask_path: Optional[str] = None,
    mask_align_to_image: bool = False,
    device: torch.device | str = "cpu",
    requested_head: Optional[str] = None,
) -> torch.Tensor:
    """Run lazy sliding-window inference directly from disk-backed volumes."""
    roi_size = resolve_inferer_roi_size(cfg)
    if roi_size is None:
        raise ValueError(
            "Lazy sliding-window inference requires inference.sliding_window.window_size "
            "or model.output_size to be configured."
        )

    if len(roi_size) != 3:
        raise ValueError(
            f"Lazy sliding-window inference currently supports 3D only, got {roi_size}."
        )

    overlap = _coerce_overlap(resolve_inferer_overlap(cfg, roi_size), spatial_dims=3)
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    sw_batch_size = max(
        1,
        int(
            getattr(sliding_cfg, "sw_batch_size", None)
            or getattr(getattr(cfg, "data", None), "dataloader", None).batch_size
        ),
    )
    blend_mode = getattr(sliding_cfg, "blending", "gaussian")
    sigma_scale = float(getattr(sliding_cfg, "sigma_scale", 0.125))
    outer_pad_mode = getattr(sliding_cfg, "padding_mode", "constant")
    outer_pad_value = float(getattr(sliding_cfg, "cval", 0.0))
    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=forward_fn)
    if predictor.is_distributed_sharding_enabled():
        raise RuntimeError(
            "Lazy sliding-window inference does not support "
            "`inference.test_time_augmentation.distributed_sharding`. Disable it first."
        )

    infer_device = torch.device(device)
    accumulation_device = torch.device("cpu")

    with _build_accessor(cfg, image_path, kind="image", mode="test") as image_accessor:
        patch_size_cfg = getattr(cfg.data.dataloader, "patch_size", None)
        if patch_size_cfg and any(
            image_accessor.transformed_spatial_shape[idx] < int(patch_size_cfg[idx])
            for idx in range(3)
        ):
            raise ValueError(
                "Lazy sliding-window inference currently requires the transformed test volume "
                "to be at least as large as data.dataloader.patch_size in every axis. "
                f"Got transformed_shape={image_accessor.transformed_spatial_shape}, "
                f"patch_size={tuple(int(v) for v in patch_size_cfg)}."
            )

        mask_accessor = (
            _build_accessor(cfg, mask_path, kind="mask", mode="test")
            if mask_path is not None
            else None
        )
        try:
            image_size = tuple(int(v) for v in image_accessor.padded_spatial_shape)
            scan_interval = _get_scan_interval(
                image_size, roi_size, num_spatial_dims=3, overlap=overlap
            )
            patch_slices = dense_patch_slices(
                image_size, roi_size, scan_interval, return_slice=True
            )
            importance_map = (
                compute_importance_map(
                    tuple(int(v) for v in roi_size),
                    mode=blend_mode,
                    sigma_scale=overlap if blend_mode == "constant" else sigma_scale,
                    device=accumulation_device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            value_accumulator = None
            weight_accumulator = torch.zeros(
                (1, 1, *image_size), device=accumulation_device, dtype=torch.float32
            )
            num_patch_batches = math.ceil(len(patch_slices) / sw_batch_size)
            progress_bar = None
            if tqdm is not None:
                progress_bar = tqdm(
                    total=num_patch_batches,
                    desc="Lazy sliding-window",
                    leave=True,
                )

            try:
                for batch_start in range(0, len(patch_slices), sw_batch_size):
                    current_slices = patch_slices[batch_start : batch_start + sw_batch_size]
                    image_batch = []
                    mask_batch = [] if mask_accessor is not None else None
                    locations = []

                    for patch_slice in current_slices:
                        location = tuple(int(s.start) for s in patch_slice)
                        image_batch.append(
                            image_accessor.read_patch(
                                location,
                                roi_size,
                                outer_pad_mode=outer_pad_mode,
                                outer_pad_value=outer_pad_value,
                            )
                        )
                        if mask_accessor is not None:
                            mask_batch.append(
                                mask_accessor.read_patch(
                                    location,
                                    roi_size,
                                    outer_pad_mode="constant",
                                    outer_pad_value=0.0,
                                )
                            )
                        locations.append(location)

                    image_tensor = torch.from_numpy(np.stack(image_batch, axis=0)).to(
                        device=infer_device, dtype=torch.float32
                    )
                    mask_tensor = None
                    if mask_batch is not None:
                        mask_tensor = torch.from_numpy(np.stack(mask_batch, axis=0)).to(
                            device=infer_device, dtype=torch.float32
                        )

                    prediction = predictor.predict(
                        image_tensor,
                        mask=mask_tensor,
                        mask_align_to_image=mask_align_to_image,
                        requested_head=requested_head,
                    )
                    if prediction.shape[2:] != tuple(int(v) for v in roi_size):
                        raise RuntimeError(
                            "Lazy sliding-window inference currently requires model predictions to "
                            "have the same spatial shape as the sliding-window ROI. "
                            f"Got prediction.shape={tuple(prediction.shape)} and roi_size={roi_size}."
                        )

                    prediction = prediction.detach().to(
                        device=accumulation_device, dtype=torch.float32
                    )
                    if value_accumulator is None:
                        value_accumulator = torch.zeros(
                            (1, int(prediction.shape[1]), *image_size),
                            device=accumulation_device,
                            dtype=torch.float32,
                        )

                    for patch_idx, location in enumerate(locations):
                        slices = tuple(
                            slice(location[axis], location[axis] + int(roi_size[axis]))
                            for axis in range(3)
                        )
                        value_accumulator[(slice(None), slice(None), *slices)] += (
                            prediction[patch_idx : patch_idx + 1] * importance_map
                        )
                        weight_accumulator[(slice(None), slice(None), *slices)] += importance_map

                    if progress_bar is not None:
                        progress_bar.update(1)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            if value_accumulator is None:
                raise RuntimeError(
                    f"No lazy sliding-window patches were generated for {image_path}."
                )

            value_accumulator /= torch.clamp_min(weight_accumulator, 1.0e-6)
            del weight_accumulator
            return value_accumulator.cpu()
        finally:
            if mask_accessor is not None:
                mask_accessor.close()
