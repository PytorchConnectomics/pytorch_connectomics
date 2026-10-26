"""Fast CPU resampling for MONAI affine augmentation.

``NumbaAffineResample`` is a drop-in ``monai.transforms.Resample`` whose 3D
reflection-padded nearest/bilinear path runs as single-pass numba kernels in float32.
The kernels mirror torch's CPU ``grid_sampler_3d`` (``align_corners=False``): unnormalize,
reflect, clip, then round-half-even (nearest) or the 8-corner trilinear blend in torch's
corner and accumulation order, so outputs equal MONAI's ``Resample(dtype=float32)``.
Source coordinates are computed once per grid and reused by every key of one
``RandAffined`` call. Ineligible inputs fall back to MONAI's implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.config import USE_COMPILED
from monai.transforms import Resample
from monai.utils import convert_to_dst_type, convert_to_tensor

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


_HALF = np.float32(0.5)
_ONE = np.float32(1.0)
_TWO = np.float32(2.0)
_ZERO = np.float32(0.0)


@njit(cache=True, inline="always")
def _source_index(g, scale, size):
    # MONAI scales the centred grid by 2/max(2, size); torch then unnormalizes it.
    x = g * scale
    x = ((x + _ONE) * size - _ONE) / _TWO
    # torch reflect_coordinates(x, twice_low=-1, twice_high=2*size-1), then clip.
    c = abs(x + _HALF)
    extra = np.fmod(c, size)
    flips = np.floor(c / size)
    if int(flips) % 2 == 0:
        x = extra - _HALF
    else:
        x = size - extra - _HALF
    if x < _ZERO:
        return _ZERO
    if x > size - _ONE:
        return size - _ONE
    return x


@njit(cache=True)
def _source_coords(grid, scale_d, scale_h, scale_w, depth, height, width):
    out = np.empty((3,) + grid.shape[1:], dtype=np.float32)
    for d in range(grid.shape[1]):
        for h in range(grid.shape[2]):
            for w in range(grid.shape[3]):
                out[0, d, h, w] = _source_index(grid[0, d, h, w], scale_d, depth)
                out[1, d, h, w] = _source_index(grid[1, d, h, w], scale_h, height)
                out[2, d, h, w] = _source_index(grid[2, d, h, w], scale_w, width)
    return out


@njit(cache=True)
def _sample_nearest(src, coords):
    out = np.empty((src.shape[0],) + coords.shape[1:], dtype=src.dtype)
    for d in range(coords.shape[1]):
        for h in range(coords.shape[2]):
            for w in range(coords.shape[3]):
                iz = int(np.rint(coords[0, d, h, w]))
                iy = int(np.rint(coords[1, d, h, w]))
                ix = int(np.rint(coords[2, d, h, w]))
                for c in range(src.shape[0]):
                    out[c, d, h, w] = src[c, iz, iy, ix]
    return out


@njit(cache=True)
def _sample_trilinear(src, coords):
    depth, height, width = src.shape[1], src.shape[2], src.shape[3]
    out = np.empty((src.shape[0],) + coords.shape[1:], dtype=np.float32)
    for d in range(coords.shape[1]):
        for h in range(coords.shape[2]):
            for w in range(coords.shape[3]):
                iz = coords[0, d, h, w]
                iy = coords[1, d, h, w]
                ix = coords[2, d, h, w]
                x0, y0, z0 = np.floor(ix), np.floor(iy), np.floor(iz)
                x1, y1, z1 = x0 + _ONE, y0 + _ONE, z0 + _ONE
                tnw = (x1 - ix) * (y1 - iy) * (z1 - iz)
                tne = (ix - x0) * (y1 - iy) * (z1 - iz)
                tsw = (x1 - ix) * (iy - y0) * (z1 - iz)
                tse = (ix - x0) * (iy - y0) * (z1 - iz)
                bnw = (x1 - ix) * (y1 - iy) * (iz - z0)
                bne = (ix - x0) * (y1 - iy) * (iz - z0)
                bsw = (x1 - ix) * (iy - y0) * (iz - z0)
                bse = (ix - x0) * (iy - y0) * (iz - z0)
                xi, yi, zi = int(x0), int(y0), int(z0)
                in_x, in_y, in_z = xi + 1 < width, yi + 1 < height, zi + 1 < depth
                for c in range(src.shape[0]):
                    acc = _ZERO
                    acc += src[c, zi, yi, xi] * tnw
                    if in_x:
                        acc += src[c, zi, yi, xi + 1] * tne
                    if in_y:
                        acc += src[c, zi, yi + 1, xi] * tsw
                    if in_x and in_y:
                        acc += src[c, zi, yi + 1, xi + 1] * tse
                    if in_z:
                        acc += src[c, zi + 1, yi, xi] * bnw
                        if in_x:
                            acc += src[c, zi + 1, yi, xi + 1] * bne
                        if in_y:
                            acc += src[c, zi + 1, yi + 1, xi] * bsw
                        if in_x and in_y:
                            acc += src[c, zi + 1, yi + 1, xi + 1] * bse
                    out[c, d, h, w] = acc
    return out


class NumbaAffineResample(Resample):
    """MONAI ``Resample`` with a numba fast path for 3D reflection nearest/bilinear."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Holding the grid object (not its address) keeps the identity check safe: its
        # storage cannot be reused by a later grid while this reference is alive.
        self._cached_grid = None
        self._cached_key = None
        self._cached_coords = None

    def _is_fast_path(self, img, grid, mode, padding_mode, align_corners) -> bool:
        mode = self.mode if mode is None else mode
        padding_mode = self.padding_mode if padding_mode is None else padding_mode
        align_corners = self.align_corners if align_corners is None else align_corners
        return (
            NUMBA_AVAILABLE
            and not USE_COMPILED
            and self.norm_coords
            and str(mode) in ("nearest", "bilinear")
            and str(padding_mode) == "reflection"
            and align_corners is False
            and isinstance(grid, torch.Tensor)
            and grid.ndim == 4
            and grid.shape[0] == 4
            and grid.device.type == "cpu"
            and not grid.requires_grad
            and isinstance(img, torch.Tensor)
            and img.ndim == 4
            and img.device.type == "cpu"
            and not img.requires_grad
            and all(s >= 1 for s in img.shape[1:])
            and img.dtype
            in (
                torch.float32,
                torch.float64,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
            )
        )

    def _coords(self, grid: torch.Tensor, spatial_shape) -> np.ndarray:
        key = (grid._version, tuple(spatial_shape))
        if grid is not self._cached_grid or key != self._cached_key:
            depth, height, width = (np.float32(s) for s in spatial_shape)
            scales = [np.float32(2.0 / max(2, int(s))) for s in spatial_shape]
            centred = np.ascontiguousarray(grid[:3].detach().to(torch.float32).numpy())
            self._cached_coords = _source_coords(centred, *scales, depth, height, width)
            self._cached_grid, self._cached_key = grid, key
        return self._cached_coords

    def __call__(
        self, img, grid=None, mode=None, padding_mode=None, dtype=None, align_corners=None
    ):
        if (
            (dtype if dtype is not None else self.dtype) not in (torch.float32, np.float32)
            or grid is None
            or not self._is_fast_path(img, grid, mode, padding_mode, align_corners)
        ):
            return super().__call__(img, grid, mode, padding_mode, dtype, align_corners)
        mode = self.mode if mode is None else mode
        src = convert_to_tensor(img, track_meta=False)
        coords = self._coords(grid, src.shape[1:])
        if str(mode) == "nearest":
            if src.dtype == torch.bool:
                src = src.to(torch.uint8)
            out = _sample_nearest(np.ascontiguousarray(src.numpy()), coords).astype(np.float32)
        else:
            out = _sample_trilinear(np.ascontiguousarray(src.to(torch.float32).numpy()), coords)
        out_val, *_ = convert_to_dst_type(torch.from_numpy(out), dst=img, dtype=np.float32)
        return out_val
