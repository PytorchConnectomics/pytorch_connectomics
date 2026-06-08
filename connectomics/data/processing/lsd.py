"""Local Shape Descriptor (LSD) target generation for instance segmentation.

Ported from ``lib/lsd/lsd/train/local_shape_descriptor.py`` (Sheridan et al.,
Nature Methods 2022, ``funkelab/lsd``). The original implementation depends
on ``gunpowder.Coordinate`` / ``gunpowder.Roi``; this port strips those types
so PyTC does not pick up a gunpowder dependency.

LSDs encode per-voxel statistics over a local segment neighborhood (Gaussian-
weighted or spherical). In 3D the descriptor has 10 channels:

    [0, 1, 2]  mean offset in (z, y, x)         (normalized to [0, 1])
    [3, 4, 5]  variance along (z, y, x)         (normalized to [0, 1])
    [6, 7, 8]  Pearson covariance (zy, zx, yx)  (normalized to [0, 1])
    [9]        local segment size               (gaussian-aggregated count)

2D has 6 channels with the same grouping ([0,1] / [2,3] / [4] / [5]).

The ``components`` argument selects a subset by index — e.g. ``"0129"`` returns
just mean offset + size in 3D.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union, cast

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve, find_objects, gaussian_filter

__all__ = ["LsdExtractor", "seg_to_lsd"]

TRUNCATE = 3.0


def seg_to_lsd(
    label: np.ndarray,
    sigma: Union[float, Sequence[float]] = 8.0,
    components: Optional[str] = None,
    voxel_size: Optional[Sequence[int]] = None,
    mode: str = "gaussian",
    downsample: int = 1,
    labels: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Compute Local Shape Descriptors for an instance segmentation.

    Args:
        label: ``(D, H, W)`` for 3D or ``(H, W)`` for 2D, integer labels with
            ``0`` reserved for background.
        sigma: Neighborhood radius in world units (Gaussian std-dev or sphere
            radius). Scalar or per-axis tuple. For connectomics typical values
            are ~80 nm; with isotropic voxel size 1 this becomes ``sigma=80``.
        components: Optional component subset as a digit string (e.g.
            ``"0129"`` for mean-offset + size in 3D). ``None`` = all components.
        voxel_size: Per-axis voxel resolution in world units. Defaults to 1.
        mode: ``"gaussian"`` (default) or ``"sphere"``.
        downsample: Compute on a downsampled volume for speed; ``1`` = none.
        labels: Optional explicit label set; defaults to ``np.unique(label)``.

    Returns:
        ``(C, D, H, W)`` or ``(C, H, W)`` ``float32`` array in [0, 1], where
        ``C = len(components)`` if given, else 10 (3D) or 6 (2D).
    """
    sigma_seq = _coerce_sigma(sigma, label.ndim)
    return LsdExtractor(sigma_seq, mode=mode, downsample=downsample).get_descriptors(
        label, components=components, voxel_size=voxel_size, labels=labels
    )


def _coerce_sigma(sigma: Union[float, Sequence[float]], ndim: int) -> tuple:
    """Broadcast a scalar sigma into per-axis tuple matching ``ndim``."""
    if np.isscalar(sigma):
        return tuple(float(cast(Any, sigma)) for _ in range(ndim))
    sigma_tuple = tuple(float(v) for v in cast(Sequence[float], sigma))
    if len(sigma_tuple) != ndim:
        raise ValueError(f"sigma length {len(sigma_tuple)} does not match label dim {ndim}")
    return sigma_tuple


class LsdExtractor:
    """Compute LSDs with cached coord grids across repeated calls.

    Use this when computing descriptors on many segments with the same shape +
    voxel size — the coordinate meshgrid is reused. For one-off use, call
    :func:`seg_to_lsd` instead.
    """

    def __init__(
        self,
        sigma: Sequence[float],
        mode: str = "gaussian",
        downsample: int = 1,
    ) -> None:
        self.sigma = tuple(float(v) for v in sigma)
        if mode not in ("gaussian", "sphere"):
            raise ValueError(f"mode must be 'gaussian' or 'sphere', got {mode!r}")
        self.mode = mode
        self.downsample = int(downsample)
        self._coords_cache: dict = {}

    def get_descriptors(
        self,
        segmentation: np.ndarray,
        *,
        components: Optional[str] = None,
        voxel_size: Optional[Sequence[int]] = None,
        labels: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        dims = segmentation.ndim
        if dims not in (2, 3):
            raise ValueError(f"segmentation must be 2D or 3D, got {dims}D")

        if dims == 2 and len(self.sigma) == 3:
            # Trim to the 2D sigma if a 3D one was supplied.
            self.sigma = self.sigma[:2]

        voxel_size_t = (
            tuple(1 for _ in range(dims))
            if voxel_size is None
            else tuple(int(v) for v in voxel_size)
        )
        if len(voxel_size_t) != dims:
            raise ValueError(f"voxel_size length {len(voxel_size_t)} != label dim {dims}")

        if labels is None:
            labels_arr = np.unique(segmentation)
        else:
            labels_arr = np.asarray(list(labels))

        if components is None:
            num_channels = 10 if dims == 3 else 6
        else:
            num_channels = len(components)

        descriptors = np.zeros((num_channels,) + segmentation.shape, dtype=np.float32)

        df = self.downsample
        if any(s % df != 0 for s in segmentation.shape):
            raise ValueError(
                f"segmentation shape {segmentation.shape} is not divisible by "
                f"downsample factor {df}"
            )
        sub_voxel_size = tuple(v * df for v in voxel_size_t)
        sub_sigma_voxel = tuple(s / v for s, v in zip(self.sigma, sub_voxel_size))

        if df == 1:
            self._accumulate_bbox(
                descriptors,
                segmentation,
                labels_arr,
                sub_sigma_voxel,
                sub_voxel_size,
                components,
                dims,
            )
        else:
            self._accumulate_full(
                descriptors,
                segmentation,
                labels_arr,
                sub_sigma_voxel,
                sub_voxel_size,
                components,
                df,
                dims,
            )

        # Normalize to [0, 1]: mean offsets and Pearson coefficients have signed
        # ranges that we shift into [0, 1] for prediction.
        if self.mode == "gaussian":
            # Farthest weighted voxel ≈ sigma (3-sigma cap is rarely reached).
            max_distance = np.asarray(self.sigma, dtype=np.float32)
        else:  # sphere
            max_distance = np.asarray([0.5 * s for s in self.sigma], dtype=np.float32)

        seg_mask = (segmentation != 0).astype(np.float32)

        if dims == 3:
            self._normalize_3d(descriptors, max_distance, seg_mask, components)
        else:
            self._normalize_2d(descriptors, max_distance, seg_mask, components)

        np.clip(descriptors, 0.0, 1.0, out=descriptors)
        return descriptors

    def _accumulate_full(
        self,
        descriptors: np.ndarray,
        segmentation: np.ndarray,
        labels_arr: np.ndarray,
        sub_sigma_voxel: tuple,
        sub_voxel_size: tuple,
        components: Optional[str],
        df: int,
        dims: int,
    ) -> None:
        sub_shape = tuple(s // df for s in segmentation.shape)
        coords = self._get_or_build_coords(sub_shape, sub_voxel_size)

        for raw_label in labels_arr:
            label = int(raw_label)
            if label == 0:
                continue

            mask = (segmentation == label).astype(np.float32)
            if df == 1:
                sub_mask = mask
            elif dims == 3:
                sub_mask = mask[::df, ::df, ::df]
            else:
                sub_mask = mask[::df, ::df]

            sub_descriptor = np.concatenate(
                self._get_stats(coords, sub_mask, sub_sigma_voxel, components)
            )
            descriptor = self._upsample(sub_descriptor, df)
            descriptors += descriptor * mask

    def _accumulate_bbox(
        self,
        descriptors: np.ndarray,
        segmentation: np.ndarray,
        labels_arr: np.ndarray,
        sub_sigma_voxel: tuple,
        sub_voxel_size: tuple,
        components: Optional[str],
        dims: int,
    ) -> None:
        present = [int(raw_label) for raw_label in labels_arr if int(raw_label) != 0]
        if not present:
            return

        radius = tuple(int(np.ceil(TRUNCATE * sigma)) for sigma in sub_sigma_voxel)
        max_label = int(segmentation.max())
        use_find_objects = (
            np.issubdtype(segmentation.dtype, np.integer)
            and max_label >= 1
            and max_label <= max(64, 8 * len(present))
        )
        objects = find_objects(segmentation) if use_find_objects else None

        for label in present:
            bbox = None
            if objects is not None:
                if 1 <= label <= len(objects):
                    bbox = objects[label - 1]
                if bbox is None:
                    continue
            else:
                eq = segmentation == label
                if not np.any(eq):
                    continue
                slices: list[slice] = []
                for axis in range(dims):
                    other_axes = tuple(d for d in range(dims) if d != axis)
                    occupied = np.where(eq.any(axis=other_axes))[0]
                    if occupied.size == 0:
                        slices = []
                        break
                    slices.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
                if not slices:
                    continue
                bbox = tuple(slices)

            crop = tuple(
                slice(
                    max(0, bbox[d].start - radius[d]),
                    min(segmentation.shape[d], bbox[d].stop + radius[d]),
                )
                for d in range(dims)
            )
            sub = segmentation[crop]
            mask = (sub == label).astype(np.float32)
            coords_local = self._get_or_build_coords(mask.shape, sub_voxel_size)
            offset = np.asarray(
                [crop[d].start * sub_voxel_size[d] for d in range(dims)],
                dtype=np.float32,
            ).reshape((dims,) + (1,) * dims)
            coords_local = coords_local + offset
            desc = np.concatenate(self._get_stats(coords_local, mask, sub_sigma_voxel, components))
            descriptors[(slice(None),) + crop] += desc * mask[None]

    def _get_or_build_coords(self, sub_shape: tuple, sub_voxel_size: tuple) -> np.ndarray:
        key = (sub_shape, sub_voxel_size)
        cached = self._coords_cache.get(key)
        if cached is not None:
            return cached
        axes = [
            np.arange(0, sub_shape[d] * sub_voxel_size[d], sub_voxel_size[d])
            for d in range(len(sub_shape))
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        coords = np.array(grid, dtype=np.float32)
        self._coords_cache[key] = coords
        return coords

    def _get_stats(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        sigma_voxel: tuple,
        components: Optional[str],
    ):
        # Per-voxel masked coordinates: ``c_i`` if inside the segment, else 0.
        masked_coords = coords * mask

        count = self._aggregate(mask, sigma_voxel)
        count_len = len(count.shape)
        count = np.where(count == 0, 1.0, count)

        # Mean (center-of-mass per voxel) along each axis.
        mean = np.stack([self._aggregate(masked_coords[d], sigma_voxel) for d in range(count_len)])
        mean = mean / count

        need_mean_offset = components is None or any(str(c) in components for c in range(count_len))
        need_cov = components is None or any(
            str(c) in components for c in range(count_len, 4 * count_len - 3)
        )

        if need_mean_offset:
            mean_offset = mean - coords
        else:
            mean_offset = None

        if need_cov:
            coords_outer = self._outer_product(masked_coords)
            entries = [0, 4, 8, 1, 2, 5] if count_len == 3 else [0, 3, 1]
            covariance = np.stack([self._aggregate(coords_outer[d], sigma_voxel) for d in entries])
            covariance = covariance / count
            covariance -= self._outer_product(mean)[entries]

            if count_len == 3:
                variance = covariance[[0, 1, 2]]
                pearson = covariance[[3, 4, 5]]
            else:
                variance = covariance[[0, 1]]
                pearson = covariance[[2]]

            # Numerical-stability floor on variance before normalizing Pearson.
            variance = np.where(variance < 1e-3, 1e-3, variance)
            if count_len == 3:
                pearson[0] /= np.sqrt(variance[0] * variance[1])
                pearson[1] /= np.sqrt(variance[0] * variance[2])
                pearson[2] /= np.sqrt(variance[1] * variance[2])
                for d in range(3):
                    variance[d] /= self.sigma[d] ** 2
            else:
                pearson /= np.sqrt(variance[0] * variance[1])
                for d in range(2):
                    variance[d] /= self.sigma[d] ** 2
        else:
            variance = None
            pearson = None

        if components is None:
            # Full descriptor: (mean_offset, variance, pearson, size).
            return (mean_offset, variance, pearson, count[None, :])

        ret = []
        for token in components:
            i = int(token)
            if count_len == 3:
                if 0 <= i < 3:
                    ret.append(mean_offset[[i]])
                elif 3 <= i < 6:
                    ret.append(variance[[i - 3]])
                elif 6 <= i < 9:
                    ret.append(pearson[[i - 6]])
                elif i == 9:
                    ret.append(count[None, :])
                else:
                    raise ValueError(f"3D LSD components must be in 0..9, got {i}")
            else:  # 2D
                if 0 <= i < 2:
                    ret.append(mean_offset[[i]])
                elif 2 <= i < 4:
                    ret.append(variance[[i - 2]])
                elif i == 4:
                    ret.append(pearson)
                elif i == 5:
                    ret.append(count[None, :])
                else:
                    raise ValueError(f"2D LSD components must be in 0..5, got {i}")
        return tuple(ret)

    def _aggregate(self, array: np.ndarray, sigma: tuple) -> np.ndarray:
        if self.mode == "gaussian":
            return gaussian_filter(array, sigma=sigma, mode="constant", cval=0.0, truncate=TRUNCATE)
        radius = sigma[0]
        if any(s != radius for s in sigma):
            raise ValueError("mode='sphere' requires isotropic sigma")
        sphere = self._make_sphere(int(radius))
        return convolve(array, sphere, mode="constant", cval=0.0)

    @staticmethod
    def _make_sphere(radius: int) -> np.ndarray:
        r2: np.ndarray = np.arange(-radius, radius) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        return (dist2 <= radius**2).astype(np.float32)

    @staticmethod
    def _outer_product(array: np.ndarray) -> np.ndarray:
        """Per-voxel outer product of the channel axis (k×k flattened)."""
        k = array.shape[0]
        outer = np.einsum("i...,j...->ij...", array, array)
        return outer.reshape((k * k,) + array.shape[1:])

    @staticmethod
    def _upsample(array: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return array
        shape = array.shape
        stride = array.strides
        sh: tuple[int, ...]
        st: tuple[int, ...]
        if array.ndim == 4:
            sh = (shape[0], shape[1], factor, shape[2], factor, shape[3], factor)
            st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
        else:
            sh = (shape[0], shape[1], factor, shape[2], factor)
            st = (stride[0], stride[1], 0, stride[2], 0)
        view = as_strided(array, sh, st)
        out_shape = [shape[0]] + [shape[i + 1] * factor for i in range(array.ndim - 1)]
        return view.reshape(out_shape)

    @staticmethod
    def _normalize_3d(
        descriptors: np.ndarray,
        max_distance: np.ndarray,
        seg_mask: np.ndarray,
        components: Optional[str],
    ) -> None:
        if components is None:
            descriptors[[0, 1, 2]] = (
                descriptors[[0, 1, 2]] / max_distance[:, None, None, None] * 0.5 + 0.5
            )
            descriptors[[6, 7, 8]] = descriptors[[6, 7, 8]] * 0.5 + 0.5
            descriptors[[0, 1, 2, 6, 7, 8]] *= seg_mask
            return
        for slot, token in enumerate(components):
            c = int(token)
            if 0 <= c < 3:
                descriptors[slot] = (descriptors[slot] / max_distance[c] * 0.5 + 0.5) * seg_mask
            elif 6 <= c < 9:
                descriptors[slot] = (descriptors[slot] * 0.5 + 0.5) * seg_mask

    @staticmethod
    def _normalize_2d(
        descriptors: np.ndarray,
        max_distance: np.ndarray,
        seg_mask: np.ndarray,
        components: Optional[str],
    ) -> None:
        if components is None:
            descriptors[[0, 1]] = descriptors[[0, 1]] / max_distance[:, None, None] * 0.5 + 0.5
            descriptors[[4]] = descriptors[[4]] * 0.5 + 0.5
            descriptors[[0, 1, 4]] *= seg_mask
            return
        for slot, token in enumerate(components):
            c = int(token)
            if 0 <= c < 2:
                descriptors[slot] = (descriptors[slot] / max_distance[c] * 0.5 + 0.5) * seg_mask
            elif c == 4:
                descriptors[slot] = (descriptors[slot] * 0.5 + 0.5) * seg_mask
