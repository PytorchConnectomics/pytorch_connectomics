"""
Lazy zarr-backed volume dataset for random patch sampling.

Keeps zarr array handles open and reads only requested crops per sample,
avoiding full-volume preload into RAM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from monai.transforms import Compose

from .base import PatchDataset

logger = logging.getLogger(__name__)


def _require_zarr():
    try:
        import zarr  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Lazy zarr dataset requires `zarr`. Install with: pip install zarr"
        ) from exc
    return zarr


def _is_channel_last_4d(shape: Tuple[int, ...]) -> bool:
    """Heuristic for 4D arrays: treat trailing small dim as channel."""
    return shape[-1] <= 8 and shape[0] > 8 and shape[1] > 8 and shape[2] > 8


class LazyZarrVolumeDataset(PatchDataset):
    """
    Lazy zarr dataset that samples random crops directly from zarr stores.

    Notes:
    - Input image arrays may be 3D or 4D (channel-last or channel-first).
    - Label/mask arrays are expected to be 3D (or 4D with singleton channel).
    - Output is channel-first: image/label/mask shapes are [C, D, H, W].
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        label_aux_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        patch_size: Tuple[int, int, int] = (112, 112, 112),
        iter_num: int = 500,
        transforms: Optional[Compose] = None,
        mode: str = "train",
        max_attempts: int = 10,
        foreground_threshold: float = 0.0,
        transpose_axes: Optional[Sequence[int]] = None,
    ):
        self.zarr = _require_zarr()

        super().__init__(
            patch_size=patch_size,
            iter_num=iter_num if iter_num > 0 else len(image_paths),
            transforms=transforms,
            mode=mode,
            max_attempts=max_attempts,
            foreground_threshold=foreground_threshold,
        )

        self.image_paths = image_paths
        self.label_paths = label_paths if label_paths else [None] * len(image_paths)
        self.label_aux_paths = label_aux_paths if label_aux_paths else [None] * len(image_paths)
        self.mask_paths = mask_paths if mask_paths else [None] * len(image_paths)
        self.transpose_axes = self._normalize_transpose_axes(transpose_axes)
        self.inverse_transpose_axes = self._invert_transpose_axes(self.transpose_axes)

        self.images = []
        self.labels = []
        self.label_auxs = []
        self.masks = []
        self.image_channel_last = []

        logger.info("Opening %d zarr volumes (lazy, no preload)...", len(image_paths))
        for i, (img_path, lbl_path, aux_path, mask_path) in enumerate(
            zip(self.image_paths, self.label_paths, self.label_aux_paths, self.mask_paths)
        ):
            img_arr = self._open_array(img_path)
            lbl_arr = self._open_array(lbl_path) if lbl_path else None
            aux_arr = self._open_array(aux_path) if aux_path else None
            mask_arr = self._open_array(mask_path) if mask_path else None

            img_channel_last = False
            if img_arr.ndim == 4:
                img_channel_last = _is_channel_last_4d(tuple(img_arr.shape))
                spatial_raw = tuple(img_arr.shape[:3] if img_channel_last else img_arr.shape[1:])
            elif img_arr.ndim == 3:
                spatial_raw = tuple(img_arr.shape)
            else:
                raise ValueError(f"Unsupported image ndim={img_arr.ndim} for {img_path}")

            spatial = self._transpose_shape(spatial_raw)

            if lbl_arr is not None:
                lbl_raw = self._get_label_spatial_shape(lbl_arr)
                if lbl_raw != spatial_raw:
                    raise ValueError(f"Image/label spatial mismatch: {spatial_raw} vs {lbl_raw}")
            if aux_arr is not None:
                aux_raw = self._get_label_spatial_shape(aux_arr)
                if aux_raw != spatial_raw:
                    raise ValueError(f"Image/label_aux spatial mismatch: {spatial_raw} vs {aux_raw}")
            if mask_arr is not None:
                mask_raw = self._get_label_spatial_shape(mask_arr)
                if mask_raw != spatial_raw:
                    raise ValueError(f"Image/mask spatial mismatch: {spatial_raw} vs {mask_raw}")

            self.images.append(img_arr)
            self.labels.append(lbl_arr)
            self.label_auxs.append(aux_arr)
            self.masks.append(mask_arr)
            self.image_channel_last.append(img_channel_last)
            self.volume_sizes.append(spatial)

            logger.info(
                "    Volume %s/%s: image=%s, spatial=%s->%s",
                i + 1,
                len(image_paths),
                img_arr.shape,
                spatial_raw,
                spatial,
            )

    # -- PatchDataset abstract methods --

    def _crop_volumes(self, vol_idx: int, pos: Tuple[int, ...]) -> Dict[str, Any]:
        image_crop = self._crop_image(vol_idx, pos)
        label_crop = (
            self._crop_label_like(self.labels[vol_idx], pos)
            if self.labels[vol_idx] is not None
            else None
        )
        label_aux_crop = (
            self._crop_label_like(self.label_auxs[vol_idx], pos)
            if self.label_auxs[vol_idx] is not None
            else None
        )
        mask_crop = (
            self._crop_label_like(self.masks[vol_idx], pos)
            if self.masks[vol_idx] is not None
            else None
        )
        return {
            "image": image_crop,
            "label": label_crop,
            "label_aux": label_aux_crop,
            "mask": mask_crop,
        }

    def _has_labels(self, vol_idx: int) -> bool:
        return self.labels[vol_idx] is not None

    # -- Zarr I/O helpers --

    def _open_array(self, path: Optional[str]):
        if path is None:
            return None
        if ".zarr" not in str(path):
            # Non-zarr files (e.g. precomputed skeleton .h5): load eagerly.
            from ..io.io import read_volume
            return read_volume(str(path))
        return self.zarr.open(str(path), mode="r")

    @staticmethod
    def _get_label_spatial_shape(arr) -> Tuple[int, int, int]:
        if arr.ndim == 3:
            return tuple(arr.shape)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            return tuple(arr.shape[:3])
        if arr.ndim == 4 and arr.shape[0] == 1:
            return tuple(arr.shape[1:])
        raise ValueError(f"Unsupported label/mask shape: {arr.shape}")

    # -- Transpose helpers --

    @staticmethod
    def _normalize_transpose_axes(transpose_axes: Optional[Sequence[int]]) -> List[int]:
        if transpose_axes is None:
            return []
        axes = [int(a) for a in transpose_axes]
        if not axes:
            return []
        if len(axes) != 3 or sorted(axes) != [0, 1, 2]:
            raise ValueError(
                f"transpose_axes must be a permutation of [0,1,2], got {transpose_axes}"
            )
        return axes

    @staticmethod
    def _invert_transpose_axes(transpose_axes: List[int]) -> List[int]:
        if not transpose_axes:
            return []
        inverse = [0, 0, 0]
        for out_axis, in_axis in enumerate(transpose_axes):
            inverse[in_axis] = out_axis
        return inverse

    def _transpose_shape(self, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if not self.transpose_axes:
            return shape
        return tuple(shape[i] for i in self.transpose_axes)

    def _transpose_spatial_array(self, arr: np.ndarray) -> np.ndarray:
        if not self.transpose_axes:
            return arr
        return np.transpose(arr, self.transpose_axes)

    def _logical_to_raw_slices(self, pos: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
        if not self.transpose_axes:
            return tuple(slice(pos[i], pos[i] + self.patch_size[i]) for i in range(3))
        raw_slices = []
        for raw_axis in range(3):
            logical_axis = self.inverse_transpose_axes[raw_axis]
            start = pos[logical_axis]
            size = self.patch_size[logical_axis]
            raw_slices.append(slice(start, start + size))
        return tuple(raw_slices)

    # -- Crop methods --

    def _crop_image(self, vol_idx: int, pos: Tuple[int, int, int]) -> np.ndarray:
        s0, s1, s2 = self._logical_to_raw_slices(pos)
        img_arr = self.images[vol_idx]

        if img_arr.ndim == 3:
            crop = np.asarray(img_arr[s0, s1, s2])
            crop = self._transpose_spatial_array(crop)
            return crop[None, ...]

        if self.image_channel_last[vol_idx]:
            crop = np.asarray(img_arr[s0, s1, s2, :])
            if self.transpose_axes:
                crop = np.transpose(crop, [*self.transpose_axes, 3])
            return np.moveaxis(crop, -1, 0)

        crop = np.asarray(img_arr[:, s0, s1, s2])
        if self.transpose_axes:
            spatial_transpose = [a + 1 for a in self.transpose_axes]
            crop = np.transpose(crop, [0, *spatial_transpose])
        return crop

    def _crop_label_like(self, arr, pos: Tuple[int, int, int]) -> np.ndarray:
        s0, s1, s2 = self._logical_to_raw_slices(pos)

        if arr.ndim == 3:
            crop = np.asarray(arr[s0, s1, s2])
            crop = self._transpose_spatial_array(crop)
            return crop[None, ...]

        if arr.ndim == 4 and arr.shape[-1] == 1:
            crop = np.asarray(arr[s0, s1, s2, 0])
            crop = self._transpose_spatial_array(crop)
            return crop[None, ...]

        if arr.ndim == 4 and arr.shape[0] == 1:
            crop = np.asarray(arr[:, s0, s1, s2])
            if self.transpose_axes:
                spatial_transpose = [a + 1 for a in self.transpose_axes]
                crop = np.transpose(crop, [0, *spatial_transpose])
            return crop

        raise ValueError(f"Unsupported label/mask shape: {arr.shape}")


__all__ = ["LazyZarrVolumeDataset"]
