"""
Lazy zarr-backed volume dataset for random patch sampling.

This dataset keeps zarr array handles open and reads only requested crops
per sample, avoiding full-volume preload into RAM.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from monai.data import Dataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep


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


class LazyZarrVolumeDataset(Dataset):
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

        self.image_paths = image_paths
        self.label_paths = label_paths if label_paths else [None] * len(image_paths)
        self.mask_paths = mask_paths if mask_paths else [None] * len(image_paths)
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.iter_num = iter_num if iter_num > 0 else len(image_paths)
        self.transforms = transforms
        self.mode = mode
        self.max_attempts = max_attempts
        self.foreground_threshold = foreground_threshold
        self.transpose_axes = self._normalize_transpose_axes(transpose_axes)
        self.inverse_transpose_axes = self._invert_transpose_axes(self.transpose_axes)

        self.base_seed = 0
        self.current_epoch = 0

        self.images = []
        self.labels = []
        self.masks = []
        self.image_channel_last = []
        self.volume_sizes = []

        print(f"  Opening {len(image_paths)} zarr volumes (lazy mode, no preload)...")
        for i, (img_path, lbl_path, mask_path) in enumerate(
            zip(self.image_paths, self.label_paths, self.mask_paths)
        ):
            img_arr = self._open_array(img_path)
            lbl_arr = self._open_array(lbl_path) if lbl_path else None
            mask_arr = self._open_array(mask_path) if mask_path else None

            img_channel_last = False
            if img_arr.ndim == 4:
                img_channel_last = _is_channel_last_4d(tuple(img_arr.shape))
                spatial_shape_raw = tuple(img_arr.shape[:3] if img_channel_last else img_arr.shape[1:])
            elif img_arr.ndim == 3:
                spatial_shape_raw = tuple(img_arr.shape)
            else:
                raise ValueError(f"Unsupported image ndim={img_arr.ndim} for {img_path}")
            spatial_shape = self._transpose_shape(spatial_shape_raw)

            if lbl_arr is not None:
                lbl_shape_raw = self._get_label_spatial_shape(lbl_arr)
                if lbl_shape_raw != spatial_shape_raw:
                    raise ValueError(
                        f"Image/label spatial mismatch for {img_path} vs {lbl_path}: "
                        f"{spatial_shape_raw} vs {lbl_shape_raw}"
                    )

            if mask_arr is not None:
                mask_shape_raw = self._get_label_spatial_shape(mask_arr)
                if mask_shape_raw != spatial_shape_raw:
                    raise ValueError(
                        f"Image/mask spatial mismatch for {img_path} vs {mask_path}: "
                        f"{spatial_shape_raw} vs {mask_shape_raw}"
                    )

            self.images.append(img_arr)
            self.labels.append(lbl_arr)
            self.masks.append(mask_arr)
            self.image_channel_last.append(img_channel_last)
            self.volume_sizes.append(spatial_shape)

            print(
                f"    Volume {i + 1}/{len(image_paths)}: "
                f"image={img_arr.shape}, label={None if lbl_arr is None else lbl_arr.shape}, "
                f"spatial(raw->model)={spatial_shape_raw}->{spatial_shape}"
            )

    def _open_array(self, path: Optional[str]):
        if path is None:
            return None
        if ".zarr" not in str(path):
            raise ValueError(
                f"LazyZarrVolumeDataset expects zarr paths (got: {path}). "
                "Use standard dataset path for non-zarr inputs."
            )
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

    @staticmethod
    def _normalize_transpose_axes(
        transpose_axes: Optional[Sequence[int]],
    ) -> List[int]:
        if transpose_axes is None:
            return []
        axes = [int(a) for a in transpose_axes]
        if not axes:
            return []
        if len(axes) != 3 or sorted(axes) != [0, 1, 2]:
            raise ValueError(
                "transpose_axes must be a permutation of [0, 1, 2], "
                f"got {transpose_axes}"
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
            return tuple(
                slice(pos[i], pos[i] + self.patch_size[i]) for i in range(3)
            )
        raw_slices = []
        for raw_axis in range(3):
            logical_axis = self.inverse_transpose_axes[raw_axis]
            start = pos[logical_axis]
            size = self.patch_size[logical_axis]
            raw_slices.append(slice(start, start + size))
        return tuple(raw_slices)

    def __len__(self) -> int:
        return self.iter_num

    def set_epoch(self, epoch: int, base_seed: int = 0):
        if self.mode == "val":
            self.base_seed = base_seed
            self.current_epoch = epoch
            effective_seed = self.base_seed + epoch
            random.seed(effective_seed)
            print(
                f"[Validation] Set epoch={epoch}, base_seed={base_seed}, effective_seed={effective_seed}"
            )
            print(
                f"[Validation] Dataset: {type(self).__name__}@{id(self)}, mode={self.mode}, iter_num={self.iter_num}"
            )

    def get_sampling_fingerprint(self, num_samples: int = 5) -> str:
        if self.mode != "val":
            return "N/A (training mode)"
        state = random.getstate()
        try:
            samples = []
            for _ in range(num_samples):
                vol_idx = random.randint(0, len(self.images) - 1)
                pos = self._get_random_crop_position(vol_idx)
                samples.append((vol_idx, pos))
            return ", ".join([f"v{v}@{p}" for v, p in samples])
        finally:
            random.setstate(state)

    def _get_random_crop_position(self, vol_idx: int) -> Tuple[int, int, int]:
        vol_size = self.volume_sizes[vol_idx]
        starts = []
        for dim in range(3):
            max_start = max(0, vol_size[dim] - self.patch_size[dim])
            starts.append(random.randint(0, max_start))
        return tuple(starts)

    def _get_center_crop_position(self, vol_idx: int) -> Tuple[int, int, int]:
        vol_size = self.volume_sizes[vol_idx]
        return tuple(max(0, (vol_size[d] - self.patch_size[d]) // 2) for d in range(3))

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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        vol_idx = random.randint(0, len(self.images) - 1)
        use_fg_sampling = (
            self.mode == "train"
            and self.labels[vol_idx] is not None
            and self.foreground_threshold > 0
        )

        attempts = self.max_attempts if use_fg_sampling else 1
        chosen_pos = None
        label_crop = None
        for _ in range(attempts):
            pos = (
                self._get_random_crop_position(vol_idx)
                if self.mode == "train"
                else self._get_center_crop_position(vol_idx)
            )
            if not use_fg_sampling:
                chosen_pos = pos
                break

            candidate = self._crop_label_like(self.labels[vol_idx], pos)
            fg_ratio = float((candidate > 0).sum()) / float(candidate.size)
            if fg_ratio >= self.foreground_threshold:
                chosen_pos = pos
                label_crop = candidate
                break

        if chosen_pos is None:
            chosen_pos = (
                self._get_random_crop_position(vol_idx)
                if self.mode == "train"
                else self._get_center_crop_position(vol_idx)
            )

        image_crop = self._crop_image(vol_idx, chosen_pos)
        if label_crop is None:
            if self.labels[vol_idx] is not None:
                label_crop = self._crop_label_like(self.labels[vol_idx], chosen_pos)
            else:
                label_crop = np.zeros((1, *self.patch_size), dtype=np.uint8)

        if self.masks[vol_idx] is not None:
            mask_crop = self._crop_label_like(self.masks[vol_idx], chosen_pos)
        else:
            mask_crop = np.zeros((1, *self.patch_size), dtype=np.uint8)

        data = {"image": image_crop, "label": label_crop, "mask": mask_crop}
        if self.transforms:
            data = self.transforms(data)
        return data


__all__ = ["LazyZarrVolumeDataset"]
