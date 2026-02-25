"""
MONAI transforms for connectomics I/O operations.

This module provides MONAI-compatible transforms for:
- Volume loading (HDF5, TIFF, PNG)
- Volume saving
- Tile-based loading for large datasets
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

from .io import read_volume, save_volume
from .tiles import reconstruct_volume_from_tiles


class LoadVolumed(MapTransform):
    """
    MONAI loader for connectomics volume data (HDF5, TIFF, etc.).

    This transform uses the connectomics read_volume function to load various
    file formats and ensures the data has a channel dimension.

    Args:
        keys: Keys to load from the data dictionary
        transpose_axes: Axis permutation for transposing loaded volumes
            (e.g., [2,1,0] for xyz->zyx). Empty list or None means no transpose.
            Applied BEFORE adding channel dimension.
        allow_missing_keys: Whether to allow missing keys in the dictionary

    Examples:
        >>> transform = LoadVolumed(keys=['image', 'label'])
        >>> data = {'image': 'img.h5', 'label': 'lbl.h5'}
        >>> result = transform(data)
        >>> # result['image'] shape: (C, D, H, W)

        >>> # With transpose (e.g., xyz to zyx)
        >>> transform = LoadVolumed(keys=['image'], transpose_axes=[2,1,0])
        >>> data = {'image': 'img.h5'}  # xyz order
        >>> result = transform(data)
        >>> # result['image'] is now in zyx order
    """

    def __init__(
        self,
        keys: KeysCollection,
        transpose_axes: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.transpose_axes = list(transpose_axes) if transpose_axes else []

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load volume data from file paths."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], str):
                source_path = d[key]
                volume = read_volume(source_path)

                # Apply transpose if specified (before adding channel dimension)
                if self.transpose_axes:
                    if volume.ndim == 3:
                        # 3D volume: transpose spatial dimensions
                        volume = np.transpose(volume, self.transpose_axes)
                    elif volume.ndim == 4:
                        # 4D volume (C, D, H, W): transpose only spatial dimensions
                        # Keep channel dimension first, transpose spatial dims
                        spatial_transpose = [i + 1 for i in self.transpose_axes]
                        volume = np.transpose(volume, [0] + spatial_transpose)

                # Ensure channel dimension exists (add channel if needed)
                # 2D: (H, W) → (1, H, W)
                # 3D: (D, H, W) → (1, D, H, W)
                if volume.ndim == 2:
                    volume = np.expand_dims(volume, axis=0)  # Add channel for 2D
                elif volume.ndim == 3:
                    volume = np.expand_dims(volume, axis=0)  # Add channel for 3D
                d[key] = volume
                meta_key = f"{key}_meta_dict"
                meta_dict = dict(d.get(meta_key, {}))
                meta_dict.update(
                    {
                        "filename_or_obj": source_path,
                        "original_shape": tuple(volume.shape),
                        "spatial_shape": tuple(volume.shape[1:]),
                        "channels_first": True,
                        "transpose_axes": self.transpose_axes if self.transpose_axes else None,
                    }
                )
                d[meta_key] = meta_dict
        return d


class NNUNetPreprocessd(MapTransform):
    """nnU-Net style preprocessing transform for inference/training parity.

    This transform optionally applies:
    - foreground crop
    - spacing-aware resampling
    - optional percentile clipping
    - intensity normalization

    It stores enough metadata in ``image_meta_dict['nnunet_preprocess']`` to
    invert preprocessing before writing outputs.
    """

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str = "image",
        enabled: bool = False,
        crop_to_nonzero: bool = True,
        source_spacing: Optional[Sequence[float]] = None,
        target_spacing: Optional[Sequence[float]] = None,
        normalization: str = "zscore",
        normalization_use_nonzero_mask: bool = True,
        clip_percentile_low: float = 0.0,
        clip_percentile_high: float = 1.0,
        force_separate_z: Optional[bool] = None,
        anisotropy_threshold: float = 3.0,
        image_order: int = 3,
        label_order: int = 0,
        order_z: int = 0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.enabled = enabled
        self.crop_to_nonzero = crop_to_nonzero
        self.source_spacing = list(source_spacing) if source_spacing is not None else None
        self.target_spacing = list(target_spacing) if target_spacing is not None else None
        self.normalization = normalization
        self.normalization_use_nonzero_mask = normalization_use_nonzero_mask
        self.clip_percentile_low = float(clip_percentile_low)
        self.clip_percentile_high = float(clip_percentile_high)
        self.force_separate_z = force_separate_z
        self.anisotropy_threshold = anisotropy_threshold
        self.image_order = image_order
        self.label_order = label_order
        self.order_z = order_z
        self._debug_stats_printed = False  # Print one-line pre/post stats once per transform instance
        if not (0.0 <= self.clip_percentile_low <= 1.0):
            raise ValueError(
                f"clip_percentile_low must be in [0, 1], got {self.clip_percentile_low}"
            )
        if not (0.0 <= self.clip_percentile_high <= 1.0):
            raise ValueError(
                f"clip_percentile_high must be in [0, 1], got {self.clip_percentile_high}"
            )
        if self.clip_percentile_low > self.clip_percentile_high:
            raise ValueError(
                "clip_percentile_low must be <= clip_percentile_high, got "
                f"{self.clip_percentile_low} > {self.clip_percentile_high}"
            )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if not self.enabled or self.image_key not in d:
            return d

        image_np, image_state = self._as_numpy(d[self.image_key])
        spatial_dims = self._infer_spatial_dims(image_np)
        image_spatial_shape = self._get_spatial_shape(image_np, spatial_dims)

        meta_key = f"{self.image_key}_meta_dict"
        meta_dict = dict(d.get(meta_key, {}))

        source_spacing = self._normalize_spacing(
            self.source_spacing or meta_dict.get("spacing") or meta_dict.get("resolution"),
            spatial_dims,
        )
        target_spacing = self._normalize_spacing(self.target_spacing, spatial_dims)

        preprocess_meta = {
            "enabled": True,
            "spatial_dims": spatial_dims,
            "original_spatial_shape": list(image_spatial_shape),
            "source_spacing": source_spacing,
            "target_spacing": target_spacing,
            "normalization": self.normalization,
            "normalization_use_nonzero_mask": self.normalization_use_nonzero_mask,
            "clip_percentile_low": self.clip_percentile_low,
            "clip_percentile_high": self.clip_percentile_high,
            "crop_bbox": None,
            "cropped_spatial_shape": list(image_spatial_shape),
            "resampled_spatial_shape": list(image_spatial_shape),
            "applied_crop": False,
            "applied_resample": False,
            "transpose_axes": meta_dict.get("transpose_axes"),
        }

        bbox = None
        if self.crop_to_nonzero:
            nonzero_mask = self._create_nonzero_mask(image_np, spatial_dims)
            if nonzero_mask is not None and np.any(nonzero_mask):
                bbox = self._bbox_from_mask(nonzero_mask)
                preprocess_meta["crop_bbox"] = [[int(b.start), int(b.stop)] for b in bbox]
                preprocess_meta["applied_crop"] = True
                for key in self.key_iterator(d):
                    if key not in d:
                        continue
                    value_np, state = self._as_numpy(d[key])
                    value_np = self._crop_to_bbox(value_np, bbox, spatial_dims)
                    d[key] = self._from_numpy(value_np, state)
                image_np = self._crop_to_bbox(image_np, bbox, spatial_dims)
                preprocess_meta["cropped_spatial_shape"] = list(
                    self._get_spatial_shape(image_np, spatial_dims)
                )

        target_shape = None
        if source_spacing is not None and target_spacing is not None:
            current_shape = np.array(preprocess_meta["cropped_spatial_shape"], dtype=np.float32)
            spacing_factors = np.array(source_spacing, dtype=np.float32) / np.array(
                target_spacing, dtype=np.float32
            )
            target_shape = np.maximum(np.round(current_shape * spacing_factors).astype(int), 1)
            if np.any(target_shape != current_shape.astype(int)):
                separate_z, lowres_axis = self._resolve_separate_z(
                    source_spacing, target_spacing, spatial_dims
                )
                preprocess_meta["applied_resample"] = True
                preprocess_meta["separate_z"] = separate_z
                preprocess_meta["lowres_axis"] = lowres_axis
                for key in self.key_iterator(d):
                    if key not in d:
                        continue
                    value_np, state = self._as_numpy(d[key])
                    is_seg = key != self.image_key
                    value_np = self._resample_to_shape(
                        value_np,
                        target_shape=tuple(int(v) for v in target_shape),
                        spatial_dims=spatial_dims,
                        is_seg=is_seg,
                        separate_z=separate_z,
                        lowres_axis=lowres_axis,
                    )
                    d[key] = self._from_numpy(value_np, state)
                image_np = self._resample_to_shape(
                    image_np,
                    target_shape=tuple(int(v) for v in target_shape),
                    spatial_dims=spatial_dims,
                    is_seg=False,
                    separate_z=separate_z,
                    lowres_axis=lowres_axis,
                )
                preprocess_meta["resampled_spatial_shape"] = list(
                    self._get_spatial_shape(image_np, spatial_dims)
                )

        pre_stats = None
        if not self._debug_stats_printed:
            pre_stats = self._format_debug_stats(image_np)

        # Optional clipping before normalization (mirrors SmartNormalizeIntensityd order).
        image_np = self._clip_image_percentiles(
            image_np,
            spatial_dims=spatial_dims,
            low=self.clip_percentile_low,
            high=self.clip_percentile_high,
            use_nonzero_mask=self.normalization_use_nonzero_mask,
        )

        # Normalize image after spatial transforms and clipping.
        image_np = self._normalize_image(
            image_np,
            spatial_dims=spatial_dims,
            mode=self.normalization,
            use_nonzero_mask=self.normalization_use_nonzero_mask,
        )
        if not self._debug_stats_printed:
            post_stats = self._format_debug_stats(image_np)
            print(
                "[NNUNetPreprocessd] "
                f"key={self.image_key} mode={self.normalization} "
                f"clip=({self.clip_percentile_low:.3f},{self.clip_percentile_high:.3f}) "
                f"nonzero_mask={self.normalization_use_nonzero_mask} "
                f"pre={pre_stats} post={post_stats}",
                flush=True,
            )
            self._debug_stats_printed = True
        d[self.image_key] = self._from_numpy(image_np, image_state)

        meta_dict["nnunet_preprocess"] = preprocess_meta
        d[meta_key] = meta_dict
        return d

    @staticmethod
    def _infer_spatial_dims(array: np.ndarray) -> int:
        if array.ndim <= 2:
            return array.ndim
        # Treat channel-first arrays as (C, spatial...).
        return array.ndim - 1

    @staticmethod
    def _get_spatial_shape(array: np.ndarray, spatial_dims: int) -> tuple:
        if array.ndim == spatial_dims:
            return tuple(int(v) for v in array.shape)
        return tuple(int(v) for v in array.shape[-spatial_dims:])

    @staticmethod
    def _as_numpy(value: Any) -> tuple[np.ndarray, Dict[str, Any]]:
        if isinstance(value, np.ndarray):
            return value, {"type": "numpy"}
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy(), {
                    "type": "torch",
                    "device": value.device,
                    "dtype": value.dtype,
                }
        except Exception:
            pass
        raise TypeError(f"NNUNetPreprocessd expects numpy arrays or tensors, got {type(value)}")

    @staticmethod
    def _from_numpy(value: np.ndarray, state: Dict[str, Any]) -> Any:
        if state["type"] == "numpy":
            return value
        import torch

        out = torch.from_numpy(value)
        if "dtype" in state:
            out = out.to(dtype=state["dtype"])
        if "device" in state:
            out = out.to(device=state["device"])
        return out

    @staticmethod
    def _normalize_spacing(
        spacing: Optional[Sequence[float]], spatial_dims: int
    ) -> Optional[List[float]]:
        if spacing is None:
            return None
        values = [float(v) for v in spacing]
        if len(values) == spatial_dims:
            return values
        if len(values) > spatial_dims:
            return values[-spatial_dims:]
        return None

    @staticmethod
    def _format_debug_stats(array: np.ndarray) -> str:
        arr = np.asarray(array, dtype=np.float32)
        if arr.size == 0:
            return "empty"
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "all_nonfinite"
        return (
            f"shape={tuple(int(v) for v in arr.shape)} "
            f"min={float(finite.min()):.4f} max={float(finite.max()):.4f} "
            f"mean={float(finite.mean()):.4f} std={float(finite.std()):.4f}"
        )

    @staticmethod
    def _create_nonzero_mask(image: np.ndarray, spatial_dims: int) -> Optional[np.ndarray]:
        if image.ndim == spatial_dims + 1:
            mask = np.any(image != 0, axis=0)
        elif image.ndim == spatial_dims:
            mask = image != 0
        else:
            return None
        if not np.any(mask):
            return None
        from scipy.ndimage import binary_fill_holes

        return binary_fill_holes(mask)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple[slice, ...]:
        coords = np.where(mask)
        return tuple(slice(int(np.min(c)), int(np.max(c)) + 1) for c in coords)

    @staticmethod
    def _crop_to_bbox(array: np.ndarray, bbox: tuple[slice, ...], spatial_dims: int) -> np.ndarray:
        if array.ndim == spatial_dims + 1:
            return array[(slice(None), *bbox)]
        if array.ndim == spatial_dims:
            return array[bbox]
        return array

    def _resolve_separate_z(
        self, source_spacing: Sequence[float], target_spacing: Sequence[float], spatial_dims: int
    ) -> tuple[bool, Optional[int]]:
        if spatial_dims != 3:
            return False, None
        if self.force_separate_z is not None:
            if not self.force_separate_z:
                return False, None
            axis = int(np.argmax(np.asarray(source_spacing)))
            return True, axis
        spacing = np.asarray(source_spacing, dtype=np.float32)
        ratio = float(np.max(spacing) / np.maximum(np.min(spacing), 1e-8))
        if ratio <= self.anisotropy_threshold:
            spacing = np.asarray(target_spacing, dtype=np.float32)
            ratio = float(np.max(spacing) / np.maximum(np.min(spacing), 1e-8))
        if ratio <= self.anisotropy_threshold:
            return False, None
        axis = int(np.argmax(np.asarray(source_spacing)))
        return True, axis

    def _resample_to_shape(
        self,
        array: np.ndarray,
        target_shape: tuple[int, ...],
        spatial_dims: int,
        is_seg: bool,
        separate_z: bool,
        lowres_axis: Optional[int],
    ) -> np.ndarray:
        if self._get_spatial_shape(array, spatial_dims) == target_shape:
            return array
        order = self.label_order if is_seg else self.image_order

        if array.ndim == spatial_dims + 1:
            channels = [
                self._resample_spatial(
                    array[c],
                    target_shape=target_shape,
                    order=order,
                    is_seg=is_seg,
                    separate_z=separate_z,
                    lowres_axis=lowres_axis,
                )[None]
                for c in range(array.shape[0])
            ]
            return np.vstack(channels).astype(array.dtype, copy=False)

        if array.ndim == spatial_dims:
            return self._resample_spatial(
                array,
                target_shape=target_shape,
                order=order,
                is_seg=is_seg,
                separate_z=separate_z,
                lowres_axis=lowres_axis,
            ).astype(array.dtype, copy=False)

        return array

    def _resample_spatial(
        self,
        array: np.ndarray,
        target_shape: tuple[int, ...],
        order: int,
        is_seg: bool,
        separate_z: bool,
        lowres_axis: Optional[int],
    ) -> np.ndarray:
        from scipy.ndimage import zoom

        current_shape = np.array(array.shape, dtype=np.float32)
        target = np.array(target_shape, dtype=np.float32)
        if np.all(current_shape == target):
            return array

        if not separate_z or lowres_axis is None:
            factors = target / current_shape
            return zoom(
                array.astype(np.float32, copy=False),
                zoom=factors,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )

        # nnU-Net style anisotropic resample: in-plane first, then low-res axis.
        axis = int(lowres_axis)
        inplane_axes = [i for i in range(3) if i != axis]
        slices: List[np.ndarray] = []
        for i in range(array.shape[axis]):
            if axis == 0:
                part = array[i, :, :]
            elif axis == 1:
                part = array[:, i, :]
            else:
                part = array[:, :, i]

            plane_factors = [
                target[inplane_axes[0]] / max(part.shape[0], 1),
                target[inplane_axes[1]] / max(part.shape[1], 1),
            ]
            resized = zoom(
                part.astype(np.float32, copy=False),
                zoom=plane_factors,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )
            slices.append(resized)

        stacked = np.stack(slices, axis=axis)
        if stacked.shape[axis] != int(target[axis]):
            axis_factors = np.ones(3, dtype=np.float32)
            axis_factors[axis] = target[axis] / max(stacked.shape[axis], 1)
            stacked = zoom(
                stacked.astype(np.float32, copy=False),
                zoom=axis_factors,
                order=self.order_z if not is_seg else self.label_order,
                mode="nearest",
                prefilter=(self.order_z > 1) and (not is_seg),
            )
        return stacked

    @staticmethod
    def _clip_image_percentiles(
        image: np.ndarray,
        spatial_dims: int,
        low: float,
        high: float,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        if low <= 0.0 and high >= 1.0:
            return image

        image = image.astype(np.float32, copy=False)
        low_q = float(low) * 100.0
        high_q = float(high) * 100.0

        def _apply(volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
            if mask is not None and np.any(mask):
                values = volume[mask]
            else:
                values = volume.reshape(-1)
            if values.size == 0:
                return volume
            low_val = float(np.percentile(values, low_q))
            high_val = float(np.percentile(values, high_q))
            if high_val < low_val:
                low_val, high_val = high_val, low_val
            return np.clip(volume, low_val, high_val)

        if image.ndim == spatial_dims + 1:
            nonzero_mask = np.any(image != 0, axis=0) if use_nonzero_mask else None
            for c in range(image.shape[0]):
                image[c] = _apply(image[c], nonzero_mask)
                if nonzero_mask is not None:
                    image[c][~nonzero_mask] = 0
            return image

        mask = image != 0 if use_nonzero_mask else None
        image = _apply(image, mask)
        if mask is not None:
            image[~mask] = 0
        return image

    @staticmethod
    def _normalize_image(
        image: np.ndarray,
        spatial_dims: int,
        mode: str,
        use_nonzero_mask: bool,
    ) -> np.ndarray:
        if mode in (None, "none"):
            return image

        image = image.astype(np.float32, copy=False)

        def _apply(volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
            mode_lower = str(mode).lower().strip()
            if mask is not None and np.any(mask):
                values = volume[mask]
            else:
                values = volume.reshape(-1)
            if values.size == 0:
                return volume

            if mode_lower in ("zscore", "normal"):
                mean = float(values.mean())
                std = float(values.std())
                if std > 1e-8:
                    volume = (volume - mean) / std
            elif mode_lower == "0-1":
                low = float(values.min())
                high = float(values.max())
                if high > low:
                    volume = (volume - low) / (high - low)
            elif mode_lower.startswith("divide-"):
                scale = float(mode_lower.split("-", 1)[1])
                if abs(scale) > 1e-8:
                    volume = volume / scale
            return volume

        if image.ndim == spatial_dims + 1:
            nonzero_mask = np.any(image != 0, axis=0) if use_nonzero_mask else None
            for c in range(image.shape[0]):
                image[c] = _apply(image[c], nonzero_mask)
                if nonzero_mask is not None:
                    image[c][~nonzero_mask] = 0
            return image

        mask = image != 0 if use_nonzero_mask else None
        image = _apply(image, mask)
        if mask is not None:
            image[~mask] = 0
        return image


class SaveVolumed(MapTransform):
    """
    MONAI transform for saving volume data.

    Args:
        keys: Keys to save from the data dictionary
        output_dir: Output directory for saved volumes
        output_format: File format ('h5' or 'png')
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = SaveVolumed(
        ...     keys=['prediction'],
        ...     output_dir='./outputs',
        ...     output_format='h5'
        ... )
        >>> data = {'prediction': np.random.rand(1, 32, 128, 128)}
        >>> result = transform(data)
        >>> # Saves to ./outputs/prediction.h5
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        output_format: str = "h5",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_dir = output_dir
        self.output_format = output_format

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save volume data to files."""
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], np.ndarray):
                filename = os.path.join(self.output_dir, f"{key}.{self.output_format}")
                save_volume(filename, d[key], file_format=self.output_format)
        return d


class TileLoaderd(MapTransform):
    """
    MONAI transform for loading tile-based data.

    This transform reconstructs volumes from tiles based on chunk coordinates
    and metadata information.

    Args:
        keys: Keys to process from the data dictionary
        allow_missing_keys: Whether to allow missing keys

    Examples:
        >>> transform = TileLoaderd(keys=['image'])
        >>> data = {
        ...     'image': {
        ...         'metadata': tile_metadata,
        ...         'chunk_coords': (0, 10, 0, 128, 0, 128)
        ...     }
        ... }
        >>> result = transform(data)
        >>> # result['image'] is reconstructed volume from tiles
    """

    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load tile data for specified keys."""
        d = dict(data)

        for key in self.key_iterator(d):
            if key in d and isinstance(d[key], dict):
                tile_info = d[key]
                if "metadata" in tile_info and "chunk_coords" in tile_info:
                    metadata = tile_info["metadata"]
                    coords = tile_info["chunk_coords"]
                    volume = self._load_tiles_for_chunk(metadata, coords)
                    d[key] = volume

        return d

    def _load_tiles_for_chunk(
        self,
        metadata: Dict[str, Any],
        coords: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        """Load and reconstruct volume chunk from tiles."""
        z_start, z_end, y_start, y_end, x_start, x_end = coords

        tile_paths = metadata["image"][z_start:z_end]
        volume_coords = [z_start, z_end, y_start, y_end, x_start, x_end]
        tile_coords = [0, metadata["depth"], 0, metadata["height"], 0, metadata["width"]]

        volume = reconstruct_volume_from_tiles(
            tile_paths=tile_paths,
            volume_coords=volume_coords,
            tile_coords=tile_coords,
            tile_size=metadata["tile_size"],
            data_type=np.dtype(metadata["dtype"]),
            tile_start=metadata.get("tile_st", [0, 0]),
            tile_ratio=metadata.get("tile_ratio", 1.0),
        )

        return volume


__all__ = [
    "LoadVolumed",
    "NNUNetPreprocessd",
    "SaveVolumed",
    "TileLoaderd",
]
