"""nnU-Net style preprocessing transform.

This transform optionally applies:
- foreground crop
- spacing-aware resampling
- optional percentile clipping
- intensity normalization

It stores metadata in ``image_meta_dict['nnunet_preprocess']``
to allow inverting preprocessing before writing outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


def _infer_spatial_dims_from_array(array: np.ndarray) -> int:
    if array.ndim <= 2:
        return array.ndim
    if array.ndim == 3:
        return 3
    return array.ndim - 1


def _spatial_shape(array: np.ndarray, spatial_dims: int) -> tuple:
    if array.ndim == spatial_dims:
        return tuple(int(v) for v in array.shape)
    return tuple(int(v) for v in array.shape[-spatial_dims:])


def _resample_array_to_shape(
    array: np.ndarray,
    target_shape: Sequence[int],
    spatial_dims: int,
    order: int,
) -> np.ndarray:
    from scipy.ndimage import zoom

    target = tuple(int(v) for v in target_shape)
    if _spatial_shape(array, spatial_dims) == target:
        return array

    def _zoom_single(vol: np.ndarray) -> np.ndarray:
        factors = np.array(target, dtype=np.float32) / np.maximum(
            np.array(vol.shape, dtype=np.float32), 1.0
        )
        return zoom(
            vol.astype(np.float32, copy=False),
            zoom=factors,
            order=order,
            mode="nearest",
            prefilter=order > 1,
        )

    if array.ndim == spatial_dims + 1:
        channels = [_zoom_single(array[c])[None] for c in range(array.shape[0])]
        return np.vstack(channels).astype(array.dtype, copy=False)
    if array.ndim == spatial_dims:
        return _zoom_single(array).astype(array.dtype, copy=False)
    return array


def _fit_array_to_shape(
    array: np.ndarray, target_shape: Sequence[int], spatial_dims: int
) -> np.ndarray:
    target = tuple(int(v) for v in target_shape)
    if _spatial_shape(array, spatial_dims) == target:
        return array

    if array.ndim == spatial_dims + 1:
        out = np.zeros((array.shape[0], *target), dtype=array.dtype)
        in_shape = array.shape[1:]
        write_shape = tuple(min(int(in_shape[d]), target[d]) for d in range(spatial_dims))
        out_slices = (slice(None),) + tuple(slice(0, w) for w in write_shape)
        in_slices = (slice(None),) + tuple(slice(0, w) for w in write_shape)
        out[out_slices] = array[in_slices]
        return out

    if array.ndim == spatial_dims:
        out = np.zeros(target, dtype=array.dtype)
        in_shape = array.shape
        write_shape = tuple(min(int(in_shape[d]), target[d]) for d in range(spatial_dims))
        out_slices = tuple(slice(0, w) for w in write_shape)
        in_slices = tuple(slice(0, w) for w in write_shape)
        out[out_slices] = array[in_slices]
        return out

    return array


def restore_prediction_to_input_space(sample: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Invert nnU-Net preprocessing metadata for one prediction sample."""
    preprocess_meta = meta.get("nnunet_preprocess")
    if not isinstance(preprocess_meta, dict) or not preprocess_meta.get("enabled", False):
        return sample

    array = sample
    spatial_dims = int(preprocess_meta.get("spatial_dims", _infer_spatial_dims_from_array(array)))
    is_integer = np.issubdtype(array.dtype, np.integer)
    interp_order = 0 if is_integer else 1

    if preprocess_meta.get("applied_resample", False):
        cropped_shape = preprocess_meta.get("cropped_spatial_shape")
        if isinstance(cropped_shape, (list, tuple)) and len(cropped_shape) == spatial_dims:
            array = _resample_array_to_shape(
                array,
                target_shape=cropped_shape,
                spatial_dims=spatial_dims,
                order=interp_order,
            )

    if preprocess_meta.get("applied_crop", False):
        bbox = preprocess_meta.get("crop_bbox")
        original_shape = preprocess_meta.get("original_spatial_shape")
        if (
            isinstance(bbox, (list, tuple))
            and isinstance(original_shape, (list, tuple))
            and len(bbox) == spatial_dims
            and len(original_shape) == spatial_dims
        ):
            crop_target_shape = tuple(int(b[1]) - int(b[0]) for b in bbox)
            array = _fit_array_to_shape(array, crop_target_shape, spatial_dims=spatial_dims)

            if array.ndim == spatial_dims + 1:
                restored = np.zeros((array.shape[0], *original_shape), dtype=array.dtype)
                slices = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
                restored[(slice(None), *slices)] = array
            else:
                restored = np.zeros(tuple(int(v) for v in original_shape), dtype=array.dtype)
                slices = tuple(slice(int(b[0]), int(b[1])) for b in bbox)
                restored[slices] = array
            array = restored

    transpose_axes = preprocess_meta.get("transpose_axes")
    if isinstance(transpose_axes, (list, tuple)) and len(transpose_axes) == spatial_dims:
        inverse_axes = np.argsort(np.asarray(transpose_axes))
        if array.ndim == spatial_dims + 1:
            perm = [0] + [int(i) + 1 for i in inverse_axes]
            array = np.transpose(array, perm)
        elif array.ndim == spatial_dims:
            array = np.transpose(array, tuple(int(i) for i in inverse_axes))

    return array


class NNUNetPreprocessd(MapTransform):
    """nnU-Net style preprocessing transform."""

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
        if not (0.0 <= self.clip_percentile_low <= 1.0):
            raise ValueError(
                "clip_percentile_low must be in [0, 1], " f"got {self.clip_percentile_low}"
            )
        if not (0.0 <= self.clip_percentile_high <= 1.0):
            raise ValueError(
                "clip_percentile_high must be in [0, 1], " f"got {self.clip_percentile_high}"
            )
        if self.clip_percentile_low > self.clip_percentile_high:
            raise ValueError(
                "clip_percentile_low must be <= "
                "clip_percentile_high, got "
                f"{self.clip_percentile_low} > "
                f"{self.clip_percentile_high}"
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
            "normalization_use_nonzero_mask": (self.normalization_use_nonzero_mask),
            "clip_percentile_low": self.clip_percentile_low,
            "clip_percentile_high": (self.clip_percentile_high),
            "crop_bbox": None,
            "cropped_spatial_shape": list(image_spatial_shape),
            "resampled_spatial_shape": list(image_spatial_shape),
            "applied_crop": False,
            "applied_resample": False,
            "transpose_axes": meta_dict.get("transpose_axes"),
        }

        # --- Crop to nonzero ---
        if self.crop_to_nonzero:
            nonzero_mask = self._create_nonzero_mask(image_np, spatial_dims)
            if nonzero_mask is not None and np.any(nonzero_mask):
                bbox = self._bbox_from_mask(nonzero_mask)
                preprocess_meta["crop_bbox"] = [[int(b.start), int(b.stop)] for b in bbox]
                preprocess_meta["applied_crop"] = True
                # Crop non-image keys only
                for key in self.key_iterator(d):
                    if key not in d or key == self.image_key:
                        continue
                    val_np, st = self._as_numpy(d[key])
                    val_np = self._crop_to_bbox(val_np, bbox, spatial_dims)
                    d[key] = self._from_numpy(val_np, st)
                # Crop image
                image_np = self._crop_to_bbox(image_np, bbox, spatial_dims)
                preprocess_meta["cropped_spatial_shape"] = list(
                    self._get_spatial_shape(image_np, spatial_dims)
                )

        # --- Resample ---
        if source_spacing is not None and target_spacing is not None:
            current_shape = np.array(
                preprocess_meta["cropped_spatial_shape"],
                dtype=np.float32,
            )
            spacing_factors = np.array(source_spacing, dtype=np.float32) / np.array(
                target_spacing, dtype=np.float32
            )
            target_shape = np.maximum(
                np.round(current_shape * spacing_factors).astype(int),
                1,
            )
            if np.any(target_shape != current_shape.astype(int)):
                separate_z, lowres_axis = self._resolve_separate_z(
                    source_spacing,
                    target_spacing,
                    spatial_dims,
                )
                preprocess_meta["applied_resample"] = True
                preprocess_meta["separate_z"] = separate_z
                preprocess_meta["lowres_axis"] = lowres_axis
                tgt = tuple(int(v) for v in target_shape)
                # Resample non-image keys only
                for key in self.key_iterator(d):
                    if key not in d or key == self.image_key:
                        continue
                    val_np, st = self._as_numpy(d[key])
                    val_np = self._resample_to_shape(
                        val_np,
                        target_shape=tgt,
                        spatial_dims=spatial_dims,
                        is_seg=True,
                        separate_z=separate_z,
                        lowres_axis=lowres_axis,
                    )
                    d[key] = self._from_numpy(val_np, st)
                # Resample image
                image_np = self._resample_to_shape(
                    image_np,
                    target_shape=tgt,
                    spatial_dims=spatial_dims,
                    is_seg=False,
                    separate_z=separate_z,
                    lowres_axis=lowres_axis,
                )
                preprocess_meta["resampled_spatial_shape"] = list(
                    self._get_spatial_shape(image_np, spatial_dims)
                )

        # --- Clip and normalize image ---
        pre_stats = self._format_debug_stats(image_np)
        image_np = self._clip_image_percentiles(
            image_np,
            spatial_dims=spatial_dims,
            low=self.clip_percentile_low,
            high=self.clip_percentile_high,
            use_nonzero_mask=(self.normalization_use_nonzero_mask),
        )
        image_np = self._normalize_image(
            image_np,
            spatial_dims=spatial_dims,
            mode=self.normalization,
            use_nonzero_mask=(self.normalization_use_nonzero_mask),
        )
        post_stats = self._format_debug_stats(image_np)
        logger.info(
            "[NNUNetPreprocessd] key=%s mode=%s "
            "clip=(%.3f,%.3f) nonzero_mask=%s "
            "pre=%s post=%s",
            self.image_key,
            self.normalization,
            self.clip_percentile_low,
            self.clip_percentile_high,
            self.normalization_use_nonzero_mask,
            pre_stats,
            post_stats,
        )
        d[self.image_key] = self._from_numpy(image_np, image_state)

        meta_dict["nnunet_preprocess"] = preprocess_meta
        d[meta_key] = meta_dict
        return d

    # --- Static helpers ---

    @staticmethod
    def _infer_spatial_dims(array: np.ndarray) -> int:
        if array.ndim <= 2:
            return array.ndim
        return array.ndim - 1

    @staticmethod
    def _get_spatial_shape(array: np.ndarray, spatial_dims: int) -> tuple:
        if array.ndim == spatial_dims:
            return tuple(int(v) for v in array.shape)
        return tuple(int(v) for v in array.shape[-spatial_dims:])

    @staticmethod
    def _as_numpy(
        value: Any,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
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
        raise TypeError("NNUNetPreprocessd expects numpy or tensor, " f"got {type(value)}")

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
        spacing: Optional[Sequence[float]],
        spatial_dims: int,
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
            f"min={float(finite.min()):.4f} "
            f"max={float(finite.max()):.4f} "
            f"mean={float(finite.mean()):.4f} "
            f"std={float(finite.std()):.4f}"
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
    def _bbox_from_mask(
        mask: np.ndarray,
    ) -> tuple[slice, ...]:
        coords = np.where(mask)
        return tuple(slice(int(np.min(c)), int(np.max(c)) + 1) for c in coords)

    @staticmethod
    def _crop_to_bbox(
        array: np.ndarray,
        bbox: tuple[slice, ...],
        spatial_dims: int,
    ) -> np.ndarray:
        if array.ndim == spatial_dims + 1:
            return array[(slice(None), *bbox)]
        if array.ndim == spatial_dims:
            return array[bbox]
        return array

    def _resolve_separate_z(
        self,
        source_spacing: Sequence[float],
        target_spacing: Sequence[float],
        spatial_dims: int,
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

        current = np.array(array.shape, dtype=np.float32)
        target = np.array(target_shape, dtype=np.float32)
        if np.all(current == target):
            return array

        if not separate_z or lowres_axis is None:
            factors = target / current
            return zoom(
                array.astype(np.float32, copy=False),
                zoom=factors,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )

        # nnU-Net anisotropic: in-plane first, then
        # low-res axis.
        axis = int(lowres_axis)
        inplane = [i for i in range(3) if i != axis]
        slices: List[np.ndarray] = []
        for i in range(array.shape[axis]):
            if axis == 0:
                part = array[i, :, :]
            elif axis == 1:
                part = array[:, i, :]
            else:
                part = array[:, :, i]

            pf = [
                target[inplane[0]] / max(part.shape[0], 1),
                target[inplane[1]] / max(part.shape[1], 1),
            ]
            resized = zoom(
                part.astype(np.float32, copy=False),
                zoom=pf,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )
            slices.append(resized)

        stacked = np.stack(slices, axis=axis)
        if stacked.shape[axis] != int(target[axis]):
            af = np.ones(3, dtype=np.float32)
            af[axis] = target[axis] / max(stacked.shape[axis], 1)
            z_order = self.order_z if not is_seg else self.label_order
            stacked = zoom(
                stacked.astype(np.float32, copy=False),
                zoom=af,
                order=z_order,
                mode="nearest",
                prefilter=(z_order > 1),
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

        def _apply(
            vol: np.ndarray,
            mask: Optional[np.ndarray],
        ) -> np.ndarray:
            if mask is not None and np.any(mask):
                values = vol[mask]
            else:
                values = vol.reshape(-1)
            if values.size == 0:
                return vol
            lo = float(np.percentile(values, low_q))
            hi = float(np.percentile(values, high_q))
            if hi < lo:
                lo, hi = hi, lo
            return np.clip(vol, lo, hi)

        if image.ndim == spatial_dims + 1:
            nz = np.any(image != 0, axis=0) if use_nonzero_mask else None
            if nz is not None and np.all(nz):
                nz = None  # Skip masking if all nonzero
            for c in range(image.shape[0]):
                image[c] = _apply(image[c], nz)
                if nz is not None:
                    image[c][~nz] = 0
            return image

        mask = image != 0 if use_nonzero_mask else None
        if mask is not None and np.all(mask):
            mask = None
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

        def _apply(
            vol: np.ndarray,
            mask: Optional[np.ndarray],
        ) -> np.ndarray:
            ml = str(mode).lower().strip()
            if mask is not None and np.any(mask):
                values = vol[mask]
            else:
                values = vol.reshape(-1)
            if values.size == 0:
                return vol

            if ml in ("zscore", "normal"):
                mean = float(values.mean())
                std = float(values.std())
                if std > 1e-8:
                    vol = (vol - mean) / std
            elif ml == "0-1":
                lo = float(values.min())
                hi = float(values.max())
                if hi > lo:
                    vol = (vol - lo) / (hi - lo)
            elif ml.startswith("divide-"):
                scale = float(ml.split("-", 1)[1])
                if abs(scale) > 1e-8:
                    vol = vol / scale
            return vol

        if image.ndim == spatial_dims + 1:
            nz = np.any(image != 0, axis=0) if use_nonzero_mask else None
            if nz is not None and np.all(nz):
                nz = None
            for c in range(image.shape[0]):
                image[c] = _apply(image[c], nz)
                if nz is not None:
                    image[c][~nz] = 0
            return image

        mask = image != 0 if use_nonzero_mask else None
        if mask is not None and np.all(mask):
            mask = None
        image = _apply(image, mask)
        if mask is not None:
            image[~mask] = 0
        return image
