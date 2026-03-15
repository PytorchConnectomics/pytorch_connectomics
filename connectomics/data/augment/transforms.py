"""
MONAI-native transforms for connectomics-specific augmentations.

Each transform is a thin MONAI wrapper that delegates business logic
to pure functions in augment_ops.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
from scipy.ndimage import binary_dilation, generate_binary_structure

from . import augment_ops


def _to_numpy(img: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, bool, Any]:
    """Convert to numpy, returning (array, was_tensor, device)."""
    if isinstance(img, torch.Tensor):
        return img.cpu().numpy(), True, img.device
    return img, False, None


def _from_numpy(img: np.ndarray, was_tensor: bool, device: Any) -> Union[np.ndarray, torch.Tensor]:
    """Convert back to tensor if input was tensor."""
    if was_tensor:
        return torch.from_numpy(img).to(device)
    return img


def _infer_depth_axis(arr: np.ndarray) -> int:
    """Infer the depth axis for channel-first volumes."""
    if arr.ndim >= 4 and arr.shape[0] <= 4:
        return 1
    return 0


def _sample_count(
    rng: np.random.RandomState,
    spec: Union[int, Tuple[int, int], List[int]],
    max_count: int,
) -> int:
    """Sample a non-negative count from an int or inclusive [min, max] pair."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 0:
            return 0
        if len(spec) == 1:
            count = int(spec[0])
        else:
            low = int(spec[0])
            high = int(spec[1])
            if high < low:
                low, high = high, low
            count = int(rng.randint(low, high + 1))
    else:
        count = int(spec)
    return min(max(count, 0), max_count)


class RandMisAlignmentd(RandomizableTransform, MapTransform):
    """Random misalignment augmentation simulating EM section alignment artifacts."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        displacement: int = 16,
        rotate_ratio: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.displacement = displacement
        self.rotate_ratio = rotate_ratio

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.prob <= 0:
            self._do_transform = False
            return data
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        use_rotation = self.R.rand() < self.rotate_ratio

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 3:
                    continue

                depth_axis = _infer_depth_axis(arr)
                depth = arr.shape[depth_axis]
                if depth <= 2:
                    continue

                split_idx = int(self.R.choice(np.arange(1, depth - 1)))
                mode = "slip" if self.R.rand() < 0.5 else "translation"

                if use_rotation:
                    height = arr.shape[-2]
                    angle_range = augment_ops.compute_misalignment_angle_range(
                        self.displacement, height
                    )
                    rand_angle = (self.R.rand() - 0.5) * 2.0 * angle_range
                    result = augment_ops.apply_misalignment_rotation(
                        arr,
                        self.displacement,
                        rand_angle,
                        split_idx,
                        mode,
                        depth_axis=depth_axis,
                    )
                else:
                    dy0 = int(self.R.randint(-self.displacement, self.displacement + 1))
                    dx0 = int(self.R.randint(-self.displacement, self.displacement + 1))
                    dy1 = int(self.R.randint(-self.displacement, self.displacement + 1))
                    dx1 = int(self.R.randint(-self.displacement, self.displacement + 1))
                    result = augment_ops.apply_misalignment_translation(
                        arr,
                        self.displacement,
                        dy0,
                        dx0,
                        dy1,
                        dx1,
                        split_idx,
                        mode,
                        depth_axis=depth_axis,
                    )
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class RandMissingSectiond(RandomizableTransform, MapTransform):
    """Random missing section augmentation with paper-style fill values."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        num_sections: Union[int, Tuple[int, int]] = 2,
        full_section_prob: float = 0.5,
        partial_ratio_range: Tuple[float, float] = (0.25, 0.75),
        fill_value_range: Tuple[float, float] = (0.0, 1.0),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.num_sections = num_sections
        self.full_section_prob = full_section_prob
        self.partial_ratio_range = partial_ratio_range
        self.fill_value_range = fill_value_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.prob <= 0:
            self._do_transform = False
            return data
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 3:
                    continue

                depth_axis = _infer_depth_axis(arr)
                depth = arr.shape[depth_axis]
                if depth <= 3:
                    continue

                num_to_fill = _sample_count(self.R, self.num_sections, depth - 2)
                if num_to_fill == 0:
                    continue

                indices = self.R.choice(np.arange(1, depth - 1), size=num_to_fill, replace=False)
                result = arr
                for idx in indices:
                    fill_value = float(self.R.uniform(*self.fill_value_range))
                    if self.R.rand() < self.full_section_prob:
                        result = augment_ops.fill_sections(
                            result,
                            np.asarray([idx]),
                            fill_value=fill_value,
                            depth_axis=depth_axis,
                        )
                        continue

                    hole_h = max(
                        1,
                        int(arr.shape[-2] * self.R.uniform(*self.partial_ratio_range)),
                    )
                    hole_w = max(
                        1,
                        int(arr.shape[-1] * self.R.uniform(*self.partial_ratio_range)),
                    )
                    y_start = int(self.R.randint(0, max(1, arr.shape[-2] - hole_h + 1)))
                    x_start = int(self.R.randint(0, max(1, arr.shape[-1] - hole_w + 1)))
                    result = augment_ops.fill_region(
                        result,
                        y_start,
                        x_start,
                        hole_h,
                        hole_w,
                        section_axis=depth_axis,
                        section_idx=int(idx),
                        fill_value=fill_value,
                    )
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class RandMissingPartsd(RandomizableTransform, MapTransform):
    """Random missing parts — creates rectangular holes in sections."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        hole_range: Tuple[float, float] = (0.1, 0.3),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.hole_range = hole_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.prob <= 0:
            self._do_transform = False
            return data
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 2:
                    continue

                # Determine section axis
                section_axis: Optional[int] = None
                section_idx: Optional[int] = None
                if arr.ndim == 2:
                    pass
                elif arr.ndim == 3 and arr.shape[0] <= 4:
                    pass  # 2D channel-first
                elif arr.ndim >= 4 and arr.shape[0] <= 4:
                    section_axis = 1
                else:
                    section_axis = 0

                hole_ratio = self.R.uniform(*self.hole_range)
                hole_h = max(1, int(arr.shape[-2] * hole_ratio))
                hole_w = max(1, int(arr.shape[-1] * hole_ratio))
                y_start = int(self.R.randint(0, max(1, arr.shape[-2] - hole_h + 1)))
                x_start = int(self.R.randint(0, max(1, arr.shape[-1] - hole_w + 1)))

                if section_axis is not None:
                    section_idx = int(self.R.randint(0, arr.shape[section_axis]))

                result = augment_ops.create_missing_hole(
                    arr, y_start, x_start, hole_h, hole_w, section_axis, section_idx
                )
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class RandMotionBlurd(RandomizableTransform, MapTransform):
    """Legacy name for paper-style out-of-focus Gaussian blur augmentation."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        sections: Union[int, Tuple[int, int]] = 2,
        kernel_size: int = 11,
        sigma_range: Tuple[float, float] = (1.0, 3.0),
        full_section_prob: float = 0.5,
        partial_ratio_range: Tuple[float, float] = (0.25, 0.75),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sections = sections
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.full_section_prob = full_section_prob
        self.partial_ratio_range = partial_ratio_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 3:
                    continue

                depth_axis = _infer_depth_axis(arr)
                depth = arr.shape[depth_axis]
                num_sections = _sample_count(self.R, self.sections, depth)
                if num_sections == 0:
                    continue

                section_indices = self.R.choice(depth, size=num_sections, replace=False)
                result = arr
                for idx in section_indices:
                    sigma = float(self.R.uniform(*self.sigma_range))
                    if self.R.rand() < self.full_section_prob:
                        result = augment_ops.blur_sections(
                            result,
                            np.asarray([idx]),
                            kernel_size=self.kernel_size,
                            sigma=sigma,
                            depth_axis=depth_axis,
                        )
                        continue

                    hole_h = max(
                        1,
                        int(arr.shape[-2] * self.R.uniform(*self.partial_ratio_range)),
                    )
                    hole_w = max(
                        1,
                        int(arr.shape[-1] * self.R.uniform(*self.partial_ratio_range)),
                    )
                    y_start = int(self.R.randint(0, max(1, arr.shape[-2] - hole_h + 1)))
                    x_start = int(self.R.randint(0, max(1, arr.shape[-1] - hole_w + 1)))
                    result = augment_ops.blur_region(
                        result,
                        section_idx=int(idx),
                        y_start=y_start,
                        x_start=x_start,
                        hole_h=hole_h,
                        hole_w=hole_w,
                        kernel_size=self.kernel_size,
                        sigma=sigma,
                        depth_axis=depth_axis,
                    )
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class RandCutNoised(RandomizableTransform, MapTransform):
    """Random cut noise — adds noise to random cuboid regions."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        length_ratio: float = 0.25,
        noise_scale: float = 0.2,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.length_ratio = length_ratio
        self.noise_scale = noise_scale

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 2 or any(s == 0 for s in arr.shape):
                    continue

                # Build cuboid slices (channel dim + spatial dims)
                spatial_shape = arr.shape[1:]
                slices = [slice(None)]
                noise_shape = [arr.shape[0]]
                for s in spatial_shape:
                    length = max(1, int(self.length_ratio * s))
                    start = int(self.R.randint(0, max(1, s - length + 1)))
                    slices.append(slice(start, start + length))
                    noise_shape.append(length)

                noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
                result = augment_ops.apply_cut_noise(arr, slices, noise)
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class RandCutBlurd(RandomizableTransform, MapTransform):
    """Random CutBlur — downsample+upsample cuboid regions for super-resolution learning."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        length_ratio: float = 0.25,
        down_ratio_range: Tuple[float, float] = (2.0, 8.0),
        downsample_z: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.length_ratio = length_ratio
        self.down_ratio_range = down_ratio_range
        self.downsample_z = downsample_z

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        # Get random params once from first available key
        random_params = None
        for key in self.key_iterator(d):
            if key in d:
                random_params = self._get_random_params(d[key])
                break

        if random_params is None:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                zl, zh, yl, yh, xl, xh, down_ratio = random_params
                result = augment_ops.apply_cutblur(
                    arr, zl, zh, yl, yh, xl, xh, down_ratio, self.downsample_z
                )
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob

    def _get_random_params(self, img: Union[np.ndarray, torch.Tensor]) -> Tuple:
        shape = img.shape
        zdim = shape[0] if len(shape) == 3 else 1

        if zdim > 1:
            zl, zh = self._random_region(shape[0])
        else:
            zl, zh = None, None

        yl, yh = self._random_region(shape[1] if len(shape) == 3 else shape[0])
        xl, xh = self._random_region(shape[2] if len(shape) == 3 else shape[1])
        down_ratio = self.R.uniform(*self.down_ratio_range)
        return zl, zh, yl, yh, xl, xh, down_ratio

    def _random_region(self, vol_len: int) -> Tuple[int, int]:
        cuboid_len = max(1, int(self.length_ratio * vol_len))
        low = int(self.R.randint(0, max(1, vol_len - cuboid_len + 1)))
        return low, low + cuboid_len


class RandMixupd(RandomizableTransform, MapTransform):
    """Random Mixup — linear interpolation between batch samples.

    Warning: This transform requires a batch dimension (ndim >= 4) and at least
    2 samples along that dimension. In standard per-sample MONAI pipelines
    (where each dict is one sample with ndim=3), this is a no-op. For true
    cross-sample mixup, use a collate-level or batch-level transform instead.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        alpha_range: Tuple[float, float] = (0.7, 0.9),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.alpha_range = alpha_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_mixup(d[key])
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob

    def _apply_mixup(
        self, volume: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply mixup to a batched volume (requires batch dim)."""
        if volume.ndim < 4:
            return volume

        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = torch.from_numpy(volume)

        batch_size = volume.shape[0]
        if batch_size < 2:
            return volume.numpy() if is_numpy else volume

        alpha = self.R.uniform(*self.alpha_range)
        indices = torch.randperm(batch_size)
        mixed = alpha * volume + (1 - alpha) * volume[indices]
        return mixed.numpy() if is_numpy else mixed


class RandCopyPasted(RandomizableTransform, MapTransform):
    """Random Copy-Paste — copies transformed objects to non-overlapping regions."""

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str = "label",
        prob: float = 0.5,
        max_obj_ratio: float = 0.7,
        rotation_angles: List[int] = list(range(30, 360, 30)),
        border: int = 3,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.label_key = label_key
        self.max_obj_ratio = max_obj_ratio
        self.rotation_angles = rotation_angles
        self.border = border
        self.dil_struct = self._generate_binary_structure()

    @staticmethod
    def _generate_binary_structure():
        return generate_binary_structure(3, 3)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        if self.label_key not in d:
            return d

        label = d[self.label_key]
        if isinstance(label, torch.Tensor):
            obj_ratio = label.float().mean().item()
        else:
            obj_ratio = float(label.astype(np.float32).mean())

        if obj_ratio > self.max_obj_ratio:
            return d

        for key in self.key_iterator(d):
            if key in d and key != self.label_key:
                d[key], d[self.label_key] = self._apply_copy_paste(d[key], label)
        return d

    def _apply_copy_paste(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        label: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Apply copy-paste augmentation."""
        is_numpy = isinstance(volume, np.ndarray)

        if is_numpy:
            volume = torch.from_numpy(volume.copy())
            label = torch.from_numpy(label.copy())

        label = label.bool()

        if label.ndim != 3 or volume.ndim not in [3, 4]:
            return volume.numpy() if is_numpy else volume, (label.numpy() if is_numpy else label)

        label_flipped = label.flip(0)

        if volume.ndim == 4:
            neuron_tensor = volume * label.unsqueeze(0)
        else:
            neuron_tensor = volume * label

        neuron_tensor, label_paste = self._find_best_paste(neuron_tensor, label, label_flipped)

        if volume.ndim == 4:
            label_paste = label_paste.unsqueeze(0)
            volume = volume * (~label_paste) + neuron_tensor * label_paste
        else:
            volume = volume * (~label_paste) + neuron_tensor * label_paste

        if is_numpy:
            return volume.numpy(), (
                label_paste.squeeze().numpy() if label_paste.ndim > 3 else label_paste.numpy()
            )
        return volume, label_paste.squeeze() if label_paste.ndim > 3 else label_paste

    def _find_best_paste(
        self,
        neuron_tensor: torch.Tensor,
        label_orig: torch.Tensor,
        label_flipped: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best rotation and position with minimal overlap."""
        labels = torch.stack([label_orig, label_flipped])
        best_overlap = torch.logical_and(label_flipped, label_orig).int().sum()
        best_angle = 0
        best_idx = 1

        for angle in self.rotation_angles:
            rotated = self._rotate_3d(labels, angle)
            overlap0 = torch.logical_and(rotated[0], label_orig).int().sum()
            overlap1 = torch.logical_and(rotated[1], label_orig).int().sum()

            if overlap0 < best_overlap:
                best_overlap = overlap0
                best_angle = angle
                best_idx = 0
            if overlap1 < best_overlap:
                best_overlap = overlap1
                best_angle = angle
                best_idx = 1

        if best_idx == 1:
            neuron_tensor = (
                neuron_tensor.flip(0) if neuron_tensor.ndim == 3 else neuron_tensor.flip(1)
            )

        label_paste = labels[best_idx : best_idx + 1]

        if best_angle != 0:
            label_paste = self._rotate_3d(label_paste, best_angle)
            if neuron_tensor.ndim == 4:
                neuron_tensor = self._rotate_3d(neuron_tensor.unsqueeze(0), best_angle).squeeze(0)
            else:
                neuron_tensor = self._rotate_3d(neuron_tensor.unsqueeze(0), best_angle).squeeze(0)

        label_paste = label_paste.squeeze(0)

        gt_dilated = torch.tensor(
            binary_dilation(label_orig.numpy(), structure=self.dil_struct, iterations=self.border)
        )
        overlap_mask = torch.logical_and(label_paste, gt_dilated)
        label_paste[overlap_mask] = False

        if neuron_tensor.ndim == 4:
            neuron_tensor[:, overlap_mask] = 0
        else:
            neuron_tensor[overlap_mask] = 0

        return neuron_tensor, label_paste

    @staticmethod
    def _rotate_3d(tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate 3D volume around z-axis."""
        if tensor.ndim == 4:  # (C, Z, Y, X)
            c, z, y, x = tensor.shape
            reshaped = tensor.reshape(1, c * z, y, x)
            rotated = tvf.rotate(reshaped, angle)
            return rotated.reshape(c, z, y, x)
        elif tensor.ndim == 5:  # (B, C, Z, Y, X)
            b, c, z, y, x = tensor.shape
            rotated_list = []
            for i in range(b):
                reshaped = tensor[i].reshape(1, c * z, y, x)
                rot = tvf.rotate(reshaped, angle)
                rotated_list.append(rot.reshape(c, z, y, x))
            return torch.stack(rotated_list)
        return tensor

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class NormalizeLabelsd(MapTransform):
    """Convert labels to binary {0, 1} integers."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                if isinstance(d[key], np.ndarray):
                    d[key] = (d[key] > 0).astype(np.int32)
                elif isinstance(d[key], torch.Tensor):
                    d[key] = (d[key] > 0).int()
        return d


class SmartNormalizeIntensityd(MapTransform):
    """Smart intensity normalization with multiple modes and percentile clipping.

    Modes: "none", "normal" (z-score), "0-1" (min-max), "divide-K" (divide by K).
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "0-1",
        clip_percentile_low: float = 0.0,
        clip_percentile_high: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.divide_value = None
        if mode.startswith("divide-"):
            try:
                self.divide_value = float(mode.split("-", 1)[1])
                self.mode = "divide"
            except ValueError:
                raise ValueError(
                    f"Invalid divide mode '{mode}'. Format should be 'divide-K' "
                    f"where K is a number (e.g., 'divide-255')"
                )
        elif mode not in ["none", "normal", "0-1"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'none', 'normal', '0-1', or 'divide-K'"
            )
        else:
            self.mode = mode

        self.clip_percentile_low = clip_percentile_low
        self.clip_percentile_high = clip_percentile_high

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                result = augment_ops.smart_normalize(
                    arr,
                    self.mode,
                    self.divide_value,
                    self.clip_percentile_low,
                    self.clip_percentile_high,
                )
                d[key] = _from_numpy(result, was_tensor, device)
        return d


class RandStriped(RandomizableTransform, MapTransform):
    """Random stripe augmentation simulating EM curtaining/scan line artifacts."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.3,
        num_stripes_range: Tuple[int, int] = (2, 10),
        thickness_range: Tuple[int, int] = (1, 5),
        intensity_range: Tuple[float, float] = (-0.2, 0.2),
        angle_range: Optional[Tuple[float, float]] = None,
        orientation: str = "random",
        mode: str = "add",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.num_stripes_range = num_stripes_range
        self.thickness_range = thickness_range
        self.intensity_range = intensity_range
        self.angle_range = angle_range

        if orientation not in ["horizontal", "vertical", "random"]:
            raise ValueError(
                f"Invalid orientation '{orientation}'. Must be 'horizontal', "
                f"'vertical', or 'random'"
            )
        self.orientation = orientation

        if mode not in ["add", "replace"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'add' or 'replace'")
        self.mode = mode

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                arr, was_tensor, device = _to_numpy(d[key])
                if arr.ndim < 2:
                    continue

                # Determine angle
                if self.angle_range is not None:
                    angle = float(self.R.uniform(*self.angle_range))
                elif self.orientation == "random":
                    angle = 0.0 if self.R.rand() > 0.5 else 90.0
                elif self.orientation == "horizontal":
                    angle = 0.0
                else:
                    angle = 90.0

                # Number of stripes
                if self.num_stripes_range[0] == self.num_stripes_range[1]:
                    num_stripes = self.num_stripes_range[0]
                else:
                    num_stripes = int(
                        self.R.randint(self.num_stripes_range[0], self.num_stripes_range[1] + 1)
                    )

                # Generate stripe parameters
                # We need coord range for positioning — compute from a representative slice
                h, w = arr.shape[-2:]
                angle_rad = np.deg2rad(angle)
                y_coords, x_coords = np.ogrid[:h, :w]
                rotated_coords = x_coords * np.sin(angle_rad) - y_coords * np.cos(angle_rad)
                coord_min = float(rotated_coords.min())
                coord_max = float(rotated_coords.max())

                stripe_params = []
                for _ in range(num_stripes):
                    center = float(self.R.uniform(coord_min, coord_max))
                    if self.thickness_range[0] == self.thickness_range[1]:
                        thickness = self.thickness_range[0]
                    else:
                        thickness = int(
                            self.R.randint(self.thickness_range[0], self.thickness_range[1] + 1)
                        )
                    intensity = float(self.R.uniform(*self.intensity_range))
                    stripe_params.append((center, thickness, intensity))

                result = augment_ops.apply_stripes(arr, stripe_params, angle, self.mode)
                d[key] = _from_numpy(result, was_tensor, device)
        return d

    def randomize(self, _: Any = None) -> None:
        self._do_transform = self.R.rand() < self.prob


class ResizeByFactord(MapTransform):
    """Resize images by scale factors using F.interpolate."""

    def __init__(
        self,
        keys: KeysCollection,
        scale_factors: List[float],
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scale_factors = scale_factors
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d:
                continue
            arr = d[key]
            is_numpy = isinstance(arr, np.ndarray)
            if is_numpy:
                arr = torch.from_numpy(arr)

            inp = arr.unsqueeze(0).float()
            spatial_dims = inp.ndim - 2
            interp_mode = self.mode
            if interp_mode == "bilinear":
                if spatial_dims == 3:
                    interp_mode = "trilinear"
                elif spatial_dims == 1:
                    interp_mode = "linear"

            out = torch.nn.functional.interpolate(
                inp,
                scale_factor=[float(f) for f in self.scale_factors],
                mode=interp_mode,
                align_corners=None if interp_mode == "nearest" else self.align_corners,
            ).squeeze(0)

            d[key] = out.numpy() if is_numpy else out.to(arr.dtype)
        return d


class RandElasticd(MapTransform, RandomizableTransform):
    """Unified elastic deformation wrapping MONAI's Rand2DElasticd/Rand3DElasticd."""

    def __init__(
        self,
        keys,
        do_2d: bool = False,
        prob: float = 0.5,
        sigma_range: tuple = (5.0, 8.0),
        magnitude_range: tuple = (50.0, 150.0),
        allow_missing_keys: bool = False,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.do_2d = do_2d
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.mode = mode
        self.padding_mode = padding_mode

    def _build_inner_transform(self):
        from monai.transforms import Rand2DElasticd, Rand3DElasticd

        if self.do_2d:
            return Rand2DElasticd(
                keys=self.keys,
                prob=1.0,
                spacing=self.sigma_range,
                magnitude_range=self.magnitude_range,
                mode=self.mode,
                padding_mode=self.padding_mode,
                allow_missing_keys=self.allow_missing_keys,
            )
        return Rand3DElasticd(
            keys=self.keys,
            prob=1.0,
            sigma_range=self.sigma_range,
            magnitude_range=self.magnitude_range,
            mode=self.mode,
            padding_mode=self.padding_mode,
            allow_missing_keys=self.allow_missing_keys,
        )

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        if not hasattr(self, "_inner_transform"):
            self._inner_transform = self._build_inner_transform()
        return self._inner_transform(d)


__all__ = [
    "RandMisAlignmentd",
    "RandMissingSectiond",
    "RandMissingPartsd",
    "RandMotionBlurd",
    "RandCutNoised",
    "RandCutBlurd",
    "RandMixupd",
    "RandCopyPasted",
    "RandStriped",
    "NormalizeLabelsd",
    "SmartNormalizeIntensityd",
    "ResizeByFactord",
    "RandElasticd",
]
