"""
Complete MONAI-native transforms for connectomics-specific augmentations.

This module provides fully implemented MONAI MapTransform versions of connectomics
augmentations, replacing the legacy custom augmentor system entirely.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform


class RandMisAlignmentd(RandomizableTransform, MapTransform):
    """
    Random misalignment augmentation for connectomics data.

    Simulates section misalignment artifacts common in EM volumes.
    """

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

        # Generate random parameters
        use_rotation = self.R.rand() < self.rotate_ratio

        for key in self.key_iterator(d):
            if key in d:
                if use_rotation:
                    d[key] = self._apply_misalignment_rotation(d[key])
                else:
                    d[key] = self._apply_misalignment_translation(d[key])
        return d

    def randomize(self, _: Any = None) -> None:
        """Randomly decide whether to apply the transform."""
        self._do_transform = self.R.rand() < self.prob

    def _apply_misalignment_translation(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply translation-based misalignment."""
        if img.ndim < 3:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.cpu().numpy()

        # Skip if volume is too small (need at least 3 sections in first dim)
        if (
            img.shape[0] <= 2
            or img.shape[1] <= self.displacement
            or img.shape[2] <= self.displacement
        ):
            if is_tensor:
                return torch.from_numpy(img).to(device)
            return img

        out_shape = (
            img.shape[0],
            img.shape[1] - self.displacement,
            img.shape[2] - self.displacement,
        )

        x0 = self.R.randint(self.displacement)
        y0 = self.R.randint(self.displacement)
        x1 = self.R.randint(self.displacement)
        y1 = self.R.randint(self.displacement)
        idx = self.R.choice(np.arange(1, out_shape[0] - 1))
        mode = "slip" if self.R.rand() < 0.5 else "translation"

        output = np.zeros(out_shape, img.dtype)
        if mode == "slip":
            output = img[:, y0 : y0 + out_shape[1], x0 : x0 + out_shape[2]]
            output[idx] = img[idx, y1 : y1 + out_shape[1], x1 : x1 + out_shape[2]]
        else:
            output[:idx] = img[:idx, y0 : y0 + out_shape[1], x0 : x0 + out_shape[2]]
            output[idx:] = img[idx:, y1 : y1 + out_shape[1], x1 : x1 + out_shape[2]]

        if is_tensor:
            output = torch.from_numpy(output).to(device)

        return output

    def _apply_misalignment_rotation(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply rotation-based misalignment."""
        if img.ndim < 3:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.clone().cpu().numpy()
        else:
            img = img.copy()

        # Skip if volume is too small (need at least 3 sections in first dim)
        if img.shape[0] <= 2:
            if is_tensor:
                img = torch.from_numpy(img).to(device)
            return img

        height, width = img.shape[-2:]
        if height != width:
            if is_tensor:
                img = torch.from_numpy(img).to(device)
            return img  # Skip if not square

        # Generate rotation matrix
        x = self.displacement / 2.0
        y = ((height - self.displacement) / 2.0) * 1.42
        angle = math.asin(x / y) * 2.0 * 57.2958
        rand_angle = (self.R.rand() - 0.5) * 2.0 * angle
        M = cv2.getRotationMatrix2D((height / 2, height / 2), rand_angle, 1)

        idx = self.R.choice(np.arange(1, img.shape[0] - 1))
        mode = "slip" if self.R.rand() < 0.5 else "translation"

        interpolation = cv2.INTER_LINEAR if img.dtype == np.float32 else cv2.INTER_NEAREST

        if mode == "slip":
            img[idx] = cv2.warpAffine(
                img[idx],
                M,
                (height, width),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            for i in range(idx, img.shape[0]):
                img[i] = cv2.warpAffine(
                    img[i],
                    M,
                    (height, width),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                )

        if is_tensor:
            img = torch.from_numpy(img).to(device)

        return img


class RandMissingSectiond(RandomizableTransform, MapTransform):
    """
    Random missing section augmentation for connectomics data.

    Simulates missing or damaged sections in EM volumes.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        num_sections: int = 2,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.num_sections = num_sections

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
                d[key] = self._apply_missing_section(d[key])
        return d

    def randomize(self, _: Any = None) -> None:
        """Randomly decide whether to apply the transform."""
        self._do_transform = self.R.rand() < self.prob

    def _apply_missing_section(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Remove random sections from volume."""
        if img.ndim < 3:
            return img  # Skip 2D or very small volumes

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)

        depth_axis = 0
        if img.ndim >= 4 and img.shape[0] <= 3:
            depth_axis = 1  # Channel-first; depth is axis 1

        depth = img.shape[depth_axis]
        if depth <= 3:
            return img

        # Select sections to remove (avoid first and last)
        num_to_remove = min(self.num_sections, depth - 2)
        indices_to_remove = self.R.choice(
            np.arange(1, depth - 1), size=num_to_remove, replace=False
        )

        if is_tensor:
            keep_mask = torch.ones(depth, dtype=torch.bool, device=img.device)
            keep_mask[indices_to_remove] = False
            return torch.index_select(img, dim=depth_axis, index=keep_mask.nonzero(as_tuple=False).squeeze(-1))
        else:
            return np.delete(img, indices_to_remove, axis=depth_axis)


class RandMissingPartsd(RandomizableTransform, MapTransform):
    """
    Random missing parts augmentation for connectomics data.

    Creates rectangular missing regions in random sections.
    """

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
        d = dict(data)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_missing_parts(d[key])
        return d

    def _apply_missing_parts(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Create missing rectangular regions."""
        if img.ndim < 3:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            img = img.clone()
        else:
            img = img.copy()

        # Select random section
        section_idx = self.R.randint(0, img.shape[0])

        # Generate hole size
        hole_ratio = self.R.uniform(*self.hole_range)
        hole_h = int(img.shape[1] * hole_ratio)
        hole_w = int(img.shape[2] * hole_ratio)

        # Generate hole position
        y_start = self.R.randint(0, img.shape[1] - hole_h + 1)
        x_start = self.R.randint(0, img.shape[2] - hole_w + 1)

        # Create hole (set to 0 or mean value)
        img[section_idx, y_start : y_start + hole_h, x_start : x_start + hole_w] = 0

        return img


class RandMotionBlurd(RandomizableTransform, MapTransform):
    """
    Random motion blur augmentation for connectomics data.

    Applies directional blur to simulate motion artifacts.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        sections: Union[int, Tuple[int, int]] = 2,
        kernel_size: int = 11,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sections = sections
        self.kernel_size = kernel_size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_motion_blur(d[key])
        return d

    def _apply_motion_blur(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply motion blur to random sections."""
        if img.ndim < 3:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.clone().cpu().numpy()
        else:
            img = img.copy()

        # Generate motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        if self.R.rand() > 0.5:  # horizontal kernel
            kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        else:  # vertical kernel
            kernel[:, int((self.kernel_size - 1) / 2)] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size

        # Select sections to blur
        if isinstance(self.sections, tuple):
            num_sections = self.R.randint(*self.sections)
        else:
            num_sections = self.sections

        num_sections = min(num_sections, img.shape[0])
        section_indices = self.R.choice(img.shape[0], size=num_sections, replace=False)

        # Apply blur (cv2 requires numpy and 2D data)
        for idx in section_indices:
            section = img[idx]
            # Handle channel dimension if present
            if section.ndim == 3:  # [C, H, W]
                # Apply to each channel
                for c in range(section.shape[0]):
                    section[c] = cv2.filter2D(section[c], -1, kernel)
            else:  # [H, W]
                img[idx] = cv2.filter2D(section, -1, kernel)

        if is_tensor:
            img = torch.from_numpy(img).to(device)

        return img


class RandCutNoised(RandomizableTransform, MapTransform):
    """
    Random cut noise augmentation for connectomics data.

    Adds noise to random cuboid regions to improve model robustness.
    """

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
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_cut_noise(d[key])
        return d

    def _apply_cut_noise(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Add noise to random cuboid region."""
        if img.ndim < 2:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.clone().cpu().numpy()
        else:
            img = img.copy()

        # Skip if any dimension is 0 (empty tensor)
        if any(s == 0 for s in img.shape):
            return torch.from_numpy(img).to(device) if is_tensor else img

        # Calculate spatial dimensions (last 2 or 3 dims)
        y_len = max(1, int(self.length_ratio * img.shape[-2]))  # Ensure at least 1
        x_len = max(1, int(self.length_ratio * img.shape[-1]))  # Ensure at least 1
        y_start = self.R.randint(0, max(1, img.shape[-2] - y_len + 1))
        x_start = self.R.randint(0, max(1, img.shape[-1] - x_len + 1))

        # Generate noise and apply based on dimensionality
        if img.ndim == 4:
            # 4D case: (C, Z, Y, X) for 3D with channels or (C, H, W) for 2D with channels
            # Heuristic: if second dim is large (>10), assume it's depth (3D with channels)
            # Otherwise assume it's 2D with channels
            if img.shape[1] > 10:
                # (C, Z, Y, X) - 3D with channels
                z_len = int(self.length_ratio * img.shape[1])
                z_len = max(1, z_len)  # Ensure at least 1
                z_start = self.R.randint(0, max(1, img.shape[1] - z_len + 1))
                noise_shape = (img.shape[0], z_len, y_len, x_len)
                noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
                region = img[
                    :,
                    z_start : z_start + z_len,
                    y_start : y_start + y_len,
                    x_start : x_start + x_len,
                ]
                noisy_region = np.clip(region + noise, 0, 1)
                img[
                    :,
                    z_start : z_start + z_len,
                    y_start : y_start + y_len,
                    x_start : x_start + x_len,
                ] = noisy_region
            else:
                # (C, H, W) - 2D with channels
                noise_shape = (img.shape[0], y_len, x_len)
                noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
                region = img[:, y_start : y_start + y_len, x_start : x_start + x_len]
                noisy_region = np.clip(region + noise, 0, 1)
                img[:, y_start : y_start + y_len, x_start : x_start + x_len] = noisy_region
        elif img.ndim == 3:
            # 3D case: (Z, Y, X) or (C, H, W)
            # Heuristic: if first dim is small (<=4), assume it's channel (2D with channels)
            # Otherwise assume it's depth (3D)
            if img.shape[0] <= 4:
                # (C, H, W) - 2D with channels
                noise_shape = (img.shape[0], y_len, x_len)
                noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
                region = img[:, y_start : y_start + y_len, x_start : x_start + x_len]
                noisy_region = np.clip(region + noise, 0, 1)
                img[:, y_start : y_start + y_len, x_start : x_start + x_len] = noisy_region
            else:
                # (Z, Y, X) - 3D
                z_len = max(1, int(self.length_ratio * img.shape[0]))  # Ensure at least 1
                z_start = self.R.randint(0, max(1, img.shape[0] - z_len + 1))
                noise_shape = (z_len, y_len, x_len)
                noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
                region = img[
                    z_start : z_start + z_len, y_start : y_start + y_len, x_start : x_start + x_len
                ]
                noisy_region = np.clip(region + noise, 0, 1)
                img[
                    z_start : z_start + z_len, y_start : y_start + y_len, x_start : x_start + x_len
                ] = noisy_region
        else:
            # 2D case: (H, W)
            noise_shape = (y_len, x_len)
            noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
            region = img[y_start : y_start + y_len, x_start : x_start + x_len]
            noisy_region = np.clip(region + noise, 0, 1)
            img[y_start : y_start + y_len, x_start : x_start + x_len] = noisy_region

        if is_tensor:
            img = torch.from_numpy(img).to(device)

        return img


# Note: Standard transforms like Grayscale, Elastic, and Rescale are available in MONAI
# as RandShiftIntensityd, RandAdjustContrastd, Rand3DElasticd, and RandZoomd
# We only implement connectomics-specific transforms that aren't available in MONAI


class RandCutBlurd(RandomizableTransform, MapTransform):
    """
    Random CutBlur augmentation for connectomics data.

    Randomly downsample cuboid regions to force super-resolution learning.
    Adapted from https://arxiv.org/abs/2004.00448.

    Args:
        keys: Keys to apply CutBlur to
        prob: Probability of applying the augmentation
        length_ratio: Ratio of cuboid length compared to volume length
        down_ratio_range: Range for downsample ratio (min, max)
        downsample_z: Whether to downsample along z-axis
        allow_missing_keys: Whether to allow missing keys
    """

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
        if not self._do_transform:
            return d

        # Get random parameters once for all keys
        if "image" in d:
            random_params = self._get_random_params(d["image"])
        else:
            # Fallback if no image key
            random_params = None

        for key in self.key_iterator(d):
            if key in d and random_params is not None:
                d[key] = self._apply_cutblur(d[key], random_params)
        return d

    def _get_random_params(self, img: Union[np.ndarray, torch.Tensor]) -> Tuple:
        """Get random parameters for CutBlur transformation."""
        # Handle tensor conversion for shape checking
        if isinstance(img, torch.Tensor):
            shape = img.shape
        else:
            shape = img.shape

        zdim = shape[0] if len(shape) == 3 else 1

        # Generate random cuboid region
        if zdim > 1:
            zl, zh = self._random_region(shape[0])
        else:
            zl, zh = None, None

        yl, yh = self._random_region(shape[1] if len(shape) == 3 else shape[0])
        xl, xh = self._random_region(shape[2] if len(shape) == 3 else shape[1])

        # Generate random downsampling ratio
        down_ratio = self.R.uniform(*self.down_ratio_range)

        return zl, zh, yl, yh, xl, xh, down_ratio

    def _random_region(self, vol_len: int) -> Tuple[int, int]:
        """Generate random region coordinates."""
        cuboid_len = int(self.length_ratio * vol_len)
        if cuboid_len <= 0:
            cuboid_len = 1
        low = self.R.randint(0, max(1, vol_len - cuboid_len + 1))
        high = low + cuboid_len
        return low, high

    def _apply_cutblur(
        self, img: Union[np.ndarray, torch.Tensor], random_params: Tuple
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply CutBlur transformation with given parameters."""
        from scipy.ndimage import zoom

        zl, zh, yl, yh, xl, xh, down_ratio = random_params

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.clone().cpu().numpy()
        else:
            img = img.copy()

        # Apply CutBlur based on original implementation
        # Handle 4D data (C, Z, Y, X) or 3D data (Z, Y, X)
        if img.ndim == 4:
            # 4D case: (C, Z, Y, X)
            temp = img[:, zl:zh, yl:yh, xl:xh].copy()
            if self.downsample_z:
                out_shape = np.array(temp.shape) / np.array([1, down_ratio, down_ratio, down_ratio])
            else:
                out_shape = np.array(temp.shape) / np.array([1, 1, down_ratio, down_ratio])
        elif img.ndim == 3:
            # 3D case: (Z, Y, X)
            temp = img[zl:zh, yl:yh, xl:xh].copy()
            if self.downsample_z:
                out_shape = np.array(temp.shape) / down_ratio
            else:
                out_shape = np.array(temp.shape) / np.array([1, down_ratio, down_ratio])
        else:
            # 2D case: (Y, X)
            temp = img[yl:yh, xl:xh].copy()
            out_shape = np.array(temp.shape) / np.array([down_ratio, down_ratio])

        out_shape = out_shape.astype(int)
        # Ensure minimum size of 1
        out_shape = np.maximum(out_shape, 1)

        # Downsample with linear interpolation and anti-aliasing
        zoom_factors = [out_size / in_size for out_size, in_size in zip(out_shape, temp.shape)]
        downsampled = zoom(temp, zoom_factors, order=1, mode="reflect", prefilter=True)

        # Upsample with nearest neighbor (no anti-aliasing to preserve sharp edges)
        zoom_factors = [
            out_size / in_size for out_size, in_size in zip(temp.shape, downsampled.shape)
        ]
        upsampled = zoom(downsampled, zoom_factors, order=0, mode="reflect", prefilter=False)

        # Put back into original image
        if img.ndim == 4:
            # 4D case: (C, Z, Y, X)
            img[:, zl:zh, yl:yh, xl:xh] = upsampled
        elif img.ndim == 3:
            # 3D case: (Z, Y, X)
            img[zl:zh, yl:yh, xl:xh] = upsampled
        else:
            # 2D case: (Y, X)
            img[yl:yh, xl:xh] = upsampled

        if is_tensor:
            img = torch.from_numpy(img).to(device)

        return img


class RandMixupd(RandomizableTransform, MapTransform):
    """
    Random Mixup augmentation for connectomics data.

    Conducts linear interpolation between two samples in a batch to improve
    model robustness and generalization. This is a batch-level augmentation.

    Note: This transform operates on batched data and mixes samples within the batch.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        alpha_range: Tuple[float, float] = (0.7, 0.9),
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to apply mixup to (typically 'image')
            prob: Probability of applying mixup
            alpha_range: Range for mixing ratio (min, max)
            allow_missing_keys: Whether to allow missing keys
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.alpha_range = alpha_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Note: This expects data to be a dictionary with batched tensors.
        For batch-level augmentation, use this in a collate_fn or after batching.
        """
        d = dict(data)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_mixup(d[key])
        return d

    def _apply_mixup(
        self, volume: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply mixup to a batched volume."""
        # Check if batched (first dimension is batch)
        if volume.ndim < 4:
            # Not batched, return as is
            return volume

        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = torch.from_numpy(volume)

        batch_size = volume.shape[0]
        if batch_size < 2:
            # Need at least 2 samples to mix
            return volume.numpy() if is_numpy else volume

        # Generate random mixing ratio
        alpha = self.R.uniform(*self.alpha_range)

        # Create random permutation for pairing samples
        indices = torch.randperm(batch_size)

        # Mix samples
        mixed = alpha * volume + (1 - alpha) * volume[indices]

        return mixed.numpy() if is_numpy else mixed


class RandCopyPasted(RandomizableTransform, MapTransform):
    """
    Random Copy-Paste augmentation for connectomics data.

    Copies objects from the image (based on segmentation mask), transforms them
    (rotation/flip), and pastes them back in non-overlapping regions to increase
    object diversity and improve instance segmentation performance.

    This augmentation requires both image and label (segmentation mask).
    """

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
        """
        Args:
            keys: Keys to apply copy-paste to (typically 'image')
            label_key: Key for segmentation labels
            prob: Probability of applying copy-paste
            max_obj_ratio: Maximum fractional size of object (skip if too large)
            rotation_angles: List of rotation angles to try
            border: Border size for dilation when checking overlap
            allow_missing_keys: Whether to allow missing keys
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.label_key = label_key
        self.max_obj_ratio = max_obj_ratio
        self.rotation_angles = rotation_angles
        self.border = border
        self.dil_struct = self._generate_binary_structure()

    def _generate_binary_structure(self):
        """Generate 3D binary structure for dilation."""
        from scipy.ndimage import generate_binary_structure

        return generate_binary_structure(3, 3)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if not self._do_transform:
            return d

        # Check if label exists
        if self.label_key not in d:
            return d

        label = d[self.label_key]

        # Check object size
        if isinstance(label, torch.Tensor):
            obj_ratio = label.float().mean().item()
        else:
            obj_ratio = float(label.astype(np.float32).mean())

        if obj_ratio > self.max_obj_ratio:
            # Object too large, skip augmentation
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

        # Convert to torch for processing
        if is_numpy:
            volume = torch.from_numpy(volume.copy())
            label = torch.from_numpy(label.copy())

        # Ensure label is boolean
        label = label.bool()

        # Check dimensions
        if label.ndim != 3 or volume.ndim not in [3, 4]:
            return volume.numpy() if is_numpy else volume, (label.numpy() if is_numpy else label)

        # Create flipped version for pasting
        label_flipped = label.flip(0)  # Flip along z-axis

        # Extract object
        if volume.ndim == 4:
            neuron_tensor = volume * label.unsqueeze(0)
        else:
            neuron_tensor = volume * label

        # Find best rotation and position
        neuron_tensor, label_paste = self._find_best_paste(neuron_tensor, label, label_flipped)

        # Paste into image
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
        from scipy.ndimage import binary_dilation

        labels = torch.stack([label_orig, label_flipped])
        best_overlap = torch.logical_and(label_flipped, label_orig).int().sum()
        best_angle = 0
        best_idx = 1

        # Try different rotations
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

        # Apply best transformation
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

        # Crop overlapping regions
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

    def _rotate_3d(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate 3D volume around z-axis."""
        import torchvision.transforms.functional as tf

        # Handle different tensor shapes
        if tensor.ndim == 4:  # (C, Z, Y, X)
            c, z, y, x = tensor.shape
            # Reshape to (1, C*Z, Y, X) for rotation
            reshaped = tensor.reshape(1, c * z, y, x)
            rotated = tf.rotate(reshaped, angle)
            return rotated.reshape(c, z, y, x)
        elif tensor.ndim == 5:  # (B, C, Z, Y, X)
            b, c, z, y, x = tensor.shape
            rotated_list = []
            for i in range(b):
                reshaped = tensor[i].reshape(1, c * z, y, x)
                rot = tf.rotate(reshaped, angle)
                rotated_list.append(rot.reshape(c, z, y, x))
            return torch.stack(rotated_list)
        else:
            return tensor


class ConvertToFloatd(MapTransform):
    """
    Convert specified keys to float32 to avoid Byte tensor issues.

    This is needed because PyTorch's interpolate function doesn't support
    Byte tensors with trilinear mode, but labels are often loaded as uint8.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: np.dtype = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to convert
            dtype: Target data type (default: float32)
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert specified keys to target dtype."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                if isinstance(d[key], np.ndarray):
                    d[key] = d[key].astype(self.dtype)
                elif isinstance(d[key], torch.Tensor):
                    d[key] = d[key].to(dtype=torch.from_numpy(np.array([], dtype=self.dtype)).dtype)
        return d


class SqueezeLabeld(MapTransform):
    """
    Squeeze channel dimension from labels for 2D/3D compatibility.

    CrossEntropyLoss expects labels without channel dimension:
    - 2D: (B, H, W) instead of (B, 1, H, W)
    - 3D: (B, D, H, W) instead of (B, 1, D, H, W)

    This transform removes the channel dimension (dim=1) if it equals 1.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to squeeze (typically ['label'])
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Squeeze channel dimension if size == 1."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                label = d[key]
                # Check if it has a channel dimension of size 1
                if isinstance(label, np.ndarray):
                    if label.ndim >= 3 and label.shape[0] == 1:
                        d[key] = label.squeeze(0)  # Remove channel dim
                elif isinstance(label, torch.Tensor):
                    if label.ndim >= 3 and label.shape[0] == 1:
                        d[key] = label.squeeze(0)  # Remove channel dim
        return d


class NormalizeLabelsd(MapTransform):
    """
    Convert labels to binary {0, 1} integers.

    This transform converts label values to binary {0, 1} integers.
    - 0: background
    - 1: foreground
    Useful for binary segmentation tasks with CrossEntropyLoss.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to normalize
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert specified keys to binary {0, 1} integers."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                if isinstance(d[key], np.ndarray):
                    # Convert to binary: 0 for background, 1 for foreground
                    d[key] = (d[key] > 0).astype(np.int32)
                elif isinstance(d[key], torch.Tensor):
                    # Convert to binary: 0 for background, 1 for foreground
                    d[key] = (d[key] > 0).int()
        return d


class SmartNormalizeIntensityd(MapTransform):
    """
    Smart intensity normalization with multiple modes and percentile clipping.

    Normalization modes:
    - "none": No normalization
    - "normal": Z-score normalization (x - mean) / std
    - "0-1": Min-max scaling to [0, 1] (default)
    - "divide-K": Simple divide by K (e.g., "divide-255" for uint8 images)

    Percentile clipping is applied BEFORE normalization when low > 0.0 or high < 1.0.

    Args:
        keys: Keys to normalize
        mode: Normalization mode ("none", "normal", "0-1", or "divide-K")
        clip_percentile_low: Lower percentile (0.0 = no clip, 0.05 = 5th percentile)
        clip_percentile_high: Upper percentile (1.0 = no clip, 0.95 = 95th percentile)
        allow_missing_keys: Whether to allow missing keys

    Examples:
        # Min-max to [0, 1] (default, no clipping)
        SmartNormalizeIntensityd(keys=['image'], mode="0-1")

        # Z-score normalization
        SmartNormalizeIntensityd(keys=['image'], mode="normal")

        # Min-max with percentile clipping (clip 5%-95%)
        SmartNormalizeIntensityd(keys=['image'], mode="0-1",
                                 clip_percentile_low=0.05, clip_percentile_high=0.95)

        # No normalization
        SmartNormalizeIntensityd(keys=['image'], mode="none")
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

        # Parse mode - support "divide-K" format where K is a number
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
        """Normalize specified keys."""
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._normalize(d[key])
        return d

    def _normalize(
        self, volume: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply normalization to volume."""
        is_numpy = isinstance(volume, np.ndarray)
        if not is_numpy:
            volume = volume.numpy()

        # Step 1: Percentile clipping (if enabled by non-default values)
        if self.clip_percentile_low > 0.0 or self.clip_percentile_high < 1.0:
            low_val = np.percentile(volume, self.clip_percentile_low * 100)
            high_val = np.percentile(volume, self.clip_percentile_high * 100)
            volume = np.clip(volume, low_val, high_val)

        # Step 2: Normalization based on mode
        if self.mode == "none":
            # No normalization
            pass
        elif self.mode == "normal":
            # Z-score normalization
            data_mean = volume.mean()
            data_std = volume.std()
            if data_std > 1e-8:
                volume = (volume - data_mean) / data_std
        elif self.mode == "0-1":
            # Min-max to [0, 1]
            min_val = volume.min()
            max_val = volume.max()
            if max_val > min_val:
                volume = (volume - min_val) / (max_val - min_val)
        elif self.mode == "divide":
            # Simple divide by K (e.g., divide-255 for uint8 images)
            volume = volume / self.divide_value

        return volume if is_numpy else torch.from_numpy(volume)


__all__ = [
    # Connectomics-specific transforms (not available in standard MONAI)
    "RandMisAlignmentd",
    "RandMissingSectiond",
    "RandMissingPartsd",
    "RandMotionBlurd",
    "RandCutNoised",
    "RandCutBlurd",
    "RandMixupd",
    "RandCopyPasted",
    "RandStriped",
    "ConvertToFloatd",
    "NormalizeLabelsd",
    "SmartNormalizeIntensityd",
    "ResizeByFactord",
]


class RandStriped(RandomizableTransform, MapTransform):
    """
    Random stripe augmentation for connectomics data.

    Adds random stripes at arbitrary angles to simulate imaging artifacts
    common in electron microscopy, such as curtaining or scan line artifacts.

    Args:
        keys: Keys to apply stripe augmentation to
        prob: Probability of applying the augmentation
        num_stripes_range: Range for number of stripes as (min, max)
        thickness_range: Range for stripe thickness in pixels as (min, max)
        intensity_range: Range for stripe intensity values as (min, max)
            For grayscale: single value added/multiplied
            For color: can be per-channel or uniform
        angle_range: Range for stripe angles in degrees as (min, max)
            0° = horizontal, 90° = vertical, 45° = diagonal
            Use None for predefined orientations (horizontal/vertical/random)
        orientation: Stripe orientation when angle_range is None
            - 'horizontal': 0° stripes
            - 'vertical': 90° stripes
            - 'random': randomly choose between horizontal and vertical
            Ignored if angle_range is specified
        mode: How to apply stripes - 'add' (additive) or 'replace' (replacement)
        allow_missing_keys: Whether to allow missing keys

    Examples:
        # Horizontal curtaining artifacts (typical in EM)
        RandStriped(keys=['image'], prob=0.3, num_stripes_range=(2, 5),
                   thickness_range=(1, 3), intensity_range=(-0.2, 0.2),
                   orientation='horizontal', mode='add')

        # Vertical scan line artifacts
        RandStriped(keys=['image'], prob=0.3, num_stripes_range=(5, 15),
                   thickness_range=(1, 2), intensity_range=(-0.15, 0.15),
                   orientation='vertical', mode='add')

        # Random angle stripes (diagonal artifacts)
        RandStriped(keys=['image'], prob=0.5, num_stripes_range=(3, 10),
                   thickness_range=(1, 5), intensity_range=(0, 0.5),
                   angle_range=(0, 180), mode='add')

        # Diagonal stripes only (45° to 135°)
        RandStriped(keys=['image'], prob=0.4, num_stripes_range=(5, 8),
                   thickness_range=(2, 4), intensity_range=(-0.1, 0.1),
                   angle_range=(45, 135), mode='add')
    """

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
                d[key] = self._apply_stripes(d[key])
        return d

    def _apply_stripes(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply random stripe artifacts to image."""
        if img.ndim < 2:
            return img

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            device = img.device
            img = img.clone().cpu().numpy()
        else:
            img = img.copy()

        # Determine angle
        if self.angle_range is not None:
            # Random angle from range
            angle = self.R.uniform(*self.angle_range)
        else:
            # Use predefined orientation
            if self.orientation == "random":
                angle = 0.0 if self.R.rand() > 0.5 else 90.0
            elif self.orientation == "horizontal":
                angle = 0.0
            else:  # vertical
                angle = 90.0

        # Generate number of stripes
        if self.num_stripes_range[0] == self.num_stripes_range[1]:
            num_stripes = self.num_stripes_range[0]
        else:
            num_stripes = self.R.randint(self.num_stripes_range[0], self.num_stripes_range[1] + 1)

        # Apply stripes to all sections if 3D, or to single image if 2D
        if img.ndim == 3:  # 3D volume (Z, Y, X)
            for z in range(img.shape[0]):
                img[z] = self._add_stripes_to_slice(img[z], num_stripes, angle)
        elif img.ndim == 4:  # 4D with channels (C, Z, Y, X)
            for c in range(img.shape[0]):
                for z in range(img.shape[1]):
                    img[c, z] = self._add_stripes_to_slice(img[c, z], num_stripes, angle)
        else:  # 2D image (Y, X)
            img = self._add_stripes_to_slice(img, num_stripes, angle)

        if is_tensor:
            img = torch.from_numpy(img).to(device)

        return img

    def _add_stripes_to_slice(
        self, slice_2d: np.ndarray, num_stripes: int, angle: float
    ) -> np.ndarray:
        """Add stripes at arbitrary angle to a single 2D slice."""
        if slice_2d.ndim != 2:
            return slice_2d

        height, width = slice_2d.shape

        # Convert angle to radians
        angle_rad = np.deg2rad(angle)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]

        # For horizontal stripes (angle=0), we use y coordinates
        # For vertical stripes (angle=90), we use x coordinates
        # For arbitrary angles, we rotate the coordinate system
        # Stripe equation: x*sin(θ) - y*cos(θ) = constant
        rotated_coords = x_coords * np.sin(angle_rad) - y_coords * np.cos(angle_rad)

        # Determine the range of rotated coordinates
        coord_min = rotated_coords.min()
        coord_max = rotated_coords.max()
        coord_range = coord_max - coord_min

        if coord_range == 0:
            return slice_2d

        # Generate random stripe positions
        stripe_positions = []
        for _ in range(num_stripes):
            # Random position along the perpendicular direction
            stripe_center = self.R.uniform(coord_min, coord_max)

            # Random stripe thickness
            if self.thickness_range[0] == self.thickness_range[1]:
                thickness = self.thickness_range[0]
            else:
                thickness = self.R.randint(self.thickness_range[0], self.thickness_range[1] + 1)

            # Random stripe intensity
            intensity = self.R.uniform(*self.intensity_range)

            stripe_positions.append((stripe_center, thickness, intensity))

        # Apply each stripe
        for stripe_center, thickness, intensity in stripe_positions:
            # Create mask for this stripe
            half_thickness = thickness / 2.0
            stripe_mask = np.abs(rotated_coords - stripe_center) <= half_thickness

            # Apply stripe
            if self.mode == "add":
                slice_2d[stripe_mask] = np.clip(slice_2d[stripe_mask] + intensity, 0, 1)
            else:  # replace
                # For replace mode, use absolute intensity value
                if intensity < 0:
                    intensity = 0
                elif intensity > 1:
                    intensity = 1
                slice_2d[stripe_mask] = intensity

        return slice_2d


class ResizeByFactord(MapTransform):
    """
    Resize images by scale factors using MONAI's Resized transform.

    This transform computes the target spatial size based on input size and scale factors,
    then uses MONAI's Resized for the actual resizing operation.

    Args:
        keys: Keys to transform
        scale_factors: Scale factors for each spatial dimension
            (e.g., [0.25, 0.25] for 2D, [0.5, 0.5, 0.5] for 3D)
        mode: Interpolation mode ('bilinear', 'nearest', 'area', etc.)
        align_corners: Whether to align corners (True for bilinear, None for nearest)
        allow_missing_keys: Whether to allow missing keys
    """

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
        from monai.transforms import Resized

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                # Get input spatial shape (excluding channel dimension)
                input_array = d[key]
                if isinstance(input_array, torch.Tensor):
                    spatial_shape = input_array.shape[
                        1:
                    ]  # (C, H, W) -> (H, W) or (C, D, H, W) -> (D, H, W)
                else:
                    spatial_shape = input_array.shape[1:]  # Same for numpy

                # Compute target size
                target_size = [int(s * f) for s, f in zip(spatial_shape, self.scale_factors)]

                # Apply resize using MONAI's Resized
                resizer = Resized(
                    keys=[key],
                    spatial_size=target_size,
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
                d = resizer(d)

        return d


class RandElasticd(MapTransform, RandomizableTransform):
    """
    Unified elastic deformation transform that supports both 2D and 3D data.

    Automatically selects between Rand2DElasticd and Rand3DElasticd based on the do_2d flag
    and input data shape. This provides a consistent API for both 2D and 3D elastic deformations.

    Args:
        keys: Keys of the corresponding items to be transformed (e.g., ["image", "label"])
        do_2d: Whether to apply 2D (True) or 3D (False) elastic deformation.
               Defaults to False (3D behavior).
        prob: Probability of applying the transform. Default: 0.5
        sigma_range: Range for Gaussian smoothing sigma (for 3D) or spacing (for 2D).
                     For 3D: tuple of (min, max) for sigma values
                     For 2D: tuple of (min, max) for pixel spacing
                     Default: (5.0, 8.0)
        magnitude_range: Range for deformation magnitude (displacement in voxels).
                        Tuple of (min, max). Default: (50.0, 150.0)
        allow_missing_keys: Don't raise exception if key is missing. Default: False
        mode: Interpolation mode for image deformation. Default: "bilinear"
        padding_mode: Padding mode for out-of-bounds voxels. Default: "reflection"

    Example:
        >>> # 3D elastic deformation
        >>> elastic_3d = RandElasticd(
        ...     keys=["image", "label"],
        ...     do_2d=False,
        ...     prob=0.5,
        ...     sigma_range=(5.0, 8.0),
        ...     magnitude_range=(50.0, 150.0)
        ... )
        >>>
        >>> # 2D elastic deformation (slice-by-slice)
        >>> elastic_2d = RandElasticd(
        ...     keys=["image", "label"],
        ...     do_2d=True,
        ...     prob=0.5,
        ...     sigma_range=(1.0, 2.0),
        ...     magnitude_range=(10.0, 30.0)
        ... )

    Note:
        - For 3D: Input should have shape (C, D, H, W)
        - For 2D: Input should have shape (C, H, W)
        - The transform automatically adapts based on do_2d flag
    """

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

    def __call__(self, data):
        from monai.transforms import Rand2DElasticd, Rand3DElasticd

        d = dict(data)
        self.randomize(None)

        if not self._do_transform:
            return d

        # Select appropriate transform based on do_2d flag
        if self.do_2d:
            # For 2D elastic deformation
            transform = Rand2DElasticd(
                keys=self.keys,
                prob=1.0,  # Already randomized above
                spacing=self.sigma_range,  # sigma_range used as spacing for 2D
                magnitude_range=self.magnitude_range,
                mode=self.mode,
                padding_mode=self.padding_mode,
                allow_missing_keys=self.allow_missing_keys,
            )
        else:
            # For 3D elastic deformation
            transform = Rand3DElasticd(
                keys=self.keys,
                prob=1.0,  # Already randomized above
                sigma_range=self.sigma_range,
                magnitude_range=self.magnitude_range,
                mode=self.mode,
                padding_mode=self.padding_mode,
                allow_missing_keys=self.allow_missing_keys,
            )

        return transform(d)
