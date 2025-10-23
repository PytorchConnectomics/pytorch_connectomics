"""
Complete MONAI-native transforms for connectomics-specific augmentations.

This module provides fully implemented MONAI MapTransform versions of connectomics
augmentations, replacing the legacy custom augmentor system entirely.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Tuple
import math
import numpy as np
import torch
import cv2
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils import ensure_tuple_rep


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
        d = dict(data)
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

        interpolation = (
            cv2.INTER_LINEAR if img.dtype == np.float32 else cv2.INTER_NEAREST
        )

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
        d = dict(data)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_missing_section(d[key])
        return d

    def _apply_missing_section(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Remove random sections from volume."""
        if img.ndim < 3 or img.shape[0] <= 3:
            return img  # Skip 2D or very small volumes

        # Handle both numpy and torch tensors
        is_tensor = isinstance(img, torch.Tensor)

        # Select sections to remove (avoid first and last)
        num_to_remove = min(self.num_sections, img.shape[0] - 2)
        indices_to_remove = self.R.choice(
            np.arange(1, img.shape[0] - 1), size=num_to_remove, replace=False
        )

        if is_tensor:
            # Keep sections that are NOT in indices_to_remove
            keep_mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
            keep_mask[indices_to_remove] = False
            return img[keep_mask]
        else:
            return np.delete(img, indices_to_remove, axis=0)


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

        # Generate cuboid dimensions and position
        if img.ndim == 3:
            z_len = int(self.length_ratio * img.shape[0])
            z_start = self.R.randint(0, img.shape[0] - z_len + 1)
        else:
            z_len = z_start = 0

        y_len = int(self.length_ratio * img.shape[-2])
        x_len = int(self.length_ratio * img.shape[-1])
        y_start = self.R.randint(0, img.shape[-2] - y_len + 1)
        x_start = self.R.randint(0, img.shape[-1] - x_len + 1)

        # Generate noise (numpy operations)
        if img.ndim == 3:
            noise_shape = (z_len, y_len, x_len)
            noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
            region = img[
                z_start : z_start + z_len,
                y_start : y_start + y_len,
                x_start : x_start + x_len,
            ]
            noisy_region = np.clip(region + noise, 0, 1)
            img[
                z_start : z_start + z_len,
                y_start : y_start + y_len,
                x_start : x_start + x_len,
            ] = noisy_region
        else:
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
                out_shape = np.array(temp.shape) / np.array(
                    [1, down_ratio, down_ratio, down_ratio]
                )
            else:
                out_shape = np.array(temp.shape) / np.array(
                    [1, 1, down_ratio, down_ratio]
                )
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
        zoom_factors = [
            out_size / in_size for out_size, in_size in zip(out_shape, temp.shape)
        ]
        downsampled = zoom(temp, zoom_factors, order=1, mode="reflect", prefilter=True)

        # Upsample with nearest neighbor (no anti-aliasing to preserve sharp edges)
        zoom_factors = [
            out_size / in_size
            for out_size, in_size in zip(temp.shape, downsampled.shape)
        ]
        upsampled = zoom(
            downsampled, zoom_factors, order=0, mode="reflect", prefilter=False
        )

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
            return volume.numpy() if is_numpy else volume, (
                label.numpy() if is_numpy else label
            )

        # Create flipped version for pasting
        label_flipped = label.flip(0)  # Flip along z-axis

        # Extract object
        if volume.ndim == 4:
            neuron_tensor = volume * label.unsqueeze(0)
        else:
            neuron_tensor = volume * label

        # Find best rotation and position
        neuron_tensor, label_paste = self._find_best_paste(
            neuron_tensor, label, label_flipped
        )

        # Paste into image
        if volume.ndim == 4:
            label_paste = label_paste.unsqueeze(0)
            volume = volume * (~label_paste) + neuron_tensor * label_paste
        else:
            volume = volume * (~label_paste) + neuron_tensor * label_paste

        if is_numpy:
            return volume.numpy(), (
                label_paste.squeeze().numpy()
                if label_paste.ndim > 3
                else label_paste.numpy()
            )
        return volume, label_paste.squeeze() if label_paste.ndim > 3 else label_paste

    def _find_best_paste(
        self,
        neuron_tensor: torch.Tensor,
        label_orig: torch.Tensor,
        label_flipped: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best rotation and position with minimal overlap."""
        import torchvision.transforms.functional as tf
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
                neuron_tensor.flip(0)
                if neuron_tensor.ndim == 3
                else neuron_tensor.flip(1)
            )

        label_paste = labels[best_idx : best_idx + 1]

        if best_angle != 0:
            label_paste = self._rotate_3d(label_paste, best_angle)
            if neuron_tensor.ndim == 4:
                neuron_tensor = self._rotate_3d(
                    neuron_tensor.unsqueeze(0), best_angle
                ).squeeze(0)
            else:
                neuron_tensor = self._rotate_3d(
                    neuron_tensor.unsqueeze(0), best_angle
                ).squeeze(0)

        label_paste = label_paste.squeeze(0)

        # Crop overlapping regions
        gt_dilated = torch.tensor(
            binary_dilation(
                label_orig.numpy(), structure=self.dil_struct, iterations=self.border
            )
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
                    d[key] = d[key].to(
                        dtype=torch.from_numpy(np.array([], dtype=self.dtype)).dtype
                    )
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

    Percentile clipping is applied BEFORE normalization when low > 0.0 or high < 1.0.

    Args:
        keys: Keys to normalize
        mode: Normalization mode ("none", "normal", "0-1")
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
        if mode not in ["none", "normal", "0-1"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'none', 'normal', or '0-1'"
            )
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
    "ConvertToFloatd",
    "NormalizeLabelsd",
    "SmartNormalizeIntensityd",
    "ResizeByFactord",
]


class ResizeByFactord(MapTransform):
    """
    Resize images by scale factors using MONAI's Resized transform.

    This transform computes the target spatial size based on input size and scale factors,
    then uses MONAI's Resized for the actual resizing operation.

    Args:
        keys: Keys to transform
        scale_factors: Scale factors for each spatial dimension (e.g., [0.25, 0.25] for 2D, [0.5, 0.5, 0.5] for 3D)
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
                target_size = [
                    int(s * f) for s, f in zip(spatial_shape, self.scale_factors)
                ]

                # Apply resize using MONAI's Resized
                resizer = Resized(
                    keys=[key],
                    spatial_size=target_size,
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
                d = resizer(d)

        return d
