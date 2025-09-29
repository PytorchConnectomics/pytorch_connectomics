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

    def _apply_misalignment_translation(self, img: np.ndarray) -> np.ndarray:
        """Apply translation-based misalignment."""
        if img.ndim < 3:
            return img

        out_shape = (img.shape[0],
                    img.shape[1] - self.displacement,
                    img.shape[2] - self.displacement)

        x0 = self.R.randint(self.displacement)
        y0 = self.R.randint(self.displacement)
        x1 = self.R.randint(self.displacement)
        y1 = self.R.randint(self.displacement)
        idx = self.R.choice(np.arange(1, out_shape[0] - 1))
        mode = 'slip' if self.R.rand() < 0.5 else 'translation'

        output = np.zeros(out_shape, img.dtype)
        if mode == 'slip':
            output = img[:, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            output[idx] = img[idx, y1:y1+out_shape[1], x1:x1+out_shape[2]]
        else:
            output[:idx] = img[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
            output[idx:] = img[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]

        return output

    def _apply_misalignment_rotation(self, img: np.ndarray) -> np.ndarray:
        """Apply rotation-based misalignment."""
        if img.ndim < 3:
            return img

        height, width = img.shape[-2:]
        if height != width:
            return img  # Skip if not square

        # Generate rotation matrix
        x = self.displacement / 2.0
        y = ((height - self.displacement) / 2.0) * 1.42
        angle = math.asin(x/y) * 2.0 * 57.2958
        rand_angle = (self.R.rand() - 0.5) * 2.0 * angle
        M = cv2.getRotationMatrix2D((height/2, height/2), rand_angle, 1)

        idx = self.R.choice(np.arange(1, img.shape[0] - 1))
        mode = 'slip' if self.R.rand() < 0.5 else 'translation'

        interpolation = cv2.INTER_LINEAR if img.dtype == np.float32 else cv2.INTER_NEAREST

        if mode == 'slip':
            img[idx] = cv2.warpAffine(img[idx], M, (height, width),
                                     flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        else:
            for i in range(idx, img.shape[0]):
                img[i] = cv2.warpAffine(img[i], M, (height, width),
                                       flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
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

    def _apply_missing_section(self, img: np.ndarray) -> np.ndarray:
        """Remove random sections from volume."""
        if img.ndim < 3 or img.shape[0] <= 3:
            return img  # Skip 2D or very small volumes

        # Select sections to remove (avoid first and last)
        num_to_remove = min(self.num_sections, img.shape[0] - 2)
        indices_to_remove = self.R.choice(
            np.arange(1, img.shape[0] - 1),
            size=num_to_remove,
            replace=False
        )

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

    def _apply_missing_parts(self, img: np.ndarray) -> np.ndarray:
        """Create missing rectangular regions."""
        if img.ndim < 3:
            return img

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
        img[section_idx, y_start:y_start+hole_h, x_start:x_start+hole_w] = 0

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

    def _apply_motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply motion blur to random sections."""
        if img.ndim < 3:
            return img

        img = img.copy()

        # Generate motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        if self.R.rand() > 0.5:  # horizontal kernel
            kernel[int((self.kernel_size-1)/2), :] = np.ones(self.kernel_size)
        else:  # vertical kernel
            kernel[:, int((self.kernel_size-1)/2)] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size

        # Select sections to blur
        if isinstance(self.sections, tuple):
            num_sections = self.R.randint(*self.sections)
        else:
            num_sections = self.sections

        num_sections = min(num_sections, img.shape[0])
        section_indices = self.R.choice(img.shape[0], size=num_sections, replace=False)

        # Apply blur
        for idx in section_indices:
            img[idx] = cv2.filter2D(img[idx], -1, kernel)

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

    def _apply_cut_noise(self, img: np.ndarray) -> np.ndarray:
        """Add noise to random cuboid region."""
        if img.ndim < 2:
            return img

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

        # Generate noise
        if img.ndim == 3:
            noise_shape = (z_len, y_len, x_len)
            noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
            region = img[z_start:z_start+z_len, y_start:y_start+y_len, x_start:x_start+x_len]
            noisy_region = np.clip(region + noise, 0, 1)
            img[z_start:z_start+z_len, y_start:y_start+y_len, x_start:x_start+x_len] = noisy_region
        else:
            noise_shape = (y_len, x_len)
            noise = self.R.uniform(-self.noise_scale, self.noise_scale, noise_shape)
            region = img[y_start:y_start+y_len, x_start:x_start+x_len]
            noisy_region = np.clip(region + noise, 0, 1)
            img[y_start:y_start+y_len, x_start:x_start+x_len] = noisy_region

        return img


# Note: Standard transforms like Grayscale, Elastic, and Rescale are available in MONAI
# as RandShiftIntensityd, RandAdjustContrastd, Rand3DElasticd, and RandZoomd
# We only implement connectomics-specific transforms that aren't available in MONAI


class RandCutBlurd(RandomizableTransform, MapTransform):
    """
    Random CutBlur augmentation for connectomics data.

    Randomly downsample cuboid regions to force super-resolution learning.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
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

        for key in self.key_iterator(d):
            if key in d:
                d[key] = self._apply_cutblur(d[key])
        return d

    def _apply_cutblur(self, img: np.ndarray) -> np.ndarray:
        """Apply CutBlur transformation."""
        from skimage.transform import resize

        img = img.copy()

        # Generate cuboid region
        if img.ndim == 3:
            z_len = int(self.length_ratio * img.shape[0]) if self.downsample_z else 1
            y_len = int(self.length_ratio * img.shape[1])
            x_len = int(self.length_ratio * img.shape[2])

            z_start = self.R.randint(0, img.shape[0] - z_len + 1) if self.downsample_z else self.R.randint(0, img.shape[0])
            y_start = self.R.randint(0, img.shape[1] - y_len + 1)
            x_start = self.R.randint(0, img.shape[2] - x_len + 1)
        else:
            z_len = z_start = 0
            y_len = int(self.length_ratio * img.shape[0])
            x_len = int(self.length_ratio * img.shape[1])
            y_start = self.R.randint(0, img.shape[0] - y_len + 1)
            x_start = self.R.randint(0, img.shape[1] - x_len + 1)

        # Generate downsampling ratio
        down_ratio = self.R.uniform(*self.down_ratio_range)

        # Extract and downsample region
        if img.ndim == 3:
            if self.downsample_z:
                region = img[z_start:z_start+z_len, y_start:y_start+y_len, x_start:x_start+x_len]
                down_shape = (int(z_len/down_ratio), int(y_len/down_ratio), int(x_len/down_ratio))
            else:
                region = img[z_start, y_start:y_start+y_len, x_start:x_start+x_len]
                down_shape = (int(y_len/down_ratio), int(x_len/down_ratio))
        else:
            region = img[y_start:y_start+y_len, x_start:x_start+x_len]
            down_shape = (int(y_len/down_ratio), int(x_len/down_ratio))

        # Downsample and upsample
        region_down = resize(region, down_shape, order=1, preserve_range=True, anti_aliasing=True)
        region_up = resize(region_down, region.shape, order=1, preserve_range=True, anti_aliasing=False)

        # Put back into image
        if img.ndim == 3:
            if self.downsample_z:
                img[z_start:z_start+z_len, y_start:y_start+y_len, x_start:x_start+x_len] = region_up
            else:
                img[z_start, y_start:y_start+y_len, x_start:x_start+x_len] = region_up
        else:
            img[y_start:y_start+y_len, x_start:x_start+x_len] = region_up

        return img


__all__ = [
    # Connectomics-specific transforms (not available in standard MONAI)
    'RandMisAlignmentd',
    'RandMissingSectiond',
    'RandMissingPartsd',
    'RandMotionBlurd',
    'RandCutNoised',
    'RandCutBlurd',
]