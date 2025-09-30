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
    
    def _apply_mixup(self, volume: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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
        label_key: str = 'label',
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
        from scipy.ndimage.morphology import generate_binary_structure
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
                d[key], d[self.label_key] = self._apply_copy_paste(
                    d[key], label
                )
        return d
    
    def _apply_copy_paste(
        self, 
        volume: Union[np.ndarray, torch.Tensor],
        label: Union[np.ndarray, torch.Tensor]
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
            return volume.numpy() if is_numpy else volume, label.numpy() if is_numpy else label
            
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
            return volume.numpy(), label_paste.squeeze().numpy() if label_paste.ndim > 3 else label_paste.numpy()
        return volume, label_paste.squeeze() if label_paste.ndim > 3 else label_paste
    
    def _find_best_paste(
        self,
        neuron_tensor: torch.Tensor,
        label_orig: torch.Tensor,
        label_flipped: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best rotation and position with minimal overlap."""
        import torchvision.transforms.functional as tf
        from scipy.ndimage.morphology import binary_dilation
        
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
            neuron_tensor = neuron_tensor.flip(0) if neuron_tensor.ndim == 3 else neuron_tensor.flip(1)
            
        label_paste = labels[best_idx:best_idx+1]
        
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
            reshaped = tensor.reshape(1, c*z, y, x)
            rotated = tf.rotate(reshaped, angle)
            return rotated.reshape(c, z, y, x)
        elif tensor.ndim == 5:  # (B, C, Z, Y, X)
            b, c, z, y, x = tensor.shape
            rotated_list = []
            for i in range(b):
                reshaped = tensor[i].reshape(1, c*z, y, x)
                rot = tf.rotate(reshaped, angle)
                rotated_list.append(rot.reshape(c, z, y, x))
            return torch.stack(rotated_list)
        else:
            return tensor


__all__ = [
    # Connectomics-specific transforms (not available in standard MONAI)
    'RandMisAlignmentd',
    'RandMissingSectiond',
    'RandMissingPartsd',
    'RandMotionBlurd',
    'RandCutNoised',
    'RandCutBlurd',
    'RandMixupd',
    'RandCopyPasted',
]