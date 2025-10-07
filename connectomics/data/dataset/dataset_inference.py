"""
Inference dataset for sliding-window prediction with patch blending.

This module provides specialized dataset classes for inference that generate
patches using stride-based grid sampling with position metadata for blending.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

import torch
from monai.data import Dataset
from monai.transforms import Compose

from ..utils.sampling import calculate_inference_grid
from ..io import read_image_file


class InferenceVolumeDataset(Dataset):
    """
    Dataset for inference with sliding-window sampling and position tracking.

    This dataset generates patches from volumes using a stride-based grid,
    returning position metadata required for patch blending.

    Args:
        image_paths: List of image volume file paths
        label_paths: Optional list of label volume file paths
        transforms: MONAI transforms pipeline
        patch_size: Size of patches to extract (D, H, W)
        stride: Stride between patch centers (D, H, W)
        preload: Whether to preload volumes into memory. Default: True

    Returns:
        Dictionary containing:
        - 'image': Patch tensor of shape (C, D, H, W)
        - 'pos': Position tensor [vol_id, z, y, x]
        - 'label': Optional label patch (if labels provided)
        - 'vol_shape': Original volume shape (for buffer initialization)

    Examples:
        >>> from connectomics.data.dataset import InferenceVolumeDataset
        >>> dataset = InferenceVolumeDataset(
        ...     image_paths=['test.tif'],
        ...     patch_size=(112, 112, 112),
        ...     stride=(56, 56, 56),
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # (1, 112, 112, 112)
        >>> print(sample['pos'])  # [0, 0, 0, 0] for first patch
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        transforms: Optional[Compose] = None,
        patch_size: Tuple[int, int, int] = (112, 112, 112),
        stride: Tuple[int, int, int] = (56, 56, 56),
        preload: bool = True,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.preload = preload

        # Load or get volume shapes
        self.volumes = []
        self.labels = []
        self.volume_shapes = []

        for i, img_path in enumerate(image_paths):
            # Load image volume
            img_data = read_image_file(img_path)

            # Handle channel dimension
            if img_data.ndim == 3:
                img_data = img_data[np.newaxis, :]  # Add channel: (C, D, H, W)

            self.volume_shapes.append(img_data.shape[1:])  # Store (D, H, W)

            if preload:
                # Convert to float32 and normalize
                img_data = img_data.astype(np.float32) / 255.0
                self.volumes.append(img_data)
            else:
                self.volumes.append(img_path)

            # Load label if provided
            if label_paths and i < len(label_paths) and label_paths[i]:
                label_data = read_image_file(label_paths[i])
                if label_data.ndim == 3:
                    label_data = label_data[np.newaxis, :]
                if preload:
                    label_data = label_data.astype(np.float32)
                    self.labels.append(label_data)
                else:
                    self.labels.append(label_paths[i])
            else:
                self.labels.append(None)

        # Calculate all patch positions
        self.positions = []
        self.position_to_volume = []

        for vol_id, vol_shape in enumerate(self.volume_shapes):
            positions, grid_shape = calculate_inference_grid(
                vol_shape,
                patch_size,
                stride
            )

            print(f"Volume {vol_id}: shape={vol_shape}, "
                  f"grid={grid_shape}, patches={len(positions)}")

            # Store positions with volume ID
            for pos in positions:
                self.positions.append(np.array([vol_id] + list(pos), dtype=np.int32))
                self.position_to_volume.append(vol_id)

        # Initialize base Dataset with dummy data_dicts
        data_dicts = [{'index': i} for i in range(len(self.positions))]
        super().__init__(data=data_dicts, transform=None)

        print(f"InferenceVolumeDataset initialized:")
        print(f"  Volumes: {len(self.volumes)}")
        print(f"  Total patches: {len(self.positions)}")
        print(f"  Patch size: {patch_size}")
        print(f"  Stride: {stride}")

    def __len__(self) -> int:
        """Return total number of patches."""
        return len(self.positions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a patch with position metadata.

        Args:
            index: Patch index

        Returns:
            Dictionary with 'image', 'pos', and optionally 'label'
        """
        # Get position [vol_id, z, y, x]
        pos = self.positions[index]
        vol_id = int(pos[0])
        z, y, x = int(pos[1]), int(pos[2]), int(pos[3])

        # Load volume if not preloaded
        if self.preload:
            volume = self.volumes[vol_id]
        else:
            volume = read_image_file(self.volumes[vol_id])
            if volume.ndim == 3:
                volume = volume[np.newaxis, :]
            volume = volume.astype(np.float32) / 255.0

        # Extract patch
        d, h, w = self.patch_size
        patch = volume[:, z:z+d, y:y+h, x:x+w]

        # Create data dict
        data = {
            'image': torch.from_numpy(patch.copy()),
            'pos': torch.from_numpy(pos.copy()),
            'vol_shape': torch.tensor(self.volume_shapes[vol_id]),
        }

        # Add label if available
        if self.labels[vol_id] is not None:
            if self.preload:
                label_volume = self.labels[vol_id]
            else:
                label_volume = read_image_file(self.labels[vol_id])
                if label_volume.ndim == 3:
                    label_volume = label_volume[np.newaxis, :]
                label_volume = label_volume.astype(np.float32)

            label_patch = label_volume[:, z:z+d, y:y+h, x:x+w]
            data['label'] = torch.from_numpy(label_patch.copy())

        # Apply transforms if provided
        if self.transforms is not None:
            data = self.transforms(data)

        return data


__all__ = [
    'InferenceVolumeDataset',
]
