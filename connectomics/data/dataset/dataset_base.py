"""
MONAI-native dataset module for PyTorch Connectomics.

This module provides a comprehensive dataset infrastructure built on MONAI datasets
with PyTorch Lightning DataModule functionality for different data loading scenarios.
The legacy dataset creation functions have been removed to keep the codebase clean.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Callable, Sequence, Tuple
from abc import ABC, abstractmethod
import os
import numpy as np

import torch
import torch.utils.data
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# MONAI imports
from monai.data import Dataset, CacheDataset, PersistentDataset
from monai.transforms import Compose
from monai.utils import ensure_tuple_rep

from ...transforms.process import create_binary_segmentation_pipeline, create_affinity_segmentation_pipeline
from ..io import read_volume


class MonaiConnectomicsDataset(Dataset):
    """
    MONAI-native base dataset for all connectomics data loading scenarios.

    This class extends MONAI's Dataset with connectomics-specific functionality:
    - Volume-based loading with sampling strategies
    - Tile-based loading for large-scale data
    - Cloud-based loading support
    - Advanced rejection sampling for training
    - Modern MONAI Compose transform pipelines

    Args:
        data_dicts (Sequence[Dict]): List of data dictionaries with 'image' and optionally 'label' keys
        transforms (Compose, optional): MONAI transforms pipeline
        sample_size (Tuple[int, ...]): Size of samples to extract (z, y, x)
        mode (str): Dataset mode ('train', 'val', 'test'). Default: 'train'
        iter_num (int): Number of iterations per epoch (-1 for inference). Default: -1
        valid_ratio (float): Volume ratio threshold for valid samples. Default: 0.5
        reject_size_thres (int): Threshold for foreground objects. Default: 0
        reject_diversity (int): Threshold for multiple objects. Default: 0
        reject_p (float): Probability of rejecting non-foreground volumes. Default: 0.95
        do_2d (bool): Load 2D samples from 3D volumes. Default: False
        do_relabel (bool): Reduce mask indices in sampled volumes. Default: True
        cache_rate (float): MONAI cache rate for performance. Default: 0.0
    """

    def __init__(
        self,
        data_dicts: Sequence[Dict[str, Any]],
        transforms: Optional[Compose] = None,
        sample_size: Tuple[int, int, int] = (32, 256, 256),
        mode: str = 'train',
        iter_num: int = -1,
        valid_ratio: float = 0.5,
        reject_size_thres: int = 0,
        reject_diversity: int = 0,
        reject_p: float = 0.95,
        do_2d: bool = False,
        do_relabel: bool = True,
        cache_rate: float = 0.0,
    ):
        # Initialize MONAI Dataset
        super().__init__(data=data_dicts, transform=transforms)

        # Store connectomics-specific parameters
        self.sample_size = ensure_tuple_rep(sample_size, 3)
        self.mode = mode
        self.iter_num = iter_num
        self.valid_ratio = valid_ratio
        self.reject_size_thres = reject_size_thres
        self.reject_diversity = reject_diversity
        self.reject_p = reject_p
        self.do_2d = do_2d
        self.do_relabel = do_relabel

        # Calculate dataset length for training
        if iter_num > 0:
            self.dataset_length = iter_num
        else:
            self.dataset_length = len(data_dicts)

    def __len__(self) -> int:
        """Return dataset length based on mode and iteration settings."""
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data sample with MONAI transforms applied.

        For training mode, implements rejection sampling to increase
        foreground sample frequency.
        """
        if self.mode == 'train' and self.iter_num > 0:
            # Rejection sampling for training
            return self._get_sample_with_rejection(index)
        else:
            # Standard MONAI dataset behavior
            return super().__getitem__(index % len(self.data))

    def _get_sample_with_rejection(self, index: int) -> Dict[str, Any]:
        """
        Implement rejection sampling to favor foreground regions.

        This method repeatedly samples until finding a valid sample
        based on the rejection criteria.
        """
        max_attempts = 50

        for _ in range(max_attempts):
            # Random sample from data
            data_index = np.random.randint(len(self.data))
            sample = super().__getitem__(data_index)

            # Check if sample meets quality criteria
            if self._is_valid_sample(sample):
                return sample

        # Fallback: return last sample if no valid sample found
        return sample

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a sample meets the validity criteria for training.

        Args:
            sample: Sample dictionary from MONAI transforms

        Returns:
            bool: True if sample is valid for training
        """
        # If no label, always accept (unsupervised case)
        if 'label' not in sample:
            return True

        label = sample['label']

        # If label is still a string (path), skip validation (no transforms applied)
        if isinstance(label, str):
            return True

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        # Check foreground ratio
        if self.valid_ratio > 0:
            foreground_ratio = np.sum(label > 0) / label.size
            if foreground_ratio < self.valid_ratio:
                if np.random.rand() > (1 - self.reject_p):
                    return False

        # Check foreground size threshold
        if self.reject_size_thres > 0:
            if np.sum(label > 0) < self.reject_size_thres:
                if np.random.rand() > (1 - self.reject_p):
                    return False

        # Check object diversity
        if self.reject_diversity > 0:
            unique_labels = len(np.unique(label)) - 1  # Exclude background
            if unique_labels < self.reject_diversity:
                if np.random.rand() > (1 - self.reject_p):
                    return False

        return True


class MonaiCachedConnectomicsDataset(CacheDataset):
    """
    MONAI CacheDataset-based implementation for improved performance.

    This dataset caches transformed data in memory for faster training.
    Suitable for smaller datasets that fit in memory.

    Args:
        data_dicts (Sequence[Dict]): List of data dictionaries
        transforms (Compose, optional): MONAI transforms pipeline
        cache_rate (float): Percentage of data to cache. Default: 1.0
        num_workers (int): Number of workers for caching. Default: 0
        **kwargs: Additional arguments passed to MonaiConnectomicsDataset
    """

    def __init__(
        self,
        data_dicts: Sequence[Dict[str, Any]],
        transforms: Optional[Compose] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        **kwargs,
    ):
        # Initialize MONAI CacheDataset
        super().__init__(
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # Store connectomics parameters
        self.connectomics_params = kwargs
        self.sample_size = kwargs.get('sample_size', (32, 256, 256))
        self.mode = kwargs.get('mode', 'train')
        self.iter_num = kwargs.get('iter_num', -1)

        # Calculate dataset length
        if self.iter_num > 0:
            self.dataset_length = self.iter_num
        else:
            self.dataset_length = len(data_dicts)

    def __len__(self) -> int:
        return self.dataset_length


class MonaiPersistentConnectomicsDataset(PersistentDataset):
    """
    MONAI PersistentDataset-based implementation for very large datasets.

    This dataset persists transformed data to disk for memory efficiency
    with large datasets that don't fit in memory.

    Args:
        data_dicts (Sequence[Dict]): List of data dictionaries
        transforms (Compose, optional): MONAI transforms pipeline
        cache_dir (str): Directory for persistent cache
        **kwargs: Additional arguments passed to MonaiConnectomicsDataset
    """

    def __init__(
        self,
        data_dicts: Sequence[Dict[str, Any]],
        transforms: Optional[Compose] = None,
        cache_dir: str = './cache',
        **kwargs,
    ):
        # Initialize MONAI PersistentDataset
        super().__init__(
            data=data_dicts,
            transform=transforms,
            cache_dir=cache_dir,
        )

        # Store connectomics parameters
        self.connectomics_params = kwargs
        self.sample_size = kwargs.get('sample_size', (32, 256, 256))
        self.mode = kwargs.get('mode', 'train')
        self.iter_num = kwargs.get('iter_num', -1)

        # Calculate dataset length
        if self.iter_num > 0:
            self.dataset_length = self.iter_num
        else:
            self.dataset_length = len(data_dicts)

    def __len__(self) -> int:
        return self.dataset_length


def create_data_dicts_from_paths(
    image_paths: List[str],
    label_paths: Optional[List[str]] = None,
    mask_paths: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create MONAI-style data dictionaries from file paths.

    Args:
        image_paths: List of image file paths
        label_paths: Optional list of label file paths
        mask_paths: Optional list of mask file paths

    Returns:
        List of dictionaries with 'image', 'label', and/or 'mask' keys
    """
    data_dicts = []

    for i, image_path in enumerate(image_paths):
        data_dict = {'image': image_path}

        if label_paths is not None:
            data_dict['label'] = label_paths[i]

        if mask_paths is not None:
            data_dict['mask'] = mask_paths[i]

        data_dicts.append(data_dict)

    return data_dicts


def create_connectomics_dataset(
    data_dicts: Sequence[Dict[str, Any]],
    transforms: Optional[Compose] = None,
    dataset_type: str = 'standard',
    **kwargs,
) -> Union[MonaiConnectomicsDataset, MonaiCachedConnectomicsDataset, MonaiPersistentConnectomicsDataset]:
    """
    Factory function to create appropriate MONAI connectomics dataset.

    Args:
        data_dicts: List of data dictionaries
        transforms: MONAI transforms pipeline
        dataset_type: Type of dataset ('standard', 'cached', 'persistent')
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Appropriate MONAI connectomics dataset instance
    """
    if dataset_type == 'cached':
        return MonaiCachedConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )
    elif dataset_type == 'persistent':
        return MonaiPersistentConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )
    else:
        return MonaiConnectomicsDataset(
            data_dicts=data_dicts,
            transforms=transforms,
            **kwargs,
        )


__all__ = [
    'MonaiConnectomicsDataset',
    'MonaiCachedConnectomicsDataset',
    'MonaiPersistentConnectomicsDataset',
    'create_data_dicts_from_paths',
    'create_connectomics_dataset',
]