"""
MONAI-native volume dataset for PyTorch Connectomics.

This module provides volume-based dataset classes using MONAI's native dataset
infrastructure with connectomics-specific sampling and augmentation strategies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from monai.data import CacheDataset
from monai.transforms import CenterSpatialCropd, Compose, RandSpatialCropd
from monai.utils import ensure_tuple_rep

from ..io.monai_transforms import LoadVolumed
from .data_dicts import create_data_dicts_from_paths
from .dataset_base import MonaiConnectomicsDataset


class MonaiVolumeDataset(MonaiConnectomicsDataset):
    """
    MONAI-native dataset for volumetric connectomics data.

    This class extends the base MONAI connectomics dataset with volume-specific
    functionality including:
    - Random spatial cropping for training
    - Sliding window sampling for inference
    - Rejection sampling based on foreground content
    - Support for multiple volumes with different sizes

    Args:
        image_paths (List[str]): List of image volume file paths
        label_paths (List[str], optional): List of label volume file paths
        mask_paths (List[str], optional): List of valid mask file paths
        transforms (Compose, optional): MONAI transforms pipeline
        sample_size (Tuple[int, int, int]): Size of samples to extract (z, y, x)
        mode (str): Dataset mode ('train', 'val', 'test'). Default: 'train'
        iter_num (int): Number of iterations per epoch (-1 for inference). Default: -1
        valid_ratio (float): Volume ratio threshold for valid samples. Default: 0.5
        reject_size_thres (int): Threshold for foreground objects. Default: 0
        reject_diversity (int): Threshold for multiple objects. Default: 0
        reject_p (float): Probability of rejecting non-foreground volumes. Default: 0.95
        do_2d (bool): Load 2D samples from 3D volumes. Default: False
        do_relabel (bool): Reduce mask indices in sampled volumes. Default: True
        data_mean (float): Mean for normalization. Default: 0.5
        data_std (float): Standard deviation for normalization. Default: 0.5
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        transforms: Optional[Compose] = None,
        sample_size: Tuple[int, int, int] = (32, 256, 256),
        mode: str = "train",
        iter_num: int = -1,
        valid_ratio: float = 0.5,
        reject_size_thres: int = 0,
        reject_diversity: int = 0,
        reject_p: float = 0.95,
        do_2d: bool = False,
        do_relabel: bool = True,
        data_mean: float = 0.5,
        data_std: float = 0.5,
        transpose_axes: Optional[List[int]] = None,
        **kwargs,
    ):
        # Create MONAI data dictionaries
        data_dicts = create_data_dicts_from_paths(
            image_paths=image_paths,
            label_paths=label_paths,
            mask_paths=mask_paths,
        )

        # Store data dictionaries temporarily for transform creation
        self._data_dicts = data_dicts
        self._transpose_axes = transpose_axes

        # Create transforms if not provided
        if transforms is None:
            transforms = self._create_default_transforms(
                sample_size=sample_size,
                mode=mode,
                do_2d=do_2d,
                data_mean=data_mean,
                data_std=data_std,
                transpose_axes=transpose_axes,
            )

        # Initialize base dataset
        super().__init__(
            data_dicts=data_dicts,
            transforms=transforms,
            sample_size=sample_size,
            mode=mode,
            iter_num=iter_num,
            valid_ratio=valid_ratio,
            reject_size_thres=reject_size_thres,
            reject_diversity=reject_diversity,
            reject_p=reject_p,
            do_2d=do_2d,
            do_relabel=do_relabel,
            **kwargs,
        )

        # Clean up temporary references
        delattr(self, "_data_dicts")
        delattr(self, "_transpose_axes")

        # Store volume-specific parameters
        self.data_mean = data_mean
        self.data_std = data_std

    def _create_default_transforms(
        self,
        sample_size: Tuple[int, int, int],
        mode: str,
        do_2d: bool,
        data_mean: float,
        data_std: float,
        transpose_axes: Optional[List[int]] = None,
    ) -> Compose:
        """
        Create default MONAI transforms pipeline for volume data.

        Args:
            sample_size: Size of samples to extract
            mode: Dataset mode ('train', 'val', 'test')
            do_2d: Whether to extract 2D samples
            data_mean: Mean for normalization
            data_std: Standard deviation for normalization
            transpose_axes: Axis permutation for transposing loaded volumes

        Returns:
            MONAI Compose transforms pipeline
        """
        keys = ["image"]
        if any("label" in data_dict for data_dict in self._data_dicts):
            keys.append("label")
        if any("mask" in data_dict for data_dict in self._data_dicts):
            keys.append("mask")

        transforms = [
            # Load images using custom connectomics loader (adds channel dim)
            LoadVolumed(keys=keys, transpose_axes=transpose_axes),
        ]

        # Add spatial cropping for both training and validation
        # This prevents loading full volumes which cause OOM errors
        crop_size = sample_size
        if do_2d:
            crop_size = (1, sample_size[1], sample_size[2])

        if mode == "train":
            # Random cropping for training
            transforms.append(
                RandSpatialCropd(
                    keys=keys,
                    roi_size=crop_size,
                    random_center=True,
                    random_size=False,
                )
            )
        elif mode == "val":
            # Center cropping for validation to ensure consistent patch size
            transforms.append(
                CenterSpatialCropd(
                    keys=keys,
                    roi_size=crop_size,
                )
            )
        # For test mode, return full volumes to enable sliding-window inference

        # TODO: Add normalization transforms here if needed
        # Could use ScaleIntensityd, NormalizeIntensityd, etc.

        return Compose(transforms)


class MonaiCachedVolumeDataset(CacheDataset):
    """
    Cached version of MONAI volume dataset for improved performance.

    This dataset caches transformed volumes in memory for faster access
    during training. Suitable for datasets that fit in available memory.

    Args:
        cache_rate (float): Percentage of data to cache. Default: 1.0
        num_workers (int): Number of workers for caching. Default: 0
        **kwargs: Arguments passed to dataset creation
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        transforms: Optional[Compose] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        **kwargs,
    ):
        # Get parameters for dataset creation
        sample_size = kwargs.get("sample_size", (32, 256, 256))
        mode = kwargs.get("mode", "train")
        do_2d = kwargs.get("do_2d", False)
        kwargs.get("data_mean", 0.5)
        kwargs.get("data_std", 0.5)
        transpose_axes = kwargs.get("transpose_axes", None)

        # Create data dictionaries
        data_dicts = create_data_dicts_from_paths(
            image_paths=image_paths,
            label_paths=label_paths,
            mask_paths=mask_paths,
        )

        # Create transforms if not provided
        if transforms is None:
            keys = ["image"]
            if label_paths:
                keys.append("label")
            if mask_paths:
                keys.append("mask")

            transforms = [
                # Use custom connectomics loader (adds channel dim)
                LoadVolumed(keys=keys, transpose_axes=transpose_axes),
            ]

            # Add spatial cropping for both training and validation
            crop_size = sample_size
            if do_2d:
                crop_size = (1, sample_size[1], sample_size[2])

            if mode == "train":
                # Random cropping for training
                transforms.append(
                    RandSpatialCropd(
                        keys=keys,
                        roi_size=crop_size,
                        random_center=True,
                        random_size=False,
                    )
                )
            elif mode == "val":
                # Center cropping for validation
                transforms.append(
                    CenterSpatialCropd(
                        keys=keys,
                        roi_size=crop_size,
                    )
                )
            # For test mode, return full volumes to enable sliding-window inference

            transforms = Compose(transforms)

        # Initialize MONAI CacheDataset
        CacheDataset.__init__(
            self,
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

        # Store connectomics parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.sample_size = ensure_tuple_rep(sample_size, 3)
        self.mode = mode
        self.iter_num = kwargs.get("iter_num", -1)

        # Calculate dataset length
        if self.iter_num > 0:
            self.dataset_length = self.iter_num
        else:
            self.dataset_length = len(data_dicts)

    def __len__(self) -> int:
        return self.dataset_length


__all__ = [
    "MonaiVolumeDataset",
    "MonaiCachedVolumeDataset",
]
