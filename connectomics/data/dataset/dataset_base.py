"""
Unified dataset module for PyTorch Connectomics.

This module provides a comprehensive dataset infrastructure combining PyTorch Lightning
DataModule functionality with modular dataset classes for different data loading scenarios.
The legacy dataset creation functions have been removed to keep the codebase clean.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import os
import numpy as np

import torch
import torch.utils.data
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ...config import CfgNode
from ...transforms.augment import build_augmentor
from ...transforms.process import DataProcessor, create_processor_from_config
from ..io import read_volume


class BaseConnectomicsDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for all connectomics datasets.

    This class provides common functionality for different data loading scenarios:
    - Volume-based loading
    - Tile-based loading
    - Cloud-based loading

    Args:
        data_processor (DataProcessor): Modern data processor for targets and weights
        augmentor (callable, optional): Data augmentation function
        sample_size (list): Size of samples to extract [z, y, x]
        mode (str): Dataset mode ('train', 'val', 'test')
        iter_num (int): Number of iterations per epoch (-1 for inference)
        valid_ratio (float): Volume ratio threshold for valid samples
        reject_size_thres (int): Threshold for foreground objects
        reject_diversity (int): Threshold for multiple objects
        reject_p (float): Probability of rejecting non-foreground volumes
        data_mean (float): Mean for normalization
        data_std (float): Standard deviation for normalization
        data_match_act (str): Activation matching mode
        do_2d (bool): Load 2D samples from 3D volumes
        do_relabel (bool): Reduce mask indices in sampled volumes
    """

    def __init__(self,
                 data_processor: DataProcessor,
                 augmentor: Optional[Callable] = None,
                 sample_size: List[int] = [128, 128, 128],
                 mode: str = 'train',
                 iter_num: int = -1,
                 valid_ratio: float = 0.5,
                 reject_size_thres: int = 0,
                 reject_diversity: int = 0,
                 reject_p: float = 0.95,
                 data_mean: float = 0.5,
                 data_std: float = 0.5,
                 data_match_act: str = 'none',
                 do_2d: bool = False,
                 do_relabel: bool = True):

        # Core configuration
        self.data_processor = data_processor
        self.augmentor = augmentor
        self.sample_size = self._ensure_list(sample_size)
        self.mode = mode
        self.iter_num = iter_num
        self.valid_ratio = valid_ratio

        # Rejection sampling
        self.reject_size_thres = reject_size_thres
        self.reject_diversity = reject_diversity
        self.reject_p = reject_p

        # Normalization
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_match_act = data_match_act

        # Advanced options
        self.do_2d = do_2d
        self.do_relabel = do_relabel

    def _ensure_list(self, x) -> List[int]:
        """Ensure input is a list of length 3."""
        if isinstance(x, int):
            return [x] * 3
        elif isinstance(x, (list, tuple)):
            if len(x) == 1:
                return list(x) * 3
            elif len(x) == 3:
                return list(x)
            else:
                raise ValueError(f"Size must be int, or list/tuple of length 1 or 3, got {len(x)}")
        else:
            raise ValueError(f"Size must be int, list, or tuple, got {type(x)}")

    def __len__(self) -> int:
        """Return dataset length based on mode and configuration."""
        if self.mode == 'train':
            return self.iter_num if self.iter_num > 0 else 1000
        else:
            return self._calculate_inference_length()

    @abstractmethod
    def _calculate_inference_length(self) -> int:
        """Calculate number of inference samples. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _extract_patches(self, index: int) -> tuple:
        """Extract image and label patches. Must be implemented by subclasses."""
        pass

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        try:
            # Extract patches (implemented by subclasses)
            image_patch, label_patch, valid_mask_patch = self._extract_patches(index)

            # Create data dictionary
            data = {
                'image': image_patch,
                'label': label_patch
            }

            if valid_mask_patch is not None:
                data['valid_mask'] = valid_mask_patch

            # Apply augmentation
            if self.augmentor and self.mode == 'train':
                data = self.augmentor(data)

            # Apply modern data processing
            processed_data = self.data_processor(data)
            data.update(processed_data)

            # Ensure proper tensor format
            data = self._ensure_tensor_format(data)

            return data

        except Exception as e:
            print(f"Warning: Error processing sample {index}: {e}")
            return self._get_fallback_sample()

    def _ensure_tensor_format(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Ensure all data is in proper tensor format."""
        # Convert image to tensor
        if not isinstance(data['image'], torch.Tensor):
            data['image'] = torch.from_numpy(data['image']).float()

        # Ensure image has channel dimension
        if data['image'].ndim == 3:
            data['image'] = data['image'].unsqueeze(0)

        # Convert targets if present
        if 'targets' in data:
            if not isinstance(data['targets'], list):
                data['targets'] = [data['targets']]
            data['targets'] = [
                torch.from_numpy(t).float() if not isinstance(t, torch.Tensor) else t
                for t in data['targets']
            ]

        # Convert weights if present
        if 'weights' in data:
            if not isinstance(data['weights'], list):
                data['weights'] = [data['weights']]
            data['weights'] = [
                torch.from_numpy(w).float() if not isinstance(w, torch.Tensor) else w
                for w in data['weights']
            ]

        return data

    def _get_fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Get a fallback sample in case of errors."""
        return {
            'image': torch.randn(1, *self.sample_size),
            'targets': [torch.randn(1, *self.sample_size)],
            'weights': [torch.ones(1, *self.sample_size)]
        }


class ConnectomicsDataModule(pl.LightningDataModule):
    """
    Unified Lightning DataModule for connectomics tasks.

    This DataModule integrates with different dataset types (volume, tile, cloud)
    and provides consistent data loading across training and inference.

    Args:
        cfg: Configuration object containing dataset and augmentation parameters
        dataset_type: Type of dataset ('volume', 'tile', 'cloud')
        **dataset_kwargs: Additional arguments passed to the dataset constructor
    """

    def __init__(self,
                 cfg: CfgNode,
                 dataset_type: str = 'volume',
                 **dataset_kwargs):
        super().__init__()
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs
        self.save_hyperparameters(logger=False)

        # Dataset paths
        self.data_dir = cfg.DATASET.INPUT_PATH
        self.train_image_path = os.path.join(self.data_dir, cfg.DATASET.IMAGE_NAME)
        self.train_label_path = os.path.join(self.data_dir, cfg.DATASET.LABEL_NAME)

        # Data configuration
        self.batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
        self.num_workers = cfg.SYSTEM.NUM_CPUS
        self.sample_size = cfg.MODEL.INPUT_SIZE

        # Modern unified processor
        self.data_processor = create_processor_from_config(cfg)

        # Transforms
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for different stages."""
        self._setup_transforms()
        self._setup_datasets(stage)

    def _setup_transforms(self) -> None:
        """Setup transforms using unified augmentor."""
        augmentors = build_augmentor(self.cfg)
        self.train_transforms = augmentors['train']
        self.val_transforms = augmentors['val']
        self.test_transforms = augmentors['test']

    def _setup_datasets(self, stage: Optional[str] = None) -> None:
        """Setup datasets based on dataset type."""
        # Import dataset classes here to avoid circular imports
        if self.dataset_type == 'volume':
            from .dataset.dataset_volume import VolumeDataset as DatasetClass
        elif self.dataset_type == 'tile':
            from .dataset.dataset_tile import TileDataset as DatasetClass
        elif self.dataset_type == 'cloud':
            from .dataset.dataset_cloud import CloudVolumeDataset as DatasetClass
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # Create data dictionaries
        train_data = [{
            'image': self.train_image_path,
            'label': self.train_label_path
        }]

        # For now, use same data for validation (in practice, use separate val set)
        val_data = train_data

        # Common dataset arguments
        common_args = {
            'data_processor': self.data_processor,
            'sample_size': self.sample_size,
            **self.dataset_kwargs
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(
                data=train_data,
                augmentor=self.train_transforms,
                mode='train',
                **common_args
            )

            self.val_dataset = DatasetClass(
                data=val_data,
                augmentor=self.val_transforms,
                mode='val',
                **common_args
            )

        if stage == 'test' or stage is None:
            self.test_dataset = DatasetClass(
                data=val_data,
                augmentor=self.test_transforms,
                mode='test',
                **common_args
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        return self.test_dataloader()


# Factory functions for different dataset types
def create_volume_datamodule(cfg: CfgNode, **kwargs) -> ConnectomicsDataModule:
    """Create a DataModule for volume-based datasets."""
    return ConnectomicsDataModule(cfg, dataset_type='volume', **kwargs)


def create_tile_datamodule(cfg: CfgNode, **kwargs) -> ConnectomicsDataModule:
    """Create a DataModule for tile-based datasets."""
    return ConnectomicsDataModule(cfg, dataset_type='tile', **kwargs)


def create_cloud_datamodule(cfg: CfgNode, **kwargs) -> ConnectomicsDataModule:
    """Create a DataModule for cloud-based datasets."""
    return ConnectomicsDataModule(cfg, dataset_type='cloud', **kwargs)


def create_datamodule_from_config(cfg: CfgNode) -> ConnectomicsDataModule:
    """Create appropriate DataModule based on configuration."""
    dataset_type = getattr(cfg.DATASET, 'TYPE', 'volume').lower()

    if dataset_type == 'volume':
        return create_volume_datamodule(cfg)
    elif dataset_type == 'tile':
        return create_tile_datamodule(cfg)
    elif dataset_type == 'cloud':
        return create_cloud_datamodule(cfg)
    else:
        # Default to volume
        return create_volume_datamodule(cfg)


# =============================================================================
# Utility Classes for Multi-Dataset Handling
# =============================================================================

class WeightedConcatDataset(torch.utils.data.Dataset):
    """
    A dataset that concatenates multiple datasets and samples from them with specified weights.

    Args:
        datasets: List of datasets to concatenate
        weights: List of weights for sampling from each dataset (must sum to 1)
        sample_separately: If True, sample from each dataset independently
    """

    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        weights: List[float],
        sample_separately: bool = True
    ):
        assert len(datasets) == len(weights), "Number of datasets must match number of weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

        self.datasets = datasets
        self.weights = np.array(weights)
        self.sample_separately = sample_separately

        # Pre-compute dataset lengths for efficient sampling
        self.dataset_lengths = [len(d) for d in datasets]
        self.min_length = min(self.dataset_lengths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Sample from datasets according to weights."""
        # Choose dataset based on weights
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)

        # Sample from chosen dataset
        if self.sample_separately:
            # Use the provided index modulo dataset length
            sample_idx = index % self.dataset_lengths[dataset_idx]
        else:
            # Random sampling within dataset
            sample_idx = np.random.randint(0, self.dataset_lengths[dataset_idx])

        sample = self.datasets[dataset_idx][sample_idx]

        # Add dataset metadata
        if isinstance(sample, dict):
            sample['dataset_id'] = torch.tensor(dataset_idx, dtype=torch.long)
        else:
            # Handle tuple/list format
            if isinstance(sample, (list, tuple)):
                sample = list(sample) + [torch.tensor(dataset_idx, dtype=torch.long)]
            else:
                sample = (sample, torch.tensor(dataset_idx, dtype=torch.long))

        return sample

    def __len__(self) -> int:
        """Return minimum length to ensure all datasets can contribute."""
        return self.min_length


class DataModule(pl.LightningDataModule):
    """
    Legacy Lightning DataModule for backward compatibility.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
    """

    def __init__(
        self,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> Optional[DataLoader]:
        """Create training dataloader."""
        if self.train_dataset is None:
            return None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # For stable training
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Usually test with batch size 1
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_multi_dataset(
    dataset_configs: List[Dict[str, Any]],
    weights: List[float],
    dataset_type: str = 'volume'
) -> WeightedConcatDataset:
    """
    Factory function to create a weighted concatenated dataset.

    Args:
        dataset_configs: List of configuration dicts for each dataset
        weights: List of sampling weights for each dataset
        dataset_type: Type of dataset to create ('volume' or 'tile')

    Returns:
        WeightedConcatDataset combining all datasets
    """
    datasets = []

    for config in dataset_configs:
        if dataset_type == 'volume':
            from .dataset.dataset_volume import VolumeDataset
            dataset = VolumeDataset(**config)
        elif dataset_type == 'tile':
            from .dataset.dataset_tile import TileDataset
            dataset = TileDataset(**config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        datasets.append(dataset)

    return WeightedConcatDataset(datasets, weights)


def create_datamodule_from_configs(
    train_config: Dict[str, Any],
    val_config: Optional[Dict[str, Any]] = None,
    test_config: Optional[Dict[str, Any]] = None,
    dataset_type: str = 'volume',
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataModule:
    """
    Create a legacy DataModule from dataset configurations.

    Args:
        train_config: Configuration for training dataset
        val_config: Configuration for validation dataset
        test_config: Configuration for test dataset
        dataset_type: Type of dataset ('volume' or 'tile')
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        Configured DataModule
    """
    # Create datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if train_config:
        if dataset_type == 'volume':
            from .dataset.dataset_volume import VolumeDataset
            train_dataset = VolumeDataset(**train_config)
        elif dataset_type == 'tile':
            from .dataset.dataset_tile import TileDataset
            train_dataset = TileDataset(**train_config)

    if val_config:
        if dataset_type == 'volume':
            from .dataset.dataset_volume import VolumeDataset
            val_dataset = VolumeDataset(**val_config)
        elif dataset_type == 'tile':
            from .dataset.dataset_tile import TileDataset
            val_dataset = TileDataset(**val_config)

    if test_config:
        if dataset_type == 'volume':
            from .dataset.dataset_volume import VolumeDataset
            test_dataset = VolumeDataset(**test_config)
        elif dataset_type == 'tile':
            from .dataset.dataset_tile import TileDataset
            test_dataset = TileDataset(**test_config)

    return DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# Legacy dataset creation functions removed as per user request to make codebase clean
# Only keep modern DataModule and factory functions above


__all__ = [
    # Core dataset infrastructure
    'BaseConnectomicsDataset',
    'ConnectomicsDataModule',

    # Factory functions for main DataModule
    'create_volume_datamodule',
    'create_tile_datamodule',
    'create_cloud_datamodule',
    'create_datamodule_from_config',

    # Utility classes
    'WeightedConcatDataset',
    'DataModule',

    # Legacy factory functions
    'create_multi_dataset',
    'create_datamodule_from_configs'
]