"""
PyTorch Lightning DataModules for MONAI-native connectomics datasets.

This module provides Lightning DataModule wrappers around the MONAI connectomics
datasets, enabling seamless integration with PyTorch Lightning training workflows.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.transforms import Compose

from connectomics.data.dataset import (
    MonaiConnectomicsDataset,
    MonaiCachedConnectomicsDataset,
    MonaiPersistentConnectomicsDataset,
    MonaiVolumeDataset,
    MonaiCachedVolumeDataset,
    MonaiTileDataset,
    MonaiCachedTileDataset,
    create_connectomics_dataset,
    create_data_dicts_from_paths,
    create_volume_dataset,
    create_tile_dataset,
)
from connectomics.data.process import create_label_transform_pipeline


class ConnectomicsDataModule(pl.LightningDataModule):
    """
    Base Lightning DataModule for MONAI connectomics datasets.

    This DataModule provides a unified interface for different types of connectomics
    datasets (volume, tile, cloud) with automatic train/val/test splits and
    MONAI transform pipelines.

    Args:
        train_data_dicts (List[Dict]): Training data dictionaries
        val_data_dicts (List[Dict], optional): Validation data dictionaries
        test_data_dicts (List[Dict], optional): Test data dictionaries
        transforms (Dict[str, Compose], optional): Transform pipelines for each split
        dataset_type (str): Type of dataset ('standard', 'cached', 'persistent')
        batch_size (int): Batch size for data loaders. Default: 1
        num_workers (int): Number of workers for data loading. Default: 0
        pin_memory (bool): Whether to use pinned memory. Default: True
        persistent_workers (bool): Whether to use persistent workers. Default: False
        cache_rate (float): Cache rate for cached datasets. Default: 1.0
        cache_dir (str): Cache directory for persistent datasets. Default: './cache'
        **dataset_kwargs: Additional arguments for dataset initialization
    """

    def __init__(
        self,
        train_data_dicts: List[Dict[str, Any]],
        val_data_dicts: Optional[List[Dict[str, Any]]] = None,
        test_data_dicts: Optional[List[Dict[str, Any]]] = None,
        transforms: Optional[Dict[str, Compose]] = None,
        dataset_type: str = "standard",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        cache_rate: float = 1.0,
        cache_dir: str = "./cache",
        train_transpose_axes: Optional[List[int]] = None,
        val_transpose_axes: Optional[List[int]] = None,
        test_transpose_axes: Optional[List[int]] = None,
        **dataset_kwargs,
    ):
        super().__init__()

        # Store data dictionaries
        self.train_data_dicts = train_data_dicts
        self.val_data_dicts = val_data_dicts
        self.test_data_dicts = test_data_dicts

        val_has_entries = val_data_dicts is not None and len(val_data_dicts) > 0
        self.skip_validation = not val_has_entries

        # Store transforms
        self.transforms = transforms or {}

        # Store dataset and dataloader configuration
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cache_rate = cache_rate
        self.cache_dir = cache_dir
        self.dataset_kwargs = dataset_kwargs

        # Store mode-specific transpose axes
        self.train_transpose_axes = train_transpose_axes
        self.val_transpose_axes = val_transpose_axes
        self.test_transpose_axes = test_transpose_axes

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the specified stage."""

        val_has_entries = self.val_data_dicts is not None and len(self.val_data_dicts) > 0
        self.skip_validation = not val_has_entries

        if stage == "fit" or stage is None:
            # Setup training dataset
            if self.train_data_dicts:
                self.train_dataset = self._create_dataset(
                    data_dicts=self.train_data_dicts,
                    transforms=self.transforms.get("train"),
                    mode="train",
                )

            # Setup validation dataset
            if self.val_data_dicts:
                self.val_dataset = self._create_dataset(
                    data_dicts=self.val_data_dicts,
                    transforms=self.transforms.get("val"),
                    mode="val",
                )

        if stage == "test" or stage is None:
            # Setup test dataset
            if self.test_data_dicts:
                self.test_dataset = self._create_dataset(
                    data_dicts=self.test_data_dicts,
                    transforms=self.transforms.get("test"),
                    mode="test",
                )

    def _create_dataset(
        self,
        data_dicts: List[Dict[str, Any]],
        transforms: Optional[Compose],
        mode: str,
    ) -> Union[
        MonaiConnectomicsDataset, MonaiCachedConnectomicsDataset, MonaiPersistentConnectomicsDataset
    ]:
        """Create appropriate dataset based on configuration."""

        # Prepare dataset arguments
        dataset_args = {
            "data_dicts": data_dicts,
            "transforms": transforms,
            "mode": mode,
            **self.dataset_kwargs,
        }

        # Note: transpose_axes should be embedded in the transforms (via LoadVolumed)
        # and NOT passed as a dataset parameter. The transform builders handle this.

        # Add type-specific arguments
        if self.dataset_type == "cached":
            dataset_args["cache_rate"] = self.cache_rate
        elif self.dataset_type == "persistent":
            dataset_args["cache_dir"] = self.cache_dir

        return create_connectomics_dataset(
            dataset_type=self.dataset_type,
            **dataset_args,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create validation data loader."""

        if self.skip_validation:
            return []

        dataloader = self._create_dataloader(self.val_dataset, shuffle=False)

        # If validation dataset exists but dataloader creation failed,
        # skip validation rather than using dummy data
        if dataloader is None:
            warnings.warn(
                "Validation dataloader creation failed despite validation dataset being provided. "
                "Skipping validation. Check your data configuration.",
                UserWarning
            )
            return []

        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Create predict data loader (same as test dataloader)."""
        return self.test_dataloader()

    def _create_dataloader(
        self,
        dataset: Optional[torch.utils.data.Dataset],
        shuffle: bool,
    ) -> Optional[DataLoader]:
        """Create data loader for the given dataset."""

        if dataset is None:
            return None

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for MONAI data."""
        # Stack samples while preserving dict structure
        if not batch:
            return {}

        # Get all keys from first sample
        keys = list(batch[0].keys())
        result = {}

        for key in keys:
            # Stack all values for this key
            values = [sample[key] for sample in batch]
            # Convert to tensor if not already
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                result[key] = torch.stack([torch.from_numpy(v) for v in values])
            else:
                result[key] = values  # Keep as list if not tensor/array

        return result


class VolumeDataModule(ConnectomicsDataModule):
    """
    Lightning DataModule specifically for volume-based connectomics datasets.

    This DataModule provides volume-specific functionality with optimized
    data loading for volumetric connectomics data.

    Args:
        train_image_paths (List[str]): Training image file paths
        train_label_paths (List[str], optional): Training label file paths
        train_mask_paths (List[str], optional): Training mask file paths
        val_image_paths (List[str], optional): Validation image file paths
        val_label_paths (List[str], optional): Validation label file paths
        val_mask_paths (List[str], optional): Validation mask file paths
        test_image_paths (List[str], optional): Test image file paths
        test_label_paths (List[str], optional): Test label file paths
        test_mask_paths (List[str], optional): Test mask file paths
        sample_size (Tuple[int, int, int]): Size of samples to extract (z, y, x)
        **kwargs: Additional arguments passed to ConnectomicsDataModule
    """

    def __init__(
        self,
        train_image_paths: List[str],
        train_label_paths: Optional[List[str]] = None,
        train_mask_paths: Optional[List[str]] = None,
        val_image_paths: Optional[List[str]] = None,
        val_label_paths: Optional[List[str]] = None,
        val_mask_paths: Optional[List[str]] = None,
        test_image_paths: Optional[List[str]] = None,
        test_label_paths: Optional[List[str]] = None,
        test_mask_paths: Optional[List[str]] = None,
        sample_size: Tuple[int, int, int] = (32, 256, 256),
        **kwargs,
    ):
        # Create data dictionaries from paths
        train_data_dicts = create_data_dicts_from_paths(
            image_paths=train_image_paths,
            label_paths=train_label_paths,
            mask_paths=train_mask_paths,
        )

        val_data_dicts = None
        if val_image_paths:
            val_data_dicts = create_data_dicts_from_paths(
                image_paths=val_image_paths,
                label_paths=val_label_paths,
                mask_paths=val_mask_paths,
            )

        test_data_dicts = None
        if test_image_paths:
            test_data_dicts = create_data_dicts_from_paths(
                image_paths=test_image_paths,
                label_paths=test_label_paths,
                mask_paths=test_mask_paths,
            )

        # Add sample size to dataset kwargs
        kwargs["sample_size"] = sample_size

        super().__init__(
            train_data_dicts=train_data_dicts,
            val_data_dicts=val_data_dicts,
            test_data_dicts=test_data_dicts,
            **kwargs,
        )

    def _create_dataset(
        self,
        data_dicts: List[Dict[str, Any]],
        transforms: Optional[Compose],
        mode: str,
    ) -> Union[MonaiVolumeDataset, MonaiCachedVolumeDataset]:
        """Create volume-specific dataset."""

        # Extract paths from data dictionaries
        image_paths = [d["image"] for d in data_dicts]
        label_paths = [d["label"] for d in data_dicts if "label" in d]
        mask_paths = [d["mask"] for d in data_dicts if "mask" in d]

        # Ensure we have label and mask paths or None
        if not label_paths:
            label_paths = None
        if not mask_paths:
            mask_paths = None

        return create_volume_dataset(
            image_paths=image_paths,
            label_paths=label_paths,
            mask_paths=mask_paths,
            transforms=transforms,
            dataset_type=self.dataset_type,
            cache_rate=self.cache_rate,
            mode=mode,
            **self.dataset_kwargs,
        )


class TileDataModule(ConnectomicsDataModule):
    """
    Lightning DataModule specifically for tile-based connectomics datasets.

    This DataModule provides tile-specific functionality for large-scale
    connectomics datasets stored as individual tiles.

    Args:
        train_volume_json (str): Training volume JSON metadata file
        train_label_json (str, optional): Training label JSON metadata file
        train_mask_json (str, optional): Training mask JSON metadata file
        val_volume_json (str, optional): Validation volume JSON metadata file
        val_label_json (str, optional): Validation label JSON metadata file
        val_mask_json (str, optional): Validation mask JSON metadata file
        test_volume_json (str, optional): Test volume JSON metadata file
        test_label_json (str, optional): Test label JSON metadata file
        test_mask_json (str, optional): Test mask JSON metadata file
        chunk_num (Tuple[int, int, int]): Volume splitting parameters (z, y, x)
        **kwargs: Additional arguments passed to ConnectomicsDataModule
    """

    def __init__(
        self,
        train_volume_json: str,
        train_label_json: Optional[str] = None,
        train_mask_json: Optional[str] = None,
        val_volume_json: Optional[str] = None,
        val_label_json: Optional[str] = None,
        val_mask_json: Optional[str] = None,
        test_volume_json: Optional[str] = None,
        test_label_json: Optional[str] = None,
        test_mask_json: Optional[str] = None,
        chunk_num: Tuple[int, int, int] = (2, 2, 2),
        **kwargs,
    ):
        # Store tile-specific parameters
        self.train_jsons = {
            "volume": train_volume_json,
            "label": train_label_json,
            "mask": train_mask_json,
        }

        self.val_jsons = None
        if val_volume_json:
            self.val_jsons = {
                "volume": val_volume_json,
                "label": val_label_json,
                "mask": val_mask_json,
            }

        self.test_jsons = None
        if test_volume_json:
            self.test_jsons = {
                "volume": test_volume_json,
                "label": test_label_json,
                "mask": test_mask_json,
            }

        # Add chunk_num to dataset kwargs
        kwargs["chunk_num"] = chunk_num

        # Create dummy data dictionaries (actual data will be created in _create_dataset)
        train_data_dicts = [{"dummy": "train"}]
        val_data_dicts = [{"dummy": "val"}] if self.val_jsons else None
        test_data_dicts = [{"dummy": "test"}] if self.test_jsons else None

        super().__init__(
            train_data_dicts=train_data_dicts,
            val_data_dicts=val_data_dicts,
            test_data_dicts=test_data_dicts,
            **kwargs,
        )

    def _create_dataset(
        self,
        data_dicts: List[Dict[str, Any]],
        transforms: Optional[Compose],
        mode: str,
    ) -> Union[MonaiTileDataset, MonaiCachedTileDataset]:
        """Create tile-specific dataset."""

        # Get appropriate JSON files for the mode
        if mode == "train":
            jsons = self.train_jsons
        elif mode == "val":
            jsons = self.val_jsons
        elif mode == "test":
            jsons = self.test_jsons
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if jsons is None:
            return None

        return create_tile_dataset(
            volume_json=jsons["volume"],
            label_json=jsons["label"],
            mask_json=jsons["mask"],
            transforms=transforms,
            dataset_type=self.dataset_type,
            cache_rate=self.cache_rate,
            mode=mode,
            **self.dataset_kwargs,
        )


def create_volume_datamodule(
    train_image_paths: List[str],
    train_label_paths: Optional[List[str]] = None,
    train_mask_paths: Optional[List[str]] = None,
    val_image_paths: Optional[List[str]] = None,
    val_label_paths: Optional[List[str]] = None,
    val_mask_paths: Optional[List[str]] = None,
    sample_size: Tuple[int, int, int] = (32, 256, 256),
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_type: str = "standard",
    task_type: str = "binary",
    **kwargs,
) -> VolumeDataModule:
    """
    Factory function to create volume-based DataModule with default transforms.

    Args:
        train_image_paths: Training image file paths
        train_label_paths: Optional training label file paths
        train_mask_paths: Optional training mask file paths
        val_image_paths: Optional validation image file paths
        val_label_paths: Optional validation label file paths
        val_mask_paths: Optional validation mask file paths
        sample_size: Size of samples to extract (z, y, x)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        dataset_type: Type of dataset ('standard', 'cached', 'persistent')
        task_type: Type of segmentation task ('binary', 'affinity', 'instance')
        **kwargs: Additional arguments for DataModule

    Returns:
        Configured VolumeDataModule instance
    """

    # Create default transforms based on task type
    transforms = {}

    if train_label_paths:
        from types import SimpleNamespace

        if task_type == "binary":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[{"name": "binary"}],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)
        elif task_type == "affinity":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[
                    {
                        "name": "affinity",
                        "kwargs": {"offsets": ["1-0-0", "0-1-0", "0-0-1"]},
                    }
                ],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)
        elif task_type == "instance":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[
                    {"name": "binary"},
                    {
                        "name": "instance_boundary",
                        "kwargs": {"thickness": 1, "edge_mode": "seg-all"},
                    },
                    {"name": "instance_edt", "kwargs": {"mode": "2d", "quantize": False}},
                ],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)

    # Use same transforms for validation if available
    if val_image_paths and transforms.get("train"):
        transforms["val"] = transforms["train"]

    return VolumeDataModule(
        train_image_paths=train_image_paths,
        train_label_paths=train_label_paths,
        train_mask_paths=train_mask_paths,
        val_image_paths=val_image_paths,
        val_label_paths=val_label_paths,
        val_mask_paths=val_mask_paths,
        sample_size=sample_size,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_type=dataset_type,
        transforms=transforms,
        **kwargs,
    )


def create_tile_datamodule(
    train_volume_json: str,
    train_label_json: Optional[str] = None,
    train_mask_json: Optional[str] = None,
    val_volume_json: Optional[str] = None,
    val_label_json: Optional[str] = None,
    val_mask_json: Optional[str] = None,
    chunk_num: Tuple[int, int, int] = (2, 2, 2),
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_type: str = "standard",
    task_type: str = "binary",
    **kwargs,
) -> TileDataModule:
    """
    Factory function to create tile-based DataModule with default transforms.

    Args:
        train_volume_json: Training volume JSON metadata file
        train_label_json: Optional training label JSON metadata file
        val_volume_json: Optional validation volume JSON metadata file
        val_label_json: Optional validation label JSON metadata file
        chunk_num: Volume splitting parameters (z, y, x)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        dataset_type: Type of dataset ('standard', 'cached', 'persistent')
        task_type: Type of segmentation task ('binary', 'affinity', 'instance')
        **kwargs: Additional arguments for DataModule

    Returns:
        Configured TileDataModule instance
    """

    # Create default transforms based on task type
    transforms = {}

    if train_label_json:
        from types import SimpleNamespace

        if task_type == "binary":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[{"name": "binary"}],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)
        elif task_type == "affinity":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[
                    {
                        "name": "affinity",
                        "kwargs": {"offsets": ["1-0-0", "0-1-0", "0-0-1"]},
                    }
                ],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)
        elif task_type == "instance":
            cfg = SimpleNamespace(
                keys=["label"],
                targets=[
                    {"name": "binary"},
                    {
                        "name": "instance_boundary",
                        "kwargs": {"thickness": 1, "edge_mode": "seg-all"},
                    },
                    {"name": "instance_edt", "kwargs": {"mode": "2d", "quantize": False}},
                ],
            )
            transforms["train"] = create_label_transform_pipeline(cfg)

    # Use same transforms for validation if available
    if val_volume_json and transforms.get("train"):
        transforms["val"] = transforms["train"]

    return TileDataModule(
        train_volume_json=train_volume_json,
        train_label_json=train_label_json,
        train_mask_json=train_mask_json,
        val_volume_json=val_volume_json,
        val_label_json=val_label_json,
        val_mask_json=val_mask_json,
        chunk_num=chunk_num,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_type=dataset_type,
        transforms=transforms,
        **kwargs,
    )


def create_datamodule_from_config(
    config: Dict[str, Any],
) -> Union[VolumeDataModule, TileDataModule]:
    """
    Create DataModule from configuration dictionary.

    Args:
        config: Configuration dictionary with dataset parameters

    Returns:
        Appropriate DataModule instance based on configuration
    """

    dataset_config = config.get("DATASET", {})

    # Determine dataset type
    if "volume_json" in dataset_config or "train_volume_json" in dataset_config:
        # Tile-based dataset
        return TileDataModule(
            train_volume_json=dataset_config.get("train_volume_json"),
            train_label_json=dataset_config.get("train_label_json"),
            train_mask_json=dataset_config.get("train_mask_json"),
            val_volume_json=dataset_config.get("val_volume_json"),
            val_label_json=dataset_config.get("val_label_json"),
            val_mask_json=dataset_config.get("val_mask_json"),
            chunk_num=dataset_config.get("chunk_num", (2, 2, 2)),
            batch_size=config.get("SOLVER", {}).get("batch_size", 1),
            num_workers=config.get("SYSTEM", {}).get("num_workers", 0),
            dataset_type=dataset_config.get("dataset_type", "standard"),
        )
    else:
        # Volume-based dataset
        return VolumeDataModule(
            train_image_paths=dataset_config.get("train_image_paths", []),
            train_label_paths=dataset_config.get("train_label_paths"),
            train_mask_paths=dataset_config.get("train_mask_paths"),
            val_image_paths=dataset_config.get("val_image_paths"),
            val_label_paths=dataset_config.get("val_label_paths"),
            val_mask_paths=dataset_config.get("val_mask_paths"),
            sample_size=dataset_config.get("sample_size", (32, 256, 256)),
            batch_size=config.get("SOLVER", {}).get("batch_size", 1),
            num_workers=config.get("SYSTEM", {}).get("num_workers", 0),
            dataset_type=dataset_config.get("dataset_type", "standard"),
        )


__all__ = [
    "ConnectomicsDataModule",
    "VolumeDataModule",
    "TileDataModule",
    "create_volume_datamodule",
    "create_tile_datamodule",
    "create_datamodule_from_config",
]
