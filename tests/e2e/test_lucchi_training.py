#!/usr/bin/env python3
"""
Test dataset integration with Lucchi-style data.

Creates dummy data and validates that datasets and PyTorch Lightning
DataModules work correctly end-to-end.
"""

import os

import h5py
import numpy as np
import pytest
from monai.transforms import Compose

from connectomics.data.datasets import create_data_dicts_from_paths
from connectomics.data.io.transforms import LoadVolumed
from connectomics.training.lightning.data import ConnectomicsDataModule


@pytest.fixture
def data_paths(tmp_path):
    """Provide synthetic Lucchi-style data paths for downstream tests."""
    return create_dummy_lucchi_data(str(tmp_path))


def create_dummy_lucchi_data(base_path):
    """Create dummy data that matches the Lucchi dataset structure."""
    os.makedirs(base_path, exist_ok=True)

    shape = (32, 64, 64)
    np.random.seed(42)

    img_data = np.random.normal(128, 30, shape).astype(np.uint8)

    for i in range(5):
        center_z = np.random.randint(3, shape[0] - 3)
        center_y = np.random.randint(8, shape[1] - 8)
        center_x = np.random.randint(8, shape[2] - 8)
        zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        ellipsoid = ((zz - center_z) / 2) ** 2 + ((yy - center_y) / 4) ** 2 + (
            (xx - center_x) / 4
        ) ** 2 < 1
        img_data[ellipsoid] = np.random.randint(180, 220)

    label_data = np.zeros(shape, dtype=np.uint8)
    for i in range(4):
        center_z = np.random.randint(2, shape[0] - 2)
        center_y = np.random.randint(6, shape[1] - 6)
        center_x = np.random.randint(6, shape[2] - 6)
        zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        mito = ((zz - center_z) / 1.5) ** 2 + ((yy - center_y) / 3) ** 2 + (
            (xx - center_x) / 3
        ) ** 2 < 1
        label_data[mito] = 1

    train_img_path = os.path.join(base_path, "train_image.h5")
    train_label_path = os.path.join(base_path, "train_label.h5")
    val_img_path = os.path.join(base_path, "val_image.h5")
    val_label_path = os.path.join(base_path, "val_label.h5")

    with h5py.File(train_img_path, "w") as f:
        f.create_dataset("main", data=img_data, compression="gzip")
    with h5py.File(train_label_path, "w") as f:
        f.create_dataset("main", data=label_data, compression="gzip")

    val_img_data = img_data + np.random.normal(0, 5, shape).astype(np.int8)
    val_img_data = np.clip(val_img_data, 0, 255).astype(np.uint8)

    with h5py.File(val_img_path, "w") as f:
        f.create_dataset("main", data=val_img_data, compression="gzip")
    with h5py.File(val_label_path, "w") as f:
        f.create_dataset("main", data=label_data, compression="gzip")

    return {
        "train_image_paths": [train_img_path],
        "train_label_paths": [train_label_path],
        "val_image_paths": [val_img_path],
        "val_label_paths": [val_label_path],
    }


def test_data_dicts_creation(data_paths):
    """Test MONAI data dictionary creation."""
    train_data_dicts = create_data_dicts_from_paths(
        image_paths=data_paths["train_image_paths"],
        label_paths=data_paths["train_label_paths"],
    )
    assert len(train_data_dicts) == 1
    assert "image" in train_data_dicts[0]
    assert "label" in train_data_dicts[0]


def test_connectomics_datamodule(data_paths):
    """Test ConnectomicsDataModule with MONAI datasets."""
    transforms = Compose(
        [
            LoadVolumed(keys=["image", "label"]),
        ]
    )

    train_data_dicts = create_data_dicts_from_paths(
        image_paths=data_paths["train_image_paths"],
        label_paths=data_paths["train_label_paths"],
    )
    val_data_dicts = create_data_dicts_from_paths(
        image_paths=data_paths["val_image_paths"],
        label_paths=data_paths["val_label_paths"],
    )

    datamodule = ConnectomicsDataModule(
        train_data_dicts=train_data_dicts,
        val_data_dicts=val_data_dicts,
        transforms={"train": transforms, "val": transforms},
        dataset_type="standard",
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None

    train_loader = datamodule.train_dataloader()
    sample = next(iter(train_loader))
    assert "image" in sample
    assert "label" in sample


def test_cached_datamodule(data_paths):
    """Test ConnectomicsDataModule with cached datasets."""
    transforms = Compose(
        [
            LoadVolumed(keys=["image", "label"]),
        ]
    )

    train_data_dicts = create_data_dicts_from_paths(
        image_paths=data_paths["train_image_paths"],
        label_paths=data_paths["train_label_paths"],
    )

    datamodule = ConnectomicsDataModule(
        train_data_dicts=train_data_dicts,
        transforms={"train": transforms},
        dataset_type="cached",
        cache_rate=1.0,
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None

    train_loader = datamodule.train_dataloader()
    sample = next(iter(train_loader))
    assert "image" in sample


def test_transform_integration(data_paths):
    """Test MONAI transform pipeline integration."""
    transforms = Compose(
        [
            LoadVolumed(keys=["image", "label"]),
        ]
    )

    train_data_dicts = create_data_dicts_from_paths(
        image_paths=data_paths["train_image_paths"],
        label_paths=data_paths["train_label_paths"],
    )

    datamodule = ConnectomicsDataModule(
        train_data_dicts=train_data_dicts,
        transforms={"train": transforms},
        dataset_type="standard",
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    sample = next(iter(train_loader))

    assert "image" in sample
    assert "label" in sample
    assert sample["image"].ndim >= 3
    assert sample["label"].ndim >= 3
