"""
MONAI-native filename-based dataset for PyTorch Connectomics.

This module provides a dataset class that loads individual images from JSON file lists
instead of cropping from large volumes. Ideal for datasets with pre-tiled images like
MitoLab, CEM500K, etc.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import json
import random
from pathlib import Path
import warnings

import torch
from monai.data import Dataset
from monai.transforms import Compose


class MonaiFilenameDataset(Dataset):
    """
    MONAI-native dataset for loading individual images from JSON file lists.

    Instead of loading large volumes and cropping patches, this dataset loads
    individual pre-tiled images directly from disk. Useful for datasets like:
    - MitoLab (21,871 image/mask pairs)
    - CEM500K
    - Pre-processed tile datasets

    The JSON file should have the following structure:
    {
        "base_path": "/path/to/data",
        "images": ["relative/path/to/image1.png", ...],
        "masks": ["relative/path/to/mask1.png", ...]  # optional
    }

    Args:
        json_path (str): Path to JSON file containing file lists
        transforms (Compose, optional): MONAI transforms pipeline to apply
        mode (str): Dataset mode ('train', 'val', 'test'). Default: 'train'
        images_key (str): Key in JSON for image file list. Default: 'images'
        labels_key (str, optional): Key in JSON for label file list. Default: 'masks'
        base_path_key (str): Key in JSON for base path. Default: 'base_path'
        train_val_split (float, optional): Fraction for train split (0.0-1.0).
                                          If provided, automatically splits data.
        random_seed (int): Random seed for train/val split. Default: 42
        use_labels (bool): Whether to load labels. Default: True

    Example:
        >>> # Create dataset from JSON
        >>> dataset = MonaiFilenameDataset(
        ...     json_path='datasets/cem-mitolab/train.json',
        ...     transforms=my_transforms,
        ...     mode='train'
        ... )
        >>>
        >>> # With automatic train/val split
        >>> train_ds = MonaiFilenameDataset(
        ...     json_path='datasets/cem-mitolab/files.json',
        ...     mode='train',
        ...     train_val_split=0.9,
        ... )
        >>> val_ds = MonaiFilenameDataset(
        ...     json_path='datasets/cem-mitolab/files.json',
        ...     mode='val',
        ...     train_val_split=0.9,
        ... )
    """

    def __init__(
        self,
        json_path: str,
        transforms: Optional[Compose] = None,
        mode: str = "train",
        images_key: str = "images",
        labels_key: str = "masks",
        base_path_key: str = "base_path",
        train_val_split: Optional[float] = None,
        random_seed: int = 42,
        use_labels: bool = True,
    ):
        self.json_path = Path(json_path)
        self.mode = mode
        self.images_key = images_key
        self.labels_key = labels_key
        self.use_labels = use_labels

        # Load JSON file
        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

        # Get base path
        self.base_path = Path(self.json_data.get(base_path_key, ""))

        # Get file lists
        image_files = self.json_data.get(images_key, [])
        label_files = self.json_data.get(labels_key, []) if use_labels else []

        if not image_files:
            raise ValueError(f"No images found in JSON file under key '{images_key}'")

        # Check that images and labels match if labels are provided
        if use_labels and label_files and len(image_files) != len(label_files):
            warnings.warn(
                f"Number of images ({len(image_files)}) doesn't match "
                f"number of labels ({len(label_files)}). Proceeding with available pairs."
            )

        # Create paired data
        if use_labels and label_files:
            pairs = list(zip(image_files, label_files))
        else:
            pairs = [(img, None) for img in image_files]

        # Apply train/val split if requested
        if train_val_split is not None:
            if not 0.0 < train_val_split < 1.0:
                raise ValueError(f"train_val_split must be between 0 and 1, got {train_val_split}")

            # Deterministic shuffle for reproducibility
            rng = random.Random(random_seed)
            pairs_shuffled = pairs.copy()
            rng.shuffle(pairs_shuffled)

            # Split
            n_train = int(len(pairs_shuffled) * train_val_split)
            if mode == "train":
                pairs = pairs_shuffled[:n_train]
            elif mode in ["val", "validation"]:
                pairs = pairs_shuffled[n_train:]
            else:  # test mode uses all data
                pairs = pairs_shuffled

        # Create MONAI data dictionaries
        data_dicts = []
        for img_file, label_file in pairs:
            data_dict = {"image": str(self.base_path / img_file)}
            if label_file is not None:
                data_dict["label"] = str(self.base_path / label_file)
            data_dicts.append(data_dict)

        # Initialize parent MONAI Dataset
        super().__init__(data=data_dicts, transform=transforms)

        print(f"📋 MonaiFilenameDataset initialized:")
        print(f"   Mode: {mode}")
        print(f"   Samples: {len(data_dicts)}")
        print(f"   Base path: {self.base_path}")
        if train_val_split is not None:
            print(f"   Train/val split: {train_val_split:.1%} (seed={random_seed})")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data sample with transforms applied.

        Args:
            index: Sample index

        Returns:
            Dictionary with 'image' and optionally 'label' keys
        """
        return super().__getitem__(index)


class MonaiFilenameIterableDataset(torch.utils.data.IterableDataset):
    """
    Iterable version of MonaiFilenameDataset for infinite sampling during training.

    This dataset randomly samples from the file list indefinitely, useful for
    training with a fixed number of iterations rather than epochs.

    Args:
        json_path (str): Path to JSON file containing file lists
        transforms (Compose, optional): MONAI transforms pipeline to apply
        images_key (str): Key in JSON for image file list. Default: 'images'
        labels_key (str, optional): Key in JSON for label file list. Default: 'masks'
        base_path_key (str): Key in JSON for base path. Default: 'base_path'
        random_seed (int, optional): Random seed. If None, uses random sampling.
        use_labels (bool): Whether to load labels. Default: True

    Example:
        >>> # Infinite sampling for training
        >>> dataset = MonaiFilenameIterableDataset(
        ...     json_path='datasets/cem-mitolab/train.json',
        ...     transforms=my_transforms,
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=8, num_workers=4)
        >>> for batch in dataloader:  # Infinite loop
        ...     train_step(batch)
    """

    def __init__(
        self,
        json_path: str,
        transforms: Optional[Compose] = None,
        images_key: str = "images",
        labels_key: str = "masks",
        base_path_key: str = "base_path",
        random_seed: Optional[int] = None,
        use_labels: bool = True,
    ):
        super().__init__()
        self.json_path = Path(json_path)
        self.images_key = images_key
        self.labels_key = labels_key
        self.use_labels = use_labels
        self.transforms = transforms
        self.random_seed = random_seed

        # Load JSON file
        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

        # Get base path
        self.base_path = Path(self.json_data.get(base_path_key, ""))

        # Get file lists
        self.image_files = self.json_data.get(images_key, [])
        self.label_files = self.json_data.get(labels_key, []) if use_labels else []

        if not self.image_files:
            raise ValueError(f"No images found in JSON file under key '{images_key}'")

        # Check that images and labels match
        if use_labels and self.label_files:
            if len(self.image_files) != len(self.label_files):
                raise ValueError(
                    f"Number of images ({len(self.image_files)}) must match "
                    f"number of labels ({len(self.label_files)})"
                )

        print(f"📋 MonaiFilenameIterableDataset initialized:")
        print(f"   Samples: {len(self.image_files)}")
        print(f"   Base path: {self.base_path}")
        print(f"   Infinite sampling: True")

    def __iter__(self):
        """Infinite iterator over dataset."""
        # Set random seed if provided (for reproducibility)
        if self.random_seed is not None:
            rng = random.Random(self.random_seed)
        else:
            rng = random.Random()

        # Infinite loop
        while True:
            # Randomly sample an index
            idx = rng.randint(0, len(self.image_files) - 1)

            # Create data dict
            data_dict = {"image": str(self.base_path / self.image_files[idx])}
            if self.use_labels and self.label_files:
                data_dict["label"] = str(self.base_path / self.label_files[idx])

            # Apply transforms if provided
            if self.transforms is not None:
                data_dict = self.transforms(data_dict)

            yield data_dict


def create_filename_datasets(
    json_path: str,
    train_transforms: Optional[Compose] = None,
    val_transforms: Optional[Compose] = None,
    train_val_split: float = 0.9,
    random_seed: int = 42,
    images_key: str = "images",
    labels_key: str = "masks",
    use_labels: bool = True,
) -> Tuple[MonaiFilenameDataset, MonaiFilenameDataset]:
    """
    Convenience function to create train and validation datasets from a single JSON file.

    Args:
        json_path: Path to JSON file with all files
        train_transforms: Transforms for training data
        val_transforms: Transforms for validation data
        train_val_split: Fraction of data for training (0.0-1.0)
        random_seed: Random seed for reproducible splits
        images_key: Key in JSON for images
        labels_key: Key in JSON for labels
        use_labels: Whether to load labels

    Returns:
        Tuple of (train_dataset, val_dataset)

    Example:
        >>> train_ds, val_ds = create_filename_datasets(
        ...     json_path='datasets/cem-mitolab/files.json',
        ...     train_transforms=train_aug,
        ...     val_transforms=val_aug,
        ...     train_val_split=0.9,
        ... )
    """
    train_dataset = MonaiFilenameDataset(
        json_path=json_path,
        transforms=train_transforms,
        mode="train",
        images_key=images_key,
        labels_key=labels_key,
        train_val_split=train_val_split,
        random_seed=random_seed,
        use_labels=use_labels,
    )

    val_dataset = MonaiFilenameDataset(
        json_path=json_path,
        transforms=val_transforms,
        mode="val",
        images_key=images_key,
        labels_key=labels_key,
        train_val_split=train_val_split,
        random_seed=random_seed,
        use_labels=use_labels,
    )

    return train_dataset, val_dataset


__all__ = [
    "MonaiFilenameDataset",
    "MonaiFilenameIterableDataset",
    "create_filename_datasets",
]
