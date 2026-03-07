"""
Filename-based dataset for PyTorch Connectomics.

Loads individual images from JSON file lists instead of cropping from large
volumes. Ideal for datasets with pre-tiled images like MitoLab, CEM500K, etc.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from monai.data import Dataset
from monai.transforms import Compose

logger = logging.getLogger(__name__)


class MonaiFilenameDataset(Dataset):
    """
    MONAI dataset for loading individual images from JSON file lists.

    JSON format::

        {
            "base_path": "/path/to/data",
            "images": ["relative/path/to/image1.png", ...],
            "masks": ["relative/path/to/mask1.png", ...]
        }

    Args:
        json_path: Path to JSON file containing file lists.
        transforms: MONAI transforms pipeline.
        mode: 'train', 'val', or 'test'.
        images_key: Key in JSON for image file list.
        labels_key: Key in JSON for label file list.
        base_path_key: Key in JSON for base path.
        train_val_split: Fraction for train split (0.0-1.0).
        random_seed: Random seed for train/val split.
        use_labels: Whether to load labels.
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

        with open(self.json_path, "r") as f:
            json_data = json.load(f)

        base_path = Path(json_data.get(base_path_key, ""))
        image_files = json_data.get(images_key, [])
        label_files = (
            json_data.get(labels_key, []) if use_labels else []
        )

        if not image_files:
            raise ValueError(
                f"No images found in JSON under key '{images_key}'"
            )

        # Create paired data
        if use_labels and label_files:
            if len(image_files) != len(label_files):
                raise ValueError(
                    f"Image count ({len(image_files)}) != "
                    f"label count ({len(label_files)})"
                )
            pairs = list(zip(image_files, label_files))
        else:
            pairs = [(img, None) for img in image_files]

        # Apply train/val split if requested
        if train_val_split is not None:
            if not 0.0 < train_val_split < 1.0:
                raise ValueError(
                    f"train_val_split must be in (0, 1), "
                    f"got {train_val_split}"
                )
            rng = random.Random(random_seed)
            pairs_shuffled = pairs.copy()
            rng.shuffle(pairs_shuffled)

            n_train = int(len(pairs_shuffled) * train_val_split)
            if mode == "train":
                pairs = pairs_shuffled[:n_train]
            elif mode in ("val", "validation"):
                pairs = pairs_shuffled[n_train:]
            else:
                pairs = pairs_shuffled

        # Create MONAI data dictionaries
        data_dicts = []
        for img_file, label_file in pairs:
            d: Dict[str, Any] = {
                "image": str(base_path / img_file),
            }
            if label_file is not None:
                d["label"] = str(base_path / label_file)
            data_dicts.append(d)

        super().__init__(data=data_dicts, transform=transforms)

        logger.info(
            "MonaiFilenameDataset: mode=%s, samples=%d, base=%s",
            mode, len(data_dicts), base_path,
        )


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
    """Create train and val datasets from a single JSON."""
    train_ds = MonaiFilenameDataset(
        json_path=json_path,
        transforms=train_transforms,
        mode="train",
        images_key=images_key,
        labels_key=labels_key,
        train_val_split=train_val_split,
        random_seed=random_seed,
        use_labels=use_labels,
    )
    val_ds = MonaiFilenameDataset(
        json_path=json_path,
        transforms=val_transforms,
        mode="val",
        images_key=images_key,
        labels_key=labels_key,
        train_val_split=train_val_split,
        random_seed=random_seed,
        use_labels=use_labels,
    )
    return train_ds, val_ds


__all__ = [
    "MonaiFilenameDataset",
    "create_filename_datasets",
]
