"""
Configuration and factory functions for PyTorch Lightning training.

This module contains all "create" functions that build training components:
- create_datamodule(): Build Lightning DataModule from config
- setup_seed_everything(): Handle seed_everything across Lightning versions
- Helper utilities used by main training scripts
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import List, Optional

import torch

# Import data and model utilities
from ...config import Config
from ...data.augment.build import (
    build_test_transforms,
    build_train_transforms,
    build_val_transforms,
)
from ...data.dataset import create_data_dicts_from_paths
from .data import ConnectomicsDataModule
from .path_utils import expand_file_paths as _expand_file_paths


def setup_seed_everything():
    """
    Setup seed_everything function with fallback for older Lightning versions.

    Returns:
        seed_everything function
    """
    try:
        from pytorch_lightning.utilities.seed import seed_everything

        return seed_everything
    except ImportError:
        try:
            from pytorch_lightning import seed_everything

            return seed_everything
        except ImportError:
            # Fallback for older versions
            def seed_everything(seed, workers=True):
                import random

                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            return seed_everything


def expand_file_paths(path_or_pattern) -> List[str]:
    """
    Backward-compatible wrapper for shared path expansion helper.

    Args:
        path_or_pattern: Single file path, glob pattern, or list of paths/patterns

    Returns:
        List of expanded file paths, sorted alphabetically
    """
    return _expand_file_paths(path_or_pattern)


def _calculate_validation_iter_num(
    val_data_dicts: List[dict],
    patch_size: tuple[int, int, int],
    min_iter: int = 50,
    max_iter: Optional[int] = 200,
    default_iter_num: int = 100,
    fallback_volume_shape: Optional[tuple[int, int, int]] = None,
    return_default_on_error: bool = True,
) -> int:
    """
    Calculate validation iter_num based on validation volume size and patch size.

    Args:
        val_data_dicts: Validation data dictionaries
        patch_size: Patch size (D, H, W)
        min_iter: Minimum iterations per epoch
        max_iter: Maximum iterations per epoch
        default_iter_num: Default iter_num when calculation fails
        fallback_volume_shape: Volume shape fallback for unknown file formats
        return_default_on_error: Return default_iter_num on errors instead of raising

    Returns:
        Calculated validation iter_num
    """
    try:
        # Get first validation volume size
        img_path = Path(val_data_dicts[0]["image"])

        # Load volume to get shape
        if img_path.suffix in [".nii", ".gz"]:
            # NIfTI file
            import nibabel as nib

            vol = nib.load(str(img_path))
            vol_shape = vol.shape
        elif img_path.suffix in [".h5", ".hdf5"]:
            # HDF5 file
            import h5py

            with h5py.File(img_path, "r") as f:
                vol_shape = f[list(f.keys())[0]].shape
        elif img_path.suffix in [".tif", ".tiff"]:
            # TIFF file
            import tifffile

            vol = tifffile.imread(img_path)
            vol_shape = vol.shape
        elif fallback_volume_shape is not None:
            vol_shape = fallback_volume_shape
        else:
            # Unknown format, use default
            print(
                f"  ⚠️  Unknown file format {img_path.suffix}, "
                f"using default val_iter_num={default_iter_num}"
            )
            return default_iter_num

        # Handle channel dimension if present
        if len(vol_shape) == 4:
            vol_shape = vol_shape[1:]  # Remove channel dim: (C, D, H, W) -> (D, H, W)

        # Calculate number of possible patches (with 50% overlap)
        stride = tuple(p // 2 for p in patch_size)  # 50% overlap
        num_patches_per_dim = [
            max(1, (vol_shape[i] - patch_size[i]) // stride[i] + 1) for i in range(3)
        ]
        total_possible_patches = (
            num_patches_per_dim[0] * num_patches_per_dim[1] * num_patches_per_dim[2]
        )

        # Calculate validation iter_num as a fraction of possible patches
        val_iter_num = int(total_possible_patches * 0.075)  # 7.5% of possible patches
        val_iter_num = max(min_iter, val_iter_num)
        if max_iter is not None:
            val_iter_num = min(max_iter, val_iter_num)

        print(f"    Validation volume shape: {vol_shape}")
        print(f"    Patch size: {patch_size}")
        print(f"    Stride (50% overlap): {stride}")
        print(f"    Possible patches per dim: {num_patches_per_dim}")
        print(f"    Total possible patches: {total_possible_patches}")
        print(f"    Using 7.5% of patches: {val_iter_num}")

        return val_iter_num

    except Exception as e:
        if not return_default_on_error:
            raise
        print(f"  ⚠️  Error calculating validation iter_num: {e}")
        print(f"  ℹ️  Using default val_iter_num={default_iter_num}")
        return default_iter_num


def create_datamodule(
    cfg: Config, mode: str = "train", fast_dev_run: bool = False
) -> ConnectomicsDataModule:
    """
    Create Lightning DataModule from config.

    Args:
        cfg: Hydra Config object
        mode: 'train', 'test', or 'tune'
        fast_dev_run: If True, config overrides have already been applied in setup_config()

    Returns:
        ConnectomicsDataModule instance
    """
    print("Creating datasets...")

    # Auto-download tutorial data if missing
    if mode == "train" and cfg.data.train_image:
        from pathlib import Path as PathLib

        # Check if data exists (support glob patterns and lists)
        data_exists = False

        # Handle list of files
        if isinstance(cfg.data.train_image, list):
            # Check if at least one file in the list exists
            data_exists = any(PathLib(img).exists() for img in cfg.data.train_image)
        # Handle glob pattern
        elif "*" in cfg.data.train_image or "?" in cfg.data.train_image:
            # Glob pattern - check if any files match
            matched_files = glob(cfg.data.train_image)
            data_exists = len(matched_files) > 0
        # Handle single file path
        else:
            data_exists = PathLib(cfg.data.train_image).exists()

        if not data_exists:
            print(f"\n⚠️  Training data not found: {cfg.data.train_image}")

            # Try to infer dataset name from path
            from ...utils.download import DATASETS, download_dataset

            path_str = str(cfg.data.train_image).lower()
            dataset_name = None
            for name in DATASETS.keys():
                if name in path_str and not name.endswith("++"):  # Skip aliases
                    dataset_name = name
                    break

            if dataset_name:
                print(f"💡 Attempting to auto-download '{dataset_name}' dataset...")
                print("   (You can disable auto-download by manually downloading data)")

                # Prompt user
                try:
                    size_mb = DATASETS[dataset_name]["size_mb"]
                    prompt = f"   Download {dataset_name} dataset (~{size_mb} MB)? [Y/n]: "
                    response = input(prompt).strip().lower()
                    if response in ["", "y", "yes"]:
                        if download_dataset(dataset_name, base_dir=PathLib.cwd()):
                            print("✅ Data downloaded successfully!")
                        else:
                            print("❌ Download failed. Please download manually:")
                            print(f"   wget {DATASETS[dataset_name]['url']}")
                            raise FileNotFoundError(
                                f"Training data not found: {cfg.data.train_image}"
                            )
                    else:
                        print("❌ Download cancelled. Please download manually.")
                        raise FileNotFoundError(f"Training data not found: {cfg.data.train_image}")
                except KeyboardInterrupt:
                    print("\n❌ Download cancelled by user")
                    raise FileNotFoundError(f"Training data not found: {cfg.data.train_image}")
            else:
                print("💡 Available datasets:")
                from ...utils.download import list_datasets

                list_datasets()
                raise FileNotFoundError(f"Training data not found: {cfg.data.train_image}")

    # Check dataset type early
    dataset_type = getattr(cfg.data, "dataset_type", None)

    # Build transforms
    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)
    test_transforms = build_test_transforms(cfg) if mode in ["test", "tune"] else val_transforms

    print(f"  Train transforms: {len(train_transforms.transforms)} steps")
    print(f"  Val transforms: {len(val_transforms.transforms)} steps")
    if mode in ["test", "tune"]:
        print(
            f"  Test transforms: {len(test_transforms.transforms)} steps (no sliding-window crop)"
        )

    # For test/tune modes, skip training data setup entirely
    if mode in ["test", "tune"]:
        train_data_dicts = []
        val_data_dicts = None
    # Check if automatic train/val split is enabled
    elif cfg.data.split_enabled and not cfg.data.val_image:
        print("🔀 Using automatic train/val split (DeepEM-style)")
        # Load full volume
        import h5py
        import tifffile

        from ...data.utils.split import split_volume_train_val

        train_path = Path(cfg.data.train_image)
        if train_path.suffix in [".h5", ".hdf5"]:
            with h5py.File(train_path, "r") as f:
                volume_shape = f[list(f.keys())[0]].shape
        elif train_path.suffix in [".tif", ".tiff"]:
            volume = tifffile.imread(train_path)
            volume_shape = volume.shape
        else:
            raise ValueError(f"Unsupported file format: {train_path.suffix}")

        print(f"  Volume shape: {volume_shape}")

        # Calculate split ranges
        train_ratio = cfg.data.split_train_range[1] - cfg.data.split_train_range[0]

        train_slices, val_slices = split_volume_train_val(
            volume_shape=volume_shape,
            train_ratio=train_ratio,
            axis=cfg.data.split_axis,
        )

        # Calculate train and val regions
        axis = cfg.data.split_axis
        train_start = int(volume_shape[axis] * cfg.data.split_train_range[0])
        train_end = int(volume_shape[axis] * cfg.data.split_train_range[1])
        val_start = int(volume_shape[axis] * cfg.data.split_val_range[0])
        val_end = int(volume_shape[axis] * cfg.data.split_val_range[1])

        print(f"  Split axis: {axis} ({'Z' if axis == 0 else 'Y' if axis == 1 else 'X'})")
        print(f"  Train region: [{train_start}:{train_end}] ({train_end - train_start} slices)")
        print(f"  Val region: [{val_start}:{val_end}] ({val_end - val_start} slices)")

        if cfg.data.split_pad_val:
            target_size = tuple(cfg.data.patch_size)
            print(f"  Val padding enabled: target size = {target_size}")

        # Create data dictionaries with split info
        train_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train_image],
            label_paths=[cfg.data.train_label] if cfg.data.train_label else None,
        )

        # Add split metadata to train dict
        train_data_dicts[0]["split_slices"] = train_slices
        train_data_dicts[0]["split_mode"] = "train"

        # Create validation data dicts using same volume
        val_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train_image],
            label_paths=[cfg.data.train_label] if cfg.data.train_label else None,
        )

        # Add split metadata to val dict
        val_data_dicts[0]["split_slices"] = val_slices
        val_data_dicts[0]["split_mode"] = "val"
        val_data_dicts[0]["split_pad"] = cfg.data.split_pad_val
        val_data_dicts[0]["split_pad_mode"] = cfg.data.split_pad_mode
        if cfg.data.split_pad_val:
            val_data_dicts[0]["split_pad_size"] = tuple(cfg.data.patch_size)

    else:
        # Check dataset type to determine how to load data
        if dataset_type == "filename":
            # Check if train_json is empty or doesn't exist
            train_json_empty = False
            if cfg.data.train_json is None or cfg.data.train_json == "":
                train_json_empty = True
            else:
                try:
                    import json

                    json_path = Path(cfg.data.train_json)
                    if not json_path.exists():
                        train_json_empty = True
                    else:
                        # Check if JSON file is empty or has no images
                        with open(json_path, "r") as f:
                            json_data = json.load(f)
                        image_files = json_data.get(cfg.data.train_image_key, [])
                        if not image_files:
                            train_json_empty = True
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    train_json_empty = True

            if train_json_empty:
                # Fallback to volume-based dataset when train_json is empty
                print("  ⚠️  Train JSON is empty or invalid, falling back to volume-based dataset")
                print(f"  Train JSON: {cfg.data.train_json}")
                dataset_type = None  # Switch to volume-based
            else:
                # Filename-based dataset: uses JSON file lists
                print("  Using filename-based dataset")
                print(f"  Train JSON: {cfg.data.train_json}")
                print(f"  Image key: {cfg.data.train_image_key}")
                print(f"  Label key: {cfg.data.train_label_key}")

                # For filename dataset, we'll create data dicts later in the DataModule
                # Here we just need placeholder dicts
                train_data_dicts = [{"dataset_type": "filename"}]
                val_data_dicts = None  # Handled by train_val_split in DataModule

        if dataset_type != "filename":
            # Standard mode: separate train and val files (supports glob patterns)
            if cfg.data.train_image is None:
                raise ValueError(
                    "For volume-based datasets, data.train_image must be specified.\n"
                    "Either set data.train_image or use data.dataset_type='filename' with "
                    "data.train_json"
                )

            train_image_paths = expand_file_paths(cfg.data.train_image)
            train_label_paths = (
                expand_file_paths(cfg.data.train_label) if cfg.data.train_label else None
            )
            train_mask_paths = (
                expand_file_paths(cfg.data.train_mask) if cfg.data.train_mask else None
            )

            print(f"  Training volumes: {len(train_image_paths)} files")
            if len(train_image_paths) <= 5:
                for path in train_image_paths:
                    print(f"    - {path}")
            else:
                print(f"    - {train_image_paths[0]}")
                print(f"    - ... ({len(train_image_paths) - 2} more files)")
                print(f"    - {train_image_paths[-1]}")

            if train_mask_paths:
                print(f"  Training masks: {len(train_mask_paths)} files")

            train_data_dicts = create_data_dicts_from_paths(
                image_paths=train_image_paths,
                label_paths=train_label_paths,
                mask_paths=train_mask_paths,
            )

            val_data_dicts = None
            if cfg.data.val_image:
                val_image_paths = expand_file_paths(cfg.data.val_image)
                val_label_paths = (
                    expand_file_paths(cfg.data.val_label) if cfg.data.val_label else None
                )
                val_mask_paths = expand_file_paths(cfg.data.val_mask) if cfg.data.val_mask else None

                print(f"  Validation volumes: {len(val_image_paths)} files")
                if val_mask_paths:
                    print(f"  Validation masks: {len(val_mask_paths)} files")

                val_data_dicts = create_data_dicts_from_paths(
                    image_paths=val_image_paths,
                    label_paths=val_label_paths,
                    mask_paths=val_mask_paths,
                )

    # Create test data dicts if in test or tune mode
    test_data_dicts = None
    if mode == "test":
        if (
            not hasattr(cfg, "test")
            or cfg.test is None
            or not hasattr(cfg.test, "data")
            or not cfg.test.data.test_image
        ):
            test_image_val = (
                cfg.test.data.test_image
                if hasattr(cfg, "test") and cfg.test and hasattr(cfg.test, "data")
                else "N/A"
            )
            raise ValueError(
                f"Test mode requires test.data.test_image to be set in config.\n"
                f"Current config has: test.data.test_image = {test_image_val}"
            )
        print(f"  🧪 Creating test dataset from: {cfg.test.data.test_image}")

        # Expand glob patterns for test data (same as train data)
        test_image_paths = expand_file_paths(cfg.test.data.test_image)
        test_label_paths = (
            expand_file_paths(cfg.test.data.test_label) if cfg.test.data.test_label else None
        )
        test_mask_paths = (
            expand_file_paths(cfg.test.data.test_mask)
            if hasattr(cfg.test.data, "test_mask") and cfg.test.data.test_mask
            else None
        )
    elif mode == "tune":
        # For tune mode, read from cfg.tune.data
        if (
            not hasattr(cfg, "tune")
            or cfg.tune is None
            or not hasattr(cfg.tune, "data")
            or not cfg.tune.data.tune_image
        ):
            tune_image_val = (
                cfg.tune.data.tune_image
                if hasattr(cfg, "tune") and cfg.tune and hasattr(cfg.tune, "data")
                else "N/A"
            )
            raise ValueError(
                f"Tune mode requires tune.data.tune_image to be set in config.\n"
                f"Current config has tune.data.tune_image: {tune_image_val}"
            )

        print(f"  🎯 Creating tune dataset from: {cfg.tune.data.tune_image}")

        # Expand glob patterns for tune data
        test_image_paths = expand_file_paths(cfg.tune.data.tune_image)
        test_label_paths = (
            expand_file_paths(cfg.tune.data.tune_label) if cfg.tune.data.tune_label else None
        )
        test_mask_paths = (
            expand_file_paths(cfg.tune.data.tune_mask) if cfg.tune.data.tune_mask else None
        )

    # Common printing and data dict creation for test and tune modes
    if mode in ["test", "tune"]:
        mode_label = "Test" if mode == "test" else "Tune"
        print(f"  {mode_label} volumes: {len(test_image_paths)} files")
        if len(test_image_paths) <= 5:
            for path in test_image_paths:
                print(f"    - {path}")
        else:
            print(f"    - {test_image_paths[0]}")
            print(f"    - ... ({len(test_image_paths) - 2} more files)")
            print(f"    - {test_image_paths[-1]}")

        if test_mask_paths:
            print(f"  {mode_label} masks: {len(test_mask_paths)} files")

        test_data_dicts = create_data_dicts_from_paths(
            image_paths=test_image_paths,
            label_paths=test_label_paths,
            mask_paths=test_mask_paths,
        )
        print(f"  {mode_label} dataset size: {len(test_data_dicts)}")

    if mode == "train":
        print(f"  Train dataset size: {len(train_data_dicts)}")
        if val_data_dicts:
            print(f"  Val dataset size: {len(val_data_dicts)}")

    # Auto-compute iter_num from volume size if not specified (only for training)
    iter_num = None
    if mode == "train":
        iter_num = cfg.data.iter_num_per_epoch
        if iter_num == -1 and dataset_type != "filename":
            # For filename datasets, iter_num is determined by the number of files
            print("📊 Auto-computing iter_num from volume size...")
            import h5py
            import tifffile

            from ...data.utils import compute_total_samples

            # Get volume sizes
            volume_sizes = []
            for data_dict in train_data_dicts:
                img_path = Path(data_dict["image"])
                if img_path.suffix in [".h5", ".hdf5"]:
                    with h5py.File(img_path, "r") as f:
                        vol_shape = f[list(f.keys())[0]].shape
                elif img_path.suffix in [".tif", ".tiff"]:
                    vol = tifffile.imread(img_path)
                    vol_shape = vol.shape
                else:
                    raise ValueError(f"Unsupported file format: {img_path.suffix}")

                # Handle both (z, y, x) and (c, z, y, x)
                if len(vol_shape) == 4:
                    vol_shape = vol_shape[1:]  # Skip channel dim
                volume_sizes.append(vol_shape)

            # Compute total possible samples
            total_samples, samples_per_vol = compute_total_samples(
                volume_sizes=volume_sizes,
                patch_size=tuple(cfg.data.patch_size),
                stride=tuple(cfg.data.stride),
            )

            iter_num = total_samples
            print(f"  Volume sizes: {volume_sizes}")
            print(f"  Patch size: {cfg.data.patch_size}")
            print(f"  Stride: {cfg.data.stride}")
            print(f"  Samples per volume: {samples_per_vol}")
            print(f"  ✅ Total possible samples (iter_num): {iter_num:,}")
            print(f"  ✅ Batches per epoch: {iter_num // cfg.system.training.batch_size:,}")
        elif iter_num == -1 and dataset_type == "filename":
            # For filename datasets, iter_num will be determined by dataset length
            print("  Filename dataset: iter_num will be determined by number of files in JSON")

    # Create DataModule
    print("Creating data loaders...")

    # For test/tune modes, disable iter_num (process full volumes once)
    if mode in ["test", "tune"]:
        iter_num_for_dataset = -1  # Process full volumes without random sampling
    else:
        iter_num_for_dataset = iter_num

    # Select appropriate batch_size and num_workers based on mode
    # For test/tune modes, use system.inference settings instead of system.training
    if mode in ["test", "tune", "tune-test"]:
        batch_size = cfg.system.inference.batch_size
        num_workers = cfg.system.inference.num_workers
        print(f"  Using inference settings: batch_size={batch_size}, num_workers={num_workers}")
    else:
        batch_size = cfg.system.training.batch_size
        num_workers = cfg.system.training.num_workers
        print(f"  Using training settings: batch_size={batch_size}, num_workers={num_workers}")

    # Use optimized pre-loaded cache when iter_num > 0 (only for training mode and volume datasets)
    use_preloaded = (
        cfg.data.use_preloaded_cache
        and iter_num is not None
        and iter_num > 0
        and mode == "train"
        and dataset_type != "filename"
    )

    if use_preloaded:
        print("  ⚡ Using pre-loaded volume cache (loads once, crops in memory)")
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        from ...data.dataset.dataset_volume_cached import CachedVolumeDataset

        # Build transforms without loading/cropping (handled by dataset)
        augment_only_transforms = build_train_transforms(cfg, skip_loading=True)

        # Get padding parameters from config
        pad_size = getattr(cfg.data.image_transform, "pad_size", None)
        pad_mode = getattr(cfg.data.image_transform, "pad_mode", "reflect")

        # Create optimized cached datasets
        train_dataset = CachedVolumeDataset(
            image_paths=[d["image"] for d in train_data_dicts],
            label_paths=[d.get("label") for d in train_data_dicts],
            patch_size=tuple(cfg.data.patch_size),
            iter_num=iter_num,
            transforms=augment_only_transforms,
            mode="train",
            pad_size=tuple(pad_size) if pad_size else None,
            pad_mode=pad_mode,
        )

        # Use fewer workers since we're loading from memory
        preloaded_num_workers = min(num_workers, 2)
        print(f"  Using {preloaded_num_workers} workers (in-memory operations are fast)")

        # Create simple dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Already random
            num_workers=preloaded_num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=preloaded_num_workers > 0,
        )

        # Create validation dataset and loader if validation data exists
        val_loader = None
        if val_data_dicts and len(val_data_dicts) > 0:
            print("  Creating validation dataset with pre-loaded cache...")

            # Build validation transforms (no augmentation, only normalization)
            val_only_transforms = build_val_transforms(cfg, skip_loading=True)

            # Get validation iter_num (auto-calculate if not specified)
            val_iter_num = (
                cfg.data.val_iter_num
                if hasattr(cfg.data, "val_iter_num") and cfg.data.val_iter_num
                else None
            )

            if val_iter_num is None:
                # Auto-calculate validation iter_num from volume size
                print("  📊 Auto-calculating validation iter_num from volume size...")
                val_iter_num = _calculate_validation_iter_num(
                    val_data_dicts=val_data_dicts,
                    patch_size=tuple(cfg.data.patch_size),
                    min_iter=1,
                    max_iter=None,
                    default_iter_num=100,
                    fallback_volume_shape=(100, 4096, 4096),
                    return_default_on_error=False,
                )
                print(f"  ✅ Validation iter_num: {val_iter_num} (auto-calculated)")

            # Create validation dataset
            val_dataset = CachedVolumeDataset(
                image_paths=[d["image"] for d in val_data_dicts],
                label_paths=[d.get("label") for d in val_data_dicts],
                patch_size=tuple(cfg.data.patch_size),
                iter_num=val_iter_num,
                transforms=val_only_transforms,
                mode="val",
                pad_size=tuple(pad_size) if pad_size else None,
                pad_mode=pad_mode,
            )

            # Create validation dataloader
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=preloaded_num_workers,
                pin_memory=cfg.data.pin_memory,
                persistent_workers=preloaded_num_workers > 0,
            )
            print(f"  ✅ Validation dataloader created with {val_iter_num} iterations")

        # Create data module wrapper that inherits from LightningDataModule
        class SimpleDataModule(pl.LightningDataModule):
            def __init__(self, train_loader, val_loader=None):
                super().__init__()
                self.train_loader = train_loader
                self._val_loader = val_loader

            def train_dataloader(self):
                return self.train_loader

            def val_dataloader(self):
                if self._val_loader is not None:
                    return self._val_loader
                return []

            def test_dataloader(self):
                # For test mode, return empty list (user should use standard datamodule)
                return []

            def setup(self, stage=None):
                pass

        datamodule = SimpleDataModule(train_loader, val_loader)
    elif dataset_type == "filename":
        # Filename-based dataset using JSON file lists
        print("  Creating filename-based datamodule...")
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        from ...data.dataset.dataset_filename import create_filename_datasets

        # Create train and val datasets from JSON
        train_dataset, val_dataset = create_filename_datasets(
            json_path=cfg.data.train_json,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            train_val_split=(
                cfg.data.train_val_split if hasattr(cfg.data, "train_val_split") else 0.9
            ),
            random_seed=cfg.system.seed if hasattr(cfg.system, "seed") else 42,
            images_key=cfg.data.train_image_key,
            labels_key=cfg.data.train_label_key,
            use_labels=True,
        )

        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Val dataset size: {len(val_dataset)}")

        # Create simple datamodule wrapper
        class FilenameDataModule(pl.LightningDataModule):
            def __init__(
                self, train_ds, val_ds, batch_size, num_workers, pin_memory, persistent_workers
            ):
                super().__init__()
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.batch_size = batch_size
                self.num_workers = num_workers
                self.pin_memory = pin_memory
                self.persistent_workers = persistent_workers

            def train_dataloader(self):
                return DataLoader(
                    self.train_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers and self.num_workers > 0,
                )

            def val_dataloader(self):
                if self.val_ds is None or len(self.val_ds) == 0:
                    return []
                return DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers and self.num_workers > 0,
                )

            def test_dataloader(self):
                return []

            def setup(self, stage=None):
                pass

        datamodule = FilenameDataModule(
            train_ds=train_dataset,
            val_ds=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
        )
    else:
        # Standard data module
        # Disable caching for test/tune modes to avoid issues with partial cache returning 0 length
        use_cache = cfg.data.use_cache and mode == "train"

        if mode in ["test", "tune"] and cfg.data.use_cache:
            print("  ⚠️  Caching disabled for test/tune mode (incompatible with partial cache)")

        # Note: transpose_axes handled in transform builders (build_train/val/test_transforms)
        # They embed the transpose in LoadVolumed, so no need to pass it here

        # Get validation iter_num (separate from training iter_num)
        val_iter_num = getattr(cfg.data, "val_iter_num", None)
        if val_iter_num is None and val_data_dicts:
            # Auto-calculate validation iter_num based on volume size and patch size
            print("  📊 Auto-calculating validation iter_num from volume size...")
            val_iter_num = _calculate_validation_iter_num(
                val_data_dicts=val_data_dicts,
                patch_size=tuple(cfg.data.patch_size),
                min_iter=50,
                max_iter=200,
            )
            print(f"  ✅ Validation iter_num: {val_iter_num} (auto-calculated)")

        datamodule = ConnectomicsDataModule(
            train_data_dicts=train_data_dicts,
            val_data_dicts=val_data_dicts,
            test_data_dicts=test_data_dicts,
            transforms={
                "train": train_transforms,
                "val": val_transforms,
                "test": test_transforms,
            },
            dataset_type="cached" if use_cache else "standard",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
            cache_rate=cfg.data.cache_rate if use_cache else 0.0,
            iter_num=iter_num_for_dataset,
            val_iter_num=val_iter_num,
            seed=cfg.system.seed,  # [FIX 1] Pass seed for validation reseeding
            sample_size=tuple(cfg.data.patch_size),
            do_2d=cfg.data.do_2d,
        )
        # Setup datasets based on mode
        if mode == "train":
            datamodule.setup(stage="fit")
        elif mode in ["test", "tune"]:
            datamodule.setup(stage="test")

    # Print dataset info based on mode
    if mode == "train":
        print(f"  Train batches: {len(datamodule.train_dataloader())}")
        if val_data_dicts:
            print(f"  Val batches: {len(datamodule.val_dataloader())}")
    elif mode in ["test", "tune"]:
        print(f"  Test batches: {len(datamodule.test_dataloader())}")

    return datamodule


def setup_run_directory(mode: str, cfg, checkpoint_dirpath: str):
    """
    Setup run directory with timestamp for training mode.
    Handles DDP subprocess coordination via timestamp files.

    Args:
        mode: 'train', 'test', 'tune', or 'tune-test'
        cfg: Config object (will be modified in-place for training mode)
        checkpoint_dirpath: Path to checkpoint/output directory from config

    Returns:
        Path: Run directory (timestamped for train, created for tune modes, dummy for test)
    """
    import os
    from datetime import datetime

    from ...config import save_config

    checkpoint_dir = Path(checkpoint_dirpath)
    checkpoint_subdir = checkpoint_dir.name or "checkpoints"
    output_base = checkpoint_dir.parent

    if mode == "train":
        # Check if this is a DDP re-launch (LOCAL_RANK is set by PyTorch Lightning)
        is_ddp_subprocess = "LOCAL_RANK" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        timestamp_file = output_base / ".latest_timestamp"

        if not is_ddp_subprocess:
            # First invocation (main process) - create new timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_base / timestamp
            checkpoint_path = run_dir / checkpoint_subdir
            cfg.monitor.checkpoint.dirpath = str(checkpoint_path)

            checkpoint_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Run directory: {run_dir}")

            # Save config to run directory
            config_save_path = run_dir / "config.yaml"
            save_config(cfg, config_save_path)
            print(f"💾 Config saved to: {config_save_path}")

            # Save timestamp for DDP subprocesses to read
            output_base.mkdir(parents=True, exist_ok=True)
            timestamp_file.write_text(timestamp)
        else:
            # DDP subprocess - read existing timestamp
            import time

            max_wait = 30  # Maximum 30 seconds
            waited = 0
            while not timestamp_file.exists() and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            if timestamp_file.exists():
                timestamp = timestamp_file.read_text().strip()
                run_dir = output_base / timestamp
                checkpoint_path = run_dir / checkpoint_subdir
                cfg.monitor.checkpoint.dirpath = str(checkpoint_path)
                print(f"📁 [DDP Rank {local_rank}] Using run directory: {run_dir}")
            else:
                raise RuntimeError(
                    f"DDP subprocess (LOCAL_RANK={local_rank}) timed out waiting for timestamp file"
                )
    elif mode in ["tune", "tune-test"]:
        # For tune modes, create the directory for Optuna outputs
        run_dir = Path(checkpoint_dirpath)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Tuning output directory: {run_dir}")
    elif mode == "test":
        # For test mode, create the results directory
        run_dir = Path(checkpoint_dirpath)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Test output directory: {run_dir}")
    else:
        # Fallback for unknown modes
        run_dir = output_base / f"{mode}_run"
        print(f"📝 Running in {mode} mode")

    return run_dir


def cleanup_run_directory(output_base: Path):
    """
    Clean up timestamp file after training completes.
    Only runs in main process (not DDP subprocesses).

    Args:
        output_base: Base output directory containing timestamp file
    """
    import os

    is_ddp_subprocess = "LOCAL_RANK" in os.environ
    if not is_ddp_subprocess:
        timestamp_file = output_base / ".latest_timestamp"
        if timestamp_file.exists():
            try:
                timestamp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


def modify_checkpoint_state(
    checkpoint_path: Optional[str],
    run_dir: Path,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
    reset_epoch: bool = False,
    reset_early_stopping: bool = False,
) -> Optional[str]:
    """
    Modify checkpoint state for training resumption with selective resets.

    Args:
        checkpoint_path: Path to checkpoint file to modify (None if no checkpoint)
        run_dir: Run directory for saving modified checkpoint
        reset_optimizer: If True, reset optimizer state
        reset_scheduler: If True, reset scheduler state
        reset_epoch: If True, reset epoch counter
        reset_early_stopping: If True, reset early stopping patience

    Returns:
        Path to modified checkpoint file, or original path if no modifications needed
    """
    import torch

    # Early return if no checkpoint or no resets requested
    if not checkpoint_path:
        return None

    if not (reset_optimizer or reset_scheduler or reset_epoch or reset_early_stopping):
        return checkpoint_path

    print("\n🔄 Modifying checkpoint state:")
    if reset_optimizer:
        print("   - Resetting optimizer state")
    if reset_scheduler:
        print("   - Resetting scheduler state")
    if reset_epoch:
        print("   - Resetting epoch counter")
    if reset_early_stopping:
        print("   - Resetting early stopping patience counter")

    # Load checkpoint (weights_only=False needed for PyTorch 2.6+ to load Lightning
    # checkpoints with custom configs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reset optimizer state
    if reset_optimizer and "optimizer_states" in checkpoint:
        del checkpoint["optimizer_states"]

    # Reset scheduler state
    if reset_scheduler and "lr_schedulers" in checkpoint:
        del checkpoint["lr_schedulers"]

    # Reset epoch counter
    if reset_epoch:
        if "epoch" in checkpoint:
            checkpoint["epoch"] = 0
        if "global_step" in checkpoint:
            checkpoint["global_step"] = 0

    # Reset early stopping state
    if reset_early_stopping and "callbacks" in checkpoint:
        for callback_state in checkpoint["callbacks"].values():
            if "wait_count" in callback_state:
                callback_state["wait_count"] = 0
            if "best_score" in callback_state:
                callback_state["best_score"] = None

    # Save modified checkpoint to temporary file
    temp_ckpt_path = run_dir / "temp_modified_checkpoint.ckpt"
    torch.save(checkpoint, temp_ckpt_path)
    print(f"   ✅ Modified checkpoint saved to: {temp_ckpt_path}")

    return str(temp_ckpt_path)


__all__ = [
    "setup_seed_everything",
    "expand_file_paths",
    "create_datamodule",
    "setup_run_directory",
    "cleanup_run_directory",
    "modify_checkpoint_state",
]
