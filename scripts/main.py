#!/usr/bin/env python3
"""
PyTorch Connectomics training script with Hydra configuration and Lightning framework.

This script provides modern deep learning training with:
- Hydra-based configuration management
- Automatic distributed training and mixed precision
- MONAI-based data augmentation
- PyTorch Lightning callbacks and logging

Usage:
    # Basic training
    python scripts/main.py --config tutorials/lucchi.yaml

    # Testing mode
    python scripts/main.py --config tutorials/lucchi.yaml --mode test --checkpoint path/to/checkpoint.ckpt

    # Fast dev run (1 batch for debugging, auto-sets num_gpus=1, num_cpus=1, num_workers=1)
    python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run
    python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run 2  # Run 2 batches

    # Override config parameters
    python scripts/main.py --config tutorials/lucchi.yaml data.batch_size=8 optimization.max_epochs=200

    # Resume training with different max_epochs
    python scripts/main.py --config tutorials/lucchi.yaml --checkpoint path/to/ckpt.ckpt --reset-max-epochs 500
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Import visualization callback
from connectomics.lightning.callbacks import VisualizationCallback

# Handle different Lightning versions
try:
    from pytorch_lightning.utilities.seed import seed_everything
except ImportError:
    try:
        from pytorch_lightning import seed_everything
    except ImportError:
        # Fallback for older versions
        def seed_everything(seed, workers=True):
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


# Import Hydra config system
from connectomics.config import (
    load_config,
    save_config,
    update_from_cli,
    validate_config,
    resolve_data_paths,
    Config,
)

# Import data and model utilities
from connectomics.data.dataset import create_data_dicts_from_paths
from connectomics.lightning import (
    ConnectomicsDataModule,
    ConnectomicsModule,
)
from connectomics.data.augment.build import (
    build_train_transforms,
    build_val_transforms,
    build_test_transforms,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Connectomics Training with Hydra Config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to Hydra YAML config file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo with synthetic data (30 seconds, no config needed)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "predict"],
        default="train",
        help="Mode: train, test, or predict (default: train)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to checkpoint for resuming/testing/prediction",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Reset optimizer state when loading checkpoint (useful for changing learning rate)",
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="Reset scheduler state when loading checkpoint",
    )
    parser.add_argument(
        "--reset-epoch",
        action="store_true",
        help="Reset epoch counter when loading checkpoint (start from epoch 0)",
    )
    parser.add_argument(
        "--reset-early-stopping",
        action="store_true",
        help="Reset early stopping patience counter when loading checkpoint",
    )
    parser.add_argument(
        "--reset-max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config (useful when resuming training with different epoch count)",
    )
    parser.add_argument(
        "--fast-dev-run",
        type=int,
        nargs='?',
        const=1,
        default=0,
        help="Run N batches for quick debugging (default: 0, no argument defaults to 1)",
    )
    parser.add_argument(
        "--external-prefix",
        type=str,
        default=None,
        help="Prefix to strip from external checkpoint keys (e.g., 'model.' for BANIS checkpoints)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., data.batch_size=8)",
    )

    return parser.parse_args()


def setup_config(args) -> Config:
    """
    Setup configuration from YAML file and CLI overrides.

    Args:
        args: Command line arguments

    Returns:
        Validated Config object
    """
    # Load base config from YAML
    print(f"📄 Loading config: {args.config}")
    cfg = load_config(args.config)

    # Resolve data paths (combine train_path with train_image, etc.)
    cfg = resolve_data_paths(cfg)

    # Extract config file name and set output folder
    config_path = Path(args.config)
    config_name = config_path.stem  # Get filename without extension
    output_folder = f"outputs/{config_name}/"

    # Update checkpoint dirpath only if not provided by the user
    if not getattr(cfg.monitor.checkpoint, "dirpath", None):
        cfg.monitor.checkpoint.dirpath = str(Path(output_folder) / "checkpoints")
    else:
        cfg.monitor.checkpoint.dirpath = str(Path(cfg.monitor.checkpoint.dirpath))

    # Update inference output path only if not provided by the user
    if not getattr(cfg.inference.data, "output_path", None):
        cfg.inference.data.output_path = str(Path(output_folder) / "results")
    else:
        cfg.inference.data.output_path = str(Path(cfg.inference.data.output_path))

    # Note: We handle timestamping manually in main() to create run directories
    # Set this to False to prevent PyTorch Lightning from adding its own timestamp
    cfg.monitor.checkpoint.use_timestamp = False

    print(f"📁 Checkpoints base directory: {cfg.monitor.checkpoint.dirpath}")
    print(f"📂 Inference output directory: {cfg.inference.data.output_path}")

    # Apply CLI overrides
    if args.overrides:
        print(f"⚙️  Applying {len(args.overrides)} CLI overrides")
        cfg = update_from_cli(cfg, args.overrides)

    # Override max_epochs if --reset-max-epochs is specified
    if args.reset_max_epochs is not None:
        print(f"⚙️  Overriding max_epochs: {cfg.optimization.max_epochs} → {args.reset_max_epochs}")
        cfg.optimization.max_epochs = args.reset_max_epochs

    # Handle external weights loading (when --external-prefix is specified with --checkpoint)
    if args.external_prefix is not None and args.checkpoint:
        print(f"🔧 Loading external weights from checkpoint with prefix '{args.external_prefix}'")
        cfg.model.external_weights_path = args.checkpoint
        cfg.model.external_weights_key_prefix = args.external_prefix

    # Override config for fast-dev-run mode
    if args.fast_dev_run:
        print(f"🔧 Fast-dev-run mode: Overriding config for debugging")
        print(f"   - num_gpus: {cfg.system.training.num_gpus} → 1")
        print(f"   - num_cpus: {cfg.system.training.num_cpus} → 1")
        print(f"   - num_workers: {cfg.system.training.num_workers} → 1")
        print(
            f"   - batch_size: Controlled by PyTorch Lightning (--fast-dev-run={args.fast_dev_run})"
        )
        cfg.system.training.num_gpus = 1
        cfg.system.training.num_cpus = 1
        cfg.system.training.num_workers = 1
        cfg.system.inference.num_gpus = 1
        cfg.system.inference.num_cpus = 1
        cfg.system.inference.num_workers = 1

    # Apply inference-specific overrides if in test/predict mode
    if args.mode in ["test", "predict"]:
        if cfg.inference.num_gpus >= 0:
            print(f"🔧 Inference override: num_gpus={cfg.inference.num_gpus}")
            cfg.system.training.num_gpus = cfg.inference.num_gpus
        if cfg.inference.num_cpus >= 0:
            print(f"🔧 Inference override: num_cpus={cfg.inference.num_cpus}")
            cfg.system.training.num_cpus = cfg.inference.num_cpus
        if cfg.inference.batch_size >= 0:
            print(f"🔧 Inference override: batch_size={cfg.inference.batch_size}")
            cfg.system.inference.batch_size = cfg.inference.batch_size
        if cfg.inference.num_workers >= 0:
            print(f"🔧 Inference override: num_workers={cfg.inference.num_workers}")
            cfg.system.inference.num_workers = cfg.inference.num_workers

    # Auto-planning (if enabled)
    if hasattr(cfg.system, "auto_plan") and cfg.system.auto_plan:
        print("🤖 Running automatic configuration planning...")
        from connectomics.config import auto_plan_config

        print_results = (
            cfg.system.print_auto_plan if hasattr(cfg.system, "print_auto_plan") else True
        )
        cfg = auto_plan_config(cfg, print_results=print_results)

    # Validate configuration
    print("✅ Validating configuration...")
    validate_config(cfg)

    # Note: Output directory will be created later in main() with timestamp
    # (see lines around "Create run directory only for training mode")

    return cfg


def expand_file_paths(path_or_pattern) -> List[str]:
    """
    Expand glob patterns to list of file paths.

    Args:
        path_or_pattern: Single file path, glob pattern, or list of paths/patterns

    Returns:
        List of expanded file paths, sorted alphabetically
    """
    from glob import glob

    # If already a list, return it (may have been expanded by resolve_data_paths)
    if isinstance(path_or_pattern, list):
        return path_or_pattern

    # Check if pattern contains wildcards
    if "*" in path_or_pattern or "?" in path_or_pattern:
        # Expand glob pattern
        paths = sorted(glob(path_or_pattern))
        if not paths:
            raise FileNotFoundError(f"No files found matching pattern: {path_or_pattern}")
        return paths
    else:
        # Single file path
        return [path_or_pattern]


def create_datamodule(
    cfg: Config, mode: str = "train", fast_dev_run: bool = False
) -> ConnectomicsDataModule:
    """
    Create Lightning DataModule from config.

    Args:
        cfg: Hydra Config object
        mode: 'train', 'test', or 'predict'
        fast_dev_run: If True, config overrides have already been applied in setup_config()

    Returns:
        VolumeDataModule instance
    """
    print("Creating datasets...")

    # Auto-download tutorial data if missing
    if mode == "train" and cfg.data.train_image:
        from glob import glob
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
            from connectomics.utils.download import DATASETS, download_dataset

            path_str = str(cfg.data.train_image).lower()
            dataset_name = None
            for name in DATASETS.keys():
                if name in path_str and not name.endswith("++"):  # Skip aliases
                    dataset_name = name
                    break

            if dataset_name:
                print(f"💡 Attempting to auto-download '{dataset_name}' dataset...")
                print(f"   (You can disable auto-download by manually downloading data)")

                # Prompt user
                try:
                    response = (
                        input(
                            f"   Download {dataset_name} dataset (~{DATASETS[dataset_name]['size_mb']} MB)? [Y/n]: "
                        )
                        .strip()
                        .lower()
                    )
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
                from connectomics.utils.download import list_datasets

                list_datasets()
                raise FileNotFoundError(f"Training data not found: {cfg.data.train_image}")

    # Check dataset type early
    dataset_type = getattr(cfg.data, "dataset_type", None)

    # Build transforms
    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)
    test_transforms = build_test_transforms(cfg) if mode in ["test", "predict"] else val_transforms

    print(f"  Train transforms: {len(train_transforms.transforms)} steps")
    print(f"  Val transforms: {len(val_transforms.transforms)} steps")
    if mode in ["test", "predict"]:
        print(
            f"  Test transforms: {len(test_transforms.transforms)} steps (no cropping for sliding window)"
        )

    # For test/predict mode, skip training data setup entirely
    if mode in ["test", "predict"]:
        train_data_dicts = []
        val_data_dicts = None
    # Check if automatic train/val split is enabled
    elif cfg.data.split_enabled and not cfg.data.val_image:
        print("🔀 Using automatic train/val split (DeepEM-style)")
        from connectomics.data.utils.split import (
            split_volume_train_val,
        )

        # Load full volume
        import h5py
        import tifffile

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
        cfg.data.split_train_range[1]

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
                    from pathlib import Path
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
                print(f"  ⚠️  Train JSON is empty or invalid, falling back to volume-based dataset")
                print(f"  Train JSON: {cfg.data.train_json}")
                dataset_type = None  # Switch to volume-based
            else:
                # Filename-based dataset: uses JSON file lists
                print(f"  Using filename-based dataset")
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
                    "Either set data.train_image or use data.dataset_type='filename' with data.train_json"
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

    # Create test data dicts if in test/predict mode
    test_data_dicts = None
    if mode in ["test", "predict"]:
        if (
            not hasattr(cfg, "inference")
            or not hasattr(cfg.inference, "data")
            or not cfg.inference.data.test_image
        ):
            raise ValueError(
                f"Test mode requires inference.data.test_image to be set in config.\n"
                f"Current config has: inference.data.test_image = {cfg.inference.data.test_image if hasattr(cfg, 'inference') and hasattr(cfg.inference, 'data') else 'N/A'}"
            )
        print(f"  🧪 Creating test dataset from: {cfg.inference.data.test_image}")

        # Expand glob patterns for test data (same as train data)
        test_image_paths = expand_file_paths(cfg.inference.data.test_image)
        test_label_paths = (
            expand_file_paths(cfg.inference.data.test_label)
            if cfg.inference.data.test_label
            else None
        )
        test_mask_paths = (
            expand_file_paths(cfg.inference.data.test_mask)
            if hasattr(cfg.inference.data, "test_mask") and cfg.inference.data.test_mask
            else None
        )

        print(f"  Test volumes: {len(test_image_paths)} files")
        if len(test_image_paths) <= 5:
            for path in test_image_paths:
                print(f"    - {path}")
        else:
            print(f"    - {test_image_paths[0]}")
            print(f"    - ... ({len(test_image_paths) - 2} more files)")
            print(f"    - {test_image_paths[-1]}")

        if test_mask_paths:
            print(f"  Test masks: {len(test_mask_paths)} files")

        test_data_dicts = create_data_dicts_from_paths(
            image_paths=test_image_paths,
            label_paths=test_label_paths,
            mask_paths=test_mask_paths,
        )
        print(f"  Test dataset size: {len(test_data_dicts)}")

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
            from connectomics.data.utils import compute_total_samples
            import h5py
            import tifffile

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

    # For test/predict mode, disable iter_num (process full volumes once)
    if mode in ["test", "predict"]:
        iter_num_for_dataset = -1  # Process full volumes without random sampling
    else:
        iter_num_for_dataset = iter_num

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
        from connectomics.data.dataset.dataset_volume_cached import CachedVolumeDataset

        # Build transforms without loading/cropping (handled by dataset)
        augment_only_transforms = build_train_transforms(cfg, skip_loading=True)

        # Get padding parameters from config
        pad_size = getattr(cfg.data.image_transform, 'pad_size', None)
        pad_mode = getattr(cfg.data.image_transform, 'pad_mode', 'reflect')

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
        num_workers = min(cfg.system.training.num_workers, 2)
        print(f"  Using {num_workers} workers (in-memory operations are fast)")

        # Create simple dataloader
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.system.training.batch_size,
            shuffle=False,  # Already random
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=num_workers > 0,
        )

        # Create data module wrapper that inherits from LightningDataModule
        import pytorch_lightning as pl

        class SimpleDataModule(pl.LightningDataModule):
            def __init__(self, train_loader):
                super().__init__()
                self.train_loader = train_loader

            def train_dataloader(self):
                return self.train_loader

            def val_dataloader(self):
                return []

            def test_dataloader(self):
                # For test mode, return empty list (user should use standard datamodule)
                return []

            def setup(self, stage=None):
                pass

        datamodule = SimpleDataModule(train_loader)
    elif dataset_type == "filename":
        # Filename-based dataset using JSON file lists
        print("  Creating filename-based datamodule...")
        from connectomics.data.dataset.dataset_filename import create_filename_datasets
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

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
            batch_size=cfg.system.training.batch_size,
            num_workers=cfg.system.training.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
        )
    else:
        # Standard data module
        use_cache = cfg.data.use_cache

        # Note: transpose_axes is now handled in the transform builders (build_train/val/test_transforms)
        # which embed the transpose in LoadVolumed, so no need to pass it here

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
            batch_size=cfg.system.training.batch_size,
            num_workers=cfg.system.training.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers,
            cache_rate=cfg.data.cache_rate if use_cache else 0.0,
            iter_num=iter_num_for_dataset,
            sample_size=tuple(cfg.data.patch_size),
            do_2d=cfg.data.do_2d,
        )
        # Setup datasets based on mode
        if mode == "train":
            datamodule.setup(stage="fit")
        elif mode in ["test", "predict"]:
            datamodule.setup(stage="test")

    # Print dataset info based on mode
    if mode == "train":
        print(f"  Train batches: {len(datamodule.train_dataloader())}")
        if val_data_dicts:
            print(f"  Val batches: {len(datamodule.val_dataloader())}")
    elif mode in ["test", "predict"]:
        print(f"  Test batches: {len(datamodule.test_dataloader())}")

    return datamodule


def extract_best_score_from_checkpoint(ckpt_path: str, monitor_metric: str) -> Optional[float]:
    """
    Extract best score from checkpoint filename.

    Args:
        ckpt_path: Path to checkpoint file
        monitor_metric: Metric name to extract (e.g., 'train_loss_total_epoch', 'val/loss')

    Returns:
        Extracted score or None if not found
    """
    import re
    from pathlib import Path

    if not ckpt_path:
        return None

    filename = Path(ckpt_path).stem  # Get filename without extension

    # Replace '/' with underscore for metric name (e.g., 'val/loss' -> 'val_loss')
    metric_pattern = monitor_metric.replace("/", "_")

    # Try multiple patterns to extract the metric value:
    # 1. Full metric name: "train_loss_total_epoch=0.1234"
    # 2. Abbreviated in filename: "loss=0.1234" (when metric is "train_loss_total_epoch")
    # 3. Other common abbreviations

    patterns = [
        rf"{metric_pattern}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",  # Full name
    ]

    # Add abbreviated patterns by extracting the last part after '_' or '/'
    if "_" in monitor_metric or "/" in monitor_metric:
        short_name = monitor_metric.split("_")[-1].split("/")[-1]
        patterns.append(rf"{short_name}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def create_trainer(
    cfg: Config,
    run_dir: Path,
    fast_dev_run: bool = False,
    ckpt_path: Optional[str] = None,
    mode: str = "train",
) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer.

    Args:
        cfg: Hydra Config object
        run_dir: Directory for this training run
        fast_dev_run: Whether to run quick debug mode
        ckpt_path: Path to checkpoint for resuming (used to extract best_score)
        mode: 'train' or 'test' - determines which system config to use

    Returns:
        Configured Trainer instance
    """
    print(f"Creating Lightning trainer (mode={mode})...")

    # Setup callbacks (only for training mode)
    callbacks = []

    if mode == "train":
        # Setup checkpoint directory
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model checkpoint (in run_dir/checkpoints/)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=cfg.monitor.checkpoint.checkpoint_filename,
            monitor=cfg.monitor.checkpoint.monitor,
            mode=cfg.monitor.checkpoint.mode,
            save_top_k=cfg.monitor.checkpoint.save_top_k,
            save_last=cfg.monitor.checkpoint.save_last,
            every_n_epochs=cfg.monitor.checkpoint.save_every_n_epochs,
            verbose=True,
            save_on_train_epoch_end=True,  # Save based on training metrics
        )
        callbacks.append(checkpoint_callback)

        # Early stopping (training only)
        if cfg.monitor.early_stopping.enabled:
            # Extract best_score from checkpoint filename if resuming
            best_score = None
            if ckpt_path:
                best_score = extract_best_score_from_checkpoint(
                    ckpt_path, cfg.monitor.early_stopping.monitor
                )
                if best_score is not None:
                    print(
                        f"  Early stopping: Extracted best_score={best_score:.6f} from checkpoint"
                    )

            early_stop_callback = EarlyStopping(
                monitor=cfg.monitor.early_stopping.monitor,
                patience=cfg.monitor.early_stopping.patience,
                mode=cfg.monitor.early_stopping.mode,
                min_delta=cfg.monitor.early_stopping.min_delta,
                verbose=True,
                check_on_train_epoch_end=True,  # Check at end of train epoch (not validation)
                check_finite=cfg.monitor.early_stopping.check_finite,  # Stop on NaN/inf
                stopping_threshold=cfg.monitor.early_stopping.threshold,
                divergence_threshold=cfg.monitor.early_stopping.divergence_threshold,
                strict=False,  # Don't crash if metric not available (wait for it)
            )

            # Manually set best_score if extracted from checkpoint
            if best_score is not None:
                early_stop_callback.best_score = torch.tensor(best_score)

            callbacks.append(early_stop_callback)

        # Learning rate monitor (training only)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Visualization callback (training only, end-of-epoch only)
        if cfg.monitor.logging.images.enabled:
            vis_callback = VisualizationCallback(
                cfg=cfg,
                max_images=cfg.monitor.logging.images.max_images,
                num_slices=cfg.monitor.logging.images.num_slices,
                log_every_n_epochs=cfg.monitor.logging.images.log_every_n_epochs,
            )
            callbacks.append(vis_callback)
            print(
                f"  Visualization: Enabled (every {cfg.monitor.logging.images.log_every_n_epochs} epoch(s))"
            )
        else:
            print(f"  Visualization: Disabled")

    # Progress bar (optional - requires rich package)
    try:
        callbacks.append(RichProgressBar())
    except (ImportError, ModuleNotFoundError):
        pass  # Use default progress bar

    # Setup logger (training only - in run_dir/logs/)
    logger = None
    if mode == "train":
        logger = TensorBoardLogger(
            save_dir=str(run_dir),
            name="",  # No name subdirectory
            version="logs",  # Logs go directly to run_dir/logs/
        )

    # Create trainer
    # Select system config based on mode
    system_cfg = cfg.system.training if mode == "train" else cfg.system.inference

    # Check if GPU is actually available
    use_gpu = system_cfg.num_gpus > 0 and torch.cuda.is_available()

    # Check if anomaly detection is enabled (useful for debugging NaN)
    detect_anomaly = getattr(cfg.monitor, "detect_anomaly", False)
    if detect_anomaly:
        print("  ⚠️  PyTorch anomaly detection ENABLED (training will be slower)")
        print("      This helps pinpoint the exact operation causing NaN in backward pass")

    # Configure DDP strategy for multi-GPU training with deep supervision
    strategy = "auto"  # Default strategy
    if system_cfg.num_gpus > 1:
        # Multi-GPU training: configure DDP
        deep_supervision_enabled = getattr(cfg.model, "deep_supervision", False)
        ddp_find_unused_params = getattr(cfg.model, "ddp_find_unused_parameters", False)
        architecture = getattr(cfg.model, "architecture", "")
        is_mednext = architecture.startswith("mednext")

        # MedNeXt always creates deep supervision layers internally (even when disabled)
        # so it always needs find_unused_parameters=True
        if is_mednext or deep_supervision_enabled or ddp_find_unused_params:
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=True)

            # Determine reason for using find_unused_parameters
            if is_mednext and not deep_supervision_enabled:
                reason = "MedNeXt (has unused DS layers)"
            elif deep_supervision_enabled:
                reason = "deep supervision enabled"
            else:
                reason = "explicit config"
            print(f"  Strategy: DDP with find_unused_parameters=True ({reason})")
        else:
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)
            print("  Strategy: DDP (standard)")

    trainer = pl.Trainer(
        max_epochs=cfg.optimization.max_epochs,
        max_steps=getattr(cfg.optimization, "max_steps", None) or -1,
        accelerator="gpu" if use_gpu else "cpu",
        devices=system_cfg.num_gpus if use_gpu else 1,
        strategy=strategy,
        precision=cfg.optimization.precision,
        gradient_clip_val=cfg.optimization.gradient_clip_val,
        accumulate_grad_batches=cfg.optimization.accumulate_grad_batches,
        val_check_interval=cfg.optimization.val_check_interval,
        log_every_n_steps=cfg.optimization.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.optimization.deterministic,
        benchmark=cfg.optimization.benchmark,
        fast_dev_run=bool(fast_dev_run),
        detect_anomaly=detect_anomaly,
    )

    print(f"  Max epochs: {cfg.optimization.max_epochs}")
    print(f"  Devices: {system_cfg.num_gpus if system_cfg.num_gpus > 0 else 1} ({mode} mode)")
    print(f"  Precision: {cfg.optimization.precision}")

    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Handle demo mode
    if args.demo:
        from connectomics.utils.demo import run_demo

        run_demo()
        return

    # Validate that config is provided for non-demo modes
    if not args.config:
        print("❌ Error: --config is required (or use --demo for a quick test)")
        print("\nUsage:")
        print("  python scripts/main.py --config tutorials/lucchi.yaml")
        print("  python scripts/main.py --demo")
        sys.exit(1)

    # Setup config
    print("\n" + "=" * 60)
    print("🚀 PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)

    # Run preflight checks for training mode
    if args.mode == "train":
        from connectomics.utils.errors import preflight_check, print_preflight_issues

        issues = preflight_check(cfg)
        if issues:
            print_preflight_issues(issues)

    # Create run directory only for training mode
    # Structure: outputs/experiment_name/YYYYMMDD_HHMMSS/{checkpoints,logs,config.yaml}
    # IMPORTANT: When using DDP (multi-GPU), PyTorch Lightning re-launches this script
    # multiple times. The first invocation (no LOCAL_RANK env var) creates the timestamp.
    # Subsequent invocations (with LOCAL_RANK set) reuse the existing timestamp.
    if args.mode == "train":
        # Extract output folder from checkpoint dirpath (remove /checkpoints suffix)
        checkpoint_dir = Path(cfg.monitor.checkpoint.dirpath)
        checkpoint_subdir = checkpoint_dir.name or "checkpoints"
        output_base = checkpoint_dir.parent  # Base directory containing timestamped runs

        # Check if this is a DDP re-launch (LOCAL_RANK is set by PyTorch Lightning)
        import os

        is_ddp_subprocess = "LOCAL_RANK" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        timestamp_file = output_base / ".latest_timestamp"

        if not is_ddp_subprocess:
            # First invocation (main process) - create new timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_base / timestamp

            # Update checkpoint dirpath to use the timestamped directory (preserve leaf name)
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
    else:
        # For test/predict mode, use a dummy run_dir (won't be created)
        output_base = Path(cfg.monitor.checkpoint.dirpath).parent
        run_dir = output_base / "test_run"
        print(f"📝 Running in {args.mode} mode (no output directory created)")

    # Set random seed
    if cfg.system.seed is not None:
        print(f"🎲 Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    # Create datamodule
    datamodule = create_datamodule(cfg, mode=args.mode, fast_dev_run=bool(args.fast_dev_run))

    # Create model
    print(f"Creating model: {cfg.model.architecture}")
    model = ConnectomicsModule(cfg)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Create trainer (pass run_dir for checkpoints and logs, and checkpoint path for resume)
    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=args.checkpoint,
        mode=args.mode,
    )

    print("\n" + "=" * 60)
    print("🏃 STARTING TRAINING")
    print("=" * 60)

    # Handle checkpoint state resets if requested
    ckpt_path = args.checkpoint
    if (
        args.checkpoint
        and args.mode == "train"
        and (
            args.reset_optimizer
            or args.reset_scheduler
            or args.reset_epoch
            or args.reset_early_stopping
        )
    ):
        print(f"\n🔄 Modifying checkpoint state:")
        if args.reset_optimizer:
            print("   - Resetting optimizer state")
        if args.reset_scheduler:
            print("   - Resetting scheduler state")
        if args.reset_epoch:
            print("   - Resetting epoch counter")
        if args.reset_early_stopping:
            print("   - Resetting early stopping patience counter")
        if args.reset_max_epochs is not None:
            print(f"   - Overriding max_epochs to: {args.reset_max_epochs}")

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        # Reset optimizer state
        if args.reset_optimizer and "optimizer_states" in checkpoint:
            del checkpoint["optimizer_states"]

        # Reset scheduler state
        if args.reset_scheduler and "lr_schedulers" in checkpoint:
            del checkpoint["lr_schedulers"]

        # Reset epoch counter
        if args.reset_epoch:
            if "epoch" in checkpoint:
                checkpoint["epoch"] = 0
            if "global_step" in checkpoint:
                checkpoint["global_step"] = 0

        # Reset early stopping state
        if args.reset_early_stopping and "callbacks" in checkpoint:
            # EarlyStopping callback state is stored in callbacks dict
            for callback_state in checkpoint["callbacks"].values():
                if "wait_count" in callback_state:
                    callback_state["wait_count"] = 0
                if "best_score" in callback_state:
                    # Reset to None or worst possible value
                    callback_state["best_score"] = None

        # Save modified checkpoint to temporary file
        temp_ckpt_path = run_dir / "temp_modified_checkpoint.ckpt"
        torch.save(checkpoint, temp_ckpt_path)
        ckpt_path = str(temp_ckpt_path)
        print(f"   ✅ Modified checkpoint saved to: {temp_ckpt_path}")

    # Train
    try:
        if args.mode == "train":
            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
            print("\n✅ Training completed successfully!")

        elif args.mode == "test":
            print("\n" + "=" * 60)
            print("🧪 RUNNING TEST")
            print("=" * 60)

            # Check if test dataset exists
            test_loader = datamodule.test_dataloader()

            if test_loader is None or len(test_loader) == 0:
                print("⚠️  No test dataset found!")
                print("   Make sure inference.test_image is set in config")
                return

            print(f"Test batches: {len(test_loader)}")

            # Don't pass ckpt_path if using external weights (already loaded into model)
            test_ckpt_path = None if args.external_prefix else args.checkpoint
            trainer.test(
                model,
                datamodule=datamodule,
                ckpt_path=test_ckpt_path,
            )

        elif args.mode == "predict":
            print("Running prediction...")
            # Don't pass ckpt_path if using external weights (already loaded into model)
            predict_ckpt_path = None if args.external_prefix else args.checkpoint
            predictions = trainer.predict(
                model,
                datamodule=datamodule,
                ckpt_path=predict_ckpt_path,
            )

            # Save predictions
            output_dir = Path(cfg.inference.data.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract checkpoint name for output filename
            ckpt_name = "default"
            if args.checkpoint:
                ckpt_name = Path(
                    args.checkpoint
                ).stem  # e.g., "epoch=099-step=0012345" from "epoch=099-step=0012345.ckpt"

            for i, pred in enumerate(predictions):
                output_file = output_dir / f"prediction_{ckpt_name}_{i:04d}.h5"

                # Convert to numpy if needed
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()

                # Save using connectomics IO
                from connectomics.data.io import write_h5

                write_h5(str(output_file), pred)

            print(f"Predictions saved to: {output_dir}")

    except Exception as e:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n❌ {mode_name} failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: Remove timestamp file (only in main process, not DDP subprocesses)
        if args.mode == "train":
            import os

            is_ddp_subprocess = "LOCAL_RANK" in os.environ
            if not is_ddp_subprocess and "output_base" in locals():
                timestamp_file = output_base / ".latest_timestamp"
                if timestamp_file.exists():
                    try:
                        timestamp_file.unlink()
                    except Exception:
                        pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
