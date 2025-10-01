#!/usr/bin/env python3
"""
PyTorch Connectomics training script with Hydra configuration and Lightning framework.

This script provides modern deep learning training with:
- Hydra-based configuration management
- Automatic distributed training and mixed precision
- MONAI-based data augmentation
- PyTorch Lightning callbacks and logging

Usage:
    python scripts/main.py --config tutorials/lucchi.yaml
    python scripts/main.py --config tutorials/lucchi.yaml --mode test --checkpoint path/to/checkpoint.ckpt
    python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run
    python scripts/main.py --config tutorials/lucchi.yaml data.batch_size=8 training.max_epochs=200
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
    print_config,
    validate_config,
    Config,
)

# Import data and model utilities
from connectomics.data.dataset import create_data_dicts_from_paths
from connectomics.lightning import (
    ConnectomicsDataModule,
    ConnectomicsModule,
    create_trainer as create_lightning_trainer,
)
from connectomics.data.augment.build import (
    build_train_transforms,
    build_val_transforms,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Connectomics Training with Hydra Config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to Hydra YAML config file",
    )
    parser.add_argument(
        "--mode",
        choices=['train', 'test', 'predict'],
        default='train',
        help="Mode: train, test, or predict (default: train)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to checkpoint for resuming/testing/prediction",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 batch for quick debugging",
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

    # Apply CLI overrides
    if args.overrides:
        print(f"⚙️  Applying {len(args.overrides)} CLI overrides")
        cfg = update_from_cli(cfg, args.overrides)

    # Auto-planning (if enabled)
    if hasattr(cfg.system, 'auto_plan') and cfg.system.auto_plan:
        print("🤖 Running automatic configuration planning...")
        from connectomics.config import auto_plan_config
        print_results = cfg.system.print_auto_plan if hasattr(cfg.system, 'print_auto_plan') else True
        cfg = auto_plan_config(cfg, print_results=print_results)

    # Validate configuration
    print("✅ Validating configuration...")
    validate_config(cfg)

    # Ensure output directory exists
    output_dir = Path(cfg.checkpoint.dirpath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_save_path = output_dir / "checkpoints" / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, config_save_path)
    print(f"💾 Config saved to: {config_save_path}")

    return cfg


def create_datamodule(cfg: Config) -> ConnectomicsDataModule:
    """
    Create Lightning DataModule from config.

    Args:
        cfg: Hydra Config object

    Returns:
        VolumeDataModule instance
    """
    print("Creating datasets...")

    # Build transforms
    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)

    print(f"  Train transforms: {len(train_transforms.transforms)} steps")
    print(f"  Val transforms: {len(val_transforms.transforms)} steps")

    # Check if automatic train/val split is enabled
    if cfg.data.split_enabled and not cfg.data.val_image:
        print("🔀 Using automatic train/val split (DeepEM-style)")
        from connectomics.data.utils.split import (
            split_volume_train_val,
            apply_volumetric_split,
        )

        # Load full volume
        import h5py
        import tifffile
        from pathlib import Path

        train_path = Path(cfg.data.train_image)
        if train_path.suffix in ['.h5', '.hdf5']:
            with h5py.File(train_path, 'r') as f:
                volume_shape = f[list(f.keys())[0]].shape
        elif train_path.suffix in ['.tif', '.tiff']:
            volume = tifffile.imread(train_path)
            volume_shape = volume.shape
        else:
            raise ValueError(f"Unsupported file format: {train_path.suffix}")

        print(f"  Volume shape: {volume_shape}")

        # Calculate split ranges
        train_ratio = cfg.data.split_train_range[1] - cfg.data.split_train_range[0]
        split_point = cfg.data.split_train_range[1]

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
        train_data_dicts[0]['split_slices'] = train_slices
        train_data_dicts[0]['split_mode'] = 'train'

        # Create validation data dicts using same volume
        val_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train_image],
            label_paths=[cfg.data.train_label] if cfg.data.train_label else None,
        )

        # Add split metadata to val dict
        val_data_dicts[0]['split_slices'] = val_slices
        val_data_dicts[0]['split_mode'] = 'val'
        val_data_dicts[0]['split_pad'] = cfg.data.split_pad_val
        val_data_dicts[0]['split_pad_mode'] = cfg.data.split_pad_mode
        if cfg.data.split_pad_val:
            val_data_dicts[0]['split_pad_size'] = tuple(cfg.data.patch_size)

    else:
        # Standard mode: separate train and val files
        train_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train_image],
            label_paths=[cfg.data.train_label] if cfg.data.train_label else None,
        )

        val_data_dicts = None
        if cfg.data.val_image:
            val_data_dicts = create_data_dicts_from_paths(
                image_paths=[cfg.data.val_image],
                label_paths=[cfg.data.val_label] if cfg.data.val_label else None,
            )

    print(f"  Train dataset size: {len(train_data_dicts)}")
    if val_data_dicts:
        print(f"  Val dataset size: {len(val_data_dicts)}")

    # Create DataModule
    print("Creating data loaders...")
    datamodule = ConnectomicsDataModule(
        train_data_dicts=train_data_dicts,
        val_data_dicts=val_data_dicts,
        transforms={
            'train': train_transforms,
            'val': val_transforms,
            'test': val_transforms,
        },
        dataset_type='cached' if cfg.data.use_cache else 'standard',
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        cache_rate=cfg.data.cache_rate if cfg.data.use_cache else 0.0,
    )

    # Setup datasets
    datamodule.setup(stage='fit')

    print(f"  Train batches: {len(datamodule.train_dataloader())}")
    if val_data_dicts:
        print(f"  Val batches: {len(datamodule.val_dataloader())}")

    return datamodule


def create_trainer(cfg: Config, fast_dev_run: bool = False) -> pl.Trainer:
    """
    Create PyTorch Lightning Trainer.

    Args:
        cfg: Hydra Config object
        fast_dev_run: Whether to run quick debug mode

    Returns:
        Configured Trainer instance
    """
    print("Creating Lightning trainer...")

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            min_delta=cfg.early_stopping.min_delta,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # Progress bar (optional - requires rich package)
    try:
        callbacks.append(RichProgressBar())
    except (ImportError, ModuleNotFoundError):
        pass  # Use default progress bar

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=Path(cfg.checkpoint.dirpath).parent,
        name='logs',
        version=cfg.experiment_name,
    )

    # Create trainer
    # Check if GPU is actually available
    use_gpu = cfg.system.num_gpus > 0 and torch.cuda.is_available()
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps if cfg.training.max_steps else -1,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.system.num_gpus if use_gpu else 1,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.benchmark,
        fast_dev_run=fast_dev_run,
    )

    print(f"  Max epochs: {cfg.training.max_epochs}")
    print(f"  Devices: {cfg.system.num_gpus if cfg.system.num_gpus > 0 else 1}")
    print(f"  Precision: {cfg.training.precision}")

    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup config
    print("\n" + "=" * 60)
    print("🚀 PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)

    # Set random seed
    if cfg.system.seed is not None:
        print(f"🎲 Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    # Create datamodule
    datamodule = create_datamodule(cfg)

    # Create model
    print(f"Creating model: {cfg.model.architecture}")
    model = ConnectomicsModule(cfg)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Create trainer
    trainer = create_trainer(cfg, fast_dev_run=args.fast_dev_run)

    print("\n" + "=" * 60)
    print("🏃 STARTING TRAINING")
    print("=" * 60)

    # Train
    try:
        if args.mode == 'train':
            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=args.checkpoint,
            )
            print("\n✅ Training completed successfully!")

        elif args.mode == 'test':
            print("Running test...")
            results = trainer.test(
                model,
                datamodule=datamodule,
                ckpt_path=args.checkpoint,
            )
            print("Test results:", results)

        elif args.mode == 'predict':
            print("Running prediction...")
            predictions = trainer.predict(
                model,
                datamodule=datamodule,
                ckpt_path=args.checkpoint,
            )

            # Save predictions
            output_dir = Path(cfg.inference.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, pred in enumerate(predictions):
                output_file = output_dir / f'prediction_{i:04d}.h5'

                # Convert to numpy if needed
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()

                # Save using connectomics IO
                from connectomics.data.io import write_h5
                write_h5(str(output_file), pred)

            print(f"Predictions saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()