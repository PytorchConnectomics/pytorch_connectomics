"""
Demo utilities for PyTorch Connectomics.

Provides synthetic data generation and quick demo runs for testing installation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple


def create_synthetic_volume(
    shape: Tuple[int, int, int] = (64, 128, 128),
    num_objects: int = 20,
    object_size_range: Tuple[int, int] = (3, 8),
    noise_level: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic 3D volume with object-like structures (mitochondria-like).

    Args:
        shape: Volume shape (Z, Y, X)
        num_objects: Number of objects to generate
        object_size_range: (min, max) size of objects
        noise_level: Gaussian noise level (0-1)

    Returns:
        Tuple of (image, label) as numpy arrays
    """
    # Create empty volumes
    image = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    # Generate random objects (ellipsoids)
    for obj_id in range(1, num_objects + 1):
        # Random center position
        center_z = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
        center_y = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)
        center_x = np.random.randint(shape[2] // 4, 3 * shape[2] // 4)

        # Random ellipsoid radii
        radius_z = np.random.randint(*object_size_range)
        radius_y = np.random.randint(*object_size_range)
        radius_x = np.random.randint(*object_size_range)

        # Create ellipsoid mask
        z, y, x = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        mask = (
            ((z - center_z) / radius_z) ** 2
            + ((y - center_y) / radius_y) ** 2
            + ((x - center_x) / radius_x) ** 2
        ) <= 1

        # Add to label (binary segmentation)
        label[mask] = 1

        # Add to image with intensity variation
        intensity = np.random.uniform(0.6, 1.0)
        image[mask] = intensity

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, shape).astype(np.float32)
    image = np.clip(image + noise, 0, 1)

    # Add background intensity
    background = np.random.uniform(0.1, 0.3, shape).astype(np.float32)
    image = np.where(label > 0, image, background)

    return image, label


def create_demo_config():
    """
    Create a minimal demo configuration.

    Returns:
        Config object for demo training
    """
    from connectomics.config import Config
    from connectomics.config.hydra_config import (
        SystemConfig,
        SystemTrainingConfig,
        SystemInferenceConfig,
        ModelConfig,
        DataConfig,
        OptimizationConfig,
        MonitorConfig,
        CheckpointConfig,
        EarlyStoppingConfig,
        LoggingConfig,
        ImageLoggingConfig,
        InferenceConfig,
        InferenceDataConfig,
    )

    cfg = Config(
        system=SystemConfig(
            seed=42,
            training=SystemTrainingConfig(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus=2,
                batch_size=2,
                num_workers=0,  # 0 for demo to avoid multiprocessing issues
            ),
            inference=SystemInferenceConfig(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus=2,
                batch_size=2,
                num_workers=0,
            ),
        ),
        model=ModelConfig(
            architecture="monai_basic_unet3d",
            in_channels=1,
            out_channels=2,
            spatial_dims=3,
            filters=[16, 32, 64, 128],  # Smaller for demo
            dropout=0.1,
            loss_functions=["DiceLoss"],
            loss_weights=[1.0],
        ),
        data=DataConfig(
            train_image=None,  # Will be generated
            train_label=None,
            val_image=None,
            val_label=None,
            patch_size=[32, 64, 64],  # Small for demo
            stride=[16, 32, 32],
            iter_num_per_epoch=10,  # Just 10 iterations per epoch
            use_cache=False,
            use_preloaded_cache=False,
            pin_memory=False,
            persistent_workers=False,
        ),
        optimization=OptimizationConfig(
            max_epochs=5,  # Quick demo
            precision="32",  # Use FP32 for stability
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            log_every_n_steps=1,
            deterministic=False,
            benchmark=True,
        ),
        optimizer={"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
        scheduler={"name": "ConstantLR", "warmup_epochs": 0},
        monitor=MonitorConfig(
            checkpoint=CheckpointConfig(
                dirpath="outputs/demo/checkpoints",
                checkpoint_filename="demo-{epoch:02d}-{step:06d}",
                monitor="train_loss_total_epoch",
                mode="min",
                save_top_k=1,
                save_last=False,
                save_every_n_epochs=1,
                use_timestamp=False,
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=False,  # Disable for demo
                monitor="train_loss_total_epoch",
                patience=10,
                mode="min",
                min_delta=0.0001,
            ),
            logging=LoggingConfig(
                images=ImageLoggingConfig(
                    enabled=False,  # Disable image logging for demo
                    max_images=1,
                    num_slices=3,
                    log_every_n_epochs=1,
                ),
            ),
        ),
        inference=InferenceConfig(
            num_gpus=-1,
            num_cpus=-1,
            batch_size=-1,
            num_workers=-1,
            data=InferenceDataConfig(
                test_image=None,
                test_label=None,
                output_path="outputs/demo/predictions",
            ),
        ),
    )

    return cfg


def run_demo():
    """
    Run a quick demo training with synthetic data.

    This creates synthetic 3D volumes, trains a small model for 5 epochs,
    and validates the installation.
    """
    print("\n" + "=" * 60)
    print("🎯 PyTorch Connectomics Demo Mode")
    print("=" * 60)
    print("\nThis demo will:")
    print("  1. Generate synthetic 3D volumes (mitochondria-like structures)")
    print("  2. Train a small 3D U-Net for 5 epochs")
    print("  3. Validate your installation")
    print("\n" + "-" * 60 + "\n")

    # Create demo config
    print("📝 Creating demo configuration...")
    cfg = create_demo_config()

    # Set seed
    from pytorch_lightning import seed_everything
    seed_everything(cfg.system.seed, workers=True)
    print(f"🎲 Random seed: {cfg.system.seed}")

    # Generate synthetic data
    print("\n🔧 Generating synthetic training data...")
    train_image, train_label = create_synthetic_volume(
        shape=(64, 128, 128), num_objects=20
    )
    print(f"   Image shape: {train_image.shape}")
    print(f"   Label shape: {train_label.shape}")
    print(f"   Num objects: {train_label.sum() / train_label.size * 100:.1f}% foreground")

    print("\n🔧 Generating synthetic validation data...")
    val_image, val_label = create_synthetic_volume(shape=(64, 128, 128), num_objects=15)

    # Save to temporary files
    import tempfile
    import h5py

    temp_dir = Path(tempfile.mkdtemp(prefix="pytc_demo_"))
    print(f"\n💾 Saving to temporary directory: {temp_dir}")

    train_image_path = temp_dir / "train_image.h5"
    train_label_path = temp_dir / "train_label.h5"
    val_image_path = temp_dir / "val_image.h5"
    val_label_path = temp_dir / "val_label.h5"

    with h5py.File(train_image_path, "w") as f:
        f.create_dataset("main", data=train_image, compression="gzip")
    with h5py.File(train_label_path, "w") as f:
        f.create_dataset("main", data=train_label, compression="gzip")
    with h5py.File(val_image_path, "w") as f:
        f.create_dataset("main", data=val_image, compression="gzip")
    with h5py.File(val_label_path, "w") as f:
        f.create_dataset("main", data=val_label, compression="gzip")

    print("   ✓ train_image.h5")
    print("   ✓ train_label.h5")
    print("   ✓ val_image.h5")
    print("   ✓ val_label.h5")

    # Update config with file paths
    cfg.data.train_image = str(train_image_path)
    cfg.data.train_label = str(train_label_path)
    cfg.data.val_image = str(val_image_path)
    cfg.data.val_label = str(val_label_path)

    # Create datamodule
    print("\n📊 Creating data loaders...")
    from connectomics.lightning import ConnectomicsDataModule
    from connectomics.data.augment.build import (
        build_train_transforms,
        build_val_transforms,
    )
    from connectomics.data.dataset import create_data_dicts_from_paths

    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)

    train_data_dicts = create_data_dicts_from_paths(
        image_paths=[str(train_image_path)], label_paths=[str(train_label_path)]
    )
    val_data_dicts = create_data_dicts_from_paths(
        image_paths=[str(val_image_path)], label_paths=[str(val_label_path)]
    )

    datamodule = ConnectomicsDataModule(
        train_data_dicts=train_data_dicts,
        val_data_dicts=val_data_dicts,
        transforms={"train": train_transforms, "val": val_transforms},
        dataset_type="standard",
        batch_size=cfg.system.training.batch_size,
        num_workers=cfg.system.training.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=False,
        iter_num=cfg.data.iter_num_per_epoch,
        sample_size=tuple(cfg.data.patch_size),
    )
    datamodule.setup(stage="fit")

    # Create model
    print(f"\n🏗️  Building model: {cfg.model.architecture}")
    from connectomics.lightning import ConnectomicsModule

    model = ConnectomicsModule(cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")

    # Create trainer
    print("\n⚡ Creating Lightning trainer...")
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

    # Create output directory
    output_dir = Path(cfg.monitor.checkpoint.dirpath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename=cfg.monitor.checkpoint.checkpoint_filename,
            monitor=cfg.monitor.checkpoint.monitor,
            mode=cfg.monitor.checkpoint.mode,
            save_top_k=1,
            save_last=False,
        ),
    ]

    try:
        callbacks.append(RichProgressBar())
    except (ImportError, ModuleNotFoundError):
        pass

    trainer = pl.Trainer(
        max_epochs=cfg.optimization.max_epochs,
        accelerator="gpu" if cfg.system.training.num_gpus > 0 else "cpu",
        devices=cfg.system.training.num_gpus if cfg.system.training.num_gpus > 0 else 1,
        precision=cfg.optimization.precision,
        callbacks=callbacks,
        logger=False,  # Disable logging for demo
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"   Max epochs: {cfg.optimization.max_epochs}")
    print(f"   Device: {'GPU' if cfg.system.training.num_gpus > 0 else 'CPU'}")
    print(f"   Precision: {cfg.optimization.precision}")

    # Train
    print("\n" + "=" * 60)
    print("🏃 STARTING DEMO TRAINING")
    print("=" * 60 + "\n")

    try:
        trainer.fit(model, datamodule=datamodule)

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYour installation is working correctly! 🎉")
        print("\n📚 Next steps:")
        print("  1. Try a tutorial:")
        print("     python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run")
        print("\n  2. Download tutorial data:")
        print("     just download-data lucchi  # Or see README for manual download")
        print("\n  3. Train on real data:")
        print("     python scripts/main.py --config tutorials/lucchi.yaml")
        print("\n  4. Read the documentation:")
        print("     https://connectomics.readthedocs.io")
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        print("\n💡 If you see errors, please:")
        print("  1. Check your installation: python -c 'import connectomics'")
        print("  2. Report issues: https://github.com/zudi-lin/pytorch_connectomics/issues")
        raise

    finally:
        # Cleanup temporary files
        import shutil

        print(f"\n🧹 Cleaning up temporary files: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"   Warning: Could not remove temp dir: {e}")
