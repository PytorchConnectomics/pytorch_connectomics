#!/usr/bin/env python3
"""
Modern PyTorch Connectomics training script with Lightning framework.

This script provides modern deep learning training with:
- Automatic distributed training and mixed precision
- nnUNet model architectures (MedNeXt, UNETR, Swin UNETR)
- MONAI-based data augmentation
- Lightning callbacks and logging

Usage:
    python scripts/main.py --config-file configs/MedNeXt-Mitochondria.yaml
    python scripts/main.py --config-file configs/UNETR-Neuron.yaml --mode train
    python scripts/main.py --config-file configs/MedNeXt-Mitochondria.yaml --mode test --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl

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

from connectomics.config import load_cfg
from connectomics.engine.lightning_trainer import create_trainer
from connectomics.engine.lightning_module import create_lightning_module
from connectomics.engine.lightning_datamodule import ConnectomicsDataModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch Connectomics Lightning Training")

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--mode",
        choices=['train', 'test', 'predict'],
        default='train',
        help="Mode: train, test, or predict"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint for testing/prediction"
    )

    return parser.parse_args()


def setup_config(args):
    """Setup configuration from file and command line arguments."""
    # Convert our args to match the expected format for load_cfg
    class MockArgs:
        def __init__(self, config_file, opts, inference=False, distributed=False):
            self.config_file = config_file
            self.config_base = None  # Not used in our case
            self.opts = opts or []
            self.inference = inference
            self.distributed = distributed

    mock_args = MockArgs(args.config_file, args.opts,
                        inference=(args.mode != 'train'),
                        distributed=False)

    cfg = load_cfg(mock_args)

    # Ensure output directory exists
    os.makedirs(cfg.DATASET.OUTPUT_PATH, exist_ok=True)

    return cfg


def main():
    """Main training function."""
    args = parse_args()
    cfg = setup_config(args)

    # Set seed for reproducibility
    if hasattr(cfg.SYSTEM, 'RANDOM_SEED'):
        seed_everything(cfg.SYSTEM.RANDOM_SEED, workers=True)

    print("=" * 50)
    print("PyTorch Connectomics Lightning Training")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config_file}")
    print(f"Output: {cfg.DATASET.OUTPUT_PATH}")
    print(f"GPUs: {cfg.SYSTEM.NUM_GPUS}")
    print("=" * 50)

    # Create Lightning components
    lightning_module = create_lightning_module(cfg)
    datamodule = ConnectomicsDataModule(cfg)

    # Setup trainer
    trainer = create_trainer(cfg)
    trainer.setup()

    if args.mode == 'train':
        print("Starting training...")
        trainer.train()

        # Run final test if validation data available
        if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset is not None:
            print("Running final test...")
            trainer.test()

    elif args.mode == 'test':
        print("Running test...")
        if args.checkpoint:
            results = trainer.test(ckpt_path=args.checkpoint)
        else:
            results = trainer.test()
        print("Test results:", results)

    elif args.mode == 'predict':
        print("Running prediction...")
        if args.checkpoint:
            predictions = trainer.predict(ckpt_path=args.checkpoint)
        else:
            predictions = trainer.predict()
        print(f"Generated {len(predictions)} predictions")

        # Save predictions
        output_dir = os.path.join(cfg.DATASET.OUTPUT_PATH, 'predictions')
        os.makedirs(output_dir, exist_ok=True)

        for i, pred in enumerate(predictions):
            output_file = os.path.join(output_dir, f'prediction_{i:04d}.h5')

            # Convert to numpy if needed
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()

            # Save using connectomics IO
            from connectomics.data.io import write_h5
            write_h5(output_file, pred)

        print(f"Predictions saved to: {output_dir}")

    print("Done!")


def test_lightning_setup():
    """Test function to verify Lightning setup works."""
    print("Testing Lightning setup...")

    # Create minimal config using proper approach
    from connectomics.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.SYSTEM.NUM_GPUS = 1
    cfg.SOLVER.SAMPLES_PER_BATCH = 1
    cfg.MODEL.INPUT_SIZE = [64, 64, 64]
    cfg.MODEL.OUTPUT_SIZE = [64, 64, 64]
    cfg.DATASET.OUTPUT_PATH = "/tmp/test_output"
    cfg.DATASET.INPUT_PATH = "/tmp/test_data"
    cfg.DATASET.IMAGE_NAME = "test_image.h5"
    cfg.DATASET.LABEL_NAME = "test_label.h5"
    cfg.freeze()

    try:
        # Test Lightning module creation
        module = create_lightning_module(cfg)
        print("✓ Lightning module created successfully")

        # Test forward pass
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        output = module(dummy_input)
        print("✓ Forward pass successful")

        # Test datamodule creation
        datamodule = ConnectomicsDataModule(cfg)
        print("✓ DataModule created successfully")

        print("✓ All Lightning components working!")
        return True

    except Exception as e:
        print(f"✗ Lightning setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # If no arguments provided, run test
    if len(sys.argv) == 1:
        test_lightning_setup()
    else:
        main()