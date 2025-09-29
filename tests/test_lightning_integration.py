#!/usr/bin/env python3
"""
Test script for PyTorch Lightning integration.

This script validates that the Lightning components work correctly
with the existing PyTorch Connectomics codebase.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from unittest.mock import MagicMock

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectomics.config import get_cfg_defaults
from connectomics.config.lightning_config import (
    adapt_cfg_for_lightning,
    validate_lightning_config,
    print_config_comparison
)


def create_test_config():
    """Create a minimal test configuration."""
    cfg = get_cfg_defaults()

    # Set minimal required values
    cfg.MODEL.ARCHITECTURE = 'unet_3d'
    cfg.MODEL.IN_PLANES = 1
    cfg.MODEL.OUT_PLANES = 1
    cfg.MODEL.INPUT_SIZE = [64, 64, 64]
    cfg.MODEL.OUTPUT_SIZE = [64, 64, 64]
    cfg.MODEL.FILTERS = [16, 32, 64]
    cfg.MODEL.BLOCKS = [2, 2, 2]
    cfg.MODEL.ISOTROPY = [False, False, True]  # Match length of filters
    cfg.MODEL.LOSS_OPTION = [['WeightedBCEWithLogitsLoss']]  # Use logits-compatible loss

    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.SAMPLES_PER_BATCH = 2
    cfg.SOLVER.ITERATION_TOTAL = 100

    cfg.SYSTEM.NUM_GPUS = 0  # Force CPU for testing to avoid device issues
    cfg.SYSTEM.NUM_CPUS = 2
    cfg.SYSTEM.PARALLEL = 'NONE'  # No parallelism for CPU testing

    # Disable monitoring for cleaner testing
    cfg.MONITOR = None

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    cfg.DATASET.OUTPUT_PATH = temp_dir
    cfg.DATASET.IMAGE_NAME = 'test_image.h5'
    cfg.DATASET.LABEL_NAME = 'test_label.h5'

    return cfg


def test_config_validation():
    """Test configuration validation and conversion."""
    print("Testing configuration validation...")

    cfg = create_test_config()

    # Test validation
    is_valid = validate_lightning_config(cfg)
    print(f"Configuration validation: {'PASSED' if is_valid else 'FAILED'}")

    # Test conversion
    lightning_cfg = adapt_cfg_for_lightning(cfg)
    print(f"Configuration conversion: {'PASSED' if lightning_cfg else 'FAILED'}")

    # Print comparison
    print_config_comparison(cfg)

    return is_valid and lightning_cfg


def test_lightning_module_creation():
    """Test Lightning module creation."""
    print("\nTesting Lightning module creation...")

    try:
        from connectomics.engine.lightning_module import ConnectomicsModule

        cfg = create_test_config()
        module = ConnectomicsModule(cfg)

        print(f"Lightning module creation: PASSED")
        print(f"Model architecture: {module.model.__class__.__name__}")

        return True

    except Exception as e:
        import traceback
        print(f"Lightning module creation: FAILED - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_lightning_trainer_creation():
    """Test Lightning trainer creation."""
    print("\nTesting Lightning trainer creation...")

    try:
        from connectomics.engine.lightning_trainer import LightningTrainer

        cfg = create_test_config()
        trainer = LightningTrainer(cfg)

        print(f"Lightning trainer creation: PASSED")
        print(f"Trainer accelerator: {trainer.trainer.accelerator}")
        print(f"Trainer devices: {trainer.trainer.num_devices}")

        return True

    except Exception as e:
        print(f"Lightning trainer creation: FAILED - {e}")
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")

    try:
        from connectomics.engine.lightning_module import ConnectomicsModule

        cfg = create_test_config()
        module = ConnectomicsModule(cfg)

        # Create dummy input
        batch_size = 1
        input_tensor = torch.randn(
            batch_size,
            cfg.MODEL.IN_PLANES,
            *cfg.MODEL.INPUT_SIZE
        )

        # Test forward pass
        with torch.no_grad():
            output = module.forward(input_tensor)

        expected_shape = (batch_size, cfg.MODEL.OUT_PLANES, *cfg.MODEL.OUTPUT_SIZE)
        shape_correct = output.shape == expected_shape

        print(f"Forward pass: {'PASSED' if shape_correct else 'FAILED'}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")

        return shape_correct

    except Exception as e:
        print(f"Forward pass: FAILED - {e}")
        return False


def test_training_step():
    """Test training step with dummy data."""
    print("\nTesting training step...")

    try:
        from connectomics.engine.lightning_module import ConnectomicsModule

        cfg = create_test_config()
        module = ConnectomicsModule(cfg)

        # Create dummy batch (mimicking connectomics data format)
        batch = MagicMock()
        batch.out_input = torch.rand(1, cfg.MODEL.IN_PLANES, *cfg.MODEL.INPUT_SIZE)  # Use rand for [0,1] range
        batch.out_target_l = [torch.randint(0, 2, (1, cfg.MODEL.OUT_PLANES, *cfg.MODEL.OUTPUT_SIZE)).float()]  # Binary targets
        batch.out_weight_l = [torch.ones(1, cfg.MODEL.OUT_PLANES, *cfg.MODEL.OUTPUT_SIZE)]

        # Test training step
        loss = module.training_step(batch, 0)

        step_successful = isinstance(loss, torch.Tensor) and loss.dim() == 0

        print(f"Training step: {'PASSED' if step_successful else 'FAILED'}")
        print(f"Loss value: {loss.item() if step_successful else 'N/A'}")

        return step_successful

    except Exception as e:
        import traceback
        print(f"Training step: FAILED - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_optimizer_configuration():
    """Test optimizer configuration."""
    print("\nTesting optimizer configuration...")

    try:
        from connectomics.engine.lightning_module import ConnectomicsModule

        cfg = create_test_config()
        module = ConnectomicsModule(cfg)

        optimizer_config = module.configure_optimizers()

        config_valid = optimizer_config is not None

        print(f"Optimizer configuration: {'PASSED' if config_valid else 'FAILED'}")

        if isinstance(optimizer_config, dict):
            print(f"Optimizer type: {type(optimizer_config['optimizer']).__name__}")
            if 'lr_scheduler' in optimizer_config:
                print(f"Scheduler type: {type(optimizer_config['lr_scheduler']['scheduler']).__name__}")
        else:
            print(f"Optimizer type: {type(optimizer_config).__name__}")

        return config_valid

    except Exception as e:
        print(f"Optimizer configuration: FAILED - {e}")
        return False


def test_enhanced_data_module():
    """Test enhanced ConnectomicsDataModule with MONAI integration."""
    print("\nTesting enhanced data module with MONAI...")

    try:
        from connectomics.engine.lightning_module import ConnectomicsDataModule
        import h5py

        cfg = create_test_config()

        # Create dummy data files for testing
        dummy_data = np.random.rand(64, 64, 64).astype(np.float32)
        dummy_label = np.random.randint(0, 2, (64, 64, 64)).astype(np.uint8)

        # Create the image file
        img_path = os.path.join(cfg.DATASET.OUTPUT_PATH, cfg.DATASET.IMAGE_NAME)
        with h5py.File(img_path, 'w') as f:
            f.create_dataset('main', data=dummy_data)

        # Create the label file
        label_path = os.path.join(cfg.DATASET.OUTPUT_PATH, cfg.DATASET.LABEL_NAME)
        with h5py.File(label_path, 'w') as f:
            f.create_dataset('main', data=dummy_label)

        # Update config to use absolute paths
        cfg.DATASET.INPUT_PATH = [cfg.DATASET.OUTPUT_PATH]
        cfg.DATASET.IS_ABSOLUTE_PATH = False

        # Test with MONAI enabled
        data_module = ConnectomicsDataModule(cfg, use_monai=True)

        print(f"Data module created: {'PASSED' if data_module else 'FAILED'}")
        print(f"MONAI enabled: {data_module.use_monai}")

        # Test basic functionality without full setup (which requires actual data)
        try:
            # Test MONAI transforms setup
            data_module._setup_monai_transforms()
            print(f"MONAI transforms setup: {'PASSED' if data_module.train_transforms else 'FALLBACK'}")
        except Exception as e:
            print(f"MONAI transforms setup: FAILED - {e}")

        # Test fallback dataloader creation (using existing system)
        try:
            train_loader = data_module.train_dataloader()
            print(f"Train dataloader creation: {'PASSED' if train_loader else 'FAILED'}")
        except Exception as e:
            print(f"Train dataloader creation: FAILED - {e}")
            return False

        return True

    except Exception as e:
        import traceback
        print(f"Enhanced data module: FAILED - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def run_all_tests():
    """Run all Lightning integration tests."""
    print("PyTorch Connectomics - Lightning Integration Tests")
    print("=" * 60)

    tests = [
        ("Configuration Validation", test_config_validation),
        ("Lightning Module Creation", test_lightning_module_creation),
        ("Lightning Trainer Creation", test_lightning_trainer_creation),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Optimizer Configuration", test_optimizer_configuration),
        ("Enhanced Data Module", test_enhanced_data_module),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:30}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Lightning integration is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)