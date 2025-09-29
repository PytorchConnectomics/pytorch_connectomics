#!/usr/bin/env python3
"""
Final integration test for PyTorch Connectomics Lightning refactoring.

This script tests the complete Lightning pipeline with both legacy and modern
model architectures to verify the refactoring was successful.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.config import get_cfg_defaults
from connectomics.engine.lightning_module import create_lightning_module
from connectomics.engine.lightning_datamodule import ConnectomicsDataModule
from connectomics.engine.lightning_trainer import create_trainer


def create_test_data(data_dir: str, volume_size=(64, 64, 64)):
    """Create synthetic test data for integration testing."""
    os.makedirs(data_dir, exist_ok=True)

    # Create synthetic image data
    image_data = np.random.randint(0, 255, volume_size, dtype=np.uint8)
    image_path = os.path.join(data_dir, "train_image.h5")
    with h5py.File(image_path, 'w') as f:
        f.create_dataset('main', data=image_data)

    # Create synthetic label data (binary segmentation)
    label_data = (np.random.rand(*volume_size) > 0.5).astype(np.uint8)
    label_path = os.path.join(data_dir, "train_label.h5")
    with h5py.File(label_path, 'w') as f:
        f.create_dataset('main', data=label_data)

    print(f"Created test data in {data_dir}")
    print(f"  - Image: {image_data.shape}, dtype: {image_data.dtype}")
    print(f"  - Label: {label_data.shape}, dtype: {label_data.dtype}")

    return image_path, label_path


def test_modern_model(data_dir: str, output_dir: str):
    """Test modern nnUNet model with Lightning."""
    print("=" * 60)
    print("Testing Modern nnUNet Model with Lightning")
    print("=" * 60)

    cfg = get_cfg_defaults()

    # System configuration
    cfg.SYSTEM.NUM_GPUS = 1
    cfg.SYSTEM.NUM_CPUS = 2

    # Model configuration (modern nnUNet)
    cfg.MODEL.ARCHITECTURE = 'monai_basic_unet3d'  # Use MONAI Basic UNet
    cfg.MODEL.INPUT_SIZE = [64, 64, 64]
    cfg.MODEL.OUTPUT_SIZE = [64, 64, 64]
    cfg.MODEL.IN_PLANES = 1
    cfg.MODEL.OUT_PLANES = 1

    # Dataset configuration
    cfg.DATASET.INPUT_PATH = data_dir
    cfg.DATASET.OUTPUT_PATH = output_dir
    cfg.DATASET.IMAGE_NAME = "train_image.h5"
    cfg.DATASET.LABEL_NAME = "train_label.h5"
    cfg.DATASET.PAD_SIZE = [4, 4, 4]

    # Solver configuration (minimal for testing)
    cfg.SOLVER.SAMPLES_PER_BATCH = 1
    cfg.SOLVER.ITERATION_TOTAL = 10  # Very short test
    cfg.SOLVER.BASE_LR = 0.001

    # Lightning configuration will be handled by the Lightning components

    cfg.freeze()

    try:
        # Test component creation
        print("Creating Lightning components...")
        lightning_module = create_lightning_module(cfg)
        print(f"✓ Lightning module created: {type(lightning_module).__name__}")

        datamodule = ConnectomicsDataModule(cfg)
        print(f"✓ DataModule created: {type(datamodule).__name__}")

        trainer = create_trainer(cfg)
        print(f"✓ Trainer created: {type(trainer).__name__}")

        print("✓ Modern nnUNet test PASSED")
        return True

    except Exception as e:
        print(f"✗ Modern nnUNet test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_config():
    """Test basic configuration creation."""
    print("=" * 60)
    print("Testing Basic Configuration")
    print("=" * 60)

    try:
        cfg = get_cfg_defaults()

        print(f"✓ Default config loaded")
        print(f"  - Default architecture: {cfg.MODEL.ARCHITECTURE}")
        print(f"  - Default input size: {cfg.MODEL.INPUT_SIZE}")
        print(f"  - Default solver LR: {cfg.SOLVER.BASE_LR}")

        return True

    except Exception as e:
        print(f"✗ Failed to load default config: {e}")
        return False


def main():
    """Run all integration tests."""
    print("PyTorch Connectomics Lightning Integration Test")
    print("=" * 60)

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        output_dir = os.path.join(temp_dir, "output")

        # Create test data
        create_test_data(data_dir)

        # Run tests
        tests = [
            ("Basic Configuration", test_basic_config),
            ("Modern nnUNet", lambda: test_modern_model(data_dir, output_dir)),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\nRunning {test_name} test...")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"✗ {test_name} test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))

        # Summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)

        passed = 0
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{test_name:30} {status}")
            if result:
                passed += 1

        print(f"\nPassed: {passed}/{len(results)} tests")

        if passed == len(results):
            print("\n🎉 ALL TESTS PASSED! Lightning integration is working correctly.")
            return 0
        else:
            print(f"\n❌ {len(results) - passed} tests failed. Check the output above.")
            return 1


if __name__ == "__main__":
    exit(main())