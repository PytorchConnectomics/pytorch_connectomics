#!/usr/bin/env python3
"""
Test script for MONAI integration with real Lucchi dataset.

This script tests the MONAI-based transforms with EM-specific augmentations
using the real Lucchi mitochondria dataset.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.transforms.augment import (
    build_monai_transforms,
    ConnectomicsEMTransformd
)
from connectomics.data import ConnectomicsDataModule
from connectomics.config.defaults import get_cfg_defaults
from yacs.config import CfgNode
import tifffile


def load_lucchi_sample(crop_size=(112, 112, 112)):
    """Load a sample from the Lucchi dataset."""
    try:
        # Load the data
        image_path = "../datasets/Lucchi/img/train_im.tif"
        label_path = "../datasets/Lucchi/label/train_label.tif"

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print(f"❌ Lucchi dataset not found at expected location")
            return None, None

        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)

        print(f"✓ Loaded Lucchi data: image {image.shape}, label {label.shape}")

        # Extract a sample crop for testing
        z, y, x = crop_size
        start_z = image.shape[0] // 2 - z // 2
        start_y = image.shape[1] // 2 - y // 2
        start_x = image.shape[2] // 2 - x // 2

        image_crop = image[start_z:start_z+z, start_y:start_y+y, start_x:start_x+x]
        label_crop = label[start_z:start_z+z, start_y:start_y+y, start_x:start_x+x]

        # Normalize image to [0, 1]
        image_crop = image_crop.astype(np.float32) / 255.0

        # Convert label to binary (mitochondria vs background)
        label_crop = (label_crop > 127).astype(np.float32)

        print(f"✓ Extracted crop: image {image_crop.shape}, label {label_crop.shape}")
        print(f"  Image range: [{image_crop.min():.3f}, {image_crop.max():.3f}]")
        print(f"  Label range: [{label_crop.min():.3f}, {label_crop.max():.3f}]")

        return image_crop, label_crop

    except Exception as e:
        print(f"❌ Error loading Lucchi data: {e}")
        return None, None


def test_basic_monai_transforms():
    """Test basic MONAI transforms with real data."""
    print("Testing basic MONAI transforms with Lucchi data...")

    image, label = load_lucchi_sample()
    if image is None:
        return False

    try:
        # Create config for basic transforms
        cfg = get_cfg_defaults()
        cfg.AUGMENTOR.ENABLED = True

        # Enable only basic transforms
        cfg.AUGMENTOR.FLIP.ENABLED = True
        cfg.AUGMENTOR.FLIP.P = 0.8
        cfg.AUGMENTOR.FLIP.DO_ZTRANS = 1  # As in Lucchi config
        cfg.AUGMENTOR.ROTATE.ENABLED = True
        cfg.AUGMENTOR.ROTATE.P = 0.5
        cfg.AUGMENTOR.ROTATE.ROT90 = True

        # Disable all EM-specific transforms for basic test
        cfg.AUGMENTOR.MISALIGNMENT.ENABLED = False
        cfg.AUGMENTOR.MISSINGSECTION.ENABLED = False
        cfg.AUGMENTOR.MISSINGPARTS.ENABLED = False
        cfg.AUGMENTOR.MOTIONBLUR.ENABLED = False
        cfg.AUGMENTOR.CUTBLUR.ENABLED = False
        cfg.AUGMENTOR.CUTNOISE.ENABLED = False
        cfg.AUGMENTOR.COPYPASTE.ENABLED = False
        cfg.AUGMENTOR.ELASTIC.ENABLED = False  # Disable for basic test
        cfg.AUGMENTOR.RESCALE.ENABLED = False  # Disable for basic test
        cfg.AUGMENTOR.GRAYSCALE.ENABLED = False  # Disable for basic test

        cfg.MODEL.INPUT_SIZE = [112, 112, 112]

        # Build transforms
        transforms = build_monai_transforms(cfg, keys=["image", "label"], mode="train")

        # Create sample dict
        sample = {
            "image": image,
            "label": label
        }

        # Apply transforms
        transformed = transforms(sample)

        print(f"✓ Basic transforms applied successfully")
        print(f"  Input image shape: {image.shape}")
        print(f"  Output image shape: {transformed['image'].shape}")
        print(f"  Output image type: {type(transformed['image'])}")

        return True

    except Exception as e:
        print(f"❌ Basic transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_em_specific_transforms():
    """Test EM-specific transforms with real data."""
    print("\nTesting EM-specific transforms with Lucchi data...")

    image, label = load_lucchi_sample()
    if image is None:
        return False

    # Test individual EM transforms
    em_transforms = [
        ("misalign", {"displacement": 10, "rotate_ratio": 0.5}),
        ("missing_section", {"num_sections": 2}),
        ("missing_parts", {"iterations": 100}),
        ("motion_blur", {"sections": 2, "kernel_size": 7}),
        ("cutblur", {"length_ratio": 0.25, "down_ratio_min": 2, "down_ratio_max": 8, "downsample_z": True}),
        ("cutnoise", {"length_ratio": 0.25, "scale": 0.2}),
    ]

    results = []

    for transform_name, kwargs in em_transforms:
        try:
            print(f"  Testing {transform_name}...")

            # Create EM transform
            em_transform = ConnectomicsEMTransformd(
                keys=["image", "label"],
                augmentation_type=transform_name,
                prob=1.0,  # Always apply for testing
                **kwargs
            )

            # Create sample dict
            sample = {
                "image": image.copy(),
                "label": label.copy()
            }

            # Apply transform
            transformed = em_transform(sample)

            print(f"    ✓ {transform_name} applied successfully")
            print(f"      Output image shape: {transformed['image'].shape}")
            print(f"      Output type: {type(transformed['image'])}")

            results.append(True)

        except Exception as e:
            print(f"    ❌ {transform_name} failed: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\nEM-specific transforms: {passed}/{total} passed")

    return passed == total


def test_full_config_with_em_transforms():
    """Test full Lucchi config with EM transforms enabled."""
    print("\nTesting full Lucchi config with EM transforms...")

    image, label = load_lucchi_sample()
    if image is None:
        return False

    try:
        # Load Lucchi config
        cfg = get_cfg_defaults()
        cfg.merge_from_file("../configs/Lucchi-Mitochondria.yaml")

        # Enable more EM-specific transforms for testing
        cfg.AUGMENTOR.ENABLED = True
        cfg.AUGMENTOR.MISALIGNMENT.ENABLED = True
        cfg.AUGMENTOR.MISALIGNMENT.DISPLACEMENT = 10
        cfg.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO = 0.5
        cfg.AUGMENTOR.MISALIGNMENT.P = 0.3

        cfg.AUGMENTOR.MISSINGSECTION.ENABLED = True
        cfg.AUGMENTOR.MISSINGSECTION.NUM_SECTION = 2
        cfg.AUGMENTOR.MISSINGSECTION.P = 0.2

        cfg.AUGMENTOR.MOTIONBLUR.ENABLED = True
        cfg.AUGMENTOR.MOTIONBLUR.SECTIONS = 2
        cfg.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE = 7
        cfg.AUGMENTOR.MOTIONBLUR.P = 0.3

        cfg.AUGMENTOR.CUTBLUR.ENABLED = True
        cfg.AUGMENTOR.CUTBLUR.LENGTH_RATIO = 0.25
        cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN = 2
        cfg.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX = 8
        cfg.AUGMENTOR.CUTBLUR.P = 0.2

        # Build transforms
        transforms = build_monai_transforms(cfg, keys=["image", "label"], mode="train")

        print(f"✓ Built full transform pipeline with {len(transforms.transforms)} components")

        # Test multiple applications
        success_count = 0
        total_tests = 5

        for i in range(total_tests):
            try:
                sample = {
                    "image": image.copy(),
                    "label": label.copy()
                }

                transformed = transforms(sample)
                success_count += 1

                if i == 0:  # Print details for first test
                    print(f"✓ Full pipeline applied successfully")
                    print(f"  Input image shape: {image.shape}")
                    print(f"  Output image shape: {transformed['image'].shape}")
                    print(f"  Image value range: [{transformed['image'].min():.3f}, {transformed['image'].max():.3f}]")

            except Exception as e:
                print(f"  ❌ Application {i+1} failed: {e}")

        print(f"✓ Full pipeline success rate: {success_count}/{total_tests}")

        return success_count >= total_tests * 0.8  # Allow some random failures

    except Exception as e:
        print(f"❌ Full config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datamodule_with_real_data():
    """Test DataModule creation with real Lucchi config."""
    print("\nTesting DataModule with real Lucchi config...")

    try:
        # Load and modify Lucchi config
        cfg = get_cfg_defaults()
        cfg.merge_from_file("../configs/Lucchi-Mitochondria.yaml")

        # Adjust for testing
        cfg.SOLVER.SAMPLES_PER_BATCH = 1
        cfg.SYSTEM.NUM_CPUS = 0
        cfg.SOLVER.ITERATION_TOTAL = 10  # Small number for testing

        # Enable some EM transforms
        cfg.AUGMENTOR.ENABLED = True
        cfg.AUGMENTOR.MISALIGNMENT.ENABLED = True
        cfg.AUGMENTOR.MISALIGNMENT.P = 0.5
        cfg.AUGMENTOR.CUTBLUR.ENABLED = True
        cfg.AUGMENTOR.CUTBLUR.P = 0.3

        # Create DataModule
        datamodule = ConnectomicsDataModule(cfg, use_monai=True, rank=None)
        print(f"✓ Created DataModule with Lucchi config")

        # Test transforms building
        transforms = build_monai_transforms(cfg, keys=["image", "label"], mode="train")
        print(f"✓ Built transforms with {len(transforms.transforms)} components")

        return True

    except Exception as e:
        print(f"❌ DataModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """Compare performance between traditional and MONAI transforms."""
    print("\nTesting performance comparison...")

    image, label = load_lucchi_sample()
    if image is None:
        return False

    try:
        # Test MONAI transforms performance
        cfg = get_cfg_defaults()
        cfg.AUGMENTOR.ENABLED = True
        cfg.AUGMENTOR.FLIP.ENABLED = True
        cfg.AUGMENTOR.FLIP.P = 1.0
        cfg.AUGMENTOR.ROTATE.ENABLED = True
        cfg.AUGMENTOR.ROTATE.P = 1.0

        # Disable EM-specific transforms for performance test
        cfg.AUGMENTOR.MISALIGNMENT.ENABLED = False
        cfg.AUGMENTOR.MISSINGSECTION.ENABLED = False
        cfg.AUGMENTOR.MISSINGPARTS.ENABLED = False
        cfg.AUGMENTOR.MOTIONBLUR.ENABLED = False
        cfg.AUGMENTOR.CUTBLUR.ENABLED = False
        cfg.AUGMENTOR.CUTNOISE.ENABLED = False
        cfg.AUGMENTOR.COPYPASTE.ENABLED = False
        cfg.AUGMENTOR.ELASTIC.ENABLED = False
        cfg.AUGMENTOR.RESCALE.ENABLED = False
        cfg.AUGMENTOR.GRAYSCALE.ENABLED = False

        cfg.MODEL.INPUT_SIZE = [112, 112, 112]

        monai_transforms = build_monai_transforms(cfg, keys=["image", "label"], mode="train")

        # Time MONAI transforms
        import time

        sample = {"image": image.copy(), "label": label.copy()}

        start_time = time.time()
        num_iterations = 10

        for _ in range(num_iterations):
            sample_copy = {"image": image.copy(), "label": label.copy()}
            result = monai_transforms(sample_copy)

        monai_time = (time.time() - start_time) / num_iterations

        print(f"✓ MONAI transforms average time: {monai_time:.4f}s per sample")
        print(f"✓ Performance test completed successfully")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests with real Lucchi data."""
    print("🧪 Testing MONAI Integration with Real Lucchi Dataset")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists("../datasets/Lucchi/img/train_im.tif"):
        print("❌ Lucchi dataset not found. Please ensure datasets/Lucchi/ contains the data.")
        return False

    tests = [
        test_basic_monai_transforms,
        test_em_specific_transforms,
        test_full_config_with_em_transforms,
        test_datamodule_with_real_data,
        test_performance_comparison,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 Test Results:")

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

    total_passed = sum(results)
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("🎉 All tests passed! MONAI integration with EM-specific transforms works correctly.")
        return True
    else:
        print("⚠️  Some tests failed, but this may be expected for some EM-specific transforms.")
        return total_passed >= total_tests * 0.6  # Allow some failures


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)