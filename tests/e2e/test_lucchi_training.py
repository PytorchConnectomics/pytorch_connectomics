#!/usr/bin/env python3
"""
Test MONAI-native dataset integration with Lucchi-style data.

This script creates dummy data and validates that our refactored MONAI datasets
and PyTorch Lightning DataModules work correctly end-to-end.
"""

import os
import sys
import tempfile
import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import Compose

# Import our refactored MONAI-native datasets and DataModules
from connectomics.data.dataset import (
    create_volume_dataset,
    create_data_dicts_from_paths,
)
from connectomics.training.lightning.data import VolumeDataModule
import pytest


@pytest.fixture
def data_paths(tmp_path):
    """Provide synthetic Lucchi-style data paths for downstream tests."""
    return create_dummy_lucchi_data(str(tmp_path))


def create_dummy_lucchi_data(base_path):
    """Create dummy data that matches the Lucchi dataset structure for MONAI."""
    print("Creating dummy Lucchi dataset for MONAI testing...")

    # Create directory structure
    os.makedirs(base_path, exist_ok=True)

    # Create dummy 3D data (smaller than real dataset for quick testing)
    # Real Lucchi: 165x768x1024, we'll use 32x64x64 for testing
    shape = (32, 64, 64)

    # Generate synthetic EM-like data
    print(f"Generating synthetic data with shape: {shape}")

    # Create synthetic EM data (grayscale, 0-255)
    np.random.seed(42)  # For reproducible results

    # Base noise
    img_data = np.random.normal(128, 30, shape).astype(np.uint8)

    # Add some structure (simulate mitochondria-like patterns)
    for i in range(5):  # Add some blob-like structures
        center_z = np.random.randint(3, shape[0]-3)
        center_y = np.random.randint(8, shape[1]-8)
        center_x = np.random.randint(8, shape[2]-8)

        # Create ellipsoid-like structure
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        ellipsoid = ((zz - center_z)/2)**2 + ((yy - center_y)/4)**2 + ((xx - center_x)/4)**2 < 1

        img_data[ellipsoid] = np.random.randint(180, 220)  # Brighter regions

    # Create corresponding binary labels (mitochondria segmentation)
    label_data = np.zeros(shape, dtype=np.uint8)

    # Create mitochondria-like segmentations
    for i in range(4):
        center_z = np.random.randint(2, shape[0]-2)
        center_y = np.random.randint(6, shape[1]-6)
        center_x = np.random.randint(6, shape[2]-6)

        # Create mitochondria-like shapes
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        mito = ((zz - center_z)/1.5)**2 + ((yy - center_y)/3)**2 + ((xx - center_x)/3)**2 < 1

        label_data[mito] = 1  # Binary segmentation

    # Save as HDF5 files (MONAI-compatible format)
    print("Saving training data...")
    train_img_path = os.path.join(base_path, "train_image.h5")
    train_label_path = os.path.join(base_path, "train_label.h5")
    val_img_path = os.path.join(base_path, "val_image.h5")
    val_label_path = os.path.join(base_path, "val_label.h5")

    # Save training data
    with h5py.File(train_img_path, 'w') as f:
        f.create_dataset('main', data=img_data, compression='gzip')

    with h5py.File(train_label_path, 'w') as f:
        f.create_dataset('main', data=label_data, compression='gzip')

    # Create validation data (slightly different)
    val_img_data = img_data + np.random.normal(0, 5, shape).astype(np.int8)
    val_img_data = np.clip(val_img_data, 0, 255).astype(np.uint8)

    # Save validation data
    with h5py.File(val_img_path, 'w') as f:
        f.create_dataset('main', data=val_img_data, compression='gzip')

    with h5py.File(val_label_path, 'w') as f:
        f.create_dataset('main', data=label_data, compression='gzip')

    print(f"✅ Created dummy Lucchi dataset at: {base_path}")
    print(f"   - Training image: {train_img_path} (shape: {shape})")
    print(f"   - Training labels: {train_label_path} (shape: {shape})")
    print(f"   - Validation image: {val_img_path} (shape: {shape})")
    print(f"   - Validation labels: {val_label_path} (shape: {shape})")

    return {
        'train_image_paths': [train_img_path],
        'train_label_paths': [train_label_path],
        'val_image_paths': [val_img_path],
        'val_label_paths': [val_label_path],
    }


def test_monai_dataset_creation(data_paths):
    """Test MONAI dataset creation with dummy data."""
    print("\n🧪 Testing MONAI Dataset Creation")
    print("=" * 50)

    try:
        # Test 1: Create data dictionaries
        print("1. Creating MONAI data dictionaries...")
        train_data_dicts = create_data_dicts_from_paths(
            image_paths=data_paths['train_image_paths'],
            label_paths=data_paths['train_label_paths'],
        )
        print(f"   ✅ Created {len(train_data_dicts)} training data dictionaries")
        print(f"   Sample keys: {list(train_data_dicts[0].keys())}")

        # Test 2: Create standard MONAI dataset
        print("2. Creating standard MONAI volume dataset...")
        dataset = create_volume_dataset(
            image_paths=data_paths['train_image_paths'],
            label_paths=data_paths['train_label_paths'],
            sample_size=(16, 32, 32),
            dataset_type='standard',
            mode='train',
            iter_num=10,
        )
        print(f"   ✅ Created dataset with {len(dataset)} samples")
        print(f"   Dataset type: {type(dataset).__name__}")

        # Test 3: Create cached MONAI dataset
        print("3. Creating cached MONAI volume dataset...")
        cached_dataset = create_volume_dataset(
            image_paths=data_paths['train_image_paths'],
            label_paths=data_paths['train_label_paths'],
            sample_size=(16, 32, 32),
            dataset_type='cached',
            cache_rate=1.0,
            mode='train',
            iter_num=5,
        )
        print(f"   ✅ Created cached dataset with {len(cached_dataset)} samples")

        # Test 4: Test dataset sampling
        print("4. Testing dataset sampling...")
        sample = dataset[0]
        print(f"   ✅ Sample keys: {list(sample.keys())}")
        if 'image' in sample:
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Image dtype: {sample['image'].dtype}")
        if 'label' in sample:
            print(f"   Label shape: {sample['label'].shape}")
            print(f"   Label dtype: {sample['label'].dtype}")

        return True

    except Exception as e:
        print(f"   ❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lightning_datamodule(data_paths):
    """Test PyTorch Lightning DataModule with MONAI datasets."""
    print("\n⚡ Testing Lightning DataModule")
    print("=" * 50)

    try:
        # Test 1: Create volume DataModule
        print("1. Creating VolumeDataModule...")
        datamodule = VolumeDataModule(
            train_image_paths=data_paths['train_image_paths'],
            train_label_paths=data_paths['train_label_paths'],
            val_image_paths=data_paths['val_image_paths'],
            val_label_paths=data_paths['val_label_paths'],
            sample_size=(16, 32, 32),
            batch_size=2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            dataset_type='standard',
        )
        print(f"   ✅ Created VolumeDataModule: {type(datamodule).__name__}")

        # Test 2: Setup datamodule
        print("2. Setting up DataModule...")
        datamodule.setup(stage='fit')
        print(f"   ✅ DataModule setup complete")
        print(f"   Train dataset length: {len(datamodule.train_dataset)}")
        print(f"   Val dataset length: {len(datamodule.val_dataset)}")

        # Test 3: Create DataLoaders
        print("3. Creating DataLoaders...")
        train_loader = datamodule.train_dataloader()
        datamodule.val_dataloader()
        print(f"   ✅ Created train DataLoader")
        print(f"   ✅ Created val DataLoader")

        # Test 4: Sample from DataLoader
        print("4. Testing DataLoader sampling...")
        train_batch = next(iter(train_loader))
        print(f"   ✅ Train batch keys: {list(train_batch.keys())}")
        if 'image' in train_batch:
            print(f"   Train batch image shape: {train_batch['image'].shape}")
        if 'label' in train_batch:
            print(f"   Train batch label shape: {train_batch['label'].shape}")

        return True

    except Exception as e:
        print(f"   ❌ DataModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datamodule_factory(data_paths):
    """Test DataModule factory function with different configurations."""
    print("\n🏭 Testing DataModule Factory Functions")
    print("=" * 50)

    try:
        # Test 1: Basic DataModule without transforms
        print("1. Creating basic DataModule without task-specific transforms...")
        basic_datamodule = VolumeDataModule(
            train_image_paths=data_paths['train_image_paths'],
            train_label_paths=data_paths['train_label_paths'],
            val_image_paths=data_paths['val_image_paths'],
            val_label_paths=data_paths['val_label_paths'],
            sample_size=(16, 32, 32),
            batch_size=1,
            dataset_type='standard',
        )
        print(f"   ✅ Created basic DataModule")

        # Test 2: DataModule with manual transforms (avoiding factory transforms for now)
        print("2. Creating DataModule with manual transforms...")
        manual_datamodule = VolumeDataModule(
            train_image_paths=data_paths['train_image_paths'],
            train_label_paths=data_paths['train_label_paths'],
            sample_size=(16, 32, 32),
            batch_size=1,
            dataset_type='standard',
        )
        print(f"   ✅ Created manual transform DataModule")

        # Test 3: Setup and test sampling
        print("3. Testing DataModule functionality...")
        basic_datamodule.setup(stage='fit')
        train_loader = basic_datamodule.train_dataloader()

        # Get a sample
        sample = next(iter(train_loader))
        print(f"   ✅ Sample keys: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"   ❌ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transform_integration(data_paths):
    """Test MONAI transform pipeline integration."""
    print("\n🔄 Testing MONAI Transform Integration")
    print("=" * 50)

    try:
        # Test 1: Create transform pipeline
        print("1. Creating binary segmentation transform pipeline...")
        # Import custom connectomics loader for HDF5 files
        from connectomics.data.dataset.dataset_volume import LoadVolumed

        # Create a simple transform that doesn't require target generation for testing
        transforms = Compose([
            LoadVolumed(keys=['image', 'label']),
        ])
        print(f"   ✅ Created transform pipeline: {type(transforms).__name__}")

        # Test 2: Create DataModule with custom transforms
        print("2. Creating DataModule with custom transforms...")
        datamodule = VolumeDataModule(
            train_image_paths=data_paths['train_image_paths'],
            train_label_paths=data_paths['train_label_paths'],
            sample_size=(16, 32, 32),
            batch_size=1,
            transforms={'train': transforms},
            dataset_type='standard',
        )
        print(f"   ✅ Created DataModule with transforms")

        # Test 3: Setup and sample
        print("3. Testing transform application...")
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()

        sample = next(iter(train_loader))
        print(f"   ✅ Transformed sample keys: {list(sample.keys())}")

        # Check if standard keys exist
        if 'image' in sample:
            print(f"   ✅ Image loaded with shape: {sample['image'].shape}")
        if 'label' in sample:
            print(f"   ✅ Label loaded with shape: {sample['label'].shape}")

        return True

    except Exception as e:
        print(f"   ❌ Transform integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Testing MONAI-Native Dataset Integration")
    print("=" * 70)

    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")

        # Set up paths
        dataset_path = os.path.join(temp_dir, "lucchi_test")

        # Create dummy data
        data_paths = create_dummy_lucchi_data(dataset_path)

        # Run all tests
        tests = [
            ("MONAI Dataset Creation", test_monai_dataset_creation),
            ("Lightning DataModule", test_lightning_datamodule),
            ("DataModule Factory", test_datamodule_factory),
            ("Transform Integration", test_transform_integration),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")

            try:
                result = test_func(data_paths)
                results.append((test_name, result))

                if result:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")

            except Exception as e:
                print(f"💥 {test_name} ERROR: {e}")
                results.append((test_name, False))

        # Summary
        print("\n" + "=" * 70)
        print("🎯 Test Summary:")
        print("=" * 70)

        passed = 0
        total = len(results)

        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All tests passed! MONAI integration is working correctly!")
            return True
        else:
            print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
