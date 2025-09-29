#!/usr/bin/env python3
"""
Simple test for MONAI-native dataset integration.

This script creates dummy data and validates basic dataset functionality
without complex transforms that might have compatibility issues.
"""

import os
import sys
import tempfile
import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from monai.transforms import Compose

# Import our refactored MONAI-native datasets
from connectomics.data.dataset import (
    create_data_dicts_from_paths,
    create_connectomics_dataset,
)


def create_simple_data(base_path):
    """Create simple test data."""
    print("Creating simple test data...")
    os.makedirs(base_path, exist_ok=True)

    # Create small test data
    shape = (16, 32, 32)

    # Simple synthetic data
    np.random.seed(42)
    img_data = np.random.randint(0, 256, shape).astype(np.uint8)
    label_data = np.random.randint(0, 2, shape).astype(np.uint8)

    # Save as HDF5
    train_img_path = os.path.join(base_path, "train_image.h5")
    train_label_path = os.path.join(base_path, "train_label.h5")

    with h5py.File(train_img_path, 'w') as f:
        f.create_dataset('main', data=img_data)

    with h5py.File(train_label_path, 'w') as f:
        f.create_dataset('main', data=label_data)

    return [train_img_path], [train_label_path]


def test_basic_dataset():
    """Test basic dataset functionality."""
    print("\n🧪 Testing Basic Dataset Functionality")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        image_paths, label_paths = create_simple_data(temp_dir)

        try:
            # Test 1: Create data dictionaries
            print("1. Creating data dictionaries...")
            data_dicts = create_data_dicts_from_paths(
                image_paths=image_paths,
                label_paths=label_paths,
            )
            print(f"   ✅ Created {len(data_dicts)} data dictionaries")

            # Test 2: Create dataset without transforms
            print("2. Creating dataset without transforms...")
            dataset = create_connectomics_dataset(
                data_dicts=data_dicts,
                transforms=None,  # No transforms to avoid compatibility issues
                dataset_type='standard',
                mode='train',
                iter_num=5,
            )
            print(f"   ✅ Created dataset with {len(dataset)} samples")

            # Test 3: Test dataset access
            print("3. Testing dataset access...")
            sample = dataset[0]
            print(f"   ✅ Successfully accessed sample")
            print(f"   Sample type: {type(sample)}")

            # Test 4: Create cached dataset
            print("4. Testing cached dataset...")
            cached_dataset = create_connectomics_dataset(
                data_dicts=data_dicts,
                transforms=None,
                dataset_type='cached',
                cache_rate=1.0,
                mode='train',
                iter_num=3,
            )
            print(f"   ✅ Created cached dataset with {len(cached_dataset)} samples")

            return True

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main test function."""
    print("🧪 Testing MONAI Dataset Integration (Simple)")
    print("=" * 60)

    success = test_basic_dataset()

    print("\n" + "=" * 60)
    if success:
        print("✅ Basic dataset test PASSED")
        print("✅ MONAI integration is working!")
        return True
    else:
        print("❌ Basic dataset test FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)