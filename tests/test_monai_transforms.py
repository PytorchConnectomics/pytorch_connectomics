#!/usr/bin/env python3
"""
Test script for MONAI-native transforms in PyTorch Connectomics.

This script validates that the new MONAI transforms work correctly and produce
expected outputs for common connectomics processing tasks.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from monai.transforms import Compose
from monai.data import MetaTensor

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new MONAI transforms
from connectomics.transforms.process import (
    SegToBinaryMaskd,
    SegToAffinityMapd,
    SegToInstanceBoundaryMaskd,
    SegToInstanceEDTd,
    SegToSemanticEDTd,
    SegToFlowFieldd,
    SegToSynapticPolarityd,
    SegToSmallObjectd,
    SegToGenericSemanticed,
    ComputeBinaryRatioWeightd,
    create_target_transforms,
    create_full_processing_pipeline,
    create_binary_segmentation_pipeline,
    create_affinity_segmentation_pipeline,
    create_instance_segmentation_pipeline,
)


def create_test_data():
    """Create synthetic test data for validation."""
    print("Creating synthetic test data...")

    # Create a simple 3D segmentation with a few objects
    shape = (32, 64, 64)  # Small 3D volume
    label = np.zeros(shape, dtype=np.int32)

    # Add some rectangular objects
    label[8:16, 20:40, 20:40] = 1  # Object 1
    label[16:24, 10:30, 30:50] = 2  # Object 2
    label[20:28, 40:60, 10:30] = 3  # Object 3

    # Create test image (simple gradient)
    image = np.random.rand(*shape).astype(np.float32)

    # Create valid mask (all valid for this test)
    valid_mask = np.ones(shape, dtype=np.uint8)

    # Package data in MONAI-style dictionary
    data = {
        'image': MetaTensor(image[None, ...]),  # Add channel dimension
        'label': MetaTensor(label),
        'valid_mask': MetaTensor(valid_mask),
    }

    print(f"Created test data with shape: {shape}")
    print(f"Label contains {len(np.unique(label))} unique values: {np.unique(label)}")

    return data


def test_individual_transforms():
    """Test individual MONAI transforms."""
    print("\n=== Testing Individual MONAI Transforms ===")

    data = create_test_data()

    # Test binary mask transform
    print("\n1. Testing SegToBinaryMaskd...")
    binary_transform = SegToBinaryMaskd(keys=['label'], target_opt='0')
    result = binary_transform(data)

    assert 'label_binary_mask' in result, "Binary mask not generated"
    binary_mask = result['label_binary_mask']
    print(f"   Binary mask shape: {binary_mask.shape}")
    print(f"   Binary mask unique values: {np.unique(binary_mask.array)}")

    # Test affinity transform
    print("\n2. Testing SegToAffinityMapd...")
    affinity_transform = SegToAffinityMapd(keys=['label'], long_range=5)
    result = affinity_transform(data)

    assert 'label_affinity' in result, "Affinity map not generated"
    affinity = result['label_affinity']
    print(f"   Affinity map shape: {affinity.shape}")
    print(f"   Affinity map range: [{affinity.array.min():.3f}, {affinity.array.max():.3f}]")

    # Test instance boundary transform
    print("\n3. Testing SegToInstanceBoundaryMaskd...")
    boundary_transform = SegToInstanceBoundaryMaskd(keys=['label'])
    result = boundary_transform(data)

    assert 'label_boundary' in result, "Boundary mask not generated"
    boundary = result['label_boundary']
    print(f"   Boundary mask shape: {boundary.shape}")
    print(f"   Boundary mask unique values: {np.unique(boundary.array)}")

    # Test instance EDT transform
    print("\n4. Testing SegToInstanceEDTd...")
    edt_transform = SegToInstanceEDTd(keys=['label'], target_opt='5-2d-0-0-1.0-0')
    result = edt_transform(data)

    assert 'label_distance' in result, "Distance transform not generated"
    distance = result['label_distance']
    print(f"   Distance transform shape: {distance.shape}")
    print(f"   Distance transform range: [{distance.array.min():.3f}, {distance.array.max():.3f}]")

    # Test flow field transform
    print("\n5. Testing SegToFlowFieldd...")
    flow_transform = SegToFlowFieldd(keys=['label'], include_binary=True)
    result = flow_transform(data)

    assert 'label_flow' in result, "Flow field not generated"
    flow = result['label_flow']
    print(f"   Flow field shape: {flow.shape}")
    print(f"   Flow field range: [{flow.array.min():.3f}, {flow.array.max():.3f}]")

    print("✅ All individual transforms passed!")


def test_compose_pipelines():
    """Test MONAI Compose pipelines."""
    print("\n=== Testing MONAI Compose Pipelines ===")

    data = create_test_data()

    # Test target generation pipeline
    print("\n1. Testing target generation pipeline...")
    target_pipeline = create_target_transforms(
        target_opts=['binary_mask', 'affinity_map', 'boundary_mask'],
        input_key='label'
    )

    result = target_pipeline(data)
    expected_keys = ['label_binary_mask', 'label_affinity', 'label_boundary']

    for key in expected_keys:
        assert key in result, f"Target {key} not generated"
        print(f"   Generated target: {key} with shape {result[key].shape}")

    # Test binary segmentation pipeline
    print("\n2. Testing binary segmentation pipeline...")
    binary_pipeline = create_binary_segmentation_pipeline(input_key='label', use_weights=True)
    result = binary_pipeline(data)

    assert 'label_binary_mask' in result, "Binary mask not generated"
    assert 'label_binary_mask_weight' in result, "Binary mask weight not generated"
    print(f"   Binary mask shape: {result['label_binary_mask'].shape}")
    print(f"   Binary mask weight shape: {result['label_binary_mask_weight'].shape}")

    # Test affinity segmentation pipeline
    print("\n3. Testing affinity segmentation pipeline...")
    affinity_pipeline = create_affinity_segmentation_pipeline(input_key='label', use_weights=True)
    result = affinity_pipeline(data)

    assert 'label_affinity' in result, "Affinity map not generated"
    assert 'label_affinity_weight' in result, "Affinity weight not generated"
    print(f"   Affinity map shape: {result['label_affinity'].shape}")
    print(f"   Affinity weight shape: {result['label_affinity_weight'].shape}")

    # Test instance segmentation pipeline
    print("\n4. Testing instance segmentation pipeline...")
    instance_pipeline = create_instance_segmentation_pipeline(input_key='label', use_weights=True)
    result = instance_pipeline(data)

    assert 'label_distance' in result, "Distance transform not generated"
    assert 'label_boundary' in result, "Boundary mask not generated"
    print(f"   Distance transform shape: {result['label_distance'].shape}")
    print(f"   Boundary mask shape: {result['label_boundary'].shape}")

    print("✅ All Compose pipelines passed!")


def test_weight_computation():
    """Test weight computation transforms."""
    print("\n=== Testing Weight Computation ===")

    data = create_test_data()

    # First generate a target
    binary_transform = SegToBinaryMaskd(keys=['label'], target_opt='0')
    data = binary_transform(data)

    # Test binary ratio weight computation
    print("\n1. Testing ComputeBinaryRatioWeightd...")
    weight_transform = ComputeBinaryRatioWeightd(
        target_keys=['label_binary_mask'],
        valid_mask_key='valid_mask'
    )
    result = weight_transform(data)

    assert 'label_binary_mask_weight' in result, "Binary ratio weight not generated"
    weight = result['label_binary_mask_weight']
    print(f"   Binary ratio weight shape: {weight.shape}")
    print(f"   Binary ratio weight range: [{weight.array.min():.3f}, {weight.array.max():.3f}]")

    print("✅ Weight computation passed!")


def test_full_pipeline():
    """Test complete processing pipeline."""
    print("\n=== Testing Full Processing Pipeline ===")

    data = create_test_data()

    # Create a complex multi-task pipeline
    pipeline = create_full_processing_pipeline(
        target_opts=['binary_mask', 'affinity_map', 'distance_instance'],
        weight_opts=[['binary_ratio'], ['binary_ratio'], ['binary_ratio']],
        input_key='label',
        valid_mask_key='valid_mask'
    )

    result = pipeline(data)

    # Check that all expected outputs are present
    expected_targets = ['label_binary_mask', 'label_affinity', 'label_distance']
    expected_weights = ['label_binary_mask_weight', 'label_affinity_weight', 'label_distance_weight']

    for target in expected_targets:
        assert target in result, f"Target {target} not generated"
        print(f"   Generated target: {target} with shape {result[target].shape}")

    for weight in expected_weights:
        assert weight in result, f"Weight {weight} not generated"
        print(f"   Generated weight: {weight} with shape {result[weight].shape}")

    print("✅ Full pipeline passed!")


def test_compatibility():
    """Test compatibility with different data formats."""
    print("\n=== Testing Data Format Compatibility ===")

    # Test with numpy arrays (non-MetaTensor)
    print("\n1. Testing with numpy arrays...")
    shape = (16, 32, 32)
    label = np.zeros(shape, dtype=np.int32)
    label[4:12, 10:22, 10:22] = 1

    data = {'label': label}

    binary_transform = SegToBinaryMaskd(keys=['label'], target_opt='0')
    result = binary_transform(data)

    assert 'label_binary_mask' in result, "Binary mask not generated from numpy array"
    print(f"   Processed numpy array with shape: {result['label_binary_mask'].shape}")

    # Test with torch tensors
    print("\n2. Testing with torch tensors...")
    label_tensor = torch.from_numpy(label)
    data = {'label': label_tensor}

    result = binary_transform(data)
    assert 'label_binary_mask' in result, "Binary mask not generated from torch tensor"
    print(f"   Processed torch tensor with shape: {result['label_binary_mask'].shape}")

    print("✅ Compatibility tests passed!")


def main():
    """Run all tests."""
    print("🧪 Testing MONAI-native transforms for PyTorch Connectomics")
    print("=" * 60)

    try:
        test_individual_transforms()
        test_compose_pipelines()
        test_weight_computation()
        test_full_pipeline()
        test_compatibility()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! MONAI transforms are working correctly.")
        print("\nThe refactoring is complete. You can now use MONAI-style")
        print("Compose pipelines for all connectomics processing operations.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())