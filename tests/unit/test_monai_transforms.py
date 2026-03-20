#!/usr/bin/env python3
"""
Test script for MONAI-native transforms in PyTorch Connectomics.

This script validates that the new MONAI transforms work correctly and produce
expected outputs for common connectomics processing tasks.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.config.schema import LabelTargetConfig, LabelTransformConfig  # noqa: E402

# Import the new MONAI transforms
from connectomics.data.processing import (  # noqa: E402
    MultiTaskLabelTransformd,
    SegErosionInstanced,
    SegToAffinityMapd,
    SegToBinaryMaskd,
    SegToFlowFieldd,
    SegToInstanceBoundaryMaskd,
    SegToInstanceEDTd,
    create_label_transform_pipeline,
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
        "image": MetaTensor(image[None, ...]),  # Add channel dimension
        "label": MetaTensor(label),
        "valid_mask": MetaTensor(valid_mask),
    }

    print(f"Created test data with shape: {shape}")
    print(f"Label contains {len(np.unique(label))} unique values: {np.unique(label)}")

    return data


def test_individual_transforms():
    """Test individual MONAI transforms (in-place modifications)."""
    print("\n=== Testing Individual MONAI Transforms ===")
    print("   NOTE: Individual transforms modify keys in-place")

    data = create_test_data()

    # Test binary mask transform (modifies 'label' in-place)
    print("\n1. Testing SegToBinaryMaskd...")
    binary_transform = SegToBinaryMaskd(keys=["label"])
    result = binary_transform(data.copy())

    assert "label" in result, "Label key missing"
    binary_mask = result["label"]
    print(f"   Binary mask shape: {binary_mask.shape}")
    print(f"   Binary mask dtype: {binary_mask.dtype}")

    # Test affinity transform (modifies 'label' in-place)
    print("\n2. Testing SegToAffinityMapd...")
    affinity_transform = SegToAffinityMapd(keys=["label"], offsets=["1-0-0", "0-1-0", "0-0-1"])
    result = affinity_transform(create_test_data())

    assert "label" in result, "Label key missing"
    affinity = result["label"]
    print(f"   Affinity map shape: {affinity.shape}")

    # Test instance boundary transform (modifies 'label' in-place)
    print("\n3. Testing SegToInstanceBoundaryMaskd (3D mode)...")
    boundary_transform = SegToInstanceBoundaryMaskd(
        keys=["label"], thickness=1, edge_mode="all", mode="3d"
    )
    result = boundary_transform(create_test_data())

    assert "label" in result, "Label key missing"
    boundary = result["label"]
    print(f"   3D boundary mask shape: {boundary.shape}")

    # Test 2D mode as well
    print("\n4. Testing SegToInstanceBoundaryMaskd (2D mode)...")
    boundary_transform_2d = SegToInstanceBoundaryMaskd(
        keys=["label"], thickness=1, edge_mode="all", mode="2d"
    )
    result = boundary_transform_2d(create_test_data())

    assert "label" in result, "Label key missing"
    boundary_2d = result["label"]
    print(f"   2D boundary mask shape: {boundary_2d.shape}")

    # Test instance EDT transform (modifies 'label' in-place)
    print("\n5. Testing SegToInstanceEDTd...")
    edt_transform = SegToInstanceEDTd(keys=["label"], mode="2d", quantize=False)
    result = edt_transform(create_test_data())

    assert "label" in result, "Label key missing"
    distance = result["label"]
    print(f"   Distance transform shape: {distance.shape}")

    # Test flow field transform (modifies 'label' in-place)
    print("\n6. Testing SegToFlowFieldd...")
    flow_transform = SegToFlowFieldd(keys=["label"])
    result = flow_transform(create_test_data())

    assert "label" in result, "Label key missing"
    flow = result["label"]
    print(f"   Flow field shape: {flow.shape}")

    print("[OK]All individual transforms passed!")


def test_compose_pipelines():
    """Test MONAI Compose pipelines."""
    print("\n=== Testing MONAI Compose Pipelines ===")

    data = create_test_data()

    # Test label transform pipeline with binary target
    print("\n2. Testing label transform pipeline (binary)...")

    binary_cfg = {
        "keys": ["label"],
        "targets": [{"name": "binary"}],
    }
    binary_pipeline = create_label_transform_pipeline(binary_cfg)
    result = binary_pipeline(data)
    assert "label" in result, "Binary mask not generated"
    print(f"   Binary mask shape: {result['label'].shape}")

    # Test label transform pipeline with affinity target
    print("\n3. Testing label transform pipeline (affinity)...")
    affinity_cfg = {
        "keys": ["label"],
        "targets": [
            {
                "name": "affinity",
                "kwargs": {"offsets": ["1-0-0", "0-1-0", "0-0-1"]},
            }
        ],
    }
    affinity_pipeline = create_label_transform_pipeline(affinity_cfg)
    result = affinity_pipeline(data)
    assert "label" in result, "Affinity map not generated"
    print(f"   Affinity map shape: {result['label'].shape}")

    # Test label transform pipeline with multi-task instance segmentation
    print("\n4. Testing label transform pipeline (instance multi-task)...")
    instance_cfg = {
        "keys": ["label"],
        "targets": [
            {"name": "binary"},
            {"name": "instance_boundary", "kwargs": {"thickness": 1, "edge_mode": "seg-all"}},
            {"name": "instance_edt", "kwargs": {"mode": "2d", "quantize": False}},
        ],
    }
    instance_pipeline = create_label_transform_pipeline(instance_cfg)
    result = instance_pipeline(data)
    assert "label" in result, "Multi-task output not generated"
    assert tuple(result["label"].shape) == (3, 32, 64, 64)
    print(f"   Multi-task output shape: {result['label'].shape}")  # Should be [3, D, H, W]

    print("[OK]All label transform pipelines passed!")


def test_affinity_pipeline_ignores_deepem_crop_control_kwarg():
    data = create_test_data()
    affinity_cfg = {
        "keys": ["label"],
        "targets": [
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                    "deepem_crop": True,
                },
            }
        ],
    }

    affinity_pipeline = create_label_transform_pipeline(affinity_cfg)
    result = affinity_pipeline(data)

    assert "label" in result
    assert tuple(result["label"].shape) == (3, 32, 64, 64)


def test_label_transform_pipeline_accepts_structured_config_dataclass():
    data = create_test_data()
    cfg = LabelTransformConfig(
        keys=["label"],
        targets=[
            LabelTargetConfig(
                name="affinity",
                kwargs={"offsets": ["1-0-0", "0-1-0", "0-0-1"]},
            )
        ],
    )

    pipeline = create_label_transform_pipeline(cfg)
    result = pipeline(data)

    assert "label" in result
    assert tuple(result["label"].shape) == (3, 32, 64, 64)


def test_multitask_label_transform_raw_3d_input_stacks_channels_correctly():
    label = np.zeros((6, 20, 20), dtype=np.int32)
    label[1:5, 4:16, 4:16] = 1
    label[2:6, 10:18, 10:18] = 2

    transform = MultiTaskLabelTransformd(
        keys=["label"],
        tasks=[
            {"name": "affinity", "kwargs": {"offsets": ["1-0-0", "0-1-0", "0-0-1"]}},
            {"name": "instance_edt", "kwargs": {"mode": "3d"}},
        ],
    )

    result = transform({"label": label})

    assert tuple(result["label"].shape) == (4, 6, 20, 20)


def test_seg_erosion_instanced_preserves_single_channel_layout():
    label = np.zeros((1, 6, 20, 20), dtype=np.int32)
    label[:, 1:5, 4:16, 4:16] = 1
    label[:, 2:6, 10:18, 10:18] = 2

    transform = SegErosionInstanced(keys=["label"], tsz_h=1)
    result = transform({"label": label})

    assert tuple(result["label"].shape) == (1, 6, 20, 20)


def test_weight_computation():
    """Test weight computation transforms."""
    print("\n=== Testing Weight Computation ===")
    print("   NOTE: This test is deprecated. Weight computation is now handled by loss functions.")
    print("   [WARN]SKIPPED")
    # Weight computation has been integrated into loss functions
    # The ComputeBinaryRatioWeightd API has changed
    return


def test_full_pipeline():
    """Test complete processing pipeline.

    NOTE: This test is deprecated. The create_full_processing_pipeline function
    no longer exists. Use create_label_transform_pipeline instead.
    """
    print("\n=== Testing Full Processing Pipeline (DEPRECATED - SKIPPED) ===")
    print("   [WARN]This test is deprecated. Use test_compose_pipelines() instead.")
    # Test skipped - function no longer exists
    return

    print("[OK]Full pipeline passed!")


def test_compatibility():
    """Test compatibility with different data formats."""
    print("\n=== Testing Data Format Compatibility ===")

    # Test with numpy arrays (non-MetaTensor)
    print("\n1. Testing with numpy arrays...")
    shape = (16, 32, 32)
    label = np.zeros(shape, dtype=np.int32)
    label[4:12, 10:22, 10:22] = 1

    data = {"label": label}

    # Use pipeline API instead of individual transforms
    binary_cfg = {
        "keys": ["label"],
        "targets": [{"name": "binary"}],
    }
    binary_pipeline = create_label_transform_pipeline(binary_cfg)
    result = binary_pipeline(data)

    assert "label" in result, "Binary mask not generated from numpy array"
    print(f"   Processed numpy array with shape: {result['label'].shape}")

    # Test with torch tensors
    print("\n2. Testing with torch tensors...")
    label_tensor = torch.from_numpy(label)
    data = {"label": label_tensor}

    result = binary_pipeline(data)
    assert "label" in result, "Binary mask not generated from torch tensor"
    print(f"   Processed torch tensor with shape: {result['label'].shape}")

    print("[OK]Compatibility tests passed!")


def main():
    """Run all tests."""
    print("Testing MONAI-native transforms for PyTorch Connectomics")
    print("=" * 60)

    try:
        test_individual_transforms()
        test_compose_pipelines()
        test_weight_computation()
        test_full_pipeline()
        test_compatibility()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! MONAI transforms are working correctly.")
        print("\nThe refactoring is complete. You can now use MONAI-style")
        print("Compose pipelines for all connectomics processing operations.")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
