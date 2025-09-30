#!/usr/bin/env python3
"""
MONAI-style Transform Usage Examples for PyTorch Connectomics

This script demonstrates how to use the new MONAI-native transforms for
various connectomics processing tasks. The refactored system provides:

1. Native MONAI transforms following MONAI conventions
2. MONAI Compose pipelines for easy chaining
3. Factory functions for common workflows
4. Backward compatibility with existing configs

Author: PyTorch Connectomics Team
"""

import numpy as np
from monai.data import MetaTensor
from monai.transforms import Compose

# Import the new MONAI-native transforms
from connectomics.data.process import (
    # Individual transforms
    SegToBinaryMaskd,
    SegToAffinityMapd,
    SegToInstanceBoundaryMaskd,
    SegToInstanceEDTd,
    SegToFlowFieldd,
    ComputeBinaryRatioWeightd,

    # Factory functions for common workflows
    create_binary_segmentation_pipeline,
    create_affinity_segmentation_pipeline,
    create_instance_segmentation_pipeline,
    create_multi_task_pipeline,
    create_full_processing_pipeline,

    # Custom composition functions
    create_target_transforms,
    create_weight_transforms,
)


def create_sample_data():
    """Create synthetic 3D connectomics data for demonstration."""
    print("📦 Creating sample 3D connectomics data...")

    # Create a 3D volume with multiple objects
    shape = (16, 64, 64)
    label = np.zeros(shape, dtype=np.int32)

    # Add some rectangular objects (simulating mitochondria, etc.)
    label[4:8, 16:32, 16:32] = 1      # Object 1
    label[8:12, 32:48, 20:36] = 2     # Object 2
    label[10:14, 16:32, 40:56] = 3    # Object 3
    label[2:6, 48:60, 8:24] = 4       # Object 4

    # Create corresponding image (just noise for this example)
    image = np.random.rand(*shape).astype(np.float32)

    # Package in MONAI format
    data = {
        'image': MetaTensor(image[None, ...]),  # Add channel dimension
        'label': MetaTensor(label),
        'valid_mask': MetaTensor(np.ones(shape, dtype=np.uint8))
    }

    print(f"   Created volume with shape: {shape}")
    print(f"   Label contains {len(np.unique(label))} objects: {np.unique(label)}")

    return data


def example_individual_transforms():
    """Demonstrate individual MONAI transforms."""
    print("\n🔧 Example 1: Individual MONAI Transforms")
    print("=" * 50)

    data = create_sample_data()

    # Binary mask generation
    print("\n1️⃣ Binary Mask Generation:")
    binary_transform = SegToBinaryMaskd(keys=['label'], target_opt='0')
    result = binary_transform(data)

    print(f"   Input label shape: {data['label'].shape}")
    print(f"   Binary mask shape: {result['label_binary_mask'].shape}")
    print(f"   Binary mask values: {np.unique(result['label_binary_mask'].array)}")

    # Affinity map generation
    print("\n2️⃣ Affinity Map Generation:")
    affinity_transform = SegToAffinityMapd(keys=['label'], long_range=5)
    result = affinity_transform(data)

    print(f"   Affinity map shape: {result['label_affinity'].shape}")
    print(f"   Affinity channels: 6 (3 short-range + 3 long-range)")
    print(f"   Affinity range: [{result['label_affinity'].array.min():.3f}, {result['label_affinity'].array.max():.3f}]")

    # Instance boundary generation
    print("\n3️⃣ Instance Boundary Generation:")
    boundary_transform = SegToInstanceBoundaryMaskd(keys=['label'])
    result = boundary_transform(data)

    print(f"   Boundary mask shape: {result['label_boundary'].shape}")
    print(f"   Boundary values: {np.unique(result['label_boundary'].array)}")

    # Distance transform generation
    print("\n4️⃣ Distance Transform Generation:")
    distance_transform = SegToInstanceEDTd(keys=['label'])
    result = distance_transform(data)

    print(f"   Distance transform shape: {result['label_distance'].shape}")
    print(f"   Distance range: [{result['label_distance'].array.min():.3f}, {result['label_distance'].array.max():.3f}]")

    # Weight computation
    print("\n5️⃣ Weight Computation:")
    # First generate a target
    binary_result = binary_transform(data)
    weight_transform = ComputeBinaryRatioWeightd(
        target_keys=['label_binary_mask'],
        valid_mask_key='valid_mask'
    )
    result = weight_transform(binary_result)

    print(f"   Weight map shape: {result['label_binary_mask_weight'].shape}")
    print(f"   Weight range: [{result['label_binary_mask_weight'].array.min():.3f}, {result['label_binary_mask_weight'].array.max():.3f}]")


def example_compose_pipelines():
    """Demonstrate MONAI Compose pipelines."""
    print("\n🔗 Example 2: MONAI Compose Pipelines")
    print("=" * 50)

    data = create_sample_data()

    # Create custom target pipeline
    print("\n1️⃣ Custom Target Pipeline:")
    target_pipeline = create_target_transforms(
        target_opts=['binary_mask', 'affinity_map', 'boundary_mask'],
        input_key='label'
    )

    result = target_pipeline(data)
    generated_targets = [k for k in result.keys() if 'label_' in k and k != 'label']
    print(f"   Generated {len(generated_targets)} targets:")
    for target in generated_targets:
        print(f"     - {target}: {result[target].shape}")

    # Custom weight pipeline
    print("\n2️⃣ Custom Weight Pipeline:")
    target_keys = ['label_binary_mask', 'label_affinity', 'label_boundary']
    weight_pipeline = create_weight_transforms(
        weight_opts=[['binary_ratio'], ['binary_ratio'], ['binary_ratio']],
        target_keys=target_keys,
        valid_mask_key='valid_mask'
    )

    result = weight_pipeline(result)
    generated_weights = [k for k in result.keys() if '_weight' in k]
    print(f"   Generated {len(generated_weights)} weight maps:")
    for weight in generated_weights:
        print(f"     - {weight}: {result[weight].shape}")


def example_factory_functions():
    """Demonstrate factory functions for common workflows."""
    print("\n🏭 Example 3: Factory Functions for Common Workflows")
    print("=" * 50)

    data = create_sample_data()

    # Binary segmentation workflow
    print("\n1️⃣ Binary Segmentation Workflow:")
    binary_pipeline = create_binary_segmentation_pipeline(
        input_key='label',
        use_weights=True,
        weight_type='binary_ratio'
    )

    result = binary_pipeline(data)
    binary_keys = [k for k in result.keys() if 'binary' in k]
    print(f"   Generated outputs: {binary_keys}")

    # Affinity segmentation workflow
    print("\n2️⃣ Affinity Segmentation Workflow:")
    affinity_pipeline = create_affinity_segmentation_pipeline(
        input_key='label',
        use_weights=True
    )

    result = affinity_pipeline(data)
    affinity_keys = [k for k in result.keys() if 'affinity' in k]
    print(f"   Generated outputs: {affinity_keys}")

    # Instance segmentation workflow
    print("\n3️⃣ Instance Segmentation Workflow:")
    instance_pipeline = create_instance_segmentation_pipeline(
        input_key='label',
        use_weights=True
    )

    result = instance_pipeline(data)
    instance_keys = [k for k in result.keys() if k not in ['image', 'label', 'valid_mask']]
    print(f"   Generated outputs: {instance_keys}")

    # Multi-task workflow
    print("\n4️⃣ Multi-Task Learning Workflow:")
    multitask_pipeline = create_multi_task_pipeline(
        target_tasks=['binary_mask', 'affinity_map', 'distance_instance'],
        input_key='label',
        use_weights=True
    )

    result = multitask_pipeline(data)
    multitask_keys = [k for k in result.keys() if k not in ['image', 'label', 'valid_mask']]
    print(f"   Generated {len(multitask_keys)} outputs for multi-task learning:")
    for key in sorted(multitask_keys):
        print(f"     - {key}: {result[key].shape}")


def example_full_pipeline():
    """Demonstrate full processing pipeline with all features."""
    print("\n🚀 Example 4: Full Processing Pipeline")
    print("=" * 50)

    data = create_sample_data()

    # Create comprehensive pipeline
    print("\n🔧 Creating comprehensive processing pipeline...")
    full_pipeline = create_full_processing_pipeline(
        target_opts=[
            'binary_mask',           # Binary segmentation
            'affinity_map',          # Affinity-based segmentation
            'distance_instance',     # Instance distance transform
            'boundary_mask',         # Instance boundaries
            'flow_field'             # Flow fields (cellpose-style)
        ],
        weight_opts=[
            ['binary_ratio'],        # Binary ratio weights for binary mask
            ['binary_ratio'],        # Binary ratio weights for affinity
            ['binary_ratio'],        # Binary ratio weights for distance
            ['binary_ratio'],        # Binary ratio weights for boundary
            ['uniform']              # No weighting for flow fields
        ],
        input_key='label',
        valid_mask_key='valid_mask'
    )

    print("\n⚡ Processing data through full pipeline...")
    result = full_pipeline(data)

    # Analyze results
    all_outputs = [k for k in result.keys() if k not in ['image', 'label', 'valid_mask']]
    targets = [k for k in all_outputs if '_weight' not in k]
    weights = [k for k in all_outputs if '_weight' in k]

    print(f"\n📊 Pipeline Results:")
    print(f"   Generated {len(targets)} targets and {len(weights)} weight maps")
    print(f"\n   🎯 Targets:")
    for target in sorted(targets):
        shape = result[target].shape
        dtype = result[target].array.dtype
        value_range = f"[{result[target].array.min():.3f}, {result[target].array.max():.3f}]"
        print(f"     - {target:25} | Shape: {str(shape):20} | Type: {dtype} | Range: {value_range}")

    print(f"\n   ⚖️  Weights:")
    for weight in sorted(weights):
        if result[weight].array.size > 1:  # Skip empty weight arrays
            shape = result[weight].shape
            dtype = result[weight].array.dtype
            value_range = f"[{result[weight].array.min():.3f}, {result[weight].array.max():.3f}]"
            print(f"     - {weight:25} | Shape: {str(shape):20} | Type: {dtype} | Range: {value_range}")


def example_monai_compatibility():
    """Demonstrate compatibility with MONAI ecosystem."""
    print("\n🤝 Example 5: MONAI Ecosystem Compatibility")
    print("=" * 50)

    data = create_sample_data()

    # Combine with standard MONAI transforms
    print("\n🔗 Combining with standard MONAI transforms...")

    from monai.transforms import (
        EnsureChannelFirstd, ScaleIntensityRanged,
        RandRotate90d, ToTensord
    )

    # Create a pipeline mixing MONAI and connectomics transforms
    mixed_pipeline = Compose([
        # Standard MONAI preprocessing
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        ScaleIntensityRanged(keys=['image'], a_min=0.1, a_max=0.9, b_min=0.0, b_max=1.0),

        # Standard MONAI augmentation
        RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(1, 2)),

        # Our connectomics processing
        SegToBinaryMaskd(keys=['label'], target_opt='0'),
        SegToAffinityMapd(keys=['label'], long_range=3),
        ComputeBinaryRatioWeightd(target_keys=['label_binary_mask']),

        # Standard MONAI final conversion
        ToTensord(keys=['image', 'label_binary_mask', 'label_affinity'])
    ])

    result = mixed_pipeline(data)

    print(f"   Successfully processed with mixed MONAI pipeline!")
    print(f"   Final outputs: {list(result.keys())}")
    print(f"   All outputs are torch tensors: {all(hasattr(v, 'shape') for v in result.values())}")


def main():
    """Run all examples."""
    print("🧬 MONAI-style Transform Examples for PyTorch Connectomics")
    print("=" * 70)
    print()
    print("This script demonstrates the refactored MONAI-native transform system")
    print("that replaces the custom processor approach with standard MONAI patterns.")
    print()

    # Run all examples
    example_individual_transforms()
    example_compose_pipelines()
    example_factory_functions()
    example_full_pipeline()
    example_monai_compatibility()

    print("\n" + "=" * 70)
    print("🎉 All examples completed successfully!")
    print()
    print("📚 Key Benefits of the Refactored System:")
    print("   ✅ Native MONAI transforms following standard conventions")
    print("   ✅ MONAI Compose pipelines for easy chaining")
    print("   ✅ Factory functions for common connectomics workflows")
    print("   ✅ Full compatibility with MONAI ecosystem")
    print("   ✅ Type hints and proper documentation")
    print("   ✅ Cleaner, more maintainable code architecture")
    print()
    print("🔗 You can now use MONAI-style Compose() for all connectomics processing!")


if __name__ == "__main__":
    main()