"""
Example: 80/20 Train/Val Split with Automatic Padding

This example demonstrates how to use PyTorch Connectomics' volumetric split
utilities, inspired by DeepEM's approach of spatial train/val splitting.

The approach:
1. Use first 80% of volume (along Z-axis) for training
2. Use last 20% for validation
3. Automatically pad validation data to model input size if needed
"""

import numpy as np
import torch
from connectomics.data.utils.split import (
    split_volume_train_val,
    create_split_masks,
    pad_volume_to_size,
    split_and_pad_volume,
    save_split_masks_h5
)


def example_basic_split():
    """Example 1: Basic 80/20 split"""
    print("=" * 60)
    print("Example 1: Basic 80/20 Split")
    print("=" * 60)

    # Simulate a volume
    volume_shape = (100, 256, 256)  # [D, H, W]
    print(f"Volume shape: {volume_shape}")

    # Get split slices
    train_slices, val_slices = split_volume_train_val(volume_shape, train_ratio=0.8)

    print(f"Train slices: {train_slices}")
    print(f"Val slices: {val_slices}")

    # Apply to actual volume
    volume = np.random.randn(*volume_shape)
    train_data = volume[train_slices]  # Shape: (80, 256, 256)
    val_data = volume[val_slices]      # Shape: (20, 256, 256)

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print()


def example_split_masks():
    """Example 2: Create binary masks (DeepEM style)"""
    print("=" * 60)
    print("Example 2: Binary Masks for Train/Val")
    print("=" * 60)

    volume_shape = (100, 256, 256)
    train_mask, val_mask = create_split_masks(volume_shape, train_ratio=0.8)

    print(f"Train mask shape: {train_mask.shape}")
    print(f"Val mask shape: {val_mask.shape}")
    print(f"Train region voxels: {train_mask.sum():,}")
    print(f"Val region voxels: {val_mask.sum():,}")
    print(f"Total voxels: {(train_mask.sum() + val_mask.sum()):,}")
    print()


def example_padding_validation():
    """Example 3: Pad validation data to model input size"""
    print("=" * 60)
    print("Example 3: Padding Validation Data")
    print("=" * 60)

    # Small validation volume (last 20 slices)
    val_volume = np.random.randn(20, 256, 256)
    print(f"Original val volume: {val_volume.shape}")

    # Model requires minimum 32 slices
    model_input_size = (32, 256, 256)

    # Pad validation volume
    val_padded = pad_volume_to_size(val_volume, model_input_size, mode='reflect')
    print(f"Padded val volume: {val_padded.shape}")
    print(f"Padding mode: reflect (mirrors edges)")
    print()


def example_split_and_pad():
    """Example 4: Combined split and pad (recommended)"""
    print("=" * 60)
    print("Example 4: Split + Pad in One Step")
    print("=" * 60)

    # Full volume
    volume = np.random.randn(100, 256, 256)
    model_input_size = (32, 256, 256)

    # Split 80/20 and pad validation
    train_vol, val_vol = split_and_pad_volume(
        volume,
        train_ratio=0.8,
        target_size=model_input_size,
        pad_mode='reflect'
    )

    print(f"Original volume: {volume.shape}")
    print(f"Train volume: {train_vol.shape}")
    print(f"Val volume (padded): {val_vol.shape}")
    print()


def example_channel_dimension():
    """Example 5: Handle volumes with channel dimension"""
    print("=" * 60)
    print("Example 5: Volumes with Channel Dimension")
    print("=" * 60)

    # Volume with channel: [C, D, H, W]
    volume = np.random.randn(1, 100, 256, 256)
    model_input_size = (32, 256, 256)

    train_vol, val_vol = split_and_pad_volume(
        volume,
        train_ratio=0.8,
        target_size=model_input_size
    )

    print(f"Original volume: {volume.shape}")
    print(f"Train volume: {train_vol.shape}")
    print(f"Val volume (padded): {val_vol.shape}")
    print()


def example_custom_axis():
    """Example 6: Split along different axis"""
    print("=" * 60)
    print("Example 6: Split Along Different Axis")
    print("=" * 60)

    volume_shape = (100, 256, 256)

    # Split along Z-axis (default, axis=0)
    train_z, val_z = split_volume_train_val(volume_shape, train_ratio=0.8, axis=0)
    print(f"Split along Z (axis=0):")
    print(f"  Train: {train_z}")
    print(f"  Val: {val_z}")

    # Split along Y-axis (axis=1)
    train_y, val_y = split_volume_train_val(volume_shape, train_ratio=0.8, axis=1)
    print(f"Split along Y (axis=1):")
    print(f"  Train: {train_y}")
    print(f"  Val: {val_y}")

    # Split along X-axis (axis=2)
    train_x, val_x = split_volume_train_val(volume_shape, train_ratio=0.8, axis=2)
    print(f"Split along X (axis=2):")
    print(f"  Train: {train_x}")
    print(f"  Val: {val_x}")
    print()


def example_minimum_val_size():
    """Example 7: Ensure minimum validation size"""
    print("=" * 60)
    print("Example 7: Minimum Validation Size")
    print("=" * 60)

    volume_shape = (50, 256, 256)  # Small volume

    # Normal 80/20 split would give 10 slices for val
    # But we want at least 20 slices
    train_slices, val_slices = split_volume_train_val(
        volume_shape,
        train_ratio=0.8,
        min_val_size=20  # Ensure at least 20 slices for validation
    )

    print(f"Volume shape: {volume_shape}")
    print(f"Requested train_ratio: 0.8")
    print(f"Minimum val size: 20")
    print(f"Actual train slices: {train_slices[0]}")
    print(f"Actual val slices: {val_slices[0]}")
    print()


def example_save_masks():
    """Example 8: Save masks to HDF5 (DeepEM compatible)"""
    print("=" * 60)
    print("Example 8: Save Masks to HDF5")
    print("=" * 60)

    volume_shape = (100, 256, 256)
    output_dir = '/tmp/connectomics_splits'

    # Save masks
    save_split_masks_h5(
        output_dir=output_dir,
        volume_shape=volume_shape,
        train_ratio=0.8,
        train_filename='msk_train.h5',
        val_filename='msk_val.h5'
    )
    print()


def example_torch_tensors():
    """Example 9: Works with PyTorch tensors too"""
    print("=" * 60)
    print("Example 9: PyTorch Tensor Support")
    print("=" * 60)

    # Create PyTorch tensor
    volume = torch.randn(100, 256, 256)
    model_input_size = (32, 256, 256)

    # Split and pad (works seamlessly with tensors)
    train_vol, val_vol = split_and_pad_volume(
        volume,
        train_ratio=0.8,
        target_size=model_input_size,
        pad_mode='reflect'
    )

    print(f"Input type: {type(volume)}")
    print(f"Train volume: {train_vol.shape}, type: {type(train_vol)}")
    print(f"Val volume: {val_vol.shape}, type: {type(val_vol)}")
    print(f"Both outputs maintain PyTorch tensor type!")
    print()


def example_practical_use_case():
    """Example 10: Practical training workflow"""
    print("=" * 60)
    print("Example 10: Practical Training Workflow")
    print("=" * 60)

    print("""
    Typical workflow for training with 80/20 split:

    1. Load full volume
    2. Split into train/val (80/20 along Z-axis)
    3. Pad validation to model input size if needed
    4. Use train portion for training with random crops
    5. Use val portion for validation (with padding)

    Code example:
    """)

    code = '''
    from connectomics.data.utils.split import split_and_pad_volume
    from connectomics.data.io import read_volume

    # Load volume
    volume = read_volume('path/to/volume.h5')
    label = read_volume('path/to/label.h5')

    # Split 80/20 with automatic padding
    model_input = (32, 256, 256)

    train_img, val_img = split_and_pad_volume(
        volume, train_ratio=0.8, target_size=model_input
    )
    train_label, val_label = split_and_pad_volume(
        label, train_ratio=0.8, target_size=model_input
    )

    # Training: use random crops from train portion
    # Validation: use padded val portion for inference
    '''

    print(code)


if __name__ == "__main__":
    # Run all examples
    example_basic_split()
    example_split_masks()
    example_padding_validation()
    example_split_and_pad()
    example_channel_dimension()
    example_custom_axis()
    example_minimum_val_size()
    example_save_masks()
    example_torch_tensors()
    example_practical_use_case()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
