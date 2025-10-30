"""
Basic tests for architecture registry (no pytest required).
"""

import sys
import torch
from omegaconf import OmegaConf

from connectomics.models.arch import (
    list_architectures,
    is_architecture_available,
    get_architecture_builder,
    ConnectomicsModel,
)


def test_monai_models_registered():
    """Test that MONAI models are registered."""
    print("Testing MONAI model registration...")

    expected_models = [
        'monai_basic_unet3d',
        'monai_unet',
        'monai_unetr',
        'monai_swin_unetr',
    ]

    for model_name in expected_models:
        assert is_architecture_available(model_name), f"{model_name} not registered!"
        print(f"  ✓ {model_name} is registered")

    print("  All MONAI models registered successfully!\n")


def test_list_architectures():
    """Test listing all architectures."""
    print("Testing list_architectures()...")

    archs = list_architectures()
    print(f"  Found {len(archs)} architectures")
    print(f"  Architectures: {archs}\n")

    # Should have 4 MONAI + 2 MedNeXt = 6 total
    assert len(archs) >= 4, f"Expected at least 4 architectures (MONAI), got {len(archs)}"

    # Count by type
    monai_count = sum(1 for a in archs if a.startswith('monai_'))
    mednext_count = sum(1 for a in archs if a.startswith('mednext'))
    print(f"  ✓ Found {monai_count} MONAI models")
    print(f"  ✓ Found {mednext_count} MedNeXt models")
    print(f"  ✓ Total: {len(archs)} architectures\n")


def test_get_builder():
    """Test getting architecture builder."""
    print("Testing get_architecture_builder()...")

    builder = get_architecture_builder('monai_basic_unet3d')
    assert callable(builder), "Builder should be callable"
    print("  ✓ Builder is callable\n")


def test_build_model():
    """Test building a model."""
    print("Testing building a model...")

    # Create minimal config
    cfg = OmegaConf.create({
        'model': {
            'architecture': 'monai_basic_unet3d',
            'in_channels': 1,
            'out_channels': 2,
            'filters': [32, 64, 128, 256, 512],
            'dropout': 0.0,
            'activation': 'relu',
            'norm': 'batch',
        }
    })

    builder = get_architecture_builder('monai_basic_unet3d')
    model = builder(cfg)

    assert isinstance(model, ConnectomicsModel), "Model should inherit from ConnectomicsModel"
    print("  ✓ Model inherits from ConnectomicsModel")

    # Check model info
    info = model.get_model_info()
    print(f"  Model: {info['name']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Deep Supervision: {info['deep_supervision']}")

    assert info['parameters'] > 0, "Model should have parameters"
    print("  ✓ Model has parameters\n")


def test_forward_pass():
    """Test forward pass through model."""
    print("Testing forward pass...")

    # Create minimal config
    cfg = OmegaConf.create({
        'model': {
            'architecture': 'monai_basic_unet3d',
            'in_channels': 1,
            'out_channels': 2,
            'filters': [16, 32, 64, 128, 256],  # Smaller for fast test
        }
    })

    builder = get_architecture_builder('monai_basic_unet3d')
    model = builder(cfg)

    # Small input for fast test
    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 2, 32, 32, 32), f"Expected shape (1, 2, 32, 32, 32), got {output.shape}"
    print(f"  ✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}\n")


def test_missing_architecture():
    """Test error handling for missing architecture."""
    print("Testing missing architecture error handling...")

    try:
        get_architecture_builder('nonexistent_model')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e), "Error message should mention 'not found'"
        print("  ✓ Correctly raises ValueError for missing architecture\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Architecture Registry System")
    print("=" * 60 + "\n")

    try:
        test_monai_models_registered()
        test_list_architectures()
        test_get_builder()
        test_build_model()
        test_forward_pass()
        test_missing_architecture()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())