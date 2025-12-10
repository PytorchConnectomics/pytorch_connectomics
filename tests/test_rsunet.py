"""
Unit tests for RSUNet architecture.

Tests various configurations:
- Anisotropic and isotropic
- Different normalization types
- Different activations
- 2D/3D hybrid
- Deep supervision
"""

import torch


def test_rsunet_basic():
    """Test basic RSUNet forward pass."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64],
    )

    x = torch.randn(2, 1, 32, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 32, 64, 64)
    print("✓ Basic RSUNet test passed")


def test_rsunet_isotropic():
    """Test RSUNet with isotropic downsampling."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64, 128],
        down_factors=[(2, 2, 2)] * 3,  # Isotropic
    )

    x = torch.randn(1, 1, 64, 128, 128)
    y = model(x)

    assert y.shape == (1, 2, 64, 128, 128)
    print("✓ Isotropic RSUNet test passed")


def test_rsunet_group_norm():
    """Test RSUNet with GroupNorm."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64],
        norm='group',
        num_groups=8,
    )

    x = torch.randn(2, 1, 32, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 32, 64, 64)
    print("✓ GroupNorm RSUNet test passed")


def test_rsunet_prelu():
    """Test RSUNet with PReLU activation."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64],
        activation='prelu',
        init=0.1,
    )

    x = torch.randn(2, 1, 32, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 32, 64, 64)
    print("✓ PReLU RSUNet test passed")


def test_rsunet_2d3d_hybrid():
    """Test RSUNet with 2D/3D hybrid convolutions."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64, 128],
        depth_2d=2,  # First 2 layers use 2D
        kernel_2d=(1, 3, 3),
    )

    x = torch.randn(1, 1, 32, 64, 64)
    y = model(x)

    assert y.shape == (1, 2, 32, 64, 64)
    print("✓ 2D/3D Hybrid RSUNet test passed")


def test_rsunet_deep_supervision():
    """Test RSUNet with deep supervision."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64, 128, 256],
        deep_supervision=True,
    )

    x = torch.randn(1, 1, 32, 64, 64)
    outputs = model(x)

    assert isinstance(outputs, dict)
    assert 'output' in outputs
    assert outputs['output'].shape == (1, 2, 32, 64, 64)
    # Should have deep supervision outputs
    assert 'ds_0' in outputs or 'ds_1' in outputs
    print("✓ Deep supervision RSUNet test passed")


def test_rsunet_model_info():
    """Test model info retrieval."""
    from connectomics.models.arch.rsunet import RSUNet

    model = RSUNet(
        in_channels=1,
        out_channels=2,
        width=[16, 32, 64],
    )

    info = model.get_model_info()

    assert 'name' in info
    assert 'parameters' in info
    assert info['parameters'] > 0
    print(f"✓ Model info test passed: {info['parameters']:,} parameters")


def test_rsunet_registry():
    """Test that RSUNet is registered."""
    from connectomics.models.arch import list_architectures

    archs = list_architectures()

    assert 'rsunet' in archs
    assert 'rsunet_iso' in archs
    print(f"✓ Registry test passed: {len(archs)} architectures available")


if __name__ == '__main__':
    print("Running RSUNet tests...\n")

    test_rsunet_basic()
    test_rsunet_isotropic()
    test_rsunet_group_norm()
    test_rsunet_prelu()
    test_rsunet_2d3d_hybrid()
    test_rsunet_deep_supervision()
    test_rsunet_model_info()
    test_rsunet_registry()

    print("\n✅ All RSUNet tests passed!")
