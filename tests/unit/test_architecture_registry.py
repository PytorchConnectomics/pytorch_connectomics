"""
Tests for architecture registry system.

Tests:
- Architecture registration and discovery
- Error handling for missing architectures
- Registry utilities (list, check availability, etc.)
- Base model interface
"""

import pytest
import torch
import torch.nn as nn
from connectomics.models.architectures import (
    register_architecture,
    get_architecture_builder,
    list_architectures,
    is_architecture_available,
    unregister_architecture,
    get_architecture_info,
    ConnectomicsModel,
)


class DummyModel(ConnectomicsModel):
    """Dummy model for testing."""

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.supports_deep_supervision = False
        self.output_scales = 1

    def forward(self, x):
        return self.conv(x)


class DummyDeepSupervisionModel(ConnectomicsModel):
    """Dummy model with deep supervision for testing."""

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.supports_deep_supervision = True
        self.output_scales = 2

    def forward(self, x):
        output = self.conv1(x)
        ds_1 = self.conv2(x)
        return {'output': output, 'ds_1': ds_1}


def test_register_architecture():
    """Test architecture registration."""

    @register_architecture('test_model')
    def build_test_model(cfg):
        return DummyModel()

    assert is_architecture_available('test_model')
    assert 'test_model' in list_architectures()

    # Cleanup
    unregister_architecture('test_model')


def test_get_architecture_builder():
    """Test getting architecture builder."""

    @register_architecture('test_model_2')
    def build_test_model(cfg):
        return DummyModel()

    builder = get_architecture_builder('test_model_2')
    assert callable(builder)

    model = builder(None)
    assert isinstance(model, DummyModel)

    # Cleanup
    unregister_architecture('test_model_2')


def test_missing_architecture():
    """Test error handling for missing architecture."""

    with pytest.raises(ValueError, match="not found"):
        get_architecture_builder('nonexistent_model')


def test_list_architectures():
    """Test listing all architectures."""

    # Should include MONAI models registered at import time
    archs = list_architectures()
    assert isinstance(archs, list)
    assert len(archs) > 0  # Should have at least MONAI models

    # Check that MONAI models are registered
    monai_archs = [a for a in archs if a.startswith('monai_')]
    assert len(monai_archs) > 0


def test_is_architecture_available():
    """Test checking architecture availability."""

    # MONAI models should be available
    assert is_architecture_available('monai_basic_unet3d')

    # Non-existent model should not be available
    assert not is_architecture_available('fake_model_xyz')


def test_architecture_info():
    """Test getting architecture information."""

    @register_architecture('test_model_3')
    def build_test_model(cfg):
        """Test model builder."""
        return DummyModel()

    info = get_architecture_info()
    assert isinstance(info, dict)
    assert 'test_model_3' in info
    assert info['test_model_3']['name'] == 'test_model_3'
    assert 'Test model builder' in info['test_model_3']['doc']

    # Cleanup
    unregister_architecture('test_model_3')


def test_unregister_architecture():
    """Test unregistering architecture."""

    @register_architecture('test_model_4')
    def build_test_model(cfg):
        return DummyModel()

    assert is_architecture_available('test_model_4')

    unregister_architecture('test_model_4')
    assert not is_architecture_available('test_model_4')

    # Trying to unregister again should raise error
    with pytest.raises(ValueError, match="not registered"):
        unregister_architecture('test_model_4')


def test_overwrite_registration_warning():
    """Test warning when overwriting registration."""

    @register_architecture('test_model_5')
    def build_test_model_v1(cfg):
        return DummyModel()

    # Re-registering should produce a warning
    with pytest.warns(UserWarning, match="already registered"):
        @register_architecture('test_model_5')
        def build_test_model_v2(cfg):
            return DummyModel()

    # Cleanup
    unregister_architecture('test_model_5')


def test_base_model_interface():
    """Test ConnectomicsModel base interface."""

    model = DummyModel(in_channels=1, out_channels=3)

    # Test model info
    info = model.get_model_info()
    assert info['name'] == 'DummyModel'
    assert info['deep_supervision'] == False
    assert info['output_scales'] == 1
    assert info['parameters'] > 0
    assert info['trainable_parameters'] > 0

    # Test summary
    summary = model.summary(input_shape=(1, 1, 32, 32, 32))
    assert 'DummyModel' in summary
    assert 'Parameters' in summary

    # Test repr
    repr_str = repr(model)
    assert 'DummyModel' in repr_str


def test_base_model_forward():
    """Test forward pass through base model."""

    model = DummyModel(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 32, 32, 32)

    output = model(x)
    assert output.shape == (1, 2, 32, 32, 32)


def test_deep_supervision_model():
    """Test model with deep supervision."""

    model = DummyDeepSupervisionModel(in_channels=1, out_channels=2)

    # Check model info
    info = model.get_model_info()
    assert info['deep_supervision'] == True
    assert info['output_scales'] == 2

    # Test forward pass
    x = torch.randn(1, 1, 32, 32, 32)
    outputs = model(x)

    assert isinstance(outputs, dict)
    assert 'output' in outputs
    assert 'ds_1' in outputs
    assert outputs['output'].shape == (1, 2, 32, 32, 32)
    assert outputs['ds_1'].shape == (1, 2, 16, 16, 16)  # 2x downsampled


def test_monai_models_registered():
    """Test that MONAI models are registered at import."""

    expected_monai_models = [
        'monai_basic_unet3d',
        'monai_unet',
        'monai_unetr',
        'monai_swin_unetr',
    ]

    for model_name in expected_monai_models:
        assert is_architecture_available(model_name), f"{model_name} should be registered"

    # Test we can get builders for them
    for model_name in expected_monai_models:
        builder = get_architecture_builder(model_name)
        assert callable(builder)