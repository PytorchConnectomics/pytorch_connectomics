#!/usr/bin/env python3
"""Simple integration test for config system."""

import pytest
from connectomics.config import load_config, Config, from_dict
from connectomics.training.lit import ConnectomicsModule, create_trainer


def test_config_creation():
    """Test basic config creation."""
    cfg = Config()
    assert cfg is not None
    assert hasattr(cfg, 'system')
    assert hasattr(cfg, 'model')


def test_config_from_dict():
    """Test creating config from dict."""
    cfg = from_dict({
        'system': {'num_gpus': 0},
        'model': {'architecture': 'monai_basic_unet3d'}
    })
    assert cfg.system.num_gpus == 0
    assert cfg.model.architecture == 'monai_basic_unet3d'


def test_config_from_yaml():
    """Test loading config from YAML."""
    try:
        cfg = load_config('tutorials/lucchi.yaml')
        assert cfg is not None
    except FileNotFoundError:
        pytest.skip("Example config not found")


def test_lightning_module_creation():
    """Test creating Lightning module."""
    cfg = from_dict({
        'system': {'num_gpus': 0},
        'model': {
            'architecture': 'monai_basic_unet3d',
            'in_channels': 1,
            'out_channels': 2,
            'filters': [8, 16],
            'loss_functions': ['DiceLoss'],
            'loss_weights': [1.0]
        },
        'optimizer': {'name': 'AdamW', 'lr': 1e-4},
        'training': {'max_epochs': 1}
    })
    
    module = ConnectomicsModule(cfg)
    assert module is not None


def test_trainer_creation():
    """Test creating trainer."""
    cfg = from_dict({
        'system': {'num_gpus': 0},
        'training': {'max_epochs': 1}
    })
    
    trainer = create_trainer(cfg)
    assert trainer is not None
    assert trainer.max_epochs == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
