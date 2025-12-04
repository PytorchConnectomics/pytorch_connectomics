"""
Test the new Hydra-based configuration system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from connectomics.config import (
    Config,
    load_config,
    save_config,
    merge_configs,
    update_from_cli,
    to_dict,
    from_dict,
    print_config,
    validate_config,
    get_config_hash,
    create_experiment_name,
)


def test_default_config_creation():
    """Test creating default config."""
    cfg = Config()
    
    assert cfg.model.architecture == 'monai_basic_unet3d'
    assert cfg.system.training.batch_size == 4
    assert cfg.optimization.optimizer.name == 'AdamW'
    assert cfg.optimization.max_epochs == 100
    print("✅ Default config creation works")


def test_config_validation():
    """Test config validation."""
    cfg = Config()
    
    # Valid config should pass
    try:
        validate_config(cfg)
        print("✅ Valid config passes validation")
    except ValueError as e:
        raise AssertionError(f"Valid config failed validation: {e}")
    
    # Invalid config should fail
    cfg.system.training.batch_size = -1
    try:
        validate_config(cfg)
        raise AssertionError("Invalid config should have failed validation")
    except ValueError:
        print("✅ Invalid config fails validation")


def test_config_dict_conversion():
    """Test converting config to/from dict."""
    cfg = Config()
    cfg.experiment_name = "test_experiment"
    cfg.model.architecture = "custom_unet"
    
    # To dict
    cfg_dict = to_dict(cfg)
    assert isinstance(cfg_dict, dict)
    assert cfg_dict['experiment_name'] == "test_experiment"
    assert cfg_dict['model']['architecture'] == "custom_unet"
    print("✅ Config to dict works")
    
    # From dict
    cfg_restored = from_dict(cfg_dict)
    assert cfg_restored.experiment_name == "test_experiment"
    assert cfg_restored.model.architecture == "custom_unet"
    print("✅ Dict to config works")


def test_config_cli_updates():
    """Test updating config from CLI arguments."""
    cfg = Config()
    
    overrides = [
        'system.training.batch_size=8',
        'model.architecture=unetr',
        'optimization.optimizer.lr=0.001'
    ]
    
    updated_cfg = update_from_cli(cfg, overrides)
    
    assert updated_cfg.system.training.batch_size == 8
    assert updated_cfg.model.architecture == 'unetr'
    assert updated_cfg.optimization.optimizer.lr == 0.001
    print("✅ CLI updates work")


def test_config_merge():
    """Test merging configs."""
    base_cfg = Config()
    base_cfg.experiment_name = "base"
    base_cfg.system.training.batch_size = 2
    
    override_dict = {
        'experiment_name': 'merged',
        'system': {'training': {'batch_size': 4}},
        'model': {'architecture': 'custom'}
    }
    
    merged_cfg = merge_configs(base_cfg, override_dict)
    
    assert merged_cfg.experiment_name == "merged"
    assert merged_cfg.system.training.batch_size == 4
    assert merged_cfg.model.architecture == "custom"
    print("✅ Config merge works")


def test_config_save_load(tmp_path):
    """Test saving and loading config."""
    cfg = Config()
    cfg.experiment_name = "save_test"
    cfg.model.filters = (16, 32, 64)
    
    # Save
    config_path = tmp_path / "test_config.yaml"
    save_config(cfg, config_path)
    assert config_path.exists()
    print("✅ Config save works")
    
    # Load
    loaded_cfg = load_config(config_path)
    assert loaded_cfg.experiment_name == "save_test"
    assert tuple(loaded_cfg.model.filters) == (16, 32, 64)
    print("✅ Config load works")


def test_config_hash():
    """Test config hashing."""
    cfg1 = Config()
    cfg2 = Config()
    
    # Same configs should have same hash
    hash1 = get_config_hash(cfg1)
    hash2 = get_config_hash(cfg2)
    assert hash1 == hash2
    print("✅ Same configs have same hash")
    
    # Different configs should have different hash
    cfg2.system.training.batch_size = 999
    hash3 = get_config_hash(cfg2)
    assert hash1 != hash3
    print("✅ Different configs have different hash")


def test_experiment_name_generation():
    """Test automatic experiment name generation."""
    cfg = Config()
    cfg.model.architecture = "unet3d"
    cfg.system.training.batch_size = 4
    cfg.optimization.optimizer.lr = 0.001
    
    name = create_experiment_name(cfg)
    
    assert "unet3d" in name
    assert "bs4" in name
    assert "1e-03" in name
    assert len(name.split('_')[-1]) == 8  # Hash
    print(f"✅ Generated experiment name: {name}")


def test_augmentation_config():
    """Test augmentation configuration."""
    cfg = Config()
    
    # Enable EM-specific augmentations
    cfg.data.augmentation.misalignment.enabled = True
    cfg.data.augmentation.misalignment.prob = 0.7
    cfg.data.augmentation.missing_section.enabled = True
    cfg.data.augmentation.mixup.enabled = True
    cfg.data.augmentation.copy_paste.enabled = True
    
    assert cfg.data.augmentation.misalignment.enabled
    assert cfg.data.augmentation.misalignment.prob == 0.7
    assert cfg.data.augmentation.missing_section.enabled
    assert cfg.data.augmentation.mixup.enabled
    assert cfg.data.augmentation.copy_paste.enabled
    print("✅ Augmentation config works")


def test_load_example_configs():
    """Test loading example configs."""
    configs_dir = project_root / "configs" / "hydra"
    
    # Test default config
    default_config = configs_dir / "default.yaml"
    if default_config.exists():
        cfg = load_config(default_config)
        assert cfg.experiment_name == "connectomics_default"
        validate_config(cfg)
        print("✅ Default config loads and validates")
    
    # Test Lucchi config
    lucchi_config = configs_dir / "lucchi.yaml"
    if lucchi_config.exists():
        cfg = load_config(lucchi_config)
        assert cfg.experiment_name == "lucchi_mitochondria"
        assert cfg.model.input_size == [18, 160, 160]
        assert cfg.augmentation.misalignment.enabled
        print("✅ Lucchi config loads and validates")


def test_print_config():
    """Test config printing."""
    cfg = Config()
    cfg.experiment_name = "print_test"
    
    print("\n" + "="*50)
    print("Sample Config YAML:")
    print("="*50)
    print_config(cfg)
    print("="*50)
    print("✅ Config printing works")


def main():
    """Run all tests."""
    print("Testing Hydra Config System\n")
    
    test_default_config_creation()
    test_config_validation()
    test_config_dict_conversion()
    test_config_cli_updates()
    test_config_merge()
    test_config_hash()
    test_experiment_name_generation()
    test_augmentation_config()
    test_load_example_configs()
    
    # Test with temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_config_save_load(Path(tmp_dir))
    
    test_print_config()
    
    print("\n" + "="*50)
    print("🎉 All Hydra config tests passed!")
    print("="*50)


if __name__ == "__main__":
    main()
