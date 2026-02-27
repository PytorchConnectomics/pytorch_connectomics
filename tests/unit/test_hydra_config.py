"""
Test the new Hydra-based configuration system.
"""

import sys
from pathlib import Path
import numpy as np

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
    resolve_shared_profiles,
)
from connectomics.config.hydra_config import TestConfig as HydraTestConfig, TuneConfig
from connectomics.data.augment.build import build_test_transforms


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


def test_shared_profile_resolution():
    """Test resolving shared profiles into train/test/tune runtime sections."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.tune = TuneConfig()

    cfg.shared.system_profiles = {
        "train_default": {"num_gpus": 2, "num_workers": 3, "batch_size": 7},
        "infer_default": {"num_gpus": 1, "num_workers": 4, "batch_size": 1},
    }
    cfg.shared.data_transform_profiles = {
        "default": {
            "image_transform": {
                "normalize": "none",
                "clip_percentile_low": 0.1,
                "clip_percentile_high": 0.9,
            },
            "mask_transform": {
                "resize": [1.0, 1.0, 1.0],
                "binarize": True,
                "threshold": 0.0,
            },
            "nnunet_preprocessing": {"enabled": True},
        }
    }
    cfg.shared.inference_profiles = {
        "default": {
            "test_time_augmentation": {"enabled": False},
            "sliding_window": {"overlap": 0.25},
        }
    }

    cfg = resolve_shared_profiles(cfg, mode="train")
    assert cfg.system.training.batch_size == 7
    assert cfg.data.image_transform.normalize == "none"

    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.system.inference.num_workers == 4
    assert cfg.test.data.image_transform.normalize == "none"
    assert cfg.test.data.mask_transform.resize == [1.0, 1.0, 1.0]
    assert cfg.test.data.mask_transform.binarize is True
    assert cfg.test.data.mask_transform.threshold == 0.0
    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.sliding_window.overlap == 0.25
    print("✅ Shared profile resolution works")


def test_yaml_shared_profile_selectors(tmp_path):
    """Test YAML selector-only shared keys for arch/loss profiles."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
arch_profiles:
  mednext:
    model:
      architecture: mednext
loss_profiles:
  loss_unit:
    - function: DiceLoss
      weight: 1.5
      pred_slice: [0, 1]
      target_slice: [0, 1]
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
shared:
  arch_profile: mednext
  loss_profile: loss_unit
_base_: {base_yaml.name}
""".strip()
    )

    cfg = load_config(config_yaml)
    assert cfg.shared.arch_profile == "mednext"
    assert cfg.shared.loss_profile == "loss_unit"
    assert cfg.model.architecture == "mednext"
    assert cfg.model.losses is not None and len(cfg.model.losses) == 1
    assert cfg.model.losses[0]["function"] == "DiceLoss"
    print("✅ YAML shared profile selectors work")


def test_build_test_transforms_with_mask_transform_resize_binarize():
    """Test test-mode mask_transform resize+binarize pipeline creation."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.test.data.test_image = "dummy_image.h5"
    cfg.test.data.test_mask = "dummy_mask.h5"
    cfg.test.data.mask_transform.resize = [1, 2, 2]
    cfg.test.data.mask_transform.binarize = True
    cfg.test.data.mask_transform.threshold = 0.0

    transforms = build_test_transforms(cfg)
    transform_names = [type(t).__name__ for t in transforms.transforms]

    assert "ResizeByFactord" in transform_names
    assert "Lambdad" in transform_names
    print("✅ Test transforms include mask resize+binarize")


def test_mask_binarize_uses_strict_greater_than_threshold():
    """Ensure mask binarization preserves zeros when threshold=0.0 (mask > 0)."""
    cfg = Config()
    cfg.data.patch_size = [0, 0, 0]  # Disable padding for this unit test.
    cfg.test = HydraTestConfig()
    cfg.test.data.test_image = "dummy_image.h5"
    cfg.test.data.test_mask = "dummy_mask.h5"
    cfg.test.data.image_transform.normalize = "none"
    cfg.test.data.mask_transform.resize = None
    cfg.test.data.mask_transform.binarize = True
    cfg.test.data.mask_transform.threshold = 0.0

    transforms = build_test_transforms(cfg)

    sample = {
        "image": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "mask": np.array([[[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]]], dtype=np.float32),
    }
    out = transforms(sample)
    mask = out["mask"].numpy()

    assert mask.min() == 0.0
    assert mask.max() == 1.0
    # Zero-valued voxels must stay zero with strict > 0 binarization.
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 1, 1] == 0.0
    assert mask[0, 0, 0, 1] == 1.0
    print("✅ Mask binarization uses strict > threshold semantics")


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
    test_shared_profile_resolution()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_yaml_shared_profile_selectors(Path(tmp_dir))
    test_build_test_transforms_with_mask_transform_resize_binarize()
    
    print("\n" + "="*50)
    print("🎉 All Hydra config tests passed!")
    print("="*50)


if __name__ == "__main__":
    main()
