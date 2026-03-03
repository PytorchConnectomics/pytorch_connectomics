"""
Test the new Hydra-based configuration system.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

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
    
    assert cfg.model.arch.type == 'monai_basic_unet3d'
    assert cfg.data.dataloader.batch_size == 4
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
    cfg.data.dataloader.batch_size = -1
    try:
        validate_config(cfg)
        raise AssertionError("Invalid config should have failed validation")
    except ValueError:
        print("✅ Invalid config fails validation")


def test_config_dict_conversion():
    """Test converting config to/from dict."""
    cfg = Config()
    cfg.experiment_name = "test_experiment"
    cfg.model.arch.type = "custom_unet"
    
    # To dict
    cfg_dict = to_dict(cfg)
    assert isinstance(cfg_dict, dict)
    assert cfg_dict['experiment_name'] == "test_experiment"
    assert cfg_dict['model']['arch']['type'] == "custom_unet"
    print("✅ Config to dict works")
    
    # From dict
    cfg_restored = from_dict(cfg_dict)
    assert cfg_restored.experiment_name == "test_experiment"
    assert cfg_restored.model.arch.type == "custom_unet"
    print("✅ Dict to config works")


def test_config_cli_updates():
    """Test updating config from CLI arguments."""
    cfg = Config()
    
    overrides = [
        'data.dataloader.batch_size=8',
        'model.arch.type=unetr',
        'optimization.optimizer.lr=0.001'
    ]
    
    updated_cfg = update_from_cli(cfg, overrides)
    
    assert updated_cfg.data.dataloader.batch_size == 8
    assert updated_cfg.model.arch.type == 'unetr'
    assert updated_cfg.optimization.optimizer.lr == 0.001
    print("✅ CLI updates work")


def test_config_merge():
    """Test merging configs."""
    base_cfg = Config()
    base_cfg.experiment_name = "base"
    base_cfg.data.dataloader.batch_size = 2
    
    override_dict = {
        'experiment_name': 'merged',
        'data': {'dataloader': {'batch_size': 4}},
        'model': {'arch': {'type': 'custom'}}
    }
    
    merged_cfg = merge_configs(base_cfg, override_dict)
    
    assert merged_cfg.experiment_name == "merged"
    assert merged_cfg.data.dataloader.batch_size == 4
    assert merged_cfg.model.arch.type == "custom"
    print("✅ Config merge works")


def test_config_save_load(tmp_path):
    """Test saving and loading config."""
    cfg = Config()
    cfg.experiment_name = "save_test"
    cfg.model.monai.filters = (16, 32, 64)
    
    # Save
    config_path = tmp_path / "test_config.yaml"
    save_config(cfg, config_path)
    assert config_path.exists()
    print("✅ Config save works")
    
    # Load
    loaded_cfg = load_config(config_path)
    assert loaded_cfg.experiment_name == "save_test"
    assert tuple(loaded_cfg.model.monai.filters) == (16, 32, 64)
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
    cfg2.data.dataloader.batch_size = 999
    hash3 = get_config_hash(cfg2)
    assert hash1 != hash3
    print("✅ Different configs have different hash")


def test_experiment_name_generation():
    """Test automatic experiment name generation."""
    cfg = Config()
    cfg.model.arch.type = "unet3d"
    cfg.data.dataloader.batch_size = 4
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
        "single-gpu-cpu": {"num_gpus": 1, "num_workers": 1},
        "all-gpu-cpu": {"num_gpus": -1, "num_workers": -1},
    }
    cfg.shared.system.profile = "single-gpu-cpu"
    cfg.shared.system.num_workers = 2
    cfg.train.data = {"dataloader": {"batch_size": 7}}
    cfg.test.system.num_workers = 4
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
    assert cfg.data.dataloader.batch_size == 7
    assert cfg.system.num_gpus == 1
    assert cfg.system.num_workers == 2
    assert cfg.data.image_transform.normalize == "none"

    cfg.test.data.batch_size = 1
    cfg._explicit_field_paths = {"test.data.batch_size"}
    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.system.num_gpus == 1
    assert cfg.data.dataloader.batch_size == 1
    assert cfg.system.num_workers == 4
    assert cfg.test.data.image_transform.normalize == "none"
    assert cfg.test.data.data_transform.resize == [1.0, 1.0, 1.0]
    assert cfg.test.data.data_transform.binarize is True
    assert cfg.test.data.data_transform.threshold == 0.0
    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.sliding_window.overlap == 0.25
    print("✅ Shared profile resolution works")


def test_system_profile_no_implicit_legacy_default():
    """System profiles require explicit shared/system profile selection."""
    cfg = Config()
    cfg.shared.system_profiles = {
        "train_default": {"num_gpus": 2, "num_workers": 3},
        "infer_default": {"num_gpus": 1, "num_workers": 4},
    }

    # Without shared.system.profile (or stage profile), resolver should not auto-apply anything.
    cfg = resolve_shared_profiles(cfg, mode="train")
    assert cfg.data.dataloader.batch_size == 4
    assert cfg.system.num_workers == 8
    print("✅ System profiles are explicit (no implicit legacy defaults)")


def test_yaml_shared_profile_selectors(tmp_path):
    """Test YAML selector-only shared keys for arch/aug/loss/label profiles."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
arch_profiles:
  mednext:
    type: mednext
    variant: S
loss_profiles:
  loss_unit:
    - function: DiceLoss
      weight: 1.5
      pred_slice: [0, 1]
      target_slice: [0, 1]
label_profiles:
  label_unit:
    targets:
      - name: binary
        kwargs: {}
augmentation_profiles:
  aug_unit:
    preset: some
    flip:
      enabled: true
system_profiles:
  single-gpu-cpu:
    num_gpus: 1
    num_workers: 1
decoding_profiles:
  decode_unit:
    - name: decode_instance_binary_contour_distance
      kwargs:
        min_instance_size: 5
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
shared:
  arch_profile: mednext
  augmentation_profile: aug_unit
  loss_profile: loss_unit
  label_profile: label_unit
  decoding_profile: decode_unit
_base_: {base_yaml.name}
""".strip()
    )

    cfg = load_config(config_yaml)
    assert cfg.shared.arch_profile == "mednext"
    assert cfg.shared.augmentation_profile == "aug_unit"
    assert cfg.shared.loss_profile == "loss_unit"
    assert cfg.shared.label_profile == "label_unit"
    assert cfg.shared.decoding_profile == "decode_unit"
    assert "single-gpu-cpu" in cfg.shared.system_profiles
    assert cfg.shared.system_profiles["single-gpu-cpu"].num_gpus == 1
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.mednext.size == "S"
    assert cfg.data.augmentation.preset == "some"
    assert cfg.data.augmentation.flip.enabled is True
    assert cfg.model.loss.losses is not None and len(cfg.model.loss.losses) == 1
    assert cfg.model.loss.losses[0]["function"] == "DiceLoss"
    assert cfg.data.label_transform.targets is not None and len(cfg.data.label_transform.targets) == 1
    assert cfg.data.label_transform.targets[0]["name"] == "binary"
    assert cfg.inference.decoding is not None and len(cfg.inference.decoding) == 1
    assert cfg.inference.decoding[0].name == "decode_instance_binary_contour_distance"
    print("✅ YAML shared profile selectors work")


def test_arch_profile_rejects_non_model_sections(tmp_path):
    """Arch profiles should reject invalid non-ModelConfig keys."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
arch_profiles:
  bad_arch:
    type: mednext
    optimization:
      max_epochs: 999
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
shared:
  arch_profile: bad_arch
_base_: {base_yaml.name}
""".strip()
    )

    with pytest.raises(ValueError, match="invalid model keys"):
        load_config(config_yaml)
    print("✅ Arch profile key boundary enforcement works")


def test_arch_profile_precedence_explicit_model_fields_win(tmp_path):
    """Explicit model fields should override profile-applied model defaults."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
arch_profiles:
  mednext:
    type: mednext
    variant: S
    dropout: 0.4
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
_base_: {base_yaml.name}
shared:
  arch_profile: mednext
model:
  dropout: 0.1
""".strip()
    )

    cfg = load_config(config_yaml)
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.monai.dropout == 0.1
    print("✅ Arch profile precedence works (explicit model field wins)")


def test_system_profile_precedence_shared_then_stage_overrides():
    """Precedence: shared profile < shared override < stage profile < stage override."""
    cfg = Config()
    cfg.shared.system_profiles = {
        "shared_profile": {"num_gpus": 0, "num_workers": 1},
        "train_profile": {"num_gpus": 2, "num_workers": 8},
    }

    cfg.shared.system.profile = "shared_profile"
    cfg.shared.system.num_workers = 3
    cfg.train.system.profile = "train_profile"
    cfg.train.system.num_workers = 10

    cfg = resolve_shared_profiles(cfg, mode="train")
    assert cfg.system.num_gpus == 2
    assert cfg.system.num_workers == 10
    print("✅ System profile precedence works")


def test_data_transform_profile_precedence_stage_overrides_win():
    """Stage image_transform values should win over selected profile defaults."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.shared.data_transform_profiles = {
        "default": {
            "image_transform": {"normalize": "none", "clip_percentile_low": 0.25},
        }
    }
    cfg.test.data.image_transform.transform_profile = "default"
    cfg.test.data.image_transform.normalize = "normal"

    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.test.data.image_transform.normalize == "normal"
    print("✅ Data transform profile precedence works")


def test_yaml_dataloader_optimizer_profiles_apply(tmp_path):
    """Dataloader/optimizer profile selectors should apply from YAML profile registries."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
dataloader_profiles:
  cached:
    use_preloaded_cache_train: true
    persistent_workers: true
optimizer_profiles:
  warmup_cosine_lr:
    gradient_clip_val: 3.0
    optimizer:
      lr: 0.0003
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
_base_: {base_yaml.name}
shared:
  dataloader_profile: cached
  optimizer_profile: warmup_cosine_lr
""".strip()
    )

    cfg = load_config(config_yaml)
    assert cfg.data.dataloader.use_preloaded_cache_train is True
    assert cfg.data.dataloader.persistent_workers is True
    assert cfg.optimization.gradient_clip_val == 3.0
    assert cfg.optimization.optimizer.lr == 0.0003
    print("✅ Dataloader/optimizer profiles apply from YAML selectors")


def test_runtime_merge_shared_then_mode_for_train_sections():
    """Runtime section precedence: defaults < shared < train."""
    cfg = Config()
    cfg.shared.model = {"arch": {"type": "mednext"}, "monai": {"dropout": 0.4}}
    cfg.shared.monitor = {"detect_anomaly": True}
    cfg.train.model = {"monai": {"dropout": 0.2}}
    cfg.train.monitor = {"detect_anomaly": False}

    cfg = resolve_shared_profiles(cfg, mode="train")
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.monai.dropout == 0.2
    assert cfg.monitor.detect_anomaly is False
    print("✅ Generic runtime merge precedence works for train mode")


def test_runtime_merge_and_inference_profile_for_test_mode():
    """Test mode should merge shared/test runtime overrides and inference profile."""
    cfg = Config()
    cfg.test = HydraTestConfig()

    cfg.shared.model = {"arch": {"type": "mednext"}}
    cfg.test.model = {"monai": {"dropout": 0.15}}

    cfg.shared.inference = {"sliding_window": {"overlap": 0.2}}
    cfg.shared.inference_profiles = {
        "default": {
            "test_time_augmentation": {"enabled": False},
            "sliding_window": {"overlap": 0.3},
        }
    }
    cfg.test.inference = {"profile": "default", "sliding_window": {"overlap": 0.4}}

    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.monai.dropout == 0.15
    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.sliding_window.overlap == 0.4
    print("✅ Test mode runtime + inference profile merge works")


def test_runtime_merge_test_data_section_overrides_runtime_data(tmp_path):
    """Mode-specific data merge should read overrides from test.data (not test.data_overrides)."""
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text(
        """
shared:
  data:
    image_transform:
      normalize: "0-1"
test:
  data:
    image_transform:
      normalize: none
    val:
      image: "dummy.h5"
""".strip()
    )

    cfg = load_config(config_yaml)
    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.data.image_transform.normalize == "none"
    print("✅ test.data drives runtime data overrides")


def test_inference_system_overrides_runtime_system_in_test_mode():
    """inference.system should override runtime cfg.system in test mode."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.shared.inference = {"system": {"num_gpus": 2, "num_workers": 3}}
    cfg.test.inference = {"system": {"num_workers": 5}}

    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.system.num_gpus == 2
    assert cfg.system.num_workers == 5
    print("✅ inference.system overrides runtime system in test mode")


def test_auto_enable_when_section_has_keys(tmp_path):
    """Sections with `enabled` default False/None auto-enable when YAML provides sibling keys."""
    config_yaml = tmp_path / "auto_enable.yaml"
    config_yaml.write_text(
        """
inference:
  test_time_augmentation:
    flip_axes: all
  save_prediction:
    output_formats: [h5]
  evaluation:
    metrics: [dice]
monitor:
  logging:
    images:
      max_images: 2
  early_stopping:
    enabled: false
    patience: 5
optimization:
  ema:
    decay: 0.99
""".strip()
    )

    cfg = load_config(config_yaml)
    cfg = resolve_shared_profiles(cfg, mode="train")

    assert cfg.inference.test_time_augmentation.enabled is True
    assert cfg.inference.save_prediction.enabled is True
    assert cfg.inference.evaluation.enabled is True
    assert cfg.monitor.logging.images.enabled is True
    assert cfg.optimization.ema.enabled is True
    # Explicit value in YAML should always win.
    assert cfg.monitor.early_stopping.enabled is False
    print("✅ Auto-enable from YAML keys works")


def test_shared_inference_decoding_profile_list_ref(tmp_path):
    """Allow list refs like `- profile: decoding_bcd` under shared.inference.decoding."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        """
decoding_profiles:
  decoding_bcd:
    - name: decode_instance_binary_contour_distance
      kwargs:
        min_instance_size: 3
""".strip()
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        f"""
_base_: {base_yaml.name}
shared:
  inference:
    decoding:
      - profile: decoding_bcd
""".strip()
    )

    cfg = load_config(config_yaml)
    cfg = resolve_shared_profiles(cfg, mode="test")
    assert cfg.inference.decoding is not None and len(cfg.inference.decoding) == 1
    assert cfg.inference.decoding[0].name == "decode_instance_binary_contour_distance"
    assert cfg.inference.decoding[0].kwargs["min_instance_size"] == 3
    print("✅ Shared inference decoding profile-list ref resolves")


def test_build_test_transforms_with_mask_transform_resize_binarize():
    """Test test-mode mask_transform resize+binarize pipeline creation."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.test.data.val.image = "dummy_image.h5"
    cfg.test.data.val.mask = "dummy_mask.h5"
    cfg.test.data.data_transform.resize = [1, 2, 2]
    cfg.test.data.data_transform.binarize = True
    cfg.test.data.data_transform.threshold = 0.0

    transforms = build_test_transforms(cfg)
    transform_names = [type(t).__name__ for t in transforms.transforms]

    assert "ResizeByFactord" in transform_names
    assert "Lambdad" in transform_names
    print("✅ Test transforms include mask resize+binarize")


def test_mask_binarize_uses_strict_greater_than_threshold():
    """Ensure mask binarization preserves zeros when threshold=0.0 (mask > 0)."""
    cfg = Config()
    cfg.data.dataloader.patch_size = [0, 0, 0]  # Disable padding for this unit test.
    cfg.test = HydraTestConfig()
    cfg.test.data.val.image = "dummy_image.h5"
    cfg.test.data.val.mask = "dummy_mask.h5"
    cfg.test.data.image_transform.normalize = "none"
    cfg.test.data.data_transform.resize = None
    cfg.test.data.data_transform.binarize = True
    cfg.test.data.data_transform.threshold = 0.0

    transforms = build_test_transforms(cfg)

    sample = {
        "image": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "mask": np.array([[[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]]], dtype=np.float32),
    }
    out = transforms(sample)
    mask_arr = out["mask"]
    mask = mask_arr.numpy() if hasattr(mask_arr, "numpy") else np.asarray(mask_arr)

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
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_auto_enable_when_section_has_keys(Path(tmp_dir))
    test_build_test_transforms_with_mask_transform_resize_binarize()
    
    print("\n" + "="*50)
    print("🎉 All Hydra config tests passed!")
    print("="*50)


if __name__ == "__main__":
    main()
