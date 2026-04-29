"""
Test the new Hydra-based configuration system.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from connectomics.config import (  # noqa: E402
    Config,
    load_config,
    resolve_default_profiles,
    save_config,
    validate_config,
)
from connectomics.config.pipeline import (  # noqa: E402
    create_experiment_name,
    from_dict,
    get_config_hash,
    merge_configs,
    print_config,
    to_dict,
    update_from_cli,
)
from connectomics.config.schema import TestConfig as HydraTestConfig  # noqa: E402
from connectomics.config.schema import TuneConfig  # noqa: E402
from connectomics.data.augmentation.build import build_test_transforms  # noqa: E402
from connectomics.data.processing.build import create_label_transform_pipeline  # noqa: E402
from connectomics.runtime.preflight import validate_runtime_coherence  # noqa: E402


def test_default_config_creation():
    """Test creating default config."""
    cfg = Config()

    assert cfg.model.arch.type == "monai_basic_unet3d"
    assert cfg.data.dataloader.batch_size == 4
    assert cfg.optimization.optimizer.name == "AdamW"
    assert cfg.optimization.max_epochs == 200
    assert cfg.monitor.logging.images.channel_mode == "all"
    print("[OK]Default config creation works")


def test_config_validation():
    """Test config validation."""
    cfg = Config()

    # Valid config should pass
    try:
        validate_config(cfg)
        print("[OK]Valid config passes validation")
    except ValueError as e:
        raise AssertionError(f"Valid config failed validation: {e}")

    # Invalid config should fail
    cfg.data.dataloader.batch_size = -1
    try:
        validate_config(cfg)
        raise AssertionError("Invalid config should have failed validation")
    except ValueError:
        print("[OK]Invalid config fails validation")


def test_cross_section_validation_rejects_input_patch_mismatch():
    """input_size and dataloader.patch_size must stay coherent."""
    cfg = Config()
    cfg.model.input_size = [64, 64, 64]
    cfg.data.dataloader.patch_size = [128, 128, 128]

    with pytest.raises(ValueError, match="model.input_size .* must match .*patch_size"):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_out_channel_mismatch():
    """Resolved pipeline channel requirements must fit model.out_channels."""
    cfg = Config()
    cfg.model.out_channels = 1
    cfg.model.loss.losses = [
        {
            "function": "DiceLoss",
            "weight": 1.0,
            "pred_slice": "0:3",
            "target_slice": "0:3",
        }
    ]

    with pytest.raises(ValueError, match="model.out_channels .* require at least 3 channels"):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_out_channel_mismatch_with_negative_slice():
    """Negative slice bounds should still enforce a concrete channel lower bound."""
    cfg = Config()
    cfg.model.out_channels = 2
    cfg.model.loss.losses = [
        {
            "function": "DiceLoss",
            "weight": 1.0,
            "pred_slice": "0:-2",
            "target_slice": "0:-2",
        }
    ]

    with pytest.raises(ValueError, match="model.out_channels .* require at least 3 channels"):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_label_target_channel_mismatch():
    """Stacked label targets impose a minimum output-channel requirement."""
    cfg = Config()
    cfg.model.out_channels = 2
    cfg.data.label_transform.targets = [
        {"name": "binary"},
        {"name": "boundary"},
        {"name": "distance"},
    ]

    with pytest.raises(ValueError, match="require at least 3 channels"):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_decoding_channel_mismatch():
    """Decoding channel selectors must fit model.out_channels."""
    cfg = from_dict(
        {
            "model": {"out_channels": 2},
            "decoding": {
                "steps": [
                    {
                        "name": "decode_instance_binary_contour_distance",
                        "kwargs": {"distance_channels": [2]},
                    }
                ],
            },
        }
    )

    with pytest.raises(ValueError, match="decoding\\[0\\].kwargs.distance_channels"):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_accepts_tta_full_span_selector_for_single_channel():
    """TTA should accept Python-style full-span selectors for binary outputs."""
    cfg = Config()
    cfg.model.out_channels = 1
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": ":", "activation": "sigmoid"}
    ]

    validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_invalid_tta_channel_range():
    """TTA channel selectors must still fit model.out_channels."""
    cfg = Config()
    cfg.model.out_channels = 1
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": "1:", "activation": "sigmoid"}
    ]

    with pytest.raises(
        ValueError,
        match="inference.test_time_augmentation.channel_activations\\[0\\]\\.channels",
    ):
        validate_runtime_coherence(cfg)


def test_cross_section_validation_rejects_incompatible_deep_supervision_arch():
    """deep_supervision requires an architecture that supports it."""
    cfg = Config()
    cfg.model.arch.type = "monai_unet"
    cfg.model.loss.deep_supervision = True

    with pytest.raises(ValueError, match="does not support deep supervision"):
        validate_runtime_coherence(cfg)


def test_config_dict_conversion():
    """Test converting config to/from dict."""
    cfg = Config()
    cfg.experiment_name = "test_experiment"
    cfg.model.arch.type = "custom_unet"

    # To dict
    cfg_dict = to_dict(cfg)
    assert isinstance(cfg_dict, dict)
    assert cfg_dict["experiment_name"] == "test_experiment"
    assert cfg_dict["model"]["arch"]["type"] == "custom_unet"
    print("[OK]Config to dict works")

    # From dict
    cfg_restored = from_dict(cfg_dict)
    assert cfg_restored.experiment_name == "test_experiment"
    assert cfg_restored.model.arch.type == "custom_unet"
    print("[OK]Dict to config works")


def test_config_cli_updates():
    """Test updating config from CLI arguments."""
    cfg = Config()

    overrides = [
        "data.dataloader.batch_size=8",
        "model.arch.type=unetr",
        "optimization.optimizer.lr=0.001",
    ]

    updated_cfg = update_from_cli(cfg, overrides)

    assert updated_cfg.data.dataloader.batch_size == 8
    assert updated_cfg.model.arch.type == "unetr"
    assert updated_cfg.optimization.optimizer.lr == 0.001
    print("[OK]CLI updates work")


def test_config_merge():
    """Test merging configs."""
    base_cfg = Config()
    base_cfg.experiment_name = "base"
    base_cfg.data.dataloader.batch_size = 2

    override_dict = {
        "experiment_name": "merged",
        "data": {"dataloader": {"batch_size": 4}},
        "model": {"arch": {"type": "custom"}},
    }

    merged_cfg = merge_configs(base_cfg, override_dict)

    assert merged_cfg.experiment_name == "merged"
    assert merged_cfg.data.dataloader.batch_size == 4
    assert merged_cfg.model.arch.type == "custom"
    print("[OK]Config merge works")


def test_config_save_load(tmp_path):
    """Test saving and loading config."""
    cfg = Config()
    cfg.experiment_name = "save_test"
    cfg.model.monai.filters = (16, 32, 64)

    # Save
    config_path = tmp_path / "test_config.yaml"
    save_config(cfg, config_path)
    assert config_path.exists()
    print("[OK]Config save works")

    # Load
    loaded_cfg = load_config(config_path)
    assert loaded_cfg.experiment_name == "save_test"
    assert tuple(loaded_cfg.model.monai.filters) == (16, 32, 64)
    print("[OK]Config load works")


def test_config_hash():
    """Test config hashing."""
    cfg1 = Config()
    cfg2 = Config()

    # Same configs should have same hash
    hash1 = get_config_hash(cfg1)
    hash2 = get_config_hash(cfg2)
    assert hash1 == hash2
    print("[OK]Same configs have same hash")

    # Different configs should have different hash
    cfg2.data.dataloader.batch_size = 999
    hash3 = get_config_hash(cfg2)
    assert hash1 != hash3
    print("[OK]Different configs have different hash")


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
    assert len(name.split("_")[-1]) == 8  # Hash
    print(f"[OK]Generated experiment name: {name}")


def test_augmentation_config():
    """Test augmentation configuration."""
    cfg = Config()

    # Enable BANIS-style and legacy EM augmentations
    cfg.data.augmentation.axis_permute.enabled = True
    cfg.data.augmentation.rotate90_all.enabled = True
    cfg.data.augmentation.slice_shift.enabled = True
    cfg.data.augmentation.slice_drop.enabled = True
    cfg.data.augmentation.slice_shift_z.enabled = True
    cfg.data.augmentation.slice_drop_z.enabled = True
    cfg.data.augmentation.misalignment.enabled = True
    cfg.data.augmentation.misalignment.prob = 0.7
    cfg.data.augmentation.missing_section.enabled = True
    cfg.data.augmentation.mixup.enabled = True
    cfg.data.augmentation.copy_paste.enabled = True

    assert cfg.data.augmentation.axis_permute.enabled
    assert cfg.data.augmentation.rotate90_all.enabled
    assert cfg.data.augmentation.slice_shift.enabled
    assert cfg.data.augmentation.slice_drop.enabled
    assert cfg.data.augmentation.slice_shift_z.enabled
    assert cfg.data.augmentation.slice_drop_z.enabled
    assert cfg.data.augmentation.misalignment.enabled
    assert cfg.data.augmentation.misalignment.prob == 0.7
    assert cfg.data.augmentation.missing_section.enabled
    assert cfg.data.augmentation.mixup.enabled
    assert cfg.data.augmentation.copy_paste.enabled
    print("[OK]Augmentation config works")


def test_load_example_configs():
    """Test loading example configs."""
    configs_dir = project_root / "configs" / "hydra"

    # Test default config
    default_config = configs_dir / "default.yaml"
    if default_config.exists():
        cfg = load_config(default_config)
        assert cfg.experiment_name == "connectomics_default"
        validate_config(cfg)
        print("[OK]Default config loads and validates")

    # Test Lucchi config
    lucchi_config = configs_dir / "lucchi.yaml"
    if lucchi_config.exists():
        cfg = load_config(lucchi_config)
        assert cfg.experiment_name == "lucchi_mitochondria"
        assert cfg.model.input_size == [18, 160, 160]
        assert cfg.augmentation.misalignment.enabled
        print("[OK]Lucchi config loads and validates")


def test_print_config():
    """Test config printing."""
    cfg = Config()
    cfg.experiment_name = "print_test"

    print("\n" + "=" * 50)
    print("Sample Config YAML:")
    print("=" * 50)
    print_config(cfg)
    print("=" * 50)
    print("[OK]Config printing works")


def test_shared_profile_resolution():
    """Runtime merge should apply default stage, then mode-specific overrides."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.tune = TuneConfig()

    cfg.default.system.num_gpus = 1
    cfg.default.system.num_workers = 2
    cfg.train.data = {"dataloader": {"batch_size": 7}}
    cfg.test.system.num_workers = 4
    cfg.default.data.image_transform.normalize = "none"
    cfg.default.inference.test_time_augmentation.enabled = False
    cfg.default.inference.sliding_window.overlap = 0.25
    cfg._merge_context.explicit_field_paths = {
        "default.system.num_gpus",
        "default.system.num_workers",
        "default.data.image_transform.normalize",
        "default.inference.test_time_augmentation.enabled",
        "default.inference.sliding_window.overlap",
        "train.data.dataloader.batch_size",
        "test.system.num_workers",
        "test.data.dataloader.batch_size",
    }

    cfg = resolve_default_profiles(cfg, mode="train")
    assert cfg.data.dataloader.batch_size == 7
    assert cfg.system.num_gpus == 1
    assert cfg.system.num_workers == 2
    assert cfg.data.image_transform.normalize == "none"

    cfg.test.data.dataloader.batch_size = 1
    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.system.num_gpus == 1
    assert cfg.data.dataloader.batch_size == 1
    assert cfg.system.num_workers == 4
    assert cfg.data.image_transform.normalize == "none"
    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.sliding_window.overlap == 0.25
    print("[OK]Shared profile resolution works")


def test_system_profile_no_implicit_legacy_default():
    """System profiles require explicit shared/system profile selection."""
    cfg = Config()
    cfg.default.system_profiles = {
        "train_default": {"num_gpus": 2, "num_workers": 3},
        "infer_default": {"num_gpus": 1, "num_workers": 4},
    }

    # Without default.system.profile (or stage profile), resolver should not auto-apply anything.
    cfg = resolve_default_profiles(cfg, mode="train")
    assert cfg.data.dataloader.batch_size == 4
    assert cfg.system.num_workers == 8
    print("[OK]System profiles are explicit (no implicit legacy defaults)")


def test_yaml_shared_profile_selectors(tmp_path):
    """Test YAML selector-only shared keys for arch/aug/loss/label profiles."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
arch_profiles:
  mednext:
    arch:
      type: mednext
    mednext:
      size: S
loss_profiles:
  loss_unit:
    losses:
      - function: DiceLoss
        weight: 1.5
        pred_slice: "0:1"
        target_slice: "0:1"
label_profiles:
  label_unit:
    targets:
      - name: binary
        kwargs: {}
augmentation_profiles:
  aug_unit:
    flip:
      enabled: true
system_profiles:
  single-gpu-cpu:
    num_gpus: 1
    num_workers: 1
decoding_templates:
  decode_unit:
    steps:
      - name: decode_instance_binary_contour_distance
        kwargs:
          min_instance_size: 5
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
default:
  model:
    arch:
      profile: mednext
    loss:
      profile: loss_unit
  data:
    augmentation:
      profile: aug_unit
    label_transform:
      profile: label_unit
  decoding:
    steps:
      - template: decode_unit
  system:
    profile: single-gpu-cpu
_base_: {base_yaml.name}
""".strip())

    cfg = load_config(config_yaml)
    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.mednext.size == "S"
    assert cfg.data.augmentation.flip.enabled is True
    assert cfg.decoding.steps is not None and len(cfg.decoding.steps) == 1
    assert cfg.decoding.steps[0].kwargs["min_instance_size"] == 5
    print("[OK]YAML shared profile selectors work")


def test_label_transform_profile_can_inherit_top_level_resolution(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    base_profiles = repo_root / "connectomics" / "config" / "all_profiles.yaml"
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
_base_:
  - {base_profiles}

default:
  data:
    label_transform:
      profile: label_aff9_sdt
      resolution: [30, 6, 6]
""".strip())

    cfg = load_config(config_yaml)

    targets = cfg.default.data.label_transform.targets
    assert len(targets) == 2
    assert targets[0]["name"] == "affinity"
    assert targets[1]["name"] == "skeleton_aware_edt"
    assert cfg.default.data.label_transform.resolution == [30.0, 6.0, 6.0]

    pipeline = create_label_transform_pipeline(cfg.default.data.label_transform)
    assert pipeline.task_specs[1]["kwargs"]["resolution"] == [30.0, 6.0, 6.0]


def test_arch_profile_rejects_non_model_sections(tmp_path):
    """Arch profiles with invalid keys are rejected by OmegaConf schema merge."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
arch_profiles:
  bad_arch:
    arch:
      type: mednext
    optimization:
      max_epochs: 999
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
default:
  model:
    arch:
      profile: bad_arch
_base_: {base_yaml.name}
""".strip())

    with pytest.raises(Exception):
        load_config(config_yaml)
    print("[OK]Arch profile key boundary enforcement works")


def test_arch_profile_precedence_explicit_model_fields_win(tmp_path):
    """Explicit model fields should override profile-applied model defaults."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
arch_profiles:
  mednext:
    arch:
      type: mednext
    mednext:
      size: S
    monai:
      dropout: 0.4
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
_base_: {base_yaml.name}
default:
  model:
    arch:
      profile: mednext
model:
  monai:
    dropout: 0.1
""".strip())

    cfg = load_config(config_yaml)
    assert cfg.model.arch.type == "monai_basic_unet3d"
    assert cfg.model.monai.dropout == 0.1
    print("[OK]Arch profile precedence works (explicit model field wins)")


def test_system_profile_precedence_shared_then_stage_overrides():
    """Precedence: default system override < train system override."""
    cfg = Config()
    cfg.default.system.num_gpus = 0
    cfg.default.system.num_workers = 3
    cfg.train.system.num_gpus = 2
    cfg.train.system.num_workers = 10
    cfg._merge_context.explicit_field_paths = {
        "default.system.num_gpus",
        "default.system.num_workers",
        "train.system.num_gpus",
        "train.system.num_workers",
    }

    cfg = resolve_default_profiles(cfg, mode="train")
    assert cfg.system.num_gpus == 2
    assert cfg.system.num_workers == 10
    print("[OK]System profile precedence works")


def test_data_transform_profile_precedence_stage_overrides_win():
    """Stage image_transform values should win over default stage values."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.default.data.image_transform.normalize = "none"
    cfg.default.data.image_transform.clip_percentile_low = 0.25
    cfg.test.data.image_transform.normalize = "normal"
    cfg._merge_context.explicit_field_paths = {
        "default.data.image_transform.normalize",
        "default.data.image_transform.clip_percentile_low",
        "test.data.image_transform.normalize",
    }

    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.data.image_transform.normalize == "normal"
    print("[OK]Data transform profile precedence works")


def test_yaml_dataloader_optimizer_profiles_apply(tmp_path):
    """Default-stage dataloader/optimization values should apply from YAML."""
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
default:
  data:
    dataloader:
      use_preloaded_cache_train: true
      persistent_workers: true
  optimization:
    gradient_clip_val: 3.0
    optimizer:
      lr: 0.0003
""".strip())

    cfg = load_config(config_yaml)
    assert cfg.default.data.dataloader.use_preloaded_cache_train is True
    assert cfg.default.data.dataloader.persistent_workers is True
    assert cfg.default.optimization.gradient_clip_val == 3.0
    assert cfg.default.optimization.optimizer.lr == 0.0003
    print("[OK]Dataloader/optimizer profiles apply from YAML selectors")


def test_runtime_merge_shared_then_mode_for_train_sections():
    """Runtime section precedence: defaults < shared < train."""
    cfg = Config()
    cfg.default.model.arch.type = "mednext"
    cfg.default.model.monai.dropout = 0.4
    cfg.default.monitor.detect_anomaly = True
    cfg.train.model.monai.dropout = 0.2
    cfg.train.monitor.detect_anomaly = False
    cfg._merge_context.explicit_field_paths = {
        "default.model.arch.type",
        "default.model.monai.dropout",
        "default.monitor.detect_anomaly",
        "train.model.monai.dropout",
        "train.monitor.detect_anomaly",
    }

    cfg = resolve_default_profiles(cfg, mode="train")
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.monai.dropout == 0.2
    assert cfg.monitor.detect_anomaly is False
    print("[OK]Generic runtime merge precedence works for train mode")


def test_runtime_merge_and_inference_profile_for_test_mode():
    """Test mode should merge shared/test runtime overrides for inference."""
    cfg = Config()
    cfg.test = HydraTestConfig()

    cfg.default.model.arch.type = "mednext"
    cfg.test.model.monai.dropout = 0.15
    cfg.default.inference.test_time_augmentation.enabled = False
    cfg.default.inference.sliding_window.overlap = 0.3
    test_inference_cfg = getattr(cfg.test, "inference")
    test_inference_cfg.sliding_window.overlap = 0.4
    cfg._merge_context.explicit_field_paths = {
        "default.model.arch.type",
        "test.model.monai.dropout",
        "default.inference.test_time_augmentation.enabled",
        "default.inference.sliding_window.overlap",
        "test.inference.sliding_window.overlap",
    }

    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.model.arch.type == "mednext"
    assert cfg.model.monai.dropout == 0.15
    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.sliding_window.overlap == 0.4
    print("[OK]Test mode runtime + inference profile merge works")


def test_runtime_merge_test_data_section_overrides_runtime_data(tmp_path):
    """Mode-specific data merge should read overrides from test.data (not test.data_overrides)."""
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text("""
default:
  data:
    image_transform:
      normalize: "0-1"
test:
  data:
    image_transform:
      normalize: none
    val:
      image: "dummy.h5"
""".strip())

    cfg = load_config(config_yaml)
    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.data.image_transform.normalize == "none"
    print("[OK]test.data drives runtime data overrides")


def test_inference_system_overrides_runtime_system_in_test_mode():
    """inference.system stores test-mode resource overrides for inference."""
    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.default.inference.system.num_gpus = 2
    cfg.default.inference.system.num_workers = 3
    test_inference_cfg = getattr(cfg.test, "inference")
    test_inference_cfg.system.num_workers = 5
    cfg._merge_context.explicit_field_paths = {
        "default.inference.system.num_gpus",
        "default.inference.system.num_workers",
        "test.inference.system.num_workers",
    }

    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.system.num_gpus == 1
    assert cfg.system.num_workers == 8
    assert cfg.inference.system.num_gpus == 2
    assert cfg.inference.system.num_workers == 5
    print("[OK]inference.system stores test-mode overrides")


def test_enabled_flags_require_explicit_opt_in(tmp_path):
    """Most sections require explicit opt-in; image logging defaults to enabled."""
    config_yaml = tmp_path / "auto_enable.yaml"
    config_yaml.write_text("""
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
""".strip())

    cfg = load_config(config_yaml)
    cfg = resolve_default_profiles(cfg, mode="train")

    assert cfg.inference.test_time_augmentation.enabled is False
    assert cfg.inference.save_prediction.enabled is False
    assert cfg.evaluation.enabled is False
    assert cfg.monitor.logging.images.enabled is True
    assert cfg.optimization.ema.enabled is False
    # Explicit value in YAML should always win.
    assert cfg.monitor.early_stopping.enabled is False
    print("[OK]Enabled flags require explicit opt-in")


def test_removed_shared_stage_is_rejected(tmp_path):
    config_yaml = tmp_path / "removed_shared.yaml"
    config_yaml.write_text("""
shared:
  model:
    arch:
      type: mednext
""".strip())

    with pytest.raises(ValueError, match="shared.*removed"):
        load_config(config_yaml)


def test_nested_inference_decoding_is_rejected(tmp_path):
    config_yaml = tmp_path / "nested_inference_decoding.yaml"
    config_yaml.write_text("""
inference:
  decoding:
    - name: decode_semantic
""".strip())

    with pytest.raises(Exception, match="decoding"):
        load_config(config_yaml)


def test_nested_inference_evaluation_is_rejected(tmp_path):
    config_yaml = tmp_path / "nested_inference_evaluation.yaml"
    config_yaml.write_text("""
inference:
  evaluation:
    enabled: true
""".strip())

    with pytest.raises(Exception, match="evaluation"):
        load_config(config_yaml)


def test_removed_wandb_prefix_fields_are_rejected(tmp_path):
    config_yaml = tmp_path / "removed_wandb_prefix.yaml"
    config_yaml.write_text("""
monitor:
  wandb:
    wandb_project: connectomics
""".strip())

    with pytest.raises(Exception, match="wandb_project"):
        load_config(config_yaml)


def test_removed_optimizer_step_aliases_are_rejected(tmp_path):
    config_yaml = tmp_path / "removed_optimizer_alias.yaml"
    config_yaml.write_text("""
optimization:
  iter_num_per_epoch: 100
""".strip())

    with pytest.raises(Exception, match="iter_num_per_epoch"):
        load_config(config_yaml)


def test_shared_decoding_template_list_ref(tmp_path):
    """Allow list refs like `- template: decoding_bcd` under default.decoding."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
decoding_templates:
  decoding_bcd:
    steps:
      - name: decode_instance_binary_contour_distance
        kwargs:
          min_instance_size: 3
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
_base_: {base_yaml.name}
default:
  decoding:
    steps:
      - template: decoding_bcd
""".strip())

    cfg = load_config(config_yaml)
    cfg = resolve_default_profiles(cfg, mode="test")
    assert cfg.decoding.steps is not None and len(cfg.decoding.steps) == 1
    assert cfg.decoding.steps[0].name == "decode_instance_binary_contour_distance"
    assert cfg.decoding.steps[0].kwargs["min_instance_size"] == 3
    print("[OK]Shared decoding profile-list ref resolves")


def test_top_level_decoding_template_list_ref(tmp_path):
    """Allow list refs like `- template: decoding_bcd` under root decoding."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
decoding_templates:
  decoding_bcd:
    steps:
      - name: decode_instance_binary_contour_distance
        kwargs:
          min_instance_size: 7
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
_base_: {base_yaml.name}
decoding:
  steps:
    - template: decoding_bcd
""".strip())

    cfg = load_config(config_yaml)
    assert cfg.decoding.steps is not None and len(cfg.decoding.steps) == 1
    assert cfg.decoding.steps[0].name == "decode_instance_binary_contour_distance"
    assert cfg.decoding.steps[0].kwargs["min_instance_size"] == 7


def test_loss_profile_positional_overrides(tmp_path):
    """Loss profile + overrides dict patches individual list entries by index."""
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
loss_profiles:
  loss_binary:
    losses:
      - function: WeightedBCEWithLogitsLoss
        weight: 1.0
        kwargs: {reduction: mean}
      - function: DiceLoss
        weight: 1.0
        kwargs: {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
""".strip())

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
_base_: {base_yaml.name}
default:
  model:
    loss:
      profile: loss_binary
      overrides:
        0: {{pos_weight: auto}}
        1: {{weight: 0.5}}
""".strip())

    cfg = load_config(config_yaml)
    cfg = resolve_default_profiles(cfg, mode="train")
    losses = cfg.model.loss.losses
    assert len(losses) == 2

    # Entry 0: BCE fields merged (pos_weight added, kwargs kept)
    assert losses[0]["function"] == "WeightedBCEWithLogitsLoss"
    assert losses[0]["pos_weight"] == "auto"
    assert losses[0]["kwargs"]["reduction"] == "mean"
    assert losses[0]["weight"] == 1.0

    # Entry 1: Dice weight overridden from 1.0 to 0.5
    assert losses[1]["function"] == "DiceLoss"
    assert losses[1]["weight"] == 0.5
    assert losses[1]["kwargs"]["sigmoid"] is True
    print("[OK]Loss profile positional overrides work")


def test_build_test_transforms_with_mask_transform_resize_binarize():
    """Test test-mode mask_transform resize+binarize pipeline creation."""
    cfg = Config()
    # build_test_transforms reads from cfg.data (after stage resolution in production)
    cfg.data.test.image = "dummy_image.h5"
    cfg.data.test.mask = "dummy_mask.h5"
    cfg.data.data_transform.resize = [1, 2, 2]
    cfg.data.data_transform.binarize = True
    cfg.data.data_transform.threshold = 0.0

    transforms = build_test_transforms(cfg)
    transform_names = [type(t).__name__ for t in transforms.transforms]

    assert "ResizeByFactord" in transform_names
    assert "Lambdad" in transform_names
    print("[OK]Test transforms include mask resize+binarize")


def test_build_test_transforms_applies_context_pad_to_image_and_mask_only():
    """Test test-mode context padding applies to image/mask but not label."""
    cfg = Config()
    cfg.data.dataloader.patch_size = [0, 0, 0]  # Isolate explicit context padding.
    cfg.data.test.image = "dummy_image.h5"
    cfg.data.test.label = "dummy_label.h5"
    cfg.data.test.mask = "dummy_mask.h5"
    cfg.data.image_transform.normalize = "none"
    cfg.data.data_transform.pad_size = [1, 2, 3]
    cfg.data.data_transform.pad_mode = "reflect"

    transforms = build_test_transforms(cfg)

    sample = {
        "image": np.ones((1, 2, 3, 4), dtype=np.float32),
        "label": np.ones((1, 2, 3, 4), dtype=np.float32),
        "mask": np.ones((1, 2, 3, 4), dtype=np.float32),
    }
    out = transforms(sample)
    image = out["image"].numpy() if hasattr(out["image"], "numpy") else np.asarray(out["image"])
    label = out["label"].numpy() if hasattr(out["label"], "numpy") else np.asarray(out["label"])
    mask = out["mask"].numpy() if hasattr(out["mask"], "numpy") else np.asarray(out["mask"])

    assert image.shape == (1, 4, 7, 10)
    assert label.shape == (1, 2, 3, 4)
    assert mask.shape == (1, 4, 7, 10)
    assert mask[0, 0, 0, 0] == 0.0
    print("[OK]Test transforms apply explicit context padding for image/mask")


def test_mask_binarize_uses_strict_greater_than_threshold():
    """Ensure mask binarization preserves zeros when threshold=0.0 (mask > 0)."""
    cfg = Config()
    cfg.data.dataloader.patch_size = [0, 0, 0]  # Disable padding for this unit test.
    # build_test_transforms reads from cfg.data (after stage resolution in production)
    cfg.data.test.image = "dummy_image.h5"
    cfg.data.test.mask = "dummy_mask.h5"
    cfg.data.image_transform.normalize = "none"
    cfg.data.data_transform.resize = None
    cfg.data.data_transform.binarize = True
    cfg.data.data_transform.threshold = 0.0

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
    print("[OK]Mask binarization uses strict > threshold semantics")


def test_build_tune_transforms_reads_mask_from_val_split():
    """Tune-mode transforms should load mask keys from data.val, not data.test."""
    cfg = Config()
    cfg.data.test.image = "dummy_test_image.h5"
    cfg.data.test.mask = None
    cfg.data.val.image = "dummy_val_image.h5"
    cfg.data.val.mask = "dummy_val_mask.h5"
    cfg.data.image_transform.normalize = "none"
    cfg.data.dataloader.patch_size = [0, 0, 0]

    transforms = build_test_transforms(cfg, mode="tune")
    sample = {
        "image": np.zeros((1, 2, 2, 2), dtype=np.float32),
        "mask": np.ones((1, 2, 2, 2), dtype=np.float32),
    }
    out = transforms(sample)

    assert torch.is_tensor(out["mask"])
    print("[OK]Tune transforms read mask from val split")


def test_build_tune_transforms_do_not_fallback_to_test_split():
    """Tune-mode transforms should not infer label/mask keys from data.test."""
    cfg = Config()
    cfg.data.test.image = "dummy_test_image.h5"
    cfg.data.test.label = "dummy_test_label.h5"
    cfg.data.test.mask = "dummy_test_mask.h5"
    cfg.data.val.image = None
    cfg.data.val.label = None
    cfg.data.val.mask = None
    cfg.data.image_transform.normalize = "none"
    cfg.data.dataloader.patch_size = [0, 0, 0]

    transforms = build_test_transforms(cfg, mode="tune")
    keyed_steps = [
        tuple(getattr(transform, "keys", ()))
        for transform in transforms.transforms
        if hasattr(transform, "keys")
    ]

    assert keyed_steps
    assert all("label" not in keys for keys in keyed_steps)
    assert all("mask" not in keys for keys in keyed_steps)
    print("[OK]Tune transforms do not fall back to test split keys")


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
        test_enabled_flags_require_explicit_opt_in(Path(tmp_dir))
    test_build_test_transforms_with_mask_transform_resize_binarize()

    print("\n" + "=" * 50)
    print("All Hydra config tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
