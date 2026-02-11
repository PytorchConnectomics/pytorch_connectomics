from pathlib import Path

import pytest

from connectomics.config import load_config


PRESET_CONFIGS = [
    Path("tutorials/presets/aug_light.yaml"),
    Path("tutorials/presets/aug_realistic.yaml"),
    Path("tutorials/presets/aug_heavy.yaml"),
    Path("tutorials/presets/aug_superres.yaml"),
    Path("tutorials/presets/aug_instance.yaml"),
]


@pytest.mark.parametrize("config_path", PRESET_CONFIGS)
def test_augmentation_presets_load(config_path: Path) -> None:
    """All augmentation presets should load with the current Hydra schema."""
    cfg = load_config(config_path)

    assert cfg.system.training.batch_size > 0
    assert cfg.system.training.num_workers >= 0
    assert cfg.data.augmentation.preset in {"none", "some", "all"}

    if config_path.name == "aug_heavy.yaml":
        assert tuple(cfg.data.augmentation.motion_blur.sections) == (1, 3)
    if config_path.name == "aug_superres.yaml":
        assert tuple(cfg.data.augmentation.motion_blur.sections) == (1, 2)
