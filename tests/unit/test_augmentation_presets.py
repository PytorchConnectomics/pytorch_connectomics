from pathlib import Path

import pytest

from connectomics.config import load_config


PRESET_CONFIGS = [
    pytest.param(
        Path("tutorials/bases/augmentation_profiles.yaml"),
        True,
        id="augmentation_profiles",
    ),
    pytest.param(Path("tutorials/presets/aug_light.yaml"), False, id="aug_light_removed"),
    pytest.param(Path("tutorials/presets/aug_realistic.yaml"), False, id="aug_realistic_legacy_scalar_sections"),
    pytest.param(Path("tutorials/presets/aug_heavy.yaml"), False, id="aug_heavy_legacy_scalar_ranges"),
    pytest.param(Path("tutorials/presets/aug_superres.yaml"), False, id="aug_superres_legacy_scalar_ranges"),
    pytest.param(Path("tutorials/presets/aug_instance.yaml"), False, id="aug_instance_removed"),
]


@pytest.mark.parametrize(("config_path", "should_load"), PRESET_CONFIGS)
def test_augmentation_presets_load(config_path: Path, should_load: bool) -> None:
    """Preset configs either load or fail with explicit schema errors."""
    if not should_load:
        with pytest.raises(Exception):
            load_config(config_path)
        return

    cfg = load_config(config_path)

    assert cfg.data.dataloader.batch_size > 0
    assert cfg.system.num_workers >= 0
    assert cfg.data.augmentation.preset in {"none", "some", "all"}
