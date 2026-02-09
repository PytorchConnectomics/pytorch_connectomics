from pathlib import Path

import pytest

from connectomics.config import load_config


def test_load_config_with_base_relative_path(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """
system:
  seed: 123
model:
  architecture: monai_unet
"""
    )

    child = tmp_path / "child.yaml"
    child.write_text(
        """
_base_: base.yaml
system:
  seed: 42
"""
    )

    cfg = load_config(child)
    assert cfg.system.seed == 42
    assert cfg.model.architecture == "monai_unet"


def test_load_config_with_multiple_bases_order(tmp_path: Path):
    base_a = tmp_path / "base_a.yaml"
    base_a.write_text(
        """
system:
  seed: 1
model:
  architecture: monai_unet
"""
    )

    base_b = tmp_path / "base_b.yaml"
    base_b.write_text(
        """
system:
  seed: 2
model:
  architecture: mednext
"""
    )

    child = tmp_path / "child.yaml"
    child.write_text(
        """
_base_:
  - base_a.yaml
  - base_b.yaml
system:
  seed: 3
"""
    )

    cfg = load_config(child)
    # Later bases override earlier ones, child overrides both.
    assert cfg.system.seed == 3
    assert cfg.model.architecture == "mednext"


def test_load_config_with_cyclic_base_raises(tmp_path: Path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("_base_: b.yaml\n")
    b.write_text("_base_: a.yaml\n")

    with pytest.raises(ValueError, match="cyclic _base_ config inheritance"):
        load_config(a)
