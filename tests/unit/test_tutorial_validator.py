"""All tutorial configs must satisfy the schema and portable-path contract."""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_SPEC = importlib.util.spec_from_file_location(
    "tutorial_validator",
    Path(__file__).resolve().parents[2] / "scripts/validate_tutorial_configs.py",
)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


@pytest.mark.parametrize(
    "path", ["/absolute/data.h5", r"C:\data\image.h5", r"\\server\data\image.h5"]
)
def test_rejects_absolute_paths_in_nested_values(path):
    assert validator._invalid_absolute_paths({"data": [{"image": path}]}) == [
        f"data[0].image: {path!r}"
    ]


def test_accepts_relative_placeholder_and_remote_paths():
    assert (
        validator._invalid_absolute_paths(
            ["datasets/image.h5", "/path/to/images.h5", "gs://bucket/image", "https://example.org"]
        )
        == []
    )


def test_explicit_glob_replaces_default(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tutorials").mkdir()
    (tmp_path / "tutorials/invalid.yaml").write_text("data: /invalid/path\n")
    (tmp_path / "selected.yaml").write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["validator", "--glob", "selected.yaml"])
    monkeypatch.setattr(validator, "load_config", lambda path: OmegaConf.create({}))
    monkeypatch.setattr(validator, "validate_runtime_coherence", lambda cfg: None)
    assert validator.main() == 0
    assert "Validated 1 tutorial configs" in capsys.readouterr().out


def test_unknown_workflow_is_rejected_instead_of_skipped(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom.yaml").write_text("large_decode: {}\n")
    monkeypatch.setattr(sys, "argv", ["validator", "--glob", "custom.yaml"])
    assert validator.main() == 1
    assert "failed to load" in capsys.readouterr().out


def test_inherited_absolute_path_is_rejected(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "selected.yaml").write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["validator", "--glob", "selected.yaml"])
    monkeypatch.setattr(
        validator, "load_config", lambda path: OmegaConf.create({"data": "/absolute/image.h5"})
    )
    monkeypatch.setattr(validator, "validate_runtime_coherence", lambda cfg: None)
    assert validator.main() == 1
    assert "absolute path must start with /path/to/" in capsys.readouterr().out
