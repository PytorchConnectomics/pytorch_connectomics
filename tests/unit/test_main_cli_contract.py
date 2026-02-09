import sys

import pytest

from connectomics.training.lit.utils import parse_args


def _parse_with_argv(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["scripts/main.py", *argv])
    return parse_args()


@pytest.mark.parametrize("mode", ["train", "test", "tune", "tune-test"])
def test_parse_args_accepts_documented_modes(monkeypatch, mode):
    args = _parse_with_argv(monkeypatch, ["--mode", mode])
    assert args.mode == mode


def test_parse_args_rejects_infer_mode(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_with_argv(monkeypatch, ["--mode", "infer"])


def test_parse_args_fast_dev_run_default_and_explicit(monkeypatch):
    args = _parse_with_argv(monkeypatch, [])
    assert args.fast_dev_run == 0

    args = _parse_with_argv(monkeypatch, ["--fast-dev-run"])
    assert args.fast_dev_run == 1

    args = _parse_with_argv(monkeypatch, ["--fast-dev-run", "3"])
    assert args.fast_dev_run == 3


def test_parse_args_preserves_overrides_passthrough(monkeypatch):
    args = _parse_with_argv(
        monkeypatch,
        [
            "--config",
            "tutorials/lucchi++.yaml",
            "data.batch_size=8",
            "optimization.max_epochs=3",
        ],
    )

    assert args.config == "tutorials/lucchi++.yaml"
    assert args.overrides == ["data.batch_size=8", "optimization.max_epochs=3"]


def test_parse_args_demo_mode_requires_no_config(monkeypatch):
    args = _parse_with_argv(monkeypatch, ["--demo"])
    assert args.demo is True
    assert args.config is None
