"""The viewer defaults to loopback and warns on explicit public exposure."""

import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "viewer_script", Path(__file__).resolve().parents[2] / "scripts/visualize_neuroglancer.py"
)
assert _SPEC is not None and _SPEC.loader is not None
viewer_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viewer_script)


def test_viewer_defaults_to_loopback(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["viewer"])
    assert viewer_script.parse_args().bind_address == "127.0.0.1"
    assert (
        inspect.signature(viewer_script.visualize_volumes).parameters["ip"].default == "127.0.0.1"
    )


def test_public_binding_requires_explicit_flag_and_warns(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["viewer", "--bind-address", "0.0.0.0"])
    args = viewer_script.parse_args()
    neuroglancer = MagicMock()
    monkeypatch.setattr(viewer_script, "neuroglancer", neuroglancer)
    monkeypatch.setattr(viewer_script, "create_neuroglancer_layer", MagicMock())
    viewer_script.visualize_volumes({"image": (np.zeros((2, 2, 2)), "image")}, ip=args.bind_address)
    neuroglancer.set_server_bind_address.assert_called_once_with(
        bind_address="0.0.0.0", bind_port=9999
    )
    assert "unauthenticated" in capsys.readouterr().out
