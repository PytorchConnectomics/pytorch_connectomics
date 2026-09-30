"""Tests for decode_abiss external wrapper."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomics.config import load_config, resolve_default_profiles
from connectomics.decoding import abiss_runner, decode_abiss
from connectomics.decoding.decoders.abiss import _resolve_python_script_path


def test_decode_abiss_with_list_command_writes_npy_output(tmp_path):
    pred = np.zeros((3, 6, 8, 10), dtype=np.float32)
    pred[0, 1:4, 2:6, 3:8] = 0.9

    command = [
        sys.executable,
        "-c",
        (
            "import h5py, numpy as np; "
            "x = h5py.File('{input_h5}', 'r')['{input_dataset}'][:]; "
            "y = (x[0] > 0.5).astype(np.uint64); "
            "np.save('{output_npy}', y)"
        ),
    ]

    seg = decode_abiss(pred, command=command, abiss_home=str(tmp_path))
    assert seg.shape == (6, 8, 10)
    assert np.issubdtype(seg.dtype, np.integer)
    assert seg.max() == 1
    assert seg[2, 3, 4] == 1


def test_decode_abiss_with_string_command_writes_h5_output(tmp_path):
    pred = np.zeros((3, 5, 7, 9), dtype=np.float32)
    pred[1, 1:4, 2:5, 3:7] = 1.0

    command = (
        f'{sys.executable} -c "'
        "import h5py, numpy as np; "
        "x = h5py.File('{input_h5}', 'r')['{input_dataset}'][:]; "
        "y = (x[1] > 0.5).astype(np.uint64); "
        "f = h5py.File('{output_h5}', 'w'); "
        "f.create_dataset('{output_dataset}', data=y); "
        'f.close()"'
    )

    seg = decode_abiss(pred, command=command, abiss_home=str(tmp_path))
    assert seg.shape == (5, 7, 9)
    assert seg.max() == 1
    assert seg[2, 3, 4] == 1


def test_decode_abiss_raises_if_output_missing(tmp_path):
    pred = np.zeros((3, 4, 4, 4), dtype=np.float32)

    command = [sys.executable, "-c", "print('no output written')"]

    with pytest.raises(FileNotFoundError, match="did not produce output file"):
        decode_abiss(pred, command=command, abiss_home=str(tmp_path))


def test_decode_abiss_requires_explicit_installation(monkeypatch):
    monkeypatch.delenv("PYTC_ABISS_HOME", raising=False)
    with pytest.raises(RuntimeError, match="abiss_home or PYTC_ABISS_HOME"):
        decode_abiss(np.zeros((3, 2, 2, 2)), command=[sys.executable, "-c", "pass"])


def test_decode_abiss_with_relative_script_command_outside_repo_cwd(monkeypatch, tmp_path):
    installation = tmp_path / "installation"
    installation.mkdir()
    (installation / "decode.py").write_text(
        "import sys, numpy as np\n"
        "x = np.load(sys.argv[1])\n"
        "np.save(sys.argv[2], (x[0] > 0.5).astype(np.uint64))\n"
    )
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv("PYTC_ABISS_HOME", str(installation))
    pred = np.ones((3, 6, 8, 10), dtype=np.float32)
    seg = decode_abiss(pred, command=[sys.executable, "decode.py", "{input_npy}", "{output_npy}"])
    assert seg.shape == (6, 8, 10)
    assert np.issubdtype(seg.dtype, np.integer)
    assert np.all(seg == 1)


def test_decode_abiss_explicit_home_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTC_ABISS_HOME", str(tmp_path / "missing"))
    command = [
        sys.executable,
        "-c",
        "import numpy as np; np.save('{output_npy}', np.ones((2, 2, 2), dtype=np.uint64))",
    ]
    seg = decode_abiss(np.ones((3, 2, 2, 2)), command=command, abiss_home=str(tmp_path))
    assert np.all(seg == 1)


@pytest.mark.parametrize("as_string", [False, True])
def test_python_module_command_is_not_rewritten(tmp_path, as_string):
    tokens = [sys.executable, "-m", "connectomics.decoding.abiss_runner", "--help"]
    command = shlex.join(tokens) if as_string else tokens
    assert _resolve_python_script_path(command, [tmp_path]) == command


def test_default_template_invokes_packaged_abiss_runner(monkeypatch, tmp_path):
    templates = (
        Path(__file__).resolve().parents[2]
        / "connectomics/config/templates/decoding_templates.yaml"
    )
    config_path = tmp_path / "decode.yaml"
    config_path.write_text(
        f"_base_: {templates.as_posix()}\n"
        "default:\n"
        "  decoding:\n"
        "    steps:\n"
        "      - template: decoding_abiss\n"
    )
    cfg = resolve_default_profiles(load_config(config_path), mode="test")
    step = cfg.decoding.steps[0]
    assert step.name == "decode_abiss"
    installation = tmp_path / "abiss"
    binary = installation / "build/ws"
    binary.parent.mkdir(parents=True)
    binary.touch()
    predictions = np.ones((3, 2, 3, 4), dtype=np.float32)
    expected = np.arange(24, dtype=np.uint64).reshape(2, 3, 4)
    commands = []

    def fake_ws(**kwargs):
        assert kwargs["ws_binary"] == binary
        np.testing.assert_array_equal(kwargs["predictions_czyx"][0], predictions)
        assert kwargs["ws_size_threshold"] == 800
        return expected

    def run_packaged_module(command, *, shell, env, cwd, check, timeout):
        assert shell and check
        tokens = shlex.split(command)
        assert tokens[:3] == [sys.executable, "-m", "connectomics.decoding.abiss_runner"]
        assert tokens[3:5] == ["--abiss-home", str(installation)]
        commands.append(tokens)
        # Exercise the module's real argument parser and HDF5 IO, substituting
        # only the unavailable external watershed binary.
        monkeypatch.setattr(sys, "argv", [abiss_runner.__file__, *tokens[3:]])
        assert abiss_runner.main() == 0

    monkeypatch.setattr(abiss_runner, "_run_abiss_ws", fake_ws)
    monkeypatch.setattr("connectomics.decoding.decoders.abiss.subprocess.run", run_packaged_module)
    actual = decode_abiss(predictions, abiss_home=str(installation), **step.kwargs)
    assert len(commands) == 1
    np.testing.assert_array_equal(actual, expected)
