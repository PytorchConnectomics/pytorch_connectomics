"""ABISS output width, argv compatibility, and subprocess failure contracts."""

import importlib.util
import subprocess
import weakref
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[2] / "scripts/run_abiss_volume.py"
    spec = importlib.util.spec_from_file_location("run_abiss_volume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
@pytest.mark.parametrize("with_halo", [False, True])
def test_reader(script, tmp_path, dtype, with_halo):
    shape = (3, 4, 5)
    values = np.arange(60, dtype=dtype).reshape(shape, order="F")
    disk = np.pad(values, 1) if with_halo else values
    path = tmp_path / "seg.data"
    disk.ravel(order="F").tofile(path)
    actual = script._read_segmentation_xyz(path, shape, dtype=dtype)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, values)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Unexpected ABISS segmentation file size"):
        script._read_segmentation_xyz(path, shape, dtype=dtype)


def test_auto_bound(script):
    assert script._resolve_seg_dtype("auto", 2**32 - 1 - 60, (3, 4, 5)) == np.uint32
    assert script._resolve_seg_dtype("auto", 2**32 - 60, (3, 4, 5)) == np.uint64
    assert script._resolve_seg_dtype("auto", 2**32, (3, 4, 5)) == np.uint64
    with pytest.raises(ValueError, match="Unknown segmentation dtype"):
        script._resolve_seg_dtype("uint16", 0, (3, 4, 5))


def run_kwargs(tmp_path):
    return dict(
        predictions_czyx=np.ones((3, 5, 4, 3), dtype=np.float32),
        ws_binary=Path("/fake/ws"),
        ws_high_threshold=0.9,
        ws_low_threshold=0.1,
        ws_size_threshold=10,
        ws_dust_threshold=0,
        boundary_flags=[1] * 6,
        offset=7,
        workdir=tmp_path,
    )


@pytest.mark.parametrize("dtype", ["uint64", "uint32", "auto"])
@pytest.mark.parametrize("batch", [False, True])
def test_argv_and_callback(script, tmp_path, monkeypatch, dtype, batch):
    expected_dtype = np.uint64 if dtype == "uint64" else np.uint32
    seen = []

    def fake_run(cmd, cwd, check):
        expected = [
            "/fake/ws",
            str(tmp_path / "param.txt"),
            str(tmp_path / "aff.raw"),
            "0.9",
            "0.1",
            "10",
            "0",
            script._ABISS_TAG,
        ]
        if batch:
            expected += ["max", "0.2", "0.4", "0.6"]
        if expected_dtype == np.uint32:
            expected += ["--seg-dtype=uint32"]
        assert cmd == expected
        assert check is True
        for suffix in ("_0", "_1", "_2") if batch else ("",):
            np.arange(60, dtype=expected_dtype).tofile(
                Path(cwd) / f"seg_{script._ABISS_TAG}{suffix}.data"
            )

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    kwargs = run_kwargs(tmp_path)
    if dtype != "uint64":  # Exercise the actual default argument too.
        kwargs["seg_dtype"] = dtype
    if batch:
        kwargs.update(
            ws_merge_function="max",
            ws_merge_thresholds=[0.2, 0.4, 0.6],
            on_batch_result=lambda i, mt, seg: seen.append((i, mt, seg.dtype)),
        )
    result = script._run_abiss_ws(**kwargs)
    if batch:
        assert result == {}
        assert seen == [(i, mt, expected_dtype) for i, mt in enumerate([0.2, 0.4, 0.6])]
    else:
        assert result.dtype == expected_dtype


def test_failed_binary_never_calls_callback(script, tmp_path):
    binary = tmp_path / "fail_ws"
    binary.write_text("#!/bin/sh\nexit 3\n")
    binary.chmod(0o755)
    kwargs = run_kwargs(tmp_path)
    seen = []
    kwargs.update(
        ws_binary=binary,
        seg_dtype="uint32",
        ws_merge_thresholds=[0.1, 0.5, 0.9],
        on_batch_result=lambda *args: seen.append(args),
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        script._run_abiss_ws(**kwargs)
    assert error.value.returncode == 3
    assert seen == []


def test_input_released_before_subprocess(script, tmp_path, monkeypatch):
    kwargs = run_kwargs(tmp_path)
    predictions = kwargs.pop("predictions_czyx")
    reference = weakref.ref(predictions)
    holder = [predictions]
    del predictions

    def fake_run(cmd, cwd, check):
        assert reference() is None
        np.zeros(60, dtype=np.uint64).tofile(Path(cwd) / f"seg_{script._ABISS_TAG}.data")

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    script._run_abiss_ws(predictions_czyx=holder, **kwargs)
    assert holder == []
