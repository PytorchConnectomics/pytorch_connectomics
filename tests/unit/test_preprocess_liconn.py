import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "preprocess_liconn.py"
SPEC = importlib.util.spec_from_file_location("preprocess_liconn", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preprocess_liconn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocess_liconn)


def test_preprocess_xy_plane_uses_requested_percentiles(monkeypatch):
    plane = np.arange(100, dtype=np.uint16).reshape(10, 10)
    captured = {}

    def fake_clahe(image, *, clip_limit):
        captured["image"] = image.copy()
        captured["clip_limit"] = clip_limit
        return image

    monkeypatch.setattr(preprocess_liconn.exposure, "equalize_adapthist", fake_clahe)
    output = preprocess_liconn.preprocess_xy_plane(plane, clip_intensity_range=None)

    low, high = np.percentile(plane.astype(np.float32), (1.0, 99.0))
    expected = np.clip((plane.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    np.testing.assert_allclose(captured["image"], expected)
    assert captured["clip_limit"] == 0.03
    np.testing.assert_array_equal(output, np.rint(expected * 255.0).astype(np.uint8))


def test_preprocess_xy_plane_uses_fixed_intensity_range(monkeypatch):
    plane = np.array([[100, 120], [235, 400]], dtype=np.uint16)
    captured = {}

    def fake_clahe(image, *, clip_limit):
        captured["image"] = image.copy()
        captured["clip_limit"] = clip_limit
        return image

    monkeypatch.setattr(preprocess_liconn.exposure, "equalize_adapthist", fake_clahe)
    output = preprocess_liconn.preprocess_xy_plane(
        plane,
        clip_intensity_range=(120.0, 350.0),
    )

    expected = np.clip((plane.astype(np.float32) - 120.0) / 230.0, 0.0, 1.0)
    np.testing.assert_allclose(captured["image"], expected)
    assert captured["clip_limit"] == 0.03
    np.testing.assert_array_equal(output, np.rint(expected * 255.0).astype(np.uint8))


def test_preprocess_xy_plane_rejects_invalid_fixed_range():
    with pytest.raises(ValueError, match="low < high"):
        preprocess_liconn.preprocess_xy_plane(
            np.ones((4, 4), dtype=np.uint16),
            clip_intensity_range=(350.0, 120.0),
        )


def test_volume_average_uint8_uses_aligned_zyx_blocks():
    volume = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6)

    output = preprocess_liconn.volume_average_uint8(volume, (2, 2, 3))

    expected = np.rint(
        volume.astype(np.uint32).reshape(1, 2, 2, 2, 2, 3).mean(axis=(1, 3, 5))
    ).astype(np.uint8)
    np.testing.assert_array_equal(output, expected)


def test_volume_average_uint8_rejects_implicit_crop():
    volume = np.zeros((3, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="must be divisible"):
        preprocess_liconn.volume_average_uint8(volume, (2, 2, 2))


def test_preprocess_liconn_selects_channel_and_writes_uint8_h5(monkeypatch, tmp_path):
    input_path = tmp_path / "input.nd2"
    input_path.touch()
    output_path = tmp_path / "output.h5"
    raw = np.stack(
        [
            np.zeros((2, 16, 16), dtype=np.uint16),
            np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16),
        ]
    )
    monkeypatch.setattr(preprocess_liconn, "read_volume", lambda _: raw)

    result = preprocess_liconn.preprocess_liconn(input_path, output_path, channel=1)

    assert result == output_path
    with h5py.File(output_path, "r") as handle:
        data = handle["main"][:]
        assert data.shape == (2, 16, 16)
        assert data.dtype == np.uint8
        assert handle["main"].attrs["axes"] == "ZYX"
        assert handle["main"].attrs["source_channel"] == 1
        np.testing.assert_array_equal(
            handle["main"].attrs["clip_intensity_range"], (120.0, 350.0)
        )


def test_preprocess_liconn_streams_volume_average_and_records_spacing(monkeypatch, tmp_path):
    input_path = tmp_path / "input.nd2"
    input_path.touch()
    output_path = tmp_path / "output.h5"
    raw = np.arange(4 * 4 * 6, dtype=np.uint8).reshape(4, 4, 6)
    monkeypatch.setattr(preprocess_liconn, "read_volume", lambda _: raw)
    monkeypatch.setattr(
        preprocess_liconn,
        "preprocess_xy_plane",
        lambda plane, **_: plane,
    )

    preprocess_liconn.preprocess_liconn(
        input_path,
        output_path,
        downsample_factor=(2, 2, 3),
        input_spacing_nm=(12.0, 9.0, 9.0),
    )

    with h5py.File(output_path, "r") as handle:
        output = handle["main"]
        np.testing.assert_array_equal(
            output[:], preprocess_liconn.volume_average_uint8(raw, (2, 2, 3))
        )
        np.testing.assert_array_equal(output.attrs["source_shape_zyx"], raw.shape)
        np.testing.assert_array_equal(output.attrs["downsample_factor_zyx"], (2, 2, 3))
        np.testing.assert_allclose(output.attrs["input_spacing_nm_zyx"], (12.0, 9.0, 9.0))
        np.testing.assert_allclose(output.attrs["spacing_nm_zyx"], (24.0, 18.0, 27.0))
        assert output.attrs["downsample_method"] == "aligned_arithmetic_volume_average"
        assert output.attrs["downsample_rounding"] == "nearest_even"


def test_preprocess_liconn_refuses_existing_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.nd2"
    input_path.touch()
    output_path = tmp_path / "output.h5"
    output_path.touch()
    monkeypatch.setattr(preprocess_liconn, "read_volume", lambda _: np.zeros((2, 8, 8)))

    with pytest.raises(FileExistsError, match="--overwrite"):
        preprocess_liconn.preprocess_liconn(input_path, output_path)
