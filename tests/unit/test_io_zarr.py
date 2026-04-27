from pathlib import Path

import numpy as np
import pytest

from connectomics.data.io import get_vol_shape, read_volume, save_volume, volume_exists
from connectomics.data.processing.distance import sdt_path_for_label

zarr = pytest.importorskip("zarr")


def test_sdt_path_for_label_preserves_zarr_backend():
    assert sdt_path_for_label("/tmp/labels.h5", mode="skeleton") == "/tmp/labels_skeleton.h5"
    assert (
        sdt_path_for_label("/tmp/labels.h5", mode="skeleton", cache_dir="/cache")
        == "/cache/labels_skeleton.h5"
    )
    assert (
        sdt_path_for_label("/tmp/data.zarr/seg", mode="skeleton") == "/tmp/data.zarr/seg_skeleton"
    )
    assert (
        sdt_path_for_label("/tmp/data.zarr/seg", mode="skeleton", cache_dir="/cache")
        == "/cache/data.zarr/seg_skeleton"
    )
    assert (
        sdt_path_for_label("/tmp/data.zarr/group/seg", mode="sdt") == "/tmp/data.zarr/group/seg_sdt"
    )
    assert (
        sdt_path_for_label("/tmp/data.zarr/group/seg", mode="sdt", cache_dir="/cache")
        == "/cache/data.zarr/group/seg_sdt"
    )
    assert sdt_path_for_label("/tmp/data.zarr", mode="skeleton") == "/tmp/data_skeleton.zarr"
    assert (
        sdt_path_for_label("/tmp/data.zarr", mode="skeleton", cache_dir="/cache")
        == "/cache/data_skeleton.zarr"
    )


def test_zarr_subkey_round_trip_and_shape(tmp_path: Path):
    path = tmp_path / "data.zarr" / "seg_skeleton"
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    save_volume(str(path), volume, file_format="zarr")

    loaded = read_volume(str(path))
    np.testing.assert_array_equal(loaded, volume)
    assert get_vol_shape(str(path)) == volume.shape
    assert volume_exists(str(path))


def test_detect_format_prefers_real_suffix_inside_zarr_directory(tmp_path: Path):
    h5_path = tmp_path / "data.zarr" / "seg_skeleton.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    save_volume(str(h5_path), np.ones((2, 2, 2), dtype=np.uint8), file_format="h5")

    loaded = read_volume(str(h5_path))

    assert loaded.shape == (2, 2, 2)
