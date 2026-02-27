from pathlib import Path

import numpy as np
import pytest

from connectomics.data.io import get_vol_shape, read_volume

tifffile = pytest.importorskip("tifffile")


def _write_tiff_pages(path: Path, volume: np.ndarray) -> None:
    with tifffile.TiffWriter(path) as writer:
        for z in range(volume.shape[0]):
            writer.write(volume[z])


def test_read_volume_stacks_multipage_tiff_series(tmp_path):
    tiff_path = tmp_path / "multipage.tif"
    volume = np.random.randint(0, 255, size=(5, 16, 12), dtype=np.uint8)
    _write_tiff_pages(tiff_path, volume)

    loaded = read_volume(str(tiff_path))
    assert loaded.shape == volume.shape
    np.testing.assert_array_equal(loaded, volume)

    shape = get_vol_shape(str(tiff_path))
    assert shape == volume.shape


def test_read_volume_keeps_single_page_tiff_2d(tmp_path):
    tiff_path = tmp_path / "single_page.tif"
    image = np.random.randint(0, 255, size=(16, 12), dtype=np.uint8)
    tifffile.imwrite(tiff_path, image)

    loaded = read_volume(str(tiff_path))
    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)

    shape = get_vol_shape(str(tiff_path))
    assert shape == image.shape
