import sys
from types import SimpleNamespace

import numpy as np
import pytest

from connectomics.data.io import get_vol_shape, read_volume


class _FakeND2File:
    sizes = {"Z": 2, "C": 3, "Y": 4, "X": 5}
    array = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def asarray(self):
        return self.array


def test_read_volume_normalizes_nd2_to_channel_first(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nd2", SimpleNamespace(ND2File=_FakeND2File))
    path = tmp_path / "volume.nd2"
    path.touch()

    loaded = read_volume(str(path))

    expected = _FakeND2File.array.transpose(1, 0, 2, 3)
    assert loaded.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(loaded, expected)
    assert get_vol_shape(str(path)) == loaded.shape


def test_read_volume_rejects_multi_position_nd2(monkeypatch, tmp_path):
    class MultiPositionND2File(_FakeND2File):
        sizes = {"P": 2, "Z": 2, "Y": 4, "X": 5}
        array = np.zeros((2, 2, 4, 5), dtype=np.uint16)

    monkeypatch.setitem(sys.modules, "nd2", SimpleNamespace(ND2File=MultiPositionND2File))
    path = tmp_path / "tiles.nd2"
    path.touch()

    with pytest.raises(ValueError, match="one XY position/tile"):
        read_volume(str(path))
