"""Tests for decode_abiss external wrapper."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from connectomics.decoding import decode_abiss


def test_decode_abiss_with_list_command_writes_npy_output():
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

    seg = decode_abiss(pred, command=command)
    assert seg.shape == (6, 8, 10)
    assert np.issubdtype(seg.dtype, np.integer)
    assert seg.max() == 1
    assert seg[2, 3, 4] == 1


def test_decode_abiss_with_string_command_writes_h5_output():
    pred = np.zeros((3, 5, 7, 9), dtype=np.float32)
    pred[1, 1:4, 2:5, 3:7] = 1.0

    command = (
        f"{sys.executable} -c \""
        "import h5py, numpy as np; "
        "x = h5py.File('{input_h5}', 'r')['{input_dataset}'][:]; "
        "y = (x[1] > 0.5).astype(np.uint64); "
        "f = h5py.File('{output_h5}', 'w'); "
        "f.create_dataset('{output_dataset}', data=y); "
        "f.close()\""
    )

    seg = decode_abiss(pred, command=command)
    assert seg.shape == (5, 7, 9)
    assert seg.max() == 1
    assert seg[2, 3, 4] == 1


def test_decode_abiss_raises_if_output_missing():
    pred = np.zeros((3, 4, 4, 4), dtype=np.float32)

    command = [sys.executable, "-c", "print('no output written')"]

    with pytest.raises(FileNotFoundError, match="did not produce output file"):
        decode_abiss(pred, command=command)

