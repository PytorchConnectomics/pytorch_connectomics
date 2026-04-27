"""
Lazy HDF5-backed volume dataset for random patch sampling.

Keeps HDF5 handles closed at init time (only reads shape/dtype), and opens
them lazily per-worker on first access. This is required because h5py file
handles are not fork-safe: a handle opened in the parent process becomes
corrupted after DataLoader workers are forked.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import h5py
import numpy as np

from .dataset_volume_zarr_lazy import LazyZarrVolumeDataset

logger = logging.getLogger(__name__)


def _split_h5_path(path: str) -> Tuple[str, Optional[str]]:
    """Split "file.h5/dataset_key" into ("file.h5", "dataset_key").

    If no subkey is provided, returns (path, None) and the dataset name
    is resolved on open (first dataset in the file).
    """
    s = str(path)
    for ext in (".h5", ".hdf5"):
        idx = s.find(ext)
        if idx >= 0:
            store = s[: idx + len(ext)]
            sub = s[idx + len(ext) :].strip("/") or None
            return store, sub
    return s, None


class _H5LazyArray:
    """Pickleable, fork-safe lazy wrapper over an HDF5 dataset.

    Exposes the minimal interface LazyZarrVolumeDataset needs:
    ``shape``, ``ndim``, and ``__getitem__``. The underlying file handle
    is reopened lazily whenever the process id changes (i.e. after a
    DataLoader worker fork) or when the wrapper is unpickled.
    """

    __slots__ = ("path", "key", "shape", "dtype", "ndim", "_pid", "_file", "_ds")

    def __init__(self, path: str, key: str, shape: Tuple[int, ...], dtype: np.dtype):
        self.path = path
        self.key = key
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)
        self._pid: Optional[int] = None
        self._file: Optional[h5py.File] = None
        self._ds = None

    def _ensure_open(self):
        pid = os.getpid()
        if self._file is not None and self._pid == pid:
            return
        self._file = h5py.File(self.path, "r")
        self._ds = self._file[self.key]
        self._pid = pid

    def __getitem__(self, idx):
        self._ensure_open()
        return self._ds[idx]

    def __getstate__(self):
        return {
            "path": self.path,
            "key": self.key,
            "shape": self.shape,
            "dtype": self.dtype,
            "ndim": self.ndim,
        }

    def __setstate__(self, state):
        self.path = state["path"]
        self.key = state["key"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.ndim = state["ndim"]
        self._pid = None
        self._file = None
        self._ds = None


class LazyH5VolumeDataset(LazyZarrVolumeDataset):
    """Lazy HDF5 dataset that samples random crops directly from .h5 files.

    Mirrors :class:`LazyZarrVolumeDataset` but opens HDF5 stores instead of
    Zarr stores. Paths may point at a file ("vol.h5") — the first dataset
    in the file is used — or include an explicit dataset key
    ("vol.h5/main").
    """

    def _open_array(self, path):
        if path is None:
            return None
        store, key = _split_h5_path(str(path))
        with h5py.File(store, "r") as fh:
            if key is None:
                key = list(fh.keys())[0]
            ds = fh[key]
            shape = tuple(ds.shape)
            dtype = ds.dtype
        return _H5LazyArray(store, key, shape, dtype)


__all__ = ["LazyH5VolumeDataset"]
