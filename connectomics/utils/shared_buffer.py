"""Private file-backed arrays and safe process pools.

Storage is reserved before mapping where fallocate is supported. The truncate
fallback does not reserve storage; callers must ensure sufficient free space.
"""

import errno
import mmap
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def default_buffer_dir():
    return (
        "/dev/shm"
        if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
        else tempfile.gettempdir()
    )


class SharedArray:
    """Own a mapped array and its optional attachment path.

    Arrays use a memoryview base so closing with live views safely defers unmapping
    until those views are collected. Drop views before close for immediate release.
    """

    def __init__(self, path, mapping, shape, dtype, owner):
        self.path = path
        self._mapping = mapping
        self._owner = owner
        self.array = np.frombuffer(mapping, dtype=dtype).reshape(shape)

    @classmethod
    def create(cls, shape, dtype, buffer_dir=None):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        fd, path = tempfile.mkstemp(
            dir=buffer_dir or default_buffer_dir(), prefix="pytc-", suffix=".buf"
        )
        mapping = None
        try:
            try:
                if hasattr(os, "posix_fallocate"):
                    os.posix_fallocate(fd, 0, size)
                else:
                    os.ftruncate(fd, size)
            except OSError as exc:
                if exc.errno not in (errno.EOPNOTSUPP, errno.EINVAL):
                    raise
                os.ftruncate(fd, size)
            mapping = mmap.mmap(fd, size)
            return cls(path, mapping, shape, dtype, True)
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.unlink(path)
            raise
        finally:
            os.close(fd)

    def release_path(self):
        if self._owner:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self._owner = False

    def close(self):
        self.release_path()
        self.array = None
        if self._mapping is not None:
            try:
                self._mapping.close()
            except BufferError:
                pass
            self._mapping = None


def attach(path, shape, dtype, writable=False):
    with open(path, "r+b" if writable else "rb") as stream:
        mapping = mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        )
    try:
        return SharedArray(path, mapping, shape, dtype, False)
    except BaseException:
        mapping.close()
        raise


def process_parallel_available():
    return os.name == "posix" and not mp.current_process().daemon


def process_pool_context():
    return mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")


def run_process_tasks(fn, tasks, num_workers, preload=()):
    """Run ordered tasks with safe worker-failure propagation.

    On forkserver contexts this replaces multiprocessing's process-global preload
    list, including any list configured by the application. It only affects a
    server started after this call; an already running server is unchanged.
    """
    context = process_pool_context()
    if context.get_start_method() == "forkserver":
        context.set_forkserver_preload(list(preload))
    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=context)
    futures = []
    try:
        futures = [executor.submit(fn, task) for task in tasks]
        return [future.result() for future in futures]
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
