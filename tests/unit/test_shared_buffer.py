"""Storage ownership and worker failure contracts."""

import errno
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomics.utils import shared_buffer as sb


def _fd_count():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0


@pytest.mark.parametrize("failure", ["reserve", "truncate", "map", "view"])
def test_create_failure(tmp_path, monkeypatch, failure):
    before = _fd_count()

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected")

    if failure == "reserve":
        monkeypatch.setattr(os, "posix_fallocate", fail, raising=False)
    elif failure == "truncate":
        monkeypatch.delattr(os, "posix_fallocate", raising=False)
        monkeypatch.setattr(os, "ftruncate", fail)
    elif failure == "map":
        monkeypatch.setattr(sb.mmap, "mmap", fail)
    else:
        monkeypatch.setattr(sb.np, "frombuffer", fail)
    with pytest.raises(OSError, match="injected"):
        sb.SharedArray.create((16,), np.uint32, tmp_path)
    assert not list(tmp_path.iterdir())
    assert _fd_count() == before


def test_unsupported_reservation_and_lifetime(tmp_path, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(os, "posix_fallocate", unsupported, raising=False)
    buf = sb.SharedArray.create((4,), np.uint32, tmp_path)
    buf.array[:] = [1, 2, 3, 4]
    attached = sb.attach(buf.path, (4,), np.uint32)
    np.testing.assert_array_equal(attached.array, buf.array)
    attached.close()
    array = buf.array
    buf.release_path()
    buf.release_path()
    buf.close()
    buf.close()
    np.testing.assert_array_equal(array, [1, 2, 3, 4])
    assert not list(tmp_path.iterdir())


def _run_child(tmp_path, code):
    script = tmp_path / "child.py"
    root = Path(__file__).resolve().parents[2]
    script.write_text(
        "import connectomics\n"
        "from pathlib import Path\n"
        f"assert Path(connectomics.__file__).resolve().is_relative_to(Path({str(root)!r}))\n" + code
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not hasattr(os, "posix_fallocate") or sys.platform != "linux", reason="Linux reservation test"
)
def test_real_reservation_failure(tmp_path):
    buf = tmp_path / "buf"
    buf.mkdir()
    _run_child(
        tmp_path,
        f"""
import resource, signal, os
from connectomics.utils.shared_buffer import SharedArray
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
try:
    SharedArray.create((16 << 20,), "u1", {str(buf)!r})
except OSError:
    assert not os.listdir({str(buf)!r})
else:
    raise AssertionError("Reservation unexpectedly succeeded")
""",
    )


@pytest.mark.skipif(not sb.process_parallel_available(), reason="Processes unavailable")
def test_worker_failure_and_order(tmp_path):
    _run_child(
        tmp_path,
        """
import os
from concurrent.futures.process import BrokenProcessPool
from connectomics.utils.shared_buffer import run_process_tasks

def die(value):
    os._exit(1)

if __name__ == "__main__":
    assert run_process_tasks(abs, [-3, -1, -2], 2) == [3, 1, 2]
    try:
        run_process_tasks(die, [1], 1)
    except BrokenProcessPool:
        pass
    else:
        raise AssertionError("Worker death did not propagate")
""",
    )
