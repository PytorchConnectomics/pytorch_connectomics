#!/usr/bin/env python3
"""Run `prepare_volume.py` for one entry of `volumes.PENDING` (SLURM array index)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # `--index` is how the SLURM array addresses a volume; `--volume` is how
    # the cloud driver does, since it runs one named volume and never an array.
    ap.add_argument("--index", type=int, help="index into volumes.PENDING")
    ap.add_argument("--volume", help="volume name (see volumes.py)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if (a.index is None) == (a.volume is None):
        raise SystemExit("give exactly one of --index / --volume")
    name = a.volume or V.PENDING[a.index]
    p = V.plan(name)
    print(f"{name}: {p['native_shape']} {p['native_spacing_zyx']} -> "
          f"{p['shape']} {p['spacing_zyx']}", flush=True)

    cmd = [
        sys.executable, str(Path(__file__).parent / "prepare_volume.py"),
        "--input", str(p["source"]), "--output", str(p["output"]), *p["prepare_args"],
    ]
    if a.overwrite:
        cmd.append("--overwrite")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # A prepared volume that is all one value means the resample silently
    # produced nothing; the inference stage would not notice.
    import h5py
    import numpy as np

    with h5py.File(p["output"], "r") as f:
        d = f["main"]
        assert tuple(d.shape) == tuple(p["shape"]), (d.shape, p["shape"])
        mid = np.asarray(d[d.shape[0] // 2])
    print(f"OK {p['output']}  mid-plane mean {mid.mean():.2f} std {mid.std():.2f} "
          f"min {mid.min()} max {mid.max()}", flush=True)
    if mid.std() < 1.0:
        raise SystemExit(f"prepared volume mid-plane is flat (std {mid.std():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
