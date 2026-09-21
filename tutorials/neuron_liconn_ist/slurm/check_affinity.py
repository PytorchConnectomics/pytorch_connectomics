"""Sanity-check a freshly written affinity volume before spending a decode on it.

Usage: check_affinity.py <affinity.h5> [reference-p25,p50,p75]

Fails loudly on a constant or degenerate mid-plane. A decode of a dead affinity
costs ~43 min and ~147 GB and produces a plausible-looking but meaningless
segmentation, so this runs first.
"""
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np

path = sys.argv[1]
reference = sys.argv[2] if len(sys.argv) > 2 else "0.26/0.53/0.68"

with h5py.File(path, "r") as handle:
    data = handle["main"]
    print(f"affinity {path}")
    print(f"  shape {data.shape} dtype {data.dtype} size {os.path.getsize(path) / 1e9:.2f} GB")
    mid = np.asarray(data[:, data.shape[1] // 2]).astype(np.float32)

q25, q50, q75 = np.percentile(mid, [25, 50, 75])
print(f"  mid-plane p25/p50/p75 = {q25:.3f}/{q50:.3f}/{q75:.3f}   (eb2 val reference: {reference})")
print(f"  mid-plane min/max = {mid.min():.3f}/{mid.max():.3f}")

# scale_sigmoid compression means the stored value never reaches 0 or 1; a plain
# sigmoid would, and would mean the activation changed under us.
if mid.std() <= 0.01:
    sys.exit("FAIL: affinity mid-plane is constant -- do not decode this")
if mid.max() > 0.95:
    sys.exit(f"FAIL: max {mid.max():.3f} exceeds the scale_sigmoid range -- activation changed")
print("OK")
