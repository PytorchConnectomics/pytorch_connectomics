#!/usr/bin/env python3
"""Convert a PyTC affinity prediction (BANIS layout) to ABISS layout.

Input:  HDF5 dataset of shape (C, Z, Y, X), C >= 3, raw logits, channels ordered
        z, y, x, each edge (i, i+1) stored at the *source* voxel i ("x -> x+1").
Output: HDF5 dataset of shape (3, Z, Y, X), float32 probabilities, channels ordered
        x, y, z, each edge (i-1, i) stored at the *destination* voxel i ("x -> x-1"),
        which is what ABISS ws reads (aff[x][y][z][0] = edge between x-1 and x).
        The exposed face at index 0 along each channel's axis is set to 0.

Only the first 3 (short-range) channels are kept. Needs numpy + h5py.

  python pytc_aff_to_abiss.py pred.h5 pred_abiss.h5 [--dataset main] [--uint8]
"""

import argparse

import h5py
import numpy as np


def convert(pred_czyx, already_sigmoid=False):
    """(C,Z,Y,X) BANIS-order source-stored logits -> (3,Z,Y,X) xyz-order destination-stored probs."""
    out = np.empty((3,) + pred_czyx.shape[1:], dtype=np.float32)
    # output channel k = edge along x, y, z; its source channel and ZYX axis
    for k, (src_c, axis) in enumerate([(2, 2), (1, 1), (0, 0)]):
        a = np.asarray(pred_czyx[src_c], dtype=np.float32)
        if not already_sigmoid:
            a = 1.0 / (1.0 + np.exp(-a))
        a = np.roll(a, shift=1, axis=axis)  # edge (i, i+1) moves from i to i+1
        face = [slice(None)] * 3
        face[axis] = 0
        a[tuple(face)] = 0.0  # wrapped-in value is not a real edge
        out[k] = a
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--dataset", default=None, help="input dataset name (default: first one)")
    p.add_argument("--already-sigmoid", action="store_true", help="input is already in [0,1]")
    p.add_argument("--uint8", action="store_true", help="write uint8 (prob*255) instead of float32")
    args = p.parse_args()

    with h5py.File(args.input, "r") as f:
        name = args.dataset or list(f.keys())[0]
        pred = f[name][:3]
    if pred.ndim != 4 or pred.shape[0] < 3:
        raise SystemExit(f"expected (C>=3, Z, Y, X), got {pred.shape}")

    aff = convert(pred, args.already_sigmoid)
    if args.uint8:
        aff = np.round(aff * 255).astype(np.uint8)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("main", data=aff, compression="gzip", chunks=True)
    print(f"{args.input}:{name} {pred.shape} -> {args.output}:main {aff.shape} {aff.dtype}")


if __name__ == "__main__":
    main()
