#!/usr/bin/env python3
"""Per-label voxel counts for the volume's segmentation -> `label_sizes.npz`.

`build_skeletons.py` picks its cohort and balances its shards from this file, so
it must be the whole inventory: every positive ID, background excluded. One
Z-slab pass, so peak memory is a slab, not the volume.

    LICONN_EA_RUN=<run dir> python tutorials/neuron_liconn_moe/error_analysis/label_sizes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=V.SIZES)
    parser.add_argument("--slab", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")

    totals: dict[int, int] = {}
    with h5py.File(V.SEG, "r") as handle:
        dataset = handle["main"]
        for z0 in range(0, dataset.shape[0], args.slab):
            ids, counts = np.unique(np.asarray(dataset[z0 : z0 + args.slab]), return_counts=True)
            for label, count in zip(ids.tolist(), counts.tolist()):
                totals[label] = totals.get(label, 0) + count
    totals.pop(0, None)
    ids = np.array(sorted(totals), dtype=np.uint64)
    counts = np.array([totals[i] for i in ids.tolist()], dtype=np.int64)
    np.savez(args.output, ids=ids, counts=counts)
    print(f"{V.NAME}: {len(ids)} labels, {int(counts.sum())} foreground voxels -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
