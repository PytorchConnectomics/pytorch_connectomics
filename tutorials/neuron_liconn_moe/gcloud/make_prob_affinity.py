#!/usr/bin/env python3
"""Create the canonical ABISS probability-space affinity HDF5 artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from scripts.run_abiss_volume import _shift_to_destination_storage, _select_affinity_channels
from tutorials.neuron_liconn_ist.merge_fn_sweep import EPS, uncompress


def convert(source: Path, output: Path, dataset: str | None = None, slab_z: int = 32) -> None:
    with h5py.File(source, "r") as src:
        if dataset is None:
            names = [key for key, value in src.items() if isinstance(value, h5py.Dataset)]
            if len(names) != 1:
                raise ValueError(f"source must contain exactly one dataset, got {names}")
            dataset = names[0]
        inp = src[dataset]
        if inp.ndim != 4 or inp.shape[0] < 3:
            raise ValueError(f"source dataset must be CZYX with at least 3 channels, got {inp.shape}")
        c, zdim, ydim, xdim = (int(v) for v in inp.shape)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as dst:
            out = dst.create_dataset("main", shape=(3, zdim, ydim, xdim), dtype="float32")
            for z0 in range(0, zdim, slab_z):
                z1 = min(zdim, z0 + slab_z)
                # The one-voxel source halo supplies the global Z predecessor.
                read0 = max(0, z0 - 1)
                raw = np.asarray(inp[:, read0:z1, :, :])
                selected = _select_affinity_channels(raw, [2, 1, 0])
                xy = np.transpose(selected, (3, 2, 1, 0))
                shifted = _shift_to_destination_storage(xy)
                shifted = shifted[:, :, z0 - read0 : z1 - read0, :]
                if z0 > 0:
                    # The imported helper zeros each temporary slab's first face;
                    # restore it from the one-voxel global-Z halo. This writes the
                    # same value already present in a full-volume shift, not a
                    # new edge, and keeps the slab conversion bitwise equivalent.
                    shifted[:, :, 0, 2] = selected[2, 0].T
                values = uncompress(shifted, 0.2)
                # The destination face is a real volume face, not a clipped tail.
                if z0 == 0:
                    values[:, :, 0, 2] = 0.0
                values = np.transpose(values, (3, 2, 1, 0))
                values[0, :, :, 0] = 0.0
                values[1, :, 0, :] = 0.0
                if z0 == 0:
                    values[2, 0, :, :] = 0.0
                out[:, z0:z1] = values.astype(np.float32, copy=False)
            out.attrs["affinity_space"] = "probability"
            out.attrs["edge_storage"] = "destination"
            out.attrs["uncompress_scale"] = 0.2
            out.attrs["uncompress_eps"] = EPS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    convert(args.source, args.output, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
