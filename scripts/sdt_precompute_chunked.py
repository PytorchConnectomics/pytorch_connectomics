"""Chunked SDT precompute of NISB seg_filled: Z-slab parallel + concurrent zarr write.

The whole-volume sdt_precompute.py path does not fit cluster limits. This splits seg_filled along Z
into bands aligned to the zarr's native 256 Z-chunks. Each SLURM array task lazily loads ONE band
plus a +/-halo overlap from data.zarr/seg_filled and data.zarr/seg_filled_skeleton, runs the
project's skeleton_aware_edt_from_skeleton_vol on the slab, and writes ONLY the core band to
data.zarr/seg_filled_sdt. Because the output chunks equal the source chunks (256^3) and bands are
chunk-aligned, tasks write DISJOINT zarr chunks -> safe concurrent ("async") writes, no assemble
pass.

Correctness: scipy.ndimage.distance_transform_edt measures to the nearest actual background or
skeleton voxel and does not treat an array edge as background. So Z-chunking is exact for core
voxels whose nearest same-instance boundary and nearest rasterized skeleton voxel both lie within
the halo-padded slab. Thin neurites satisfy this with halo=96 (~1.9um). A structure thicker in Z
than the halo near a band boundary (for example a soma), or a sparse skeleton outside the slab, is
approximate because that instance falls back to regular boundary EDT in that slab. Raise --halo if
QC finds boundary-band artifacts.

Two phases (chain with a SLURM dependency):
    python scripts/sdt_precompute_chunked.py --init --seeds 0 1 2 3 4 100
    python scripts/sdt_precompute_chunked.py --task $SLURM_ARRAY_TASK_ID --seeds 0 1 2 3 4 100 --halo 96

Run from repo root (env: pytc).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import zarr

BASE = "/projects/weilab/dataset/nisb/base"


def zpath_for(seed: int, path: str | None = None) -> str:
    if path is not None:
        return path
    sub = f"train/seed{seed}" if seed < 100 else f"val/seed{seed}"
    return f"{BASE}/{sub}/data.zarr"


def z_bands(n_z: int, chunk_z: int) -> list[tuple[int, int]]:
    """Bands aligned to the zarr's native Z chunk size (last band may be short)."""
    return [(z0, min(z0 + chunk_z, n_z)) for z0 in range(0, n_z, chunk_z)]


def decode_task(task: int, seeds: list[int], bands: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    n_bands = len(bands)
    n_tasks = len(seeds) * n_bands
    if task < 0 or task >= n_tasks:
        raise ValueError(f"--task must be in [0, {n_tasks - 1}] for {len(seeds)} seeds x "
                         f"{n_bands} bands")
    seed = seeds[task // n_bands]
    z0, z1 = bands[task % n_bands]
    return seed, task % n_bands, z0, z1


def open_zarr_array(path: str, role: str, mode: str = "r"):
    try:
        return zarr.open(path, mode=mode)
    except Exception as exc:
        raise FileNotFoundError(f"missing {role}: {path}") from exc


def zarr_subkey(path: str) -> str:
    zarr_idx = path.index(".zarr")
    sub_key = path[zarr_idx + 5:].strip("/")
    if not sub_key:
        raise ValueError(f"expected zarr sub-key path, got {path}")
    return sub_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="seeds (train 0-4, val 100) defining the flat task grid")
    ap.add_argument("--init", action="store_true",
                    help="create empty seg_filled_sdt zarrs for all --seeds, then exit")
    ap.add_argument("--task", type=int, default=None,
                    help="flat task id = seed_index*n_bands + band_index")
    ap.add_argument("--halo", type=int, default=96, help="Z overlap (vox) loaded around each band")
    ap.add_argument("--band-z", type=int, default=None,
                    help="Z-band thickness (vox); default = native label Z chunk. SDT needs a large "
                         "halo, so prefer fat bands (e.g. 256) to amortize the fixed halo cost; thin "
                         "bands recompute the EDT redundantly. seg_filled_sdt is created with this Z "
                         "chunk so bands write disjoint chunks (safe concurrent).")
    ap.add_argument("--label-key", default="seg_filled")
    ap.add_argument("--resolution", type=float, nargs=3, default=[9, 9, 20],
                    help="array-axis order")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--bg-value", type=float, default=-1.0)
    ap.add_argument("--path", default=None,
                    help="override data.zarr path for a single-seed offline check")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="threads for the per-instance EDT (0=serial; set to CPU count "
                         "-- a few giant neurons otherwise serialize the whole band)")
    args = ap.parse_args()

    from connectomics.data.processing.distance import (
        sdt_path_for_label,
        skeleton_aware_edt_from_skeleton_vol,
    )

    # Probe one volume for shape / chunks (all NISB seeds share the layout).
    probe_zpath = zpath_for(args.seeds[0], args.path)
    probe = open_zarr_array(f"{probe_zpath}/{args.label_key}", "hole-filled label volume")
    shape, chunks = probe.shape, probe.chunks
    n_z, chunk_z = shape[2], chunks[2]
    band_z = args.band_z or chunk_z
    out_chunks = (chunks[0], chunks[1], band_z)  # Z-chunk = band -> disjoint concurrent writes
    bands = z_bands(n_z, band_z)
    n_bands = len(bands)

    if args.init:
        for seed in args.seeds:
            zpath = zpath_for(seed, args.path)
            label_path = f"{zpath}/{args.label_key}"
            open_zarr_array(label_path, "hole-filled label volume")
            out_path = sdt_path_for_label(label_path, mode="sdt")
            g = zarr.open(zpath, mode="a")
            g.create_dataset(zarr_subkey(out_path), shape=shape, chunks=out_chunks, dtype=np.float32,
                             fill_value=args.bg_value, overwrite=True)
            print(f"created {out_path} shape={shape} chunks={out_chunks} "
                  f"dtype=float32 fill_value={args.bg_value}", flush=True)
        print(f"init done; {len(args.seeds)} seeds x {n_bands} bands = "
              f"{len(args.seeds) * n_bands} tasks", flush=True)
        return

    if args.task is None:
        ap.error("provide --task (or --init)")
    try:
        seed, band_index, z0, z1 = decode_task(args.task, args.seeds, bands)
    except ValueError as exc:
        ap.error(str(exc))

    lo, hi = max(0, z0 - args.halo), min(n_z, z1 + args.halo)
    zpath = zpath_for(seed, args.path)
    label_path = f"{zpath}/{args.label_key}"
    skel_path = sdt_path_for_label(label_path, mode="skeleton")
    out_path = sdt_path_for_label(label_path, mode="sdt")

    src = open_zarr_array(label_path, "hole-filled label volume")
    skel = open_zarr_array(skel_path, "skeleton volume")
    out = open_zarr_array(out_path, "initialized SDT output", mode="r+")
    if skel.shape != src.shape:
        raise ValueError(f"skeleton shape {skel.shape} does not match label shape {src.shape}")
    if out.shape != src.shape:
        raise ValueError(f"output shape {out.shape} does not match label shape {src.shape}")

    print(f"seed{seed} band{band_index} z[{z0}:{z1}] slab z[{lo}:{hi}] of {n_z} ...",
          flush=True)
    t0 = time.time()
    label_slab = np.asarray(src[:, :, lo:hi])
    skel_slab = np.asarray(skel[:, :, lo:hi])
    print(f"loaded label/skel slabs {label_slab.shape} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    sdt_slab = skeleton_aware_edt_from_skeleton_vol(
        label_slab,
        skel_slab,
        resolution=tuple(args.resolution),
        alpha=args.alpha,
        bg_value=args.bg_value,
        max_parallel=args.num_workers,
    )
    print(f"computed SDT slab in {time.time()-t0:.0f}s "
          f"range=[{sdt_slab.min():.3f}, {sdt_slab.max():.3f}]", flush=True)

    core = sdt_slab[:, :, z0 - lo: z0 - lo + (z1 - z0)]
    out[:, :, z0:z1] = core.astype(np.float32, copy=False)
    print(f"wrote {out_path}[:,:,{z0}:{z1}]", flush=True)


if __name__ == "__main__":
    main()
