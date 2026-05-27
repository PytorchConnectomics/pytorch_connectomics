#!/usr/bin/env python3
"""Precompute kimimaro skeleton volumes for label_aux_type=skeleton training.

For each label path, writes a sibling ``*_skeleton`` array (zarr sub-key or
``*_skeleton.h5``) matching ``connectomics.data.processing.distance``'s
``sdt_path_for_label`` convention, so training picks it up as a cache hit
instead of recomputing in-process on the GPU node.

Examples (CPU node):
    # All 6 NISB base volumes, sequentially in one job:
    python scripts/precompute_skeleton_volumes.py \\
        --label '/projects/weilab/dataset/nisb/base/train/seed*/data.zarr/seg' \\
        --label '/projects/weilab/dataset/nisb/base/val/seed*/data.zarr/seg'

    # Single volume (use this form with one SLURM job per volume for parallelism):
    python scripts/precompute_skeleton_volumes.py \\
        --label /projects/weilab/dataset/nisb/base/train/seed0/data.zarr/seg

    # Sharded across SLURM tasks:
    python scripts/precompute_skeleton_volumes.py \\
        --label '/projects/weilab/dataset/nisb/base/train/seed*/data.zarr/seg' \\
        --num-shards $SLURM_NTASKS --shard-index $SLURM_PROCID
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from connectomics.chunked import SkeletonVolumeConfig, SkeletonVolumeProcessor
from connectomics.data.io.io import volume_exists
from connectomics.data.processing.distance import (
    precompute_skeleton_volume,
    sdt_path_for_label,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label",
        action="append",
        required=True,
        help=(
            "Label volume path or glob (zarr sub-key like data.zarr/seg or HDF5). "
            "Repeat to pass multiple."
        ),
    )
    p.add_argument(
        "--resolution",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
        help=(
            "Voxel resolution passed to kimimaro. Must match what training uses; "
            "the v2 skeleton_aware_edt target falls back to (1.0, 1.0, 1.0) when "
            "no resolution is set in label_transform, so leave the default for "
            "base_banis_v2*.yaml."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if the *_skeleton cache already exists.",
    )
    p.add_argument(
        "--chunk-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help=(
            "When set, route through the chunked precompute "
            "(SkeletonVolumeProcessor) instead of the single-process function. "
            "Required for volumes larger than node RAM. Example: 1024 1024 256."
        ),
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of local worker processes for chunked mode (default 1).",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=0,
        help=(
            "Per-axis halo (voxels) for skeletonization context near chunk "
            "boundaries. Rasterization still writes only the inner chunk "
            "region. Default 0 (acceptable for SDT supervision)."
        ),
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shards (for splitting across SLURM tasks).",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This shard index in [0, num-shards).",
    )
    return p.parse_args()


def _expand_label_path(pattern: str) -> list[str]:
    """Expand a label path or glob.

    Zarr sub-keys (``store.zarr/seg``) need the glob applied to the store part
    only, since ``glob.glob`` won't descend into a zarr directory the way we
    want for sub-arrays.
    """
    if ".zarr" in pattern:
        zarr_idx = pattern.index(".zarr") + len(".zarr")
        store_pat = pattern[:zarr_idx]
        sub_key = pattern[zarr_idx:].lstrip("/")
        stores = sorted(glob.glob(store_pat))
        if not stores:
            return []
        return [f"{s}/{sub_key}" if sub_key else s for s in stores]
    return sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        print(f"bad shard args: {args.num_shards=} {args.shard_index=}", file=sys.stderr)
        return 2

    label_paths: list[str] = []
    for pat in args.label:
        matches = _expand_label_path(pat)
        if not matches:
            print(f"WARNING: no labels matched: {pat}", file=sys.stderr)
        label_paths.extend(matches)

    if not label_paths:
        print("No label paths matched any --label argument.", file=sys.stderr)
        return 1

    shard = [
        p for i, p in enumerate(label_paths) if i % args.num_shards == args.shard_index
    ]
    print(
        f"[shard {args.shard_index}/{args.num_shards}] "
        f"{len(shard)}/{len(label_paths)} label paths assigned:",
        flush=True,
    )
    for p in shard:
        print(f"  {p}")

    chunked = args.chunk_shape is not None
    if chunked:
        mode_label = (
            f"chunked (chunk_shape={tuple(args.chunk_shape)}, "
            f"parallel={args.parallel}, overlap={args.overlap})"
        )
    else:
        mode_label = "single-process"
    print(f"Mode: {mode_label}", flush=True)

    failures: list[str] = []
    for lp in shard:
        sp = sdt_path_for_label(lp, mode="skeleton")
        if volume_exists(sp) and not args.overwrite:
            print(f"\n[cache hit] {sp} exists; skipping", flush=True)
            continue
        print(f"\n=== {lp} -> {sp} ===", flush=True)
        t0 = time.time()
        try:
            if chunked:
                cfg = SkeletonVolumeConfig(
                    input_path=lp,
                    output_path=sp,
                    chunk_shape=tuple(int(v) for v in args.chunk_shape),
                    parallel=int(args.parallel),
                    overlap=int(args.overlap),
                    overwrite=bool(args.overwrite),
                    resolution=tuple(float(v) for v in args.resolution),
                )
                SkeletonVolumeProcessor(cfg).run()
            else:
                precompute_skeleton_volume(
                    lp, sp, resolution=tuple(args.resolution)
                )
        except Exception as e:
            print(f"FAILED {lp}: {e}", file=sys.stderr, flush=True)
            failures.append(lp)
            continue
        print(f"  Elapsed: {time.time() - t0:.1f}s", flush=True)

    if failures:
        print(f"\n{len(failures)} volume(s) FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
