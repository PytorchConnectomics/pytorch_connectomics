#!/usr/bin/env python3
"""GT-free percolation guard for a precomputed segmentation.

2_abiss.yaml's own instruction: fit AGG_THRESHOLD by the largest-segment volume
fraction, not by VOI on 100^3 cubes, because a cube cannot see percolation at all.
This is that measurement. It samples blocks instead of reading 158 Gvoxels -- a
segment holding tens of percent of the volume is visible in any sample; one that is
invisible in 0.3% of the volume is not the failure this guards against.

    python scripts/seg_percolation_guard.py <precomputed_seg_dir> [--blocks 200]
"""
from __future__ import annotations

import argparse
import collections
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("--block", type=int, nargs=3, default=[128, 128, 32], help="XYZ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write the numbers here as JSON")
    a = ap.parse_args()

    from cloudvolume import CloudVolume

    path = a.seg if "://" in a.seg else "file://" + a.seg
    vol = CloudVolume(path, mip=0, bounded=True, progress=False, fill_missing=False)
    size = [int(v) for v in vol.shape[:3]]
    off = [int(v) for v in vol.voxel_offset]
    rng = np.random.default_rng(a.seed)

    counts = collections.Counter()
    sampled = 0
    for i in range(a.blocks):
        start = [int(rng.integers(off[k], off[k] + size[k] - a.block[k])) for k in range(3)]
        cut = np.asarray(vol[start[0]:start[0] + a.block[0],
                             start[1]:start[1] + a.block[1],
                             start[2]:start[2] + a.block[2]]).ravel()
        ids, n = np.unique(cut, return_counts=True)
        counts.update(dict(zip(ids.tolist(), n.tolist())))
        sampled += cut.size
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{a.blocks} blocks, {sampled/1e6:.1f} Mvox", flush=True)

    bg = counts.pop(0, 0)
    top = counts.most_common(5)
    fg = sum(counts.values())
    print(f"\n{path}")
    print(f"  sampled            {sampled/1e6:.1f} Mvox in {a.blocks} blocks "
          f"({100.0 * sampled / np.prod(size):.3f}% of the volume)")
    print(f"  background (id 0)  {bg / sampled:.4%}")
    print(f"  distinct labels    {len(counts)}")
    if not fg:
        print("  no foreground sampled")
        return 1
    print(f"  largest label      id {top[0][0]}  {top[0][1] / sampled:.4%} of all voxels, "
          f"{top[0][1] / fg:.4%} of foreground")
    for label, n in top[1:]:
        print(f"    next             id {label}  {n / sampled:.4%}")
    if a.out:
        import json

        Path(a.out).write_text(json.dumps({
            "segmentation": path,
            "blocks": a.blocks,
            "block_xyz": list(a.block),
            "sampled_voxels": int(sampled),
            "background_fraction": bg / sampled,
            "distinct_labels": len(counts),
            "largest_label": int(top[0][0]),
            "largest_share_of_all": top[0][1] / sampled,
            "largest_share_of_foreground": top[0][1] / fg,
        }, indent=2) + "\n")
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
