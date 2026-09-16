#!/usr/bin/env python3
"""Is the biggest segment a merge chain or a real object? Shape says, size does not.

`sweep_merge_threshold.py` ranks thresholds by the largest segment's SHARE of the
volume, a rule calibrated on one volume. It does not survive contact with the
rest of the batch, for two reasons:

1. Share is not comparable across fields. These volumes run 1035-5624 um^3, so a
   fixed 2% cap is 3x stricter on a 32x field than on the published 18x one.
2. Size alone cannot tell a merge chain from a soma. The published volume simply
   had no soma in frame; volumes that do have one show a "runaway" segment at
   every threshold below 0.70 and the rule then picks the shattered end of the
   sweep.

What separates them is SHAPE. A merge chain threads the whole field through thin
bridges: bounding box close to the full volume, fill fraction (voxels / bbox
voxels) of a few percent. A soma is compact: a bbox a few um across and a fill
fraction upwards of 0.2. This prints both for the top segments of every swept
threshold so the operating point is chosen on evidence.

    python tutorials/neuron_liconn_moe/inspect_top_segments.py --volume <name>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def top_segments(seg: np.ndarray, spacing_zyx, top: int) -> list[dict]:
    import fastremap
    from scipy import ndimage as ndi

    vox_um3 = float(np.prod(spacing_zyx)) / 1e9
    # find_objects needs contiguous labels; ABISS ids are sparse uint64.
    small, _ = fastremap.renumber(seg, in_place=False)
    ids, cnt = np.unique(small, return_counts=True)
    keep = ids != 0
    ids, cnt = ids[keep], cnt[keep]
    order = np.argsort(cnt)[::-1][:top]
    slices = ndi.find_objects(small.astype(np.int32))

    out = []
    for i in order:
        lid = int(ids[i])
        sl = slices[lid - 1]
        ext_vox = np.array([s.stop - s.start for s in sl], dtype=np.float64)
        ext_um = ext_vox * np.asarray(spacing_zyx) / 1000.0
        vol = float(cnt[i]) * vox_um3
        out.append({
            "volume_um3": vol,
            "share": float(cnt[i]) / seg.size,
            "bbox_um_zyx": [round(float(v), 1) for v in ext_um],
            "bbox_span_frac": float(np.prod(ext_vox) / seg.size),
            "fill": vol / (float(np.prod(ext_vox)) * vox_um3),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--top", type=int, default=5)
    # The FULL grid, not a subset: `seg_mt{i}.h5` is indexed by position in the
    # grid that was swept, so a shortened list silently mislabels every file.
    ap.add_argument("--grid", help="comma list of the full swept grid; "
                                   "default = the grid recorded in mt_sweep.json")
    a = ap.parse_args()

    name = a.volume
    p = V.plan(name)
    d = V.sweep_dir(name)
    sj = V.work_dir(name) / "mt_sweep.json"
    if a.grid:
        grid = [float(x) for x in a.grid.split(",")]
    elif sj.exists():
        grid = [r["mt"] for r in json.loads(sj.read_text())["rows"]]
    elif name in V.LEGACY:
        grid = V.LEGACY[name]["sweep_grid"]
    else:
        raise SystemExit(f"no {sj}; pass --thresholds")

    field = float(np.prod(p["shape"])) * float(np.prod(p["spacing_zyx"])) / 1e9
    box_um = [round(s * sp / 1000, 1) for s, sp in zip(p["shape"], p["spacing_zyx"])]
    print(f"{name}  field {field:.0f} um^3  box {box_um} um ZYX", flush=True)
    for i, mt in enumerate(grid):
        f = d / f"seg_mt{i}.h5"
        if not f.exists():
            continue
        with h5py.File(f, "r") as fh:
            seg = np.asarray(fh["main"])
        rows = top_segments(seg, p["spacing_zyx"], a.top)
        del seg
        print(f"\n  mt {mt:.3f}", flush=True)
        print(f"    {'rank':>4} {'um^3':>9} {'%vol':>7} {'bbox um ZYX':>20} "
              f"{'bbox/field':>10} {'fill':>6}")
        for r, row in enumerate(rows, 1):
            print(f"    {r:>4} {row['volume_um3']:>9.2f} {row['share']*100:>6.2f}% "
                  f"{str(row['bbox_um_zyx']):>20} {row['bbox_span_frac']:>10.3f} "
                  f"{row['fill']:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
