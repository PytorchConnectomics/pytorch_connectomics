"""Route B label damage: what the (3,2,2) mode downsample does to the proofread GT.

Card MSIDEPLOY-MODEL-002, the Stage 2 PRECONDITION listed in its "Next decision"
table and owned by the executing session. This measures it; it does not start
Stage 2 and it does not authorise anything.

Route B trains at [36, 18, 18] nm by taking the mip0 export at [12, 9, 9] and
block-downsampling by integer (3, 2, 2). The image half of that is an exact block
average. The LABEL half cannot be averaged, so it is a mode -- and the card flags
that as unavoidably lossy and makes quantifying it a precondition of the launch.

THE REDUCTION THAT MAKES THIS CHEAP. The mip0 `seg` is the 18 nm proofread
segmentation nearest-upsampled 2x on every axis and bit-exact to it. Under that:

  * XY: a factor-2 mode over a 2x-repeated array takes all four sub-voxels of one
    block from the SAME original voxel, so XY is an identity. Route B does not
    touch the labels in XY at all.
  * Z: output plane j covers mip0 planes 3j, 3j+1, 3j+2, which are 18 nm planes
    floor(3j/2), floor((3j+1)/2), floor((3j+2)/2) -- always two copies of one
    plane and one copy of its neighbour, so the mode is the doubled plane,
    uniquely and with no tie.

So route B's label transform, written on the 18 nm label grid, is exactly a
1.5x Z PLANE SELECTION: keep two planes out of every three. This script derives
that map from the definition rather than assuming it, VERIFIES the bit-exactness
the reduction rests on against the actual mip0 export, and then reuses the same
damage scan the 1.3889x measurement used.

    python z600_routeb_label_damage.py --json <out.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
ROOT = "/projects/weilab/dataset/liconn/pytc"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from z600_oracle_floor import label_damage  # noqa: E402
from z600_zresample import routeb_z_selection  # noqa: E402


def verify_bit_exact(seg18, seg0, rng, n_blocks: int, tag: str) -> list[dict]:
    """Check mip0 seg == np.repeat(seg18, 2) on random blocks."""
    out = []
    z18, y18, x18 = (int(v) for v in seg18.shape)
    for _ in range(n_blocks):
        z = int(rng.integers(0, z18 - 4))
        y = int(rng.integers(0, y18 - 64))
        x = int(rng.integers(0, x18 - 64))
        a = np.asarray(seg18[z:z + 4, y:y + 64, x:x + 64])
        want = np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)
        got = np.asarray(seg0[2 * z:2 * z + 8, 2 * y:2 * y + 128, 2 * x:2 * x + 128])
        ok = bool(np.array_equal(want, got))
        out.append({"split": tag, "zyx_18nm": [z, y, x], "bit_exact": ok})
        print(f"  [{tag}] block z={z} y={y} x={x}: "
              f"{'bit-exact' if ok else 'MISMATCH'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--min-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import zarr

    rng = np.random.default_rng(a.seed)
    out = {"card": "MSIDEPLOY-MODEL-002",
           "purpose": "Stage 2 precondition: route B (3,2,2) label damage",
           "route": "B -- mip0 [12,9,9] -> (3,2,2) -> [36,18,18] nm ZYX",
           "reduction": ("on the 18 nm label grid this is a 1.5x Z plane selection "
                         "with XY identity, valid because the mip0 seg is a bit-exact "
                         "2x nearest upsample of the 18 nm seg"),
           "splits": {}}

    for split in a.splits.split(","):
        seg18 = zarr.open(f"{ROOT}/final_proofread/{split}/data.zarr/seg", mode="r")
        seg0 = zarr.open(f"{ROOT}/final_proofread_mip0/{split}/data.zarr/seg", mode="r")
        n18 = int(seg18.shape[0])
        print(f"[{split}] 18 nm {seg18.shape}  mip0 {seg0.shape}", flush=True)
        if tuple(2 * int(v) for v in seg18.shape) != tuple(int(v) for v in seg0.shape):
            raise SystemExit(f"[{split}] mip0 is not exactly 2x the 18 nm shape")

        checks = verify_bit_exact(seg18, seg0, rng, a.blocks, split)
        if not all(c["bit_exact"] for c in checks):
            raise SystemExit(
                f"[{split}] mip0 seg is NOT a bit-exact 2x upsample -- the reduction "
                "this script rests on does not hold; measure on mip0 directly instead.")

        sel, n_out = routeb_z_selection(n18)
        dropped = sorted(set(range(n18)) - set(sel.tolist()))
        runs = np.diff(np.array(dropped)) if len(dropped) > 1 else np.array([])
        print(f"[{split}] 18 nm {n18} planes -> {n_out} at 36 nm; "
              f"{len(dropped)} dropped, min gap between dropped planes "
              f"{int(runs.min()) if runs.size else 'n/a'}", flush=True)

        t0 = time.time()
        dmg = label_damage(seg18, sel, n18, a.min_size)
        print(f"[{split}] damage scan {time.time() - t0:.0f}s", flush=True)
        print(json.dumps(dmg["all_objects"], indent=1), flush=True)
        print(json.dumps(dmg[f"objects_ge_{a.min_size}_voxels"], indent=1), flush=True)

        out["splits"][split] = {
            "n_planes_18nm": n18, "n_planes_36nm": n_out,
            "n_dropped_18nm_planes": len(dropped),
            "min_gap_between_dropped_planes": int(runs.min()) if runs.size else None,
            "bit_exactness_checks": checks,
            "label_damage": dmg,
        }
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}", flush=True)


main()
