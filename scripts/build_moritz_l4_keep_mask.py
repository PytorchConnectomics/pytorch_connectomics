#!/usr/bin/env python3
"""Keep-mask for the Moritz L4 ABISS decode: NOT(blood vessel) AND tissue.

The j0126 playbook masks affinity before the watershed sees it, so segments cannot
grow through vessel lumen or the zero-padding border and chain two neurons together
(`dev/zebrafinch/build_bv_border_mask.py` is the same construction from a vessel
volume we own). This is that mask for Moritz L4:

    keep = (blood_vessel == 0)  AND  (y >= 103 and x >= 103)

Blood vessel comes from Peng's TriSAM run (prediction only, no GT), on the
`mip0_ds_z4y8x8` grid -- the SAME grid, origin and frame as the affinity, at ratio
(4, 8, 8). The mask is written at that grid and upsampled on the fly by ABISS's
volume backend (AFF_KEEP_MASK_RATIO), so no 158-Gvoxel full-resolution copy exists.

FRAME. Volume frame, ZYX, origin = global (118, 0, 0). The downsampled grid is the
FULL 9x6 tile grid (1152 x 768 cells = 9216 x 6144 voxels) while the affinity is
cropped on the far face to 8534 x 5599, so the mask is cropped to
ceil(volume / ratio) = (827, 1067, 700) here. Leaving the overhang in would make the
mask larger than the volume, which the backend rejects -- correctly, because that is
also what a mask from the wrong frame looks like.

BORDER. The tissue bbox recorded by `make_ds_h5.py` is y >= 103, x >= 103 (global
z 118-3423 == the whole volume in z). 103 is not a multiple of 8, so whole mask cells
are dropped: y, x cell index < ceil(103/8) = 13. That drops full-resolution y, x < 104,
i.e. one row of real tissue, and never leaves padding unmasked.

    python scripts/build_moritz_l4_keep_mask.py
"""
from __future__ import annotations

import argparse
import math

import h5py
import numpy as np

BV = "/projects/weilab/liupeng/runs/trisam-moritz-segem/blood_vessel_trisam_clean_r8.h5"
OUT = "/projects/weilab/dataset/segEM/Moritz_l4_2019/em/keep_mask_z4y8x8.h5"
RATIO_ZYX = (4, 8, 8)
VOLUME_ZYX = (3306, 8534, 5599)          # affinity extent, volume frame
TISSUE_START_ZYX = (0, 103, 103)         # volume frame; far face is the volume itself


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bv", default=BV)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    shape = tuple(-(-VOLUME_ZYX[i] // RATIO_ZYX[i]) for i in range(3))

    with h5py.File(a.bv, "r") as f:
        bv = f["main"]
        src = tuple(int(v) for v in bv.shape)
        if any(src[i] < shape[i] for i in range(3)):
            raise SystemExit(f"blood-vessel mask {src} does not cover {shape}")
        vessel = np.asarray(bv[: shape[0], : shape[1], : shape[2]]) != 0

    keep = ~vessel
    vessel_frac = float(vessel.mean())

    border = np.zeros(shape, dtype=bool)
    cut = [int(math.ceil(TISSUE_START_ZYX[i] / RATIO_ZYX[i])) for i in range(3)]
    border[: cut[0], :, :] = True
    border[:, : cut[1], :] = True
    border[:, :, : cut[2]] = True
    keep &= ~border

    with h5py.File(a.out, "w") as f:
        d = f.create_dataset("main", data=keep.astype(np.uint8), chunks=(32, 128, 128),
                             compression="gzip")
        d.attrs["axis_order"] = "ZYX"
        d.attrs["downsample_factors_zyx"] = np.array(RATIO_ZYX)
        d.attrs["global_offset_zyx"] = np.array([118, 0, 0])
        d.attrs["volume_shape_zyx_fullres"] = np.array(VOLUME_ZYX)
        d.attrs["semantics"] = "1 = keep (affinity used), 0 = drop (blood vessel or border)"
        d.attrs["source_blood_vessel"] = a.bv
        d.attrs["tissue_start_zyx_fullres"] = np.array(TISSUE_START_ZYX)

    print(f"blood vessel   {a.bv}")
    print(f"  source grid  {src} -> cropped to {shape} (= ceil({VOLUME_ZYX} / {RATIO_ZYX}))")
    print(f"wrote          {a.out}  {shape} uint8")
    print(f"  vessel       {vessel_frac:.4%} of cells")
    print(f"  border       {float(border.mean()):.4%} of cells")
    print(f"  keep         {float(keep.mean()):.4%} of cells "
          f"({1.0 - float(keep.mean()):.4%} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
