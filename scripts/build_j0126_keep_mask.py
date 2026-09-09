#!/usr/bin/env python3
"""Build j0126's tissue-and-border keep-mask, the exclusion mask ABISS decodes under.

Two sharded stages, run in order:

  --stage tissue  stream FFN's own `tissue_classification` layer and threshold it to
                  NOT(blood vessel | myelin | out-of-bounds), at FFN's native
                  18 x 18 x 20 nm. This is FFN's published tissue_mask recipe.
  --stage keep    upsample that 2x in XY onto the mip-0 grid and AND it with the
                  0/255 border ring of the aligned EM, giving the mip-0 keep-mask.

Both stages shard by grid cell and resume: each shard writes `<out>.done.<id>`
when it finishes, and the driver treats a missing sentinel as an unfinished
shard rather than trusting the zarr, whose fill_value is "keep".

Run `--init` once before an array job so the shards never race on creating the
array:

    python scripts/build_j0126_keep_mask.py --stage tissue --init --out tissue.zarr
    python scripts/build_j0126_keep_mask.py --stage tissue --out tissue.zarr \
        --shard-id $SLURM_ARRAY_TASK_ID --num-shards 16

The masks removes 15.64% of the volume. Because it comes from FFN's own CNN it
weakens a "we beat FFN" comparison; `dev/zebrafinch/build_bv_border_mask.py` is
the alternative built from a vessel volume we own (no myelin masked, 1.38%
removed).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter

# FFN's J0126 cloud-volume mirror. tissue_classification is 6-channel uint8 prob:
# 0=unused, 1=blood_vessel, 2=cell_body, 3=myelin, 4=neuropil, 5=out-of-bounds.
FFN_TISSUE = (
    "https://storage.googleapis.com/j0126-nature-methods-data/"
    "GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/tissue_classification"
)
EXCLUDE_CHANNELS = (1, 3, 5)
TISSUE_SHAPE_ZYX = (5700, 5456, 5332)   # native 18 x 18 x 20 nm
# True mip-0 EM extent, ZYX. Same origin as the tissue layer: Z is 1:1, Y and X
# are half-resolution there, so the tissue mask upsamples 2x in Y/X only.
KEEP_SHAPE_ZYX = (5700, 10912, 10664)
KEEP_CHUNKS = (126, 252, 252)
CELL = 1008                             # matches the affinity chunk grid


def _sentinel(out: Path, shard_id: int) -> Path:
    return Path(f"{out}.done.{shard_id}")


def _border_pad_slice(sl: np.ndarray, offset: int) -> np.ndarray:
    """2D padding mask for one z-section of aligned EM.

    The aligned volume has pure 0 (black) / 255 (white) padding at the faces that
    the FFN tissue mask does not fully cover. Mark {0,255} plus the anti-aliased
    0<->255 seam, dilate by `offset`, then keep only the straight run that reaches
    an image edge -- interior black is real tissue, not padding.
    """
    b = (sl == 0) | (sl == 255)
    b |= (minimum_filter(sl, 3) == 0) & (maximum_filter(sl, 3) == 255)
    if offset:
        b = binary_dilation(b, iterations=offset)
    nb = ~b
    return (
        (np.cumsum(nb, 1) == 0)
        | (np.cumsum(nb[:, ::-1], 1) == 0)[:, ::-1]
        | (np.cumsum(nb, 0) == 0)
        | (np.cumsum(nb[::-1], 0) == 0)[::-1]
    )


def build_keep_cell(start, shape, index_zyx, tissue_ds, em_ds, *, border_offset=1):
    """Per-cell boolean keep-mask (True = keep) = FFN tissue AND 0/255 border."""
    z0, y0, x0 = start
    depth, height, width = shape
    keep = np.ones((depth, height, width), bool)

    # Tissue: Z 1:1, XY half-res -> nearest 2x upsample; beyond bounds -> keep.
    tz, ty, tx = tissue_ds.shape
    y0h, x0h = y0 // 2, x0 // 2
    z1c, y1c, x1c = (
        min(z0 + depth, tz),
        min(y0h + (height + 1) // 2, ty),
        min(x0h + (width + 1) // 2, tx),
    )
    native = np.ones((depth, (height + 1) // 2, (width + 1) // 2), np.uint8)
    if z1c > z0 and y1c > y0h and x1c > x0h:
        sub = np.asarray(tissue_ds[z0:z1c, y0h:y1c, x0h:x1c]).astype(np.uint8)
        native[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
    up = np.repeat(np.repeat(native, 2, axis=1), 2, axis=2)[:depth, :height, :width]
    if up.shape != (depth, height, width):
        up = np.pad(
            up,
            [(0, (depth, height, width)[k] - up.shape[k]) for k in range(3)],
            mode="edge",
        )
    keep &= up > 0

    # Border ring only: the interior is full tissue, and reading EM for it is the
    # expensive half of this stage.
    ymax = (KEEP_SHAPE_ZYX[1] - 1) // CELL
    xmax = (KEEP_SHAPE_ZYX[2] - 1) // CELL
    if index_zyx[1] in (0, ymax) or index_zyx[2] in (0, xmax):
        em = np.asarray(em_ds[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width])
        if em.shape != (depth, height, width):
            pad = [(0, (depth, height, width)[k] - em.shape[k]) for k in range(3)]
            em = np.pad(em, pad)          # off-volume reads as 0 == padding == masked
        for zi in range(depth):
            keep[zi] &= ~_border_pad_slice(em[zi], border_offset)
    return keep


def run_tissue(args) -> None:
    import cloudvolume as cv

    out = zarr.open(str(args.out), mode="r+")
    vol = cv.CloudVolume(FFN_TISSUE, mip=0, bounded=True, progress=False, fill_missing=True)
    got = tuple(int(s) for s in vol.shape[:3])
    want = (TISSUE_SHAPE_ZYX[2], TISSUE_SHAPE_ZYX[1], TISSUE_SHAPE_ZYX[0])
    if got != want or int(vol.num_channels) != 6:
        raise SystemExit(f"unexpected tissue_classification geometry: {vol.shape}")

    slabs = [
        (z, min(z + args.z_slab, TISSUE_SHAPE_ZYX[0]))
        for z in range(0, TISSUE_SHAPE_ZYX[0], args.z_slab)
    ]
    mine = slabs[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(slabs)} slabs", flush=True)
    t0 = time.time()
    for i, (z0, z1) in enumerate(mine, 1):
        # cloud-volume is XYZC; channel selection has to be a post-fetch numpy index.
        arr = vol[:, :, z0:z1]
        excluded = (arr[..., list(EXCLUDE_CHANNELS)] > args.threshold).any(axis=-1)
        out[z0:z1, :, :] = np.transpose((~excluded).astype(np.uint8), (2, 1, 0))
        print(f"  [{i}/{len(mine)}] z={z0}:{z1} ({(time.time()-t0)/i:.1f}s/slab)", flush=True)


def run_keep(args) -> None:
    out = zarr.open(str(args.out), mode="r+")
    tissue = zarr.open(str(args.tissue), mode="r")
    em = zarr.open(str(args.em), mode="r")

    cells = [
        (cz, cy, cx)
        for cz in range(0, -(-KEEP_SHAPE_ZYX[0] // CELL))
        for cy in range(0, -(-KEEP_SHAPE_ZYX[1] // CELL))
        for cx in range(0, -(-KEEP_SHAPE_ZYX[2] // CELL))
    ]
    mine = cells[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(cells)} cells", flush=True)
    t0 = time.time()
    for i, (cz, cy, cx) in enumerate(mine, 1):
        gz, gy, gx = cz * CELL, cy * CELL, cx * CELL
        ez = min(gz + CELL, KEEP_SHAPE_ZYX[0])
        ey = min(gy + CELL, KEEP_SHAPE_ZYX[1])
        ex = min(gx + CELL, KEEP_SHAPE_ZYX[2])
        # Always compute the FULL cell and clip on write, so a face cell gets the
        # same border decision it would get in the interior of a padded volume.
        keep = build_keep_cell(
            (gz, gy, gx), (ez - gz, CELL, CELL), (cz, cy, cx),
            tissue, em, border_offset=args.border_offset,
        )
        out[gz:ez, gy:ey, gx:ex] = keep[:, : ey - gy, : ex - gx].astype(np.uint8)
        print(
            f"  [{i}/{len(mine)}] z{cz}_y{cy}_x{cx} keep={keep.mean():.4f} "
            f"({(time.time()-t0)/i:.1f}s/cell)",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("tissue", "keep"), required=True)
    ap.add_argument("--out", type=Path, required=True, help="output zarr")
    ap.add_argument("--tissue", type=Path, help="tissue zarr (--stage keep)")
    ap.add_argument("--em", type=Path, help="mip-0 EM zarr array (--stage keep)")
    ap.add_argument("--init", action="store_true", help="create the zarr and exit")
    ap.add_argument("--threshold", type=int, default=128, help="tissue channel threshold")
    ap.add_argument("--z-slab", type=int, default=128, help="Z per cloud-volume read")
    ap.add_argument("--border-offset", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.init:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            print(f"{args.out} already exists, leaving it alone")
            return 0
        # fill_value 1 = KEEP, so an unwritten region never silently deletes data.
        if args.stage == "tissue":
            zarr.open(
                str(args.out), mode="w", shape=TISSUE_SHAPE_ZYX, dtype="uint8",
                chunks=(args.z_slab, 512, 512), fill_value=1,
                compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
            )
        else:
            zarr.open(
                str(args.out), mode="w", shape=KEEP_SHAPE_ZYX, dtype="uint8",
                chunks=KEEP_CHUNKS, fill_value=1,
            )
        print(f"created {args.out}")
        return 0

    if not args.out.exists():
        raise SystemExit(f"{args.out} does not exist; run --init first")
    if args.stage == "keep":
        if args.tissue is None or args.em is None:
            raise SystemExit("--stage keep needs --tissue and --em")
        for path in (args.tissue, args.em):
            if not path.exists():
                raise SystemExit(f"missing input: {path}")
        run_keep(args)
    else:
        run_tissue(args)

    _sentinel(args.out, args.shard_id).write_text("done\n")
    print(f"shard {args.shard_id} done -> {_sentinel(args.out, args.shard_id)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
