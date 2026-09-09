"""Download a Neuroglancer `precomputed` volume into a local zarr (ZYX).

The source location is the only required argument, so this works for any public
precomputed layer, image or segmentation:

  # whole j0126 EM volume at mip 0 (9 x 9 x 20 nm) -- ~660 GB uint8, shard it
  python scripts/download_precompute.py \
      gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned \
      --out /path/to/j0126_em.zarr --mip 0 --tile-xy 2048 --slab 64

  # one 1008^3 test chunk instead of the whole volume (a few minutes, ~1 GB)
  python scripts/download_precompute.py <source> --out crop.zarr --mip 0 \
      --bbox 2900 3908 5000 6008 5000 6008

Point a config at the array inside the store, e.g. `image: /path/to/crop.zarr/main`.

Notes carried over from the measured full-volume runs
(`dev/zebrafinch/download_ffn_gcs.py`, which this generalizes):

* `--parallel` stays 1 by default. CloudVolume's multiprocessing path measured
  ~10x SLOWER here, and combined with `--fill-missing` a failed worker returns
  SILENT ZEROS. Shard the job instead of raising `--parallel`.
* `--fill-missing` is OFF by default so a gap in the source raises instead of
  writing zeros that look like real EM downstream.
* Jobs are (z-slab x XY-tile) blocks. They are disjoint and chunk-aligned, so
  several `--shard-id` processes can fill one store concurrently, and each shard
  resumes from its own `.progress.<shard>` sidecar.
* Run `--init-only` once before launching a sharded array job, so the shards
  never race on creating the array.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import zarr
from cloudvolume import CloudVolume


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="precomputed location, e.g. gs://bucket/path/layer (precomputed:// optional)")
    ap.add_argument("--out", required=True, help="output zarr store path")
    ap.add_argument("--dataset", default="main", help="array name inside the store (default: main)")
    ap.add_argument("--mip", type=int, default=0, help="source mip level (default: 0, full resolution)")
    ap.add_argument(
        "--bbox",
        type=int,
        nargs=6,
        metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
        help="ZYX voxel bounds at the chosen mip; default is the whole volume",
    )
    ap.add_argument("--slab", type=int, default=64, help="Z voxels per job (default: 64)")
    ap.add_argument("--tile-xy", type=int, default=0, help="XY tile per job; 0 = whole XY plane per slab")
    ap.add_argument("--parallel", type=int, default=1, help="CloudVolume workers (see module docstring)")
    ap.add_argument("--fill-missing", action="store_true", help="return zeros for missing source chunks")
    ap.add_argument("--init-only", action="store_true", help="create the zarr array and exit")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    source = args.source if "://" in args.source else f"precomputed://{args.source}"

    cv = CloudVolume(
        source,
        mip=args.mip,
        parallel=args.parallel,
        use_https=True,
        progress=False,
        fill_missing=args.fill_missing,
    )
    size_x, size_y, size_z = (int(v) for v in cv.volume_size)
    res_xyz = [int(v) for v in cv.resolution]
    res_zyx = [res_xyz[2], res_xyz[1], res_xyz[0]]

    if args.bbox:
        z0, z1, y0, y1, x0, x1 = args.bbox
    else:
        z0, z1, y0, y1, x0, x1 = 0, size_z, 0, size_y, 0, size_x
    for lo, hi, limit, axis in ((z0, z1, size_z, "z"), (y0, y1, size_y, "y"), (x0, x1, size_x, "x")):
        if not 0 <= lo < hi <= limit:
            raise SystemExit(f"--bbox {axis} range [{lo}, {hi}) is outside the mip{args.mip} extent [0, {limit})")
    shape = (z1 - z0, y1 - y0, x1 - x0)

    print(
        f"{source}\n  mip{args.mip}: volume (Z,Y,X)=({size_z},{size_y},{size_x}) "
        f"res(zyx)={res_zyx} nm dtype={cv.dtype}\n"
        f"  writing (Z,Y,X)={shape} from z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] -> {args.out}/{args.dataset}",
        flush=True,
    )

    store = zarr.open(args.out, mode="a")
    if args.dataset in store:
        arr = store[args.dataset]
        assert tuple(arr.shape) == shape, (arr.shape, shape)
    else:
        arr = store.create_array(
            args.dataset,
            shape=shape,
            chunks=(min(args.slab, 64), 256, 256),
            dtype=cv.dtype,
        )
        arr.attrs["resolution_zyx_nm"] = res_zyx
        arr.attrs["source"] = source
        arr.attrs["mip"] = args.mip
        arr.attrs["bbox_zyx"] = [z0, z1, y0, y1, x0, x1]

    if args.init_only:
        print(f"initialized {args.out}/{args.dataset}", flush=True)
        return

    tile = args.tile_xy or max(shape[1], shape[2])
    jobs = [
        (zs, ys, xs)
        for zs in range(z0, z1, args.slab)
        for ys in range(y0, y1, tile)
        for xs in range(x0, x1, tile)
    ]
    mine = jobs[args.shard_id :: args.num_shards]

    suffix = "" if args.num_shards == 1 else f".{args.shard_id}"
    prog = Path(f"{args.out}.progress{suffix}")
    done = set(prog.read_text().split()) if prog.exists() else set()
    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(mine)} of {len(jobs)} jobs "
        f"(slab {args.slab}, tile {tile}); resuming: {len(done)} already done",
        flush=True,
    )

    t0 = time.time()
    n = 0
    for zs, ys, xs in mine:
        key = f"{zs}_{ys}_{xs}"
        if key in done:
            continue
        ze, ye, xe = min(zs + args.slab, z1), min(ys + tile, y1), min(xs + tile, x1)
        block = cv[xs:xe, ys:ye, zs:ze]                        # (dx, dy, dz, channels)
        block = np.asarray(block[..., 0]).transpose(2, 1, 0)   # -> (dz, dy, dx)
        arr[zs - z0 : ze - z0, ys - y0 : ye - y0, xs - x0 : xe - x0] = block
        with open(prog, "a") as f:
            f.write(f"{key}\n")
        n += 1
        elapsed = time.time() - t0
        left = (len(mine) - len(done) - n) * elapsed / n / 60
        print(
            f"  z[{zs}:{ze}] y[{ys}:{ye}] x[{xs}:{xe}] done "
            f"({n} this run, {elapsed:.0f}s, {elapsed / n:.0f}s/job, ~{left:.0f} min left)",
            flush=True,
        )
    print(f"DONE -> {args.out}/{args.dataset}", flush=True)


if __name__ == "__main__":
    main()
