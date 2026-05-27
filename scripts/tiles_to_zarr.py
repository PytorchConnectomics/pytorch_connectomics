"""Convert a tiled PNG volume (e.g. zebrafinch im_align_10nm) to a multiscale OME-Zarr.

Why: the tiled-PNG reader decodes a full 4096x4096 PNG per z-section for every
ROI, so large-volume inference spends most of its wall-time at 0% GPU. Converting
once to a chunked, compressed zarr makes reads true random-access and lets the GPU
stay fed. The output is an OME-NGFF multiscale group (full-res + 2x/4x/8x ...),
directly usable by inference (`data.test.image: <out>.zarr/0`), neuroglancer, and
napari.

Workflow (the destructive `remove-png` step is intentionally separate and gated):

    # 1. create the group + per-level arrays + OME metadata (run once)
    python scripts/tiles_to_zarr.py --source SRC --output OUT.zarr --stage init

    # 2. fill full resolution from the PNGs (shard across the cluster)
    python scripts/tiles_to_zarr.py --source SRC --output OUT.zarr --stage base \
        --shard-id $i --num-shards $N --workers 16

    # 3. build the downsampled pyramid from level 0 (cheap; single job is fine)
    python scripts/tiles_to_zarr.py --source SRC --output OUT.zarr --stage pyramid

    # 4. verify level 0 byte-for-byte against the PNGs (shard across the cluster)
    python scripts/tiles_to_zarr.py --source SRC --output OUT.zarr --stage verify \
        --shard-id $i --num-shards $N --workers 16

    # 5. ONLY after every verify shard signed off: delete the raw PNGs
    python scripts/tiles_to_zarr.py --source SRC --output OUT.zarr --stage remove-png \
        --num-shards $N --yes

Steps 2 and 4 slot directly into `just slurm-sharded` (one task per shard id).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np

from connectomics.data.io.tiles import reconstruct_volume_from_tiles
from connectomics.inference.lazy import _load_tile_metadata

VERIFY_DIR = "_verify_ok"


def _level_shapes(base_shape: tuple[int, int, int], levels: int) -> list[tuple[int, int, int]]:
    shapes = [tuple(int(v) for v in base_shape)]
    for _ in range(1, levels):
        prev = shapes[-1]
        shapes.append(tuple(max(1, math.ceil(n / 2)) for n in prev))
    return shapes


def _iter_blocks(shape, block):
    """Yield (zslice, yslice, xslice) covering `shape` in `block`-sized tiles."""
    zb, yb, xb = block
    for z0 in range(0, shape[0], zb):
        for y0 in range(0, shape[1], yb):
            for x0 in range(0, shape[2], xb):
                yield (
                    slice(z0, min(z0 + zb, shape[0])),
                    slice(y0, min(y0 + yb, shape[1])),
                    slice(x0, min(x0 + xb, shape[2])),
                )


def _my_blocks(blocks, shard_id, num_shards):
    return [b for i, b in enumerate(blocks) if i % num_shards == shard_id]


def _read_source_block(meta, block) -> np.ndarray:
    """Decode the source PNG region for `block` (parallel over z-sections)."""
    tile_h, tile_w = (int(meta["tile_size"][0]), int(meta["tile_size"][1])) \
        if isinstance(meta["tile_size"], (list, tuple)) else (int(meta["tile_size"]),) * 2
    zs, ys, xs = block
    return reconstruct_volume_from_tiles(
        tile_paths=meta["image"],
        volume_coords=[zs.start, zs.stop, ys.start, ys.stop, xs.start, xs.stop],
        tile_coords=[0, int(meta["depth"]), 0, int(meta["height"]), 0, int(meta["width"])],
        tile_size=[tile_h, tile_w],
        data_type=np.dtype(meta.get("dtype", "uint8")),
        tile_start=meta.get("tile_st", [0, 0]),
        tile_ratio=float(meta.get("tile_ratio", 1.0)),
        is_image=True,
        num_workers=int(os.environ.get("TILE_READ_WORKERS", "1")),
    )


def _png_file_list(meta) -> list[str]:
    """Expand the tile patterns into concrete source PNG paths."""
    tile_h, tile_w = int(meta["tile_size"][0]), int(meta["tile_size"][1])
    n_rows = int(meta["height"]) // tile_h
    n_cols = int(meta["width"]) // tile_w
    r0, c0 = meta.get("tile_st", [0, 0])
    files = []
    for pattern in meta["image"]:
        if "{row}_{column}" in pattern:
            for r in range(n_rows):
                for c in range(n_cols):
                    files.append(pattern.format(row=r + r0, column=c + c0))
        else:
            files.append(pattern)
    return files


def stage_init(args, meta):
    import numcodecs
    import zarr

    base_shape = (int(meta["depth"]), int(meta["height"]), int(meta["width"]))
    shapes = _level_shapes(base_shape, args.levels)
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to recreate metadata/arrays.")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    root = zarr.open_group(args.output, mode="a")
    for level, shp in enumerate(shapes):
        chunks = tuple(min(c, s) for c, s in zip(args.chunk, shp))
        root.create_dataset(
            str(level), shape=shp, chunks=chunks, dtype="uint8",
            compressor=compressor, overwrite=args.force, fill_value=0,
        )
    res = [float(r) for r in args.resolution]
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": Path(args.output).stem,
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in ("z", "y", "x")],
        "datasets": [
            {"path": str(l),
             "coordinateTransformations": [{"type": "scale", "scale": [r * (2 ** l) for r in res]}]}
            for l in range(args.levels)
        ],
    }]
    print(f"Initialized {args.output} with {args.levels} levels:")
    for l, shp in enumerate(shapes):
        print(f"  level {l}: shape={shp} chunks={tuple(min(c, s) for c, s in zip(args.chunk, shp))}")


def stage_base(args, meta):
    import zarr

    os.environ["TILE_READ_WORKERS"] = str(args.workers)
    arr = zarr.open(args.output, mode="r+")["0"]
    blocks = list(_iter_blocks(arr.shape, (args.z_block, args.xy_block, args.xy_block)))
    mine = _my_blocks(blocks, args.shard_id, args.num_shards)
    print(f"[base shard {args.shard_id}/{args.num_shards}] {len(mine)}/{len(blocks)} blocks")
    for n, block in enumerate(mine):
        arr[block] = _read_source_block(meta, block)
        if n % 5 == 0 or n == len(mine) - 1:
            print(f"  wrote block {n + 1}/{len(mine)} z={block[0]} y={block[1]} x={block[2]}", flush=True)


def stage_pyramid(args, meta):
    from skimage.measure import block_reduce
    import zarr

    root = zarr.open(args.output, mode="r+")
    levels = range(1, args.levels) if args.level is None else [args.level]
    for level in levels:
        src, dst = root[str(level - 1)], root[str(level)]
        blocks = list(_iter_blocks(dst.shape, (args.z_block, args.xy_block, args.xy_block)))
        mine = _my_blocks(blocks, args.shard_id, args.num_shards)
        print(f"[pyramid L{level} shard {args.shard_id}/{args.num_shards}] {len(mine)}/{len(blocks)} blocks")
        for n, (zs, ys, xs) in enumerate(mine):
            region = src[2 * zs.start: min(2 * zs.stop, src.shape[0]),
                         2 * ys.start: min(2 * ys.stop, src.shape[1]),
                         2 * xs.start: min(2 * xs.stop, src.shape[2])]
            down = block_reduce(region.astype(np.float32), (2, 2, 2), np.mean)
            down = np.round(down).astype(np.uint8)[: zs.stop - zs.start, : ys.stop - ys.start, : xs.stop - xs.start]
            dst[zs, ys, xs] = down
            if n % 5 == 0 or n == len(mine) - 1:
                print(f"  L{level} block {n + 1}/{len(mine)}", flush=True)


def stage_verify(args, meta):
    import zarr

    os.environ["TILE_READ_WORKERS"] = str(args.workers)
    arr = zarr.open(args.output, mode="r")["0"]
    base_shape = (int(meta["depth"]), int(meta["height"]), int(meta["width"]))
    if tuple(arr.shape) != base_shape:
        raise SystemExit(f"shape mismatch: zarr {tuple(arr.shape)} vs source {base_shape}")
    blocks = list(_iter_blocks(arr.shape, (args.z_block, args.xy_block, args.xy_block)))
    mine = _my_blocks(blocks, args.shard_id, args.num_shards)
    print(f"[verify shard {args.shard_id}/{args.num_shards}] {len(mine)}/{len(blocks)} blocks")
    for n, block in enumerate(mine):
        if not np.array_equal(arr[block], _read_source_block(meta, block)):
            raise SystemExit(f"MISMATCH at z={block[0]} y={block[1]} x={block[2]} — do NOT delete PNGs")
        if n % 5 == 0 or n == len(mine) - 1:
            print(f"  verified block {n + 1}/{len(mine)}", flush=True)
    ok_dir = Path(args.output) / VERIFY_DIR
    ok_dir.mkdir(exist_ok=True)
    (ok_dir / f"shard_{args.shard_id}_of_{args.num_shards}.ok").write_text("ok\n")
    print(f"[verify shard {args.shard_id}] OK")


def stage_remove_png(args, meta):
    if not args.yes:
        raise SystemExit("Refusing to delete PNGs without --yes.")
    ok_dir = Path(args.output) / VERIFY_DIR
    missing = [i for i in range(args.num_shards)
               if not (ok_dir / f"shard_{i}_of_{args.num_shards}.ok").exists()]
    if missing:
        raise SystemExit(
            f"Refusing to delete: verify sign-off missing for shards {missing} "
            f"(expected {args.num_shards} '.ok' files in {ok_dir}). Run --stage verify first."
        )
    files = [f for f in _png_file_list(meta) if os.path.exists(f)]
    print(f"All {args.num_shards} verify shards signed off. Deleting {len(files)} PNG files...")
    for n, f in enumerate(files):
        os.remove(f)
        if n % 5000 == 0:
            print(f"  removed {n}/{len(files)}", flush=True)
    # Drop now-empty section directories.
    for d in sorted({os.path.dirname(f) for f in files}):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    print(f"Removed {len(files)} PNG files.")


STAGES = {
    "init": stage_init, "base": stage_base, "pyramid": stage_pyramid,
    "verify": stage_verify, "remove-png": stage_remove_png,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Tile metadata JSON or tiled directory.")
    p.add_argument("--output", required=True, help="Output .zarr path.")
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--levels", type=int, default=4, help="Number of pyramid levels (full-res + downsamples).")
    p.add_argument("--chunk", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--z-block", type=int, default=128, help="Z extent of a write block (multiple of chunk z).")
    p.add_argument("--xy-block", type=int, default=4096, help="Y/X extent of a write block (multiple of chunk).")
    p.add_argument("--resolution", type=float, nargs=3, default=[10.0, 10.0, 10.0], help="Level-0 voxel size (nm).")
    p.add_argument("--workers", type=int, default=16, help="Threads for PNG decode in base/verify.")
    p.add_argument("--level", type=int, default=None, help="pyramid: build only this level (for sharding).")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--force", action="store_true", help="init: overwrite existing arrays.")
    p.add_argument("--yes", action="store_true", help="remove-png: confirm deletion.")
    args = p.parse_args()

    for name, value in (("z_block", args.z_block), ("xy_block", args.xy_block)):
        if value % min(args.chunk) != 0 and value % args.chunk[0] != 0:
            print(f"  WARNING: --{name.replace('_', '-')}={value} is not chunk-aligned; "
                  "sharded writes may touch shared chunks and race.")

    meta = _load_tile_metadata(args.source)
    STAGES[args.stage](args, meta)


if __name__ == "__main__":
    main()
