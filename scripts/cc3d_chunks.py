"""Per-chunk cc3d connected-components on chunked affinity predictions.

For each `chunk_z*_y*_x*.h5` under a chunked raw-prediction `.h5.chunks` dir,
load the (3, Z, Y, X) short-range affinity, optionally zero out masked voxels,
and label connected components of the thresholded foreground via
`cc3d.connected_components((aff[:3] > threshold).any(0))` — bit-identical to
connectomics' `decode_affinity_cc(backend="cc3d")` (the nisb-base operation).
Writes a uint32 label volume per chunk to a sibling output dir. Labels are LOCAL
per chunk (1..N, 0=bg); cross-chunk stitching is a separate step.

Mask layers (mirroring lib/em_pipeline tasks/waterz.py), applied to the affinity
BEFORE cc3d, exactly as `aff *= mask`:

  v1 (default): no masking.
  v2 (--border-mask): per z-section bbox `[y_min, y_max, x_min, x_max]` from
      `mask_align_10nm_thres/%04d.txt`; keep only `[y_min+W : y_max-W,
      x_min+W : x_max-W]` with W=`--border-width` (128).
  v3 (--vessel-mask, in addition to border): zero voxels where the blood-vessel
      mask is set. Read from the 80nm `yl_bv_80nm.h5` (uint8, nonzero=vessel),
      upsampled by `--vessel-ratio` (4 8 8) to the 10nm frame. Masks align
      top-left with the affinity tile frame.

Shards by chunk index for `just slurm-sharded` (appends --shard-id/--num-shards);
re-runnable (chunks whose output exists are skipped):

    just slurm-sharded short 8 0 20 \
        "python scripts/cc3d_chunks.py --chunks-dir <...>.h5.chunks --threshold 0.66 \
         --border-mask --vessel-mask" "" 48G
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cc3d
import h5py
import numpy as np

_CHUNK_RE = re.compile(r"chunk_(z\d+_y\d+_x\d+)\.h5$")


def decode_affinity_cc3d(aff: np.ndarray, threshold: float) -> np.ndarray:
    """Foreground connected components on thresholded short-range affinities.

    Bit-identical to connectomics' decode_affinity_cc(backend="cc3d"). Inlined so
    this stays a standalone script (no connectomics/torch import).
    """
    foreground = (aff[:3] > threshold).any(axis=0)
    return cc3d.connected_components(foreground)


def _chunk_files(chunks_dir: Path) -> list[Path]:
    return sorted(chunks_dir.glob("chunk_z*_y*_x*.h5"))


def _load_chunk_starts(chunks_dir: Path) -> dict[str, tuple[int, int, int]]:
    """Map chunk key (e.g. 'z0_y1_x2') -> core start (z, y, x) from the index json."""
    index_path = chunks_dir.parent / chunks_dir.name.replace(".h5.chunks", ".h5.index.json")
    if not index_path.exists():
        raise SystemExit(f"Masking needs the chunk index json (chunk start coords): {index_path}")
    idx = json.load(open(index_path))
    return {c["key"]: tuple(int(v) for v in c["start_zyx"]) for c in idx["chunks"]}


def _load_border_bboxes(glob_pattern: str, n_sections: int) -> np.ndarray:
    """Preload per-section border bboxes [y_min, y_max, x_min, x_max]; (n_sections, 4)."""
    bboxes = np.zeros((n_sections, 4), np.int64)
    for z in range(n_sections):
        bboxes[z] = np.loadtxt(glob_pattern % z).astype(np.int64)
    return bboxes


def _build_keep_mask(start, shape, *, bboxes, border_width, vessel_ds, vessel_ratio):
    """Per-chunk boolean keep-mask (True=keep) for border and/or vessel masking."""
    z0, y0, x0 = start
    Z, Y, X = shape
    keep = np.ones((Z, Y, X), bool)

    if bboxes is not None:
        for zi in range(Z):
            ymin, ymax, xmin, xmax = bboxes[z0 + zi]
            vy0 = max(ymin + border_width - y0, 0)
            vy1 = min(ymax - border_width - y0, Y)
            vx0 = max(xmin + border_width - x0, 0)
            vx1 = min(xmax - border_width - x0, X)
            keep[zi] = False
            if vy1 > vy0 and vx1 > vx0:
                keep[zi, vy0:vy1, vx0:vx1] = True

    if vessel_ds is not None:
        rz, ry, rx = vessel_ratio
        bz, by, bx = vessel_ds.shape
        # 1008 is divisible by 4 and 8, so chunk starts land on 80nm voxel edges.
        z80, y80, x80 = z0 // rz, y0 // ry, x0 // rx
        z81 = min(-(-(z0 + Z) // rz), bz)  # ceil division, clipped to volume
        y81 = min(-(-(y0 + Y) // ry), by)
        x81 = min(-(-(x0 + X) // rx), bx)
        vessel = np.zeros((Z, Y, X), bool)  # beyond 80nm coverage = no vessel (border handles padding)
        if z81 > z80 and y81 > y80 and x81 > x80:
            sub = vessel_ds[z80:z81, y80:y81, x80:x81]
            up = np.repeat(np.repeat(np.repeat(sub, rz, 0), ry, 1), rx, 2) > 0
            uz, uy, ux = min(Z, up.shape[0]), min(Y, up.shape[1]), min(X, up.shape[2])
            vessel[:uz, :uy, :ux] = up[:uz, :uy, :ux]
        keep &= ~vessel

    return keep


def _write_labels(path: Path, labels: np.ndarray) -> None:
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h:
        h.create_dataset(
            "main", data=labels, compression="gzip", compression_opts=1,
            chunks=(min(64, labels.shape[0]), min(64, labels.shape[1]), min(64, labels.shape[2])),
        )
    os.replace(tmp, path)  # atomic: a partial file never looks "done" to skip-existing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks-dir", required=True, help="The raw-prediction <...>.h5.chunks directory.")
    p.add_argument("--threshold", type=float, default=0.66, help="Affinity foreground threshold.")
    p.add_argument("--dataset", default="main", help="HDF5 dataset key in each chunk file.")
    p.add_argument("--output-dir", default=None, help="Override output dir (default: sibling tagged by mode).")
    p.add_argument("--border-mask", action="store_true", help="v2: apply per-section border bbox mask.")
    p.add_argument("--border-glob", default="/projects/weilab/dataset/zebrafinch/mask_align_10nm_thres/%04d.txt")
    p.add_argument("--border-width", type=int, default=128)
    p.add_argument("--n-sections", type=int, default=5700, help="Number of z-sections (for border preload).")
    p.add_argument("--vessel-mask", action="store_true", help="v3: zero blood-vessel voxels (implies border).")
    p.add_argument("--vessel-h5", default="/projects/weilab/dataset/zebrafinch/yl_bv_80nm.h5")
    p.add_argument("--vessel-ratio", type=int, nargs=3, default=[4, 8, 8], help="80nm->10nm upsample (z y x).")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.is_dir():
        raise SystemExit(f"--chunks-dir not found: {chunks_dir}")
    masking = args.border_mask or args.vessel_mask

    # Output dir tagged by mode so v1/v2/v3 don't collide.
    thr_tag = f"t{args.threshold:.2f}".replace(".", "")
    mode_tag = thr_tag + ("_bd" if args.border_mask else "") + ("_bv" if args.vessel_mask else "")
    out_dir = (Path(args.output_dir) if args.output_dir
               else chunks_dir.parent / chunks_dir.name.replace(".h5.chunks", f".cc3d_{mode_tag}.chunks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    starts = _load_chunk_starts(chunks_dir) if masking else None
    bboxes = _load_border_bboxes(args.border_glob, args.n_sections) if args.border_mask else None
    vessel_file = h5py.File(args.vessel_h5, "r") if args.vessel_mask else None
    vessel_ds = vessel_file[list(vessel_file)[0]] if vessel_file is not None else None

    files = _chunk_files(chunks_dir)
    mine = files[args.shard_id::args.num_shards]
    print(f"[cc3d{('+'+mode_tag) if masking else ''} shard {args.shard_id}/{args.num_shards}] "
          f"{len(mine)}/{len(files)} chunks -> {out_dir} (threshold={args.threshold})", flush=True)

    done = 0
    skipped = 0
    for n, f in enumerate(mine):
        out_path = out_dir / f.name
        if out_path.exists():
            continue
        try:
            with h5py.File(f, "r") as h:
                key = args.dataset if args.dataset in h else list(h)[0]
                aff = h[key][:]
        except OSError as exc:
            # A corrupt raw chunk (e.g. gzip "inflate() failed") must not abort the
            # whole shard; skip it so the remaining readable chunks still decode.
            skipped += 1
            print(f"  [skip] {f.name}: unreadable ({exc})", flush=True)
            continue
        if aff.ndim != 4 or aff.shape[0] < 3:
            raise SystemExit(f"{f.name}: expected (C>=3, Z, Y, X), got {aff.shape}")

        if masking:
            m = _CHUNK_RE.search(f.name)
            keep = _build_keep_mask(
                starts[m.group(1)], aff.shape[1:],
                bboxes=bboxes, border_width=args.border_width,
                vessel_ds=vessel_ds, vessel_ratio=tuple(args.vessel_ratio) if args.vessel_mask else None,
            )
            aff[:, ~keep] = 0  # em_pipeline: zero affinity at masked voxels before segmentation

        # float16 compares fine against the threshold; avoid a 2x memory blow-up.
        labels = decode_affinity_cc3d(aff, args.threshold)
        _write_labels(out_path, labels.astype(np.uint32, copy=False))
        done += 1
        if done % 5 == 1 or n == len(mine) - 1:
            print(f"  {f.name}: labels={int(labels.max())} ({n + 1}/{len(mine)})", flush=True)

    if vessel_file is not None:
        vessel_file.close()
    print(f"[cc3d shard {args.shard_id}/{args.num_shards}] wrote {done} new chunk labels"
          f"{f' ({skipped} unreadable skipped)' if skipped else ''}.", flush=True)


if __name__ == "__main__":
    main()
