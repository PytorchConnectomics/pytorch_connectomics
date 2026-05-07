"""Re-stitch a chunked raw prediction back into the canonical CZYX h5.

Mirrors `connectomics/inference/chunked.py::_stitch_chunk_prediction_files`
without requiring SLURM/Lightning, so a corrupt or interrupted stitch output
can be rebuilt from the intact ``<base>.h5.chunks/`` directory and
``<base>.h5.index.json``.

Usage::

    python scripts/stitch_chunked_prediction.py path/to/prediction.h5
    python scripts/stitch_chunked_prediction.py path/to/prediction.h5 --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py

from connectomics.inference.artifact import write_prediction_artifact


def stitch(base: Path, *, slab: int = 64, force: bool = False) -> Path:
    chunks_dir = Path(str(base) + ".chunks")
    index_path = Path(str(base) + ".index.json")

    if not chunks_dir.is_dir():
        raise SystemExit(f"Chunks directory missing: {chunks_dir}")
    if not index_path.is_file():
        raise SystemExit(f"Index missing: {index_path}")

    if base.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing {base}; pass --force.")
        base.unlink()

    idx = json.loads(index_path.read_text())
    final_shape = tuple(int(v) for v in idx["final_shape"])
    chunks = idx["chunks"]
    if not chunks:
        raise SystemExit("Index has no chunks.")

    first_path = chunks_dir / Path(chunks[0]["path"]).name
    with h5py.File(first_path, "r") as f:
        first = f["main"]
        channel_count = int(first.shape[0])
        out_dtype = first.dtype

    print(
        f"Stitching {len(chunks)} chunks → "
        f"({channel_count}, {final_shape[0]}, {final_shape[1]}, {final_shape[2]}) {out_dtype}"
    )

    t0 = time.time()

    def writer(dset) -> None:
        for chunk_idx, c in enumerate(chunks, start=1):
            chunk_path = chunks_dir / Path(c["path"]).name
            s = [int(v) for v in c["start_zyx"]]
            e = [int(v) for v in c["stop_zyx"]]
            spatial = (e[0] - s[0], e[1] - s[1], e[2] - s[2])
            with h5py.File(chunk_path, "r") as f:
                src = f["main"]
                if int(src.shape[0]) != channel_count:
                    raise SystemExit(
                        f"Channel mismatch in {chunk_path.name}: "
                        f"{int(src.shape[0])} vs {channel_count}"
                    )
                if tuple(int(v) for v in src.shape[-3:]) != spatial:
                    raise SystemExit(
                        f"Spatial mismatch in {chunk_path.name}: "
                        f"{tuple(int(v) for v in src.shape[-3:])} vs {spatial}"
                    )
                for z0 in range(0, spatial[0], slab):
                    z1 = min(z0 + slab, spatial[0])
                    dset[
                        :,
                        s[0] + z0 : s[0] + z1,
                        s[1] : e[1],
                        s[2] : e[2],
                    ] = src[:, z0:z1, :, :]
            elapsed = time.time() - t0
            print(f"  [{chunk_idx}/{len(chunks)}] {c['key']} done ({elapsed:.0f}s elapsed)")

    write_prediction_artifact(
        base,
        data=None,
        dataset="main",
        compression="gzip",
        shape=(channel_count, *final_shape),
        dtype=out_dtype,
        chunks=(channel_count, slab, slab, slab),
        writer=writer,
    )

    elapsed = time.time() - t0
    size_gb = base.stat().st_size / 1e9
    print(f"Stitched in {elapsed:.1f}s → {base}")
    print(f"Output size: {size_gb:.2f} GB")

    with h5py.File(base, "r") as f:
        d = f["main"]
        print(f"Verified: shape={d.shape}, dtype={d.dtype}")

    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base",
        type=Path,
        help="Path to the canonical stitched h5 (the file alongside <base>.chunks/ and "
        "<base>.index.json).",
    )
    parser.add_argument(
        "--slab",
        type=int,
        default=64,
        help="Z-slab size for streaming each chunk into the output (default 64).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the stitched h5 if it already exists.",
    )
    args = parser.parse_args()
    stitch(args.base, slab=args.slab, force=args.force)


if __name__ == "__main__":
    main()
