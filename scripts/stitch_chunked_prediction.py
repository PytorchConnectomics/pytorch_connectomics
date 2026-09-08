"""Present a chunked raw prediction as the canonical CZYX h5.

Mirrors `connectomics/inference/chunked.py::_stitch_chunk_prediction_files`
without requiring SLURM/Lightning, so a corrupt or interrupted stitch output
can be rebuilt from the intact ``<base>.h5.chunks/`` directory and
``<base>.h5.index.json``.

``--vds`` writes an HDF5 *virtual* dataset over the same chunk files instead of
copying them. It is the only usable mode at whole-volume scale -- j0126's
affinity is ~4 TB, so a real stitch has nowhere to go -- and it costs seconds
because no voxel is read. The result opens through h5py exactly like a stitched
file, which is what `run_abiss_chunk.py` expects of `source_affinity_h5`.

``--discover DIR`` finds the store under DIR instead of naming it, so a job
submitted before inference has run can still resolve the timestamped output.

Usage::

    python scripts/stitch_chunked_prediction.py path/to/prediction.h5
    python scripts/stitch_chunked_prediction.py path/to/prediction.h5 --force
    python scripts/stitch_chunked_prediction.py --vds --discover outputs/x/affinity \
        --out outputs/x/affinity/affinity.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectomics.inference.artifact import write_prediction_artifact  # noqa: E402


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


def _read_index(base: Path) -> tuple[Path, dict]:
    """Return (chunks_dir, index payload) for a chunk store base path."""
    chunks_dir = Path(str(base) + ".chunks")
    index_path = Path(str(base) + ".index.json")
    if not chunks_dir.is_dir():
        raise SystemExit(f"Chunks directory missing: {chunks_dir}")
    if not index_path.is_file():
        raise SystemExit(f"Index missing: {index_path}")
    return chunks_dir, json.loads(index_path.read_text())


def discover(root: Path) -> Path:
    """Return the newest chunk-store base under `root`.

    Inference writes to a checkpoint-named subdirectory, so the exact path is
    not known until the job has run; a driver can name the run directory up
    front and let this resolve the store afterwards.
    """
    # Skip the stable-name symlinks a previous --vds run left, so this always
    # resolves the real store rather than a link back to itself.
    indexes = [p for p in sorted(root.glob("**/*.h5.index.json")) if not p.is_symlink()]
    if not indexes:
        raise SystemExit(f"No *.h5.index.json under {root}")
    newest = max(indexes, key=lambda p: p.stat().st_mtime)
    return newest.parent / newest.name[: -len(".index.json")]


def virtual_stitch(base: Path, out: Path, *, force: bool = False) -> Path:
    """Write `out` as an HDF5 virtual dataset over `base`'s per-chunk files.

    Layout is identical to `stitch`: one (C, Z, Y, X) dataset named ``main``,
    each chunk mapped at its own ``start_zyx``. Source paths are stored
    absolute, so the chunk store must not move afterwards.
    """
    chunks_dir, idx = _read_index(base)
    final_shape = tuple(int(v) for v in idx["final_shape"])
    chunks = idx["chunks"]
    if not chunks:
        raise SystemExit("Index has no chunks.")

    missing = [c["key"] for c in chunks if not (chunks_dir / Path(c["path"]).name).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(chunks)} chunks are missing, first {missing[:3]}; "
            "finish inference before building the virtual dataset."
        )

    first = chunks_dir / Path(chunks[0]["path"]).name
    with h5py.File(first, "r") as f:
        channel_count = int(f["main"].shape[0])
        dtype = f["main"].dtype

    if out.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing {out}; pass --force.")
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    layout = h5py.VirtualLayout(shape=(channel_count, *final_shape), dtype=dtype)
    for c in chunks:
        s = [int(v) for v in c["start_zyx"]]
        e = [int(v) for v in c["stop_zyx"]]
        spatial = (e[0] - s[0], e[1] - s[1], e[2] - s[2])
        source = h5py.VirtualSource(
            str((chunks_dir / Path(c["path"]).name).resolve()),
            "main",
            shape=(channel_count, *spatial),
            dtype=dtype,
        )
        layout[:, s[0] : e[0], s[1] : e[1], s[2] : e[2]] = source

    with h5py.File(out, "w") as f:
        f.create_virtual_dataset("main", layout, fillvalue=0)

    # Give the whole store one stable name. Downstream configs address the chunk
    # directory and index as `<base>.chunks` / `<base>.index.json`, and `base` is
    # a timestamped, checkpoint-named path nobody can write down in advance.
    if out.resolve() != base.resolve():
        for suffix, target in ((".chunks", chunks_dir), (".index.json", Path(str(base) + ".index.json"))):
            link = Path(str(out) + suffix)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve())

    print(
        f"Virtual dataset {out} -> {len(chunks)} chunks in {chunks_dir}\n"
        f"  shape=({channel_count}, {final_shape[0]}, {final_shape[1]}, {final_shape[2]}) "
        f"dtype={np.dtype(dtype).name}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base",
        type=Path,
        nargs="?",
        help="Path to the canonical stitched h5 (the file alongside <base>.chunks/ and "
        "<base>.index.json). Omit it when passing --discover.",
    )
    parser.add_argument(
        "--vds",
        action="store_true",
        help="Write a virtual dataset over the chunk files instead of copying them.",
    )
    parser.add_argument(
        "--discover",
        type=Path,
        help="Directory to search for the chunk store instead of naming <base>.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Destination path (--vds only; defaults to <base>).",
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
    if (args.base is None) == (args.discover is None):
        parser.error("pass exactly one of <base> or --discover")
    base = args.base if args.base is not None else discover(args.discover)
    if args.vds:
        virtual_stitch(base, args.out or base, force=args.force)
    else:
        if args.out is not None:
            parser.error("--out is only meaningful with --vds")
        stitch(base, slab=args.slab, force=args.force)


if __name__ == "__main__":
    main()
