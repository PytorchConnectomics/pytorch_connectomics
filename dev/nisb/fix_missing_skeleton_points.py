"""Chunked marker-watershed fill for NISB v3 erosion2 missing skeleton points.

The reusable algorithm lives in
``connectomics.decoding.decoders.segmentation_grow``. This script is the
large-volume driver for seed101: cc=0.66 labels are markers, raw ch0-1-2
affinities above 0.3 define foreground, and each chunk writes its non-
overlapping core directly into a Zarr output with per-chunk success/failure
sentinels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import zarr

from connectomics.decoding.decoders.segmentation_grow import grow_segmentation_from_affinity

RUN_DIR = Path("outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101")
DEFAULT_SEED = RUN_DIR / "decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66.h5"
DEFAULT_AFFINITY = RUN_DIR / "raw_x1_ch0-1-2.h5"
DEFAULT_WORK_ROOT = RUN_DIR / "seg_fusion/segmentation_grow"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--affinity", type=Path, default=DEFAULT_AFFINITY)
    ap.add_argument("--dataset", default="main")
    ap.add_argument("--affinity-dataset", default="main")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--output-format", choices=("zarr", "h5"), default="zarr")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--chunk-shape", type=int, nargs=3, default=(188, 376, 172))
    ap.add_argument(
        "--halo",
        type=int,
        default=None,
        help="Chunk halo. Defaults to --max-fill-steps.",
    )
    ap.add_argument("-j", "--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--foreground-threshold", type=float, default=0.3)
    ap.add_argument(
        "--channel-reduction",
        choices=("max", "min", "mean"),
        default="max",
    )
    ap.add_argument(
        "--max-fill-steps",
        type=int,
        default=64,
        help="Local geodesic growth cap; required for chunk-exact output.",
    )
    ap.add_argument("--connectivity", type=int, default=1)
    ap.add_argument("--cost-power", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stitch-only",
        action="store_true",
        help="Skip chunk processing and stitch existing chunk results.",
    )
    ap.add_argument(
        "--no-stitch",
        action="store_true",
        help="Only write per-chunk results.",
    )
    return ap.parse_args()


def open_h5(path: Path, mode: str):
    try:
        return h5py.File(path, mode, locking=False)
    except TypeError:
        return h5py.File(path, mode)


def slices_from_bounds(bounds: Sequence[Sequence[int]]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(start), int(stop)) for start, stop in bounds)  # type: ignore[return-value]


def bounds_from_slices(slc: tuple[slice, slice, slice]) -> tuple[tuple[int, int], ...]:
    return tuple((int(axis.start or 0), int(axis.stop or 0)) for axis in slc)


def pad_slices(
    slc: tuple[slice, slice, slice],
    shape: Sequence[int],
    halo: int,
) -> tuple[slice, slice, slice]:
    padded = []
    for axis, dim in zip(slc, shape):
        padded.append(
            slice(
                max(0, int(axis.start or 0) - int(halo)),
                min(int(dim), int(axis.stop or 0) + int(halo)),
            )
        )
    return tuple(padded)  # type: ignore[return-value]


def relative_core(
    core: tuple[slice, slice, slice],
    expanded: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    out = []
    for core_axis, expanded_axis in zip(core, expanded):
        start = int(core_axis.start or 0) - int(expanded_axis.start or 0)
        stop = int(core_axis.stop or 0) - int(expanded_axis.start or 0)
        out.append(slice(start, stop))
    return tuple(out)  # type: ignore[return-value]


def iter_core_slices(
    shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> Iterable[tuple[int, tuple[slice, slice, slice]]]:
    index = 0
    for z0 in range(0, int(shape[0]), int(chunk_shape[0])):
        for y0 in range(0, int(shape[1]), int(chunk_shape[1])):
            for x0 in range(0, int(shape[2]), int(chunk_shape[2])):
                index += 1
                yield index, (
                    slice(z0, min(z0 + int(chunk_shape[0]), int(shape[0]))),
                    slice(y0, min(y0 + int(chunk_shape[1]), int(shape[1]))),
                    slice(x0, min(x0 + int(chunk_shape[2]), int(shape[2]))),
                )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    tag = (
        f"fg{args.foreground_threshold:g}_steps{args.max_fill_steps}_"
        f"chunk{'x'.join(str(v) for v in args.chunk_shape)}"
    )
    work_dir = args.work_dir or (DEFAULT_WORK_ROOT / tag)
    suffix = ".zarr" if args.output_format == "zarr" else ".h5"
    output = args.output or (work_dir / f"{args.seed.stem}_segmentation_grow_{tag}{suffix}")
    return work_dir, output


def inspect_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[int, int, int], np.dtype, tuple[int, ...] | None]:
    with open_h5(args.seed, "r") as seed_h5, open_h5(args.affinity, "r") as aff_h5:
        seed_ds = seed_h5[args.dataset]
        aff_ds = aff_h5[args.affinity_dataset]
        seed_shape = tuple(int(v) for v in seed_ds.shape)
        if aff_ds.ndim == 4:
            aff_shape = tuple(int(v) for v in aff_ds.shape[1:])
        elif aff_ds.ndim == 3:
            aff_shape = tuple(int(v) for v in aff_ds.shape)
        else:
            raise ValueError(f"Affinity dataset must be 3D or CZYX 4D, got shape {aff_ds.shape}.")
        if seed_shape != aff_shape:
            raise ValueError(f"Seed shape {seed_shape} does not match affinity shape {aff_shape}.")
        return seed_shape, np.dtype(seed_ds.dtype), seed_ds.chunks


def chunk_result_path(chunk_dir: Path, index: int) -> Path:
    return chunk_dir / f"chunk_{index:05d}.npz"


def chunk_done_path(chunk_dir: Path, index: int) -> Path:
    return chunk_dir / f"chunk_{index:05d}.done"


def chunk_fail_path(chunk_dir: Path, index: int) -> Path:
    return chunk_dir / f"chunk_{index:05d}.fail"


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def create_or_open_zarr_output(
    output_path: Path,
    *,
    shape: Sequence[int],
    dtype: np.dtype,
    chunk_shape: Sequence[int],
    force: bool,
) -> bool:
    """Create the direct-write Zarr output.

    Returns True when an existing output was reused, which allows the caller to
    honor existing per-chunk ``.done`` sentinels.
    """
    existed = output_path.exists()
    if existed and force:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
        existed = False

    if existed:
        arr = zarr.open(str(output_path), mode="r+")
        if tuple(arr.shape) != tuple(int(v) for v in shape):
            raise ValueError(f"Existing Zarr shape {arr.shape} != expected {tuple(shape)}")
        if np.dtype(arr.dtype) != np.dtype(dtype):
            raise ValueError(f"Existing Zarr dtype {arr.dtype} != expected {dtype}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = zarr.open(
        str(output_path),
        mode="w",
        shape=tuple(int(v) for v in shape),
        chunks=tuple(int(v) for v in chunk_shape),
        dtype=np.dtype(dtype),
    )
    arr.attrs.update(
        {
            "writer": "dev/nisb/fix_missing_skeleton_points.py",
            "chunk_shape": [int(v) for v in chunk_shape],
        }
    )
    return False


def run_chunk(spec: dict[str, object]) -> dict[str, object]:
    index = int(spec["index"])
    chunk_dir = Path(str(spec["chunk_dir"]))
    done_path = chunk_done_path(chunk_dir, index)
    fail_path = chunk_fail_path(chunk_dir, index)
    output_format = str(spec["output_format"])
    result_path = chunk_result_path(chunk_dir, index)
    can_skip = bool(spec["can_skip_done"]) and not bool(spec["force"])
    if can_skip and done_path.exists():
        if output_format == "zarr" or result_path.exists():
            return {"index": index, "status": "skipped", "changed_voxels": 0}

    try:
        if fail_path.exists():
            fail_path.unlink()
        if bool(spec["force"]) and done_path.exists():
            done_path.unlink()

        core = slices_from_bounds(spec["core_bounds"])  # type: ignore[arg-type]
        expanded = slices_from_bounds(spec["expanded_bounds"])  # type: ignore[arg-type]
        rel = relative_core(core, expanded)

        with (
            open_h5(Path(str(spec["seed"])), "r") as seed_h5,
            open_h5(Path(str(spec["affinity"])), "r") as aff_h5,
        ):
            seed_ds = seed_h5[str(spec["dataset"])]
            aff_ds = aff_h5[str(spec["affinity_dataset"])]
            seed_crop = np.asarray(seed_ds[expanded])
            if aff_ds.ndim == 4:
                aff_crop = np.asarray(aff_ds[(slice(None),) + expanded])
            else:
                aff_crop = np.asarray(aff_ds[expanded])

        filled = grow_segmentation_from_affinity(
            seed_crop,
            aff_crop,
            foreground_threshold=float(spec["foreground_threshold"]),
            channel_reduction=str(spec["channel_reduction"]),  # type: ignore[arg-type]
            max_fill_steps=int(spec["max_fill_steps"]),
            connectivity=int(spec["connectivity"]),
            cost_power=float(spec["cost_power"]),
        )
        core_seed = seed_crop[rel]
        core_filled = filled[rel].astype(core_seed.dtype, copy=False)
        changed = int(np.count_nonzero(core_filled != core_seed))

        payload = {
            "index": index,
            "status": "done",
            "changed_voxels": changed,
            "core_bounds": bounds_from_slices(core),
            "expanded_bounds": bounds_from_slices(expanded),
            "output_format": output_format,
        }
        if output_format == "zarr":
            out = zarr.open(str(spec["output"]), mode="r+")
            out[core] = core_filled
        else:
            np.savez(
                result_path,
                core=core_filled,
                core_bounds=np.asarray(bounds_from_slices(core), dtype=np.int64),
                expanded_bounds=np.asarray(bounds_from_slices(expanded), dtype=np.int64),
                changed_voxels=np.asarray(changed, dtype=np.uint64),
            )
            payload["result"] = str(result_path)

        write_status(done_path, payload)
        return {"index": index, "status": "done", "changed_voxels": changed}
    except Exception as exc:  # noqa: BLE001 - worker should record the failing chunk.
        payload = {
            "index": index,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_status(fail_path, payload)
        return {"index": index, "status": "failed", "changed_voxels": 0, "error": repr(exc)}


def process_chunks(
    specs: list[dict[str, object]],
    *,
    workers: int,
) -> list[dict[str, object]]:
    if workers <= 1:
        return [run_chunk(spec) for spec in specs]

    results = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(run_chunk, spec) for spec in specs]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if i == 1 or i == len(futures) or i % 25 == 0:
                print(
                    f"  chunks {i}/{len(futures)}: "
                    f"#{result['index']} {result['status']} "
                    f"changed={result['changed_voxels']}",
                    flush=True,
                )
    return results


def stitch_output(
    *,
    seed_path: Path,
    dataset: str,
    output_path: Path,
    chunk_dir: Path,
    specs: list[dict[str, object]],
    force: bool,
) -> None:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {output_path}")
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(seed_path, "r") as seed_h5, h5py.File(output_path, "w") as out_h5:
        seed_ds = seed_h5[dataset]
        kwargs: dict[str, object] = {}
        if seed_ds.chunks is not None:
            kwargs["chunks"] = seed_ds.chunks
        if seed_ds.compression is not None:
            kwargs["compression"] = seed_ds.compression
            kwargs["compression_opts"] = seed_ds.compression_opts
        if seed_ds.shuffle:
            kwargs["shuffle"] = seed_ds.shuffle
        if seed_ds.fletcher32:
            kwargs["fletcher32"] = seed_ds.fletcher32

        out_ds = out_h5.create_dataset(
            dataset,
            shape=seed_ds.shape,
            dtype=seed_ds.dtype,
            **kwargs,
        )
        for key, value in seed_h5.attrs.items():
            out_h5.attrs[key] = value
        for key, value in seed_ds.attrs.items():
            out_ds.attrs[key] = value

        for i, spec in enumerate(specs, start=1):
            index = int(spec["index"])
            result_path = chunk_result_path(chunk_dir, index)
            if not result_path.exists():
                raise FileNotFoundError(f"Missing chunk result: {result_path}")
            data = np.load(result_path, allow_pickle=False)
            core = slices_from_bounds(np.asarray(data["core_bounds"], dtype=np.int64))
            out_ds[core] = data["core"]
            if i == 1 or i == len(specs) or i % 50 == 0:
                print(f"  stitched {i}/{len(specs)} chunks", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_fill_steps < 0:
        raise ValueError("--max-fill-steps must be nonnegative for chunked execution.")
    halo = int(args.max_fill_steps if args.halo is None else args.halo)
    if halo < int(args.max_fill_steps):
        raise ValueError("--halo must be >= --max-fill-steps for chunk-local correctness.")

    shape, seed_dtype, seed_chunks = inspect_inputs(args)
    work_dir, output = resolve_paths(args)
    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format == "zarr" and args.stitch_only:
        raise ValueError("--stitch-only only applies to --output-format h5.")

    specs: list[dict[str, object]] = []
    for index, core in iter_core_slices(shape, args.chunk_shape):
        expanded = pad_slices(core, shape, halo)
        specs.append(
            {
                "index": index,
                "seed": str(args.seed),
                "affinity": str(args.affinity),
                "dataset": args.dataset,
                "affinity_dataset": args.affinity_dataset,
                "chunk_dir": str(chunk_dir),
                "output": str(output),
                "output_format": args.output_format,
                "can_skip_done": False,
                "core_bounds": bounds_from_slices(core),
                "expanded_bounds": bounds_from_slices(expanded),
                "foreground_threshold": args.foreground_threshold,
                "channel_reduction": args.channel_reduction,
                "max_fill_steps": args.max_fill_steps,
                "connectivity": args.connectivity,
                "cost_power": args.cost_power,
                "force": args.force,
            }
        )

    metadata = {
        "seed": str(args.seed),
        "affinity": str(args.affinity),
        "dataset": args.dataset,
        "affinity_dataset": args.affinity_dataset,
        "shape": shape,
        "seed_dtype": str(seed_dtype),
        "seed_chunks": seed_chunks,
        "chunk_shape": tuple(int(v) for v in args.chunk_shape),
        "halo": halo,
        "foreground_threshold": float(args.foreground_threshold),
        "channel_reduction": args.channel_reduction,
        "max_fill_steps": int(args.max_fill_steps),
        "connectivity": int(args.connectivity),
        "cost_power": float(args.cost_power),
        "num_chunks": len(specs),
        "output_format": args.output_format,
        "output": str(output),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(
        f"shape={shape} chunks={len(specs)} chunk_shape={tuple(args.chunk_shape)} "
        f"halo={halo} workers={args.workers}",
        flush=True,
    )
    print(f"work_dir={work_dir}", flush=True)
    print(f"output={output}", flush=True)

    if args.dry_run:
        return

    zarr_reused = False
    if args.output_format == "zarr":
        zarr_reused = create_or_open_zarr_output(
            output,
            shape=shape,
            dtype=seed_dtype,
            chunk_shape=args.chunk_shape,
            force=bool(args.force),
        )
        for spec in specs:
            spec["can_skip_done"] = zarr_reused

    results: list[dict[str, object]] = []
    if not args.stitch_only:
        results = process_chunks(specs, workers=int(args.workers))
        total_changed = int(sum(int(r["changed_voxels"]) for r in results))
        failures = [r for r in results if r["status"] == "failed"]
        (work_dir / "chunk_summary.json").write_text(
            json.dumps(
                {
                    "total_changed_voxels": total_changed,
                    "failed_chunks": len(failures),
                    "chunks": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"changed_voxels={total_changed}", flush=True)
        if failures:
            failed_ids = ",".join(str(r["index"]) for r in failures[:20])
            raise RuntimeError(
                f"{len(failures)} chunks failed; see {chunk_dir}/*.fail. "
                f"First failed chunk ids: {failed_ids}"
            )

    if args.output_format == "zarr":
        print(f"wrote {output}", flush=True)
    elif not args.no_stitch:
        stitch_output(
            seed_path=args.seed,
            dataset=args.dataset,
            output_path=output,
            chunk_dir=chunk_dir,
            specs=specs,
            force=args.force,
        )
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
