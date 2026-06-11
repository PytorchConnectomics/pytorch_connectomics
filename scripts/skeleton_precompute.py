"""Precompute NISB skeleton volumes from hole-filled GT seg (seg_filled).

Phase 1 of the chunked SDT pipeline. Skeletonization is global per instance (not spatially
decomposable), so each SLURM array task processes one seed, reads data.zarr/seg_filled, and writes
data.zarr/seg_filled_skeleton via the standard sdt_path_for_label(..., mode="skeleton") cache path.

Run AFTER the chunked fill array:
    python scripts/skeleton_precompute.py --task 0 --seeds 0 1 2 3 4 100

Run from repo root (env: pytc). One seed per task (SLURM array).
"""
from __future__ import annotations

import argparse

BASE = "/projects/weilab/dataset/nisb/base"


def zpath_for(seed: int, path: str | None = None) -> str:
    if path is not None:
        return path
    sub = f"train/seed{seed}" if seed < 100 else f"val/seed{seed}"
    return f"{BASE}/{sub}/data.zarr"


def require_zarr_array(path: str, role: str) -> None:
    import zarr

    try:
        zarr.open(path, mode="r")
    except Exception as exc:
        raise FileNotFoundError(f"missing {role}: {path}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="seeds (train 0-4, val 100) defining the flat task grid")
    ap.add_argument("--task", type=int, required=True,
                    help="flat task id = seed_index")
    ap.add_argument("--label-key", default="seg_filled")
    ap.add_argument("--resolution", type=float, nargs=3, default=[9, 9, 20],
                    help="array-axis order")
    ap.add_argument("--path", default=None,
                    help="override data.zarr path for a single-seed offline check")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="processes for label-parallel skeletonization (default 1)")
    ap.add_argument("--downsample-xy", type=int, default=1,
                    help="subsample the two fine axes by this factor for skeletonization "
                         "(2 ~5x faster; output skeleton stays full-res)")
    args = ap.parse_args()

    if args.task < 0 or args.task >= len(args.seeds):
        ap.error(f"--task must be in [0, {len(args.seeds) - 1}] for {len(args.seeds)} seeds")

    from connectomics.data.processing.distance import precompute_skeleton_volume, sdt_path_for_label

    seed = args.seeds[args.task]
    zpath = zpath_for(seed, args.path)
    label_path = f"{zpath}/{args.label_key}"
    out = sdt_path_for_label(label_path, mode="skeleton")

    require_zarr_array(label_path, "hole-filled label volume")
    print(f"Skeleton: {label_path} -> {out}  (res={tuple(args.resolution)})", flush=True)
    precompute_skeleton_volume(label_path, out, resolution=tuple(args.resolution),
                               num_workers=args.num_workers, downsample_xy=args.downsample_xy)
    print("done", flush=True)


if __name__ == "__main__":
    main()
