#!/usr/bin/env python3
"""
Crop a sub-volume from a NISB data.zarr and save as HDF5.

Input pattern:
  /projects/weilab/dataset/nisb/base/{split}/{seed}/data.zarr/{img,seg}
  Stored axes are (x, y, z) for seg and (x, y, z, c) for img.

Output (default: next to source):
  {data_zarr_path}/img_crop_{cx}-{cy}-{cz}_x{nx}_y{ny}_z{nz}.h5
  {data_zarr_path}/seg_crop_{cx}-{cy}-{cz}_x{nx}_y{ny}_z{nz}.h5

Crop is specified by size in xyz; origin defaults to the volume center.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from connectomics.data.io import write_hdf5


def _require_zarr():
    try:
        import zarr  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This script requires the `zarr` package. Install with: pip install zarr"
        ) from exc
    return zarr


def _resolve_origin(
    origin_xyz: tuple[int, int, int] | None,
    size_xyz: tuple[int, int, int],
    full_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
    if origin_xyz is None:
        return tuple((full - size) // 2 for full, size in zip(full_xyz, size_xyz))  # type: ignore[return-value]
    return origin_xyz


def crop_data_zarr(
    data_zarr_path: Path,
    *,
    size_xyz: tuple[int, int, int],
    origin_xyz: tuple[int, int, int] | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
    dataset_key: str = "main",
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> tuple[Path, Path]:
    """Crop one seed's `data.zarr` and write img/seg HDF5 outputs in xyz layout.

    Segmentation crops are relabeled with connected components before saving so
    disconnected fragments do not share one instance ID inside the crop.
    """
    zarr = _require_zarr()

    root = zarr.open(str(data_zarr_path), mode="r")
    if "img" not in root or "seg" not in root:
        raise KeyError(f"Expected 'img' and 'seg' in {data_zarr_path}")

    img = root["img"]  # shape: (x, y, z, c)
    seg = root["seg"]  # shape: (x, y, z)
    if img.ndim != 4 or seg.ndim != 3:
        raise ValueError(f"Unexpected ndim in {data_zarr_path}: img={img.shape}, seg={seg.shape}")
    if img.shape[:3] != seg.shape:
        raise ValueError(
            f"img/seg spatial shape mismatch in {data_zarr_path}: {img.shape[:3]} vs {seg.shape}"
        )

    full_xyz = tuple(int(s) for s in seg.shape)
    nx, ny, nz = size_xyz
    if min(nx, ny, nz) <= 0:
        raise ValueError(f"Crop size must be positive, got {size_xyz}")
    for full, size, axis in zip(full_xyz, size_xyz, "xyz"):
        if size > full:
            raise ValueError(
                f"Crop size {size} exceeds full size {full} on axis {axis} in {data_zarr_path}"
            )

    x0, y0, z0 = _resolve_origin(origin_xyz, size_xyz, full_xyz)
    x1, y1, z1 = x0 + nx, y0 + ny, z0 + nz
    for v0, v1, full, axis in zip((x0, y0, z0), (x1, y1, z1), full_xyz, "xyz"):
        if v0 < 0 or v1 > full:
            raise ValueError(
                f"Crop on axis {axis} out of bounds: [{v0}:{v1}] vs full {full} in {data_zarr_path}"
            )

    out_dir = Path(output_dir) if output_dir is not None else data_zarr_path
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"crop_{x0}-{y0}-{z0}_x{nx}_y{ny}_z{nz}.h5"
    img_out_path = out_dir / f"img_{suffix}"
    seg_out_path = out_dir / f"seg_{suffix}"

    if not overwrite and img_out_path.exists() and seg_out_path.exists():
        print(f"Skip (exists): {img_out_path}, {seg_out_path}")
        return img_out_path, seg_out_path

    print(
        f"Processing {data_zarr_path}\n"
        f"  Full shape (x,y,z): {full_xyz}\n"
        f"  Crop origin (x,y,z): ({x0}, {y0}, {z0})\n"
        f"  Crop size   (x,y,z): ({nx}, {ny}, {nz})"
    )

    img_xyz = np.asarray(img[x0:x1, y0:y1, z0:z1, 0])
    seg_xyz = np.asarray(seg[x0:x1, y0:y1, z0:z1])
    try:
        import cc3d
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Cropping segmentation requires cc3d for connected-component relabeling. "
            "Install connected-components-3d."
        ) from exc

    seg_xyz = cc3d.connected_components(
        seg_xyz,
        connectivity=6,
        out_dtype=np.uint32,
    )

    write_hdf5(
        str(img_out_path),
        img_xyz,
        dataset=dataset_key,
        compression=compression,
        compression_level=compression_level,
    )
    write_hdf5(
        str(seg_out_path),
        seg_xyz,
        dataset=dataset_key,
        compression=compression,
        compression_level=compression_level,
    )
    print(f"  Wrote: {img_out_path} ({img_xyz.dtype})")
    print(f"  Wrote: {seg_out_path} ({seg_xyz.dtype}, cc3d relabeled)")
    return img_out_path, seg_out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a sub-volume from NISB data.zarr and write HDF5 outputs."
    )
    parser.add_argument(
        "--data-zarr",
        type=Path,
        required=True,
        help="Path to a single data.zarr (e.g. /.../test/seed101/data.zarr).",
    )
    parser.add_argument(
        "--size-xyz",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Crop size in xyz, e.g. 1024 1024 200.",
    )
    parser.add_argument(
        "--origin-xyz",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Crop origin in xyz. Defaults to volume center.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to writing inside the source data.zarr path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-key", type=str, default="main")
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
    )
    parser.add_argument("--compression-level", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    crop_data_zarr(
        args.data_zarr,
        size_xyz=tuple(args.size_xyz),
        origin_xyz=tuple(args.origin_xyz) if args.origin_xyz is not None else None,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dataset_key=args.dataset_key,
        compression=compression,
        compression_level=args.compression_level,
    )


# Example:
# python scripts/crop_nisb.py \
#   --data-zarr /projects/weilab/dataset/nisb/base/test/seed101/data.zarr \
#   --size-xyz 1024 1024 200 \
#   --output-dir /projects/weilab/weidf/lib/pytorch_connectomics/dev/nisb/data
if __name__ == "__main__":
    main()
