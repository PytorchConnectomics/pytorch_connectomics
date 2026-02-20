#!/usr/bin/env python3
"""
Downsample NISB base zarr volumes and save as HDF5.

Input pattern:
  /projects/weilab/dataset/nisb/base/{train,val,test}/*/data.zarr/{img,seg}

Output files (written next to each source array):
  /projects/weilab/dataset/nisb/base/{train,val,test}/*/data.zarr/img_{z}-{y}-{x}nm.h5
  /projects/weilab/dataset/nisb/base/{train,val,test}/*/data.zarr/seg_{z}-{y}-{x}nm.h5

Downsampling rule:
  - Default resolution conversion: (z,y,x) 20x9x9 nm -> 40x36x36 nm
    using --downsample-ratio-zyx 2 4 4
  - Downsample ratio in zyx order is configurable.
  - Stored data axes are (x,y,z) for seg and (x,y,z,c) for img
  - img: ndimage.zoom downsample in xy + strided sampling in z
  - seg: strided sampling in xyz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from connectomics.data.io import write_hdf5


def _require_zarr():
    try:
        import zarr  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "This script requires the `zarr` package. Install with: pip install zarr"
        ) from exc
    return zarr


def _len_from_step(size: int, step: int) -> int:
    return (size + step - 1) // step


def zoom_downsample_xy(
    image_xy: np.ndarray,
    out_shape_xy: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """
    Downsample one XY slice using scipy.ndimage.zoom.

    Args:
        image_xy: 2D image slice.
        out_shape_xy: target shape (x_out, y_out).
    """
    zoom_factors = (
        out_shape_xy[0] / image_xy.shape[0],
        out_shape_xy[1] / image_xy.shape[1],
    )
    down = ndimage.zoom(
        image_xy,
        zoom=zoom_factors,
        order=order,
        mode="nearest",
        prefilter=(order > 1),
    )
    return down.astype(image_xy.dtype, copy=False)


def downsample_data_zarr(
    data_zarr_path: Path,
    *,
    downsample_ratio_zyx: tuple[int, int, int] = (2, 4, 4),
    overwrite: bool = False,
    dataset_key: str = "main",
    compression: str | None = "gzip",
    compression_level: int = 4,
    img_zoom_order: int = 1,
) -> tuple[Path, Path]:
    """
    Downsample one seed's `data.zarr` and write required HDF5 outputs.
    """
    zarr = _require_zarr()

    input_resolution_zyx_nm = (20, 9, 9)
    output_resolution_zyx_nm = tuple(
        in_nm * ratio for in_nm, ratio in zip(input_resolution_zyx_nm, downsample_ratio_zyx)
    )
    resolution_suffix = "-".join(str(v) for v in output_resolution_zyx_nm) + "nm"
    img_out_path = data_zarr_path / f"img_{resolution_suffix}.h5"
    seg_out_path = data_zarr_path / f"seg_{resolution_suffix}.h5"

    if not overwrite and img_out_path.exists() and seg_out_path.exists():
        print(f"Skip (exists): {data_zarr_path}")
        return img_out_path, seg_out_path

    root = zarr.open(str(data_zarr_path), mode="r")
    if "img" not in root or "seg" not in root:
        raise KeyError(f"Expected 'img' and 'seg' in {data_zarr_path}")

    img = root["img"]  # shape: (x, y, z, c)
    seg = root["seg"]  # shape: (x, y, z)

    if img.ndim != 4:
        raise ValueError(f"Expected img ndim=4, got {img.ndim} in {data_zarr_path}")
    if seg.ndim != 3:
        raise ValueError(f"Expected seg ndim=3, got {seg.ndim} in {data_zarr_path}")
    if img.shape[:3] != seg.shape:
        raise ValueError(
            f"img/seg spatial shape mismatch in {data_zarr_path}: {img.shape[:3]} vs {seg.shape}"
        )
    if img.shape[3] < 1:
        raise ValueError(f"img has no channel dimension in {data_zarr_path}")

    # Stored axes are (x, y, z). User provides ratio in (z, y, x).
    z_factor, y_factor, x_factor = downsample_ratio_zyx
    if min(z_factor, y_factor, x_factor) <= 0:
        raise ValueError(f"All downsample ratios must be positive, got {downsample_ratio_zyx}")

    x_in, y_in, z_in = seg.shape
    x_out = _len_from_step(x_in, x_factor)
    y_out = _len_from_step(y_in, y_factor)
    z_out = _len_from_step(z_in, z_factor)

    print(
        f"Processing {data_zarr_path}\n"
        f"  Downsample ratio (z,y,x): {downsample_ratio_zyx}\n"
        f"  Input shape  (x,y,z): ({x_in}, {y_in}, {z_in})\n"
        f"  Output shape (z,y,x): ({z_out}, {y_out}, {x_out})"
    )

    # Save outputs in zyx order.
    out_img = np.empty((z_out, y_out, x_out), dtype=img.dtype)
    out_seg = np.empty((z_out, y_out, x_out), dtype=seg.dtype)

    for out_z, in_z in enumerate(range(0, z_in, z_factor)):
        # seg: strided downsample in xyz using configured ratios
        seg_xy = np.asarray(seg[::x_factor, ::y_factor, in_z])
        out_seg[out_z, :, :] = seg_xy.T

        # img: zoom downsample in xy and ::2 in z
        img_xy = np.asarray(img[:, :, in_z, 0])
        img_ds = zoom_downsample_xy(
            img_xy,
            out_shape_xy=(x_out, y_out),
            order=img_zoom_order,
        )
        out_img[out_z, :, :] = img_ds.T

        if out_z % 50 == 0 or out_z == z_out - 1:
            print(f"  z-slices: {out_z + 1}/{z_out}")

    write_hdf5(
        str(img_out_path),
        out_img,
        dataset=dataset_key,
        compression=compression,
        compression_level=compression_level,
    )
    write_hdf5(
        str(seg_out_path),
        out_seg,
        dataset=dataset_key,
        compression=compression,
        compression_level=compression_level,
    )

    print(f"  Wrote: {img_out_path}")
    print(f"  Wrote: {seg_out_path}")
    return img_out_path, seg_out_path


def iter_data_zarr_paths(base_dir: Path, splits: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        split_paths = sorted((base_dir / split).glob("*/data.zarr"))
        paths.extend(split_paths)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample NISB data.zarr and write H5 outputs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/projects/weilab/dataset/nisb/base"),
        help="Base NISB directory containing train/val/test.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--downsample-ratio-zyx",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=[2, 4, 4],
        help="Downsample ratio in zyx order. Default: 2 4 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="main",
        help="Dataset key to write in each output h5.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression method.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (used only when --compression gzip).",
    )
    parser.add_argument(
        "--img-zoom-order",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="Interpolation order for ndimage.zoom when downsampling img in XY.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of work shards (for parallel SLURM tasks).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, num-shards).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compression = None if args.compression == "none" else args.compression
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be > 0, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    paths = iter_data_zarr_paths(args.base_dir, args.splits)
    if not paths:
        raise FileNotFoundError(
            f"No data.zarr found under {args.base_dir} for splits {args.splits}"
        )

    shard_paths = [
        path for idx, path in enumerate(paths) if idx % args.num_shards == args.shard_index
    ]
    print(
        f"Found {len(paths)} data.zarr directories total; "
        f"shard {args.shard_index}/{args.num_shards} will process {len(shard_paths)}"
    )
    for path in shard_paths:
        downsample_data_zarr(
            path,
            downsample_ratio_zyx=tuple(args.downsample_ratio_zyx),
            overwrite=args.overwrite,
            dataset_key=args.dataset_key,
            compression=compression,
            compression_level=args.compression_level,
            img_zoom_order=args.img_zoom_order,
        )


# Example:
# python scripts/downsample_nisb.py --base-dir /projects/weilab/dataset/nisb/base --splits train val test --downsample-ratio-zyx 2 4 4
# Parallel shard example (task 3 of 7):
# python scripts/downsample_nisb.py --base-dir /projects/weilab/dataset/nisb/base --splits train val test --downsample-ratio-zyx 2 4 4 --num-shards 7 --shard-index 3
if __name__ == "__main__":
    main()
