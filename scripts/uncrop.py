#!/usr/bin/env python3
"""Pad a cropped HDF5 volume with a constant value.

This is intended for fixing old prediction/segmentation outputs that were
cropped by a known number of voxels during inference.

Example:
    python scripts/uncrop.py \
        img_x1_ch0-1-2_ckpt-step-step=00050000_decoding_affinity_cc_numba-0-0.7.h5 \
        img_x1_ch0-1-2_ckpt-step-step=00050000_decoding_affinity_cc_numba-0-0.7_uncrop1.h5 \
        --k 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def _resolve_dataset(handle: h5py.File, dataset: str | None) -> str:
    if dataset is not None:
        if dataset not in handle:
            raise KeyError(f"Dataset {dataset!r} not found. Available: {list(handle.keys())}")
        return dataset

    datasets = [key for key, value in handle.items() if isinstance(value, h5py.Dataset)]
    if len(datasets) != 1:
        raise ValueError(
            "Input must contain exactly one top-level dataset when --dataset is omitted. "
            f"Available datasets: {datasets}"
        )
    return datasets[0]


def _parse_pad_width(k: int, side: str, ndim: int, spatial_dims: int) -> list[tuple[int, int]]:
    if k < 0:
        raise ValueError(f"--k must be >= 0, got {k}")
    if spatial_dims <= 0:
        raise ValueError(f"--spatial-dims must be > 0, got {spatial_dims}")
    if spatial_dims > ndim:
        raise ValueError(f"--spatial-dims={spatial_dims} exceeds input ndim={ndim}")

    if side == "before":
        spatial_pad = (k, 0)
    elif side == "after":
        spatial_pad = (0, k)
    elif side == "both":
        spatial_pad = (k, k)
    else:
        raise ValueError(f"Unknown --side {side!r}")

    leading_dims = ndim - spatial_dims
    return [(0, 0)] * leading_dims + [spatial_pad] * spatial_dims


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _compression_kwargs(compression: str | None, compression_level: int) -> dict:
    if compression in (None, "", "none"):
        return {}
    kwargs = {"compression": compression}
    if compression == "gzip":
        kwargs["compression_opts"] = compression_level
    return kwargs


def uncrop_h5(
    input_path: str,
    output_path: str,
    *,
    dataset: str | None = "main",
    k: int = 1,
    side: str = "after",
    value: float = 0,
    dtype: str | None = None,
    spatial_dims: int = 3,
    chunk_axis_size: int | None = None,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    with h5py.File(input_path, "r") as f_in:
        dataset_name = _resolve_dataset(f_in, dataset)
        dset_in = f_in[dataset_name]
        pad_width = _parse_pad_width(k, side, dset_in.ndim, spatial_dims)
        output_shape = tuple(
            int(size) + int(before) + int(after)
            for size, (before, after) in zip(dset_in.shape, pad_width)
        )
        output_dtype = np.dtype(dtype) if dtype is not None else dset_in.dtype
        write_slices = tuple(
            slice(before, before + size) for size, (before, _after) in zip(dset_in.shape, pad_width)
        )

        spatial_axis = dset_in.ndim - spatial_dims
        n_axis = int(dset_in.shape[spatial_axis])
        if chunk_axis_size is None:
            if dset_in.chunks is not None:
                chunk_axis_size = int(dset_in.chunks[spatial_axis])
            else:
                chunk_axis_size = 16
        chunk_axis_size = max(1, min(int(chunk_axis_size), n_axis))

        fill_value = np.asarray(value, dtype=output_dtype).item()
        output_parent = Path(output_path).resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f_out:
            _copy_attrs(f_in, f_out)
            dset_out = f_out.create_dataset(
                dataset_name,
                shape=output_shape,
                dtype=output_dtype,
                chunks=True,
                fillvalue=fill_value,
                **_compression_kwargs(compression, compression_level),
            )
            _copy_attrs(dset_in, dset_out)
            dset_out.attrs["uncrop_k"] = int(k)
            dset_out.attrs["uncrop_side"] = str(side)
            dset_out.attrs["uncrop_value"] = fill_value

            print(f"Input:   {input_path}")
            print(f"Dataset: {dataset_name}")
            print(f"Shape:   {tuple(dset_in.shape)} -> {output_shape}")
            print(f"Dtype:   {dset_in.dtype} -> {output_dtype}")
            print(f"Pad:     {pad_width}")
            print(f"Value:   {fill_value}")
            print(f"Output:  {output_path}")

            n_chunks = (n_axis + chunk_axis_size - 1) // chunk_axis_size
            for idx, start in enumerate(range(0, n_axis, chunk_axis_size), start=1):
                stop = min(start + chunk_axis_size, n_axis)
                read = [slice(None)] * dset_in.ndim
                read[spatial_axis] = slice(start, stop)
                write = list(write_slices)
                write[spatial_axis] = slice(
                    write_slices[spatial_axis].start + start,
                    write_slices[spatial_axis].start + stop,
                )
                dset_out[tuple(write)] = dset_in[tuple(read)].astype(output_dtype, copy=False)
                print(f"  [{idx}/{n_chunks}] axis{spatial_axis}={start}:{stop}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input cropped HDF5 file.")
    parser.add_argument("output", help="Output uncropped HDF5 file.")
    parser.add_argument("--dataset", default="main", help="HDF5 dataset name (default: main).")
    parser.add_argument("--k", type=int, default=1, help="Pad voxels per selected side.")
    parser.add_argument(
        "--side",
        default="after",
        choices=["before", "after", "both"],
        help="Where to add padding on each spatial axis (default: after).",
    )
    parser.add_argument("--value", type=float, default=0, help="Constant pad value (default: 0).")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Output dtype, e.g. uint32, uint64, float16, float32. Default: preserve input dtype.",
    )
    parser.add_argument(
        "--spatial-dims",
        type=int,
        default=3,
        help="Number of trailing spatial dimensions to uncrop (default: 3).",
    )
    parser.add_argument(
        "--chunk-axis-size",
        type=int,
        default=None,
        help="Chunk size along the first spatial axis (default: input chunk size or 16).",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="Output HDF5 compression: gzip, lzf, or none (default: gzip).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        uncrop_h5(
            args.input,
            args.output,
            dataset=args.dataset,
            k=args.k,
            side=args.side,
            value=args.value,
            dtype=args.dtype,
            spatial_dims=args.spatial_dims,
            chunk_axis_size=args.chunk_axis_size,
            compression=args.compression,
            compression_level=args.compression_level,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
