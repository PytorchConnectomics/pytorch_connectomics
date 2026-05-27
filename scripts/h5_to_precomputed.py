#!/usr/bin/env python3
"""Convert an HDF5 volume to Neuroglancer precomputed format.

Reads the HDF5 input with ``connectomics.data.io.read_volume`` and writes
z-slab chunks to a local CloudVolume precomputed directory. Volumes are
assumed to be in connectomics (z, y, x) layout; CloudVolume metadata and
indexing use (x, y, z), so CLI triplets are accepted in z y x order and
reversed internally.

Usage:
    python scripts/h5_to_precomputed.py input.h5 output_precomputed --resolution 40 4 4
    python scripts/h5_to_precomputed.py labels.h5 labels_precomputed \
        --layer-type segmentation --resolution 40 4 4
    python scripts/h5_to_precomputed.py pred.h5 pred_precomputed \
        --dataset main --channel 1 --resolution 40 4 4
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from cloudvolume import CloudVolume
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "cloud-volume is required to write Neuroglancer precomputed data. "
        "Install it with: pip install cloud-volume"
    ) from exc

from connectomics.data.io import read_volume


def load_volume(input_path: str, dataset: str | None, channel: int) -> np.ndarray:
    """Load a 3D (z, y, x) volume, selecting one channel from 4D input."""
    volume = read_volume(input_path, dataset=dataset)
    if volume.ndim == 4:
        if channel < 0 or channel >= volume.shape[0]:
            raise ValueError(
                f"Channel index {channel} is out of bounds for input shape {volume.shape}"
            )
        volume = volume[channel]

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume or 4D (C,Z,Y,X) volume, got shape {volume.shape}")

    return volume


def zyx_to_xyz(values: list[int] | list[float]) -> list[int] | list[float]:
    """Convert a zyx triplet to xyz order."""
    return list(reversed(values))


def prepare_output_volume(
    volume: np.ndarray,
    layer_type: str,
    encoding: str | None,
) -> tuple[np.ndarray, np.dtype, str]:
    """Apply deterministic dtype and encoding rules for CloudVolume output."""
    input_dtype = volume.dtype

    if layer_type == "image":
        return volume, input_dtype, encoding or "raw"

    if layer_type != "segmentation":
        raise ValueError(f"Unsupported layer type: {layer_type}")

    if input_dtype in (np.dtype("uint32"), np.dtype("uint64")):
        return volume, input_dtype, encoding or "compressed_segmentation"

    if input_dtype in (np.dtype("bool"), np.dtype("uint8"), np.dtype("uint16")):
        print(
            f"Upcasting segmentation labels from {input_dtype} to uint32 "
            "for compressed_segmentation."
        )
        return volume.astype(np.uint32, copy=False), np.dtype("uint32"), (
            encoding or "compressed_segmentation"
        )

    raise ValueError(f"segmentation labels must be unsigned integer; got {input_dtype}")


def convert_h5_to_precomputed(
    input_path: str,
    output_path: str,
    *,
    dataset: str | None,
    channel: int,
    layer_type: str,
    resolution_zyx: list[float],
    offset_zyx: list[int],
    chunk_size_zyx: list[int],
    chunk_z: int,
    encoding: str | None,
) -> None:
    """Convert one HDF5 volume to a base-mip precomputed directory."""
    if chunk_z <= 0:
        raise ValueError(f"--chunk-z must be positive, got {chunk_z}")
    if min(chunk_size_zyx) <= 0:
        raise ValueError(f"--chunk-size values must be positive, got {chunk_size_zyx}")
    if min(resolution_zyx) <= 0:
        raise ValueError(f"--resolution values must be positive, got {resolution_zyx}")

    volume = load_volume(input_path, dataset=dataset, channel=channel)
    input_dtype = volume.dtype
    output_volume, output_dtype, resolved_encoding = prepare_output_volume(
        volume,
        layer_type=layer_type,
        encoding=encoding,
    )

    size_xyz = list(output_volume.shape[::-1])
    resolution_xyz = zyx_to_xyz(resolution_zyx)
    offset_xyz = zyx_to_xyz(offset_zyx)
    chunk_size_xyz = zyx_to_xyz(chunk_size_zyx)
    abs_output = os.path.abspath(output_path)
    cloudpath = f"file://{abs_output}"

    print(f"Input:  {input_path}")
    print(f"  dataset:          {dataset if dataset is not None else '<first dataset>'}")
    print(f"  shape (z,y,x):    {tuple(volume.shape)}")
    print(f"  input dtype:      {input_dtype}")
    print(f"  output dtype:     {output_dtype}")
    print(f"  layer type:       {layer_type}")
    print(f"  encoding:         {resolved_encoding}")
    print(f"  resolution zyx:   {resolution_zyx}")
    print(f"  resolution xyz:   {resolution_xyz}")
    print(f"  offset zyx:       {offset_zyx}")
    print(f"  offset xyz:       {offset_xyz}")
    print(f"  chunk size zyx:   {chunk_size_zyx}")
    print(f"  chunk size xyz:   {chunk_size_xyz}")
    print(f"Output: {abs_output}")

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=str(output_dtype),
        encoding=resolved_encoding,
        resolution=resolution_xyz,
        voxel_offset=offset_xyz,
        volume_size=size_xyz,
        chunk_size=chunk_size_xyz,
        max_mip=0,
    )
    vol = CloudVolume(
        cloudpath,
        info=info,
        mip=0,
        bounded=True,
        progress=False,
        compress=False,
        fill_missing=True,
    )
    vol.commit_info()
    vol.non_aligned_writes = True

    z_size, y_size, x_size = output_volume.shape
    ox, oy, oz = offset_xyz
    n_chunks = (z_size + chunk_z - 1) // chunk_z
    print(f"Writing chunks: {n_chunks} z-slabs")

    for chunk_idx, z0 in enumerate(range(0, z_size, chunk_z), start=1):
        z1 = min(z0 + chunk_z, z_size)
        slab_xyz = np.ascontiguousarray(output_volume[z0:z1].transpose(2, 1, 0))
        vol[ox : ox + x_size, oy : oy + y_size, oz + z0 : oz + z1] = slab_xyz[..., np.newaxis]
        pct = (z1 / z_size) * 100
        print(f"  [{chunk_idx}/{n_chunks}] z={z0}:{z1}  ({pct:.0f}%)", flush=True)

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 3D HDF5 volume to Neuroglancer precomputed format.",
    )
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("output", help="Output precomputed directory")
    parser.add_argument(
        "--dataset",
        default=None,
        help="HDF5 dataset name (default: first dataset)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to select for 4D (C,Z,Y,X) input (default: 0)",
    )
    parser.add_argument(
        "--layer-type",
        choices=["image", "segmentation"],
        default="image",
        help="Neuroglancer layer type (default: image)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        required=True,
        help="Voxel resolution in nanometers, z y x order (required)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=[0, 0, 0],
        help="Voxel offset in z y x order (default: 0 0 0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=[64, 64, 64],
        help="Precomputed chunk size in z y x order (default: 64 64 64)",
    )
    parser.add_argument(
        "--chunk-z",
        type=int,
        default=64,
        help="Source z-slab thickness for writes (default: 64)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help=(
            "Override precomputed encoding (default: raw for image, "
            "compressed_segmentation for segmentation)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_h5_to_precomputed(
        args.input,
        args.output,
        dataset=args.dataset,
        channel=args.channel,
        layer_type=args.layer_type,
        resolution_zyx=args.resolution,
        offset_zyx=args.offset,
        chunk_size_zyx=args.chunk_size,
        chunk_z=args.chunk_z,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
