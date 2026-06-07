#!/usr/bin/env python3
"""Stream a (possibly large, multi-channel) HDF5 or Zarr volume to precomputed.

Unlike ``scripts/h5_to_precomputed.py`` (which loads the whole volume via
``read_volume``), this reads the source dataset in z-slabs so volumes larger
than RAM can be converted, supports multi-channel inputs as a multi-channel
layer, converts the on-disk dtype to a precomputed-valid dtype, and writes
to any CloudVolume backend (``file://``, ``gs://``, ``s3://``).

The source axis order is given explicitly via ``--src-axes`` (a permutation
of ``cxyz`` for 4D input or ``xyz`` for 3D). CloudVolume stores ``(X,Y,Z,C)``
and CLI resolution/offset/chunk triplets are given in z y x order.

Input format is auto-detected: a path ending in ``.zarr`` (or any directory
that is not an HDF5 file) is opened with ``zarr``; everything else is
opened with ``h5py``. ``--dataset`` selects the array name inside either.

GCS auth: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import h5py
import numpy as np

try:
    from cloudvolume import CloudVolume
except ImportError as exc:  # pragma: no cover
    raise SystemExit("cloud-volume is required. Install: pip install cloud-volume") from exc


def _is_zarr_path(path: str) -> bool:
    if path.rstrip("/").endswith(".zarr"):
        return True
    # Treat any directory input as zarr; HDF5 is a single file.
    return os.path.isdir(path)


@contextlib.contextmanager
def _open_dataset(path: str, dataset: str | None):
    """Yield ``(dset, resolved_name)`` for either HDF5 or Zarr inputs."""
    if _is_zarr_path(path):
        import zarr

        root = zarr.open(path, mode="r")
        name = dataset if dataset is not None else list(root)[0]
        yield root[name], name
    else:
        with h5py.File(path, "r") as f:
            name = dataset if dataset is not None else list(f)[0]
            yield f[name], name


def zyx_to_xyz(values):
    return list(reversed(values))


def _to_dtype(arr: np.ndarray, out_dtype: str) -> np.ndarray:
    target = np.dtype(out_dtype)
    if arr.dtype == target:
        return arr
    if out_dtype == "float32":
        return arr.astype(np.float32)
    if out_dtype == "uint8":
        # Treat as [0, 1] -> [0, 255] only when source is floating-point
        # (e.g. affinities). Integer sources pass through with a plain cast.
        if np.issubdtype(arr.dtype, np.floating):
            return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return arr.astype(np.uint8)
    return arr.astype(target)


def convert(
    input_path: str,
    cloudpath: str,
    *,
    dataset: str | None,
    src_axes: str,
    reverse_channels: bool,
    out_dtype: str,
    layer_type: str,
    resolution_zyx: list[float],
    offset_zyx: list[int],
    chunk_size_zyx: list[int],
    write_z: int,
    start_z: int,
    compress: bool,
) -> None:
    if not (cloudpath.startswith(("gs://", "s3://", "file://")) or "://" in cloudpath):
        cloudpath = f"file://{os.path.abspath(cloudpath)}"

    src_axes = src_axes.lower()
    src_kind = "zarr" if _is_zarr_path(input_path) else "hdf5"

    with _open_dataset(input_path, dataset) as (dset, resolved_name):
        dataset = resolved_name

        # Resolve which source axis is c/x/y/z.
        if dset.ndim == 4:
            if sorted(src_axes) != list("cxyz"):
                raise ValueError(f"--src-axes must be a permutation of 'cxyz' for 4D input, got {src_axes!r}")
            cax = src_axes.index("c")
            num_channels = dset.shape[cax]
        elif dset.ndim == 3:
            if sorted(src_axes) != list("xyz"):
                raise ValueError(f"--src-axes must be a permutation of 'xyz' for 3D input, got {src_axes!r}")
            cax = None
            num_channels = 1
        else:
            raise ValueError(f"Expected 3D or 4D dataset, got shape {dset.shape}")

        xax, yax, zax = (src_axes.index("x"), src_axes.index("y"), src_axes.index("z"))
        x_size, y_size, z_size = dset.shape[xax], dset.shape[yax], dset.shape[zax]

        resolution_xyz = zyx_to_xyz(resolution_zyx)
        offset_xyz = zyx_to_xyz(offset_zyx)
        chunk_size_xyz = zyx_to_xyz(chunk_size_zyx)
        ox, oy, oz = offset_xyz

        print(f"Input:  {input_path}  ({src_kind})")
        print(f"  dataset:        {dataset!r}")
        print(f"  shape:          {tuple(dset.shape)}  (src-axes={src_axes})")
        print(f"  in dtype:       {dset.dtype}")
        print(f"  out dtype:      {out_dtype}")
        print(f"  num_channels:   {num_channels}")
        print(f"  layer_type:     {layer_type}")
        print(f"  reverse_chans:  {reverse_channels}")
        print(f"  volume xyz:     [{x_size}, {y_size}, {z_size}]")
        print(f"  resolution xyz: {resolution_xyz}")
        print(f"  offset xyz:     {offset_xyz}")
        print(f"  chunk xyz:      {chunk_size_xyz}")
        print(f"  write_z:        {write_z}  (compress={compress})")
        print(f"Output: {cloudpath}")

        info = CloudVolume.create_new_info(
            num_channels=num_channels,
            layer_type=layer_type,
            data_type=out_dtype,
            encoding="raw",
            resolution=resolution_xyz,
            voxel_offset=offset_xyz,
            volume_size=[x_size, y_size, z_size],
            chunk_size=chunk_size_xyz,
            max_mip=0,
        )
        vol = CloudVolume(
            cloudpath, info=info, mip=0,
            compress=compress, progress=False, fill_missing=True,
        )
        vol.commit_info()
        vol.non_aligned_writes = True

        n_chunks = (z_size + write_z - 1) // write_z
        for idx, z0 in enumerate(range(0, z_size, write_z), start=1):
            z1 = min(z0 + write_z, z_size)
            if z1 <= start_z:
                print(f"  [{idx}/{n_chunks}] z={z0}:{z1} skipped (resume)", flush=True)
                continue
            sl = [slice(None)] * dset.ndim
            sl[zax] = slice(z0, z1)
            slab = dset[tuple(sl)]                      # source order, z axis sliced
            if reverse_channels and cax is not None:
                slab = np.flip(slab, axis=cax)          # e.g. zyx-affinity -> xyz-affinity
            slab = _to_dtype(slab, out_dtype)
            if cax is None:
                slab = np.ascontiguousarray(slab.transpose(xax, yax, zax))[..., np.newaxis]
            else:
                slab = np.ascontiguousarray(slab.transpose(xax, yax, zax, cax))  # (X, Y, dz, C)
            vol[ox:ox + x_size, oy:oy + y_size, oz + z0:oz + z1, :] = slab
            pct = (z1 / z_size) * 100
            print(f"  [{idx}/{n_chunks}] z={z0}:{z1}  ({pct:.1f}%)", flush=True)

    print("Done!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream an HDF5 or Zarr volume to precomputed (cloud-capable).")
    p.add_argument("input", help="Input HDF5 file or Zarr directory (auto-detected)")
    p.add_argument("output", help="Output cloudpath (gs://, s3://, file://, or local dir)")
    p.add_argument("--dataset", default=None, help="Source dataset / zarr array name (default: first)")
    p.add_argument("--src-axes", default="czyx",
                   help="Source axis order: permutation of cxyz (4D) or xyz (3D). Default: czyx")
    p.add_argument("--reverse-channels", action="store_true",
                   help="Reverse channel order on write (e.g. zyx-affinity -> xyz-affinity)")
    p.add_argument("--out-dtype", default="float32",
                   help="float32, uint8 (scale [0,1]->[0,255]), or any numpy dtype (default: float32)")
    p.add_argument("--layer-type", default="image", choices=["image", "segmentation"],
                   help="Precomputed layer type (default: image)")
    p.add_argument("--resolution", type=float, nargs=3, metavar=("Z", "Y", "X"), required=True,
                   help="Voxel resolution in nm, z y x order")
    p.add_argument("--offset", type=int, nargs=3, metavar=("Z", "Y", "X"), default=[0, 0, 0],
                   help="Voxel offset, z y x order (default: 0 0 0)")
    p.add_argument("--chunk-size", type=int, nargs=3, metavar=("Z", "Y", "X"), default=[64, 128, 128],
                   help="Precomputed chunk size, z y x order (default: 64 128 128)")
    p.add_argument("--write-z", type=int, default=64,
                   help="Source z-slab thickness per write; keep a multiple of chunk z (default: 64)")
    p.add_argument("--start-z", type=int, default=0, help="Resume: skip slabs ending at/below this z")
    p.add_argument("--no-compress", action="store_true", help="Disable gzip chunk compression")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    convert(
        a.input, a.output,
        dataset=a.dataset,
        src_axes=a.src_axes,
        reverse_channels=a.reverse_channels,
        out_dtype=a.out_dtype,
        layer_type=a.layer_type,
        resolution_zyx=a.resolution,
        offset_zyx=a.offset,
        chunk_size_zyx=a.chunk_size,
        write_z=a.write_z,
        start_z=a.start_z,
        compress=not a.no_compress,
    )


if __name__ == "__main__":
    main()
