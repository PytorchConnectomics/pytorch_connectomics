"""Convert a float32 HDF5 affinity volume to uint8 in chunks.

Reads z-slice chunks that fit in available RAM, converts float [0,1] to
uint8 [0,255], and writes to a new HDF5 file.  Never loads the entire
volume into memory.

Usage:
    python scripts/convert_h5_to_uint8.py input.h5 output.h5
    python scripts/convert_h5_to_uint8.py input.h5 output.h5 --ram-fraction 0.5
    python scripts/convert_h5_to_uint8.py input.h5 output.h5 --dataset main --chunk-z 10
"""

import argparse
import os
import sys

import h5py
import numpy as np
import psutil


def get_available_ram_bytes():
    """Return available RAM in bytes."""
    return psutil.virtual_memory().available


def estimate_chunk_z(shape, dtype_in, dtype_out, ram_fraction=0.3):
    """Estimate how many z-slices can be read + converted at once.

    Needs memory for one input chunk + one output chunk simultaneously.
    """
    available = get_available_ram_bytes() * ram_fraction
    # Per z-slice: input (float32) + output (uint8)
    bytes_per_z_in = int(np.prod(shape[1:])) * np.dtype(dtype_in).itemsize
    bytes_per_z_out = int(np.prod(shape[1:])) * np.dtype(dtype_out).itemsize
    bytes_per_z = bytes_per_z_in + bytes_per_z_out
    chunk_z = max(1, int(available / bytes_per_z))
    return min(chunk_z, shape[0])  # don't exceed total z


def convert(input_path, output_path, dataset="main", ram_fraction=0.3,
            chunk_z=None, compression="gzip"):
    """Convert float32 HDF5 to uint8 in z-chunks."""
    with h5py.File(input_path, "r") as f_in:
        if dataset not in f_in:
            datasets = list(f_in.keys())
            if len(datasets) == 1:
                dataset = datasets[0]
                print(f"Using dataset: {dataset}")
            else:
                print(f"Available datasets: {datasets}")
                sys.exit(1)

        dset_in = f_in[dataset]
        shape = dset_in.shape
        dtype_in = dset_in.dtype
        ndim = len(shape)

        print(f"Input:  {input_path}")
        print(f"  dataset: {dataset}")
        print(f"  shape:   {shape}")
        print(f"  dtype:   {dtype_in}")
        print(f"  size:    {dset_in.id.get_storage_size() / 1e9:.2f} GB (compressed)")
        print(f"  raw:     {np.prod(shape) * dtype_in.itemsize / 1e9:.2f} GB")

        if dtype_in == np.uint8:
            print("Already uint8, nothing to do.")
            return

        # Determine z-axis (first axis for (C,Z,Y,X) or (Z,Y,X))
        if ndim == 4:
            z_axis = 1  # shape = (C, Z, Y, X)
            n_z = shape[1]
            slice_shape = (shape[0], 1, shape[2], shape[3])
        elif ndim == 3:
            z_axis = 0  # shape = (Z, Y, X)
            n_z = shape[0]
            slice_shape = (1, shape[1], shape[2])
        else:
            print(f"Unsupported ndim={ndim}, expected 3 or 4")
            sys.exit(1)

        # Estimate chunk size
        avail = get_available_ram_bytes()
        if chunk_z is None:
            chunk_z = estimate_chunk_z(
                (n_z,) + shape[z_axis + 1:] if ndim == 3 else (n_z,) + (shape[0],) + shape[z_axis + 1:],
                dtype_in, np.uint8, ram_fraction,
            )
        chunk_z = min(chunk_z, n_z)

        print(f"\nAvailable RAM:  {avail / 1e9:.1f} GB (using {ram_fraction*100:.0f}%)")
        print(f"Chunk size:     {chunk_z} z-slices")
        print(f"Total z-slices: {n_z}")
        print(f"Chunks:         {(n_z + chunk_z - 1) // chunk_z}")

        print(f"\nOutput: {output_path}")

        with h5py.File(output_path, "w") as f_out:
            # Create output dataset
            chunks = True  # let h5py pick chunk shape
            dset_out = f_out.create_dataset(
                dataset,
                shape=shape,
                dtype=np.uint8,
                chunks=chunks,
                compression=compression,
            )

            # Copy attributes
            for key, val in dset_in.attrs.items():
                dset_out.attrs[key] = val

            # Convert in chunks
            n_chunks = (n_z + chunk_z - 1) // chunk_z
            for i in range(n_chunks):
                z0 = i * chunk_z
                z1 = min(z0 + chunk_z, n_z)

                if ndim == 4:
                    chunk_in = dset_in[:, z0:z1, :, :]
                else:
                    chunk_in = dset_in[z0:z1, :, :]

                # Convert: clip to [0,1], scale to [0,255], cast to uint8
                chunk_out = np.clip(chunk_in, 0, 1)
                chunk_out = (chunk_out * 255).astype(np.uint8)

                if ndim == 4:
                    dset_out[:, z0:z1, :, :] = chunk_out
                else:
                    dset_out[z0:z1, :, :] = chunk_out

                pct = (z1 / n_z) * 100
                print(f"  [{i+1}/{n_chunks}] z={z0}:{z1}  ({pct:.0f}%)", flush=True)

            out_size = os.path.getsize(output_path)
            in_size = os.path.getsize(input_path)
            print(f"\nDone!")
            print(f"  Input:  {in_size / 1e9:.2f} GB")
            print(f"  Output: {out_size / 1e9:.2f} GB")
            print(f"  Ratio:  {in_size / max(out_size, 1):.1f}x smaller")


def main():
    parser = argparse.ArgumentParser(
        description="Convert float32 HDF5 affinity volume to uint8")
    parser.add_argument("input", help="Input HDF5 file (float32)")
    parser.add_argument("output", help="Output HDF5 file (uint8)")
    parser.add_argument("--dataset", default="main", help="HDF5 dataset name")
    parser.add_argument("--ram-fraction", type=float, default=0.3,
                        help="Fraction of available RAM to use (default: 0.3)")
    parser.add_argument("--chunk-z", type=int, default=None,
                        help="Override: z-slices per chunk (auto if not set)")
    parser.add_argument("--compression", default="gzip",
                        help="HDF5 compression (default: gzip)")
    args = parser.parse_args()

    convert(args.input, args.output,
            dataset=args.dataset,
            ram_fraction=args.ram_fraction,
            chunk_z=args.chunk_z,
            compression=args.compression)


if __name__ == "__main__":
    main()
