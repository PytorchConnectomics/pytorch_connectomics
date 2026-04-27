#!/usr/bin/env python3
"""
Downsample 3D volumes stored in any format supported by
``connectomics.data.io.read_volume`` (HDF5, TIFF, PNG stack, NIfTI, Zarr).
Outputs are always written as HDF5.

Example:
  python scripts/downsample_data.py \
      /projects/weilab/dataset/MitoLE/betaSeg/high_c1_im.h5 \
      /projects/weilab/dataset/MitoLE/betaSeg/high_c1_mito.h5 \
      --downsample-ratio-zyx 2 2 2

  python scripts/downsample_data.py my_volume.tif --downsample-ratio-zyx 1 4 4

By default each input `<name><ext>` is written to `<name>_d{z}x{y}x{x}.h5`
next to it. Output dataset key defaults to `main`.

Mode handling (image vs label):
  - `auto` (default): integer dtypes other than uint8 are treated as labels
    (strided sampling); everything else is treated as image (zoom).
  - `image`: ndimage.zoom on each z-slice (reuses zoom_downsample_xy from
    downsample_nisb) + strided z sampling.
  - `label`: strided sampling in (z, y, x).

Volumes are assumed to be stored in (Z, Y, X) layout, which is the standard
connectomics EM convention and matches what ``read_volume`` returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from connectomics.data.io import read_volume, write_hdf5

from downsample_nisb import zoom_downsample_xy  # reuse YX zoom helper


def _len_from_step(size: int, step: int) -> int:
    return (size + step - 1) // step


def _detect_mode(volume: np.ndarray) -> str:
    return "label" if np.issubdtype(volume.dtype, np.integer) and volume.dtype != np.uint8 else "image"


def downsample_volume_zyx(
    volume: np.ndarray,
    downsample_ratio_zyx: tuple[int, int, int],
    mode: str,
    img_zoom_order: int = 1,
) -> np.ndarray:
    """Downsample a 3D (Z, Y, X) volume.

    Args:
        volume: numpy array with shape (Z, Y, X).
        downsample_ratio_zyx: integer strides in (z, y, x).
        mode: "image" (ndimage.zoom in yx + strided z) or "label" (strided zyx).
        img_zoom_order: interpolation order for image mode.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got ndim={volume.ndim}")
    z_factor, y_factor, x_factor = downsample_ratio_zyx
    if min(z_factor, y_factor, x_factor) <= 0:
        raise ValueError(f"All downsample ratios must be positive, got {downsample_ratio_zyx}")

    z_in, y_in, x_in = volume.shape
    z_out = _len_from_step(z_in, z_factor)
    y_out = _len_from_step(y_in, y_factor)
    x_out = _len_from_step(x_in, x_factor)

    if mode == "label":
        # Strided nearest-neighbour sampling preserves label ids.
        return np.ascontiguousarray(volume[::z_factor, ::y_factor, ::x_factor])

    if mode != "image":
        raise ValueError(f"Unknown mode: {mode!r} (expected 'image' or 'label')")

    # Image mode: zoom in YX per sampled z-slice.
    out = np.empty((z_out, y_out, x_out), dtype=volume.dtype)
    for out_z, in_z in enumerate(range(0, z_in, z_factor)):
        yx = volume[in_z]  # (Y, X)
        # zoom_downsample_xy treats its args as a generic 2D plane; the
        # (y, x) layout maps directly to (out_shape[0], out_shape[1]).
        out[out_z] = zoom_downsample_xy(
            yx,
            out_shape_xy=(y_out, x_out),
            order=img_zoom_order,
        )
        if out_z % 50 == 0 or out_z == z_out - 1:
            print(f"  z-slices: {out_z + 1}/{z_out}")
    return out


def downsample_file(
    input_path: Path,
    *,
    downsample_ratio_zyx: tuple[int, int, int],
    mode: str,
    output_path: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    dataset_in: Optional[str] = None,
    dataset_out: str = "main",
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
    img_zoom_order: int = 1,
) -> Path:
    """Downsample one volume file and write an HDF5 output."""
    if output_path is None:
        z, y, x = downsample_ratio_zyx
        out_name = f"{input_path.stem}_d{z}x{y}x{x}.h5"
        if output_folder is not None:
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / out_name
        else:
            output_path = input_path.with_name(out_name)

    if output_path.exists() and not overwrite:
        print(f"Skip (exists): {output_path}")
        return output_path

    print(f"Reading {input_path}" + (f" [{dataset_in}]" if dataset_in else ""))
    # read_volume dispatches by extension: h5/tiff/png/nifti/zarr. The
    # ``dataset`` arg is only consumed by the h5 backend; other backends
    # ignore it.
    volume = read_volume(str(input_path), dataset=dataset_in)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume from {input_path}, got shape {volume.shape}. "
            "4D (multi-channel) inputs are not supported."
        )

    resolved_mode = _detect_mode(volume) if mode == "auto" else mode
    print(
        f"  Input shape (z,y,x): {tuple(volume.shape)}, dtype={volume.dtype}, "
        f"mode={resolved_mode}, ratio={downsample_ratio_zyx}"
    )

    out_volume = downsample_volume_zyx(
        volume,
        downsample_ratio_zyx=downsample_ratio_zyx,
        mode=resolved_mode,
        img_zoom_order=img_zoom_order,
    )

    print(f"  Output shape (z,y,x): {tuple(out_volume.shape)}, dtype={out_volume.dtype}")
    write_hdf5(
        str(output_path),
        out_volume,
        dataset=dataset_out,
        compression=compression,
        compression_level=compression_level,
    )
    print(f"  Wrote: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample 3D volumes in HDF5 files by strided/zoom sampling.",
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input volume file paths (h5, tiff, png, nifti, or zarr).",
    )
    parser.add_argument(
        "--downsample-ratio-zyx",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=[2, 2, 2],
        help="Downsample ratio in zyx order. Default: 2 2 2.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "image", "label"],
        help="Downsampling mode (auto picks label for integer dtype, image otherwise).",
    )
    parser.add_argument(
        "--dataset-in",
        type=str,
        default=None,
        help="Input dataset key inside each .h5 (ignored for non-h5 inputs; "
        "defaults to the first dataset).",
    )
    parser.add_argument(
        "--dataset-out",
        type=str,
        default="main",
        help="Output dataset key in each written .h5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        nargs="+",
        default=None,
        help="Optional explicit output paths, one per input. If omitted, "
        "the output filename is '<stem>_d{z}x{y}x{x}.h5' (placed next to the "
        "input, or in --output-folder if given).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Directory to write outputs into (created if missing). "
        "Ignored when --output is given.",
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
        help="Gzip compression level (ignored for lzf/none).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--img-zoom-order",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="Interpolation order for ndimage.zoom in image mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression

    output_paths = args.output
    if output_paths is not None and len(output_paths) != len(args.inputs):
        raise ValueError(
            f"--output expects one path per input ({len(args.inputs)}), got {len(output_paths)}"
        )
    if output_paths is not None and args.output_folder is not None:
        print("Warning: --output-folder is ignored because --output was provided.")

    ratio = tuple(args.downsample_ratio_zyx)
    for idx, input_path in enumerate(args.inputs):
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        out_path = output_paths[idx] if output_paths is not None else None
        downsample_file(
            input_path,
            downsample_ratio_zyx=ratio,
            mode=args.mode,
            output_path=out_path,
            output_folder=args.output_folder,
            dataset_in=args.dataset_in,
            dataset_out=args.dataset_out,
            compression=compression,
            compression_level=args.compression_level,
            overwrite=args.overwrite,
            img_zoom_order=args.img_zoom_order,
        )


# Examples:
# python scripts/downsample_data.py \
#     /projects/weilab/dataset/MitoLE/betaSeg/high_c1_im.h5 \
#     /projects/weilab/dataset/MitoLE/betaSeg/high_c1_mito.h5 \
#     --downsample-ratio-zyx 2 2 2
#
# python scripts/downsample_data.py vol.h5 --mode label --downsample-ratio-zyx 1 4 4
if __name__ == "__main__":
    main()
