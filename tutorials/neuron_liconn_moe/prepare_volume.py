#!/usr/bin/env python3
"""Resample a LICONN ExPID96 (moe) volume onto the checkpoint's training grid.

WHY THIS STEP EXISTS. The banis+ checkpoint reused here was trained on the IST
LICONN final_proofread volume at 18x18x24 nm XYZ = [24, 18, 18] nm ZYX. The moe
volumes are all far finer in XY (5.08-9.03 nm), so a neurite would appear 2-3.5x
too thick to the model. moe spacings are *biological* nm (the ND2 records the
pre-expansion pixel size; the expansion factor is divided out), so matching them
to the training grid in nm really does match neurite caliber in voxels.

TWO MODES, and the choice per volume is in `volumes.py`:

`--factor Z Y X`  exact aligned block average, no interpolation. The 18x volumes
    are the only ones an integer factor from the grid:

        moe 18x native   [22.22,  9.03,  9.03] nm ZYX
        x (1, 2, 2)   -> [22.22, 18.06, 18.06] nm ZYX
        training grid    [24.00, 18.00, 18.00] nm ZYX   (Z 7.4% finer, XY 0.3%)

`--target-spacing Z Y X`  area-average resample to an arbitrary spacing, for the
    22x / 28x / 32x volumes whose factors to [24, 18, 18] are 1.32-3.54. Done as
    `cv2.INTER_AREA`, which is the exact area average of the source footprint --
    for an integer factor it reproduces the block average to within one gray
    level, and for a fractional one it is the anti-aliased generalisation. XY is
    resampled plane by plane (planes are independent) and Z afterwards on the
    XY-reduced volume, so peak memory is one uint8 copy of the reduced volume.

    The OUTPUT SPACING IS NOT EXACTLY THE TARGET. `out = round(n / f)` cannot in
    general tile the source extent, and INTER_AREA maps the *whole* source range
    onto the *whole* output range. The effective spacing `n * s / out` -- within
    ~0.1% of the target here -- is what is recorded in the attrs and what the
    neuroglancer layer must use, or the segmentation will not overlay the native
    image group.

WHY zarr -> zarr AND NOT nd2 -> zarr. `scripts/preprocess_liconn.py` applies CLAHE
per XY plane *before* block-averaging, and level 0 of the source zarr is exactly
that CLAHE'd native uint8 volume. Resampling level 0 is therefore identical to
re-running the ND2 at this grid, at minutes instead of ~1 hour, and without
re-deriving the per-plane percentile clip. The NGFF pyramid already in the source
cannot supply it either: its level 1 halves Z as well.

OUTPUT FORMAT IS CHOSEN BY EXTENSION, and `.h5` is not a convenience.
`inference/output.py::resolve_output_filenames` names the per-volume output
folder from the *last component* of the image path, which for `<vol>.zarr/0` is
`"0"` for every volume -- so a second moe volume through the same checkpoint
would overwrite the first. `<vol>.h5` gives the folder the volume's own name.

    python tutorials/neuron_liconn_moe/prepare_volume.py \
        --input  .../zarr/ExPID96_2ndgel_S1_40XW001_18x.zarr \
        --output .../zarr_ds1-2-2/ExPID96_2ndgel_S1_40XW001_18x.zarr \
        --factor 1 2 2

    python tutorials/neuron_liconn_moe/prepare_volume.py \
        --input  .../zarr/ExPID96_2ndgel_S2_40XW002_22x.zarr \
        --output .../prepared/ExPID96_2ndgel_S2_40XW002_22x.h5 \
        --target-spacing 24 18 18
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr


def _ome_multiscales(shapes, spacing_zyx, name):
    """OME-NGFF 0.4 multiscales metadata for a ZYX uint8 pyramid."""
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    return [
        {
            "version": "0.4",
            "name": name,
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": (spacing * 2**level).tolist()}
                    ],
                }
                for level in range(len(shapes))
            ],
        }
    ]


def block_average_uint8(volume: np.ndarray, factor: tuple[int, int, int]) -> np.ndarray:
    """Aligned arithmetic block average, nearest-even rounding (matches
    ``scripts/preprocess_liconn.py::volume_average_uint8``)."""
    fz, fy, fx = factor
    z, y, x = volume.shape
    if (z % fz, y % fy, x % fx) != (0, 0, 0):
        raise ValueError(f"shape {volume.shape} is not divisible by factor {factor}")
    blocks = volume.astype(np.uint32).reshape(z // fz, fz, y // fy, fy, x // fx, fx)
    return np.rint(blocks.mean(axis=(1, 3, 5), dtype=np.float64)).astype(np.uint8)


def resample_area_uint8(lvl0, out_shape, z_slab: int = 64) -> np.ndarray:
    """Area-average resample a ZYX uint8 zarr array to ``out_shape``.

    XY first, plane by plane (planes never mix), then Z on the XY-reduced volume
    in Y slabs so the 1D-along-Z resize stays inside cv2's 2D interface.
    """
    import cv2

    nz, ny, nx = lvl0.shape
    oz, oy, ox = out_shape
    xy = np.empty((nz, oy, ox), dtype=np.uint8)
    for z0 in range(0, nz, z_slab):
        z1 = min(z0 + z_slab, nz)
        src = np.asarray(lvl0[z0:z1])
        for i in range(z1 - z0):
            xy[z0 + i] = cv2.resize(src[i], (ox, oy), interpolation=cv2.INTER_AREA)
        print(f"  xy {z1}/{nz}", flush=True)
    del src

    if oz == nz:
        return xy
    out = np.empty(out_shape, dtype=np.uint8)
    y_slab = max(1, 2_000_000 // max(ox, 1))
    for y0 in range(0, oy, y_slab):
        y1 = min(y0 + y_slab, oy)
        flat = xy[:, y0:y1, :].reshape(nz, (y1 - y0) * ox)
        out[:, y0:y1, :] = cv2.resize(
            flat, ((y1 - y0) * ox, oz), interpolation=cv2.INTER_AREA
        ).reshape(oz, y1 - y0, ox)
    print(f"  z {nz} -> {oz}", flush=True)
    return out


def _write_h5(path: Path, vol: np.ndarray, attrs: dict) -> None:
    import h5py
    import os

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    tmp = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp, "w") as f:
        d = f.create_dataset("main", data=vol, chunks=(min(64, vol.shape[0]), 128, 128),
                             compression="gzip", compression_opts=1)
        for k, v in attrs.items():
            d.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    tmp.replace(path)


def _write_zarr_pyramid(path: Path, a0_src, out_shape, factor, spacing, levels, chunk, z_slab, attrs):
    tmp = path.with_name(f".{path.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    grp = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    chunks = tuple(min(chunk, s) for s in out_shape)
    a0 = grp.create_array("0", shape=out_shape, chunks=chunks, dtype="uint8")

    fz, fy, fx = factor
    nz = a0_src.shape[0]
    slab = z_slab - (z_slab % fz)
    for z0 in range(0, nz - nz % fz, slab):
        z1 = min(z0 + slab, out_shape[0] * fz)
        a0[z0 // fz : z1 // fz] = block_average_uint8(
            np.asarray(a0_src[z0:z1, : out_shape[1] * factor[1], : out_shape[2] * factor[2]]), factor)
        print(f"  source planes {z1}/{nz}", flush=True)

    shapes = [out_shape]
    prev = np.asarray(a0[:])
    for level in range(1, levels + 1):
        crop = prev[: prev.shape[0] // 2 * 2, : prev.shape[1] // 2 * 2, : prev.shape[2] // 2 * 2]
        prev = block_average_uint8(crop, (2, 2, 2))
        grp.create_array(
            str(level), shape=prev.shape, chunks=tuple(min(chunk, s) for s in prev.shape),
            dtype="uint8",
        )[:] = prev
        shapes.append(prev.shape)
        print(f"  level {level}: {prev.shape}", flush=True)

    attrs["multiscales"] = _ome_multiscales(shapes, spacing, path.stem)
    grp.attrs.update(attrs)
    if path.is_dir():
        shutil.rmtree(path)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Source OME-NGFF .zarr group")
    p.add_argument("--output", required=True, type=Path, help="Destination .zarr group or .h5 file")
    p.add_argument("--factor", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                   help="Integer block-average factor (exact, no interpolation).")
    p.add_argument("--target-spacing", type=float, nargs=3, default=None, metavar=("Z", "Y", "X"),
                   help="Area-resample to this ZYX nm spacing instead.")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--levels", type=int, default=4, help="Zarr pyramid levels below the new level 0")
    p.add_argument("--z-slab", type=int, default=64, help="Source Z planes held in memory at once")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if (args.factor is None) == (args.target_spacing is None):
        raise SystemExit("give exactly one of --factor / --target-spacing")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists (pass --overwrite)")

    src = zarr.open_group(str(args.input), mode="r")
    lvl0 = src["0"]
    src_attrs = dict(src.attrs)
    native = np.asarray(src_attrs["spacing_nm_zyx"], dtype=np.float64)

    attrs = dict(src_attrs)
    attrs["source"] = str(args.input)

    if args.factor is not None:
        factor = tuple(args.factor)
        out_shape = tuple(s // f for s, f in zip(lvl0.shape, factor))
        spacing = native * np.asarray(factor, dtype=np.float64)
        attrs["downsample_factor_zyx"] = list(factor)
        attrs["downsample_method"] = "aligned_arithmetic_volume_average"
        attrs["downsample_rounding"] = "nearest_even"
    else:
        target = np.asarray(args.target_spacing, dtype=np.float64)
        out_shape = tuple(
            max(1, int(round(n / (t / s)))) for n, t, s in zip(lvl0.shape, target, native)
        )
        # Effective, not nominal: INTER_AREA stretches the full source extent
        # over the full output extent, so this is the spacing that keeps the
        # physical corners of source and output coincident.
        spacing = native * np.asarray(lvl0.shape, dtype=np.float64) / np.asarray(out_shape)
        attrs["target_spacing_nm_zyx"] = target.tolist()
        attrs["downsample_method"] = "cv2_INTER_AREA_xy_then_z"
        # Inherited from the source's own (1,1,1) provenance; meaningless here.
        for stale in ("downsample_factor_zyx", "downsample_rounding"):
            attrs.pop(stale, None)
    attrs["spacing_nm_zyx"] = spacing.tolist()

    print(f"{lvl0.shape} {np.round(native, 4).tolist()} nm  ->  {out_shape} "
          f"{np.round(spacing, 4).tolist()} nm", flush=True)

    if args.output.suffix == ".h5":
        if args.factor is not None:
            vol = np.empty(out_shape, dtype=np.uint8)
            fz = args.factor[0]
            slab = args.z_slab - (args.z_slab % fz)
            for z0 in range(0, lvl0.shape[0] - lvl0.shape[0] % fz, slab):
                z1 = min(z0 + slab, out_shape[0] * fz)
                # Drop a trailing partial block in XY, as Z already does: an
                # odd-sized crop (e.g. the z-aligned 28x, 2267x2269) at factor 2.
                vol[z0 // fz : z1 // fz] = block_average_uint8(
                    np.asarray(lvl0[z0:z1, : out_shape[1] * args.factor[1],
                                    : out_shape[2] * args.factor[2]]), tuple(args.factor))
                print(f"  source planes {z1}/{lvl0.shape[0]}", flush=True)
        else:
            vol = resample_area_uint8(lvl0, out_shape, z_slab=args.z_slab)
        _write_h5(args.output, vol, attrs)
    else:
        if args.factor is None:
            raise SystemExit("--target-spacing currently writes .h5 only")
        _write_zarr_pyramid(args.output, lvl0, out_shape, tuple(args.factor), spacing,
                            args.levels, args.chunk_size, args.z_slab, attrs)

    print(f"wrote {args.output}")
    print(json.dumps({k: v for k, v in attrs.items() if k != "multiscales"}, indent=2))


if __name__ == "__main__":
    main()
