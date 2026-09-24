#!/usr/bin/env python3
"""Write the published LICONN image group: a uint8 OME-Zarr pyramid, zarr v2.

THIS FILE'S JOB IS TO NOT INVENT A LAYOUT. Twelve image groups already sit
under `gs://donglai_public/liconn/moe/<expid>/image/`, and a viewer state, a
`volumes.py` entry and a segmentation run all key off their exact shape. The
constants below were read back off the published
`ExPID96_2ndgel_S1_40XW001_18x.zarr` rather than chosen:

    .zgroup              {"zarr_format": 2}
    <level>/.zarray      chunks [128,128,128], dtype |u1, fill_value 0,
                         order C, filters null, dimension_separator ".",
                         compressor blosc/zstd/clevel 5/shuffle 1
    .zattrs              OME-NGFF 0.4 multiscales, levels 0..4, ZYX nanometer,
                         scale = spacing * 2**level, plus the provenance keys
                         `scripts/preprocess_liconn.py` writes on its HDF5.

`test_zarr_writer.py` asserts this against a recorded copy of that group's
metadata, so a dependency bump that changes a default fails the build rather
than the next volume.

STREAMING, NOT BECAUSE IT IS ELEGANT. A 1199x2304x2304 uint16 ND2 is 12.7 GB
and the machine driving this has ~24 GB free. Level 0 is written one z-chunk
row at a time and every finer-to-coarser reduction reads the level below off
disk, so peak memory is a couple of chunk rows and does not scale with Z.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# NOTE: numpy is imported INSIDE the functions that need it. `ome_multiscales`,
# `provenance_attrs` and `read_group_metadata` are pure JSON arithmetic, and
# `refold` in ingest.py calls them from the HOST, which deliberately has no
# numpy -- supplying it is the image's job. Keeping the import lazy is what
# lets a fold correction be a metadata rewrite instead of a container round
# trip and a re-conversion.

# Read back off the published group. Changing any of these forks the bucket.
CHUNK = 128
LEVELS_BELOW_ZERO = 4
COMPRESSOR = {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1, "blocksize": 0}


def _compressor():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0)


def ome_multiscales(shapes, spacing_zyx, name):
    """OME-NGFF 0.4 multiscales for a ZYX uint8 pyramid.

    Byte-identical in form to `prepare_volume.py::_ome_multiscales`, which
    wrote the published groups -- except that `name` is the clean group name.
    The published `name` fields read `.<group>.zarr.<pid>` because that writer
    took `path.stem` of its own `.tmp` staging path; that is a cosmetic bug and
    is not reproduced.
    """
    spacing = [float(v) for v in spacing_zyx]
    return [
        {
            "version": "0.4",
            "name": name,
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [v * 2**level for v in spacing]}
                    ],
                }
                for level in range(len(shapes))
            ],
        }
    ]


def block_average_uint8(volume: np.ndarray, factor=(2, 2, 2)) -> np.ndarray:
    """Aligned arithmetic block average, nearest-even rounding.

    Same contract as `prepare_volume.py::block_average_uint8` and
    `preprocess_liconn.py::volume_average_uint8`: the accumulate is exact in
    uint32, the mean is float64, and `np.rint` rounds halves to even. Anything
    else drifts one gray level per level against the published pyramid.
    """
    import numpy as np

    fz, fy, fx = factor
    z, y, x = volume.shape
    if (z % fz, y % fy, x % fx) != (0, 0, 0):
        raise ValueError(f"shape {volume.shape} is not divisible by factor {factor}")
    blocks = volume.astype(np.uint32).reshape(z // fz, fz, y // fy, fy, x // fx, fx)
    return np.rint(blocks.mean(axis=(1, 3, 5), dtype=np.float64)).astype(np.uint8)


def _create(grp, level: int, shape):
    import zarr

    return grp.create_array(
        str(level),
        shape=tuple(shape),
        chunks=tuple(min(CHUNK, s) for s in shape),
        dtype="uint8",
        compressors=[_compressor()],
        fill_value=0,
    )


def _reduce_level(grp, src_level: int, dst_level: int, z_slab: int = CHUNK) -> tuple:
    """Write `dst_level` as the (2,2,2) block average of `src_level`, streaming.

    The odd trailing plane/row/column of the source is dropped, matching the
    published pyramid: `out = in // 2` exactly, never a padded edge.
    """
    import numpy as np

    src = grp[str(src_level)]
    sz, sy, sx = src.shape
    out_shape = (sz // 2, sy // 2, sx // 2)
    if min(out_shape) == 0:
        return None
    dst = _create(grp, dst_level, out_shape)
    slab = max(2, z_slab - (z_slab % 2))
    for z0 in range(0, out_shape[0] * 2, slab):
        z1 = min(z0 + slab, out_shape[0] * 2)
        block = np.asarray(src[z0:z1, : sy // 2 * 2, : sx // 2 * 2])
        dst[z0 // 2 : z1 // 2] = block_average_uint8(block)
    return out_shape


def write_pyramid(dest: Path, plane_iter, shape_zyx, spacing_zyx, attrs: dict,
                  levels: int = LEVELS_BELOW_ZERO, z_slab: int = CHUNK) -> dict:
    """Stream `plane_iter` (uint8 YX planes, Z-ordered) into a zarr v2 pyramid.

    Staged at `.<name>.tmp` and renamed, so an interrupted run never leaves a
    half-written group that `publish` would happily upload.
    """
    import numpy as np
    import zarr

    dest = Path(dest)
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    grp = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    a0 = _create(grp, 0, shape_zyx)

    nz, ny, nx = shape_zyx
    buf = np.empty((z_slab, ny, nx), dtype=np.uint8)
    filled = 0
    written = 0
    for plane in plane_iter:
        if plane.shape != (ny, nx):
            raise ValueError(f"plane {written + filled} is {plane.shape}, expected {(ny, nx)}")
        buf[filled] = plane
        filled += 1
        if filled == z_slab:
            a0[written : written + filled] = buf
            written += filled
            filled = 0
            print(f"  level 0: {written}/{nz} planes", flush=True)
    if filled:
        a0[written : written + filled] = buf[:filled]
        written += filled
    if written != nz:
        raise ValueError(f"plane iterator yielded {written} planes, expected {nz}")

    shapes = [tuple(shape_zyx)]
    for level in range(1, levels + 1):
        out_shape = _reduce_level(grp, level - 1, level, z_slab=z_slab)
        if out_shape is None:
            break
        shapes.append(out_shape)
        print(f"  level {level}: {out_shape}", flush=True)

    attrs = dict(attrs)
    attrs["multiscales"] = ome_multiscales(shapes, spacing_zyx, dest.name)
    grp.attrs.update(attrs)

    if dest.is_dir():
        shutil.rmtree(dest)
    tmp.replace(dest)
    return {"shapes": [list(s) for s in shapes], "levels": len(shapes)}


def provenance_attrs(*, source: str, source_channel: int, source_shape_zyx,
                     spacing_nm_zyx, clip_mode: str, clip_values, clip_limit: float,
                     spacing_basis: str = "biological") -> dict:
    """The non-OME half of `.zattrs`, keyed exactly as preprocess_liconn writes it.

    SPACING BASIS. Every published moe group records *biological* spacing:
    optics spacing divided by the expansion fold. The ExPID71 drop has no fold
    anywhere -- not in the filename, not in the ND2 (searched 2026-09-21: zero
    hits for expan/gel/fold/swell across custom_data and text_info) -- so those
    groups are published with raw optics spacing instead, on an explicit
    decision. When that happens the two keys below are ADDED, and only then, so
    that the anomalous groups are the ones that look anomalous and the normal
    ones keep the published key set exactly. A reader that ignores
    `spacing_basis` and overlays these on an expid96/99/107/108 layer will be
    wrong by the expansion fold, which is why the warning is in the metadata and
    not only in a log.
    """
    attrs = {
        "axes": "ZYX",
        "source": source,
        "source_channel": int(source_channel),
        "source_shape_zyx": [int(v) for v in source_shape_zyx],
        "downsample_factor_zyx": [1, 1, 1],
        "downsample_method": "aligned_arithmetic_volume_average",
        "downsample_rounding": "nearest_even",
        "input_spacing_nm_zyx": [float(v) for v in spacing_nm_zyx],
        "spacing_nm_zyx": [float(v) for v in spacing_nm_zyx],
        "clahe_clip_limit": float(clip_limit),
    }
    if clip_mode == "percentile":
        attrs["clip_mode"] = "percentile"
        attrs["clip_percentiles"] = [float(v) for v in clip_values]
    elif clip_mode == "fixed_intensity":
        attrs["clip_mode"] = "fixed_intensity"
        attrs["clip_intensity_range"] = [float(v) for v in clip_values]
    else:
        raise ValueError(f"unknown clip_mode {clip_mode!r}")

    if spacing_basis == "biological_assumed_fold":
        attrs["spacing_basis"] = "biological_assumed_fold"
        attrs["spacing_warning"] = (
            "The expansion fold used to derive spacing_nm_zyx is ASSUMED, not recorded. "
            "This specimen's fold appears in neither its filename nor its ND2 metadata, "
            "so the spacing here is only as good as that assumption. It is on the same "
            "basis as the other published groups (biological nm) and so is directly "
            "comparable with them, but confirm the fold with the submitter before "
            "treating any quantitative result from it as settled."
        )
    elif spacing_basis == "optics_uncorrected":
        attrs["spacing_basis"] = "optics_uncorrected"
        attrs["spacing_warning"] = (
            "Expansion fold UNKNOWN for this specimen. spacing_nm_zyx is raw optics "
            "spacing, NOT divided by the expansion fold, so it is not biological "
            "spacing and this volume is NOT spatially comparable with the expid96 / "
            "expid99 / expid107 / expid108 groups. Divide by the fold once it is "
            "known before any cross-volume comparison or model input."
        )
    elif spacing_basis != "biological":
        raise ValueError(f"unknown spacing_basis {spacing_basis!r}")
    return attrs


def read_group_metadata(root: Path) -> dict:
    """Group metadata as plain JSON, for tests and for `verify`."""
    root = Path(root)
    out = {"zgroup": json.loads((root / ".zgroup").read_text()),
           "zattrs": json.loads((root / ".zattrs").read_text()), "levels": {}}
    for level in sorted(p.name for p in root.iterdir() if p.is_dir()):
        out["levels"][level] = json.loads((root / level / ".zarray").read_text())
    return out
