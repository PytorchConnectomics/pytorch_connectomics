"""Keep-mask construction shared by volume playbooks.

The FFN tissue/border routines and CLI below are relocated verbatim from j0126.
Downsampled sources combine configured keep/exclude masks in a shared volume frame.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter

# fmt: off
# FFN's J0126 cloud-volume mirror. tissue_classification is 6-channel uint8 prob:
# 0=unused, 1=blood_vessel, 2=cell_body, 3=myelin, 4=neuropil, 5=out-of-bounds.
FFN_TISSUE = (
    "https://storage.googleapis.com/j0126-nature-methods-data/"
    "GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/tissue_classification"
)
EXCLUDE_CHANNELS = (1, 3, 5)
EXCLUDE_THRESHOLDS = {1: 252, 3: 252, 5: 25}
TISSUE_SHAPE_ZYX = (5700, 5456, 5332)   # native 18 x 18 x 20 nm
# True mip-0 EM extent, ZYX. Same origin as the tissue layer: Z is 1:1, Y and X
# are half-resolution there, so the tissue mask upsamples 2x in Y/X only.
KEEP_SHAPE_ZYX = (5700, 10912, 10664)
KEEP_CHUNKS = (126, 504, 504)
CELL = 1008                             # matches the affinity chunk grid
assert all(CELL % c == 0 for c in KEEP_CHUNKS)


def _create(out: Path, shape, chunks) -> None:
    """Create a zstd-compressed uint8 mask array under zarr 2 or zarr 3.

    zarr >= 3 rejects the zarr-2 `compressor=` kwarg outright
    (`ValueError: compressor cannot be used for arrays with zarr_format 3`),
    which aborted `--init` -- and therefore the whole driver -- before a single
    job was submitted.
    """
    try:
        zarr.create_array(
            store=str(out), shape=shape, chunks=chunks, dtype="uint8",
            fill_value=1, compressors=zarr.codecs.ZstdCodec(level=5),
        )
    except AttributeError:  # zarr 2
        zarr.open(
            str(out), mode="w", shape=shape, chunks=chunks, dtype="uint8",
            fill_value=1,
            compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        )



def _sentinel(out: Path, shard_id: int) -> Path:
    return Path(f"{out}.done.{shard_id}")


def _border_pad_slice(sl: np.ndarray, offset: int) -> np.ndarray:
    """2D padding mask for one z-section of aligned EM.

    The aligned volume has pure 0 (black) / 255 (white) padding at the faces that
    the FFN tissue mask does not fully cover. Mark {0,255} plus the anti-aliased
    0<->255 seam, dilate by `offset`, then keep only the straight run that reaches
    an image edge -- interior black is real tissue, not padding.
    """
    b = (sl == 0) | (sl == 255)
    b |= (minimum_filter(sl, 3) == 0) & (maximum_filter(sl, 3) == 255)
    if offset:
        b = binary_dilation(b, iterations=offset)
    nb = ~b
    return (
        (np.cumsum(nb, 1) == 0)
        | (np.cumsum(nb[:, ::-1], 1) == 0)[:, ::-1]
        | (np.cumsum(nb, 0) == 0)
        | (np.cumsum(nb[::-1], 0) == 0)[::-1]
    )


def build_keep_cell(start, shape, index_zyx, tissue_ds, em_ds, *, border_offset=1):
    """Per-cell boolean keep-mask (True = keep) = FFN tissue AND 0/255 border."""
    z0, y0, x0 = start
    depth, height, width = shape
    keep = np.ones((depth, height, width), bool)

    # Tissue: Z 1:1, XY half-res -> nearest 2x upsample; beyond bounds -> keep.
    tz, ty, tx = tissue_ds.shape
    y0h, x0h = y0 // 2, x0 // 2
    z1c, y1c, x1c = (
        min(z0 + depth, tz),
        min(y0h + (height + 1) // 2, ty),
        min(x0h + (width + 1) // 2, tx),
    )
    native = np.ones((depth, (height + 1) // 2, (width + 1) // 2), np.uint8)
    if z1c > z0 and y1c > y0h and x1c > x0h:
        sub = np.asarray(tissue_ds[z0:z1c, y0h:y1c, x0h:x1c]).astype(np.uint8)
        native[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
    up = np.repeat(np.repeat(native, 2, axis=1), 2, axis=2)[:depth, :height, :width]
    if up.shape != (depth, height, width):
        up = np.pad(
            up,
            [(0, (depth, height, width)[k] - up.shape[k]) for k in range(3)],
            mode="edge",
        )
    keep &= up > 0

    # Border ring only: the interior is full tissue, and reading EM for it is the
    # expensive half of this stage.
    ymax = (KEEP_SHAPE_ZYX[1] - 1) // CELL
    xmax = (KEEP_SHAPE_ZYX[2] - 1) // CELL
    if index_zyx[1] in (0, ymax) or index_zyx[2] in (0, xmax):
        em = np.asarray(em_ds[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width])
        if em.shape != (depth, height, width):
            pad = [(0, (depth, height, width)[k] - em.shape[k]) for k in range(3)]
            em = np.pad(em, pad)          # off-volume reads as 0 == padding == masked
        for zi in range(depth):
            keep[zi] &= ~_border_pad_slice(em[zi], border_offset)
    return keep


def run_tissue(args) -> None:
    import cloudvolume as cv

    out = zarr.open(str(args.out), mode="r+")
    vol = cv.CloudVolume(FFN_TISSUE, mip=0, bounded=True, progress=False, fill_missing=True)
    got = tuple(int(s) for s in vol.shape[:3])
    want = (TISSUE_SHAPE_ZYX[2], TISSUE_SHAPE_ZYX[1], TISSUE_SHAPE_ZYX[0])
    if got != want or int(vol.num_channels) != 6:
        raise SystemExit(f"unexpected tissue_classification geometry: {vol.shape}")

    slabs = [
        (z, min(z + args.z_slab, TISSUE_SHAPE_ZYX[0]))
        for z in range(0, TISSUE_SHAPE_ZYX[0], args.z_slab)
    ]
    mine = slabs[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(slabs)} slabs", flush=True)
    t0 = time.time()
    for i, (z0, z1) in enumerate(mine, 1):
        # cloud-volume is XYZC; channel selection has to be a post-fetch numpy index.
        arr = vol[:, :, z0:z1]
        excluded = np.zeros(arr.shape[:-1], dtype=bool)
        for channel in EXCLUDE_CHANNELS:
            cutoff = EXCLUDE_THRESHOLDS[channel] if args.threshold is None else args.threshold
            excluded |= arr[..., channel] >= cutoff
        out[z0:z1, :, :] = np.transpose((~excluded).astype(np.uint8), (2, 1, 0))
        print(f"  [{i}/{len(mine)}] z={z0}:{z1} ({(time.time()-t0)/i:.1f}s/slab)", flush=True)


def run_keep(args) -> None:
    out = zarr.open(str(args.out), mode="r+")
    tissue = zarr.open(str(args.tissue), mode="r")
    em = zarr.open(str(args.em), mode="r")

    cells = [
        (cz, cy, cx)
        for cz in range(0, -(-KEEP_SHAPE_ZYX[0] // CELL))
        for cy in range(0, -(-KEEP_SHAPE_ZYX[1] // CELL))
        for cx in range(0, -(-KEEP_SHAPE_ZYX[2] // CELL))
    ]
    mine = cells[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(cells)} cells", flush=True)
    t0 = time.time()
    for i, (cz, cy, cx) in enumerate(mine, 1):
        gz, gy, gx = cz * CELL, cy * CELL, cx * CELL
        ez = min(gz + CELL, KEEP_SHAPE_ZYX[0])
        ey = min(gy + CELL, KEEP_SHAPE_ZYX[1])
        ex = min(gx + CELL, KEEP_SHAPE_ZYX[2])
        # Always compute the FULL cell and clip on write, so a face cell gets the
        # same border decision it would get in the interior of a padded volume.
        keep = build_keep_cell(
            (gz, gy, gx), (ez - gz, CELL, CELL), (cz, cy, cx),
            tissue, em, border_offset=args.border_offset,
        )
        out[gz:ez, gy:ey, gx:ex] = keep[:, : ey - gy, : ex - gx].astype(np.uint8)
        print(
            f"  [{i}/{len(mine)}] z{cz}_y{cy}_x{cx} keep={keep.mean():.4f} "
            f"({(time.time()-t0)/i:.1f}s/cell)",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=("tissue", "keep"), required=True)
    ap.add_argument("--out", type=Path, required=True, help="output zarr")
    ap.add_argument("--tissue", type=Path, help="tissue zarr (--stage keep)")
    ap.add_argument("--em", type=Path, help="mip-0 EM zarr array (--stage keep)")
    ap.add_argument("--init", action="store_true", help="create the zarr and exit")
    ap.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="override every channel with one cutoff; default uses FFN's published "
             "per-channel thresholds (blood vessel 252, myelin 252, out-of-bounds 25)",
    )
    ap.add_argument("--z-slab", type=int, default=128, help="Z per cloud-volume read")
    ap.add_argument("--border-offset", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.init:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            print(f"{args.out} already exists, leaving it alone")
            return 0
        # fill_value 1 = KEEP, so an unwritten region never silently deletes data.
        if args.stage == "tissue":
            _create(args.out, TISSUE_SHAPE_ZYX, (args.z_slab, 512, 512))
        else:
            _create(args.out, KEEP_SHAPE_ZYX, KEEP_CHUNKS)
        print(f"created {args.out}")
        return 0

    if not args.out.exists():
        raise SystemExit(f"{args.out} does not exist; run --init first")
    if args.stage == "keep":
        if args.tissue is None or args.em is None:
            raise SystemExit("--stage keep needs --tissue and --em")
        for path in (args.tissue, args.em):
            if not path.exists():
                raise SystemExit(f"missing input: {path}")
        run_keep(args)
    else:
        run_tissue(args)

    _sentinel(args.out, args.shard_id).write_text("done\n")
    print(f"shard {args.shard_id} done -> {_sentinel(args.out, args.shard_id)}", flush=True)
    return 0


# fmt: on


@dataclass(frozen=True)
class MaskSource:
    path: Path
    dataset: str = "main"
    polarity: str = "exclude"


@dataclass(frozen=True)
class KeepMaskSpec:
    """Mask inputs in the volume's ZYX coordinate frame."""

    strategy: str
    out: Path
    volume_shape_zyx: tuple[int, int, int]
    ratio_zyx: tuple[int, int, int]
    border_start_zyx: tuple[int, int, int]
    sources: list[MaskSource | Mapping[str, Any]] = field(default_factory=list)
    global_offset_zyx: tuple[int, int, int] = (0, 0, 0)
    tissue: Path | None = None
    em: Path | None = None
    threshold: int = 128
    z_slab: int = 128
    border_offset: int = 1


def spec_from_params(params: Mapping[str, Any]) -> KeepMaskSpec:
    """Resolve the shared mask declaration from a tutorial's resolved params."""
    data = params["data"]
    masks = data["masks"]
    frame = params["frame"]
    return KeepMaskSpec(
        strategy=masks["strategy"],
        out=Path(data["keep_mask"]),
        volume_shape_zyx=tuple(frame["volume_shape_zyx"]),
        ratio_zyx=tuple(masks["ratio_zyx"]),
        border_start_zyx=tuple(masks["border_start_zyx"]),
        sources=[
            MaskSource(Path(source["path"]), source["dataset"], source["polarity"])
            for source in masks["sources"]
        ],
        global_offset_zyx=tuple(frame["volume_origin_global_zyx"]),
        tissue=Path(data["tissue_mask"]) if data.get("tissue_mask") else None,
        em=Path(data["raw_em"]) if data.get("raw_em") else None,
        threshold=masks.get("threshold", 128),
        z_slab=masks.get("z_slab", 128),
        border_offset=masks.get("border_offset", 1),
    )


def _downsampled_sources(spec: KeepMaskSpec) -> None:
    for name, values, positive in (
        ("volume_shape_zyx", spec.volume_shape_zyx, True),
        ("ratio_zyx", spec.ratio_zyx, True),
        ("border_start_zyx", spec.border_start_zyx, False),
    ):
        if len(values) != 3 or any(v < (1 if positive else 0) for v in values):
            raise ValueError(
                f"{name} must contain three {'positive' if positive else 'nonnegative'} integers"
            )
    shape = tuple(-(-spec.volume_shape_zyx[i] // spec.ratio_zyx[i]) for i in range(3))
    keep: np.ndarray = np.ones(shape, dtype=bool)
    sources = [
        source if isinstance(source, MaskSource) else MaskSource(**source)
        for source in spec.sources
    ]
    for source in sources:
        if source.polarity not in ("keep", "exclude"):
            raise ValueError(f"unknown mask polarity: {source.polarity!r}")
        with h5py.File(source.path, "r") as f:
            src = f[source.dataset]
            if src.ndim != 3 or any(src.shape[i] < shape[i] for i in range(3)):
                raise ValueError(f"mask {source.path} shape {src.shape} does not cover {shape}")
            active = np.asarray(src[: shape[0], : shape[1], : shape[2]]) != 0
        keep &= active if source.polarity == "keep" else ~active
        print(f"{source.polarity} source {source.path}: {float(active.mean()):.4%} of cells")
    border: np.ndarray = np.zeros(shape, dtype=bool)
    cut = [int(math.ceil(spec.border_start_zyx[i] / spec.ratio_zyx[i])) for i in range(3)]
    border[: cut[0], :, :] = True
    border[:, : cut[1], :] = True
    border[:, :, : cut[2]] = True
    keep &= ~border

    spec.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec.out, "w") as f:
        d = f.create_dataset(
            "main",
            data=keep.astype(np.uint8),
            chunks=tuple(min(size, chunk) for size, chunk in zip(shape, (32, 128, 128))),
            compression="gzip",
        )
        d.attrs["axis_order"] = "ZYX"
        d.attrs["downsample_factors_zyx"] = np.array(spec.ratio_zyx)
        d.attrs["global_offset_zyx"] = np.array(spec.global_offset_zyx)
        d.attrs["volume_shape_zyx_fullres"] = np.array(spec.volume_shape_zyx)
        d.attrs["tissue_start_zyx_fullres"] = np.array(spec.border_start_zyx)
        if len(sources) == 1 and sources[0].polarity == "exclude":
            d.attrs["semantics"] = "1 = keep (affinity used), 0 = drop (blood vessel or border)"
            d.attrs["source_blood_vessel"] = str(sources[0].path)
        else:
            d.attrs["semantics"] = "1 = keep (affinity used), 0 = drop (mask sources or border)"
            d.attrs["sources"] = json.dumps(
                [
                    {"path": str(s.path), "dataset": s.dataset, "polarity": s.polarity}
                    for s in sources
                ]
            )
    print(f"wrote {spec.out} {shape} uint8")
    print(f"  border {float(border.mean()):.4%} of cells")
    print(f"  keep {float(keep.mean()):.4%} of cells")


def build(
    spec: KeepMaskSpec,
    *,
    stage: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    init: bool = False,
) -> None:
    """Build a mask; FFN stage initialization and shard completion preserve the original CLI."""
    if spec.strategy == "downsampled_sources":
        if stage is not None or init or shard_id != 0 or num_shards != 1:
            raise ValueError("downsampled_sources is unsharded and does not accept stage/init")
        _downsampled_sources(spec)
        return
    if spec.strategy != "ffn_tissue_border":
        raise ValueError(f"unknown keep-mask strategy: {spec.strategy!r}")
    if stage not in ("tissue", "keep"):
        raise ValueError("ffn_tissue_border requires stage=tissue or stage=keep")
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_id < num_shards")
    args = [
        "build_keep_mask",
        "--stage",
        stage,
        "--out",
        str(spec.out),
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
        "--threshold",
        str(spec.threshold),
        "--z-slab",
        str(spec.z_slab),
        "--border-offset",
        str(spec.border_offset),
    ]
    if init:
        args.append("--init")
    if spec.tissue is not None:
        args.extend(("--tissue", str(spec.tissue)))
    if spec.em is not None:
        args.extend(("--em", str(spec.em)))
    # The original CLI body is kept verbatim as a movement regression contract.
    previous_argv = sys.argv
    try:
        sys.argv = args
        main()
    finally:
        sys.argv = previous_argv


def _load_spec(path: Path) -> KeepMaskSpec:
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["params"]
    return spec_from_params(params)


def cli_main(argv: list[str] | None = None) -> int:
    """Shared CLI for every configured cube dataset."""
    ap = argparse.ArgumentParser(description="Build a tutorial's configured keep-mask")
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stage", choices=("tissue", "keep"))
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)
    spec = _load_spec(args.params)
    if args.out is not None:
        spec = replace(spec, out=args.out)
    elif args.stage == "tissue" and spec.tissue is not None:
        spec = replace(spec, out=spec.tissue)
    build(
        spec, stage=args.stage, shard_id=args.shard_id, num_shards=args.num_shards, init=args.init
    )
    return 0


def moritz_main(argv: list[str] | None = None) -> int:
    """Keep the existing Moritz --bv/--out invocation over the shared implementation."""
    spec = _load_spec(
        Path(__file__).resolve().parents[2] / "tutorials/neuron_moritz_l4/params.yaml"
    )
    source = spec.sources[0]
    assert isinstance(source, MaskSource)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bv", type=Path, default=source.path)
    ap.add_argument("--out", type=Path, default=spec.out)
    args = ap.parse_args(argv)
    build(replace(spec, out=args.out, sources=[replace(source, path=args.bv)]))
    return 0
