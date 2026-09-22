"""Build the [36, 18, 18] nm ZYX ExPID82 train/val set from the mip0 export.

Card MSIDEPLOY-MODEL-002, Stage 2 route B. CPU only.

Source: final_proofread_mip0/{train,val}/data.zarr/{img,seg}, [12, 9, 9] nm ZYX.

Z  -- DECIMATION, no averaging: output plane j of offset o is mip0 plane 3*j + o.
      Offsets 0/1/2 of train give three volumes whose IMAGE planes are disjoint.
      Val is built at offset 0 only.
XY -- img: exact 2x2 block mean, np.rint, uint8. NOT scripts/downsample_data.py's
      image mode: that is ndimage.zoom(order=1), which on a 12x12 check against
      final_proofread/img correlates 0.932, while the 2x2 block mean correlates
      0.961 on the same single plane (0.982 when the 2-plane Z mean is also
      applied -- final_proofread/img is the source's native mip1 JPEG layer, a
      2x2x2 average; decimation deliberately drops the Z half of that).
      seg: strided [::2, ::2], via downsample_data.downsample_volume_zyx(mode="label").
      The mip0 seg is a bit-exact 2x nearest upsample of the 18x18x24 seg
      (attrs segmentation_upsample_factor_zyx, and checked 100% equal on 4 planes),
      so this recovers the original 18 nm seg in XY exactly -- no ids invented.

THE SEG IS 24 nm IN Z. mip0 plane z carries the 24 nm plane z // 2, so output
plane j of offset o carries 24 nm plane (3j + o) // 2 -- the plane that
physically contains the kept image plane. Kept 24 nm planes step irregularly
(24/48 nm), and the three offsets share some seg planes: only the images are
disjoint. Verified at the end against final_proofread/*/seg directly.

Streams 96-plane x 1024x1024 tiles (96 = lcm(3, 32): each tile writes exactly
one 32-deep output chunk row per offset, and 512 output XY is chunk-aligned), so
tiles write disjoint chunks and run in parallel.

    python z36_build.py --out /projects/weilab/dataset/liconn/pytc/final_proofread_z36
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from downsample_data import downsample_volume_zyx  # noqa: E402

SRC = "/projects/weilab/dataset/liconn/pytc/final_proofread_mip0"
REF = "/projects/weilab/dataset/liconn/pytc/final_proofread"  # 24 nm, for verification
ZF, XYF = 3, 2
ZSLAB, TILE = 96, 1024
# (source split, offset, output leaf). Val leaf is NOT `val`: checkpoint_dispatch
# takes the inference output leaf from the data path, and a `val` leaf would
# overwrite the published 18 nm val affinity under the eb2 checkpoint.
JOBS = [("train", 0, "train_o0"), ("train", 1, "train_o1"), ("train", 2, "train_o2"),
        ("val", 0, "val_z36")]


def block_mean_xy(img: np.ndarray) -> np.ndarray:
    z, y, x = img.shape
    y2, x2 = y - y % XYF, x - x % XYF
    out = img[:, :y2, :x2].astype(np.float32).reshape(z, y2 // XYF, XYF, x2 // XYF, XYF)
    return np.clip(np.rint(out.mean((2, 4))), 0, 255).astype(np.uint8)


def n_out(n_in: int, o: int) -> int:
    return len(range(o, n_in, ZF))


def _tile(args):
    split, z0, y0, x0, out_root, offsets = args
    import zarr

    src = zarr.open_group(f"{SRC}/{split}/data.zarr", mode="r")
    z1 = min(z0 + ZSLAB, src["img"].shape[0])
    ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
    img = np.asarray(src["img"][z0:z1, ys, xs])
    seg = np.asarray(src["seg"][z0:z1, ys, xs])
    for o, leaf in offsets:
        sel = np.arange(o, z1 - z0, ZF)  # z0 % 3 == 0, so global plane z0+sel = o mod 3
        if sel.size == 0:
            continue
        dst = zarr.open_group(f"{out_root}/{leaf}/data.zarr", mode="r+")
        j0 = z0 // ZF
        oi = block_mean_xy(img[sel])
        os_ = downsample_volume_zyx(seg[sel], (1, XYF, XYF), mode="label")
        dst["img"][j0:j0 + sel.size, y0 // XYF:y0 // XYF + oi.shape[1],
                   x0 // XYF:x0 // XYF + oi.shape[2]] = oi
        dst["seg"][j0:j0 + sel.size, y0 // XYF:y0 // XYF + os_.shape[1],
                   x0 // XYF:x0 // XYF + os_.shape[2]] = os_
    return z0, y0, x0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import zarr

    if a.out.exists():
        raise SystemExit(f"{a.out} already exists; refusing to overwrite")

    by_split: dict[str, list[tuple[int, str]]] = {}
    for split, o, leaf in JOBS:
        by_split.setdefault(split, []).append((o, leaf))

    shapes = {}
    for split, offsets in by_split.items():
        src = zarr.open_group(f"{SRC}/{split}/data.zarr", mode="r")
        zin, yin, xin = (int(v) for v in src["img"].shape)
        assert yin % XYF == 0 and xin % XYF == 0
        for o, leaf in offsets:
            shp = (n_out(zin, o), yin // XYF, xin // XYF)
            shapes[leaf] = shp
            # zarr_format=2, source codecs/chunks: matches every array in this dataset.
            root = zarr.open_group(str(a.out / leaf / "data.zarr"), mode="w", zarr_format=2)
            for key, dt in (("img", "|u1"), ("seg", "<u8")):
                root.create_array(key, shape=shp, chunks=tuple(src[key].chunks), dtype=dt,
                                  compressors=src[key].compressors, fill_value=0)
            root.attrs.update({
                "axes": ["z", "y", "x"],
                "resolution_nm_zyx": [36.0, 18.0, 18.0],
                "derived_from": f"{SRC}/{split}/data.zarr",
                "derived_by": "tutorials/neuron_liconn_ist/z36_build.py",
                "z_rule": f"decimation: mip0 plane {ZF}*j + {o}",
                "split": split, "z_offset": o,
            })
            print(f"{split} o{o} -> {leaf} {shp}", flush=True)

        tasks = [(split, z0, y0, x0, str(a.out), offsets)
                 for z0 in range(0, zin, ZSLAB)
                 for y0 in range(0, yin, TILE) for x0 in range(0, xin, TILE)]
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(_tile, tasks)):
                if i % 50 == 0 or i == len(tasks) - 1:
                    print(f"  {split} tile {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)

    # Verification: every output against a direct recomputation (img) and against
    # the 24 nm seg (seg), on first / middle / last planes.
    for split, o, leaf in JOBS:
        dst = zarr.open_group(str(a.out / leaf / "data.zarr"), mode="r")
        src = zarr.open_group(f"{SRC}/{split}/data.zarr", mode="r")
        ref = zarr.open_group(f"{REF}/{split}/data.zarr", mode="r")
        nz = shapes[leaf][0]
        for j in sorted({0, nz // 2, nz - 1}):
            zi = ZF * j + o
            want = block_mean_xy(np.asarray(src["img"][zi:zi + 1]))[0]
            if not np.array_equal(want, np.asarray(dst["img"][j])):
                raise SystemExit(f"img verification FAILED {leaf} plane {j}")
            if not np.array_equal(np.asarray(ref["seg"][zi // 2]), np.asarray(dst["seg"][j])):
                raise SystemExit(f"seg verification FAILED {leaf} plane {j} vs 24nm plane {zi // 2}")
            print(f"  verified {leaf} plane {j} (mip0 {zi}, 24nm seg plane {zi // 2})", flush=True)

    for split, o, leaf in JOBS:
        zin = int(zarr.open_group(f"{SRC}/{split}/data.zarr", mode="r")["img"].shape[0])
        kept = list(range(o, zin, ZF))
        prov = {
            "card": "MSIDEPLOY-MODEL-002 (Stage 2, route B)",
            "brief": "dw-research/projects/msi_liconn_deploy/brief_36nm.md",
            "source": f"{SRC}/{split}/data.zarr",
            "source_resolution_nm_zyx": [12.0, 9.0, 9.0],
            "target_resolution_nm_zyx": [36.0, 18.0, 18.0],
            "source_shape_zyx": [zin, *[2 * s for s in shapes[leaf][1:]]],
            "output_shape_zyx": list(shapes[leaf]),
            "z_rule": "decimation (single plane, no averaging): output j = mip0 plane 3*j + offset",
            "z_offset": o,
            "selected_mip0_planes": kept,
            "seg_24nm_plane_per_output": [z // 2 for z in kept],
            "image_xy_method": "exact 2x2 block mean, np.rint, clip [0,255], uint8",
            "image_xy_note": ("final_proofread/img (the [24,18,18] set) is the source's native "
                              "mip1 JPEG layer = 2x2x2 average; 2x2 block mean matches its XY "
                              "(corr 0.961 single plane / 0.982 with Z mean) better than "
                              "downsample_data.py zoom order=1 (0.932). Z mean deliberately not "
                              "applied: decimation per brief."),
            "seg_xy_method": ("strided [::2, ::2] (downsample_data.downsample_volume_zyx "
                              "mode=label); mip0 seg is a 2x nearest upsample of the 18x18x24 "
                              "seg, so this is exact, label-preserving"),
            "seg_z_note": ("GT exists only at 24 nm in Z; each output plane carries the 24 nm "
                           "plane containing its image plane. Kept 24 nm planes step 24/48 nm; "
                           "offsets share seg planes, only images are disjoint."),
            "output": str(a.out / leaf / "data.zarr"),
            "created_by": str(Path(__file__).resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (a.out / leaf / "provenance.json").write_text(json.dumps(prov, indent=2))
        print(f"wrote {a.out / leaf / 'provenance.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
