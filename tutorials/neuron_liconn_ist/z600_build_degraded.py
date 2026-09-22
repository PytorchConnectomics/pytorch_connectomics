"""Step 1 of the 600 nm gate: build the Z-degraded val image.

Card MSIDEPLOY-MODEL-002, Stage 1 step 1. CPU only.

Degrades `final_proofread/val` in Z ONLY, by an exact area average (cv2.INTER_AREA
semantics) from 145 planes at 24 nm to 104 planes at 33.3333 nm. XY is untouched:
after the standard XY x2 block average the 600 nm volumes are already on the
training grid in XY, so Z is the whole of the mismatch this gate is about.

A whole-volume cv2.resize is not viable at 2.08 Gvox, and is not needed -- the
area average along one axis IS a streamable overlap-weighted sum, 2-3 input
planes per output plane. Peak RSS is a few hundred MB.

--replicate BUILDS THE CONTROL VOLUME INSTEAD. The primary degraded volume
confounds three things at once: Z information is lost, the voxel grid no longer
matches the training grid (neurite caliber in voxels changes by 1.39x), and the
volume becomes 104 planes deep against a 128-deep inference window, so it gets
ONE Z window position where the 145-plane baseline got four. --replicate writes
the same 104 area-averaged planes back onto the ORIGINAL 145-plane grid by the
nearest-upsample map, giving a volume that has lost exactly the same Z
information but is identical to the baseline in grid, shape, window count and
scoring path. The only difference from the baseline run is then the Z
information content itself.

Read together: if the primary loses a lot and the control loses little, the cost
is grid mismatch -- which is what a 36 nm-matched model would recover. If the
control loses as much, the cost is information a retrained model cannot get back.

THE SEGMENTATION IS NOT DEGRADED. Stage 1 needs no label transform at all: the
18 nm GT stays the scoring reference and the degraded segmentation is upsampled
back to it. That is what keeps the gate clear of the A1 label-ceiling problem
that Stage 2 cannot avoid.

OUTPUT LEAF IS `val_z600`, NOT `val`, deliberately. `checkpoint_dispatch` derives
the inference output dir from the checkpoint and then takes the per-volume leaf
from the DATA PATH's last component. A `val` leaf under this checkpoint would
write over
  outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/
    raw_x1_ch0-1-2.h5
which is the published 18 nm val affinity behind the 0.9130 baseline.

    python z600_build_degraded.py --out <dir>/val_z600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

SRC = "/projects/weilab/dataset/liconn/pytc/final_proofread/val"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from z600_zresample import Z_FACTOR, Z_NATIVE_NM, provenance, zplan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--factor", type=float, default=Z_FACTOR)
    ap.add_argument("--replicate", action="store_true",
                    help="write the control volume: the same area-averaged planes "
                         "replicated back onto the original 145-plane grid")
    a = ap.parse_args()

    import zarr

    src_img = zarr.open(f"{a.src}/data.zarr/img", mode="r")
    n_in, ny, nx = (int(v) for v in src_img.shape)
    if src_img.dtype != np.uint8:
        raise SystemExit(f"expected uint8 image, got {src_img.dtype}")
    n_out, weights, sel, up = zplan(n_in, a.factor)
    n_write = n_in if a.replicate else n_out
    out_res = [24.0, 18.0, 18.0] if a.replicate else [Z_NATIVE_NM, 18.0, 18.0]
    print(f"src {src_img.shape} {src_img.dtype} chunks {src_img.chunks}", flush=True)
    print(f"-> {(n_write, ny, nx)} at {out_res} nm ZYX"
          f"{' (REPLICATED control)' if a.replicate else ''}", flush=True)

    if a.out.exists():
        raise SystemExit(f"{a.out} already exists; refusing to overwrite")
    # zarr_format=2 explicitly: the installed zarr is 3.x and defaults to v3,
    # but every other array in this dataset is v2 and the v3 writer rejects the
    # source's numcodecs Blosc outright. The written `.zarray` is byte-identical
    # to the source's apart from `shape`.
    root = zarr.open_group(str(a.out / "data.zarr"), mode="w", zarr_format=2)
    dst = root.create_array(
        "img", shape=(n_write, ny, nx), chunks=tuple(int(c) for c in src_img.chunks),
        dtype="|u1", compressors=src_img.compressors, fill_value=0, overwrite=True)
    root.attrs.update({
        "axes": ["z", "y", "x"],
        "resolution_nm_zyx": out_res,
        "derived_from": f"{a.src}/data.zarr/img",
        "derived_by": "tutorials/neuron_liconn_ist/z600_build_degraded.py",
        "note": ("Z-degraded copy for the MSIDEPLOY-MODEL-002 gate. Image only; the "
                 "18 nm segmentation is the scoring reference and is NOT copied here. "
                 "This is a software Z degradation, not a 600 nm acquisition."
                 + (" REPLICATED control: 104 area-averaged planes written back onto "
                    "the 145-plane grid, so Z information matches the degraded volume "
                    "but grid, shape and inference window geometry match the baseline."
                    if a.replicate else "")),
    })

    # Destination planes for each coarse plane j: itself, or every original plane
    # that reads j under the nearest-upsample map.
    targets = ([np.flatnonzero(up == j).tolist() for j in range(n_out)] if a.replicate
               else [[j] for j in range(n_out)])
    if a.replicate and sum(len(t) for t in targets) != n_in:
        raise AssertionError("replicate target map does not cover every original plane")

    cache: dict[int, np.ndarray] = {}
    t0 = time.time()
    for j, row in enumerate(weights):
        need = {z for z, _ in row}
        for z in list(cache):
            if z not in need:
                del cache[z]
        acc = np.zeros((ny, nx), dtype=np.float32)
        for z, w in row:
            if z not in cache:
                cache[z] = np.asarray(src_img[z], dtype=np.float32)
            acc += np.float32(w) * cache[z]
        plane = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
        for t in targets[j]:
            dst[t] = plane
        if j % 10 == 0:
            print(f"  plane {j}/{n_out}  {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {n_write} planes in {time.time() - t0:.0f}s", flush=True)

    prov = provenance(n_in, (ny, nx), a.factor)
    prov.update({
        "variant": "replicated_control" if a.replicate else "primary_degraded",
        "output_shape_zyx": [n_write, ny, nx],
        "output_resolution_nm_zyx": out_res,
        "output": str(a.out / "data.zarr/img"),
        "image_dtype": "uint8",
        "rounding": "np.rint on the float32 weighted sum, clipped to [0, 255]",
        "created_by": str(Path(__file__).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {a.out / 'provenance.json'}", flush=True)

    # Verify against a direct recomputation on three planes, including a 3-plane
    # window (j=2) -- the only structurally different case.
    for j in (0, 2, n_out - 1):
        row = weights[j]
        ref = np.zeros((ny, nx), dtype=np.float32)
        for z, w in row:
            ref += np.float32(w) * np.asarray(src_img[z], dtype=np.float32)
        ref = np.clip(np.rint(ref), 0, 255).astype(np.uint8)
        for t in targets[j]:
            if not np.array_equal(ref, np.asarray(dst[t])):
                raise SystemExit(f"verification FAILED on output plane {t} (coarse {j})")
        print(f"  verified coarse plane {j} ({len(row)} source planes) -> "
              f"output {targets[j]} exact", flush=True)
    print("DONE", flush=True)


main()
