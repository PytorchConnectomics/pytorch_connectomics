#!/usr/bin/env python3
"""ABISS merge-threshold sweep on IST LICONN val slabs, scored with VOI.

One watershed per slab, then the merge step repeated per threshold (ABISS batch
mode), so the sweep costs barely more than a single decode.

Slabs are FULL-Z (145) so GT objects are only truncated in XY; the crop-eval
trap that inverted merge-vs-coverage comparisons on the NISB liconn volume came
from re-cc3d'ing GT inside a small 3D crop, which is not done here -- GT ids are
taken as-is from the proofread volume.

    python tutorials/neuron_liconn_ist/sweep_merge_threshold.py --slabs 3
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]  # repo root (tutorials/<name>/<file>)
MAIN_REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
sys.path.insert(0, str(REPO))

AFF = MAIN_REPO / "outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = MAIN_REPO / "lib/abiss/build/ws"


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slabs", type=int, default=3)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="20%")
    ap.add_argument(
        "--merge-thresholds",
        default="0.40,0.45,0.50,0.53,0.56,0.59,0.62,0.66,0.70",
    )
    args = ap.parse_args()

    import h5py
    import zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi

    rav = _load_runner()
    mts = [float(v) for v in args.merge_thresholds.split(",")]
    S = args.size

    # Disjoint full-z slabs spread across the val face.
    origins = [(600, 600), (2000, 1600), (3200, 800), (1200, 2200), (2600, 2400)][: args.slabs]

    gt_store = zarr.open(GT, mode="r")
    rows = []
    for si, (y0, x0) in enumerate(origins):
        with h5py.File(AFF, "r") as f:
            aff = np.asarray(f["main"][:, :, y0 : y0 + S, x0 : x0 + S]).astype(np.float32)
        gt = np.asarray(gt_store[:, y0 : y0 + S, x0 : x0 + S])
        hi = rav._resolve_threshold(args.ws_high, aff, "ws_high")
        lo = rav._resolve_threshold(args.ws_low, aff, "ws_low")
        print(
            f"[slab {si}] y0={y0} x0={x0} aff{aff.shape} "
            f"range=[{aff.min():.3f},{aff.max():.3f}] ws_high={hi:.4f} ws_low={lo:.4f} "
            f"gt_ids={len(np.unique(gt))}",
            flush=True,
        )
        segs = rav._run_abiss_ws(
            aff,
            ws_binary=WS,
            ws_high_threshold=hi,
            ws_low_threshold=lo,
            ws_size_threshold=10_000_000,
            ws_dust_threshold=200,
            boundary_flags=[1, 1, 1, 1, 1, 1],
            offset=0,
            channels=[2, 1, 0],
            ws_merge_thresholds=mts,
            ws_merge_function="max",
            edge_storage="source",
        )
        for mt in mts:
            seg = segs[round(mt, 10)]
            vs, vm = voi(seg, gt)
            ar = adapted_rand(seg, gt)
            rows.append((si, mt, vs, vm, vs + vm, ar, int(seg.max())))
            print(
                f"   mt={mt:.2f}  VOI={vs + vm:.4f} (split {vs:.4f} / merge {vm:.4f})  "
                f"ARerr={ar:.4f}  nseg={int(seg.max())}",
                flush=True,
            )

    print("\n=== mean over slabs ===")
    print(f"{'mt':>6} {'VOI':>8} {'split':>8} {'merge':>8} {'ARerr':>8}")
    best = None
    for mt in mts:
        sel = [r for r in rows if r[1] == mt]
        v = float(np.mean([r[4] for r in sel]))
        s = float(np.mean([r[2] for r in sel]))
        m = float(np.mean([r[3] for r in sel]))
        a = float(np.mean([r[5] for r in sel]))
        print(f"{mt:6.2f} {v:8.4f} {s:8.4f} {m:8.4f} {a:8.4f}")
        if best is None or v < best[1]:
            best = (mt, v)
    print(f"\nbest mean VOI: mt={best[0]:.2f} -> {best[1]:.4f}")


if __name__ == "__main__":
    main()
