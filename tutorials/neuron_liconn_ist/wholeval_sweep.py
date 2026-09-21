#!/usr/bin/env python3
"""Whole-volume ABISS merge-threshold sweep on IST LICONN val, scored against GT.

Replaces the slab sweep for the FINAL comparison. Slab VOI is biased high on
splits -- GT truncated at the crop boundary leaves cross-boundary splits
unpenalised -- so a slab optimum is an upper bracket, not the operating point.
Cubes would be worse, not better: they add Z truncation on top of XY.

One watershed, N merge thresholds (ABISS batch mode), each scored. Cost is
~1 decode + N scorings, not N decodes.

    python wholeval_sweep.py --affinity <h5> --label eb8 --thresholds 0.41,0.44,0.47
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = REPO / "lib/abiss/build/ws"
sys.path.insert(0, str(REPO))


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--thresholds", default="0.41,0.44,0.47")
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="20%")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--merge-function", default="max",
                    help="ABISS edge score: max, mean, or pNN. `max` is monotone-"
                         "invariant; mean/pNN are NOT, so they require --uncompress.")
    ap.add_argument("--uncompress", action="store_true",
                    help="Invert channel_activations: scale_sigmoid before decoding, "
                         "v = sigmoid(scale*logit(p)). Required for mean/pNN.")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--oracle", action="store_true",
                    help="Also report the oracle-merge ceiling. Costs ~50 GB transient "
                         "(an int64 key array over 2.08 Gvox plus its sort), so the "
                         "sweep frees each segmentation as it is scored.")
    a = ap.parse_args()

    import h5py, zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi
    if a.oracle:
        from connectomics.metrics.oracle import oracle_merge_segmentation

    rav = _load_runner()
    mts = [float(v) for v in a.thresholds.split(",")]

    print(f"[{a.label}] reading {a.affinity}", flush=True)
    with h5py.File(a.affinity, "r") as f:
        aff = np.asarray(f["main"]).astype(np.float32)
    print(f"[{a.label}] affinity {aff.shape} range=[{aff.min():.3f},{aff.max():.3f}]", flush=True)

    if a.merge_function != "max" and not a.uncompress:
        raise SystemExit(
            f"`{a.merge_function}` is not monotone-invariant, so scoring it on the "
            "compressed affinity measures an arbitrary criterion. Pass --uncompress.")
    if a.uncompress:
        eps = 1e-7
        v = np.clip(aff.astype(np.float64), eps, 1.0 - eps)
        aff = (1.0 / (1.0 + np.exp(-(np.log(v / (1.0 - v)) / a.scale)))).astype(np.float32)
        del v
        print(f"[{a.label}] uncompressed (scale {a.scale}) -> "
              f"[{aff.min():.3e},{aff.max():.6f}]", flush=True)

    hi = rav._resolve_threshold(a.ws_high, aff, "ws_high")
    lo = rav._resolve_threshold(a.ws_low, aff, "ws_low")
    print(f"[{a.label}] ws_high={hi:.6f} ws_low={lo:.6f} mts={mts} "
          f"fn={a.merge_function} space={'probability' if a.uncompress else 'compressed'}",
          flush=True)

    segs = rav._run_abiss_ws(
        aff, ws_binary=WS, ws_high_threshold=hi, ws_low_threshold=lo,
        ws_size_threshold=10_000_000, ws_dust_threshold=200,
        boundary_flags=[1, 1, 1, 1, 1, 1], offset=0, channels=[2, 1, 0],
        ws_merge_thresholds=mts, ws_merge_function=a.merge_function,
        edge_storage="source")
    del aff

    gt = np.asarray(zarr.open(GT, mode="r")[:])
    rows = []
    for mt in mts:
        key = round(mt, 10)
        seg = segs[key]
        vs, vm = voi(seg, gt)
        ar = adapted_rand(seg, gt)
        row = {"mt": mt, "voi_split": float(vs), "voi_merge": float(vm),
               "voi": float(vs + vm), "adapted_rand_error": float(ar),
               "nseg": int(seg.max())}
        orc_msg = ""
        if a.oracle:
            orc = oracle_merge_segmentation(seg, gt)
            ovs, ovm = voi(orc, gt)
            del orc
            row["voi_oracle"] = float(ovs + ovm)
            orc_msg = f"  oracle={ovs+ovm:.4f}"
        rows.append(row)
        print(f"[{a.label}] mt={mt:.2f}  VOI={vs+vm:.4f} (split {vs:.4f} / merge {vm:.4f})  "
              f"ARerr={ar:.4f}  nseg={int(seg.max())}{orc_msg}", flush=True)
        # Free as we go: with --oracle the transient key/sort arrays are the peak.
        del seg
        del segs[key]

    best = min(rows, key=lambda r: r["voi"])
    print(f"[{a.label}] best VOI: mt={best['mt']:.2f} -> {best['voi']:.4f}", flush=True)
    if best["mt"] in (mts[0], mts[-1]):
        print(f"[{a.label}] WARNING: optimum at an END of the range -- widen it.", flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"label": a.label, "affinity": str(a.affinity),
                                      "ws_high": a.ws_high, "ws_low": a.ws_low,
                                      "merge_function": a.merge_function,
                                      "space": "probability" if a.uncompress else "compressed",
                                      "whole_volume": True, "rows": rows, "best": best}, indent=2))
        print(f"wrote {a.json}", flush=True)


main()
