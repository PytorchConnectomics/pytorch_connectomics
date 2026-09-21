#!/usr/bin/env python3
"""Does a different ABISS merge criterion beat `max`, and must the sigmoid be undone first?

TWO QUESTIONS, ONE RUN.

1. WOULD UNSCALING HELP? For `max`, provably not: `max` is monotone-invariant, so
   uncompressing the affinity and mapping the threshold through the same map
   yields the SAME segmentation. Only the threshold's numeric label changes. Task
   pairs (max_compressed, max_uncompressed) below are that proof, run rather than
   asserted -- they must agree row for row, and if they do not, the uncompression
   is wrong and nothing else here can be trusted.

   For `mean` and `pNN` it is the opposite: neither is monotone-invariant, so
   running them on sigmoid(0.2*logit(p)) scores a different and arbitrary
   criterion. Undoing the compression is a PRECONDITION for testing them, not an
   improvement in itself.

2. DOES A DIFFERENT CRITERION HELP? This is the question worth asking, because
   tuning thresholds is exhausted: on this volume the decode sits at VOI ~0.91
   with an oracle-merge ceiling of ~0.30, so ~67% of the remaining error is false
   splits that perfect agglomeration of the EXISTING fragments would fix. That
   points at the merge criterion, not at any threshold on it. `max` merges on the
   single strongest voxel across a contact surface, which one spurious voxel can
   trigger; `mean` and `p75` weigh the whole surface.

Seeding is held at the adopted 94% / 0.0 throughout, and percentile seeding is
itself monotone-invariant, so seeding is identical in both spaces and the merge
function is the only thing that varies.

Cubes and geometry are identical to cube_sweep.py, so numbers here are directly
comparable to that sweep -- and, as ever, crop-biased on splits and valid only
for comparing settings to each other.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = REPO / "lib/abiss/build/ws"
sys.path.insert(0, str(REPO))

CUBE = (96, 384, 384)
Z0 = 24
ORIGINS_YX = [(700, 700), (1500, 1900), (2400, 900),
              (3000, 2100), (1900, 1300), (2700, 1700)]
EPS = 1e-7


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def uncompress(v, scale):
    """Invert channel_activations: scale_sigmoid, v = sigmoid(scale*logit(p))."""
    v = np.clip(v.astype(np.float64), EPS, 1.0 - EPS)
    return (1.0 / (1.0 + np.exp(-(np.log(v / (1.0 - v)) / scale)))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--merge-function", default="max")
    ap.add_argument("--merge-thresholds", required=True)
    ap.add_argument("--uncompress", action="store_true")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="0.0")
    ap.add_argument("--cubes", type=int, default=len(ORIGINS_YX))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    import h5py, zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi
    from connectomics.metrics.oracle import oracle_merge_segmentation

    rav = _load_runner()
    mts = [float(x) for x in a.merge_thresholds.split(",")]
    dz, dy, dx = CUBE
    gt_store = zarr.open(GT, mode="r")
    rows = []
    space = "probability" if a.uncompress else "compressed"
    print(f"[{a.label}] merge_function={a.merge_function} space={space} mts={mts}", flush=True)

    for ci, (y0, x0) in enumerate(ORIGINS_YX[: a.cubes]):
        with h5py.File(a.affinity, "r") as f:
            aff = np.asarray(f["main"][:, Z0:Z0 + dz, y0:y0 + dy, x0:x0 + dx]).astype(np.float32)
        if a.uncompress:
            lo0, hi0 = float(aff.min()), float(aff.max())
            aff = uncompress(aff, a.scale)
            print(f"[{a.label}] cube {ci} uncompressed [{lo0:.4f},{hi0:.4f}] -> "
                  f"[{aff.min():.3e},{aff.max():.6f}]", flush=True)
        gt = np.asarray(gt_store[Z0:Z0 + dz, y0:y0 + dy, x0:x0 + dx])
        hv = rav._resolve_threshold(a.ws_high, aff, "ws_high")
        lv = rav._resolve_threshold(a.ws_low, aff, "ws_low")
        segs = rav._run_abiss_ws(
            aff, ws_binary=WS, ws_high_threshold=hv, ws_low_threshold=lv,
            ws_size_threshold=10_000_000, ws_dust_threshold=200,
            boundary_flags=[1, 1, 1, 1, 1, 1], offset=0, channels=[2, 1, 0],
            ws_merge_thresholds=mts, ws_merge_function=a.merge_function,
            edge_storage="source")
        for mt in mts:
            seg = segs[round(mt, 10)]
            vs, vm = voi(seg, gt)
            ovs, ovm = voi(oracle_merge_segmentation(seg, gt), gt)
            rows.append({"cube": ci, "merge_function": a.merge_function, "space": space,
                         "mt": mt, "voi": float(vs + vm), "voi_split": float(vs),
                         "voi_merge": float(vm), "voi_oracle": float(ovs + ovm),
                         "adapted_rand_error": float(adapted_rand(seg, gt)),
                         "nseg": int(seg.max())})
        print(f"[{a.label}] cube {ci}  " + "  ".join(
            f"mt{r['mt']:g}:{r['voi']:.3f}/{r['voi_oracle']:.3f}" for r in rows[-len(mts):]),
            flush=True)
        del aff, gt, segs

    print(f"\n=== [{a.label}] {a.merge_function} on {space}, mean over {a.cubes} cubes ===")
    print(f"{'mt':>10} {'VOI':>8} {'oracle':>8} {'split':>8} {'merge':>8} {'ARerr':>8} {'nseg':>9}")
    agg, summary = {}, []
    for r in rows:
        agg.setdefault(r["mt"], []).append(r)
    for mt, v in sorted(agg.items()):
        m = {f: float(np.mean([x[f] for x in v]))
             for f in ("voi", "voi_oracle", "voi_split", "voi_merge",
                       "adapted_rand_error", "nseg")}
        summary.append({"merge_function": a.merge_function, "space": space, "mt": mt, **m})
        print(f"{mt:10g} {m['voi']:8.4f} {m['voi_oracle']:8.4f} {m['voi_split']:8.4f} "
              f"{m['voi_merge']:8.4f} {m['adapted_rand_error']:8.4f} {m['nseg']:9.0f}")

    best = min(summary, key=lambda r: r["voi"])
    best_orc = min(summary, key=lambda r: r["voi_oracle"])
    print(f"\nbest VOI   : mt={best['mt']:g} -> {best['voi']:.4f}")
    print(f"best ORACLE: mt={best_orc['mt']:g} -> {best_orc['voi_oracle']:.4f}")
    if best["mt"] in (mts[0], mts[-1]) or best_orc["mt"] in (mts[0], mts[-1]):
        print("WARNING: an optimum sits at an END of the range -- widen it.")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "label": a.label, "merge_function": a.merge_function, "space": space,
            "scale": a.scale, "ws_high": a.ws_high, "ws_low": a.ws_low,
            "cube_shape": CUBE, "z0": Z0, "origins_yx": ORIGINS_YX[: a.cubes],
            "rows": rows, "summary": summary, "best_voi": best, "best_oracle": best_orc,
            "caveat": "interior cubes; crop-biased on splits. Relative comparison only.",
        }, indent=2))
        print(f"wrote {a.json}")


main()
