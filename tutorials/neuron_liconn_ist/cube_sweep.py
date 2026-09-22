#!/usr/bin/env python3
"""ABISS seeding-parameter sweep on interior cubes of IST LICONN val.

WHY CUBES, AND WHY INTERIOR. ws_high/ws_low need a NEW watershed per setting
(unlike ws_merge_threshold, which reuses one), so a whole-volume 2D sweep would
cost N*M decodes at ~43 min each. Cubes make the search affordable. They are
sampled away from the volume borders because near an edge the model has less
context and the proofread GT is least reliable -- that is a data-quality reason,
distinct from truncation.

WHAT THIS CAN AND CANNOT SAY. Every cube truncates GT at its own faces, so
absolute VOI here is biased high on splits exactly as the slabs were. The bias is
common to every parameter setting, so RELATIVE comparison across settings is
valid; the absolute numbers are not an operating point. Confirm the winner on the
whole volume before adopting it. GT ids are taken as-is and never re-labelled
inside the crop -- re-cc3d'ing GT in a small crop is what inverted merge-vs-
coverage comparisons on the NISB liconn volume.

WHY THE ORACLE CEILING IS THE PRIMARY METRIC. ws_high sets the FRAGMENTS; the
merge threshold only agglomerates them. Seeding too high bakes merges into the
fragments that no agglomeration can undo, which moves the ceiling rather than the
operating point. `voi_oracle` (perfect merging of the given fragments) isolates
that: it is what the decode could achieve if agglomeration were solved.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = REPO / "lib/abiss/build/ws"
sys.path.insert(0, str(REPO))

# val is (145, 4290, 3345); Z is only 145, so a "cube" is Z-limited.
CUBE = (96, 384, 384)
Z0 = 24                      # 24-voxel Z margin at both ends
ORIGINS_YX = [(700, 700), (1500, 1900), (2400, 900),
              (3000, 2100), (1900, 1300), (2700, 1700)]


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ws-high", default="88%,91%,94%,97%")
    ap.add_argument("--ws-low", default="10%,20%,30%")
    ap.add_argument("--merge-thresholds", default="0.38,0.41,0.44,0.47")
    ap.add_argument("--cubes", type=int, default=len(ORIGINS_YX))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    import h5py, zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi
    from connectomics.metrics.oracle import oracle_merge_segmentation

    rav = _load_runner()
    highs = a.ws_high.split(",")
    lows = a.ws_low.split(",")
    mts = [float(v) for v in a.merge_thresholds.split(",")]
    dz, dy, dx = CUBE
    gt_store = zarr.open(GT, mode="r")
    rows = []

    for ci, (y0, x0) in enumerate(ORIGINS_YX[: a.cubes]):
        with h5py.File(a.affinity, "r") as f:
            aff = np.asarray(f["main"][:, Z0:Z0 + dz, y0:y0 + dy, x0:x0 + dx]).astype(np.float32)
        gt = np.asarray(gt_store[Z0:Z0 + dz, y0:y0 + dy, x0:x0 + dx])
        print(f"[{a.label}] cube {ci} z{Z0} y{y0} x{x0} aff{aff.shape} gt_ids={len(np.unique(gt))}", flush=True)
        for h in highs:
            for lo in lows:
                hv = rav._resolve_threshold(h, aff, "ws_high")
                lv = rav._resolve_threshold(lo, aff, "ws_low")
                segs = rav._run_abiss_ws(
                    aff, ws_binary=WS, ws_high_threshold=hv, ws_low_threshold=lv,
                    ws_size_threshold=10_000_000, ws_dust_threshold=200,
                    boundary_flags=[1, 1, 1, 1, 1, 1], offset=0, channels=[2, 1, 0],
                    ws_merge_thresholds=mts, ws_merge_function="max", edge_storage="source")
                for mt in mts:
                    seg = segs[round(mt, 10)]
                    vs, vm = voi(seg, gt)
                    orc = oracle_merge_segmentation(seg, gt)
                    ovs, ovm = voi(orc, gt)
                    rows.append({"cube": ci, "ws_high": h, "ws_low": lo, "mt": mt,
                                 "voi": float(vs + vm), "voi_split": float(vs),
                                 "voi_merge": float(vm), "voi_oracle": float(ovs + ovm),
                                 "adapted_rand_error": float(adapted_rand(seg, gt)),
                                 "nseg": int(seg.max())})
                print(f"   high={h} low={lo}  " + "  ".join(
                    f"mt{r['mt']:.2f}:VOI {r['voi']:.3f}/orc {r['voi_oracle']:.3f}"
                    for r in rows[-len(mts):]), flush=True)
        del aff, gt

    print(f"\n=== [{a.label}] mean over {a.cubes} cubes ===")
    print(f"{'high':>5} {'low':>5} {'mt':>5} {'VOI':>8} {'oracle':>8} {'split':>8} {'merge':>8} {'ARerr':>8}")
    agg = {}
    for r in rows:
        agg.setdefault((r["ws_high"], r["ws_low"], r["mt"]), []).append(r)
    summary = []
    for k, v in sorted(agg.items()):
        m = {f: float(np.mean([x[f] for x in v]))
             for f in ("voi", "voi_oracle", "voi_split", "voi_merge", "adapted_rand_error")}
        summary.append({"ws_high": k[0], "ws_low": k[1], "mt": k[2], **m})
        print(f"{k[0]:>5} {k[1]:>5} {k[2]:>5.2f} {m['voi']:8.4f} {m['voi_oracle']:8.4f} "
              f"{m['voi_split']:8.4f} {m['voi_merge']:8.4f} {m['adapted_rand_error']:8.4f}")

    best_voi = min(summary, key=lambda r: r["voi"])
    best_orc = min(summary, key=lambda r: r["voi_oracle"])
    print(f"\nbest VOI   : high={best_voi['ws_high']} low={best_voi['ws_low']} "
          f"mt={best_voi['mt']:.2f} -> {best_voi['voi']:.4f}")
    print(f"best ORACLE: high={best_orc['ws_high']} low={best_orc['ws_low']} "
          f"-> {best_orc['voi_oracle']:.4f}   (the recoverable ceiling; ws_high owns this)")
    if best_orc["ws_high"] in (highs[0], highs[-1]) or best_orc["ws_low"] in (lows[0], lows[-1]):
        print("WARNING: an optimum sits at an EDGE of the swept grid -- widen it.")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "label": a.label, "affinity": str(a.affinity), "cube_shape": CUBE,
            "z0": Z0, "origins_yx": ORIGINS_YX[: a.cubes], "rows": rows,
            "summary": summary, "best_voi": best_voi, "best_oracle": best_orc,
            "caveat": "interior cubes; VOI biased high on splits as slabs were. "
                      "Relative comparison only; confirm on the whole volume.",
        }, indent=2))
        print(f"wrote {a.json}")


main()
