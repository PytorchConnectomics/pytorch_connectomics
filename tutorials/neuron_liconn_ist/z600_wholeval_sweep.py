"""Whole-volume ABISS merge-threshold sweep for the 600 nm gate, scored on TWO grids.

Card MSIDEPLOY-MODEL-002, Stage 1 steps 3-4. This is `wholeval_sweep.py` with one
addition, forced by what step 0 measured.

THE PROBLEM STEP 0 FOUND. The card's step 4 says to upsample the degraded
segmentation back to the 18 nm grid and score it against the GT behind 0.9130.
That round trip -- 145 -> 104 -> 145 planes -- replaces 70 of the 145 planes with
a neighbouring plane, and the measured ORACLE FLOOR through it is
**VOI 0.6235**: the best score anything can get, model or not, against a ~0.01
noise floor. Read on that grid the gate cannot distinguish a model effect from
z re-quantisation, and it would fire either way.

THE FIX, and why it is fair. Score on the DEGRADED grid instead: mode-downsample
the 18 nm GT to the same 104 planes and compare there. Its oracle floor is
exactly 0 by construction. The card warns, correctly, that VOI is not comparable
across voxel grids -- so the offset is MEASURED rather than assumed: run this
same script on the EXISTING 18 nm affinity, which reproduces 0.9130 on the 18 nm
grid and simultaneously gives that segmentation's score on the 104-plane grid.
The gate is then the difference between two numbers taken on one grid, and the
18 nm run doubles as an end-to-end check that this script reproduces the
published baseline.

The 104-plane grid preserves each object's relative volume almost exactly --
measured voxel retention is 71.72% overall and 71.72% for objects >= 1000 voxels,
i.e. 104/145 uniformly -- which is why VOI transfers onto it as well as it does.
That is an argument, not a proof, which is what the measured 18 nm offset is for.

Both grids are reported for every threshold. The 18 nm column is the card's
number and must be read WITH the 0.6235 oracle floor; the 104 column is the one
the gate verdict is taken on.

    python z600_wholeval_sweep.py --affinity <h5> --label z600 --source-grid z600 \
        --thresholds 0.38,0.41,0.44,0.47,0.50,0.53 --workdir <scratch on /projects>
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = REPO / "lib/abiss/build/ws"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from z600_zresample import Z_FACTOR, zplan  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--source-grid", choices=["18nm", "z600"], required=True,
                    help="which grid the affinity lives on: 145 planes at 24 nm, or "
                         "104 planes at 33.3333 nm")
    ap.add_argument("--thresholds", default="0.38,0.41,0.44,0.47,0.50,0.53")
    # ws_high/ws_low are the baseline's settings, not the ones the ws_low study
    # later preferred (0.0). They are kept so the comparison is to 0.9130 exactly;
    # changing them would confound the gate with MSIDEPLOY-JOB-001.
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="20%")
    ap.add_argument("--workdir", type=Path, required=True,
                    help="ABISS scratch on /projects -- node-local /tmp is too small")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--factor", type=float, default=Z_FACTOR)
    ap.add_argument("--skip-adapted-rand", action="store_true")
    ap.add_argument("--keep-workdir", action="store_true")
    a = ap.parse_args()

    import h5py
    import zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi

    rav = _load_runner()
    mts = [float(v) for v in a.thresholds.split(",")]
    a.workdir = a.workdir.resolve()
    a.workdir.mkdir(parents=True, exist_ok=True)

    print(f"[{a.label}] reading {a.affinity}", flush=True)
    with h5py.File(a.affinity, "r") as f:
        aff = np.asarray(f["main"]).astype(np.float32)
    print(f"[{a.label}] affinity {aff.shape} range=[{aff.min():.3f},{aff.max():.3f}] "
          f"p25/p50/p75={np.percentile(aff, [25, 50, 75]).round(4).tolist()}", flush=True)

    gt = np.asarray(zarr.open(a.gt, mode="r")[:])
    n_in = int(gt.shape[0])
    n_out, _, sel, up = zplan(n_in, a.factor)
    expect_z = n_in if a.source_grid == "18nm" else n_out
    if int(aff.shape[1]) != expect_z:
        raise SystemExit(
            f"--source-grid {a.source_grid} expects {expect_z} affinity planes, "
            f"got {aff.shape[1]}")
    gt104 = gt[sel]
    print(f"[{a.label}] GT {gt.shape} -> mode-downsampled {gt104.shape}", flush=True)

    hi = rav._resolve_threshold(a.ws_high, aff, "ws_high")
    lo = rav._resolve_threshold(a.ws_low, aff, "ws_low")
    print(f"[{a.label}] ws_high={hi:.6f} ws_low={lo:.6f} mts={mts} fn=max "
          f"space=compressed grid={a.source_grid}", flush=True)

    rows = []

    def score(_i: int, mt: float, seg: np.ndarray) -> None:
        """Score one threshold as ABISS produces it, then let it be freed.

        Consuming through `on_batch_result` rather than the returned dict is what
        keeps the ceiling at ONE segmentation instead of len(mts): a uint64
        segmentation is 11.9 GB at 1.49 Gvox and 16.6 GB at 2.08 Gvox, so a
        six-point grid held at once would be 71-100 GB on top of the GT, its
        104-plane copy and the scoring transients.
        """
        t0 = time.time()
        seg = np.ascontiguousarray(seg)
        row = {"mt": float(mt), "nseg": int(seg.max())}

        # --- the card's grid: everything back on 145 planes at 24 nm ---
        pred18 = seg[up] if a.source_grid == "z600" else seg
        vs, vm = voi(pred18, gt)
        row.update(grid18nm_voi_split=float(vs), grid18nm_voi_merge=float(vm),
                   grid18nm_voi=float(vs + vm))
        if not a.skip_adapted_rand:
            row["grid18nm_adapted_rand_error"] = float(adapted_rand(pred18, gt))
        if a.source_grid == "z600":
            del pred18
            gc.collect()

        # --- the readable grid: everything on 104 planes at 33.3333 nm ---
        pred104 = seg if a.source_grid == "z600" else seg[sel]
        vs4, vm4 = voi(pred104, gt104)
        row.update(grid104_voi_split=float(vs4), grid104_voi_merge=float(vm4),
                   grid104_voi=float(vs4 + vm4))
        if not a.skip_adapted_rand:
            row["grid104_adapted_rand_error"] = float(adapted_rand(pred104, gt104))
        if a.source_grid == "18nm":
            del pred104
        rows.append(row)
        print(f"[{a.label}] mt={mt:.2f}  18nm-grid VOI={row['grid18nm_voi']:.4f} "
              f"(split {vs:.4f} / merge {vm:.4f})  |  104-grid VOI={row['grid104_voi']:.4f} "
              f"(split {vs4:.4f} / merge {vm4:.4f})  nseg={row['nseg']}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps({"label": a.label, "partial": True,
                                          "rows": rows}, indent=2))
        gc.collect()

    rav._run_abiss_ws(
        aff, ws_binary=WS, ws_high_threshold=hi, ws_low_threshold=lo,
        ws_size_threshold=10_000_000, ws_dust_threshold=200,
        boundary_flags=[1, 1, 1, 1, 1, 1], offset=0, channels=[2, 1, 0],
        workdir=a.workdir, keep_workdir=a.keep_workdir,
        ws_merge_thresholds=mts, ws_merge_function="max",
        edge_storage="source", on_batch_result=score)
    del aff
    gc.collect()
    if len(rows) != len(mts):
        raise SystemExit(f"scored {len(rows)} thresholds, expected {len(mts)}")

    best18 = min(rows, key=lambda r: r["grid18nm_voi"])
    best104 = min(rows, key=lambda r: r["grid104_voi"])
    print(f"[{a.label}] best 18nm-grid VOI: mt={best18['mt']:.2f} -> "
          f"{best18['grid18nm_voi']:.4f}", flush=True)
    print(f"[{a.label}] best 104-grid  VOI: mt={best104['mt']:.2f} -> "
          f"{best104['grid104_voi']:.4f}", flush=True)
    edge = {"grid18nm": best18["mt"] in (mts[0], mts[-1]),
            "grid104": best104["mt"] in (mts[0], mts[-1])}
    for k, v in edge.items():
        if v:
            print(f"[{a.label}] WARNING: {k} optimum sits at an END of the range -- "
                  f"widen it and re-run, keeping the incumbent inside the new grid.",
                  flush=True)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "card": "MSIDEPLOY-MODEL-002", "label": a.label,
            "affinity": str(a.affinity), "source_grid": a.source_grid,
            "ws_high": a.ws_high, "ws_low": a.ws_low, "merge_function": "max",
            "space": "compressed", "whole_volume": True, "gt": a.gt,
            "z_factor": a.factor, "n_planes_18nm": n_in, "n_planes_z600": n_out,
            "grid18nm_oracle_floor_voi": 0.6235,
            "grid104_oracle_floor_voi": 0.0,
            "baseline_18nm_grid_voi": 0.9130164471789348,
            "noise_floor_voi": 0.01,
            "optimum_at_range_edge": edge,
            "rows": rows, "best_grid18nm": best18, "best_grid104": best104}, indent=2))
        print(f"wrote {a.json}", flush=True)

    if not a.keep_workdir:
        for pat in ("*.raw", "seg_*.data"):
            for p in a.workdir.glob(pat):
                p.unlink(missing_ok=True)


main()
