#!/usr/bin/env python3
"""Measure the ABISS `ws` binary's OWN peak RSS as a function of volume size.

Why this exists. The only recorded decode number for this project is "147 GB peak
at 2.08 Gvox", which is the peak of the whole *python* job -- and that job holds,
at 2.08 Gvox: the affinity as float32 (25 GB), a transposed float32 copy (25 GB),
and one uint64 segmentation per merge threshold (16.6 GB each, all retained in a
dict before `_run_abiss_ws` returns). Scaling that composite 8x to the mip0 val
volume gives ~1.2 TB and the conclusion "h200 or nothing".

But `ws` mmaps its affinity (src/ws/atomic_chunk.cpp: `MMArray<aff_t,4>`), so the
affinity is page cache, not anonymous memory, and in batch mode it frees each
threshold's segmentation right after writing it. The python wrapper's copies are
avoidable; `ws`'s own working set is not. Those are different numbers and only the
second one constrains which node this can run on.

This measures the second one. Crops of the existing 18 nm val affinity, each
written through run_abiss_volume's own helpers (so the layout, the [2,1,0] channel
reversal and the edge_storage=source shift are the real ones), then `ws` run under
/usr/bin/time -v.

    python ws_sizing.py --sizes 64x512x512,96x768x768,128x1024x1024

Reports voxels, supervoxel count, region-graph size, ws peak RSS and wall time.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
WS = REPO / "lib/abiss/build/ws"
EB2_AFF = REPO / "outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
sys.path.insert(0, str(REPO))


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, default=EB2_AFF)
    ap.add_argument("--sizes", default="48x384x384,64x512x512,96x768x768,128x1024x1024",
                    help="comma-separated ZxYxX crop sizes, taken from the volume centre")
    ap.add_argument("--thresholds", default="0.47")
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="20%")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    import h5py
    rav = _load_runner()
    mts = [float(v) for v in a.thresholds.split(",")]
    a.workdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in a.sizes.split(","):
        dz, dy, dx = (int(v) for v in spec.split("x"))
        with h5py.File(a.affinity, "r") as f:
            d = f["main"]
            _, Z, Y, X = d.shape
            z0, y0, x0 = (Z - dz) // 2, (Y - dy) // 2, (X - dx) // 2
            aff = np.asarray(d[:, z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]).astype(np.float32)
        nvox = int(dz) * int(dy) * int(dx)

        hi = rav._resolve_threshold(a.ws_high, aff, "ws_high")
        lo = rav._resolve_threshold(a.ws_low, aff, "ws_low")

        wd = a.workdir / f"ws_{dz}x{dy}x{dx}"
        wd.mkdir(parents=True, exist_ok=True)
        aff_xyzc = rav._to_abiss_affinity(aff, channels=[2, 1, 0], edge_storage="source")
        del aff
        ws_xyz = rav._write_affinity_with_halo(wd / "aff.raw", aff_xyzc, halo=1)
        del aff_xyzc
        rav._write_abiss_param_file(wd / "param.txt", ws_xyz, [1, 1, 1, 1, 1, 1], 0)

        cmd = ["/usr/bin/time", "-v", str(WS), str(wd / "param.txt"), str(wd / "aff.raw"),
               str(hi), str(lo), str(10_000_000), str(200), "probe", "max"] + [str(m) for m in mts]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            print(f"[{spec}] ws FAILED rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
            continue

        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
        peak_gb = int(m.group(1)) / 1e6 if m else float("nan")
        sv = re.search(r"num of sv:(\d+)", p.stdout)
        rg = re.search(r"size of rg:(\d+)", p.stdout)
        svn = int(sv.group(1)) if sv else None
        if svn is None:
            mm = re.search(r"sv=(\d+) rg=(\d+)", p.stdout)
            svn = int(mm.group(1)) if mm else None
            rgn = int(mm.group(2)) if mm else None
        else:
            rgn = int(rg.group(1)) if rg else None

        row = {"crop_zyx": [dz, dy, dx], "voxels": nvox, "ws_high": hi, "ws_low": lo,
               "peak_rss_gb": peak_gb, "wall_s": wall, "n_sv": svn, "rg_size": rgn,
               "bytes_per_voxel": peak_gb * 1e9 / nvox}
        rows.append(row)
        print(f"[{spec}] {nvox/1e9:.4f} Gvox  ws peak {peak_gb:7.2f} GB  "
              f"({row['bytes_per_voxel']:5.1f} B/vox)  sv={svn}  rg={rgn}  {wall:.0f}s", flush=True)

        for leftover in wd.glob("*"):
            leftover.unlink()
        wd.rmdir()

    if len(rows) >= 2:
        x = np.array([r["voxels"] for r in rows], dtype=np.float64)
        y = np.array([r["peak_rss_gb"] for r in rows], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        target = 290 * 8580 * 6690
        print(f"\nlinear fit: peak_GB = {slope*1e9:.2f} * Gvox + {intercept:.2f}", flush=True)
        print(f"extrapolated to mip0 val ({target/1e9:.2f} Gvox): "
              f"{slope*target + intercept:.0f} GB", flush=True)
        print("NOTE: this extrapolates mip1 supervoxel DENSITY. mip0 has 8x the voxels "
              "over the same anatomy, so its sv-count term should grow slower than "
              "linearly and the per-voxel seg array term exactly linearly.", flush=True)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"affinity": str(a.affinity), "thresholds": mts,
                                      "rows": rows}, indent=2))
        print(f"wrote {a.json}", flush=True)


main()
