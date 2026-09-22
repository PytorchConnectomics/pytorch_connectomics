"""Step 0 of the 600 nm gate: the oracle floor, and the label damage that comes with it.

Card MSIDEPLOY-MODEL-002, Stage 1 step 0. No GPU, no model.

WHY THIS RUNS FIRST. The gate scores the degraded segmentation by upsampling it
back to the 18 nm grid (145 -> 104 -> 145 planes). That round trip is itself
lossy: boundary positions come back quantised to the coarse Z grid. So a measured
"degraded VOI" is the sum of two things -- the model genuinely doing worse on
coarser input, which is what the gate is for, and a pure Z re-quantisation penalty
A PERFECT SEGMENTER WOULD ALSO PAY. This bounds the second exactly: push the
ground truth itself through the identical scoring path and score it against
itself. If that number is comparable to the ~0.01 VOI noise floor, the gate would
fire for a reason that has nothing to do with the model, and that is the finding.

Because the mode downsample degenerates to plane selection (see z600_zresample),
the whole round trip collapses to a per-plane index map, so the oracle is
`gt[idx]` with `idx = sel[up]`: 75 of the 145 planes come back unchanged and 70
are replaced by a neighbouring plane.

SECOND OUTPUT, and the reason this is worth more than one number: the same mode
downsample is a contained instance of the categorical label transform Stage 2
needs at full scale (card alternative A1). So this also reports how many GT
objects lose z-extent and how many vanish entirely, at 1.3889x rather than the
(3,2,2) route B would use.

    python z600_oracle_floor.py --json <out.json>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from z600_zresample import Z_FACTOR, zplan  # noqa: E402


def label_damage(gt_zarr, sel: np.ndarray, n_in: int, min_size: int) -> dict:
    """Per-object z-extent and volume, before and after the mode downsample.

    Streamed one z-plane at a time: the plane-selection degeneracy means "after"
    is just the extent restricted to the selected planes, so both numbers come
    from a single pass and nothing volume-sized is held.
    """
    kept = np.zeros(n_in, dtype=bool)
    kept[sel] = True
    vol_before, vol_after, presence = {}, {}, {}
    t0 = time.time()
    for z in range(n_in):
        ids, counts = np.unique(np.asarray(gt_zarr[z]), return_counts=True)
        nz = ids != 0
        ids, counts = ids[nz], counts[nz]
        for lab, cnt in zip(ids.tolist(), counts.tolist()):
            row = presence.get(lab)
            if row is None:
                row = presence[lab] = np.zeros(n_in, dtype=bool)
            row[z] = True
            vol_before[lab] = vol_before.get(lab, 0) + cnt
            if kept[z]:
                vol_after[lab] = vol_after.get(lab, 0) + cnt
        if z % 20 == 0:
            print(f"  label scan plane {z}/{n_in}  {time.time() - t0:.0f}s", flush=True)

    labels = np.fromiter(presence.keys(), dtype=np.int64, count=len(presence))
    eb = np.empty(labels.size, dtype=np.int64)
    ea = np.empty(labels.size, dtype=np.int64)
    # z-FRAGMENTATION, the statistic that actually bears on supervision damage.
    # "Loses z-extent" is close to a tautology once the mode downsample is a plane
    # selection -- any object touching one of the 41 dropped planes loses extent.
    # What damages a label is an object whose z-profile goes from one contiguous
    # run to several: that is a false split written into the training target.
    fb = np.empty(labels.size, dtype=np.int64)
    fa = np.empty(labels.size, dtype=np.int64)

    def nruns(mask):
        return int(np.count_nonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)) == 1))

    for i, lab in enumerate(labels.tolist()):
        row = presence[lab]
        after = row[sel]
        eb[i], ea[i] = int(row.sum()), int(after.sum())
        fb[i], fa[i] = nruns(row), nruns(after)
    vb = np.array([vol_before[int(l)] for l in labels], dtype=np.int64)
    va = np.array([vol_after.get(int(l), 0) for l in labels], dtype=np.int64)

    def summary(mask, tag):
        n = int(mask.sum())
        if n == 0:
            return {"tag": tag, "n_objects": 0}
        lost = ea[mask] < eb[mask]
        gone = ea[mask] == 0
        frag = (fa[mask] > fb[mask]) & ~gone
        return {
            "tag": tag,
            "n_objects": n,
            "n_z_fragmented": int(frag.sum()),
            "frac_z_fragmented": float(frag.mean()),
            "voxels_in_z_fragmented": int(vb[mask][frag].sum()),
            "n_contiguous_in_z_before": int((fb[mask] == 1).sum()),
            "n_lose_z_extent": int(lost.sum()),
            "frac_lose_z_extent": float(lost.mean()),
            "n_vanish": int(gone.sum()),
            "frac_vanish": float(gone.mean()),
            "n_single_plane_before": int((eb[mask] == 1).sum()),
            "voxels_before": int(vb[mask].sum()),
            "voxels_after": int(va[mask].sum()),
            "voxel_frac_retained": float(va[mask].sum() / max(1, vb[mask].sum())),
            "voxels_in_vanished_objects": int(vb[mask][gone].sum()),
            "mean_z_extent_before": float(eb[mask].mean()),
            "mean_z_extent_after": float(ea[mask].mean()),
            "median_z_extent_before": float(np.median(eb[mask])),
        }

    allm = np.ones(labels.size, dtype=bool)
    return {
        "n_gt_objects": int(labels.size),
        "min_size_threshold": min_size,
        "all_objects": summary(allm, "all"),
        f"objects_ge_{min_size}_voxels": summary(vb >= min_size, f">={min_size} vox"),
        "note": (
            "n_z_fragmented = objects whose z-profile gains contiguous runs under the "
            "downsample, i.e. a false split written into the label. This is the number "
            "that bears on supervision damage; n_lose_z_extent is near-tautological "
            "because any object touching one of the 41 dropped planes loses extent. "
            "z-extent = number of distinct z planes the object occupies. 'after' counts "
            "only the planes the weighted-mode downsample keeps (41 of 145 are dropped). "
            "This is the 1.3889x Z analogue of the route B (3,2,2) transform, NOT that "
            "transform: route B starts from mip0 and its label damage must be measured "
            "separately before any training launch."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--factor", type=float, default=Z_FACTOR)
    ap.add_argument("--min-size", type=int, default=1000,
                    help="second object-count cohort, so the headline fraction is not "
                         "dominated by single-voxel fragments")
    ap.add_argument("--skip-adapted-rand", action="store_true")
    ap.add_argument("--damage-only", action="store_true",
                    help="label-damage pass only; skip the volume-sized scoring")
    a = ap.parse_args()

    import zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi

    z = zarr.open(a.gt, mode="r")
    n_in = int(z.shape[0])
    n_out, _, sel, up = zplan(n_in, a.factor)
    idx = sel[up]
    print(f"GT {z.shape} {z.dtype}", flush=True)
    print(f"factor {a.factor} -> {n_in} -> {n_out} -> {n_in} planes; "
          f"{int((idx == np.arange(n_in)).sum())}/{n_in} planes unchanged by the round trip",
          flush=True)

    t0 = time.time()
    damage = label_damage(z, sel, n_in, a.min_size)
    print(f"label damage scan done in {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(damage["all_objects"], indent=2), flush=True)
    print(json.dumps(damage[f"objects_ge_{a.min_size}_voxels"], indent=2), flush=True)

    # Partial write first: the label-damage numbers are a Stage 2 precondition in
    # their own right and must survive an OOM in the scoring below.
    a.json.parent.mkdir(parents=True, exist_ok=True)
    out = {"card": "MSIDEPLOY-MODEL-002", "step": "0 oracle floor", "gt": a.gt,
           "z_factor": a.factor, "n_in": n_in, "n_out": n_out,
           "planes_unchanged_by_round_trip": int((idx == np.arange(n_in)).sum()),
           "label_damage": damage}
    a.json.write_text(json.dumps(out, indent=2))
    print(f"wrote partial {a.json}", flush=True)

    if a.damage_only:
        print("--damage-only: skipping the oracle scoring", flush=True)
        return

    t0 = time.time()
    gt = np.asarray(z[:])
    print(f"GT loaded {gt.nbytes / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
    oracle = gt[idx]
    print(f"oracle built {oracle.nbytes / 1e9:.1f} GB", flush=True)

    t0 = time.time()
    vs, vm = voi(oracle, gt)
    print(f"ORACLE FLOOR  VOI={vs + vm:.4f} (split {vs:.4f} / merge {vm:.4f})  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    out["oracle_voi_split"] = float(vs)
    out["oracle_voi_merge"] = float(vm)
    out["oracle_voi"] = float(vs + vm)
    a.json.write_text(json.dumps(out, indent=2))

    if not a.skip_adapted_rand:
        gc.collect()
        t0 = time.time()
        ar = adapted_rand(oracle, gt)
        print(f"ORACLE FLOOR  ARerr={ar:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        out["oracle_adapted_rand_error"] = float(ar)

    out["baseline_voi_18nm"] = 0.9130164471789348
    out["noise_floor_voi"] = 0.01
    out["interpretation"] = (
        "oracle_voi is the best VOI any segmenter can achieve through the "
        "145->104->145 scoring path. Read the degraded-val VOI against 0.9130 WITH "
        "this number stated alongside; the model-attributable share is what is left "
        "after it, and if oracle_voi is itself comparable to the ~0.01 noise floor "
        "the gate is not interpretable as a statement about the model.")
    a.json.write_text(json.dumps(out, indent=2))
    print(f"wrote {a.json}", flush=True)


# Guarded, unlike the sibling sweep scripts, because z600_routeb_label_damage.py
# imports `label_damage` from here rather than duplicating the scan.
if __name__ == "__main__":
    main()
