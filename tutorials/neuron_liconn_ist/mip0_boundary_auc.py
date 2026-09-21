#!/usr/bin/env python3
"""Does the mip0 affinity separate GT boundaries better than the mip1 affinity,
independently of the decoder?

WHY THIS EXISTS. A VOI comparison answers one question with two possible causes.
If mip0 ties mip1 on VOI, that is ambiguous between "the sharper input does not
help the model" and "the model sees more but neither the 18 nm label nor the
decoder can express it" (resolution-gate lesson §5). VOI alone cannot separate
them, and asserting one is not evidence.

This measures the first half directly. It asks only: given two voxels that are
ADJACENT ON THE 18 nm GRID, how well does each model's affinity predict whether
they belong to the same proofread object? That is a pure boundary-detection
question, scored on the grid where the GT is exact rather than upsampled, with no
watershed, no agglomeration and no merge threshold in the path.

    mip0 wins here, ties on VOI   -> the model does see more; the label/decoder
                                     is the binding constraint
    mip0 ties here, ties on VOI   -> the sharper input did not help the model
                                     at this label resolution
    mip0 loses here               -> something is wrong upstream, not a ceiling

HOW AN 18 nm EDGE IS BUILT FROM 9 nm AFFINITY. Both models store edge (i, i+1)
at voxel i (`edge_storage: source`), channel c = the edge along array axis c.
18 nm voxel (z,y,x) is mip0 voxel (2z,2y,2x). Travelling from (z,y,x) to its
+x neighbour crosses TWO mip0 edges, at 2x and 2x+1, so the 18 nm edge affinity
is their MIN -- a path is only as connected as its weakest link, which is also
exactly what the watershed's steepest-neighbour rule uses. Same along y and z.

WHAT THIS DOES NOT SHOW. The GT is still a proofread FFN segmentation quantized
to 18 nm, so an edge whose true boundary lies inside a 9 nm sub-voxel is
unlabelled here too. This tests boundary DETECTION at 18 nm, not boundary
LOCALIZATION at 9 nm -- nothing available on this volume can test the latter.
Edges with a background (id 0) endpoint are excluded: GT is 42.5% background and
those edges have no defined answer.

    python mip0_boundary_auc.py --mip1 <h5> --mip0 <h5> --planes 24
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"


def auc_mannwhitney(same: np.ndarray, diff: np.ndarray) -> float:
    """P(affinity of a same-parent edge > affinity of a different-parent edge).

    Rank-based, so it is invariant to the scale_sigmoid compression -- which is
    the point: the two models' affinities live in the same compressed space but
    need not share an operating point, and a threshold-free measure does not
    require them to.
    """
    n1, n2 = len(same), len(diff)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.concatenate([same, diff])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mip1", type=Path, required=True, help="18 nm affinity h5")
    ap.add_argument("--mip0", type=Path, required=True, help="9 nm affinity h5")
    ap.add_argument("--planes", type=int, default=24, help="18 nm z-planes to sample")
    ap.add_argument("--max-pairs-per-axis", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    import h5py, zarr
    rng = np.random.default_rng(a.seed)
    seg = zarr.open(GT, mode="r")
    Z, Y, X = seg.shape

    with h5py.File(a.mip1, "r") as f1, h5py.File(a.mip0, "r") as f0:
        d1, d0 = f1["main"], f0["main"]
        print(f"mip1 affinity {d1.shape} {d1.dtype}", flush=True)
        print(f"mip0 affinity {d0.shape} {d0.dtype}", flush=True)
        assert d0.shape[1] == 2 * d1.shape[1], "mip0 must be 2x mip1 on z"

        # Interior planes only: an edge needs z+1 in the GT and 2z+1 in mip0.
        zs = np.sort(rng.choice(np.arange(1, Z - 2), size=a.planes, replace=False))
        print(f"sampling 18 nm z-planes: {zs.tolist()}", flush=True)

        acc = {ax: {"m1_same": [], "m1_diff": [], "m0_same": [], "m0_diff": []}
               for ax in ("z", "y", "x")}

        for n, z in enumerate(zs):
            g0 = np.asarray(seg[z])                 # (Y, X)
            g1z = np.asarray(seg[z + 1])
            # mip1: channel c at (z,y,x) is the edge to +c neighbour
            a1 = np.asarray(d1[:, z]).astype(np.float32)          # (3, Y, X)
            # mip0: planes 2z and 2z+1
            a0a = np.asarray(d0[:, 2 * z]).astype(np.float32)     # (3, 2Y, 2X)
            a0b = np.asarray(d0[:, 2 * z + 1]).astype(np.float32)

            # --- X edges: (z,y,x)-(z,y,x+1) ---
            same = (g0[:, :-1] == g0[:, 1:]) & (g0[:, :-1] != 0) & (g0[:, 1:] != 0)
            valid = (g0[:, :-1] != 0) & (g0[:, 1:] != 0)
            m1 = a1[2, :, :-1]
            m0 = np.minimum(a0a[2, ::2, 0:2 * X - 2:2], a0a[2, ::2, 1:2 * X - 1:2])
            _push(acc["x"], m1, m0, same, valid)

            # --- Y edges: (z,y,x)-(z,y+1,x) ---
            same = (g0[:-1] == g0[1:]) & (g0[:-1] != 0) & (g0[1:] != 0)
            valid = (g0[:-1] != 0) & (g0[1:] != 0)
            m1 = a1[1, :-1, :]
            m0 = np.minimum(a0a[1, 0:2 * Y - 2:2, ::2], a0a[1, 1:2 * Y - 1:2, ::2])
            _push(acc["y"], m1, m0, same, valid)

            # --- Z edges: (z,y,x)-(z+1,y,x) ---
            same = (g0 == g1z) & (g0 != 0) & (g1z != 0)
            valid = (g0 != 0) & (g1z != 0)
            m1 = a1[0]
            m0 = np.minimum(a0a[0, ::2, ::2], a0b[0, ::2, ::2])
            _push(acc["z"], m1, m0, same, valid)

            del g0, g1z, a1, a0a, a0b
            print(f"  plane {n+1}/{len(zs)} (z={z}) done", flush=True)

    out = {"mip1": str(a.mip1), "mip0": str(a.mip0), "planes": zs.tolist(), "axes": {}}
    print(f"\n{'axis':<6}{'n_same':>12}{'n_diff':>12}{'mip1 AUC':>11}{'mip0 AUC':>11}"
          f"{'delta':>9}", flush=True)
    for ax in ("z", "y", "x"):
        s1 = _cap(np.concatenate(acc[ax]["m1_same"]), a.max_pairs_per_axis, rng)
        d1_ = _cap(np.concatenate(acc[ax]["m1_diff"]), a.max_pairs_per_axis, rng)
        s0 = _cap(np.concatenate(acc[ax]["m0_same"]), a.max_pairs_per_axis, rng)
        d0_ = _cap(np.concatenate(acc[ax]["m0_diff"]), a.max_pairs_per_axis, rng)
        auc1, auc0 = auc_mannwhitney(s1, d1_), auc_mannwhitney(s0, d0_)
        out["axes"][ax] = {
            "n_same": int(len(s1)), "n_diff": int(len(d1_)),
            "mip1_auc": auc1, "mip0_auc": auc0, "delta": auc0 - auc1,
            "mip1_median_same": float(np.median(s1)), "mip1_median_diff": float(np.median(d1_)),
            "mip0_median_same": float(np.median(s0)), "mip0_median_diff": float(np.median(d0_)),
        }
        print(f"{ax:<6}{len(s1):>12,}{len(d1_):>12,}{auc1:>11.4f}{auc0:>11.4f}"
              f"{auc0-auc1:>+9.4f}", flush=True)

    print("\nAUC = P(a same-parent 18 nm edge scores higher than a different-parent one).",
          flush=True)
    print("Rank-based, so the scale_sigmoid compression and any per-model offset "
          "cancel; only ordering matters.", flush=True)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}", flush=True)


def _push(store, m1, m0, same, valid):
    assert m1.shape == m0.shape == same.shape, (m1.shape, m0.shape, same.shape)
    vs, vd = valid & same, valid & ~same
    store["m1_same"].append(m1[vs]); store["m1_diff"].append(m1[vd])
    store["m0_same"].append(m0[vs]); store["m0_diff"].append(m0[vd])


def _cap(arr, n, rng):
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), size=n, replace=False)]


main()
