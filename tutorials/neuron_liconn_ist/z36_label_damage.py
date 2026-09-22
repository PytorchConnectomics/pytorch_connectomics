"""How much GT survives the [36,18,18] decimation? Card MSIDEPLOY-MODEL-002 Stage 2
precondition (measured and reported BEFORE the training launch).

Z36 offset o keeps 24 nm seg plane (3j + o) // 2 for every mip0 plane 3j + o (see
z36_build.py), and XY is exact, so the damage is purely which 24 nm planes are
kept. Computed from final_proofread/*/seg (24 nm) per plane, no z36 volume needed.

Per offset, reports: objects that VANISH (no kept plane), objects whose z-extent
SHRINKS (first or last plane dropped), and the voxel fraction kept.

    python z36_label_damage.py --json outputs/neuron_liconn_ist/z36/label_damage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fastremap
import numpy as np
import zarr

REF = "/projects/weilab/dataset/liconn/pytc/final_proofread"
MIP0_Z = {"train": 540, "val": 290}
OFFSETS = {"train": (0, 1, 2), "val": (0, 1, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    report = {}
    for split in ("train", "val"):
        seg = zarr.open_group(f"{REF}/{split}/data.zarr", mode="r")["seg"]
        nk = seg.shape[0]
        assert nk * 2 == MIP0_Z[split], (nk, MIP0_Z[split])
        ids_l, k_l, c_l = [], [], []
        for k in range(nk):
            u, c = fastremap.unique(np.asarray(seg[k]), return_counts=True)
            m = u != 0
            ids_l.append(u[m]); c_l.append(c[m]); k_l.append(np.full(m.sum(), k, np.int32))
            if k % 30 == 0:
                print(f"  {split} plane {k}/{nk}", flush=True)
        ids, ks, cs = np.concatenate(ids_l), np.concatenate(k_l), np.concatenate(c_l)
        uid, inv = np.unique(ids, return_inverse=True)
        zmin = np.full(uid.size, nk, np.int32); zmax = np.full(uid.size, -1, np.int32)
        np.minimum.at(zmin, inv, ks); np.maximum.at(zmax, inv, ks)
        vox = np.bincount(inv, weights=cs)
        rep = {"n_objects": int(uid.size), "n_planes_24nm": nk}
        for o in OFFSETS[split]:
            keep = np.zeros(nk, bool)
            keep[[z // 2 for z in range(o, MIP0_Z[split], 3)]] = True
            sel = keep[ks]
            kmin = np.full(uid.size, nk, np.int32); kmax = np.full(uid.size, -1, np.int32)
            np.minimum.at(kmin, inv[sel], ks[sel]); np.maximum.at(kmax, inv[sel], ks[sel])
            present = kmax >= 0
            shrink = present & ((kmin != zmin) | (kmax != zmax))
            vox_kept = np.bincount(inv[sel], weights=cs[sel], minlength=uid.size)
            single = (zmax == zmin)
            rep[f"offset{o}"] = {
                "planes_kept": int(keep.sum()),
                "vanished": int((~present).sum()),
                "vanished_frac": float((~present).mean()),
                "vanished_voxel_frac_of_total": float(vox[~present].sum() / vox.sum()),
                "vanished_max_voxels_24nm": int(vox[~present].max()) if (~present).any() else 0,
                "vanished_of_single_plane_objects": int((~present & single).sum()),
                "extent_shrunk": int(shrink.sum()),
                "extent_shrunk_frac": float(shrink.mean()),
                "voxel_frac_kept": float(vox_kept.sum() / vox.sum()),
            }
            print(split, o, rep[f"offset{o}"], flush=True)
        rep["single_plane_objects"] = int(single.sum())
        rep["objects_ge_1000_vox"] = int((vox >= 1000).sum())
        report[split] = rep
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
