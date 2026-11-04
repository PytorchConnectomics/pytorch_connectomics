#!/usr/bin/env python3
"""Convert EMNeuron (SegNeuron, MICCAI 2024) into PyTC training chunks.

Input is the HuggingFace dataset ``yanchaoz/EMNeuron``: ``labeled/<source>/<i>.tif`` with
``<i>_MaskIns.tif`` instance labels, and ``valid/{InDistribution,OutofDistribution}/<name>/``
with ``raw.tif`` and ``label.tif``. The output reproduces the training-data semantics of
SegNeuron's ``Train_and_Inference/supervised_provider.py``:

* ignore regions become ``-1``, which PyTC's affinity target masks out of the loss:
  label 10000 in J0126-sbem and Mira-scn, labels 121 and 122 in Kasthuri-atum volume 0;
* SegNeuron's multi-z-ratio augmentation becomes precomputed copies: Hemi-brain-fib and
  Fib-25-fib also at z-stride 2 and 3 (SegNeuron also strides J0126 by 1-2; not done here);
* SegNeuron samples a source uniformly, then a volume uniformly. PyTC's cached dataset
  samples a volume uniformly, so each source volume is tiled into the same number of
  chunks within its source, and the chunk counts per source are kept near-equal (24-36).

Layout written under ``--out`` (HDF5 dataset ``main``; image uint8, label int32):

* ``train/<name>_img.h5`` / ``<name>_seg.h5`` -- 308 training chunks;
* ``tune/`` -- the last chunk of the last volume of every source, at z-stride 1 (its
  stride-2/3 copies are dropped). Validation loss and decoding choices use only these;
* ``val/<name>_img.h5`` / ``<name>_seg.h5`` -- the 8 evaluation volumes of the paper,
  for inference (score against the original ``valid/`` labels);
* ``manifest.json`` -- chunk provenance, shapes and per-source sampling shares.
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import tifffile

# source dir -> (short name, z-strides, per-volume tiling (nz, ny, nx))
PLAN = {
    "J0126-sbem": ("j0126", [1], (1, 1, 1)),
    "Kasthuri-atum": ("kasthuri", [1], (1, 4, 4)),
    "Hemi-brain-fib": ("hemibrain", [1, 2, 3], (1, 1, 2)),
    "CREMI-sstem": ("cremi", [1], (2, 2, 2)),
    "AxonEM[M]-sstem": ("axonem_m", [1], (1, 2, 2)),
    "AxonEM[H]-atum": ("axonem_h", [1], (1, 2, 2)),
    "Mira-adwt": ("mira_adwt", [1], (1, 3, 3)),
    "Fib-25-fib": ("fib25", [1, 2, 3], (3, 2, 1)),
    "Mira-scn": ("mira_scn", [1], (1, 2, 3)),
    "Mira-fish": ("mira_fish", [1], (1, 2, 4)),
}

VAL = {
    "FAFB-1": "InDistribution/FAFB-1",
    "FAFB-2": "InDistribution/FAFB-2",
    "FIB25-1": "InDistribution/FIB25-1",
    "FIB25-2": "InDistribution/FIB25-2",
    "Kstr": "InDistribution/Kstr",
    "Harris": "OutofDistribution/Harris",
    "Ionsem": "OutofDistribution/Ionsem",
    "MicronsB": "OutofDistribution/Microns[B]",
}


def ignore_ids(source: str, vol: int) -> list[int]:
    if source in ("J0126-sbem", "Mira-scn"):
        return [10000]
    if source == "Kasthuri-atum" and vol == 0:
        return [121, 122]
    return []


def to_int32_labels(lab: np.ndarray, ignore: list[int]) -> np.ndarray:
    if lab.dtype.kind == "f" and not np.all(lab == np.round(lab)):
        raise ValueError("non-integer float labels")
    if lab.min() < 0 or lab.max() > np.iinfo(np.int32).max:
        raise ValueError(f"label range {lab.min()}..{lab.max()} does not fit int32 >= 0")
    out = lab.astype(np.int32)
    for i in ignore:
        out[out == i] = -1
    return out


def bounds(n: int, parts: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def write_h5(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "main",
            data=arr,
            compression="gzip",
            compression_opts=1,
            chunks=tuple(min(s, 64) for s in arr.shape),
        )
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="EMNeuron root holding labeled/ and valid/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-shape", type=int, nargs=3, default=[32, 128, 128])
    args = ap.parse_args()
    for split in ("train", "tune", "val"):
        os.makedirs(f"{args.out}/{split}", exist_ok=True)
    plan = {k: [v[0], v[1], list(v[2])] for k, v in PLAN.items()}
    manifest: dict = {"train": [], "tune": [], "val": [], "plan": plan}

    for source, (short, ratios, tiling) in PLAN.items():
        src_dir = f"{args.src}/labeled/{source}"
        n_vol = len([f for f in os.listdir(src_dir) if f.endswith("_MaskIns.tif")])
        holdout = (n_vol - 1, int(np.prod(tiling)) - 1)  # (volume, chunk), every z-stride
        for vol in range(n_vol):
            raw = tifffile.imread(f"{src_dir}/{vol}.tif")
            lab = tifffile.imread(f"{src_dir}/{vol}_MaskIns.tif")
            lab = to_int32_labels(lab, ignore_ids(source, vol))
            if raw.dtype != np.uint8 or raw.shape != lab.shape:
                raise ValueError(f"{source}/{vol}: raw {raw.dtype}{raw.shape} vs label {lab.shape}")
            for r in ratios:
                rr, ll = raw[::r], lab[::r]
                for iz, (z0, z1) in enumerate(bounds(rr.shape[0], tiling[0])):
                    for iy, (y0, y1) in enumerate(bounds(rr.shape[1], tiling[1])):
                        for ix, (x0, x1) in enumerate(bounds(rr.shape[2], tiling[2])):
                            k = (iz * tiling[1] + iy) * tiling[2] + ix
                            name = f"{short}_v{vol:02d}_z{r}_c{k:02d}"
                            split = "train"
                            if (vol, k) == holdout:
                                if r != 1:
                                    continue
                                split = "tune"
                            img_c = np.ascontiguousarray(rr[z0:z1, y0:y1, x0:x1])
                            seg_c = np.ascontiguousarray(ll[z0:z1, y0:y1, x0:x1])
                            write_h5(f"{args.out}/{split}/{name}_img.h5", img_c)
                            write_h5(f"{args.out}/{split}/{name}_seg.h5", seg_c)
                            small = any(s < m for s, m in zip(img_c.shape, args.min_shape))
                            manifest[split].append(
                                {
                                    "name": name,
                                    "source": source,
                                    "volume": vol,
                                    "z_ratio": r,
                                    "box": [
                                        int(z0) * r,
                                        int(z1) * r,
                                        int(y0),
                                        int(y1),
                                        int(x0),
                                        int(x1),
                                    ],
                                    "shape": list(img_c.shape),
                                    "ignore_frac": round(float((seg_c < 0).mean()), 4),
                                    "bg_frac": round(float((seg_c == 0).mean()), 4),
                                    "below_min_shape": small,
                                }
                            )
            print(source, vol, flush=True)

    for name, rel in VAL.items():
        raw = tifffile.imread(f"{args.src}/valid/{rel}/raw.tif")
        lab = to_int32_labels(tifffile.imread(f"{args.src}/valid/{rel}/label.tif"), [])
        write_h5(f"{args.out}/val/{name}_img.h5", raw)
        write_h5(f"{args.out}/val/{name}_seg.h5", lab)
        manifest["val"].append({"name": name, "src": rel, "shape": list(raw.shape)})
        print("val", name, flush=True)

    counts: dict[str, int] = {}
    for t in manifest["train"]:
        counts[t["source"]] = counts.get(t["source"], 0) + 1
    total = sum(counts.values())
    manifest["chunks_per_source"] = counts
    manifest["sampling_share"] = {k: round(v / total, 4) for k, v in counts.items()}
    with open(f"{args.out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["sampling_share"], indent=1))
    print("below min shape:", [t["name"] for t in manifest["train"] if t["below_min_shape"]])
    print("tune:", [(t["name"], t["shape"]) for t in manifest["tune"]])


if __name__ == "__main__":
    main()
