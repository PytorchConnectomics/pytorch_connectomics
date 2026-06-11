"""Precompute a WHOLE-VOLUME skeleton-aware SDT from the hole-filled GT (NISB).

For label_aux_type=sdt: computes the SDT on the ENTIRE volume (global distance field), from the
void-filled seg (seg_filled), un-eroded. Global (not per-crop) so the model learns a globally
consistent distance -- it must infer the neuron's unseen extent -> patches stitch consistently.
Writes data.zarr/seg_filled_sdt (the sdt_path_for_label cache the trainer loads).

Run from repo root (env: pytc). One seed per task (SLURM array):
    python scripts/sdt_precompute.py --seed 0 --resolution 9 9 20 --alpha 0.8
"""
from __future__ import annotations

import argparse

TRAIN = "/projects/weilab/dataset/nisb/base/train"


def main() -> None:
    from connectomics.data.processing.distance import precompute_sdt_volume, sdt_path_for_label

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label-key", default="seg_filled")
    ap.add_argument("--resolution", type=float, nargs=3, default=[9, 9, 20], help="array-axis order")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--bg-value", type=float, default=-1.0)
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    zpath = args.path or f"{TRAIN}/seed{args.seed}/data.zarr"
    label_path = f"{zpath}/{args.label_key}"
    out = sdt_path_for_label(label_path, mode="sdt")
    print(f"SDT: {label_path} -> {out}  (res={tuple(args.resolution)}, alpha={args.alpha})",
          flush=True)
    precompute_sdt_volume(label_path, out, resolution=tuple(args.resolution),
                          alpha=args.alpha, bg_value=args.bg_value)
    print("done", flush=True)


if __name__ == "__main__":
    main()
