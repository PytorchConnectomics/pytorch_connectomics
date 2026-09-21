"""Sweep the affinity-CC decode threshold on cached Pinky test predictions.

Inference is the expensive part and is already cached per volume as
``raw_x1_head-aff_ch0-1-2.h5``, so a threshold sweep is decode-only. For every
threshold we report the decoded score and the false-merge-free oracle ceiling,
which separates "the decode is mis-tuned" from "the affinities cannot support a
better segmentation".

Thresholds are in the saved ``scale_sigmoid`` (temperature 0.2) space, i.e.
plain-sigmoid p = sigmoid(logit(t) / 0.2).

Usage (cwd or PYTHONPATH must include the repo root):
    python scripts/pinky_decode_sweep.py --results-dir <test_step=NNN dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from connectomics.decoding.decoders.segmentation import decode_affinity_cc
from connectomics.metrics.oracle import oracle_merge_segmentation
from pinky_test_ceiling import GT_DIR, read_h5, score

THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", default=GT_DIR, type=Path)
    parser.add_argument("--thresholds", default=None, type=str,
                        help="comma-separated list; default 0.5,0.6,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--skip", default="", type=str,
                        help="comma-separated volume stems to skip (e.g. the 134M-voxel vol501)")
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    thresholds = (
        tuple(float(v) for v in args.thresholds.split(",")) if args.thresholds else THRESHOLDS
    )
    skip = {s for s in args.skip.split(",") if s}

    volume_dirs = sorted(
        p for p in args.results_dir.iterdir()
        if p.is_dir() and (p / "raw_x1_head-aff_ch0-1-2.h5").exists() and p.name not in skip
    )
    if not volume_dirs:
        raise SystemExit(f"no cached raw predictions under {args.results_dir}")

    print("threshold in saved scale_sigmoid space; plain-sigmoid equivalent in parentheses")
    for t in thresholds:
        plain = 1.0 / (1.0 + math.exp(-math.log(t / (1 - t)) / 0.2))
        print(f"  {t:.2f} -> plain {plain:.6f}")

    rows = []
    for volume_dir in volume_dirs:
        aff = read_h5(volume_dir / "raw_x1_head-aff_ch0-1-2.h5").astype(np.float32)
        stem = volume_dir.name[: -len("_image")] if volume_dir.name.endswith("_image") else volume_dir.name
        gt = read_h5(args.gt_dir / f"{stem}_label.h5")
        print(f"\n=== {volume_dir.name}  aff {aff.shape}  gt ids {int((np.unique(gt) != 0).sum())} ===",
              flush=True)
        print(f"{'thr':>6}{'frag':>9}{'ARE':>9}{'VOI':>9}{'VOIs':>9}{'VOIm':>9}"
              f"{'oARE':>9}{'oVOI':>9}{'oVOIs':>9}{'oVOIm':>9}{'sec':>7}")
        for t in thresholds:
            start = time.time()
            seg = decode_affinity_cc(aff, threshold=t, backend="numba", edge_offset=0)
            base = score(seg, gt)
            oracle = score(oracle_merge_segmentation(seg, gt), gt)
            elapsed = time.time() - start
            rows.append({
                "volume": volume_dir.name, "threshold": t, "voxels": int(gt.size),
                "base": base, "oracle_merge": oracle,
            })
            print(f"{t:>6.2f}{base['n_pred']:>9}{base['are']:>9.4f}{base['voi']:>9.4f}"
                  f"{base['voi_split']:>9.4f}{base['voi_merge']:>9.4f}"
                  f"{oracle['are']:>9.4f}{oracle['voi']:>9.4f}"
                  f"{oracle['voi_split']:>9.4f}{oracle['voi_merge']:>9.4f}{elapsed:>7.1f}",
                  flush=True)
        del aff, gt

    print("\n" + "=" * 78)
    print("voxel-weighted means across volumes")
    print(f"{'thr':>6}{'ARE':>9}{'VOI':>9}{'VOIs':>9}{'VOIm':>9}{'oARE':>9}{'oVOI':>9}")
    summary = {}
    for t in thresholds:
        subset = [r for r in rows if r["threshold"] == t]
        weights = np.array([r["voxels"] for r in subset], dtype=np.float64)
        weights /= weights.sum()

        def wavg(kind: str, key: str) -> float:
            return float((np.array([r[kind][key] for r in subset]) * weights).sum())

        summary[f"{t:.2f}"] = {
            "are": wavg("base", "are"), "voi": wavg("base", "voi"),
            "voi_split": wavg("base", "voi_split"), "voi_merge": wavg("base", "voi_merge"),
            "oracle_are": wavg("oracle_merge", "are"), "oracle_voi": wavg("oracle_merge", "voi"),
        }
        s = summary[f"{t:.2f}"]
        print(f"{t:>6.2f}{s['are']:>9.4f}{s['voi']:>9.4f}{s['voi_split']:>9.4f}"
              f"{s['voi_merge']:>9.4f}{s['oracle_are']:>9.4f}{s['oracle_voi']:>9.4f}")

    out = args.out or args.results_dir / "decode_threshold_sweep.json"
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
