#!/usr/bin/env python3
"""Report the volume's caliber distribution so the gates can be set from it.

The `ContinuityConfig` caliber gates default to EM-scale conventions
(0.35 / 0.50 um). On a fine-caliber expansion volume those can sit entirely
outside the data, in which case caliber stops discriminating and the semantic
type is decided by whatever is left. This reads only the skeleton bundle -- no
segmentation, no morphology pass -- so the gates can be chosen before paying for
the full analysis.

Length-weighted median radius per segment, the same statistic
`analyze_continuity` falls back to, without the arbor pruning: this is a survey
for choosing thresholds, not the measurement of record.

    python dev/liconn_moe/caliber_survey.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeletons", type=Path, default=V.SKELETONS)
    parser.add_argument("--min-length-um", type=float, default=0.0,
                        help="restrict to segments with at least this skeleton length")
    args = parser.parse_args()

    with np.load(args.skeletons, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        voxels = np.asarray(data["voxel_counts"], dtype=np.int64)
        v_off = np.asarray(data["vertex_offset"], dtype=np.int64)
        e_off = np.asarray(data["edge_offset"], dtype=np.int64)
        vertices = np.asarray(data["vertices_um_zyx"], dtype=np.float64)
        radii = np.asarray(data["radii_um"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.int64)

    caliber = np.full(len(labels), np.nan)
    lengths = np.zeros(len(labels))
    for index in range(len(labels)):
        v0, v1 = v_off[index], v_off[index + 1]
        e0, e1 = e_off[index], e_off[index + 1]
        block, radius = vertices[v0:v1], radii[v0:v1]
        link = edges[e0:e1]
        if not len(link):
            continue
        span = np.linalg.norm(block[link[:, 0]] - block[link[:, 1]], axis=1)
        lengths[index] = span.sum()
        weight = np.zeros(len(block))
        np.add.at(weight, link[:, 0], span / 2)
        np.add.at(weight, link[:, 1], span / 2)
        used = weight > 0
        if not np.any(used):
            continue
        order = np.argsort(radius[used], kind="stable")
        cumulative = np.cumsum(weight[used][order])
        pick = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
        caliber[index] = radius[used][order[min(pick, len(order) - 1)]]

    keep = np.isfinite(caliber) & (lengths >= args.min_length_um)
    values, size, span = caliber[keep], voxels[keep], lengths[keep]
    print(f"{keep.sum()} of {len(labels)} segments with a usable radius"
          f"{f' and length >= {args.min_length_um} um' if args.min_length_um else ''}\n")
    print("caliber radius (um)")
    for q in PERCENTILES:
        print(f"  p{q:<5} {np.percentile(values, q):.4f}")
    print(f"  max    {values.max():.4f}")

    print("\nfraction above candidate dendrite gates")
    for gate in (0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50):
        print(f"  >= {gate:.2f} um : {np.mean(values >= gate) * 100:6.2f}%  "
              f"({int(np.sum(values >= gate)):6d} segments, "
              f"{np.sum(size[values >= gate]) / np.sum(size) * 100:5.1f}% of cohort voxels)")

    print("\ncaliber by skeleton length decile (long segments are the reliable ones)")
    edges_q = np.percentile(span, np.arange(0, 101, 10))
    for i in range(10):
        m = (span >= edges_q[i]) & (span <= edges_q[i + 1])
        if m.sum():
            print(f"  length {edges_q[i]:7.2f}-{edges_q[i+1]:7.2f} um  n={m.sum():6d}  "
                  f"caliber p50={np.median(values[m]):.4f}  p90={np.percentile(values[m], 90):.4f}")

    big = np.argsort(size)[-10:][::-1]
    print("\nten largest segments")
    for i in big:
        print(f"  label {labels[keep][i]:>7d}  vox {size[i]:9d}  "
              f"caliber {values[i]:.4f}  length {span[i]:8.2f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
