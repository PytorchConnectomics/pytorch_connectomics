#!/usr/bin/env python3
"""Turn `error_analysis.json` into the four panels worth arguing with.

The caliber gates and the face margin are asserted thresholds. These plots show
where they actually fall on this volume's distributions, so they can be moved on
evidence rather than left at their defaults.

    python dev/liconn_moe/plot_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

TYPE_COLOR = {
    "axon_like": "#228833",
    "dendrite_like": "#4477aa",
    "ambiguous_caliber": "#ee7733",
    "unmeasured": "#bbbbbb",
}
COMPLETENESS_ORDER = (
    "complete",
    "broken_one_end",
    "broken_multi_end",
    "isolated_fragment",
    "unmeasured",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=V.ANALYSIS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text())
    segments = payload["segments"]
    config = payload["metadata"]["continuity_config"]
    output = args.output or args.analysis.parent / "error_analysis.png"

    radii = np.array(
        [s["continuity"]["caliber_radius_um"] for s in segments
         if s["continuity"].get("caliber_radius_um") is not None]
    )
    types = [s["semantic_type"] for s in segments]
    classes = [s["completeness_class"] for s in segments]
    gaps = np.array([e["distance_to_face_um"] for s in segments for e in s["free_ends"]])
    lengths = {
        key: np.array(
            [s["continuity"]["retained_length_um"] for s in segments
             if s["semantic_type"] == key and s["continuity"].get("retained_length_um")]
        )
        for key in TYPE_COLOR
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    ax = axes[0, 0]
    ax.hist(radii, bins=np.linspace(0, 1.2, 120), color="#555555")
    ax.axvline(config["axon_max_radius_um"], color="#228833", lw=2,
               label=f"axon <= {config['axon_max_radius_um']}")
    ax.axvline(config["dendrite_min_radius_um"], color="#4477aa", lw=2,
               label=f"dendrite >= {config['dendrite_min_radius_um']}")
    ax.set_xlabel("skeleton caliber radius (um)")
    ax.set_ylabel("segments")
    ax.set_title("Caliber vs the gates that decide semantic type\n"
                 "gates far out in a single mode = the cut is arbitrary", fontsize=10)
    ax.legend()

    ax = axes[0, 1]
    order = [key for key in TYPE_COLOR]
    bottom = np.zeros(len(order))
    for completeness in COMPLETENESS_ORDER:
        values = np.array(
            [sum(1 for t, c in zip(types, classes) if t == key and c == completeness)
             for key in order],
            dtype=float,
        )
        ax.bar(order, values, bottom=bottom, label=completeness)
        bottom += values
    ax.set_ylabel("segments")
    ax.set_title("Completeness within each semantic type\n"
                 "a free end is a candidate false split, not a proven one", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    if len(gaps):
        ax.hist(gaps, bins=np.linspace(0, float(np.percentile(gaps, 99)), 100),
                color="#cc3311")
    ax.set_xlabel("free end distance to the nearest crop face (um)")
    ax.set_ylabel("free ends")
    ax.set_title(f"{len(gaps)} free ends\n"
                 "a pile-up at zero would mean the face margin is too tight", fontsize=10)

    ax = axes[1, 1]
    for key, values in lengths.items():
        if len(values):
            ax.hist(values, bins=np.logspace(-1.5, 2, 60), histtype="step", lw=2,
                    color=TYPE_COLOR[key], label=f"{key} (n={len(values)})")
    ax.set_xscale("log")
    ax.set_xlabel("retained skeleton length (um)")
    ax.set_ylabel("segments")
    ax.set_title("Skeleton length by semantic type\n"
                 "a hard edge in dendrite_like is min_branch_length_um, not biology",
                 fontsize=10)
    ax.legend(fontsize=8)

    summary = payload["summary"]
    fig.suptitle(
        f"{payload['metadata']['volume']}  eb2 ABISS mt060  "
        f"{len(segments)} segments >= {V.MIN_VOXELS} voxels   |   "
        f"tube blind spot {summary['tube_blind_spot_count']}  "
        f"(axon-calibre and broken, invisible to the profile rule)   |   NO GROUND TRUTH",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=110)
    print(f"wrote {output}")

    print("\ncross-tab, largest cells first (profile class | semantic | completeness):")
    for row in summary["cross_tab"][:20]:
        print(f"  {row['count']:7d}  {row['morphology_class']:34s} "
              f"{row['semantic_type']:18s} {row['completeness_class']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
