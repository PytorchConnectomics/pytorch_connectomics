#!/usr/bin/env python3
"""Flag each skeleton free end as spur, border, or axon terminal -> end_evidence.json.

Reads the volume's segmentation, `skeletons_fine.npz` and `error_analysis.json`
and writes, per free end (keyed by segment id and skeleton vertex index):

spur      terminal branch (tip -> first junction) shorter than SPUR_FACTOR x the
          junction radius: surface roughness, not a process end.
at_border the label's own voxels on a crop face lie within BORDER_UM of the tip.
terminal  largest inscribed radius within BALL_UM of the tip, over the segment
          caliber, is at or above this volume's own mid-shaft control at the
          CONTROL_PERCENTILE -- swollen beyond what normal shaft reads.

The terminal cut is calibrated per volume from mid-shaft points (degree-2
vertices >= 0.5 um from any tip or junction) measured the same way, because a
max over a ball against a median caliber reads above 1 even on plain shaft
(ExPID96 S1: control median 1.23, p95 1.94 over 23,356 points).

    LICONN_EA_RUN=<run dir> python tutorials/neuron_liconn_moe/error_analysis/end_evidence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import edt
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

from connectomics.metrics.unsupervised.end_evidence import (  # noqa: E402
    ball_max_radius,
    face_planes,
    face_reach_um,
    terminal_branch,
)

SPUR_FACTOR = 3.0
BORDER_UM = 0.5
BALL_UM = 0.4
CONTROL_PERCENTILE = 95
CONTROL_CLEARANCE_UM = 0.5
CONTROLS_PER_SEGMENT = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=V.OUT / "end_evidence.json")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    spacing = np.asarray(V.SPACING_UM_ZYX)
    analysis = json.loads(V.ANALYSIS.read_text())
    with np.load(V.OUT / "skeletons_fine.npz", allow_pickle=False) as data:
        labels = data["labels"].astype(np.int64)
        voff, eoff = data["vertex_offset"], data["edge_offset"]
        verts, edges_all, radii_all = data["vertices_um_zyx"], data["edges"], data["radii_um"]
    index = {int(label): i for i, label in enumerate(labels.tolist())}

    def graph(i: int):
        v0, v1, e0, e1 = voff[i], voff[i + 1], eoff[i], eoff[i + 1]
        return verts[v0:v1], edges_all[e0:e1], radii_all[v0:v1]

    with h5py.File(V.SEG, "r") as handle:
        seg = np.asarray(handle["main"]).astype(np.uint32)
    print(f"{V.NAME}: seg {seg.shape}; EDT", flush=True)
    dist = edt.edt(seg, anisotropy=tuple(spacing), black_border=False, parallel=args.threads)
    dist = dist.astype(np.float32, copy=False)

    planes = face_planes(seg)
    rng = np.random.default_rng(0)
    control = []
    for segment in analysis["segments"]:
        i = index.get(int(segment["id"]))
        caliber = (segment.get("continuity") or {}).get("caliber_radius_um")
        if i is None or not caliber:
            continue
        vertices, edges, _ = graph(i)
        edges = edges[edges[:, 0] != edges[:, 1]]
        degree = np.bincount(edges.ravel(), minlength=len(vertices))
        special, middle = vertices[degree != 2], np.flatnonzero(degree == 2)
        if not len(special) or not len(middle):
            continue
        clearance = np.min(np.linalg.norm(vertices[middle][:, None] - special[None], axis=-1), axis=1)
        far = middle[clearance >= CONTROL_CLEARANCE_UM]
        for v in rng.choice(far, size=min(CONTROLS_PER_SEGMENT, len(far)), replace=False):
            label = int(segment["id"])
            control.append(ball_max_radius(seg, dist, spacing, label, vertices[v], BALL_UM) / caliber)
    control = np.asarray(control)
    terminal_ratio = float(np.percentile(control, CONTROL_PERCENTILE))
    print(f"control n={len(control)} median {np.median(control):.2f} "
          f"p{CONTROL_PERCENTILE} {terminal_ratio:.2f}", flush=True)

    ends: dict[str, list[dict]] = {}
    for k, segment in enumerate(analysis["segments"]):
        if not segment["free_ends"]:
            continue
        label = int(segment["id"])
        i = index.get(label)
        caliber = (segment.get("continuity") or {}).get("caliber_radius_um")
        rows = []
        for end in segment["free_ends"]:
            tip = np.asarray(end["position_um_zyx"])
            length, junction_radius, at_junction = (
                terminal_branch(*graph(i), end["vertex_index"]) if i is not None else (None, None, False)
            )
            reach = face_reach_um(planes, seg.shape, spacing, label, tip)
            ratio = (ball_max_radius(seg, dist, spacing, label, tip, BALL_UM) / caliber) if caliber else None
            rows.append({
                "vertex_index": end["vertex_index"],
                "spur": bool(at_junction and length < SPUR_FACTOR * junction_radius),
                "at_border": bool(reach <= BORDER_UM),
                "terminal": bool(ratio is not None and ratio >= terminal_ratio),
                "branch_length_um": round(length, 4) if length is not None else None,
                "face_reach_um": None if not np.isfinite(reach) else round(reach, 4),
                "tip_swelling_ratio": None if ratio is None else round(ratio, 3),
            })
        ends[segment["id"]] = rows
        if (k + 1) % 2000 == 0:
            print(f"  {k + 1} / {len(analysis['segments'])} segments", flush=True)

    flat = [row for rows in ends.values() for row in rows]
    payload = {
        "schema_name": "pytc.nogt.end_evidence",
        "schema_version": "1.0.0",
        "metadata": {
            "volume": V.NAME,
            "segmentation_sha256": analysis["metadata"]["segmentation_sha256"],
            "spur_factor": SPUR_FACTOR,
            "border_um": BORDER_UM,
            "ball_um": BALL_UM,
            "terminal_ratio": terminal_ratio,
            "terminal_ratio_source": f"p{CONTROL_PERCENTILE} of {len(control)} mid-shaft controls",
            "control_median": float(np.median(control)),
            "free_end_count": len(flat),
            "spur_count": sum(r["spur"] for r in flat),
            "at_border_count": sum(r["at_border"] for r in flat),
            "terminal_count": sum(r["terminal"] for r in flat),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "ends": ends,
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(args.output)
    m = payload["metadata"]
    print(f"wrote {args.output}: {m['free_end_count']} free ends, spur {m['spur_count']}, "
          f"border {m['at_border_count']}, terminal {m['terminal_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
