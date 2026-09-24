#!/usr/bin/env python3
"""Propose joins for the free ends found by `analyze.py`, under semantic constraints.

An axon end may only be joined to an axon; a dendrite only to a dendrite. On top
of that the pair must be close, of agreeing caliber, aligned with the free end's
outward direction, and -- decisively -- supported by affinity across the gap.

The affinity gate is not decoration. Tip geometry alone proposes well and decides
badly: on zebrafinch the same geometric linker topped out near 0.82 precision
against a break-even bar of about 0.95, and shape-only "facing" models failed
outright, while one-tip plus an affinity floor produced the first clean gain.
Joins are asymmetric -- one wrong join fuses two processes over their whole
length -- so every gate here is a veto and the defaults are conservative.

The affinity volume is the same `scale_sigmoid` prediction ABISS agglomerated.
The default floor is set to that run's merge threshold (0.60) because it is the
one value on this scale already calibrated for this volume -- but the two are not
the same quantity: ABISS thresholds a region-graph edge weight between watershed
fragments, while this samples voxels along the straight gap. Treat the default as
an anchor of the right order, not an equivalence, and read the reported affinity
distribution of accepted versus rejected pairs before trusting it. The volume's
own per-voxel median is about 0.61, so this floor is permissive on its own and
does its work in combination with the geometric gates. Both the stored
(compressed) value and the restored probability are reported.

This writes proposals only. `--apply` additionally writes a relabeled volume, and
is deliberately a separate step: there is no ground truth here, so the precision
of these proposals on this volume is unknown and unmeasurable.

    python dev/liconn_moe/link_splits.py
    python dev/liconn_moe/link_splits.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import volume as V  # noqa: E402

from connectomics.decoding.error_correction.affinity import restore_sigmoid  # noqa: E402
from connectomics.decoding.error_correction.split_links import (  # noqa: E402
    LinkConfig,
    LinkSite,
    connected_groups,
    propose_links,
    select_links,
)

SPACING_UM = np.asarray(V.SPACING_UM_ZYX)


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GapAffinity:
    """Mean affinity sampled along the straight segment between two free ends.

    The whole affinity volume is 3 x 585 x 1152 x 1152 float16, about 4.7 GB in
    memory, which is cheaper than issuing two random HDF5 reads per candidate.
    Channel c is the edge along the input array's axis c; a gap has no preferred
    axis, so the three channels are averaged at each sample.
    """

    def __init__(self, path: Path, samples: int = 9):
        with h5py.File(path, "r") as handle:
            self.volume = np.asarray(handle["main"], dtype=np.float16)
        if self.volume.shape[1:] != V.SHAPE_ZYX:
            raise ValueError(f"affinity shape {self.volume.shape} does not match {V.SHAPE_ZYX}")
        self.samples = int(samples)
        self.shape = np.asarray(V.SHAPE_ZYX)

    def __call__(self, left_um: np.ndarray, right_um: np.ndarray) -> dict[str, float] | None:
        points = np.linspace(left_um, right_um, self.samples)
        index = np.rint(points / SPACING_UM - 0.5).astype(np.int64)
        inside = np.all((index >= 0) & (index < self.shape), axis=1)
        index = index[inside]
        if not len(index):
            return None
        values = self.volume[:, index[:, 0], index[:, 1], index[:, 2]].astype(np.float32)
        per_sample = values.mean(axis=0)
        restored = restore_sigmoid(per_sample)
        return {
            "mean": float(per_sample.mean()),
            "min": float(per_sample.min()),
            "restored_mean": float(restored.mean()),
            "restored_min": float(restored.min()),
            "samples": int(len(index)),
        }


def load_analysis(
    path: Path, *, skip_bouton: bool = False
) -> tuple[list[LinkSite], dict[int, str], dict]:
    payload = json.loads(path.read_text())
    sites: list[LinkSite] = []
    types: dict[int, str] = {}
    for segment in payload["segments"]:
        label = int(segment["id"])
        types[label] = segment["semantic_type"]
        for end in segment["free_ends"]:
            if skip_bouton and end.get("shape") == "bouton_head":
                continue
            sites.append(
                LinkSite(
                    label=label,
                    vertex_index=int(end["vertex_index"]),
                    position_um_zyx=tuple(float(v) for v in end["position_um_zyx"]),
                    outward_tangent_zyx=tuple(float(v) for v in end["outward_tangent_zyx"]),
                    radius_um=float(end["radius_um"]),
                    semantic_type=segment["semantic_type"],
                    shaft_radius_um=end.get("shaft_radius_um"),
                    shape=end.get("shape"),
                )
            )
    return sites, types, payload["metadata"]


def load_skeleton_points(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        offset = np.asarray(data["vertex_offset"], dtype=np.int64)
        vertices = np.asarray(data["vertices_um_zyx"], dtype=np.float64)
        radii = np.asarray(data["radii_um"], dtype=np.float64)
    points, calibers = {}, {}
    for index, label in enumerate(labels.tolist()):
        v0, v1 = offset[index], offset[index + 1]
        points[label] = vertices[v0:v1]
        calibers[label] = radii[v0:v1]
    return points, calibers


def apply_groups(groups: list[list[int]], destination: Path) -> dict:
    """Write a relabeled volume in which each group takes its smallest member ID."""
    mapping = {member: group[0] for group in groups for member in group}
    with h5py.File(V.SEG, "r") as source:
        dataset = source["main"]
        maximum = 0
        for z0 in range(0, dataset.shape[0], 32):
            maximum = max(maximum, int(np.asarray(dataset[z0 : z0 + 32]).max()))
        lut = np.arange(maximum + 1, dtype=np.uint64)
        for member, target in mapping.items():
            if member <= maximum:
                lut[member] = target
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".TMP.h5")
        with h5py.File(temporary, "w") as sink:
            out = sink.create_dataset(
                "main", shape=dataset.shape, dtype=np.uint64, chunks=dataset.chunks,
                compression="gzip", compression_opts=1,
            )
            for z0 in range(0, dataset.shape[0], 32):
                out[z0 : z0 + 32] = lut[np.asarray(dataset[z0 : z0 + 32])]
    temporary.replace(destination)
    return {
        "output": str(destination),
        "output_sha256": sha256(destination),
        "groups_applied": len(groups),
        "labels_absorbed": len(mapping) - len(groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=V.ANALYSIS)
    parser.add_argument("--skeletons", type=Path, default=V.SKELETONS)
    parser.add_argument("--output", type=Path, default=V.LINKS)
    parser.add_argument("--max-gap-um", type=float, default=0.5)
    parser.add_argument("--min-caliber-ratio", type=float, default=0.5)
    parser.add_argument("--max-tangent-deg", type=float, default=50.0)
    parser.add_argument(
        "--min-affinity",
        type=float,
        default=0.60,
        help="floor on stored scale_sigmoid affinity; default is this volume's merge threshold",
    )
    parser.add_argument("--max-group-size", type=int, default=4)
    parser.add_argument("--no-side-attach", action="store_true", help="tip-to-tip pairs only")
    parser.add_argument("--no-affinity", action="store_true", help="geometry only; not advised")
    parser.add_argument(
        "--skip-bouton-ends",
        action="store_true",
        help="do not offer a bouton-headed free end as a link source. A swelling at "
        "the tip is positive evidence the process really ends there, so joining it "
        "is more likely to be a false merge than a repair.",
    )
    parser.add_argument(
        "--require-mutual",
        action="store_true",
        help="keep only tip-to-tip pairs that chose each other; this deletes every "
        "side attachment, so it is off by default",
    )
    parser.add_argument(
        "--no-semantic-gate",
        action="store_true",
        help="do not veto on axon/dendrite class. Use when the caliber gates are not "
        "supported by the volume's own distribution, where the class is noise and "
        "vetoing on it only removes good candidates.",
    )
    parser.add_argument("--apply", action="store_true", help="also write the relabeled volume")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")
    started = time.monotonic()

    sites, types, analysis_metadata = load_analysis(
        args.analysis, skip_bouton=args.skip_bouton_ends
    )
    log(f"{len(sites)} free ends over {len(types)} segments")
    points, calibers = ({}, {})
    if not args.no_side_attach:
        points, calibers = load_skeleton_points(args.skeletons)
        log(f"side-attachment enabled against {len(points)} skeletons")
    probe = None
    if not args.no_affinity:
        log(f"loading affinity {V.AFFINITY}")
        probe = GapAffinity(V.AFFINITY)
        log("affinity loaded")

    config = LinkConfig(
        max_gap_um=args.max_gap_um,
        min_caliber_ratio=args.min_caliber_ratio,
        max_tangent_deg=args.max_tangent_deg,
        min_affinity=None if args.no_affinity else args.min_affinity,
        max_group_size=args.max_group_size,
        require_mutual=args.require_mutual,
        apply_semantic_gate=not args.no_semantic_gate,
    )
    candidates = propose_links(
        sites,
        config=config,
        skeleton_points=points or None,
        skeleton_radii=calibers or None,
        label_semantic_types=types,
        affinity_probe=probe,
    )
    log(f"{len(candidates)} free ends had a partner within {args.max_gap_um} um")
    reasons = Counter(c.reject_reason or "accepted" for c in candidates)
    pairs = select_links(candidates, require_mutual=args.require_mutual)
    groups, dropped = connected_groups(pairs, max_group_size=config.max_group_size)
    kept = [c for c in pairs if c not in dropped]
    log(f"{len(pairs)} pairs ({'mutual only' if args.require_mutual else 'one-sided allowed'}) "
        f"-> {len(groups)} groups ({len(dropped)} edges dropped by cap)")

    applied = None
    if args.apply:
        destination = V.OUT / f"{V.NAME}_seg_abiss_mt060_split_linked.h5"
        log(f"applying {len(groups)} groups -> {destination}")
        applied = apply_groups(groups, destination)
        log(f"wrote {destination}")

    payload = {
        "schema_name": "pytc.nogt.split_links",
        "schema_version": "1.0.0",
        "metadata": {
            "volume": V.NAME,
            "segmentation": str(V.SEG),
            "affinity": str(V.AFFINITY) if probe else None,
            "affinity_scale": "stored scale_sigmoid = sigmoid(0.2 * logit); "
            "`restored_*` undoes it",
            "analysis": str(args.analysis),
            "analysis_metadata": analysis_metadata,
            "config": asdict(config),
            "side_attachment": not args.no_side_attach,
            "skip_bouton_ends": args.skip_bouton_ends,
            "require_mutual": args.require_mutual,
            "semantic_gate": not args.no_semantic_gate,
            "script_sha256": sha256(Path(__file__)),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        },
        "summary": {
            "free_end_count": len(sites),
            "ends_with_a_partner": len(candidates),
            "outcome_counts": dict(reasons),
            "pairs": len(pairs),
            "groups": len(groups),
            "edges_dropped_by_group_cap": len(dropped),
            "labels_in_groups": sum(len(group) for group in groups),
            "group_size_histogram": dict(Counter(len(group) for group in groups)),
        },
        "limitations": [
            "No ground truth: the precision of these proposals is unknown.",
            "Accepted means every configured gate passed, never verified.",
            "A real axon terminal also presents a free end and can be joined in error.",
            "Affinity is read at sampled voxels along a straight line, not a path search.",
        ],
        "groups": [[str(member) for member in group] for group in groups],
        "proposals": [
            {
                **{
                    key: (round(value, 5) if isinstance(value, float) else value)
                    for key, value in asdict(candidate).items()
                    if key != "affinity"
                },
                "left_label": str(candidate.left_label),
                "right_label": str(candidate.right_label),
                "affinity": candidate.affinity,
            }
            for candidate in kept
        ],
        "rejected_sample": [
            {
                "left_label": str(candidate.left_label),
                "right_label": str(candidate.right_label),
                "reject_reason": candidate.reject_reason,
                "gap_um": round(candidate.gap_um, 5),
                "tangent_deg": round(candidate.tangent_deg, 2),
                "caliber_ratio": round(candidate.caliber_ratio, 4),
                "affinity": candidate.affinity,
            }
            for candidate in candidates
            if candidate.reject_reason
        ][:2000],
        "applied": applied,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".TMP.json")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=1) + "\n")
    temporary.replace(args.output)

    print(f"\nwrote {args.output}")
    print("outcome per free end with a partner:")
    for reason, count in reasons.most_common():
        print(f"  {reason:34s} {count:7d}")
    print(f"pairs {len(pairs)}, groups {len(groups)}, "
          f"labels absorbed {sum(len(g) for g in groups) - len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
