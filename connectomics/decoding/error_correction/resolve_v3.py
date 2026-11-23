#!/usr/bin/env python3
"""Add atomic two-host connector contractions to the frozen GT-free v2 proposal.

A connector is a physical-interior 10--100 um segment with exactly two eligible contacts to
distinct, at-least-2x larger hosts.  Both contacts must pass repaired-skeleton continuation,
caliber, spine, gap, and affinity floors; their junctions must lie at distinct ends of the
connector.  The two edges are accepted or rejected atomically.  Existing v2 edges are never
dropped: connector groups are added deterministically only while the component owner and size
firewalls remain valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .resolve import (
    DEFAULT_CANDIDATES,
    DEFAULT_JUNCTIONS,
    DEFAULT_NUCLEI,
    UnionFind,
    load_gt_free,
    load_nucleus_owners,
    reject_evaluation_path,
    sha256_file,
)
from .resolve_v2 import DEFAULT_BOUNDARY, lookup_boundary

DEV = Path(__file__).resolve().parent
V2_ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v2"
ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v3"
DEFAULT_V2_PROPOSALS = V2_ROOT / "frozen_junction_merges.npz"
DEFAULT_V2_REPORT = V2_ROOT / "frozen_junction_merges.json"
DEFAULT_AUDIT = ROOT / "junction_merge_candidates.npz"
DEFAULT_REPORT = ROOT / "junction_merge_audit.json"
DEFAULT_PROPOSALS = ROOT / "frozen_junction_merges.npz"
DEFAULT_FROZEN_REPORT = ROOT / "frozen_junction_merges.json"


def connector_groups(
    candidate: dict[str, np.ndarray],
    junction: dict[str, np.ndarray],
    rows: np.ndarray,
    left_volume_border: np.ndarray,
    right_volume_border: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, tuple[int, int]]], np.ndarray, np.ndarray]:
    """Return two-edge groups, oriented source labels, and the raw edge-eligible mask."""
    left_length = np.asarray(candidate["left_length_nm"][rows], dtype=np.float64)
    right_length = np.asarray(candidate["right_length_nm"][rows], dtype=np.float64)
    left_voxels = np.asarray(candidate["left_voxels"][rows], dtype=np.float64)
    right_voxels = np.asarray(candidate["right_voxels"][rows], dtype=np.float64)
    source_left = (
        ~left_volume_border
        & ~candidate["left_nucleus_anchor"][rows].astype(bool)
        & (left_length >= args.min_connector_length_nm)
        & (left_length <= args.max_connector_length_nm)
        & (right_length >= args.min_connector_host_length_nm)
        & (right_length >= args.min_connector_host_ratio * left_length)
        & (right_voxels >= args.min_connector_host_ratio * left_voxels)
        & (junction["a_leaf_dist_nm"] <= args.max_leaf_distance_nm)
    )
    source_right = (
        ~right_volume_border
        & ~candidate["right_nucleus_anchor"][rows].astype(bool)
        & (right_length >= args.min_connector_length_nm)
        & (right_length <= args.max_connector_length_nm)
        & (left_length >= args.min_connector_host_length_nm)
        & (left_length >= args.min_connector_host_ratio * right_length)
        & (left_voxels >= args.min_connector_host_ratio * right_voxels)
        & (junction["b_leaf_dist_nm"] <= args.max_leaf_distance_nm)
    )
    source = np.where(source_left, candidate["left"][rows], candidate["right"][rows])
    host = np.where(source_left, candidate["right"][rows], candidate["left"][rows])
    edge_eligible = (
        (source_left ^ source_right)
        & np.isfinite(junction["turn_short_deg"])
        & np.isfinite(junction["radius_ratio"])
        & (junction["gap_junction_nm"] <= args.max_junction_gap_nm)
        & (junction["turn_short_deg"] > args.min_turn_short_deg)
        & (junction["radius_ratio"] >= args.min_radius_ratio)
        & ~junction["spine_veto"].astype(bool)
        & (candidate["affinity_mean"][rows] >= args.min_connector_affinity_mean)
        & (candidate["affinity_ge08_fraction"][rows] >= args.min_connector_affinity_ge08_fraction)
        & (candidate["affinity_ge09_fraction"][rows] >= args.min_connector_affinity_ge09_fraction)
    )
    incident: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(edge_eligible).tolist():
        incident[int(source[index])].append(index)
    groups = []
    for source_label, indices in sorted(incident.items()):
        if len(indices) != 2:
            continue
        first, second = indices
        if int(host[first]) == int(host[second]):
            continue
        separation = float(
            np.linalg.norm(junction["junction_zyx_nm"][first] - junction["junction_zyx_nm"][second])
        )
        source_length = float(left_length[first] if source_left[first] else right_length[first])
        if separation < args.min_connector_end_separation_nm:
            continue
        if separation < args.min_connector_end_separation_ratio * source_length:
            continue
        groups.append((source_label, (first, second)))
    return groups, source, edge_eligible


def add_guarded_groups(
    left: np.ndarray,
    right: np.ndarray,
    base: np.ndarray,
    groups: list[tuple[int, tuple[int, int]]],
    owners: dict[int, frozenset[int]],
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add atomic connector groups without ever invalidating or dropping base edges."""
    union = UnionFind()
    for index in np.flatnonzero(base).tolist():
        union.union(int(left[index]), int(right[index]))
    labels_by_root: dict[int, set[int]] = defaultdict(set)
    owners_by_root: dict[int, set[int]] = defaultdict(set)
    for label in union.parent:
        root = union.find(label)
        labels_by_root[root].add(label)
        owners_by_root[root].update(owners.get(label, ()))

    accepted = np.asarray(base, dtype=bool).copy()
    owner_rejected = np.zeros(len(left), dtype=bool)
    size_rejected = np.zeros(len(left), dtype=bool)
    for _, indices in groups:
        labels = {int(left[index]) for index in indices} | {int(right[index]) for index in indices}
        roots = {union.find(label) for label in labels}
        combined_labels = set().union(*(labels_by_root.get(root, {root}) for root in roots))
        combined_labels.update(labels)
        combined_owners = set().union(*(owners_by_root.get(root, set()) for root in roots))
        for label in labels:
            combined_owners.update(owners.get(label, ()))
        if len(combined_owners) > 1:
            owner_rejected[list(indices)] = True
            continue
        if len(combined_labels) > max_segments:
            size_rejected[list(indices)] = True
            continue
        for index in indices:
            union.union(int(left[index]), int(right[index]))
            accepted[index] = True
        root = union.find(next(iter(labels)))
        for old_root in roots:
            labels_by_root.pop(old_root, None)
            owners_by_root.pop(old_root, None)
        labels_by_root[root] = combined_labels
        owners_by_root[root] = combined_owners
    return accepted, owner_rejected, size_rejected


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        DEV / "resolve.py",
        DEV / "resolve_v2.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--junctions", type=Path, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--boundary-inventory", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--nucleus-manifest", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument("--base-proposals", type=Path, default=DEFAULT_V2_PROPOSALS)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_V2_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--max-junction-gap-nm", type=float, default=750.0)
    parser.add_argument("--max-leaf-distance-nm", type=float, default=1_500.0)
    parser.add_argument("--min-turn-short-deg", type=float, default=97.36936936936937)
    parser.add_argument("--min-radius-ratio", type=float, default=0.45)
    parser.add_argument("--min-connector-length-nm", type=float, default=10_000.0)
    parser.add_argument("--max-connector-length-nm", type=float, default=100_000.0)
    parser.add_argument("--min-connector-host-length-nm", type=float, default=50_000.0)
    parser.add_argument("--min-connector-host-ratio", type=float, default=2.0)
    parser.add_argument("--min-connector-affinity-mean", type=float, default=0.25)
    parser.add_argument("--min-connector-affinity-ge08-fraction", type=float, default=0.15)
    parser.add_argument("--min-connector-affinity-ge09-fraction", type=float, default=0.10)
    parser.add_argument("--min-connector-end-separation-nm", type=float, default=5_000.0)
    parser.add_argument("--min-connector-end-separation-ratio", type=float, default=0.20)
    parser.add_argument("--max-component-segments", type=int, default=8)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    input_paths = (
        args.candidates,
        args.junctions,
        args.boundary_inventory,
        args.nucleus_manifest,
        args.base_proposals,
        args.base_report,
    )
    for path in (*input_paths, args.audit, args.report, args.proposals, args.frozen_report):
        reject_evaluation_path(path)
    if args.freeze and (args.proposals.exists() or args.frozen_report.exists()):
        raise FileExistsError("refusing to overwrite an already-frozen proposal")

    base_digest = sha256_file(args.base_proposals)
    base_report = json.loads(args.base_report.read_text())
    if base_report.get("proposal_sha256") != base_digest:
        raise ValueError("base proposal hash differs from its frozen report")
    base_proposal = load_gt_free(args.base_proposals)
    if not bool(base_proposal["frozen_before_evaluation"].item()):
        raise ValueError("base proposal was not frozen before evaluation")
    candidate = load_gt_free(args.candidates)
    junction = load_gt_free(args.junctions)
    boundary = load_gt_free(args.boundary_inventory)
    rows = np.asarray(junction["row"], dtype=np.int64)
    left = candidate["left"][rows]
    right = candidate["right"][rows]
    row_to_position = {int(row): position for position, row in enumerate(rows.tolist())}
    base = np.zeros(len(rows), dtype=bool)
    for row in np.asarray(base_proposal["candidate_row"], dtype=np.int64).tolist():
        if int(row) not in row_to_position:
            raise ValueError(f"base proposal row {row} is outside junction scope")
        base[row_to_position[int(row)]] = True
    if not np.array_equal(left[base], base_proposal["left"]) or not np.array_equal(
        right[base], base_proposal["right"]
    ):
        raise ValueError("base proposal labels do not align with candidate rows")

    left_border = lookup_boundary(boundary, left)
    right_border = lookup_boundary(boundary, right)
    groups, connector_source, connector_edge_eligible = connector_groups(
        candidate, junction, rows, left_border, right_border, args
    )
    connector = np.zeros(len(rows), dtype=bool)
    for _, indices in groups:
        connector[list(indices)] = True
    owners = load_nucleus_owners(args.nucleus_manifest)
    accepted, connector_owner_rejected, connector_size_rejected = add_guarded_groups(
        left, right, base, groups, owners, args.max_component_segments
    )
    connector_accepted = connector & accepted & ~base

    connector_parameters = {
        name: getattr(args, name)
        for name in (
            "max_junction_gap_nm",
            "max_leaf_distance_nm",
            "min_turn_short_deg",
            "min_radius_ratio",
            "min_connector_length_nm",
            "max_connector_length_nm",
            "min_connector_host_length_nm",
            "min_connector_host_ratio",
            "min_connector_affinity_mean",
            "min_connector_affinity_ge08_fraction",
            "min_connector_affinity_ge09_fraction",
            "min_connector_end_separation_nm",
            "min_connector_end_separation_ratio",
            "max_component_segments",
        )
    }
    parameters = {
        "base_v2": json.loads(str(base_proposal["parameters"].item())),
        "connector_v3": connector_parameters,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.audit,
        row=rows,
        left=left,
        right=right,
        base=base,
        connector_edge_eligible=connector_edge_eligible,
        connector_group=connector,
        connector_source=connector_source,
        connector_owner_rejected=connector_owner_rejected,
        connector_size_rejected=connector_size_rejected,
        connector_accepted=connector_accepted,
        accepted=accepted,
        gt_free=np.asarray(True),
    )
    report = {
        "scope": "GT-free v2 plus atomic two-host internal connector contractions",
        "inputs": {
            name: str(path.resolve())
            for name, path in zip(
                ("candidates", "junctions", "boundary", "nuclei", "base", "base_report"),
                input_paths,
            )
        },
        "input_sha256": {path.name: sha256_file(path) for path in input_paths},
        "parameters": parameters,
        "threshold_provenance": (
            "physical branch/host scale and unlabeled lower-quartile contact floors, "
            "required independently at both connector ends; no evaluation or pseudo-GT"
        ),
        "base_pairs": int(np.count_nonzero(base)),
        "eligible_connector_edges": int(np.count_nonzero(connector_edge_eligible)),
        "connector_groups": int(len(groups)),
        "connector_edges": int(np.count_nonzero(connector)),
        "connector_owner_rejected": int(np.count_nonzero(connector_owner_rejected)),
        "connector_size_rejected": int(np.count_nonzero(connector_size_rejected)),
        "connector_edges_accepted": int(np.count_nonzero(connector_accepted)),
        "accepted_pairs": int(np.count_nonzero(accepted)),
        "gt_free": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    if args.freeze:
        impl_digest = implementation_digest()
        proposal_tier = np.where(base[accepted], 2, 3).astype(np.uint8)
        np.savez_compressed(
            args.proposals,
            left=left[accepted],
            right=right[accepted],
            candidate_row=rows[accepted],
            proposal_tier=proposal_tier,
            parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
            implementation_sha256=np.asarray(impl_digest),
            gt_free=np.asarray(True),
            frozen_before_evaluation=np.asarray(True),
        )
        proposal_digest = sha256_file(args.proposals)
        frozen = {
            **report,
            "scope": "GT-free v3 proposal frozen before evaluation",
            "implementation_sha256": impl_digest,
            "proposal_sha256": proposal_digest,
            "proposal_path": str(args.proposals.resolve()),
        }
        args.frozen_report.write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"frozen {int(np.count_nonzero(accepted))} v3 merges sha256={proposal_digest}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
