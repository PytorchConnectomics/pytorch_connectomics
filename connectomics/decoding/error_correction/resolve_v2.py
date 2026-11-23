#!/usr/bin/env python3
"""Freeze two score-free, GT-free junction merge tiers on matchguard ABISS.

Tier 1 is the v1 high-confidence two-ended continuation rule.  Tier 2 implements the
physical-internal-fragment prior: a long neurite fragment that does not touch the actual volume
boundary may attach to a much larger process only at its single eligible skeleton continuation.
Touching the tissue/keep mask does not count as reaching the physical volume boundary.

No weighted score, evaluation skeleton, FFN output, or evaluation LUT is read.  Thresholds are
physical or unlabeled-distribution constants: 750/1500 nm are three/six 250-nm skeleton samples;
30/100 um define a real branch and a long host; 4x defines asymmetric ownership; and the tier-2
0.30 interface mean is the unlabeled contact population's upper quartile (0.295), with at least
20% of faces above both 0.8 and 0.9 affinity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .resolve import (
    DEFAULT_CANDIDATES,
    DEFAULT_JUNCTIONS,
    DEFAULT_NUCLEI,
    component_firewall,
    hard_gate_conditions,
    load_gt_free,
    load_nucleus_owners,
    local_ambiguity_mask,
    reject_evaluation_path,
    sha256_file,
)

DEV = Path(__file__).resolve().parent
ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v2"
DEFAULT_BOUNDARY = ROOT / "boundary_inventory.npz"
DEFAULT_AUDIT = ROOT / "junction_merge_candidates.npz"
DEFAULT_REPORT = ROOT / "junction_merge_audit.json"
DEFAULT_PROPOSALS = ROOT / "frozen_junction_merges.npz"
DEFAULT_FROZEN_REPORT = ROOT / "frozen_junction_merges.json"


def lookup_boundary(inventory: dict[str, np.ndarray], labels: np.ndarray) -> np.ndarray:
    available = np.asarray(inventory["segment_label"], dtype=np.uint64)
    if not np.all(available[:-1] < available[1:]):
        raise ValueError("boundary labels must be unique and sorted")
    index = np.searchsorted(available, labels)
    valid = (index < len(available)) & (available[np.minimum(index, len(available) - 1)] == labels)
    if not np.all(valid):
        raise ValueError(f"boundary inventory lacks {int(np.count_nonzero(~valid))} labels")
    return np.asarray(inventory["touches_volume_boundary"], dtype=bool)[index]


def unique_source_mask(source: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Accept an internal source only when it has exactly one eligible host."""
    counts = Counter(int(value) for value in source[eligible].tolist())
    result = np.zeros(len(source), dtype=bool)
    for index in np.flatnonzero(eligible).tolist():
        result[index] = counts[int(source[index])] == 1
    return result


def forced_internal_tier(
    candidate: dict[str, np.ndarray],
    junction: dict[str, np.ndarray],
    rows: np.ndarray,
    left_volume_border: np.ndarray,
    right_volume_border: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    left_length = np.asarray(candidate["left_length_nm"][rows], dtype=np.float64)
    right_length = np.asarray(candidate["right_length_nm"][rows], dtype=np.float64)
    left_voxels = np.asarray(candidate["left_voxels"][rows], dtype=np.float64)
    right_voxels = np.asarray(candidate["right_voxels"][rows], dtype=np.float64)
    source_left = (
        ~left_volume_border
        & ~candidate["left_nucleus_anchor"][rows].astype(bool)
        & (left_length >= args.min_internal_source_length_nm)
        & (right_length >= args.min_internal_host_length_nm)
        & (right_length >= args.min_internal_host_ratio * left_length)
        & (right_voxels >= args.min_internal_host_ratio * left_voxels)
    )
    source_right = (
        ~right_volume_border
        & ~candidate["right_nucleus_anchor"][rows].astype(bool)
        & (right_length >= args.min_internal_source_length_nm)
        & (left_length >= args.min_internal_host_length_nm)
        & (left_length >= args.min_internal_host_ratio * right_length)
        & (left_voxels >= args.min_internal_host_ratio * right_voxels)
    )
    exactly_one_source = source_left ^ source_right
    source_terminates = (
        source_left & (junction["a_leaf_dist_nm"] <= args.max_leaf_distance_nm)
    ) | (source_right & (junction["b_leaf_dist_nm"] <= args.max_leaf_distance_nm))
    finite = np.isfinite(junction["turn_short_deg"]) & np.isfinite(junction["radius_ratio"])
    conditions = {
        "finite_junction": finite,
        "physical_internal_asymmetric_source": exactly_one_source,
        "source_terminates": source_terminates,
        "junction_gap": junction["gap_junction_nm"] <= args.max_junction_gap_nm,
        "continuation_angle": junction["turn_short_deg"] > args.min_turn_short_deg,
        "caliber_match": junction["radius_ratio"] >= args.min_radius_ratio,
        "not_spine": ~junction["spine_veto"].astype(bool),
        "affinity_mean": candidate["affinity_mean"][rows] >= args.min_internal_affinity_mean,
        "affinity_ge08_fraction": candidate["affinity_ge08_fraction"][rows]
        >= args.min_internal_affinity_ge08_fraction,
        "affinity_ge09_fraction": candidate["affinity_ge09_fraction"][rows]
        >= args.min_internal_affinity_ge09_fraction,
    }
    eligible = np.logical_and.reduce(list(conditions.values()))
    source = np.where(source_left, candidate["left"][rows], candidate["right"][rows])
    unique = unique_source_mask(source, eligible)
    return eligible & unique, source, conditions


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), DEV / "resolve.py"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--junctions", type=Path, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--boundary-inventory", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--nucleus-manifest", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--scope-min-affinity-ge08-fraction", type=float, default=0.05)
    parser.add_argument("--min-affinity-mean", type=float, default=0.70)
    parser.add_argument("--min-affinity-ge09-fraction", type=float, default=0.50)
    parser.add_argument("--max-junction-gap-nm", type=float, default=750.0)
    parser.add_argument("--max-leaf-distance-nm", type=float, default=1_500.0)
    parser.add_argument("--min-turn-short-deg", type=float, default=97.36936936936937)
    parser.add_argument("--min-radius-ratio", type=float, default=0.45)
    parser.add_argument("--local-competition-nm", type=float, default=2_000.0)
    # A real alternating big/small repair can contain the host plus several forced fragments.
    # Eight is still a small bounded correction and admits the largest (seven-label) provisional
    # component seen in this unlabeled proposal graph without permitting a cascade.
    parser.add_argument("--max-component-segments", type=int, default=8)
    parser.add_argument("--min-internal-source-length-nm", type=float, default=30_000.0)
    parser.add_argument("--min-internal-host-length-nm", type=float, default=100_000.0)
    parser.add_argument("--min-internal-host-ratio", type=float, default=4.0)
    parser.add_argument("--min-internal-affinity-mean", type=float, default=0.30)
    parser.add_argument("--min-internal-affinity-ge08-fraction", type=float, default=0.20)
    parser.add_argument("--min-internal-affinity-ge09-fraction", type=float, default=0.20)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    paths = (
        args.candidates,
        args.junctions,
        args.boundary_inventory,
        args.nucleus_manifest,
        args.audit,
        args.report,
        args.proposals,
        args.frozen_report,
    )
    for path in paths:
        reject_evaluation_path(path)
    if args.freeze and (args.proposals.exists() or args.frozen_report.exists()):
        raise FileExistsError("refusing to overwrite an already-frozen proposal")

    candidate = load_gt_free(args.candidates)
    junction = load_gt_free(args.junctions)
    boundary = load_gt_free(args.boundary_inventory)
    if not bool(boundary.get("complete", np.asarray(False)).item()):
        raise ValueError("boundary projection is incomplete")
    rows = np.asarray(junction["row"], dtype=np.int64)
    if not np.all(junction["source"] == 1):
        raise ValueError("junction scope must contain decoder-only source rows")
    if not np.array_equal(junction["left"], candidate["left"][rows]) or not np.array_equal(
        junction["right"], candidate["right"][rows]
    ):
        raise ValueError("junction labels do not align to candidate rows")
    expected = np.flatnonzero(
        candidate["affinity_ge08_fraction"] >= args.scope_min_affinity_ge08_fraction
    )
    if not np.array_equal(np.sort(rows), expected):
        raise ValueError("junction scope is not the exhaustive GT-free affinity selection")

    left = candidate["left"][rows]
    right = candidate["right"][rows]
    left_border = lookup_boundary(boundary, left)
    right_border = lookup_boundary(boundary, right)
    tier1_conditions = hard_gate_conditions(candidate, junction, rows, args)
    tier1_hard = np.logical_and.reduce(list(tier1_conditions.values()))
    tier1_ambiguous = local_ambiguity_mask(
        left, right, junction["junction_zyx_nm"], tier1_hard, args.local_competition_nm
    )
    tier1 = tier1_hard & ~tier1_ambiguous
    tier2, tier2_source, tier2_conditions = forced_internal_tier(
        candidate, junction, rows, left_border, right_border, args
    )
    preliminary = tier1 | tier2
    owners = load_nucleus_owners(args.nucleus_manifest)
    owner_conflict, component_too_large, component_size = component_firewall(
        left, right, preliminary, owners, args.max_component_segments
    )
    accepted = preliminary & ~owner_conflict & ~component_too_large

    parameter_names = (
        "scope_min_affinity_ge08_fraction",
        "min_affinity_mean",
        "min_affinity_ge09_fraction",
        "max_junction_gap_nm",
        "max_leaf_distance_nm",
        "min_turn_short_deg",
        "min_radius_ratio",
        "local_competition_nm",
        "max_component_segments",
        "min_internal_source_length_nm",
        "min_internal_host_length_nm",
        "min_internal_host_ratio",
        "min_internal_affinity_mean",
        "min_internal_affinity_ge08_fraction",
        "min_internal_affinity_ge09_fraction",
    )
    parameters = {name: getattr(args, name) for name in parameter_names}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.audit,
        row=rows,
        left=left,
        right=right,
        junction_zyx_nm=junction["junction_zyx_nm"],
        left_volume_border=left_border,
        right_volume_border=right_border,
        tier1=tier1,
        tier1_local_ambiguous=tier1_ambiguous,
        tier2=tier2,
        tier2_source=tier2_source,
        preliminary=preliminary,
        owner_conflict=owner_conflict,
        component_too_large=component_too_large,
        component_size=component_size,
        accepted=accepted,
        gt_free=np.asarray(True),
    )
    report = {
        "scope": "GT-free hard junction plus forced physical-internal-fragment correction",
        "inputs": {
            name: str(path.resolve())
            for name, path in zip(
                ("candidates", "junctions", "boundary_inventory", "nucleus_manifest"), paths[:4]
            )
        },
        "input_sha256": {
            "candidates": sha256_file(args.candidates),
            "junctions": sha256_file(args.junctions),
            "boundary_inventory": sha256_file(args.boundary_inventory),
            "nucleus_manifest": sha256_file(args.nucleus_manifest),
        },
        "parameters": parameters,
        "threshold_provenance": (
            "physical skeleton sampling/process scale plus unlabeled contact-distribution "
            "upper quartile; no evaluation or pseudo-GT"
        ),
        "scope_pairs": int(len(rows)),
        "tier1_pairs": int(np.count_nonzero(tier1)),
        "tier2_pairs": int(np.count_nonzero(tier2)),
        "tier_overlap": int(np.count_nonzero(tier1 & tier2)),
        "owner_conflict_pairs": int(np.count_nonzero(preliminary & owner_conflict)),
        "oversize_component_pairs": int(np.count_nonzero(preliminary & component_too_large)),
        "accepted_pairs": int(np.count_nonzero(accepted)),
        "gt_free": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    if args.freeze:
        impl_digest = implementation_digest()
        np.savez_compressed(
            args.proposals,
            left=left[accepted],
            right=right[accepted],
            junction_zyx_nm=junction["junction_zyx_nm"][accepted],
            candidate_row=rows[accepted],
            proposal_tier=np.where(tier1[accepted], 1, 2).astype(np.uint8),
            parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
            implementation_sha256=np.asarray(impl_digest),
            gt_free=np.asarray(True),
            frozen_before_evaluation=np.asarray(True),
        )
        proposal_digest = sha256_file(args.proposals)
        frozen = {
            **report,
            "scope": "GT-free v2 proposal frozen before evaluation",
            "implementation_sha256": impl_digest,
            "proposal_sha256": proposal_digest,
            "proposal_path": str(args.proposals.resolve()),
        }
        args.frozen_report.write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"frozen {int(np.count_nonzero(accepted))} v2 merges sha256={proposal_digest}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
