#!/usr/bin/env python3
"""Extend frozen v4 with unique terminating branches attaching to one nucleus host."""

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
    load_gt_free,
    load_nucleus_owners,
    reject_evaluation_path,
    sha256_file,
)
from .resolve_v3 import add_guarded_groups

DEV = Path(__file__).resolve().parent
V4_ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v4"
ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v5"
DEFAULT_BASE_PROPOSALS = V4_ROOT / "frozen_junction_merges.npz"
DEFAULT_BASE_REPORT = V4_ROOT / "frozen_junction_merges.json"
DEFAULT_AUDIT = ROOT / "junction_merge_candidates.npz"
DEFAULT_REPORT = ROOT / "junction_merge_audit.json"
DEFAULT_PROPOSALS = ROOT / "frozen_junction_merges.npz"
DEFAULT_FROZEN_REPORT = ROOT / "frozen_junction_merges.json"


def unique_nucleus_host_tier(
    candidate: dict[str, np.ndarray],
    junction: dict[str, np.ndarray],
    rows: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    left_anchor = candidate["left_nucleus_anchor"][rows].astype(bool)
    right_anchor = candidate["right_nucleus_anchor"][rows].astype(bool)
    left_source = (
        ~left_anchor
        & right_anchor
        & (candidate["left_length_nm"][rows] >= args.min_source_length_nm)
        & (candidate["right_length_nm"][rows] >= args.min_nucleus_host_length_nm)
        & (junction["a_leaf_dist_nm"] <= args.max_leaf_distance_nm)
    )
    right_source = (
        ~right_anchor
        & left_anchor
        & (candidate["right_length_nm"][rows] >= args.min_source_length_nm)
        & (candidate["left_length_nm"][rows] >= args.min_nucleus_host_length_nm)
        & (junction["b_leaf_dist_nm"] <= args.max_leaf_distance_nm)
    )
    source = np.where(left_source, candidate["left"][rows], candidate["right"][rows])
    eligible = (
        (left_source ^ right_source)
        & np.isfinite(junction["turn_short_deg"])
        & np.isfinite(junction["radius_ratio"])
        & (junction["gap_junction_nm"] <= args.max_junction_gap_nm)
        & (junction["turn_short_deg"] > args.min_turn_short_deg)
        & (junction["radius_ratio"] >= args.min_radius_ratio)
        & ~junction["spine_veto"].astype(bool)
        & (candidate["affinity_mean"][rows] >= args.min_affinity_mean)
        & (candidate["affinity_ge08_fraction"][rows] >= args.min_affinity_ge08_fraction)
        & (candidate["affinity_ge09_fraction"][rows] >= args.min_affinity_ge09_fraction)
    )
    counts = Counter(int(value) for value in source[eligible].tolist())
    unique = np.zeros(len(rows), dtype=bool)
    for index in np.flatnonzero(eligible).tolist():
        unique[index] = counts[int(source[index])] == 1
    return unique, source


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        DEV / "resolve.py",
        DEV / "resolve_v3.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--junctions", type=Path, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--nucleus-manifest", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument("--base-proposals", type=Path, default=DEFAULT_BASE_PROPOSALS)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--max-junction-gap-nm", type=float, default=750.0)
    parser.add_argument("--max-leaf-distance-nm", type=float, default=1_500.0)
    parser.add_argument("--min-turn-short-deg", type=float, default=97.36936936936937)
    parser.add_argument("--min-radius-ratio", type=float, default=0.45)
    parser.add_argument("--min-source-length-nm", type=float, default=10_000.0)
    parser.add_argument("--min-nucleus-host-length-nm", type=float, default=100_000.0)
    parser.add_argument("--min-affinity-mean", type=float, default=0.30)
    parser.add_argument("--min-affinity-ge08-fraction", type=float, default=0.20)
    parser.add_argument("--min-affinity-ge09-fraction", type=float, default=0.20)
    parser.add_argument("--max-component-segments", type=int, default=8)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    input_paths = (
        args.candidates,
        args.junctions,
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
    rows = np.asarray(junction["row"], dtype=np.int64)
    left = candidate["left"][rows]
    right = candidate["right"][rows]
    row_to_position = {int(row): position for position, row in enumerate(rows.tolist())}
    base = np.zeros(len(rows), dtype=bool)
    base_tier = np.zeros(len(rows), dtype=np.uint8)
    for row, tier in zip(
        np.asarray(base_proposal["candidate_row"], dtype=np.int64).tolist(),
        np.asarray(base_proposal["proposal_tier"], dtype=np.uint8).tolist(),
    ):
        position = row_to_position.get(int(row))
        if position is None:
            raise ValueError(f"base proposal row {row} is outside junction scope")
        base[position] = True
        base_tier[position] = int(tier)
    if not np.array_equal(left[base], base_proposal["left"]) or not np.array_equal(
        right[base], base_proposal["right"]
    ):
        raise ValueError("base proposal labels do not align with candidate rows")

    nucleus_tier, source = unique_nucleus_host_tier(candidate, junction, rows, args)
    additional = nucleus_tier & ~base
    groups = [
        (int(source[index]), (index,))
        for index in sorted(
            np.flatnonzero(additional).tolist(), key=lambda value: (int(source[value]), value)
        )
    ]
    owners = load_nucleus_owners(args.nucleus_manifest)
    accepted, owner_rejected, size_rejected = add_guarded_groups(
        left, right, base, groups, owners, args.max_component_segments
    )
    tier_accepted = additional & accepted
    tier_parameters = {
        name: getattr(args, name)
        for name in (
            "max_junction_gap_nm",
            "max_leaf_distance_nm",
            "min_turn_short_deg",
            "min_radius_ratio",
            "min_source_length_nm",
            "min_nucleus_host_length_nm",
            "min_affinity_mean",
            "min_affinity_ge08_fraction",
            "min_affinity_ge09_fraction",
            "max_component_segments",
        )
    }
    parameters = {
        "base_v4": json.loads(str(base_proposal["parameters"].item())),
        "nucleus_host_v5": tier_parameters,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.audit,
        row=rows,
        left=left,
        right=right,
        base=base,
        nucleus_host=nucleus_tier,
        additional=additional,
        source=source,
        owner_rejected=owner_rejected,
        size_rejected=size_rejected,
        tier_accepted=tier_accepted,
        accepted=accepted,
        gt_free=np.asarray(True),
    )
    report = {
        "scope": "GT-free v4 plus unique terminating branch to one nucleus-bearing host",
        "inputs": {
            name: str(path.resolve())
            for name, path in zip(
                ("candidates", "junctions", "nuclei", "base", "base_report"), input_paths
            )
        },
        "input_sha256": {path.name: sha256_file(path) for path in input_paths},
        "parameters": parameters,
        "threshold_provenance": (
            "external nucleus identity plus physical skeleton scale and unlabeled upper-quartile "
            "contact confidence; no evaluation or pseudo-GT"
        ),
        "base_pairs": int(np.count_nonzero(base)),
        "unique_nucleus_host_edges": int(np.count_nonzero(nucleus_tier)),
        "additional_edges": int(np.count_nonzero(additional)),
        "owner_rejected": int(np.count_nonzero(owner_rejected)),
        "size_rejected": int(np.count_nonzero(size_rejected)),
        "tier_edges_accepted": int(np.count_nonzero(tier_accepted)),
        "accepted_pairs": int(np.count_nonzero(accepted)),
        "gt_free": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    if args.freeze:
        impl_digest = implementation_digest()
        proposal_tier = base_tier.copy()
        proposal_tier[tier_accepted] = 5
        np.savez_compressed(
            args.proposals,
            left=left[accepted],
            right=right[accepted],
            candidate_row=rows[accepted],
            proposal_tier=proposal_tier[accepted],
            parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
            implementation_sha256=np.asarray(impl_digest),
            gt_free=np.asarray(True),
            frozen_before_evaluation=np.asarray(True),
        )
        proposal_digest = sha256_file(args.proposals)
        frozen = {
            **report,
            "scope": "GT-free v5 proposal frozen before evaluation",
            "implementation_sha256": impl_digest,
            "proposal_sha256": proposal_digest,
            "proposal_path": str(args.proposals.resolve()),
        }
        args.frozen_report.write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"frozen {int(np.count_nonzero(accepted))} v5 merges sha256={proposal_digest}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
