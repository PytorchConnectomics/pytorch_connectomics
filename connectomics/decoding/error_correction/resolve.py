#!/usr/bin/env python3
"""Freeze GT-free contact-anchored merges on the matchguard ABISS segmentation.

This resolver deliberately has no learned or weighted score.  A pair is eligible only when
all hard physical tests pass: a majority-high-confidence direct interface, two repaired
skeletons meeting at the same junction, nearby branch terminations, continuation rather than
a perpendicular spine, and compatible local caliber.  Competition is local to a junction;
distant contacts of one long segment never compete with each other.

External nucleus instances are a cannot-link firewall.  A provisional merge component that
contains more than one owner is quarantined in full.  Evaluation skeletons and their LUTs are
forbidden inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import reject_evaluation_path

REPO = Path(__file__).resolve().parents[3]
DEV = REPO / "dev" / "zebrafinch"
ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v1"
DEFAULT_CANDIDATES = (
    DEV / "matchguard_error_correction" / "decoder_gtfree" / "contact_merge_candidates.npz"
)
DEFAULT_JUNCTIONS = ROOT / "junction_features_raw.npz"
DEFAULT_NUCLEI = (
    DEV
    / "wholevol_arm0_native96_nuc_matchguard"
    / "seg_arm0_native96_nuc_matchguard"
    / "nucleus_competition"
    / "manifest.json"
)
DEFAULT_AUDIT = ROOT / "junction_merge_candidates.npz"
DEFAULT_REPORT = ROOT / "junction_merge_audit.json"
DEFAULT_PROPOSALS = ROOT / "frozen_junction_merges.npz"
DEFAULT_FROZEN_REPORT = ROOT / "frozen_junction_merges.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gt_free(path: Path) -> dict[str, np.ndarray]:
    reject_evaluation_path(path)
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if "gt_free" not in result or not bool(result["gt_free"].item()):
        raise ValueError(f"{path} is not explicitly marked GT-free")
    return result


def load_nucleus_owners(path: Path) -> dict[int, frozenset[int]]:
    """Map final prediction labels to external nucleus identities."""
    reject_evaluation_path(path)
    payload = json.loads(path.read_text())
    qualified = payload.get("qualified_segment_labels")
    qualified_owners = payload.get("qualified_segment_owners")
    if not isinstance(qualified, dict) or not isinstance(qualified_owners, dict):
        raise ValueError(f"{path}: missing qualified segment owner tables")
    owners_by_final: dict[int, set[int]] = defaultdict(set)
    for raw_label, owners in qualified_owners.items():
        territories = qualified.get(raw_label)
        if isinstance(territories, dict):
            for owner in owners:
                final_label = int(territories.get(str(owner), 0))
                if final_label:
                    owners_by_final[final_label].add(int(owner))
        else:
            owners_by_final[int(raw_label)].update(int(owner) for owner in owners)
    return {label: frozenset(owners) for label, owners in owners_by_final.items()}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        parent = self.parent.setdefault(value, value)
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left


def local_ambiguity_mask(
    left: np.ndarray,
    right: np.ndarray,
    junction_zyx_nm: np.ndarray,
    eligible: np.ndarray,
    radius_nm: float,
) -> np.ndarray:
    """Mark multiple candidates competing for the same local branch junction.

    Contacts farther apart than ``radius_nm`` on a long segment are independent breaks and
    are intentionally allowed.  No candidate score is consulted.
    """
    ambiguous = np.zeros(len(left), dtype=bool)
    incident: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(eligible).tolist():
        incident[int(left[index])].append(index)
        incident[int(right[index])].append(index)
    for indices in incident.values():
        if len(indices) < 2:
            continue
        union = UnionFind()
        for index in indices:
            union.find(index)
        for position, first in enumerate(indices):
            distance = np.linalg.norm(
                junction_zyx_nm[np.asarray(indices[position + 1 :], dtype=np.int64)]
                - junction_zyx_nm[first],
                axis=1,
            )
            for second in np.asarray(indices[position + 1 :], dtype=np.int64)[
                distance <= radius_nm
            ].tolist():
                union.union(first, second)
        groups: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            groups[union.find(index)].append(index)
        for group in groups.values():
            if len(group) > 1:
                ambiguous[group] = True
    return ambiguous


def component_firewall(
    left: np.ndarray,
    right: np.ndarray,
    eligible: np.ndarray,
    owners: dict[int, frozenset[int]],
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quarantine complete provisional components with multiple owners or excessive size."""
    union = UnionFind()
    for index in np.flatnonzero(eligible).tolist():
        union.union(int(left[index]), int(right[index]))
    labels_by_root: dict[int, set[int]] = defaultdict(set)
    owners_by_root: dict[int, set[int]] = defaultdict(set)
    for label in union.parent:
        root = union.find(label)
        labels_by_root[root].add(label)
        owners_by_root[root].update(owners.get(label, ()))
    owner_conflict = np.zeros(len(left), dtype=bool)
    component_too_large = np.zeros(len(left), dtype=bool)
    component_size = np.zeros(len(left), dtype=np.uint16)
    for index in np.flatnonzero(eligible).tolist():
        root = union.find(int(left[index]))
        size = len(labels_by_root[root])
        component_size[index] = size
        owner_conflict[index] = len(owners_by_root[root]) > 1
        component_too_large[index] = size > max_segments
    return owner_conflict, component_too_large, component_size


def hard_gate_conditions(
    candidate: dict[str, np.ndarray],
    junction: dict[str, np.ndarray],
    rows: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    finite = np.ones(len(rows), dtype=bool)
    for key in (
        "gap_junction_nm",
        "turn_short_deg",
        "radius_ratio",
        "a_leaf_dist_nm",
        "b_leaf_dist_nm",
    ):
        finite &= np.isfinite(junction[key])
    return {
        "finite_junction": finite,
        "affinity_mean": candidate["affinity_mean"][rows] >= args.min_affinity_mean,
        "affinity_ge09_fraction": candidate["affinity_ge09_fraction"][rows]
        >= args.min_affinity_ge09_fraction,
        "junction_gap": junction["gap_junction_nm"] <= args.max_junction_gap_nm,
        "both_branches_terminate": (junction["a_leaf_dist_nm"] <= args.max_leaf_distance_nm)
        & (junction["b_leaf_dist_nm"] <= args.max_leaf_distance_nm),
        "continuation_angle": junction["turn_short_deg"] > args.min_turn_short_deg,
        "caliber_match": junction["radius_ratio"] >= args.min_radius_ratio,
        "not_spine": ~junction["spine_veto"].astype(bool),
        "candidate_nucleus_firewall": ~(
            candidate["left_nucleus_anchor"][rows].astype(bool)
            & candidate["right_nucleus_anchor"][rows].astype(bool)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--junctions", type=Path, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--nucleus-manifest", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--min-affinity-mean", type=float, default=0.70)
    parser.add_argument("--min-affinity-ge09-fraction", type=float, default=0.50)
    parser.add_argument("--scope-min-affinity-ge08-fraction", type=float, default=0.05)
    parser.add_argument("--max-junction-gap-nm", type=float, default=750.0)
    parser.add_argument("--max-leaf-distance-nm", type=float, default=1_500.0)
    parser.add_argument("--min-turn-short-deg", type=float, default=97.36936936936937)
    parser.add_argument("--min-radius-ratio", type=float, default=0.45)
    parser.add_argument("--local-competition-nm", type=float, default=2_000.0)
    parser.add_argument("--max-component-segments", type=int, default=4)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    for path in (
        args.candidates,
        args.junctions,
        args.nucleus_manifest,
        args.audit,
        args.report,
        args.proposals,
        args.frozen_report,
    ):
        reject_evaluation_path(path)
    if args.freeze and (args.proposals.exists() or args.frozen_report.exists()):
        raise FileExistsError("refusing to overwrite an already-frozen proposal")

    candidate = load_gt_free(args.candidates)
    junction = load_gt_free(args.junctions)
    rows = np.asarray(junction["row"], dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("junction feature rows are not unique")
    if np.any(rows < 0) or np.any(rows >= len(candidate["left"])):
        raise IndexError("junction feature row outside candidate table")
    if not np.array_equal(junction["left"], candidate["left"][rows]) or not np.array_equal(
        junction["right"], candidate["right"][rows]
    ):
        raise ValueError("junction labels do not align to candidate rows")
    if not np.all(np.asarray(junction["source"]) == 1):
        raise ValueError("junction scope must contain decoder-only source rows")
    expected = np.flatnonzero(
        candidate["affinity_ge08_fraction"] >= args.scope_min_affinity_ge08_fraction
    )
    if not np.array_equal(np.sort(rows), expected):
        raise ValueError("junction scope is not the exhaustive GT-free affinity selection")

    conditions = hard_gate_conditions(candidate, junction, rows, args)
    hard_gate = np.logical_and.reduce(list(conditions.values()))
    left = candidate["left"][rows]
    right = candidate["right"][rows]
    local_ambiguous = local_ambiguity_mask(
        left,
        right,
        junction["junction_zyx_nm"],
        hard_gate,
        args.local_competition_nm,
    )
    owners = load_nucleus_owners(args.nucleus_manifest)
    preliminary = hard_gate & ~local_ambiguous
    owner_conflict, component_too_large, component_size = component_firewall(
        left,
        right,
        preliminary,
        owners,
        args.max_component_segments,
    )
    accepted = preliminary & ~owner_conflict & ~component_too_large

    parameters = {
        name: getattr(args, name)
        for name in (
            "min_affinity_mean",
            "min_affinity_ge09_fraction",
            "scope_min_affinity_ge08_fraction",
            "max_junction_gap_nm",
            "max_leaf_distance_nm",
            "min_turn_short_deg",
            "min_radius_ratio",
            "local_competition_nm",
            "max_component_segments",
        )
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.audit,
        row=rows,
        left=left,
        right=right,
        junction_zyx_nm=junction["junction_zyx_nm"],
        gap_junction_nm=junction["gap_junction_nm"],
        turn_short_deg=junction["turn_short_deg"],
        radius_ratio=junction["radius_ratio"],
        a_leaf_dist_nm=junction["a_leaf_dist_nm"],
        b_leaf_dist_nm=junction["b_leaf_dist_nm"],
        spine_veto=junction["spine_veto"],
        affinity_mean=candidate["affinity_mean"][rows],
        affinity_ge08_fraction=candidate["affinity_ge08_fraction"][rows],
        affinity_ge09_fraction=candidate["affinity_ge09_fraction"][rows],
        **{f"gate_{key}": value for key, value in conditions.items()},
        hard_gate=hard_gate,
        local_ambiguous=local_ambiguous,
        owner_conflict=owner_conflict,
        component_too_large=component_too_large,
        component_size=component_size,
        accepted=accepted,
        gt_free=np.asarray(True),
    )
    report = {
        "scope": "GT-free hard-gated contact/junction merge correction; no score or GT",
        "inputs": {
            "candidates": str(args.candidates.resolve()),
            "junctions": str(args.junctions.resolve()),
            "nucleus_manifest": str(args.nucleus_manifest.resolve()),
        },
        "input_sha256": {
            "candidates": sha256_file(args.candidates),
            "junctions": sha256_file(args.junctions),
            "nucleus_manifest": sha256_file(args.nucleus_manifest),
        },
        "parameters": parameters,
        "selection": "all hard gates; local ties and multi-owner components are quarantined",
        "scope_pairs": int(len(rows)),
        "hard_gate_pairs": int(np.count_nonzero(hard_gate)),
        "local_ambiguous_pairs": int(np.count_nonzero(hard_gate & local_ambiguous)),
        "owner_conflict_pairs": int(np.count_nonzero(preliminary & owner_conflict)),
        "oversize_component_pairs": int(np.count_nonzero(preliminary & component_too_large)),
        "accepted_pairs": int(np.count_nonzero(accepted)),
        "independent_gate_counts": {
            key: int(np.count_nonzero(value)) for key, value in conditions.items()
        },
        "gt_free": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    if args.freeze:
        implementation_sha256 = sha256_file(Path(__file__))
        np.savez_compressed(
            args.proposals,
            left=left[accepted],
            right=right[accepted],
            junction_zyx_nm=junction["junction_zyx_nm"][accepted],
            candidate_row=rows[accepted],
            parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
            implementation_sha256=np.asarray(implementation_sha256),
            gt_free=np.asarray(True),
            frozen_before_evaluation=np.asarray(True),
        )
        proposal_sha256 = sha256_file(args.proposals)
        frozen = {
            **report,
            "scope": "GT-free proposal frozen before evaluation",
            "implementation_sha256": implementation_sha256,
            "proposal_sha256": proposal_sha256,
            "proposal_path": str(args.proposals.resolve()),
        }
        args.frozen_report.write_text(json.dumps(frozen, indent=2) + "\n")
        print(
            f"frozen {int(np.count_nonzero(accepted))} hard-gated junction merges "
            f"sha256={proposal_sha256}",
            flush=True,
        )
    else:
        print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
