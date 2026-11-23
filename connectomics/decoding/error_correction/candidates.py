#!/usr/bin/env python3
"""Rank GT-free arm0_96 merge proposals on the exact segmentation contact graph.

Touching final-segment pairs are candidates. A candidate can pass through either a strict
endpoint/contact-surface route or a smooth cubic interpolation between compatible non-spine
skeleton endpoints. The interpolation route does not use contact-face shape, normal, endpoint
proximity, or the straight-chord facing heuristic. High-confidence glia, multi-nucleus
segments, and pairs containing two distinct nucleus anchors are vetoed. Evaluation skeletons
are forbidden inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .artifacts import reject_evaluation_path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "dev" / "zebrafinch" / "arm096_error_correction" / "decoder_gtfree_v2"
DEFAULT_FEATURES = ROOT / "segment_morphology.npz"
DEFAULT_ENDPOINTS = ROOT / "segment_endpoints.npz"
DEFAULT_INTERIORS = ROOT / "segment_interiors.npz"
DEFAULT_CONTACTS = ROOT / "contact_graph.npz"
DEFAULT_CANDIDATES = ROOT / "contact_merge_candidates.npz"
DEFAULT_AUDIT = ROOT / "contact_merge_audit.json"
DEFAULT_PROPOSALS = ROOT / "frozen_contact_merges.npz"
DEFAULT_FROZEN_REPORT = ROOT / "frozen_contact_merges.json"
RESOLUTION_ZYX_NM = np.asarray([20.0, 9.0, 9.0])
FACE_AREA_NM2 = np.asarray([81.0, 180.0, 180.0])
TYPE_CODE = {"axon_axon": 1, "dendrite_dendrite": 2, "ambiguous": 3, "mixed": 4}


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gt_free(path: Path, *, require_complete: bool = False) -> dict[str, np.ndarray]:
    reject_evaluation_path(path)
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]) for name in data.files}
    if "gt_free" not in result or not bool(result["gt_free"].item()):
        raise ValueError(f"{path} is not explicitly marked GT-free")
    if require_complete and ("complete" not in result or not bool(result["complete"].item())):
        raise ValueError(f"{path} is not a complete whole-volume artifact")
    return result


def lookup(labels: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(labels)
    sorted_labels = labels[order]
    indices = np.searchsorted(sorted_labels, query)
    safe = np.minimum(indices, len(sorted_labels) - 1)
    valid = (indices < len(sorted_labels)) & (sorted_labels[safe] == query)
    return order[safe], valid


def group_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    if not len(order):
        return {}
    starts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1]
    return {
        int(sorted_labels[start]): order[start:stop]
        for start, stop in zip(starts, np.r_[starts[1:], len(order)])
    }


def caliber_class(radius_nm: float, branch_density: float) -> str:
    if radius_nm <= 300.0 and branch_density < 5.0:
        return "axon"
    if radius_nm >= 500.0 or branch_density >= 8.0:
        return "dendrite"
    return "ambiguous"


def connection_type(
    left_radius: float,
    right_radius: float,
    left_branch_density: float,
    right_branch_density: float,
) -> str:
    left_class = caliber_class(left_radius, left_branch_density)
    right_class = caliber_class(right_radius, right_branch_density)
    if left_class == right_class == "axon":
        return "axon_axon"
    if left_class == right_class == "dendrite":
        return "dendrite_dendrite"
    ratio = min(left_radius, right_radius) / max(left_radius, right_radius, 1e-6)
    if ratio >= 0.65 and "ambiguous" in (left_class, right_class):
        return "ambiguous"
    return "mixed"


def skeleton_interpolation_metrics(
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_outward_tangent: np.ndarray,
    right_outward_tangent: np.ndarray,
    samples: int = 33,
) -> dict[str, float]:
    """Fit the smoothest bounded cubic continuation between two skeleton endpoints.

    The curve leaves the left endpoint along its outward tangent and arrives at the right
    endpoint opposite its outward tangent. Several physical handle lengths are evaluated;
    the lowest-bending non-looping shape is reported. No contact-surface geometry is used.
    """
    left_position = np.asarray(left_position, dtype=np.float64)
    right_position = np.asarray(right_position, dtype=np.float64)
    delta = right_position - left_position
    gap = float(np.linalg.norm(delta))
    if gap <= 0:
        return {
            "interpolation_turn_radians": float("inf"),
            "interpolation_length_ratio": float("inf"),
            "interpolation_min_progress": -1.0,
            "interpolation_min_radius_nm": 0.0,
            "interpolation_handle_ratio": 0.0,
            "interpolation_score": 0.0,
        }
    chord = delta / gap
    left_tangent = np.asarray(left_outward_tangent, dtype=np.float64)
    left_tangent /= max(float(np.linalg.norm(left_tangent)), 1e-12)
    # Curve derivative at the right endpoint points into the object.
    right_tangent = -np.asarray(right_outward_tangent, dtype=np.float64)
    right_tangent /= max(float(np.linalg.norm(right_tangent)), 1e-12)
    u = np.linspace(0.0, 1.0, samples)
    u2 = u * u
    u3 = u2 * u
    best: tuple[float, dict[str, float]] | None = None
    for handle_ratio in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        left_handle = left_tangent * gap * handle_ratio
        right_handle = right_tangent * gap * handle_ratio
        points = (
            (2 * u3 - 3 * u2 + 1)[:, None] * left_position
            + (u3 - 2 * u2 + u)[:, None] * left_handle
            + (-2 * u3 + 3 * u2)[:, None] * right_position
            + (u3 - u2)[:, None] * right_handle
        )
        velocity = (
            (6 * u2 - 6 * u)[:, None] * left_position
            + (3 * u2 - 4 * u + 1)[:, None] * left_handle
            + (-6 * u2 + 6 * u)[:, None] * right_position
            + (3 * u2 - 2 * u)[:, None] * right_handle
        )
        acceleration = (
            (12 * u - 6)[:, None] * left_position
            + (6 * u - 4)[:, None] * left_handle
            + (-12 * u + 6)[:, None] * right_position
            + (6 * u - 2)[:, None] * right_handle
        )
        speed = np.linalg.norm(velocity, axis=1)
        if np.any(speed <= 1e-9):
            continue
        unit_velocity = velocity / speed[:, None]
        turn = float(
            np.arccos(
                np.clip(np.einsum("ij,ij->i", unit_velocity[:-1], unit_velocity[1:]), -1, 1)
            ).sum()
        )
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        length_ratio = length / gap
        min_progress = float(np.min(unit_velocity @ chord))
        curvature = np.linalg.norm(np.cross(velocity, acceleration), axis=1) / np.maximum(
            speed**3, 1e-12
        )
        max_curvature = float(np.max(curvature))
        min_radius = 1.0 / max(max_curvature, 1e-12)
        objective = (
            turn
            + 2.0 * max(length_ratio - 1.0, 0.0)
            + 0.25 * gap / max(min_radius, 1.0)
            + 5.0 * max(-min_progress, 0.0)
        )
        metrics = {
            "interpolation_turn_radians": turn,
            "interpolation_length_ratio": length_ratio,
            "interpolation_min_progress": min_progress,
            "interpolation_min_radius_nm": min_radius,
            "interpolation_handle_ratio": handle_ratio,
            "interpolation_score": math.exp(-objective),
        }
        candidate = (objective, metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return {
            "interpolation_turn_radians": float("inf"),
            "interpolation_length_ratio": float("inf"),
            "interpolation_min_progress": -1.0,
            "interpolation_min_radius_nm": 0.0,
            "interpolation_handle_ratio": 0.0,
            "interpolation_score": 0.0,
        }
    return best[1]


def best_endpoint_profile_pair(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    positions: np.ndarray,
    tangents: np.ndarray,
    radii: np.ndarray,
    radius_cv: np.ndarray,
    radius_alternation: np.ndarray,
    branch_density: np.ndarray,
    roles: np.ndarray,
    max_gap_nm: float,
    min_caliber_ratio: float,
    min_radius_profile_similarity: float,
    min_branch_similarity: float,
    max_interpolation_turn_radians: float = math.pi,
    max_interpolation_length_ratio: float = 1.5,
    min_interpolation_progress: float = 0.0,
    min_interpolation_radius_nm: float = 500.0,
) -> dict[str, float | int | bool | str] | None:
    """Choose a continuation using only bounded endpoint-local morphology.

    A touching segment pair can have several contacts and many endpoints. Selecting the
    endpoints nearest one representative contact patch is brittle for large, folded objects.
    This scan first looks for endpoint pairs that satisfy the physical continuation gates,
    then ranks them by tangent, radius-profile, branch-profile, and gap agreement.
    Whole-segment caliber and branch counts are intentionally absent.
    """
    if not len(left_indices) or not len(right_indices):
        return None
    # At most eight nearby endpoints per left endpoint are useful. More alternatives in a
    # 6-um ball indicate a locally ambiguous branch/crossing, and an all-pairs product can be
    # enormous for bushy objects.
    from scipy.spatial import cKDTree

    tree = cKDTree(positions[right_indices])
    neighbor_count = min(8, len(right_indices))
    neighbor_gap, neighbor_local = tree.query(
        positions[left_indices],
        k=neighbor_count,
        distance_upper_bound=max_gap_nm,
    )
    neighbor_gap = np.asarray(neighbor_gap).reshape(len(left_indices), neighbor_count)
    neighbor_local = np.asarray(neighbor_local).reshape(len(left_indices), neighbor_count)
    left_local = np.repeat(np.arange(len(left_indices)), neighbor_count)
    right_local = neighbor_local.ravel()
    nearby = np.isfinite(neighbor_gap.ravel()) & (right_local < len(right_indices))
    if np.any(nearby):
        left_flat = left_indices[left_local[nearby]]
        right_flat = right_indices[right_local[nearby]]
    else:
        # Retain the closest pair for diagnostics even though it must fail the gap gate.
        closest_gap, closest_local = tree.query(positions[left_indices], k=1)
        closest_left = int(np.argmin(closest_gap))
        left_flat = np.asarray([left_indices[closest_left]])
        right_flat = np.asarray([right_indices[int(closest_local[closest_left])]])
    delta = positions[right_flat] - positions[left_flat]
    gap = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(gap) & (gap > 0)
    direction = np.zeros_like(delta)
    direction[finite] = delta[finite] / gap[finite, None]
    left_tangent = tangents[left_flat]
    left_tangent = left_tangent / np.maximum(
        np.linalg.norm(left_tangent, axis=1, keepdims=True), 1e-12
    )
    right_tangent = tangents[right_flat]
    right_tangent = right_tangent / np.maximum(
        np.linalg.norm(right_tangent, axis=1, keepdims=True), 1e-12
    )
    facing = np.minimum(
        np.einsum("ij,ij->i", left_tangent, direction),
        np.einsum("ij,ij->i", right_tangent, -direction),
    )
    collinear = -np.einsum("ij,ij->i", left_tangent, right_tangent)
    left_radius = np.maximum(radii[left_flat], 1.0)
    right_radius = np.maximum(radii[right_flat], 1.0)
    caliber_ratio = np.minimum(left_radius, right_radius) / np.maximum(left_radius, right_radius)
    cv_similarity = np.exp(
        -np.abs(np.log1p(radius_cv[left_flat]) - np.log1p(radius_cv[right_flat]))
    )
    alternation_similarity = 1.0 - np.minimum(
        np.abs(radius_alternation[left_flat] - radius_alternation[right_flat]), 1.0
    )
    radius_profile_similarity = (
        0.60 * caliber_ratio + 0.25 * cv_similarity + 0.15 * alternation_similarity
    )
    branch_similarity = np.exp(
        -np.abs(np.log1p(branch_density[left_flat]) - np.log1p(branch_density[right_flat]))
    )
    kinds = np.asarray(
        [
            connection_type(
                float(left_radius[index]),
                float(right_radius[index]),
                float(branch_density[left_flat[index]]),
                float(branch_density[right_flat[index]]),
            )
            for index in range(len(left_flat))
        ]
    )
    spine_veto = roles[left_flat].astype(bool) | roles[right_flat].astype(bool)
    qualified = (
        finite
        & (gap <= max_gap_nm)
        & (collinear >= 0.0)
        & (caliber_ratio >= min_caliber_ratio)
        & (radius_profile_similarity >= min_radius_profile_similarity)
        & (branch_similarity >= min_branch_similarity)
        & np.isin(kinds, ("axon_axon", "dendrite_dendrite"))
        & ~spine_veto
    )
    gap_similarity = np.exp(-gap / max(max_gap_nm, 1.0))
    score = (
        0.40 * np.clip((collinear + 1.0) / 2.0, 0.0, 1.0)
        + 0.25 * radius_profile_similarity
        + 0.20 * branch_similarity
        + 0.15 * gap_similarity
    )
    pool = np.flatnonzero(qualified)
    if not len(pool):
        pool = np.flatnonzero(finite)
    if not len(pool):
        return None
    # Evaluate only the strongest local-profile alternatives with the more expensive cubic
    # interpolation. This avoids choosing a closer crossing whose tangents require a loop.
    ordered = pool[np.argsort(score[pool])[-min(16, len(pool)) :]]
    interpolation_by_index: dict[int, dict[str, float]] = {}
    smooth: list[int] = []
    for index in ordered.tolist():
        metrics = skeleton_interpolation_metrics(
            positions[left_flat[index]],
            positions[right_flat[index]],
            tangents[left_flat[index]],
            tangents[right_flat[index]],
        )
        interpolation_by_index[index] = metrics
        if (
            metrics["interpolation_turn_radians"] <= max_interpolation_turn_radians
            and metrics["interpolation_length_ratio"] <= max_interpolation_length_ratio
            and metrics["interpolation_min_progress"] >= min_interpolation_progress
            and metrics["interpolation_min_radius_nm"] >= min_interpolation_radius_nm
        ):
            smooth.append(index)
    if smooth:
        chosen = max(
            smooth,
            key=lambda index: (
                0.6 * float(score[index])
                + 0.4 * interpolation_by_index[index]["interpolation_score"],
                -float(gap[index]),
            ),
        )
    else:
        chosen = int(pool[int(np.argmax(score[pool]))])
    chosen_interpolation = interpolation_by_index.get(chosen)
    if chosen_interpolation is None:
        chosen_interpolation = skeleton_interpolation_metrics(
            positions[left_flat[chosen]],
            positions[right_flat[chosen]],
            tangents[left_flat[chosen]],
            tangents[right_flat[chosen]],
        )
    return {
        "left": int(left_flat[chosen]),
        "right": int(right_flat[chosen]),
        "gap_nm": float(gap[chosen]),
        "facing": float(facing[chosen]),
        "collinear": float(collinear[chosen]),
        "caliber_ratio": float(caliber_ratio[chosen]),
        "radius_cv_similarity": float(cv_similarity[chosen]),
        "radius_alternation_similarity": float(alternation_similarity[chosen]),
        "radius_profile_similarity": float(radius_profile_similarity[chosen]),
        "branch_similarity": float(branch_similarity[chosen]),
        "kind": str(kinds[chosen]),
        "spine_veto": bool(spine_veto[chosen]),
        "qualified": bool(qualified[chosen]),
        "score": float(score[chosen]),
        **chosen_interpolation,
    }


def nearest(index: np.ndarray, positions: np.ndarray, point: np.ndarray) -> tuple[int, float]:
    distance = np.linalg.norm(positions[index] - point, axis=1)
    local = int(np.argmin(distance))
    return int(index[local]), float(distance[local])


def distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {
        name: float(np.quantile(finite, quantile))
        for name, quantile in (
            ("min", 0.0),
            ("p10", 0.10),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("max", 1.0),
        )
    }


def select_internal_orphans(
    candidate: dict[str, np.ndarray],
    gated: np.ndarray,
    min_target_length_ratio: float,
    min_target_voxel_ratio: float,
    ambiguity_margin: float,
    competition: dict[str, np.ndarray] | None = None,
    min_competition_margin: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Return side-coded orphan sources and each source's unambiguous best join."""
    orphan_source = np.zeros(len(candidate["left"]), dtype=np.uint8)  # 1=left, 2=right
    left_source = (
        candidate["left_internal"]
        & ~candidate["left_nucleus_anchor"]
        & (candidate["right_length_nm"] >= candidate["left_length_nm"] * min_target_length_ratio)
        & (candidate["right_voxels"] >= candidate["left_voxels"] * min_target_voxel_ratio)
    )
    right_source = (
        candidate["right_internal"]
        & ~candidate["right_nucleus_anchor"]
        & (candidate["left_length_nm"] >= candidate["right_length_nm"] * min_target_length_ratio)
        & (candidate["left_voxels"] >= candidate["right_voxels"] * min_target_voxel_ratio)
    )
    orphan_source[left_source & ~right_source] = 1
    orphan_source[right_source & ~left_source] = 2
    if competition is not None:
        source_label = np.where(orphan_source == 1, candidate["left"], candidate["right"])
        target_label = np.where(orphan_source == 1, candidate["right"], candidate["left"])
        competition_index, competition_valid = lookup(competition["source"], source_label)
        supported = (
            competition_valid
            & (competition["best_target"][competition_index] == target_label)
            & (competition["affinity_margin"][competition_index] >= min_competition_margin)
        )
        orphan_source[~supported] = 0

    incident: dict[int, list[int]] = {}
    for index in gated.tolist():
        side = int(orphan_source[index])
        if side == 0:
            continue
        source = int(candidate["left"][index] if side == 1 else candidate["right"][index])
        incident.setdefault(source, []).append(index)
    preliminary: list[int] = []
    for indices in incident.values():
        ordered = sorted(indices, key=lambda index: (-float(candidate["score"][index]), index))
        runner = float(candidate["score"][ordered[1]]) if len(ordered) > 1 else 0.0
        if float(candidate["score"][ordered[0]]) - runner >= ambiguity_margin:
            preliminary.append(ordered[0])
    return orphan_source, preliminary


def tiered_matching(
    candidate: dict[str, np.ndarray], orphan: list[int], mutual: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Prioritize the completeness prior, then keep a one-edge-per-label first round."""
    accepted: list[int] = []
    accepted_type = np.zeros(len(candidate["left"]), dtype=np.uint8)
    used_labels: set[int] = set()
    for proposal_type, indices in ((2, orphan), (1, mutual)):
        for index in sorted(indices, key=lambda value: (-float(candidate["score"][value]), value)):
            left_label = int(candidate["left"][index])
            right_label = int(candidate["right"][index])
            if left_label in used_labels or right_label in used_labels:
                continue
            accepted.append(index)
            accepted_type[index] = proposal_type
            used_labels.update((left_label, right_label))
    return np.asarray(sorted(accepted), dtype=np.int64), accepted_type


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--interiors", type=Path, default=DEFAULT_INTERIORS)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--rg-competition", type=Path)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--min-segment-length-nm", type=float, default=10_000.0)
    parser.add_argument("--max-endpoint-contact-nm", type=float, default=4_000.0)
    parser.add_argument("--max-endpoint-gap-nm", type=float, default=6_000.0)
    parser.add_argument("--endpoint-side-margin-nm", type=float, default=750.0)
    parser.add_argument("--min-facing", type=float, default=0.25)
    parser.add_argument("--min-collinear", type=float, default=0.25)
    parser.add_argument("--min-caliber-ratio", type=float, default=0.45)
    parser.add_argument("--min-radius-profile-similarity", type=float, default=0.55)
    parser.add_argument("--min-branch-similarity", type=float, default=0.35)
    parser.add_argument("--max-interpolation-turn-radians", type=float, default=math.pi)
    parser.add_argument("--max-interpolation-length-ratio", type=float, default=1.5)
    parser.add_argument("--min-interpolation-progress", type=float, default=0.0)
    parser.add_argument("--min-interpolation-radius-nm", type=float, default=500.0)
    parser.add_argument("--min-normal-coherence", type=float, default=0.15)
    parser.add_argument("--min-normal-alignment", type=float, default=0.10)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-area-ratio", type=float, default=20.0)
    parser.add_argument("--min-affinity-ge08-fraction", type=float, default=0.10)
    # Independent physical/affinity gates decide; this score only ranks competitors.
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--enable-internal-orphan", action="store_true")
    parser.add_argument("--min-orphan-target-length-ratio", type=float, default=1.25)
    parser.add_argument("--min-orphan-target-voxel-ratio", type=float, default=2.0)
    parser.add_argument("--orphan-ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--min-orphan-rg-affinity-margin", type=float, default=0.05)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    for path in (
        args.features,
        args.endpoints,
        args.interiors,
        args.contacts,
        args.candidates,
        args.audit_report,
        args.proposals,
        args.frozen_report,
    ):
        reject_evaluation_path(path)
    if args.rg_competition is not None:
        reject_evaluation_path(args.rg_competition)
    if args.enable_internal_orphan and args.rg_competition is None:
        raise ValueError("internal-orphan mode requires --rg-competition")
    if args.freeze and (args.proposals.exists() or args.frozen_report.exists()):
        raise FileExistsError("refusing to overwrite an already-frozen proposal")

    features = load_gt_free(args.features)
    endpoints = load_gt_free(args.endpoints)
    interiors = load_gt_free(args.interiors)
    contacts = load_gt_free(args.contacts, require_complete=True)
    competition = load_gt_free(args.rg_competition) if args.rg_competition is not None else None
    if competition is not None:
        required_competition = {"source", "best_target", "affinity_margin"}
        missing_competition = required_competition - set(competition)
        if missing_competition:
            raise ValueError(f"region-graph competition lacks {missing_competition}")
        if not np.all(competition["source"][:-1] < competition["source"][1:]):
            raise ValueError("region-graph competition sources must be unique and sorted")
        competition_ratio = float(competition["min_target_voxel_ratio"].item())
        if not math.isclose(competition_ratio, args.min_orphan_target_voxel_ratio):
            raise ValueError(
                "region-graph competition target ratio differs from orphan target ratio: "
                f"{competition_ratio} != {args.min_orphan_target_voxel_ratio}"
            )
    feature_labels = np.asarray(features["label"], dtype=np.uint64)
    left_labels = np.asarray(contacts["left"], dtype=np.uint64)
    right_labels = np.asarray(contacts["right"], dtype=np.uint64)
    left_feature, left_valid = lookup(feature_labels, left_labels)
    right_feature, right_valid = lookup(feature_labels, right_labels)
    length = np.asarray(features["largest_component_length_nm"], dtype=np.float64)
    voxels = np.asarray(features["voxels"], dtype=np.uint64)
    glia = np.asarray(features["glia_quarantine"], dtype=bool)
    nucleus_anchor = np.asarray(features["nucleus_anchor"], dtype=bool)
    bushiness = np.asarray(features["bushiness_score"], dtype=np.float64)
    feature_internal = np.zeros(len(feature_labels), dtype=bool)
    if args.enable_internal_orphan:
        required_boundary_fields = {
            "segment_label",
            "touches_volume_boundary",
            "touches_keep_boundary",
            "complete_boundary_inventory",
        }
        missing_boundary_fields = required_boundary_fields - set(contacts)
        if missing_boundary_fields:
            raise ValueError(
                f"internal-orphan mode requires boundary inventory: {missing_boundary_fields}"
            )
        if not bool(contacts["complete_boundary_inventory"].item()):
            raise ValueError("internal-orphan mode requires a complete boundary inventory")
        boundary_labels = np.asarray(contacts["segment_label"], dtype=np.uint64)
        boundary_feature, boundary_valid = lookup(boundary_labels, feature_labels)
        if not np.all(boundary_valid):
            missing = feature_labels[~boundary_valid]
            raise ValueError(
                f"boundary inventory lacks {len(missing):,} morphology labels; "
                f"examples={missing[:10].tolist()}"
            )
        feature_internal = (
            ~np.asarray(contacts["touches_volume_boundary"], dtype=bool)[boundary_feature]
            & ~np.asarray(contacts["touches_keep_boundary"], dtype=bool)[boundary_feature]
        )
    valid = (
        left_valid
        & right_valid
        & (length[left_feature] >= args.min_segment_length_nm)
        & (length[right_feature] >= args.min_segment_length_nm)
        & ~glia[left_feature]
        & ~glia[right_feature]
        & ~(nucleus_anchor[left_feature] & nucleus_anchor[right_feature])
    )
    contact_indices = np.flatnonzero(valid)
    affinity_thresholds = np.asarray(contacts["affinity_thresholds"], dtype=np.float32)
    affinity_08 = np.flatnonzero(np.isclose(affinity_thresholds, 0.8))
    affinity_09 = np.flatnonzero(np.isclose(affinity_thresholds, 0.9))
    if len(affinity_08) != 1 or len(affinity_09) != 1:
        raise ValueError("contact graph must contain affinity thresholds 0.8 and 0.9")
    required_patch_fields = {
        "best_patch_count",
        "best_patch_centroid_nm_zyx",
        "best_patch_normal_sum_zyx",
        "best_patch_area_nm2",
        "best_patch_affinity_sum",
        "best_patch_affinity_sq_sum",
        "best_patch_affinity_ge_count",
        "representative_patch_policy",
    }
    missing_patch_fields = required_patch_fields - set(contacts)
    if missing_patch_fields:
        raise ValueError(f"contact graph lacks representative patches: {missing_patch_fields}")
    log(
        f"contact pairs={len(left_labels):,} morphology-eligible={len(contact_indices):,}",
        "glia/nucleus firewall applied",
    )

    endpoint_labels = np.asarray(endpoints["label"], dtype=np.uint64)
    endpoint_positions = np.asarray(endpoints["position_nm"], dtype=np.float64)
    endpoint_tangents = np.asarray(endpoints["outward_tangent"], dtype=np.float64)
    endpoint_groups = group_indices(endpoint_labels)
    interior_labels = np.asarray(interiors["label"], dtype=np.uint64)
    interior_positions = np.asarray(interiors["position_nm"], dtype=np.float64)
    interior_groups = group_indices(interior_labels)
    endpoint_feature, endpoint_feature_valid = lookup(feature_labels, endpoint_labels)
    if not np.all(endpoint_feature_valid):
        raise ValueError("an endpoint label is absent from morphology features")
    required_local_profiles = {
        "profile_radius_median_nm",
        "profile_radius_cv",
        "profile_radius_alternation_fraction",
        "profile_branch_density_per_100um",
    }
    missing_local_profiles = required_local_profiles - set(endpoints)
    if missing_local_profiles:
        raise ValueError(
            "endpoint-local profile artifact is required; refusing whole-segment fallback: "
            f"{missing_local_profiles}"
        )
    endpoint_radius = np.maximum(
        np.asarray(endpoints["profile_radius_median_nm"], dtype=np.float64), 1.0
    )
    branch_density = np.asarray(endpoints["profile_branch_density_per_100um"], dtype=np.float64)
    radius_cv = np.asarray(endpoints["profile_radius_cv"], dtype=np.float64)
    radius_alternation = np.asarray(
        endpoints["profile_radius_alternation_fraction"], dtype=np.float64
    )
    endpoint_roles = np.asarray(endpoints["role"], dtype=np.uint8)

    rows: list[tuple[object, ...]] = []
    for number, contact_index in enumerate(contact_indices.tolist(), 1):
        left_label = int(left_labels[contact_index])
        right_label = int(right_labels[contact_index])
        left_group = endpoint_groups.get(left_label)
        right_group = endpoint_groups.get(right_label)
        if left_group is None or right_group is None:
            continue
        contact = np.asarray(
            contacts["best_patch_centroid_nm_zyx"][contact_index], dtype=np.float64
        )
        profile_pair = best_endpoint_profile_pair(
            left_group,
            right_group,
            endpoint_positions,
            endpoint_tangents,
            endpoint_radius,
            radius_cv,
            radius_alternation,
            branch_density,
            endpoint_roles,
            args.max_endpoint_gap_nm,
            args.min_caliber_ratio,
            args.min_radius_profile_similarity,
            args.min_branch_similarity,
            args.max_interpolation_turn_radians,
            args.max_interpolation_length_ratio,
            args.min_interpolation_progress,
            args.min_interpolation_radius_nm,
        )
        if profile_pair is None:
            continue
        left_endpoint = int(profile_pair["left"])
        right_endpoint = int(profile_pair["right"])
        left_contact_distance = float(np.linalg.norm(endpoint_positions[left_endpoint] - contact))
        right_contact_distance = float(np.linalg.norm(endpoint_positions[right_endpoint] - contact))
        left_interior_group = interior_groups.get(left_label)
        right_interior_group = interior_groups.get(right_label)
        left_interior_distance = (
            nearest(left_interior_group, interior_positions, contact)[1]
            if left_interior_group is not None
            else float("inf")
        )
        right_interior_distance = (
            nearest(right_interior_group, interior_positions, contact)[1]
            if right_interior_group is not None
            else float("inf")
        )
        delta = endpoint_positions[right_endpoint] - endpoint_positions[left_endpoint]
        gap = float(profile_pair["gap_nm"])
        direction = delta / gap
        facing = float(profile_pair["facing"])
        collinear = float(profile_pair["collinear"])
        left_radius = float(endpoint_radius[left_endpoint])
        right_radius = float(endpoint_radius[right_endpoint])
        left_radius_cv = float(radius_cv[left_endpoint])
        right_radius_cv = float(radius_cv[right_endpoint])
        left_radius_alternation = float(radius_alternation[left_endpoint])
        right_radius_alternation = float(radius_alternation[right_endpoint])
        caliber_ratio = float(profile_pair["caliber_ratio"])
        left_branch_density = float(branch_density[left_endpoint])
        right_branch_density = float(branch_density[right_endpoint])
        branch_similarity = float(profile_pair["branch_similarity"])
        radius_profile_similarity = float(profile_pair["radius_profile_similarity"])
        kind = str(profile_pair["kind"])
        interpolation = profile_pair
        area = float(contacts["best_patch_area_nm2"][contact_index])
        area_ratio = area / max(math.pi * min(left_radius, right_radius) ** 2, 1.0)
        area_normal = (
            np.asarray(contacts["best_patch_normal_sum_zyx"][contact_index], dtype=np.float64)
            * FACE_AREA_NM2
        )
        normal_magnitude = float(np.linalg.norm(area_normal))
        normal_coherence = normal_magnitude / max(area, 1.0)
        normal_alignment = (
            float(np.dot(area_normal / normal_magnitude, direction))
            if normal_magnitude > 0
            else 0.0
        )
        endpoint_endpoint = (
            left_contact_distance <= left_interior_distance + args.endpoint_side_margin_nm
            and right_contact_distance <= right_interior_distance + args.endpoint_side_margin_nm
        )
        spine_veto = bool(profile_pair["spine_veto"])
        max_contact_distance = max(left_contact_distance, right_contact_distance)
        max_bushiness = max(
            float(bushiness[left_feature[contact_index]]),
            float(bushiness[right_feature[contact_index]]),
        )
        affinity_count = max(float(contacts["best_patch_count"][contact_index]), 1.0)
        affinity_mean = float(contacts["best_patch_affinity_sum"][contact_index]) / affinity_count
        affinity_variance = max(
            float(contacts["best_patch_affinity_sq_sum"][contact_index]) / affinity_count
            - affinity_mean**2,
            0.0,
        )
        affinity_std = math.sqrt(affinity_variance)
        affinity_ge08_fraction = (
            float(contacts["best_patch_affinity_ge_count"][contact_index, affinity_08[0]])
            / affinity_count
        )
        affinity_ge09_fraction = (
            float(contacts["best_patch_affinity_ge_count"][contact_index, affinity_09[0]])
            / affinity_count
        )
        left_feature_index = int(left_feature[contact_index])
        right_feature_index = int(right_feature[contact_index])
        area_score = math.exp(-abs(math.log(max(area_ratio, 1e-6))))
        contact_score = float(
            0.14 * math.exp(-max_contact_distance / 2_500.0)
            + 0.07 * math.exp(-gap / 5_000.0)
            + 0.10 * np.clip((facing + 1.0) / 2.0, 0.0, 1.0)
            + 0.07 * np.clip((collinear + 1.0) / 2.0, 0.0, 1.0)
            + 0.10 * np.clip(normal_alignment, 0.0, 1.0)
            + 0.07 * np.clip(normal_coherence, 0.0, 1.0)
            + 0.08 * radius_profile_similarity
            + 0.06 * area_score
            + 0.05 * (1.0 - max_bushiness)
            + 0.14 * affinity_mean
            + 0.12 * affinity_ge08_fraction
        )
        skeleton_score = float(
            0.25 * float(interpolation["interpolation_score"])
            + 0.15 * math.exp(-gap / 5_000.0)
            + 0.15 * np.clip((collinear + 1.0) / 2.0, 0.0, 1.0)
            + 0.15 * radius_profile_similarity
            + 0.10 * branch_similarity
            + 0.10 * affinity_mean
            + 0.10 * affinity_ge08_fraction
        )
        common_profile_gate = (
            not spine_veto
            and kind in ("axon_axon", "dendrite_dendrite")
            and gap <= args.max_endpoint_gap_nm
            and caliber_ratio >= args.min_caliber_ratio
            and radius_profile_similarity >= args.min_radius_profile_similarity
            and branch_similarity >= args.min_branch_similarity
        )
        contact_surface_route = (
            endpoint_endpoint
            and max_contact_distance <= args.max_endpoint_contact_nm
            and facing >= args.min_facing
            and collinear >= args.min_collinear
            and normal_coherence >= args.min_normal_coherence
            and normal_alignment >= args.min_normal_alignment
            and args.min_area_ratio <= area_ratio <= args.max_area_ratio
        )
        skeleton_interpolation_route = (
            float(interpolation["interpolation_turn_radians"])
            <= args.max_interpolation_turn_radians
            and float(interpolation["interpolation_length_ratio"])
            <= args.max_interpolation_length_ratio
            and float(interpolation["interpolation_min_progress"])
            >= args.min_interpolation_progress
            and float(interpolation["interpolation_min_radius_nm"])
            >= args.min_interpolation_radius_nm
        )
        score = max(contact_score, skeleton_score)
        gate = (
            common_profile_gate
            and affinity_ge08_fraction >= args.min_affinity_ge08_fraction
            and (contact_surface_route or skeleton_interpolation_route)
            and score >= args.min_score
        )
        rows.append(
            (
                contact_index,
                left_label,
                right_label,
                left_endpoint,
                right_endpoint,
                TYPE_CODE[kind],
                left_contact_distance,
                right_contact_distance,
                left_interior_distance,
                right_interior_distance,
                gap,
                facing,
                collinear,
                caliber_ratio,
                left_radius,
                right_radius,
                left_radius_cv,
                right_radius_cv,
                left_radius_alternation,
                right_radius_alternation,
                float(profile_pair["radius_cv_similarity"]),
                float(profile_pair["radius_alternation_similarity"]),
                radius_profile_similarity,
                left_branch_density,
                right_branch_density,
                branch_similarity,
                float(profile_pair["score"]),
                bool(profile_pair["qualified"]),
                float(interpolation["interpolation_turn_radians"]),
                float(interpolation["interpolation_length_ratio"]),
                float(interpolation["interpolation_min_progress"]),
                float(interpolation["interpolation_min_radius_nm"]),
                float(interpolation["interpolation_handle_ratio"]),
                float(interpolation["interpolation_score"]),
                area,
                area_ratio,
                normal_coherence,
                normal_alignment,
                max_bushiness,
                affinity_mean,
                affinity_std,
                affinity_ge08_fraction,
                affinity_ge09_fraction,
                endpoint_endpoint,
                spine_veto,
                contact_score,
                skeleton_score,
                contact_surface_route,
                skeleton_interpolation_route,
                score,
                gate,
                feature_internal[left_feature_index],
                feature_internal[right_feature_index],
                nucleus_anchor[left_feature_index],
                nucleus_anchor[right_feature_index],
                length[left_feature_index],
                length[right_feature_index],
                voxels[left_feature_index],
                voxels[right_feature_index],
            )
        )
        if number % 100_000 == 0:
            log(f"ranked {number:,}/{len(contact_indices):,} rows={len(rows):,}")

    names = (
        "contact_index",
        "left",
        "right",
        "endpoint_left",
        "endpoint_right",
        "kind",
        "left_contact_nm",
        "right_contact_nm",
        "left_interior_nm",
        "right_interior_nm",
        "gap_nm",
        "facing",
        "collinear",
        "caliber_ratio",
        "left_local_radius_nm",
        "right_local_radius_nm",
        "left_local_radius_cv",
        "right_local_radius_cv",
        "left_local_radius_alternation",
        "right_local_radius_alternation",
        "radius_cv_similarity",
        "radius_alternation_similarity",
        "radius_profile_similarity",
        "left_local_branch_density_per_100um",
        "right_local_branch_density_per_100um",
        "local_branch_similarity",
        "endpoint_profile_score",
        "endpoint_profile_gate",
        "interpolation_turn_radians",
        "interpolation_length_ratio",
        "interpolation_min_progress",
        "interpolation_min_radius_nm",
        "interpolation_handle_ratio",
        "interpolation_score",
        "area_nm2",
        "area_ratio",
        "normal_coherence",
        "normal_alignment",
        "max_bushiness",
        "affinity_mean",
        "affinity_std",
        "affinity_ge08_fraction",
        "affinity_ge09_fraction",
        "endpoint_endpoint",
        "spine_veto",
        "contact_score",
        "skeleton_score",
        "contact_surface_route",
        "skeleton_interpolation_route",
        "score",
        "gate",
        "left_internal",
        "right_internal",
        "left_nucleus_anchor",
        "right_nucleus_anchor",
        "left_length_nm",
        "right_length_nm",
        "left_voxels",
        "right_voxels",
    )
    dtypes = (
        np.int64,
        np.uint64,
        np.uint64,
        np.int64,
        np.int64,
        np.uint8,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        bool,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float64,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        bool,
        bool,
        np.float32,
        np.float32,
        bool,
        bool,
        np.float32,
        bool,
        bool,
        bool,
        bool,
        bool,
        np.float32,
        np.float32,
        np.uint64,
        np.uint64,
    )
    candidate = {
        name: np.asarray([row[index] for row in rows], dtype=dtype)
        for index, (name, dtype) in enumerate(zip(names, dtypes))
    }
    gated = np.flatnonzero(candidate["gate"])
    incident: dict[int, list[int]] = {}
    for index in gated.tolist():
        incident.setdefault(int(candidate["left"][index]), []).append(index)
        incident.setdefault(int(candidate["right"][index]), []).append(index)
    best: dict[int, int] = {}
    runner_up: dict[int, float] = {}
    for label, indices in incident.items():
        ordered = sorted(indices, key=lambda index: (-float(candidate["score"][index]), index))
        best[label] = ordered[0]
        runner_up[label] = float(candidate["score"][ordered[1]]) if len(ordered) > 1 else 0.0
    mutual = []
    for index in gated.tolist():
        left_label = int(candidate["left"][index])
        right_label = int(candidate["right"][index])
        if best.get(left_label) != index or best.get(right_label) != index:
            continue
        margin = float(candidate["score"][index]) - max(
            runner_up[left_label], runner_up[right_label]
        )
        if margin >= args.ambiguity_margin:
            mutual.append(index)

    # A sufficiently long prediction segment that touches neither the physical volume nor the
    # tissue keep-mask boundary and owns no nucleus is incomplete by construction. It may claim
    # a larger continuation without reciprocal-best status, but only when the same strict
    # endpoint/contact gates leave one clearly superior target. This is the alternating
    # big/small branch case; it never turns the completeness prior into blind nearest-neighbor
    # absorption.
    orphan_source = np.zeros(len(rows), dtype=np.uint8)  # 1=left, 2=right
    orphan_preliminary: list[int] = []
    if args.enable_internal_orphan:
        orphan_source, orphan_preliminary = select_internal_orphans(
            candidate,
            gated,
            args.min_orphan_target_length_ratio,
            args.min_orphan_target_voxel_ratio,
            args.orphan_ambiguity_margin,
            competition,
            args.min_orphan_rg_affinity_margin,
        )

    # First round is deliberately a matching: no segment participates in two accepted joins.
    # This prevents a chain of individually legal pairs from joining two nucleus anchors or
    # allowing a connector fragment to bridge two unrelated processes in one evaluation.
    accepted, accepted_type = tiered_matching(candidate, orphan_preliminary, mutual)

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.candidates,
        **candidate,
        accepted=np.isin(np.arange(len(rows)), accepted),
        accepted_type=accepted_type,
        orphan_source=orphan_source,
        gt_free=np.asarray(True),
    )
    numeric_audit = {
        name: distribution(candidate[name].astype(np.float64))
        for name in (
            "left_contact_nm",
            "right_contact_nm",
            "gap_nm",
            "facing",
            "collinear",
            "caliber_ratio",
            "radius_profile_similarity",
            "local_branch_similarity",
            "endpoint_profile_score",
            "interpolation_turn_radians",
            "interpolation_length_ratio",
            "interpolation_min_progress",
            "interpolation_min_radius_nm",
            "interpolation_score",
            "area_nm2",
            "area_ratio",
            "normal_coherence",
            "normal_alignment",
            "max_bushiness",
            "affinity_mean",
            "affinity_std",
            "affinity_ge08_fraction",
            "affinity_ge09_fraction",
            "contact_score",
            "skeleton_score",
            "score",
        )
    }
    diagnostic_conditions = {
        "endpoint_endpoint": candidate["endpoint_endpoint"],
        "endpoint_contact_distance": np.maximum(
            candidate["left_contact_nm"], candidate["right_contact_nm"]
        )
        <= args.max_endpoint_contact_nm,
        "endpoint_gap": candidate["gap_nm"] <= args.max_endpoint_gap_nm,
        "facing": candidate["facing"] >= args.min_facing,
        "collinear": candidate["collinear"] >= args.min_collinear,
        "caliber_ratio": candidate["caliber_ratio"] >= args.min_caliber_ratio,
        "radius_profile_similarity": candidate["radius_profile_similarity"]
        >= args.min_radius_profile_similarity,
        "local_branch_similarity": candidate["local_branch_similarity"]
        >= args.min_branch_similarity,
        "normal_coherence": candidate["normal_coherence"] >= args.min_normal_coherence,
        "normal_alignment": candidate["normal_alignment"] >= args.min_normal_alignment,
        "contact_area_ratio": (candidate["area_ratio"] >= args.min_area_ratio)
        & (candidate["area_ratio"] <= args.max_area_ratio),
        "interpolation_turn": candidate["interpolation_turn_radians"]
        <= args.max_interpolation_turn_radians,
        "interpolation_length": candidate["interpolation_length_ratio"]
        <= args.max_interpolation_length_ratio,
        "interpolation_progress": candidate["interpolation_min_progress"]
        >= args.min_interpolation_progress,
        "interpolation_curvature": candidate["interpolation_min_radius_nm"]
        >= args.min_interpolation_radius_nm,
        "contact_surface_route": candidate["contact_surface_route"],
        "skeleton_interpolation_route": candidate["skeleton_interpolation_route"],
    }
    gate_conditions = {
        "not_spine": ~candidate["spine_veto"],
        "same_process_class": np.isin(
            candidate["kind"],
            (TYPE_CODE["axon_axon"], TYPE_CODE["dendrite_dendrite"]),
        ),
        "endpoint_gap": candidate["gap_nm"] <= args.max_endpoint_gap_nm,
        "caliber_ratio": candidate["caliber_ratio"] >= args.min_caliber_ratio,
        "radius_profile_similarity": candidate["radius_profile_similarity"]
        >= args.min_radius_profile_similarity,
        "local_branch_similarity": candidate["local_branch_similarity"]
        >= args.min_branch_similarity,
        "affinity_p90_proxy": candidate["affinity_ge08_fraction"]
        >= args.min_affinity_ge08_fraction,
        "contact_or_interpolation_route": candidate["contact_surface_route"]
        | candidate["skeleton_interpolation_route"],
        "score": candidate["score"] >= args.min_score,
    }
    cumulative = np.ones(len(rows), dtype=bool)
    cumulative_gate_counts = {}
    for name, condition in gate_conditions.items():
        cumulative &= condition
        cumulative_gate_counts[name] = int(np.count_nonzero(cumulative))
    if not np.array_equal(cumulative, candidate["gate"]):
        raise AssertionError("reported gate decomposition differs from proposal gate")
    parameters = {
        name: getattr(args, name)
        for name in (
            "min_segment_length_nm",
            "max_endpoint_contact_nm",
            "max_endpoint_gap_nm",
            "endpoint_side_margin_nm",
            "min_facing",
            "min_collinear",
            "min_caliber_ratio",
            "min_radius_profile_similarity",
            "min_branch_similarity",
            "max_interpolation_turn_radians",
            "max_interpolation_length_ratio",
            "min_interpolation_progress",
            "min_interpolation_radius_nm",
            "min_normal_coherence",
            "min_normal_alignment",
            "min_area_ratio",
            "max_area_ratio",
            "min_affinity_ge08_fraction",
            "min_score",
            "ambiguity_margin",
            "enable_internal_orphan",
            "min_orphan_target_length_ratio",
            "min_orphan_target_voxel_ratio",
            "orphan_ambiguity_margin",
            "min_orphan_rg_affinity_margin",
        )
    }
    report = {
        "scope": "GT-free whole-volume contact/morphology audit; no evaluation skeleton read",
        "decoder_inputs": {
            "features": str(args.features.resolve()),
            "endpoints": str(args.endpoints.resolve()),
            "interiors": str(args.interiors.resolve()),
            "contacts": str(args.contacts.resolve()),
            "rg_competition": (
                str(args.rg_competition.resolve()) if args.rg_competition is not None else None
            ),
        },
        "parameters": parameters,
        "representative_patch_policy": str(contacts["representative_patch_policy"].item()),
        "contact_pairs": int(len(left_labels)),
        "morphology_eligible_pairs": int(len(contact_indices)),
        "ranked_pairs": int(len(rows)),
        "endpoint_endpoint": int(np.count_nonzero(candidate["endpoint_endpoint"])),
        "spine_veto": int(np.count_nonzero(candidate["spine_veto"])),
        "gated_pairs": int(len(gated)),
        "mutual_unambiguous_before_matching": int(len(mutual)),
        "internal_orphan_unambiguous_before_matching": int(len(orphan_preliminary)),
        "accepted_mutual_unambiguous": int(np.count_nonzero(accepted_type == 1)),
        "accepted_internal_orphan": int(np.count_nonzero(accepted_type == 2)),
        "accepted_matching_total": int(len(accepted)),
        "accepted_contact_surface_route": int(
            np.count_nonzero(candidate["contact_surface_route"][accepted])
        ),
        "accepted_skeleton_interpolation_route": int(
            np.count_nonzero(candidate["skeleton_interpolation_route"][accepted])
        ),
        "independent_gate_counts": {
            name: int(np.count_nonzero(condition))
            for name, condition in {**diagnostic_conditions, **gate_conditions}.items()
        },
        "cumulative_gate_counts": cumulative_gate_counts,
        "kind_counts": {
            name: int(np.count_nonzero(candidate["kind"] == code))
            for name, code in TYPE_CODE.items()
        },
        "accepted_kind_counts": {
            name: int(np.count_nonzero(candidate["kind"][accepted] == code))
            for name, code in TYPE_CODE.items()
        },
        "distributions": numeric_audit,
    }
    args.audit_report.write_text(json.dumps(report, indent=2) + "\n")
    log(
        json.dumps(
            {key: value for key, value in report.items() if key != "distributions"}, indent=2
        )
    )

    if args.freeze:
        implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        input_sha256 = {
            name: sha256_file(path)
            for name, path in (
                ("features", args.features),
                ("endpoints", args.endpoints),
                ("interiors", args.interiors),
                ("contacts", args.contacts),
            )
        }
        if args.rg_competition is not None:
            input_sha256["rg_competition"] = sha256_file(args.rg_competition)
        np.savez_compressed(
            args.proposals,
            left=candidate["left"][accepted],
            right=candidate["right"][accepted],
            endpoint_left=candidate["endpoint_left"][accepted],
            endpoint_right=candidate["endpoint_right"][accepted],
            kind=candidate["kind"][accepted],
            score=candidate["score"][accepted],
            proposal_type=accepted_type[accepted],
            orphan_source=orphan_source[accepted],
            parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
            input_sha256=np.asarray(json.dumps(input_sha256, sort_keys=True)),
            implementation_sha256=np.asarray(implementation_sha256),
            gt_free=np.asarray(True),
            frozen_before_evaluation=np.asarray(True),
        )
        proposal_sha256 = hashlib.sha256(args.proposals.read_bytes()).hexdigest()
        frozen_report = dict(report)
        frozen_report.update(
            scope="GT-free whole-volume proposal frozen before funlib evaluation",
            implementation_sha256=implementation_sha256,
            input_sha256=input_sha256,
            proposal_sha256=proposal_sha256,
        )
        args.frozen_report.write_text(json.dumps(frozen_report, indent=2) + "\n")
        log(f"frozen {len(accepted):,} proposals sha256={proposal_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
