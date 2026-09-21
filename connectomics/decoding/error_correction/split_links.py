"""Propose joins across free skeleton ends under a semantic-type constraint.

A segment that stops inside the crop has an unexplained end. This module pairs
such an end with somewhere on another segment and scores the pairing; it does not
apply anything, read any file, or claim a join is correct.

Two design choices carry over from measured failures of earlier tip linkers and
are not free parameters:

*One tip, not two.* Requiring both sides to present a facing terminal ("tip-tip")
throws away the common case where a broken process should rejoin the *side* of
its continuation, and shape-only facing models did not reach usable precision.
A candidate therefore needs one free end on the left and merely a reachable
skeleton point on the right; :class:`LinkConfig` can still demand a mutual pair.

*Geometry proposes, affinity decides.* Gap length, caliber agreement and tangent
alignment rank candidates well but do not separate true continuations from
neighbours running alongside. The affinity probe is injected by the caller, so
this module stays I/O-free, but a candidate with no affinity evidence is reported
with ``affinity=None`` and is never accepted when a floor is configured.

Joins are asymmetric in cost: one wrong join can fuse two processes for their
whole length, while a missed join costs only that one gap. The defaults are
therefore conservative, and ``accepted`` means "passed every configured gate",
never "verified". Without ground truth the precision of this proposer on any
given volume is unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ...metrics.unsupervised.continuity import SegmentContinuity, semantic_link_allowed

__all__ = [
    "LinkConfig",
    "LinkSite",
    "LinkCandidate",
    "collect_link_sites",
    "propose_links",
    "select_links",
    "select_mutual_best",
    "connected_groups",
]


@dataclass(frozen=True)
class LinkConfig:
    """Gates applied in order; the first failure is recorded as the reason.

    ``max_gap_um`` is the straight distance between the free end and the point it
    would attach to. ``min_caliber_ratio`` compares the two local *shaft* radii,
    thinner over thicker -- see :attr:`LinkSite.caliber_um` for why the terminal
    vertex's own radius must not be used here. ``max_tangent_deg`` bounds the angle between the free end's
    outward direction and the direction to its partner: a continuation leaves
    roughly the way it was heading, a neighbour running alongside does not.

    ``min_affinity`` is compared against whatever scale the caller's probe
    returns; passing the same threshold the decoder agglomerated at makes the
    result readable as "this gap would have merged had the boundary not been
    there".

    ``require_mutual`` keeps only pairs that are each other's best partner. It is
    a large precision gain for tip-to-tip pairs and a total loss for the one-tip
    route, which has no reverse candidate to be mutual with; see
    :func:`select_links`.

    ``apply_semantic_gate`` can switch the axon/dendrite veto off. Do that when
    the volume's caliber distribution does not actually support the class
    distinction: a veto driven by an unsupported label removes good candidates
    and adds no safety. The caliber *ratio* still applies either way, and is the
    weaker but better-founded constraint.
    """

    max_gap_um: float = 1.0
    min_caliber_ratio: float = 0.5
    max_tangent_deg: float = 60.0
    min_affinity: float | None = None
    require_mutual: bool = True
    apply_semantic_gate: bool = True
    allow_ambiguous_semantic: bool = True
    max_group_size: int = 8

    def __post_init__(self) -> None:
        for name in ("max_gap_um", "min_caliber_ratio", "max_tangent_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.min_caliber_ratio > 1:
            raise ValueError("min_caliber_ratio must be at most one")
        if self.max_tangent_deg > 180:
            raise ValueError("max_tangent_deg must be at most 180")
        if self.min_affinity is not None:
            value = float(self.min_affinity)
            if not math.isfinite(value):
                raise ValueError("min_affinity must be finite or None")
            object.__setattr__(self, "min_affinity", value)
        if isinstance(self.max_group_size, bool) or self.max_group_size < 2:
            raise ValueError("max_group_size must be an integer of at least two")
        object.__setattr__(self, "max_group_size", int(self.max_group_size))


@dataclass(frozen=True)
class LinkSite:
    """One free end offered for linking, with the segment context it carries."""

    label: int
    vertex_index: int
    position_um_zyx: tuple[float, float, float]
    outward_tangent_zyx: tuple[float, float, float]
    radius_um: float
    semantic_type: str
    shaft_radius_um: float | None = None
    shape: str | None = None

    @property
    def caliber_um(self) -> float:
        """The radius to compare against a partner.

        **Never the terminal vertex's own radius.** A skeletonizer places the
        terminal vertex at the object surface, so its distance-to-boundary is
        about one voxel regardless of how thick the process is -- measured on
        LICONN mip1 it was 0.0181 um at p10, p50 *and* p90, i.e. a constant. A
        caliber ratio built on it compares an artifact with a real radius and
        rejects almost everything. Use the local shaft radius from the terminal
        profile when it exists.
        """
        if self.shaft_radius_um is not None and self.shaft_radius_um > 0:
            return float(self.shaft_radius_um)
        return float(self.radius_um)


@dataclass(frozen=True)
class LinkCandidate:
    """One proposed join and every measurement behind it.

    ``right_vertex_index`` is the attachment point on the right segment: its own
    free end when ``mutual`` is true, otherwise the nearest point of its skeleton.
    ``reject_reason`` is empty exactly when ``accepted`` is true.
    """

    left_label: int
    right_label: int
    left_vertex_index: int
    right_vertex_index: int
    left_semantic_type: str
    right_semantic_type: str
    gap_um: float
    caliber_ratio: float
    left_radius_um: float
    right_radius_um: float
    tangent_deg: float
    mutual: bool
    affinity: dict[str, float] | None
    accepted: bool
    reject_reason: str


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros(3)


def collect_link_sites(segments: Iterable[SegmentContinuity]) -> list[LinkSite]:
    """Every free end of every measured segment, in input order."""
    sites: list[LinkSite] = []
    for segment in segments:
        if not segment.measured:
            continue
        for end in segment.free_ends:
            sites.append(
                LinkSite(
                    label=segment.label,
                    vertex_index=end.vertex_index,
                    position_um_zyx=end.position_um_zyx,
                    outward_tangent_zyx=end.outward_tangent_zyx,
                    radius_um=end.radius_um,
                    semantic_type=segment.semantic_type,
                    shaft_radius_um=end.shaft_radius_um,
                    shape=end.shape,
                )
            )
    return sites


def propose_links(
    sites: Sequence[LinkSite],
    *,
    config: LinkConfig,
    skeleton_points: dict[int, np.ndarray] | None = None,
    skeleton_radii: dict[int, np.ndarray] | None = None,
    label_semantic_types: dict[int, str] | None = None,
    affinity_probe: Callable[[np.ndarray, np.ndarray], dict[str, float] | None] | None = None,
) -> list[LinkCandidate]:
    """Score every free end against nearby partners, keeping the best per end.

    ``skeleton_points`` maps a label to its ``(N, 3)`` vertices in micrometers and
    enables the one-tip route, where a free end attaches to the side of another
    segment. Without it, only free-end to free-end pairs are considered. All
    labels' vertices go into a single spatial index, so cost scales with the
    number of vertices near each end rather than with the number of segments.

    ``label_semantic_types`` supplies the type of a side-attachment target, which
    may be a segment with no free end of its own and therefore absent from
    ``sites``; such a label is treated as ``ambiguous_caliber`` when not given.
    ``affinity_probe`` receives the two physical endpoints and returns a dict of
    statistics, or ``None`` when the gap cannot be sampled.

    Returns one candidate per free end that had any partner within
    ``max_gap_um``, accepted or not, so the rejection reasons can be counted.
    """
    if not sites:
        return []
    from scipy.spatial import cKDTree

    positions = np.asarray([site.position_um_zyx for site in sites], dtype=np.float64)
    labels = np.asarray([site.label for site in sites], dtype=np.int64)
    end_tree = cKDTree(positions)

    types: dict[int, str] = {site.label: site.semantic_type for site in sites}
    if label_semantic_types:
        types.update(label_semantic_types)

    point_tree = None
    point_xyz = np.zeros((0, 3))
    point_label = np.zeros(0, dtype=np.int64)
    point_radius = np.zeros(0)
    point_local = np.zeros(0, dtype=np.int64)
    if skeleton_points:
        blocks, block_labels, block_radii, block_local = [], [], [], []
        for label, points in skeleton_points.items():
            points = np.asarray(points, dtype=np.float64)
            if not len(points):
                continue
            blocks.append(points)
            block_labels.append(np.full(len(points), label, dtype=np.int64))
            block_local.append(np.arange(len(points), dtype=np.int64))
            radii = skeleton_radii.get(label) if skeleton_radii else None
            block_radii.append(
                np.asarray(radii, dtype=np.float64)
                if radii is not None
                else np.full(len(points), np.nan)
            )
        if blocks:
            point_xyz = np.concatenate(blocks)
            point_label = np.concatenate(block_labels)
            point_radius = np.concatenate(block_radii)
            point_local = np.concatenate(block_local)
            point_tree = cKDTree(point_xyz)

    candidates: list[LinkCandidate] = []
    for index, site in enumerate(sites):
        origin = positions[index]
        tangent = np.asarray(site.outward_tangent_zyx, dtype=np.float64)
        neighbours = end_tree.query_ball_point(origin, config.max_gap_um)
        partners: dict[int, tuple[float, int, bool]] = {}
        for other in neighbours:
            if labels[other] == site.label:
                continue
            gap = float(np.linalg.norm(positions[other] - origin))
            best = partners.get(int(labels[other]))
            if best is None or gap < best[0]:
                partners[int(labels[other])] = (gap, other, True)
        if point_tree is not None:
            nearby = point_tree.query_ball_point(origin, config.max_gap_um)
            if nearby:
                nearby = np.asarray(nearby, dtype=np.int64)
                nearby = nearby[point_label[nearby] != site.label]
            if len(nearby):
                gaps = np.linalg.norm(point_xyz[nearby] - origin, axis=1)
                order = np.argsort(gaps, kind="stable")
                for position in order:
                    point = int(nearby[position])
                    label = int(point_label[point])
                    best = partners.get(label)
                    gap = float(gaps[position])
                    if best is None or gap < best[0]:
                        partners[label] = (gap, point, False)
        if not partners:
            continue

        scored: list[LinkCandidate] = []
        for label, (gap, target, is_end) in partners.items():
            if is_end:
                right_radius = sites[target].caliber_um
                right_position = positions[target]
                right_vertex = sites[target].vertex_index
                right_type = sites[target].semantic_type
            else:
                right_position = point_xyz[target]
                # Report the index within the owning label's own vertex array, so
                # the caller can address it in the graph it supplied.
                right_vertex = int(point_local[target])
                radius = float(point_radius[target])
                right_radius = radius if math.isfinite(radius) else site.caliber_um
                # A side attachment has no free end of its own to read a type
                # from; it inherits the owning segment's type when known.
                right_type = types.get(label, "ambiguous_caliber")
            direction = _unit(right_position - origin)
            if np.linalg.norm(tangent) == 0 or np.linalg.norm(direction) == 0:
                angle = 180.0
            else:
                angle = float(
                    math.degrees(math.acos(float(np.clip(np.dot(tangent, direction), -1.0, 1.0))))
                )
            left_caliber = site.caliber_um
            ratio = min(left_caliber, right_radius) / max(left_caliber, right_radius, 1e-9)

            # Cheap geometric vetoes first. The affinity probe reads a volume and
            # is the expensive step, so it only runs for a pair that has already
            # survived everything else; a pair rejected earlier carries
            # ``affinity: None`` with its own, accurate reason.
            reason = ""
            if config.apply_semantic_gate and not semantic_link_allowed(
                site.semantic_type, right_type, allow_ambiguous=config.allow_ambiguous_semantic
            ):
                reason = f"semantic:{site.semantic_type}->{right_type}"
            elif gap > config.max_gap_um:
                reason = "gap"
            elif ratio < config.min_caliber_ratio:
                reason = "caliber"
            elif angle > config.max_tangent_deg:
                reason = "tangent"
            affinity = None
            if not reason and affinity_probe is not None:
                affinity = affinity_probe(origin, right_position)
            if not reason and config.min_affinity is not None:
                if affinity is None or affinity.get("mean") is None:
                    reason = "no_affinity"
                elif float(affinity["mean"]) < config.min_affinity:
                    reason = "affinity"
            scored.append(
                LinkCandidate(
                    left_label=site.label,
                    right_label=label,
                    left_vertex_index=site.vertex_index,
                    right_vertex_index=right_vertex,
                    left_semantic_type=site.semantic_type,
                    right_semantic_type=right_type,
                    gap_um=gap,
                    caliber_ratio=ratio,
                    left_radius_um=left_caliber,
                    right_radius_um=right_radius,
                    tangent_deg=angle,
                    mutual=is_end,
                    affinity=affinity,
                    accepted=not reason,
                    reject_reason=reason,
                )
            )
        passing = [candidate for candidate in scored if candidate.accepted]
        pool = passing or scored
        candidates.append(min(pool, key=lambda item: (not item.accepted, item.gap_um)))
    return candidates


def select_links(
    candidates: Sequence[LinkCandidate], *, require_mutual: bool
) -> list[LinkCandidate]:
    """One accepted candidate per unordered pair, shortest gap first.

    ``require_mutual`` is not free: a side attachment has no reverse candidate to
    be mutual *with*, because the partner is a point on a shaft rather than a
    terminal reaching back. Demanding mutuality therefore deletes the entire
    one-tip route, which is the route that exists precisely because tip-to-tip
    shape models do not work in dense neuropil. Enable it only for a tip-to-tip
    run; otherwise the gates in :func:`propose_links` are the whole filter.
    """
    if require_mutual:
        return select_mutual_best(candidates)
    kept: dict[tuple[int, int], LinkCandidate] = {}
    for candidate in candidates:
        if not candidate.accepted:
            continue
        key = (
            min(candidate.left_label, candidate.right_label),
            max(candidate.left_label, candidate.right_label),
        )
        if key not in kept or candidate.gap_um < kept[key].gap_um:
            kept[key] = candidate
    return sorted(kept.values(), key=lambda item: item.gap_um)


def select_mutual_best(candidates: Sequence[LinkCandidate]) -> list[LinkCandidate]:
    """Keep accepted pairs that chose each other, deduplicated to one per pair.

    A free end reaching for a partner that is reaching back is far better
    evidence than a one-sided nearest neighbour, which any dense neuropil
    supplies in quantity.
    """
    # A segment with several free ends can offer several candidates towards the
    # same partner; keep its shortest, not whichever was enumerated last.
    best: dict[tuple[int, int], LinkCandidate] = {}
    for candidate in candidates:
        if not candidate.accepted:
            continue
        key = (candidate.left_label, candidate.right_label)
        if key not in best or candidate.gap_um < best[key].gap_um:
            best[key] = candidate
    kept: dict[tuple[int, int], LinkCandidate] = {}
    for (left, right), candidate in best.items():
        reverse = best.get((right, left))
        if reverse is None:
            continue
        key = (min(left, right), max(left, right))
        if key not in kept or candidate.gap_um < kept[key].gap_um:
            kept[key] = candidate
    return sorted(kept.values(), key=lambda item: item.gap_um)


def connected_groups(
    candidates: Sequence[LinkCandidate], *, max_group_size: int
) -> tuple[list[list[int]], list[LinkCandidate]]:
    """Union the accepted pairs, refusing any join that overgrows a group.

    Chains are how a join pipeline turns a handful of plausible repairs into one
    implausible object. Edges are taken shortest first and an edge is dropped
    when it would push a component past ``max_group_size``; the dropped edges are
    returned rather than silently discarded.
    """
    parent: dict[int, int] = {}
    size: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        size.setdefault(node, 1)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    dropped: list[LinkCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.gap_um):
        left, right = find(candidate.left_label), find(candidate.right_label)
        if left == right:
            continue
        if size[left] + size[right] > max_group_size:
            dropped.append(candidate)
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(members) for members in groups.values() if len(members) > 1], dropped
