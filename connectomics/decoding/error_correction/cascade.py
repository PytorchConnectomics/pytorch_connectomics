"""Repair split axons in confidence order, aiming for ends that reach the crop.

Two ideas, both cheap and both checkable without ground truth.

**The objective is border contact.** A neurite crossing a crop should stop at the
crop, not in the middle of it. A segment whose every terminal is explained by a
crop face is *anchored*; one that spans two different faces is *through*. A
fragment floating with no censored end is the clearest kind of unfinished
reconstruction. So the thing to maximise is the share of skeleton length sitting
in anchored components, and that can be counted before and after any repair.

**Confidence is the absence of a choice.** A free end with one admissible partner
and no competition for it is a repair with nothing to get wrong; the same end
with six admissible partners is a guess, whichever scores best. So each pass
accepts only the uncontested cases and leaves the contested ones for a later
pass, by which point earlier joins may have removed the competition -- a fragment
absorbed into a chain stops competing for anything.

Note what "uncontested" must *not* mean here. Requiring the partner to reach back
-- mutual agreement -- deletes the one-tip route, because a process rejoining the
*side* of its continuation has no terminal there to reach back with, and on
LICONN mip1 tip-to-tip produced exactly zero accepted joins. Confidence is
therefore one-sided: the source end has no alternative, **and** no other end is
competing for the same attachment site.

Passes run a gate schedule from tight to loose. Each pass iterates to a fixpoint
before the next begins, so the easy repairs are all taken at the strictest gates
available and a loose gate never gets to pre-empt a decision a tight one could
have made.

Nothing here is verified. A join accepted by this cascade passed gates and had no
rival; with no ground truth its precision is unknown, and the border objective
rewards a wrong join that happens to reach a face exactly as much as a right one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .split_links import LinkCandidate, LinkConfig, LinkSite, propose_links

__all__ = [
    "CascadePass",
    "CascadeConfig",
    "BorderState",
    "PassReport",
    "CascadeResult",
    "default_schedule",
    "border_state",
    "run_cascade",
]


@dataclass(frozen=True)
class CascadePass:
    """One pass of the schedule: its gates, its confidence rule, and its risk.

    ``allow_union`` is off by default, and that default is the most important
    setting here. Two operations look alike and are not:

    *Assignment* gives a small fragment the label of a much longer host. Getting
    it wrong misplaces that fragment and nothing else.

    *Union* welds two substantial components. Getting it wrong destroys both.

    On zebrafinch every union variant measured net-negative -- affinity-gated
    anchor-anchor joins scored between -0.015 and -0.263, and a continuity-gated
    version at 91.2% precision scored -0.085 and zeroed 13 neurons -- while
    assignment was the operation that paid. Break-even precision for a union
    there was about 96.5% against a best achievable 81-86%. A candidate is
    treated as an assignment only when the source component is at most
    ``assignment_max_length_um`` long and the host is at least
    ``assignment_min_host_ratio`` times longer; everything else is a union and
    needs this flag set deliberately.
    """

    name: str
    link: LinkConfig
    site_grid_um: float = 0.10
    max_claims_per_site: int = 1
    require_border_gain: bool = False
    allow_union: bool = False
    assignment_max_length_um: float = 3.0
    assignment_min_host_ratio: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.link, LinkConfig):
            raise TypeError("link must be a LinkConfig")
        for name in ("assignment_max_length_um", "assignment_min_host_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.assignment_min_host_ratio < 1:
            raise ValueError("assignment_min_host_ratio must be at least one")
        value = float(self.site_grid_um)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("site_grid_um must be finite and positive")
        object.__setattr__(self, "site_grid_um", value)
        if isinstance(self.max_claims_per_site, bool) or self.max_claims_per_site < 1:
            raise ValueError("max_claims_per_site must be a positive integer")
        object.__setattr__(self, "max_claims_per_site", int(self.max_claims_per_site))


@dataclass(frozen=True)
class CascadeConfig:
    """The schedule, the group cap, and which segments may be repaired at all."""

    passes: tuple[CascadePass, ...]
    max_group_size: int = 8
    repair_semantic_types: tuple[str, ...] = ("axon_like", "ambiguous_caliber")
    max_iterations_per_pass: int = 8

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("at least one pass is required")
        for name in ("max_group_size", "max_iterations_per_pass"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.max_group_size < 2:
            raise ValueError("max_group_size must be at least two")
        object.__setattr__(self, "repair_semantic_types", tuple(self.repair_semantic_types))


def default_schedule(
    *, min_affinity: float, max_group_size: int = 8, allow_union: bool = False
) -> CascadeConfig:
    """Tight to loose, with the gap opening last. Numbers have provenance.

    Gap is relaxed last on purpose. In dense neuropil almost every free end has
    *some* neighbour within half a micron, so a wider search adds rivals far
    faster than it adds true continuations -- it erodes the very uniqueness this
    cascade relies on. Tangent and caliber are loosened first because they
    discriminate, and the affinity floor is never loosened at all: it is the only
    gate that separates a continuation from a neighbour running alongside.

    Where the values come from, all from zebrafinch measurements in
    ``dev/zebrafinch``:

    * **60 deg is the outer tangent bound**, not a round number: the whole-volume
      graph linker fixed its tangent floor at 0.5, i.e. cos 60 deg
      (``graph_link_whole.README.md``). This schedule ends there and starts at
      30 deg.
    * **The gap should be small because splits abut.** Classifying every GT
      skeleton edge on the matchguard run put 86.7% of breaks in the *contact*
      class -- both pieces labelled, touching along the neurite -- against 0.37%
      true gaps across unlabelled voxels (``error_analysis_matchguard.md``, and
      the same result on Ran-0719). A contact split has a gap near zero, so the
      first passes at 0.25 um already cover the dominant class; the 0.60 um pass
      is reaching for the 0.37% minority and is the one to drop first if
      precision disappoints.
    * **Deferring the ambiguous case is the linker's own rule.** That linker
      deferred any candidate whose ambiguity exceeded 0.8 rather than scoring it.
      The uniqueness rule here is the same idea taken to its limit, and it is
      the direct mitigation for the measured failure that 46% of assignments
      pick the wrong host when the fragment has a choice
      (``HANDOFF_gtfree_ceiling.md``).
    * **The group cap is not cosmetic.** Blind top-k joining accumulated enough
      stray nodes from neighbours to trip a merge threshold and collapse a
      neuron's score to zero at k=20, from 0.93 at k=12, even though no single
      piece contributed more than 4 stray nodes
      (``error_analysis_matchguard.md``).

    What the same source says *against* this whole family: gap, collinearity,
    perpendicular offset, clutter, size and contact affinity together did not
    carry enough signal to decide "same neuron?" for two touching anchors at a
    3% prior, and the ceiling was feature-limited rather than data-limited. That
    is the argument for assignment over union, and for treating any union result
    from this cascade as unproven until reviewed.
    """
    common = dict(
        min_affinity=min_affinity,
        require_mutual=False,
        apply_semantic_gate=False,
        max_group_size=max_group_size,
    )
    # Assignment only. See CascadePass for the measured reason: unions were
    # net-negative on zebrafinch at every setting tried, assignment was not.
    risk = dict(allow_union=allow_union)
    return CascadeConfig(
        passes=(
            CascadePass(
                "strict",
                LinkConfig(max_gap_um=0.25, min_caliber_ratio=0.6, max_tangent_deg=30.0,
                           **common),
                require_border_gain=False,
                **risk,
            ),
            CascadePass(
                "relaxed-angle",
                LinkConfig(max_gap_um=0.25, min_caliber_ratio=0.5, max_tangent_deg=50.0,
                           **common),
                **risk,
            ),
            CascadePass(
                "relaxed-caliber",
                LinkConfig(max_gap_um=0.35, min_caliber_ratio=0.35, max_tangent_deg=60.0,
                           **common),
                **risk,
            ),
            CascadePass(
                "wide-gap",
                LinkConfig(max_gap_um=0.60, min_caliber_ratio=0.35, max_tangent_deg=60.0,
                           **common),
                **risk,
            ),
        ),
        max_group_size=max_group_size,
    )


@dataclass(frozen=True)
class BorderState:
    """How much of the reconstruction currently ends where it should.

    ``through`` components carry censored terminals on two or more *different*
    faces, which is what a neurite crossing the crop looks like. ``anchored``
    reach at least one face. ``floating`` reach none: nothing about where they
    begin or end is explained.
    """

    components: int
    through: int
    anchored: int
    floating: int
    interior_free_ends: int
    length_um_total: float
    length_um_anchored: float
    length_um_through: float

    @property
    def anchored_length_fraction(self) -> float:
        return self.length_um_anchored / self.length_um_total if self.length_um_total else 0.0

    @property
    def through_length_fraction(self) -> float:
        return self.length_um_through / self.length_um_total if self.length_um_total else 0.0


@dataclass(frozen=True)
class PassReport:
    name: str
    iterations: int
    candidates_scored: int
    accepted_gates: int
    unique_source: int
    uncontested: int
    joins_taken: int
    assignments_taken: int
    unions_taken: int
    refused_as_union: int
    dropped_by_group_cap: int
    border_after: BorderState


@dataclass(frozen=True)
class CascadeResult:
    groups: tuple[tuple[int, ...], ...]
    joins: tuple[LinkCandidate, ...]
    passes: tuple[PassReport, ...]
    border_before: BorderState
    border_after: BorderState
    label_to_group: dict[int, int] = field(default_factory=dict)


class _Union:
    """Union-find that also carries each component's set of crop faces.

    The face set is maintained on merge rather than recomputed. Recomputing it
    per candidate means walking every label in the volume for every join
    considered, which on a 16k-label cohort is a five-order-of-magnitude waste
    and makes the schedule too slow to run.
    """

    def __init__(
        self,
        faces: dict[int, frozenset[str]] | None = None,
        lengths: dict[int, float] | None = None,
    ) -> None:
        self.parent: dict[int, int] = {}
        self.size: dict[int, int] = {}
        self.faces: dict[int, set[str]] = {}
        self.length: dict[int, float] = {}
        self._initial = faces or {}
        self._lengths = lengths or {}

    def find(self, node: int) -> int:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            self.faces[node] = set(self._initial.get(node, ()))
            self.length[node] = float(self._lengths.get(node, 0.0))
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.faces[a] |= self.faces.pop(b)
        self.length[a] += self.length.pop(b)
        return True

    def component_size(self, node: int) -> int:
        return self.size[self.find(node)]

    def component_faces(self, node: int) -> set[str]:
        return self.faces[self.find(node)]

    def component_length(self, node: int) -> float:
        return self.length[self.find(node)]


def border_state(
    labels: Sequence[int],
    *,
    faces: dict[int, frozenset[str]],
    free_ends: dict[int, int],
    length_um: dict[int, float],
    union: _Union | None = None,
) -> BorderState:
    """Aggregate per-label border contact over the current components.

    ``faces`` is the set of distinct crop faces a label's *censored* terminals
    touch, ``free_ends`` its count of uncensored terminals, both as measured
    before any joining; joining changes which labels share a component, not the
    geometry of any one label.
    """
    groups: dict[int, list[int]] = {}
    for label in labels:
        root = union.find(label) if union is not None else label
        groups.setdefault(root, []).append(label)
    through = anchored = floating = 0
    total = anchored_length = through_length = 0.0
    interior = 0
    for members in groups.values():
        member_faces: set[str] = set()
        member_length = 0.0
        for label in members:
            member_faces |= faces.get(label, frozenset())
            member_length += length_um.get(label, 0.0)
            interior += free_ends.get(label, 0)
        total += member_length
        if len(member_faces) >= 2:
            through += 1
            through_length += member_length
            anchored_length += member_length
            anchored += 1
        elif member_faces:
            anchored += 1
            anchored_length += member_length
        else:
            floating += 1
    return BorderState(
        components=len(groups),
        through=through,
        anchored=anchored,
        floating=floating,
        interior_free_ends=interior,
        length_um_total=total,
        length_um_anchored=anchored_length,
        length_um_through=through_length,
    )


def _group_by_site(
    claims: list[tuple[tuple[int, int], LinkCandidate, np.ndarray]], radius_um: float
) -> list[list[tuple[tuple[int, int], LinkCandidate]]]:
    """Cluster claims that land on the same place of the same target segment.

    Two ends attaching two vertices apart are attaching to the same place; at
    this volume's ~0.029 um vertex spacing a raw vertex index would call them
    distinct and both would look uncontested.

    Clustering is by distance, not by a grid. A grid cell has edges, and two
    points 0.02 um apart can sit either side of one: 5.30/0.1 floors to 52 in
    IEEE doubles while 5.32/0.1 floors to 53, so the pair that most needs to
    collide is exactly the pair a grid may separate. Single-linkage within
    ``radius_um`` has no such boundary.
    """
    by_label: dict[int, list[tuple[tuple[int, int], LinkCandidate, np.ndarray]]] = {}
    for source, candidate, position in claims:
        by_label.setdefault(candidate.right_label, []).append((source, candidate, position))
    clusters: list[list[tuple[tuple[int, int], LinkCandidate]]] = []
    for entries in by_label.values():
        remaining = list(entries)
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                for entry in list(remaining):
                    if any(
                        float(np.linalg.norm(entry[2] - member[2])) <= radius_um
                        for member in cluster
                    ):
                        cluster.append(entry)
                        remaining.remove(entry)
                        changed = True
            clusters.append([(source, candidate) for source, candidate, _ in cluster])
    return clusters


def run_cascade(
    sites: Sequence[LinkSite],
    *,
    config: CascadeConfig,
    faces: dict[int, frozenset[str]],
    length_um: dict[int, float],
    skeleton_points: dict[int, np.ndarray],
    skeleton_radii: dict[int, np.ndarray] | None = None,
    label_semantic_types: dict[int, str] | None = None,
    affinity_probe: Callable[[np.ndarray, np.ndarray], dict[str, float] | None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CascadeResult:
    """Run the schedule and return the groups, the joins and the border change.

    ``sites`` are all free ends; ``faces`` and ``length_um`` describe every label
    that exists, including ones with no free end, because a fragment's repair is
    only worth anything if the thing it attaches to reaches a face.
    """
    labels = sorted(set(length_um))
    free_end_counts: dict[int, int] = {}
    for site in sites:
        free_end_counts[site.label] = free_end_counts.get(site.label, 0) + 1

    union = _Union(faces, length_um)
    for label in labels:
        union.find(label)
    before = border_state(
        labels, faces=faces, free_ends=free_end_counts, length_um=length_um
    )

    allowed = set(config.repair_semantic_types)
    open_sites = [site for site in sites if site.semantic_type in allowed]
    consumed: set[tuple[int, int]] = set()
    joins: list[LinkCandidate] = []
    reports: list[PassReport] = []

    for step in config.passes:
        iterations = scored_total = accepted_total = unique_total = 0
        uncontested_total = taken_total = dropped_total = 0
        assigned_total = united_total = refused_total = 0
        for _ in range(config.max_iterations_per_pass):
            live = [
                site
                for site in open_sites
                if (site.label, site.vertex_index) not in consumed
            ]
            if not live:
                break
            scored = propose_links(
                live,
                config=step.link,
                skeleton_points=skeleton_points,
                skeleton_radii=skeleton_radii,
                label_semantic_types=label_semantic_types,
                affinity_probe=affinity_probe,
                keep_all=True,
            )
            iterations += 1
            scored_total += len(scored)

            # Same component already: not a repair, and unioning it again would
            # count a self-join as progress.
            passing = [
                candidate
                for candidate in scored
                if candidate.accepted
                and union.find(candidate.left_label) != union.find(candidate.right_label)
            ]
            accepted_total += len(passing)

            by_source: dict[tuple[int, int], list[LinkCandidate]] = {}
            for candidate in passing:
                key = (candidate.left_label, candidate.left_vertex_index)
                by_source.setdefault(key, []).append(candidate)
            # "No other choice": exactly one admissible partner for this end.
            unique = {
                source: options[0]
                for source, options in by_source.items()
                if len(options) == 1
            }
            unique_total += len(unique)

            claims: list[tuple[tuple[int, int], LinkCandidate, np.ndarray]] = []
            for source, candidate in unique.items():
                points = skeleton_points.get(candidate.right_label)
                if points is None or candidate.right_vertex_index >= len(points):
                    continue
                claims.append(
                    (source, candidate,
                     np.asarray(points[candidate.right_vertex_index], dtype=np.float64))
                )
            confident = [
                (source, candidate)
                for cluster in _group_by_site(claims, step.site_grid_um)
                if len(cluster) <= step.max_claims_per_site
                for source, candidate in cluster
            ]
            uncontested_total += len(confident)
            if not confident:
                break

            def gain(candidate: LinkCandidate) -> int:
                """Distinct faces this join would add to the source component."""
                left_faces = union.component_faces(candidate.left_label)
                right_faces = union.component_faces(candidate.right_label)
                return len(left_faces | right_faces) - len(left_faces)

            ranked = sorted(
                confident,
                key=lambda item: (-gain(item[1]), item[1].gap_um),
            )
            def is_assignment(candidate: LinkCandidate) -> bool:
                """A small fragment taking a much longer host's label."""
                source_length = union.component_length(candidate.left_label)
                host_length = union.component_length(candidate.right_label)
                return (
                    source_length <= step.assignment_max_length_um
                    and host_length >= step.assignment_min_host_ratio * max(source_length, 1e-9)
                )

            taken = assigned = united = refused = 0
            for source, candidate in ranked:
                if (
                    union.component_size(candidate.left_label)
                    + union.component_size(candidate.right_label)
                    > config.max_group_size
                ):
                    dropped_total += 1
                    continue
                if step.require_border_gain and gain(candidate) <= 0:
                    continue
                assignment = is_assignment(candidate)
                if not assignment and not step.allow_union:
                    refused += 1
                    continue
                if not union.union(candidate.left_label, candidate.right_label):
                    continue
                consumed.add(source)
                if candidate.mutual:
                    consumed.add((candidate.right_label, candidate.right_vertex_index))
                joins.append(candidate)
                taken += 1
                if assignment:
                    assigned += 1
                else:
                    united += 1
            taken_total += taken
            assigned_total += assigned
            united_total += united
            refused_total += refused
            if not taken:
                break

        state = border_state(
            labels, faces=faces, free_ends=free_end_counts, length_um=length_um, union=union
        )
        reports.append(
            PassReport(
                name=step.name,
                iterations=iterations,
                candidates_scored=scored_total,
                accepted_gates=accepted_total,
                unique_source=unique_total,
                uncontested=uncontested_total,
                joins_taken=taken_total,
                assignments_taken=assigned_total,
                unions_taken=united_total,
                refused_as_union=refused_total,
                dropped_by_group_cap=dropped_total,
                border_after=state,
            )
        )
        if progress is not None:
            progress(
                f"{step.name}: {iterations} iter, {accepted_total} pass gates, "
                f"{unique_total} unique, {uncontested_total} uncontested, "
                f"{assigned_total} assigned + {united_total} unioned "
                f"({refused_total} refused as union) -> {state.through} through / "
                f"{state.anchored} anchored / {state.floating} floating"
            )

    grouped: dict[int, list[int]] = {}
    for label in labels:
        grouped.setdefault(union.find(label), []).append(label)
    groups = tuple(
        tuple(sorted(members)) for members in grouped.values() if len(members) > 1
    )
    label_to_group = {
        label: index for index, members in enumerate(groups) for label in members
    }
    after = border_state(
        labels, faces=faces, free_ends=free_end_counts, length_um=length_um, union=union
    )
    return CascadeResult(
        groups=groups,
        joins=tuple(joins),
        passes=tuple(reports),
        border_before=before,
        border_after=after,
        label_to_group=label_to_group,
    )
