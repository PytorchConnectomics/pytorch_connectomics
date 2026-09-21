"""Semantic type and endpoint continuity for one segment, without ground truth.

Two questions are answered independently, because conflating them is what makes
short, blobby split fragments invisible:

``semantic_type``
    What kind of process the geometry is consistent with, decided from *local*
    skeleton caliber and branch density. It deliberately does not use global PCA
    elongation, so a bouton-shaped axon stub is still axon-calibre.
``completeness``
    Whether every skeleton terminal is explained by the crop. A terminal further
    than its own radius plus a margin from all six faces is a *free end*: the
    segment stops inside the volume, and something is missing there.

Both are geometric candidates. A free end is not a proven false split -- a real
axon terminal bouton also ends inside the volume, and telling the two apart needs
evidence this module does not have. A ``done`` segment is not a verified neurite
either; an end-to-end false merge is ``done`` by this definition.

Skeleton measurements come from :mod:`.arbor`, which owns pruning and the
protected geodesic diameter. This module adds only the crop-face test and the
caliber decision. All lengths, radii and coordinates are micrometers, z, y, x.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .arbor import ArborConfig, ArborMetrics, analyze_arbor

__all__ = [
    "SEMANTIC_TYPES",
    "COMPLETENESS_CLASSES",
    "TERMINAL_SHAPES",
    "FACE_NAMES",
    "ContinuityConfig",
    "FreeEnd",
    "SegmentContinuity",
    "analyze_continuity",
    "semantic_link_allowed",
]

SEMANTIC_TYPES = (
    "axon_like",
    "dendrite_like",
    "ambiguous_caliber",
    "unmeasured",
)
COMPLETENESS_CLASSES = (
    "complete",
    "broken_one_end",
    "broken_multi_end",
    "isolated_fragment",
    "unmeasured",
)
TERMINAL_SHAPES = (
    "bouton_head",
    "blunt",
    "tapered",
    "unresolved",
)
FACE_NAMES = (("z0", "zmax"), ("y0", "ymax"), ("x0", "xmax"))


@dataclass(frozen=True)
class ContinuityConfig:
    """Crop extent, face tolerance and the caliber gates, all in micrometers.

    ``volume_extent_um`` is the physical size of the crop, ``shape * voxel_size``.
    Vertex coordinates must be in the same frame, i.e. voxel centers are
    ``(index + 0.5) * voxel_size_um``, so a terminal at the very first voxel
    center sits half a voxel from the face.

    A terminal is censored by the crop when its distance to the nearest face is
    at most its own skeleton radius plus ``face_margin_um``. Using the radius
    matters: a medial-axis vertex of a thick process is inset from the surface by
    roughly one radius even when the object plainly touches the face.

    The caliber gates are provisional and grid-dependent. ``axon_max_radius_um``
    and ``dendrite_min_radius_um`` leave a deliberate gap that reports
    ``ambiguous_caliber`` rather than forcing a side. **Check them against the
    measured radius distribution of your own volume before trusting them**: the
    defaults are EM-scale conventions, and on a finer grid they can sit entirely
    outside the data, in which case caliber silently stops discriminating.

    Branch density is branch points per micrometer of retained skeleton. It is a
    rate, and a rate measured over a short skeleton is noise: two branch points
    on a 1.4 um fragment is 1.4 per micrometer, which would outrank a real
    dendrite. ``min_branch_length_um`` and ``min_branch_points`` are therefore
    both required before branch density is allowed to decide anything; below
    either, the decision falls back to caliber alone. Branch density is also
    raised by false merges, so it is never used to rule a segment *out*.
    """

    volume_extent_um: tuple[float, float, float]
    face_margin_um: float = 0.05
    axon_max_radius_um: float = 0.35
    dendrite_min_radius_um: float = 0.50
    axon_max_branch_per_um: float = 0.25
    dendrite_min_branch_per_um: float = 0.50
    min_branch_length_um: float = 5.0
    min_branch_points: int = 3
    head_window_um: float = 0.30
    shaft_window_um: float = 0.70
    bouton_head_ratio: float = 1.5
    taper_ratio: float = 0.7
    arbor: ArborConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        extent = np.asarray(self.volume_extent_um, dtype=float)
        if extent.shape != (3,) or not np.all(np.isfinite(extent) & (extent > 0)):
            raise ValueError("volume_extent_um must contain three finite positive values")
        object.__setattr__(self, "volume_extent_um", tuple(float(v) for v in extent))
        if not np.isfinite(self.face_margin_um) or self.face_margin_um < 0:
            raise ValueError("face_margin_um must be finite and nonnegative")
        object.__setattr__(self, "face_margin_um", float(self.face_margin_um))
        for name in (
            "axon_max_radius_um",
            "dendrite_min_radius_um",
            "axon_max_branch_per_um",
            "dendrite_min_branch_per_um",
            "min_branch_length_um",
            "head_window_um",
            "shaft_window_um",
            "bouton_head_ratio",
            "taper_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.dendrite_min_radius_um < self.axon_max_radius_um:
            raise ValueError("dendrite_min_radius_um must be at least axon_max_radius_um")
        if self.dendrite_min_branch_per_um < self.axon_max_branch_per_um:
            raise ValueError(
                "dendrite_min_branch_per_um must be at least axon_max_branch_per_um"
            )
        if self.bouton_head_ratio <= 1:
            raise ValueError("bouton_head_ratio must exceed one to mean a swelling")
        if self.taper_ratio >= 1:
            raise ValueError("taper_ratio must be below one to mean a narrowing")
        if isinstance(self.min_branch_points, bool) or self.min_branch_points < 1:
            raise ValueError("min_branch_points must be a positive integer")
        object.__setattr__(self, "min_branch_points", int(self.min_branch_points))
        if self.arbor is None:
            object.__setattr__(self, "arbor", ArborConfig())
        elif not isinstance(self.arbor, ArborConfig):
            raise TypeError("arbor must be an ArborConfig")


@dataclass(frozen=True)
class FreeEnd:
    """One skeleton terminal, with the outward direction a link would leave by.

    ``vertex_index`` addresses the caller's original vertex array, so a proposal
    can be drawn on the same graph. ``outward_tangent_zyx`` is the unit vector
    pointing away from the segment along the terminal branch; it is the zero
    vector when the terminal has no neighbor at a different position.

    ``shape`` describes the radius profile walking inward from the tip, which is
    the one local signal that bears on whether this end is a real ending or a
    cut. ``head_radius_um`` is the largest radius within ``head_window_um`` of the
    tip and ``shaft_radius_um`` the median over the window beyond it; their ratio
    is ``head_ratio``. The radius *at* the tip vertex is not used for this,
    because a skeletonizer puts terminal vertices near the surface and that value
    is small whether the object ends in a bulb or is sliced across.
    """

    vertex_index: int
    position_um_zyx: tuple[float, float, float]
    outward_tangent_zyx: tuple[float, float, float]
    radius_um: float
    distance_to_face_um: float
    nearest_face: str
    censored: bool
    shape: str
    head_radius_um: float | None
    shaft_radius_um: float | None
    head_ratio: float | None
    shaft_source: str


@dataclass(frozen=True)
class SegmentContinuity:
    """Geometric candidate type and endpoint budget for one segment.

    ``free_ends`` lists only the uncensored terminals, ordered by descending
    distance from the crop, so the first entry is the most interior. Every count
    refers to the pruned skeleton: twigs removed by :mod:`.arbor` are not ends.
    ``unmeasured`` on either axis means the skeleton could not support the
    decision, never that the segment was found to be correct.
    """

    label: int
    voxel_count: int
    measured: bool
    semantic_type: str
    semantic_basis: str
    completeness_class: str
    done: bool
    caliber_radius_um: float | None
    caliber_source: str
    branch_points_per_um: float | None
    branch_density_usable: bool
    retained_length_um: float
    terminal_count: int
    censored_end_count: int
    free_end_count: int
    bouton_head_end_count: int
    terminal_shape_counts: dict[str, int]
    free_ends: tuple[FreeEnd, ...]
    arbor: ArborMetrics | None


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _clean_graph(
    vertices: np.ndarray, edges: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop self-loops, duplicates and zero-length edges that ``arbor`` rejects.

    Skeletonizer output routinely contains all three after simplification. They
    carry no information, so removing them is not a measurement choice; the
    alternative is refusing to measure the segment at all.
    """
    if not len(edges):
        return edges.reshape(0, 2).astype(np.int64), radii
    pairs = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    # Bounds are checked here rather than left to the indexing below, which would
    # fail with a bare IndexError naming neither the edge nor the vertex count.
    if pairs.min() < 0 or pairs.max() >= len(vertices):
        raise ValueError(
            f"edge indices must address existing vertices: "
            f"range [{pairs.min()}, {pairs.max()}] against {len(vertices)} vertices"
        )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs):
        pairs = np.unique(pairs, axis=0)
    if len(pairs):
        lengths = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1)
        pairs = pairs[np.isfinite(lengths) & (lengths > 0)]
    return pairs.reshape(-1, 2), radii


def _weighted_median_radius(
    vertices: np.ndarray, edges: np.ndarray, radii: np.ndarray
) -> float | None:
    """Length-weighted median radius over the given edges, or None when empty.

    Each edge contributes half its length to both endpoints, so a densely sampled
    stretch does not outvote a sparsely sampled one of the same physical length.
    """
    if not len(edges):
        return None
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    weights = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(weights, edges[:, 0], lengths / 2)
    np.add.at(weights, edges[:, 1], lengths / 2)
    used = weights > 0
    if not np.any(used):
        return None
    order = np.argsort(radii[used], kind="stable")
    cumulative = np.cumsum(weights[used][order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(radii[used][order[min(index, len(order) - 1)]])


def _face_distance(
    point: np.ndarray, extent: np.ndarray
) -> tuple[float, str]:
    """Distance from ``point`` to the nearest of the six crop faces, and its name."""
    low = np.maximum(point, 0.0)
    high = np.maximum(extent - point, 0.0)
    distances = np.concatenate((low, high))
    index = int(np.argmin(distances))
    axis, side = index % 3, index // 3
    return float(distances[index]), FACE_NAMES[axis][side]


def _outward_tangent(
    terminal: int, adjacency: list[list[int]], vertices: np.ndarray
) -> np.ndarray:
    neighbors = adjacency[terminal]
    if not neighbors:
        return np.zeros(3)
    direction = vertices[terminal] - vertices[neighbors[0]]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else np.zeros(3)


def _branch_density_is_usable(
    branch_per_um: float | None,
    length_um: float,
    branch_points: int,
    config: ContinuityConfig,
) -> bool:
    """Whether the branch rate rests on enough skeleton to mean anything.

    A rate needs both a denominator long enough to be a sample and a numerator
    above counting noise. Two branch points on a one-micrometre fragment is a
    skeletonization artifact, not a dendritic arbor, and admitting it makes the
    class a proxy for "short and messy".
    """
    return (
        branch_per_um is not None
        and length_um >= config.min_branch_length_um
        and branch_points >= config.min_branch_points
    )


def _terminal_profile(
    terminal: int,
    adjacency: list[list[int]],
    vertices: np.ndarray,
    radii: np.ndarray,
    reach_um: float,
) -> list[tuple[float, float]]:
    """``(arc distance from the tip, radius)`` walking inward along one branch.

    The walk stops at a branch point: past a fork the radius belongs to a
    different process, and averaging across it would wash out exactly the local
    swelling this is measuring.
    """
    samples = [(0.0, float(radii[terminal]))]
    neighbours = adjacency[terminal]
    if not neighbours:
        return samples
    previous, current = terminal, neighbours[0]
    distance = float(np.linalg.norm(vertices[current] - vertices[previous]))
    while True:
        samples.append((distance, float(radii[current])))
        if distance >= reach_um:
            break
        onward = [node for node in adjacency[current] if node != previous]
        if len(onward) != 1:
            break
        previous, current = current, onward[0]
        distance += float(np.linalg.norm(vertices[current] - vertices[previous]))
    return samples


def _terminal_shape(
    samples: list[tuple[float, float]],
    config: ContinuityConfig,
    fallback_shaft_um: float | None = None,
) -> tuple[str, float | None, float | None, float | None, str]:
    """Classify a terminal from its radius profile.

    A presynaptic bouton is a swelling: walking back from the tip the radius
    rises above the shaft it hangs off, then returns to it. A process cut by a
    false split has no such maximum -- the radius climbs to the shaft caliber and
    stays there. A genuine thin ending narrows instead.

    ``bouton_head`` is therefore positive evidence that an end is real, and a
    ``blunt`` end is the shape a clean cut makes. Neither is proof: a bouton can
    still be split through its middle, and a thin axon can genuinely stop without
    swelling. The class is a shape, not a verdict.
    """
    head = [radius for distance, radius in samples if distance <= config.head_window_um]
    shaft = [
        radius
        for distance, radius in samples
        if config.head_window_um < distance <= config.head_window_um + config.shaft_window_um
    ]
    source = "profile"
    if not head:
        return "unresolved", None, None, None, "none"
    if len(shaft) < 2:
        # The terminal branch is shorter than the windows. Walking past the fork
        # is not an option -- beyond it the radius may belong to a sibling branch
        # -- but the process this twig hangs off is exactly what the segment's own
        # caliber measures, so use that as the reference and say so. Without this
        # fallback a volume whose terminal branches are shorter than the window
        # reports almost every end `unresolved`, which is a statement about the
        # window, not about the terminals.
        if fallback_shaft_um is None or fallback_shaft_um <= 0:
            return "unresolved", float(max(head)), None, None, "none"
        shaft_radius = float(fallback_shaft_um)
        source = "segment_caliber"
    else:
        shaft_radius = float(np.median(shaft))
    head_radius = float(max(head))
    if shaft_radius <= 0:
        return "unresolved", head_radius, shaft_radius, None, "none"
    ratio = head_radius / shaft_radius
    if ratio >= config.bouton_head_ratio:
        shape = "bouton_head"
    elif ratio <= config.taper_ratio:
        shape = "tapered"
    else:
        shape = "blunt"
    return shape, head_radius, shaft_radius, ratio, source


def _semantic_type(
    radius_um: float | None,
    branch_per_um: float | None,
    length_um: float,
    branch_points: int,
    config: ContinuityConfig,
) -> tuple[str, str]:
    """Decide the caliber class; usable branch density can only override upward.

    Radius is the primary signal because it is a direct physical measurement of
    the process. Branch density promotes a thin but densely branching object to
    ``dendrite_like``: a spiny dendrite measured between its spines can read thin,
    while nothing about a bare axon shaft produces that branch rate. The reverse
    demotion is not applied, because a low branch count is equally consistent
    with a short dendrite fragment. Branch density is ignored entirely unless it
    passes :func:`_branch_density_is_usable`.
    """
    if radius_um is None:
        return "unmeasured", "no_skeleton_radius"
    usable = _branch_density_is_usable(branch_per_um, length_um, branch_points, config)
    if usable and branch_per_um >= config.dendrite_min_branch_per_um:
        return "dendrite_like", "branch_density"
    if radius_um >= config.dendrite_min_radius_um:
        return "dendrite_like", "caliber"
    if radius_um <= config.axon_max_radius_um:
        if usable and branch_per_um > config.axon_max_branch_per_um:
            return "ambiguous_caliber", "thin_but_branching"
        return "axon_like", "caliber"
    return "ambiguous_caliber", "caliber"


def _completeness(free_ends: int, censored: int) -> tuple[str, bool]:
    if free_ends == 0:
        return "complete", True
    if censored == 0 and free_ends >= 2:
        return "isolated_fragment", False
    return ("broken_one_end" if free_ends == 1 else "broken_multi_end"), False


def _unmeasured(label: int, voxel_count: int, arbor: ArborMetrics | None) -> SegmentContinuity:
    return SegmentContinuity(
        label=label,
        voxel_count=voxel_count,
        measured=False,
        semantic_type="unmeasured",
        semantic_basis="no_skeleton",
        completeness_class="unmeasured",
        done=False,
        caliber_radius_um=None,
        caliber_source="none",
        branch_points_per_um=None,
        branch_density_usable=False,
        retained_length_um=0.0,
        terminal_count=0,
        censored_end_count=0,
        free_end_count=0,
        bouton_head_end_count=0,
        terminal_shape_counts={name: 0 for name in TERMINAL_SHAPES},
        free_ends=(),
        arbor=arbor,
    )


def analyze_continuity(
    label: int,
    vertices_um_zyx,
    edges,
    radii_um,
    voxel_count: int,
    config: ContinuityConfig,
) -> SegmentContinuity:
    """Measure one segment's caliber class and unexplained skeleton terminals.

    ``vertices_um_zyx``, ``edges`` and ``radii_um`` are the segment's own skeleton
    in the volume's physical frame, in the form :func:`.arbor.analyze_arbor`
    accepts; degenerate edges are dropped first. ``voxel_count`` is carried
    through for the caller's size rules and is not used in any decision here.

    Returns an ``unmeasured`` record instead of raising when the skeleton has no
    usable edges, so a batch can report the unmeasured population rather than
    silently dropping it.
    """
    label = int(label)
    voxel_count = _positive_integer(voxel_count, "voxel_count")
    if not isinstance(config, ContinuityConfig):
        raise TypeError("config must be a ContinuityConfig")
    vertices = np.asarray(vertices_um_zyx, dtype=np.float64)
    radii = np.asarray(radii_um, dtype=np.float64)
    edges = np.asarray(edges)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_um_zyx must have shape (N, 3)")
    if radii.shape != (len(vertices),):
        raise ValueError("radii_um must supply one radius per vertex")
    edges, radii = _clean_graph(vertices, edges.reshape(-1, 2), radii)
    if not len(edges):
        return _unmeasured(label, voxel_count, None)

    arbor = analyze_arbor(vertices, edges, radii, config.arbor)
    retained = edges[list(arbor.retained_edge_indices)]
    if not len(retained):
        return _unmeasured(label, voxel_count, arbor)

    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for left, right in retained.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)
    degree = np.asarray([len(values) for values in adjacency], dtype=np.int64)
    terminals = np.flatnonzero(degree == 1)

    # Computed up front: it is the fallback shaft reference for terminals whose
    # own branch is too short to supply one.
    caliber = arbor.diameter_median_radius_um
    caliber_source = "diameter"
    if caliber is None:
        caliber = _weighted_median_radius(vertices, retained, radii)
        caliber_source = "retained_skeleton" if caliber is not None else "none"

    extent = np.asarray(config.volume_extent_um, dtype=float)
    ends: list[FreeEnd] = []
    censored_count = 0
    reach = config.head_window_um + config.shaft_window_um
    shape_counts: dict[str, int] = {name: 0 for name in TERMINAL_SHAPES}
    for terminal in terminals.tolist():
        distance, face = _face_distance(vertices[terminal], extent)
        radius = float(radii[terminal])
        censored = distance <= radius + config.face_margin_um
        profile = _terminal_profile(terminal, adjacency, vertices, radii, reach)
        shape, head_radius, shaft_radius, ratio, shaft_source = _terminal_shape(
            profile, config, fallback_shaft_um=caliber
        )
        shape_counts[shape] += 1
        if censored:
            censored_count += 1
            continue
        ends.append(
            FreeEnd(
                vertex_index=int(terminal),
                position_um_zyx=tuple(float(v) for v in vertices[terminal]),
                outward_tangent_zyx=tuple(
                    float(v) for v in _outward_tangent(terminal, adjacency, vertices)
                ),
                radius_um=radius,
                distance_to_face_um=distance,
                nearest_face=face,
                censored=False,
                shape=shape,
                head_radius_um=head_radius,
                shaft_radius_um=shaft_radius,
                head_ratio=ratio,
                shaft_source=shaft_source,
            )
        )
    ends.sort(key=lambda end: -end.distance_to_face_um)

    source = caliber_source
    branch_per_um = (
        arbor.retained_branch_point_count / arbor.retained_length_um
        if arbor.retained_length_um > 0
        else None
    )
    branch_density_usable = _branch_density_is_usable(
        branch_per_um, float(arbor.retained_length_um), arbor.retained_branch_point_count, config
    )
    semantic_type, semantic_basis = _semantic_type(
        caliber,
        branch_per_um,
        float(arbor.retained_length_um),
        arbor.retained_branch_point_count,
        config,
    )
    completeness_class, done = _completeness(len(ends), censored_count)
    return SegmentContinuity(
        label=label,
        voxel_count=voxel_count,
        measured=True,
        semantic_type=semantic_type,
        semantic_basis=semantic_basis,
        completeness_class=completeness_class,
        done=done,
        caliber_radius_um=caliber,
        caliber_source=source,
        branch_points_per_um=branch_per_um,
        branch_density_usable=branch_density_usable,
        retained_length_um=float(arbor.retained_length_um),
        terminal_count=int(len(terminals)),
        censored_end_count=censored_count,
        free_end_count=len(ends),
        bouton_head_end_count=sum(1 for end in ends if end.shape == "bouton_head"),
        terminal_shape_counts=shape_counts,
        free_ends=tuple(ends),
        arbor=arbor,
    )


def semantic_link_allowed(left: str, right: str, *, allow_ambiguous: bool = True) -> bool:
    """Whether two segment types may be joined to repair a split.

    An axon is only ever joined to an axon and a dendrite only to a dendrite.
    ``ambiguous_caliber`` is permitted to pair with anything measured by default,
    because the class exists precisely where caliber cannot decide; set
    ``allow_ambiguous`` to ``False`` for a run that must not guess. ``unmeasured``
    never links, since there is no evidence to constrain it.
    """
    for value in (left, right):
        if value not in SEMANTIC_TYPES:
            raise ValueError(f"Unknown semantic type: {value!r}")
    if "unmeasured" in (left, right):
        return False
    if left == right:
        return left != "ambiguous_caliber" or allow_ambiguous
    if "ambiguous_caliber" in (left, right):
        return allow_ambiguous
    return False
