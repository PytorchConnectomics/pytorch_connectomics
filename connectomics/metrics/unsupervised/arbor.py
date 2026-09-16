"""Physical skeleton measurements after conservative terminal-twig pruning.

Skeleton trees do not establish cell identity: many skeletonizers construct
trees by design. These features describe a possible backbone independently of
whole-object cross sections, which can be inflated by spines and branching.
All lengths and radii are in micrometers; vertex coordinates use z, y, x order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ArborConfig",
    "ArborMetrics",
    "BackboneConfig",
    "analyze_arbor",
    "is_dendrite_backbone_candidate",
]


@dataclass(frozen=True)
class ArborConfig:
    """A twig must satisfy all three limits; component diameters are preserved.

    Length limits include previously removed distal paths, preventing iterative
    pruning from consuming a long arbor in several individually short steps.
    """

    max_twig_length_um: float = 1.0
    max_twig_radius_um: float = 0.10
    max_twig_radius_ratio: float = 0.6

    def __post_init__(self) -> None:
        for name in ("max_twig_length_um", "max_twig_radius_um", "max_twig_radius_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.max_twig_radius_ratio > 1:
            raise ValueError("max_twig_radius_ratio must be at most one")


@dataclass(frozen=True)
class BackboneConfig:
    """Minimum physical size and slenderness of a candidate dendrite backbone."""

    min_diameter_length_um: float = 5.0
    min_diameter_radius_um: float = 0.15
    min_diameter_to_width_ratio: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "min_diameter_length_um",
            "min_diameter_radius_um",
            "min_diameter_to_width_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ArborMetrics:
    """JSON-safe graph features, with original graph indices for visualization.

    Branch points are vertices of degree at least three. Branches are maximal
    chains whose interiors have degree two; a closed degree-two ring counts as
    one branch. Long branches exceed ``max_twig_length_um``. Isolated vertices
    count as components, not terminals. Radius statistics use each edge's half
    length as the weight of each endpoint (a vertex quadrature approximation).

    Every forest component has an exact weighted geodesic diameter. If any
    component has a cycle, pruning is disabled and the overall diameter and
    associated measurements are unknown; acyclic component diameters remain
    available. Neither acyclicity nor any returned feature is a cell-type or
    correctness assertion.
    """

    vertex_count: int
    edge_count: int
    component_count: int
    has_cycles: bool
    cyclic_component_count: int
    total_length_um: float
    retained_length_um: float
    pruned_length_um: float
    pruned_length_fraction: float
    pruning_pass_count: int
    terminal_count: int
    branch_point_count: int
    retained_terminal_count: int
    retained_branch_point_count: int
    retained_branch_count: int
    retained_long_branch_count: int
    component_diameters_um: tuple[float | None, ...]
    diameter_length_um: float | None
    diameter_chord_um: float | None
    diameter_tortuosity: float | None
    diameter_median_radius_um: float | None
    diameter_radius_p10_um: float | None
    diameter_radius_p90_um: float | None
    diameter_radius_cv: float | None
    diameter_total_bending_deg: float | None
    diameter_fraction_of_retained_length: float | None
    retained_edge_indices: tuple[int, ...]
    retained_vertex_indices: tuple[int, ...]
    diameter_vertex_indices: tuple[int, ...]


def _validate_graph(vertices, edges, radii) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    edges = np.asarray(edges)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices_um_zyx must be a finite array of shape (N, 3)")
    if radii.shape != (len(vertices),) or not np.all(np.isfinite(radii) & (radii > 0)):
        raise ValueError("radii_um must contain one finite positive radius per vertex")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edges must be an integer array of shape (M, 2)")
    if edges.size and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError("edge indices must address existing vertices")
    edges = edges.astype(np.int64)
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self-loop edges are not supported")
    if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        raise ValueError("duplicate undirected edges are not supported")
    if len(edges):
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        if not np.all(np.isfinite(lengths) & (lengths > 0)):
            raise ValueError("edge lengths must be finite and positive")
    return vertices, edges, radii


def _components(adjacency: list[list[tuple[int, int]]]) -> list[tuple[list[int], bool]]:
    seen: set[int] = set()
    components = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        vertices = []
        degree_sum = 0
        while stack:
            vertex = stack.pop()
            vertices.append(vertex)
            degree_sum += len(adjacency[vertex])
            for neighbor, _ in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append((vertices, degree_sum // 2 >= len(vertices)))
    return components


def _farthest(start: int, adjacency, lengths) -> tuple[int, dict[int, int], dict[int, float]]:
    parents = {start: -1}
    distances = {start: 0.0}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for neighbor, edge in adjacency[vertex]:
            if neighbor not in parents:
                parents[neighbor] = vertex
                distances[neighbor] = distances[vertex] + lengths[edge]
                stack.append(neighbor)
    maximum = max(distances.values())
    farthest = min(
        vertex
        for vertex, distance in distances.items()
        if np.isclose(distance, maximum, rtol=1e-12, atol=1e-12)
    )
    return farthest, parents, distances


def _tree_diameter(start: int, adjacency, lengths) -> tuple[float, list[int]]:
    first, _, _ = _farthest(start, adjacency, lengths)
    last, parents, distances = _farthest(first, adjacency, lengths)
    path = [last]
    while path[-1] != first:
        path.append(parents[path[-1]])
    if path[0] > path[-1]:
        path.reverse()
    return float(distances[last]), path


def _path_radius_quantiles(path, vertices, radii, quantiles) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(vertices[path], axis=0), axis=1)
    weights: np.ndarray = np.zeros(len(path), dtype=np.float64)
    weights[:-1] += lengths / 2
    weights[1:] += lengths / 2
    order = np.argsort(radii[path], kind="stable")
    cumulative = np.cumsum(weights[order])
    targets = (np.asarray(quantiles) - 1e-12) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return radii[np.asarray(path)[order[indices]]]


def _exceeds(value: float, limit: float) -> bool:
    """Ignore floating-point noise from rigid coordinate transformations."""
    return bool(value > limit + max(1e-12, abs(limit) * 1e-12))


def is_dendrite_backbone_candidate(
    metrics: ArborMetrics, config: BackboneConfig | None = None
) -> bool:
    """Gate a thick, extended backbone hypothesis, not biological cell identity.

    A connected acyclic graph is required for known backbone measurements, but
    tree topology alone is never evidence of a neuron. Thin spiny axons and
    thick compact objects fail the independent caliber and slenderness gates.
    """
    if config is None:
        config = BackboneConfig()
    length = metrics.diameter_length_um
    radius = metrics.diameter_median_radius_um
    if (
        metrics.component_count != 1
        or metrics.has_cycles
        or length is None
        or radius is None
        or not np.all(np.isfinite([length, radius]))
        or length <= 0
        or radius <= 0
    ):
        return False
    ratio = length / (2 * radius)
    return bool(
        np.isfinite(ratio)
        and not _exceeds(config.min_diameter_length_um, length)
        and not _exceeds(config.min_diameter_radius_um, radius)
        and not _exceeds(config.min_diameter_to_width_ratio, ratio)
    )


def _prune(adjacency, edges, vertices, radii, lengths, protected, config):
    active = np.ones(len(edges), dtype=bool)
    removed_reach = np.zeros(len(vertices), dtype=np.float64)
    passes = 0
    while True:
        current = [[(n, e) for n, e in neighbors if active[e]] for neighbors in adjacency]
        degree = np.array([len(neighbors) for neighbors in current])
        proposals = []
        for leaf in np.flatnonzero(degree == 1):
            chain_vertices = [int(leaf)]
            chain_edges = []
            previous = -1
            vertex = int(leaf)
            while True:
                neighbor, edge = next((n, e) for n, e in current[vertex] if n != previous)
                chain_edges.append(edge)
                chain_vertices.append(neighbor)
                previous, vertex = vertex, neighbor
                if degree[vertex] != 2:
                    break
            if degree[vertex] < 3 or any(edge in protected for edge in chain_edges):
                continue
            distance_from_attachment = np.cumsum(lengths[chain_edges][::-1])[::-1]
            reach = float(np.max(distance_from_attachment + removed_reach[chain_vertices[:-1]]))
            if _exceeds(reach, config.max_twig_length_um):
                continue
            radius = float(_path_radius_quantiles(chain_vertices, vertices, radii, [0.5])[0])
            if not _exceeds(radius, config.max_twig_radius_um) and not _exceeds(
                radius, config.max_twig_radius_ratio * radii[vertex]
            ):
                proposals.append((chain_edges, vertex, reach))
        if not proposals:
            return active, passes
        for chain_edges, attachment, reach in proposals:
            active[chain_edges] = False
            removed_reach[attachment] = max(removed_reach[attachment], reach)
        passes += 1


def _branch_lengths(adjacency, active, lengths) -> list[float]:
    current = [[(n, e) for n, e in neighbors if active[e]] for neighbors in adjacency]
    visited: set[int] = set()
    branch_lengths = []
    starts = sorted(range(len(current)), key=lambda vertex: len(current[vertex]) == 2)
    for start in starts:
        for neighbor, first_edge in current[start]:
            if first_edge in visited:
                continue
            visited.add(first_edge)
            length = lengths[first_edge]
            vertex = neighbor
            while len(current[vertex]) == 2:
                choices = [(n, e) for n, e in current[vertex] if e not in visited]
                if not choices:
                    break
                vertex, edge = choices[0]
                visited.add(edge)
                length += lengths[edge]
            branch_lengths.append(float(length))
    return branch_lengths


def analyze_arbor(
    vertices_um_zyx: np.ndarray,
    edges: np.ndarray,
    radii_um: np.ndarray,
    config: ArborConfig | None = None,
) -> ArborMetrics:
    """Measure a skeleton forest without connecting disconnected components.

    Duplicate edges, self loops, zero-length edges, invalid indices, nonfinite
    coordinates, and nonpositive radii raise ``ValueError``. Cycles are reported
    rather than silently replaced with spanning trees. Runtime and working
    memory do not require an all-pairs distance matrix.
    """
    if config is None:
        config = ArborConfig()
    vertices, edges, radii = _validate_graph(vertices_um_zyx, edges, radii_um)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in vertices]
    edge_lookup = {}
    for index, (left, right) in enumerate(edges):
        adjacency[left].append((int(right), index))
        adjacency[right].append((int(left), index))
        edge_lookup[tuple(sorted((int(left), int(right))))] = index
    components = _components(adjacency)
    cyclic_count = sum(cyclic for _, cyclic in components)
    component_diameters: list[float | None] = []
    protected: set[int] = set()
    diameter = 0.0
    diameter_path: list[int] = []
    for members, cyclic in components:
        if cyclic:
            component_diameters.append(None)
            continue
        length, path = _tree_diameter(min(members), adjacency, lengths)
        component_diameters.append(length)
        protected.update(edge_lookup[tuple(sorted(pair))] for pair in zip(path[:-1], path[1:]))
        if not diameter_path or length > diameter + 1e-12:
            diameter, diameter_path = length, path
    active: np.ndarray
    if cyclic_count:
        active, passes = np.ones(len(edges), dtype=bool), 0
        diameter_path = []
    else:
        active, passes = _prune(adjacency, edges, vertices, radii, lengths, protected, config)
    before_degree = np.array([len(neighbors) for neighbors in adjacency])
    after_degree = np.array([sum(active[edge] for _, edge in row) for row in adjacency])
    retained_vertices = np.flatnonzero((after_degree > 0) | (before_degree == 0))
    branches = _branch_lengths(adjacency, active, lengths)
    total_length = float(lengths.sum())
    retained_length = float(lengths[active].sum())
    pruned_length = float(lengths[~active].sum())
    chord = None if cyclic_count else 0.0
    tortuosity = radius_median = radius_p10 = radius_p90 = radius_cv = None
    bending = None if cyclic_count else 0.0
    if len(diameter_path) >= 2:
        points = vertices[diameter_path]
        differences = np.diff(points, axis=0)
        path_lengths = np.linalg.norm(differences, axis=1)
        chord = float(np.linalg.norm(points[-1] - points[0]))
        tortuosity = diameter / chord if chord > 0 else None
        radius_p10, radius_median, radius_p90 = (
            float(value)
            for value in _path_radius_quantiles(diameter_path, vertices, radii, [0.1, 0.5, 0.9])
        )
        weights: np.ndarray = np.zeros(len(points), dtype=np.float64)
        weights[:-1] += path_lengths / 2
        weights[1:] += path_lengths / 2
        mean = float(np.average(radii[diameter_path], weights=weights))
        radius_cv = float(
            np.sqrt(np.average((radii[diameter_path] - mean) ** 2, weights=weights)) / mean
        )
        directions = differences / path_lengths[:, None]
        turns = np.einsum("ij,ij->i", directions[:-1], directions[1:])
        bending = float(np.degrees(np.arccos(np.clip(turns, -1, 1))).sum())
    return ArborMetrics(
        vertex_count=len(vertices),
        edge_count=len(edges),
        component_count=len(components),
        has_cycles=bool(cyclic_count),
        cyclic_component_count=cyclic_count,
        total_length_um=total_length,
        retained_length_um=retained_length,
        pruned_length_um=pruned_length,
        pruned_length_fraction=pruned_length / total_length if total_length else 0.0,
        pruning_pass_count=passes,
        terminal_count=int(np.count_nonzero(before_degree == 1)),
        branch_point_count=int(np.count_nonzero(before_degree >= 3)),
        retained_terminal_count=int(np.count_nonzero(after_degree == 1)),
        retained_branch_point_count=int(np.count_nonzero(after_degree >= 3)),
        retained_branch_count=len(branches),
        retained_long_branch_count=sum(
            _exceeds(length, config.max_twig_length_um) for length in branches
        ),
        component_diameters_um=tuple(component_diameters),
        diameter_length_um=None if cyclic_count else diameter,
        diameter_chord_um=chord,
        diameter_tortuosity=tortuosity,
        diameter_median_radius_um=radius_median,
        diameter_radius_p10_um=radius_p10,
        diameter_radius_p90_um=radius_p90,
        diameter_radius_cv=radius_cv,
        diameter_total_bending_deg=bending,
        diameter_fraction_of_retained_length=(
            None
            if cyclic_count
            else min(1.0, diameter / retained_length) if retained_length else 0.0
        ),
        retained_edge_indices=tuple(int(index) for index in np.flatnonzero(active)),
        retained_vertex_indices=tuple(int(index) for index in retained_vertices),
        diameter_vertex_indices=tuple(diameter_path),
    )
