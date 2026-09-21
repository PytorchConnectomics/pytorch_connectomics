"""Chop a segment's skeleton into pieces and describe what the pieces are.

A whole-object shape statistic cannot describe a composite object. An axon that
ends in a terminal swelling has one thin elongated part and one short thick part;
its overall elongation and its median radius both land between the two and match
neither, so every single-shape gate rejects it. On LICONN mip1 that is what the
`unclassified` bucket is: 22.7% of segments, 82% of them axon-calibre, failing on
elongation (65%) and length (84%) but almost never on thickness (1.9%), with
section-area variability p50 0.53 -- composite, not uniform.

This module splits the skeleton at branch points, cuts long branches into pieces
of roughly fixed physical length, classifies each piece against a caliber
reference, and reports the *composition*. That separates the two cases a single
class cannot:

``axon_with_terminal``
    a shaft piece and a terminal swelling -- an axon that still carries its ending
``terminal_only``
    a swelling with no shaft -- an ending detached from its axon

Both are geometry. A swelling is not a verified presynaptic bouton, a shaft is
not a verified axon, and ``terminal_only`` does not prove the ending was split
from anything: the reference caliber is supplied by the caller and the thresholds
are ratios against it. All lengths and radii are micrometers, z, y, x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PIECE_KINDS",
    "COMPOSITION_CLASSES",
    "PieceConfig",
    "Piece",
    "PieceDecomposition",
    "decompose_segment",
]

PIECE_KINDS = ("shaft", "swelling", "thin")
COMPOSITION_CLASSES = (
    "axon_with_terminal",
    "terminal_only",
    "shaft_only",
    "beaded",
    "mixed",
    "unresolved",
)


@dataclass(frozen=True)
class PieceConfig:
    """Piece length and the ratios that name a piece, all against a reference.

    ``piece_length_um`` is a target, not a guarantee: a branch shorter than it
    yields one piece, and the last piece of a branch absorbs the remainder rather
    than being emitted as a sliver.

    ``swelling_ratio`` and ``thin_ratio`` compare a piece's median radius to the
    **caller-supplied** ``reference_caliber_um``. Using a per-segment reference
    would hide exactly the case of interest: a segment that is *entirely* one
    swelling has a per-segment median equal to the swelling, so nothing would read
    as thick. Pass the volume's typical axon caliber instead, and a lone ending
    reads as a swelling because it is thick *for an axon*.
    """

    piece_length_um: float = 0.4
    swelling_ratio: float = 1.5
    thin_ratio: float = 0.7
    min_pieces_for_composition: int = 2

    def __post_init__(self) -> None:
        for name in ("piece_length_um", "swelling_ratio", "thin_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.swelling_ratio <= 1:
            raise ValueError("swelling_ratio must exceed one")
        if self.thin_ratio >= 1:
            raise ValueError("thin_ratio must be below one")
        if (
            isinstance(self.min_pieces_for_composition, bool)
            or self.min_pieces_for_composition < 1
        ):
            raise ValueError("min_pieces_for_composition must be a positive integer")
        object.__setattr__(
            self, "min_pieces_for_composition", int(self.min_pieces_for_composition)
        )


@dataclass(frozen=True)
class Piece:
    """One stretch of skeleton, with the indices needed to draw it.

    ``straightness`` is the end-to-end chord over the arc length, so 1 is a
    straight run and a low value is a coil or a doubling back.
    """

    index: int
    length_um: float
    median_radius_um: float
    max_radius_um: float
    radius_ratio: float
    straightness: float
    kind: str
    has_terminal: bool
    vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class PieceDecomposition:
    """What the pieces are and, from their arrangement, what the segment is.

    ``composition`` is ``unresolved`` when there are too few pieces to say
    anything about arrangement -- never a guess. ``terminal_swelling_count``
    counts swellings that contain a skeleton terminal; ``interior_swelling_count``
    counts those that do not, which is what a run of en-passant swellings looks
    like.
    """

    label: int
    reference_caliber_um: float
    piece_count: int
    total_length_um: float
    kind_counts: dict[str, int]
    terminal_swelling_count: int
    interior_swelling_count: int
    shaft_length_um: float
    swelling_length_um: float
    composition: str
    pieces: tuple[Piece, ...]


def _branches(adjacency: list[list[int]]) -> list[list[int]]:
    """Maximal chains whose interior vertices all have degree two.

    Splitting here first matters: a piece that straddled a fork would mix two
    processes' radii into one median and describe neither.
    """
    keys = {index for index, neighbours in enumerate(adjacency) if len(neighbours) != 2}
    if not keys:
        keys = {0} if adjacency else set()
    seen: set[tuple[int, int]] = set()
    chains: list[list[int]] = []
    for start in sorted(keys):
        for neighbour in adjacency[start]:
            edge = (min(start, neighbour), max(start, neighbour))
            if edge in seen:
                continue
            seen.add(edge)
            chain = [start, neighbour]
            previous, current = start, neighbour
            while len(adjacency[current]) == 2:
                onward = [node for node in adjacency[current] if node != previous]
                if not onward:
                    break
                following = onward[0]
                edge = (min(current, following), max(current, following))
                if edge in seen:
                    break
                seen.add(edge)
                chain.append(following)
                previous, current = current, following
            chains.append(chain)
    return chains


def _cut(chain: list[int], vertices: np.ndarray, target_um: float) -> list[list[int]]:
    """Split a chain into runs of about ``target_um``, with no trailing sliver."""
    if len(chain) < 2:
        return [chain] if chain else []
    steps = np.linalg.norm(np.diff(vertices[chain], axis=0), axis=1)
    if steps.sum() <= target_um:
        return [chain]
    runs: list[list[int]] = []
    current = [chain[0]]
    accumulated = 0.0
    for index, step in enumerate(steps):
        current.append(chain[index + 1])
        accumulated += float(step)
        if accumulated >= target_um and index + 2 < len(chain):
            runs.append(current)
            current = [chain[index + 1]]
            accumulated = 0.0
    if len(current) >= 2:
        runs.append(current)
    elif runs:
        # A single leftover vertex joins the previous run instead of becoming a
        # one-vertex piece with no length and no radius median.
        runs[-1].extend(current[1:])
    return runs


def _kind(ratio: float, config: PieceConfig) -> str:
    if ratio >= config.swelling_ratio:
        return "swelling"
    if ratio <= config.thin_ratio:
        return "thin"
    return "shaft"


def _composition(
    kind_counts: dict[str, int],
    terminal_swellings: int,
    interior_swellings: int,
    piece_count: int,
    config: PieceConfig,
) -> str:
    if piece_count < config.min_pieces_for_composition:
        return "unresolved"
    shafts = kind_counts["shaft"]
    swellings = kind_counts["swelling"]
    if shafts and terminal_swellings:
        return "axon_with_terminal"
    if terminal_swellings and not shafts:
        return "terminal_only"
    if interior_swellings >= 2:
        return "beaded"
    if shafts and not swellings:
        return "shaft_only"
    return "mixed"


def decompose_segment(
    label: int,
    vertices_um_zyx,
    edges,
    radii_um,
    *,
    reference_caliber_um: float,
    config: PieceConfig | None = None,
) -> PieceDecomposition:
    """Split one segment's skeleton into pieces and name their arrangement.

    ``edges`` must already be the graph the caller wants described -- typically
    the retained edges from :func:`.arbor.analyze_arbor`, so pruned twigs do not
    become pieces. ``reference_caliber_um`` is the caliber a piece is thick or
    thin *relative to*; see :class:`PieceConfig` for why it is not derived from
    this segment.
    """
    config = config or PieceConfig()
    reference = float(reference_caliber_um)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_caliber_um must be finite and positive")
    vertices = np.asarray(vertices_um_zyx, dtype=np.float64)
    radii = np.asarray(radii_um, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if radii.shape != (len(vertices),):
        raise ValueError("radii_um must supply one radius per vertex")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError("edge indices must address existing vertices")

    empty = PieceDecomposition(
        label=int(label),
        reference_caliber_um=reference,
        piece_count=0,
        total_length_um=0.0,
        kind_counts={name: 0 for name in PIECE_KINDS},
        terminal_swelling_count=0,
        interior_swelling_count=0,
        shaft_length_um=0.0,
        swelling_length_um=0.0,
        composition="unresolved",
        pieces=(),
    )
    if not len(edges):
        return empty

    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for left, right in edges.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)
    degree = np.asarray([len(values) for values in adjacency], dtype=np.int64)
    terminals = set(np.flatnonzero(degree == 1).tolist())

    pieces: list[Piece] = []
    kind_counts = {name: 0 for name in PIECE_KINDS}
    terminal_swellings = interior_swellings = 0
    shaft_length = swelling_length = total_length = 0.0
    for chain in _branches(adjacency):
        for run in _cut(chain, vertices, config.piece_length_um):
            if len(run) < 2:
                continue
            points = vertices[run]
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            length = float(steps.sum())
            if length <= 0:
                continue
            chord = float(np.linalg.norm(points[-1] - points[0]))
            median = float(np.median(radii[run]))
            ratio = median / reference
            kind = _kind(ratio, config)
            has_terminal = bool(terminals.intersection(run))
            kind_counts[kind] += 1
            total_length += length
            if kind == "shaft":
                shaft_length += length
            elif kind == "swelling":
                swelling_length += length
                if has_terminal:
                    terminal_swellings += 1
                else:
                    interior_swellings += 1
            pieces.append(
                Piece(
                    index=len(pieces),
                    length_um=length,
                    median_radius_um=median,
                    max_radius_um=float(np.max(radii[run])),
                    radius_ratio=ratio,
                    straightness=chord / length,
                    kind=kind,
                    has_terminal=has_terminal,
                    vertex_indices=tuple(int(v) for v in run),
                )
            )
    if not pieces:
        return empty
    return PieceDecomposition(
        label=int(label),
        reference_caliber_um=reference,
        piece_count=len(pieces),
        total_length_um=total_length,
        kind_counts=kind_counts,
        terminal_swelling_count=terminal_swellings,
        interior_swelling_count=interior_swellings,
        shaft_length_um=shaft_length,
        swelling_length_um=swelling_length,
        composition=_composition(
            kind_counts, terminal_swellings, interior_swellings, len(pieces), config
        ),
        pieces=tuple(pieces),
    )
