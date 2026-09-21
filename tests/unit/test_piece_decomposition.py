"""Contract tests for piecewise skeleton decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.metrics.unsupervised.pieces import (
    COMPOSITION_CLASSES,
    PIECE_KINDS,
    PieceConfig,
    decompose_segment,
)

AXON = 0.08  # the volume's typical axon caliber, used as the reference


def chain(length_um, radius, samples=None, start=(5.0, 0.0, 5.0), axis=1):
    """A straight chain of ``length_um`` at constant ``radius``."""
    samples = samples or max(3, int(length_um / 0.03) + 1)
    stop = list(start)
    stop[axis] += length_um
    points = np.linspace(np.asarray(start, float), np.asarray(stop, float), samples)
    edges = np.stack((np.arange(samples - 1), np.arange(1, samples)), axis=1)
    return points, edges, np.full(samples, float(radius))


def join(*parts):
    """Concatenate chains end to end into one skeleton."""
    points, edges, radii = [], [], []
    offset = 0
    tail = None
    for block, block_edges, block_radii in parts:
        if tail is not None:
            block = block + (tail - block[0])
            edges.append(np.array([[offset - 1, offset]]))
        points.append(block)
        edges.append(block_edges + offset)
        radii.append(block_radii)
        offset += len(block)
        tail = block[-1]
    return np.vstack(points), np.vstack(edges), np.concatenate(radii)


def run(vertices, edges, radii, **kwargs):
    return decompose_segment(
        1, vertices, edges, radii, reference_caliber_um=AXON, config=PieceConfig(**kwargs)
    )


def test_axon_ending_in_a_swelling_is_axon_with_terminal():
    """The case a whole-object statistic cannot describe.

    Overall elongation and median radius both fall between the shaft and the
    ending and match neither, which is how these land in `unclassified`.
    """
    result = run(*join(chain(2.0, AXON), chain(0.3, AXON * 3)))
    assert result.composition == "axon_with_terminal"
    assert result.kind_counts["shaft"] >= 1
    assert result.terminal_swelling_count == 1
    assert result.interior_swelling_count == 0
    assert result.shaft_length_um > result.swelling_length_um


def test_a_lone_swelling_is_terminal_only():
    """An ending with no shaft: the 'without axon' case."""
    result = run(*chain(0.9, AXON * 3))
    assert result.composition == "terminal_only"
    assert result.kind_counts["shaft"] == 0
    assert result.terminal_swelling_count >= 1


def test_the_reference_is_not_derived_from_the_segment():
    """A per-segment reference would make a lone swelling invisible.

    Its own median radius *is* the swelling, so relative to itself nothing is
    thick. Passing the volume's axon caliber is what lets it read as an ending.
    """
    vertices, edges, radii = chain(0.9, AXON * 3)
    against_axon = run(vertices, edges, radii)
    against_itself = decompose_segment(
        1, vertices, edges, radii, reference_caliber_um=AXON * 3
    )
    assert against_axon.composition == "terminal_only"
    assert against_itself.kind_counts["swelling"] == 0
    assert against_itself.composition == "shaft_only"


def test_uniform_axon_is_shaft_only():
    result = run(*chain(3.0, AXON))
    assert result.composition == "shaft_only"
    assert result.kind_counts["swelling"] == 0
    assert result.terminal_swelling_count == 0


def test_a_run_of_interior_swellings_is_beaded():
    """En-passant swellings: thick stretches that do not contain a terminal."""
    result = run(*join(
        chain(0.5, AXON), chain(0.3, AXON * 3), chain(0.5, AXON),
        chain(0.3, AXON * 3), chain(0.5, AXON),
    ))
    assert result.interior_swelling_count >= 2
    assert result.terminal_swelling_count == 0
    assert result.composition == "beaded"


def test_too_few_pieces_is_unresolved_not_a_guess():
    result = run(*chain(0.2, AXON), min_pieces_for_composition=3)
    assert result.composition == "unresolved"
    assert result.composition in COMPOSITION_CLASSES


def test_pieces_are_cut_to_about_the_target_length():
    result = run(*chain(2.0, AXON), piece_length_um=0.4)
    assert result.piece_count == pytest.approx(5, abs=1)
    assert all(p.length_um >= 0.2 for p in result.pieces)
    assert result.total_length_um == pytest.approx(2.0, rel=0.05)


def test_no_trailing_sliver_piece():
    """A leftover vertex joins the previous piece rather than becoming one."""
    result = run(*chain(1.05, AXON), piece_length_um=0.5)
    assert all(len(p.vertex_indices) >= 2 for p in result.pieces)
    assert all(p.length_um > 0 for p in result.pieces)


def test_a_piece_never_straddles_a_branch_point():
    """Mixing two processes' radii into one median would describe neither."""
    shaft = chain(1.0, AXON)
    vertices, edges, radii = shaft
    # A thick branch hanging off the middle.
    mid = len(vertices) // 2
    extra = vertices[mid] + np.array([0.0, 0.0, 0.5])
    vertices = np.vstack((vertices, extra))
    radii = np.concatenate((radii, [AXON * 6]))
    edges = np.vstack((edges, [[mid, len(vertices) - 1]]))
    result = run(vertices, edges, radii)
    stub = len(vertices) - 1
    # A branch point is an endpoint of every branch meeting there, so sharing it
    # is correct. What must not happen is a piece spanning the stub and the
    # shaft's interior, which would average two processes into one median.
    shaft_interior = {mid - 1, mid + 1}
    for piece in result.pieces:
        members = set(piece.vertex_indices)
        assert not (stub in members and members & shaft_interior)
    # The stub is its own piece and reads as thick.
    stub_pieces = [p for p in result.pieces if stub in set(p.vertex_indices)]
    assert len(stub_pieces) == 1
    assert stub_pieces[0].kind == "swelling"


def test_straightness_detects_a_doubling_back():
    straight = run(*chain(1.0, AXON))
    assert all(p.straightness > 0.95 for p in straight.pieces)
    # A hairpin: out and back along the same line.
    out = np.linspace([5, 0, 5], [5, 0.5, 5], 20)
    back = np.linspace([5, 0.5, 5], [5, 0.02, 5], 20)
    points = np.vstack((out, back))
    edges = np.stack((np.arange(len(points) - 1), np.arange(1, len(points))), axis=1)
    hairpin = run(points, edges, np.full(len(points), AXON), piece_length_um=5.0)
    assert hairpin.pieces[0].straightness < 0.3


def test_validation():
    vertices, edges, radii = chain(1.0, AXON)
    with pytest.raises(ValueError):
        decompose_segment(1, vertices, edges, radii, reference_caliber_um=0)
    with pytest.raises(ValueError):
        decompose_segment(1, vertices, edges, radii[:-1], reference_caliber_um=AXON)
    with pytest.raises(ValueError):
        decompose_segment(
            1, vertices, np.vstack((edges, [[0, 999]])), radii, reference_caliber_um=AXON
        )
    with pytest.raises(ValueError):
        PieceConfig(swelling_ratio=1.0)
    with pytest.raises(ValueError):
        PieceConfig(thin_ratio=1.0)


def test_empty_graph_is_empty_not_an_error():
    result = decompose_segment(
        1, np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int64), np.full(1, AXON),
        reference_caliber_um=AXON,
    )
    assert result.piece_count == 0
    assert result.composition == "unresolved"
    assert set(result.kind_counts) == set(PIECE_KINDS)
