from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from connectomics.metrics.unsupervised.arbor import (
    ArborConfig,
    BackboneConfig,
    analyze_arbor,
    is_dendrite_backbone_candidate,
)


def _spiny_trunk(trunk_radius=0.4, twig_radius=0.06, long_branch=False):
    vertices = [[z, 0, 0] for z in np.linspace(-2, 2, 9)]
    radii = [trunk_radius] * len(vertices)
    edges = [[index, index + 1] for index in range(8)]
    for attachment in (2, 4, 6):
        previous = attachment
        for offset in (0.2, 0.4):
            vertex = len(vertices)
            vertices.append([vertices[attachment][0], 0, offset])
            radii.append(twig_radius)
            edges.append([previous, vertex])
            previous = vertex
    if long_branch:
        previous = 4
        for offset in (0.4, 0.8, 1.2, 1.6):
            vertex = len(vertices)
            vertices.append([0, offset, 0])
            radii.append(0.25)
            edges.append([previous, vertex])
            previous = vertex
    return np.array(vertices), np.array(edges), np.array(radii)


def test_arbor_public_api():
    import connectomics.metrics.unsupervised.arbor as arbor

    assert set(arbor.__all__) == {
        "ArborConfig",
        "ArborMetrics",
        "BackboneConfig",
        "analyze_arbor",
        "is_dendrite_backbone_candidate",
    }


def test_thick_backbone_remains_after_short_thin_twigs_are_pruned():
    result = analyze_arbor(*_spiny_trunk())

    assert result.total_length_um == pytest.approx(5.2)
    assert result.retained_length_um == pytest.approx(4)
    assert result.pruned_length_fraction == pytest.approx(1.2 / 5.2)
    assert result.diameter_length_um == pytest.approx(4)
    assert result.diameter_median_radius_um == pytest.approx(0.4)
    assert result.diameter_tortuosity == pytest.approx(1)
    assert result.diameter_total_bending_deg == pytest.approx(0)
    assert result.terminal_count == 5
    assert result.branch_point_count == 3
    assert result.retained_terminal_count == 2
    assert result.retained_branch_point_count == 0
    assert result.retained_long_branch_count == result.retained_branch_count == 1
    assert result.retained_edge_indices == tuple(range(8))
    assert result.diameter_vertex_indices == result.retained_vertex_indices == tuple(range(9))
    json.dumps(asdict(result), allow_nan=False)


def test_thick_long_side_arbor_is_retained():
    result = analyze_arbor(*_spiny_trunk(long_branch=True))

    assert result.retained_length_um == pytest.approx(5.6)
    assert result.pruned_length_um == pytest.approx(1.2)
    assert result.retained_branch_point_count == 1
    assert result.retained_terminal_count == 3
    assert result.retained_branch_count == result.retained_long_branch_count == 3
    assert result.diameter_fraction_of_retained_length == pytest.approx(4 / 5.6)


def test_spiny_thin_axon_does_not_acquire_thick_backbone_radius():
    result = analyze_arbor(*_spiny_trunk(trunk_radius=0.07, twig_radius=0.025))

    assert result.diameter_median_radius_um == pytest.approx(0.07)
    assert result.diameter_radius_p90_um == pytest.approx(0.07)
    assert result.pruned_length_um == pytest.approx(1.2)
    assert result.retained_branch_point_count == 0


@pytest.mark.parametrize("radius, ratio", [(0.2, 0.6), (0.06, 0.1)])
def test_twig_absolute_radius_and_attachment_ratio_are_both_required(radius, ratio):
    result = analyze_arbor(
        *_spiny_trunk(twig_radius=radius),
        config=ArborConfig(max_twig_radius_ratio=ratio),
    )

    assert result.pruned_length_um == 0
    assert result.retained_terminal_count == 5


def test_accumulated_distal_reach_prevents_iterative_peeling_of_long_arbor():
    vertices = np.array(
        [
            [-3, 0, 0],
            [0, 0, 0],
            [3, 0, 0],
            [0, 0.4, 0],
            [0, 0.8, 0],
            [0, 1.2, 0],
            [0, 0.8, 0.3],
            [0, 0.8, -0.3],
        ]
    )
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6], [4, 7]])
    radii = np.array([0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05])
    result = analyze_arbor(vertices, edges, radii, ArborConfig(max_twig_radius_ratio=1.0))

    assert result.pruning_pass_count == 1
    assert result.retained_edge_indices == (0, 1, 2, 3)
    assert result.pruned_length_um == pytest.approx(1.0)
    assert result.retained_length_um == pytest.approx(6.8)


@pytest.mark.parametrize("twig_length_limit", [0.4, 1.0])
def test_physical_rotation_translation_and_edge_order_do_not_change_measurements(twig_length_limit):
    vertices, edges, radii = _spiny_trunk(long_branch=True)
    config = ArborConfig(max_twig_length_um=twig_length_limit)
    rotation, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(3, 3)))
    baseline = asdict(analyze_arbor(vertices, edges, radii, config))
    rotated = asdict(analyze_arbor(vertices @ rotation + [7, -4, 19], edges, radii, config))

    for key, value in baseline.items():
        if isinstance(value, float) or key == "component_diameters_um":
            assert rotated[key] == pytest.approx(value, abs=1e-5)
        else:
            assert rotated[key] == value

    shuffled = asdict(analyze_arbor(vertices, edges[::-1, ::-1], radii, config))
    shuffled.pop("retained_edge_indices")
    baseline.pop("retained_edge_indices")
    for key, value in baseline.items():
        if isinstance(value, float) or key == "component_diameters_um":
            assert shuffled[key] == pytest.approx(value)
        else:
            assert shuffled[key] == value


def test_radius_statistics_weight_arc_length_instead_of_vertex_count():
    vertices = np.zeros((6, 3))
    vertices[:, 0] = [0, 0.01, 0.02, 0.03, 5, 10]
    edges = np.column_stack((np.arange(5), np.arange(1, 6)))
    radii = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2])
    result = analyze_arbor(vertices, edges, radii)

    assert np.median(radii) == 0.8
    assert result.diameter_median_radius_um == pytest.approx(0.2)
    assert result.diameter_radius_p10_um == pytest.approx(0.2)
    assert result.diameter_radius_p90_um == pytest.approx(0.8)


def test_bending_and_tortuosity_follow_diameter_path():
    result = analyze_arbor(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        np.array([[0, 1], [1, 2]]),
        np.full(3, 0.2),
    )

    assert result.diameter_length_um == pytest.approx(2)
    assert result.diameter_chord_um == pytest.approx(np.sqrt(2))
    assert result.diameter_tortuosity == pytest.approx(np.sqrt(2))
    assert result.diameter_total_bending_deg == pytest.approx(90)


def test_diameter_fraction_remains_bounded_despite_path_sum_roundoff():
    # Reversed path accumulation and NumPy's edge sum differ by about 1e-15.
    vertices = np.cumsum(np.random.default_rng(5).normal(size=(501, 3)), axis=0)
    edges = np.column_stack((np.arange(500), np.arange(1, 501)))
    result = analyze_arbor(vertices, edges, np.full(501, 0.2))

    assert result.diameter_fraction_of_retained_length <= 1
    assert result.diameter_fraction_of_retained_length == pytest.approx(1)
    assert result.retained_edge_indices == tuple(range(500))


def test_disconnected_components_preserve_their_own_diameters_without_gap_length():
    result = analyze_arbor(
        np.array([[0, 0, 0], [4, 0, 0], [100, 0, 0], [100.5, 0, 0], [200, 0, 0]]),
        np.array([[0, 1], [2, 3]]),
        np.full(5, 0.05),
    )

    assert result.component_count == 3
    assert result.component_diameters_um == (4, 0.5, 0)
    assert result.diameter_length_um == 4
    assert result.total_length_um == result.retained_length_um == 4.5
    assert result.retained_vertex_indices == (0, 1, 2, 3, 4)
    assert result.terminal_count == 4


def test_cycles_are_explicit_and_never_silently_replaced_with_tree_backbones():
    result = analyze_arbor(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0, 0], [9, 0, 0]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]),
        np.full(6, 0.05),
    )

    assert result.has_cycles
    assert result.cyclic_component_count == 1
    assert result.component_diameters_um == (None, 4)
    assert result.diameter_length_um is None
    assert result.diameter_median_radius_um is None
    assert result.diameter_vertex_indices == ()
    assert result.pruned_length_um == 0
    assert result.retained_branch_count == 2
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("vertex_count", [0, 1, 3])
def test_empty_and_isolated_vertex_graphs_have_json_safe_results(vertex_count):
    result = analyze_arbor(
        np.zeros((vertex_count, 3)), np.empty((0, 2), dtype=int), np.full(vertex_count, 0.1)
    )

    assert result.component_count == vertex_count
    assert result.total_length_um == result.diameter_length_um == 0
    assert result.diameter_median_radius_um is None
    assert result.retained_vertex_indices == tuple(range(vertex_count))
    assert result.retained_terminal_count == 0
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize(
    "vertices, edges, radii, message",
    [
        ([[0, 0]], np.empty((0, 2), int), [1], "vertices_um_zyx"),
        ([[0, 0, np.nan]], np.empty((0, 2), int), [1], "vertices_um_zyx"),
        ([[0, 0, 0]], np.empty((0, 2), int), [0], "radii_um"),
        ([[0, 0, 0]], np.empty((0, 2), int), [np.inf], "radii_um"),
        ([[0, 0, 0]], np.empty((0, 2), int), [], "radii_um"),
        ([[0, 0, 0]], np.empty((0, 2), float), [1], "integer array"),
        ([[0, 0, 0]], [[0, 1]], [1], "indices"),
        ([[0, 0, 0]], [[0, -1]], [1], "indices"),
        ([[0, 0, 0]], [[0, 0]], [1], "self-loop"),
        ([[0, 0, 0], [1, 0, 0]], [[0, 1], [1, 0]], [1, 1], "duplicate"),
        ([[0, 0, 0], [0, 0, 0]], [[0, 1]], [1, 1], "edge lengths"),
    ],
)
def test_invalid_graphs_raise(vertices, edges, radii, message):
    with pytest.raises(ValueError, match=message):
        analyze_arbor(vertices, np.asarray(edges), radii)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_twig_length_um": 0},
        {"max_twig_radius_um": -1},
        {"max_twig_radius_um": np.nan},
        {"max_twig_radius_ratio": np.inf},
        {"max_twig_radius_ratio": 1.01},
    ],
)
def test_invalid_physical_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        ArborConfig(**kwargs)


def test_thick_spiny_backbone_passes_candidate_gate_independent_of_orientation():
    vertices, edges, radii = _spiny_trunk()
    rotation, _ = np.linalg.qr(np.random.default_rng(53).normal(size=(3, 3)))
    original = analyze_arbor(vertices * 2, edges, radii)
    rotated = analyze_arbor(vertices @ rotation * 2, edges, radii)

    assert is_dendrite_backbone_candidate(original)
    assert is_dendrite_backbone_candidate(rotated)


def test_thin_spiny_axon_and_compact_thick_object_fail_independent_backbone_gates():
    vertices, edges, radii = _spiny_trunk(trunk_radius=0.07, twig_radius=0.025)
    thin_axon = analyze_arbor(vertices * 2, edges, radii)
    short_thick = analyze_arbor(*_spiny_trunk())
    wide_object = replace(short_thick, diameter_length_um=6, diameter_median_radius_um=0.4)

    assert not is_dendrite_backbone_candidate(thin_axon)
    assert not is_dendrite_backbone_candidate(short_thick)
    assert not is_dendrite_backbone_candidate(wide_object)
    assert is_dendrite_backbone_candidate(
        short_thick, BackboneConfig(min_diameter_length_um=3, min_diameter_to_width_ratio=4)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"component_count": 0},
        {"component_count": 2},
        {"has_cycles": True},
        {"diameter_length_um": None},
        {"diameter_median_radius_um": None},
        {"diameter_length_um": np.nan},
        {"diameter_median_radius_um": np.inf},
        {"diameter_length_um": 0},
        {"diameter_median_radius_um": -1},
    ],
)
def test_unknown_or_invalid_backbone_measurements_do_not_classify(changes):
    vertices, edges, radii = _spiny_trunk()
    metrics = analyze_arbor(vertices * 2, edges, radii)

    assert not is_dendrite_backbone_candidate(replace(metrics, **changes))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_diameter_length_um": 0},
        {"min_diameter_radius_um": -1},
        {"min_diameter_to_width_ratio": np.nan},
    ],
)
def test_invalid_backbone_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        BackboneConfig(**kwargs)
