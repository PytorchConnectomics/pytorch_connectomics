"""Contract tests for semantic type and endpoint continuity without ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.metrics.unsupervised.continuity import (
    COMPLETENESS_CLASSES,
    SEMANTIC_TYPES,
    TERMINAL_SHAPES,
    ContinuityConfig,
    analyze_continuity,
    semantic_link_allowed,
)

EXTENT = (10.0, 10.0, 10.0)


def straight(start, stop, radius, samples=11):
    """A straight chain of ``samples`` vertices from ``start`` to ``stop``."""
    points = np.linspace(np.asarray(start, float), np.asarray(stop, float), samples)
    edges = np.stack((np.arange(samples - 1), np.arange(1, samples)), axis=1)
    return points, edges, np.full(samples, float(radius))


def analyze(vertices, edges, radii, voxels=5000, **kwargs):
    config = ContinuityConfig(volume_extent_um=EXTENT, **kwargs)
    return analyze_continuity(1, vertices, edges, radii, voxels, config)


def test_through_axon_touching_two_faces_is_complete():
    result = analyze(*straight((5, 0.02, 5), (5, 9.98, 5), 0.1))
    assert result.semantic_type == "axon_like"
    assert result.completeness_class == "complete"
    assert result.done is True
    assert result.free_end_count == 0
    assert result.censored_end_count == 2


def test_axon_with_one_interior_end_is_broken_and_reports_a_link_site():
    result = analyze(*straight((5, 0.02, 5), (5, 6.0, 5), 0.1))
    assert result.semantic_type == "axon_like"
    assert result.completeness_class == "broken_one_end"
    assert result.done is False
    assert result.free_end_count == 1
    (end,) = result.free_ends
    assert end.position_um_zyx == pytest.approx((5.0, 6.0, 5.0))
    # Outward tangent points away from the segment, i.e. further along +y.
    assert end.outward_tangent_zyx == pytest.approx((0.0, 1.0, 0.0))
    assert end.nearest_face == "ymax"
    assert end.distance_to_face_um == pytest.approx(4.0)


def test_fragment_with_no_censored_end_is_isolated():
    result = analyze(*straight((5, 3.0, 5), (5, 6.0, 5), 0.1))
    assert result.completeness_class == "isolated_fragment"
    assert result.free_end_count == 2
    assert result.censored_end_count == 0
    # Most interior end first.
    assert [end.distance_to_face_um for end in result.free_ends] == sorted(
        [end.distance_to_face_um for end in result.free_ends], reverse=True
    )


def test_short_blobby_stub_is_still_axon_like_and_broken():
    """The case the profile classifier misses: too short and round to be a tube.

    Global PCA elongation here is ~1, so an elongation-gated rule cannot call this
    an axon at all, and its brokenness becomes invisible. Caliber still can.
    """
    vertices, edges, radii = straight((5, 5.0, 5), (5, 5.3, 5), 0.12, samples=4)
    result = analyze(vertices, edges, radii, voxels=1200)
    assert result.semantic_type == "axon_like"
    assert result.caliber_radius_um == pytest.approx(0.12)
    assert result.done is False
    assert result.free_end_count == 2


def test_thick_process_is_dendrite_like_by_caliber():
    result = analyze(*straight((5, 1.0, 5), (5, 8.0, 5), 0.6))
    assert result.semantic_type == "dendrite_like"
    assert result.semantic_basis == "caliber"


def test_thin_but_densely_branching_is_promoted_to_dendrite():
    """A spiny dendrite measured between spines reads thin; branch rate does not.

    The shaft is 8 um because branch density is only consulted over a skeleton
    long enough for the rate to be a measurement rather than counting noise.
    """
    vertices, edges, radii = straight((5, 1.0, 5), (5, 9.0, 5), 0.2, samples=21)
    extra_vertices = []
    extra_edges = []
    extra_radii = []
    for index in range(2, 19, 2):  # nine branch points on an 8 um shaft
        extra_edges.append((index, len(vertices) + len(extra_vertices)))
        extra_vertices.append(vertices[index] + np.array([0.0, 0.0, 0.5]))
        extra_radii.append(0.18)
    vertices = np.vstack((vertices, np.asarray(extra_vertices)))
    radii = np.concatenate((radii, np.asarray(extra_radii)))
    edges = np.vstack((edges, np.asarray(extra_edges)))
    result = analyze(vertices, edges, radii)
    assert result.branch_points_per_um is not None
    assert result.branch_points_per_um >= 0.5
    assert result.semantic_type == "dendrite_like"
    assert result.semantic_basis == "branch_density"


def test_radius_inset_prevents_a_thick_touching_end_from_reading_as_free():
    """A medial-axis vertex of a thick process is inset by about one radius."""
    thick = analyze(*straight((5, 0.55, 5), (5, 8.0, 5), 0.6))
    assert thick.free_ends[0].position_um_zyx[1] == pytest.approx(8.0)
    assert thick.censored_end_count == 1
    # The same geometry on a thin process is not censored at that distance.
    thin = analyze(*straight((5, 0.55, 5), (5, 8.0, 5), 0.1))
    assert thin.censored_end_count == 0


def test_twigs_removed_by_pruning_are_not_free_ends():
    vertices, edges, radii = straight((5, 1.0, 5), (5, 8.0, 5), 0.2, samples=15)
    vertices = np.vstack((vertices, vertices[7] + np.array([0.0, 0.0, 0.05])))
    radii = np.concatenate((radii, [0.02]))
    edges = np.vstack((edges, [[7, len(vertices) - 1]]))
    result = analyze(vertices, edges, radii)
    assert result.terminal_count == 2
    assert result.free_end_count == 2


def test_degenerate_edges_are_dropped_not_fatal():
    vertices, edges, radii = straight((5, 1.0, 5), (5, 8.0, 5), 0.2)
    edges = np.vstack((edges, edges[:1], [[3, 3]]))
    result = analyze(vertices, edges, radii)
    assert result.measured is True
    assert result.free_end_count == 2


def test_empty_graph_is_unmeasured_rather_than_an_error():
    vertices = np.zeros((1, 3))
    result = analyze(vertices, np.zeros((0, 2), dtype=np.int64), np.full(1, 0.1))
    assert result.measured is False
    assert result.semantic_type == "unmeasured"
    assert result.completeness_class == "unmeasured"
    assert result.done is False


def test_vocabularies_are_closed():
    assert set(SEMANTIC_TYPES) == {
        "axon_like",
        "dendrite_like",
        "ambiguous_caliber",
        "unmeasured",
    }
    assert set(COMPLETENESS_CLASSES) == {
        "complete",
        "broken_one_end",
        "broken_multi_end",
        "isolated_fragment",
        "unmeasured",
    }


@pytest.mark.parametrize(
    "left,right,allowed",
    [
        ("axon_like", "axon_like", True),
        ("dendrite_like", "dendrite_like", True),
        ("axon_like", "dendrite_like", False),
        ("axon_like", "ambiguous_caliber", True),
        ("axon_like", "unmeasured", False),
        ("unmeasured", "unmeasured", False),
    ],
)
def test_semantic_link_constraint(left, right, allowed):
    assert semantic_link_allowed(left, right) is allowed


def test_ambiguous_can_be_forbidden():
    assert semantic_link_allowed("axon_like", "ambiguous_caliber", allow_ambiguous=False) is False
    assert semantic_link_allowed("axon_like", "axon_like", allow_ambiguous=False) is True


def test_unknown_semantic_type_raises():
    with pytest.raises(ValueError):
        semantic_link_allowed("axon_like", "glia")


def test_config_rejects_inverted_caliber_gates():
    with pytest.raises(ValueError):
        ContinuityConfig(volume_extent_um=EXTENT, axon_max_radius_um=0.8)


def branchy(length_um, branch_count, radius=0.2, samples_per_um=10):
    """A straight shaft of ``length_um`` carrying ``branch_count`` side branches."""
    n = max(3, int(length_um * samples_per_um) + 1)
    vertices = np.linspace([5, 1.0, 5], [5, 1.0 + length_um, 5], n)
    edges = [[i, i + 1] for i in range(n - 1)]
    radii = [radius] * n
    step = max(1, (n - 2) // max(branch_count, 1))
    for k in range(branch_count):
        anchor = 1 + k * step
        if anchor >= n - 1:
            break
        edges.append([anchor, len(vertices)])
        vertices = np.vstack((vertices, vertices[anchor] + np.array([0.0, 0.0, 0.6])))
        radii.append(radius * 0.95)
    return vertices, np.asarray(edges), np.asarray(radii)


def test_short_fragment_is_not_called_dendrite_by_branch_rate():
    """Two branch points on a ~1.4 um fragment is 1.4/um -- and is meaningless.

    This was a real defect: on a fine-caliber volume it made 139 of 300 segments
    `dendrite_like` purely from skeletonization noise, and those bogus types then
    vetoed 43% of all link candidates on the semantic gate.
    """
    result = analyze(*branchy(1.4, 2, radius=0.08))
    assert result.retained_length_um < 5.0
    assert result.branch_points_per_um > 0.5  # the rate alone would say dendrite
    assert result.branch_density_usable is False
    assert result.semantic_type == "axon_like"
    assert result.semantic_basis == "caliber"


def test_long_densely_branching_shaft_still_reaches_dendrite():
    """The guard must not silence branch density where it is actually measured."""
    result = analyze(*branchy(8.0, 6, radius=0.2))
    assert result.retained_length_um >= 5.0
    assert result.arbor.retained_branch_point_count >= 3
    assert result.branch_density_usable is True
    assert result.semantic_type == "dendrite_like"
    assert result.semantic_basis == "branch_density"


def test_long_but_barely_branching_is_not_promoted():
    result = analyze(*branchy(8.0, 1, radius=0.1))
    assert result.branch_density_usable is False  # too few branch points
    assert result.semantic_type == "axon_like"


def test_branch_gates_are_configurable():
    vertices, edges, radii = branchy(1.4, 2, radius=0.08)
    permissive = analyze(vertices, edges, radii, min_branch_length_um=0.5, min_branch_points=1)
    assert permissive.branch_density_usable is True
    assert permissive.semantic_type == "dendrite_like"


def test_config_rejects_bad_branch_gates():
    with pytest.raises(ValueError):
        ContinuityConfig(volume_extent_um=EXTENT, min_branch_points=0)
    with pytest.raises(ValueError):
        ContinuityConfig(volume_extent_um=EXTENT, min_branch_length_um=0)


def test_out_of_range_edge_index_raises_a_clear_error():
    vertices, edges, radii = straight((5, 1.0, 5), (5, 8.0, 5), 0.2, samples=5)
    edges = np.vstack((edges, [[0, 99]]))
    with pytest.raises(ValueError, match="edge indices must address existing vertices"):
        analyze(vertices, edges, radii)


def shaft_with_tip(tip_radius, shaft_radius=0.08, length_um=2.0, head_um=0.25, samples=41):
    """A shaft of ``shaft_radius`` whose last ``head_um`` swells to ``tip_radius``.

    The very last vertex keeps the shaft radius scaled down, mimicking a
    skeletonizer putting the terminal vertex near the surface — so a rule that
    merely read the tip vertex would see nothing.
    """
    points = np.linspace([5, 1.0, 5], [5, 1.0 + length_um, 5], samples)
    edges = np.stack((np.arange(samples - 1), np.arange(1, samples)), axis=1)
    arc = np.linspace(0, length_um, samples)
    distance_from_tip = arc[-1] - arc
    radii = np.full(samples, float(shaft_radius))
    swell = distance_from_tip <= head_um
    radii[swell] = tip_radius
    radii[-1] = shaft_radius * 0.5  # tip vertex sits near the surface
    return points, edges, radii


def test_bouton_head_terminal_is_detected():
    """A swelling above the shaft is positive evidence the end is a real ending."""
    result = analyze(*shaft_with_tip(tip_radius=0.20, shaft_radius=0.08))
    end = next(e for e in result.free_ends if e.position_um_zyx[1] > 2.0)
    assert end.shape == "bouton_head"
    assert end.head_radius_um == pytest.approx(0.20)
    assert end.shaft_radius_um == pytest.approx(0.08)
    assert end.head_ratio == pytest.approx(2.5)
    assert result.bouton_head_end_count >= 1


def test_cut_cylinder_reads_blunt_not_bouton():
    """A uniform shaft sliced across has no swelling: this is the split shape."""
    result = analyze(*shaft_with_tip(tip_radius=0.08, shaft_radius=0.08))
    end = next(e for e in result.free_ends if e.position_um_zyx[1] > 2.0)
    assert end.shape == "blunt"
    assert end.head_ratio == pytest.approx(1.0)
    assert result.bouton_head_end_count == 0


def test_narrowing_terminal_reads_tapered():
    """The taper must span the head window: the head statistic is a max.

    Using `max` over the head window is what makes a *swelling* detectable
    anywhere in it. The cost is that a narrowing shorter than the window is not a
    taper by this measure, because the window still contains full-caliber shaft.
    """
    result = analyze(*shaft_with_tip(tip_radius=0.03, shaft_radius=0.10, head_um=0.45))
    end = next(e for e in result.free_ends if e.position_um_zyx[1] > 2.0)
    assert end.shape == "tapered"
    assert end.head_ratio is not None and end.head_ratio < 0.7


def test_narrowing_shorter_than_the_head_window_reads_blunt():
    """Pinned deliberately: this is a limit of the statistic, not a bug.

    A 0.25 um taper inside a 0.30 um window leaves shaft-caliber samples in the
    window, so the max equals the shaft and the end reads blunt. Shrink
    `head_window_um` if sub-window tapers matter for a given grid.
    """
    result = analyze(*shaft_with_tip(tip_radius=0.03, shaft_radius=0.10, head_um=0.25))
    end = next(e for e in result.free_ends if e.position_um_zyx[1] > 2.0)
    assert end.shape == "blunt"
    tighter = analyze(
        *shaft_with_tip(tip_radius=0.03, shaft_radius=0.10, head_um=0.25),
        head_window_um=0.15,
    )
    assert next(
        e for e in tighter.free_ends if e.position_um_zyx[1] > 2.0
    ).shape == "tapered"


def test_tip_vertex_radius_alone_cannot_distinguish_them():
    """Guards the design: the discriminator must be the profile, not the tip.

    Both geometries end on the same small tip-vertex radius, so any rule reading
    only `radius_um` would be forced to call them identical.
    """
    bouton = analyze(*shaft_with_tip(tip_radius=0.20, shaft_radius=0.08))
    cut = analyze(*shaft_with_tip(tip_radius=0.08, shaft_radius=0.08))
    b = next(e for e in bouton.free_ends if e.position_um_zyx[1] > 2.0)
    c = next(e for e in cut.free_ends if e.position_um_zyx[1] > 2.0)
    assert b.radius_um == pytest.approx(c.radius_um)  # identical at the tip
    assert b.shape != c.shape  # but separated by the profile


def test_short_branch_falls_back_to_the_segment_caliber():
    """A branch shorter than the windows still has a reference: the segment itself.

    Without this, a volume whose terminal branches are shorter than the window
    reports nearly every end `unresolved` — a statement about the window, not the
    terminals. On LICONN mip1 that was 68% of ends. The uniform cylinder here is
    correctly `blunt`, and `shaft_source` says the reference was not the profile.
    """
    vertices, edges, radii = straight((5, 5.0, 5), (5, 5.2, 5), 0.1, samples=4)
    result = analyze(vertices, edges, radii, voxels=1200)
    for end in result.free_ends:
        assert end.shape == "blunt"
        assert end.shaft_source == "segment_caliber"
        assert end.head_ratio == pytest.approx(1.0)


def test_shaft_source_is_the_profile_when_the_branch_is_long_enough():
    result = analyze(*shaft_with_tip(tip_radius=0.20, shaft_radius=0.08))
    end = next(e for e in result.free_ends if e.position_um_zyx[1] > 2.0)
    assert end.shaft_source == "profile"


def test_a_swelling_is_still_found_through_the_fallback():
    """The fallback must not blind the bouton test on short branches."""
    vertices, edges, radii = straight((5, 5.0, 5), (5, 5.2, 5), 0.1, samples=4)
    radii = radii.copy()
    radii[-1] = 0.30  # a bulb at one end, on a branch too short to profile
    result = analyze(vertices, edges, radii, voxels=1200)
    assert any(end.shape == "bouton_head" for end in result.free_ends)


def test_shape_counts_cover_every_terminal_including_censored():
    result = analyze(*shaft_with_tip(tip_radius=0.20, shaft_radius=0.08))
    assert sum(result.terminal_shape_counts.values()) == result.terminal_count
    assert set(result.terminal_shape_counts) == set(TERMINAL_SHAPES)


def test_profile_walk_stops_at_a_branch_point():
    """Past a fork the radius belongs to another process and must not be averaged."""
    vertices, edges, radii = straight((5, 1.0, 5), (5, 3.0, 5), 0.08, samples=21)
    # A thick side branch hanging off the midpoint.
    vertices = np.vstack((vertices, vertices[10] + np.array([0.0, 0.0, 0.4])))
    radii = np.concatenate((radii, [0.9]))
    edges = np.vstack((edges, [[10, len(vertices) - 1]]))
    result = analyze(vertices, edges, radii)
    for end in result.free_ends:
        if end.shaft_radius_um is not None:
            assert end.shaft_radius_um < 0.5  # never picked up the 0.9 branch


def test_bouton_thresholds_are_configurable_and_validated():
    vertices, edges, radii = shaft_with_tip(tip_radius=0.12, shaft_radius=0.08)
    assert analyze(vertices, edges, radii, bouton_head_ratio=1.4).free_ends[0].shape == "bouton_head"
    assert analyze(vertices, edges, radii, bouton_head_ratio=2.0).free_ends[0].shape == "blunt"
    with pytest.raises(ValueError):
        ContinuityConfig(volume_extent_um=EXTENT, bouton_head_ratio=1.0)
    with pytest.raises(ValueError):
        ContinuityConfig(volume_extent_um=EXTENT, taper_ratio=1.0)
