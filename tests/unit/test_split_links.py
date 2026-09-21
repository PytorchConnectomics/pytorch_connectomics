"""Contract tests for semantic-constrained split-link proposals."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.decoding.error_correction.split_links import (
    LinkCandidate,
    LinkConfig,
    LinkSite,
    collect_link_sites,
    connected_groups,
    propose_links,
    select_links,
    select_mutual_best,
)
from connectomics.metrics.unsupervised.continuity import (
    ContinuityConfig,
    analyze_continuity,
)

EXTENT = (10.0, 10.0, 10.0)


def site(label, position, tangent, radius=0.1, semantic="axon_like", vertex=0):
    return LinkSite(
        label=label,
        vertex_index=vertex,
        position_um_zyx=tuple(float(v) for v in position),
        outward_tangent_zyx=tuple(float(v) for v in tangent),
        radius_um=radius,
        semantic_type=semantic,
    )


def facing_pair(gap=0.3, **kwargs):
    """Two axon ends pointing at each other across ``gap``."""
    left = site(1, (5, 5, 5), (0, 1, 0), **kwargs)
    right = site(2, (5, 5 + gap, 5), (0, -1, 0), **kwargs)
    return [left, right]


def test_facing_axons_across_a_small_gap_are_accepted():
    candidates = propose_links(facing_pair(), config=LinkConfig())
    assert len(candidates) == 2
    assert all(candidate.accepted for candidate in candidates)
    assert candidates[0].gap_um == pytest.approx(0.3)
    assert candidates[0].tangent_deg == pytest.approx(0.0, abs=1e-6)
    pairs = select_mutual_best(candidates)
    assert [(p.left_label, p.right_label) for p in pairs] == [(1, 2)]


def test_axon_to_dendrite_is_refused_by_the_semantic_gate():
    sites = facing_pair()
    sites[1] = site(2, (5, 5.3, 5), (0, -1, 0), radius=0.6, semantic="dendrite_like")
    candidates = propose_links(sites, config=LinkConfig())
    left = next(c for c in candidates if c.left_label == 1)
    assert left.accepted is False
    assert left.reject_reason == "semantic:axon_like->dendrite_like"
    assert select_mutual_best(candidates) == []


def test_ambiguous_can_be_forbidden_by_config():
    sites = facing_pair()
    sites[1] = site(2, (5, 5.3, 5), (0, -1, 0), semantic="ambiguous_caliber")
    assert propose_links(sites, config=LinkConfig())[0].accepted is True
    strict = LinkConfig(allow_ambiguous_semantic=False)
    rejected = propose_links(sites, config=strict)[0]
    assert rejected.accepted is False
    assert rejected.reject_reason.startswith("semantic:")


def test_gap_beyond_the_limit_yields_no_candidate_at_all():
    assert propose_links(facing_pair(gap=2.0), config=LinkConfig(max_gap_um=1.0)) == []


def test_mismatched_caliber_is_refused():
    sites = [site(1, (5, 5, 5), (0, 1, 0), radius=0.30), site(2, (5, 5.3, 5), (0, -1, 0), radius=0.05)]
    candidate = propose_links(sites, config=LinkConfig())[0]
    assert candidate.accepted is False
    assert candidate.reject_reason == "caliber"
    assert candidate.caliber_ratio == pytest.approx(1 / 6)


def test_a_neighbour_running_alongside_fails_the_tangent_gate():
    """The partner is close, but the free end is not heading towards it."""
    sites = [site(1, (5, 5, 5), (0, 1, 0)), site(2, (5, 5, 5.3), (0, -1, 0))]
    candidate = propose_links(sites, config=LinkConfig())[0]
    assert candidate.accepted is False
    assert candidate.reject_reason == "tangent"
    assert candidate.tangent_deg == pytest.approx(90.0)


def test_affinity_floor_rejects_a_geometrically_perfect_pair():
    """Geometry proposes; affinity decides. A clean gap with dead affinity fails."""
    config = LinkConfig(min_affinity=0.5)
    weak = propose_links(
        facing_pair(), config=config, affinity_probe=lambda a, b: {"mean": 0.1}
    )[0]
    assert weak.accepted is False
    assert weak.reject_reason == "affinity"
    strong = propose_links(
        facing_pair(), config=config, affinity_probe=lambda a, b: {"mean": 0.9}
    )[0]
    assert strong.accepted is True
    assert strong.affinity == {"mean": 0.9}


def test_missing_affinity_is_not_silently_treated_as_passing():
    config = LinkConfig(min_affinity=0.5)
    candidate = propose_links(facing_pair(), config=config, affinity_probe=lambda a, b: None)[0]
    assert candidate.accepted is False
    assert candidate.reject_reason == "no_affinity"


def test_one_tip_route_attaches_to_the_side_of_another_segment():
    """A broken end rejoining the shaft of its continuation has no facing tip."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    shaft = np.array([[5, 5.3, 4.0], [5, 5.3, 5.0], [5, 5.3, 6.0]], dtype=float)
    candidates = propose_links(
        sites,
        config=LinkConfig(),
        skeleton_points={2: shaft},
        skeleton_radii={2: np.full(3, 0.1)},
    )
    assert len(candidates) == 1
    assert candidates[0].accepted is True
    assert candidates[0].right_label == 2
    assert candidates[0].mutual is False
    assert candidates[0].gap_um == pytest.approx(0.3)
    # A one-sided attachment is not a mutual pair and select_mutual_best drops it.
    assert select_mutual_best(candidates) == []


def test_a_segment_never_links_to_itself():
    sites = [site(1, (5, 5, 5), (0, 1, 0), vertex=0), site(1, (5, 5.3, 5), (0, -1, 0), vertex=9)]
    assert propose_links(sites, config=LinkConfig()) == []


def test_one_candidate_per_free_end_and_the_best_partner_wins():
    sites = [
        site(1, (5, 5, 5), (0, 1, 0)),
        site(2, (5, 5.6, 5), (0, -1, 0)),
        site(3, (5, 5.2, 5), (0, -1, 0)),
    ]
    candidates = propose_links(sites, config=LinkConfig())
    left = [c for c in candidates if c.left_label == 1]
    assert len(left) == 1
    assert left[0].right_label == 3  # nearest accepted partner


def test_rejected_candidates_are_reported_not_dropped():
    """Every free end with a nearby partner returns a row, so reasons can be counted."""
    sites = facing_pair()
    sites[1] = site(2, (5, 5.3, 5), (0, -1, 0), radius=0.6, semantic="dendrite_like")
    candidates = propose_links(sites, config=LinkConfig())
    assert len(candidates) == 2
    assert {c.reject_reason for c in candidates} == {
        "semantic:axon_like->dendrite_like",
        "semantic:dendrite_like->axon_like",
    }


def accepted_pair(left, right, gap):
    return LinkCandidate(
        left_label=left,
        right_label=right,
        left_vertex_index=0,
        right_vertex_index=0,
        left_semantic_type="axon_like",
        right_semantic_type="axon_like",
        gap_um=gap,
        caliber_ratio=1.0,
        left_radius_um=0.1,
        right_radius_um=0.1,
        tangent_deg=0.0,
        mutual=True,
        affinity=None,
        accepted=True,
        reject_reason="",
    )


def test_groups_refuse_to_grow_past_the_cap():
    """A chain of individually plausible joins is how one implausible object forms."""
    chain = [accepted_pair(i, i + 1, 0.1 * i) for i in range(1, 6)]
    groups, dropped = connected_groups(chain, max_group_size=3)
    assert all(len(group) <= 3 for group in groups)
    # 1-2-3 fills the cap, so 3-4 is dropped; 4-5 then forms its own group.
    assert [(c.left_label, c.right_label) for c in dropped] == [(3, 4)]
    assert sorted(groups) == [[1, 2, 3], [4, 5, 6]]


def test_groups_take_the_shortest_edges_first():
    groups, dropped = connected_groups(
        [accepted_pair(1, 2, 0.9), accepted_pair(2, 3, 0.1)], max_group_size=2
    )
    assert groups == [[2, 3]]
    assert [(c.left_label, c.right_label) for c in dropped] == [(1, 2)]


def test_groups_from_disjoint_pairs():
    pairs = select_mutual_best(propose_links(facing_pair(), config=LinkConfig()))
    groups, dropped = connected_groups(pairs, max_group_size=8)
    assert groups == [[1, 2]]
    assert dropped == []


def test_collect_link_sites_skips_unmeasured_segments():
    config = ContinuityConfig(volume_extent_um=EXTENT)
    points = np.linspace([5, 3.0, 5], [5, 6.0, 5], 11)
    edges = np.stack((np.arange(10), np.arange(1, 11)), axis=1)
    radii = np.full(11, 0.1)
    measured = analyze_continuity(1, points, edges, radii, 5000, config)
    empty = analyze_continuity(2, points[:1], np.zeros((0, 2), dtype=np.int64), radii[:1], 5000, config)
    sites = collect_link_sites([measured, empty])
    assert {s.label for s in sites} == {1}
    assert len(sites) == 2  # both free ends of the measured segment
    assert all(s.semantic_type == "axon_like" for s in sites)


def test_config_validation():
    with pytest.raises(ValueError):
        LinkConfig(min_caliber_ratio=1.5)
    with pytest.raises(ValueError):
        LinkConfig(max_gap_um=0)
    with pytest.raises(ValueError):
        LinkConfig(max_group_size=1)


def test_side_attachment_uses_one_index_over_all_labels():
    """Many target segments must not cost one spatial query each.

    The earlier implementation built a tree per label and queried every one of
    them for every free end, which is quadratic in the segment count and does not
    finish on a real volume. This checks the result is still correct with many
    labels present, including labels that have no free end of their own.
    """
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {
        label: np.array([[5 + 0.01 * label, 9.0, 5.0]], dtype=float)
        for label in range(2, 400)
    }
    points[7] = np.array([[5.0, 5.3, 5.0]], dtype=float)  # the only reachable one
    radii = {label: np.full(len(value), 0.1) for label, value in points.items()}
    candidates = propose_links(
        sites,
        config=LinkConfig(),
        skeleton_points=points,
        skeleton_radii=radii,
        label_semantic_types={label: "axon_like" for label in points},
    )
    assert len(candidates) == 1
    assert candidates[0].right_label == 7
    assert candidates[0].accepted is True
    assert candidates[0].gap_um == pytest.approx(0.3)


def test_side_attachment_target_type_comes_from_the_label_map():
    """A target with no free end is absent from `sites`; its type must still gate."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    shaft = {2: np.array([[5.0, 5.3, 5.0]], dtype=float)}
    radii = {2: np.full(1, 0.1)}
    rejected = propose_links(
        sites,
        config=LinkConfig(),
        skeleton_points=shaft,
        skeleton_radii=radii,
        label_semantic_types={2: "dendrite_like"},
    )[0]
    assert rejected.accepted is False
    assert rejected.reject_reason == "semantic:axon_like->dendrite_like"


def test_side_attachment_reports_the_local_vertex_index():
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    shaft = {2: np.array([[5, 9.0, 5.0], [5, 5.3, 5.0]], dtype=float)}
    candidate = propose_links(
        sites,
        config=LinkConfig(),
        skeleton_points=shaft,
        skeleton_radii={2: np.full(2, 0.1)},
        label_semantic_types={2: "axon_like"},
    )[0]
    assert candidate.right_vertex_index == 1  # index within label 2's own array


def test_affinity_probe_does_not_run_for_pairs_rejected_on_geometry():
    """The probe reads a volume; it must not be called for a doomed pair."""
    calls = []

    def probe(left, right):
        calls.append((tuple(left), tuple(right)))
        return {"mean": 0.9}

    sites = [site(1, (5, 5, 5), (0, 1, 0)), site(2, (5, 5, 5.3), (0, -1, 0))]  # 90 deg
    candidate = propose_links(
        sites, config=LinkConfig(min_affinity=0.5), affinity_probe=probe
    )[0]
    assert candidate.reject_reason == "tangent"
    assert candidate.affinity is None
    assert calls == []


def test_affinity_probe_runs_once_a_pair_survives_geometry():
    calls = []

    def probe(left, right):
        calls.append((tuple(left), tuple(right)))
        return {"mean": 0.9}

    propose_links(facing_pair(), config=LinkConfig(min_affinity=0.5), affinity_probe=probe)
    assert len(calls) == 2  # one per free end


def test_mutual_best_keeps_the_shortest_of_several_ends_to_one_partner():
    """One segment can reach the same partner from two ends; the near one wins."""
    near = accepted_pair(1, 2, 0.1)
    far = accepted_pair(1, 2, 0.8)
    reverse = accepted_pair(2, 1, 0.1)
    for ordering in ([far, near, reverse], [near, far, reverse]):
        (kept,) = select_mutual_best(ordering)
        assert kept.gap_um == pytest.approx(0.1)


def test_semantic_gate_can_be_switched_off():
    """On a volume where caliber does not separate the classes, the veto is noise."""
    sites = facing_pair()
    sites[1] = site(2, (5, 5.3, 5), (0, -1, 0), radius=0.09, semantic="dendrite_like")
    gated = propose_links(sites, config=LinkConfig())[0]
    assert gated.accepted is False
    assert gated.reject_reason.startswith("semantic:")
    ungated = propose_links(sites, config=LinkConfig(apply_semantic_gate=False))[0]
    assert ungated.accepted is True
    # The caliber ratio still applies with the gate off.
    sites[1] = site(2, (5, 5.3, 5), (0, -1, 0), radius=0.01, semantic="dendrite_like")
    assert propose_links(sites, config=LinkConfig(apply_semantic_gate=False))[0].reject_reason == "caliber"


def test_require_mutual_false_keeps_side_attachments():
    """The one-tip route has no reverse candidate; mutual selection deletes it."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    shaft = {2: np.array([[5.0, 5.3, 5.0]], dtype=float)}
    candidates = propose_links(
        sites, config=LinkConfig(), skeleton_points=shaft, skeleton_radii={2: np.full(1, 0.1)},
        label_semantic_types={2: "axon_like"},
    )
    assert candidates[0].accepted is True and candidates[0].mutual is False
    assert select_links(candidates, require_mutual=True) == []
    kept = select_links(candidates, require_mutual=False)
    assert [(c.left_label, c.right_label) for c in kept] == [(1, 2)]


def test_select_links_deduplicates_an_unordered_pair():
    kept = select_links(
        [accepted_pair(1, 2, 0.4), accepted_pair(2, 1, 0.2)], require_mutual=False
    )
    assert len(kept) == 1 and kept[0].gap_um == pytest.approx(0.2)


def test_caliber_uses_the_shaft_radius_not_the_tip_artifact():
    """A terminal vertex sits at the surface and reads ~one voxel regardless.

    Measured on LICONN mip1 the tip radius was 0.0181 um at p10, p50 and p90 —
    a constant — while the true local shaft radius there had p50 0.0901 um.
    Comparing the artifact against a partner's real radius failed 92.6% of ends
    on the gate before biology was consulted.
    """
    tip_artifact = 0.018
    left = LinkSite(1, 0, (5, 5, 5), (0, 1, 0), tip_artifact, "axon_like",
                    shaft_radius_um=0.090)
    right = LinkSite(2, 0, (5, 5.3, 5), (0, -1, 0), tip_artifact, "axon_like",
                     shaft_radius_um=0.085)
    assert left.caliber_um == pytest.approx(0.090)
    candidate = propose_links([left, right], config=LinkConfig())[0]
    assert candidate.accepted is True
    assert candidate.caliber_ratio == pytest.approx(0.085 / 0.090)
    assert candidate.left_radius_um == pytest.approx(0.090)


def test_without_a_shaft_radius_it_falls_back_to_the_tip():
    site_only_tip = LinkSite(1, 0, (5, 5, 5), (0, 1, 0), 0.1, "axon_like")
    assert site_only_tip.caliber_um == pytest.approx(0.1)


def test_a_genuine_caliber_mismatch_is_still_rejected():
    """The fix must not disable the gate, only stop it firing on an artifact."""
    left = LinkSite(1, 0, (5, 5, 5), (0, 1, 0), 0.018, "axon_like", shaft_radius_um=0.30)
    right = LinkSite(2, 0, (5, 5.3, 5), (0, -1, 0), 0.018, "axon_like", shaft_radius_um=0.05)
    candidate = propose_links([left, right], config=LinkConfig())[0]
    assert candidate.accepted is False
    assert candidate.reject_reason == "caliber"
