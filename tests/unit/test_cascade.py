"""Contract tests for the confidence-ordered split-repair cascade."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.decoding.error_correction.cascade import (
    CascadeConfig,
    CascadePass,
    border_state,
    default_schedule,
    run_cascade,
)
from connectomics.decoding.error_correction.split_links import LinkConfig, LinkSite

AXON = 0.09


def site(label, position, tangent, vertex=0, semantic="axon_like"):
    return LinkSite(
        label=label,
        vertex_index=vertex,
        position_um_zyx=tuple(float(v) for v in position),
        outward_tangent_zyx=tuple(float(v) for v in tangent),
        radius_um=0.018,
        semantic_type=semantic,
        shaft_radius_um=AXON,
    )


def one_pass(**kwargs):
    gates = dict(
        max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
        require_mutual=False, apply_semantic_gate=False,
    )
    gates.update(kwargs.pop("gates", {}))
    return CascadeConfig(passes=(CascadePass("only", LinkConfig(**gates)),), **kwargs)


def run(sites, points, *, faces, length, config=None, **kwargs):
    return run_cascade(
        sites,
        config=config or one_pass(),
        faces=faces,
        length_um=length,
        skeleton_points=points,
        skeleton_radii={k: np.full(len(v), AXON) for k, v in points.items()},
        label_semantic_types={k: "axon_like" for k in points},
        **kwargs,
    )


def test_an_end_with_one_option_is_joined():
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    result = run(sites, points, faces={1: frozenset({"y0"}), 2: frozenset()},
                 length={1: 1.0, 2: 20.0})
    assert result.groups == ((1, 2),)
    assert len(result.joins) == 1
    assert result.passes[0].unique_source == 1
    assert result.passes[0].uncontested == 1


def test_an_end_with_two_options_is_left_alone():
    """The core rule: a choice is a guess, so it waits rather than being scored."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {
        1: np.array([[5, 4.8, 5.0]]),
        2: np.array([[5, 5.30, 5.00]]),
        3: np.array([[5, 5.28, 5.03]]),
    }
    result = run(sites, points, faces={1: frozenset(), 2: frozenset(), 3: frozenset()},
                 length={1: 2.0, 2: 1.0, 3: 1.0})
    assert result.groups == ()
    assert result.passes[0].accepted_gates == 2  # both were admissible
    assert result.passes[0].unique_source == 0  # but the end had a choice
    assert result.joins == ()


def test_two_ends_competing_for_one_site_are_both_left_alone():
    """Each end is individually unique; together they contest the same place."""
    sites = [site(1, (5, 5.0, 5), (0, 1, 0)), site(2, (5, 5.6, 5), (0, -1, 0))]
    points = {
        1: np.array([[5, 4.9, 5.0]]),
        2: np.array([[5, 5.7, 5.0]]),
        3: np.array([[5, 5.30, 5.0]]),
    }
    result = run(sites, points, faces={k: frozenset() for k in (1, 2, 3)},
                 length={1: 1.0, 2: 1.0, 3: 1.0})
    assert result.passes[0].unique_source == 2
    assert result.passes[0].uncontested == 0
    assert result.groups == ()


def test_raising_the_claim_limit_admits_the_contested_pair():
    sites = [site(1, (5, 5.0, 5), (0, 1, 0)), site(2, (5, 5.6, 5), (0, -1, 0))]
    points = {
        1: np.array([[5, 4.9, 5.0]]),
        2: np.array([[5, 5.7, 5.0]]),
        3: np.array([[5, 5.30, 5.0]]),
    }
    config = CascadeConfig(
        passes=(
            CascadePass(
                "loose",
                LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
                           require_mutual=False, apply_semantic_gate=False),
                max_claims_per_site=2,
            ),
        )
    )
    result = run(sites, points, faces={k: frozenset() for k in (1, 2, 3)},
                 length={1: 1.0, 2: 1.0, 3: 30.0}, config=config)
    assert result.passes[0].uncontested == 2
    assert result.groups and set(result.groups[0]) == {1, 2, 3}


def test_site_radius_coarse_enough_to_notice_neighbours():
    """Two attachment points a vertex apart are the same place, not two places.

    These coordinates are the ones a grid gets wrong: 5.30/0.1 floors to 52 in
    IEEE doubles and 5.32/0.1 to 53, so a grid separates exactly the pair that
    must collide. Distance clustering has no cell edge to straddle.
    """
    sites = [site(1, (5, 5.0, 5), (0, 1, 0)), site(2, (5, 5.6, 5), (0, -1, 0))]
    # Distinct vertex indices, 0.02 um apart: one site at a 0.10 um grid.
    points = {1: np.array([[5, 4.9, 5.0]]), 2: np.array([[5, 5.7, 5.0]]),
              3: np.array([[5, 5.30, 5.0], [5, 5.32, 5.0]])}
    coarse = run(sites, points, faces={k: frozenset() for k in (1, 2, 3)},
                 length={1: 1.0, 2: 1.0, 3: 1.0})
    assert coarse.passes[0].uncontested == 0  # recognised as one contested site
    fine = CascadeConfig(
        passes=(
            CascadePass(
                "fine",
                LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
                           require_mutual=False, apply_semantic_gate=False),
                site_grid_um=0.001,  # smaller than the 0.02 um separation
            ),
        )
    )
    split = run(sites, points, faces={k: frozenset() for k in (1, 2, 3)},
                length={1: 1.0, 2: 1.0, 3: 1.0}, config=fine)
    assert split.passes[0].uncontested == 2  # a too-fine grid hides the contest


def test_border_objective_improves_and_is_reported():
    """A floating fragment joined to an anchored segment becomes anchored."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    result = run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
                 length={1: 1.0, 2: 20.0})
    assert result.border_before.floating == 1
    assert result.border_before.anchored == 1
    assert result.border_after.floating == 0
    assert result.border_after.anchored == 1  # now one component, still anchored
    assert result.border_after.length_um_anchored == pytest.approx(21.0)
    assert result.border_after.anchored_length_fraction == pytest.approx(1.0)


def test_through_requires_two_distinct_faces():
    """Two censored ends on the SAME face is not a crossing."""
    same = border_state([1], faces={1: frozenset({"z0"})}, free_ends={1: 0},
                        length_um={1: 1.0})
    assert same.anchored == 1 and same.through == 0
    across = border_state([1], faces={1: frozenset({"z0", "zmax"})}, free_ends={1: 0},
                          length_um={1: 1.0})
    assert across.through == 1


def test_joining_two_anchored_halves_makes_a_through_component():
    """Two substantial halves is a union by definition, so the test opts in."""
    sites = [site(1, (5, 5.0, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.9, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    config = CascadeConfig(
        passes=(
            CascadePass(
                "union",
                LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
                           require_mutual=False, apply_semantic_gate=False),
                allow_union=True,
            ),
        )
    )
    result = run(sites, points, faces={1: frozenset({"y0"}), 2: frozenset({"ymax"})},
                 length={1: 3.0, 2: 3.0}, config=config)
    assert result.border_before.through == 0
    assert result.border_after.through == 1
    assert result.border_after.through_length_fraction == pytest.approx(1.0)


def test_require_border_gain_refuses_a_join_that_reaches_nothing():
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    faces = {1: frozenset(), 2: frozenset()}
    config = CascadeConfig(
        passes=(
            CascadePass(
                "gain-only",
                LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
                           require_mutual=False, apply_semantic_gate=False),
                require_border_gain=True,
            ),
        )
    )
    assert run(sites, points, faces=faces, length={1: 1.0, 2: 20.0},
               config=config).groups == ()
    # The same join is taken once the partner reaches a face.
    assert run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
               length={1: 1.0, 2: 20.0}, config=config).groups == ((1, 2),)


def test_multiple_passes_take_the_strict_joins_first():
    """A later, looser pass must not pre-empt what a tighter one could decide."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.45, 5.0]])}
    config = CascadeConfig(
        passes=(
            CascadePass("tight", LinkConfig(max_gap_um=0.2, min_caliber_ratio=0.5,
                                            max_tangent_deg=60.0, require_mutual=False,
                                            apply_semantic_gate=False)),
            CascadePass("loose", LinkConfig(max_gap_um=0.8, min_caliber_ratio=0.5,
                                            max_tangent_deg=60.0, require_mutual=False,
                                            apply_semantic_gate=False)),
        )
    )
    result = run(sites, points, faces={1: frozenset(), 2: frozenset()},
                 length={1: 1.0, 2: 20.0}, config=config)
    assert result.passes[0].joins_taken == 0  # 0.45 um is outside the tight gap
    assert result.passes[1].joins_taken == 1
    assert result.groups == ((1, 2),)


def test_group_cap_is_enforced_across_passes():
    sites = [site(i, (5, 5 + 0.3 * i, 5), (0, 1, 0)) for i in range(1, 7)]
    points = {i: np.array([[5, 5 + 0.3 * i + 0.15, 5.0]]) for i in range(1, 7)}
    result = run(sites, points, faces={i: frozenset() for i in range(1, 7)},
                 length={i: 1.0 for i in range(1, 7)},
                 config=one_pass(max_group_size=3))
    assert all(len(group) <= 3 for group in result.groups)


def test_a_segment_never_joins_its_own_component_twice():
    sites = [site(1, (5, 5.0, 5), (0, 1, 0), vertex=0),
             site(1, (5, 5.6, 5), (0, 1, 0), vertex=1)]
    points = {1: np.array([[5, 4.9, 5.0], [5, 5.7, 5.0]]),
              2: np.array([[5, 5.30, 5.0]])}
    result = run(sites, points, faces={1: frozenset(), 2: frozenset()},
                 length={1: 1.0, 2: 1.0})
    assert len(result.joins) <= 1
    assert result.groups in ((), ((1, 2),))


def test_only_the_configured_semantic_types_are_repaired():
    sites = [site(1, (5, 5, 5), (0, 1, 0), semantic="dendrite_like")]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    config = CascadeConfig(
        passes=(CascadePass("only", LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5,
                                               max_tangent_deg=60.0, require_mutual=False,
                                               apply_semantic_gate=False)),),
        repair_semantic_types=("axon_like",),
    )
    result = run(sites, points, faces={1: frozenset(), 2: frozenset()},
                 length={1: 1.0, 2: 1.0}, config=config)
    assert result.groups == ()
    assert result.passes[0].candidates_scored == 0


def test_default_schedule_relaxes_gap_last():
    schedule = default_schedule(min_affinity=0.6)
    gaps = [p.link.max_gap_um for p in schedule.passes]
    angles = [p.link.max_tangent_deg for p in schedule.passes]
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]
    assert angles[0] < angles[-1]
    # The affinity floor is never loosened.
    assert {p.link.min_affinity for p in schedule.passes} == {0.6}


def test_config_validation():
    with pytest.raises(ValueError):
        CascadeConfig(passes=())
    with pytest.raises(ValueError):
        one_pass(max_group_size=1)
    with pytest.raises(ValueError):
        CascadePass("x", LinkConfig(), site_grid_um=0)
    with pytest.raises(TypeError):
        CascadePass("x", "not a config")


def test_progress_callback_reports_each_pass():
    lines: list[str] = []
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
        length={1: 1.0, 2: 20.0}, progress=lines.append)
    assert len(lines) == 1
    assert "assigned" in lines[0] and "anchored" in lines[0]


def test_assignment_is_allowed_by_default():
    """A short fragment taking a much longer host's label: error costs one fragment."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    result = run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
                 length={1: 1.0, 2: 20.0})
    assert result.groups == ((1, 2),)
    assert result.passes[0].assignments_taken == 1
    assert result.passes[0].unions_taken == 0


def test_union_is_refused_by_default():
    """Two substantial components welding: error destroys both, so it needs opt-in.

    On zebrafinch every union variant measured net-negative (affinity-gated
    anchor-anchor joins -0.015 to -0.263; a 91.2%-precision continuity-gated
    version -0.085 with 13 neurons zeroed), while assignment was the operation
    that paid. The default must therefore be assignment only.
    """
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    faces = {1: frozenset({"y0"}), 2: frozenset({"ymax"})}
    length = {1: 20.0, 2: 22.0}
    refused = run(sites, points, faces=faces, length=length)
    assert refused.groups == ()
    assert refused.passes[0].refused_as_union == 1
    assert refused.passes[0].assignments_taken == 0


def test_union_is_taken_when_explicitly_allowed():
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    config = CascadeConfig(
        passes=(
            CascadePass(
                "risky",
                LinkConfig(max_gap_um=0.5, min_caliber_ratio=0.5, max_tangent_deg=60.0,
                           require_mutual=False, apply_semantic_gate=False),
                allow_union=True,
            ),
        )
    )
    result = run(sites, points, faces={1: frozenset({"y0"}), 2: frozenset({"ymax"})},
                 length={1: 20.0, 2: 22.0}, config=config)
    assert result.groups == ((1, 2),)
    assert result.passes[0].unions_taken == 1
    assert result.passes[0].assignments_taken == 0


def test_a_long_fragment_is_a_union_even_against_a_longer_host():
    """The length cap, not just the ratio, decides: a long piece is never a fragment."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    # ratio is 10x, but the source is 10 um: far too much to risk misplacing.
    result = run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
                 length={1: 10.0, 2: 100.0})
    assert result.passes[0].refused_as_union == 1
    assert result.groups == ()


def test_a_similar_sized_pair_is_a_union_even_when_short():
    """The host ratio, not just the cap, decides: peers are not host and fragment."""
    sites = [site(1, (5, 5, 5), (0, 1, 0))]
    points = {1: np.array([[5, 4.8, 5.0]]), 2: np.array([[5, 5.3, 5.0]])}
    result = run(sites, points, faces={1: frozenset(), 2: frozenset({"ymax"})},
                 length={1: 1.0, 2: 1.2})
    assert result.passes[0].refused_as_union == 1


def test_component_length_accumulates_so_a_chain_stops_being_a_fragment():
    """After absorbing enough, a component is no longer safe to reassign."""
    sites = [site(i, (5, 5 + 0.3 * i, 5), (0, 1, 0)) for i in range(1, 6)]
    points = {i: np.array([[5, 5 + 0.3 * i + 0.15, 5.0]]) for i in range(1, 6)}
    result = run(sites, points, faces={i: frozenset() for i in range(1, 6)},
                 length={i: 2.0 for i in range(1, 6)},
                 config=one_pass(max_group_size=8))
    # Each is 2 um; two of them exceed the 3 um assignment cap, so the chain
    # cannot keep growing by assignment alone.
    assert all(
        sum(2.0 for _ in group) <= 4.0 + 1e-9 for group in result.groups
    ), result.groups


def test_default_schedule_is_assignment_only_unless_asked():
    assert all(not p.allow_union for p in default_schedule(min_affinity=0.6).passes)
    assert all(
        p.allow_union
        for p in default_schedule(min_affinity=0.6, allow_union=True).passes
    )
