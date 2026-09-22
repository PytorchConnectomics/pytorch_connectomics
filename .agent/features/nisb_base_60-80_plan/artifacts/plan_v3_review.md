# Plan v3 Review

## Summary

Codex (coder) reviewed `plan_v3.md`, the final plan version (p3). It **accepts the ratified
1-voxel G0 oracle** and confirms the prior correctness/scope items resolved (cached-LUT
Sanity A, cumulative B3=62, measured `BASE_NERL`, break-cluster oracle events, validated
ΔNERL, C1/C2 + denominators). It raised **4 remaining [major]** items — all narrow
implementation-spec fixes in G1 and `DESIGN.md`, explicitly stated to "not challenge the
ratified G0 decision." Full reviewer output: `state/plan_v3_review.review.raw.md`.

## Findings

Faithful summary (no material finding softened):

- [major] **G1 §1 proposal geometry not reproducible:** pin the tangent-agreement equation,
  the cubic-Hermite handle magnitudes + curvature-evaluation rule, and the boundary-EDT/KNN
  physical units under anisotropic resolution.
- [major] **G1 §4 realizer gate does not test repaired realization:** the only gated run is
  per-fragment (control); run the realizer from the **capped-oracle repaired** seed partition,
  compare to its pre-grow result, and gate dense-GT adoption precision, nonzero coverage,
  abstention, and merge safety; define how label-0 regions enter the fragment graph.
- [major] **G1 §2 "non-transitive" vs degree-1 contradiction:** a 2-tip fragment could accept
  two links → transitive chain; require **degree ≤1 per `BASE_SEG` fragment** (or an exact
  non-cascading transaction) and pin the deterministic mutual-best ranking.
- [major] **`DESIGN.md` channel targets under-defined:** pin the `d_b` ± sign convention, the
  exact medialness `0.6·r_i` cutoff, tangent-masking + veto neighborhoods, and the sparse-skip
  SAME/MUTEX/DEFER target rule.

## Questions

None. Codex states each blocker has a local specification fix.

## Verdict

VERDICT: NEEDS_CHANGES
