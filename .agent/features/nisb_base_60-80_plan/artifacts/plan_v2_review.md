# Plan v2 Review

## Summary

Codex (coder) reviewed `plan_v2.md`, the final allowed plan version. Verdict:
**READY: no / NEEDS_CHANGES**. It confirmed genuine resolutions (affinity+scorer binding,
the `get_nodes_position` coordinate API, the deterministic top-8 cap, the crop-scale
surrogate framing) but still raised major findings — a mix of real correctness/scope bugs
and a design-fidelity disagreement about how faithful G0 must be to the reply's Phase-0.
The plan-round budget (p2) is now exhausted. Full reviewer output:
`state/plan_v2_review.review.raw.md`. (The `plan_v1_review` "Artifact 2" was empty in this
review prompt due to a coordinator ordering error, now repaired; Codex's findings are
independent of it.)

## Findings

Faithful summary (no material finding softened):

Real correctness / scope bugs (coordinator concurs — should be fixed before/while coding):
- [major] **Sanity A would violate scope:** scoring whole-vol cc066 must load the cached
  node-LUT (`.../seg_fusion/oracle_lut/ch0-1-2_cc0.66_node_luts.npz`), NOT decode the
  12.15 B-voxel volume; bind the exact cached invocation or make it a nonblocking provenance check.
- [major] **B3 offset count wrong for a cumulative sweep:** the 49-offset `|o|∞=2` shell
  excludes B0's 13 nearest neighbors, so cumulative B3 = 62, not 49 (or state it replaces).
- [major] **Realizer conflates two questions and is untrustworthy:** "capped-candidate-recovered"
  merging only GT-confirmed pairs makes 0 false merges true by construction (no real matching
  tested); and with every skeleton voxel a fixed seed, realized NERL measures identity merges,
  not geodesic realization. Separate candidate-recovered *gain* from realizer *fidelity*; run an
  actual (non-transitive, capped) matching and audit its precision/merges vs GT; use the measured
  Sanity-B baseline, not rounded 0.836; define failure when full-oracle gain is nonpositive.
- [major] **450 M-voxel two-label geodesic needs a bounded implementation:** prefer an atomic
  `BASE_SEG`-adjacency-graph realizer (assign fragments to identities, O(fragments)) over a
  voxelwise search; specify reverse-step affinity indexing, runner-up margin, abstention label.
- [major] **G1 oracle event multiplicity:** one fragment pair can have several crossing skeleton
  edges → multiple break locations; define event per break cluster (or a break set) with exact
  hit aggregation; bind `ΔNERL = 2·L_a·L_b/D` and validate it against the canonical scorer.
- [major] **Geometric proposal set not reproducible:** the radius-aware distance limit, radius
  estimator, and Hermite construction need explicit equations/constants — they control every
  recall/candidate-count number.
- [major] **C1/C2 metric definitions:** restrict C1 deletion to degree-2 nodes; define C2's
  occupied section + all anchor pairs across every cut strand; state fragments/GT denominator
  (626 scored skeletons vs 783 seg IDs) and the exact retained-length numerator.

Design-fidelity disagreement (coordinator's stated deliberate deviation, see below):
- [major] **G0 uses 1-voxel skeleton nodes, not the reply's medial-band + SAME/MUTEX/DEFER +
  rasterized skip paths.** Codex holds the plan to §12's Phase-0 spec. Coordinator position: G0
  is a GT-*oracle* decode-feasibility test; the proven ancestor `centerline_graph_decode_gt.py`
  uses 1-voxel skeleton nodes + same-instance offsets + watershed and is the correct cheap
  de-risk. Medial band / signed-edge logits / path rasterization are learned-pipeline (Phase-2)
  concepts that carry no extra information when identity is already known. This is a scope choice,
  not a defect — but it is genuinely the maintainer's call.
- [major] **Channel formulas still in DESIGN.md, not the plan.** Codex wants medialness params,
  tangent estimator, DEFER rules, veto formulas in the plan. Coordinator position: DESIGN.md is
  the correct home for the model spec; the plan pins the index map (sufficient for planning). Also
  the maintainer's call.

## Questions

- Plan rounds are exhausted with the two design-fidelity items unresolved by disagreement. The
  human decision (per CCC normal mode) is whether to (a) proceed to code, treating the real
  correctness/scope findings as code-stage acceptance criteria and G0 as the proven-ancestor
  oracle; or (b) extend plan rounds to fully harden the spec first.

## Verdict

VERDICT: NEEDS_CHANGES
