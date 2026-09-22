# Plan v4 Review

## Summary
Reviewer (codex, read-only) accepts the consolidation direction but returns READY: no on four blocking
grounds: the axon v2 composition is not equivalent to the validated algorithm (full `branch_merge` also runs
stage-1 and stage-3, which bypass/extend the margin-qualified mutual stage); the backward-compatible defaults
are self-contradictory (`margin=0.15` in one section vs "off = 0.0" in verification); graph resolvability is
established only for the two `register_graph_op` names; and the verification section references absent
plan_v3 sections while omitting pre-refactor oracles and coverage of existing callers.

## Findings
Major (4):
1. Axon v2 composition not equivalent — define an explicit axon stage sequence in the canonical module while
   preserving the legacy three-stage default; an unspecified "dedicated code path" is insufficient.
2. Contradictory defaults — the shared default must remain `0.0`, with only the axon adapter passing `0.15`;
   the regression must exercise the OLD signature with all new arguments omitted.
3. Graph boundary incomplete — establish (and test) that `register_decoder` makes `axon_v0`, `axon_split`,
   `axon_complete` graph-resolvable, or give every stage an explicit adapter.
4. Verification not executable — depends on absent plan_v3 sections; add omitted-default golden regressions,
   existing decoder/unit coverage after the `seg_stats` migration, full-DAG execution, and concrete parity/
   performance inputs and pass criteria.

## Questions
- None beyond the findings; all four require decisions rather than clarification.

## Verdict
VERDICT: NEEDS_CHANGES
