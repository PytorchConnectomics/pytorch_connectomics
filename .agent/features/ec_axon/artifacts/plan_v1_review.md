# Plan v1 Review

## Summary
Reviewer (codex, read-only, round 2) confirms plan_v1 resolved prior findings 2, 3 (location/laziness), 5
(for v1-v4), 7, 8 and 14, but judges it still non-executable: several decisions were deferred rather than
made ("Q1" on the v0 seed, "to be confirmed" on the merge_threshold schema path), one assertion was wrong,
and v4/completeness parity plus bounded perf acceptance were still missing.

## Findings
Major (6):
1. Findings 1/12/13 unresolved — `axon_v0` vs `axon_tracklet_base` left open, no destination file for the v0
   port, superseded v1-v3 registrations kept, dev-path dependency preserved.
2. Finding 4 partial — stats invalidation depends on an undefined `n > 0`; return identity, dtype, adapter
   handling of change counts and both-`inplace` tests unspecified; `max(zr) <= seg.max()` cannot establish
   freshness (compares bbox data with label IDs).
3. Finding 6 deferred — `merge_threshold` schema path still "to be confirmed"; fallback commands/config/
   artifact paths/base+oracle invocation not pinned.
4. Findings 9/11 incomplete — no parity verification for v4 or completeness; the v4 node uses `enabled:
   false`, which no graph/adapter/core contract defines, so `output: v4` is not executable.
5. Finding 10 partial — `np.shares_memory` proves aliasing, not absence of a transient full-volume
   allocation; the benchmark records time/memory with no acceptance threshold.
6. Registration conflicts with the task contract — the plan needs `register_graph_op` for multi-input yet the
   task mandates `register_decoder(...)`; the plan must reconcile rather than force a violation.

## Questions
- None raised by the reviewer beyond the findings; all six require a decision rather than clarification.

## Verdict
VERDICT: NEEDS_CHANGES
