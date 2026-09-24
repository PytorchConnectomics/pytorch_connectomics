# Plan v1 Review

## Summary
Codex reviewed plan_v1 (`codex exec --sandbox read-only`; raw at
`state/plan_v1_review.review.raw.md`) and answered `READY: no`: 4 major, 2 minor. Reviewer
confirmed the exactness arguments for C2 (BFS cursor invariant), C3 (LUT), C4 (`in_rg` removal)
and C6 (MAX/MEAN accumulators, conditional on FP settings), and that v0 findings 1, 5, 8, 9 are
addressed. Remaining gaps are in the uint32 bound, memory attribution, the `dend_*` contract and
failure-path tests.

## Findings
1. [major] C7 uses `lut.size() - 1` as the max output id, but the LUT spans original watershed ids while outputs are compacted (`counts.resize(next_id)`). Use the final compact count (`counts.size() - 1`) or max LUT value; add a heavy-merge fixture.
2. [major] Memory attribution still insufficient: logging after `stable_sort` misses its temp buffer; `VmHWM` is cumulative; streamed-write page cache is not in the writer's `RssFile`; `sacct MaxRSS` is not an aggregate cgroup measure. Need phase-bracketed sampling and a defined aggregate job-memory measurement, or report it as unverified.
3. [major] `dend_*` padding: two matching stock runs don't prove padding determinism. Make record count/layout/field equality the unconditional contract, report padding diffs separately, confirm actual ABI field offsets.
4. [major] Missing failure-path tests: duplicate dtype token, `offset > UINT32_MAX`, checked-product overflow, exact `high_bit` boundary for C1, batch failure after an earlier successful threshold (no artifacts for the failed one; Python callback never invoked after failure).
5. [minor] Target bound omits the queue term: C2 is still O(P) when one search spans the volume; express via max search size/capacity; `clear()` the vector between searches (retain capacity).
6. [minor] Plan mandates creating branch `feature/ws-memory` in lib/abiss though the user did not request it.

## Questions
- uint32 validation: final compact count or max LUT value?
- Is `dend_*` field equality the unconditional contract?
- What aggregate memory measurement replaces cgroup `memory.peak` when unavailable?
- Do the stock compiler flags prohibit FP reassociation?

## Verdict
VERDICT: NEEDS_CHANGES
