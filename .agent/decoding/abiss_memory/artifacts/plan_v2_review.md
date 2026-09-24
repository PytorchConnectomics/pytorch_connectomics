# Plan v2 Review

## Summary
Codex reviewed plan_v2 (`codex exec --sandbox read-only`; raw at
`state/plan_v2_review.review.raw.md`) and answered `READY: yes`: all six plan_v1_review findings
are resolved, no blocking issues, four minor refinements to be incorporated during coding. The
coordinator forwards these four items to the coder as binding implementation notes.

## Findings
1. [minor] Memory model for C5-C6 omits the x-y plane buffer / face buffers and PERCENTILE's retained boundary observations; state the bound per scoring mode.
2. [minor] Step 5: stock binaries have no C0 markers, so baseline phase attribution is unavailable; report baseline overall peaks only (per-phase attribution for the modified build), and flush C0 markers promptly (`std::endl`/explicit flush) so brackets are not distorted by buffering.
3. [minor] Step 2: `dend_*` score equality must be identical field bytes (padding excluded), not numeric equality (signed zero, NaN). Compute ABI offsets with byte-pointer differences (`reinterpret_cast<const char*>`), not typed pointer subtraction.
4. [minor] `test_ws_bfs` checks must stay active under `-DNDEBUG` (use explicit checks that return non-zero, not `assert`).

## Questions
- Baseline phase attribution: separately instrumented or explicitly unavailable? (Coordinator: explicitly unavailable; stock overall peaks only.)
- Bitwise float comparisons in `dend_*` and accumulator tests? (Coordinator: yes, bitwise.)

## Verdict
VERDICT: APPROVE_WITH_MINOR_COMMENTS
