# Plan v0 Review

## Summary

Attested summary of `state/plan_v0_review.review.raw.md` (codex reviewer,
read-only sandbox). The reviewer maps `READY: no` to
`VERDICT: NEEDS_CHANGES`. The plan is directionally correct on C1
(rank-0 guard) and C3 (dead-branch removal), but three findings block:

- C2's proposed windowed scan does not fully align streaming with
  post-save because `scan_prediction` derives `interior_mean` over
  *stride-N sampled* z rows, while the accumulator stores every z.
  Aligning only the loop bounds leaves the cutoff potentially off.
- C7's interior-outlier test would not catch the current full-curve bug
  it is meant to demonstrate (full-curve finalize accepts isolated bad
  middle slices because there are good slices on both sides).
- C5 (drop `img=` parameter from `AffinityQCAccumulator.finalize`)
  changes a signature on an exported class; the task explicitly
  preserves the public API.

The reviewer also raised two clarifying questions and one preflight
edge-case worth resolving before plan_v1 is written.

## Findings

- **[major] C2 alignment is partial.** Post-save `interior_mean` comes
  from `_per_z_scan(..., z_stride)`; the accumulator's `means` array is
  *every* z. With default `z_stride=10`, even after fixing the loop
  bounds, the cutoff value can differ. Plan_v1 must either (a) subsample
  the accumulator's `means` by `z_stride` before computing
  `interior_mean`, or (b) intentionally change both code paths so the
  sampled-vs-full discrepancy is resolved once.

- **[major] C7 interior-outlier test does not exercise the C2 bug it
  claims to.** The proposed synthetic places bad slices at
  `(0, 1, 30, 31, 58, 59)`. The pre-fix full-curve finalize would still
  pick `low_z=2` and `high_z=58` because there are good slices both
  before and after `30/31`. To demonstrate the regression, the failing
  case needs bad slices to fill the head window with good slices
  immediately past it (or the symmetric tail variant) — then the
  windowed scan would return `low_z=refine_window` while the full-curve
  scan would return `low_z=0` (the first good slice past the head).

- **[major] C5 removes a parameter from a public class method.**
  `AffinityQCAccumulator.finalize` is exported via
  `connectomics.decoding.qc.__all__`. The task's "preserve public API"
  guard means the kwarg must remain accepted (even if unused), or the
  change must be promoted to an intentional API break with an
  appropriate test-suite update. Plan_v1 should keep the keyword.

- **[minor] C1 belt-and-suspenders catches only the zero-update misuse.**
  Partial updates already fail clearly via the existing missing-slice
  error path. Worth wording the new error specifically as "no slabs
  folded in — likely a non-rank-0 caller" so debuggers land on the
  caller contract.

- **[minor] C4 may collide with the streaming short-circuit path.**
  `run_affinity_qc` already returns an existing
  `decoding.affinity_mask_path` when `mode='streaming'` and the mask is
  present. Plan_v1 should explicitly state whether C4 exempts that
  case (preflight passes if `mask_path` already wired) or whether the
  short-circuit becomes effectively unused (delete the short-circuit).

- **[minor] V3 boundary preserved.** The reviewer confirmed no proposed
  change forces `connectomics.inference` to statically import
  `connectomics.decoding`. No action.

## Questions

1. Should streaming use the post-save sampled `z_stride` semantics
   exactly, including `g_mean`/`g_std` report stats, or only converge
   on `low_z`/`high_z`?
2. Should preflight allow `mode='streaming'` without `image_path` when
   `decoding.affinity_mask_path` is already set?
3. Is `AffinityQCAccumulator.finalize` part of the public API in
   practice? If yes, `img=` must stay; if no, the plan should declare
   the break.

## Verdict

VERDICT: NEEDS_CHANGES
