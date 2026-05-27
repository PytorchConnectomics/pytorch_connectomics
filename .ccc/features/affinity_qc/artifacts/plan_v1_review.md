# Plan v1 Review

## Summary

Attested summary of `state/plan_v1_review.review.raw.md` (codex
reviewer, read-only). The reviewer maps `READY: no` to
`VERDICT: NEEDS_CHANGES`. C5 withdrawal is accepted. C2's algorithmic
direction is correct, but two new majors remain:

- C7's `z_stride=5` variant does not deterministically distinguish the
  pre-fix from post-fix C2 behavior — both produce the same
  `low_z=30, high_z=60` because the entire head window is bad under
  either cutoff. A test that catches the subsampling bug needs an
  ambiguity slice straddling the two cutoffs, which is fragile to
  engineer. A simpler test — direct equality of `interior_mean`
  between `scan_prediction` and `AffinityQCAccumulator.finalize` — is
  the deterministic guard.
- C4 (preflight exemption when `affinity_mask_path` is pre-set)
  preserves the decode-stage short-circuit but leaves a runtime
  mismatch: `begin_streaming_qc` will still build an accumulator
  when chunked inference runs, and `finish_streaming_qc` will raise
  on the missing `image_path` (per C3). Fix: short-circuit at
  `begin_streaming_qc` too.

One minor about report-array semantics after subsampling.

## Findings

- **[major] C7 needs an `interior_mean` equality test, not a windowed-scan
  divergence synthetic.** The proposed `z_stride=5` head-fill test does
  not catch a reverted C2 subsampling fix because both cutoff
  formulations classify z=0..29 as bad and z=30..59 as good. To make
  the subsampling fix deterministically protected, assert that
  `scan_prediction(pred).interior_mean` equals
  `accumulator.finalize().interior_mean` on a non-trivial random-ish
  volume. That equality is the direct property C2 enforces.

- **[major] C4 leaves `begin_streaming_qc` / `finish_streaming_qc`
  mismatched.** Preflight now accepts `mode='streaming'` without
  `image_path` when `decoding.affinity_mask_path` is already wired,
  but `begin_streaming_qc` still builds an accumulator under those
  conditions, chunked inference still folds slabs, then
  `finish_streaming_qc` raises (C3 requires `image_path`). Fix:
  `begin_streaming_qc` must return `None` when
  `decoding.affinity_mask_path` is non-empty. That matches the
  semantic "pre-wired mask → skip streaming setup → decode-stage
  short-circuit picks it up". The runtime path stays consistent with
  preflight.

- **[minor] C2 should pin what `AffinityQCReport.z_idx/means/stds`
  contain after subsampling is introduced.** The internal head/tail
  scan operates on *raw* per-Z `means`; the cutoff is computed from
  *sampled* means. The plan should be explicit about whether the
  returned report exposes raw or sampled arrays. Recommendation:
  expose raw — that information is free in streaming and informational
  only (does not affect `low_z`/`high_z`). The intentional difference
  with post-save's sampled `z_idx/means/stds` is already noted in
  plan_v1's risk section; just spell it out where the report is
  constructed.

## Questions

1. Should a pre-wired `decoding.affinity_mask_path` disable streaming
   QC setup entirely (per the proposed fix), even when chunked
   inference runs? **Recommended yes.**
2. Should streaming reports expose raw per-Z curves
   (`z_idx`/`means`/`stds`) for diagnostic value, or match post-save's
   sampled rows for report consistency? **Recommended raw; document
   the intentional difference.**

## Verdict

VERDICT: NEEDS_CHANGES
