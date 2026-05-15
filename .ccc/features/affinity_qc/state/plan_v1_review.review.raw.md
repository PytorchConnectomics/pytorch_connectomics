# Plan v1 Review

## Summary

Plan v1 fixes the main algorithmic direction for C2: computing the streaming cutoff from `means[::z_stride]` before the windowed head/tail scan should make `low_z`/`high_z` equivalent to `scan_prediction` for the same raw volume, assuming all raw Z slices are accumulated.

C5 is correctly withdrawn. Keeping `finalize(img=...)` is the right call under the stated public API preservation requirement.

C4 is directionally right for preserving the decode-stage `run_affinity_qc` short-circuit, but the plan does not address the streaming accumulator path when `affinity_mask_path` is already set. C7 still has a blocking coverage problem: the proposed `z_stride > 1` variant is not deterministic enough to catch the subsampling bug if reverted.

## Findings

- **[major] C7 still does not reliably catch the C2 subsampling bug.**  
  The head-window test with `z_stride=1` catches the full-curve scan bug. The proposed `z_stride=5` variant uses the same `z=0..32` bad / `33..59` good volume, but reverting only the subsampling fix can still produce the same `low_z=30, high_z=60`, because the whole head window remains bad under both cutoffs. The plan even says the reverted behavior “MAY” differ, which is not an acceptable regression test. Add a deterministic stride case where sampled and full interior means straddle a head/tail candidate slice so reverting `means[::z_stride]` fails.

- **[major] C4’s exemption can pass preflight but still fail in the streaming finish path.**  
  Allowing `mode='streaming'` with no `image_path` when `decoding.affinity_mask_path` is set preserves `run_affinity_qc`’s decode-stage short-circuit. However, `begin_streaming_qc` as shown currently ignores `affinity_mask_path`; if chunked inference still runs, it will create an accumulator and `finish_streaming_qc` will later raise because C3 requires `image_path`. Plan v1 should either make `begin_streaming_qc` return `None` when an existing mask is wired, or narrow the preflight exemption to runs that will not execute streaming finalize.

- **[minor] C2 should specify what `AffinityQCReport.z_idx/means/stds` contain after subsampling.**  
  The low/high logic is sound if raw `means` are retained for the window scan and sampled `means[::z_stride]` are used only for `interior_mean`. The plan introduces `sampled_z_idx` but does not say whether the returned report keeps raw per-Z rows or sampled rows. That ambiguity can produce inconsistent report arrays if implementation mixes sampled `z_idx` with raw `means`.

## Questions

1. Should a pre-wired `decoding.affinity_mask_path` disable streaming QC setup entirely, even when chunked inference runs?
2. Should streaming reports intentionally expose full raw per-Z curves, or match post-save sampled `z_idx/means/stds` for report consistency?

## Verdict

READY: no

Reasoning: C2’s implementation plan is mostly correct, but C7 does not actually protect the `z_stride` subsampling fix, and C4 introduces a preflight/runtime mismatch unless the accumulator path also honors the existing-mask short-circuit.