# Plan v0 Review

## Summary

The plan is directionally right on the rank-0 guard and dead streaming fallback removal, but it is not ready. The proposed C2 fix only aligns the head/tail scan loop; it does not make streaming equivalent to `scan_prediction` because the cutoff/interior mean still uses a different sampled population. The proposed C7 tests also would not reliably catch the current C2 bug.

## Findings

- [major] C2 does not fully align streaming with post-save. `scan_prediction` computes `interior_mean` from `_per_z_scan(..., z_stride)`, so `k_edge` is applied to sampled Z rows. `AffinityQCAccumulator.finalize` still computes `interior_mean` from every Z slice, so with default `z_stride=10` the cutoff can differ even if the head/tail window loop matches `_refine_z_cuts`. The fix should either use `means[::params.z_stride]` for the sampled/interior calculation or intentionally change both paths.

- [major] The proposed interior-outlier test does not prove C2. The current full-curve finalize finds the first and last good slice; isolated bad slices at `30/31` do not narrow the range if there are good slices before and after them. A failing pre-fix case needs all slices in the first or last `refine_window` bad, with good slices just outside the window.

- [major] C5 removes `finalize(img=...)`, which is a signature change on the exported `AffinityQCAccumulator` class. That conflicts with the task’s public API preservation requirement and the plan’s own statement that exported signatures do not change. Keep the keyword accepted and unused.

- [minor] C1 is correct for the shown in-tree distributed call site, and the accumulator should not silently no-op based on rank. Rank policy belongs at the caller. However, the “all-zero accumulator” error only catches zero-update misuse, not every distributed misuse; partial updates are caught by the existing missing-slice error, and duplicate full writes from a future non-rank caller would not be detected.

- [minor] C4 may reject a config that would otherwise short-circuit through an already populated `decoding.affinity_mask_path`. If that short-circuit is intended public behavior, preflight should exempt existing-mask cases or the plan should explicitly retire that behavior.

- [minor] The plan preserves the v3 `inference` boundary as written. No proposed change requires `connectomics.inference` to statically import `connectomics.decoding`.

## Questions

- Should streaming use post-save’s sampled `z_stride` semantics exactly, including report stats, or only match `low_z/high_z`?
- Should preflight allow `mode='streaming'` without `image_path` when `affinity_mask_path` is already set?
- Is `AffinityQCAccumulator.finalize` considered public through the exported class? If yes, `img` must remain accepted.

## Verdict

READY: no