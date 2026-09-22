# Review v1

## Summary

`code_v1` addresses every `review_v0` finding and is proven end-to-end on real data.
The LUT persistence + zero-GCS reuse feature is implemented cleanly, the metric is
unchanged, and the offline suite (48 tests) passes. Approving. Raw notes + evidence:
`state/review_v1.review.raw.md`.

Live two-leg proof (real skeletons + real CloudVolume):
- **Leg 1 (sample + save LUT):** ERL **96,337.92** — exactly the `code_v0` baseline, so
  the LUT/mip/cache refactor did not change the computation. Saved LUT = **316 KB**
  (gzip-compressed).
- **Leg 2 (reuse LUT, `open_seg_cloudvolume` wired to raise):** **0.1 s**, ERL
  96,337.92 — identical, and provably **zero GCS** (any access would have raised).
- Speedup 8.3 min → 0.1 s; shareable artifact ~316 KB vs ~3.6 GB one-time egress.

## Diff Baseline

run_start_ref: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949

Review surface: working tree of the nested `lib/em_erl` repo vs run_start_ref
(scripts/j0126_workflow.py, scripts/README.md, tests/test_j0126_workflow.py).

## Findings

- **[resolved, major] LUT persistence + zero-GCS reuse.** `--lut`: loads via `read_vol`
  with no CloudVolume import/open when the file exists; otherwise samples, validates, and
  saves via `write_h5`. Graph always rebuilt from `-g`; `len(lut)==num_nodes` guard with a
  clear error. Verified live (leg 2, 0.1 s, no GCS).
- **[resolved, minor] `--mip` (default 0).** Faithful default preserved; passed to the
  opener; help + README document the measured ~1.5% drift for coarser mips.
- **[resolved, minor] `--cache-dir`.** `cache=PATH` when set, else `cache=False`.
- **[confirmed] Sampling strategy unchanged** — each occupied chunk fetched once (GCS cost
  floor for this unsharded `compressed_segmentation` layer).
- No new findings.

## Tests to Add

None required. The added offline tests are sufficient and genuine: the reuse test asserts
`open_seg_cloudvolume` is never called (records calls + returns a `RaisingCloudVolume`),
the mismatch test asserts a clear error, and the mip/cache tests assert the values reach
the opener/constructor. Full suite: 48 passed.

## Questions

None.

## Verdict

VERDICT: APPROVE
