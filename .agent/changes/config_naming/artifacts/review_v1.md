# Review v1

## Summary

`code_v1` fixes the direct `inference.save_compression` misses and removes the
public `tta_*` output-naming aliases from the runtime API. The remaining issues
are concentrated in the new per-volume layout: chunked test paths still bypass
the per-volume writer contract, and the stem resolver does not produce stable
dataset identities for the `data.zarr/img` paths used by the NISB tutorials.

## Diff Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9

## Findings

1. `connectomics/training/lightning/test_pipeline.py:718`-`728` and
   `connectomics/training/lightning/test_pipeline.py:810`-`816` still construct
   chunked output paths as flat `<save_path>/<volume>_<artifact>.h5` files.

   This bypasses the new `<save_path>/<volume_stem>/<artifact>.h5` contract that
   `write_outputs`, `write_decoded_outputs`, and the cache resolver now use. In
   the raw chunked branch, `raw_x...chunked-raw...h5` is written as
   `results/<volume>_raw_x...h5`, so `resolve_cached_prediction_files` later
   looks in `results/<volume>/raw_x...h5` and misses it. In the decoded chunked
   branch, `final_prediction_output_tag()` already returns a filename ending in
   `.h5`, but line 815 appends another `.h5`, producing a flat
   `<volume>_decoded_....h5.h5` path. This leaves chunked inference outside the
   central output-layout deliverable.

2. `connectomics/runtime/output_naming.py:40`-`60` does not implement the
   documented `data.zarr/img -> seed101` fallback. It only walks one parent up,
   so `/data/seed101/data.zarr/img` and `/data/seed102/data.zarr/img` both
   resolve to `data.zarr`.

   That collapses multiple NISB-style volumes into the same per-volume
   directory and contradicts `code_v1.md`'s stated behavior that
   `data.zarr/img` resolves to the enclosing seed directory. This affects
   `resolve_output_filenames`, `_resolve_tuning_prediction_files`, and any cache
   lookup using the same helper. I reproduced the current behavior with:
   `/data/seed101/data.zarr/img -> data.zarr` and
   `/data/seed102/data.zarr/img -> data.zarr`.

3. `connectomics/runtime/cache_resolver.py:246`-`253` and
   `connectomics/runtime/cache_resolver.py:319`-`321` still derive cache
   preflight volume names with `Path(...).stem` instead of the canonical
   per-volume resolver.

   For the same `data.zarr/img` and generic `img.h5` cases, preflight checks use
   `img` while the writer path uses the helper-derived directory. Even after the
   helper is fixed, these paths can still disagree unless they call the same
   canonical stem function. This means valid per-volume caches can be missed
   before `trainer.test()` and in cache-only decode/eval flows.

## Tests to Add

- Add a stem-resolution test for at least
  `/data/seed101/data.zarr/img` and `/data/seed102/data.zarr/img`, asserting
  distinct `seed101` / `seed102` volume directories.
- Add chunked test-step coverage for both `chunking.output_mode=raw_prediction`
  and `chunking.output_mode=decoded`, asserting files are passed to the chunked
  runners as `<save_path>/<volume>/<artifact>.h5` with no duplicated extension.
- Add cache-preflight coverage for `data.zarr/img` and `img.h5`-style paths so
  `preflight_test_cache_hit` and `try_cache_only_test_execution` use the same
  volume names as the writers.

## Questions

- Should `.zarr` / `.n5` container names always be treated as structural
  parents to skip when deriving `volume_stem`, or should this be controlled by
  the existing `data.{val,test}.name` field for ambiguous layouts?

## Verdict

VERDICT: NEEDS_CHANGES
