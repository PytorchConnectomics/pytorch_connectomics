# Review v0

## Summary

The schema rename and YAML migration are a substantial start, but this code version is not ready to approve. It explicitly defers the core output-layout deliverable, still reads deleted inference fields in runtime writers, and adds compatibility aliases that violate the repository's V3 clean-break contract.

## Diff Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9

## Findings

1. `connectomics/inference/stage.py:65`, `connectomics/inference/chunked.py:442`, `connectomics/decoding/streamed_chunked.py:330` still read `cfg.inference.compression` after `InferenceConfig.compression` was renamed to `save_compression`.

   Since `compression` is no longer a schema field, these paths silently fall back to `"gzip"` and ignore user configuration in `inference.save_compression`. That affects single-volume prediction artifacts, chunked raw prediction writes, and streamed chunked decoded writes. Any tutorial or CLI override setting `save_compression: null` or another supported value will not take effect in these code paths.

2. `connectomics/runtime/output_naming.py:432` exposes `raw_cache_suffix`, but it still returns `_tta_x..._prediction.h5` and `tests/unit/test_output_path_resolution.py:136` asserts that legacy prefix.

   Plan v2 resolved the naming question by choosing a single canonical `raw_` prefix for cached model predictions. The implementation renames the function to `raw_cache_suffix` while preserving the old `tta` filename token, which leaves the public API and tests inconsistent with the accepted plan. This will cause churn in code_v1 and makes strict-config/user-facing naming say one thing while on-disk names say another.

3. `connectomics/runtime/output_naming.py:452`, `connectomics/runtime/output_naming.py:529`, and `connectomics/runtime/output_naming.py:638` add `tta_*` compatibility aliases and keep them in `__all__`.

   The repo-local V3 contract explicitly says to delete duplicate import paths, facade re-exports, and compatibility shims instead of preserving them. The task also calls for strict legacy-key rejection rather than compatibility. These aliases are not just internal temporaries; they remain public exports and therefore preserve the old API surface. They should be removed, and callers/tests should use only the canonical `raw_*` names.

4. The code version does not implement the central path-layout deliverable yet: `write_outputs`, `write_decoded_outputs`, cache lookup, and evaluation report writing still use the flat legacy filename layout.

   `code_v0.md` calls this out as deferred, but the user deliverable is specifically to derive tune/test output paths into per-checkpoint and per-volume subfolders and remove dataset/checkpoint identity from filenames. Leaving `resolve_volume_save_dir` unused means the new schema exists without the behavioral change it was designed to support. This must land before approval: writers, cache resolver, decoded glob matching, and eval report output need to consume the per-volume directory helper and canonical filenames.

5. The CCC state reached `code_v0` even though the plan phase never received an approving review verdict.

   `plan_v0_review.md` and `plan_v1_review.md` both say `VERDICT: NEEDS_CHANGES`; `plan_v2.md` was the max allowed plan version and has no review stage under the protocol. The final plan may be usable, but the recorded workflow did not have an approved plan transition before code was written. Before the next code artifact, either the coordinator state should be corrected by an explicit user decision to accept `plan_v2`, or the next artifact should document that it is proceeding under user override from the max-rounds stop point.

## Tests to Add

- Add a focused test that sets `cfg.inference.save_compression = None` or a non-default value and verifies `run_prediction_inference`, chunked raw inference, and streamed chunked decoding pass that value to their HDF5 writers.
- Update `test_output_path_resolution.py` to assert the accepted canonical raw filename prefix instead of `_tta_x` once the raw cache filename change lands.
- Add writer/cache tests that verify artifacts are written and found under `<results_step>/<volume_stem>/...`, including decoded outputs and evaluation reports.
- Add a public API snapshot assertion that `tta_cache_suffix`, `tta_cache_suffix_candidates`, and `is_tta_cache_suffix` are not exported.

## Questions

- Is the user explicitly accepting `plan_v2` despite the max-rounds stop, or should the CCC run be restarted with another planning review round?
- Should `raw_cache_suffix` return a full filename stem (`raw_x1...`) or continue to return a suffix with a leading separator? The next implementation should settle this once and align all tests.

## Verdict

VERDICT: NEEDS_CHANGES
