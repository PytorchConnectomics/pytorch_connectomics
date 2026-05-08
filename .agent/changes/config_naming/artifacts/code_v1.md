# Code v1

## Overview

Address all five findings from `review_v0.md`:

1. **Finding 1**: Three `cfg.inference.compression` reads (escaped the v0
   regex because they used `getattr(cfg.inference, "compression", "gzip")`)
   now read `cfg.inference.save_compression`. Added regression test for
   the compression flow into chunked inference HDF5 writers.
2. **Finding 2**: `raw_cache_suffix` no longer returns the legacy
   `_tta_x..._prediction.h5` string. It now returns the canonical
   per-volume artifact filename: `raw_x{n}{head}{ch}.h5`.
3. **Finding 3**: All `tta_cache_suffix`, `tta_cache_suffix_candidates`,
   and `is_tta_cache_suffix` back-compat aliases removed from
   `output_naming.py` and `runtime/__init__.py`. Tests and runtime now
   import only the canonical `raw_*` names.
4. **Finding 4**: Per-volume subdirectory layout + filename rewrite landed.
   Writers (`write_outputs`, `write_decoded_outputs`) write to
   `<save_path>/<volume_stem>/<artifact>.<ext>`. Filenames lose the
   `<volume_stem>_` prefix and `_ckpt-` token (encoded by parent dirs).
   Cache resolver (`resolve_cached_prediction_files`,
   `has_cached_predictions_in_output_dir`,
   `_load_cached_predictions`) walks per-volume subdirs. Eval reports
   move to `<save_path>/<volume_name>/eval_<tag>.{txt,npz}`.
5. **Finding 5**: `plan_v2.md` is the user-approved plan (manual override
   from the `plan_rounds=2` max-rounds stop), recorded in `run.md`.
   `code_v1.md` proceeds against that approved plan.

All 447 unit tests pass; tutorial validator clean. Diff vs `run_start_ref`:
73 files changed, +1256 / −989.

## What Changed

### Findings 1–3 (output_naming + consumers)

- `connectomics/runtime/output_naming.py`:
  - `raw_cache_suffix(cfg, ...)` now returns `raw_x{n}{head}{ch}.h5`.
  - `intermediate_prediction_cache_suffix` composes
    `raw_x{n}...{chunked-raw-tag}{user_save_suffix}.h5`.
  - `final_prediction_output_tag` returns
    `decoded_x{n}{head}{ch}_<dec>{user}.h5` (or
    `prediction_x{n}{head}{ch}{user}.h5` when no decoders configured).
  - `intermediate_decode_step_output_tag` returns
    `decoded_step{idx}_x{n}{head}{ch}_<step_tag>{user}.h5` (no
    `_decoding_` infix; type word `decoded_` already leads).
  - `final_prediction_decoded_glob_suffix` returns
    `decoded_x{n}{head}{ch}<dec>*{user}.h5` for per-volume glob lookup.
  - `is_raw_cache_suffix` matches the new `raw_x` prefix.
  - `resolve_prediction_cache_suffix` simplified to delegate uniformly to
    `intermediate_prediction_cache_suffix` (per-volume layout uses one
    canonical filename across all modes).
  - `intermediate_prediction_cache_suffix_candidates` and
    `raw_cache_suffix_candidates` now return a single canonical entry
    (legacy two-variant chain dropped — checkpoint identity is encoded
    by parent dir, not the filename).
  - `_format_one_decode_step` and `format_decode_tag` unchanged in
    semantics; `format_decoding_output_suffix_tag` already updated.
  - Back-compat aliases removed: `tta_cache_suffix`,
    `tta_cache_suffix_candidates`, `is_tta_cache_suffix`. Their entries
    in `__all__` removed.

- `connectomics/runtime/__init__.py`: removed `tta_cache_suffix_candidates`
  from `_LAZY_EXPORTS` and `__all__`; added `raw_cache_suffix_candidates`.

- `connectomics/inference/{stage,chunked}.py` and
  `connectomics/decoding/streamed_chunked.py`: read
  `inference.save_compression` (was `inference.compression` due to v0
  regex miss).

### Finding 4 (per-volume layout)

- **`connectomics/inference/output.py`**:
  - `resolve_output_filenames` now uses the parent-dir fallback heuristic
    via `_stem_from_image_path` so volumes loaded as
    `data.zarr/img` resolve to their parent dir name (e.g. `seed101`)
    instead of the uninformative `img` shared across volumes.
  - `write_outputs` writes to `<output_dir>/<volume_stem>/<artifact>`.
    The `suffix` parameter is now interpreted as the artifact filename
    (with optional `.h5` extension; appended if missing).

- **`connectomics/decoding/output.py`**: `write_decoded_outputs` writes
  to `<output_dir>/<volume_stem>/<artifact>.h5`. Helper `_strip_h5`
  added.

- **`connectomics/runtime/cache_resolver.py`**:
  - `resolve_cached_prediction_files` walks
    `<output_dir>/<filename>/<artifact>` for the new layout. Glob and
    preferred-suffix paths updated.
  - `has_cached_predictions_in_output_dir` enumerates expected stems via
    the new `resolve_dataset_volume_stems` helper, falling back to
    image-path-derived stems for unit-test configs.
  - Cache-only decode/eval call sites pass
    `final_prediction_output_tag(cfg, ...)` directly (already includes
    `.h5`) instead of wrapping with `_<...>.h5`.

- **`connectomics/training/lightning/model.py:_load_cached_predictions`**:
  Updated three lookup branches (decoded-final, decoded-glob, raw-cache)
  to walk `<output_dir>/<filename>/<artifact>`. The chosen-suffix
  return value now is the artifact filename (no
  `name[len(filename):]` trimming).

- **`connectomics/training/lightning/test_pipeline.py`**:
  `_save_intermediate_prediction_outputs` passes the cache_suffix
  unchanged (no more `removeprefix("_").removesuffix(".h5")`).

- **`connectomics/evaluation/report.py`**:
  `save_metrics_to_file` writes to
  `<save_path>/<volume_name>/eval_<tag>.txt` (and `eval_<tag>_*.npz`
  for NERL per-GT files). `decode_experiments.tsv` stays at
  `<save_path>/` (cross-volume log).

- **`connectomics/decoding/tuning/optuna_tuner.py`**:
  `_resolve_tuning_prediction_files` now constructs
  `<predictions_dir>/<_stem_from_image_path(path)>/<cache_suffix>` per
  the new layout.

### Tests

- `tests/unit/test_output_path_resolution.py`: added five new cases —
  channel-selector encoding, chunked-raw token, `decoded_*` filename
  format, `prediction_*` fallback when no decoder, intermediate-step
  filename. All assert the new short-form filenames.
- `tests/unit/test_inference_stage.py`: added regression
  `test_run_prediction_inference_honors_save_compression` (Finding 1).
- `tests/unit/test_lit_utils.py`: assertions migrated to per-volume
  layout. `test_tta_cache_suffix_candidates_*` renamed to
  `test_raw_cache_suffix_candidates_returns_canonical_per_volume_filename`.
  `test_chunked_raw_intermediate_suffix_does_not_collide_with_whole_volume_cache`
  expects `raw_x1_ch0-1-2_chunked-raw_cs1000x1000x1350_chunk_raw_v1.h5`.
  Cache-resolver tests rewritten to create files at `<dir>/<volume>/<file>`.
- `tests/unit/test_connectomics_module.py`: per-volume cache file
  fixtures and assertions; `evaluation_metrics_*` filename assertions
  rewritten to `<vol>/eval_<tag>.txt`.
- `tests/unit/test_main_runtime_stage_switch.py`: tune cache detection
  test uses `volume_a.h5` (informative stem) and per-volume subdir
  layout.
- `tests/unit/test_lightning_data_collate.py`: stem-fallback test
  expectation updated to parent-dir name.
- `tests/unit/test_test_pipeline_multi_volume_eval.py`: multi-head save
  assertions use `raw_x1_head-*.h5` filenames.
- `tests/unit/test_nnunet_preprocessing.py`: per-volume subdir lookup.

## Implementation Details

### Filename token policy (locked)

| Artifact                   | Filename                                                              |
|---|---|
| Raw prediction (saved)     | `raw_x{n}{head}{ch}.h5`                                               |
| Raw prediction (chunked)   | `raw_x{n}{head}{ch}_chunked-raw_cs{...}.h5`                           |
| Decoded final              | `decoded_x{n}{head}{ch}_<decoder>_<kwargs>{user_suffix}.h5`           |
| Decoded glob               | `decoded_x{n}{head}{ch}<dec>*{user}.h5`                               |
| Decoded intermediate (N≥2) | `decoded_step{idx}_x{n}{head}{ch}_<step_tag>{user}.h5`                |
| Eval summary               | `eval_<tag>.txt` where `<tag>` = `final_prediction_output_tag` (no `.h5`) |
| Eval per-metric NPZ        | `eval_<tag>_nerl_per_gt_erl.npz`                                      |
| Decode experiments log     | `decode_experiments.tsv` (at save_path root, cross-volume)            |

### Per-volume directory layout

```
<inference.save_path>/                    # = .../results_step=<NNN>/ in test mode
  <volume_stem_a>/
    raw_x1_ch0-1-2.h5
    decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75.h5
    eval_decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75.txt
  <volume_stem_b>/
    raw_x1_ch0-1-2.h5
    decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75.h5
  decode_experiments.tsv
  affinity_mask.h5                        # cross-volume asset (kept at root)
```

`<volume_stem>` resolution: explicit `data.{val,test}.name`, then
parent-of-load-path for decode-only flows, then
`Path(image).stem` if informative, then parent-dir fallback for
uninformative stems (`img`, `image`, `raw`, `em`, `main`, `data`).

### Suffix parameter API (`write_outputs`, `write_decoded_outputs`)

Both writers now interpret `suffix` as an artifact filename (with or
without `.h5` extension). The volume stem (passed via `filenames[idx]`)
becomes a directory between `output_dir` and the artifact.

This is a behavioral change for callers that pass legacy suffixes like
`"prediction"` (no extension) — they now produce
`<output_dir>/<volume>/prediction.h5`. Tests updated.

## Files Changed

| File | Purpose |
|---|---|
| `connectomics/inference/output.py` | Per-volume subdir join in `write_outputs`; new `_strip_extension`; `resolve_output_filenames` uses parent-dir fallback |
| `connectomics/inference/{stage,chunked}.py` | Read `inference.save_compression` |
| `connectomics/decoding/output.py` | Per-volume subdir join in `write_decoded_outputs`; new `_strip_h5` |
| `connectomics/decoding/streamed_chunked.py` | Read `inference.save_compression` |
| `connectomics/decoding/tuning/optuna_tuner.py` | `_resolve_tuning_prediction_files` walks per-volume subdir |
| `connectomics/runtime/output_naming.py` | New filename format for raw / decoded / glob / intermediate-step; `is_raw_cache_suffix` matches new prefix; back-compat `tta_*` aliases removed; `_candidates` simplified |
| `connectomics/runtime/__init__.py` | `tta_cache_suffix_candidates` removed; `raw_cache_suffix_candidates` added |
| `connectomics/runtime/cache_resolver.py` | Per-volume lookup in 3 helpers; cache-only call sites pass full filenames |
| `connectomics/training/lightning/model.py` | `_load_cached_predictions` walks per-volume subdirs |
| `connectomics/training/lightning/test_pipeline.py` | `_save_intermediate_prediction_outputs` passes cache_suffix unchanged |
| `connectomics/evaluation/report.py` | Eval files in `<save_path>/<volume>/eval_<tag>.{txt,npz}` |
| `tests/unit/test_output_path_resolution.py` | +5 cases for canonical filename format |
| `tests/unit/test_inference_stage.py` | +1 case: `inference.save_compression` regression |
| `tests/unit/test_lit_utils.py` | Per-volume layout in cache-resolver tests; raw cache filename assertions |
| `tests/unit/test_connectomics_module.py` | Per-volume cache file fixtures; eval-file paths |
| `tests/unit/test_main_runtime_stage_switch.py` | Tune-cache test uses informative stem + per-volume layout |
| `tests/unit/test_lightning_data_collate.py` | Stem-fallback assertion updated |
| `tests/unit/test_test_pipeline_multi_volume_eval.py` | Multi-head save assertion uses `raw_x1_head-*.h5` |
| `tests/unit/test_nnunet_preprocessing.py` | Per-volume subdir lookup |
| `tests/unit/test_optuna_tuner.py` | Per-volume cache file fixtures |

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

(All changes in working tree; no commits in this code stage.)

## Verification

Commands actually run:

```
# Tutorial validator
python scripts/validate_tutorial_configs.py
# Result: Validated 14 canonical tutorial configs successfully; skipped 2 custom workflow YAMLs.

# Targeted: per-volume + raw filename
python -m pytest tests/unit/test_output_path_resolution.py -q
# Result: 14 passed

# Full unit suite
python -m pytest tests/unit/ -q
# Result: 447 passed, 45 warnings in 84.55s
```

## Review Focus

1. **Finding 4 path semantics** — verify the per-volume subdir layer is
   applied uniformly across writers, the cache resolver, and the
   per-volume eval file move. Edge case worth re-checking:
   `_load_cached_predictions`'s decoded-glob path uses
   `(output_dir / filename).glob(decoded_glob)`; if `<filename>` is
   missing entirely (no cached predictions for that volume), the glob
   silently returns no matches. Behaviorally same as before, but the
   user might expect the "no prediction subdir at all" case to be
   distinguishable from "subdir present but no decoded files." It isn't.
2. **Volume-stem heuristic in `resolve_output_filenames`** — the parent-
   dir fallback now triggers when the stem matches one of
   `{img, image, raw, em, main, data}`. The set is tight; widen if
   reviewer prefers (e.g. add `gt`, `seg`, `volume`). Single source of
   truth: `_UNINFORMATIVE_STEMS` in `runtime/output_naming.py`.
3. **`intermediate_prediction_cache_suffix_candidates` simplification**
   — the function now returns a single canonical entry. Plan v2 keeps
   the parameter but says checkpoint identity is encoded by parent dir.
   Reviewer should confirm no caller depends on the legacy
   two-element list. (None found by grep.)
4. **`final_prediction_output_tag` always returns a filename ending in
   `.h5`** — callers that previously stripped `.h5` to use the value as a
   sub-filename token (e.g. eval_<tag>.txt) now strip it themselves. The
   one such caller is `evaluation/report.py` which I updated. Reviewer
   may prefer a separate `final_prediction_artifact_basename` helper
   that returns without extension; current pragma keeps it inline.

## Risks and Unknowns

- **`affinity_mask.h5` placement unchanged** (cross-volume asset at
  `<save_path>/`). If a future reviewer wants per-volume masks, that
  needs a different config shape (`affinity_mask_path` is currently a
  single file, not a per-volume dict).
- **Existing on-disk caches will not be hit by the new resolver** —
  files written by code_v0 (or any pre-PR version) at the flat
  `<save_path>/<filename>_<suffix>.h5` location are now invisible to the
  cache resolver. Re-runs incur re-inference. This is the V3 trade-off
  for a cleaner layout. Document in PR description.
- **`scripts/decode_large.py` and `tutorials/waterz_decoding_large*.yaml`**
  use the custom `large_decode:` / `abiss_large:` workflow keys (still
  excluded from validator). Their `output_path` field is part of the
  custom workflow surface — not the renamed `inference.save_path` — and
  is unaffected.
- **External tooling that grepped flat `results_step=*/<vol>_*.h5`**
  will need updating. No internal CLI tool depends on the flat layout
  per audit; document in PR description.

## Changes Since Previous Code Version

Address all five `review_v0.md` findings:

- **Finding 1**: Stray `cfg.inference.compression` reads in
  `inference/stage.py:65`, `inference/chunked.py:442`, and
  `decoding/streamed_chunked.py:330` migrated to
  `inference.save_compression`. Regression test added in
  `test_inference_stage.py`.
- **Finding 2**: `raw_cache_suffix` now returns the canonical per-volume
  artifact filename `raw_x{n}{head}{ch}.h5` instead of the legacy
  `_tta_x..._prediction.h5`. Tests updated to match.
- **Finding 3**: Back-compat aliases `tta_cache_suffix`,
  `tta_cache_suffix_candidates`, `is_tta_cache_suffix` removed from
  `output_naming.py` and `runtime/__init__.py`. All callers (3 modules,
  4 tests) migrated to canonical names.
- **Finding 4**: Per-volume subdirectory layout fully wired:
  `write_outputs`, `write_decoded_outputs`, cache resolver helpers,
  Lightning's `_load_cached_predictions`, eval report file location,
  and tuning prediction-file resolver. Filenames drop `<volume_stem>_`
  prefix and `_ckpt-` token. Eval files move to per-volume subdir.
- **Finding 5**: User accepted `plan_v2` from the max-rounds stop;
  `run.md` records the override. This `code_v1` proceeds against
  `plan_v2.md`.

Tests: 447 passing (was 441 pre-v1; net +6 new tests, 0 dropped).
