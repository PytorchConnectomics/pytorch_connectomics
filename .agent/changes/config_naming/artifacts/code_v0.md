# Code v0

## Overview

First implementation pass for the config-naming + per-mode `save_path`
redesign laid out in `plan_v0.md` → `plan_v1.md` → `plan_v2.md`.

This pass lands the **schema renames + alias map + YAML migration +
tests + helper scaffolding** that the plan calls for. The
**per-volume subdirectory writer change and the leading-type-word
filename rewrite** described in plan_v1 §C/§E are deferred to
`code_v1` — see `## Risks and Unknowns` below for why and what's
needed.

All 441 unit tests pass. `scripts/validate_tutorial_configs.py`
passes. Tutorial validator and strict-key rejector now reject every
legacy key with a hint pointing at its renamed sibling.

## What Changed

1. **Schema**: storage-leaf renames on `InferenceConfig`, `DecodingConfig`,
   and a flatten of `tune.output:*` → `tune.save_*`.
   - `inference.output_path → save_path`;
     `cache_suffix → save_cache_suffix`;
     `dtype → save_dtype`;
     `backend → save_backend`;
     `compression → save_compression`;
     `tta_result_path → load_tta_path`.
   - `inference.chunks` and `inference.write_mode` deleted (dead).
   - `decoding.output_path → save_path`;
     `output_suffix → save_suffix`;
     `input_prediction_path → load_prediction_path`.
   - `TuneOutputConfig` dataclass deleted; its fields hoisted to
     `TuneConfig` as `tune.save_path`, `tune.save_predictions_path`,
     `tune.save_cache_suffix`, `tune.save_all_trials`,
     `tune.save_best_segmentation`, `tune.save_study`,
     `tune.save_visualizations`, `tune.save_report`.
   - New: `data.{train,val,test}.name: Optional[str|List[str]] = None`
     on the shared `DataInputConfig`. Train-time use is advisory; the
     validator emits a non-fatal warning when set on `data.train`.
2. **Strict-key rejector** (`config/pipeline/config_io.py`): extended
   `_INFERENCE_RUNTIME_ALIAS_REPLACEMENTS` with the 8 inference renames;
   added `_DECODING_RUNTIME_ALIAS_REPLACEMENTS` and
   `_TUNE_OUTPUT_FIELD_REPLACEMENTS`; rejector raises with hints.
3. **Validator script** (`scripts/validate_tutorial_configs.py`):
   `LEGACY_PATTERNS` covers every renamed key for `inference`, `decoding`,
   and `tune.output.*`. `ADVISORY_PATTERNS` warns on `data.train.name`.
4. **Output-naming helpers** (`runtime/output_naming.py`):
   - Renamed `tta_cache_suffix → raw_cache_suffix`,
     `tta_cache_suffix_candidates → raw_cache_suffix_candidates`,
     `is_tta_cache_suffix → is_raw_cache_suffix` (with back-compat
     aliases for the old names so external callers don't break).
   - Read `inference.save_cache_suffix` (renamed) in
     `resolve_prediction_cache_suffix`.
   - Read `decoding.save_suffix` (renamed) in
     `format_decoding_output_suffix_tag`.
   - Added `resolve_dataset_volume_stems(cfg, mode)` that enumerates
     per-volume names from `data.{val|test}.name` (explicit), then from
     decode-only `load_prediction_path` parent dir, then from
     image-filename stems with parent-dir fallback when stem is one of
     `{img, image, raw, em, main, data}`.
   - Added `resolve_volume_save_dir(cfg, mode, volume_stem)` that joins
     `<inference.save_path>/<volume_stem>`. Not yet wired into the
     writers (deferred to code_v1).
5. **Checkpoint dispatcher** (`runtime/checkpoint_dispatch.py`): writes
   `cfg.inference.save_path`, `cfg.tune.save_path`,
   `cfg.tune.save_predictions_path` instead of the old paths.
6. **Consumer migration** (~30 .py sites): bulk renamed all reads to
   the new field names across `connectomics/{inference,decoding,
   training/lightning,runtime,evaluation}/`. Updated
   `connectomics/runtime/tune_runner.py` to use flat `tune.save_*`
   fields; `connectomics/decoding/tuning/optuna_tuner.py` similarly.
   Cache resolver reads `tune.save_predictions_path` directly instead
   of the deleted `tune.output.output_pred`.
7. **YAMLs** (~20 files): tutorials, profiles, templates, and the test
   yaml updated to the new keys. The `tune_profiles.yaml` had its nested
   `output: { save_study: true }` block hoisted via a generic
   "any-output:-with-tune-fields" pass.
8. **Tests**: bulk-renamed field accesses across 8 unit test files;
   updated 2 strict-config rejection cases. Added new
   `tests/unit/test_output_path_resolution.py` covering:
   - Test/tune mode `save_path` derivation from a checkpoint.
   - `resolve_dataset_volume_stems` with explicit string,
     explicit list, uninformative-stem fallback, decode-only flow,
     and tune-mode (val) data resolution.
   - `resolve_volume_save_dir` joining.
   - `raw_cache_suffix` format + back-compat alias.

## Implementation Details

### Naming rule (locked in plan_v2)

Storage-policy leaves at section root use the `save_` prefix; input-
loading leaves use `load_` with a *specific* artifact noun
(`load_tta_path`, `load_prediction_path`); other leaves keep natural
names (`enabled`, `metrics`, `n_trials`, `affinity_mask_path`).
Nested dataclass blocks (`inference.model`, `inference.window`,
`decoding.postprocessing`) stay unprefixed concept names. The
narrowed rule is documented in `plan_v2.md §A` and reflected in the
schema docstrings.

### `resolve_dataset_volume_stems` resolution chain

```
1. data.{val|test}.name (str applied uniformly | List[str] indexed)
2. decoding.load_prediction_path.parent.name | inference.load_tta_path.parent.name
3. Path(image).stem if not in UNINFORMATIVE = {img, image, raw, em, main, data}
4. Path(image).parent.name if it is informative
5. f"volume_{idx}"
```

Implemented in `runtime/output_naming.py:resolve_dataset_volume_stems`.

### Back-compat aliases in `output_naming.py`

The `tta_cache_suffix → raw_cache_suffix` rename keeps a same-module
alias `tta_cache_suffix = raw_cache_suffix` so external imports
continue to work during the transition. Same for
`tta_cache_suffix_candidates` and `is_tta_cache_suffix`. The legacy
names remain in `__all__` to preserve the public API surface; they
can be removed after one release cycle.

### `data.train.name` advisory warning

The validator runs through `ADVISORY_PATTERNS` after the
`LEGACY_PATTERNS` strict check. When set, it prints
`"data.train.name has no effect; train mode writes no per-volume
artifacts."` and continues with exit 0. This matches plan_v2's
chosen policy (warning, not error).

### `inference.chunks`/`write_mode` deletion checkpoint fallout

Both fields confirmed unused via source grep. Deleted from
`InferenceConfig`. Plan_v2 §C noted that pre-PR checkpoints which
embedded these fields would need re-pickling via
`scripts/checkpoint_conversion.py`. Within this repo, no current
tutorial sets them and no existing checkpoints in `outputs/` reference
them (verified by grep). One-shot remediation step is documented
inline.

## Files Changed

| File | Purpose |
|---|---|
| `connectomics/config/schema/inference.py` | Rename storage leaves; delete `chunks`/`write_mode`; rename `tta_result_path → load_tta_path` |
| `connectomics/config/schema/decoding.py` | Rename `output_path/output_suffix/input_prediction_path → save_path/save_suffix/load_prediction_path` |
| `connectomics/config/schema/stages.py` | Delete `TuneOutputConfig`; hoist 8 fields onto `TuneConfig` |
| `connectomics/config/schema/data.py` | Add `name: Optional[Any]` to `DataInputConfig` |
| `connectomics/config/schema/__init__.py` | Drop `TuneOutputConfig` import + `__all__` |
| `connectomics/config/pipeline/config_io.py` | Extend rejector for inference, decoding, and tune.output renames |
| `connectomics/runtime/output_naming.py` | Add `resolve_dataset_volume_stems`, `resolve_volume_save_dir`; rename `tta_*` helpers; back-compat aliases |
| `connectomics/runtime/checkpoint_dispatch.py` | Write to renamed fields |
| `connectomics/runtime/cli.py` | Read/write `inference.save_path` |
| `connectomics/runtime/dispatch.py` | Read/write `inference.save_cache_suffix` |
| `connectomics/runtime/cache_resolver.py` | Reads from `tune.save_predictions_path`; flat field reads |
| `connectomics/runtime/tune_runner.py` | Backups use `save_results/save_cache_suffix/save_path`; reads `tune.save_path`/`save_predictions_path` |
| `connectomics/inference/{output,chunked,chunk_grid,stage}.py` | Read renamed fields |
| `connectomics/decoding/{output,streamed_chunked,stage}.py` | Read renamed fields; updated docstring |
| `connectomics/decoding/tuning/optuna_tuner.py` | Read `tune.save_path`/`save_study` (flat) |
| `connectomics/training/lightning/{model,test_pipeline,data_factory}.py` | Read renamed fields |
| `connectomics/evaluation/{context,nerl,report}.py` | Read `inference.save_path` |
| `scripts/validate_tutorial_configs.py` | Full `LEGACY_PATTERNS` for renames; `ADVISORY_PATTERNS` for `data.train.name` |
| `tutorials/**/*.yaml` (~20) | Field renames |
| `tests/inference/test_nisb/base.yaml` | Field renames |
| `connectomics/config/profiles/tune_profiles.yaml` | Hoist `output:` to root for all 3 tune profiles |
| `tests/unit/test_{hydra_config,main_runtime_stage_switch,inference_stage,prediction_transform,optuna_tuner,connectomics_module,lit_utils,nnunet_preprocessing,test_pipeline_multi_volume_eval}.py` | Bulk rename field accesses; update strict-config rejection assertions |
| `tests/unit/test_output_path_resolution.py` | NEW — covers per-mode path derivation, stem resolution, raw cache suffix |

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

# Targeted tests (focused on renamed surfaces)
python -m pytest tests/unit/test_v3_guardrails.py \
                 tests/unit/test_v2_boundaries.py \
                 tests/unit/test_hydra_config.py \
                 tests/unit/test_output_path_resolution.py \
                 tests/unit/test_main_runtime_stage_switch.py \
                 tests/unit/test_inference_stage.py \
                 tests/unit/test_prediction_transform.py \
                 tests/unit/test_optuna_tuner.py \
                 tests/unit/test_decoding_pipeline.py -q
# Result: passed after fixing 3 transient assertions (test yaml legacy keys; storage_dtype attr name).

# Full unit suite
python -m pytest tests/unit/ -q
# Result: 441 passed, 45 warnings in 88s
```

Tutorial validator clean. All 441 unit tests pass. Diff stat:
70 files changed, 993 insertions, 777 deletions.

## Review Focus

Highest-leverage review areas:

1. **Strict-key rejector** (`config_io.py` lines 113-194): the
   alias-replacement maps now have several new entries; verify the
   `tune.output.*` rejection emits a clear hint per legacy key, and
   that `_path_is_or_descendant` correctly catches descendant paths
   (so e.g. setting `tune.output.visualizations.foo: bar` is also
   rejected, not just `tune.output.visualizations`).
2. **`resolve_dataset_volume_stems`** (`output_naming.py:53-104`):
   the fallback chain has six branches; corner cases worth checking:
   - Explicit `name` is `List[str]` shorter than dataset items → silently
     falls through to filename-stem branch (intentional; reviewer may
     prefer a hard error).
   - Decode-only flow with no `decoding.load_prediction_path` *and* no
     image data — returns `[]`. Tests don't cover this.
3. **`resolve_volume_save_dir`** (lines 107-122): falls back to the
   parent of `decoding.load_prediction_path` when `inference.save_path`
   is empty. The fallback walks one level up (`.parent.parent`),
   assuming the load path lives under `<base>/<volume_stem>/...`.
   Review whether that assumption holds across all decode-only flows.
4. **Validator advisory channel** (`validate_tutorial_configs.py`):
   the new `ADVISORY_PATTERNS` block prints warnings to stdout but
   exits 0. Review whether CI consumers that parse stdout will tolerate
   the new warning format.
5. **Test suite renames**: many tests had their string literal fields
   updated (e.g. `cfg.inference.cache_suffix = "_x1_prediction.h5"` →
   `cfg.inference.save_cache_suffix`). Spot-check that the *value*
   strings (`"_x1_prediction.h5"`) weren't accidentally mutated.

## Risks and Unknowns

### Deferred from plan: per-volume subdir + filename rewrite

`plan_v1 §E` (canonical filename table) and `plan_v1 §F` (per-mode
directory layout) call for:
- New filenames `raw_x{n}{head}{ch}.h5`, `decoded_x{n}_<decoder>_<kwargs>.h5`,
  `decoded_step{idx}_*.h5`, `eval.txt`, `eval_<metric>.npz` —
  all without the `<volume_stem>_` prefix or the `_ckpt-...` token
  (encoded by parent dir).
- Per-volume subdir `<inference.save_path>/<volume_stem>/<artifact>.h5`
  in `write_outputs` and `write_decoded_outputs`.

This pass kept the **legacy filename format unchanged** (still includes
`{stem}_x1_..._ckpt-..._prediction.h5` flat under `results_step=<NNN>/`).
Reasons:

1. The writer-side change touches every cache-resolver lookup, every
   filename-glob in `final_prediction_decoded_glob_suffix`, and every
   downstream tooling assumption. Risk of breaking on-disk reads of
   existing experiments is high without a parallel migration.
2. The naming + alias-map work alone is already 70 files of churn; a
   reviewer can audit it cleanly before adding the path-layout change
   on top.
3. The new helpers (`resolve_dataset_volume_stems`,
   `resolve_volume_save_dir`) are wired into the test suite but not
   yet called from `write_outputs` or `write_decoded_outputs`. This is
   intentional — they're staged for `code_v1`.

What `code_v1` needs to land:
- Update `write_outputs` and `write_decoded_outputs` to write
  `<output_dir>/<volume_stem>/<artifact>.<ext>`.
- Refactor `final_prediction_output_tag`,
  `intermediate_decode_step_output_tag`,
  `raw_cache_suffix` to drop `_ckpt-...` and `<volume_stem>_` from
  filenames; produce the leading-type-word names (`raw_`, `decoded_`).
- Update `cache_resolver` lookups to walk into per-volume subdirs.
- Migrate eval-file writers in `evaluation/report.py` to per-volume.
- Manual smoke test on
  `tutorials/neuron_nisb/base_banis_crop.yaml --mode test
  --checkpoint <existing>` to verify the new tree appears under
  `<base>/results_step=<NNN>/<volume_stem>/`.

### Other risks

- **Back-compat aliases in `output_naming.py`**: `tta_cache_suffix`,
  `tta_cache_suffix_candidates`, `is_tta_cache_suffix` remain as
  aliases pointing at the renamed `raw_*` functions. They are still in
  `__all__`. If V3 contract demands no compat shims, these aliases
  should be deleted in `code_v1` once the new names are stable.
- **Decoded glob suffix** (`final_prediction_decoded_glob_suffix`):
  unchanged in this pass. Still produces
  `_x{n}{head}{ch}{ckpt}_decoding{dec}*{suffix}.h5`. After `code_v1`
  the parent-dir-encoded `_ckpt` part should drop.
- **Existing on-disk caches** under `outputs/<exp>/<ts>/results_step=<NNN>/`
  will continue to be hit by the cache-resolver (filename format
  unchanged). When `code_v1` lands the per-volume layout, those caches
  will silently miss; document in PR description.

## Changes Since Previous Code Version

Initial implementation.
