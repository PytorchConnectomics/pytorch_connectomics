# Plan v1

## Summary

Resolve the design ambiguities flagged in `plan_v0_review.md` and lock in
concrete answers so the code stage is mechanical rather than further
exploratory. Two coupled deliverables:

1. **Naming convention (narrowed and explicit)** — adopt a `save_*` /
   `load_*` prefix only for storage- and input-loading scalar leaves at
   stage-section root. Other leaves keep natural names; the rule
   acknowledges its scope explicitly. Rename surface is correspondingly
   tighter.

2. **Per-mode `output_path` derivation + per-volume subdir** — train
   keeps timestamp; test/tune derive from the checkpoint and add a
   per-volume subdir layer. Volume-stem resolution is deterministic via
   a new opt-in `data.test.name` field plus a documented fallback chain.
   Filenames inside per-volume dirs use a single canonical token policy
   for raw, decoded final, decoded intermediate, and eval files.

## Scope

In scope (unchanged from v0):

- Schema dataclasses: `InferenceConfig`, `DecodingConfig`,
  `EvaluationConfig`, `TuneConfig`, `TuneOutputConfig`, plus a new
  `data.test.name` / `data.val.name` field.
- Strict-key rejector + tutorial validator.
- Output-naming helpers + checkpoint-derived path resolver.
- Inference, decoding, evaluation, and runtime consumers that read
  the renamed/deleted fields.
- Tutorials, profile/template YAMLs, test YAML.
- Unit tests + public-API snapshot test.

Out of scope (unchanged from v0):

- Pipeline semantics (decoder chaining, postprocessing, TTA aggregation).
- New dataclasses for un-flattening `save_results` etc.
- Backward-compat shims (V3 contract).

## Proposed Changes

### A. Naming convention rule (narrowed)

> **Storage-policy leaves** at section root use the `save_` prefix. The
> noun suffix names which storage aspect the leaf controls.
>
> **Input-loading leaves** at section root use the `load_` prefix with a
> *specific* artifact noun, never the bare `load_path`. Examples:
> `inference.load_tta_path`, `decoding.load_prediction_path`.
>
> **Other root-level leaves** (domain knobs, switches, dataset selectors,
> tuner-engine controls) keep natural names — `enabled`, `metrics`,
> `n_trials`, `study_name`, `affinity_mask_path`, `prediction_threshold`,
> `nerl_*`. They are *exempt* from the prefix rule because they describe
> what the stage *does*, not where it stores or loads artifacts.
>
> **Nested dataclass blocks** stay as concept names without prefix
> (`inference.model`, `inference.window`, `decoding.postprocessing`).

This rule answers review-Q1 (narrow scope, not all root leaves) and
explicitly listed exemptions. The before/after tables below show the
exact rename surface.

### B. Schema before/after (tightened)

**`inference`** (`connectomics/config/schema/inference.py:InferenceConfig`):

| Before                              | After                                | Note |
|---|---|---|
| `inference.save_results: bool`      | unchanged                             |  |
| `inference.save_all_heads: bool`    | unchanged                             |  |
| `inference.output_path: str`        | `inference.save_path: str`            |  |
| `inference.cache_suffix: str`       | `inference.save_cache_suffix: str`    | namespace-consistent; verbose acceptable |
| `inference.dtype: Optional[str]`    | `inference.save_dtype: Optional[str]` |  |
| `inference.backend: str`            | `inference.save_backend: str`         |  |
| `inference.compression: Optional[str]` | `inference.save_compression: Optional[str]` |  |
| `inference.chunks: Optional[List[int]]` | **deleted**                       | dead — no consumer reads it |
| `inference.write_mode: str`         | **deleted**                           | dead — no consumer reads it |
| `inference.tta_result_path: str`    | `inference.load_tta_path: str`        | specific noun (review-finding 4) |

Other `inference.*` blocks (`model`, `execution`, `window`, `chunking`,
`test_time_augmentation`, `prediction_transform`, `memory_cleanup`,
`system`) and runtime aliases (`head`, `select_channel`, `crop_pad`,
`sliding_window`, `strategy`, `do_eval`) are **unchanged**.

**`decoding`** (`connectomics/config/schema/decoding.py:DecodingConfig`):

| Before                              | After                                | Note |
|---|---|---|
| `decoding.enabled: bool`            | unchanged                             | exempt (domain switch) |
| `decoding.save_intermediate: bool`  | unchanged                             |  |
| `decoding.save_results: bool`       | unchanged                             |  |
| `decoding.output_path: str`         | `decoding.save_path: str`             |  |
| `decoding.output_suffix: str`       | `decoding.save_suffix: str`           | filename-level user tag |
| `decoding.input_prediction_path: str` | `decoding.load_prediction_path: str` | specific noun |
| `decoding.affinity_mask_path: str`  | unchanged                             | exempt (cross-volume input asset, not save/load action) |
| `decoding.steps`, `postprocessing`, `tuning` | unchanged                  |  |

**`evaluation`** (`connectomics/config/schema/evaluation.py:EvaluationConfig`):

Schema unchanged. All leaves are domain knobs (`metrics`, `*_threshold`,
`nerl_*`) which the narrowed rule exempts. Eval **output paths** still
move into per-volume subdirs (Section C); that is a writer change, not
a schema change.

**`tune`** (`connectomics/config/schema/stages.py`):

`TuneOutputConfig` is **deleted** (review-finding 5). Its fields hoist
to `TuneConfig` with the `save_` prefix and matching defaults:

| Before                                 | After                                |
|---|---|
| `tune.output: TuneOutputConfig`         | **(block removed)**                  |
| `tune.output.output_dir`                | `tune.save_path: Optional[str]`      |
| `tune.output.output_pred`               | `tune.save_predictions_path: Optional[str]` |
| `tune.output.cache_suffix`              | `tune.save_cache_suffix: str`        |
| `tune.output.save_all_trials`           | `tune.save_all_trials: bool`         |
| `tune.output.save_best_segmentation`    | `tune.save_best_segmentation: bool`  |
| `tune.output.save_study`                | `tune.save_study: bool`              |
| `tune.output.visualizations`            | `tune.save_visualizations: Optional[Dict]` |
| `tune.output.report`                    | `tune.save_report: Optional[Dict]`   |

`TuneOutputConfig` removal:
- Drop import in `connectomics/config/schema/__init__.py` and `__all__`.
- Drop from public API snapshot (`tests/unit/test_public_api_snapshot.py`).
- Strict-key rejector raises on any `tune.output.*` YAML key with hint
  pointing at the new `tune.save_*` sibling.

Per-section `save_path` overload (review-finding from major-4):
**accepted**. `inference.save_path` (predictions dir),
`decoding.save_path` (decoded dir), `tune.save_path` (study dir),
`tune.save_predictions_path` (predictions cache dir for tune mode) live
in different sections; the section name carries the noun. Code-comment
and schema docstrings call this out.

### C. New `data.test.name` / `data.val.name` field

Add an optional explicit volume-name override to the `data.{test,val}`
sub-blocks (currently in `connectomics/config/schema/data.py`):

```python
@dataclass
class TestDataConfig:                       # (and ValDataConfig analogously)
    path: ... ; image: ... ; label: ... ; resolution: ...
    skeleton: Optional[str] = None
    name: Optional[Union[str, List[str]]] = None  # NEW
```

Semantics:
- If `name` is `None`, the runtime falls back to the deterministic
  resolution chain in section D.
- If `name` is a `str`, it is used as the per-volume subdir name for
  every dataset item.
- If `name` is `List[str]`, it must have the same length as the resolved
  dataset items (after glob expansion); each name maps to its index.

This addresses review-finding 3 by giving an explicit knob. Keep
backwards-compatible default (`None`) so all existing tutorials continue
to work via the fallback chain.

### D. Volume-stem resolution chain (deterministic)

A single helper `resolve_dataset_volume_stems(cfg, mode) -> List[str]`
in `connectomics/runtime/output_naming.py` returns the canonical list
of volume names for the active mode. Used by both writers and the
cache resolver. Resolution order **per dataset item**:

1. Explicit `data.{test|val}.name` value (string or indexed list entry).
2. **Decode-only path heuristic**: if the item is a `decoding.load_prediction_path`,
   the stem is the parent directory name of the load path.
   Same for `inference.load_tta_path`.
3. `Path(image_filename).stem` if it is informative (not in
   `{"img", "image", "raw", "em", "main", "data"}` — uninformative
   common dataset names).
4. `Path(image_filename).parent.name` if the previous fallback was
   uninformative or stem is empty.
5. `f"volume_{idx}"` as last resort.

Validator warns at config load when steps 3 and 4 yield duplicate stems
across multiple items in the same dataset, and points the user to set
`data.{test|val}.name` explicitly.

This answers review-Q3 (decode-only flow uses the load-path parent;
not a separate code path) and review-Q1 of the original v0 questions
(volume stem heuristic).

### E. Filename token policy (single canonical answer)

| Artifact                   | Filename                                                              |
|---|---|
| Raw prediction (saved)     | `raw_x{n}{head}{ch}.h5`                                               |
| Raw prediction (chunked)   | `raw_x{n}{head}{ch}_chunked-raw_cs{...}.h5`                           |
| Decoded final              | `decoded_x{n}{head}{ch}_<decoder>_<kwargs>{user_suffix}.h5`           |
| Decoded postprocessed      | `decoded_x{n}{head}{ch}_<decoder>_<kwargs>{user_suffix}_post.h5`      |
| Decoded intermediate (N≥2) | `decoded_step{idx}_x{n}{head}{ch}_<decoder>_<kwargs>{user_suffix}.h5` |
| Decoded intermediate (N=1) | **elided** (would duplicate final)                                    |
| Evaluation summary         | `eval.txt`                                                            |
| Evaluation per-metric NPZ  | `eval_<metric>.npz` (e.g. `eval_nerl.npz`)                            |

Tokens explained:
- `_x{n}` = TTA pass count (always preserved; aggregation context).
- `{head}` = `_head-<name>` if multi-head, else empty.
- `{ch}` = `_ch<indices>` if `select_channel` set, else empty.
- `<decoder>_<kwargs>` = existing `format_decode_tag` output, leading underscore stripped.
- `{user_suffix}` = `decoding.save_suffix` value if set.

Dropped tokens (now encoded by parent directory):
- `_<volume_stem>_` prefix (encoded by `<volume_stem>/` subdir).
- `_ckpt-<step>` (encoded by `results_step=<step>/` parent).

This answers review-finding 2 and locks in **`_x{n}` is preserved on
decoded outputs** so reruns with different TTA configs don't overwrite.

### F. Per-mode directory layout

#### Train mode

```
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  config.yaml
  checkpoints/
    epoch=NNN-...ckpt
    step=NNNNNNNN.ckpt
```

`<experiment>` = parent dir of `cfg.monitor.checkpoint.dirpath` (the
existing convention). The top-level `cfg.experiment_name` is metadata
only and does **not** override the dirpath. Documented as such in the
schema docstring; no new derivation logic.

#### Test mode

```
<output_base>/results_step=<NNN>/
  <volume_stem>/
    raw_x1_ch0-1-2.h5
    decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75.h5
    decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75_post.h5
    eval.txt
    eval_nerl.npz
    err_analysis/
  affinity_mask.h5                   # cross-volume; stays at results root
  decode_experiments.tsv             # cross-volume log
```

`<output_base>` derived from checkpoint path (existing
`get_output_base_from_checkpoint`). `<NNN>` from
`extract_step_from_checkpoint`.

#### Tune mode

```
<output_base>/tuning_step=<NNN>/
  predictions/
    <volume_stem>/
      tta_x1_ch0-1-2.h5              # cached intermediate prediction
  study.db
  best_params.yaml
  trials/                            # save_all_trials=true
    trial_NNN/
      <volume_stem>/
        decoded_x1_ch_..._<decoder>_<kwargs>.h5
```

### G. Cache-resolver contract (locked)

Updated functions in `connectomics/runtime/cache_resolver.py`:

- `has_cached_predictions_in_output_dir(cfg, mode, checkpoint_path)`:
  1. Compute expected stems via `resolve_dataset_volume_stems(cfg, mode)`.
  2. For each stem, check `<output_path>/<stem>/<expected_filename>`.
  3. Return `True` only when all expected files exist (all-or-nothing).
- `_try_cache_only_intermediate_eval`: same enumeration; iterate stems.
- `_resolve_first_complete_tuning_prediction_cache`: enumerate stems
  from `tune.data.val` config; iterate `predictions/<stem>/<filename>`.

Partial-hit policy: cache miss for the whole dataset (if one stem
misses, re-run inference). This avoids mismatched mixing of cached and
fresh outputs across volumes.

This answers review-finding 6.

### H. Code changes (concrete file list)

1. **`connectomics/config/schema/data.py`**: add `name: Optional[Union[str, List[str]]] = None` to `TestDataConfig` and `ValDataConfig`.
2. **`connectomics/config/schema/inference.py`**:
   - Rename fields per Section B.
   - Delete `chunks` and `write_mode`.
   - Update docstrings and inline comments referencing the dead fields and old names.
3. **`connectomics/config/schema/decoding.py`**: rename per Section B.
4. **`connectomics/config/schema/stages.py`**: delete `TuneOutputConfig`; hoist its fields to `TuneConfig`.
5. **`connectomics/config/schema/__init__.py`**: drop `TuneOutputConfig` import + `__all__` entry.
6. **`connectomics/config/pipeline/config_io.py`**: extend `_INFERENCE_RUNTIME_ALIAS_REPLACEMENTS` and add a `_DECODING_RUNTIME_ALIAS_REPLACEMENTS` and `_TUNE_RUNTIME_ALIAS_REPLACEMENTS` for the renames; reject the entire `tune.output:` sub-block with hints.
7. **`connectomics/runtime/output_naming.py`**:
   - Add `resolve_dataset_volume_stems(cfg, mode) -> List[str]`.
   - Add `resolve_volume_save_dir(cfg, mode, volume_stem) -> Path` joining `<save_path>/<volume_stem>`.
   - Refactor `final_prediction_output_tag`, `intermediate_decode_step_output_tag`, `tta_cache_suffix`, `intermediate_prediction_cache_suffix`, `resolve_prediction_cache_suffix`, `is_tta_cache_suffix` to produce the Section E canonical filenames (drop `_<stem>_` and `_ckpt-` prefixes; gain leading type-word `raw_`/`decoded_`).
8. **`connectomics/runtime/checkpoint_dispatch.py`**: write to renamed fields (`cfg.inference.save_path`, `cfg.tune.save_path`, `cfg.tune.save_predictions_path`).
9. **`connectomics/runtime/cli.py`, `dispatch.py`, `tune_runner.py`, `cache_resolver.py`, `output_naming.py`**: update all reads to the renamed fields; integrate per-volume subdir join; update cache enumeration per Section G.
10. **`connectomics/inference/output.py`**: read `inference.save_path` / `save_dtype` / `save_backend` / `save_compression`; join per-volume subdir in `write_outputs`.
11. **`connectomics/inference/{chunked,chunk_grid,stage}.py`**: read renamed fields.
12. **`connectomics/decoding/output.py`**: read `decoding.save_path` / fallback to `inference.save_path`; join per-volume subdir; gate on `decoding.save_results`. Update header comment to reference `inference.save_dtype`.
13. **`connectomics/decoding/streamed_chunked.py`**: read `inference.save_compression`.
14. **`connectomics/training/lightning/{model,test_pipeline}.py`**: rename reads; the per-volume subdir-join is via the new helper; per-step gating reads `decoding.save_intermediate` (unchanged) but skips emit when `len(decoding.steps) == 1`.
15. **`connectomics/evaluation/{context,report,nerl}.py`**: per-volume subdir join for `eval.txt` / `eval_*.npz`. Read `inference.save_path` (renamed).
16. **`scripts/validate_tutorial_configs.py`**: extend `LEGACY_PATTERNS` for the full rename surface (every renamed key) and `tune.output.*` rejection.
17. **Tests** — see Section I.
18. **YAMLs** — see Section J.

### I. Tests

- Update existing tests to use the renamed fields:
  `tests/unit/test_hydra_config.py`, `test_main_runtime_stage_switch.py`,
  `test_inference_stage.py`, `test_prediction_transform.py`,
  `test_optuna_tuner.py`, `test_connectomics_module.py`,
  `test_lit_utils.py`, `test_test_pipeline_multi_volume_eval.py`.
- Update strict-config rejection tests to expect new error messages.
- Update `tests/unit/test_public_api_snapshot.py` to drop
  `TuneOutputConfig` (review-finding 5 fallout).
- New `tests/unit/test_output_path_resolution.py`:
  - Train: `setup_run_directory` returns timestamped dir.
  - Test: `configure_checkpoint_output_paths` writes
    `cfg.inference.save_path == <base>/results_step=<NNN>` for
    `--checkpoint <base>/checkpoints/step=00200000.ckpt`.
  - Tune: `cfg.tune.save_path == <base>/tuning_step=<NNN>` and
    `cfg.tune.save_predictions_path == .../predictions`.
  - `resolve_dataset_volume_stems` cases:
    - explicit `data.test.name = "seed101"` → `["seed101"]`.
    - explicit list `data.test.name = ["a", "b"]` → `["a", "b"]`.
    - glob `seed*/data.zarr/img` with no `name` → parent-dir fallback
      (`seed0`, `seed1`, …) since stem `img` is uninformative.
    - flat `path.h5` → file stem.
    - decode-only via `load_prediction_path = ".../seed101/raw_x1.h5"`
      → `"seed101"`.
  - `resolve_volume_save_dir(cfg, "test", "seed101")`
    `== <base>/results_step=<NNN>/seed101`.
  - `final_prediction_output_tag` returns
    `decoded_x1_ch0-1-2_affinity_cc_numba-0-0.75` (no `_ckpt-`,
    no stem, leading type-word).
- New unit test for cache-resolver per-volume enumeration:
  - All stems' caches present → cache hit.
  - One stem missing → cache miss for whole dataset.

### J. YAML migration (mechanical)

Programmatic pass over 28 tutorials + 2 system YAMLs + 1 test YAML
applying the rename table from Section B and the `tune.output:` flatten
from the same section. Tutorials may need new `data.test.name` entries
where the auto-resolver would yield duplicate stems; validator warning
guides this. The migration script is a Python pass mirroring the prior
`save:` flatten.

The `tutorials/waterz_decoding_large*.yaml` files use the custom
`large_decode:` / `abiss_large:` top-level keys (already excluded from
`Config` validation by `CUSTOM_WORKFLOW_ROOTS`). Audit confirms only
`tutorials/waterz_decoding_large.yaml` declares `output_path` — it's a
**custom workflow's own field**, not the `inference.output_path` we are
renaming. No migration needed; document this in the strict-key rejector
to avoid confusion.

### K. Strict-key rejector + validator

`config_io.py` and `scripts/validate_tutorial_configs.py` jointly reject
the legacy keys with hints pointing at the renamed sibling:

| Legacy YAML path                         | Hint                                                    |
|---|---|
| `inference.output_path`                  | Use `inference.save_path`                               |
| `inference.cache_suffix`                 | Use `inference.save_cache_suffix`                       |
| `inference.dtype`                        | Use `inference.save_dtype`                              |
| `inference.backend`                      | Use `inference.save_backend`                            |
| `inference.compression`                  | Use `inference.save_compression`                        |
| `inference.chunks`                       | Field deleted (was unused)                              |
| `inference.write_mode`                   | Field deleted (was unused)                              |
| `inference.tta_result_path`              | Use `inference.load_tta_path`                           |
| `decoding.output_path`                   | Use `decoding.save_path`                                |
| `decoding.output_suffix`                 | Use `decoding.save_suffix`                              |
| `decoding.input_prediction_path`         | Use `decoding.load_prediction_path`                     |
| `tune.output`                            | Hoist fields to `tune.save_*` siblings                  |
| `tune.output.output_dir`                 | Use `tune.save_path`                                    |
| `tune.output.output_pred`                | Use `tune.save_predictions_path`                        |
| `tune.output.cache_suffix`               | Use `tune.save_cache_suffix`                            |
| `tune.output.save_all_trials`            | Use `tune.save_all_trials`                              |
| `tune.output.save_best_segmentation`     | Use `tune.save_best_segmentation`                       |
| `tune.output.save_study`                 | Use `tune.save_study`                                   |
| `tune.output.visualizations`             | Use `tune.save_visualizations`                          |
| `tune.output.report`                     | Use `tune.save_report`                                  |

## Files and Areas

Schema:
- `connectomics/config/schema/inference.py`
- `connectomics/config/schema/decoding.py`
- `connectomics/config/schema/stages.py`
- `connectomics/config/schema/data.py` (add `name`)
- `connectomics/config/schema/__init__.py` (drop `TuneOutputConfig`)

Pipeline:
- `connectomics/config/pipeline/config_io.py`

Runtime:
- `connectomics/runtime/output_naming.py`
- `connectomics/runtime/checkpoint_dispatch.py`
- `connectomics/runtime/cli.py`
- `connectomics/runtime/dispatch.py`
- `connectomics/runtime/cache_resolver.py`
- `connectomics/runtime/tune_runner.py`

Inference / decoding:
- `connectomics/inference/output.py`
- `connectomics/inference/chunked.py`
- `connectomics/inference/chunk_grid.py`
- `connectomics/inference/stage.py`
- `connectomics/decoding/output.py`
- `connectomics/decoding/streamed_chunked.py`
- `connectomics/decoding/stage.py`

Lightning glue:
- `connectomics/training/lightning/model.py`
- `connectomics/training/lightning/test_pipeline.py`
- `connectomics/training/lightning/data_factory.py` (reads `decoding.input_prediction_path`)

Evaluation:
- `connectomics/evaluation/context.py`
- `connectomics/evaluation/report.py`
- `connectomics/evaluation/nerl.py`

Tests:
- `tests/unit/test_hydra_config.py`
- `tests/unit/test_main_runtime_stage_switch.py`
- `tests/unit/test_inference_stage.py`
- `tests/unit/test_prediction_transform.py`
- `tests/unit/test_optuna_tuner.py`
- `tests/unit/test_connectomics_module.py`
- `tests/unit/test_lit_utils.py`
- `tests/unit/test_test_pipeline_multi_volume_eval.py`
- `tests/unit/test_decoding_pipeline.py` (verify; no field changes expected)
- `tests/unit/test_public_api_snapshot.py` (drop `TuneOutputConfig`)
- New: `tests/unit/test_output_path_resolution.py`

Validator:
- `scripts/validate_tutorial_configs.py`

YAMLs:
- 28 tutorials + `connectomics/config/profiles/pipeline_profiles.yaml`
  + `connectomics/config/templates/decoding_templates.yaml`
  + `tests/inference/test_nisb/base.yaml`.

## Verification Plan

1. **Static guardrails + public API**:
   ```
   python -m pytest tests/unit/test_v3_guardrails.py \
                    tests/unit/test_v2_boundaries.py \
                    tests/unit/test_public_api_snapshot.py -q
   ```
2. **Strict config + hydra**:
   ```
   python -m pytest tests/unit/test_hydra_config.py \
                    tests/unit/test_main_runtime_stage_switch.py \
                    tests/unit/test_inference_stage.py \
                    tests/unit/test_prediction_transform.py \
                    tests/unit/test_optuna_tuner.py \
                    tests/unit/test_connectomics_module.py \
                    tests/unit/test_lit_utils.py \
                    tests/unit/test_test_pipeline_multi_volume_eval.py -q
   ```
3. **New per-mode/per-volume path test**:
   ```
   python -m pytest tests/unit/test_output_path_resolution.py -q
   ```
4. **Tutorial validator**:
   ```
   python scripts/validate_tutorial_configs.py
   ```
5. **Full unit suite**:
   ```
   python -m pytest tests/unit/ -q
   ```
6. **Tutorial load smoke**:
   ```
   python -c "
   from connectomics.config import load_config
   from pathlib import Path
   for p in sorted(Path('tutorials').rglob('*.yaml')):
       if p.name.startswith('waterz_decoding_large'): continue
       cfg = load_config(str(p)); print(p.name, 'OK')
   "
   ```
7. **Synthetic path-derivation smoke** (new — addresses review-minor 12):
   ```
   python -c "
   import argparse
   from connectomics.config import Config
   from connectomics.runtime.checkpoint_dispatch import configure_checkpoint_output_paths
   cfg = Config()
   args = argparse.Namespace(mode='test', checkpoint='outputs/exp/20260101_000000/checkpoints/step=00200000.ckpt')
   configure_checkpoint_output_paths(args, cfg)
   assert cfg.inference.save_path.endswith('results_step=00200000'), cfg.inference.save_path
   print('OK', cfg.inference.save_path)
   "
   ```
8. **Manual regression spot-check** (not CI):
   ```
   python scripts/main.py --config tutorials/neuron_nisb/base_banis_crop.yaml \
        --mode test --checkpoint <ckpt> --fast-dev-run
   ```
   Verify `<base>/results_step=<NNN>/<volume_stem>/{raw_x1.h5,
   decoded_x1_*.h5, eval.txt}` appears.

## Risks and Questions

### Risks

- **External tooling breakage**: same as v0 — flat-globs in downstream
  scripts will miss the new layout. Document in the PR description.
  Note `scripts/tools/eval_curvilinear.py` is a candidate to audit
  during code stage; it currently reads files from
  `<results>/<glob>_decoding_*.h5`.
- **Public-API surface change**: removing `TuneOutputConfig` is a small
  break for any external consumer importing it directly. Acceptable per
  V3 contract; called out in the rejector hint.
- **`data.{test,val}.name` is opt-in**: existing tutorials with
  `path: seed*/data.zarr/img` rely on the parent-dir fallback. Any
  tutorial whose images all sit under indistinguishable parents will
  hit the validator warning and need to add `name` explicitly. Plan to
  pre-emptively add `name` to NISB tutorials in this same PR to avoid
  breaking smoke tests.
- **Filename `_x{n}` retention on decoded outputs** means a rerun with
  TTA flipped on/off produces different filenames (no overwrite). Good
  for safety, but it leaves stale files when iterating. Document this
  in the schema docstring.

### Open questions (none blocking)

- **`decoding.save_suffix` vs `decoding.save_filename_suffix`**: the v0
  question — kept short name for namespace consistency. Lock in
  `decoding.save_suffix`. (Resolved.)
- **`decoded_step{idx}_*.h5` when N=1**: elide. (Resolved.)
- **`affinity_mask.h5` placement**: v0 question (a) vs (b) — keep at
  results root, NOT under volume subdir, because the configured
  `decoding.affinity_mask_path` is a single file (cross-volume). If
  per-volume masks become necessary later, they'd need a different
  config shape. (Resolved.)

## Changes Since Previous Plan Version

Address all five major findings and the seven question items in
`plan_v0_review.md`:

- **Finding 1 (naming rule scope)**: rule narrowed to storage-/load-
  prefix only; non-storage leaves explicitly exempt with rationale.
  See Section A.
- **Finding 2 (filename token contradiction)**: single canonical
  filename table covers raw, decoded final, decoded postprocessed,
  decoded intermediate (with N=1 elision), and eval files. `_x{n}`
  is preserved on decoded outputs. See Section E.
- **Finding 3 (volume subdir under-specified for decode-only / cache
  paths)**: explicit fallback chain added including a decode-only
  branch that uses the load-path parent-dir name. See Section D.
  Cache-resolver contract spelled out in Section G.
- **Finding 4 (generic `load_path` ambiguity)**: renamed to specific
  artifact nouns: `inference.load_tta_path`,
  `decoding.load_prediction_path`. See Section B.
- **Finding 5 (tune dataclass decision)**: `TuneOutputConfig` deleted;
  fields hoisted to `TuneConfig`; public API snapshot updated; rejector
  raises on `tune.output.*`. See Section B and Section H step 5.
- **Q1 narrowed-rule**: storage/load-only.
- **Q2 decoded `_x{n}` token**: preserved.
- **Q3 decode-only volume dir**: load-path parent dir.
- **Q4 specific `load_*` nouns**: adopted.
- **v0 minor 5 dead-field audit**: confirmed `inference.chunks` and
  `inference.write_mode` unused; deleted in this PR rather than renamed.
  See Section B and Section K hints.
- **v0 minor 9 step-elision**: documented N=1 elision in Section E.
- **v0 minor 12 path-derivation smoke**: added Verification step 7.
- **v0 minor 14 public-API snapshot**: explicitly listed in Section H/I.
