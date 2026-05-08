# Plan v0

## Summary

Redesign two related concerns:

1. **Naming convention** for Hydra config keys so leaves and sub-blocks
   in stage sections (`inference`, `decoding`, `evaluation`, `tune`)
   read consistently. Adopt a shared `save_*` prefix for all
   storage-related scalar leaves at section root, mirroring the
   recently-flattened `inference.save_results`. Section root keeps both
   nested dataclass blocks (e.g. `inference.model`, `inference.window`)
   and these prefix-grouped scalar leaves; all other leaves either move
   inside an obvious block or drop their redundant `output_*` prefix.

2. **`output_path` derivation per stage mode**:
   - **Train**: keep timestamp-rooted construction; document its rule.
   - **Test / tune**: derive output base from `--checkpoint` (existing),
     then build a per-checkpoint, per-input-dataset directory tree.
     Move the dataset stem out of filenames into a subfolder; keep
     filenames short and identity-from-path.

The two are coupled because the `save_*` prefix carries through to the
new `*_path` leaves (`inference.save_path`, `decoding.save_path`) which
the new derivation rule writes into.

## Scope

In scope:

- Schema dataclasses: `InferenceConfig`, `DecodingConfig`,
  `EvaluationConfig`, `TuneConfig`, `TuneOutputConfig`.
- Strict-key rejector (`config_io.py`) + tutorial validator
  (`scripts/validate_tutorial_configs.py`).
- Output-naming helpers in `connectomics/runtime/output_naming.py` and
  the resolver in `connectomics/runtime/checkpoint_dispatch.py`.
- Output writers in
  `connectomics/inference/output.py`,
  `connectomics/decoding/output.py`,
  `connectomics/decoding/streamed_chunked.py`,
  `connectomics/inference/{chunked,chunk_grid,stage}.py`.
- Every consumer that reads `cfg.inference.{output_path,…}` or
  `cfg.decoding.{output_path,output_suffix,…}`.
- Tutorials, profile/template YAMLs, plus `tests/inference/test_nisb/base.yaml`.
- Affected tests.

Out of scope:

- The skeleton of the dataclass hierarchy (no new dataclasses, no
  un-flattening of the prior `save_results` flatten).
- Pipeline semantics — decoder chaining, postprocessing rules, TTA
  aggregation, evaluation metrics. This change is naming + path layout
  only.
- Backward compatibility shims. Per V3 contract, legacy keys raise.

## Proposed Changes

### A. Naming convention rule

**Rule (applied uniformly to all stage sections):**

> Section root holds two kinds of children:
>
> 1. Nested dataclass blocks named after the *concept* they represent
>    (`inference.model`, `inference.window`, `decoding.postprocessing`, …).
> 2. Scalar leaves that are siblings of an action verb. Verb is the
>    prefix; the noun is the suffix. Examples:
>    - `save_results: bool` is the verb-noun.
>    - `save_path: str`, `save_dtype`, `save_backend`, `save_compression`,
>      `save_chunks`, `save_cache_suffix`, `save_all_heads`, `save_write_mode`
>      are siblings under the same `save_*` action group.
>    - `load_path: str` (input prediction path) is a sibling under the
>      `load_*` action group.

This means: a leaf at section root is always one of `<verb>` or
`<verb>_<noun>`. It never names a free-floating noun (`output_path`,
`backend`, `dtype`, `compression`).

A reader scanning the YAML sees `save_*` and immediately knows the
field controls output storage; sees `load_*` and knows it controls
input loading; sees a key without underscore and knows it's a nested
block.

### B. Schema before/after

**`inference` section** (`connectomics/config/schema/inference.py:InferenceConfig`):

| Before                              | After                                |
|---|---|
| `inference.save_results: bool`      | `inference.save_results: bool` (unchanged) |
| `inference.output_path: str`        | `inference.save_path: str`           |
| `inference.cache_suffix: str`       | `inference.save_cache_suffix: str`   |
| `inference.dtype: Optional[str]`    | `inference.save_dtype: Optional[str]` |
| `inference.backend: str`            | `inference.save_backend: str`        |
| `inference.compression: Optional[str]` | `inference.save_compression: Optional[str]` |
| `inference.chunks: Optional[List[int]]` | `inference.save_chunks: Optional[List[int]]` |
| `inference.write_mode: str`         | `inference.save_write_mode: str`     |
| `inference.save_all_heads: bool`    | `inference.save_all_heads: bool` (unchanged) |
| `inference.tta_result_path: str`    | `inference.load_path: str` (input prediction file; symmetric with `save_path`) |

`inference.{model,execution,window,chunking,test_time_augmentation,prediction_transform,memory_cleanup,system}` and the runtime aliases (`head`, `select_channel`, `crop_pad`, `sliding_window`, `strategy`, `do_eval`) remain unchanged.

**`decoding` section** (`connectomics/config/schema/decoding.py:DecodingConfig`):

| Before                              | After                                |
|---|---|
| `decoding.enabled: bool`            | `decoding.enabled: bool` (unchanged) |
| `decoding.save_intermediate: bool`  | `decoding.save_intermediate: bool` (unchanged) |
| `decoding.save_results: bool`       | `decoding.save_results: bool` (unchanged) |
| `decoding.output_path: str`         | `decoding.save_path: str`            |
| `decoding.output_suffix: str`       | `decoding.save_suffix: str`          |
| `decoding.input_prediction_path: str` | `decoding.load_path: str`          |
| `decoding.affinity_mask_path: str`  | `decoding.affinity_mask_path: str` (unchanged — semantically a sibling input artifact, not a save/load action) |
| `decoding.steps: List[DecodeModeConfig]` | unchanged                       |
| `decoding.postprocessing: PostprocessingConfig` | unchanged                |
| `decoding.tuning: Optional[DecodingTuningConfig]` | unchanged              |

**`evaluation` section** (`connectomics/config/schema/evaluation.py:EvaluationConfig`):
no leaves require renaming; this section already reads as
`enabled / metrics / nerl_*`. No-op.

**`tune` section** (`connectomics/config/schema/stages.py:TuneOutputConfig` & `TuneConfig`):

| Before                                 | After                                |
|---|---|
| `tune.output.output_dir: Optional[str]` | `tune.save_path: Optional[str]` (study/best-params dir) |
| `tune.output.output_pred: Optional[str]`| `tune.save_predictions_path: Optional[str]` |
| `tune.output.cache_suffix: str`         | `tune.save_cache_suffix: str`        |
| `tune.output.save_all_trials: bool`     | `tune.save_all_trials: bool`         |
| `tune.output.save_best_segmentation: bool` | `tune.save_best_segmentation: bool` |
| `tune.output.save_study: bool`          | `tune.save_study: bool`              |
| `tune.output.visualizations`            | `tune.save_visualizations`           |
| `tune.output.report`                    | `tune.save_report`                   |

The `tune.output` sub-block disappears; its fields hoist to root with
the `save_*` prefix matching `inference.save_*`. This drops the awkward
`tune.output.output_*` double-noun pattern. Other `tune` leaves stay as
they are (they're tuner-engine controls, not storage).

### C. `output_path` (now `save_path`) derivation per mode

#### Train mode

Unchanged in spirit; just documented and renamed.

```
outputs/<experiment_name>/<YYYYMMDD_HHMMSS>/
  config.yaml
  checkpoints/
    epoch=NNN-...ckpt
    step=NNNNNNNN.ckpt
  results/                 # populated only if test runs after training in same run dir
  tuning/                  # populated only if tune runs after training in same run dir
```

`outputs/<experiment_name>/` is the configured `monitor.checkpoint.dirpath`'s parent (`cfg.monitor.checkpoint.dirpath = outputs/<experiment_name>/checkpoints/`). Train-time runtime creates `<timestamp>/checkpoints/` and assigns `cfg.monitor.checkpoint.dirpath = run_dir/checkpoints` so subsequent writes land under the timestamped run dir. This is the existing `setup_run_directory` behavior.

#### Test mode

Existing rule:
- `output_base = walk_up_to_timestamp(checkpoint_path)` (or `parent.parent` fallback).
- `cfg.inference.save_path = output_base / "results_step=<NNNN>"`.

**New rule** (additive — only changes filename layout, not directory derivation):

For each test volume, construct a per-volume subdirectory under `inference.save_path`:

```
<output_base>/results_step=00200000/
  <volume_stem>/
    raw_x1.h5                               # raw prediction (was {stem}_tta_x1_ch_..._ckpt-..._prediction.h5)
    raw_x1_ch0-1-2.h5                       # if channel selector active
    decoded_affinity_cc_numba-0-0.75.h5     # final decoded (was {stem}_x1_ch..._ckpt-..._decoding_affinity_cc_numba-0-0.75.h5)
    decoded_affinity_cc_numba-0-0.75_postprocessed.h5  # only if postprocessing.enabled=true
    decoded_step0_affinity_cc_numba-0-0.75.h5          # save_intermediate=true; one file per step
    eval.txt                                # was evaluation_metrics_{long_name}.txt
    eval_per_gt_erl.npz                     # was evaluation_metrics_{long_name}_nerl_per_gt_erl.npz
    err_analysis/                           # was err_analysis/ at results root
  affinity_mask.h5                          # cross-volume asset (stays at results root)
  decode_experiments.tsv                    # cross-volume log (stays at results root)
```

**Filename-token policy** (inside per-volume subdir):
- Drop `<volume_stem>_` prefix (encoded by parent dir).
- Drop `_ckpt-step=<N>` token (encoded by parent `results_step=<N>/`).
- Keep `_x{n}` (TTA pass count varies within one volume between runs).
- Keep `_ch{...}` (channel selector varies within one volume).
- Keep decoder-name + kwargs token (`affinity_cc_numba-0-0.75`).
- Keep optional user `decoding.save_suffix` tail.
- Use a leading **type word** instead of stem-prefixed naming: `raw_…`, `decoded_…`, `eval…`. This makes ls'ing a per-volume folder self-describing.

**Volume stem resolution**: `Path(image_filename).stem`, with the same fallback chain as `resolve_output_filenames` today. For lazy / dataset-name flows where the stem is e.g. `img` (from `data.zarr/img`), prefer `data.test.path`'s leaf when it disambiguates (`seed101/data.zarr/img` → `seed101`). Single rule documented in code: take the first non-empty of [`Path(meta.filename_or_obj).stem`, `Path(image_path).parent.name if stem == "img"`, fallback `volume_<idx>`].

#### Tune mode

Existing rule: `tuning_step=<N>/predictions/` and `tuning_step=<N>/` for study.

**New rule** (additive — same per-volume subfolder split):

```
<output_base>/tuning_step=00200000/
  predictions/
    <volume_stem>/
      tta_x1.h5                             # the cached intermediate prediction
  study.db
  best_params.yaml
  trials/                                   # save_all_trials=true
    trial_NNN/
      <volume_stem>/
        decoded_<decoder>_<kwargs>.h5
```

### D. Code changes

1. **`connectomics/runtime/output_naming.py`**
   - Add `format_volume_subdir(meta_or_path) -> str`.
   - Refactor `final_prediction_output_tag` to drop `_ckpt-...` and the
     volume-stem prefix; produce `decoded_<decoder>_<kwargs>` (or
     `prediction`) prefix instead. Move `_ckpt-...` derivation to the
     directory level (handled by `checkpoint_dispatch`).
   - Refactor `intermediate_decode_step_output_tag` similarly: produce
     `decoded_step{idx}_<step_tag>`.
   - Refactor `intermediate_prediction_cache_suffix` and `tta_cache_suffix`
     to produce `raw_x{n}{head}{ch}.h5` (no ckpt token, no stem).
   - `resolve_prediction_cache_suffix` and `is_tta_cache_suffix` updated
     to match new short suffixes.
   - Add `resolve_volume_save_dir(cfg, mode, volume_stem) -> Path`
     that combines `cfg.inference.save_path` with the volume subdir.

2. **`connectomics/runtime/checkpoint_dispatch.py`**
   - `configure_checkpoint_output_paths` writes `cfg.inference.save_path`
     (renamed) and `cfg.tune.save_path` / `cfg.tune.save_predictions_path`
     (renamed).
   - No new logic — only rename.

3. **`connectomics/inference/output.py`**
   - `_resolve_mode_configs` reads `cfg.inference.save_path`.
   - `write_outputs` joins per-volume subdir before writing.
   - `apply_storage_dtype_transform` reads `cfg.inference.save_dtype`.

4. **`connectomics/decoding/output.py`**
   - `_resolve_decoded_output_dir` reads `cfg.decoding.save_path` then
     `cfg.inference.save_path` then parent of `cfg.decoding.load_path`.
   - `write_decoded_outputs` joins per-volume subdir.

5. **`connectomics/inference/{chunked,chunk_grid,stage}.py`**
   - Read `cfg.inference.save_compression` / `save_backend`.

6. **`connectomics/decoding/streamed_chunked.py`**
   - Read `cfg.inference.save_compression`.

7. **`connectomics/training/lightning/{model,test_pipeline}.py`** and
   **`connectomics/runtime/{cli,dispatch,cache_resolver,output_naming,tune_runner}.py`** and
   **`connectomics/evaluation/context.py`**
   - Update field reads.
   - Test pipeline's per-volume directory join goes through the new
     helper.
   - Evaluation report writer joins the per-volume subdir for `eval.txt`
     / `eval_*.npz`.

8. **`connectomics/config/pipeline/config_io.py`**
   - Extend `_INFERENCE_RUNTIME_ALIAS_REPLACEMENTS` to redirect:
     `output_path → save_path`, `dtype → save_dtype`,
     `backend → save_backend`, `cache_suffix → save_cache_suffix`,
     `compression → save_compression`, `chunks → save_chunks`,
     `write_mode → save_write_mode`, `tta_result_path → load_path`.
   - Add equivalent mapping under `decoding.*`:
     `output_path → save_path`, `output_suffix → save_suffix`,
     `input_prediction_path → load_path`.
   - Add mapping under `tune.*`: reject the entire `output:` sub-block
     with hint redirecting each field to its `save_*` flat sibling.

9. **`scripts/validate_tutorial_configs.py`**
   - Add the same legacy-pattern rejection messages.

### E. YAML migration

- 28 tutorials + profile/template YAMLs + `tests/inference/test_nisb/base.yaml`.
- For each: rename keys per the table in section B. The flatten of
  `tune.output:` requires hoisting up to 8 keys.
- Programmatic: a single regex/Python pass; per-file diff visible in PR.

### F. Strict-key rejector

After the rename, old YAML keys must raise on load with a hint pointing
to the new key. Reuse the same machinery added in
`config_io.py:_reject_inference_runtime_alias_paths`. The validator in
`scripts/validate_tutorial_configs.py` parallels it for CI.

### G. New defaults

Default values are preserved verbatim under their renamed keys. The
new `inference.save_results: bool = False` and
`decoding.save_results: bool = True` semantics from the previous refactor
do **not** change.

## Files and Areas

Schema:
- `connectomics/config/schema/inference.py`
- `connectomics/config/schema/decoding.py`
- `connectomics/config/schema/stages.py` (TuneOutputConfig hoist)
- `connectomics/config/schema/__init__.py` (drop `TuneOutputConfig` export if unused)

Pipeline (config loader):
- `connectomics/config/pipeline/config_io.py`

Runtime:
- `connectomics/runtime/output_naming.py`
- `connectomics/runtime/checkpoint_dispatch.py`
- `connectomics/runtime/cli.py`
- `connectomics/runtime/dispatch.py`
- `connectomics/runtime/cache_resolver.py`
- `connectomics/runtime/tune_runner.py`

Inference / decoding stage code:
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

Evaluation:
- `connectomics/evaluation/context.py`
- `connectomics/evaluation/report.py` (per-volume subdir join for eval files)
- `connectomics/evaluation/nerl.py` (already touched in working tree; verify the eval-output path)

Tests:
- `tests/unit/test_hydra_config.py`
- `tests/unit/test_main_runtime_stage_switch.py`
- `tests/unit/test_inference_stage.py`
- `tests/unit/test_prediction_transform.py`
- `tests/unit/test_optuna_tuner.py`
- `tests/unit/test_connectomics_module.py`
- `tests/unit/test_lit_utils.py`
- `tests/unit/test_test_pipeline_multi_volume_eval.py`
- `tests/unit/test_decoding_pipeline.py` (no field changes, but check)
- New: `tests/unit/test_output_path_resolution.py` covering
  per-mode/per-volume subdir construction and the new filename format.

YAMLs:
- 28 files under `tutorials/` + `connectomics/config/profiles/pipeline_profiles.yaml`
  + `connectomics/config/templates/decoding_templates.yaml`
  + `tests/inference/test_nisb/base.yaml`.

Validator:
- `scripts/validate_tutorial_configs.py`

Documentation strings: comment headers in `inference.py`, `decoding.py`,
`stages.py`, `runtime/checkpoint_dispatch.py` to record the new
convention.

## Verification Plan

1. **Static guardrails** — must pass:
   ```
   python -m pytest tests/unit/test_v3_guardrails.py \
                    tests/unit/test_v2_boundaries.py \
                    tests/unit/test_public_api_snapshot.py -q
   ```
2. **Strict config + hydra** — must pass and exercise every renamed key:
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
3. **New unit test** for per-mode/per-volume path construction:
   - Train: `setup_run_directory` returns `<base>/<timestamp>/` with
     `checkpoints/` underneath.
   - Test: given `--checkpoint outputs/<exp>/<ts>/checkpoints/step=00200000.ckpt`,
     `cfg.inference.save_path == outputs/<exp>/<ts>/results_step=00200000`.
   - Tune: `cfg.tune.save_path == outputs/<exp>/<ts>/tuning_step=00200000`,
     `cfg.tune.save_predictions_path == .../predictions`.
   - Per-volume: `resolve_volume_save_dir(cfg, mode="test", "seed101")
     == outputs/<exp>/<ts>/results_step=00200000/seed101`.
   - Filename: `final_prediction_output_tag(...)` == `decoded_affinity_cc_numba-0-0.75`
     (no leading `x1`, no `_ckpt-`, no stem).
4. **Tutorial validator** — must pass:
   ```
   python scripts/validate_tutorial_configs.py
   ```
5. **Full unit suite** — must pass:
   ```
   python -m pytest tests/unit/ -q
   ```
6. **Smoke test** — load each tutorial via `Config()`:
   ```
   python -c "
   from connectomics.config import load_config
   from pathlib import Path
   for p in sorted(Path('tutorials').rglob('*.yaml')):
       if p.name.startswith('waterz_decoding_large'): continue
       cfg = load_config(str(p))
       print(p, 'OK')
   "
   ```
7. **Regression spot-check** (manual; not part of CI):
   `python scripts/main.py --config tutorials/neuron_nisb/base_banis_crop.yaml
        --mode test --checkpoint <existing> --fast-dev-run`
   and verify the new directory tree appears under
   `<base>/results_step=<N>/<volume_stem>/`.

## Risks and Questions

### Risks

- **Filesystem collisions on rerun**: today, rerunning a test with
  different decoder kwargs writes a new file in the flat results dir.
  After this change, the new filename
  `decoded_affinity_cc_numba-0-0.75.h5` lives in `<volume_stem>/`. Two
  different decoder configs still produce two filenames thanks to
  kwargs in the tag, so no collision. But two reruns with identical
  decoder kwargs overwrite (already true today).
- **External tooling breakage**: any downstream script or notebook
  globbing flat `results_step=*/img_*_decoding_*.h5` will miss the
  new layout. There is no shim per V3 contract; mention this in the
  PR description.
- **`affinity_mask.h5` placement**: currently lives at the flat
  `results_step=<N>/` root, alongside per-volume artifacts. Two reasonable
  policies: (a) keep it at the results root as a cross-volume asset; (b)
  move it under `<volume_stem>/` since the mask is dataset-specific in
  practice. Plan picks (a) because today's `decoding.affinity_mask_path`
  is configured per cfg, not per volume — moving it would require
  decode-time path templating. Open question if reviewer prefers (b).

### Questions for review

1. **`decoding.input_prediction_path` rename**: does
   `decoding.load_path` read clearly, or is `decoding.load_predictions_path`
   clearer about the artifact type? Symmetric with `inference.load_path`
   either way.
2. **`decoding.save_suffix` vs `decoding.save_filename_suffix`**: the
   field is purely a filename-level user tag (not a directory). The
   shorter name is consistent; the longer name is unambiguous. Stick
   with `save_suffix`?
3. **`inference.save_chunks` and `inference.save_write_mode`**: are these
   actively used anywhere? If dead, drop them in this same PR. (Will
   audit during code stage; flag if not used.)
4. **Volume stem fallback**: if a tutorial loads `data.zarr/img`, the
   resolved stem is `img` (uninformative). Plan proposes falling back to
   parent dir name when the stem is `img`. Reviewer should confirm this
   heuristic is acceptable, or specify a hard-coded mapping in YAML
   (e.g. a new `data.test.name` field).
5. **`tune.output` flatten**: the `output:` block today nests
   `visualizations: dict` and `report: dict` (free-form). Hoisting them
   to `tune.save_visualizations` / `tune.save_report` is fine since they
   are already opaque dicts, but reviewer should confirm none of them
   should retain nesting for OmegaConf merge semantics (lists replace,
   dicts merge).
6. **Eval file location**: the per-volume `eval.txt` and `eval_*.npz`
   moves. Any external workflow that grep'd these files at the flat root
   needs updating — confirm no internal CLI tool depends on this layout
   (`scripts/tools/eval_curvilinear.py` is a candidate to audit).

## Changes Since Previous Plan Version

Initial plan.
