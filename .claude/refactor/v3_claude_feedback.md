# Feedback on V3 Refactor Plan

Reviewed against the current tree after commit `ba0f482`. The plan has the
right architectural direction, but it should not be implemented exactly as
written. The main issue is that several "dead code" and "zero caller" claims
are too broad, while the ordering understates breakage risk.

## Executive Summary

The strongest parts of the plan are:

- Moving tuning orchestration out of `connectomics.decoding`.
- Moving evaluation logic out of Lightning.
- Moving runtime dispatch out of `scripts/main.py`.
- Enforcing strict config loading instead of warning on unknown keys.
- Removing ghost config reads such as `inference.test_time_augmentation.act`.
- Separating raw inference artifacts from decoding and evaluation concerns.

The weakest parts are:

- Theme A overclaims dead code. Some entries are public APIs, builder paths, or
  test-covered behavior rather than unreachable code.
- The proposed PR order is unsafe. Public API trimming, strict config, schema
  migration, and runtime extraction depend on each other.
- The file-size target is too mechanical. Splits should follow ownership and
  dependency direction first, not a hard line-count rule.
- The high-performance objective is not developed enough. Most items improve
  architecture, but few directly address runtime hot spots.

Bottom line: implement v3 as a boundary-first refactor with import-boundary
tests and small benchmarks, not as a broad deletion and file-split sweep.

## What The Plan Gets Right

### Decoding/Tuning Boundary

`connectomics/decoding/tuning/optuna_tuner.py` is still the worst boundary
violation. It lazy-imports naming helpers from `connectomics.training`:

- `tta_cache_suffix`
- `tuning_best_params_filename`
- `tuning_best_params_filename_candidates`
- `tuning_study_db_filename`

It also creates a datamodule and calls `trainer.test(...)` inside the decoding
package. That orchestration belongs in `connectomics.runtime`, while decoding
tuning should operate on arrays/artifacts and metric callables.

The proposed split into pure tuner modules plus `runtime/tune_runner.py` is the
right direction.

### Inference/Decoding Boundary

`connectomics/inference/chunked.py` currently contains a streamed affinity-CC
decode path. That means inference imports decoding and performs segmentation
inside an inference module. Moving `run_chunked_affinity_cc_inference` and its
decode-kwargs resolver into a decoding/runtime module is a clean boundary fix.

Keep `run_chunked_prediction_inference` in inference. It should remain raw
prediction only.

### Evaluation Boundary

`training/lightning/test_pipeline.py` owns too much evaluation logic. Moving
binary, instance, NERL, report writing, and metric aggregation into
`connectomics.evaluation` would reduce private `module._*` coupling and make
decode-only/evaluate-only workflows possible without Lightning.

This should be one of the highest-priority v3 changes.

### Strict Config

`config/pipeline/config_io.py::_warn_unconsumed_keys` still warns for unknown
top-level keys. With "no backward compatibility", unknown keys should raise.
This is a good mechanical cleanup and would catch drift earlier.

The ghost activation reads in `inference/tta.py` should also be removed:

- `cfg.inference.test_time_augmentation.act`
- `cfg.inference.output_act`

The canonical path is `inference.test_time_augmentation.channel_activations`.

## Corrections Needed Before Implementation

### Theme A Is Too Aggressive

Do not treat all listed items as safe deletion.

Examples:

- `RandMixupd` is not dead. It is imported by tests and referenced by the
  augmentation builder. Removing it is a behavior/API removal, not dead-code
  cleanup.
- `auto_plan_config`, `AutoConfigPlanner`, and `AutoPlanResult` are not dead.
  They are intentionally tested under `connectomics.config.hardware`. Removing
  them is a product decision, not a cleanup.
- `WandbConfig` may have no active logger consumer, but deleting the schema field
  still changes accepted config surface. That belongs in the schema-break PR,
  with tutorial/docs validation.
- `SaveVolumed` and `TileLoaderd` may have no production caller, but they are
  exported from `connectomics.data.io`. Removing exported names should be part
  of an API-trim PR, not mixed into dead-code deletion.
- `create_combined_loss`, `create_loss_from_config`, `CombinedLoss`, and
  `GANLoss` are exported through model/loss package APIs. If removed, update
  package exports and tests explicitly.
- `TestConfig.output_path` and `TestConfig.cache_suffix` are not obviously
  "never read". There is stage fallback behavior in `inference/output.py`, and
  tests currently set `cfg.test.output_path`.

The plan should split Theme A into three buckets:

1. Safe artifacts/dead modules: delete after grep and import checks.
2. Public API removals: delete only with export/test updates.
3. Product feature removals: require an explicit decision.

### PR Ordering Is Not Safe

The statement "PRs 1-7 are mergeable in any order" is not correct.

Safer dependencies:

- `runtime/output_naming.py` should land before splitting `optuna_tuner.py`,
  because it removes decoding's dependency on Lightning naming helpers.
- Public API trim should happen after target modules exist, not before.
- Strict config should avoid removing schema fields before tutorial migration.
- Schema split and tutorial migration must be one coordinated change.
- Evaluation extraction should happen before broad `test_pipeline.py` file
  splitting, because it removes the largest and most meaningful coupling first.

### File Size Should Be Secondary

"Every file < 600 lines" is a useful smell check, not a design rule. Splitting
large files without resolving ownership can create more import churn and weaker
locality.

Prioritize splits that reduce dependency direction problems:

- decoding no longer imports training
- inference no longer imports decoding
- config no longer imports data execution machinery
- scripts no longer own runtime orchestration
- evaluation no longer depends on Lightning module internals

After those are fixed, split remaining large files by stable concepts.

### Performance Work Needs More Specific Targets

The plan says "high-performance" but mostly proposes architecture cleanup. Add
concrete performance targets:

- Avoid unnecessary `tensor.cpu().float().numpy()` conversions where downstream
  operations can stay in torch or preserve dtype.
- Benchmark raw chunked inference write throughput before changing artifact
  writing paths.
- Verify HDF5 chunk shapes and compression settings for prediction artifacts.
- Reduce duplicated raw prediction read/write loops between standard and
  chunked inference.
- Measure TTA memory use and data movement, especially patch-first-local and
  distributed reductions.
- Avoid forcing every streamed path through an artifact abstraction if it hurts
  large-volume throughput.

## Recommended Revised Order

### PR 0: Verification Guardrails

Add tests before moving code:

- Import-boundary tests:
  - `connectomics.decoding` must not import `connectomics.training`
  - `connectomics.inference` must not import `connectomics.decoding`
  - `connectomics.config` must not import MONAI/data execution code
- Strict-config tests for unknown top-level and nested removed keys.
- Snapshot tests for public imports that are intentionally kept.
- Small chunked-inference artifact benchmark or smoke test.

### PR 1: Safe Dead Artifacts Only

Delete only files/symbols that are truly unreachable and not exported. For
example, `decoding/tuning/auto_tuning.py` looks like a strong candidate because
it imports `from .segmentation import decode_affinity_cc`, which is broken.

Required cleanup:

- Remove its re-exports from `connectomics/decoding/tuning/__init__.py`.
- Add an import test for `connectomics.decoding.tuning`.

Do not include public API removals in this PR.

### PR 2: Runtime Naming Extraction

Move cache suffix, checkpoint tag, output-head tag, and tuning filename helpers
from `training/lightning/utils.py` to `connectomics/runtime/output_naming.py`.

Then update `optuna_tuner.py`, `scripts/main.py`, `model.py`, and
`test_pipeline.py` imports.

This is the cleanest first boundary move because it removes lazy decoding to
training imports without changing tuning behavior yet.

### PR 3: Strict Config Pass

Make unknown top-level keys raise in `load_config`.

Delete verified ghost reads:

- `inference.test_time_augmentation.act`
- `inference.output_act`

Keep genuinely optional section-root access where sections may be `None`.

### PR 4: Evaluation Extraction

Move binary/instance/NERL metrics and report writing into
`connectomics.evaluation`.

Target layout:

```text
connectomics/evaluation/
  __init__.py
  stage.py
  metrics.py
  nerl.py
  report.py
  curvilinear.py
```

`training/lightning/test_pipeline.py` should call evaluation as a service, not
own the metric implementation.

### PR 5: Tuning Runtime Split

Move `run_tuning`, `load_and_apply_best_params`, and temporary inference
override orchestration into `connectomics/runtime/tune_runner.py`.

Keep `connectomics.decoding.tuning` focused on pure parameter search over saved
arrays/artifacts and metric callables.

### PR 6: Chunked Decode Boundary

Move streamed affinity-CC decode out of `connectomics.inference`.

Suggested target:

- `connectomics/decoding/streamed_chunked.py`, if it is a decoder concern
- or `connectomics/runtime/chunked_decode.py`, if it remains orchestration-heavy

`connectomics/inference/chunked.py` should only produce raw prediction
artifacts.

### PR 7: Runtime CLI Extraction

Move dispatch-heavy code out of `scripts/main.py` into runtime modules:

- `runtime/cli.py`
- `runtime/checkpoint_dispatch.py`
- `runtime/cache_resolver.py`
- `runtime/sharding.py`
- `runtime/torch_safe_globals.py`

Keep `scripts/main.py` as a thin entrypoint.

### PR 8: Schema Split And Tutorial Migration

Split decoding and evaluation config out of `config/schema/inference.py`.

Do this atomically with tutorial migration because it changes config keys:

- `inference.postprocessing.*` to `decoding.postprocessing.*`
- `inference.saved_prediction_path` to `decoding.input_prediction_path`
- `inference.decoding_path` to `decoding.output_path`
- evaluation config into its own schema module

Run tutorial config validation across all tutorial YAMLs in the same PR.

### PR 9: Public API Trim

Trim `__init__.py` exports after the new homes exist.

Do this with explicit import tests so the kept public surface is intentional.

### PR 10: File Splits

Split remaining large files only after ownership boundaries are fixed.

Focus on:

- `inference/tta.py`
- `inference/lazy.py`
- `data/augmentation/transforms.py`
- `data/processing/transforms.py`
- `training/lightning/callbacks.py`
- `config/pipeline/config_io.py`

### PR 11: Architecture Rename And Docs Refresh

Rename `nnunet_pretrained` to `nnunet` only with tutorial and docs migration.

Refresh stale docs in:

- `docs/source/tutorials/*.rst`
- `docs/source/notes/*.rst`
- `docs/source/modules/model.rst`
- `docs/source/modules/models.rst`

Delete duplicate docs only after confirming they are not linked by Sphinx
toctrees.

## Implementation Guardrails

- Every deletion must pass `rg` checks across `connectomics`, `scripts`,
  `tests`, `tutorials`, and `docs`.
- Public export removals need an explicit before/after import surface test.
- Schema changes need tutorial validation in the same commit.
- Boundary changes need import graph tests.
- Runtime changes need focused tests for cache-hit, tune, tune-test, and
  checkpoint-derived output paths.
- Performance-sensitive paths need at least smoke benchmarks before and after:
  chunked inference, TTA, lazy prediction, and artifact writes.

## Verdict

Use the v3 plan as a strong architectural roadmap, not as an execution spec.

The high-value sequence is:

1. Add guardrails.
2. Extract runtime naming.
3. Enforce strict config.
4. Move evaluation out of Lightning.
5. Move tuning orchestration out of decoding.
6. Move chunked decoding out of inference.
7. Split schema and migrate tutorials.
8. Trim public APIs.
9. Split large files by ownership.
10. Refresh docs.

This preserves the intended "no backward compatibility" cleanup while avoiding
accidental deletion of still-reachable APIs and avoiding review-hostile churn.
