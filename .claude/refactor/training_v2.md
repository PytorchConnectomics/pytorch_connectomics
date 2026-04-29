# Training V2 Plan

## Goal

Keep training focused on fitting models. Training may provide a test-mode
convenience wrapper, but it should delegate prediction, decoding, and evaluation
to their stage APIs instead of owning that logic.

Baseline: `training.md` reports good cleanup inside Lightning code. V2 focuses
on package boundaries and removal of compatibility import paths.

## Public API

Keep:

- Lightning module class used for model training
- training data module helpers
- callback classes
- optimizer/scheduler builders
- loss orchestration
- train entrypoint function
- debug hooks

Public names should be listed explicitly in `training.__init__` only if they are
intended for v2 callers.

## Delete

- Any old compatibility modules under `training.lightning`.
- Any test pipeline logic that directly decodes or evaluates instead of calling
  stage APIs.
- Any duplicate path utilities if they are only kept for old imports.
- Any `print()`-based operational output except interactive debugging hooks.
- Any private-method coupling between test pipeline and Lightning module where
  a context object or stage API can replace it.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Training loop | `connectomics.training.lightning.trainer` |
| Lightning module | `connectomics.training.lightning.model` |
| Train/test runtime dispatch | runtime/CLI layer or a small training runtime module |
| Optimizer/scheduler | `connectomics.training.optimization` |
| Loss orchestration | `connectomics.training.losses` |
| Debug hooks | `connectomics.training.debugging` |

If test orchestration remains in `training.lightning.test_pipeline`, it should
be a thin wrapper around `inference`, `decoding`, and `evaluation` stage calls.

## Config Contract

Training uses:
- `model`
- `data`
- `optimization`
- `monitor`
- `system`

Training does not own:
- decoder list;
- final evaluation metric reports;
- raw prediction artifact format;
- decode-after-inference behavior.

## Boundary Rules

- Training may import config, data, models, and metrics.
- Training may import inference only through a stage wrapper for test mode.
- Training should not import decoder internals.
- Training should not import evaluation internals except through
  `run_evaluation_stage`.
- Loss orchestration should avoid data-process coupling unless injected or
  routed through stable shared helpers.

## Implementation Order

1. Define the stage APIs for inference, decoding, and evaluation.
2. Refactor test pipeline to call those APIs.
3. Remove direct decoder/evaluation logic from training.
4. Delete compatibility import modules and update tests.
5. Keep `TestContext` or replace it with explicit stage inputs.
6. Re-run Lightning and runtime tests.

## Tests

Add or update:
- train mode does not require decoding config;
- test convenience mode calls inference, decoding, and evaluation stages in
  order;
- decode-only and evaluate-only paths do not construct a Lightning trainer;
- optimizer unknown names raise;
- deep supervision loss handling remains covered;
- debug hooks still print to stdout when entering interactive debugging.

## Open Decisions

- Whether `training.lightning.test_pipeline` should be deleted entirely after
  stage APIs exist.
- Whether train/test runtime dispatch should live in a new
  `connectomics.runtime` package.
