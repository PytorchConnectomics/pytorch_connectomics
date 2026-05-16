# Connectomics Folder Boundary and Import Structure

## Position

The package does not need fewer folders as a goal. It needs folders whose names
answer two questions clearly:

1. Who owns this concept?
2. Which direction may imports flow?

Mature PyTorch-family repositories are not small. The useful precedent is that
public namespaces map to stable concepts. PyTorch exposes component namespaces
such as `torch.nn`, `torch.autograd`, and `torch.utils`; TorchVision exposes
concept namespaces such as `datasets`, `io`, `models`, `ops`, and `transforms`.
For PyTorch Connectomics, the same lesson argues for concept-owned packages,
not a broad `deep_learning/` umbrella and not a merged application-shell bucket.

## Recommended Package Boundary

```text
connectomics/
  _core/        # tiny pure primitives; no internal dependencies
  config/       # schemas, YAML/profile expansion, config-only validation
  data/         # IO, datasets, transforms, preprocessing, target generation
  models/       # architectures, model factories, output contracts, model-side losses
  metrics/      # reusable scoring algorithms and torchmetrics wrappers
  training/     # Lightning modules, datamodules, callbacks, optimizers, online validation
  inference/    # model prediction, TTA, sliding-window and chunked prediction
  decoding/     # predictions -> segmentation/synapse artifacts and tuning
  evaluation/   # metric execution over artifacts, reports, logging, evaluation stage
  runtime/      # CLI setup, dispatch, preflight, cache/checkpoint orchestration
```

`scripts/` should stay outside the package and call `connectomics.runtime`.

## Import Direction

The target dependency graph should be a DAG:

```text
_core
config     -> _core only
data       -> _core
models     -> _core
metrics    -> _core
training   -> config, data, models, metrics
inference  -> config, data, models
decoding   -> config, data, metrics, _core
evaluation -> config, data, metrics
runtime    -> config, training, inference, decoding, evaluation
scripts    -> runtime
```

Two rules matter more than the exact folder names:

- Lower-level primitives must not import stage orchestration.
- Stage packages must not import `runtime`; runtime should pass decisions down.
- `metrics` should remain config-free; config-specific interpretation belongs
  in `training`, `evaluation`, or `runtime`, which can pass primitive arguments
  into metric functions.

## Metrics vs Evaluation

Keep `connectomics.metrics` separate from `connectomics.evaluation`.

`metrics` owns reusable primitives:

- Adapted Rand, VOI, instance matching
- skeleton and curvilinear pair metrics
- torchmetrics-compatible wrappers
- side-effect-free NERL scoring around `em_erl`

`evaluation` owns stage integration:

- `EvaluationContext`
- selecting requested metrics from config
- per-volume metric execution
- report files and experiment-log integration
- Lightning/test logging glue
- `run_evaluation_stage`

Putting `metrics/` under `evaluation/` would force `training` and `decoding`
to import the evaluation stage for primitive scoring. That is backwards.
Training legitimately needs online validation metrics such as `val_dice`,
`val_jaccard`, and `val_accuracy`; decoding tuning needs objective functions.
Neither should depend on report-writing or evaluation-stage orchestration.

## NERL Ownership

`lib/em_erl` should remain the algorithm provider: graph structures, LUT
sampling, ERL scoring, and low-level IO helpers.

PyTC still needs adapter code, but that adapter should be split:

```text
connectomics/metrics/nerl.py
  side-effect-free scoring:
    load/normalize ERLGraph inputs where needed
    prepare segmentation arrays
    compute (nerl, pred_erl, gt_erl)

connectomics/evaluation/nerl.py
  evaluation-stage integration:
    read PyTC config fields
    select per-volume skeleton and mask paths
    populate metrics_dict keys
    write/log test_nerl and per-GT ERL artifacts
```

The current smell is that `connectomics/decoding/tuning/optuna_tuner.py`
imports `connectomics.evaluation.nerl.compute_nerl_score`. That makes decoding
depend on evaluation. Move the pure scoring function to `metrics/nerl.py`, then
have both decoding tuning and evaluation import from `metrics`.

## Runtime vs Core Utilities

Do not merge `runtime` and `utils`.

They sit at opposite ends of the dependency graph:

- `runtime` is the application shell. It may know about CLI args, checkpoints,
  cache resolution, preflight checks, sharding, mode dispatch, and stage order.
- `_core` should contain only small primitives with no internal imports and at
  least two unrelated consumers.

Merging them would make it normal for low-level code to import the top-level
application shell. That removes the ability to reason about dependencies.

The current `utils` package should be dissolved or made intentionally tiny:

```text
utils/channel_slices.py -> _core/channels.py
utils/label_overlap.py  -> _core/labels.py
utils/model_outputs.py  -> models/outputs.py, with config readers split out
```

Admission rule for `_core`: if a helper does not have at least two unrelated
consumers, or if it depends on config, stage state, Lightning, file layout, or
model internals, it does not belong there.

## Training Metrics Config

Training and evaluation should not share one overloaded metric switch.

Suggested config direction:

```yaml
training:
  validation_metrics: [dice, jaccard]

evaluation:
  enabled: true
  metrics: [adapted_rand, voi, nerl]
```

Training metrics are online model-quality signals used during validation,
checkpointing, schedulers, and progress logging. Evaluation metrics are
artifact-level reports after inference and decoding. They may use the same
primitive implementations, but they are different workflows.

## Chunked Processing

The top-level `connectomics/chunked/` package should be resolved explicitly.

If the active chunked behavior is owned by inference, keep it under
`inference/`. If inference and decoding both need shared grid, halo, manifest,
or chunk-stitching primitives, promote a real shared package:

```text
connectomics/chunking/
  grid.py
  halo.py
  manifest.py
```

Avoid a half-owned top-level package that is neither clearly primitive nor
clearly stage-specific.

## Concrete Migration Targets

1. Introduce `connectomics/metrics/nerl.py` for side-effect-free scoring and
   update decoding tuning to import from it.
2. Keep `connectomics/evaluation/nerl.py` for config/report/logging adapter
   behavior only.
3. Rename `connectomics/evaluation/metrics.py` to
   `connectomics/evaluation/metric_execution.py`.
4. Move `utils/channel_slices.py` and `utils/label_overlap.py` into `_core/`
   or their true owners.
5. Split `utils/model_outputs.py`: model output selection belongs under
   `models/`; config-only readers belong under `config/`.
6. Audit `runtime/output_naming.py`. Prefer passing resolved names from
   `runtime` into stages. If lower stages truly need pure naming helpers, move
   only those helpers below `runtime`.
7. Decide whether `connectomics/chunked/` is unused, inference-owned, or a real
   shared `chunking/` package.
8. Split training validation metric config from artifact evaluation config.
9. Update boundary guardrail tests after each move so the import DAG is enforced.

## Near-Term Priority

Start with the import violations that make architectural reasoning harder:

1. Fix `decoding -> evaluation` by moving pure NERL scoring into `metrics`.
2. Stop `config` from depending on broad utility helpers.
3. Resolve `runtime` imports from lower stages, especially output naming.
4. Dissolve `utils` into `_core` or owners.
5. Rename misleading modules after ownership is clear.

This sequence is intentionally conservative. It improves the dependency graph
without creating a large umbrella package or hiding stage boundaries under a
smaller-looking tree.
