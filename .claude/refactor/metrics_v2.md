# Metrics V2 Plan

## Goal

Keep metric implementations lean and stage-agnostic. Metrics should compute
scores from arrays or tensors; evaluation should orchestrate file IO, config,
and reports.

Baseline: `metrics.md` reports major dead code removal and a clean public API.
V2 mainly formalizes the split between metrics and evaluation.

## Public API

Keep:

- adapted Rand error implementation and torchmetrics wrapper
- variation of information implementation and torchmetrics wrapper
- instance matching implementations
- skeleton metrics if still used by supported scripts
- small shared helpers only when used by multiple metric functions

Public metric names should be explicit in `metrics.__init__`.

## Delete

- Any metric functions with no v2 caller.
- Any wildcard exports.
- Any wrappers that only preserve old names.
- Any file-pair or directory evaluation orchestration that belongs in
  `connectomics.evaluation`.
- Any printing inside metrics.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Numpy segmentation metrics | `connectomics.metrics.segmentation_numpy` |
| Torchmetrics wrappers | `connectomics.metrics.metrics_seg` |
| Skeleton metrics | `connectomics.metrics.metrics_skel` |
| Report orchestration | `connectomics.evaluation` |

If directory/file-pair evaluation helpers remain, move them to evaluation unless
they are pure array-level helpers.

## Config Contract

Metrics do not own config schema directly. Evaluation config references metric
names and kwargs, then evaluation resolves those names to metric functions.

Metric kwargs must be explicit. No metric should inspect global config objects.

## Boundary Rules

- Metrics may import numpy, scipy, torch, and torchmetrics.
- Metrics may import shared label-overlap utilities.
- Metrics must not import config.
- Metrics must not import data IO.
- Metrics must not import inference, decoding, training, or evaluation stage
  runners.

## Implementation Order

1. Define evaluation-stage metric registry or resolver.
2. Move any file/directory report helpers out of metrics.
3. Trim `metrics.__init__` to supported names only.
4. Add import-boundary tests.
5. Run metric and evaluation tests.

## Tests

Add or update:
- metric functions accept arrays/tensors without config objects;
- torchmetrics wrappers update/compute correctly;
- removed old metric names are not exported;
- evaluation resolves metric names to the correct metric callables;
- multiprocessing helpers, if kept, use picklable functions only.

## Open Decisions

- Whether skeleton metric scripts remain supported v2 surface or become
  examples outside the core package.
- Whether a metric registry should live in `metrics` or `evaluation`.
