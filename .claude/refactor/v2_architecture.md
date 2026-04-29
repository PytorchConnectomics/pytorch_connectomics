# V2 Refactor Architecture Contract

## Goal

V2 is a clean-break refactor. The target is a smaller codebase with one
canonical path for each concept, strict config semantics, and explicit stage
boundaries. Backward compatibility is intentionally out of scope.

The existing refactor reports remain useful evidence. This document is the
contract that all `*_v2.md` component plans should follow.

## Non-Goals

- Preserve old import paths.
- Preserve old YAML field names.
- Preserve compatibility shims, aliases, or facade modules.
- Keep tutorial configs working without edits.
- Keep unused public APIs for external callers.
- Optimize every algorithm during the structural pass.

## Core Rules

### 1. One canonical owner per concept

Every public function, class, config field, and CLI behavior must have exactly
one owner. Duplicate import paths are deleted instead of re-exported.

Examples:
- dataset split and sampling code lives under `connectomics.data.datasets`.
- top-level `connectomics.utils` contains only code used by multiple domains.
- decoding owns conversion from prediction arrays to segmentation artifacts.
- evaluation owns metrics over artifacts and labels.

### 2. No backward-compatible shims

Delete modules that only re-export symbols from their new location. Tests and
tutorials must import from canonical paths.

Allowed `__init__.py` files:
- package-level public API declarations,
- schema export hubs,
- small import convenience for stable v2 APIs.

Disallowed `__init__.py` files:
- compatibility aliases,
- wildcard forwarding,
- old module facades,
- duplicate names maintained only for historical callers.

### 3. Config is strict

The dataclass schema is the source of truth. Runtime code must not probe ghost
fields with `getattr(..., default)` to tolerate old configs.

Rules:
- removed fields raise during config load,
- old aliases are not accepted,
- profile selectors are accepted only at canonical paths,
- stage sections are allowed only where the schema declares them,
- cross-section validation catches shape, channel, loss, decoding, and
  evaluation mismatches before execution.

### 4. Stages are separate

The pipeline stages are:

1. `train`: fit model checkpoints.
2. `infer`: load a model and write raw prediction artifacts.
3. `decode`: consume prediction artifacts and write segmentation artifacts.
4. `evaluate`: consume segmentation artifacts and labels, then write metrics.
5. `tune`: run search over decoding/postprocessing parameters.

Combined test mode may remain as a convenience wrapper, but it must call the
same stage functions in sequence. It must not contain separate hidden logic.

### 5. Dependencies flow by domain

Target dependency direction:

```text
config
utils
data
models
metrics
training     -> config, data, models, metrics
inference    -> config, data, models
decoding     -> config, data, utils
evaluation   -> config, data, metrics
runtime/CLI  -> config, training, inference, decoding, evaluation
```

Domain packages should not import from `scripts/`. Training should not own
decoding or evaluation internals. Decoding should not construct models or load
checkpoints. Evaluation should not know which decoder produced an artifact.

### 6. Public API is explicit and small

Each component plan must list the v2 public API. Anything not listed is private
or deleted. Public APIs should be stable within v2, but no v1 compatibility is
required.

### 7. Tests validate v2 contracts

Tests should be updated to the new contracts rather than preserving old import
paths. Add focused tests for:
- no removed config fields accepted,
- no ghost config field reads,
- no compatibility shim imports,
- stage split behavior,
- artifact layout and metadata,
- component boundary imports.

## Target Package Responsibilities

### `connectomics.config`

Owns schema, YAML loading, profile resolution, stage merge, validation, and path
resolution. It does not execute training, inference, decoding, or evaluation.

### `connectomics.data`

Owns volume IO, MONAI transforms, deterministic preprocessing, augmentation,
label target generation, datasets, split, and sampling.

### `connectomics.models`

Owns model and loss construction. It does not know about Lightning loops,
artifact writing, decoding, or evaluation.

### `connectomics.training`

Owns training runtime, Lightning module, callbacks, optimizers, schedulers,
debugging, and train-time metrics. It may expose a test convenience only if the
logic delegates to inference/decode/evaluate stage APIs.

### `connectomics.inference`

Owns model prediction over volumes and writing raw prediction artifacts.
Artifacts are file-backed, chunked, and use a stable `(C, Z, Y, X)` layout per
volume after crop/channel selection.

### `connectomics.decoding`

Owns converting raw predictions to segmentation/synapse artifacts. It consumes
prediction artifacts or arrays. It never constructs models or loads checkpoints.

### `connectomics.evaluation`

Owns metric execution over decoded artifacts and labels. It is a top-level
stage, not a decoding submodule.

### `connectomics.metrics`

Owns metric implementations and thin torchmetrics wrappers. It does not own
pipeline orchestration or artifact IO beyond simple array inputs.

### `connectomics.utils`

Owns only genuinely cross-domain primitives. If a helper has one domain
consumer, move it into that domain.

## V2 Documentation Set

Component plans:
- `config_v2.md`
- `data_v2.md`
- `inference_v2.md`
- `decoding_v2.md`
- `evaluation_v2.md`
- `training_v2.md`
- `models_v2.md`
- `metrics_v2.md`
- `utils_v2.md`

Each component plan should follow this template:

```text
Goal
Public API
Delete
Move/Rename
Config Contract
Boundary Rules
Implementation Order
Tests
Open Decisions
```

## Cross-Component Implementation Order

1. Freeze this architecture contract.
2. Refactor config schema and tutorials to v2 names.
3. Delete compatibility shims and update imports.
4. Split inference, decoding, and evaluation stages.
5. Simplify training so test mode delegates to stage APIs.
6. Trim model, metric, and utility public APIs.
7. Run full tests and add boundary/contract tests.

This order keeps the schema and import graph stable before deeper runtime
changes land.
