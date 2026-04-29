# Data V2 Plan

## Goal

Keep `connectomics.data` domain-based and remove compatibility leftovers. The
module should own IO, preprocessing, augmentation, label transforms, datasets,
sampling, and splitting with one canonical path for each object.

Baseline: `data.md` reports most data cleanup as complete. V2 mainly deletes
remaining shims and turns the documented structure into the enforced structure.

## Public API

Keep these public subpackages:

- `connectomics.data.io`
- `connectomics.data.augmentation`
- `connectomics.data.processing`
- `connectomics.data.datasets`

Public concepts:
- volume IO: read, save, shape, format detection
- MONAI dictionary transforms for IO, augmentation, and label processing
- pure augmentation operations in `augment.augment_ops`
- deterministic label/target operations in `process`
- dataset classes and sampling/split helpers in `dataset`

Everything else should be private or package-internal.

## Delete

- `connectomics.data.utils` compatibility shim.
- Duplicate copies of `sampling.py` and `split.py` outside `data.datasets`.
- Any `monai_transforms.py` compatibility names.
- Any unused dataset classes kept for legacy code.
- Any transform builder functions that are no longer called.
- Any old exports in `data.__init__` that point to deleted modules.

## Move/Rename

Canonical paths:

| Concept | V2 owner |
| --- | --- |
| Train/val volume split | `connectomics.data.datasets.split` |
| Sample counting | `connectomics.data.datasets.sampling` |
| IO transforms | `connectomics.data.io.transforms` |
| Augmentation transforms | `connectomics.data.augmentation.transforms` |
| Label/process transforms | `connectomics.data.processing.transforms` |
| nnUNet preprocessing | `connectomics.data.processing.nnunet_preprocess` |
| Pure augmentation functions | `connectomics.data.augmentation.augment_ops` |

## Config Contract

Data config should use domain terms:
- `data.input`
- `data.label_transform`
- `data.augmentation`
- `data.dataloader`
- `data.split`

Rules:
- no config objects inside transform classes;
- transform constructors receive plain Python parameters;
- no hidden old-key handling in builders;
- train/test/tune data shorthands must be documented in schema, not handled
  through ad hoc fallback logic.

## Boundary Rules

- `data` must not import `training`.
- `data` must not import `inference`.
- `data` must not import `decoding` except through tests or examples.
- Pure functions must not import MONAI.
- MONAI wrappers should be thin adapters around pure functions.
- Dataset code should not know about model architectures or losses.

## Implementation Order

1. Delete `connectomics/data/utils`.
2. Update tests and tutorials to import from `data.datasets`.
3. Remove stale exports from `data.__init__` and subpackage `__init__` files.
4. Search for deleted names across Python and YAML.
5. Tighten transform builders so unsupported config shapes raise early.
6. Add boundary tests for no `data.utils` import path.
7. Run data, training data-factory, and tutorial config tests.

## Tests

Add or update:
- importing `connectomics.data.utils` fails;
- production imports use `data.datasets.split` and `data.datasets.sampling`;
- every `RandomizableTransform` calls `self.randomize()` in `__call__`;
- pure augmentation ops import no MONAI symbols;
- transform builders reject unsupported config types;
- datasets preserve image/label padding semantics;
- nnUNet preprocessing round-trips metadata needed by inference output.

## Open Decisions

- Whether `connectomics.data.__init__` should export only subpackages or also
  selected high-use functions like `read_volume` and `save_volume`.
- Whether test/tune `data.batch_size` shorthand should stay or move under
  `data.dataloader.batch_size` only.
