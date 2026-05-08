# Code v3

## Overview

Address `review_v2.md` finding 1: `resolve_dataset_volume_stems` returned
`"volume"` for the realistic NISB tutorial shape
(`data.test.path = .../seed101`, `data.test.image = data.zarr/img`)
because it passed `cfg.data.test.path` as the `base` argument to
`_stem_from_image_path`. The walker reached `base` and broke out before
returning a name, falling back to `"volume"`. The writer-side helper
(called from `resolve_output_filenames`) does *not* pass `base`, so
predictions still land under `seed101/` while cache discovery looked
under `volume/` — exactly the inconsistency review_v2 flagged.

Fix: drop the `base` parameter entirely. Both the resolver and the
writer now invoke `_stem_from_image_path(path)` with the same signature
and produce identical stems for the same image path.

449 unit tests pass; tutorial validator clean. Diff vs run baseline:
74 files changed, +1387 / −1003.

## What Changed

### Stem resolver alignment

`connectomics/runtime/output_naming.py`:

- `_stem_from_image_path(path)` — `base` parameter removed. Docstring
  updated to explain why: the resolver and the writer must derive the
  same stem from the same path, and passing a base caused the walker
  to halt at the dataset root even when that root was the volume
  identity dir. The walker now climbs unconditionally past container
  directories and uninformative names until it finds an informative
  parent or exhausts the path.

- `resolve_dataset_volume_stems(cfg, mode)` — now calls
  `_stem_from_image_path(p)` without `base`. Comment in the function
  body cites the review_v2 regression for the change.

The "early break at base" behavior was the only place this parameter
flowed through, so removing it does not affect any other code path.
The `data.{val,test}.path` field continues to function for path
expansion in `_expand_paths`/`expand_file_paths`; only the stem
derivation no longer consumes it.

### Regression test

`tests/unit/test_output_path_resolution.py`:

- New `test_resolve_dataset_volume_stems_with_path_equal_to_volume_dir`
  reproduces the exact NISB shape from the review:
  ```
  cfg.data.test.path  = ".../nisb/seed101"
  cfg.data.test.image = ".../nisb/seed101/data.zarr/img"
  ```
  asserts `resolve_dataset_volume_stems(cfg, "test") == ["seed101"]`.

## Implementation Details

### `_stem_from_image_path` semantics (post-v3)

```
1. If filename stem is informative → return stem.
2. Walk up parent chain:
   - skip names that are container suffixes (.zarr / .n5 / .ome.zarr)
   - skip names listed in _UNINFORMATIVE_STEMS
   - return the first remaining informative name
3. If no informative parent found:
   - if original stem is informative → return it
   - else → return "volume"
```

The function is now base-agnostic. Both `resolve_output_filenames`
(writer) and `resolve_dataset_volume_stems` (cache resolver enumeration)
call it identically.

### Why drop `base` rather than fix the early-break check

Two equivalent fixes:

A. Drop `base` everywhere. Resolver and writer trivially agree.
B. Have `_stem_from_image_path(..., base=base)` allow returning
   `base.name` when every child below `base` was skipped as
   structural.

(A) is simpler and aligns with the writer-side semantics. (B) preserves
a knob that was never strictly needed (`resolve_output_filenames` never
had access to `base` anyway). Plan v2 didn't require the parameter
either; it was a leftover from the v1 helper draft.

## Files Changed

| File | Purpose |
|---|---|
| `connectomics/runtime/output_naming.py` | Drop `base` param from `_stem_from_image_path`; update docstring; remove `base=base` call in `resolve_dataset_volume_stems` (with explanatory comment) |
| `tests/unit/test_output_path_resolution.py` | Add `test_resolve_dataset_volume_stems_with_path_equal_to_volume_dir` covering NISB shape |

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

## Verification

```
python scripts/validate_tutorial_configs.py
# Validated 14 canonical tutorial configs successfully; skipped 2 custom workflow YAMLs.

python -m pytest tests/unit/ -q
# 449 passed, 45 warnings in 83.87s
```

Direct sanity check (NISB shape):

```
python -c "
from connectomics.config import Config
from connectomics.runtime.output_naming import (
    _stem_from_image_path, resolve_dataset_volume_stems,
)
print('direct:', _stem_from_image_path('/projects/.../seed101/data.zarr/img'))
cfg = Config()
cfg.data.test.path  = '/projects/.../seed101'
cfg.data.test.image = '/projects/.../seed101/data.zarr/img'
print('resolver:', resolve_dataset_volume_stems(cfg, mode='test'))
"
# direct: seed101
# resolver: ['seed101']
```

## Review Focus

1. **`base` parameter removal**: search for any external caller of
   `_stem_from_image_path` that still passes `base=...`. Grep result:
   none in `connectomics/`, `tests/`, or `scripts/`. Anyone importing
   the helper from outside this repo would see a `TypeError` on the
   stale call — but the helper is underscore-prefixed (private),
   so external consumers should not be relying on it.
2. **NISB tutorial**: `tutorials/neuron_nisb/base_banis.yaml` uses
   `data.test.path = .../seed101/` + `image: data.zarr/img`; this is
   the exact configuration the review flagged. Reviewer can confirm
   the regression test reproduces it before / does not reproduce it
   after the fix by running the new test alone.

## Risks and Unknowns

- The dataset-root identity-dir pattern (`path = .../seed101`,
  `image = data.zarr/img`) only appears in the NISB tutorials in
  this repo. Other tutorials use `path: .../base/test/` plus
  `image: seed101/data.zarr/img` (the seed dir is below `path`). Both
  shapes now produce `seed101` because the walker is the same in both
  cases.
- No new external API surface; one private parameter dropped. Public
  callers of `resolve_dataset_volume_stems` and `resolve_output_filenames`
  unchanged.

## Changes Since Previous Code Version

Address the single `review_v2.md` finding:

- **Finding 1** (resolver/writer divergence on NISB shape): the `base`
  parameter on `_stem_from_image_path` (and the `base=base` call in
  `resolve_dataset_volume_stems`) was the source of the divergence.
  Removed entirely. Both code paths now produce identical stems for
  the same image path. Regression test added.

Tests: 449 (was 448 in code_v2; +1 net).
