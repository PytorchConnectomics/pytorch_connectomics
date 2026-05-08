# Code v2

## Overview

Final code revision under `revision_rounds=2`, addressing all three findings
from `review_v1.md`. Net behavioral effect: chunked inference now writes
through the per-volume contract; `_stem_from_image_path` correctly skips
`.zarr` / `.n5` / `.ome.zarr` container directories so NISB-style
`/data/seed101/data.zarr/img` paths resolve to `seed101`; cache-preflight
and decode-only flows route their stem derivation through the same
canonical helper as the writers.

All 448 unit tests pass. Tutorial validator clean. Diff vs run baseline:
74 files changed, +1380 / −1003.

## What Changed

### Finding 1: chunked test paths use per-volume layout

`connectomics/training/lightning/test_pipeline.py`:

- **Raw chunked branch** (`chunking.output_mode=raw_prediction`) at
  line 718: previously stripped the leading `_` and trailing `.h5` from
  the cache suffix and rejoined `<save_path>/<volume>_<suffix>.h5` flat.
  Now treats `intermediate_prediction_cache_suffix` as the artifact
  filename it returns (already includes `.h5`) and writes
  `<save_path>/<volume>/<filename>`. `volume_dir.mkdir(parents=True,
  exist_ok=True)` ensures the per-volume directory exists before the
  chunked runner streams into it.

- **Decoded chunked branch** (`chunking.output_mode=decoded`) at line
  810: previously appended a second `.h5` to a value that already ended
  in `.h5`, producing `<volume>_decoded_....h5.h5`. Now uses the
  filename from `final_prediction_output_tag` directly and joins it
  under the per-volume subdir (`<save_path>/<volume>/<filename>`).

### Finding 2: `_stem_from_image_path` skips container parents

`connectomics/runtime/output_naming.py`:

- New `_CONTAINER_PARENT_SUFFIXES = (".zarr", ".n5", ".ome.zarr")` and
  helper `_is_container_dir_name(name)`.

- `_stem_from_image_path` rewritten as a parent-walk that skips both
  uninformative names (`img`, `data`, `raw`, …) **and** container
  directories. The previous one-step parent fallback was incorrect for
  multi-component paths like `/data/seed101/data.zarr/img` (returned
  `data.zarr`); the new walker keeps climbing until it finds an
  informative non-container name.

- Final fallback now suppresses uninformative original stems too: a
  bare `/img.h5` returns `"volume"` instead of `"img"` so multiple such
  volumes don't collide on the same per-volume directory.

Verified directly:

```
/data/seed101/data.zarr/img            → seed101
/data/seed102/data.zarr/img            → seed102
/data/seed101/img.h5                   → seed101
/data/seed101/data.n5/img              → seed101
/data/seed101/dataset.ome.zarr/img     → seed101
/data/seed101/raw_aff.h5               → raw_aff
/data/sample.h5                        → sample
/img.h5                                → volume
```

### Finding 3: cache_resolver routes through canonical helper

`connectomics/runtime/cache_resolver.py`:

- `preflight_test_cache_hit` (lines 246–253): replaced
  `Path(image_path).stem` with `_stem_from_image_path(image_path)`.
- `try_cache_only_test_execution` (line 319): same replacement for
  `[Path(p).stem for p in test_image_paths]`.
- `has_cached_predictions_in_output_dir` (line 194): the in-test
  fallback path that bypasses `resolve_dataset_volume_stems` (used when
  `data.test.name` is unset and the resolver returns empty) also routes
  through `_stem_from_image_path`.

All cache-discovery code paths now produce the same per-volume stems
as the writers.

### Tests

`tests/unit/test_output_path_resolution.py`:

- New `test_resolve_dataset_volume_stems_skips_zarr_container_parent`
  asserts both the direct helper output for `.zarr`, `.n5`, and
  `.ome.zarr` container variants and the end-to-end resolution through
  `resolve_dataset_volume_stems` for two seed dirs sharing
  `data.zarr/img` paths.

`tests/unit/test_lightning_data_collate.py`:

- `test_resolve_output_filenames_supports_single_lazy_string_image_path`
  updated for the new `.zarr` skip behavior:
  `/data/seed101/input_e.zarr/img` → `["seed101"]`.

## Implementation Details

### Container-suffix definition

`_CONTAINER_PARENT_SUFFIXES` is a tuple of lowercase suffixes; the
detector matches case-insensitively via
`name.lower().endswith(suffix)`. Adding new container types (e.g.
`.zip` for zarr-in-zip, `.toml` for some metadata containers) is a
single-line change. The current set covers the connectomics canonical
storage formats found in the tutorials and tested datasets.

### Stem resolver semantics (precise rule)

```
1. If filename stem is informative → return stem.
2. Walk up the parent chain:
   - skip names that are container suffixes (`.zarr` etc.)
   - skip names listed in _UNINFORMATIVE_STEMS
   - skip if the candidate equals the explicit dataset `base` prefix
   - return the first remaining informative name
3. If no informative parent found:
   - if original stem is informative → return it
   - else → return "volume"
```

This is pure (no I/O), so it runs before any glob expansion and is
safe to call from cache preflight paths.

### Why not `data.{val,test}.name` for everything?

The opt-in `name` field still wins when set (handled in
`resolve_dataset_volume_stems` step 1); the parent-walk is the
fallback for tutorials that haven't been migrated to set `name`
explicitly. Most existing NISB tutorials use `seed*/data.zarr/img`
which the walker now resolves correctly without requiring a tutorial
edit.

## Files Changed

| File | Purpose |
|---|---|
| `connectomics/runtime/output_naming.py` | `_stem_from_image_path` rewritten to skip `.zarr`/`.n5`/`.ome.zarr` containers via parent walk; final fallback suppresses uninformative stems; `_CONTAINER_PARENT_SUFFIXES` + `_is_container_dir_name` added |
| `connectomics/runtime/cache_resolver.py` | Three call sites migrated from `Path(...).stem` to `_stem_from_image_path` (`has_cached_predictions_in_output_dir` fallback, `preflight_test_cache_hit`, `try_cache_only_test_execution`) |
| `connectomics/training/lightning/test_pipeline.py` | Chunked raw + chunked decoded branches now write through `<save_path>/<volume>/<artifact>.h5` instead of flat `<volume>_<suffix>.h5` (raw) / `<volume>_<filename>.h5.h5` (decoded). Removed redundant `removeprefix("_").removesuffix(".h5")` and the trailing `+ ".h5"` |
| `tests/unit/test_output_path_resolution.py` | `test_resolve_dataset_volume_stems_skips_zarr_container_parent` covers `.zarr`, `.n5`, `.ome.zarr` and end-to-end glob-derived stems |
| `tests/unit/test_lightning_data_collate.py` | Single-image stem test updated for new container-skip semantics |

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

## Verification

```
python scripts/validate_tutorial_configs.py
# Validated 14 canonical tutorial configs successfully; skipped 2 custom workflow YAMLs.

python -m pytest tests/unit/ -q
# 448 passed, 45 warnings in 85.69s
```

Direct helper sanity checks (run interactively, not part of CI):

```
python -c "from connectomics.runtime.output_naming import _stem_from_image_path
for p, e in [
    ('/data/seed101/data.zarr/img', 'seed101'),
    ('/data/seed102/data.zarr/img', 'seed102'),
    ('/data/seed101/data.n5/img', 'seed101'),
    ('/data/seed101/dataset.ome.zarr/img', 'seed101'),
    ('/data/seed101/raw_aff.h5', 'raw_aff'),
    ('/img.h5', 'volume'),
]:
    g = _stem_from_image_path(p)
    print('OK' if g==e else 'FAIL', p, '->', g)"
# All OK.
```

## Review Focus

1. **Container suffix list** in `_CONTAINER_PARENT_SUFFIXES`. Currently
   `(".zarr", ".n5", ".ome.zarr")`. Reviewer may want to add `.zip`
   (zarr-in-zip), `.toml` (rarely used as a container), or others.
   Single source of truth — easy to extend.
2. **Chunked write `volume_dir.mkdir(parents=True, exist_ok=True)`**:
   the chunked runners (`run_chunked_prediction_inference`,
   `run_chunked_affinity_cc_inference`) take an explicit `output_path`
   and write through their own h5py code (not via `write_outputs`).
   Pre-creating the parent dir means downstream code reads the file
   from the same per-volume location the cache resolver looks in. No
   recursion or duplication.
3. **`has_cached_predictions_in_output_dir` fallback** (used only when
   `data.{val,test}.name` is unset *and* `resolve_dataset_volume_stems`
   returns empty — a unit-test-only configuration). The fallback is
   now consistent with the writers.

## Risks and Unknowns

- The stem resolver is purely string-based (no I/O); container detection
  uses literal suffixes. A directory genuinely named e.g. `data.zarr/`
  but not actually a Zarr container would still be skipped. Acceptable
  in this codebase since the tutorials only use these suffixes for real
  Zarr/N5 stores.
- Existing on-disk cache files written under the flat code_v0 layout
  (or any pre-PR version) are still invisible to the cache resolver —
  same risk as code_v1. Re-runs incur re-inference. Documented in PR.
- `data.train.name` advisory warning is unchanged (still emitted; train
  mode does not write per-volume artifacts).

## Changes Since Previous Code Version

Address all three `review_v1.md` findings:

- **Finding 1** (chunked paths): both raw and decoded chunked test
  branches in `test_pipeline.py:718-728` and `:810-816` migrated to
  the per-volume `<save_path>/<volume>/<artifact>.h5` layout.
  Decoded-branch double-`.h5` bug (line 815: `+ ".h5"` over a value
  already ending in `.h5`) fixed.
- **Finding 2** (stem resolver): `_stem_from_image_path` now correctly
  resolves `/data/seed101/data.zarr/img` to `seed101` (and the same for
  `.n5` / `.ome.zarr`) via a parent-walk that skips container
  directories. NISB tutorials (which use `seed*/data.zarr/img`) work
  out of the box without `data.test.name` overrides.
- **Finding 3** (cache resolver consistency): three call sites in
  `cache_resolver.py` (preflight, cache-only execution, and the
  fallback in `has_cached_predictions_in_output_dir`) now use
  `_stem_from_image_path` so preflight and writer paths agree.

Tests added: `test_resolve_dataset_volume_stems_skips_zarr_container_parent`
covering all three container suffixes, plus an end-to-end glob case.
Total tests: 448 passing (was 447 in code_v1; +1 net, 0 dropped).
