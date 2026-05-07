# waterz_dw vs lib/waterz

This note only covers what `lib/waterz_dw/` changes relative to `lib/waterz/`.
The core watershed/agglomeration backend is still broadly the same; the main
differences are in packaging, Python API shape, and extra region-graph helpers.

## Packaging and Build

- `lib/waterz_dw` uses a flat package layout: `waterz/...`.
- `lib/waterz` uses a `src/` layout: `src/waterz/...`.
- `lib/waterz_dw` keeps most JIT build logic inside `waterz/__init__.py` and
  compiles the agglomeration module manually with `distutils`, `Cython`, file
  locks, and a cache under `~/.cython/inline/`.
- `lib/waterz` moves that logic into `src/waterz/_agglomerate.py` and uses
  `witty.compile_cython(...)` instead of the older manual inline-build path.
- `lib/waterz_dw` builds only `evaluate` and `region_graph` Cython extensions in
  `setup.py`.
- `lib/waterz` builds `evaluate` and `merge` extensions, and adds a richer
  `pyproject.toml`, `setuptools-scm`, typed-package metadata, Ruff, and GitHub
  Actions CI.
- `lib/waterz_dw` still carries older release plumbing (`.travis.yml`,
  `travis/build-wheels.sh`) instead of the newer GitHub Actions/pre-commit
  toolchain.
- Versioning is older and less consistent in `lib/waterz_dw`: `setup.py`
  declares `0.9.6`, while `waterz/__init__.py` hardcodes `__version__ = '0.8'`.
  `lib/waterz` derives the package version from `setuptools-scm`.

## Public API Differences

- `lib/waterz_dw` exposes only `agglomerate()` and `evaluate()` from the top
  level.
- `lib/waterz` additionally exposes top-level `waterz()`,
  `get_region_graph()`, `get_region_graph_average()`, `merge_segments()`, and
  `merge_dust()`.
- `lib/waterz_dw` does not have the modern helper modules
  `src/waterz/_waterz.py` or `src/waterz/_merge.py`.
- `lib/waterz_dw` keeps convenience wrappers in separate legacy modules:
  `waterz/seg_waterz.py` and `waterz/seg_init.py`.

## Region Graph Workflow Differences

- `lib/waterz_dw` overloads `agglomerate()` with an extra `rg_opt` switch that
  changes the meaning of the call:
- `rg_opt=0`: normal waterz segmentation/agglomeration
- `rg_opt=1`: build a full region graph from `affs + fragments`
- `rg_opt=2`: build a z-only region graph
- `rg_opt=3`: build an xy-only region graph
- `rg_opt=4`: initialize agglomeration from a precomputed region graph instead
  of from affinities/fragments
- `lib/waterz` removes this multiplexed `rg_opt` API. Normal agglomeration,
  region-graph extraction, and post-merge cleanup are split into separate
  explicit helpers.
- `lib/waterz_dw` includes `seg_waterz.getRegionGraph(...)` and
  `seg_waterz.waterzFromRegionGraph(...)` as wrappers around those `rg_opt`
  modes. `lib/waterz` does not provide a precomputed-region-graph re-entry API.
- `lib/waterz_dw` includes `region_graph.pyx`, which exposes `merge_id(id1,id2)`
  for plain union-find relabeling from a pre-sorted edge list.
- `lib/waterz` replaces that older path with explicit merge helpers operating on
  segmentations and graph arrays: `merge_segments(...)` and `merge_dust(...)`.

## Extra Helpers Present Only in waterz_dw

- `waterz/seg_waterz.py` includes `getScoreFunc(...)`, which converts shorthand
  names like `aff50_his256` into C++ scoring-function type strings. `lib/waterz`
  does not keep this shorthand helper in the package.
- `waterz/seg_init.py` adds Python-side seed generation and 2D slice-wise
  watershed helpers using `mahotas` and `scipy.ndimage`. These utilities do not
  exist in `lib/waterz`.
- Because of those helpers, `lib/waterz_dw` effectively has extra optional
  runtime dependencies beyond the leaner modern package surface.

## What lib/waterz Adds Instead

- A dedicated top-level `waterz()` wrapper that materializes copied
  segmentations for all thresholds.
- Dedicated merge/cleanup helpers in `_merge.py` instead of the older
  `rg_opt`/`merge_id` workflow.
- Separate max-affinity and average-affinity region-graph extraction helpers.
- Cleaner package metadata, typed-package support, and more modern build/test
  tooling.

## Practical Summary

- Reach for `lib/waterz_dw` only if you specifically need its older
  direction-specific region-graph extraction (`rg_opt=2/3`), precomputed region
  graph re-entry (`rg_opt=4`), or the legacy `seg_init.py` / `seg_waterz.py`
  helpers.
- Reach for `lib/waterz` if you want the cleaner public API, maintained build
  path, dedicated merge helpers, and the package surface that the current
  PyTorch Connectomics decoder code already targets.
