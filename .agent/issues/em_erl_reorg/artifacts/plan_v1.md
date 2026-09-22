# Plan v1

## Summary

Move the genuinely-reusable functions out of `scripts/j0126_workflow.py` into the
`em_erl` package (mapped to `io.py`, `sampling.py`, `eval.py` by concern), export them
from `__init__`, and rename `scripts/` → `examples/` so the scripts become thin, runnable
usage demos. The change is **behavior-preserving**: functions are relocated verbatim;
existing package functions are not renamed or rewritten. Dataset-specific bits (the j0126
default segmentation URL) stay in the example; the reusable eval recipe is generalized only
by taking `seg_url` as a parameter.

Only `j0126_workflow.py` contributes new package functions — it is the one script that grew
real primitives (CloudVolume sampler, LUT lifecycle, scoring). `volume_eval.py`,
`seg_to_graph.py`, `skel_to_graph.py` are already thin compositions of existing library
functions, so their `run_*` bodies stay in `examples/` as illustration.

This revision addresses the plan_v0 review: (1) examples keep a minimal `sys.path`
bootstrap so they run directly from a source checkout — verified necessary because `em_erl`
is **not** installed in the target env (`import em_erl` fails from an arbitrary cwd; tests
pass only via `conftest.py`'s path insert); (2) `examples/j0126_workflow.py` keeps a thin
`run_j0126_eval(...)` wrapper preserving the old callable surface with the j0126 default URL.

## Scope

In scope:
- Add relocated functions to `em_erl/io.py`, `em_erl/sampling.py`, `em_erl/eval.py`.
- Update `em_erl/__init__.py` `__all__` with the new public names.
- `git mv scripts/ → examples/`; rewrite `examples/j0126_workflow.py` to a thin CLI + a
  `run_j0126_eval` wrapper, both over `em_erl`; keep the other three example files
  (relocated, imports unchanged) and give all four a consistent `sys.path` bootstrap.
- Rewrite `tests/test_j0126_workflow.py` to import from `em_erl` (drop the importlib loader).
- Update `examples/README.md` (was `scripts/README.md`) and top-level `README.md`.

Out of scope (avoid Kitchen-Sink):
- No changes to `erl.py`, `skel.py`, or the internals of `eval.py`/`sampling.py`/`io.py`
  beyond adding the relocated functions.
- No renaming of existing public functions; no new console_scripts entry points; no
  behavior/metric changes.
- Not moving `run_volume_eval`/`run_seg_to_graph`/`run_skel_to_graph` into the package —
  they are already thin wrappers over existing library calls and read better as examples.

## Proposed Changes

### 1. Relocate reusable functions (verbatim bodies)

| Function (from `scripts/j0126_workflow.py`) | New home | Public? |
|---|---|---|
| `normalize_seg_url` | `em_erl/io.py` | yes |
| `open_seg_cloudvolume(seg_url, mip=0, cache_dir="")` | `em_erl/io.py` | yes |
| `load_skeletons(gt_skeleton_path)` (+ `_skeleton_sort_key`, `_parse_skeleton_id`) | `em_erl/io.py` | yes (`load_skeletons`) |
| `sample_cloudvolume_lut(cv, node_zyx, num_workers=16)` (+ `_squeeze_cloudvolume_block`, `_xyz_array`) | `em_erl/sampling.py` | yes (`sample_cloudvolume_lut`) |
| `save_node_segment_lut`, `load_node_segment_lut`, `validate_node_segment_lut` | `em_erl/eval.py` | yes |
| `score_graph_with_lut`, `score_skeletons_with_lut` | `em_erl/eval.py` | yes |
| `run_j0126_eval(...)` → generalized `evaluate_skeletons_cloudvolume(gt_skeleton_path, seg_url, merge_threshold=50, num_workers=16, output_path="", lut_path="", mip=0, cache_dir="")` | `em_erl/eval.py` | yes |

Homes match existing concerns and the current dependency direction (`eval` already imports
from `io` and `sampling`; `sampling → io`):
- `io.py`: CloudVolume opening + URL normalization (next to `_read_cloudvolume`), and HDF5
  skeleton loading (next to the other readers). `cloudvolume` stays a lazy import inside
  `open_seg_cloudvolume`.
- `sampling.py`: `sample_cloudvolume_lut` beside `sample_segment_lut`.
- `eval.py`: LUT save/load/validate, LUT scoring, and the high-level recipe. `eval.py`
  already imports `read_vol, write_h5, mkdir` from `io`; add `open_seg_cloudvolume` and
  `sample_cloudvolume_lut` imports. **Call them through the `em_erl.eval` module namespace**
  (module-level `from .io import open_seg_cloudvolume` / `from .sampling import
  sample_cloudvolume_lut`) so the reuse test can monkeypatch `em_erl.eval.open_seg_cloudvolume`.
  `evaluate_skeletons_cloudvolume` is the old `run_j0126_eval` with `seg_url` promoted to a
  required parameter (no j0126 default in the package).

No import cycles: additions follow the existing `eval → {io, sampling} → io` direction.

### 2. `em_erl/__init__.py`

Add to imports + `__all__` (keep existing entries): `normalize_seg_url`,
`open_seg_cloudvolume`, `load_skeletons`, `sample_cloudvolume_lut`,
`save_node_segment_lut`, `load_node_segment_lut`, `validate_node_segment_lut`,
`score_graph_with_lut`, `score_skeletons_with_lut`, `evaluate_skeletons_cloudvolume`.

### 3. `scripts/` → `examples/`

- `git mv scripts examples` (preserve history for the moved files).
- **`sys.path` bootstrap (all four examples):** because `em_erl` is not installed in this
  env, each example keeps a minimal top-of-file bootstrap so it runs directly from a source
  checkout:
  ```python
  import sys, pathlib
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  ```
  This preserves the current behavior of `j0126_workflow.py` and fixes the inconsistency
  where the other three lacked it. (A `pip install -e .` would make this unnecessary, but is
  not assumed.)
- `examples/j0126_workflow.py`: thin CLI. Keep `DEFAULT_SEG_URL`, `build_parser`, `main`, and
  a thin `run_j0126_eval(gt_skeleton_path, seg_url=DEFAULT_SEG_URL, **kw)` wrapper that calls
  `em_erl.evaluate_skeletons_cloudvolume(...)`. `main` calls `run_j0126_eval`. Remove the
  relocated primitives. Short module docstring naming the `em_erl` functions it demonstrates.
- `examples/volume_eval.py`, `examples/seg_to_graph.py`, `examples/skel_to_graph.py`:
  relocated; add the bootstrap; fix usage comments that say `scripts/…` → `examples/…`.
  Their `run_*` bodies stay (already thin over library funcs).

### 4. Tests

- `tests/test_j0126_workflow.py`: replace `_load_j0126_workflow()`/importlib with
  `import em_erl` / `from em_erl import (…)`. Keep every current assertion:
  `sample_cloudvolume_lut` sampling+OOB, `normalize_seg_url`, LUT round-trip proving reuse
  never opens CloudVolume (monkeypatch `em_erl.eval.open_seg_cloudvolume`), length-mismatch
  guard, and mip/cache plumbing — the reuse/plumbing tests now target
  `em_erl.evaluate_skeletons_cloudvolume`. Other tests unchanged. (conftest already puts the
  repo root on `sys.path`, so `import em_erl` works under pytest.)

### 5. Docs

- Rename `scripts/README.md` → `examples/README.md`; update command paths to `examples/…`;
  note the `em_erl` API each example demonstrates and that examples run from a source
  checkout (or after `pip install -e .`).
- Top-level `README.md`: `scripts/volume_eval.py` → `examples/volume_eval.py`,
  `scripts/README.md` → `examples/README.md`.

## Files and Areas

| File | Change |
|---|---|
| `em_erl/io.py` | + `normalize_seg_url`, `open_seg_cloudvolume`, `load_skeletons` (+ 2 private helpers) |
| `em_erl/sampling.py` | + `sample_cloudvolume_lut` (+ `_squeeze_cloudvolume_block`, `_xyz_array`) |
| `em_erl/eval.py` | + LUT save/load/validate, `score_graph_with_lut`, `score_skeletons_with_lut`, `evaluate_skeletons_cloudvolume` (imports opener/sampler at module level) |
| `em_erl/__init__.py` | export the new public names |
| `examples/j0126_workflow.py` (moved) | thin CLI + `run_j0126_eval` wrapper over `em_erl.evaluate_skeletons_cloudvolume`; `sys.path` bootstrap |
| `examples/{volume_eval,seg_to_graph,skel_to_graph}.py` (moved) | relocated; `sys.path` bootstrap; comment path fixes |
| `examples/README.md` (moved) | path + API doc updates |
| `README.md` | `scripts/` → `examples/` references |
| `tests/test_j0126_workflow.py` | import from `em_erl`; drop importlib loader |

## Verification Plan

1. **Import surface:** from the repo root, `PYTHONPATH=. python -c "import em_erl; [getattr(em_erl,n)
   for n in ['normalize_seg_url','open_seg_cloudvolume','load_skeletons','sample_cloudvolume_lut',
   'save_node_segment_lut','load_node_segment_lut','validate_node_segment_lut','score_graph_with_lut',
   'score_skeletons_with_lut','evaluate_skeletons_cloudvolume']]; print('ok')"` succeeds and the names
   are in `em_erl.__all__`.
2. **Full offline suite:** `python -m pytest tests/ -q` — all 48 tests pass with rewritten
   imports (no network; fake CloudVolume). Primary regression proof that relocation preserved
   behavior.
3. **No leftover coupling:** `grep -rn "importlib\|scripts/" tests/` shows no moved-script
   reference; `grep -rn "scripts/" README.md examples/` is clean.
4. **Examples run WITHOUT install (the point of the bootstrap):** from the repo root, with
   `em_erl` not installed, `python examples/j0126_workflow.py -h`,
   `python examples/volume_eval.py -h`, `python examples/seg_to_graph.py -h`,
   `python examples/skel_to_graph.py -h` all print usage (no `ModuleNotFoundError`).
5. **Example end-to-end (offline):** `python examples/volume_eval.py -p tests/data/vol_pred.h5
   -g tests/data/gt_graph.npz -r 30,30,30` runs and prints ERL from committed test data,
   proving the relocated import path works end-to-end from an example without install.
6. **Behavior parity spot-check:** the j0126 reuse test still asserts scoring from a saved LUT
   does not open CloudVolume and yields identical ERL — now via `em_erl.evaluate_skeletons_cloudvolume`.

## Risks and Questions

- **Env has no installed `em_erl` (verified).** Hence the `sys.path` bootstrap in examples and
  `conftest.py`'s path insert for tests. Verification deliberately runs examples with `em_erl`
  uninstalled to prove the bootstrap works.
- **Monkeypatch point:** `evaluate_skeletons_cloudvolume` must reference `open_seg_cloudvolume`
  and `sample_cloudvolume_lut` via the `em_erl.eval` module namespace so the reuse test can patch
  `em_erl.eval.open_seg_cloudvolume`. Called out for the coder.
- **`load_skeletons` home:** `io.py` (HDF5 reader producing a skeleton dict); alternative
  `skel.py`. Non-blocking.
- **git history:** use `git mv` for the four moved files.
- **Scope discipline:** deliberately not moving the other `run_*` wrappers or adding
  console_scripts; keep the change a focused relocation.
- **Recently-pushed file:** `scripts/j0126_workflow.py` (`3549cad`) is relocated here; the
  example retains the same CLI surface (path changes to `examples/`).

## Changes Since Previous Plan Version

- **[major] Keep the `sys.path` bootstrap.** plan_v0 removed the shim; verified that `em_erl`
  is not installed in the env, so examples must add the repo root to `sys.path`. plan_v1 keeps a
  minimal 2-line bootstrap in all four examples (fixing the prior inconsistency) and changes
  verification to run examples with `em_erl` uninstalled.
- **[minor] Preserve the old callable surface.** plan_v1 keeps a thin `run_j0126_eval` wrapper in
  `examples/j0126_workflow.py` (j0126 default URL) over the generalized package recipe.
