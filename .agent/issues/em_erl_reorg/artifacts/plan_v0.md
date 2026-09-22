# Plan v0

## Summary

Move the genuinely-reusable functions out of `scripts/j0126_workflow.py` into the
`em_erl` package (mapped to `io.py`, `sampling.py`, `eval.py` by concern), export them
from `__init__`, and rename `scripts/` → `examples/` so the scripts become thin,
runnable usage demos. Keep the change **behavior-preserving**: functions are relocated
verbatim; existing package functions are not renamed or rewritten. Dataset-specific bits
(the j0126 default segmentation URL) stay in the example; the reusable eval recipe is
generalized only by taking `seg_url` as a parameter.

Only `j0126_workflow.py` contributes new package functions — it is the one script that
grew real primitives (CloudVolume sampler, LUT lifecycle, scoring). `volume_eval.py`,
`seg_to_graph.py`, `skel_to_graph.py` are already thin compositions of existing library
functions, so their `run_*` bodies stay in `examples/` as illustration; they only move
directory and keep importing from `em_erl`.

## Scope

In scope:
- Add relocated functions to `em_erl/io.py`, `em_erl/sampling.py`, `em_erl/eval.py`.
- Update `em_erl/__init__.py` `__all__` with the new public names.
- `git mv scripts/ → examples/`; rewrite `examples/j0126_workflow.py` to a thin CLI that
  imports from `em_erl`; keep the other three example files (relocated, imports unchanged).
- Rewrite `tests/test_j0126_workflow.py` to import from `em_erl` (drop the importlib
  script loader).
- Update `examples/README.md` (was `scripts/README.md`) and top-level `README.md`
  (`scripts/…` → `examples/…`).

Out of scope (avoid Kitchen-Sink):
- No changes to `erl.py`, `skel.py`, or the internals of `eval.py`/`sampling.py`/`io.py`
  beyond adding the relocated functions.
- No renaming of existing public functions; no new console_scripts entry points; no
  behavior changes; no metric changes.
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

Homes chosen to match existing concerns and dependency direction (`eval` already imports
from `io` and `sampling`; `sampling` and `io` already lazy-import optional deps):
- `io.py`: CloudVolume opening + URL normalization (sits with existing `_read_cloudvolume`),
  and HDF5 skeleton loading (sits with the other readers). `cloudvolume` stays a lazy
  import inside `open_seg_cloudvolume`.
- `sampling.py`: `sample_cloudvolume_lut` is a point sampler, next to `sample_segment_lut`.
- `eval.py`: the LUT lifecycle (save/load/validate), LUT scoring, and the high-level
  CloudVolume+LUT recipe — all ERL-domain. `evaluate_skeletons_cloudvolume` is the old
  `run_j0126_eval` with `seg_url` promoted to a required parameter (no j0126 default in
  the package), and it calls `open_seg_cloudvolume`/`sample_cloudvolume_lut` via the
  module namespace so tests can monkeypatch it.

No import cycles: `eval → {io, sampling}` and `sampling → io` already hold; the additions
follow the same direction.

### 2. `em_erl/__init__.py`

Add to imports + `__all__` (keep existing entries): `normalize_seg_url`,
`open_seg_cloudvolume`, `load_skeletons`, `sample_cloudvolume_lut`,
`save_node_segment_lut`, `load_node_segment_lut`, `validate_node_segment_lut`,
`score_graph_with_lut`, `score_skeletons_with_lut`, `evaluate_skeletons_cloudvolume`.

### 3. `scripts/` → `examples/`

- `git mv scripts examples` (preserve history for the moved files).
- `examples/j0126_workflow.py`: thin CLI — keep `DEFAULT_SEG_URL`, `build_parser`, `main`;
  `main` calls `em_erl.evaluate_skeletons_cloudvolume(seg_url=args.seg_url, …)`. Remove the
  now-relocated primitives and the `sys.path` shim (the package is importable). Keep a short
  module docstring pointing at the library functions it demonstrates.
- `examples/volume_eval.py`, `examples/seg_to_graph.py`, `examples/skel_to_graph.py`:
  relocated as-is; they already import from `em_erl`. Update the usage comments that say
  `scripts/…` to `examples/…`.

### 4. Tests

- `tests/test_j0126_workflow.py`: replace `_load_j0126_workflow()`/importlib with direct
  `from em_erl import (…)` / `import em_erl`. Keep all current assertions:
  `sample_cloudvolume_lut` sampling+OOB, `normalize_seg_url`, LUT round-trip proving reuse
  never opens CloudVolume (monkeypatch `em_erl.eval.open_seg_cloudvolume`), length-mismatch
  guard, and mip/cache plumbing through `evaluate_skeletons_cloudvolume`. The
  reuse/plumbing tests now target `em_erl.evaluate_skeletons_cloudvolume` (the generalized
  recipe) instead of a script module.
- Other tests unchanged.

### 5. Docs

- Rename `scripts/README.md` → `examples/README.md`; update command paths to `examples/…`
  and note the demonstrated `em_erl` API.
- Top-level `README.md`: `scripts/volume_eval.py` → `examples/volume_eval.py`,
  `scripts/README.md` → `examples/README.md`.

## Files and Areas

| File | Change |
|---|---|
| `em_erl/io.py` | + `normalize_seg_url`, `open_seg_cloudvolume`, `load_skeletons` (+ 2 private helpers) |
| `em_erl/sampling.py` | + `sample_cloudvolume_lut` (+ `_squeeze_cloudvolume_block`, `_xyz_array`) |
| `em_erl/eval.py` | + LUT save/load/validate, `score_graph_with_lut`, `score_skeletons_with_lut`, `evaluate_skeletons_cloudvolume` |
| `em_erl/__init__.py` | export the new public names |
| `examples/j0126_workflow.py` (moved) | thin CLI over `em_erl.evaluate_skeletons_cloudvolume` |
| `examples/{volume_eval,seg_to_graph,skel_to_graph}.py` (moved) | relocated; comment path fixes |
| `examples/README.md` (moved) | path + API doc updates |
| `README.md` | `scripts/` → `examples/` references |
| `tests/test_j0126_workflow.py` | import from `em_erl`; drop importlib loader |

## Verification Plan

1. **Import surface:** `python -c "import em_erl; [getattr(em_erl, n) for n in (…new names…)]"`
   succeeds; `em_erl.__all__` contains the new names.
2. **Full offline suite:** `python -m pytest tests/ -q` — all 48 tests pass with the
   rewritten imports (no network; fake CloudVolume). This is the primary regression proof
   that the relocation preserved behavior.
3. **No leftover coupling:** `grep -rn "importlib\|scripts/" tests/` shows no reference to
   a moved script path; `grep -rn "scripts/" README.md examples/` is clean.
4. **Example CLIs load:** `python examples/j0126_workflow.py -h`,
   `python examples/volume_eval.py -h`, `examples/seg_to_graph.py -h`,
   `examples/skel_to_graph.py -h` all print usage (no import errors, no `sys.path` shim
   needed).
5. **Example end-to-end (offline):** `python examples/volume_eval.py -p tests/data/vol_pred.h5
   -g tests/data/gt_graph.npz -r 30,30,30` runs and prints ERL (uses committed test data),
   proving the relocated package import path works from an example.
6. **Behavior parity spot-check:** the j0126 reuse test still asserts scoring from a saved
   LUT does not open CloudVolume and yields identical ERL — now via the package recipe.

## Risks and Questions

- **Monkeypatch point:** `evaluate_skeletons_cloudvolume` must call `open_seg_cloudvolume`
  and `sample_cloudvolume_lut` through the `em_erl.eval` module namespace (module-level
  references), so the reuse test can patch `em_erl.eval.open_seg_cloudvolume`. Flagged so
  the coder wires the call sites accordingly.
- **`load_skeletons` home:** placed in `io.py` (it is an HDF5 reader producing a skeleton
  dict). Alternative `skel.py`; `io.py` chosen for reader cohesion. Non-blocking.
- **git history:** use `git mv` for the four moved files so history follows; function moves
  across files inherently don't preserve per-function blame.
- **Scope discipline:** deliberately NOT moving the other three `run_*` wrappers into the
  package or adding console_scripts, to avoid over-abstracting already-thin glue. If the
  maintainer wants installable CLIs later, that is a separate, additive change.
- **Regenerated pushed file:** `scripts/j0126_workflow.py` was just committed (`3549cad`);
  this refactor relocates its functions. That is the intent; the example retains the same
  CLI surface so existing invocations still work (path changes to `examples/`).

## Changes Since Previous Plan Version

Initial plan.
