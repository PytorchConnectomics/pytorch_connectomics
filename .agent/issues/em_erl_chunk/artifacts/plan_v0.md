# Plan v0

## Summary

Add `examples/volume_eval_chunk.py`: a generic, config-driven orchestrator that computes the
node-segment LUT for an extra-large segmentation stored in chunks by launching **SLURM array
jobs or local multiprocess workers**, waits for the per-chunk results, combines them, and
scores ERL. It mirrors the `waterz decode_large` CLI ergonomics but is deliberately lighter:
LUT computation is embarrassingly parallel (**map → reduce → score**, no cross-chunk
stitching), so completion is tracked by **partial-LUT file presence** rather than a
task-claiming database.

The orchestrator is a thin launcher over existing, already-tested `em_erl` functions —
`compute_segment_lut_tile_zyx` (per-chunk map, writes `[ind,val]` partial LUTs, skips existing
→ idempotent), `combine_segment_lut_tile_zyx` (reduce), `skel_to_erlgraph` + `compute_erl_score`
(build/score). It is the generalization of the old `j0126_workflow.py` `map-lut/reduce-lut/score`
staging into a real parallel launcher. No package changes are required; this is a single new
example plus an example config and tests.

## Scope

In scope:
- `examples/volume_eval_chunk.py` — the orchestrator/CLI.
- `examples/volume_eval_chunk.yaml` — an example config (the j0126 FFN tile layout as the
  worked example).
- `tests/test_volume_eval_chunk.py` — offline tests incl. a synthetic chunked volume proving
  chunked ERL == monolithic ERL.
- Short `examples/README.md` entry for the new orchestrator.

Out of scope:
- No changes to `em_erl` package modules (all reuse already exists).
- Primary input is a **per-chunk file template** (the common "saved in chunks" case and the
  direct reuse path). Single chunked stores (zarr/precomputed) as a per-chunk box sampler are a
  noted future extension, not built here.
- No port of waterz's task-claiming orchestrator / border-stitch stages (not needed for
  embarrassingly-parallel LUT map).

## Proposed Changes

### 1. `examples/volume_eval_chunk.py`

Config (YAML `volume_eval_chunk:` block, consumed as a plain dict like decode_large's
`large_decode`, with CLI `key=value` overrides):

- `gt_skeleton` (HDF5 skeleton groups) or `gt_graph` (prebuilt `.npz`) — build/load the ERL graph.
- `seg_path_format` — per-chunk file template with `%` placeholders for `(z, y, x)`
  (e.g. `.../%04d/%d_%d.h5` for the j0126 tiles, or `.../z%d_y%d_x%d.h5`).
- `z_range`, `y_range`, `x_range` — chunk-index ranges (explicit lists, or a compact
  `start,stop,step` spec expanded to lists); `factor` — voxel offset per chunk index
  (e.g. `[1, 2048, 2048]`), matching `compute_segment_lut_tile_zyx`'s `factor`.
- `dataset` — HDF5 dataset name inside chunk files (optional).
- `workflow_root` — directory holding `gt_vertices.h5`, `gt_graph.npz`, per-chunk partial LUTs
  (`<root>/lut/%04d_%d_%d.h5`), and the combined `seg_lut_all.h5`.
- `merge_threshold` (default 50), `resolution` (optional; default voxel-unit ERL),
  `output_path` (optional score pickle).
- `backend` (`multiprocess`|`slurm`), `num_workers`, `slurm:{partition,mem,cpus_per_task,time}`.

Chunk enumeration: flatten `product(z_range, y_range, x_range)` into an ordered list; index
`i` ↔ `(z,y,x)` gives the SLURM `--array` mapping and `--chunk-index`.

CLI (mirrors `decode_large`):
- `--config PATH` + `key=value` overrides.
- `--init-only` — build the ERL graph, write `gt_vertices.h5` + `gt_graph.npz`, print the
  workflow summary (volume/chunk grid, #chunks, backend), exit.
- `--chunk-index N` / `--chunk-range A-B` — compute the partial LUT for chunk(s) N by calling
  `compute_segment_lut_tile_zyx` with the single tile's `([z],[y],[x])` ranges. Auto-detects N
  from `SLURM_ARRAY_TASK_ID` when present. (Idempotent: existing partial LUTs are skipped.)
- `--parallel N` / `--local` — local multiprocess `spawn` Pool; each worker computes a disjoint
  subset of chunks.
- `--sbatch` — emit an sbatch script with `--array=0-{n_chunks-1}` whose body runs
  `python examples/volume_eval_chunk.py --config … --chunk-index $SLURM_ARRAY_TASK_ID`; submit
  via `sbatch`; print the monitor command (`--wait`). `--local`/`--sbatch` override YAML `backend`.
- `--wait` — poll `workflow_root/lut` until all expected partial-LUT files exist; print
  `done/total` progress every ~10 s; exit when complete (warn on any missing after a stall).
- `--reduce` — `combine_segment_lut_tile_zyx` over all chunks → `seg_lut_all.h5`.
- `--score` — load `gt_graph.npz` + `seg_lut_all.h5`, `compute_erl_score`, print ERL, optional
  `-o` pickle. `--reduce`+`--score` may be combined (single `--assemble`-style call, like
  decode_large's `--wait --assemble`).

Backend selection: `--sbatch`/`--local` override YAML `backend` (default `multiprocess`), same
precedence as decode_large.

Recovery/robustness: per-chunk map skips existing outputs, so re-submitting after a crash only
recomputes missing chunks; `--wait` is pure file presence, no shared task DB.

### 2. `examples/volume_eval_chunk.yaml`

An example config for the j0126 FFN tile layout (`seg_path_format`, `z_range=128*range(45)`,
`y_range=range(6)`, `x_range=range(6)`, `factor=[1,2048,2048]`, `gt_skeleton=test_50_skeletons.h5`,
a `workflow_root`, and a `slurm` block), demonstrating the intended usage on a real dataset.

### 3. `tests/test_volume_eval_chunk.py` (offline)

- **Synthetic chunked volume (key correctness proof):** generate a small labeled volume, write
  it as per-chunk HDF5 files on a small grid (e.g. 2×2×1 chunks) plus a small skeleton dict;
  run the orchestrator end-to-end (serial and `--parallel 2`) → combined LUT + ERL; assert the
  ERL **equals** `run_volume_eval`/`compute_erl_score` on the same **unchunked** volume. Proves
  chunked map/reduce == monolithic.
- **Chunk enumeration / index mapping:** flat index ↔ `(z,y,x)` round-trips; `--chunk-index`
  writes exactly that chunk's partial LUT.
- **Wait / reduce:** `--wait` reports complete only when all partial LUTs exist; reduce combines
  to the correct full LUT (aligned to graph node order).
- **sbatch script generation (no real submit):** assert the emitted script contains
  `--array=0-{n-1}` and the correct `--chunk-index $SLURM_ARRAY_TASK_ID` worker command
  (monkeypatch/inspect; do not call `sbatch`).

### 4. `examples/README.md`

Add a section documenting `volume_eval_chunk.py`: the map→reduce→score model, the CLI modes,
the SLURM-array vs multiprocess backends, and a pointer to the example YAML.

## Files and Areas

| File | Change |
|---|---|
| `examples/volume_eval_chunk.py` (new) | Generic chunked-volume LUT orchestrator + CLI |
| `examples/volume_eval_chunk.yaml` (new) | Example config (j0126 tile layout) |
| `tests/test_volume_eval_chunk.py` (new) | Offline tests incl. chunked==monolithic ERL |
| `examples/README.md` | Document the new orchestrator |

## Verification Plan

1. **Chunked == monolithic (primary):** the synthetic-fixture test asserts the orchestrator's
   ERL equals the single-volume computation on the same data, in both serial and `--parallel`
   modes. `python -m pytest tests/test_volume_eval_chunk.py -q`.
2. **Full suite unaffected:** `python -m pytest tests/ -q` stays green (new file only adds tests;
   no package changes).
3. **CLI loads uninstalled:** `python examples/volume_eval_chunk.py -h` prints usage from a
   source checkout (same `sys.path` bootstrap as the other examples).
4. **End-to-end on the synthetic fixture from the CLI:** `--init-only` → `--parallel 2` →
   `--wait` → `--reduce --score` prints a finite ERL; a second run recomputes nothing
   (idempotent skip) and yields the identical ERL.
5. **sbatch dry check:** the emitted sbatch script has `--array=0-{n-1}` and the correct worker
   command; `sbatch` is not actually invoked in tests.

## Risks and Questions

- **Node-order alignment:** the combined LUT must be aligned to `skel_to_erlgraph` node order.
  Guaranteed because `compute_segment_lut_tile_*` write `[ind,val]` over the full stacked-vertex
  array (written at `--init` from `graph.get_nodes_position(None)`), and `combine_*` ORs them
  back by index. The synthetic test checks LUT length == `graph.num_nodes` and ERL parity.
- **Coordinate/factor correctness:** vertices must be in the chunk voxel grid; `factor` maps
  chunk index → voxel offset (`pts_oset = [z,y,x]*factor`). The example YAML documents this;
  the test exercises a non-trivial factor.
- **Input model scope:** per-chunk file template is primary (direct reuse); zarr/precomputed
  single-store sampling is deferred and called out, not silently missing.
- **SLURM untestable in CI:** the sbatch path is verified by script-string inspection only; the
  real launch is documented. Multiprocess/serial paths are fully tested offline.
- **Placement:** orchestrator lives in `examples/` (a workflow), reusing package primitives; no
  package changes — consistent with the just-completed reorg's "general → package, composition →
  examples" split.
- **Sequencing:** built on the approved-but-uncommitted `em_erl_reorg`; review surface is the
  delta over `run_start.diff`.

## Changes Since Previous Plan Version

Initial plan.
