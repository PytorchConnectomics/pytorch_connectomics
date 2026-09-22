# Plan v1

## Summary

Add `examples/volume_eval_chunk.py`: a generic, config-driven orchestrator that computes the
node-segment LUT for an extra-large segmentation stored in chunks by launching **SLURM array
jobs or local multiprocess workers**, waits for the per-chunk results, combines them, and
scores ERL. It mirrors `waterz decode_large` CLI ergonomics but is lighter because LUT
computation is embarrassingly parallel (**map → reduce → score**, no cross-chunk stitching);
completion is tracked by **partial-LUT file presence**, not a task database.

The orchestrator is a thin launcher over existing, tested `em_erl` functions
(`compute_segment_lut_tile_zyx` map, `combine_segment_lut_tile_zyx` reduce, `skel_to_erlgraph`
build, `score_graph_with_lut` score). No package changes: a single new example + example config
+ tests.

This revision makes the coordinate/alignment semantics precise per the plan_v0 review:
ranges are **segmentation template keys**, `factor` maps a key to a voxel offset, the exact
`pts` node array is written once and reused by every worker, `--score` uses `score_graph_with_lut`,
and the test asserts **full per-node LUT equality** (not just ERL) against a monolithic baseline.

## Scope

In scope:
- `examples/volume_eval_chunk.py` — orchestrator/CLI.
- `examples/volume_eval_chunk.yaml` — example config (j0126 FFN tile layout).
- `tests/test_volume_eval_chunk.py` — offline tests incl. full-LUT-equality vs monolithic.
- `examples/README.md` entry.

Out of scope:
- No `em_erl` package changes (all reuse exists).
- Primary input = **per-chunk file template**; zarr/precomputed single-store sampling is a noted
  future extension, not built.
- No port of waterz's task-claiming orchestrator / border-stitch stages.

## Proposed Changes

### 1. Coordinate & key model (made precise)

A "chunk" is one segmentation file addressed by `seg_path_format % (z, y, x)`. The config
`z_range`/`y_range`/`x_range` are exactly those **template key values**. The reused API computes
each tile's voxel offset as `pts_oset = [z, y, x] * factor` and its z-extent from the file's own
shape, so:

- `factor` (length-3, zyx) is the **per-axis multiplier from template key → voxel start**.
- Two supported keyings, both expressed with the same mechanism:
  - **Index-keyed** files (keys `0..n-1` per axis): set `factor = chunk_shape` (voxel start =
    index * chunk size).
  - **Voxel-start-keyed** files (keys are voxel starts, e.g. j0126 `z = 0,128,256,…`): set that
    axis's `factor = 1`. j0126 = `factor=[1, 2048, 2048]` (z voxel-start; y,x index).
- Optional `seg_oset` (default 0): added to non-zero sampled ids per tile, for chunk files whose
  labels are numbered locally rather than globally.

The flat task index `i` maps to `(z,y,x) = list(itertools.product(z_range, y_range, x_range))[i]`.
Both the per-chunk worker and the reduce use this identical enumeration and the identical
`output_path_format % (z, y, x)` for partial LUTs, so array shards and the reduce line up exactly.

### 2. `examples/volume_eval_chunk.py`

Config (YAML `volume_eval_chunk:` block; plain dict like decode_large's `large_decode`, with CLI
`key=value` overrides):

- `gt_skeleton` (HDF5 skeleton groups) **or** `gt_graph` (prebuilt `.npz`).
- `seg_path_format` — per-chunk file template with `%` placeholders `(z, y, x)`.
- `z_range` / `y_range` / `x_range` — template key values, each given as an explicit list **or**
  a `"start,stop,step"` string expanded with `range(...)` (documented coercion).
- `factor` — length-3 zyx multiplier (key→voxel start), default `[1, 1, 1]`.
- `seg_oset` — optional int (default 0).
- `dataset` — HDF5 dataset name inside chunk files (optional).
- `workflow_root` — dir holding `gt_vertices.h5`, `gt_graph.npz`, partial LUTs
  `<root>/lut/%04d_%d_%d.h5`, and combined `seg_lut_all.h5`.
- `merge_threshold` (default 50), `resolution` (optional; default voxel-unit ERL),
  `output_path` (optional score pickle).
- `backend` (`multiprocess`|`slurm`), `num_workers`, `slurm:{partition,mem,cpus_per_task,time}`.

Alignment invariant (made concrete): `--init-only` loads skeletons (`load_skeletons`) or the
`gt_graph`, builds the graph (`skel_to_erlgraph`), computes `pts = graph.get_nodes_position(None)`
(int voxel zyx in node order), writes `pts` to `workflow_root/gt_vertices.h5` (`write_h5`) and the
graph to `gt_graph.npz` (`ERLGraph.save_npz`). **Every** map worker reads that same
`gt_vertices.h5` and passes it as `pts` to `compute_segment_lut_tile_zyx`, which writes
`[ind, val]` over the full point array; `combine_segment_lut_tile_zyx` ORs partials back by index
→ full LUT aligned 1:1 to graph node order.

CLI (mirrors `decode_large`):
- `--config PATH` + `key=value` overrides (scalar int/float/str coercion; ranges/factor parsed
  from the YAML types).
- `--init-only` — build graph, write `gt_vertices.h5` + `gt_graph.npz`, print workflow summary
  (grid, #chunks, backend), exit.
- `--chunk-index N` / `--chunk-range A-B` — map chunk(s) N: call `compute_segment_lut_tile_zyx`
  with single-key ranges `([z],[y],[x])`, the shared `pts`, and `output_path_format`. Auto-detect
  N from `SLURM_ARRAY_TASK_ID`. Idempotent (existing partial LUTs skipped by the reused API).
- `--parallel N` / `--local` — local multiprocess `spawn` Pool; each worker maps a disjoint chunk
  subset.
- `--sbatch` — emit an sbatch script with `--array=0-{n_chunks-1}` whose body runs
  `python examples/volume_eval_chunk.py --config … --chunk-index $SLURM_ARRAY_TASK_ID`; submit via
  `sbatch`; print the `--wait` monitor command. `--sbatch`/`--local` override YAML `backend`.
- `--wait` — poll `workflow_root/lut` for all expected partial-LUT files; print `done/total`
  every ~10 s; **stall/timeout policy:** `--wait-timeout` (default e.g. 0 = no cap) and a
  `--stall-timeout` (default 600 s) — if no new partial appears within `stall-timeout`, exit
  non-zero listing the missing chunks (so it cannot hang forever; offline-testable by pointing at
  a complete vs incomplete lut dir).
- `--reduce` — `combine_segment_lut_tile_zyx` over all keys → `seg_lut_all.h5`.
- `--score` — load `gt_graph.npz` + `seg_lut_all.h5` and call **`score_graph_with_lut(graph, lut,
  merge_threshold, output_path)`** (validates `len==num_nodes`, runs `compute_erl()`, prints,
  optional pickle). `--reduce`+`--score` may be combined.

Recovery: per-chunk map skips existing outputs; re-submit recomputes only missing chunks;
`--wait` is pure file presence.

### 3. `examples/volume_eval_chunk.yaml`

j0126 FFN tile layout worked example: `seg_path_format=.../%04d/%d_%d.h5`,
`z_range="0,5760,128"` (→ `128*range(45)`), `y_range="0,6,1"`, `x_range="0,6,1"`,
`factor=[1,2048,2048]`, `gt_skeleton=test_50_skeletons.h5`, a `workflow_root`, `merge_threshold`,
and a `slurm` block. Comments explain the key/factor relationship.

### 4. `tests/test_volume_eval_chunk.py` (offline)

- **Full-LUT equality + ERL parity (primary):** build a small labeled volume with **nontrivial,
  high-entropy labels** and a small skeleton dict placing nodes across chunk boundaries with a
  **nonzero chunk offset** (e.g. index-keyed with `factor=chunk_shape`). Write the volume as
  per-chunk HDF5 files; run the orchestrator (serial and `--parallel 2`) to produce the combined
  LUT; assert the combined LUT **equals element-wise, in graph node order**, the monolithic LUT
  from `compute_segment_lut(full_volume, graph.get_nodes_position(None))`; only then assert equal
  ERL. This catches node-order/coordinate/factor bugs that equal ERL alone could mask.
- **Key/index mapping:** flat index ↔ `(z,y,x)` round-trips; `--chunk-index i` writes exactly the
  `output_path_format % (z,y,x)` for that key.
- **Wait policy:** `--wait` returns complete when all partials exist; with a missing partial and a
  short `--stall-timeout`, it exits non-zero naming the missing chunk (no hang).
- **Reduce:** combined LUT length == `graph.num_nodes` and matches the monolithic LUT.
- **sbatch script generation (no real submit):** emitted script contains `--array=0-{n-1}` and the
  correct `--chunk-index $SLURM_ARRAY_TASK_ID` worker command (monkeypatch `subprocess.run`/inspect
  the script string; never call `sbatch`).

### 5. `examples/README.md`

Document `volume_eval_chunk.py`: the map→reduce→score model, the key/factor coordinate rule, the
CLI modes, SLURM-array vs multiprocess backends, and the example YAML.

## Files and Areas

| File | Change |
|---|---|
| `examples/volume_eval_chunk.py` (new) | Generic chunked-volume LUT orchestrator + CLI |
| `examples/volume_eval_chunk.yaml` (new) | Example config (j0126 tile layout) |
| `tests/test_volume_eval_chunk.py` (new) | Offline tests incl. full-LUT-equality vs monolithic |
| `examples/README.md` | Document the new orchestrator |

## Verification Plan

1. **Full-LUT equality (primary):** the synthetic test asserts the combined per-node LUT equals
   the monolithic LUT element-wise in node order (nontrivial labels, nonzero offset), then equal
   ERL, in serial and `--parallel` modes. `python -m pytest tests/test_volume_eval_chunk.py -q`.
2. **Full suite unaffected:** `python -m pytest tests/ -q` stays green (new tests only).
3. **CLI loads uninstalled:** `python examples/volume_eval_chunk.py -h` prints usage from a source
   checkout (same `sys.path` bootstrap as other examples).
4. **End-to-end from the CLI on the synthetic fixture:** `--init-only` → `--parallel 2` → `--wait`
   → `--reduce --score` prints a finite ERL; a second run recomputes nothing (idempotent) and
   yields the identical ERL and identical combined LUT.
5. **Wait stall + sbatch string:** `--wait` with a missing partial exits non-zero under a short
   stall timeout; the emitted sbatch script has `--array=0-{n-1}` and the right worker command
   (no real `sbatch`).

## Risks and Questions

- **Coordinate/key correctness (highest):** resolved by defining ranges = template keys and
  `factor` = key→voxel-start multiplier (`pts_oset=[z,y,x]*factor`), documented with the two
  keyings and exercised by the test's nonzero-offset fixture.
- **Node-order alignment:** guaranteed by writing `pts=graph.get_nodes_position(None)` once at
  `--init` and having every worker read that exact array; asserted by full-LUT equality, not just
  ERL.
- **`--score` correctness:** uses `score_graph_with_lut` (validates length/order, runs
  `compute_erl()`), not raw `compute_erl_score`.
- **`--wait` cannot hang:** bounded by `--stall-timeout` (default 600 s) and optional
  `--wait-timeout`; offline-testable.
- **SLURM untestable in CI:** sbatch path verified by script-string inspection; multiprocess/serial
  fully tested offline.
- **Input model scope:** per-chunk file template primary; zarr/precomputed store deferred and
  called out.
- **Placement:** orchestrator in `examples/`, reusing package primitives — consistent with the
  reorg split.

## Changes Since Previous Plan Version

- **[major] Range/factor semantics** — defined ranges as segmentation template keys and `factor`
  as the key→voxel-start multiplier, with both keyings (index vs voxel-start) documented; j0126 =
  `factor=[1,2048,2048]`.
- **[major] Key/index consistency** — flat index↔`(z,y,x)` via `product(...)`; worker and reduce
  share the identical `output_path_format % (z,y,x)`.
- **[major] pts alignment made concrete** — `--init` writes `gt_vertices.h5` from
  `graph.get_nodes_position(None)`; every worker reads that exact array.
- **[major] `--score` uses `score_graph_with_lut`** (validate + `compute_erl()` + print).
- **[major] Verification strengthened** — assert full per-node LUT equality vs a monolithic LUT
  (nontrivial labels, nonzero offset) before comparing ERL.
- **[minor] `--wait` stall/timeout policy** added (`--stall-timeout`/`--wait-timeout`), offline-testable.
- **[minor] Coercion + `seg_oset`** — list-or-`start,stop,step` ranges, 3-list factor, scalar
  override coercion, optional `seg_oset` (default 0).
