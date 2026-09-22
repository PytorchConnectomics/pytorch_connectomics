# Task

Rewrite the old `j0126_workflow.py` tiled workflow into a **generic** orchestrator
`volume_eval_chunk.py` that launches SLURM jobs or local multiprocess workers to compute
the node-segment LUT for **extra-large volumes stored in chunks**, waits for the results,
and then evaluates ERL. Model the execution ergonomics on
`lib/waterz/src/waterz/cli/decode_large.py` (`waterz_decode_large`).

## Dependency / sequencing

This builds on the in-flight CCC run `em_erl_reorg`, which relocates the j0126 LUT
primitives into the `em_erl` package and renames `scripts/` → `examples/`. This chunk run
is **sequenced after** the reorg completes (and is committed) so it targets the final
package layout and gets a clean git baseline. Two coders must not edit `lib/em_erl`
concurrently.

## Reference pattern (waterz `decode_large`, read-only research done)

`lib/waterz/src/waterz/cli/decode_large.py` + `large_workflow.py` provide the ergonomics to
mirror:
- CLI modes: `--init-only`, `--worker`, `--chunk-index`/`--chunk-range` (for `sbatch --array`),
  `--wait` (poll until all tasks terminal, print progress), `--parallel N` (local multiprocess
  spawn Pool), `--sbatch`/`--local` (force backend; else YAML `backend`), `--assemble`.
- SLURM backend: emit an sbatch script with `--array=0-{n_chunks-1}`; each array task runs a
  worker on one chunk; print a monitor command (`--wait`).
- `build_chunk_grid(volume_shape, chunk_shape)` -> `ChunkRef(index, start, stop, key=z{}_y{}_x{})`.
- File-backed task state under a `workflow_root`; crash recovery (reset stale/failed; recover
  outputs that already exist).
- Config from a YAML `large_decode:` block via `LargeDecodeConfig.from_dict`.

## Key design simplification vs waterz

LUT computation is **embarrassingly parallel**: each chunk independently samples the GT
skeleton nodes that fall inside it and writes a partial LUT; there is **no cross-chunk
stitching** (unlike waterz's offsets/connect/region-graph stages). So the DAG is just
**map (per-chunk, parallel) → reduce (combine partial LUTs, serial) → score (serial)**. The
orchestrator should be correspondingly lighter than waterz's full task-claiming orchestrator —
a chunk grid + per-chunk partial-LUT files + a file-presence wait + combine + ERL is enough.

## Reuse points in `em_erl` (post-reorg)

- Existing tile LUT machinery in `em_erl/eval.py`: `compute_segment_lut_tile_zyx`,
  `compute_segment_lut_tile_z`, `combine_segment_lut_tile_zyx`, `combine_segment_lut_tile`
  (partial-LUT `[ind, val]` files + combine) — directly reusable for map/reduce.
- `sample_segment_lut` / `VolumeSource` (`em_erl/sampling.py`) for reading a chunk and sampling.
- Reorg additions: `load_skeletons`, `skel_to_erlgraph`, LUT save/load/validate,
  `score_graph_with_lut` / `evaluate_skeletons_cloudvolume`, `compute_erl_score`.
- The old `j0126_workflow.py` map-lut/reduce-lut/score design (pre-CloudVolume rewrite) is the
  conceptual ancestor; this generalizes it to any chunked volume with a real launcher.

## Chunked-volume input model (to design in the plan)

The segmentation is stored in chunks. Support the common cases: a per-chunk file template
(e.g. `%04d/%d_%d.h5` like the old j0126 tiles, or `z{}_y{}_x{}.h5`), and/or a chunked store
(zarr / precomputed CloudVolume). Node→chunk partitioning uses the GT graph node voxel
positions and the chunk grid.

## Constraints

- Lives in `examples/` (post-reorg) as `volume_eval_chunk.py`; generic (not j0126-specific).
- Reuse `em_erl` package functions; do not reimplement LUT tiling/combine/ERL.
- Keep the orchestration appropriately simple for an embarrassingly-parallel map/reduce;
  do not port waterz's full stitching orchestrator.
- Repo `lib/em_erl` (nested git repo). Baseline captured after the reorg is committed.
