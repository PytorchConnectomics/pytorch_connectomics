# Task

Rewrite `lib/em_erl/scripts/j0126_workflow.py`.

Instead of asking people to download FFN segmentation HDF5 tiles, query the segment
id directly with CloudVolume from
`gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/ffn_segmentation/`
and compute ERL. Simplify the workflow: it should only need the path to the ground
truth skeletons at `/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`.

## Verified facts (from research)

- `test_50_skeletons.h5`: 50 top-level groups, each with `vertices` (N,3 int64,
  **zyx voxel** order) and `edges` (M,2 int64). 500,845 vertices, 500,712 edges.
  Global vertex range zyx: min [0,14,31], max [5714,10886,10555].
- CloudVolume `ffn_segmentation` mip0: resolution (x,y,z)=(9,9,20) nm,
  volume_size (x,y,z)=(10624,10880,5700), dtype uint64, 1 channel,
  chunk_size (x,y,z)=(128,128,64).
- **Coordinate mapping**: skeleton (z,y,x) voxel indexes directly into mip0 as
  `cv[x,y,z]`. Verified: sampled skeleton points return real, consistent segment
  ids (all sampled points of skeleton "0" -> seg 1465128). No resolution scaling.
- Only 37 / 500,845 points fall outside the mip0 bounds (<=14 voxels past the z/y
  edge); these map to segment 0 ("omitted" in ERL).
- Sampling: 82,845 unique occupied chunks, ~0.14s / chunk serial; needs thread-pool
  parallelism.

## Current workflow (to be replaced)

Four subcommands over downloaded `%04d/%d_%d.h5` tiles:
`prepare-gt` -> `map-lut` (sharded) -> `reduce-lut` -> `score`, with intermediate
files `gt_vertices.h5`, `gt_graph.npz`, per-tile LUTs, `seg_lut_all.h5`.

## Constraints

- `lib/em_erl` is its own git repo (nested, gitignored from the outer pytc repo);
  it is the review surface for this run.
- Preserve the existing ERL convention used by the current `prepare-gt`:
  `skel_to_erlgraph(skel_dict)` with no resolution (voxel-unit edge lengths), so
  results stay comparable to the old pipeline.
- Reuse existing `em_erl` APIs (`skel_to_erlgraph`, `compute_erl_score`,
  `print_skeleton_assignment_zero_stats`).
- `cloud-volume` is already the `cloud` extra in `em_erl/setup.py`.
