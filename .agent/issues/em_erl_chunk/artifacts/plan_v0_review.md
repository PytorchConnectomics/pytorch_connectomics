# Plan v0 Review

## Summary

Codex (coder) reviewed plan_v0 read-only and withheld approval: the direction is right but
the coordinate/alignment semantics around the reused `compute_segment_lut_tile_zyx` /
`combine_segment_lut_tile_zyx` APIs are under-specified in ways that could silently produce
misaligned LUTs. Five `[major]` and two `[minor]` findings, all addressable in a revision.
Raw transcript: `state/plan_v0_review.review.raw.md`.

## Findings

- **[major] z/y/x range semantics conflate index vs voxel-start.** The reused API computes the
  tile offset as `[z,y,x]*factor`; j0126 uses `z_range=128*range(45)` with `factor=[1,2048,2048]`
  (z is a voxel start, y/x are indices). The plan must define the ranges as **segmentation
  template key values** and `factor` as the per-axis key→voxel-offset multiplier
  (`voxel_offset = key*factor`), and document both keyings (index-keyed → factor=chunk_shape;
  voxel-start-keyed → factor=1).
- **[major] chunk-index → (z,y,x) and output keys must be consistent.** The single-tile worker
  (`([z],[y],[x])`) and the full-range reduce must use the **same** `output_path_format % (z,y,x)`
  keys, so array shards and the reduce line up. Make the flat-index↔key mapping explicit.
- **[major] LUT node-order alignment must be concrete.** Specify that `--init` writes
  `gt_vertices.h5` from `graph.get_nodes_position(None)` (int voxel zyx, node order) and
  `gt_graph.npz`, and every map worker reads that exact `pts` array; `combine_*` ORs `[ind,val]`
  back by index → full LUT in node order.
- **[major] `--score` should use `score_graph_with_lut`.** `compute_erl_score` returns an
  `ERLScore` needing `compute_erl()`+print; the reusable `score_graph_with_lut` validates LUT
  length/order and does both. Use it.
- **[major] Verification too weak.** Equal ERL can mask node-order/coordinate bugs. The synthetic
  test must first assert the combined per-node LUT **equals a direct monolithic LUT** in graph
  node order, using nontrivial labels and nonzero chunk offsets, then compare ERL.
- **[minor] `--wait` needs a timeout/stall policy** (offline-testable), so it can't hang forever.
- **[minor] CLI override / range / factor coercion unspecified.** Define list-or-`start,stop,step`
  range expansion, factor as a 3-list, and scalar override coercion; expose optional `seg_oset`
  (default 0) for chunk files with per-chunk local label numbering.

## Questions

- What do config `z/y/x` mean — file keys, grid indices, or voxel starts? Resolved: **template
  key values**, with `factor` converting key→voxel offset.
- Is `seg_oset` needed generically? Yes — expose it (default 0) for per-chunk locally-numbered
  segmentations.

## Verdict

VERDICT: NEEDS_CHANGES
