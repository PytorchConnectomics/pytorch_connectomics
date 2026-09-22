Summary: the plan is directionally scoped, but it is not executable as written. The largest problems are coordinate semantics and output-file alignment with the reused `compute_segment_lut_tile_zyx` / `combine_segment_lut_tile_zyx` APIs.

Findings:

- [major] `z_range/y_range/x_range` semantics are internally inconsistent. The plan calls them “chunk-index ranges” with `factor` as voxel offset per chunk index, but the j0126 example uses `z_range=128*range(45)` with `factor=[1,2048,2048]`. The reused API computes offsets as `[z,y,x] * factor`, so generic chunk indices `0..n-1` require `factor=[chunk_z,chunk_y,chunk_x]`, while j0126’s stored keys use z voxel starts with factor z=1. The plan needs an explicit distinction between file/template coordinates and voxel offset coordinates, or it risks silently sampling the wrong z offsets.

- [major] The proposed per-chunk worker calls `compute_segment_lut_tile_zyx` with single-tile `([z],[y],[x])`, but the expected output format is still keyed by `% (z,y,x)`. If flattened chunk index maps to logical grid indices while `compute_segment_lut_tile_zyx` expects template keys and offset keys, SLURM array execution can write valid-looking but misaligned LUT shards.

- [major] The plan says combined LUT alignment is “guaranteed” by `graph.get_nodes_position(None)`, but it does not specify how `gt_vertices.h5` is written/read or ensure the exact same `pts` array in graph node order is passed to every map worker. This is the critical invariant for `[ind,val]` partials. The implementation plan must make this concrete.

- [major] The `--score` path says load `gt_graph.npz` + `seg_lut_all.h5`, `compute_erl_score`, but `compute_erl_score` returns an `ERLScore` object that still needs `compute_erl()` and printing/output handling. The existing reusable wrapper is `score_graph_with_lut`, which validates LUT length/order and calls `compute_erl()`. The plan should use that directly.

- [major] The “chunked == monolithic ERL” verification is too weak by itself. Equal ERL can pass despite node-order or coordinate bugs if the synthetic labels are symmetric, repeated, or low entropy. It must assert the full node LUT equals a direct monolithic LUT in graph-node order, using nontrivial labels and chunk offsets, before comparing ERL.

- [minor] `--wait` “warn on missing after a stall” is underspecified. A pure wait should have a timeout/stall policy that is testable offline, otherwise it can hang indefinitely in CI or scripts.

- [minor] The plan includes CLI `key=value` overrides and YAML parsing but does not define type coercion for lists/ranges/factor. That is implementable, but the plan should specify it enough to avoid ambiguous configs.

Questions:

- Will config coordinates represent file keys, grid indices, or voxel starts? The current plan uses all three concepts under the same `z/y/x` names.
- Is `seg_oset` needed in the generic config? The reused API supports it, but the plan omits it.

READY: no