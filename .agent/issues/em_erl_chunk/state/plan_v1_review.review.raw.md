Prior findings addressed:

- Range/factor semantics: addressed. `z/y/x_range` are template keys and `factor` is explicitly key-to-voxel offset.
- Chunk index/output key consistency: addressed via shared `product(z_range, y_range, x_range)` mapping and identical `output_path_format`.
- LUT node-order alignment: addressed. `gt_vertices.h5` is written once from `graph.get_nodes_position(None)` and reused by all workers.
- Score path: addressed. Plan uses `score_graph_with_lut`.
- Verification strength: addressed. Tests require full per-node LUT equality against monolithic `compute_segment_lut` before ERL parity.
- Wait policy: addressed with stall and optional total timeout.
- Coercion / `seg_oset`: addressed.

New findings:

None.

READY: yes