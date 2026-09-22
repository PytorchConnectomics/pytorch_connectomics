# review_v0 raw reviewer notes (in-session planner review of chunk code_v0)

Review surface: chunk-run delta over run_start.diff (the reorg substrate). New files only:
examples/volume_eval_chunk.py (594), examples/volume_eval_chunk.yaml, tests/test_volume_eval_chunk.py.
HEAD unchanged (3549cad — no driver commit). No reorg files modified.

## Independent verification (re-run)

- `python -m pytest tests/ -q` -> 55 passed (48 prior + 7 new).
- `python examples/volume_eval_chunk.py -h` OK uninstalled (bootstrap works).
- Primary correctness test (full-LUT equality) passes serial and parallel.

## Primary correctness test is genuinely strong (addresses plan_v0-review [major])

tests/test_volume_eval_chunk.py::test_chunked_lut_matches_monolithic_and_erl [num_workers 1,2]:
- factor=[3,4,5] (per-axis, nontrivial); z/y/x_range keys START at 1 => nonzero voxel offset
  (z0=1*3=3), directly exercising pts_oset=[z,y,x]*factor.
- volume = high-entropy hash (zz*100003+yy*1009+xx*9176+37)%1000003+1000 => nontrivial,
  non-repeating labels; two edge-connected vertices forced to a shared label so ERL scores a real
  correct edge.
- Splits volume into per-chunk files seg%(z,y,x); runs init -> run_chunks_local -> reduce_and_score;
  asserts combined_lut == monolithic compute_segment_lut(volume, graph.get_nodes_position(None))
  ELEMENT-WISE (assert_array_equal), then ERL + skeleton_erl parity. A coordinate/factor or
  node-order bug fails assert_array_equal. This is exactly the strengthened check the plan review
  required.
- Other tests: chunk_index<->key roundtrip + single-chunk writes exactly one file; wait completes
  when all present and raises WaitTimeoutError naming "chunk 3" + path on stall; reduce length ==
  num_nodes; sbatch script contains "--array=0-{n-1}" and "--chunk-index $SLURM_ARRAY_TASK_ID";
  config overrides incl. dotted nested (slurm.mem=4G) + type coercion.

## Wiring spot-read (matches plan_v1)

- Imports compute_segment_lut_tile_zyx / combine_segment_lut_tile_zyx / score_graph_with_lut from em_erl.eval.
- parse_factor: list/"a,b,c"/default [1,1,1], len==3 validated.
- init_workflow: pts = graph.get_nodes_position(None) -> gt_vertices.h5 + gt_graph.npz.
- run_chunk: compute_segment_lut_tile_zyx(..., factor=cfg["factor"]) with the shared pts, single-key ranges.
- run_chunks_local: multiprocess over disjoint chunks; reduce_and_score: combine_* + score_graph_with_lut.
- build_sbatch_script (testable, returns string), wait_for_completion (stall/timeout), chunk_index_to_key.
- sys.path bootstrap present (example runs uninstalled).

## Conclusion

Behavior matches plan_v1; all plan_v0-review findings are realized in code and proven by a rigorous
full-LUT-equality test with nonzero offset and nontrivial labels. Reuses existing tile machinery; no
package changes; embarrassingly-parallel map/reduce/score with file-presence completion and a bounded
wait; SLURM array + multiprocess backends. No new findings.

READY: yes
