# Review v0

## Summary

The generic chunked-volume ERL orchestrator is implemented as planned and approved.
`examples/volume_eval_chunk.py` is a thin launcher over em_erl's existing tile machinery
(`compute_segment_lut_tile_zyx` map → `combine_segment_lut_tile_zyx` reduce →
`score_graph_with_lut` score), with SLURM-array and multiprocess backends, a bounded `--wait`,
and file-presence completion. Independently re-verified: 55 tests pass (48 prior + 7 new), the
example runs uninstalled, and the primary correctness test proves chunked == monolithic. Raw
notes: `state/review_v0.review.raw.md`.

The correctness gate demanded by the plan_v0 review is genuinely met:
`test_chunked_lut_matches_monolithic_and_erl` uses `factor=[3,4,5]` with chunk keys starting at 1
(nonzero voxel offset) and high-entropy non-repeating labels, and asserts the combined per-node LUT
**equals element-wise** the monolithic `compute_segment_lut(volume, graph.get_nodes_position(None))`
before comparing ERL, in both serial and `--parallel` modes.

## Diff Baseline

run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b

Review surface: the chunk-run delta over `run_start.diff` (the approved em_erl_reorg substrate) —
three new files only (`examples/volume_eval_chunk.py`, `examples/volume_eval_chunk.yaml`,
`tests/test_volume_eval_chunk.py`) plus an `examples/README.md` section. HEAD unchanged; no reorg
file modified.

## Findings

- **[resolved] Range/factor semantics** — `parse_factor` + `pts_oset=[z,y,x]*factor`; verified by the
  nonzero-offset fixture (`factor=[3,4,5]`, keys from 1).
- **[resolved] chunk-index↔key + output-key consistency** — `chunk_index_to_key`/`chunk_key_to_index`
  round-trip; single-chunk run writes exactly the expected key file.
- **[resolved] LUT node-order alignment** — `init_workflow` writes `pts=graph.get_nodes_position(None)`;
  every worker reads it; proven by full-LUT `assert_array_equal` vs monolithic.
- **[resolved] Score path** — `reduce_and_score` uses `score_graph_with_lut`.
- **[resolved] Verification strength** — full per-node LUT equality before ERL parity.
- **[resolved] Wait policy** — `wait_for_completion` with stall/timeout raises `WaitTimeoutError` naming
  the missing chunk; testable offline.
- **[resolved] Coercion / seg_oset** — list/`start,stop,step` ranges, dotted-nested overrides, factor list.
- No new findings.

## Tests to Add

None. The offline suite covers full-LUT equality (serial+parallel), index/key mapping, wait stall,
reduce length, sbatch script generation, and config normalization. SLURM launch is verified by
script-string inspection (the real submit is documented, not run).

## Questions

None.

## Verdict

VERDICT: APPROVE
