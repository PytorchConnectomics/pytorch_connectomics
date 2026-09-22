# CCC Run

## Description

Add a generic chunked-volume ERL orchestrator `examples/volume_eval_chunk.py` that launches
SLURM array jobs or local multiprocess workers to compute the node-segment LUT for
extra-large segmentations stored in chunks, waits for the partial results, combines them, and
scores ERL. Modeled on `waterz decode_large` ergonomics; kept lighter because LUT computation
is embarrassingly parallel (map → reduce → score, no cross-chunk stitching).

The changed files live in the nested `lib/em_erl` git repo (the CCC review surface). This run
is sequenced AFTER the completed+approved `em_erl_reorg` run; that reorg's changes are the
uncommitted substrate captured in this run's `run_start.diff`, so the review surface isolates
only the new orchestrator work.

## Runtime

planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default

## Rounds

plan_rounds: 2
revision_rounds: 2

## Task Summary

`examples/volume_eval_chunk.py`: config-driven generic launcher around em_erl's existing
`compute_segment_lut_tile_zyx` (per-chunk map) + `combine_segment_lut_tile_zyx` (reduce) +
`compute_erl_score` (score). CLI mirrors decode_large (`--init-only/--chunk-index/--chunk-range/
--parallel/--sbatch/--local/--wait/--reduce/--score`); file-presence-based completion; SLURM
`--array` backend; multiprocess backend.

## Git Baseline

run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

Baseline repo: lib/em_erl (nested git repo). NOTE: run_start.diff captures the completed,
approved em_erl_reorg working-tree changes as the substrate; the chunk review surface is the
delta on top of that.

## Workflow State

current_stage: review_v0
latest_artifact: artifacts/review_v0.md
latest_verdict: APPROVE
next_action: complete

## Status

complete
