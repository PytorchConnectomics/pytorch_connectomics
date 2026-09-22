# CCC Run

## Description

Reorganize the `em_erl` repo: move reusable auxiliary functions out of `scripts/` into
the `em_erl` package (io/sampling/eval), rename `scripts/` -> `examples/` so the scripts
become thin, illustrative usage demos, update `__init__` exports, tests, and docs. The
change is a behavior-preserving relocation.

The changed files live in the nested, gitignored `lib/em_erl` git repo, which is the CCC
git baseline and review surface. The run folder lives under the outer repo's `.agent/`.

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

Move general reusable functions from `scripts/j0126_workflow.py` (CloudVolume opener/URL
normalizer, chunk-binned point sampler, LUT save/load/validate, LUT scoring, skeleton
loader, and a general CloudVolume+LUT eval recipe) into `em_erl/{io,sampling,eval}.py`;
keep dataset-specific glue (j0126 default URL + CLI) in `examples/`; rename `scripts/` ->
`examples/`; update `__init__.__all__`, tests, and READMEs. Behavior preserved.

## Git Baseline

run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

Baseline repo: lib/em_erl (nested git repo).

## Workflow State

current_stage: review_v0
latest_artifact: artifacts/review_v0.md
latest_verdict: APPROVE
next_action: complete

## Status

complete
