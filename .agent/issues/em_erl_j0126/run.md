# CCC Run

## Description

Rewrite `lib/em_erl/scripts/j0126_workflow.py` to compute j0126 ERL by querying the
FFN `ffn_segmentation` CloudVolume directly at ground-truth skeleton vertices,
replacing the four-stage tile-download workflow with a single command that only needs
the skeleton HDF5 path.

Note: the changed file lives in the nested, gitignored `lib/em_erl` git repo. That
repo (not the outer pytorch_connectomics worktree) is the CCC git baseline and review
surface. The run folder lives under the outer repo's `.agent/`.

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

Collapse `prepare-gt -> map-lut -> reduce-lut -> score` into one CloudVolume-backed
command; skeleton (z,y,x) voxel coords index mip0 as `cv[x,y,z]`; sample segment ids
into a node-aligned LUT; reuse `skel_to_erlgraph` + `compute_erl_score`.

## Git Baseline

run_start_ref: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

Baseline repo: lib/em_erl (nested git repo).

## Workflow State

current_stage: review_v1
latest_artifact: artifacts/review_v1.md
latest_verdict: APPROVE
next_action: complete

## Status

complete
