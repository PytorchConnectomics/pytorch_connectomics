# CCC Run

## Description

Snap watershed supervoxels onto the nucleus instance mask and split the ones spanning more than
one nucleus, at the watershed stage, so ABISS's nucleus constraint finally operates on supervoxels
that each belong to exactly one nucleus.

This is a separate feature folder from `abiss_nucleus`, whose run 2 (the global nucleus table)
remains open at `code_v1`. Both share the `work2/abiss` clone, symlinked here; run 2's completed
working-tree change is this run's baseline and is recorded in `state/run_start.diff`.

The decision to work at the watershed rather than with a region-graph split is measured, not
stylistic: the whole contamination sits inside a single supervoxel, and region-graph nodes are
supervoxels. See `task.md` and `../abiss_nucleus/state/evidence_conflict.md`.

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

At the atomic watershed stage, relabel every supervoxel overlapping nucleus N to a deterministic
N-derived id (so ABISS's ws remap stitches N globally for free and cross-nucleus fusion becomes
inexpressible), and split any supervoxel overlapping two or more nuclei with a seeded watershed
restricted to that supervoxel. Must run before `chunkmap` is generated, and must leave the
no-`NUC_PATH` path byte-identical.

## Git Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: review_v0
latest_artifact: artifacts/review_v0.md
latest_verdict: NEEDS_CHANGES
next_action: code_v1

## Status

active
