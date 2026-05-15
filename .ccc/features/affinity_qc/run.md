# CCC Run

## Description

Code review and follow-up update of the affinity-QC subsystem
(`connectomics/decoding/qc/affinity.py` plus its hooks in chunked inference
and the decoding stage). The implementation already exists in the working
tree; this run audits it and applies any required corrections.

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

Review the affinity-QC implementation across schema, decoding module,
chunked inference hook, training-stage wiring, preflight validation, and
unit tests. Identify correctness, boundary, API, failure-mode, coverage,
and quality issues. Apply surgical fixes.

## Git Baseline

run_start_ref: fa8fce67202394ab8b44d0f37857daabd8ba0a68
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: none
latest_artifact: artifacts/review_v0.md
latest_verdict: APPROVE_WITH_MINOR_COMMENTS
next_action: complete

## Status

complete
