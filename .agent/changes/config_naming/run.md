# CCC Run

## Metadata
protocol_version: 1

## Description
Redesign Hydra config naming for consistency across stage sections, and rework
how `output_path` is derived per mode (train vs tune/test) so input dataset
identity becomes a subfolder rather than a filename token.

## Roles
a1: Claude Code (Opus 4.7 1M, this session)
a2: Codex session (external)

## Rounds
plan_rounds: 2
revision_rounds: 3

## Task Summary
Two coupled concerns:
1. Pick a consistent leaf/sub-block layout across `inference`, `decoding`,
   `evaluation`, and `tune` sections — currently flat scalars and nested
   dataclasses share the same namespace post-flatten.
2. Make `output_path` derivation principled: train mode = timestamped run dir;
   tune/test mode = checkpoint-rooted with per-checkpoint and per-volume
   subfolders, removing dataset-name flattening from filenames.

## Git Baseline
run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State
current_stage: none
expected_role: none
latest_artifact: artifacts/code_v3.md
latest_verdict: none
next_waiting_for: max-rounds-reached

## Status
max-rounds-reached
