# CCC Run

## Metadata

protocol_version: 1

## Description

Brainstorm and rank ideas to push NISB-base NERL above the
24%@50k / 32%@200k baseline of `tutorials/neuron_nisb/base_banis.yaml`.
Output is a ranked plan-of-experiments, not a single chosen approach.

## Roles

a1: Claude Code (this session)
a2: peer Claude Code or Codex session

## Rounds

plan_rounds: 2
revision_rounds: 2

## Task Summary

See `task.md`. The user already named several candidate ideas
(MedNeXt-L, multi-head, waterz/SDT, GT erosion) and asked for a
broader brainstorm informed by `tutorials/*`. Many of those ideas are
already scaffolded as variant configs; the plan also surfaces ideas
not yet present in the codebase.

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: code_v2
expected_role: none
latest_artifact: artifacts/code_v2.md
latest_verdict: NEEDS_CHANGES
next_waiting_for: max-rounds-reached

## Status

max-rounds-reached
