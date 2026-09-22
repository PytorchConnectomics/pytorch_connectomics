# CCC Run

## Description

Implement a typed **Segmentation Checkpoint DSL** — `DESCRIBE -> CERTIFY -> ACT -> VERIFY` — with
one operator, `nucleus_anchor`, that turns nucleus-instance anchors into descriptive observations,
a deterministic identity-conflict certificate, voxel-level split + cannot-link actions, and
instance-level verification, exported for a later global agglomeration pass.

The full specification is `task.md` (unchanged from the requester). The coordinator appended a
"Run context" section pointing at `dev/zebrafinch/lesson_abiss.md` **L126**, which carries the
measurements the design rests on — the ~0.999 affinity bottleneck, the single supervoxel holding
100.0% of the contamination, and the three in-pipeline attempts that each measured exactly 0.0000.

Two structural facts the planner must not lose: the discriminating test is a **voxel-level** split
(region-graph edge rejection provably cannot fix an intra-supervoxel fusion), and **exclusion must
precede anchoring** or the consolidation step re-fuses everything it just separated.

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

Build `checkpoint/` (schema, engine, registry, actions, verification, serialization, cli,
operators/) in the tracked tree, lifting the reusable numerical split/anchor code out of the
gitignored `dev/zebrafinch/` scripts without changing their CLI behaviour. Deliver a
plan/apply/verify/run CLI, twelve named tests including idempotence, scope-safety, determinism and
minority-contamination detection, and a design document.

## Git Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: plan_v2
latest_artifact: artifacts/plan_v2.md
latest_verdict: NEEDS_CHANGES
next_action: code_v0

## Status

active
