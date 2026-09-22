# CCC Run

## Description

Fix the residual perinuclear-shell fusion in the `NUC_PATH` nucleus constraint by making
agglomeration merge order nucleus-first: a nucleus claims its own shell before neighbouring
nuclei's shells can merge with each other.

This is run 2 in this folder. Run 1 (the NUC_PATH feature itself, 7 plan versions / 5 plan
reviews / 3 code versions / 2 code reviews, Status complete) is archived intact under
`run1_nucleus_mask/` rather than overwritten.

Git baselines target the **`lib/abiss` repository**; `lib/` is gitignored by the parent so abiss
changes appear in no parent diff. Work happens in the isolated clone `work2/abiss` detached at
`312bf54` (= the pushed `feature/nucleus-mask`). The live `lib/abiss` checkout is off limits: it
sits on `main` and carries unrelated uncommitted work in `scripts/volume_backends.py`.

Task-owner decision taken on plan_v1_review question 3: the global per-supervoxel nucleus table
(plan v2) replaces the merge-ordering change the task originally mandated. Both the ordering fix
and plan v1's clause-2 veto are refuted by measurement; see `state/evidence_conflict.md`.

Task-owner instruction: make the instance-mask channel CORRECT first, tune ABISS parameters
afterwards. plan v2's V0 gate asked whether the global table fixes the reported blob; it largely
does not (the supervoxels carrying that contamination resolve consistently). But the
partition-dependence it fixes is a genuine correctness defect in the feature -- 857 supervoxels
in the crop are resolved as different nuclei in different chunks -- and correctness comes first.
Implementation proceeds on that basis. See `state/evidence_conflict.md`.

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

Nucleus-first two-phase merge ordering in `agglomerate_cc`: phase 1 processes only edges with a
PROPER endpoint (descending affinity), phase 2 the rest. Must keep the no-`NUC_PATH` path
bit-identical, and must state and test the reducibility argument that makes a lexicographic
`(has_proper_id, affinity)` key safe under ABISS's chunk-independence guarantee.

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
