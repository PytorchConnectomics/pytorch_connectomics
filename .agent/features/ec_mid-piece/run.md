# CCC Run

## Description

GT-leakage-safe Zebrafinch mid-piece absorption experiment. Tests whether the L126
mid-piece oracle rung (+0.138303 NERL, 1,887 GT 10-49-node pieces) can be recovered by
affinity-weighted assignment of small fragments to clean neuron anchors, under a strict
GT firewall (`gt_free/` proposal side, `evaluation_gt/` evaluator side).

The pre-existing `task.md` in this folder is the authoritative task specification; the
coordinator did not overwrite it. The user instruction was "read task.md and implement it".

Operative repository root is the main checkout
`/projects/weilab/weidf/lib/pytorch_connectomics` (branch `codex/gt-free-tube-analysis`),
not the bridge worktree, because `dev/` and `.agent/` are gitignored and exist only there.
All deliverables land under gitignored `dev/zebrafinch/ec_mid_piece/`, so tracked-file git
diffs are expected to stay empty; code review is served the untracked deliverable files
directly alongside the git outputs.

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

Implement `dev/zebrafinch/ec_mid_piece/` as a bounded diagnostic + error-correction
experiment over the arm0_96 segmentation:

- Phase 0: reproduce canonical arm0_96 baseline NERL and the L126 piece inventory; build a
  GT-free full-population segment inventory and a candidate/anchor topology partition.
- Phase 1: freeze a native-scale face-contact fragment-host candidate edge table with
  verified affinity indexing (no legacy `2 - axis` assumption).
- Phase 2: predeclared assignment-only absorption ablations (contact-area control, affinity
  mean/p90, winner margin, margin+morphology) with risk/coverage sweeps; one-hop first.
- Phase 3 (conditional): multiscale/larger-context affinity corroboration on the same frozen
  rows, only with verified geometry.
- Phase 4: frozen residual manifest with one reason code per unresolved fragment.

Hard constraints: no anchor-anchor unions during absorption, no GT in the honest path, no
new dependencies, no modification of canonical arm0_96 inputs/LUTs/reports, no commits.
Only the honest GT-free row may be compared with FFN 0.538003.

## Git Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

Note: the working tree was already dirty at run start (75 status lines, 4044-line unstaged
diff, 638-line staged diff). Those pre-existing modifications are recorded in the baseline
files above and are not part of this run's review surface.

## Workflow State

current_stage: review_v1
latest_artifact: artifacts/review_v1.md
latest_verdict: APPROVE
next_action: complete

## Status

complete
