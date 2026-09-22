# CCC Run

## Description

Design MESA-EM (from the 2026-07-14 deep-research reply) for the NISB `base_banis+`
0.60→0.80 NERL problem and implement + run the cheap oracle (Phase-0 GT-channel
plumbing) and banis+ small-chunk feasibility (Phase-1 candidate recall) gates on the
standard center chunk before any full/training experiment. Planner: Claude. Coder:
Codex. Deliverables are new untracked research files under
`dev/nisb/` plus this run folder; no framework code is edited and nothing is committed.

## Runtime

planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default

## Rounds

plan_rounds: 3
revision_rounds: 2

Plan budget extended p2→p3 by explicit human decision (2026-07-15) at the plan-cap block:
"extend planning first". Human also ratified G0 = proven-ancestor 1-voxel oracle (not the
reply's full medial-band Phase-0); plan_v3 must be reviewed under that constraint, not
re-litigate it.

## Task Summary

See `task.md`. Two explicit user asks: (1) design the model + experiments from the
MESA-EM reply; (2) try the oracle and a banis+ feasibility check on smaller chunks
BEFORE kicking off the whole experiment. Go/no-go by the reply's Phase-0/Phase-1
kill gates, quoting deltas off realized 0.627.

## Git Baseline

Operative repository: `/projects/weilab/weidf/lib/pytorch_connectomics` (main working
tree; holds both `.agent/features/` and the runnable `dev/nisb/` workspace with data +
harness). The coordinator session runs from a clean sibling worktree, but all run
artifacts and code target this main tree; it carries pre-existing unrelated dirt
(24 tracked files) recorded below so our (untracked) additions stay distinguishable.

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

Note: MESA-EM deliverables are new untracked files (`dev/nisb/scripts/mesa/*`, this
run folder). Tracked content must not change; code review compares embedded file
contents plus the `git status --short` delta against run_start, since the review
surface is untracked research code rather than a tracked diff.

## Workflow State

current_stage: none
latest_artifact: artifacts/review_v1.md
latest_verdict: APPROVE
next_action: complete

CCC run complete. Plan+code loop finished (review_v1 APPROVE). Coordinator gate verification ran the
full center-chunk G0+G1 gates (2 documented G1 hotfixes for clean-chunk assertion bugs). Consolidated
findings + go/no-go in artifacts/RESULTS.md. Headline: MESA-EM representation VALIDATED at the decode
level (G0 0.9936 with GT-correct edges); NO-GO for training the full head stack yet -> (1) make the
skip-edge head Z-aware (G0 C2=45%), (2) make candidate generation crumb-aware (67% of frags <12 vox
get no tips), (3) re-bench G1 on split-dominated tile_0_0_2 (center chunk only +0.037 split headroom),
(4) learned PairNet/BiLOQ required (geometry-only 79% prec / 7 merges).

Plan phase closed at p3. plan_v3_review APPROVED the ratified 1-voxel G0 oracle and confirmed
all prior correctness/scope fixes; 4 narrow G1/DESIGN implementation-spec items remained.
COORDINATOR DECISION (documented, not a silent override): rather than block again on
coder-resolvable specs after planning was already extended once, proceed to code_v0 carrying
plan_v3 PLUS `state/code_spec_addendum.md` (precise resolutions of all 4 items) as mandatory
acceptance criteria; the code review (claude) verifies them. Code stage writes the scripts +
DESIGN.md and runs only the 256^3 smoke; the coordinator runs the full center-chunk gates in
the background after review (execution-model boundary in the addendum).

## Status

complete
