# CCC Run

## Description

Implement the zebrafinch global neuron-graph agglomeration design (reconciling
`agglomeration_design_fable5.md` + `agglomeration_design_gpt5-5.md`) one
evaluable step at a time, on top of the existing `decode_v1` base, optimizing
whole-volume `test_50_skeletons` NERL. Claude plans + reviews code; Codex
(gpt-5.6-sol, ultra) reviews the plan + implements.

## Runtime

planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default
codex_model: gpt-5.6-sol
codex_reasoning_effort: ultra
repo_root: /projects/weilab/weidf/lib/pytorch_connectomics

## Rounds

plan_rounds: 5
revision_rounds: 2

## Task Summary

Read the two design docs; reconcile into an ordered sequence of implementable,
independently-evaluable steps; implement the first step with a concrete NERL (or
recall_thick/precision) evaluation on the standard test-50 / chunk_gt_skel
substrate. Reuse existing dev/zebrafinch tooling; do not change the decode_v1
base.

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: review_v0
latest_artifact: artifacts/acceptance_v0.md
latest_verdict: APPROVE_WITH_MINOR_COMMENTS
next_action: complete
note: CCC code workflow COMPLETE — code_v0 approved (code faithful to plan_v5, no defects; 4 minors). Whole-volume acceptance COLLECTED (bymr2tlee, 600/600, artifacts/acceptance_v0.md): the linker FAILS the merge-safety gate — oracle drops ~0.015 (<-eps) at every link-accepting tau (false merges among markerless neurites, which the firewall can't see but the test-50 oracle catches) AND realized base stays ~0 (does not beat the L44 tiling wall). tau0.8 oracle-flat but no-op. HONEST NEGATIVE (plan_v5's honesty clause fired). Firewall/namespace/controls all worked. Next (Step 2, new run): trajectory/multi-face path matching or halo-overlap, not adjacent-face endpoint linking + M1 per-axis radius fix.

## Status

complete
<!-- CCC plan+code+review workflow complete (review_v0 APPROVE_WITH_MINOR_COMMENTS); whole-volume acceptance collected (artifacts/acceptance_v0.md). Step-1 empirical verdict: linker code correct but FAILS the merge-safety gate at whole-volume scale (oracle -0.015, base ~0) — honest negative, confirms L44 tiling wall. Step 2 (trajectory/halo-overlap) is a separate run. -->

