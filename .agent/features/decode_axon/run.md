# CCC Run

## Description
Design and implement `dev/mit_liconn/decode_axon.py`: a unified, modular LiCONN axon decoder that
combines the strong/weak foreground split, the decode_v2 2D section→force-split→link pipeline, the
decode_p1p2 3D split-then-merge cleanup (bump-safe + parallel-veto), weak-region fastmorph recovery
with strong↔weak hysteresis, and a final crumb cleanup. GT-free; scored by the valid-tube pseudo-metric
(GT is over-split). Goal: beat decode_p1p2 (valid-tube volume 62.7%, bumps 117, parallel 8).

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
Unified `decode_axon.py`: (a) strong (aff>0.66) / weak (0.3–0.66) fg split; (b) 2D-seg tier
(sections→force-split fused→link) then 3D-seg tier for incomplete tubes (Prob-1 bump-safe split,
Prob-2 parallel-vetoed merge); (d) strong via waterz sections, weak via fastmorph opening+cc3d per
slice; (e) strong first then weak; (f) weak used only to bridge split strong tubes (hysteresis);
(g) crumb cleanup. Reuse validated dev/mit_liconn helpers; primary metric valid_tube_metric.py.

## Git Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State
current_stage: review_v0
latest_artifact: artifacts/review_v0.md
latest_verdict: APPROVE_WITH_MINOR_COMMENTS
next_action: complete

## Status
complete
