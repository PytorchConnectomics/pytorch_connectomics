# CCC Run

## Description
decode_v2 for LiCONN DL288B: integrate tube-bb base decode + false-merge error detection +
seeded-watershed force-split (waterz-propagated seeds) + relink, into a runnable
`dev/mit_liconn/decode_v2.py`. Claude plans and reviews code; Codex reviews the plan and codes.

## Runtime
planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: default

## Rounds
plan_rounds: 3
revision_rounds: 2

## Task Summary
Build `decode_v2` = tube-bb (thr-0.3 sections → conservative + best-buddy links, no z-contact,
no fill) → detect fused merge cross-sections (N-to-1 area-matched) → seeded-watershed force-split
with propagated seeds → relink pieces to their tubes without re-merging. Evaluate by NERL and
oracle-merge NERL. Reuse dev/mit_liconn helpers. Beat tube-bb NERL 0.593 or expose relink as the
knob with oracle-merge ceiling ≥0.78 documented.

## Git Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State
current_stage: review_v1
latest_artifact: artifacts/review_v1.md
latest_verdict: APPROVE
next_action: complete

## Status
complete
