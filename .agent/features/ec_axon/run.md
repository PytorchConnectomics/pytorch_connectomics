# CCC Run

## Description
Refactor the validated axon-decode pipeline (dev/mit_liconn) into connectomics/decoding/decoders/ as
first-class graph ops, chain the stages via a decoding-graph YAML, and add tutorials/neuron_axon/ comparing
naive waterz vs the axon-specific decoding. Supersedes the earlier method-doc run (archived in
_superseded_method_doc/).

## Runtime
planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: explicit

## Rounds
plan_rounds: 6   # extended by user amendments: (1) remove dev dependency, (2) consolidate with existing branch_split/branch_merge
revision_rounds: 2

## Task Summary
Vendor v1 split (link_cut_change + confident_split), v2 merge (completion + IoU-primary + margin), v3 weak,
optional v4 behind prefer_length, plus the perf helpers (seg_stats/apply_lut) and the GT-free completeness
metric. Register as decoding.graph ops; express the pipeline as a YAML DAG; tutorial compares naive waterz
(0.6530/0.7580) vs axon decode (v3: 0.8434/0.9525) at merge_threshold=10.

## Git Baseline
run_start_ref: 25ed266b7f4da881bf4d529a7c5e4252892f4573
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff
note: repo has large PRE-EXISTING unrelated WIP; review surface scoped to .agent/features/ec_axon/,
connectomics/decoding/, tutorials/. Driver makes no commits.

## Workflow State
current_stage: code_v0_review
latest_artifact: artifacts/code_v0_review.md
latest_verdict: APPROVED
next_action: none (run complete; changes are uncommitted in the working tree)

## Result
Reproduction verified through `scripts/main.py --mode test`, not the coder's own harness:

| decode | NERL base | oracle-merge | voxel diff vs research artifact |
|---|---|---|---|
| naive waterz | 0.653011 | 0.758010 | 0 (`decoded_waterz_large_test.h5`, 1223 labels) |
| staged branch decode | 0.843386 | 0.952493 | 0 (`v3_weak.h5`, 22071 labels) |

Reviewer fixed 5 defects in the delivered code (dead 3.35 GB affinity mean in `branch_link`, unused
full-volume bincount, 6 dangling cue-ladder `fn:` pointers + stale merge.py header, unexported
`metrics/{oracle,completeness}`, stale CLAUDE.md map) and recorded 5 notes. SLURM: 2774727 (waterz),
2774728 (staged), 2774741 (staged re-run on the post-edit tree — same 0.843386/0.952493, 0 differing
voxels vs both the research artifact and the pre-edit output). A 5-dimension adversarial review
workflow (25 raw findings, refute-by-default) confirmed one further defect, fixed: experiment_log.py
logged the DELETED decoders' parameters, so graph runs emitted 6 dead columns and none of the
vendored gates.

## Status
complete
