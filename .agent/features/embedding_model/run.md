# CCC Run

## Description
Add a mean-shift-style embedding head (e.g. 12 channels, Lee et al. / DeepEM mean loss) alongside the 6-channel affinity output in pytorch_connectomics, check prior NISB lessons for failed attempts, then launch a BANIS+-like training on nisb-base.

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
See task.md. (a) read Kisuk embedding wiki + DeepEM mean.py loss; (b) 6-aff + N-ch embedding model; (c) review nisb_base lessons and dev/nisb for failed attempts; (d) kick off BANIS+-like training on nisb-base.

## Git Baseline
run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State
current_stage: review_v1
latest_artifact: artifacts/review_v1.md
latest_verdict: BLOCKER
next_action: launched; awaiting training job 3014439 and its 5k/10k watchdog

## Status
launched

## Human Decisions
- 2026-09-15: plan_v2_review unresolved [major] (watchdog non-finite handling) resolved by user choice "Cancel + report": non-finite gate value => immediate verified scancel, exit 8, never PASS. Binding amendment: state/human_decision_plan_v2.md. Run resumed to code_v0 on the approved plan_v2 + amendment.
- 2026-09-15 code_v0 infra incidents (coordinator): (1) coder's multi-file `black` deadlocked in the codex bwrap sandbox 11:07-14:32 (killed); (2) gpt-6-astra "at capacity" made session 01a0a592 unresumable; code_v0 finished by a fresh codex session auditing the draft (state/code_v0.continue.prompt.md, log state/code_v0.codex.continue.log). HEAD unchanged throughout.
- 2026-09-16: review_v1 BLOCKER (gate d, 1.253x) resolved by user choice "Launch at 0.3, watchdog decides": submit the 200k run at the rule-selected weight and let the 5k/10k validation-ratio watchdog auto-cancel on a persistent regression. No code change; the failing gate is recorded as an accepted deviation in the prereg.

## Launch
- train job: 3014439 (`slurm_jobs/nisb_banis_plus_embed12_train.sbatch`, seed 43, 4 GPUs, partition long, 200k steps), submitted 2026-09-16 17:0x.
- queued behind cluster-wide maintenance reservation root_67 (2026-09-18 08:00-16:00); a 5-day request cannot fit in the 39h before it, so expected start is after 2026-09-18 16:00.
- logs: `slurm_jobs/logs/nisb_banis_plus_embed12_train_3014439.{out,err}`; run dir will appear under `outputs/nisb_base_banis+_embed12_seed43/<timestamp>/`.
- prereg: `dev/nisb/research_plan/prereg_nisb_embed12.md` (status log updated 2026-09-16).
- unit gates re-run on current HEAD 652b9795 (run baseline 1546f47e had moved under unrelated commits): 92 passed, 2 CUDA-only skipped on CPU, both green on GPU in review_v1.
- the embedding changes are still uncommitted working-tree edits in pytc; HEAD drift is a standing risk to this run.

- 2026-09-17 01:19 status check: 3014439 still PENDING (Reason=ReqNodeNotAvail, reserved for maintenance), Slurm StartTime 2026-09-18T16:00:00, EndTime 2026-09-23T16:00:00; 3014447 PENDING on `after:3014439`. pytc HEAD unchanged at 652b9795 since the gate re-run; all embedding working-tree edits and the git-excluded yaml/sbatch/watchdog files verified present. Snapshot of the 9 uncommitted files taken at `artifacts/worktree_snapshot_20260917_0119/` so a stray checkout/stash by another pytc session is recoverable before the job starts.

## Resolved Blocker
resolved 2026-09-16 by user decision "launch at 0.3, watchdog decides". review_v1 BLOCKER was: smoke gate (d) affinity interference 1.253x (bar 1.10) at the rule-selected weight 0.3; the one-shot gradient-ratio rule is spent. Human decision required among: launch at 0.3 with the 5k/10k watchdog, lower weight, code_v2 warm-up ramp, or stop. All other gates pass (speed 1.063x, 94/94 unit tests, memory 1.25GB). Gate (d) is an accepted, recorded pre-launch deviation; the 5k/10k watchdog now carries the kill decision.
