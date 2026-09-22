# CCC Run

## Description

Find ways to speed up computation and reduce the memory footprint of `lib/abiss`, grounded
in the measurements already collected in `dev/zebrafinch/lesson_efficiency.md` (L117) and
the whole-volume Seuron reproduction runs. Segmentation output must stay bit-identical:
this pipeline reproduces a Seuron provenance record.

Git baselines target the **`lib/abiss` repository**, not pytorch_connectomics, because
`lib/` is gitignored by the parent and abiss changes appear in no parent diff.

## Runtime

planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: persisted

## Rounds

plan_rounds: 2
revision_rounds: 2

## Approvals

2026-07-29, user (weidf), verbatim direction: "approve dependencies and continue to code".
Satisfies AGENTS.md:104 for adding abseil / mimalloc / jemalloc as BUILD dependencies for
lib/abiss. Scope of approval as recorded: dependency installation is unblocked and the run
advances to the code stage. Installs must use isolated prefixes per plan_v2 Phase C.2 and
must not modify the shared `pytc` conda env or `base`.

## Resume 2026-07-30

Hold condition SATISFIED: SLURM 2787008..2787013 all COMPLETED (143 shards, zero failures).
Whole-volume masked reproduction scored canonical NERL 0.355796 (reference 0.385291).

Resumed in `normal` mode. Per the standing constraint, code_v0 runs against an ISOLATED
COPY of lib/abiss at run_start_ref, never the live checkout:

    work/abiss   (git clone of lib/abiss, detached at 3c4f562, clean, no build/)

The live lib/abiss is not touched by this run. Code-stage diffs are taken inside work/abiss.

## Superseded Hold

2026-07-29, user (weidf): "hold it until the whole-volume job finishes".

The run is intentionally parked. Do NOT start code_v0 until SLURM 2787008..2787013
(whole-volume masked Seuron reproduction) reaches a terminal state. That chain executes
binaries from lib/abiss/build/, and rebuilding that directory is what broke the previous
attempt (see Incident below).

Resume condition: 2787008..2787013 all COMPLETED (or explicitly abandoned).
Resume command:  /ccc resume .agent/features/abiss_speedup/

On resume, code_v0 must run against a COPY of lib/abiss, not the live checkout, so a
rebuild can never again collide with a running job. The dependency approval recorded above
still stands; the abseil REJECTION from the read-only audit also still stands and is
preserved in state/code_v0.last_message.md.

## Incident — code_v0 broke a running production job

2026-07-29. The code_v0 stage was run with `--sandbox workspace-write` over `lib/abiss`
WHILE the whole-volume Seuron chain (SLURM 2785292..2785305) was executing binaries from
`lib/abiss/build/`. Codex reconfigured and rebuilt that directory; `build/agg` was
transiently absent, so all 16 `me_L2` shards failed with "No such file or directory" and
`me_L3`, `me_L4`, `me_L5`, `remapagg` were cancelled on dependency.

Cause: coordinator sequencing error, not sandbox scope. plan_v2 itself said Phase 0 work
should be sequenced after the production job; the coordinator ran the code stage
concurrently anyway.

Recovery performed:
- codex killed; `CMakeLists.txt` change saved to `state/code_v0_partial_CMakeLists.diff`
  and reverted; `build/` wiped and rebuilt clean at 3c4f562 with `-DEXTRACT_SIZE=ON`.
- Restored build verified equivalent to pre-incident: EXTRACT_SIZE present, no
  USE_MIMALLOC/USE_ABSL_HASHMAP, no allocator linked, TBB on agg/meme/ws2, march=nocona.
- 176 DONE flags written after the mimalloc build were invalidated (recorded in
  `state/invalidated_done_flags.txt`) because those chunks may have been produced by
  differently-built binaries; mixing builds within one run would void bit-identity.
- Chain resubmitted from me_L1: SLURM 2787008..2787013.

Standing constraint for the remainder of this run: build experiments must NOT touch
`lib/abiss` in place. Use a separate copy of the repo.

## Superseded Blocking Reason

plan_v2 is the terminal plan version under p2. plan_v1_review returned NEEDS_CHANGES with
major findings and no plan-review round remains, so per the normal-mode table this is a
human decision point. Two blocking decisions are also internal to plan_v2 and cannot be
resolved by the planner:

1. Dependency approval for abseil / mimalloc / jemalloc, required by AGENTS.md:104
   ("Do not add runtime dependencies unless explicitly approved"). Phase C is gated on it.
2. Node-ISA inventory, required before any non-empty ABISS_ARCH. Default stays empty.

Phase 0 (profiling), Phase A (ranking), and Phase B (build flags) need no new dependency
and can proceed on approval to continue.

## Task Summary

Identify and rank concrete speed/memory opportunities in `lib/abiss`, each with mechanism,
payoff estimated against the measured stage table, fidelity risk, and verification.

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff

## Workflow State

current_stage: review_v1
latest_artifact: artifacts/review_v1.md
latest_verdict: APPROVE_WITH_MINOR_COMMENTS
next_action: complete

## Status

complete
