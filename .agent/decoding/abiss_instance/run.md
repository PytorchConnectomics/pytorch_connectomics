# CCC Run

## Description

Refactor and improve the ABISS competitive-nucleus-instance feature for efficiency and
robustness, driven by the design and review document already present at
`.agent/decoding/abiss_instance/task.md`.

`task.md` was authored before this run started (2026-08-17 13:36, 17,725 bytes) and is the
substantive input the task string points at ("read task.md and ..."). The coordinator therefore
**preserved it instead of overwriting it**; the invocation's task string is recorded under
`## Task Summary` below. Nothing in `task.md` was modified by run initialization.

Run initialized 2026-08-17 as blocked (Codex was over quota). **Unblocked on resume the same
day**: `codex exec` now returns exit 0, so the intended cross-model `claude-codex` run proceeds.
`## Block Details` is retained as the record of that episode, including the PATH fix the codex
CLI needs.

## Runtime

```text
planner: claude
coder: codex
plan_code: claude-codex
session_detected: claude
plan_code_source: persisted
```

## Rounds

```text
plan_rounds: 3
revision_rounds: 4
```

## Task Summary

Invocation:

```text
/ccc .agent/decoding/abiss_instance/ "read task.md and refactor and improve the nuc instance
feature. more efficient and robust"
```

The full task specification is `task.md` in this folder. In brief: the competitive nucleus
growth feature works and has a measured positive NERL effect on two independent affinity
substrates (+0.012100/+0.019505 on native96 at mt50/mt0; +0.023654/+0.014456 on win144), but

1. `nuccomp` consumes **440.6 of 500.6 minutes = 88%** of post-watershed wall-clock on win144
   and **11h18m42s** on native96, single-node and serial, to repair 8-9 independent units;
2. the stage has **no phase timing**, so those hours cannot be attributed between the global
   scan and the per-unit floods;
3. two production bugs have already escaped the test suite, one of them a graph-versus-volume
   divergence that every natural check still passed;
4. six of the seven acceptance checks the feature's own lesson file requires are unrun.

## Git Baseline

```text
run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7
run_start_ref_kind: head
run_start_status_file: state/run_start.status
run_start_unstaged_diff: state/run_start.diff
run_start_staged_diff: state/run_start_cached.diff
```

The working tree was **not clean** at run start: 86 entries in `git status --short`, 211,183
bytes of unstaged diff and 36,053 bytes of staged diff, captured in the files above. Much of it
is this session's evaluation-standard and lessons work. Any later code review must diff against
`run_start_ref` and treat that pre-existing delta as baseline, not as this run's output.

## Workflow State

```text
current_stage: review_v4
latest_artifact: artifacts/review_v4.md
latest_verdict: APPROVE_WITH_MINOR_COMMENTS
next_action: complete
```

## Block Details

Reason: **both configured model providers are out of quota, so no cross-agent stage can run.**

```text
codex   `codex exec` exits 1:
        "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage
         to purchase more credits or try again at Aug 20th, 2026 10:54 PM."
        Account: weiddoonngglai@gmail.com, plan prolite, auth_mode chatgpt.
        Separately, the codex CLI is a node script and `node` is absent from the
        non-interactive PATH; it runs only with
        PATH=/home/weidf/.nvm/versions/node/v24.9.0/bin:$PATH  (exit 127 without it).
        That PATH fix is confirmed working -- the quota error above is from a
        successfully launched CLI, not from the missing runtime.

claude  Anthropic account weidf@bc.edu hit its monthly spend limit; parallel subagents and
        workflows fail with "You've hit your monthly spend limit". `claude --print` itself
        still succeeds (ccc-check-agent-cli.sh claude passes), so in-session and single
        CLI calls currently work while fan-out does not.
```

`plan-code=claude-codex` needs `codex` for `plan_v0_review` and `code_v0`, which is a hard
failure per the protocol's "invalid CLI or auth state". The run folder, git baseline and this
file are initialized so `/ccc resume .agent/decoding/abiss_instance/` continues cleanly.

Options, for the operator to choose:

1. **Wait for Codex** (available 2026-08-20 22:54) and then
   `/ccc resume .agent/decoding/abiss_instance/`. Keeps the intended cross-model review, which
   is most of the value here: `task.md` was written to be checked by GPT-5.6, and the feature's
   failure mode is subtle enough that a second model is worth waiting for.
2. **Raise the Codex limit** at https://chatgpt.com/codex/settings/usage, then resume. Note the
   ChatGPT subscription itself lapses 2026-08-23 23:43 UTC, six days out.
3. **`/ccc resume .agent/decoding/abiss_instance/ plan-code=claude-claude`** — every stage is
   then owned by the current session's agent and runs in-session, needing neither the codex CLI
   nor subagent fan-out, so it works today. The protocol calls this
   "protocol-discipline-only": it preserves artifact structure, validation and bounded rounds
   but provides **no cross-model review**. For a robustness audit whose whole point is a second
   opinion, that is a real loss, not a formality.

The coordinator did not silently switch `plan-code`; selection is the operator's.

## Decision Point (plan_v2 under p2) -- RESOLVED 2026-08-17 by resuming with p3-c2

`p2-c2` expands to `plan_v0 -> plan_v0_review -> plan_v1 -> plan_v1_review -> plan_v2`. There is
no `plan_v2_review` stage, so `plan_v2` is the terminal plan artifact and, per the protocol, a
decision point rather than an automatic approval in `normal` mode.

State: the last recorded verdict is `NEEDS_CHANGES` (`plan_v1_review.md`, 12 major + 2 minor).
`plan_v2` states that it closes all ten items left open there, and its
`## Changes Since Previous Plan Version` maps each one to a specific section. **No reviewer has
confirmed that.** The coordinator will not self-certify that its own plan resolved the reviewer's
findings; the protocol's rule against silently softening material findings applies to the planner
as much as to the summary of a review.

Progress this run: 5 stages complete (`plan_v0`, `plan_v0_review`, `plan_v1`, `plan_v1_review`,
`plan_v2`), two cross-model review rounds, 23 major findings raised and addressed across
revisions. Codex CLI works via
`PATH=/home/weidf/.nvm/versions/node/v24.9.0/bin:$PATH`; the mutation guard was clean on both
review calls; `HEAD` never moved from `run_start_ref`.

**Resolution: the operator chose option 2** -- resumed with `p3-c2`, which adds `plan_v2_review`
so Codex adjudicates whether the ten items are closed rather than the coordinator self-certifying.
The options as presented were:

1. **Approve `plan_v2` and proceed to `code_v0`.** Reasonable: every finding is mapped to a
   change, and the central objection (measuring efficiency without improving it) is resolved by
   selecting the per-unit flood array on robustness-plus-speed grounds. Resume and say to proceed.
2. **Allow another plan round** — `/ccc resume .agent/decoding/abiss_instance/ p3-c2` — which
   adds `plan_v2_review` so Codex actually adjudicates whether the ten items are closed. This is
   the option that preserves the cross-model guarantee the run was set up for.
3. **`auto` mode** — `/ccc resume .agent/decoding/abiss_instance/ auto` — proceeds past the
   unresolved disagreement with `VERDICT: APPROVE_AUTO_OVERRIDE` recorded in the artifact.

Also outstanding, independent of the choice: `plan_v2` §Risks Q-1 asks the operator to confirm
that `code_v0` implements and unit-tests while the **coordinator or operator** runs the ~7-11 h
cluster baseline, since a CLI call cannot wait for it.

## Diff-Surface Decision (recorded 2026-08-17, before code_v0)

The run started with a dirty tree: 211,183 bytes of unstaged diff against the protocol's
200,000-byte `CCC_REVIEW_PROMPT_MAX_BYTES`. A by-the-book `review_v0` prompt includes
`git diff <run_start_ref>...HEAD` and `git diff`, so it would exceed the ceiling on the
pre-existing baseline alone -- before any of code_v0's output -- and the protocol's response to
that is `Status: blocked`.

Overlap check between the plan's declared files and `state/run_start.status`:

```text
clean    lib/abiss/scripts/nucleus_competition.py
clean    lib/abiss/scripts/nucleus_overlay.py
clean    dev/zebrafinch/sbatch_nucleus_competition.sh
clean    dev/zebrafinch/submit_wholevol_sharded.sh
clean    dev/zebrafinch/nucleus_acceptance_report.py
OVERLAP  tests/unit/test_abiss_nucleus_competition.py
```

**Operator chose option 1: scoped-diff review.** `review_v0`'s prompt will be assembled from
`git diff -- <the files plan_v3 declares>` rather than the whole working tree. This is
protocol-compatible -- the coordinator assembles the prompt and the protocol already notes that
integrity depends on it doing so completely -- and it moves no commits and no `HEAD`.

The narrowing and its cost must be stated in `review_v0.md`: the reviewer sees only the declared
surface, so an edit made outside it would go unreviewed. The coordinator will therefore also run
`git status --short` before and after `code_v0` and report any file changed outside the declared
list, so the narrowing cannot hide a stray edit.

## Diff-Surface Correction (2026-08-17, after code_v0)

The scoped-diff plan recorded above rests on a false premise, discovered when code_v0 landed. Of
the six declared files, only two produce a git diff at all:

```text
lib/abiss/ is a SEPARATE git repository (its own toplevel). Changes to
  scripts/nucleus_competition.py and scripts/nucleus_overlay.py are invisible to the parent
  repo's git status/diff, and were therefore absent from run_start.diff. The run's git baseline
  was incomplete from the start -- a coordinator error, not a coder one.
dev/ is gitignored (.gitignore:159), so sbatch_nucleus_competition.sh, sbatch_nuccomp_flood.sh,
  submit_wholevol_sharded.sh and nucleus_acceptance_report.py yield no diff.
tests/unit/test_abiss_nucleus_competition.py is UNTRACKED (??), so it yields no diff either.
```

This does not block the run, because `review_vN` is **planner-owned** and the planner is this
session: the review is performed in-session by reading files directly, so the 200,000-byte
cross-agent prompt ceiling never applies. The ceiling would have bound only a codex-owned review.

Attribution of code_v0's edits was therefore established by mtime rather than by git:

```text
code_v0 (2026-08-17 14:58-15:14):
  lib/abiss/scripts/nucleus_competition.py      15:08
  lib/abiss/scripts/nucleus_overlay.py          15:08
  tests/unit/test_abiss_nucleus_competition.py  15:12
  dev/zebrafinch/sbatch_nuccomp_flood.sh        14:58  (new)
  dev/zebrafinch/sbatch_nucleus_competition.sh  14:59
  dev/zebrafinch/submit_wholevol_sharded.sh     14:59
  dev/zebrafinch/nucleus_acceptance_report.py   15:08
pre-existing in lib/abiss (2026-08-16, the earlier codex thread -- NOT this run):
  CMakeLists.txt, scripts/composite_chunk_me.sh, scripts/cut_chunk_agg.py,
  scripts/cut_chunk_remap.py, src/agg/mean_aggl.cpp, src/seg/match_chunks.cpp,
  tests/test_nuc_agg.py, tests/test_nuc_match.py, build/
```

Codex stayed inside the declared file list. Two outputs were misplaced at the repository root
(`artifacts/code_v0.md`, `docs/nucleus_competition_review.md`) instead of the run folder that
plan_v3 §D specified; the coordinator relocated both and left the root clean.

## Stage Recovery and Incident (2026-08-18)

Two things were found on resuming this run after an 8-hour gap.

**`code_v1` had completed 22 hours earlier and was never claimed.** Codex wrote its last message to
`/tmp/code_v1.raw` (14,750 bytes) at 2026-08-17 15:52. The owning session polled it afterwards and
reported "still RUNNING" at 23:20 and again at 2026-08-18 01:18, and on that belief declined to
launch the `ABISS_NUC_MIN_TAGGED` sweep because the scripts were thought to be mid-edit. They were
not; `nucleus_competition.py` and `nucleus_overlay.py` were last written at 15:42. The coordinator
validated the artifact against the `code_vN.md` contract (all ten sections, `run_start_ref` and
`current_head` both `c705458a`), reproduced its central claim (`pytest
tests/unit/test_abiss_nucleus_competition.py -q` -> 22 passed in 1.67 s) and confirmed all six
newly claimed deliverables exist, then landed `artifacts/code_v1.md` and `state/code_v1.done` at
2026-08-18 14:03.

**The strict overlay introduced by `code_v0` destroyed an unrelated live cluster array.** Job
2872438 (`me_L0`, 80 tasks over `dev/zebrafinch/aggsweep/b3_gate025`, submitted 13:40 on
2026-08-17, started 14:42) reads the schema-1.2 native96 manifest through
`cut_chunk_agg.py` -> `nucleus_overlay.py`. 77 of its 80 tasks show a successful manifest load
followed by `ValueError: nucleus competition manifest lacks a valid completion marker`, with no
successes afterwards; none failed from their first chunk. Task 0 completed 89 chunks before the
first rejection. Roughly 70 tasks then hit the 12-hour wall (TIMEOUT) and the downstream chain
`me_L1`-`me_L5`, `remapagg` (2872439-2872444) and `mgrag` (2872596) was cancelled: about 840
node-hours, and the `b3_gate025` arm produced nothing. `code_v0.md:28` and `:146` show the
rejection was deliberate; the failure was deploying it by editing a script that live jobs were
already executing. Recorded as `review_v1` finding G1.

The exact write instant is unrecoverable — `code_v1` overwrote the mtimes at 15:42 and `lib/abiss`
has no commit between — so run.md's earlier attribution of `nucleus_overlay.py` to 15:08 cannot be
reconciled with rejections appearing by 14:55. The finding does not depend on resolving it.

**Codex quota has recovered.** `codex exec --sandbox read-only` returns normally (codex-cli
0.145.0), so the block recorded above no longer applies and a codex-owned `code_v2` can run.


## Decision Point (code_v2 under c2) -- RESOLVED 2026-08-18 by extending to c3

`c2` makes `code_v2` the terminal code artifact, so no `review_v2` stage exists and, per the
protocol, this is a decision point rather than an automatic approval in `normal` mode.
`latest_verdict` remains `NEEDS_CHANGES` because it is the last recorded reviewer verdict
(`review_v1`); no reviewer has passed judgement on `code_v2` itself.

The coordinator will not self-certify its own review directives as satisfied. It did, however,
verify `code_v2`'s material claims independently rather than accept them:

```text
pytest tests/unit/test_abiss_nucleus_competition.py   -> 26 passed (was 22 at code_v1)
G2  parent-winner rule                                -> absent from BOTH former sites
G4  canonicalize_qualified_segments at :446 now runs BEFORE `if not repairs:` at :447
G5  realization gate reproduced by the coordinator, two ways:
      native96  units 8/8   owners 15/15
      win144    units 0/9   owners 0/17
    (independently recomputed from each run's own nucleus_shell_contamination_tol0.json
     joined to its own manifest, before running the tool -- same answer both ways)
G1  migration tested on a COPY at /tmp/migtest, never on the live publication:
      territory_*.npz aggregate md5 identical before and after
      migration completes in under a second (no flood recomputation)
      production load_validated_manifest REJECTS the original 1.2 manifest, and the message
        now names the exact migrate command with the real path
      production load_validated_manifest ACCEPTS the migrated copy (schema 2.0, 8 repairs)
parent HEAD c705458a and nested lib/abiss HEAD 452efa5f both unmoved
```

Operator decisions binding this round were: scope G1-G5 + tests 1-5 and stay at `c2`;
`residue_disposition: parent_retained` so no published label changes; relaunch `b3_gate025` after
the migration path lands.

Open items, all declared rather than discovered: gates 2-8 and the 7-11 h cluster baselines are
unrun and unclaimed; win144's schema-1.0 ids stay opaque with no lossless migration; the
repository-wide decimal-string audit was out of scope; `black --check` exited 124 on a timeout and
is not claimed as passing. The authoritative migration of the live native96 publication has NOT
been performed -- only the copy was migrated.

**Resolution: the operator chose `c3`.** `revision_rounds` is now 3, which adds `review_v2`
and would allow a `code_v3`. `review_v2` is planner-owned, so Claude reviews Codex's
`code_v2` in-session -- a genuine cross-model review, since the artifact under review is
Codex's implementation rather than the coordinator's own directives.

## Schema Version Decision (recorded 2026-08-18, before code_v4)

The operator asked what the final schema version should be. Answer: **3.0**, and `2.0` is
withdrawn.

`schema_version: "2.0"` denoted two mutually incompatible contracts inside this run. The
`code_v0`/`code_v1` form had no `identity` and no `ledger` and REQUIRED
`emitted.count(parent_id) == 1`; the `code_v2` form REQUIRES `identity` and `ledger` and FORBIDS a
territory emitting its parent id. Same number, inverted rule. A hard-reject MAJOR policy cannot
work if the number does not move when the contract does.

The decision document names 3.0 in three places. Codex extended 2.0 because neither `review_v1`
nor the coordinator's `code_v2` prompt named a version; that omission is the coordinator's.

Measured cost of fixing now: every nucleus manifest on disk is 1.0 or 1.2 -- three at 1.2
(native96 matchguard, and both `abiss_tuning/fixed_tiers/.../z4_nucleus_matchguard{,_win96}`) and
two at 1.0 (arm096 v2, arm2mix_r10 v2). **No 2.0 artifact exists anywhere**, so the rename costs
nothing today and stops being free the moment native96 is migrated or a fresh nuccomp publishes.

`2.1` was rejected: a MINOR must be additive and this inverted a rule. The deferred surface
(`inputs[].semantics`, capability negotiation, `native_id_space`, `id_encoding`) is additive and
lands as 3.1+, so the shipped subset stays policy-consistent. `required_capabilities: []` must
exist in 3.0 itself, since the policy admits a newer minor only when the consumer implements every
listed capability.

Two further schema-1.2 consumers were discovered by the same survey --
`abiss_tuning/fixed_tiers/work/z4_nucleus_matchguard` and `..._win96` -- which would hit the same
rejection `b3_gate025` did and are unblocked by the same migration.

## Status

complete
