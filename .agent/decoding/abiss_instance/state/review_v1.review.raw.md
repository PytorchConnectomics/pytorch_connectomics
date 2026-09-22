# review_v1 raw evidence log (planner-owned, in-session)

Planner is `claude` under `plan-code=claude-codex`, so this stage ran in-session per
CCC_PROTOCOL "Stage Ownership". No CLI subprocess; this file is the raw working record.

## Mutation guard / baseline

    git rev-parse HEAD              -> c705458ae5b907bb9c32a75c63c85c6aad7edec7  (== run_start_ref)
    code_v1 artifact                -> /tmp/code_v1.raw, 14750 bytes, mtime 2026-08-17 15:52 EDT
    contract check                  -> all 10 required sections present exactly once
    Git Baseline in artifact        -> run_start_ref + current_head both c705458a

The coder produced this artifact at 15:52 on 2026-08-17. The coordinator never validated it into
`artifacts/` and never wrote `state/code_v1.done`; the session that owned the run reported it as
"still RUNNING" at 2026-08-17 23:20 and again at 2026-08-18 01:18, ~7.5 h and ~9.5 h after the
process had in fact written its last message. Landed as `artifacts/code_v1.md` at 2026-08-18 14:03
after the validation below.

## Reproduced, not taken on trust

    python -m pytest tests/unit/test_abiss_nucleus_competition.py -q
    -> 22 passed in 1.67s                                    (code_v1 claims "22 passed in 2.19s")

    all 6 newly claimed deliverables exist on disk:
      dev/zebrafinch/compare_nucleus_competition.py
      tests/fixtures/abiss_nucleus_competition/legacy_emitted_labels.json
      docs/nucleus_competition_review.md
      dev/zebrafinch/sbatch_nuccomp_{scan,merge,flood}.sh

## G1 — the strict overlay destroyed a live, unrelated cluster array (measured here)

Job 2872438 `me_L0`, 80-task array over `dev/zebrafinch/aggsweep/b3_gate025/seg_agg_b3_gate025`,
submitted 2026-08-17 13:40:18, started 14:42:28. Its `param` sets

    NUC_COMPETITION_MANIFEST = .../wholevol_arm0_native96_nuc_matchguard/
                               seg_arm0_native96_nuc_matchguard/nucleus_competition/manifest.json

which is schema 1.2.

    sacct 2872438        -> 1 COMPLETED, ~9 FAILED (~12 min), ~70 TIMEOUT (12:00:25 wall)
    downstream           -> me_L1 me_L2 me_L3 me_L4 me_L5 remapagg (2872439-2872444) CANCELLED
                            mgrag (2872596) CANCELLED
    77 of 80 task logs contain
        File "lib/abiss/scripts/nucleus_overlay.py", line 73, in load_validated_manifest
        ValueError: nucleus competition manifest lacks a valid completion marker

Broke mid-flight, not at launch — this is the decisive measurement:

    tasks with BOTH a successful manifest load AND a later rejection : 77
    tasks rejected from their first chunk                            : 0
    task _0: 89 chunks logged "nucleus competition: removed ... voxels" before the first
             rejection at log line 16671; 0 successes after it.

So the array was running correctly against the 1.2 manifest, and the file changed underneath it.
`code_v0.md:28` claims the change ("Made the overlay validate the completion marker, expected plan,
authoritative watershed identity") and `code_v0.md:146` states the consequence was intended
("Schema-1.2 canonical manifests are intentionally rejected by the schema-2.0 overlay").

Cost: ~70 tasks x 12 h wall ~= 840 node-hours, plus the cancelled downstream chain; the
`b3_gate025` arm produced no result.

Timing caveat I could not resolve and am not asserting past: the coordinator's mtime attribution in
run.md records code_v0's write of `nucleus_overlay.py` at 15:08, but the first rejections appear by
14:55. mtimes were overwritten by the 15:42 code_v1 write and `lib/abiss` has no commit between,
so the exact write instant is unrecoverable. It does not change the finding: strictness entered a
shared production path under a live consumer.

## G2 — the enforced parent-id rule contradicts the production reference (verified at both sites)

    nucleus_overlay.py:106       raise ValueError("each unit must emit its parent watershed id exactly once")
    nucleus_competition.py:1191  (identical rule, duplicated)

Provenance: the design panel measured that native96 (1.2) emits `_stable_new_id(nucleus_id)` for
16/16 territories and that no repair emits the parent id; the prior session independently
confirmed nucleus 373 carries one id (1194193566422099803) under two different parents. I verified
the rule's existence and duplication, not the 16/16 recomputation.

## G3 — "competition can only divide, never merge" is false (measured by the prior session)

`canonicalize_qualified_segments` (`nucleus_overlay.py:179`) remaps multiple distinct base
watershed segments onto one owner id: 85 qualified segments -> 23 labels, 15 labels absorbing more
than one segment, up to 7:1 (nucleus 37). The claim appears as a safety invariant in
`lesson_nucleus_competition.md` and in `task.md` Section 1.

## G4 — live silent no-op (verified directly)

    nucleus_overlay.py:250   repairs = manifest.get("repairs", [])
    nucleus_overlay.py:251   if not repairs:
    nucleus_overlay.py:253       return seg_cutout, nucleus_cutout
    nucleus_overlay.py:269   canonicalize_qualified_segments(labels, manifest)

A legitimate zero-repair run returns before canonicalization and silently skips it.

## G5 — parent-absence is a non-detector (panel measurement, not re-verified here)

0 of 9 (1.0) and 0 of 8 (1.2) parent ids appear in any published segment overlapping a nucleus,
including in the run where 9 of 9 repairs were undone. Any postcondition built on parent-absence
in the published volume would have signed off on the collapsed run.

## Codex availability

    PATH=/home/weidf/.nvm/versions/node/v24.9.0/bin:$PATH codex exec --sandbox read-only
    -> CODEX_OK   (codex-cli 0.145.0)

The quota block recorded in run.md has lifted; a codex-owned `code_v2` can run.

READY: no

## Addendum (2026-08-18, coordinator) — G5 realization gate verified first-hand

G5 was inherited from the design panel and marked "not re-verified here". Now measured directly
from each run's own `nucleus_shell_contamination_tol0.json` (464 nuclei records each) joined to its
own `nucleus_competition/manifest.json`, taking each repair's anchors and comparing their
`dominant_segment` in the published volume:

    win144  (schema 1.0), 9 repairs : SEPARATED 0 of 9    owner label realized  0 of 18
    native96(schema 1.2), 8 repairs : SEPARATED 8 of 8    owner label realized 16 of 16

Every win144 unit's two anchors share a single dominant segment (e.g. 611/651 both ->
73254963610077728). Every native96 unit's anchors have distinct dominant segments, and each anchor's
`anchor_labels` entry equals that nucleus's dominant published segment.

Precision note for test 1's oracle: 16 is the number of anchor slots (8 repairs x 2); 15 is the
number of DISTINCT owner labels, because nucleus 373 appears in two repairs (parents
72481112709922836 and 72551481454100482) and carries the same id 1194193566422099803 in both. That
also confirms the per-nucleus identity scope first-hand rather than by inheritance. Both figures are
correct; a test must not conflate them.
