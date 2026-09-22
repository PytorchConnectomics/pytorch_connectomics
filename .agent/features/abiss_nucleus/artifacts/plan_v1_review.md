# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v1_review.review.raw.md`. Verdict upheld without softening; the lead finding is
correct and I have since confirmed it independently.

Plan v1 was right that `task.md`'s mechanism is inconsistent with `NucExtractor`, and wrong about
what replaces it. It mistook the final CONFLICT state for the cause. The reviewer's arithmetic is
decisive: once a cluster is CONFLICT the only permitted merge is with NONE, whose record has
`total == 0`, so every CONFLICT+NONE join **preserves** `total`. The final `total = 28,576,326`
was therefore already present when CONFLICT was created. Invariant D clause 2 explains growth in
untagged volume; it cannot explain how three nuclei became fused.

Post-review measurement (`state/evidence_conflict.md`) confirms the reviewer and goes further:
the offending SID is already one segment at the **atomic** level, appearing across 70+ atomic
chunks with locally-correct PROPER records -- 275 in some, 319 in others, 373 in others -- and the
`nonuc` arm reproduces the same SID at **640,562,148 voxels** with no nucleus mask at all. The
fusion is pre-existing and predates any nucleus evidence.

## Findings

* **[major] Clause 2 cannot explain the measured tagged contamination.** CONFLICT+NONE preserves
  `total`; `28,576,326 / 256` closely matches the 110-112K contaminated mask voxels, so that mass
  was present at conflict birth. Confirmed.
* **[major] CONFLICT is a post-hoc symptom, not the cause.** The `load_conflict_collisions`
  counter is not keyed by SID and does not establish where this segment became CONFLICT; zero
  "merge propagation violated" messages is expected and says nothing about which merge classes
  occurred. Confirmed -- and located: the fusion is already complete at level 0.
* **[major] The refutation of `task.md` was only partial.** The code refutes "missing the
  dominance ratio produces NONE" (it produces CONFLICT unless below `min_tagged`), and the
  subfloor count rules out below-floor voxels carrying 110K mask voxels; it does not establish
  clause 2 as the replacement cause. Correct.
* **[major] The treatment cannot satisfy V4's primary criterion.** `load_nuc`/`reduce_nuc` create
  CONFLICT without consulting `nuc_can_merge`, and by then the records share a SID; making the
  cluster inert does not split that SID. The proposed `conflict_absorbed` diagnostic is also
  misplaced -- the absorbed operand is NONE, so "tagged voxels absorbed" is always zero; it should
  record the NONE operand's `seg_size`, the conflict's size at birth, hierarchy level and SID.
* **[major] The proposed predicate does not preserve flag-off behaviour.**
  `if (conflict_a || conflict_b) return !nuc_strict_conflict();` admits CONFLICT+PROPER when the
  flag is off. The flag-off path must retain the other operand's NONE check. The plan also
  alternates between an unspecified global and plumbing a parameter to call sites.
* **[major] The reducibility argument is still incomplete.** A veto does have a better local
  argument than a priority key -- under strict mode no permitted in-loop join can create CONFLICT,
  so conflict clusters can be treated as removed from the active graph -- but that closure
  argument must be stated, not just commutativity/associativity. Across levels, the extractor is
  **partition-dependent**: one child emits PROPER-1, another PROPER-2, their reduction is
  CONFLICT, while the combined raw counts may be 70/30 and monolithically PROPER-1. A generic
  boundary-spanning fixture does not exercise this, so V3 would not establish chunk independence.
* **[major] `nonuc` is not the claimed discriminator and its branches are not objectively
  defined.** Removing all nucleus vetoes changes the whole mean-linkage trajectory; "comparable"
  and "much smaller" have no thresholds. The direct discriminator is per-SID size and tagged mass
  at conflict birth versus volume absorbed afterwards. `mint5000` also creates more NONE records,
  so it is not inherently safer.
* **[major] V4 is gameable and does not price the stated NERL risk.** Five shared fragments of
  0.0099 each pass criteria 1/2/4 while retaining 4.95% contamination; splitting 291M into two
  145M false objects passes the size ceiling with one added record; criteria 2 and 4 disagree on
  the clean-nucleus threshold; criterion 2 silently upgrades "does not regress" to 0.95.
* **[major] Not all nine v0 findings are resolved.** F1-F5 are legitimately mooted and F8 is
  fixed; F6 remains (no adversarial load-created-CONFLICT case), F7 is scoped away rather than
  tested, F9 remains through the subjective gate and gameable criteria. Because the plan abandons
  the task's mandated ordering change, implementation requires explicit task-owner approval.

## Questions

1. For the offending SID, what were its `seg_size`, nucleus `total`, and contributing child
   records at the exact collision that first created CONFLICT?
2. How much untagged `seg_size` did that SID absorb through CONFLICT+NONE after that point?
3. Is the task owner authorising replacement of the required ordering change with an earlier
   hierarchy fix or a post-agglomeration split, if the tagged fusion already exists at conflict
   birth?

Questions 1 and 2 are now answered in `state/evidence_conflict.md`: the SID is fused at the atomic
level and exists at 640,562,148 voxels with no nucleus mask at all, so the fusion is pre-existing;
the nucleus constraint already halves it to 291,235,662. Question 3 is the open decision and is
the reason this run is blocked.

## Verdict

VERDICT: NEEDS_CHANGES
