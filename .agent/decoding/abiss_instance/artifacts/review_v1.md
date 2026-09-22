# Review v1

## Summary

`code_v1` does what `review_v0` asked. All six findings are addressed, the legacy regression oracle
is executable again through a read-only harness that leaves production strict, and the 22 focused
tests pass — reproduced here in 1.67 s, not taken on trust. The coder was also honest about what it
did not do: gates 2–8 unrun, no speedup claimed, `ccc-validate.sh` left to the coordinator. On its
own terms this is a good implementation stage.

It is **NEEDS_CHANGES** anyway, and for reasons that are larger than `code_v1`'s own quality. Two
things came to light after it was written.

**First, the strictness it hardens has already destroyed a live experiment.** Job 2872438 — an
80-task `me_L0` array over `dev/zebrafinch/aggsweep/b3_gate025`, unrelated to this run and launched
before it — was consuming `nucleus_overlay.py` while `code_v0` rewrote that file in place. The
array did not fail at launch: **77 of 80 tasks logged successful manifest loads first and then
began rejecting**, with zero successes afterwards; task 0 completed 89 chunks before the first
rejection. About 70 tasks then burned the full 12-hour wall, and the whole downstream chain
(`me_L1`–`me_L5`, `remapagg`, `mgrag`) was cancelled. That is roughly **840 node-hours** and one
sweep arm lost, to a validator that this run introduced deliberately (`code_v0.md:28`, `:146`) but
deployed by editing a script other people's jobs were already executing.

**Second, the rule being enforced is factually wrong about the production run.** `plan_v3` §B.1
told the coder that "the largest territory retains the parent watershed id", and `code_v0`
implemented it as a hard invariant in two places. The native96 matchguard run — the arm every
current acceptance number describes — does not do that: it mints a per-nucleus id and no repair
emits the parent id at all. The invariant describes the *win144* run, whose repairs did not survive
into its published volume. So the validator rejects the good arm and would have accepted the
collapsed one. That error is mine as planner, not the coder's.

A third item is a plain defect that neither the plan nor the reviews caught until now: a zero-repair
run returns before canonicalization and silently skips it.

Scope note for `code_v2`: `c2` makes it the terminal code version, so the directives below are the
final ones. They are ordered accordingly — the two that unblock live work first.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

`git rev-parse HEAD` is unchanged at `c705458a`; no commit was created by `code_v1`. The surface
caveat from `review_v0` still holds — `lib/abiss` is a nested git repository, `dev/` is gitignored,
and the test file is untracked — so attribution again rests on mtimes and on the coder's own file
list rather than on a parent-repo diff. `code_v1`'s artifact was produced at 15:52 on 2026-08-17
and sat unclaimed for 22 h; the coordinator validated and landed it at 2026-08-18 14:03 before
this review. Evidence for every measurement below is in `state/review_v1.review.raw.md`.

## Findings

- **[major] G1 — Strictness was deployed into a shared script path under a live consumer, and cost
  ~840 node-hours.** Measured above. The defect is not the strictness itself; it is that
  `nucleus_overlay.py` is executed by `cut_chunk_agg.py` for *every* agglomeration run in this
  repository, including runs that reference a legacy manifest and were launched before the change
  existed. `code_v2` must make this class of breakage impossible to repeat: a strict production
  reader may not ship until a migration path exists that turns an existing legacy publication into
  an acceptable one **without recomputation**, and the rejection message must name that command.
  The current message tells the operator to "rerun the scan/flood/merge workflow", which for
  `b3_gate025` meant an 11-hour stage it had no reason to run.
- **[major] G2 — Delete the parent-id inheritance rule; declare identity instead.** The rule at
  `nucleus_overlay.py:106` and its duplicate at `nucleus_competition.py:1191` ("each unit must emit
  its parent watershed id exactly once") contradicts the production arm. Replace both with a
  declared `identity` block — `scope: "nucleus"`, `parent_disposition: "retired"`, and an explicit
  `mint` descriptor — and derive every id check from what the artifact declares about itself rather
  than from a convention the author remembers. The general rule worth writing into the code
  comments and the lesson: *a validator may enforce what an artifact claims about itself; it may
  not enforce what its author remembers other artifacts doing.* One canonical owner for the rule,
  not two.
- **[major] G3 — The "refinement only" safety invariant is false and must be declared, not
  repaired.** `canonicalize_qualified_segments` (`nucleus_overlay.py:179`) collapses 85 qualified
  segments into 23 labels, 15 of them absorbing more than one base segment, up to 7:1. Competition
  therefore merges by *naming*, which no region-graph cannot-link can observe. The consequence
  belongs in the docs in plain words: if a nucleus mask ever places one instance across two
  genuinely different cells, this silently fuses them. Correct the claim in
  `dev/zebrafinch/lesson_nucleus_competition.md` and in `task.md` §1, both of which assert the
  opposite, and record the consolidation explicitly in the manifest (a `sources` list of length > 1)
  so a consumer can see it.
- **[major] G4 — Zero-repair runs silently skip canonicalization.** `nucleus_overlay.py:251-253`
  returns before `canonicalize_qualified_segments` at `:269`. A legitimate zero-repair publication
  is a first-class outcome, and it currently omits a stage it should still perform. Fix and pin.
- **[major] G5 — Do not gate on parent-id absence in the published volume.** It is a non-detector:
  0 of 9 and 0 of 8 parent ids appear in any published segment overlapping a nucleus, *including*
  in the run where every repair was undone. The check that discriminates is per-nucleus
  realization on the published volume — each nucleus's dominant published segment is distinct from
  its unit-mates' and equals that nucleus's own minted label. It separates the two arms cleanly
  (0/9 versus 15/15), uses only nucleus-mask voxels, and needs no ground truth.
- **[minor] G6 — There are three id conventions, not two, and one is opaque.** The win144 (1.0)
  labels match neither `_stable_new_id` nor `_stable_territory_id`, so they cannot be recomputed
  from any recorded field. A legacy record for that arm should say so — `mint.scheme:
  "opaque_legacy", deterministic: false` — rather than imply reproducibility it does not have.
- **[minor] G7 — Ids exceed 2^53 and must round-trip as decimal strings.** Values near 1.2e18 are
  quoted in the manifests only by convention, and `nucleus_competition.py:290` writes a bare int
  into the intermediate unit dict. Any `jq` or JavaScript consumer silently rounds it.

Recorded so `code_v2` does not redo them: `review_v0`'s F1–F6 are closed. The read-only harness
(`dev/zebrafinch/compare_nucleus_competition.py`) makes gates 2 and 3 executable while leaving
production strict, which was the harder of the two options offered in Q-2 and the right one; the
`CHUNKMAP_INPUT` amendment is accepted and recorded; `NUC_MAX_UNITS = 64` is confirmed with
fail-closed coverage; and `docs/nucleus_competition_review.md` now states plainly that gates 2–8
are unrun. None of that is reopened here.

## Tests to Add

1. **A realization gate on a published volume** (G5). Given a run directory and its nucleus mask,
   assert per unit that the anchors' dominant published segments are distinct, that dominance
   clears a declared threshold, and that each equals that nucleus's own minted label. It must
   return 0/9 for `wholevol_arm096_nuc_competitive_v2` and 15/15 for
   `wholevol_arm0_native96_nuc_matchguard`; those two numbers are the test's own oracle.
2. **A legacy-consumer regression** (G1). A schema-1.2 manifest plus the migration output must be
   accepted by the production overlay with no recomputation, and the pre-migration rejection
   message must name the exact migrate command. This is the test that would have prevented the
   `b3_gate025` loss.
3. **Mint determinism and scope** (G2). Recompute every `emitted_id` from the declared
   `key_template`; assert ids are equal iff the nucleus is equal, across units *and* across the
   canonicalization table. Pin nucleus 373's single id under two parents.
4. **Ledger closure** (G3). Every retirement is recorded — the 8 split parents *and* all 77
   canonicalized single-owner segments — and every id in the emitted supervoxel layer is either a
   declared minted id or an untouched base id. A ledger that records only the splits misses 90% of
   the retirements.
5. **Zero-repair integrity** (G4). `repairs == []` still performs canonicalization, still writes a
   complete publication with a populated reason, and does not short-circuit sibling stages.
6. **Decimal-string round-trip** (G7) for every emitted id, at every write site.

## Questions

- **Q-1 (scope, and the one I would most like ruled on).** `code_v2` is terminal under `c2`. My
  recommendation is that it carry G1–G5 plus tests 1–5 — the two live-work unblockers, the three
  measured defects, and the identity declaration that makes them coherent — and that the remaining
  contract surface from the design decision (full `inputs[].semantics` provenance, capability
  negotiation, the measured native-id-space scan) be deferred to a follow-up run rather than
  crammed into a final round that no review stage will see. Alternative, if you would rather land
  the whole contract at once: extend to `c3` so `code_v2` gets a `review_v2`.
- **Q-2 (a genuine behaviour change, your call).** `residue_disposition` — the un-adjudicated
  remainder of a fused parent, outside the adjudicated box or marker-free inside it. Minting it is
  what makes "no parent id survives in the emitted supervoxel layer" a true, checkable statement,
  but it renames published objects on new runs and invalidates meshes and proofreading state keyed
  on them. It cannot be applied to the existing matchguard run by editing a manifest. Default to
  `"minted"` for new runs, or keep `"parent_retained"` and accept a weaker invariant?
- **Q-3 (not this run's, but yours).** `b3_gate025` produced nothing. Relaunch it after `code_v2`
  lands the migration path, or drop that sweep arm?
- **Q-4.** The `0/9` versus `8/8` separation difference between the two arms is confounded —
  `matchguard` changed several things at once, and the lesson attributes the v2 collapse to
  `match_chunks.cpp::process_nucs` ordering and to the final remap not replaying the overlay, not
  to id inheritance. I have written the identity findings so they do not depend on that
  attribution. Confirm you want it left that way rather than resolved by a controlled re-run.

## Verdict

VERDICT: NEEDS_CHANGES
