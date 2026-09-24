# Review v1

## Summary

Reviewer: claude (planner), performed in-session. Raw transcript:
`state/review_v1.review.raw.md`.

All eight `review_v0` findings — four major, four minor — are resolved. Every claim in
`code_v1.md` was re-run rather than trusted: 28 tests pass, the legacy pretrain reproduction
still passes, `compare_cc3d_abiss.py` is byte-identical, and no commit was made.

The riskiest change was the resolver rewrite from a per-fragment Python loop to vectorised
`lexsort`/`unique`/`bincount` reductions, because it rewrites safety-critical selection logic and
the coder's own tests share the understanding that produced the code. I wrote an independent
reference resolver from the plan_v2 R1/R2/R3 specification and compared fragment→anchor
assignments over 300 randomised trials × 5 policies — **1500 comparisons, zero mismatches** —
exercising the veto, no-direct-contact, multi-host abstention, and margin adjudication paths.

Finding 1 was answered the right way. The constant was neither silently changed nor its
tolerance widened; it is retained, annotated `status: "unsatisfiable_from_archived_artifacts"`,
and the gate now fails loudly after reproducing seven named archived aggregations whose values
match my own independent measurements exactly.

Two minor residual observations, neither affecting correctness or any reported number.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

`.gitignore:159` ignores `dev/`, so the change surface is untracked and produces no git diff; the
review read the files directly. `git diff` and `git diff --cached` were captured before and after
and are unchanged. `git log c705458a..HEAD` is empty.

## Findings

1. [minor] **The scale guard does not exercise contested hosts.** `test_resolver_scales_to_full_population_order`
   gives every fragment ten edges to a single anchor, so the R3 multi-host adjudication and the
   `np.maximum.at` loss-rank reduction run at scale only in the degenerate case. My differential
   test covers multi-host correctness but only at small n. A scale case with two competing hosts
   per fragment would close the gap.

2. [minor] **Formatting verification is weaker than the artifact implies.** An earlier Black run
   exited 124 (timeout) and the final checks used `--fast` because the `pytc` environment is
   Python 3.11 against a 3.12 target. Cosmetic only — mypy, flake8, and isort passed normally.

Resolved from `review_v0`, each verified rather than accepted: (1) tier10 constant now
fail-closed with declared provenance and seven reproduced archived aggregations; (2) resolver
vectorised, 100k edges in 0.11 s, verified equivalent by differential test; (3) C3 bboxes carried
into C5, dense `argwhere` removed; (4) bounded crops with overflow-checked `uint32` slabs, modeled
peak 15.262 GiB under a 16 GiB ceiling; (5) C5 writes a separate feature artifact and rejects
in-place overwrite; (6) freeze digest narrowed to the GT-free file list; (7) interface summary
keys consistent; (8) abstention reasons in causal ladder order.

## Tests to Add

* A second scale case with two competing hosts per fragment, so R3 and the loss-rank reduction
  are exercised at full cardinality rather than only in the single-host case.
* Fold the reviewer's differential test into the suite: an independent reference resolver checked
  against the vectorised one over randomised inputs. It is the strongest available guard against
  a future optimisation silently changing assignment semantics.

## Questions

One decision is required, and it belongs to the task author rather than the implementation.
`task.md` makes reproducing the recorded tier10 CC3D@0.70 row a Phase-0 blocker, but the quoted
0.775832 cannot be reproduced from the archived tier10 CSVs by any of seven standard
aggregations, and the string appears nowhere in the repository outside `task.md`. Stage B is now
correctly fail-closed on this, which means the data stages cannot proceed until either a
reproducible aggregation is supplied or the gate is redefined. The substrate resolution to
`arm0_win144` is unaffected, and the task's qualitative ordering still holds — measured native
0.837572 / 0.852206 beats measured win144 0.694207 / 0.749818, by a wider margin than task.md's
table implies.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
