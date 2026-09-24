# Plan v2 Review

## Summary

Reviewer: codex, read-only, exit 0, repository unmutated. Raw transcript at
`state/plan_v2_review.review.raw.md`.

**4 major, 1 minor, `READY: no`.** `plan_v2` was the final allowed plan version
(`p2`), so under `normal` mode this is a **decision point for the human**, not
another revision.

The reviewer's first finding has since become a measured fact rather than a
hypothetical, which strengthens it: **S1 (BC job `3031548`) completed during this
review and selected `max`.** Best VOI over six cubes was `max` 0.5279 against
`mean` 0.5827 — a 0.055 gap, roughly 5.5× this project's ~0.01 noise floor. The
`max_compressed` / `max_uncompressed` arms matched row for row, so the
uncompression is verified and the comparison is trustworthy.

So the branch `plan_v2` §1 called "the single highest-consequence open item" has
resolved onto its bad side: **the criterion that wins is the one ABISS's chunked
pipeline cannot execute.** That is not a planning defect; it is the project-level
decision the plan correctly refused to make on its own.

## Findings

### Major

1. **The `max` branch is an unresolved human decision.** If S1 selects `max`,
   §1 option (a) contradicts §6's requirement that both decoders use the same
   criterion: the existing whole-volume reference is `max` while chunked decode
   would be `mean`. The plan must say whether to produce a new whole-volume
   `mean` reference or stop S3, and who decides. *(Now live: S1 selected `max`.)*
2. **The uncompression precondition is not executable.** float16 layout and a
   probability-space threshold are specified, but not *where* `scale_sigmoid` is
   inverted, which artifact ABISS consumes, or how that representation is
   validated.
3. **S4 inference distribution is underspecified.** Decode has a dispatcher with
   leases, retries and barriers; inference has only "one block writes one disjoint
   region" plus a validator — no task manifest, claim/retry/resume protocol,
   in-flight missing-block detection, or explicit barrier guaranteeing affinity
   completeness before decode.
4. **The dispatcher contract gap blocks implementation.** The plan admits its
   `$PARALLEL_CMD` contract is inferred from one call site and "should be
   re-read". Exact command, quoting and environment behaviour must be fixed
   before replacing GNU parallel.

### Minor

5. Cross-region costs are not reproducible from the displayed decimal-GB sizes at
   `$0.02/GiB`. *(Accurate but unshown: the figures include a ×1.024 GB→GiB
   conversion; label the units or show the arithmetic.)*

## Questions

- Finding 1 is now a measured fork, not a hypothetical. The decision is: accept
  `mean` for the chunked path and re-establish a whole-volume `mean` reference so
  §6 stays coherent — paying ~0.055 VOI against `max` — or evaluate `rlme`
  (`ac`/`agg`), the untested third criterion ABISS's chunked path ships. Both are
  measurements, not implementations, and both belong to card
  MSIDEPLOY-SCALE-001.
- Findings 2–4 are all "specify the mechanism precisely". They are tractable in
  one more plan version, which the configured rounds do not allow. Is the right
  response `p3`, or a fresh CCC run scoped to the dispatcher alone?

## Verdict

VERDICT: NEEDS_CHANGES
