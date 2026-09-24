# Plan v3 Review

## Summary

Reviewer: codex, read-only, exit 0, repository unmutated. Raw transcript at
`state/plan_v3_review.review.raw.md`.

**5 major, 2 minor, `READY: no`.** `p3` is exhausted, so this blocks for a human
decision again.

The reviewer explicitly conceded one point — *"The quality conclusion, that S3 can
compare two mean decoders without resolving the eventual scientific choice, does
follow"* — and its remaining objections are concentrated, not scattered. Findings
2, 3 and 5 are all about **distribution (S4)**; finding 1 is about
**parameterising the criterion**; finding 4 is about the pre-decode barrier.

That concentration is the useful signal from this round. Across four plan
versions the residue has narrowed to one area, and it is the area the task
bundled in rather than the area the 100 µm decision actually needs. See
Questions.

Findings 2 and 4 are accepted without reservation: a stdin/`{}` contract really
does describe a local command wrapper and not a distribution protocol, and
"readable and non-empty at each block's bounding box" really can pass on a
partially written block.

## Findings

### Major

1. **The criterion is still chosen by the plan.** The task says treat it as an
   input; `plan_v3` fixes `mean` for the equivalence experiment. That experiment
   is accepted as coherent, but the **code must parameterise the criterion** and
   define the `max` outcome explicitly — stop, add a compatible decoder, or take a
   different path.
2. **Distributed dispatch is still not executable.** The stdin/`{}`/`--halt 2`
   contract defines a local command wrapper. No worker protocol, task assignment,
   worker registration, failure detection, retry behaviour, or means for the
   dispatcher to determine global completion and exit status.
3. **Inference resume is asserted, not specified.** ABISS's flags are established
   for ABISS tasks; the plan does not say how inference invokes the flag scripts,
   how `infer_` keys are formed, or how two workers avoid ambiguous ownership of a
   block.
4. **The pre-decode barrier is insufficient.** "Readable and non-empty at each
   block's bounding box" can pass when only part of a block exists, or when
   missing regions hold valid-looking values. Needs an exact coverage/chunk
   manifest check with expected extents and completed writes.
5. **Redis-for-flags is not shown to cover all scratch.** The plan must establish
   that every *non-flag* scratch artifact is task-local and never read by another
   worker or a later stage; otherwise distributed decode still fails despite
   shared flags.

### Minor

6. The probability-space writer is an implementation placeholder: no exact source
   artifact, clipping/quantisation rules, path schema, or writer ownership.
7. The transfer-cost table does not distinguish which artifacts are *actually*
   transferred in the fallback path from those merely costed.

## Questions

**The blocking question is scope, not detail.** Four plan versions have converged
the residue onto S4 (distribution): findings 2, 3 and 5 are all distribution, and
5 is a consequence of attempting it. S3 — chunked decode on one VM plus the
equivalence test — is by contrast nearly fully specified and carries none of
those findings.

S3 is also the part that answers the 100 µm question: whether chunked decoding
equals whole-volume decoding at all. S4 is throughput. Bundling them came from the
task, which this session authored, not from a dependency.

So: **split the run.** A fresh CCC run scoped to S3 alone — criterion-parameterised
per finding 1, single VM, no dispatcher — would carry findings 1, 4, 6 and be
tractable in one or two rounds. S4 becomes its own run whose entire subject is the
dispatcher, where findings 2, 3 and 5 are the specification rather than
objections to it.

Two smaller questions for the human:

- Finding 1 asks for an explicit `max` path. Given S1 measured `max` 0.5279 vs
  `mean` 0.5827 and ABISS's chunked path ships no max binary, the honest options
  are "accept `mean` at 100 µm", "measure `rlme` first", or "do not chunk" — the
  last of which forecloses 100 µm entirely. That is a scientific decision this
  workflow cannot make.
- Is `auto` mode wanted for a re-run? It would permit `APPROVE_AUTO_OVERRIDE` on
  unresolved disagreement. Given findings 2 and 4 are genuine correctness gaps
  rather than stylistic ones, overriding them is **not** recommended.

## Verdict

VERDICT: NEEDS_CHANGES
