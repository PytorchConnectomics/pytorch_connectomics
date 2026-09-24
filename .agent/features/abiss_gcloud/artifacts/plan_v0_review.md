# Plan v0 Review

## Summary

Reviewer: codex (coder for this run), read-only sandbox, exit 0. Repository
unmutated: `git diff` and `git diff --cached` both 0 bytes before and after, and
`HEAD` still `7d819bdb`. Raw transcript preserved at
`state/plan_v0_review.review.raw.md`.

**6 major and 7 minor findings; `READY: no`.** The reviewer's core objection is
consistent across the majors and is accepted: the plan describes the *right*
architecture but repeatedly defers the decisions that make it implementable —
what the chunk boundary semantics are, what the queue actually is, what the
affinity layout is, and what happens for each possible S1 outcome. A plan whose
proposed changes contain "determine whether X exists; if not, add it" is a
research note, not an executable plan.

One finding is a genuine gap against the task rather than against my judgement:
the task requires multi-**region** fallback and the plan only specified
multi-zone.

No finding is disputed. None is softened or dropped below.

## Findings

### Major

1. **Merge criterion pass-through is not specified per S1 outcome.** The plan
   assumes `mean` and says it will "restate" expectations otherwise. It must say
   how the selected criterion is threaded through *both* whole-volume and chunked
   decode, and what code/config changes each S1 outcome requires.
2. **Multi-region fallback is missing.** Only alternate zones, smaller shapes and
   optional on-demand are covered. Bucket replication, cross-region input
   staging, cost handling and region selection are absent — and the task named
   multi-region explicitly.
3. **Chunk-boundary semantics are underspecified.** No overlap/halo size, no
   ownership rule for boundary voxels, no cross-chunk label reconciliation, and
   no account of how `CHUNKMAP_OUTPUT` is consumed to produce globally consistent
   labels. A work queue does not by itself make independent chunk decodes
   equivalent. *(This is the finding most worth acting on: it is the actual
   mechanism behind the seam risk the plan claims to be centred on.)*
4. **The worker queue is not executable.** No backend, task schema, claim/lease
   mechanism, atomic completion marker, retry/idempotency rule, or missing-chunk
   validation — all of which preemption recovery depends on.
5. **The affinity writer is an unresolved architectural branch.** Output layout,
   chunk alignment, dtype, metadata, concurrent-write behaviour and validation
   are undefined.
6. **GCS scratch is unresolved despite being central.** Neither established as
   safe/acceptable for ABISS scratch, nor given a local-SSD fallback with its
   staging, synchronisation and upload workflow.

### Minor

7. Equivalence criteria are ambiguous: "boundary-crossing subset" and
   "whole-volume-interior subset" need precise object selection and a formula,
   and the tolerance must say split, merge, or total VOI.
8. Several checks lack numeric bounds: "label count in the expected order", "max
   inside the `scale_sigmoid` range", sizing numbers "match the task's table".
9. The one-chunk smoke test is redundant with the whole-volume path; keep it, but
   label it a plumbing test rather than an S3 acceptance test.
10. Parallel correctness needs a comparison rule tolerant of label renumbering,
    and should include interrupted/retried workers, not only 1 vs 4 workers.
11. No proof of complete inference block coverage before decode: block inventory,
    missing-block detection, duplicate-write handling, affinity metadata
    consistency.
12. The sizing report is not fully executable: no RSS model or measurement
    method, chunk dimensions, worker concurrency, temp storage, or whether
    GPU-hours include retries and preemption overhead.
13. Meshing/publication is named as a blocker with no handoff artifact or
    acceptance boundary; state whether S3/S4 deliver decoded segmentation only or
    a usable published volume.

## Questions

- Finding 3 asks for cross-chunk label reconciliation semantics. These are
  properties of ABISS's existing chunked implementation
  (`remap_watershed`, `remap_agglomeration`, `CHUNKMAP_OUTPUT`), not of new code.
  `plan_v1` should therefore **read that implementation and document the
  semantics it already has**, rather than designing them — the risk is
  mis-describing a working mechanism, not inventing a missing one.
- Finding 1 requires enumerating S1 outcomes. S1 is four arms; is enumerating all
  four proportionate, or is the correct response to make the criterion a single
  configured parameter with no per-outcome branching?
- Finding 2 asks for cross-region staging and cost handling. At 772 GB of
  affinity, cross-region transfer is ~$15 per copy — is multi-region fallback
  actually wanted, or should the plan argue for staying in-region and failing
  loudly instead?

## Verdict

VERDICT: NEEDS_CHANGES
