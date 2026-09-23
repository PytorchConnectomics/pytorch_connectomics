# Plan v1 Review

## Summary

Reviewer: codex, read-only sandbox, exit 0. Repository unmutated (`git diff` and
`git diff --cached` 0 bytes, `HEAD` still `7d819bdb`). Raw transcript at
`state/plan_v1_review.review.raw.md`.

**6 major, 3 minor, `READY: no`.** Two findings land hard and are accepted
without reservation:

- **Finding 4 exposes a contradiction I introduced.** The task requires inference
  *and decode* across many small interchangeable spot VMs; `plan_v1` §2 put decode
  on one high-memory VM. Parallel chunk *uploads* inside one process are not a
  distributed worker pool, and I conflated them.
- **Finding 2 is right that I refused a stated requirement.** More usefully, the
  refusal rested on moving the wrong artifact: I costed relocating the 772 GB
  affinity, when the thing to relocate is the ~129 GB source. That changes the
  economics and makes multi-region tractable rather than prohibitive.

The partial vindication on finding 3 ("the rejection is mostly sound") is noted
but does not excuse the gap it identifies: the `CHUNKMAP_OUTPUT` consumption
contract is asserted, not documented.

No finding is softened or dropped.

## Findings

### Major

1. **Multi-region fallback is still not addressed.** Refusing cross-region, even
   with a cost estimate, violates a stated task requirement. Needs replication or
   staging, region selection, transfer-cost handling, and resume behaviour.
2. **A non-`mean` S1 outcome is still only declared a blocker.** No required ABISS
   stage/configuration change and no executable alternative is specified. "No
   defaults" does not resolve an outcome-dependent implementation.
3. **The finding-3 rejection is mostly sound** — duplicating ABISS's
   reconciliation would be wrong — **but the `CHUNKMAP_OUTPUT` consumption
   contract is still not documented precisely enough to configure and validate.**
   Preconditions remain asserted rather than demonstrated.
4. **The finding-4 rejection is not sound.** Parallel chunk uploads are not an
   executable distributed worker queue with leases, retries, idempotency and
   missing-work detection. And putting decode on one high-memory VM conflicts
   with the task's requirement of many small interchangeable spot VMs.
5. **No concrete numbers.** Chunk dimensions, worker concurrency, VM and disk
   sizing, and scratch capacity are all required before implementation and before
   any 100 µm sizing claim.
6. **The equivalence criterion is not executable as written.**
   `VOI_boundary − VOI_interior` is undefined without a precise method for
   computing VOI on object subsets, and it is unclear whether it is a sum, a
   maximum, or independently normalised.

### Minor

7. Affinity layout narrowed but not complete: metadata, exact volume/chunk shape,
   writer validation, and the block-to-output mapping remain open.
8. Fallback policy has no concrete ordered shape/zone matrix, and the
   "explicitly enabled" on-demand interface is unspecified.
9. The ×1.3 retry allowance is a reporting convention, not a measured or modelled
   quantity.

## Questions

- Finding 4 requires a distributed decode. The Dockerfile for
  `neuron_snemi_gcloud` notes it builds "only the single-volume C++ `ws` target:
  no CloudVolume worker stack", which implies ABISS **has** a worker stack that
  this project has never built. `plan_v2` must establish whether that stack
  provides the queue semantics finding 4 asks for, because building a second
  queue beside it would repeat the `plan_v1` §2 mistake in the other direction.
- Finding 2: is multi-region wanted as *relocate the source and run wholly in the
  fallback region* (source ~129 GB at the model grid, affinity never crosses), or
  as genuine bucket replication? These have very different costs and the plan
  should commit to one.
- Finding 5 asks for concrete chunk dimensions, but they are constrained by the
  alignment rule, per-chunk RSS, and scratch capacity simultaneously. Should
  `plan_v2` fix them numerically for the 100 µm case, or specify the solver that
  derives them from a volume shape?

## Verdict

VERDICT: NEEDS_CHANGES
