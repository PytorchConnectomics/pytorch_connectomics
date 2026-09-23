# Plan v0 Review

## Summary

Reviewer: codex, read-only, exit 0, repository unmutated. Raw transcript at
`state/plan_v0_review.review.raw.md`.

**7 major, 3 minor, `READY: no`** — but the character of the findings differs
sharply from the canceled run. Every one is a *specific missing value or
definition*: name the threshold, name the second chunk size, compare key sets
rather than counts, state the clipping constant, enumerate the accepted criteria.
None is architectural, and **none concerns distribution**, so the scope split did
what it was meant to do.

Finding 3 is the sharpest: the plan says the completeness guard inspects
CloudVolume objects *and* that config resolution performs no I/O. Those
contradict, and the fix is a stated resolve/preflight boundary.

All findings accepted. One further defect, **not raised by the reviewer**, was
found while pricing finding 2 and is recorded here because it would have
invalidated the acceptance test: see Questions.

## Findings

### Major

1. **No `AGG_THRESHOLD` value.** The config has no default, so the equivalence
   test cannot run without an additional decision.
2. **The second `CHUNK_SIZE` is unnamed.** "A second alignment-valid size" is not
   executable; it needs the exact size and its storage-chunk pairing.
3. **The completeness guard contradicts the no-I/O claim.** It must inspect
   objects, while verification says resolution does no filesystem or subprocess
   I/O. Needs a concrete resolve/preflight boundary and invocation order.
4. **Count comparison is insufficient.** Equal counts can hide a missing expected
   key plus an unexpected extra one. Compare the exact expected key set, including
   edge chunks and prefix/schema handling.
5. **The S1 integrity check leaves the mapped threshold unspecified.** It needs the
   compressed→probability formula and the exact value used for both `max` decodes.
6. **The tail-clipping rule is not specified.** Referencing `EPS` in an imported
   function gives neither its value nor the clipping-before-logit behaviour; that
   is not an artifact contract.
7. **Criterion support is not concrete.** "`rlme`, `cs` as ABISS names them" does
   not define accepted values, exact stage names, or threshold semantics. The
   guard must enumerate the canonical mapping.

### Minor

8. The `B`/`I` definition needs edge conventions: whether planes at volume
   boundaries count, whether bounding boxes are half-open, and whether background
   label `0` participates.
9. "Chunk-size invariance" does not say whether the comparison is chunked-vs-
   chunked or each chunked result against the whole-volume reference.
10. The 8×-larger reuse claim has no concrete smoke-test path, particularly given
    the converter is described as non-scaling.

## Questions

**A defect the review did not find, which would have voided acceptance.** Pricing
finding 2 required checking the proposed chunk sizes against the *test volume*
rather than the 100 µm cube. `ExPID108_32x_Cortex_L1_01` prepares to
`(503, 650, 650)` ZYX = `(650, 650, 503)` XYZ, so the plan's default
`CHUNK_SIZE [2048, 2048, 80]` yields chunks **`[1, 1, 7]`** — a single chunk in X
and Y. The "multi-chunk acceptance test" would therefore have exercised **only Z
seams**, and an X/Y seam defect would have passed silently.

This is the same failure mode the plan claims to be built around — seams failing
quietly — reproduced in the test design itself. `plan_v1` must separate **test**
chunk sizes from **100 µm** chunk sizes and require multi-chunk decomposition on
all three axes. Candidates, both alignment-valid:
`[256,256,128]` / `[128,128,128]` → 3×3×4 = 36 chunks, and
`[192,192,96]` / `[64,64,96]` → 4×4×6 = 96 chunks.

Smaller: finding 7 asks for the canonical criterion mapping. Since this change
exercises only `mean` and the semantics of `rlme`/`cs` are unestablished, is the
right answer to enumerate all three, or to accept exactly `{mean}` and raise on
everything else including them?

## Verdict

VERDICT: NEEDS_CHANGES
