# Plan v1 Review

## Summary

Reviewer: codex, read-only, exit 0, repository unmutated. Raw transcript at
`state/plan_v1_review.review.raw.md`.

**2 major, 2 minor, `READY: no`** — down from 7 major and 3 minor. The residue is
now entirely *concrete inputs*: how the whole-volume `mean` reference is produced,
and real paths in place of `<test_out>`/`<run_prefix>`/`BBOX` placeholders.

Both are accepted. Finding 1 is the more interesting: `plan_v1` asserted a
reference "is produced" without naming the command, its inputs, its output
artifact, or when it runs relative to the equivalence test — which, as the
reviewer says, leaves a workflow decision to the implementer.

No finding concerns distribution. The scope split continues to hold.

## Findings

### Major

1. **The whole-volume `mean` reference is unspecified.** The volume has a
   whole-volume decode, but it is the published `max` result, which the plan
   explicitly excludes as the reference. `DECODE=whole|chunked` is named, but the
   whole-volume-`mean` command, inputs, output artifact and invocation order
   before equivalence testing are undefined.
2. **Paths are placeholders.** `<test_out>`, `<volume>`, `<run_prefix>` and `BBOX`
   are not concrete values and have no stated source, so `preflight` cannot be
   implemented or run without deciding the test-volume path, run prefix, bounding
   box and output locations.

### Minor

3. The unsupported-criterion error must explicitly include the available chunked
   stage representation (`agglomerate_mean_edge`), not only the requested value,
   the accepted set, and the `max` missing-binary note.
4. The plan places the alignment and multi-chunk guards in `resolve` but does not
   guarantee they run **before** CloudVolume allocation or layer creation in the
   decode entry point. The invocation ordering belongs in the implementation
   contract.

## Questions

- Finding 1's cleanest answer is that the reference is produced by the **same
  `DECODE=whole` path already in the driver**, run once at `merge_function: mean`
  and the configured threshold, with its output becoming a named input to the
  equivalence test. That keeps one decode implementation rather than a special
  test-only route. Confirm that is preferred over a standalone script.
- Finding 2 needs a decision about the affinity source. The VM that produced the
  2026-09-22 ExPID108 affinity is deleted, so the durable artifact is the
  published copy in GCS rather than a container-local path. `plan_v2` should name
  that object and treat the container path as the historical provenance only.

## Verdict

VERDICT: NEEDS_CHANGES
