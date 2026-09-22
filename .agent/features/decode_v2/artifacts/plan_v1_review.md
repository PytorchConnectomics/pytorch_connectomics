# Plan v1 Review

## Summary
Reviewer (Codex, read-only): constrained union-find + exact-2-input handling are improvements, but
7 majors remain — mostly precise implementation-spec gaps (node identity, upstream-only
eligibility, stale seeds, same-side continuity/terminal assignment, candidate-vs-selected fallback,
oracle threshold binding, verification coverage, CLI/slice-mode). `READY: no`. Raw:
`state/plan_v1_review.review.raw.md`.

## Findings
- [major] Node identity: use `(z,label)` or a volume-wide remap CONSISTENTLY (edges, processed set,
  run sides, relabel); do not store slice-local labels directly while assuming cross-slice distinct.
- [major] Eligibility must be an UPSTREAM-only tube test excluding the fusion transition; `tube_of`
  from the full graph + whole-component `min_len` is not that.
- [major] Run ownership still allows STALE SEEDS: an earlier run can replace a section later used as
  an incoming seed. Specify remap/recompute/deterministic conflict-skip.
- [major] Relink only adds cannot-link; it must also GUARANTEE each split piece connects to its seed
  tube (mandatory same-side continuity) and assign natural-separation TERMINAL sections to a side.
  Test connectivity, not just A/B separation.
- [major] Fallback inconsistent: `max(a,c)` makes the SELECTED artifact non-regressing, not the
  candidate `(c)`. Separate candidate vs selected; check selected vs recomputed base AND absolute
  0.593; mark GT-based selection eval-only.
- [major] The ≥0.78 oracle check must bind to the UNLINKED force-split substrate, not linked a/b/c.
- [major] Verification must cover stale seeds, overlapping/consecutive runs, terminal assignment,
  repeated slice-local IDs through the full split-and-relink path — not just the union-find dict.
- [major] CLI/pilot underspecified: `--tag` type/effect, exact ablation filenames, slice-mode
  GT/skeleton cropping + coordinate rebasing.
- [minor] Descending-IoU needs a stable node-ID tie-breaker.
- [minor] Pilot/self-test use bare `python`; the two `force_split` signatures disagree
  (`iou_edges` vs `incoming`).

## Questions
- Confirm slice-mode may report INSTANCE metrics only (NERL needs the full-volume skeleton graph),
  or require local-skeleton cropping. plan_v2 chooses instance-only for `--zslice`.

## Verdict
VERDICT: NEEDS_CHANGES
