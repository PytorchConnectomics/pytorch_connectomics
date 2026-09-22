# Plan v0 Review

## Summary
Reviewer (Codex, read-only) judged the scope appropriate but the load-bearing relink design not
yet sound or verifiable. `READY: no`. Raw transcript: `state/plan_v0_review.review.raw.md`. The
core issue: forbidden-edge deletion cannot guarantee axon separation (indirect A–X–B paths), and
the ungated "split all 539 fusions" interacts badly with a hard constraint (would convert
same-axon over-splits into persistent false splits). The relink must be a seed-anchored
constrained union-find paired with a fusion-eligibility rule, with globally-unique side IDs and
deterministic run ownership. Verification must include graph-level invariant tests, and the
acceptance rule must honor the task's no-regression requirement.

## Findings
- [major] Forbidden-edge deletion ≠ separation: union-find can reconnect A/B via A–X–B, another
  run, or an unsplit section. Use component-level cannot-link (constrained union-find), pieces
  anchored to their incoming-seed identity; reject unions whose components carry opposing anchors.
- [major] Same-axon recovery contradicts the constraint: every fusion gets A/B sides regardless of
  GT. If both sides may reconnect "through the same tube," that's the indirect path that also
  defeats true separation. With ~539 candidates but ~48 true fusions, need an explicit eligibility
  rule (e.g. independently-traced upstream/downstream tubes) or a hard-vs-soft constraint policy.
- [major] Side identity underspecified: `A_ids/B_ids` must be globally-unique node IDs `(z,label)`,
  include seed anchors + propagated + terminal pieces, and survive relabeling/`merge_id`.
- [major] Fusion-run ownership ambiguous: propagating per entry can overwrite sections handled by
  an earlier run → stale side sets. Need deterministic run-head selection, processed-section
  tracking, conflict handling, ordering.
- [major] N-way inconsistent: area test sums all incoming but only the 2 largest are split/
  constrained. Restrict to exactly-2 incoming (recompute area for that pair) or define N-way.
- [major] Acceptance rule contradicts task: `(c) < 0.593` is not acceptable (task requires
  no-regression). Must trigger another iteration or a safe fallback, not be declared success.
- [major] Evaluation deliverable incomplete: `decode_v2.py` itself must print real NERL,
  oracle-merge NERL, and instance metrics for all ablations; define exact output names and
  `--zslice` output isolation.
- [major] Verification lacks a focused invariant test: add deterministic graph-level cases (direct
  + indirect A/B paths, seed anchoring, interacting runs, union ordering, repeated slice-local
  labels, allowed same-side unions). The 96-slice smoke test can't validate the relink invariant.
- [minor] Validation commands should activate/use `pytc`; SLURM wrapper should pass the exact
  generated path to both eval commands.
- [minor] "Not much worse", "closer to 62", "~2625 instances" are not executable thresholds —
  make reporting-only or assign concrete bounds.

## Questions
- Reviewer answered the plan's question: ship the constrained union-find in v0 (seed-anchored +
  fusion eligibility); forbidden-edge deletion is only an ablation.
- Open for v1: exact fusion-eligibility rule (how to identify "true" fusions GT-free without
  re-solving the ambiguity) — v1 should specify a concrete, testable rule and accept that
  imperfect eligibility is handled by the constrained union-find preserving false-positive splits.

## Verdict
VERDICT: NEEDS_CHANGES
