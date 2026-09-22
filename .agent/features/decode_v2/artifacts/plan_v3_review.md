# Plan v3 Review

## Summary
Codex plan_v3 review; resolved 7 prior findings, raised 8 finer edge-cases. READY: no.

## Findings

Several prior findings are resolved: constrained-UF initialization, forced-union conflict checks, detector defaults, mandatory terminal links, verification gates, SLURM syntax, and documentation scope. The following material gaps remain.

- [major] The label-uniqueness assertion must exclude background label `0`, which necessarily occurs on multiple slices. As written, the full-volume load would abort.

- [major] Eligibility is still not upstream-only. Components and `min_len` use the entire graph after removing `t`, so downstream structure can alter eligibility or satisfy the required axial depth. Restrict traversal and depth measurement to upstream transitions from each seed.

- [major] Propagation is not exhaustive or reproducible. The plan selects one maximum-overlap successor before introducing a two-successor separation branch, without defining branch precedence or per-side acceptance thresholds. Already-owned successors, partially owned terminal pairs, one-sided overlaps, degenerate watershed results, and multiple successors that do not form a valid pair fit no stop/rollback rule.

- [major] “2×2 greedy, deterministic” does not define a correct terminal assignment. Greedy matching can choose a lower-total-overlap bijection, and no tie-break or minimum overlap for both matched sides is specified. Compare both bijections, validate both matches, and pin tie handling.

- [major] The task defines fusions as having at least two incoming sections and requests splitting each fused section, while plan v3 handles exactly two and unilaterally excludes `N>2`. That scope reduction needs explicit acceptance or support for higher multiplicities.

- [major] Slice bounds remain unresolved. A post-crop shape assertion does not reject negative or silently clamped Python slices. Require `0 <= z0 < z1 <= depth` and consistent bounds across affinity, sections, and GT.

- [major] Slice artifact behavior is contradictory. Stage 6 saves unsuffixed a/b/c files, while the pilot specifies a z suffix only for the selected candidate. Define whether slice a/b/c artifacts are emitted and suffix every emitted slice artifact to prevent overwriting full-volume or other-slice results.

- [major] The data-free `--self-test` conflicts with the blanket requirement that exactly one of `--full` and `--zslice` be supplied. Define self-test as an early-exit exception or a third mutually exclusive mode.


## Questions
- plan_v3 (round 4) resolved 7 prior findings; Codex raised 8 finer ones (real: background-
  label-0 assertion, slice-bounds; rest coder-resolvable detail). Review is NOT converging
  (7->7->8 findings, each finer). Decision: proceed to code (Claude or Codex) folding all
  accumulated findings in, vs another plan round.

## Verdict
VERDICT: NEEDS_CHANGES
