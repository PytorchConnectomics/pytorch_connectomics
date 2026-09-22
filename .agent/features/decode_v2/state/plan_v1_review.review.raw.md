# Plan v1 Review

The constrained union-find concept and exact-two-input handling are improved, but several prior majors remain unresolved.

## Findings

- [major] Node identity remains contradictory. Labels unique only within a slice cannot be stored directly while repeated labels across slices remain distinct. Use `(z, label)` or an explicit volume-wide remapping consistently for edges, processed sets, run sides, and relabeling.

- [major] Eligibility is not truly an independent upstream-tube test. `tube_of` comes from the full conservative graph, potentially including the candidate transition or downstream paths, and `min_len` measures whole components. Define upstream-only traces excluding the ambiguous fusion transition. Bidirectional tracing can remain optional.

- [major] Run ownership still permits stale seeds. An earlier run may replace a section later referenced as an incoming seed by another unprocessed run-head. Target-only `processed` checks do not handle this; specify remapping, recomputation, or deterministic conflict skipping.

- [major] The relink only adds cannot-link annotations; it does not guarantee each split piece is connected to its seed tube. Natural-separation terminal sections are also absent from the side sets. Define mandatory same-side continuity/terminal assignment and test resulting connectivity, not merely A/B separation.

- [major] Fallback acceptance is internally inconsistent. Selecting `max(a,c)` makes a separate final artifact non-regressing; it does not make candidate `(c) >= (a)`. Define candidate and selected artifacts separately, check the selected result against both the recomputed base and absolute `0.593`, and identify GT-based selection explicitly as evaluation-only if retained.

- [major] The `>=0.78` oracle check is not tied to the correct artifact. Evaluating oracle-merge on linked `(a)/(b)/(c)` does not establish the force-split section ceiling. Explicitly evaluate the unlinked force-split substrate and bind the threshold to that result.

- [major] Verification does not cover the remaining force-split invariants: stale incoming seeds, overlapping/consecutive runs, terminal assignment, and repeated slice-local IDs through the complete split-and-relink path. The current tests primarily exercise the union-find conflict dictionary.

- [major] CLI and pilot behavior remain underspecified: `--tag` has no defined type or filename effect, ablation filenames are not exact, and slice-mode metrics do not define GT/skeleton cropping and coordinate rebasing.

- [minor] Descending IoU needs a stable node-ID tie-breaker for deterministic equal-score edges.

- [minor] Pilot/self-test commands still use bare `python`, and the two proposed `force_split` signatures disagree on `iou_edges` versus `incoming`.

Findings tagged [minor]/[major] above.
READY: no