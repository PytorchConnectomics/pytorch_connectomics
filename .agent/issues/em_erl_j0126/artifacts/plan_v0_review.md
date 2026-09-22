# Plan v0 Review

## Summary

Codex (coder) reviewed plan_v0 read-only. The plan is close but not ready: one
material correctness gap in how node coordinates are recovered for sampling, and a
verification criterion too weak to catch a silent mis-registration. Two `[major]` and
two `[minor]` findings, all fixable in a plan revision. Raw transcript:
`state/plan_v0_review.review.raw.md`.

## Findings

- **[major] Canonical node-coordinate source.** The plan proposes
  `graph.node_coords_zyx.round().astype(int64)` as the sampling source. Use the
  canonical `graph.get_nodes_position(None)` (matches `volume_eval.py`) or the original
  integer vertices in graph order instead of a new rounding rule, so the sampling grid
  exactly matches the graph's node convention. (Note: coords are < 2^24, so float32 is
  exact here and round-vs-truncate is numerically identical — but the canonical API is
  the right, unambiguous choice.)
- **[major] Registration proof in verification.** Relying only on "assignment-zero
  ratio is small" can miss coordinate swaps/offsets/scale errors that still hit nonzero
  segments. Add explicit sentinel checks to the end-to-end run: skeleton `"0"` points
  map consistently to segment `1465128` (observed), and out-of-bounds count ≈ 37.
- **[minor] Drop `-r/--resolution`.** It is out of scope, and the task asks to preserve
  voxel-unit ERL. A physical-units option risks incomparable results; remove it.
- **[minor] CloudVolume opener under-specified.** Pin the exact URL normalization and
  opening form (make it a small testable helper), rather than leaving `use_https`/
  `gs://` handling implicit.

## Questions

- Will the implementation preserve an exact raw-vertex array in graph order, or rely
  only on `ERLGraph.node_coords_zyx`? (Resolved by adopting `get_nodes_position(None)`.)
- Can the real-run verification include the skeleton `"0"` -> `1465128` and OOB-count
  checks before computing ERL? (Yes — folding into plan_v1.)

## Verdict

VERDICT: NEEDS_CHANGES
