# Review v0

## Summary
code_v0 faithfully implements plan_v2. The reused-core extractions are proven behavior-preserving by four
full-volume (800³) relabeling-invariant partition-equality checks, the two genuinely-new paths
(prob1_carve bump-safe guard, prob2 caliber gate) are correct and covered by the self-test, and the
authoritative `decode_axon --full` run (coordinator, 1038s, exit 0) meets the project's primary goal —
false-merge reduction. Full-volume valid-tube result:

| decode | VALID vol | bumps | parallel | note |
|---|---|---|---|---|
| decode_v2 | 53.5% | 152 | 9 | base |
| decode_p1p2 | 62.7% | 117 | 8 | prior best |
| decode_p1p2eg | 65.5% | 89 | 0 | bump-safe extension |
| **decode_axon** | **62.5%** | **84** | **1** | fewest false merges; coverage ≈ decode_p1p2 |

`decode_axon` posts the LOWEST false-merge counts of any decode (bumps 84, parallel 1) at coverage flat
with decode_p1p2, real NERL 0.6114 (> tube-bb 0.593; NERL secondary, GT over-split). One minor, non-defect
finding: the new caliber gate trades ~3 pts of valid volume vs decode_p1p2eg — a tunable, not a bug.

## Diff Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
(dev/ and .agent/ are gitignored; reviewed the full current file contents directly, not a git diff.)

## Findings
- [pass] Extraction fidelity: `prob1_carve(bump_safe=False)`≡decode_p1, `prob2_merge(area_tol=None)`≡
  decode_p1p2, `prob1_extend`≡decode_p1p2eg, `decode_sections`≡decode_v2_c_constrained — all PASS as
  full-volume relabeling-invariant partitions. The legacy CLIs are unaffected.
- [pass] decode_axon.py orchestration: aff loaded once, affxy shared, strong→carve→extend→merge→drop-crumb
  data flow correct; global z0-800 section cache sliced [z0:z1]; `--zslice` requires a tag (no overwrite)
  and is labeled SMOKE; `--self-test` returns before any I/O; single required mode; thr validated.
- [pass] prob1_carve bump-safe path: `_ComponentAreas` union-find accumulates orphan-chain profiles;
  `_candidate_bump_safe` requires host bump present→removed over the run AND no new run on host or on the
  above+below+bridge relink; 3-distinct-roots guard prevents re-carving unified components.
- [pass] prob2_merge: `region_graph(seg, aff)` in-memory (channel-correct, equality-proven); the parallel
  z-overlap veto is UNCHANGED and precedes the caliber gate, so parallels can't slip through.
- [pass] score_array refactor is behavior-preserving; `report()` prints one labeled metric line (no
  double label; confirmed in run output).
- [minor] prob2 caliber uses denominator z-extent (`zmax-zmin+1`) where plan_v2 said "voxels/occupied-
  slice"; equivalent for gap-free tubes, a reasonable proxy — cosmetic wording vs impl.
- [minor] The shipped default `prob2_merge(area_tol=0.5)` yields the fewest bumps (84) but VALID volume
  62.5%, ~3 pts under decode_p1p2eg (65.5%); the caliber gate over-rejects caliber-mismatched sequential
  merges. plan_v2 made this a knob with an `area_tol=None` fallback. Recommend a coordinator follow-up
  comparing `area_tol=None` to pick the shipped default. Not a correctness defect.

## Tests to Add
None required for correctness — the self-test covers masks, carve accept/solid-reject/bump-adding-reject,
extend gate accept/reject + real extension, prob2 sequential-accept/parallel-reject/caliber-reject, crumb
drop, and all four deferred stubs. (Optional future: a full-volume regression assertion once the best
`area_tol` default is chosen.)

## Questions
None blocking.

## Verdict
VERDICT: APPROVE_WITH_MINOR_COMMENTS
