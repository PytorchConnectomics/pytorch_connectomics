# Plan v1 Review

## Summary

Plan v1 is not yet executable with trustworthy G0/G1 pass/fail results. Four prior major findings and both minor findings are resolved, but eight prior major findings remain at least partly open. Several new gate-definition problems also appear.

No repository files were inspected or changed.

## Findings

### Prior-finding audit

| Prior finding | Status | Assessment |
|---|---|---|
| 1. Harness sanity anchors | Resolved | Adds `0.601431±0.0005`, `0.545±0.001`, `0.836±0.003`; `0.772069` is supplemental. |
| 2. Canonical G0 pipeline | **[major] Open** | Source/realizer choices are narrowed, but medial-band construction, coordinate conversion, edge states, and exact offset banks remain undefined or inconsistent. |
| 3. Safe skip rasterization | **[major] Open** | Rasterization was removed instead of made collision-safe, contrary to Phase-0 step 4. |
| 4. Perturbation protocol | **[major] Open** | C1/C2 side definitions and deterministic section selection remain unclear; C3 has no named distribution or pass threshold. |
| 5. Predicted-SDT `0.985` | Resolved | Predicted SDT is deferred; the remaining `0.985` is explicitly a GT-SDT gray-zone boundary. |
| 6. One G1 starting map | Resolved logically | `BASE_SEG=cc3d@0.66` is bound consistently. Physical cache binding remains open under finding 10. |
| 7. Bridge-set/source conflict | **[major] Open** | Script role is chosen, but localized events, neighborhoods, hyperedges, deduplication, and matching remain undefined. |
| 8. Proposal-cap logic | **[major] Open** | Raw/capped reporting is added, but raw maxima of 9–12 are incorrectly allowed to pass, and two-endpoint cap arbitration is undefined. |
| 9. Executable realizer | **[major] Open** | Seeds and supervoxels are both `BASE_SEG`, so unchanged output can pass; coverage and repaired-seed gates remain absent. |
| 10. Exact affinity asset | **[major] Open** | A basename and `outputs/...` are not an exact path; dataset key, axes, halo, and channel-edge mapping are missing. |
| 11. Runtime/verdict semantics | Resolved | Smoke cannot substitute for completion; gray-zone and failure reporting are defined. |
| 12. Indexed 43-channel design | **[major] Open** | The arithmetic sums to 43, but indices, offsets, formulas, masks, and veto targets are only promised. |
| 13. Read-only prior art | Resolved | New scripts adapt/import without modifying tracked files. |
| 14. Second chunk | Resolved | Correctly deferred. |

### Remaining and new blocking problems

- [major] `plan_v1 §G0 steps 1–4` does not define the representation exactly. `node_coords_zyx` is painted into an affinity-shaped `1000×1000×450` volume without specifying permutation, global-to-local origin subtraction, or edge voxelization. A vertex volume is not necessarily the required medial band. Define the exact coordinate transform, medial-band/edge rasterization, collision policy, and assertions against `seg.h5`.

- [major] `§G0 step 4` does not match the Phase-0 sweep. It adds axis-distance 3/4 banks, omits the required full 49-offset half-shell, and never enumerates the cumulative tuples/counts. Replace the canonical sweep with `O1`, `+axis-two`, `+nine Δz=2 offsets`, and `+full half-shell`; list every tuple and deduplication rule. Extra radius-3/4 banks may only be diagnostics.

- [major] `§G0 steps 3–5` removes skip-path rasterization and therefore cannot audit “zero accepted paths traversing a foreign GT medial band.” Keep union-find roots authoritative, but rasterize accepted paths with a fixed corridor rule, reject foreign-band/root collisions, and prohibit spatial CC from creating unions.

- [major] `§G0 Perturbation protocol` still cannot yield reproducible closure rates. Require C1 events to have exactly defined side anchors, define eligible C2 sections and deterministic selection, name or derive the C3 crumb distribution, state whether RNG is reset per condition, and provide a condition-by-condition gate table. Define fragments/GT and retained length using physical erlgraph length so I2 is respected.

- [major] `§Step 0`, `§G1 header`, and the Files table lack executable asset bindings. Supply complete repo-relative affinity/cache/LUT paths, HDF5 keys, stored shapes and axes, crop/halo convention, value range, and ch0/ch1/ch2 directed-edge mapping. `outputs/.../seed101/...` does not resolve the prior blocker.

- [major] `§G1 steps 1–4` can overstate recall. “Adjacent split pair” is undefined on a branching erlgraph, while the stated hit rule checks fragment IDs only and omits the required endpoint-neighborhood match. Define localized oracle events, event coordinates, neighborhood radius/metric, boundary margin, pair/hyperedge handling, deduplication, and one-to-one matching. Also pin the search-distance formula, radius source, and Hermite-curvature threshold.

- [major] `§G1 step 4/pass` misclassifies the proposal-count gate. A forced top-8 cap makes the post-cap success condition tautological. Evaluate raw post-pruning maximum as: `≤8` pass, `9–12` gray, `>12` kill. Define whether an undirected proposal must survive both endpoints’ rankings and whether proposals/recovered-bridge uses raw or capped proposals.

- [major] `§G1 step 5(b)` uses every oracle bridge instead of bridges recovered by the capped candidate set. That bypasses the candidate generator. Produce separate full-oracle, raw-candidate-oracle, and capped-candidate-oracle segmentations; run the feasibility realizer from capped recovered repairs.

- [major] The absolute Phase-1 candidate-oracle gates (`≥0.752`, `<0.742`) are correctly recognized as meaningless on a crop already scoring `0.836`, but no local replacement is declared. Either label G1 as a partial recall-only surrogate that cannot establish full Phase-1 success, or predeclare a scale-appropriate gate such as recovered fraction of the full local oracle gain. The current plan cannot claim “every success criterion passes.”

- [major] `§G1 step 5` remains trivially passable. Because `BASE_SEG` supplies both full-body supervoxels and all seeds, abstaining everywhere can reproduce the input exactly. Define distinct sparse identity seeds and body supervoxels, the source of `sdt`, traversal and competition equations, a nonzero eligible-coverage requirement, and require both control and capped-repaired runs to remain within `0.005` with zero forbidden unions or multi-ID assignments.

- [major] `§DESIGN.md` must pin the promised channel table in the plan: indices `0–42`, all nine affinity offsets, tangent sign convention, medialness/distance formulas, all 13 edge offsets, DEFER/ignore encoding, three veto definitions, and sparse-skip query/target representation. `9+1+1+3+26+3` alone is insufficient.

- [major] `§Verification Plan` is not resource-grounded for a cheap gate: multiple 450M-voxel watersheds, voxel-level geodesics, and per-bridge full-score calls have no memory/runtime strategy. Specify reusable buffers/dtypes, exact incremental node-LUT gain computation, and either a bounded full-volume implementation or gate-equivalent tiling with halo/reconciliation rules.

## Questions

- Is G1 intended to certify full Phase 1, or only a center-crop recall surrogate? The verdict and replacement candidate-oracle gate must reflect that choice.
- What exact affinity file, HDF5 key, axes, and channel-offset mapping produced the `0.836` anchor?

## Verdict

VERDICT: NEEDS_CHANGES

READY: no