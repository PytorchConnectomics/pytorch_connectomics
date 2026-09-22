# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v0_review.review.raw.md`. Sixteen major findings and one minor, none softened here.

Placement inside `connectomics/decoding/checkpoint/` is judged sound and non-circular, and the plan
nominally covers the five commands, twelve tests, four ordered actions, seven invariants and the
design document. It is not executable safely: the regression oracle is circular (the refactored
scripts would be compared against themselves), several action and verification contracts are
vacuously satisfiable, and — the finding with teeth — the proposed tests could all pass while the
framework repairs nothing useful.

## Findings

- [major] Test #12 has no independent pre-refactor oracle. Rewiring the development scripts and then comparing them with the lifted functions risks comparing the new implementation with itself. The plan must first capture checked-in golden inputs and exact expected arrays, metrics, or hashes from the untouched scripts. Changes to gitignored `dev/` scripts are also absent from a normal review or clean checkout, so tests cannot depend on those modified files.

- [major] The plan does not require the measured `worst3` outcomes to be reproduced or asserted. Merely running dry-run and full-run commands does not demonstrate improvement. The validation should record the strict contamination, cross-nucleus sharing, and dominance figures and compare them with the supplied `115,479 → 5,071`, `1 → 0`, and dominance targets. Otherwise the implementation could be another mechanism with an effective result of 0.0000.

- [major] Synthetic test #3 is insufficient as proposed. An implementation could carve away anchor voxels, assign arbitrary checkerboard territories, or discard most of the component and still satisfy “no component contains both anchors.” The test needs a known seeded-watershed oracle, voxel conservation or explicit abstention accounting, expected anchor-driven territories, and assertions that non-anchor component mass is meaningfully assigned.

- [major] The numerical lift omits essential backend behavior. The proposed function list does not include the reusable same-anchor consolidation logic from `nucleus_anchor_merge.py`, nor does it specify cost construction from affinities, marker construction, connectivity, tie-breaking, pooling/upscaling, anisotropic z/y/x handling, or deterministic handling of unreachable voxels. These choices determine the actual watershed result.

- [major] `min_share = 0.02` is unsupported by the supplied evidence and conflicts with the instruction not to invent scientific defaults. The supplied 0.3% failure establishes that an absolute threshold was unsafe; it does not establish 2% as correct. The implementation must preserve a cited existing value or require an explicit configured threshold.

- [major] `qualifying_anchors(hist, min_share)` cannot reliably calculate a fraction of each anchor’s own mass when execution is restricted to a repair ROI or block. It needs authoritative whole-anchor totals or an explicitly defined reference scope. Summing only the cropped histogram can turn a tiny truncated overlap into 100% of the observed anchor mass.

- [major] The action dependency rule is not fail-closed. Refusing consolidation only when an overlapping split remains unexecuted allows consolidation when the split is missing from a malformed plan, when a serialized status falsely says it executed, or before `forbid_merge` has installed the cannot-links. Consolidation must require executor-recorded successful split and cannot-link actions, a resolved supporting conflict, and no unresolved distinct-anchor conflict over the exact scope.

- [major] Planning `forbid_merge` before split results exist is not defined. The plan needs deterministic symbolic territory references or an explicit, validated result-binding mechanism that resolves anchor territories after splitting without silently synthesizing new unreviewed actions. The same issue affects stable action IDs and replay.

- [major] “Every output territory is a subset of the input component” is too weak and not operationally defined. Deleting all voxels would satisfy it. Verification must use exact parent-component and scope masks and check refinement, conservation or explicit abstention, absence of voxels from other input components, exact changed-voxel accounting, and outside-scope partition equivalence. A bounding-box overlap alone is insufficient for action ordering or this postcondition.

- [major] The seven invariants are named but not made executable. In particular, the plan does not define the dominant-territory tolerance for invariant 2, authoritative execution evidence for invariant 5, what constitutes a no-op for invariant 6, or which deterministic artifacts invariant 7 compares. Hashing records that include timestamps contradicts stable IDs and identical-plan determinism unless volatile provenance and timing fields are excluded or normalized.

- [major] The placement is only partly resolved. Keeping the implementation inside `decoding` fits the supplied dependency direction and creates no inherent cycle, provided it imports only permitted dependencies and sibling modules. However, a `scripts/checkpoint.py` that dispatches directly into `decoding` conflicts with the stated `scripts → runtime` contract. The CLI should enter through `runtime`, which may call the decoding checkpoint engine. A backend constraint adapter must also avoid importing a downstream runtime or decoder owner back into `decoding`.

- [major] The scalable I/O and export layer is absent from the proposed decomposition. Pure array kernels are appropriate, but the operator still needs bounded/blockwise reads, input artifact hashing/versioning, exact scope-mask storage, corrected-segmentation output in a decoder-consumable format, append-only execution records, timing and affected-voxel counts, and avoidance of full-volume copies. None of these are covered by the synthetic tests.

- [major] The cannot-link adapter is left as “whatever the global decoder consumes,” with no verification that the exported constraints are consumable. The plan needs to identify the existing format during implementation and add an adapter-level integration test. Otherwise it can pass by writing a canonical manifest that no later global pass can use.

- [major] The action/schema coverage is incomplete. The task requires typed support for `annotate` in addition to the four ordered operational actions, but the plan only discusses the latter. It also says only three descriptor keys will be emitted without explaining how per-anchor overlap voxel counts, anchor IDs, component identity, and repair scope required by Describe are represented.

- [major] The plan applies splitting to every component satisfying `distinct_count >= 2` but does not reconcile this with the supplied result that competitive splitting is valid only for soma-contact cases and not the 16 neurite-bridge cases. It must define a conservative eligibility boundary—such as explicitly supplied contact scopes or an existing trusted selector—and record unsupported bridge conflicts without pretending to repair them.

- [major] The five command names are present, but their artifact contracts are not. The revised plan should define whether `describe` serializes descriptors only, how `plan` consumes or reproduces them, how `apply` binds a frozen plan to hashed inputs, and how `verify` locates the immutable plan, execution log, exact scope, and output segmentation.

- [major] The design-document plan does not enumerate the required contents. It must explicitly include all seven requested explanations, the four-way distinction among hard forbids, future holds, local refinement, and global reclustering, all listed future descriptor examples, and both a complete pass specification and a complete emitted-plan example.

- [minor] Updating the public API snapshot should not be assumed. A private decoding subpackage can remain outside `connectomics.decoding.__all__`; adding a facade export merely to satisfy discoverability would conflict with the small-public-API rule.

## Questions

1. What existing configuration or historical run establishes the qualifying-anchor threshold? If none does, should the CLI require it explicitly with no scientific default?
2. How will soma-contact scopes be selected without applying the known-invalid watershed repair to neurite bridges?
3. What checked-in fixture and immutable pre-lift outputs will serve as the independent regression oracle when `dev/` is unavailable in a clean checkout?

## Verdict

VERDICT: NEEDS_CHANGES
