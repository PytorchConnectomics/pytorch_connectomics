The plan identifies the right research lever and correctly recognizes that deleted `decode_v1` chunks must be regenerated. It is not yet ready for implementation because several acceptance and safety contracts remain unresolved.

- [major] The acceptance metric violates the task contract. The task requires every step to be judged by whole-volume length²-weighted NERL, while this plan declares success from block-cropped NERL and defers the 600-box evaluation. Cropping changes skeleton lengths and weights, truncates long chains, and hides ownership outside the block. Treat block evaluation as a smoke test only; either restore the full decode_v1 substrate and use the whole-volume path, or explicitly revise the task’s acceptance criterion.

- [major] The block-evaluation assembly contract is incomplete. Every foreground segment must first receive a unique identity `(chunk_key, local_label)`, with background 0 preserved, before constructing both baseline and linked variants. Otherwise equal local labels in different chunks become accidental merges. The two variants should share identical voxel support, sampled nodes, and missingness and differ only by the accepted union-find remap.

- [major] The GT crop procedure is underspecified. A valid block experiment requires a complete rectangular, half-open block; one outer-boundary crop/split of the canonical whole-volume `test_50_skeletons.h5`; one global-to-block coordinate translation; and fixed `RES=[10,10,10]`, length threshold 1000, and merge threshold 1. Per-chunk skeleton concatenation, an L-shaped block, or switching ambiguously to the “dedup combined” GT could produce misleading NERL.

- [major] The oracle gate implements the wrong condition. `linked_oracle >= baseline_oracle` is not “oracle flat.” Require equality within a stated tight numerical tolerance; an increase should trigger investigation of sampling, namespace, or evaluator inconsistency rather than pass automatically. Also, oracle-flat means no newly detected merge among sampled test-50 owners—not zero false merges globally.

- [major] The proposed algorithm remains overlap-gated rather than clearly implementing endpoint continuation. Requiring `--min-ov 50`, an IoU threshold, and mutual-best IoU prevents low- or zero-overlap continuations from becoming candidates—the cases needed to move beyond ordinary face stitching. Specify geometric candidate generation, physical search radius, tangent polarity, feature normalization/formulas, hard thresholds/defaults, mutual-best basis, deterministic tie-breaking, and endpoint ambiguity handling.

- [major] The nucleus firewall is not yet safe or executable as specified. Fix one marker source and define its axis order, resolution/origin conversion, and chunk-offset mapping. Each globally unique base segment/root must hold a set of marker IDs, not “which marker, if any”; unions are allowed only when the combined set has cardinality at most one. Pre-existing multi-nucleus decode_v1 segments and empty/misaligned marker mappings must fail or be explicitly quarantined. Add focused tests including the transitive `nucleus A → unmarked → nucleus B` case.

- [major] Substrate validation is too weak and internally contradicted by the suggested reuse of “hero” finals. Those files are not established as canonical decode_v1 and must not be substituted without proven equivalence. Matching only per-chunk oracle within ±0.01 also does not prove fidelity; compare base, oracle, missing fraction, skeleton count, and GT length using the historical evaluator and justified deterministic tolerances.

- [major] The evaluation block and resource model are not fixed. “At least four chunks,” a radius-1 command that may select 27 chunks, and a 2×2×2 estimate describe materially different runs. Predeclare an exact rectangular in-bounds block, its internal faces and expected GT crossings, before inspecting results. Because 1008³ chunks make dense assembly expensive, specify chunk-backed/streamed relabeling and evaluation, preferably through a whole-volume scorer that consumes the union-find LUT/remap.

- [minor] Define the certificate schema: stable endpoint/global IDs, face, raw cue values, thresholds, mutual ranks, commit order, pre-union marker sets, decision, and rejection reason. Nucleus conflict must be an unconditional dynamic rejection, not merely a score penalty.

- [minor] Rename Step 2: the firewall is already mandatory in Step 1, so Step 2 should describe only soma-root/identity-prior integration.

A self-contained block evaluator is useful for engineering validation, but it cannot replace the requested whole-volume acceptance measurement.

READY: no