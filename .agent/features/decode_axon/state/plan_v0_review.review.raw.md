The plan is not executable as written. The main blockers are incomplete call contracts and under-specified safety logic.

- [major] Tier 1 lacks the data flow required by the supplied APIs. `_raw_iou_table` needs `z_of`; `detect_fusions` needs `area`, `fuse_iou`, and `area_tol`; `eligible_fusions` needs `min_len`; and `constrained_relabel` needs `runs` and `candidate_links`. The plan omits `_candidate_links(...)` entirely and does not explain how force-split produces the required `runs`. Either extract one canonical pure pipeline function from `decode_v2._run_pipeline`, or specify every call, return value, and preserved default.

- [major] The proposed `force_split` and `section_index` reuse cannot be verified from the supplied contracts: neither definition/signature is shown. The plan must identify their actual import sources and signatures, especially the force-split return structure consumed by `constrained_relabel`.

- [major] `build_strong_sections(z0, z1, thr)` cannot directly reuse `_load_inputs(args)`, which accepts an argparse-like object. If calling `waterz_2d_spacefill` directly, the plan must map its required `aff_chunk`, `mask_zyx`, `small_size`, `aff_low`, and remaining parameters. For nonzero `--zslice`, it must read `[z0:z1]` from the full `z0-800` cache or use a cache named by both global bounds; `z0-{depth}` is incorrect.

- [major] The shown `decode_3d` composition is not callable: `prob1_carve` omits `affxy` and all required keyword-only values, `prob1_extend` omits `affxy`, and `prob2_merge` omits `aff_raw`. The plan promises “measured defaults” but does not state them.

- [major] Prob-1 carve does not enforce the mandatory bump-safe proposal/commit rule and does not explicitly restrict processing to incomplete tubes. Only extension is described as gated. Unsafe carves could therefore be committed or valid tubes modified.

- [major] `prob1_extend` does not define how it supplies `extend_end` parameters (`area_tol`, `host_margin`, `mem_thr`, `max_steps`) or constructs `zr`, `area`, `carved`, and `bp` for `open_ends` and `gate_bumpsafe`. Bidirectional ordering and recomputation after each accepted mutation are also unspecified. The optional choice between copying the loop and extracting it from `tube_extend.main` must be resolved.

- [major] Prob-2 omits the exact area-match statistic and threshold. It also cites `compute_bbox_all_3d`, which is not among the supplied APIs; the available helpers are `z_ranges` and `z_overlap_frac`. Since `region_graph(seg)` accepts no affinity, the extraction must explicitly preserve how channel-correct raw boundary affinities are aggregated and whether incompleteness/z-range checks apply to original labels or evolving union-find components.

- [major] Weak sectioning leaves a topology-defining choice unresolved: binary versus multilabel opening, exact fastmorph operation, structuring element/radius, border behavior, and `min_cc` default. These must be fixed for the algorithm and self-test to be reproducible.

- [major] `weak_bridge` is under-specified. It lacks numeric limits for z-gap, centroid distance, area match, strong-to-weak and weak-to-weak overlap/contact, and skipped slices. It also does not state how existing IoU/link helpers are reused. The phrase “never weak-weak” conflicts with the required multi-slice weak chain; the correct contract is that weak-to-weak paths may exist only when anchored by exactly two accepted strong endpoints.

- [major] Weak bridging lacks global exclusivity. A branched weak component could connect three or more strong endpoints and cause multiple pairwise merges. Require two-anchor-only paths plus weak-node and strong-endpoint mutexes, deterministic conflict resolution, and rejection of ambiguous chains.

- [major] `gate_bumpsafe(area, oid, carved, bp)` is not a generic merge checker. The plan does not explain how a strong–weak–strong merge or crumb absorption maps to that signature, which object is `oid`, or whether both original strong objects are checked. A concrete adapter or separately defined bridge/absorption safety test is required.

- [major] Crumb absorption is not deterministic: “adjacent,” “larger,” and “unambiguous” lack connectivity/contact definitions, and bump safety alone does not prevent persistent parallel merges. Define drop-only as the safe default, or give absorption its own flag and explicit affinity, z-overlap, contact, area, and exclusivity rules. Also state whether an unsafe/unmatched orphan is dropped or retained.

- [major] Metric calls are invalid. `valid_tube_metric.score` accepts a path plus nine required metric arguments, but the plan calls `score(out)` with an ndarray. Fix the corrected-baseline parameters and either save every ablation tier in the expected HDF5 layout before scoring or explicitly add an array-level metric API to scope.

- [major] Full-run NERL remains unspecified. The plan must identify the `liconn_nerl` callable or valid CLI, prediction/GT arguments, dataset keys, parameters, and execution point after saving.

- [major] Default-output validation is incomplete. Crumb cleanup is default-on but the gate checks only strong+3D and optional weak results; the final saved artifact must itself satisfy all three metric limits. If weak succeeds, the plan should state that it becomes default for `decode_axon.h5`; if it fails, it remains disabled as required.

- [major] CLI/output semantics need completion: require one execution mode, validate slice bounds, ensure `--self-test` exits before data I/O, prevent an untagged z-slice run from overwriting `decode_axon.h5`, and define whether artificial slab boundaries count as faces. Z-slice metrics should be labeled smoke-only because they are not comparable with full-volume face counts.

- [major] Self-tests omit load-bearing contracts: Prob-1 carve rejection, Prob-2 sequential acceptance/concurrent rejection, weak-only-chain removal, ambiguous-chain rejection, accepted-chain exclusivity, and safe crumb fallback. The referenced bump-safe fixture is not part of the supplied API and must be constructed explicitly.

- [major] Behavior-preserving extraction should compare the saved `main` arrays, or normalized partitions if relabeling is intentional. HDF5 byte comparison is metadata-sensitive, while aggregate metric equality is too weak. Because `decode_v3_merge.py` and `decode_v4_split.py` are also changed and gitignored, their complete contents/diffs must accompany the later code review—not only `decode_axon.py`.

- [minor] “Strong-only reproduces `decode_p1p2`” is incorrect terminology. Tier 1 corresponds to `decode_v2`; strong+3D corresponds to `decode_p1p2`.

- [minor] Verification commands should consistently use `conda run -n pytc ...` as required by the repository instructions.

READY: no