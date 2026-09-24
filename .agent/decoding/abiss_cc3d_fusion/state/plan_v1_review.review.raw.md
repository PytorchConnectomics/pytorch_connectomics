# Plan v1 Review

## v0 finding disposition

1. **Resolved as originally stated:** Stage A no longer reads GT, and fidelity measurements moved to `evaluation_gt/`. A new temporal-freeze problem remains because Stage B precedes assignment freezing.
2. **Partially resolved:** assignment candidates now use all 726 chunks, but the face-join and split candidate families needed by the directional oracles are still undefined.
3. **Resolved:** Stage C no longer reuses evaluator-selected artifacts.
4. **Partially resolved:** the ten-chunk reproduction is now genuine, but the fusion-mask bridge does not validate or replace the tier10-masked dense store used for the actual experiment.
5. **Partially resolved:** many thresholds are numeric, but several classifications, feature definitions, robustness variants, and veto inputs still require implementation-time choices.
6. **Not resolved:** the batch resolver remains internally inconsistent across P2–P6 and does not meaningfully represent the conflict cases its tests claim to cover.
7. **Partially resolved:** manifest coverage is much better, but the histogram-based alignment test still cannot prove spatial correspondence.
8. **Partially resolved:** deliverables are named, but Stage E precedes diagnostics needed for questions 5–6 and the split oracle.
9. **Not resolved:** namespace subset plus baseline-score equality does not prove identical segmentation identity, and composition conflict semantics remain unspecified.
10. **Resolved:** determinism uses canonical arrays/content hashes and atomic publication.

## Findings

1. [major] The reordered stages invalidate the claimed freeze chronology. Stage B reads evaluator GT before Stage D creates the policy and assignment artifacts, yet those artifacts are stamped `frozen_before_evaluation: true`. More importantly, B3 is not a reproduction of a published number: it produces a new human-GT score and candidate-topology deltas, and the risk section allows that result to change the decoding path if the difference is “material.” That term has no threshold. Stage B also precedes the global segment table needed to define those topology deltas. The assignments and implementation hash must be frozen before any new evaluator result, or B3 must be moved after Stage D and forbidden from changing the current run.

2. [major] The primary candidate graph still uses the wrong masked partition. The dense store was built with the tier10 tissue-only recipe, while fusion is supposed to use the ABISS tissue+border keep-mask. A ten-chunk bridge does not correct the 726-chunk store or prove equivalence on every accepted candidate. A P6 `touches_ring` veto also cannot repair upstream component topology, zero-anchor counts, Gate-A oracles, or P2–P5 results: extra kept voxels may join components before the veto, while voxels removed during store construction cannot be recovered. The unsupported assertion that only ring chunks differ must be replaced by a voxelwise mask comparison and deterministic re-decoding of every mismatching fusion chunk under the ABISS mask.

3. [major] Stage C cannot compute all mask-related vetoes from its declared inputs. It reads dense CC3D labels and ABISS crops, but claims to emit tissue-mask-boundary flags. A zero label does not reveal whether foreground was removed by the mask or failed the affinity threshold. Both masks must be read explicitly. The dense store itself also lacks a fidelity gate comparing selected stored partitions against fresh affinity-plus-mask decoding; provenance and file completeness do not establish correct contents or alignment.

4. [major] Gate A is not executable for all three promised directional families. Stage C defines assignment overlaps and retains face slabs, but specifies neither reciprocal face matching nor a best-with-margin threshold. It creates no GT-free suspect detector or cut population for the split oracle. Those are deferred to Stage F, after Stage E supposedly evaluates the three oracles and writes the final answers. “Incompatible centerlines,” glia/lamellar morphology, and the alternative Gate-A criterion of recovering an “important” break have no frozen definitions. GT-free face and split proposals must be generated before evaluator overlay; Stage E cannot score candidate families that do not yet exist.

5. [major] The batch-global resolver does not define consistent behavior for the ablation ladder. Step 2 always requires “the margin rule,” but P2–P4 do not contain a margin gate. Applying 0.20 silently changes those rows; disabling it leaves multiple-host and tie handling unspecified. Further, alternative claims are discarded before graph construction. With disjoint immutable anchor/fragment roles and one surviving anchor per fragment, a multi-anchor connected component, cycle, or transitive anchor bridge is normally impossible, making Step 4 and several permutation tests vacuous on valid extractor output. Inconsistent chunk-local hosts are also winner-selected rather than necessarily abstained. Each policy needs an explicit raw-claim resolution rule, score, tie-breaker, and final remap invariant.

6. [major] Several supposedly predeclared policy definitions still require choices:

   - Inclusive endpoints make a label exactly at an anchor threshold potentially both anchor and mid-fragment.
   - Labels above the mid upper bound but failing the bbox criterion are unclassified.
   - The plan never says whether dust is assignable or diagnostic-only.
   - “Every robustness neighbour” could mean six one-factor variants or a Cartesian product, and its interaction with three bands is unspecified.
   - The affinity second pass does not define whether “surviving” means the union of candidates needed by every robustness variant.
   - Nucleus-mask resolution, coordinate conversion, instance semantics, and interpolation are absent. Moreover, the recorded ABISS nucleus counts do not establish whether a particular CC3D component touches a nucleus.

7. [major] The P4 transition-affinity statistic can silently measure the wrong evidence. Taking `max` over channels at voxels within two voxels of the other label is not an affinity across the fragment–anchor transition: the maximum may point along the fragment or in an unrelated axis, and distance-two voxels may have an intervening label or background. The plan must identify the directed affinity edges and channel orientations that actually cross the interface, and separately define any distance-two path aggregation.

8. [minor] The volume arithmetic supporting the bands is correct: a 150 nm-radius, 10 µm cylinder occupies approximately \(4.36\times10^5\) voxels at 9×9×20 nm, so the quoted 23 µm and 0.23 µm conversions follow. It is not a general physical calibration, however. It assumes a fixed-radius, fully occupied, unbranched cylinder and does not justify the bbox thresholds or tight/loose multipliers for variable-caliber neurites, branches, or somata. Present it as a reference-caliber interpretation, not a physical validation of all three bands.

9. [major] Finding 9 is still only partly restated. A correction map can have labels that are a subset of the current label universe and reproduce the same aggregate NERL while referring to a different segmentation revision. The gate needs a recorded source URI plus segmentation/LUT content identity or an equivalent exact substrate digest. “Define conflict semantics” is also not a definition: the plan must state what each composition order does when both maps claim a label and whether an invariant-violating composition is rejected or reported only as an unsafe diagnostic.

10. [major] B5 does not close the highest-risk coordinate bug. Histograms discard spatial arrangement and are invariant to permutations; a transposed or otherwise misordered crop can have the correct histogram. Requiring every shifted-box histogram to differ is also not guaranteed even around a boundary and can false-fail a correct implementation. Use independently specified global XYZ probes and elementwise comparison of a non-symmetric subvolume, with coordinate conversion code independent of the loader under test.

11. [major] Stage E writes `results.md` before Stage F generates split proposals, face diagnostics, and visual cases, yet the report promises to answer primary questions 5–6 fully. Gate C also needs both the autonomous suspect detector and its evaluator precision, but the plan contains no evaluator pass after Stage F. Split Stage F into GT-free proposal generation before evaluation and evaluator-only diagnostics afterward, then generate the final report last.

12. [minor] The leakage audit is self-contradictory: it scans all `gt_free/*` artifacts for `NERL` references, while Stage A intentionally writes `expected_base_nerl`. The allowed declared constant needs an explicit audit exception; provenance enforcement should not rely only on keyword scanning.

## Questions

No external decision is required; these are plan-definition and sequencing issues.

## Summary

The revision materially improves provenance separation, full-population assignment evidence, historical fidelity reproduction, deliverable routing, and deterministic sharding. It is still not executable without consequential choices. The most serious risks are evaluating before the assignments are actually frozen, building the graph from the tier10-masked CC3D store, an inconsistent resolver across P2–P6, undefined face/split candidate populations, and an affinity statistic that does not necessarily measure the proposed transition. Those issues can silently change candidate topology, accepted assignments, Gate-A headroom, and final NERL.

READY: no