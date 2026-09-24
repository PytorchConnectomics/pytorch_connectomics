# Plan v1 Review

## Summary

Reviewer: codex. Raw transcript: `state/plan_v1_review.review.raw.md`.

The reviewer credits v1 with materially improving provenance separation, full-population
assignment evidence, historical fidelity reproduction, deliverable routing, and deterministic
sharding. It returns NEEDS_CHANGES: the plan is still not executable without consequential
implementation-time choices, and five risks can silently change candidate topology, accepted
assignments, Gate-A headroom, or final NERL — evaluating before assignments are frozen,
building the graph from the tier10-masked CC3D store, a resolver that is inconsistent across
P2–P6, undefined face and split candidate populations, and a P4 affinity statistic that need
not measure the proposed transition.

Disposition of the ten v0 findings, as recorded by the reviewer: **resolved** — 1 (with a new
temporal-freeze problem), 3, 10; **partially resolved** — 2, 4, 5, 7, 8; **not resolved** —
6, 9.

Ten new findings are tagged major and two minor. The coordinator judges all twelve material
and has softened none.

## Findings

1. [major] The reordered stages invalidate the claimed freeze chronology. Stage B reads
   evaluator GT before Stage D creates the policy and assignment artifacts, yet those artifacts
   are stamped `frozen_before_evaluation: true`. B3 is worse than a chronology problem: it is
   not a reproduction of a published number but a *new* human-GT score plus candidate-topology
   deltas, and the risk section lets that result change the decoding path if the difference is
   "material" — a term with no threshold. Stage B also precedes the global segment table needed
   to define those topology deltas. Either freeze the assignments and implementation hash
   before any new evaluator result, or move B3 after Stage D and forbid it from changing the
   current run.

2. [major] The primary candidate graph still uses the wrong masked partition. The dense store
   was built with the tier10 tissue-only recipe while fusion is supposed to use the ABISS
   tissue+border keep-mask. A ten-chunk bridge neither corrects the 726-chunk store nor proves
   equivalence on every accepted candidate, and a P6 `touches_ring` veto cannot repair upstream
   component topology, zero-anchor counts, Gate-A oracles, or the P2–P5 rows: extra kept voxels
   may join components *before* the veto applies, and voxels removed during store construction
   cannot be recovered afterwards. Replace the unsupported "only ring chunks differ" assertion
   with a voxelwise mask comparison and deterministic re-decoding of every mismatching fusion
   chunk under the ABISS mask.

3. [major] Stage C cannot compute all mask-related vetoes from its declared inputs. It reads
   dense CC3D labels and ABISS crops but claims to emit tissue-mask-boundary flags; a zero label
   does not distinguish "removed by the mask" from "failed the affinity threshold". Both masks
   must be read explicitly. The dense store also lacks a fidelity gate comparing selected stored
   partitions against fresh affinity-plus-mask decoding — provenance and file completeness do
   not establish correct contents or alignment.

4. [major] Gate A is not executable for all three promised directional families. Stage C defines
   assignment overlaps and retains face slabs but specifies neither reciprocal face matching nor
   a best-with-margin threshold, and it creates no GT-free suspect detector or cut population for
   the split oracle. Those are deferred to Stage F, *after* Stage E is supposed to score all
   three oracles and write the final answers. "Incompatible centerlines", glia/lamellar
   morphology, and the alternative Gate-A criterion of recovering an "important" break have no
   frozen definitions. GT-free face and split proposals must exist before the evaluator overlay.

5. [major] The batch-global resolver does not define consistent behaviour across the ablation
   ladder. Step 2 always applies "the margin rule", but P2–P4 contain no margin gate: applying
   0.20 silently changes those rows, and disabling it leaves multiple-host and tie handling
   unspecified. Worse, discarding alternative claims *before* graph construction means that,
   with disjoint immutable anchor/fragment roles and one surviving anchor per fragment, a
   multi-anchor component, cycle, or transitive anchor bridge is normally impossible — making
   step 4 and several permutation tests vacuous on valid extractor output. Inconsistent
   chunk-local hosts are winner-selected rather than necessarily abstained. Each policy needs an
   explicit raw-claim resolution rule, score, tie-breaker, and final remap invariant.

6. [major] Several supposedly predeclared definitions still require choices: inclusive endpoints
   let a label at an anchor threshold be both anchor and mid-fragment; labels above the mid upper
   bound that fail the bbox criterion are unclassified; the plan never says whether dust is
   assignable or diagnostic-only; "every robustness neighbour" could mean six one-factor variants
   or a Cartesian product, and its interaction with the three bands is unspecified; the affinity
   second pass does not say whether "surviving" means the union over all robustness variants; and
   nucleus-mask resolution, coordinate conversion, instance semantics, and interpolation are
   absent — recorded ABISS nucleus counts do not establish whether a given CC3D component touches
   a nucleus.

7. [major] The P4 transition-affinity statistic can silently measure the wrong evidence. A `max`
   over channels at voxels within two voxels of the other label is not an affinity across the
   fragment–anchor transition: the maximum may point along the fragment or in an unrelated axis,
   and distance-two voxels may have an intervening label or background. Identify the directed
   affinity edges and channel orientations that actually cross the interface, and define any
   distance-two path aggregation separately.

8. [minor] The band volume arithmetic checks out — a 150 nm-radius, 10 µm cylinder is ≈4.36×10⁵
   voxels at 9×9×20 nm, so the 23 µm and 0.23 µm conversions follow. But it is not a general
   physical calibration: it assumes a fixed-radius, fully occupied, unbranched cylinder and
   justifies neither the bbox thresholds nor the tight/loose multipliers for variable-caliber
   neurites, branches, or somata. Present it as a reference-caliber interpretation.

9. [major] The composition identity gate is still insufficient. A correction map can have labels
   that are a subset of the current universe and reproduce the same aggregate NERL while
   referring to a different segmentation revision. The gate needs a recorded source URI plus
   segmentation/LUT content identity, or an equivalent exact substrate digest. "Define conflict
   semantics" is not itself a definition: state what each composition order does when both maps
   claim a label, and whether an invariant-violating composition is rejected or reported only as
   an unsafe diagnostic.

10. [major] B5 does not close the highest-risk coordinate bug. Histograms discard spatial
    arrangement and are permutation-invariant, so a transposed or otherwise misordered crop can
    produce the correct histogram; and requiring every shifted-box histogram to differ is not
    guaranteed even near a boundary, so it can false-fail a correct implementation. Use
    independently specified global XYZ probes and elementwise comparison of a non-symmetric
    subvolume, with coordinate-conversion code independent of the loader under test.

11. [major] Stage E writes `results.md` before Stage F generates split proposals, face
    diagnostics, and visual cases, yet the report promises to answer primary questions 5–6 fully.
    Gate C additionally needs both the autonomous suspect detector and its evaluator precision,
    and the plan contains no evaluator pass after Stage F. Split Stage F into GT-free proposal
    generation before evaluation and evaluator-only diagnostics afterwards, and generate the
    final report last.

12. [minor] The leakage audit is self-contradictory: it scans all `gt_free/*` artifacts for
    `NERL` references while Stage A intentionally writes `expected_base_nerl`. The declared
    constant needs an explicit audit exception, and provenance enforcement should not rely on
    keyword scanning alone.

## Questions

The reviewer states no external decision is required; all twelve are plan-definition and
sequencing issues the planner can resolve.

## Verdict

VERDICT: NEEDS_CHANGES
