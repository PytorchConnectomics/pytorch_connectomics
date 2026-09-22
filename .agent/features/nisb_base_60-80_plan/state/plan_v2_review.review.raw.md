# Plan v2 Review

## Summary

Several revisions are genuinely resolved: the affinity/scorer binding (#10), coordinate API, deterministic top-8 cap (#8), and crop-scale surrogate framing. However, core G0/G1 choices still alter the measured verdict, so the plan is not implementation-ready.

Artifact 2 is empty, so findings #1/#5/#6/#11/#13/#14 cannot be independently verified from the supplied material.

## Findings

- [major] **G0 does not implement the required Phase-0 representation.** `G0 — Phase-0 oracle plumbing` explicitly replaces medial-band nodes with one-voxel skeleton nodes, omits explicit SAME/MUTEX/DEFER inputs, and never rasterizes accepted skip paths into markers. This contradicts `task.md` and supplied §12. Bind the GT medial-band construction, signed edge encoding, exact 3-D corridor rasterizer, and post-UF path painting.

- [major] **Blocking Sanity A remains unbound.** Step 0 names a “whole-vol node-LUT harness” but gives no exact command/API or cached LUT/segmentation inputs. Decoding the bound 12.15-billion-voxel affinity volume would violate the cheap/full-volume scope. Bind the exact cheap invocation and assets or make this a nonblocking provenance check.

- [major] **The final G0 bank is incorrect for the cumulative sweep.** The 49-offset `|o|∞=2` shell excludes B0’s 13 nearest-neighbor offsets. Because §12 says to add the shell, B3 must contain 62 offsets, not replace earlier offsets with 49.

- [major] **G0 perturbation and metric definitions remain verdict-changing.** C1 must restrict deletion to degree-2 nodes; C2 must define an occupied section and all anchor pairs across every cut strand. `fragments/GT` must specify whether its denominator is the 626 scored skeletons or 783 segmentation IDs, and retained-length needs an exact numerator formula.

- [major] **The G1 oracle event remains ambiguous.** One fragment pair can have multiple crossing skeleton edges and therefore multiple break locations. Define an event per break cluster or retain a set of break locations per pair, with exact boundary-exclusion/hit aggregation. Bind the edge-length allocation and exact `ΔNERL = 2L_aL_b/D` normalization, then validate it against the canonical scorer on a deterministic sample.

- [major] **The geometric proposal set is not reproducible.** “Radius-aware limit” has no equation/constants or radius estimator; Hermite construction and degeneracy handling are also unspecified. These choices directly control every recall and candidate-count gate.

- [major] **The realizer remains underbound and lacks a credible full-chunk execution path.** It leaves the foreground as “GT-SDT proxy or aff-fg,” does not define reverse-step affinity indexing, distinct-identity runner-up margin, abstention label, or atomic supervoxel policy. A two-label voxelwise geodesic search over 450 million voxels also needs a bound implementation and peak-resource estimate. Prefer an explicitly defined atomic `BASE_SEG` adjacency-graph realizer if that is the intended scalable implementation.

- [major] **The matching and realizer gates are not trustworthy as written.** `capped-candidate-recovered` merges only GT-confirmed proposals, making zero false merges true by construction without a matching algorithm. Furthermore, every scored skeleton voxel is already a fixed seed, so realized NERL measures candidate identity merges, not geodesic realization. Separate candidate-oracle recovered gain from realizer checks; audit coverage, cross-GT labels, and multi-ID supervoxels for all three realizations. Use the measured Sanity-B baseline rather than rounded `0.836`, and define failure when full-oracle gain is nonpositive.

- [major] **The claimed #12 channel binding remains a promise.** The nine affinity offsets/targets, medialness parameters, tangent estimator, DEFER rules, and veto formulas are not specified. “DESIGN.md expands” leaves substantive model design to the coder.

## Questions

None; the required fixes are specification decisions, not implementation details.

## Verdict

VERDICT: NEEDS_CHANGES

READY: no