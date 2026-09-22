# Plan v3 Review

## Summary

The ratified 1-voxel G0 oracle is accepted. Cached-LUT Sanity A, cumulative B3=62, measured `BASE_NERL`, break-cluster events, validated ΔNERL, and C1/C2 denominators are resolved.

Four implementation blockers remain in G1 and `DESIGN.md`.

## Findings

- [major] **G1 §1 — proposal geometry is not reproducible.** “Tangent agreement” lacks an equation; the cubic Hermite curve lacks tangent-derivative magnitudes and a curvature-evaluation rule; and “boundary EDT … ×res” is ambiguous under anisotropic resolution. These choices directly control the gated recall and candidate maximum. Pin physical EDT/KNN units, the tangent score, Hermite handle scale, and curvature computation.

- [major] **G1 §4 — the realizer gate does not test repaired realization.** The only gated run starts with one identity per fragment. Since orphans contain no skeleton nodes, adopting or abstaining on them normally cannot change NERL; an abstain-everywhere implementation can pass because coverage has no threshold. Run the realizer from the capped-oracle repaired seed partition, compare against its pre-grow result, and gate dense-GT adoption precision, nonzero coverage, abstention, and merge safety. Define how label-0 regions enter the fragment graph.

- [major] **G1 §2 — “non-transitive” matching contradicts endpoint degree-1.** A fragment with two tips can accept two links and create a transitive union chain. Require degree ≤1 per `BASE_SEG` fragment—or otherwise define an exact non-cascading transaction rule—and pin the deterministic mutual-best ranking.

- [major] **`DESIGN.md` channel table — several GT targets remain undefined or incorrect.** `sign(v∈G)` is not a ± signed-distance convention; medialness uses an approximate `~0.6·r_i` cutoff; tangent masking and veto neighborhoods are undefined; and the sparse skip head lacks a SAME/MUTEX/DEFER target rule. Pin these definitions so the promised 43-channel and skip-edge targets are deterministically derivable.

## Questions

None. Each blocker has a local specification fix and does not challenge the ratified G0 decision.

## Verdict

VERDICT: NEEDS_CHANGES

READY: no