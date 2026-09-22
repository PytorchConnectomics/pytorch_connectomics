## Summary

Plan v2 is not yet executable as written. It genuinely closes four of the ten open items (#1, #10, #13, #14), partially closes four (#6, #7, #8, #9), and leaves #4 and #11 materially open.

Selecting the flood array is defensible even at ~1.1× if it truly provides resumable per-unit execution. The current plan preserves per-unit artifacts but does not define or test the retry workflow, so its robustness justification is not yet established. The explicit overlay mapping and voxel-level equality gate do close the substance of finding #1.

## Findings

1. [minor] Item #1 — CLOSED. `nucleus_overlay.py` is now in scope, one manifest mapping is authoritative, and the voxel-level sharded-versus-serial gate would catch graph/volume disagreement. Validation should require the mapping domain to equal the exact territory-ID set, exactly one emitted parent ID per unit, and unique remaining emitted IDs—not merely say “non-bijective.”

2. [major] Item #4 — NOT CLOSED. The fingerprint still omits the affinity/cost input explicitly required by the prior review, even though flooding consumes it. `WS_PATH` is only a location, and “watershed manifest identity” is not defined as a digest or immutable artifact ID. The nucleus input may also be a store rather than one hashable file. The plan digest must cover authoritative identities for watershed, nucleus, affinity/cost, parameters, and all relevant Python/native code.

3. [major] Item #6 — PARTIALLY CLOSED. The selection rule is mutually exclusive, but B.5 only shards nucleus-to-watershed mapping. Instrumentation distinguishes “scan instance geometry” from “map nuclei→watershed ids”; if geometry scanning dominates, combining both into `s` can select an optimization that does not address the dominant work. The rule must use separate geometry-scan and mapping fractions, with a concrete design for whichever dominates.

4. [major] Item #7 — PARTIALLY CLOSED. The `CHUNKMAP_INPUT` precedence repair is concrete and implementable. The ordered efficiency plan is still incomplete: “bounded by shard count” is a ceiling, not an expected scan-sharding speedup, and the stated flood prediction is numerically wrong. For `f=0.7`, `1/(1-f+f/9)` is about 2.65×, not 3.9×.

5. [major] Item #8 — PARTIALLY CLOSED. The outside-box gate and method for establishing rejected-edge references are now specified adequately. Comparing `nucleus_shell_contamination.py` reports for nuclei absent from `units.json` does not by itself prove already-clean nuclei are unchanged voxel-for-voxel. The plan must name the exact report fields or voxel regions compared and require exact equality or a defined label-equivalence relation.

6. [major] Item #9 — PARTIALLY CLOSED. R3 is conservatively handled by defaulting to `local_only`, and R4 qualification is explicit. R5 names useful invariance dimensions but gives no owning code/test target, fixture, or ownership oracle. R6 merely says to test whether local scoring sees cross-chunk merges; it provides no method, input, expected result, or verification step. Those two remain promises rather than executable planned work.

7. [minor] Item #10 — CLOSED. “Audit, add missing production validation, then test” repairs the tests-without-checks problem, and incompatible-manifest reachability is now stated. Each resulting validation must satisfy verification 5: abort and publish no canonical manifest.

8. [major] Item #11 — NOT CLOSED. Verification 6 has neither a correct prediction nor a pass/fail threshold. Array critical time is approximately `nonflood time + max(per-unit flood time) + scheduler/merge overhead`, not `nonflood time + total flood time/9`; units need not be balanced, and native96 has eight rather than nine. The plan must define what result counts as acceptable, including what happens if the array is slower.

9. [minor] Item #13 — CLOSED. The durable result is emitted by the stage and judged after completion by the acceptance/reporting step, rather than incorrectly at submission time.

10. [minor] Item #14 — CLOSED. “Produced” no longer contradicts the no-commit rule.

11. [major] The retryability justification is sound in principle but incomplete in this plan. Nothing specifies how a failed array element is requeued or resubmitted, how the merge dependency is repaired after an `afterok` failure, or how completed unit artifacts are reused without rerunning scan. Add a failure-injection gate: fail unit 8, rerun only that unit, prove prior unit artifacts are unchanged, then merge successfully. Without this, the flood array may deliver only ~1.1× and no demonstrated robustness benefit.

12. [major] The artifact gates conflict with the new ID representation. B.1 changes the winning territory inside `terr_*.npz` from the parent ID to a preallocated high ID, while verification 3 expects the old manifest to differ only by additive keys. If existing territory records contain that ID, their values necessarily change. The plan must define whether existing manifest fields store internal territory IDs or emitted IDs and replace the additive-only comparison with an exact semantic comparison after applying the translation.

## Questions

Q-1: Yes. The coordinator or operator can execute the 7–11-hour cluster baselines after code_v0 supplies exact commands, frozen fingerprints, fresh output namespaces, and expected artifacts. Code_v0 may complete implementation and focused tests, but full efficiency verification must remain pending until both parameter-specific baselines and array runs are recorded; it cannot claim that gate passed beforehand.

Q-2: Use a fully separate run-scoped staging directory and publish only after validation. Keep the old canonical manifest as the last successful publication, require consumers to validate the expected plan digest/completion marker, and atomically replace the canonical manifest at the end. A staged manifest and a canonical manifest are not ambiguous when only the canonical path is published. Quarantining the old manifest before execution creates an unnecessary period with no valid canonical result and can disrupt concurrent readers.

READY: no