## Summary

Instrumentation-first is the correct order and does not waste a round, provided it is a real decision gate. The plan should instrument the existing serial path, run it, then revise the optimization based on measured phase costs. It should not commit to the array architecture before that evidence exists.

The two reference runs’ different `ABISS_NUC_MIN_TAGGED` values do not prevent regression testing: each run can be compared with its own frozen parameters and inputs. They do prevent treating the runs as equivalent configurations.

Plan v0 is not executable as written. It leaves required deliverables unresolved, has an unsolved dynamic Slurm-array dependency, and does not sufficiently preserve or verify territory-level determinism.

## Findings

- [major] The scope does not satisfy `task.md`. It omits the required watershed-reuse recommendation, explicit R1–R7 answers, and ship-readiness verdict. It also silently turns a stated review/design task into implementation. If the CCC task string authorizes implementation despite that, the plan must explicitly reconcile the conflict.

- [major] Section B is conditional on instrumentation results, but no decision threshold or revision checkpoint exists. If scanning dominates, the proposed flood array may deliver negligible benefit and the alternative “shard by nucleus range” is not designed concretely enough to implement. Instrument the unchanged serial path first, measure it, and then approve a concrete optimization plan.

- [major] The proposed static three-job Slurm chain is not executable as described. `submit_wholevol_sharded.sh` cannot know the array bounds until `nuccomp_scan` has produced `units.json`. The plan needs a concrete dynamic-submission mechanism or fixed task scheme, including how the merge obtains the resulting array job ID. It also does not handle zero units, for which no ordinary 8–9-element array can be submitted.

- [major] The territory-ID mitigation is incomplete. The specification says the largest flooded territory retains the parent ID, but scan cannot know which territory is largest before flooding. The plan does not define how IDs are preallocated while preserving that rule, existing tie-breaking, and existing global allocation order. Sorting the final manifest does not prove deterministic flood results.

- [major] “Bit-identical manifest” is internally inconsistent with “ignore new keys,” possible addition of `zero_repairs`, and imposing newly sorted serialization. More importantly, an identical manifest does not prove that `terr_<unit>.npz` contains identical voxel labels. The gate must cover territory-array contents and preferably the complete materialized output, not only manifest metadata.

- [major] Split execution introduces stale-artifact hazards that are not addressed. A record or territory file from an earlier run could satisfy fan-in after a retry, and an old completed manifest could remain visible when a later merge fails. The plan needs run/config/input identity, per-record plan-digest validation, atomic writes, an exact expected artifact set, and isolation or invalidation of prior outputs.

- [major] The R7 accounting is wrong. The claimed abort conditions include invalid nucleus coordinates, but the proposed tests omit it. The existing missing-manifest test is not a substitute for one of those five conditions. The plan also does not answer which conditions are practically reachable, as requested.

- [major] The proposed B2 test lacks a reliable oracle. Merely asserting that the aggregation and remap paths produce the same labels could pass if both paths return the same unsplit result. The test must assert the expected competitive labels and that distinct owners remain distinct after the real final-remap path.

- [major] R3, R5, and R6 are unaddressed. The plan neither establishes global containment nor records `separation_claim = local_only`; it does not verify cannot-link ownership determinism across shard counts; and it does not investigate the local bit-identical scoring null. R4’s sweep can reasonably remain experimental, but the parameter non-equivalence must constrain the ship verdict.

- [major] Deferring all remaining acceptance work is too broad. At minimum, unchanged material outside repair boxes and nucleus-rejected edge counts are relevant regression gates for a behavior-preserving refactor. Checking only the 611/651 pair cannot establish “no downstream drift.”

- [major] The efficiency acceptance criterion is incomplete. There is no expected speedup per proposed change, and an old uninstrumented run cannot supply the promised before/after phase table. The plan needs an instrumented serial baseline on identical inputs and must compare end-to-end critical-path elapsed time, not summed array-task time.

- [minor] RSS reporting needs defined semantics. `ru_maxrss` has platform-dependent units and measures one process; after sharding, the report should state units and distinguish scan, individual flood-task, merge, maximum-task, and concurrent aggregate resource requirements.

## Questions

- **Q-1:** Keep scan single for the instrumentation milestone. Do not implement either scan sharding or the flood array until the phase table determines which is justified. This open branch currently prevents the full plan from being executable.

- **Q-2:** Use a separate immutable `units.json`, not a `planned` state inside the final manifest. It should carry a schema version, input/config fingerprints, plan digest, canonical unit ordering, and exact expected outputs. Publish the existing manifest only after validated completion.

- **Q-3:** Yes, bit-identity can be checked per run using each run’s complete original materialized parameters, including `ABISS_NUC_MIN_TAGGED=50` or `1024`, and its frozen inputs. The two outputs must not be compared to each other or treated as equivalent experimental replicates. Exact byte identity additionally requires preserving the existing serializer; otherwise the gate must honestly be called normalized semantic equality.

READY: no