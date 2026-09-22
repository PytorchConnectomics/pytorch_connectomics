## Summary

Plan v1 materially improves the fidelity gate, separates WS/ME memory correctly, and fixes the fp16 accounting. However, several prior findings remain only partially addressed. The plan is not yet executable as written, principally because profiling cannot support the promised ranking, architecture safety remains unresolved, and dependency experiments are not isolated or fully specified.

## Findings

- [major] The required ranked deliverable is still missing. Phase 0 explicitly produces “a table, not a recommendation,” and the plan defines neither a post-profiling ranking step nor decision thresholds for advancing candidates. The parallel-scheduling opportunity still has no conditional payoff estimate. Add a final synthesis step that computes stage-level and whole-run savings from measured per-binary times, ranks all candidates, and states accept/reject thresholds.

- [major] Phase 0 is not representative or resource-safe enough to support scheduling conclusions. A single L4 chunk cannot characterize a layer known to have non-uniform chunk costs, while concurrent execution of the single L5 chunk is not a production workload. The required `{1,4,16} × {1,2,4}` sweep could demand roughly 392 GB at four concurrent L5 processes, without a node-memory admission check. Use multiple representative L4 chunks, treat L5 as latency-only, isolate duplicate outputs if used experimentally, and omit any matrix point exceeding the node’s CPU or memory capacity.

- [major] The architecture prerequisite remains unresolved. The evidence establishes model heterogeneity between the build node and `c134`, but `sinfo` labels such as `48c`, `64c`, and `96c` do not establish ISA heterogeneity or prove that every eligible node supports x86-64-v3. The sampled Cascade Lake/Ice Lake pair both support v3, so that pair alone does not make `-march=native` unsafe. Inventory the effective ISA of every eligible node class before choosing a default; until then, the default must remain unset or use a demonstrated common baseline.

- [major] The proposed flag mechanism contradicts its verification criterion. Appending the project’s `-march=x86-64-v3` after inherited `-march=nocona` produces two `-march` options, while verification requires exactly one. The plan must specify how inherited architecture flags are removed or how configuration occurs with sanitized `CXXFLAGS`, and must verify actual compile commands for every affected target, not merely assume one glob covers them all.

- [major] The Abseil-absence finding is overstated and the integration remains under-specified. Checking three include prefixes and shared libraries does not exclude nonstandard `CMAKE_PREFIX_PATH` entries, compiler include paths, CMake package registries, or static Abseil libraries. More importantly, the plan still does not identify the exact CMake option, package target, compile definition, and link requirement that activate `MapContainer = absl::flat_hash_map`. Confirm absence through the actual configured toolchain and define the complete enablement path.

- [major] The container and allocator payoff estimates are not soundly tied to the supplied measurements. Applying generic “30–50% map memory” results to 39–98 GB whole-process RSS assumes an unknown fraction of RSS belongs to those maps. The allocator’s 5–25% estimate lacks an allocation profile, an absolute saving against the 143-minute composite total, and any numerical memory range. Phase 0’s timing/RSS measurements alone cannot establish these fractions; add heap/allocation attribution or present explicit conditional formulas instead of unsupported whole-process percentages.

- [major] The allocator experiments are not sufficiently isolated. Because the existing logic is described as `if/elseif`, installing both allocators into one discoverable environment can cause the jemalloc build to select mimalloc. Separate build directories alone do not prevent that. Specify isolated prefixes or explicit mutually exclusive enable/disable controls, then verify at runtime which allocator is active through loaded-library or allocator-specific evidence.

- [major] Installing Abseil, mimalloc, and jemalloc adds dependencies without the explicit approval required by the supplied AGENTS.md. The plan needs an approval checkpoint and should use isolated, reproducible environments so installing one candidate cannot silently alter the baseline or another candidate build.

- [major] `run_start_ref` is not a sufficient performance or fidelity baseline for build-environment experiments because Git does not capture the compiler, inherited flags, CMake cache, dependency resolution, or loaded allocator. Preserve immutable baseline outputs and record the full build/runtime manifest. Otherwise dependency installation or reconfiguration can change multiple variables simultaneously and invalidate both timing attribution and fidelity comparison.

- [minor] Phase 0 calls itself “no code change,” but the file table proposes modifying two scripts and leaves “revert or keep behind a flag” undecided. Choose a non-persistent external wrapper or specify an opt-in profiling flag and its final disposition. Also make the CPU-affinity command concrete: `taskset -pc` requires a PID or an equivalent recorded `Cpus_allowed_list`.

## Fidelity assessment

The strengthened gate does address the prior core finding: identical shape, dtype, and exhaustive unrelabelled element values across the complete segmentation prove exact segmentation-array identity for the compared runs. Chunk hashes provide an additional integrity check, and repeated candidate runs are an appropriate practical nondeterminism test. This proves array-content identity, not byte identity of container metadata or compression, which is sufficient for the stated segmentation constraint.

READY: no