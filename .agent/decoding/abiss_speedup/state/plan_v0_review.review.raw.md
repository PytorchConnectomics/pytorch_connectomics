## Summary

The plan has promising leads, but it is not executable or fidelity-safe as written. The verification method does not prove bit-identical segmentation, several opportunities lack payoff estimates, and P0-A’s ranking precedes the measurement needed to justify it.

## Findings

- [major] The fidelity test is insufficient. Equality only after `relabel-sequential` proves partition equivalence, not bit-identical label values. Testing one known chunk also cannot establish whole-volume fidelity for compiler, hash-container, allocator, or parallel-execution changes. Require direct shape/dtype/value equality of the unrelabelled segmentation across the complete affected output, preferably chunkwise hashes plus an exhaustive array comparison. P0-A should additionally be repeated to detect scheduling-dependent nondeterminism.

- [major] P0-A is ranked as the “largest measured payoff” before its load-bearing experiment has occurred. The 64% upper-layer share is not an estimate of parallelizable time: the ME composite invokes seven binaries, while the evidence only establishes TBB linkage for `agg` and `meme`. Per-binary timing and CPU-utilization/thread-count measurements must precede the payoff estimate and scheduling recommendation.

- [major] The P0-A resource model is under-specified and partly unsafe. `--cpus-per-task` alone does not demonstrate that TBB observes the allocation; the experiment must verify CPU binding and effective parallel utilization. The plan also conflates CPUs per process with chunk concurrency. Raising both can increase aggregate memory pressure and is not inherently required. Benchmark latency, throughput, CPU efficiency, and RSS over an explicit CPUs-per-chunk × concurrent-chunks matrix.

- [major] The L4 memory premise is not consistent with the supplied table. `me_L4` is approximately 39 GB per concurrent chunk (`78.2/2`), but `ws_L4` is approximately 80 GB (`160/2`). Using a single 39 GB estimate for L4 could produce unsafe scheduling recommendations; WS and ME must be modeled separately.

- [major] The deliverable requires an estimated payoff for every opportunity, but P0-B, P1-A, and P1-B provide no numerical time or memory range tied to the measurements. P0-A supplies only an unmeasured upside premise. Add conservative/best-case estimates, state which measured stage each applies to, and rerank after the profiling gates.

- [major] P0-B is mis-sequenced for cluster safety. The plan proposes committing `-march=native` before resolving whether build and compute nodes are homogeneous. Node compatibility must be a prerequisite, and the architecture should be configurable rather than globally hard-coded. Verification must establish the effective flags for every affected target, including exactly one intended architecture choice, `-ffp-contract=off`, and absence of unsafe floating-point flags.

- [major] P1-A is not yet executable. Absence of `USE_ABSL_HASHMAP` does not by itself prove that Abseil is uninstalled rather than disabled or incompletely detected. The plan must identify the exact configuration switch/package target, audit every affected `MapContainer` use rather than only `mean_aggl.cpp`, define the pass/fail criterion for ordering and ties, and reject the change if determinism cannot be established.

- [major] P1-B combines mimalloc and jemalloc without specifying which integration path, how each is enabled, or how their effects will be isolated. “Allocator choice cannot alter results” is too absolute: changed layout can expose address-dependent behavior or latent undefined behavior. Benchmark each allocator independently against the same baseline and retain the full fidelity gate.

- [major] The fp16 memory rationale contains a factual error: touched file-backed mmap pages do contribute to process RSS, although they are not heap allocations. The 805 MB mapping is about 10% of an 8 GB chunk footprint; converting storage to fp16 would save roughly 402 MB, or about 5%. The rejection may remain defensible, but it must use the corrected accounting and mention possible I/O/page-cache savings.

- [minor] The summary cites `lesson_efficiency.md` L117, while the supplied context assigns L117 to `lesson_abiss.md`. Also, `grep ... | wc -l` establishes 58 matching lines, not necessarily 58 distinct call sites.

READY: no