# Plan v1 Review

## Summary

Plan v1 resolves prior findings 2, 3’s location/laziness concern, 5 for v1–v4, 7, 8, and 14. It remains non-executable due to unresolved implementation choices and incomplete verification contracts.

## Findings

1. [major] Prior findings 1, 12, and 13 remain unresolved. The plan still leaves `axon_v0` versus `axon_tracklet_base` as an open decision, provides no destination file for the v0 port, and keeps all superseded v1–v3 registrations intact. That also preserves the dev-path dependency the task requires replacing or retiring.

2. [major] Prior finding 4 is only partially addressed. The stages are described as segmentation-in/segmentation-out, yet stats invalidation depends on an undefined `n > 0` result. Return identity, dtype preservation, adapter handling of change counts, and tests for both `inplace` modes remain unspecified. `max(zr) <= seg.max()` cannot establish freshness and may compare bounding-box data with label IDs.

3. [major] Prior finding 6 remains deferred. The actual `merge_threshold` schema path is still “to be confirmed,” while the fallback promises—but does not specify—the exact commands, configuration, artifact paths, and base/oracle-merge invocation. The requested runnable comparison therefore is not pinned.

4. [major] Prior findings 9 and 11 are incomplete. Neither v4 nor completeness receives parity verification. The v4 node also uses `enabled: false`, although no graph, adapter, or core-function contract defines `enabled`; selecting `output: v4` is therefore not executable as written.

5. [major] Prior finding 10 is not fully resolved. `np.shares_memory` proves output aliasing, not absence of a transient full-volume allocation. The alternative benchmark records time and memory without an acceptance threshold, so it cannot establish preservation of the measured performance requirement.

6. [major] Registration still conflicts with the explicit task contract. The plan correctly needs native `register_graph_op` adapters for multi-input nodes, but it simultaneously directs the coder not to use the required `register_decoder(...)` calls. The plan must explicitly reconcile those two requirements rather than forcing the implementation to violate one.

READY: no