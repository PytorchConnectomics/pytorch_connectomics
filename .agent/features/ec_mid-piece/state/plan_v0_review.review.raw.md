- [major] The GT-informed scope selector directly violates the firewall and the explicit requirement to build candidates over the full segment population. Passing `evaluation_gt/scope_segment_ids.txt`, derived from the LUT, into candidate generation uses GT to choose candidates. Calling it a compute-only restriction does not make the resulting experiment GT-free.

- [major] The proposed scope-invariance assertion proves only that assignments to fragments without sampled nodes do not directly change the current LUT. It does not prove proposal or policy invariance. Omitted segments can be anchors, competitors, contacts, relays, or contributors to RAG degree, quarantine, winner margins, and component conflicts. Consequently, both the one-hop ranking and the promised complete-graph multi-hop resolution can change under this scope.

- [major] Several essential policy contracts remain undefined: the numerical fragment/anchor bands, anchor eligibility, “extreme” quarantine thresholds, morphology score/fusion formula, reciprocal-best definition, residual-reason precedence, and naming/freezing of individual grid-point assignments. Implementing these would require inventing scientifically consequential rules.

- [major] The affinity verification convention is internally unclear. The plan declares channel `c` to represent array axis `c` at `v→v+1`, but then says the ABISS cross-check should return channel `2-c` at an unspecified shifted position. Without an exact source-indexed coordinate/channel truth table, the coder must infer the intended conversion. The real-crop boundary statistic would not reliably detect an axis swap, sign flip, or one-voxel anchor error.

- [major] Phase 3’s gate is insufficient for the named `r10_minpool` artifact. The task forbids reusing historically buggy r10 TTA/min output unless it passes the referenced affinity-TTA contract; the plan substitutes a newly implemented, incompletely specified geometry validation. Running Phase 3 on that basis would be out of scope or violate the artifact restriction.

- [minor] The report promises precision and sample counts but does not specify the required uncertainty calculation. The “material share” escalation threshold and quantitative corridor-evidence gate are also undefined, making the final recommendation non-reproducible.

- [minor] Calling Stage 2 a “pure LUT-level rescore” conflicts with the firewall unless this means candidate-table-only scoring. Any actual LUT access before freezing would violate the required stage separation.

The plan is not executable as written because its central compute compromise breaks the task’s GT firewall, and multiple load-bearing selection and affinity contracts would need to be invented.

READY: no