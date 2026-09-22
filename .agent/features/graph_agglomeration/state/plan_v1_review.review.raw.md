Several prior findings are genuinely resolved: substrate reuse, whole-volume acceptance/crop, endpoint-oriented candidate generation, transitive marker-set logic, and streamed evaluation direction. Step 1 is also a reasonable bounded vertical increment. However, key execution contracts remain incomplete.

- [major] The global-ID namespace remains underspecified. The linker sees face-crossing labels, while evaluation encounters every foreground label. The plan must define either a total mapping or a collision-free fallback for untouched `(chunk_key, local_label)` pairs, including ID allocation and dtype. It must also prove the remap is applied exactly once before both linked base NERL and `branch_merge`. Otherwise repeated local labels can silently merge across chunks and produce misleading gains.

- [major] Add a whole-volume identity/no-edge remap control. Running the new remap path without committed unions must reproduce the existing no-remap base/oracle NERL, sampled/missing counts, and GT length within numerical roundoff. The proposed 2–3 chunk substrate check does not validate global namespacing or the modified scorer.

- [major] The marker contract is not executable. The plan calls `yl_cb_80nm.h5` fixed, then reopens whether `yl_cb_80nm_neuron.h5` is correct. “8× scale plus confirmed axis order/origin” is not an exact mapping: axis permutation, origin, coarse-voxel block/center convention, boundary handling, and marker-to-segment assignment remain unspecified. It must also establish that values are distinct marker IDs rather than binary seeds. Finally, ownership is inconsistent: extraction supposedly writes membership, but only the later linker receives `--markers`. Specify one concrete sparse marker-sampling stage.

- [major] Quarantine contradicts the proposed assertion. A pre-existing segment containing multiple markers remains a multi-marker global component even when quarantined, so “no global_id holds ≥2 markers” cannot pass. The invariant should instead assert that quarantined roots remain unchanged, accept no edges, and no accepted union creates or absorbs a multi-marker component.

- [major] The Step-1 operating point is not reproducible. Weights, feature normalization, ambiguity, `tau_score`, and the validation/selection protocol are undefined, while tuning is assigned to Step 5. Code_v0 would have to invent parameters or tune opportunistically. Predeclare formulas and constants or define a bounded calibration protocol. `tau_iou_min` should be exactly zero or removed; “approximately zero” can reinstate an overlap gate.

- [major] The oracle gate remains internally inconsistent. `|Δ| ≤ 0.002` permits a negative change that “any drop ⇒ fail” simultaneously rejects, and 0.002 can conceal a real false merge. Use one justified numerical-roundoff tolerance: substantive negative change fails, while a positive change indicates namespace/scoring error and also fails pending investigation.

- [minor] Oracle-flat detects only bad links represented by distinct sampled test-50 owners. Markerless or off-GT false merges remain invisible. That limitation is compatible with the requested sparse-GT metric, but the plan must not describe it as global merge safety.

- [minor] Validate all 600 symlink targets and regenerated crossing schemas, and record current root marker sets—not merely original endpoint sets—in certificates so transitive firewall decisions remain auditable.

The polarity-aware tangent rule and dynamic per-root marker-set union rule themselves are sound, including rejection of `A → unmarked → B`.

READY: no