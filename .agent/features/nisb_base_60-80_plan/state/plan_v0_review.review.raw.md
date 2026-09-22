# Plan v0 Review

## Findings

- [major] **Step 0 / Verification 1:** The required scorer calibration `cc3d@0.75 → 0.545` is missing. Add it alongside `0.66 → 0.601431`, with canonical cache paths/commands and numeric tolerances. Treat `0.772069` as supplemental unless its extra precision is grounded in a named artifact. Define a tolerance for the blocking center-chunk `≈0.833` check.

- [major] **G0 steps 1–4:** Material implementation choices remain unresolved: erlgraph versus kimimaro skeletons, watershed versus geodesic growth, medial-band construction, edge ignore/unknown states, corridor rasterization, coordinate order/anisotropy, and cumulative offset definitions. Select one canonical pipeline and enumerate every offset bank, including half-space rules and deduplication.

- [major] **G0 step 3:** “Rasterize skip paths, CC → seed IDs” can accidentally unite different graph components when paths cross. Union-find graph components must remain authoritative; rasterized voxels should inherit component IDs, with collisions rejected or left unknown. Specify MUTEX checks during unions and prohibit spatial CC from inventing identity unions.

- [major] **G0 perturbation protocol:** The 95% and 90% closure gates are not reproducible. Define eligible events, trial counts, RNG seed, boundary exclusions, whether section deletion is per object or global, the crumb-distribution source, when labels are generated, the closure criterion, denominators, and whether every condition must pass independently.

- [major] **G0 / Verification 2:** Predicted-SDT growth is never run, but the unexplained `0.985` kill threshold remains. Since `task.md` narrows executable G0 to GT-SDT, explicitly defer predicted-SDT and remove that kill criterion from this gate; otherwise add and report the predicted-SDT experiment.

- [major] **G1 steps 2–5:** Bind one exact starting label map and LUT throughout. The plan does not resolve whether proposals operate on cc3d@0.66 fragments or the 0.627 intersection-cut firewall. Define that relationship before constructing bridge events, extracting endpoints, scoring gains, or realizing bodies.

- [major] **G1 bridge recall:** The task identifies `ec_endpoint_bridge.py` as the bridge-event source, while the plan derives the set from `banis+_oracle_merge.py`. Use the former for canonical local endpoint/bridge events and the latter only where appropriate for identities and exact node-LUT gains. Specify endpoint neighborhoods, crop-boundary exclusions, pair/hyperedge handling, duplicate matching, and how each marginal gain is computed.

- [major] **G1 proposal caps:** Hard-capping at eight candidates makes the success count automatic and the `>12` kill criterion impossible. Report post-geometric-pruning counts before capping, then separately measure recall after a deterministic ranked top-eight cap, including ranking and tie-breaking.

- [major] **G1 realizer:** “aff_r1 threshold” and “reproduces base local NERL within the harness” are not executable gates and can pass trivially with singleton supervoxels or unchanged labels. Specify the affinity channels, threshold sweep/freeze rule, supervoxel algorithm, geodesic cost, seeds, competition/abstention policy, and minimum coverage. Run both a base-preservation control and candidate-oracle-repaired seeds; require grown NERL within 0.005 of seed NERL, zero seed-ID unions, and zero multi-ID assignments.

- [major] **Risks / missing affinity:** GT-derived fragments are not a valid substitute for the required banis+ feasibility experiment. Resolve an exact cached affinity path or an inference configuration/checkpoint, including crop axes, halo, key, and channel order. If neither exists, G1 is blocked/no-go—not a surrogate pass.

- [major] **Runtime guard / verdict:** Smoke results cannot replace completed center-chunk gates. A background run must be monitored through successful exit and validated output before final results are written. Also define intermediate outcomes between success and kill thresholds: training proceeds only when every success criterion passes; gray-zone results remain no-go for training and trigger cheap refinement. Provide an overall result location even when G0 failure prevents `results_g1.md`.

- [major] **DESIGN.md:** The asserted “43 dense” channels cannot be verified from the plan. Add an indexed channel accounting that sums to 43 and defines every target, edge class/ignore mask, veto target, and sparse-skip representation.

- [minor] **Proposed Changes:** “Extend” existing scripts conflicts with their read-only status. Say explicitly that new `mesa/` scripts call, import, or adapt their logic without modifying tracked files, and specify how the `banis+` filename is loaded or invoked.

- [minor] **Second-chunk question:** Do not add a harder chunk in this run; the task explicitly scopes execution to the standard center chunk. It can be recommended as a later validation gate.

READY: no