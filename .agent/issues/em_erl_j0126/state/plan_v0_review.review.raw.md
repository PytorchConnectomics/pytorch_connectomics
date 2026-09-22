Summary: the plan is close, but not ready. It has one material correctness gap around node-coordinate recovery and one weak verification criterion that could let a silent mis-registration pass.

Findings:

- [major] `graph.node_coords_zyx.round().astype(int64)` is not a safe canonical sampling source. The API excerpt says `ERLGraph.get_nodes_position(None)` returns `node_coords_zyx.astype(int)`, while the plan uses `round()`. If any graph construction path stores float32 coordinates that are near half/inexact values, rounding can sample different voxels than the current graph/node convention. For scientific registration, the plan should use the original integer vertices in graph order or exactly `graph.get_nodes_position(None)`, not a new rounding rule.

- [major] The live verification relies on “assignment-zero ratio is small” as the key registration proof. That can miss silent coordinate swaps, offsets, or scale mistakes if the wrong locations still hit nonzero FFN segments. The task includes a stronger verified anchor: sampled points from skeleton `"0"` all map to segment `1465128`. The plan should include an explicit end-to-end registration assertion/check for known sentinel points or skeleton `"0"` segment consistency, plus expected OOB count near `37`, not just a low zero ratio.

- [minor] The optional `-r/--resolution` flag is out of scope and increases coordinate-risk surface. The task explicitly asks to preserve voxel-unit ERL. Keeping a physical-units option in this script could lead users to produce incomparable ERL results unless very clearly guarded or removed.

- [minor] The CloudVolume opener details are plausible but underspecified for executability. `use_https=True` with a `gs://` source may or may not be the right CloudVolume access mode for this public bucket. The plan should make the exact URL normalization/opening behavior testable, or at least specify the expected final CloudVolume path form.

Questions:

- Will the implementation preserve an exact raw-vertex array in graph order, or rely only on `ERLGraph.node_coords_zyx`?
- Can the real-run verification include the known skeleton `"0"` -> `1465128` registration check and OOB count check before computing ERL?

READY: no