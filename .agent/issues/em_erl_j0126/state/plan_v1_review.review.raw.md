**Prior Findings**

- [major] Canonical node-coordinate source: addressed. Plan now uses `graph.get_nodes_position(None)` as the sampling source and preserves graph node order for the LUT.
- [major] Registration proof: addressed. Plan now requires skeleton `"0"` -> segment `1465128`, OOB count near the verified `37 / 500,845`, and zero-ratio checks before trusting ERL.
- [minor] Drop `-r/--resolution`: addressed. The plan is voxel-unit only.
- [minor] CloudVolume opener under-specified: addressed. `normalize_seg_url()` and explicit `CloudVolume(...)` construction are planned and unit-tested.

**New Findings**

No new major or minor findings. The plan is executable, appropriately scoped, reuses the ERL APIs, keeps LUT order aligned by construction, and adds verification strong enough to catch silent coordinate swaps/offsets/scaling errors.

READY: yes