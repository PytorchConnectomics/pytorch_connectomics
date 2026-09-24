1. [major] **C7 uses the wrong maximum label.** C3’s LUT has one entry per **original watershed ID**, whereas `merge_segments` compacts output IDs and executes `counts.resize(next_id)`. Consequently, `lut.size() - 1` is not the maximum output ID. After merges or dust removal, C7 can reject valid uint32 outputs and fail Verification step 3’s exact-fit test based on `meta_*`. Use the final compact label count (`counts.size() - 1` before freeing counts), or the maximum LUT value. Add a fixture where merging substantially reduces the label count.

2. [major] **C0 and Verification step 5 still do not establish the requested peak-memory attribution.** Logging after `stable_sort` misses its temporary allocation; cumulative `VmHWM` cannot identify its anonymous/file-backed composition. Also, streamed writes create page-cache charges that do **not** appear in the writer’s `RssFile`, contrary to the C5 risk discussion. `sacct MaxRSS` is not an equivalent fallback for aggregate parent/child plus page-cache cgroup usage. Add phase-bracketed sampling and a defined aggregate job-memory measurement; if unavailable, explicitly report that metric as unverified. This leaves previous finding 4 unresolved.

3. [major] **The graph-padding contract remains unreliable.** Two identical stock runs do not establish that tuple padding is deterministic. Changing allocations can change padding bytes even when both stock runs happened to match. Verification step 2 would then reject semantically exact results for an irrelevant reason. Require identical record count, layout, and field values for `dend_*` unconditionally; report padding-byte differences separately. Confirm the ABI’s actual field offsets rather than inferring them from the tuple’s template argument order. Previous finding 7 is only partially resolved.

4. [major] **Verification still omits several explicitly planned failure paths.** Add duplicate dtype-token rejection, `offset > UINT32_MAX`, checked-product overflow, and batch failure after an earlier successful threshold. For C1, test the exact `high_bit` boundary as well as a larger product. The batch fixture should verify that the failed threshold creates no artifacts and that the Python callback is never invoked after subprocess failure. Previous finding 6 is substantially improved, but not fully closed.

5. [minor] **The revised memory target overstates the bound.** C2 still needs O(P) queue storage when one search spans the volume, even if N and E are small. Therefore “anon ≈ S·P + O(N + E) throughout” omits a potentially dominant allocation. Express the queue term using maximum search size and allocated capacity. Explicitly clear the vector’s logical size between searches while retaining capacity; resetting cursors alone is insufficient when appending with `push_back`.

6. [minor] **The proposed new branch conflicts with the supplied repository instructions.** Scope mandates creating `feature/ws-memory`, but the user has not requested branch creation. Remove that step and preserve the existing branch.

The requested exactness arguments check out as follows:

- **C2 BFS:** The cursor invariant is correct. A zero seed advances all cursors together; hitting an existing label drains `bfs_start` and sets `bfs_index = bfs_end`; normal exhaustion leaves `bfs_index == bfs_end`, followed by assignment draining `bfs_start`. Already-labeled seeds preserve equality. Resetting between searches therefore preserves traversal and assignment order, provided queue contents and logical size are reset consistently. Keep neighbor arithmetic signed.
- **C3 LUT:** `lut[id] = remaps[find_set(id)]` reproduces both the voxel remapping and graph endpoint mapping. Build it over the original ID domain before freeing union-find/remaps. Path compression changes representation, not component membership. Each threshold must still receive a fresh copy of the original counts. The LUT transformation is valid; C7’s interpretation of its length is not.
- **C4 `in_rg`:** The removal is sound. An inserted edge immediately joins its endpoints’ MST components, and those components never split. A later occurrence of the same unordered endpoint pair cannot pass the different-component test.
- **C6 accumulators:** MAX matches the supplied first-element initialization and strict `<` update, including signed-zero and NaN selection behavior. MEAN matches the source-level addition order, initial value, count conversion, and division. Machine-level exactness requires FP settings that preserve that order; copying compiler flags alone is insufficient if those flags permit reassociation. Preserve first-seen pair order and add direct accumulator comparisons with rounding-sensitive values. PERCENTILE remains unchanged.

Previous findings 1, 5, 8, and 9 are addressed; finding 2 has a sound conditional solution. Findings 3 and 4 remain incomplete around peak accounting, finding 6 has remaining coverage gaps, and finding 7 needs a stronger contract.

Questions:

- Will uint32 validation use the final compact label count or the maximum LUT value?
- Will `dend_*` field equality become the unconditional correctness contract?
- What aggregate memory measurement replaces cgroup `memory.peak` when unavailable?
- Do the complete stock compiler flags prohibit floating-point reassociation?

READY: no