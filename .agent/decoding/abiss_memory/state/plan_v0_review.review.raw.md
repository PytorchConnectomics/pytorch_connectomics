1. **[major] C5 breaks default calls to the preserved binaries.** `_run_abiss_ws` would pass `--seg-dtype=uint64`, but the supplied stock parser treats that token as a threshold and calls `std::stof` on it. This contradicts “default behaviour unchanged” while stock binaries remain installed. Omit the token for uint64; require the new binary only for uint32. Test default command construction against the old CLI contract.

2. **[major] C4’s mean accumulator can overflow and invalidate bit-exactness.** A `uint32` boundary-face count is not sufficient for supported large `ws64` volumes; an edge can collect more than `UINT32_MAX` faces. Use `size_t` or another proven sufficient width, retaining the original conversion to `F` for division. For MAX, explicitly initialize from the first observed value and preserve `std::max_element` comparison behavior, including negative values, signed zeros, and NaNs. For MEAN, matching source-level addition order is necessary, but the claim also requires compatible floating-point compiler settings. Real-crop comparisons are evidence, not a general proof.

3. **[major] The peak-memory model omits a full output copy and mapped output pages.** `utils.hpp::write_volume` materializes an owning, cropped `boost::multi_array<uint64_t,3>`. `write_multi_array` then maps the output and copies into it. During single-threshold writing, live allocations include the padded relabeled volume **plus an 8 B/interior-voxel crop**, with another potentially resident 8 B/interior-voxel output mapping. Multi-threshold writing additionally retains the original segmentation. The table also omits union-find arrays, remaps, counts, graph construction/sort storage, and graph copies. Consequently, the asserted attribution of the measured 24 B/voxel peak is unsupported.

   Revise the table using separate padded/interior voxel counts and separate anonymous allocations, resident mappings, and cgroup page-cache charges. C2 potentially saves substantially more than its stated copy alone.

4. **[major] C0 and the memory gate cannot establish the proposed phase attribution.** Logging after watershed misses the live BFS queue; logging after region-graph construction misses its edge containers and sorting workspace. `VmHWM` is cumulative, so it cannot identify later phase peaks below an earlier high-water mark. Instrument allocation-heavy points inside those phases or sample with phase markers. Measure the Python parent and cgroup/job peak separately from binary RSS: the supplied driver retains `predictions_czyx` throughout the subprocess, and `/usr/bin/time -v` around the binary does not capture that composite footprint. Add a concrete acceptance criterion for memory reduction and report runtime impact.

5. **[major] C1 relies on a size limit that Release builds need not enforce.** The supplied limit is an `assert`, so “ws always qualifies” is not an enforceable contract under `-DNDEBUG`. Moreover, the cited 3.82 Gvox `ws` measurement exceeds the stated internal-ID limit and needs explanation before serving as a valid baseline. Require checked dimension multiplication and runtime validation against the internal-ID and signed-index limits. Test dispatch immediately around its actual cutoff near **2³²**, plus both queue implementations on identical small inputs. A ≥2³¹ run alone does not exercise the wide queue branch.

6. **[major] Verification misses important changed behavior and assumes a nondeterministic overflow fixture.**
   - `1 1 1 1 1 1` and `1 1 1 0 0 0` never exercise front segmentation faces or any `aff_i_*` output. Add false front flags, preferably all-false and mixed cases.
   - Add deterministic synthetic inputs covering background-only results, isolated components, dust removal, ties, repeated edge observations, and multiple independently merged thresholds.
   - The overflow test at `2**32 - 10` fails only if enough labels survive. Construct that condition explicitly; test exact-fit success, one-past failure, and overflow-safe checking without computing an unchecked unsigned `offset + max_id`.
   - Exercise dtype-option placement and invalid values, both binary widths, batch callbacks, and both reader layouts.

7. **[major] “Every output byte identical” needs a baseline determinism check.** `write_vector` writes raw C++ object representations, including `std::tuple<F,ID,ID>` graph records. Such records can contain padding—for example with float scores and uint64 IDs—which is not guaranteed deterministic across allocations or builds. First compare repeated stock runs and a stock-source build with matching compiler settings. If padding differs, the plan must resolve the artifact contract explicitly; neither silently weakening comparison nor changing serialization is consistent with the current promise.

8. **[minor] C3 is logically sound, but its memory bound needs qualification.** Precomputing `remaps[find_set(id)]` and applying it during output preserves label values if zero handling, graph traversal order, and boundary transforms remain unchanged. However, supervoxel count can approach voxel count, so the new LUT is not necessarily small. Specify its lifetime and release union-find/remap storage as soon as possible. Also remove the avoidable by-value input copy in `relabel_region_graph`; the supplied implementation copies the graph before constructing another graph.

9. **[minor] Several additional exact memory reductions deserve evaluation.**
   - The BFS cursors coincide after a completed search. Resetting them and using reusable, incrementally allocated storage could scale queue memory with the largest search instead of the entire volume. Preserve traversal order and benchmark allocation behavior.
   - `in_rg` appears redundant: once an edge joins two MST components, the same endpoint pair cannot pass the different-component test again. Prove that invariant before removing its per-label sets.
   - The Python reader materializes a complete copy from a memmap. An explicitly managed mapped-output consumption path could reduce memory further, but requires a clear lifetime contract around temporary-directory cleanup.
   - After graph construction, only boundary affinity data is needed. Evaluate retaining those faces and releasing the full affinity mapping, distinguishing reduced process RSS from cgroup cache reclamation.

Questions:

- Why is the 3.82 Gvox measurement attributed to `ws` despite its internal-ID limit?
- Which compiler flags and source revisions produced both stock binaries?
- Is the acceptance metric binary RSS, total job/cgroup memory, or both?
- If stock graph-file padding is nondeterministic, what byte-exact contract should replace the current unconditional promise?
- For batch overflow, may earlier completed threshold outputs remain, or must the invocation leave no segmentation outputs?

READY: no