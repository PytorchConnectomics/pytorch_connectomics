# review_v0 — raw planner notes (in-session; planner == claude == this session)

Method: read the full tracked diff vs 3c4f5621 file by file, plus the untracked
src/seg/NucExtractor.hpp; then independently re-ran the load-bearing checks rather than
relying on code_v0.md.

## Independent reproduction (not taken from the coder's report)

* V2: built baseline binaries at 3c4f5621 in a throwaway `git worktree`, same cmake flags
  (Release, EXTRACT_SIZE=ON, conda pytc boost), ran make_fixture.py --nuc none + the atomic
  staging under both builds.
    RESULT: identical=35  differing=0  missing=0
    modified-only: done_nuc.data size=0, nuc_cuts.data size=0, ongoing_nuc.data size=0
  Matches the coder's "35 baseline files byte-identical" exactly.
* V5/V6: recompiled both unit binaries myself.
    test_nuc_algebra: PASS   (incl. "nuc: voxel count overflow: 18446744073709551615 + 1")
    test_nuc_extractor: PASS (conflict_sv 1 / minority_sv 1 / subfloor_sv 1 / subfloor_voxels 10)
* V3/V4: all five nucleus modes.
    none      remaps=1 nuc_cuts=0  MERGED
    different remaps=0 nuc_cuts=1  SEPARATE cut=(100,200)
    same      remaps=1 nuc_cuts=0  MERGED
    one       remaps=1 nuc_cuts=0  MERGED
    max       remaps=0 nuc_cuts=1  SEPARATE cut=(100,200)
  Fixture affinity is ~0.97, so the veto demonstrably is NOT affinity-gated. 0xFFFFFFFF
  behaves as an ordinary id, confirming the H5 fix.

## Code reading — correct

* Types.h: nuc_wire_t 29 bytes with all five offset static_asserts; nuc_is_dominant uses
  __uint128_t (exact, closes K1a); nuc_add checked (closes K1b); nuc_join matches plan_v6 A4;
  nuc_can_merge branches carry the Invariant D clause annotations and match the truth table.
  nuc_can_merge IS in Types.h — the K4 contradiction is resolved.
* NucExtractor: NONE branch leaves the default record, so total==0 — the J1 fix is real.
  Tie-break `count > max || (count == max && id < max_id)` is order-independent (checked both
  hash orders). Output sorted by sid, so the file is deterministic despite absl iteration order.
* mean_aggl: veto placed after the frozen block, before the sem check, ungated by affinity;
  propagation mirrors the sem swap discipline exactly (v0/v1 not yet swapped at that point);
  nuc files opened with trunc unconditionally, content gated on !nuc_ids.empty() — B5 honored.
  The frozen-edge condition at 686-687 was NOT touched, as plan_v6 required.
* filesize() returns 0 for a missing file (Utils.hpp:12-17), so load_nuc / reduce_nuc /
  process_nucs all degrade gracefully rather than aborting when the payload is absent.
* README carries Invariant D's three clauses, the 99/1 counterexample, Bound C's corrected
  "recorded usable evidence" scope, clause 3's real over-segmentation cost, and the
  perinuclear-tagging caveat.

## Defects found

1. [major] atomic_chunk_ME.cpp:104-107 — nuc_ratio_from_env() and nuc_min_tagged_from_env()
   are evaluated when constructing the extractor, OUTSIDE the `exists("nuc.raw")` guard. A
   stale/malformed ABISS_NUC_DOMINANCE now aborts acme on a run with NO nucleus input. New
   failure mode on the default path. V2 cannot catch it because V2 runs a clean environment.
2. [major] No re-runnable V2 harness. work/test/ has make_fixture.py, run_atomic_fixture.sh and
   the per-feature tests, but nothing that builds the baseline and does the comparison. V2 is
   the Seuron bit-identity gate; I could only confirm it by reconstructing the whole thing by
   hand. That is exactly the step a future change will skip.
3. [minor] Collision counters over-count: load_nuc (mean_aggl), reduce_nuc (reduce_chunk),
   process_nucs (match_chunks) all increment whenever the RESULT is CONFLICT, so a third record
   joining an already-CONFLICT accumulator increments again. These counters are the designated
   observable for a veto failure (plan_v6 B6/Q2); inflating them is misleading. Increment only
   on the transition into CONFLICT.
4. [minor] Default-path log noise: NucExtractor::output prints four `nuc:` lines and load_nuc
   prints "nuc: no nucleus labels" on EVERY chunk with no nucleus input. ~600 extra lines on a
   143-shard run. Gate on having a source.
5. [minor] load_nuc aborts on a nucleus sid absent from seg_indices. Mirrors load_sem
   (mean_aggl.cpp:255-266) so it is consistent, but it is worth confirming the ongoing/done
   partition keeps the sets aligned in the COMPOSITE path, where seg_indices comes from
   ongoing_supervoxel_counts.data rather than ns.data.
6. [minor] merge_chunks_me.py now merges ongoing_nuclei_labels unconditionally, so a pipeline
   mixing pre-change and post-change binaries breaks at the merge. Out of scope to fix; should
   be a README upgrade note.

None of 1-6 is a correctness defect in the nucleus feature itself. 1 and 2 are default-path /
verification issues, 3-6 are hygiene.
