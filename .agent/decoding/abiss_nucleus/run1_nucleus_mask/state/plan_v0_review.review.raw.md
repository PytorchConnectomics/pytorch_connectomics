## Summary

The plan is not executable safely yet. Its nucleus-state invariant breaks under the proposed threshold, distributed combination is underspecified, and verification does not exercise the hierarchy where silent constraint loss is most likely.

## Findings

- [major] B3 and B4 directly conflict. `nuc_can_merge` permits different nonzero IDs whenever either count is below the threshold, but B4 then aborts whenever merging different nonzero IDs. That state is explicitly permitted, not unreachable.

- [major] The fixed record is not exact under the proposed behavior. Without `NUC_WS`, A2 discards minority IDs in a mixed supervoxel. Threshold-permitted merges can also create mixed-ID clusters. Once reduced to `(dominant id, count, total)`, the discarded identities cannot participate in later vetoes, violating the hard cannot-link requirement.

- [major] The proposed winner-selection reduction is not associative. For example, records `(id=1,count=6)`, `(id=2,count=10)`, `(id=1,count=6)` should make ID 1 dominant after aggregation, but pairwise “keep the larger count” can retain ID 2. Chunk order can therefore change the veto state.

- [major] C1/C2 cannot reuse semantic reduction merely because both records occupy 24 bytes. Semantic payloads support element-wise summation; nucleus IDs do not. The plan must define and implement a valid combination operation at every remapping/reduction point, including `reduce_chunk` and `match_chunks`. The current record may be incapable of such an operation without a stronger invariant or richer state.

- [major] `NUC_MIP` is exported but never used. C5 instead passes `AFF_RESOLUTION` as the `mip` argument. The plan must define the exact source-mip/alignment contract and use `NUC_MIP`, including its required/default behavior.

- [major] Default-path file handling is contradictory. B5 guards nucleus output streams with `!nuc_ids.empty()` but also requires files to exist unconditionally, while C4 unconditionally moves `ongoing_nuc.data`, `done_nuc.data`, and `nuc_cuts.data`. Their creation and conditional writing must be specified separately. Likewise, adding optional keys to `set_env.py` must explicitly preserve operation when `NUC_PATH` and `NUC_MIP` are absent.

- [major] The dtype contract is inconsistent. C5 says valid inputs are cast to `uint32`, but V6 expects a `uint16` input to be rejected. Validation also mentions only values above the maximum, leaving negative, fractional, and non-finite values unspecified. Choose strict `uint32` input or validated lossless conversion and test that exact behavior.

- [major] A1 serializes the ID as `uint64_t`, contrary to the task’s specified `nuc_t`/`uint32` record field and the success criterion. Equal byte size with `sem_array_t` is not itself a serialization requirement. The on-disk layout and all readers/writers need one explicit, stable contract.

- [major] The verification plan explicitly declines to execute the distributed hierarchy. Syntax checks and “mirroring by construction” do not prove nucleus records survive reduce/match or that OVERLAP=2 veto feedback prevents a later merge. A synthetic multi-chunk test—or focused executable tests of reduction, matching, veto-stream ingestion, and final segmentation—is required.

- [major] V2 does not fully prove default-path invariance. The modified run intentionally creates additional `.data` files, so “cmp every produced `.data` output” has no defined one-to-one comparison. It also bypasses config, `set_env.py`, cutout generation, and shell drivers. The comparison set, treatment of new empty sidecars, and end-to-end no-`NUC_PATH` driver test must be specified.

- [major] V3 does not ensure that both tagged groups satisfy the default 1000-voxel threshold. It must either construct sufficiently large groups or set `ABISS_NUC_MIN_VOXELS` explicitly; otherwise the primary veto test may not exercise the veto.

## Questions

- Should different IDs always veto, or may weak conflicting IDs merge? If weak merges are allowed, what state preserves all evidence for later hierarchy levels?
- Is `NUC_WS` required to establish the one-ID-per-supervoxel invariant, or should the payload representation handle mixed supervoxels exactly?
- Does default-path invariance permit new empty nucleus sidecar files, provided all pre-existing outputs remain byte-identical?

READY: no