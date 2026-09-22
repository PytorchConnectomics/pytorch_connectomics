# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v0_review.review.raw.md`.

The reviewer returned `READY: no` with eleven findings, all tagged `[major]`, and no minor
findings. The verdict summary: "The plan is not executable safely yet. Its nucleus-state
invariant breaks under the proposed threshold, distributed combination is underspecified, and
verification does not exercise the hierarchy where silent constraint loss is most likely."

Three of the findings (F1, F2, F3) are the same root defect seen from different angles:
`min_voxel_threshold` was applied at *merge* time, which legalizes clusters holding two
different nucleus ids, which in turn destroys the exactness of the fixed-width record and the
associativity of its combination. The remaining findings are genuine internal contradictions
in the plan (file creation, dtype contract, `NUC_MIP`, id width) and real gaps in the
verification plan.

All findings are accepted. None are softened or discarded.

## Findings

**F1 [major] — B3 and B4 directly conflict.** `nuc_can_merge` permits different nonzero ids
whenever either count is below `min_voxel_threshold`, but B4 then aborts whenever a merge
combines different nonzero ids. The plan simultaneously permits and treats-as-unreachable the
same state.

**F2 [major] — the fixed record is not exact under the proposed behavior.** Without `NUC_WS`,
A2 discards minority ids in a mixed supervoxel, and threshold-permitted merges can also create
mixed-id clusters. Once reduced to `(dominant id, count, total)`, the discarded identities can
never participate in a later veto, so the "hard cannot-link" claim is not actually delivered.

**F3 [major] — the winner-selection reduction is not associative.** Concrete counterexample
from the reviewer: records `(id=1,count=6)`, `(id=2,count=10)`, `(id=1,count=6)` should make id
1 dominant after aggregation (12 vs 10), but pairwise "keep the larger count" retains id 2.
Chunk processing order can therefore change the veto state, which breaks the
chunking-independence property the design claims to preserve.

**F4 [major] — C1/C2 cannot reuse the semantic reduction merely because both records occupy 24
bytes.** Semantic payloads support element-wise summation; nucleus ids do not. A valid
combination operation must be defined and implemented at every remapping/reduction point,
including `reduce_chunk` and `match_chunks`. The reviewer notes the current record may be
incapable of such an operation without a stronger invariant.

**F5 [major] — `NUC_MIP` is exported but never used.** C5 passes `AFF_RESOLUTION` as the `mip`
argument instead. The plan must define the exact source-mip and alignment contract.

**F6 [major] — default-path file handling is contradictory.** B5 guards the nucleus output
streams with `!nuc_ids.empty()`, but also requires the files to exist unconditionally, while C4
unconditionally `mv`s `ongoing_nuc.data`, `done_nuc.data`, and `nuc_cuts.data` under
`set -euo pipefail`. File creation and conditional content must be specified separately.
Adding optional keys to `set_env.py` must also explicitly preserve operation when `NUC_PATH`
is absent.

**F7 [major] — the dtype contract is inconsistent.** C5 says valid inputs are cast to `uint32`;
V6 expects a `uint16` input to be rejected. Validation also covers only values above the
maximum, leaving negative, fractional, and non-finite values unspecified.

**F8 [major] — A1 serializes the id as `uint64_t`, contrary to the task's `uint32` contract**
and to success criterion 4. Byte-size equality with `sem_array_t` is not itself a serialization
requirement. One explicit, stable on-disk layout is needed.

**F9 [major] — the verification plan explicitly declines to execute the distributed
hierarchy.** Syntax checks and "mirroring by construction" do not prove that nucleus records
survive `reduce_chunk` / `match_chunks`, nor that OVERLAP=2 veto feedback actually prevents a
later merge. Executable tests of reduction, matching, veto-stream ingestion, and final
segmentation are required.

**F10 [major] — V2 does not fully prove default-path invariance.** The modified run
intentionally creates additional `.data` files, so "cmp every produced `.data` output" has no
defined one-to-one comparison set. V2 also bypasses config, `set_env.py`, cutout generation,
and the shell drivers.

**F11 [major] — V3 does not ensure both tagged groups clear the default 1000-voxel
threshold.** As written the primary veto test may not exercise the veto at all.

## Questions

The reviewer asked three questions, all of which are load-bearing for plan_v1:

1. Should different ids always veto, or may weak conflicting ids merge? If weak merges are
   allowed, what state preserves all evidence for later hierarchy levels?
2. Is `NUC_WS` required to establish the one-id-per-supervoxel invariant, or should the payload
   representation handle mixed supervoxels exactly?
3. Does default-path invariance permit new empty nucleus sidecar files, provided all
   pre-existing outputs remain byte-identical?

## Verdict

VERDICT: NEEDS_CHANGES
