# Review v3

## Summary

- **Reviewer:** planner (claude), `claude --print --output-format text
  --no-session-persistence --tools ""`, exit 0.
- **Repository unmutated:** tracked and staged diffs were identical before and
  after the call.
- **Prompt:** `state/review_v3.prompt.txt`, 104,617 bytes. It included the full
  current `run_volume.sh` and the vendored ABISS `check_task_flag.py` and
  `update_task_flag.py`.
- **Raw transcript:** `state/review_v3.review.raw.md`.

**Result: 0 major, `READY: yes`.** The only major finding from review_v2 (M1) is
fixed in the diff, and code_v3 introduced no new major defects. What remains is
minor, and code_v3.md openly lists most of it as not done.

**Coordinator verification:**

- The offline suites pass, 31/31, re-run independently.
- `bash -n run_volume.sh` passes.
- `git diff 7d819bdb -- run_volume.sh` has no hunk inside the GPU half.

The VM acceptance items (plan items 9–14) have **not been run**. This approval
covers the offline implementation only. It does not show that chunked decoding
equals whole-volume decoding.

## Diff Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Findings

### Status of review_v2 findings

| # | Status |
|---|---|
| M1 | FIXED: GPU half byte-identical to baseline; discovery inside the S3 branch; a stub test guards it |
| m-new-1 | FIXED: relabel offset; the no-background case is tested |
| m-new-2 | FIXED: the resolved `ws_high`/`ws_low` are recorded under their own names |
| m-new-3 | PARTIAL: only the key format was addressed; the other four items are unchanged, as disclosed |
| m2 | NOT FIXED, documented: integer `resolution_xyz`, which affects metadata only |
| m5 | PARTIAL: real two-way flag comparison implemented and tested, but never called from `run_volume.sh` |
| m6 | FIXED |
| m8 | PARTIAL: dtype test isolated and stub test added; the I/O-patch and empty-`tmp_path` tests are still vacuous |

### Minor (carry forward)

1. `compare_task_keys` is not called in the pipeline, and nothing generates the
   expected keys, so plan §6's key-set check does not run on the VM.
2. `preflight` runs before `write_affinity_diagnostic` but refers to the
   diagnostic in its error text.
3. `preflight`, the diagnostic and `resolve_thresholds` each load the whole
   array, about 60 GB at 8×. That fits a high-memory box but will not scale to
   100 µm.
4. **Unverified:** the one-chunk variant pairs `CHUNK_SIZE [650,650,503]` with
   `seg_chunk_size_xyz [128,128,128]`. If the driver's `prepare()` enforces
   alignment, VM item 12 fails before acceptance. Check this first on the VM.
5. m-new-3 leftovers: the `MERGE_CRITERION` key in the payload, the
   `AGG_THRESHOLD` literal written twice, the missing channel check on
   `resolve_variant("one")`, and non-resumable references.

## Tests to Add

- An expected-task-key generator from `BBOX`/`CHUNK_SIZE`/stages, checked with
  `compare_task_keys` after each chunked run.
- A `prepare_config` plus driver-alignment test for variant `one`.

## Questions

None. Before the VM acceptance run, settle the variant `one` alignment (minor 4).

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
