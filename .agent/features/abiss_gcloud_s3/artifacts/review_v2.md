# Review v2

## Summary

- **Reviewer:** planner (claude), `claude --print --output-format text
  --no-session-persistence --tools ""`, exit 0.
- **Repository unmutated:** tracked and staged diffs were identical before and
  after the call.
- **Prompt:** `state/review_v2.prompt.txt`, 98,484 bytes.
- **Raw transcript:** `state/review_v2.review.raw.md`.

**Result: 1 major, 6 minor, `READY: no`.**

Of the 16 review_v1 findings, 12 are fixed (N1–N7, m1, m3, m4, m7, m9). Three
are partial (m2, m6, m8) and one is not fixed (m5). Every defect from review_v1
that blocked the VM run is fixed in the diff. The reviewer also checked the
acceptance arithmetic and found it correct:

- masked VOI with 0 counted as a label;
- half-open plane crossing;
- chunk-size invariance under the `W≠0` mask;
- one set of absolute thresholds shared by both decoders.

The reviewer confirmed that threshold resolution uses `run_abiss_volume`'s own
percentile rule over the same set of values the decoder sees.

**Coordinator verification of M1: confirmed.** `run_volume.sh:118-131` opens
`$AFF_H5` with `h5py` unconditionally, at the start of the GPU half and before
the `if [[ -s "$AFF_H5" ]]` inference-skip check at line 132. `set -euo pipefail`
is in force (line 15). On a volume whose affinity does not exist yet, the default
path therefore aborts before inference, which is a regression of existing
behaviour. With `STAGES=cpu`, the S3 branch at line 219 reads `$SOURCE_DATASET`
while it is unset.

`revision_rounds: 2` is exhausted, and the finding is major. In `normal` mode the
run **blocks for a human decision**.

## Diff Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Findings

### Status of review_v1 findings

| # | Status |
|---|---|
| N1 N2 N3 N4 N5 N6 N7 | FIXED |
| m1 m3 m4 m7 m9 | FIXED (m7 by documentation; m9 introduces m-new-1) |
| m2 | PARTIAL: `resolution_xyz` still becomes an integer and the difference is not recorded |
| m5 | NOT FIXED: `compare_task_keys` compares a constant with itself and never reads the task flags under `SCRATCH_PATH` |
| m6 | PARTIAL: the dataset name is discovered, but in the wrong place (M1) |
| m8 | PARTIAL: see the list below |

Weak tests that m8 still leaves:

- the I/O-patch test only patches `Path.is_file`, which `resolve` never calls;
- the empty-`tmp_path` assertion is vacuous;
- the dtype case is not isolated, because both fixtures fail on shape first;
- there is still no stub test for `run_volume.sh`.

### Major

1. **M1: source-dataset discovery runs before the affinity file can exist.**
   - It breaks the default path on a fresh volume (confirmed above).
   - With `STAGES=cpu`, `$SOURCE_DATASET` is unset when `ref_max_compressed` uses
     it.
   - Fix: move the discovery into the S3 branch, right after it recomputes
     `AFF_H5`.

### Minor

2. **m-new-1: the relabel-before-`find_objects` step misclassifies a label when
   `W` has no 0.** The smallest label maps to 0 and silently lands in `I`. The
   `all_boundary` test passes for this reason. Fix: offset the relabel by
   `unique_labels[0] != 0`.
3. **m-new-2: the diagnostic percentiles are not the thresholds ws receives.**
   They exclude the zero faces, while `resolve_thresholds` and the decoders
   include them. Record the resolved `ws_high`/`ws_low` in the diagnostic.
4. **m-new-3: small defects.**
   - `MERGE_CRITERION` is a new, unverified key in the ABISS param payload.
   - `AGG_THRESHOLD` is duplicated as a literal in the shell and the template.
   - `resolve_variant("one")` skips the `AFF_CHANNELS` check.
   - On spot resume, the converter and the references run again from scratch.
5. **m2** (partial, above).
6. **m5** (not fixed, above): the key-set check needs to read the real task flags.
7. **m8** (partial, above).

## Tests to Add

- A stub test for `run_volume.sh`: with no affinity file present and `DECODE`
  empty, the script reaches inference. With `STAGES=cpu DECODE=whole`,
  `SOURCE_DATASET` is set before it is used.
- `boundary_and_interior` on a reference with no 0 voxels.
- A dtype-only `preflight` failure with a correctly shaped fixture.
- A task-flag key-set check against a fake `SCRATCH_PATH` flag directory:
  one flag missing, and one extra flag present.

## Questions

- How should M1 be closed now that code rounds are exhausted? The fix is a
  one-block move with an obvious form. The human decides whether to allow a
  directed fix outside the rounds, add a round (`c3`), or leave the run blocked.

## Verdict

VERDICT: NEEDS_CHANGES
