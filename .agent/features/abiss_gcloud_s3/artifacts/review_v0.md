# Review v0

## Summary

- **Reviewer:** planner (claude), run as `claude --print --output-format text
  --no-session-persistence --tools ""` from the repository root, exit 0.
- **Repository unmutated:** tracked and staged diffs were identical before and
  after the call.
- **Prompt:** `state/review_v0.prompt.txt`, 58,484 bytes. It contained the task,
  `run.md`, `plan_v3`, `plan_v3_review`, `code_v0`, every git output the protocol
  requires, the full text of the five untracked new files, and excerpts of the
  existing code the diff depends on.
- **Raw transcript:** `state/review_v0.review.raw.md`.

**Result: 7 major, 4 minor, `READY: no`.**

The whole-volume half is sound. The converter semantics, the layout and
global-shift tests, the `resolve` guards, the §4 masks and both plan_v3_review
minors are implemented correctly. The chunked half is not wired:

- the driver cannot load the chunked config;
- the chunked watershed thresholds sit at a different operating point from the
  reference's;
- chunk-size B, the one-chunk run and the acceptance comparison never run.

The default `run_volume.sh` path is also broken.

The coordinator checked two findings against the repository:

- **#7 confirmed.** `upload_seg_precomputed.py:72-75` exits with
  `mt_sweep.json missing -- run sweep_merge_threshold.py first`. The default
  `DECODE=whole` path therefore now fails at section 3 for every volume.
- **#8 resolved.** `run_volume.sh:15` is `set -euo pipefail`, so a failed
  integrity gate does stop the pipeline. No change needed.

## Diff Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

The review surface is the working tree against `run_start_ref`: 2 tracked files
modified and 5 new untracked files. `HEAD` was checked to equal `run_start_ref`
before the review.

## Findings

### Major

1. **The chunked decode cannot run.**
   - `chunked_abiss.yaml` has no `abiss_chunk:` section, so `prepare_config`
     rejects it.
   - It has no `param:` block (`CHUNK_SIZE`, `AFF_CHANNELS`, `BBOX`, agglomeration
     stage), and no `abiss_home`, `workdir`, `secrets_dir`, `source_affinity_h5`
     or `source_dataset`.
   - `${RUN_PREFIX}` is a literal string that nothing expands.
   - The driver ignores the top-level keys.
2. **The criterion is not configured once and threaded through.**
   - Both `resolve` heredocs hard-code `mean` and chunk size A.
   - The shell pre-check rejects a bad criterion with a message that leaves out
     §5's accepted set, backing stage, stage list and `max` note.
   - Nothing carries the criterion into the chunked stage selection.
3. **The chunked watershed thresholds differ from the reference's.** The YAML sets
   `WS_HIGH_THRESHOLD 0.94` and `WS_LOW_THRESHOLD 0.20` as absolute values, while
   the references use whole-volume percentiles `94%` and `20%`. The chunked run
   must use the absolute values those global percentiles resolve to on
   `aff_canon.h5`, computed once and passed to both sides. Otherwise the
   equivalence test measures a threshold mismatch instead of chunking.
4. **The §6 pipeline and acceptance steps are missing.**
   - Only one chunk-size variant runs.
   - `equivalence_test.py` is never run on chunked outputs.
   - Nothing reads the precomputed chunked segmentation back into ZYX; the test
     reads HDF5 `main` only.
   - The one-chunk plumbing run is not wired.
   - `compare_task_keys` is never called.
5. **The boundary set uses the wrong planes for the second chunk size.** A single
   `--chunk-size` applies to every `--chunked` input, so B's boundary set is built
   from A's planes. Each candidate needs its own chunk size, and the invariance
   check must still run.
6. **`boundary_and_interior` is O(labels × voxels).** It calls
   `np.where(reference == label)` once per label, so on 0.26 Gvox it would not
   finish. Use `scipy.ndimage.find_objects`, or a single pass computing min/max
   per label.
7. **Regression (confirmed by the coordinator).** The production
   `sweep_merge_threshold.py` step is removed and the S3 path replaces the
   default. Section 3 then fails on the missing `mt_sweep.json`. The S3 path must
   be opt-in, and the existing sweep must remain the default.

### Minor

8. The integrity gate stops the pipeline only if the script exits on error.
   **Coordinator: resolved.** `set -euo pipefail` is at `run_volume.sh:15`.
9. `preflight` does not check that the source object exists. VM item 9's checks
   (`min > 0` off the zero face, `max < 1`) are not implemented anywhere.
10. Some tests are weaker than the plan specifies:
    - test 1 patches only `Path.is_file`, not filesystem and network access in
      general;
    - test 7's empty-`tmp_path` check is vacuous, because nothing writes there;
    - there is no `preflight` test.
11. Small deviations:
    - `--input-dataset main` is omitted on the canonical whole-volume calls;
    - `merge_fn_sweep.py` gained a `__main__` guard although the plan listed it as
      unmodified; the change is justified and causes no regression;
    - the `z0 > 0` face restore in `make_prob_affinity.py` rewrites a value that is
      already correct, and needs a comment saying so.

## Tests to Add

- The chunked config loads through `prepare_config` (the `abiss_chunk:` schema),
  and each variant A, B and one-chunk resolves from the YAML, not from literals.
- The chunked watershed thresholds equal the global-percentile values computed
  from `aff_canon.h5`, and the whole-volume reference records the same values.
- Precomputed-to-ZYX readback of a chunked segmentation, with a round trip on a
  small `file://` layer.
- `equivalence_test` with two candidates and two different chunk sizes: each
  candidate's boundary set uses its own planes.
- A performance guard: `boundary_and_interior` on something like
  `(128,512,512)` with thousands of labels finishes in seconds.
- `run_volume.sh` with S3 not opted in still calls `sweep_merge_threshold.py`.
- A `preflight` test with a wrong shape, a wrong dtype and a missing file, and a
  test that a guard failure through the real entry point leaves no output layer.

## Questions

- Where do the global-percentile watershed thresholds come from? Either
  `run_abiss_volume.py` exposes the absolute values it resolves `94%`/`20%` to, or
  a small helper computes them from `aff_canon.h5` with the same percentile rule.
  code_v1 should reuse the existing resolution rather than reimplement it, if
  possible, and record the chosen values in the run output.

## Verdict

VERDICT: NEEDS_CHANGES
