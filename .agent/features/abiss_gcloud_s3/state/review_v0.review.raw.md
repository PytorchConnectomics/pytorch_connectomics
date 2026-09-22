# Review of code_v0 (S3, plan_v3 + plan_v3_review minors)

The whole-volume half is mostly correct. The converter matches plan_v3's canonical tensor semantics, and the guards and mask definitions follow §3 and §4. The chunked half is not wired: the chunked config can't be loaded by the driver, only one chunk size is run, and the equivalence test can't read a chunked output. The default `run_volume.sh` path also no longer runs the existing production sweep.

## Findings

1. **[major] The chunked decode cannot run.**
   - `chunked_abiss.yaml` has no `abiss_chunk:` section, so `prepare_config` raises "has no abiss_chunk section" straight away.
   - It also lacks `abiss_home`, `workdir`, `secrets_dir`, `source_affinity_h5`, `source_dataset` and a `param:` block holding `CHUNK_SIZE`, `AFF_CHANNELS`, `BBOX` and the agglomeration stage.
   - `${RUN_PREFIX}` is a literal string. Nothing shown expands environment variables.
   - The driver never reads the top-level keys (`merge_criterion`, `CHUNK_SIZE`, `chunk_variants`, …).

2. **[major] The merge criterion is not configured once and threaded through.**
   - Both `resolve` heredocs hard-code `"merge_criterion": "mean"` and chunk size A. They don't read `$MERGE_CRITERION` or the YAML, so variant B is never resolved.
   - The shell check `[[ "$MERGE_CRITERION" == mean ]]` rejects bad criteria before `resolve` runs. Its message leaves out the accepted set, the backing stage, the list of chunked stages and the `max` note. That breaks Requirement 1 and §5.
   - Nothing carries the criterion into the chunked decoder's stage selection.

3. **[major] The chunked watershed thresholds differ from the reference's.**
   - `chunked_abiss.yaml` sets `WS_HIGH_THRESHOLD: 0.94` and `WS_LOW_THRESHOLD: 0.20` as absolute values.
   - The references use `94%` and `20%`, which are percentiles of the whole volume.
   - On probability-space affinity these are different operating points. Percentiles computed per chunk would also differ from global ones.
   - The chunked side needs the absolute values that the global percentiles resolve to on `aff_canon.h5`, fixed once. Otherwise the equivalence test measures a threshold mismatch, not chunking.

4. **[major] The pipeline order and acceptance steps from §6 are missing.**
   - `DECODE=chunked` runs one variant only. A and B never both run.
   - Nothing runs `equivalence_test.py` on chunked outputs.
   - Nothing reads the chunked result back from its precomputed layer into ZYX. The test only reads HDF5 `main`.
   - The one-chunk plumbing run (item 12) is not wired.
   - `compare_task_keys` is defined but never called, so the §6 two-way key-set check is missing.

5. **[major] The boundary set uses the wrong planes for the second chunk size.**
   - `equivalence_test.py` takes one `--chunk-size` for every `--chunked` input. When A and B are passed together for the invariance check, B's boundary set is built from A's planes, so B's seams aren't tested as boundary.
   - Running them separately avoids that, but then the invariance check `VOI_total(C_A, C_B)` never runs.
   - Each candidate needs its own chunk size.

6. **[major] `boundary_and_interior` is too slow for the acceptance volume.**
   - It calls `np.where(reference == label)` over the whole volume once per label, which is O(labels × voxels).
   - On ExPID108 (0.26 Gvox, likely thousands of labels) item 13 would not finish in practice.
   - Using `scipy.ndimage.find_objects`, or one pass computing min/max per label, fixes it.

7. **[major] Regression: the production CPU stage is replaced.**
   - `sweep_merge_threshold.py --volume "$VOL"` is removed. The default `DECODE=whole` now runs S3 reference decodes and the converter on every volume.
   - Section 3 (`upload_seg_precomputed.py --volume "$VOL"`) is unchanged and presumably expects the sweep's output, which no longer exists.
   - The S3 path should sit behind an explicit opt-in, such as `DECODE` being set or an `S3=1` flag, and the old behaviour should stay the default.

8. **[minor] The integrity gate only stops the pipeline if `run_volume.sh` exits on error.** That depends on `set -e` being in force, which the excerpt doesn't show. Please confirm it, or add an explicit `|| exit`.

9. **[minor] `preflight` checks less than plan §3 and VM item 9 require.** It doesn't check that the source object exists. It also doesn't check `min > 0` off the zero face or `max < 1`.

10. **[minor] Several offline tests are weaker than the plan asks.**
    - Test 1 patches only `Path.is_file`, not filesystem and network access in general.
    - Test 7's ordering check (`tmp_path` is empty) is vacuous, because nothing ever points output at `tmp_path`.
    - There is no test for `preflight`.

11. **[minor] Small deviations from the plan.**
    - The whole-volume calls on `aff_canon.h5` leave out `--input-dataset main`, which plan §1 specifies. It probably works anyway because `main` is the only dataset.
    - `merge_fn_sweep.py` was changed although the plan listed it as not modified. The `__main__` guard is justified and doesn't cause a regression.
    - In `make_prob_affinity.py`, the `z0 > 0` face restore rewrites the same value the halo slice already supplies. It is harmless but misleading, and the comment should say so.

## Parts that are correct

- **Converter semantics:** channel reversal, then the global shift using the Z halo, then inversion, then exact zero faces. Output is float32 CZYX with dataset `main`.
- **Layout test:** the bitwise layout-equivalence test and the global-vs-per-chunk shift test.
- **Guards:** the `resolve` checks for criterion, alignment, the ≥2-chunks rule and `AFF_CHANNELS`, and the §5 error text inside `resolve`.
- **Masks and VOI:** the §4 plane and crossing definitions; `0` in the candidate counted as its own label; `W ≠ 0` for the invariance check; `|B| > 0` and `|I| > 0` both enforced (the two plan_v3_review minors).
- **Integrity references:** both `max` references use the flags and thresholds from plan §2.

READY: no
