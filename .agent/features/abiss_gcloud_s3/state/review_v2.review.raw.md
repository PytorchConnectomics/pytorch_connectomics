# Review v2: S3 chunked ABISS decode (code_v2)

**Verdict: 1 major and 6 minor findings, `READY: no`.** Of the 16 review_v1 findings, 12 are fixed, 3 are partial and 1 is not fixed. The one major finding is new: the source-dataset discovery was put in the wrong place, and it breaks the default path. The fix is to move a single block. I could only check what the prompt includes, so I did not see the top of `run_volume.sh` or the full `abiss_chunk.py`. Where a finding depends on unseen code, I say so.

## Threshold resolution (the question you asked)

**Yes. Threshold resolution reuses `run_abiss_volume`'s percentile rule over the same set of values.**

- `resolve_thresholds.resolve` calls the imported `_resolve_threshold`, which is `np.percentile` over the whole array, zero faces included. It runs on `_read_array(aff_canon.h5, "main")`.
- With `--channels 0,1,2 --edge-storage destination`, the whole-volume decoder would compute its percentile over `_to_abiss_affinity(canon, [0,1,2], "destination")`. That is a float32 transpose of the same 3 channels with no shift, so it holds the same multiset of values. Percentile does not depend on order, so the two thresholds are identical.
- `test_percentile_thresholds_are_resolved_from_canonical_artifact` pins exactly this. The code comment explaining why the layout conversion is not applied a second time is correct.
- The same absolute `WS_HIGH`/`WS_LOW` values reach `ref_max_canon`, `ref_mean` and every chunked variant.
- `ref_max_compressed` takes its percentiles over the shifted compressed tensor. `uncompress` is monotone and the zero faces match on both sides, so both runs sort voxels the same way. The only exception is float32 saturation ties (N3), which the new diagnostic covers.

One minor inconsistency here is listed as m-new-2 below.

## Status of review_v1 findings

| # | Status | Notes |
|---|---|---|
| N1 | FIXED | The masks in `preflight` and `write_affinity_diagnostic` are `[0,:,:,0]`, `[1,:,0,:]`, `[2,0,:,:]`, which match the converter. The fixture is built with `canon.convert`. |
| N2 | FIXED | Readback asserts a 4-D cutout with `C == 1`, drops the channel axis, then transposes. The fake returns `(4,3,2,1)`. |
| N3 | FIXED | Checks `max ≤ 1`, reports the saturated count and fraction, and writes the diagnostic before the reference decodes. The integrity failure message points to it. (See m-new-2 for a flaw in the diagnostic.) |
| N4 | FIXED | `exit 0` after S3 in both the whole and chunked modes. The `if` nesting is balanced, and the default path still reaches section 3. |
| N5 | FIXED | `BBOX` and the one-chunk size come from `artifact_bbox`. The references and the equivalence test are gated by `S3_ACCEPT`, which defaults to 1 only for `ExPID108*`. |
| N6 | FIXED | `resolve` runs before the GPU half. `$MERGE_CRITERION` reaches `ref_mean` and `write_variant_config`, and the stages come from `stages_for_criterion`. |
| N7 | FIXED | `AFF_PATH` is `file://<run>/chunked_<v>/aff`, `aff_chunk_size_xyz` equals the storage chunk size, and a test asserts the path. |
| m1 | FIXED | `--chunked` is parsed as a string, and `file://` is added explicitly. |
| m2 | PARTIAL | `AGG_THRESHOLD` and `ABISS_HOME` are fixed. The template's `resolution_xyz [18.0, 18.0, 23.9811]` still goes through `_maybe_int_list` and ends up as an integer (23 if it truncates). This affects only layer metadata, but the review asked for it to be recorded and it was not. |
| m3 | FIXED | The shell derives chunk sizes from `variant_specs`. |
| m4 | FIXED | `--plumbing` mode asserts `VOI_total == 0` exactly and is wired to `chunked_one`. It only runs after A and B have been decoded, so a plumbing failure is found late. That wastes compute but is not wrong. |
| m5 | **NOT FIXED** | `compare_task_keys(CHUNKED_STAGES, stages_for_criterion(...))` compares a constant with itself and can never fail. The plan (§6) asks for a two-way comparison of the per-chunk task-flag keys under `SCRATCH_PATH`. Nothing reads those flags. |
| m6 | PARTIAL | The dataset name is now discovered, but the discovery sits in the wrong place (**M1** below). |
| m7 | FIXED (documented) | Covered by a docstring in `resolve_thresholds`. `preflight` and `write_affinity_diagnostic` also copy full arrays, which the docstring does not mention. |
| m8 | PARTIAL | Four of the five weak tests are unchanged (details below). |
| m9 | FIXED, with a new edge case | Labels are relabelled with `np.unique` before `find_objects`. See m-new-1. |

Details for m8:
- The I/O-patch test only patches `Path.is_file`, and `resolve` never calls it.
- The empty-`tmp_path` assertion is still vacuous.
- The dtype case is still not isolated. Both fixtures fail on **shape** against the default `BBOX` of 650×650×503, so `match="float32"` passes whether or not the dtype check works.
- There is still no `run_volume.sh` stub test, which would have caught M1.

## New findings

### [major] M1: source-dataset discovery runs before the affinity file can exist

The new `SOURCE_DATASET=$(python … h5py.File(sys.argv[1]) …)` block is placed unconditionally before `if [[ -s "$AFF_H5" ]]`. That check exists because the file may not be there yet.

- **Fresh volume, `STAGES=all|gpu`, any `DECODE`, including the default empty one:** `h5py.File` raises and the substitution exits non-zero. This relies on the script running under `set -e`, which its style suggests; I could not see the top of the file. If it does, the script aborts before inference. That breaks existing default behaviour.
- **`STAGES=cpu`** (the GPU/CPU split from the last commit): the block sits inside the GPU half, so it never runs. `ref_max_compressed` then gets an unbound `$SOURCE_DATASET`, which is fatal under `set -u` and otherwise produces `--input-dataset ""`. The S3 branch already recomputes `AFF_H5` for this reason but does not recompute `SOURCE_DATASET`.
- **Fix:** move the discovery into the S3 branch, right after that branch recomputes `AFF_H5`.

### [minor] m-new-1: labels are misclassified when the reference has no background

`boundary_and_interior` relabels with `inverse`, so the smallest label maps to 0. If `W` contains no zeros, `find_objects` treats that label as background. It is skipped and silently lands in `I`.

The `all_boundary = np.ones(...)` test passes for this reason, not because the label is interior: label 1 is never boxed. Real whole-volume output almost always contains 0, so the risk is low. The fix is to relabel as `inverse + int(unique_labels[0] != 0)`.

### [minor] m-new-2: the diagnostic percentiles are not the thresholds actually used

`write_affinity_diagnostic` computes p20 and p94 with the zero faces excluded. `resolve_thresholds`, like the decoders, includes them. So the JSON's `percentile_94` is not the `WS_HIGH` that ws receives, which weakens the N3 tie diagnostic. The fix is to record the resolved `ws_high`/`ws_low` in the diagnostic, or to compute it over the same full array.

### [minor] m-new-3: other small defects

- `write_variant_config` adds a new `MERGE_CRITERION` key to the ABISS param payload. The stages ignore it, so it is only a tag. It is probably harmless if the param loader tolerates unknown keys, but that is unverified.
- `AGG_THRESHOLD = 0.1394546` is still written literally in both the shell (`ref_mean`) and the template.
- `resolve_variant("one")` skips the `AFF_CHANNELS` check.
- On spot resume, `make_prob_affinity` and all three whole-volume references run again from scratch because nothing checks whether their output already exists. This costs time but does not cause a failure.

## Verdict

- Every VM-blocking defect from review_v1 (N1, N2, N7) is fixed in the diff.
- The acceptance arithmetic is right: masked VOI with 0 counted as a label, half-open plane crossing, invariance under the `W≠0` mask, and shared absolute thresholds.
- M1 still has to be fixed before the default path or a `STAGES=cpu` VM run can work. It is a one-block move.

READY: no
