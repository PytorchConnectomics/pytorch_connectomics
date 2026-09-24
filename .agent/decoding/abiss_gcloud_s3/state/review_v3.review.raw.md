# Review v3

## Summary

code_v3 closes M1. The GPU half is byte-identical to the baseline: the diff has no hunk between the `if runs gpu` block and `fi # end GPU half`. `SOURCE_DATASET` discovery now runs inside the S3 branch, right after `AFF_H5` is recomputed. So a fresh volume reaches inference, and `STAGES=cpu` sets `SOURCE_DATASET` before `ref_max_compressed` uses it.

The new stub test can catch M1 coming back. The old code passed `$AFF_H5` to `python -`, so it would appear in the GPU-only call log, and the test asserts that it does not.

code_v3 introduced no major defects. Of the review_v2 findings still open, all are minor, and code_v3.md openly lists most of them as not done.

## Status of review_v2 findings

| # | Status | Evidence in the diff |
|---|---|---|
| **M1** | **FIXED** | `run_volume.sh`: the GPU `AFF_H5` block is unchanged from the baseline. The `h5py` discovery sits inside `else` of `if [[ -z "$DECODE" ]]`, after `AFF_H5=` is recomputed. The stub test covers three cases: `STAGES=gpu` with no `DECODE` reaches `scripts/main.py` without passing the affinity path; `STAGES=cpu DECODE=whole` runs `h5py` before `run_abiss_volume.py` with `--input-dataset main`; and the default path still reaches `sweep_merge_threshold.py`. |
| m-new-1 | FIXED | `relabeled = inverse + (unique[0] != 0)`, `object_index = r + offset - 1`. I checked both cases by hand. With a background label, label index `r` maps to `objects[r-1]`. Without one, it maps to `objects[r]`. The new `[3,3,4,4,4,5,6,6]` case correctly gives B={4} and I={3,5,6}. The `all_boundary` test now fails for the right reason: label 1 spans `[0,8)` and crosses plane 4. |
| m-new-2 | FIXED | `write_affinity_diagnostic(..., ws_high, ws_low)` is called with the exact values from `ws_thresholds.json`, and those same values go to `ws` and to both chunked configs. The face-excluded percentiles stay in the file under separate names (`percentile_20`/`percentile_94`), so they can no longer be mistaken for the thresholds. |
| m-new-3 | PARTIAL | Only the task-key format was addressed. Four items are unchanged, as code_v3.md admits: `MERGE_CRITERION` is still passed through into the ABISS param payload via `payload = dict(param)`; `AGG_THRESHOLD` still appears as a literal in both the shell and the YAML; `resolve_variant("one")` still skips the `AFF_CHANNELS` check; and resume still re-runs the converter and references. |
| m2 | NOT FIXED | `resolution_xyz` is still turned into integers by `_maybe_int_list`, so `23.9811` becomes 23. This is now documented under Risks. It only affects layer metadata, not any VOI number. |
| m5 | PARTIAL | `read_task_flag_keys` and `compare_task_keys` now read the real `SCRATCH_PATH/done/*.txt` markers, in the same format as the vendored `check_task_flag.py`/`update_task_flag.py`, and the check runs in both directions with a test. **But nothing in `run_volume.sh` calls it**, and no code builds the expected key set from `BBOX`/`CHUNK_SIZE`/stages. On the VM, plan §6's key-set comparison never runs. |
| m6 | FIXED | Same change as M1. |
| m8 | PARTIAL | Fixed: the dtype-only fixture now has the correct shape `(3,2,4,4)` against `BBOX [...,4,4,2]`, and a real `run_volume.sh` stub test exists. Still weak: the I/O-patch test still patches only `Path.is_file`, which `resolve` never calls, and `assert not list(tmp_path.iterdir())` is still vacuous because nothing could have written there. |

## New findings in code_v3

1. **[minor]** `compare_task_keys` is dead code in the pipeline (see m5). Either call it after each `run_abiss_chunk.py`, with an expected-key generator, or drop the claim that the key-set comparison is part of the run.
2. **[minor]** `preflight` in `run_volume.sh` runs before `write_affinity_diagnostic`. If it fails, its error text points at `affinity_diagnostic.json`, which does not exist yet. Write the diagnostic first, or reword the message.
3. **[minor]** Carried from v2, not new, but relevant to the "8× larger, unmodified" criterion: `preflight`, `write_affinity_diagnostic` and `resolve_thresholds` each load the whole canonical artifact, plus a full boolean mask and a copy of the active values. At 8× (~2 Gvox × 3 channels × float32 ≈ 25 GB per copy), peak memory is roughly 60 GB per call. That fits a high-memory CPU box, so nothing breaks, but it will not scale to 100 µm, unlike the converter, which streams by Z-slab.
4. **[minor] Unverified risk for VM item 12 (plumbing test).** Variant `one` uses `CHUNK_SIZE = [650,650,503]` with `seg_chunk_size_xyz [128,128,128]`, which are not multiples of each other. The offline `prepare_config` test passes, but I can't see from here whether the driver's alignment check lives in `prepare()` and would reject this pairing. If it does, item 12 fails before acceptance. This is not a code_v3 regression.

What I checked and found correct:
- Default behaviour is unchanged: with `DECODE` empty, the early resolve block is skipped, the GPU half is untouched, and the CPU half runs the sweep as before.
- The `merge_fn_sweep.py` `__main__` guard keeps script behaviour.
- Both decoders get the same absolute `WS_HIGH`/`WS_LOW`. Because `max` is monotone-invariant, the integrity gate is still meaningful even though `ref_max_compressed` uses `94%`/`20%` and `ref_max_canon` uses absolute values. An interpolated percentile falls between the same two neighbouring stored values under a monotone map, so both sides keep the same set of edges.

## Verdict

The one major finding (M1) is fixed in the diff, and no new major defects were introduced. The remaining items are minor and do not affect whether the VM run can execute or whether the acceptance numbers mean anything.

READY: yes
