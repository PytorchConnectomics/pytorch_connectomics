# Review v1: final code version

I checked code_v1 against the full working-tree diff from `7d819bdb`, using only the artifacts in the prompt. The code_v1.md description of what changed is accurate. But the new code has a few defects that the offline tests either don't exercise or pass by building the same mistake into their fixtures. Two of them would stop the VM run before any decode, and a third would stop it at the acceptance step.

## Status of the review_v0 findings

| # | Status | Notes |
|---|---|---|
| 1 | **PARTIAL** | The generated config now loads through `prepare_config`, but the affinity path it points at is wrong (see N7). |
| 2 | **PARTIAL** | The guard exists and its error text meets §5. But the criterion still comes from three places, and the guard runs after GPU inference (see N6). |
| 3 | **FIXED** | `resolve_thresholds.py` computes the absolute 94%/20% values once from `aff_canon.h5`, using the existing `_resolve_threshold`. They are saved to JSON and passed to `ref_max_canon`, `ref_mean` and every chunked variant. `ref_max_compressed` keeps percentile strings on the compressed input. Any value strictly between two neighbouring sorted values selects the same voxels in both spaces, so that part is sound. Float32 rounding still weakens it (see N3). |
| 4 | **PARTIAL** | A and B now run and are compared, and precomputed readback exists. But the one-chunk run is decoded and never checked; `compare_task_keys` is still never called; and the readback itself is broken (N2). |
| 5 | **FIXED** | `--chunk-size` is now given once per `--chunked` input and matched by index. The chunk-size invariance check uses the `W≠0` mask. |
| 6 | **FIXED** | Bounding boxes now come from a single `find_objects` pass. The requested performance test was not added (see minor m8). |
| 7 | **FIXED** | With `DECODE` empty, the script calls `sweep_merge_threshold.py` exactly as before. The opt-in path gains a new problem (see N4). |
| 8 | **FIXED** | No change was needed: `set -euo pipefail` already stops the pipeline. |
| 9 | **PARTIAL** | The range checks were added, but they check the wrong faces, so preflight fails on every real artifact (N1). Whether the GCS source object exists is still not checked; only `aff_canon.h5` is. |
| 10 | **PARTIAL** | Tests were added for preflight, thresholds and readback. But the patched-I/O test and the empty-directory test are still vacuous, and there is still no "guard failure through the real entry point leaves no layer" test and no "default path still calls the sweep" test. Two of the new tests build the defects into their fixtures (N1, N2). |
| 11 | **FIXED** | `--input-dataset main` was added to the canonical calls, and the face-restore line now has a comment. One side effect is listed as minor m6. |

## New or remaining defects

### Major

**N1. Preflight's face mask is transposed, so it rejects every real `aff_canon.h5`.**
- In `resolve_chunked.py::preflight` the excluded faces are `exposed[0,0,:,:]`, `exposed[1,:,0,:]` and `exposed[2,:,:,0]`.
- In CZYX layout, channel 0 is the X edge, whose zero face is X=0: `[0,:,:,0]`. Channel 2 is the Z edge, whose zero face is Z=0: `[2,0,:,:]`. `make_prob_affinity.convert` writes exactly those faces, and `test_layout_equivalence_and_zero_faces` asserts them.
- Preflight has channels 0 and 2 swapped. The real zeros at `[0,:,:,0]` therefore count as active voxels, `min == 0`, and preflight raises.
- That preflight heredoc runs for both `DECODE=whole` and `DECODE=chunked`, so no S3 decode would ever start.
- `test_preflight_and_prepare_config_for_each_variant` builds its fixture with the same swapped faces (`data[0,0]=0`, `data[2,:,:,0]=0`), so it passes.
- Fix: `exposed[0,:,:,0]=False; exposed[1,:,0,:]=False; exposed[2,0,:,:]=False`. Build the test fixture with `make_prob_affinity.convert` rather than by hand.

**N2. `read_precomputed_zyx` fails on real CloudVolume output.**
- `CloudVolume[:, :, :]` returns an array of shape `(X, Y, Z, C)`, with a trailing channel axis. `np.transpose(..., (2,1,0))` on a 4-D array raises `ValueError: axes don't match array`.
- The fake volume in `test_precomputed_readback_returns_zyx` returns a 3-D array, which hides the bug.
- Fix: check that `C == 1`, drop that axis with `[..., 0]`, then transpose. Change the fake to return shape `(4,3,2,1)`.
- As a result, acceptance item 13 cannot run as written.

**N3. Float32 rounding breaks both preflight's `max < 1` check and the integrity gate's invariance argument.**
- `p = sigmoid(logit(v)/0.2)` rounds to exactly `1.0f` once `1−p < 2^-25 ≈ 3e-8`. That happens when `logit(p) > 17.3`, i.e. when `logit(v) > 3.47`, i.e. for any compressed value `v > 0.970`. Confident affinities in the interior of neurites very plausibly exceed that.
- Consequence 1: preflight's `max < 1` check (plan VM item 9) fails even after N1 is fixed.
- Consequence 2: every compressed value above 0.970 collapses to the same 1.0. If more than 6% of voxels do so, the 94th percentile on the canonical artifact is exactly 1.0. The set of voxels passing `WS_HIGH` then differs between `ref_max_canon` and `ref_max_compressed`. The gate would then either fail for no real reason, or pass only because the threshold happened to land somewhere the ties don't matter.
- Fix: relax the check to `max ≤ 1` and report how many voxels saturated. Before the reference decodes, add a diagnostic of how many ties the saturation creates (or of the 94th percentile), so a gate failure caused by rounding can be told apart from a real layout bug. The alternative is to reconsider float32 for the canonical artifact. The plan fixed float32, so this is a plan inconsistency, but it blocks the run.

**N4. The S3 path falls through into section 3.**
- Section 3 (building the precomputed layer and meshes) sits inside the same `runs cpu` block and runs whatever `DECODE` is set to.
- With `DECODE` set, `upload_seg_precomputed.py` then either exits on a missing `mt_sweep.json`, or, since ExPID108 already has a sweep, rebuilds the old sweep segmentation into the publish path. The first makes the S3 run's exit status meaningless. The second is unwanted work next to a published layer.
- Fix: exit, or skip sections 3 and later, once the S3 branch completes.

**N5. The same code path cannot run unmodified on a larger volume.**
- The task's success criterion says it must. But `BBOX [0,0,0,650,650,503]` is hard-coded in both `resolve` heredocs, in the variant loop, in `write_variant_config`'s default argument, and in `VARIANTS["one"]` (`[650,650,503]`).
- The chunked branch also always runs all three whole-volume references and the integrity gate. A volume with no whole-volume answer would still attempt whole-volume `ws` runs (about 148 GB of RSS at 8× the size), and a larger volume would exceed the `uint32` cap outright.
- Fix: derive `BBOX` from the shape of `aff_canon.h5` (preflight already reads it). Derive the one-chunk size from `BBOX`. Put the references and the equivalence test behind a switch such as `S3_ACCEPT=1`.

**N6. Criterion threading (review_v0 finding 2, still open).**
- **The guard runs after GPU inference.** The criterion check sits in section 2. With the default `STAGES=all`, the GPU half (section 1) has already finished. Requirement 1 says the failure must come at config resolution, before any VM work. Fix: move the `resolve` call to the top of `run_volume.sh`, or into the host launcher.
- **The criterion comes from three places.**
  - `write_variant_config` takes `MERGE_CRITERION` from the template's top-level `merge_criterion`, not from the `$MERGE_CRITERION` environment variable, which is only passed to `resolve_variant`.
  - The shell passes a literal `--stages … agglomerate_mean_edge …` instead of using `stages_for_criterion(...)`.
  - Today the guard restricts everything to `mean`, so the three sources cannot disagree. Still, this is not "configured once and threaded identically".

**N7. The chunked config points `AFF_PATH` at the source HDF5 instead of a per-run precomputed layer.**
- `write_variant_config` sets `AFF_PATH = file://<aff_canon.h5>` while `source_affinity_h5` is also set.
- Plan §1 puts the copied layer at `${RUN_PREFIX}/chunked_<tag>/aff/`. With the source set, the driver's HDF5-to-precomputed copy targets `AFF_PATH`, which here is a path to an existing *file*. Depending on the driver, it will either fail to create `…/aff_canon.h5/info`, or treat the path as an HDF5 backend and skip the copy. In the second case, all three variants read one shared input, which is not what the plan specifies.
- `prepare_config` passing in the unit test does not exercise the copy.
- Fix: set `AFF_PATH` to `file://{root}/aff`. Consider also setting `aff_chunk_size_xyz` to the storage chunk size, as plan §1 says.

### Minor

- **m1. Readback paths.** `--chunked "$RUN_PREFIX/chunked_A/seg"` is passed without `file://`, and `argparse type=Path` would turn `file:///x` into `file:/x`. Parse candidates as strings and add `file://` explicitly for precomputed paths.
- **m2. Template values that are silently ignored or wrong.**
  - `AGG_THRESHOLD` is read from `raw` (the top level), so the template's `param.AGG_THRESHOLD` is always replaced by the default.
  - `abiss_home: /opt/abiss` ignores `$ABISS_HOME`.
  - `resolution_xyz [18,18,24]` does not match the plan's `23.9811`, since the driver takes integers. This is metadata only, but the difference should be recorded.
- **m3. Chunk sizes are duplicated.** The shell hard-codes `256 256 128` and `192 192 96` separately from `VARIANTS`, so the two can drift apart. Emit them from `resolve_chunked`.
- **m4. The one-chunk plumbing check (VM item 12) is not wired.** `compare()` would raise on it anyway, because with one chunk there are no boundary labels and `|B|=0`. Add a `--plumbing` mode that asserts `VOI_total == 0`.
- **m5. `compare_task_keys` is still never called.** It is required by §6.
- **m6. `ref_max_compressed` now assumes the source dataset is named `main`.** The plan says to discover it, as `make_prob_affinity` does. Pass the name that was discovered.
- **m7. Memory.** `preflight` loads the full array plus two same-sized copies, and `resolve_thresholds` makes about three full copies. That is fine at 0.26 Gvox and about 60–75 GB at 8×. Stream by slab, or document the memory needed.
- **m8. Still-weak tests.**
  - The patched-I/O test only patches `Path.is_file`, and `resolve` never calls it anyway.
  - The empty-`tmp_path` assertion is vacuous.
  - The shape/dtype preflight test uses a mismatched `BBOX`, so the dtype case never isolates the dtype check.
  - The performance test for `boundary_and_interior` was not added.
  - There is no test that `run_volume.sh` still calls the sweep when `DECODE` is empty.
- **m9. `find_objects` allocates a list as long as the largest label ID.** That is fine for contiguous ABISS IDs but costly for sparse, large IDs. Relabel first with `fastremap.renumber`.

## Assessment

Requirement 3's layout reasoning (the layout test, the global shift, the slab halo) and the threshold resolution are correct. The per-candidate boundary sets, the efficient masks and the restored default path are also correct. But:
- N1 stops both S3 modes at preflight;
- N2 stops acceptance at readback;
- N3 very likely stops preflight even after N1 is fixed, and makes the integrity gate unreliable.

The offline suite passes only because its fixtures contain the same mistakes as N1 and N2. N4 through N7 are wiring gaps against the task's stated criteria. Each fix is local, but this was the last allowed code version, and as it stands the pipeline cannot reach VM items 9–14.

READY: no
