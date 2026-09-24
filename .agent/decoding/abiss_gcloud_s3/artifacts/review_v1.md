# Review v1

## Summary

Reviewer: planner (claude), `claude --print --output-format text
--no-session-persistence --tools ""`, exit 0. Tracked and staged diffs were
unchanged by the call. Prompt: `state/review_v1.prompt.txt` (90,314 bytes; full
diff against `run_start_ref`, all untracked files, and excerpts of the existing
code). Raw transcript: `state/review_v1.review.raw.md`.

**7 major, 9 minor, `READY: no`.** Of the 11 review_v0 findings, 6 are fixed
(3, 5, 6, 7, 8, 11) and 5 are partially fixed (1, 2, 4, 9, 10).

Correction to the prompt: it told the reviewer that code_v1 was the final
allowed version. That is wrong, because `revision_rounds: 2` allows `code_v2`.
The framing does not change any finding.

**Coordinator verification** of the findings that stop the run:

- **N1 confirmed.** `resolve_chunked.py:96-98` excludes
  `[0,0,:,:]`/`[1,:,0,:]`/`[2,:,:,0]`, but the converter zeroes
  `[0,:,:,0]`/`[1,:,0,:]`/`[2,0,:,:]`. Channels 0 and 2 are swapped.
- **N2 confirmed.** `equivalence_test.py:55` transposes the 4-D CloudVolume cutout
  with a 3-axis permutation.
- **N3 confirmed numerically.** A compressed value of `0.9705` maps to float32
  `1.0` after inversion.
- **N7 confirmed.** `resolve_chunked.py:157` sets `AFF_PATH = file://<aff_canon.h5>`.

The offline suite passes 28/28 (re-run by the coordinator). It passes because the
N1 and N2 test fixtures contain the same mistakes as the code.

## Diff Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Findings

### Status of review_v0 findings

| # | Status | Notes |
|---|---|---|
| 1 | PARTIAL | loads through `prepare_config`; `AFF_PATH` is wrong (N7) |
| 2 | PARTIAL | §5 guard exists; criterion still has 3 sources and the guard runs after GPU work (N6) |
| 3 | FIXED | absolute thresholds resolved once and shared; float32 saturation undermines it (N3) |
| 4 | PARTIAL | A/B run and are compared; one-chunk not checked, key-set never called, readback broken (N2) |
| 5 | FIXED | per-candidate chunk size; invariance check under `W≠0` |
| 6 | FIXED | `find_objects` one-pass bounding boxes |
| 7 | FIXED | default path calls the sweep again; opt-in path falls through (N4) |
| 8 | FIXED | no change needed |
| 9 | PARTIAL | range checks use the wrong faces (N1); GCS source existence not checked |
| 10 | PARTIAL | new tests added; several still vacuous, and two build in the N1/N2 defects |
| 11 | FIXED | side effect recorded as m6 |

### Major

1. **N1. The preflight face mask is transposed.** Every real `aff_canon.h5` is
   rejected, so neither S3 mode can start. Fix the mask to
   `[0,:,:,0]`/`[1,:,0,:]`/`[2,0,:,:]` and build the test fixture with
   `make_prob_affinity.convert`.
2. **N2. `read_precomputed_zyx` fails on real output.** The CloudVolume cutout is
   `(X,Y,Z,C)`. Assert `C == 1`, drop the channel axis, then transpose. The test
   fake must return 4-D. Acceptance cannot run until this is fixed.
3. **N3. Float32 saturation.** `p` rounds to `1.0` for compressed `v > 0.970`.
   - Preflight's `max < 1` check fails on real data.
   - Ties at 1.0 can shift the 94th-percentile high threshold, so the integrity
     gate becomes unreliable.
   - Relax the check to `max ≤ 1`, report the saturated fraction, and add a
     tie/percentile diagnostic before the reference decodes.
   - This is a tension in the plan itself, which fixed float32.
4. **N4. The S3 path falls through into section 3.** It then publishes or fails
   on the old sweep's artifacts. It must exit, or skip section 3, after S3
   completes.
5. **N5. The code does not run unmodified on a larger volume.**
   - `BBOX` and the one-chunk size are hard-coded in five places. Derive them from
     `aff_canon.h5`.
   - The chunked mode always runs the whole-volume references. Put them behind an
     acceptance switch.
6. **N6. The criterion guard runs after GPU inference, and the criterion has
   three sources.**
   - Move `resolve` to the start of the script.
   - Thread `$MERGE_CRITERION` into `write_variant_config`.
   - Use `stages_for_criterion` instead of a literal stage list.
7. **N7. `AFF_PATH` points at the source HDF5.** It should point at the per-run
   precomputed `${RUN_PREFIX}/chunked_<tag>/aff/`, so the driver's copy runs.
   Also set `aff_chunk_size_xyz` to match the storage chunk size.

### Minor

- **m1.** `argparse type=Path` mangles `file://` URLs. Parse them as strings and
  add the scheme explicitly.
- **m2.** Template values are ignored:
  - `param.AGG_THRESHOLD` is read from the wrong level;
  - `abiss_home` ignores `$ABISS_HOME`;
  - the integer `resolution_xyz [18,18,24]` differs from `23.9811`; record this.
- **m3.** Chunk sizes are duplicated between the shell and `VARIANTS`.
- **m4.** The one-chunk plumbing check is not wired, and `compare()` would raise
  on `|B|=0`. Add a `--plumbing` mode that asserts `VOI_total == 0`.
- **m5.** `compare_task_keys` is never called.
- **m6.** `ref_max_compressed` assumes the dataset name `main` instead of
  discovering it.
- **m7.** Full-array copies in `preflight` and `resolve_thresholds` need about
  60–75 GB at 8× volume. Stream them, or document the requirement.
- **m8.** Tests remain weak:
  - the I/O-patch test is vacuous;
  - the empty-`tmp_path` assertion is vacuous;
  - the dtype case is not isolated;
  - there is no `boundary_and_interior` performance test;
  - there is no test that the default path calls the sweep.
- **m9.** `find_objects` allocates up to the maximum label ID. Renumber the
  labels first.

## Tests to Add

- Preflight on a fixture built by `make_prob_affinity.convert`, including
  saturated inputs (`v ≥ 0.98`).
- Precomputed readback with a 4-D `(X,Y,Z,1)` fake, plus a real `file://`
  CloudVolume round trip if `cloudvolume` is available in the env.
- `write_variant_config` puts `AFF_PATH` under the variant root, not on the HDF5
  file.
- `BBOX` derived from the artifact: a non-650 fixture produces matching `BBOX`
  and one-chunk sizes.
- `--plumbing` mode passes on identical volumes and fails on a single changed
  voxel.
- `run_volume.sh`, with `python` stubbed:
  - empty `DECODE` calls the sweep;
  - set `DECODE` does not reach section 3;
  - a bad `MERGE_CRITERION` exits before the GPU half.

## Questions

- For N3: is float32 saturation acceptable if the diagnostic shows the 94th
  percentile falls below the saturated band? Or should the canonical artifact
  store float64, or store the logit? code_v2 should apply the relaxed check and
  the diagnostic, and report which applies. It should not change the plan's dtype
  unilaterally.

## Verdict

VERDICT: NEEDS_CHANGES
