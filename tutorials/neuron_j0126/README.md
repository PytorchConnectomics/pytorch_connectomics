# j0126 / zebrafinch

This directory contains affinity and ABISS examples plus the reproducible morphology error
correction (EC) used for the current native-resolution result. The EC is prediction-only at
runtime: it reads the segmentation, its affinity, predicted-segment morphology, and an external
nucleus-instance manifest. It never reads evaluation skeletons, an evaluation LUT, or FFN output.

The supervised affinity checkpoint in `1_affinity_supervised.yaml` was trained on zebrafinch
dense labels, so the complete image-to-segmentation experiment is supervised. “GT-free” below
describes the EC proposal generator, not the affinity model or an unbiased test-set claim. The
v7 rule was developed while the 50-skeleton benchmark was visible and its result is therefore a
development-set result.

## What step 3 runs

`3_merge.yaml` replaces the old fragment-growth decoder with the frozen v7 pipeline:

1. Aggregate ABISS segment sizes and skeletonize every segment with at least 100k voxels in
   `[512,512,256]` XYZ cores with halo.
2. Reconnect only same-label skeleton pieces at chunk faces. Derive branch length, local caliber,
   backbone/twig structure, bushiness/glia flags, endpoints, and nucleus ownership.
3. Build the full voxel-face RAG for all segments at the 200-voxel ABISS dust floor, including
   source-indexed affinity evidence and physical-volume boundary flags.
4. Examine every contact with `affinity_ge08_fraction >= 0.05` (20,857 contacts on the reference
   run). At the closest skeleton approach, measure multiscale tangent continuation, leaf distance,
   caliber compatibility, and the short/thin/perpendicular spine veto.
5. Freeze five hard-gated tiers: direct continuation, unique internal branch, atomic two-host
   connector, nucleus-host attachment, and stable long-branch continuation. A component may
   contain at most 12 input labels and never more than one external nucleus identity.
6. Apply all unions to the dense volume, then optionally erase only boundaries between remaining
   distinct objects. The reported cleanup uses a 3x3x3 window (`erosion_radius_zyx: [1,1,1]`).

The proposal is hash-checked before the dense write. The output manifest records the proposal
hash and operation order. The package deliberately exposes no oracle-selected junction-scope
command.

## Run

First inspect the exact commands resolved from YAML:

```bash
python scripts/run_error_correction.py \
  --config tutorials/neuron_j0126/3_merge.yaml \
  --stage all --num-tasks 1 --dry-run
```

For the whole volume, run the three expensive passes as arrays and the reductions serially. The
commands are restartable; completed chunk artifacts are skipped.

```bash
CFG=tutorials/neuron_j0126/3_merge.yaml

python scripts/run_error_correction.py --config "$CFG" --stage sizes

# Submit each command as array 0-79 (or run task ids manually).
python scripts/run_error_correction.py --config "$CFG" --stage skeletonize \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80
python scripts/run_error_correction.py --config "$CFG" --stage contacts \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80

python scripts/run_error_correction.py --config "$CFG" --stage skeletons
python scripts/run_error_correction.py --config "$CFG" --stage contact_graph
python scripts/run_error_correction.py --config "$CFG" --stage candidates
python scripts/run_error_correction.py --config "$CFG" --stage junction_scope
python scripts/run_error_correction.py --config "$CFG" --stage junction_features
python scripts/run_error_correction.py --config "$CFG" --stage boundary
python scripts/run_error_correction.py --config "$CFG" --stage resolve
python scripts/run_error_correction.py --config "$CFG" --stage prepare_output

# Submit as array 0-79 after prepare_output.
python scripts/run_error_correction.py --config "$CFG" --stage postprocess \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80

python scripts/run_error_correction.py --config "$CFG" --stage verify
```

For a one-core smoke test, append `--max-owned-chunks 1` to an array stage. A completely serial
run is also supported with `--stage all --num-tasks 1`, but is not practical for j0126.

`3_merge.yaml` currently names the exact native-96 matchguard base used by the paper experiment.
For a new ABISS decode, change `segmentation`, `affinity_chunks`, `keep_mask`,
`nucleus_manifest`, and `size_glob` together. `size_glob` may name either all atomic ABISS
`seg_size_*.data` tables or an already aggregated table; the workflow always emits a new sorted,
hash-reported size inventory under its own work directory.

## Reference result

The reference run freezes 749 branch unions. Evaluation is intentionally a separate downstream
operation and is not present in the EC YAML.

| segmentation | NERL mt=0 | NERL mt=5 | VOI split | VOI merge | VOI sum |
|---|---:|---:|---:|---:|---:|
| frozen v7 | 0.300729 | 0.539311 | 2.354895 | 0.019229 | 2.374124 |
| v7 + 3x3x3 inter-object erosion | 0.440506 | 0.527696 | 2.311561 | 0.128437 | 2.439998 |

Use `[0,0,0]` for `erosion_radius_zyx` when mt=5 NERL or VOI sum is the target. The 3x3x3
variant is a strict-mt=0 cleanup: it gains +0.139777 mt=0 NERL but costs -0.011615 mt=5 and
+0.065874 VOI sum.
