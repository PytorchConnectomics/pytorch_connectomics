# j0126: conservative segmentation, morphology-based reconnecting

This tutorial turns a 10 nm j0126 EM volume into a neuron segmentation in three steps:

1. Predict voxel affinities.
2. Decode them conservatively with ABISS, preferring splits to false merges.
3. Reconnect only high-confidence, morphologically continuous branches.

The last step is prediction-only at runtime. It uses the segmentation, affinities, predicted morphology, and an external nucleus-instance manifest; it does not read evaluation skeletons, their lookup table, or an FFN segmentation.

## Before running

Edit **only** [params.yaml](params.yaml). It contains the repository checkout, dataset root, writeable output root, and the existing artifacts required to replay the frozen EC recipe. Every step inherits this file, so paths are not duplicated across the workflow YAMLs. Keep the algorithmic thresholds in the step YAMLs unchanged when reproducing the reference recipe.

The zero-shot affinity path needs a NISB-trained checkpoint. The supervised affinity YAML is included as a target-domain reference only: it uses j0126 dense labels, so it is not part of the zero-shot pipeline.

## Step 1 — affinity prediction

Run zero-shot inference with an NISB checkpoint:

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml \
  --mode test --checkpoint /path/to/nisb_affinity.ckpt
```

For the full volume, run independent one-GPU shards:

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml \
  --mode test --checkpoint /path/to/nisb_affinity.ckpt \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 80
```

The output is chunked float16, three-channel affinity under `output_root/affinity` after resolving `params.yaml`.

## Step 2 — conservative ABISS decode

First inspect the planned ABISS commands:

```bash
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml --prepare-only
```

Then run them:

```bash
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml
```

The fixed watershed and agglomeration settings intentionally under-merge. This leaves recoverable fragments for step 3 instead of welding uncertain neurons.

## Step 3 — morphology-based error correction

Step 3 builds skeletons for large predicted segments, evaluates every sufficiently confident contact, and accepts only hard-gated branch continuations. It protects external nucleus identities and never joins two different identities.

Preview the complete plan first:

```bash
python scripts/run_error_correction.py \
  --config tutorials/neuron_j0126/3_merge.yaml --stage all --num-tasks 1 --dry-run
```

For the full volume, run the array stages with the same task count configured in `3_merge.yaml` (80 in the reference run):

```bash
CFG=tutorials/neuron_j0126/3_merge.yaml

python scripts/run_error_correction.py --config "$CFG" --stage sizes
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

python scripts/run_error_correction.py --config "$CFG" --stage postprocess \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80
python scripts/run_error_correction.py --config "$CFG" --stage verify
```

Stages are restartable: completed chunk artifacts are reused. For a one-core smoke test, append `--max-owned-chunks 1` to an array-stage command.

## Frozen reference result

The recipe freezes 749 branch unions. Evaluation is deliberately outside the EC config and should be run only after the proposal has been frozen.

| segmentation | NERL mt=0 | NERL mt=5 | VOI split | VOI merge | VOI sum |
|---|---:|---:|---:|---:|---:|
| v7 unions | 0.300729 | 0.539311 | 2.354895 | 0.019229 | 2.374124 |
| v7 + 3x3x3 boundary erosion | 0.440506 | 0.527696 | 2.311561 | 0.128437 | 2.439998 |

Use `erosion_radius_zyx: [0, 0, 0]` for the first row and `[1, 1, 1]` for the strict-mt=0 cosmetic cleanup in the second row. The cleanup improves mt=0 NERL, but is not the best choice for mt=5 NERL or VOI sum.
