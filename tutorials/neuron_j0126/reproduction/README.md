# Reproduce the frozen j0126 Zebrafinch decode

This directory replays the two ABISS rows used by the PyTC2 j0126 ablation:

1. conservative decoding with a tissue-border exclusion mask; and
2. the same decode with nucleus-instance competition enabled.

The workflow reuses an existing three-channel affinity prediction. It does not
train a network and does not read the test skeletons until the final evaluation
job. All volume computation is submitted as non-interactive Slurm jobs.

## Frozen inputs

The replay requires:

- the complete grid-indexed `float16` affinity store and its sibling
  `.h5.index.json`;
- the full-resolution tissue keep mask;
- `yl_cb_80nm_neuron_v2.h5` for the nucleus-aware row;
- the test-50 skeleton HDF5 only for evaluation; and
- ABISS commit `452efa5f87f9d3cb241891ee44010d966a33b316`.

The captured algorithmic parameters are in [seuron_provenance.json](seuron_provenance.json).
Do not change the `BBOX`, resolution, chunk size, watershed thresholds, dust
threshold, or agglomeration threshold for the frozen replay.

## 1. Make run configs

Copy [exclusion.example.yaml](exclusion.example.yaml),
[nucleus.example.yaml](nucleus.example.yaml), and `seuron_provenance.json` to a
writeable run-config directory. Replace every `/path/to` value. The two YAML
files must use different output roots. Keep the affinity store in place; the
workflow reads it but does not copy or modify it.

Use a Python environment containing the PyTorch Connectomics runtime
dependencies, CloudVolume, HDF5, NumPy, SciPy, NetworkX, PyYAML, and `em_erl`.
Set `PYTHON` to that environment's interpreter in every command below.

## 2. Build the frozen ABISS dependency

Clone ABISS into the ignored `lib/` directory and pin it before submitting the
build:

```bash
git clone https://github.com/PytorchConnectomics/ABISS.git lib/abiss
git -C lib/abiss checkout 452efa5f87f9d3cb241891ee44010d966a33b316

sbatch --export=ALL,REPOSITORY="$PWD",ABISS_HOME="$PWD/lib/abiss",PYTC_PREFIX=/path/to/conda/env \
  tutorials/neuron_j0126/reproduction/slurm/00_build_abiss.sbatch
```

`PYTC_PREFIX` supplies Boost headers and libraries. The job builds `ws` and
`agg`, then runs all three ABISS tests. Record the successful job ID.

## 3. Run fail-closed preflight

Submit [01_preflight.sbatch](slurm/01_preflight.sbatch) after the build. Its
required environment variables are listed at the top of that script. Preflight
checks the exact ABISS commit, all 726 affinity chunks and edge reads, the keep
mask and nucleus shapes, the test skeleton file, and available output capacity.
It writes a machine-readable JSON report and exits nonzero on any mismatch.

## 4. Run the ABISS runtime smoke test

Use [02_smoke.sbatch](slurm/02_smoke.sbatch) with an exclusion-only smoke YAML.
A cropped run cannot use the frozen full-volume `AFF_KEEP_MASK` with this HDF5
chunk backend, so omit that field in the smoke config. The default is one
origin-anchored native ABISS chunk (512×512×256). It checks input decoding,
the frozen watershed binary, remapping, and output writing.

The exact frozen ABISS commit cannot complete a representative cropped replay:
nonzero origins and cropped multi-chunk hierarchies produce watershed IDs that
lack the full-volume global RAG namespace expected by later mean-edge
processing. A nucleus crop can validate scan/competition and write its
manifest, but `agg` then fails closed on the unmatched ID. Therefore the
single-chunk watershed smoke is only a runtime check; use the regression
evaluation of the existing frozen full nucleus result as the semantic gate,
then run the fresh full-volume configs. Do not remove `AFF_KEEP_MASK` or change
`ABISS_NUC_MIN_TAGGED: 1024` in either full-volume config.

## 5. Prepare and submit a full replay

Preparation creates the precomputed output layers and deterministic ABISS task
lists but does not run chunks:

```bash
RUN_ROOT=/path/to/run/exclusion
RUNTIME_JSON="$RUN_ROOT/runtime.json"

sbatch --export=ALL,REPOSITORY="$PWD",PYTHON=/path/to/python,CONFIG=/path/to/exclusion.yaml,OUT_ROOT="$RUN_ROOT",RUN_NAME=abiss_exclusion_frozen,RUNTIME_JSON="$RUNTIME_JSON" \
  --output="$RUN_ROOT/prepare_%j.out" \
  tutorials/neuron_j0126/reproduction/slurm/03_prepare_full.sbatch
```

After that job succeeds, submit the dependency graph:

```bash
RUNTIME_JSON="$RUNTIME_JSON" \
RUN_ROOT="$RUN_ROOT" \
PYTHON=/path/to/python \
ABISS_LIBRARY_PREFIX=/path/to/conda/env \
SKELETONS=/path/to/test_50_skeletons.h5 \
JOB_PREFIX=zf-exclusion \
PARTITION=medium \
bash tutorials/neuron_j0126/reproduction/submit_full.sh
```

The wrapper submits watershed and mean-edge arrays with `afterok`
dependencies, followed by a short evaluation job. The nucleus config inserts a
single high-memory nucleus-competition job on `long` between watershed remap
and agglomeration. `submission.tsv`, `final_decode_jobid.txt`, and
`evaluation_jobid.txt` record the complete graph. To serialize the second run,
set `INIT_DEP` to the first run's final decode job ID.

## 6. Check outputs and compare metrics

The final segmentation is the local precomputed directory named by `SEG_PATH`
inside `runtime.json`. `evaluation.json` reports NERL at strict `mt=0` and
five-node tolerance `mt=5`, plus background-inclusive and foreground-only
skeleton-node VOI.

The frozen targets are:

| Decode | NERL mt=0 | NERL mt=5 | VOI split | VOI merge | VOI sum |
|---|---:|---:|---:|---:|---:|
| Exclusion | 0.267679 | 0.469513 | 2.542 | 0.042 | 2.584 |
| Nucleus-aware | 0.287184 | 0.481614 | 2.542840 | 0.019130 | 2.561970 |

Treat deviations larger than normal floating-point rounding as a reproduction
failure and inspect the manifest, ABISS build ID, input hashes, and node
coverage before tuning anything.

The same evaluator accepts the canonical FFN graph-order node LUT instead of a
volume. Set `NODE_LUT` rather than `SEGMENTATION` when submitting
`05_evaluate.sbatch`. It sorts numeric skeleton IDs exactly as the `em_erl` LUT
generator does. The canonical reference reproduces `NERL mt=0 0.525766`,
`NERL mt=5 0.538003`, and background-inclusive `VOI 1.855795`.

The canonical LUT is a matched native/mip0 sample. This was verified by
regenerating all 500,845 node assignments from the public FFN CloudVolume with
`--mip 0`: the regenerated file was byte-identical to the canonical artifact
(`sha256 744c7ee42bbae97234b386756f0b91453f9839ba5cd4658f740b45a439ada1c2`)
and reproduced the metrics above exactly. The restartable Slurm entry point
[05_ffn_native_lut.sbatch](slurm/05_ffn_native_lut.sbatch) performs that
provenance check. It uses a persistent CloudVolume cache, writes the mip0 LUT
and SHA-256, and evaluates it unchanged with the same public evaluator.

## Restart behavior

Every Slurm edge uses `afterok`, so a failed stage prevents downstream jobs
from running. Inspect its log, fix the cause, and prepare a new output namespace
for a clean replay. Use `--mode resume` only when the existing manifest matches
the exact inputs and ABISS runtime identity; the replay code rejects mismatches.
