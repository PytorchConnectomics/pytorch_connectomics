# Cube segmentation playbooks

`cube_decode.py` owns two sequences. `runtime/volume_pipeline.py` owns completion
checks and local/Slurm execution, and `data/keep_mask.py` owns mask computation.
The scripts are entry points; adding a cube dataset needs no Python.

| `params.pipeline` | Sequence |
|---|---|
| `cube` | mask → smoke → abiss → guard → score |
| `cube_from_scratch` | fetch → em → tissue → keep → train → infer → abiss → ec |

Moritz L4 selects `cube`. j0126 selects `cube_from_scratch`, retaining its published
FFN mask geometry, optional download/training steps, shard sentinels, and existing
commands. The old driver and mask-builder invocation paths remain available.

## Add a cube dataset

Create `tutorials/neuron_<name>/params.yaml`, `2_abiss.yaml`, and dataset documentation.
The example below uses a hypothetical 4096³ volume at 25 nm; replace its paths and
frame with the measured dataset frame. All mask sources must share that frame.

```yaml
# params.yaml
params:
  pipeline: cube
  paths:
    repository: /path/to/pytorch_connectomics
    dataset_root: /path/to/neuron_example
    output_root: ${params.paths.repository}/outputs/neuron_example
  data:
    # Already in ABISS's precomputed affinity convention: channels x,y,z and
    # edges v -> v-1. Converting another convention is a separate data step.
    affinity_precomputed: ${params.paths.dataset_root}/affinity
    gt_skeletons: ${params.paths.dataset_root}/manual-neuron-reconstructions
    keep_mask: ${params.paths.dataset_root}/keep_mask.h5
    masks:
      strategy: downsampled_sources
      ratio_zyx: [4, 8, 8]
      border_start_zyx: [0, 0, 0]
      sources:
        - path: ${params.paths.dataset_root}/vessels.h5
          dataset: main
          polarity: exclude
        - path: ${params.paths.dataset_root}/tissue.h5
          dataset: main
          polarity: keep
  frame:
    volume_shape_zyx: [4096, 4096, 4096]
    volume_origin_global_zyx: [0, 0, 0]
    resolution_xyz_nm: [25, 25, 25]
  cluster:
    partition: long
    mask: {cpus: 4, memory: 32G, time: "01:00:00"}
    smoke: {cpus: 8, memory: 64G, time: "04:00:00"}
    abiss: {cpus: 16, memory: 128G, time: "24:00:00"}
    score: {cpus: 8, memory: 64G, time: "02:00:00"}
```

The masks are ANDed: a `keep` source includes its nonzero cells, an `exclude`
source removes its nonzero cells. The output is cropped to `ceil(shape / ratio)`.
The border drops every cell before `ceil(border_start / ratio)` on each axis.
This is conservative when the border is between mask cells. Cluster resources
must fit the chosen logical chunks and the concurrency; the example is not a
memory estimate for another dataset.

In the following `2_abiss.yaml`, replace every `<...>` placeholder with a
dataset-specific measurement or fit. They intentionally have no shared value.

```yaml
_base_:
  - ../_base/abiss.yaml
  - params.yaml

abiss_chunk:
  # Storage chunks: affinity must match its precomputed info. Segmentation
  # storage boundaries must align with internal logical CHUNK_SIZE boundaries.
  copy_block_shape_xyz: [128, 128, 32]
  aff_chunk_size_xyz: [128, 128, 32]
  seg_chunk_size_xyz: [128, 128, 32]
  param:
    NAME: neuron_example_abiss
    BBOX: [0, 0, 0, 4096, 4096, 4096]
    CHUNK_SIZE: [<fitted X>, <fitted Y>, <fitted Z>]
    WS_HIGH_THRESHOLD: <dataset fit>
    WS_LOW_THRESHOLD: <dataset fit>
    AGG_THRESHOLD: <dataset fit>
    WS_SIZE_THRESHOLD: <dataset choice>
    WS_DUST_THRESHOLD: <dataset choice>
```

`WS_HIGH_THRESHOLD`, `WS_LOW_THRESHOLD`, `AGG_THRESHOLD`, and `CHUNK_SIZE` are
required dataset values. The playbook checks them before ABISS can supply its
own fallback values. An absent key and an explicit null both fail, with separate
diagnostics. The shared base supplies neither fitted values nor null placeholders.
Do not copy a threshold from a small validation cube and assume it transfers to
the full volume; record how it was selected in the dataset documentation.

The smoke step uses `2_abiss_smoke.yaml` when the tutorial already supplies one.
Otherwise it generates `output_root/abiss_smoke/config.yaml` from `2_abiss.yaml`,
capping the extent at 1024 × 1024 × 1792 XYZ voxels while retaining the dataset's
logical chunk size and thresholds. Generation happens during execution, so the
new tutorial still has only two input YAMLs. Review the pre-flight extent against
the documented frame and mask coverage before running the whole volume.

The guard and score steps come with `cube`. Scoring expects `.nml` reconstructions
with integer global voxel coordinates, and receives the supplied shape, origin,
and full-precision resolution. It uses the existing
`scripts/score_moritz_l4_nerl.py` entry point with explicit frame arguments for new
datasets. This sequence does not provide a reader for other skeleton formats.

```bash
python scripts/run_playbook.py --params tutorials/neuron_example/params.yaml --check
python scripts/run_playbook.py --params tutorials/neuron_example/params.yaml --dry-run
python scripts/run_playbook.py --params tutorials/neuron_example/params.yaml
python scripts/run_playbook.py --params tutorials/neuron_example/params.yaml --local
python scripts/build_keep_mask.py --params tutorials/neuron_example/params.yaml
conda run -n pytc python scripts/validate_tutorial_configs.py \
  --glob 'tutorials/neuron_example/*.yaml'
```

`--steps` selects steps for resuming or debugging; the YAML sequence itself cannot
be reordered or shortened. `--force` reruns a completed step. Local execution
checks step inputs immediately; Slurm chains submitted jobs with `afterok` so
earlier jobs can create later inputs.
