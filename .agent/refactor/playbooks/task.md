# Task

Move the reusable parts of the 100 um cube segmentation playbook into a shared location so
that adding a new cube dataset requires supplying data paths only -- no new code.

## Context

Three 100 um cube datasets will run the same playbook. Two instances exist today and
already duplicate each other:

| | `neuron_j0126` | `neuron_moritz_l4` |
|---|---:|---:|
| driver | `scripts/run_j0126.py`, 480 lines | `scripts/run_moritz_l4.py`, 235 lines |
| keep-mask builder | `scripts/build_j0126_keep_mask.py`, 233 | `scripts/build_moritz_l4_keep_mask.py`, 92 |

`Status` and `check_paths` are byte-identical between the two drivers; `run_slurm` and
`parse_args` differ only by a job-name prefix and an `--array` flag. The keep-mask builders
perform the same operation (combine downsampled masks with a polarity at ratio R, add a
border box, crop to the volume) from different inputs.

`scripts/` holds 56 files, of which 6 are multi-step orchestrators and the rest are
single-purpose tools. The repository already demonstrates the intended split twice, and the
correlation with test coverage is exact:

| driver | lines | logic lives in | tested |
|---|---:|---|---|
| `scripts/run_abiss_chunk.py` | 12 | `connectomics/runtime/abiss_chunk.py` | yes |
| `scripts/run_error_correction.py` | 12 | `connectomics/decoding/error_correction/` | yes |
| `scripts/run_moritz_l4.py` | 235 | itself | no |
| `scripts/run_j0126.py` | 480 | itself | no |
| `scripts/run_abiss_volume.py` | 606 | itself | no |
| `scripts/run_seuron_provenance.py` | 887 | itself | no |

Logic that lives in `scripts/` does not get tested. Any new home must not repeat that.

## Naming constraint

Do NOT introduce a top-level `workflows/` directory. "Workflow" already has two meanings in
this repository: `scripts/validate_tutorial_configs.py` reports "skipped 5 custom workflow
YAMLs" about `tutorials/neuron_moritz_l4/2_abiss.yaml` itself, `connectomics/utils/yaml_config.py`
documents "standalone workflow scripts", and `connectomics/decoding/error_correction/workflow.py`
is the error-correction stage graph. Use `playbooks/` for the per-playbook definitions.

## Required outcome

Adding cube dataset #3 must require only:

1. `tutorials/neuron_<name>/params.yaml` -- paths, frame (shape, origin, resolution), mask
   paths and ratio, cluster resources.
2. `tutorials/neuron_<name>/2_abiss.yaml` -- extent, chunk grid, and the fitted thresholds.
3. Dataset documentation.

No new Python. The playbook (mask -> pre-flight -> abiss -> guard -> score) is ONE artifact
shared by all three datasets, not a copy per dataset.

## Hard constraint on shared defaults

A shared base may declare a fitted parameter and explain it. It must NOT supply a value for
one. `WS_HIGH 0.5` was fitted on one dataset's 100^3 val cubes, looked like a sane default,
and cost a week: it makes a single watershed object own ~70% of every chunk, which the
agglomeration then stitches into a segment holding 67.5% of the finished volume. Fitted
values (`WS_HIGH_THRESHOLD`, `WS_LOW_THRESHOLD`, `AGG_THRESHOLD`, `CHUNK_SIZE`) must be
required-per-dataset, ideally so that omitting one fails loudly rather than inheriting.

## Out of scope

- Changing any fitted value, threshold, or numerical behaviour of the existing two tutorials.
- The vendored ABISS tree under `lib/` (a separate git repository, gitignored here).
- `run_abiss_volume.py` and `run_seuron_provenance.py` may stay where they are; migrating
  them is optional and must not be a prerequisite.

## Definition of done

- `scripts/run_j0126.py` and `scripts/run_moritz_l4.py` no longer carry orchestration logic.
- Both existing tutorials still resolve and still produce the same planned commands.
- The engine is importable and unit-tested.
- A new dataset needs config only.
