# Code v0

## Overview

Implemented the approved shared cube playbook, importable execution engine, and both keep-mask strategies. The four existing driver/mask invocation paths are now 12-line wrappers. A third decode dataset supplies two YAML files and documentation, with no new Python.

All work remains uncommitted in the working tree. Step 0 fixtures were captured and saved before either driver, any other production file, or any tutorial configuration was modified. The instruction to commit fixtures was interpreted as saving fixture files, subordinate to the explicit prohibition on git commits.

## What Changed

- Added one engine for completion checks, resource resolution, local/Slurm execution, arrays, dependency chaining, preparation, input gating, and the CLI.
- Added canonical `cube` and `cube_from_scratch` sequences and a builder registry. YAML selects a sequence name; it cannot replace its ordering or remove stages.
- Moved both mask implementations into `connectomics/data/keep_mask.py`. The j0126 function sources remain verbatim, including its original CLI main, initialization, and shard sentinels.
- Added dataset-local fitted-value enforcement, generic mask configuration, a shared ABISS base without fitted values, and onboarding documentation.
- Added scorer frame arguments so a config-only dataset uses its own shape, origin, and resolution. The scorer invocation path, Moritz defaults, and numerical formulas are preserved.

## Implementation Details

`volume_pipeline.py` owns `Status`, `Step`, completion predicates, both existing resource schemas, and execution. Playbook construction validates all four fitted keys before any ABISS preparation. Absent and null values produce distinct diagnostics naming every offending key. Raw YAML ownership also rejects fitted values inherited from outside the dataset directory; a dataset-local smoke YAML may inherit its own full-volume recipe. `abiss_chunk.prepare_config` is unchanged.

`cube_decode.py` owns both sequences and their step contents. Existing Moritz commands retain their exact wrapper and relative-path spelling; other cube tutorials use the generic mask CLI and explicit scoring frame arguments. Existing smoke YAMLs remain in use. When a third tutorial supplies only `params.yaml` and `2_abiss.yaml`, a preparation callback writes a derived smoke config under its output directory. The derived extent preserves fitted thresholds and logical chunk size, isolates output paths, and leaves existing precomputed affinity shared. For HDF5 affinity input, the smoke conversion gets a separate output affinity path. Storage completion counts use the same storage-chunk fallback as the ABISS executor.

`KeepMaskSpec` supports downsampled keep/exclude sources, ratio, border, cropping, and frame metadata. Its FFN adapter calls the unchanged relocated CLI body and restores `sys.argv` in `finally`. Original j0126 geometry, threshold defaults, source channels, and numeric operations are unchanged. The new dataset route uses the configurable downsampled-source strategy.

`params.pipeline` does not collide with structured pipeline profiles: tutorial params are resolved and removed before schema validation. Existing ABISS tutorial YAMLs and every existing fitted value are unchanged. No dependencies, branches, commits, or CCC sentinels were added.

## Files Changed

| File | Purpose |
|---|---|
| `connectomics/runtime/volume_pipeline.py` | Shared engine, completion/resource helpers, fitted-value validation, CLI |
| `connectomics/playbooks/__init__.py` | Playbook package |
| `connectomics/playbooks/cube_decode.py` | Canonical sequences, builders, generic smoke preparation |
| `connectomics/playbooks/README.md` | Complete two-YAML onboarding example and execution contract |
| `connectomics/data/keep_mask.py` | Shared mask specification, both strategies, relocated FFN CLI |
| `scripts/run_playbook.py` | Generic playbook entry point |
| `scripts/build_keep_mask.py` | Generic configured-mask entry point |
| `scripts/run_j0126.py` | Thin wrapper preserving the existing path and job prefix |
| `scripts/run_moritz_l4.py` | Thin wrapper preserving the existing path and job prefix |
| `scripts/build_j0126_keep_mask.py` | Thin wrapper preserving all existing FFN flags |
| `scripts/build_moritz_l4_keep_mask.py` | Thin wrapper preserving `--bv` and `--out` |
| `scripts/score_moritz_l4_nerl.py` | Configurable frame arguments and checkout-relative imports; invocation path retained |
| `tutorials/_base/abiss.yaml` | Shared structural/path values, with all four fitted keys absent |
| `tutorials/neuron_j0126/params.yaml` | Sequence, mask strategy, unchanged FFN frame |
| `tutorials/neuron_moritz_l4/params.yaml` | Sequence and mask-source declaration |
| `tests/unit/test_volume_pipeline.py` | 39 engine, resource, CLI, and fitted-value cases |
| `tests/unit/test_cube_playbook.py` | 17 baseline, onboarding, frame, smoke, and completion cases |
| `tests/unit/test_keep_mask.py` | 10 numeric, movement, initialization, shard, and CLI cases |
| `tests/fixtures/playbook_baseline/j0126.json` | Pre-migration serialized steps in three filesystem states |
| `tests/fixtures/playbook_baseline/moritz_l4.json` | Pre-migration serialized steps in three filesystem states |
| `CLAUDE.md` | One Agent Quick Reference row |
| `.agent/refactor/playbooks/artifacts/code_v0.md` | This code-stage artifact |

## Git Baseline

run_start_ref: a692d2e37a5eeffb1eee0590613d6a35f5cfc412
current_head: a692d2e37a5eeffb1eee0590613d6a35f5cfc412

The initial worktree contained 26 modified tracked files plus unrelated untracked work. SHA-256 comparison confirmed that every initially modified tracked file remained byte-identical to its task-start state. Existing unrelated changes remain user-owned. Nothing was staged or committed.

## Verification

The conda executable used was `/projects/weilab/weidf/lib/miniconda3/bin/conda`; commands below abbreviate this executable as `conda`.

0. **Pre-edit capture: passed.** Ran `conda run -n pytc python /tmp/playbook_capture.py` against the untouched drivers and wrote both JSON fixtures before production/config edits. The same synthetic scenario and serializer are retained in `test_cube_playbook.py`. Config loaders were substituted; completion predicates used real temporary files. The Moritz main storage count was reduced to 2 to match the synthetic BBOX; its 3584-chunk smoke count was unchanged. All specified fields, including `prepare_fn_present`, were captured for 39 step records. Immediately after capture, `git diff --quiet -- scripts/run_j0126.py scripts/run_moritz_l4.py` passed. Fixture hashes remained unchanged through implementation:

   - j0126: `eeec839df9a63848d9b82d5ec06a06da6b5caf68fdbf26a02f1860be8a9f8908`
   - Moritz: `956226a2dced76e1ba28b6b45732935292a538d45b66fc59d0960390ed6fe967`

1. **Binding baseline gate: passed.** All six dataset/state comparisons match the captured JSON exactly. This includes commands, resources, inputs, arrays, preparation, skip reasons, and status details/completion.

2. **Engine: passed, 39 tests.** Covers pending/complete/disabled/forced execution, canonical selection, local and Slurm arrays, first/subsequent dependency IDs, preparation ordering and callback counts, local input blocking versus Slurm dependencies, exact resource flag lists, and zero subprocess calls in dry-run. Direct `conda run -n pytc pytest -q tests/unit/test_volume_pipeline.py` passed.

3. **Fitted enforcement: passed.** Both sequences reject absent/null keys and accept complete configurations. Engine tests additionally reject shared/cross-dataset fitted inheritance and accept dataset-local smoke inheritance.

4. **Masks: passed with the initialization limitation below.** Ten unit cases cover numeric source combination/crop/border behavior, exact source movement, initialization and sentinels, and old/new CLIs. The movement test compares all six original function source segments byte-for-byte against `git show a692d2e3:scripts/build_j0126_keep_mask.py`, and checks every original module constant. Tissue initialization/sharding used explicit Zarr format 2 and a stub CloudVolume; keep-stage initialization/sharding used real local arrays.

   Ran `conda run -n pytc python /tmp/verify_moritz_keep_mask.py` on the actual Moritz vessel input and existing keep-mask reference. The generated `/tmp/playbook-moritz-keep-mask.h5` matches every voxel of shape `(827, 1067, 700)`, dtype, and all seven attributes. Vessel: 1.9945%; border: 3.0529%; keep: 94.95355604%. This is bitwise mask-data equality; entire HDF5 container-byte equality was not asserted.

5. **Config-only onboarding: passed.** A synthetic third tutorial contains exactly two input YAMLs, resolves all five `CUBE` stages, references its supplied data/output paths, carries explicit frame flags, preserves its fitted values in generated smoke config, and produces no files/subprocess calls under dry-run. Its test runs the real validator from the temporary directory: **1 canonical config validated, 1 custom YAML skipped**. Scorer tests check coordinates, physical edge lengths, default frame behavior, and out-of-bounds filtering. A separate numerical harness also matched the old scorer's complete graph arrays with unchanged defaults.

6. **ABISS regressions: passed, 48 tests.** Direct execution of `conda run -n pytc pytest -q tests/unit/test_abiss_chunk_executor.py tests/unit/test_abiss_nucleus_competition.py` passed.

   **Final combined execution: 114 passed, 3 SWIG deprecation warnings.** Exact command:

   ```bash
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/pytc-mpl \
     /projects/weilab/weidf/lib/miniconda3/bin/conda run --no-capture-output -n pytc \
     python /tmp/playbook_pytest_localio.py -q \
     tests/unit/test_volume_pipeline.py tests/unit/test_cube_playbook.py \
     tests/unit/test_keep_mask.py tests/unit/test_abiss_chunk_executor.py \
     tests/unit/test_abiss_nucleus_competition.py
   ```

   The temporary launcher is an environment workaround: this sandbox denies asyncio's local socketpair wakeup, causing ordinary Zarr calls to hang. Before starting each new asyncio loop, it schedules a recurring 10 ms timer, then runs pytest unchanged. It replaces no Zarr operation, filesystem data, completion predicate, fixture, or test assertion. Direct baseline runs were interrupted at that hang. An earlier combined attempt passed 113 tests but its validator child hit the shared host's thread limit; the displayed single-threaded command resolved that environmental failure. These failures are not claimed as passing runs.

   **Existing tutorial validation: passed for the targeted command.** `conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_j0126/*.yaml' --glob 'tutorials/neuron_moritz_l4/*.yaml'` reported **24 canonical configs validated, 7 custom YAMLs skipped** (the validator also includes its default top-level glob).

   **Full tutorial validation: pre-existing failure reproduced before and after.** Both runs of `conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'` fail because `tutorials/neuron_nisb/base_banis_v4_erosion2.yaml` references missing `base_banis_v3_erosion2.yaml`. No full-tree success is claimed. Independent raw-root classification is **98 canonical/8 custom before, 98/9 after**: the new shared base adds one custom YAML. Literally unchanged total counts are incompatible with adding that file; the validator was not modified to hide it.

7. **Advisory dry runs: passed, exact output equality for both drivers.** Ran each driver's `--dry-run` before and after migration; both exited 0 and complete captured outputs matched byte-for-byte. Evidence is in `/tmp/playbook-before/{j0126,moritz_l4}.txt` and `/tmp/playbook-after/{j0126,moritz_l4}.txt`.

8. **Scope and source checks: passed with pre-existing scorer style/type debt.** `git diff --check` passed. `git status --short`, initial dirty-file hashes, and explicit diffs confirmed no task edits to out-of-scope files or existing unrelated work. No vendored ABISS source was edited.

   Ran changed-file `black --check`, `isort --check-only`, `flake8 --max-line-length=100`, and `mypy --config-file .github/mypy_changed.ini`. Black and mypy pass for the 13 new/moved/thin-wrapper Python files; isort and flake8 pass for all 14 changed Python files including the scorer. Black and mypy on the scorer still fail on existing formatting and two missing annotations (`node_coords`, `out`); running the same checks on its `git show` baseline reproduced those failures. Unrelated scorer reformatting/type cleanup was not performed. Black also prints the repository's Python 3.11 versus configured Python 3.12 target warning.

## Review Focus

- Review the captured fixtures and their retained synthetic harness first; neither fixture was regenerated after driver edits.
- Confirm fitted-value provenance checks run on both full and smoke ABISS configurations before execution, and that the shared base contains no fitted values.
- Inspect generic smoke output isolation and storage completion geometry, including the conditional HDF5 conversion path.
- Verify the FFN source-movement gate and the narrow scorer frame extension. The scorer content change is needed for config-only onboarding; its invocation path remains unchanged as required.
- Compare task-owned changes separately from the 26 pre-existing modified tracked files and unrelated untracked work.

## Risks and Unknowns

- Full-volume execution, remote FFN downloads, real Slurm submission/dependency execution, and end-to-end guard/scoring were not run. CloudVolume behavior in FFN shard tests is stubbed; exact source movement cannot prove remote service behavior or eliminate import-state risk.
- The unchanged FFN tissue initializer fails under this environment's default Zarr 3 format: its original `compressor=` API requires format 2. This failure was explicitly reproduced. Format-2 initialization and shard completion passed; changing the initializer would violate the approved pure-movement constraint.
- The synchronous FFN adapter temporarily changes `sys.argv`; callers must not invoke it concurrently from threads.
- Existing guard-script inspection found `Path(a.out)` without a `Path` import and no enforcement of the title's 5% cutoff. Those pre-existing issues were left untouched, so this refactor does not establish that a real complete pipeline run succeeds.
- The existing missing NISB base, scorer style/type debt, and socket/thread restrictions described above limit clean global verification. None were silently fixed or hidden.
- Both existing resource schemas persist. j0126's FFN geometry and from-scratch behavior are retained; new decode datasets use the configurable `cube` sequence.
- The ignored frozen Moritz YAML under `dev/` remains outside this change.

## Changes Since Previous Code Version

Initial implementation.
