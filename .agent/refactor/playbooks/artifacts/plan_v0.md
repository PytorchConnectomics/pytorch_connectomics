# Plan v0

## Summary

Extract the orchestration engine out of the two per-dataset drivers into
`connectomics/runtime/volume_pipeline.py`, express each pipeline as one declarative
playbook under a new top-level `playbooks/`, and reduce `scripts/run_*.py` to thin CLIs
matching the repository's existing `run_abiss_chunk.py` pattern. After this, a third
100 um cube dataset is added with `tutorials/neuron_<name>/` config only.

The engine is the part that is provably duplicated: `Status` and `check_paths` are
byte-identical across the two drivers, and `run_slurm`/`parse_args` differ only by a
job-name prefix and an `--array` flag. The mask *builders* are NOT duplicated in the same
way and this plan deliberately leaves them alone -- see Scope.

## Scope

In scope:

1. `connectomics/runtime/volume_pipeline.py` -- engine: `Status`, `Step`, completion
   checks, resource resolution, local/slurm launching with `afterok` chaining, and the
   shared CLI (`--steps`, `--force`, `--check`, `--dry-run`, `--local`).
2. `playbooks/cube_decode.py` -- the 100 um cube playbook (mask -> smoke -> abiss ->
   guard -> score), shared by all three cube datasets.
3. `playbooks/j0126_full.py` -- j0126's existing pipeline (fetch/em/tissue/keep/train/
   infer/abiss/ec) re-expressed on the same engine, so its 480 lines stop being bespoke.
4. `scripts/run_playbook.py` -- thin CLI. `scripts/run_j0126.py` and
   `scripts/run_moritz_l4.py` become thin wrappers that delegate, preserving their
   documented commands.
5. `tutorials/_base/abiss.yaml` -- the invariant ABISS block with every FITTED value set
   to `null`, plus a validator in the playbook that fails loudly listing any null.
6. `tests/unit/test_volume_pipeline.py` -- engine unit tests.
7. One row in `CLAUDE.md`'s Agent Quick Reference for `playbooks/`, which that file calls
   the single source of truth for agent navigation.
8. `playbooks/README.md` -- what a playbook is and what may not go in one.

Explicitly OUT of scope, with reasons:

- **Unifying the two keep-mask builders.** They look like duplicates and are not.
  `build_j0126_keep_mask.py` is a two-stage, shard-resumable build that streams a remote
  CloudVolume `tissue_classification` layer at 18x18x20 nm and writes a full-resolution
  zarr; `build_moritz_l4_keep_mask.py` is a single in-memory pass over a 4.9 MB
  downsampled h5 that writes a downsampled h5. Forcing one implementation would rewrite
  j0126's sharding and resume behaviour, which the task puts out of scope. The shared
  contract is the *step* (`mask` produces the keep-mask artifact), not the builder.
- Any change to a fitted value or numeric behaviour of the two existing tutorials.
- `lib/` (separate git repository, gitignored here).
- `run_abiss_volume.py`, `run_seuron_provenance.py`.
- Moving `scripts/run_abiss_chunk.py`, `scripts/seg_percolation_guard.py`,
  `scripts/score_moritz_l4_nerl.py`, `scripts/build_*_keep_mask.py`. These are invoked by
  path from sbatch wrappers outside this repository; their paths are a public contract.

## Proposed Changes

### 1. Engine (`connectomics/runtime/volume_pipeline.py`)

Lift from the two drivers, taking the union of their features so neither regresses:

```
@dataclass Status(done: bool, detail: str)
@dataclass Step(name, title, command, status, resources="", inputs=[], array=0,
                prepare="", prepare_fn=None, skip="")
```

`Step` must keep j0126's `array`, `prepare`, `prepare_fn`, `inputs` and `skip` fields --
they are load-bearing there (`array=int(dl.num_shards)` for download, `array=params.mask.num_shards`
for the mask, `skip=download_off` when a step is disabled in params). Dropping them is the
most likely way to silently break j0126.

Shared completion checks: `check_paths`, `check_shards`, `check_layer`, `check_download`,
`check_checkpoint`, `check_affinity`, `input_exists`. These are already generic over paths
and counts; they move unchanged.

`sbatch_resources(params, block, gpus=0)` reads a resource block. It must accept BOTH
existing shapes: j0126's flat `params.<step>.{slurm_partition,cpus,memory,time}` and
moritz's nested `params.cluster.<step>.{partition,cpus,memory,time}` with a
`params.cluster.partition` fallback. Normalising the two params schemas is a config
migration, not an orchestration change, and doing it here would change j0126's resolved
sbatch flags -- out of scope.

`main(playbook, tutorial_dir, argv)` holds the CLI and the run loop.

### 2. Playbooks (`playbooks/`)

A playbook is one module exposing:

```python
ORDER: tuple[str, ...]
def build_steps(params, tutorial: Path) -> list[Step]
```

No dataset constants, no thresholds, no paths. Everything comes from `params`.
`playbooks/cube_decode.py` is `build_steps` lifted from `run_moritz_l4.py`;
`playbooks/j0126_full.py` is `build_steps` lifted from `run_j0126.py`.

### 3. Selection

`tutorials/neuron_<name>/params.yaml` gains one key:

```yaml
params:
  playbook: cube_decode
```

so the invocation is uniform and dataset #3 adds no Python:

```bash
python scripts/run_playbook.py --tutorial tutorials/neuron_<name>
```

`scripts/run_j0126.py` and `scripts/run_moritz_l4.py` remain as thin wrappers that call
the engine with their tutorial pinned, so every command already documented in the two
READMEs keeps working.

### 4. Fitted values cannot be inherited

`tutorials/_base/abiss.yaml` carries the invariant ABISS block and sets
`WS_HIGH_THRESHOLD`, `WS_LOW_THRESHOLD`, `AGG_THRESHOLD` and `CHUNK_SIZE` to `null`, with
the reasoning as comments. `cube_decode.build_steps` validates the resolved abiss config
and raises listing every null key.

The check belongs in the playbook, NOT in `abiss_chunk.prepare_config`, which currently
does `payload.setdefault("WS_HIGH_THRESHOLD", 0.9)`. Changing that setdefault would alter
behaviour for every other config in the repository that relies on it.

Neither existing tutorial is required to adopt `_base_` in this change; both already state
all four values explicitly. Adopting it is a follow-up.

## Files and Areas

| File | Change |
|---|---|
| `connectomics/runtime/volume_pipeline.py` | new -- engine |
| `playbooks/__init__.py` | new |
| `playbooks/cube_decode.py` | new -- the shared 100 um cube playbook |
| `playbooks/j0126_full.py` | new -- j0126 steps on the engine |
| `playbooks/README.md` | new -- the contract |
| `scripts/run_playbook.py` | new -- thin CLI (~12 lines) |
| `scripts/run_moritz_l4.py` | shrink to a wrapper |
| `scripts/run_j0126.py` | shrink to a wrapper |
| `tutorials/_base/abiss.yaml` | new -- invariants; fitted values `null` |
| `tutorials/neuron_moritz_l4/params.yaml` | add `playbook: cube_decode` |
| `tutorials/neuron_j0126/params.yaml` | add `playbook: j0126_full` |
| `tests/unit/test_volume_pipeline.py` | new |
| `CLAUDE.md` | one Agent Quick Reference row |

Untouched: every `scripts/build_*_keep_mask.py`, `seg_percolation_guard.py`,
`score_moritz_l4_nerl.py`, `run_abiss_chunk.py`, all `2_abiss*.yaml`, `lib/`.

## Verification Plan

1. **Command-level equivalence, the primary gate.** Before changing anything, capture the
   planned commands of both drivers:

   ```bash
   python scripts/run_j0126.py     --dry-run | grep '^  \$' > /tmp/j0126.before
   python scripts/run_moritz_l4.py --dry-run | grep '^  \$' > /tmp/moritz.before
   ```

   After the refactor the same commands must produce byte-identical output. Compare only
   the `  $` command lines: the surrounding `[done]/[todo]` status lines embed live
   filesystem state (chunk counts, marker counts) that changes between runs and is not a
   refactor regression.

2. `python scripts/run_moritz_l4.py --check` and `python scripts/run_j0126.py --check`
   both exit 0 and report every step.

3. `python scripts/run_playbook.py --tutorial tutorials/neuron_moritz_l4 --dry-run`
   produces the same command lines as (1).

4. `pytest tests/unit/test_volume_pipeline.py` -- new tests covering: step ordering;
   `--steps` subsetting; `--force`; a `skip=` step is skipped; an `array=N` step emits
   `--array=0-{N-1}` and the shard-id suffix; `afterok` chaining passes the previous job
   id; `--dry-run` performs no subprocess call; the null-fitted-value validator raises and
   names every missing key.

5. `pytest tests/unit/test_abiss_chunk_executor.py tests/unit/test_abiss_nucleus_competition.py`
   still pass (19 + 29 currently).

6. `python scripts/validate_tutorial_configs.py` still reports the same counts.

7. `git status --short` shows no modification to any file listed as untouched above.

## Risks and Questions

1. **j0126 regression is the main risk.** Its driver has five features moritz's lacks
   (`array`, `prepare`, `prepare_fn`, `inputs`, `skip`) and no test coverage at all today.
   Verification step 1 is the only thing standing between this refactor and a silent
   change to a pipeline that costs GPU-days. If byte-identical command output cannot be
   achieved, stop and report rather than accepting a diff.
2. **Two params schemas.** j0126 uses flat per-step resource blocks; moritz uses nested
   `cluster:`. The engine supports both rather than migrating either. This is deliberate
   debt: it keeps the diff reviewable and defers a config migration that would change
   j0126's resolved flags.
3. **A new top-level directory.** `playbooks/` only pays off if both fat drivers migrate;
   a half-migration leaves three locations instead of two. Both are in scope here.
   `workflows/` is rejected: the word already denotes `2_abiss.yaml`-style configs in
   `validate_tutorial_configs.py` and the EC stage graph in
   `connectomics/decoding/error_correction/workflow.py`.
4. **Untracked `dev/`.** `dev/` is gitignored, so the 14 sweep configs and the frozen
   whole-volume resume config `dev/moritz_l4/abiss_wholevol_frozen_c1x1x104.yaml` are not
   under version control. Not caused by this change and not fixed by it, but that frozen
   config is load-bearing for resuming an interrupted whole-volume decode.
5. **Open question for the reviewer:** should `playbooks/` live at the repository root, or
   as `connectomics/playbooks/`? Root matches `tutorials/` and keeps playbooks visibly
   non-library; `connectomics/playbooks/` makes them importable without a path insert and
   automatically in scope for the packaging config. This plan assumes root.

## Changes Since Previous Plan Version

Initial plan.
