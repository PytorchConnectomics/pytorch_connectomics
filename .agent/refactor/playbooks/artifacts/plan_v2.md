# Plan v2

## Summary

One engine, one canonical cube sequence defined in code, one importable mask package with
two strategies, and a mandatory filesystem-independent regression gate captured from the
current drivers before they are touched. `scripts/` keeps only thin CLIs.

`plan_v1` claimed "none are deferred" while its Risk 3 offered to abandon the shared mask
spec. That claim was false and is withdrawn. This version removes the deferral rather than
restating it.

## Scope

In scope:

1. `connectomics/runtime/volume_pipeline.py` -- engine: `Status`, `Step`, completion
   checks, resource resolution, local/slurm launching, CLI, `require_dataset_values`.
2. `connectomics/playbooks/cube_decode.py` -- step-builder registry **and** the canonical
   sequences.
3. `connectomics/data/keep_mask.py` -- importable mask computation, two strategies.
4. `scripts/run_playbook.py`, `scripts/build_keep_mask.py` -- thin CLIs;
   `scripts/run_j0126.py`, `scripts/run_moritz_l4.py`, `scripts/build_j0126_keep_mask.py`
   reduced to thin wrappers **at their existing paths**.
5. `tutorials/_base/abiss.yaml`; `pipeline:` key in both tutorials' `params.yaml`.
6. `tests/unit/test_volume_pipeline.py`, `tests/unit/test_cube_playbook.py`,
   `tests/unit/test_keep_mask.py`, plus committed step-baseline fixtures.
7. `CLAUDE.md` Agent Quick Reference row.

Out of scope: numeric behaviour of either tutorial; `lib/`; `run_abiss_volume.py`;
`run_seuron_provenance.py`; the invocation PATHS of `run_abiss_chunk.py`,
`seg_percolation_guard.py`, `score_moritz_l4_nerl.py`, `build_j0126_keep_mask.py`.

## Proposed Changes

### 1. The canonical sequence lives in code, not in each tutorial (finding 1)

```python
# connectomics/playbooks/cube_decode.py
CUBE            = ("mask", "smoke", "abiss", "guard", "score")
CUBE_FROM_SCRATCH = ("fetch", "em", "tissue", "keep", "train", "infer", "abiss", "ec")
SEQUENCES = {"cube": CUBE, "cube_from_scratch": CUBE_FROM_SCRATCH}
```

`params.yaml` selects a sequence by name and may not reorder or drop from it:

```yaml
params:
  pipeline: cube                 # moritz, and dataset #3
  pipeline: cube_from_scratch    # j0126, its exact current order
```

Dataset #3 therefore cannot omit `guard` or `score`; they come with the name. j0126's
emitted commands are unchanged because `cube_from_scratch` is its present order verbatim.
An unknown name raises and lists the registry. Per-step enablement stays where it already
is -- j0126's `download.enabled` / `train.enabled` drive `Step.skip`, not the sequence.

### 2. Mask computation becomes an importable package with both strategies (finding 2)

`connectomics/data/keep_mask.py`:

```python
@dataclass KeepMaskSpec:
    strategy: str                    # "downsampled_sources" | "ffn_tissue_border"
    out: Path
    volume_shape_zyx, ratio_zyx, border_start_zyx
    sources: list[{path, dataset, polarity}]     # polarity: keep | exclude
def build(spec, *, stage=None, shard_id=0, num_shards=1, init=False) -> None
```

Both existing behaviours are exposed through that one interface:

- `downsampled_sources` -- the Moritz family. Combine N downsampled sources by polarity at
  `ratio_zyx`, AND a border box, crop to `volume_shape_zyx`.
- `ffn_tissue_border` -- j0126's existing two-stage, shard-resumable CloudVolume-to-zarr
  build, **relocated verbatim**. Its `--stage tissue|keep`, `--init`, `--shard-id`,
  `--num-shards` semantics and its `<out>.done.<id>` sentinels are preserved, because the
  driver's `check_shards` and its Slurm array depend on them.

`scripts/build_j0126_keep_mask.py` and `scripts/build_keep_mask.py` become thin CLIs over
the package, keeping their paths and flags so external sbatch wrappers are unaffected.

The j0126 relocation is **pure movement**: no logic edit, no reformatting, no renamed
locals. Verification 4 enforces that with a diff-shaped check, because this is the one
place in the plan where a silent numeric change could hide.

### 3. Fitted values cannot be inherited, on every route

Unchanged from `plan_v1`: `require_dataset_values` lives in the engine, runs for any step
resolving an ABISS config, executes **before** `abiss_chunk.prepare_config` can
`setdefault` a value, and treats **absent** and **null** as distinct failures naming every
offending key. `tutorials/_base/abiss.yaml` omits the four keys rather than nulling them.
`abiss_chunk.prepare_config` is not modified.

### 4. Package location

`connectomics/playbooks/` and `connectomics/data/keep_mask.py`, per the reviewer.
`tutorials/` and `scripts/` resolve from the repository root via
`Path(__file__).resolve().parents[2]`, matching `run_abiss_chunk.py`.

## Files and Areas

| File | Change |
|---|---|
| `connectomics/runtime/volume_pipeline.py` | new -- engine |
| `connectomics/playbooks/{__init__,cube_decode}.py`, `README.md` | new -- registry + sequences |
| `connectomics/data/keep_mask.py` | new -- both mask strategies |
| `scripts/run_playbook.py`, `scripts/build_keep_mask.py` | new -- thin CLIs |
| `scripts/run_j0126.py`, `scripts/run_moritz_l4.py`, `scripts/build_j0126_keep_mask.py` | reduced to wrappers, paths kept |
| `tutorials/_base/abiss.yaml` | new |
| `tutorials/neuron_{j0126,moritz_l4}/params.yaml` | add `pipeline:`, `data.masks` |
| `tests/unit/test_{volume_pipeline,cube_playbook,keep_mask}.py` | new |
| `tests/fixtures/playbook_baseline/{j0126,moritz_l4}.json` | new -- pre-refactor step baselines |
| `CLAUDE.md` | one row |

## Verification Plan

**Step 0, before any file is modified** (this is what makes finding 3's gate possible):
run a throwaway harness that imports `build_steps` from today's `scripts/run_j0126.py` and
`scripts/run_moritz_l4.py` against a fixed synthetic params object and a `tmp_path`
filesystem seeded three ways -- all steps pending, all complete, and disabled steps
skipped. Serialize every resulting `Step` as
`{name, title, command, resources, array, prepare, inputs, skip, status.done, status.detail}`
and commit the JSON to `tests/fixtures/playbook_baseline/`.

1. **Migrated builders reproduce the baseline exactly** (mandatory, filesystem-independent).
   The new builders run against the same synthetic params and seeded `tmp_path` and must
   equal the committed JSON field for field, in all three seed states. This is the gate for
   commands, `prepare`, `prepare_fn` presence, `inputs`, `skip`, `array` and completion
   predicates -- everything `plan_v1` left to manual review.
2. **Engine unit tests**: pending emits a command, complete emits none, `skip=` emits none
   and reports why; `array=N` emits `--array=0-{N-1}` plus the shard-id suffix; `afterok`
   chaining passes the previous job id and the first step has none; `prepare` runs before
   launch and `prepare_fn` exactly once and never under `--dry-run`; `inputs` gating blocks
   a local run and not a slurm run; both resource schemas resolve to the exact sbatch flag
   list; `--dry-run` makes zero `subprocess.run` calls.
3. **Fitted-value enforcement**: absent raises; null raises; both name every offending key;
   complete passes. Run for both sequences.
4. **Mask package**: (a) `downsampled_sources` on the existing Moritz inputs reproduces the
   committed `keep_mask_z4y8x8.h5` bit for bit (2.0% vessel, 3.05% border, 94.95% keep);
   (b) the `ffn_tissue_border` relocation is verified as movement, by diffing the moved
   function bodies against `git show a692d2e3:scripts/build_j0126_keep_mask.py` and
   requiring no change outside imports and the module docstring; (c) `--init` still creates
   the array and a shard still writes `<out>.done.<id>`.
5. **Config-only onboarding**, strengthened per finding 4: build a synthetic third tutorial
   in `tmp_path` from `params.yaml` + `2_abiss.yaml` alone with `pipeline: cube`; resolve
   the **complete** `CUBE` sequence; assert every emitted command references that
   tutorial's supplied paths and none of Moritz's or j0126's; and run the real
   `validate_tutorial_configs.py` over both of its YAML files.
6. `pytest tests/unit/test_abiss_chunk_executor.py tests/unit/test_abiss_nucleus_competition.py`
   still pass; `validate_tutorial_configs.py` reports unchanged counts.
7. Golden `--dry-run` comparison for both drivers, advisory only -- a difference is
   investigated, not automatically a failure, because live filesystem state moves. Test 1
   is the binding gate.
8. `git status --short` shows no modification to any out-of-scope file.

## Risks and Questions

1. **Relocating j0126's mask builder is the largest single risk.** 233 lines of
   shard-resumable CloudVolume streaming, no test coverage today, and out of scope to
   change behaviourally. Verification 4(b) treats it as movement and will catch an edit;
   it cannot catch a subtle import-order or module-state difference. If 4(b) cannot be made
   to pass cleanly, the correct response is to stop and report, not to hand-fix the diff.
2. **Step 0 must run before any modification.** If the baseline is captured after the
   drivers are touched, the whole gate is worthless. This ordering is the plan's main
   procedural dependency.
3. **Two params schemas persist.** The engine reads both; migrating j0126 to the nested
   `cluster:` shape would change its resolved sbatch flags and stays deferred -- stated as
   a deferral, not claimed as resolved.
4. **`pipeline:` name collision.** Moritz's `params.yaml` has no `pipeline` key today but
   `connectomics/config/profiles/pipeline_profiles.yaml` exists; confirm during
   implementation that the tutorial-level key does not collide with a schema field, and
   rename to `playbook:` if it does.
5. `dev/` stays gitignored, so `dev/moritz_l4/abiss_wholevol_frozen_c1x1x104.yaml` remains
   untracked. Out of scope; flagged for a separate decision.

## Changes Since Previous Plan Version

Addresses the three major and one minor finding in `plan_v1_review.md`.

- **Finding 1 (canonical sequence)** -- accepted. `CUBE` and `CUBE_FROM_SCRATCH` are now
  defined in `cube_decode.py` and selected by name; a tutorial can no longer reorder or
  drop steps, so dataset #3 gets `guard` and `score` mandatorily, while j0126's order is
  reproduced verbatim (Proposed Changes 1).
- **Finding 2 (mask interface, and where it lives)** -- accepted in full, deferral removed.
  Mask computation moves into `connectomics/data/keep_mask.py` with both strategies behind
  one spec; both scripts become thin CLIs at their existing paths. `plan_v1`'s fallback of
  keeping the Moritz builder as-is is withdrawn (Proposed Changes 2).
- **Finding 3 (builder-level regression proof)** -- accepted. Step 0 captures a serialized
  baseline from both current drivers under three seeded filesystem states before anything
  is edited, and Verification 1 makes exact reproduction the mandatory gate. The
  filesystem-dependent dry-run comparison is explicitly advisory (Verification 0, 1, 7).
- **Finding 4 (onboarding assertion)** -- accepted. The synthetic tutorial now resolves the
  complete canonical sequence, asserts commands use its own paths, and is run through the
  real tutorial validator (Verification 5).
- **Withdrawn claim.** `plan_v1` stated "none are deferred" while deferring part of finding
  2. The reviewer was right; the statement is retracted, and the two remaining deferrals
  (params schema migration, untracked `dev/`) are now named as deferrals in Risks 3 and 5.
