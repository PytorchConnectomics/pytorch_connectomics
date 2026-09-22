# Plan v1

## Summary

One engine in `connectomics/runtime/volume_pipeline.py`, **one** cube playbook in
`connectomics/playbooks/cube_decode.py`, and a per-dataset ordered step list in
`params.yaml`. Both existing datasets run that single playbook; they differ only by which
steps they list. `scripts/run_*.py` become thin CLIs.

Two of the reviewer's findings were checked against the code and changed the design:

- **j0126 is a ~100 um cube too** -- 10664 x 10913 x 5700 at 9x9x20 nm is 96 x 98 x 114 um,
  against Moritz L4's 63 x 96 x 93 um. The reviewer was right that two playbooks is wrong;
  there is one playbook with optional steps.
- **Tutorial validation needs no registration.** `scripts/validate_tutorial_configs.py:128`
  keys `CUSTOM_WORKFLOW_ROOTS = {"large_decode", "abiss_chunk", "seuron_replay",
  "error_correction"}` on the top-level YAML key, not on a tutorial path. A new
  `2_abiss.yaml` uses the already-registered `abiss_chunk:` root, so dataset #3 needs no
  Python change to pass validation.

## Scope

In scope:

1. `connectomics/runtime/volume_pipeline.py` -- engine.
2. `connectomics/playbooks/cube_decode.py` -- the single cube playbook: a registry of step
   builders, assembled per dataset from `params.pipeline.steps`.
3. `connectomics/playbooks/__init__.py`, `README.md`.
4. `scripts/run_playbook.py`; `scripts/run_j0126.py` and `scripts/run_moritz_l4.py` reduced
   to wrappers that pin their tutorial.
5. `scripts/build_keep_mask.py` -- the cube-family mask builder, generalized from
   `build_moritz_l4_keep_mask.py` over a declarative mask spec.
6. `tutorials/_base/abiss.yaml` -- invariants, fitted values absent.
7. `tests/unit/test_volume_pipeline.py`, `tests/unit/test_cube_playbook.py`.
8. `CLAUDE.md` Agent Quick Reference row.

Out of scope, unchanged from v0: numeric behaviour of either tutorial; `lib/`;
`run_abiss_volume.py`; `run_seuron_provenance.py`; and the PATHS of
`scripts/run_abiss_chunk.py`, `seg_percolation_guard.py`, `score_moritz_l4_nerl.py`,
`build_j0126_keep_mask.py`, which are invoked by path from sbatch wrappers in another
repository.

## Proposed Changes

### 1. One playbook, per-dataset step lists

`cube_decode` exposes a registry mapping step name -> builder function:

```
fetch, em, tissue, keep, mask, train, infer, smoke, abiss, guard, score, ec
```

`params.yaml` states the ordered subset:

```yaml
params:
  pipeline:
    steps: [mask, smoke, abiss, guard, score]        # moritz
    steps: [fetch, em, tissue, keep, train, infer, abiss, ec]   # j0126, today's exact list
```

Both datasets therefore run the same playbook module and the same engine, and j0126's
emitted commands are unchanged because its list is exactly what it emits today. Dataset #3
lists the cube subset. This is the mechanism the reviewer asked for in finding 1, and it
does not contradict the command-equivalence gate.

A step builder receives `(params, tutorial)` and returns a `Step`. Unknown step names in
`params.pipeline.steps` raise, naming the unknown step and listing the registry.

### 2. Mask extraction behind a declarative spec (finding 2)

`params.data.masks` becomes the shared interface:

```yaml
masks:
  ratio_zyx: [4, 8, 8]
  border_start_zyx: [0, 103, 103]
  out: ${params.paths.dataset_root}/em/keep_mask_z4y8x8.h5
  sources:
    - {path: .../blood_vessel_trisam_clean_r8.h5, dataset: main, polarity: exclude}
```

`scripts/build_keep_mask.py` consumes exactly that: N downsampled sources with a polarity,
a ratio, a border box, cropped to `params.frame.volume_shape_zyx`. It is a generalization
of `build_moritz_l4_keep_mask.py` (92 lines, single pass, no sharding), whose current
behaviour must be reproduced bit-for-bit on the existing Moritz inputs.

`build_j0126_keep_mask.py` is NOT folded in and is NOT modified. Its inputs are a remote
CloudVolume `tissue_classification` layer at a different resolution, built in two
shard-resumable stages into a full-resolution zarr. The `tissue`/`keep` step builders keep
invoking it by path. Honest residual, stated for the record: a future dataset whose mask
comes from a source of that kind still needs its own builder; "config only" holds for the
vessel/border family, which is what the three cube datasets use.

### 3. Fitted values cannot be inherited, on every route (finding 3)

The check lives in the **engine**, not in one playbook, and runs for any step that resolves
an ABISS config, so `j0126`'s route is covered too:

```
require_dataset_values(cfg, keys=("WS_HIGH_THRESHOLD", "WS_LOW_THRESHOLD",
                                  "AGG_THRESHOLD", "CHUNK_SIZE"))
```

It reads the merged tutorial YAML **before** `abiss_chunk.prepare_config` runs, because
that function does `payload.setdefault("WS_HIGH_THRESHOLD", 0.9)` and would mask an absent
key. **Absent** and **null** are distinct failures and both raise, listing every offending
key. `abiss_chunk.prepare_config` itself is not modified: other configs in the repository
rely on its defaults.

`tutorials/_base/abiss.yaml` carries the invariant block with those four keys **absent**
(not `null`), so inheriting the base without stating them fails the check.

### 4. Package location (reviewer's answer to the open question)

`connectomics/playbooks/`, as the reviewer directed: executable logic, normal imports, and
covered by the packaging config. This also moots the naming objection, since it does not
collide with the repository's two existing meanings of "workflow". Assets stay where they
are: `tutorials/` and `scripts/` are resolved from the repository root, which the engine
derives from `Path(__file__).resolve().parents[2]`, matching `run_abiss_chunk.py`.

## Files and Areas

| File | Change |
|---|---|
| `connectomics/runtime/volume_pipeline.py` | new -- engine, `Step`, checks, launchers, CLI, `require_dataset_values` |
| `connectomics/playbooks/__init__.py` | new -- registry lookup |
| `connectomics/playbooks/cube_decode.py` | new -- all 12 step builders |
| `connectomics/playbooks/README.md` | new -- contract |
| `scripts/run_playbook.py` | new -- thin CLI |
| `scripts/run_moritz_l4.py`, `scripts/run_j0126.py` | reduced to wrappers |
| `scripts/build_keep_mask.py` | new -- generalized cube mask builder |
| `tutorials/_base/abiss.yaml` | new -- invariants; four fitted keys absent |
| `tutorials/neuron_moritz_l4/params.yaml` | add `pipeline.steps`, `data.masks` |
| `tutorials/neuron_j0126/params.yaml` | add `pipeline.steps` |
| `tests/unit/test_volume_pipeline.py` | new |
| `tests/unit/test_cube_playbook.py` | new |
| `CLAUDE.md` | one row |

## Verification Plan

Deterministic fixtures are the primary gate; golden dry-run output is secondary (finding 4).

1. **Fixture-based step resolution** (`test_volume_pipeline.py`). A synthetic params object
   and a `tmp_path` filesystem, exercising every case the live gate cannot reach:
   - a **pending** step emits its command; a **completed** step emits none and reports
     `[done]`; a `skip=` step emits none and reports why. Live `--dry-run` cannot cover
     completed steps, which is exactly the reviewer's objection.
   - `array=N` emits `--array=0-{N-1}` and appends `--shard-id $SLURM_ARRAY_TASK_ID
     --num-shards N`.
   - `afterok` chaining passes the previous job id; the first step has no dependency.
   - `prepare` runs before launch; `prepare_fn` is invoked exactly once and not under
     `--dry-run`.
   - `inputs` gating blocks a local run with a missing input and does not block a slurm run.
   - both resource schemas resolve: j0126's flat `params.<step>.{slurm_partition,cpus,...}`
     and moritz's nested `params.cluster.<step>.{partition,...}`, compared as the exact
     resolved sbatch flag list, not as a substring.
   - `--dry-run` performs zero `subprocess.run` calls (monkeypatched to raise).
2. **Fitted-value enforcement** (`test_cube_playbook.py`): a config with a key **absent**
   raises; a config with it **null** raises; both name every offending key; a complete
   config passes. Run for both a cube-style and a j0126-style step list.
3. **Config-only onboarding** (`test_cube_playbook.py`): construct a synthetic third
   tutorial from `params.yaml` + `2_abiss.yaml` alone in `tmp_path`, resolve every step in
   its `pipeline.steps`, and assert no Python file outside `tutorials/` was needed. Also
   assert its `2_abiss.yaml` is accepted by `validate_tutorial_configs.py`'s
   `CUSTOM_WORKFLOW_ROOTS` logic.
4. **Mask builder equivalence**: `scripts/build_keep_mask.py` on the existing Moritz inputs
   must produce a file byte-identical to the committed `keep_mask_z4y8x8.h5` (2.0% vessel,
   3.05% border, 94.95% keep). This is the only check that the generalization did not
   change a mask.
5. **Golden command comparison**, secondary: capture before the refactor
   ```
   python scripts/run_j0126.py     --dry-run | grep '^  \$' > /tmp/j0126.before
   python scripts/run_moritz_l4.py --dry-run | grep '^  \$' > /tmp/moritz.before
   ```
   and require byte-identical output after. Filesystem state can change between captures,
   so a difference here is investigated, not automatically a failure; test 1 is the gate
   that must hold.
6. `pytest tests/unit/test_abiss_chunk_executor.py tests/unit/test_abiss_nucleus_competition.py`
   still pass; `python scripts/validate_tutorial_configs.py` reports unchanged counts.
7. `git status --short` shows no modification to any file listed out of scope.

## Risks and Questions

1. **j0126 has no test coverage today and costs GPU-days to run.** Test 1 replaces the
   command-string gate the reviewer correctly called insufficient, but it verifies the
   engine, not j0126's step definitions. Lifting `build_steps` is mechanical and must be
   diff-reviewed line by line rather than trusted to tests.
2. **Two params schemas persist.** The engine reads both. Migrating j0126 to the nested
   `cluster:` shape would change its resolved sbatch flags and is deliberately deferred.
3. **`build_keep_mask.py` generalization can silently change a mask.** Verification 4 is
   bit-exact for that reason; if it cannot be made to pass, keep the Moritz builder as-is
   and treat the shared spec as the follow-up.
4. **"Config only" is scoped**, not absolute: it holds for cube datasets whose keep mask is
   a downsampled vessel/border family mask. A different mask source needs a builder.
5. `dev/` remains gitignored, so `dev/moritz_l4/abiss_wholevol_frozen_c1x1x104.yaml` -- the
   config for resuming an interrupted whole-volume decode -- stays untracked. Out of scope
   here; worth a separate decision.

## Changes Since Previous Plan Version

Addresses all five findings in `plan_v0_review.md`; none are deferred.

- **Finding 1 (one shared cube playbook)** -- accepted, and the design changed. Checked the
  volumes: j0126 is 96 x 98 x 114 um, so it is a cube dataset and `plan_v0`'s split into
  `cube_decode` + `j0126_full` was wrong. There is now one playbook with a step registry
  and a per-dataset `params.pipeline.steps` list, which keeps j0126's emitted commands
  identical while both datasets share the module (Proposed Changes 1).
- **Finding 2 (mask interface)** -- accepted in part, with the boundary stated. Added the
  declarative `params.data.masks` spec and `scripts/build_keep_mask.py` generalized from
  the Moritz builder, gated by a bit-exact equivalence test. `build_j0126_keep_mask.py` is
  still not folded in, and the residual limit on "config only" is now written down rather
  than implied (Proposed Changes 2).
- **Finding 3 (enforcement gaps)** -- accepted. The check moved from the playbook to the
  engine so both routes are covered, runs before `prepare_config` can supply a default, and
  treats absent and null as distinct failures with separate tests. The shared base now
  omits the four keys rather than setting them null (Proposed Changes 3, Verification 2).
- **Finding 4 (verification gate)** -- accepted. Deterministic fixtures are now the primary
  gate and cover pending/completed/skipped steps, `prepare`/`prepare_fn`, input gating,
  array submission, `afterok` chaining and both resource schemas, comparing resolved sbatch
  flags. Golden command comparison is demoted to a secondary signal (Verification 1, 5).
- **Finding 5 (onboarding gate and registration)** -- accepted, and the registration half
  is answered with evidence: `validate_tutorial_configs.py:128` keys `CUSTOM_WORKFLOW_ROOTS`
  on the top-level YAML key `abiss_chunk`, not on a tutorial path, so no Python change is
  needed. A synthetic third-tutorial integration test is added (Verification 3).
- **Open question** -- answered as directed: `connectomics/playbooks/`, not root
  `playbooks/` (Proposed Changes 4).
