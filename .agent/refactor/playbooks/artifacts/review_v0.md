# Review v0

## Summary

The implementation meets the approved plan. Every binding gate in `plan_v3` was re-executed
by the reviewer rather than accepted from `code_v0.md`, including the one claim that could
silently change science: the refactored mask builder reproduces the committed
`keep_mask_z4y8x8.h5` **bit for bit** (`np.array_equal` True, keep fraction 0.949536 both).

The definition of done holds. `run_j0126.py`, `run_moritz_l4.py`, `build_j0126_keep_mask.py`
and `build_moritz_l4_keep_mask.py` are each **12 lines**, matching the repository's existing
`run_abiss_chunk.py` thin-CLI precedent exactly; the logic lives in
`connectomics/runtime/volume_pipeline.py` (373), `connectomics/playbooks/cube_decode.py`
(398) and `connectomics/data/keep_mask.py` (423), all importable and unit-tested.

Two minor findings, neither blocking.

## Diff Baseline

run_start_ref: a692d2e37a5eeffb1eee0590613d6a35f5cfc412

`git rev-parse HEAD` equals `run_start_ref`; `git log a692d2e3..HEAD` is empty, so the
coder created no commit. The mutation guard (`git diff` and `git diff --cached` captured
before and after this review) is clean. Diffing `git status --short` against
`state/run_start.status` shows the coder touched only the planned paths plus one noted
below; the 26 pre-existing modified files from other sessions are untouched, which the
coder independently confirmed by SHA-256.

## Findings

1. **[minor] `scripts/score_moritz_l4_nerl.py` was modified although `plan_v3` named it out
   of scope.** Disclosed by the coder in Files Changed. The change is
   `REPO = Path("/projects/weilab/...")` -> `Path(__file__).resolve().parents[1]`, plus
   keyword frame arguments on `build_graph` defaulting to the existing module constants.
   Both are behaviour-preserving and in the spirit of the task, and the invocation path --
   which an sbatch wrapper in another repository depends on -- is unchanged. Accepted as a
   scope note, not a defect.

2. **[minor] The coder's test runs required an environment workaround; mine did not.**
   `code_v0.md` reports that its sandbox denied asyncio's socketpair wakeup, hanging
   ordinary Zarr calls, so pytest ran behind a temporary 10 ms timer launcher, and it
   states plainly that the interrupted direct runs are not claimed as passes. Re-running
   the same three test files in the project conda environment with no workaround gives
   **66 passed**, so the hang was specific to the coder's sandbox rather than a property of
   the tests. Recorded so the workaround is not mistaken for a test dependency.

Verified independently, all passing:

- 66 new tests; 48 pre-existing ABISS tests (no regression).
- `run_moritz_l4.py --check` exits 0 and reports live state correctly;
  `run_j0126.py --dry-run` resolves and emits its command lines.
- `CUBE`, `CUBE_FROM_SCRATCH` and `SEQUENCES` are defined in `cube_decode.py:30-32`; the
  tutorials select by name (`pipeline: cube`, `pipeline: cube_from_scratch`) and therefore
  cannot reorder or drop a step -- dataset #3 gets `guard` and `score` unavoidably.
- `require_dataset_values` builds separate `absent` and `null` lists and reports both, so a
  fitted value cannot be inherited silently. `tutorials/_base/abiss.yaml` omits all four
  keys; both tutorials still state their own.
- The Step 0 ordering was honoured: `tests/fixtures/playbook_baseline/{j0126,moritz_l4}.json`
  hold three seeded states each with `<ROOT>`/`<REPO>` templating and a
  `prepare_fn_present` field -- the `plan_v3` correction, applied.

## Tests to Add

None required for this change. Two follow-ups, both outside this run's scope:

- An end-to-end test that a `cube` sequence submits in `--local` mode against stub
  commands, closing the gap between step resolution and actual execution.
- Coverage for `run_abiss_volume.py` and `run_seuron_provenance.py`, which remain the two
  untested fat drivers under `scripts/` and were deliberately left out of scope.

## Questions

None blocking. One deferral to restate rather than let it disappear: the two params
schemas persist by design (j0126's flat per-step resource blocks, moritz's nested
`cluster:`), because migrating j0126 would change its resolved sbatch flags. `plan_v3`
Risk 3 names this as a deferral and it remains one.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
