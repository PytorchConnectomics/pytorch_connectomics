# review_v0 raw notes (planner, in-session)

Claims in code_v0.md were re-run independently rather than accepted.

## Independently executed
- pytest tests/unit/test_volume_pipeline.py test_cube_playbook.py test_keep_mask.py
  -> 66 passed, 23.66s, in the pytc conda env, WITHOUT Codex's asyncio workaround.
- pytest tests/unit/test_abiss_chunk_executor.py test_abiss_nucleus_competition.py
  -> 48 passed. No regression in the pre-existing suites.
- python scripts/run_moritz_l4.py --check -> exit 0, live state correct
  (mask found; smoke 3584/3584 chunks).
- python scripts/run_j0126.py --dry-run -> 10 emitted command lines, resolves.
- scripts/build_keep_mask.py --params tutorials/neuron_moritz_l4/params.yaml
  -> np.array_equal against the committed keep_mask_z4y8x8.h5 is True.
     keep fraction 0.949536 both. BIT-IDENTICAL.
- git rev-parse HEAD == a692d2e3... == run_start_ref. git log a692d2e3..HEAD -> 0 commits.
- git status --short diffed against state/run_start.status: the only paths Codex touched
  are the planned ones plus scripts/score_moritz_l4_nerl.py.
- wc -l: run_j0126.py 12, run_moritz_l4.py 12, build_j0126_keep_mask.py 12,
  build_moritz_l4_keep_mask.py 12. Engine 373, cube_decode 398, keep_mask 423.
- tutorials/_base/abiss.yaml: all four fitted keys absent. Both tutorials still state
  their own values (moritz 14 occurrences, j0126 7).
- connectomics/playbooks/cube_decode.py:30-32 defines CUBE, CUBE_FROM_SCRATCH, SEQUENCES.
  params.yaml selects by name: moritz `pipeline: cube`, j0126 `pipeline: cube_from_scratch`.
- volume_pipeline.py:66-72 builds separate `absent` and `null` lists and reports both.
- tests/fixtures/playbook_baseline/{j0126,moritz_l4}.json: 3 seed states each, paths
  templated as <ROOT>/<REPO>, fields include prepare_fn_present.

## Concerns
- scripts/score_moritz_l4_nerl.py modified though the plan named it out of scope.
  Disclosed in Files Changed. Diff is REPO -> Path(__file__).resolve().parents[1] and
  build_graph gaining keyword frame args defaulting to the module constants. Both
  behaviour-preserving; the file is invoked by path from an sbatch wrapper in another repo
  and that path is unchanged.
- Codex reports its sandbox denied asyncio's socketpair wakeup, hanging Zarr, and that it
  ran pytest behind a temporary 10 ms timer launcher. My own runs used no such workaround
  and passed, so the hang was environmental to the coder's sandbox, not a property of the
  tests. Codex explicitly did not claim the interrupted runs as passes.

## Verdict reasoning
Every binding gate in plan_v3 verified independently. Two minor, non-blocking findings.
