# Review v1

## Summary

`code_v1` fixes the SDT loss routing and the E5/citation text issues from
`review_v0`. The new overlays also resolve through the CLI setup path. However,
the waterz tune overlay still does not replace the inherited CC tuning
parameter map, so the highest-priority decode sweep remains misleading and needs
one more revision.

## Diff Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9

Reviewed actual repository state against the run baseline. The tracked
`connectomics/evaluation/nerl.py` diff and untracked `.claude/` path are
outside this CCC code stage. The CCC-stage changes are under
`.agent/benchmark/nisb-base/`.

## Findings

- High: `base_banis_waterz_tune.yaml` still inherits the base CC `threshold`
  tuning parameter and CC defaults, despite the document saying the parameter
  map is replaced. Loading the overlay through `setup_config(..., mode="tune")`
  resolves `parameters=['threshold', 'thresholds']` and defaults containing
  stale `backend` / `edge_offset`. This happens because the overlay only defines
  `thresholds` (`.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml:35-52`)
  on top of the base config's `threshold` parameter
  (`tutorials/neuron_nisb/base_banis.yaml:223-234`), and OmegaConf merges dicts.
  In waterz batch mode, the tuner skips sampling `thresholds` but still samples
  every other inherited parameter (`connectomics/decoding/tuning/optuna_tuner.py:1217-1225`).
  The 25 trials therefore vary an unused CC `threshold`, while waterz evaluates
  the same threshold list each trial. The saved best params are also wrong:
  waterz stores the selected candidate in `best_threshold`, but result saving
  only injects ABISS `best_ws_merge_threshold`
  (`connectomics/decoding/tuning/optuna_tuner.py:1053-1061` and `:1489-1498`).
  Revise the overlay/documentation so E3/E4 are honest and launchable: either
  use `tune.n_trials: 1` for waterz batch sweeps and document that
  `best_threshold` is read from trial attrs/logs, or create a non-batch waterz
  search that samples `thresholds` directly without the inherited CC `threshold`.
  Also remove or neutralize the inherited `threshold` entry in the resolved tune
  parameter space, rather than claiming the map was replaced.

## Tests to Add

- Re-run the CLI setup check for the final waterz overlay and confirm the
  resolved tune parameter list no longer includes inherited `threshold`, or that
  the document explicitly treats the run as a one-trial waterz batch sweep.

- Keep the SDT setup check from this review: the resolved SDT overlay should
  show `out_channels=7`, affinity BCE sliced to `0:6`, SmoothL1 sliced to `6:7`,
  and `select_channel=[0, 1, 2]`.

## Questions

- Should the waterz row optimize only `thresholds`, or should the matrix expand
  it to tune other waterz parameters (`merge_function`, `aff_threshold_low/high`)
  using the existing `tune_waterz` profile pattern?

## Verdict

VERDICT: NEEDS_CHANGES
