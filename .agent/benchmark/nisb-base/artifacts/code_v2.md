# Code v2

## Overview

`code_v2` addresses the single High-severity finding in `review_v1.md`:
the waterz tune overlay did not actually replace the inherited CC
parameter map. OmegaConf merges dicts rather than replacing them, so
the base config's `threshold` parameter and `backend`/`edge_offset`
defaults survived into the resolved tune config and would have been
sampled on every Optuna trial. Worse, the document falsely claimed the
parameter map was replaced.

The fix accepts that constraint instead of fighting it: the overlay
now uses **`tune.n_trials: 1`** with waterz batch mode, which
evaluates the full threshold list inside a single decode call. The
inherited CC parameter is sampled exactly once and silently ignored by
`decode_waterz`'s `**kwargs`; the chosen best threshold is read from
the trial's `best_threshold` user attribute. Documentation is updated
to be honest about this rather than pretending the merge replaced the
map.

This is a `revision_rounds: 2` run. `code_v2` is the final allowed
code version — under the protocol, there is no `review_v2` stage.

## What Changed

- **High — waterz tune parameter-map merge.** Set `tune.n_trials: 1`
  in `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml`
  (was 25). Added an in-file comment block explaining: (a) OmegaConf
  dict-merge means inherited CC parameters survive; (b) `decode_waterz`
  silently drops unknown args via `**kwargs`; (c) waterz batch mode
  enumerates all thresholds in one call so additional Optuna trials
  only re-sample irrelevant inherited params; (d) the chosen
  `best_threshold` is in trial user attrs, not `best_params`.
- Updated `experiments.md` E3 / E4 table rows: cost field now reads
  "one waterz batch decode (~minutes)" instead of "minutes per trial
  × 25 trials". Description column reflects the single-trial
  + internal sweep.
- Rewrote the launch paragraph for E3 in `experiments.md`: removed
  the false claim that the overlay "replaces" the parameter map.
  Replaced with a precise description of the merge behaviour, why
  `n_trials: 1` is the right choice, where to read `best_threshold`
  (trial user attribute), and the per-threshold ARE values
  (`are_thr_<value>` user attributes).
- Acknowledged the reviewer's open question on broader waterz tuning
  (`merge_function`, `aff_threshold` low/high): out of scope this
  iteration because expanding the parameter set requires disabling
  the batch shortcut. Flagged as future work.

No source files under `connectomics/`, `tutorials/`, or `scripts/`
were edited.

## Implementation Details

The chosen fix (Path A from `review_v1`'s suggestions) was preferred
over Path B (non-batch waterz search with the inherited `threshold`
neutralized) because:

1. Waterz batch mode is the documented "right" way to sweep
   thresholds — the comment in
   `connectomics/decoding/decoders/waterz.py:62-66` explicitly says
   "When multiple thresholds are provided, waterz performs watershed
   and region-graph extraction once and incrementally merges for each
   threshold — making multi-threshold evaluation nearly as fast as
   single. This is especially useful for Optuna parameter tuning."
   Disabling the batch shortcut to clean up an inherited parameter
   would trade the protocol's intent for a cosmetic fix.
2. The OmegaConf merge behaviour is structural — there is no `null`
   sentinel that deletes a key, and no `_replace_:` directive in
   `connectomics/config/pipeline/profile_engine.py` (verified). The
   only honest options were "stop pretending" + accept the merge
   (this fix), or restructure the `_base_:` chain to not inherit
   `tune` at all (much larger surface area).
3. With `n_trials: 1`, the inherited CC `threshold` parameter is
   sampled exactly once and silently dropped by `decode_waterz`'s
   `**kwargs`. The cost is one wasted Optuna sample call; correctness
   and result are unaffected.

The trial-attribute-vs-best_params distinction is real. Result
inspection now requires reading the Optuna study/journal:

```python
import optuna
study = optuna.load_study(
    study_name="nisb_base_banis_waterz_tuning",
    storage="sqlite:///outputs/nisb_base_banis/tune/optuna.db",
)
trial = study.best_trial
print(trial.user_attrs["best_threshold"])
# Per-threshold ARE values:
for k, v in trial.user_attrs.items():
    if k.startswith("are_thr_"):
        print(k, v)
```

(Path to `optuna.db` depends on the run-time `tune.output_path`; the
overlay does not pin it.)

## Files Changed

| File | Purpose |
|---|---|
| `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml` | Updated — `n_trials: 25 → 1`; added comment block explaining the merge behaviour and why a single trial is correct. |
| `.agent/benchmark/nisb-base/experiments.md` | Updated — fixed the E3 launch paragraph (removed the false "replaces the parameter map" claim, added accurate merge/n_trials/best_threshold guidance); updated E3/E4 table rows. |
| `.agent/benchmark/nisb-base/artifacts/code_v2.md` | New — this artifact. |

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

No commits were made. The CCC run folder remains untracked.

## Verification

What was verified:

- `connectomics/decoding/tuning/optuna_tuner.py:562-583` — confirmed
  waterz batch mode reads `parameters.thresholds.range/step` directly
  to enumerate the threshold list and sets
  `_waterz_batch_enabled = True`.
- `connectomics/decoding/tuning/optuna_tuner.py:1018-1021` — confirmed
  the batch trial replaces `decoding_params["thresholds"]` with the
  full list and sets `return_all_thresholds=True`.
- `connectomics/decoding/tuning/optuna_tuner.py:1053-1063` — confirmed
  `best_threshold` is stored as a trial `user_attr`, alongside
  per-threshold `are_thr_<value>` attributes.
- `connectomics/decoding/decoders/waterz.py:31-56` — confirmed
  `decode_waterz` accepts `**kwargs` and silently drops unknown args.
  So inherited `threshold`, `backend`, `edge_offset` are harmless.
- `connectomics/config/pipeline/config_io.py:82-86` and
  `profile_engine.py:21-29` — confirmed `_base_` resolution uses pure
  `OmegaConf.merge(base, child)`. No replace semantics; no `null`
  delete behaviour.

What was **not** verified:

- The overlay was not loaded through
  `connectomics.config.load_config`; the static-only review of the
  edit is the verification. Smoke-test recommendation in
  `experiments.md` (one trial, then read trial user attrs) remains
  the right pre-launch step.
- The `channel_order: xyz` decision is still untested at the data
  level. With `n_trials: 1` the user can run the smoke test and read
  `are_thr_*` to spot a degenerate sweep before committing further
  GPU time.

No tests run.

## Review Focus

`code_v2` is the final allowed code version under
`revision_rounds: 2`; per protocol there is no `review_v2` stage. If
the user wants another revision pass, `run.md` `Rounds` can be bumped
to `2,3` or higher and a `code_v3` invoked.

If anyone later does want to spot-check this artifact, the most
useful focus points are:

1. The reasoning for choosing `n_trials: 1` over neutralizing the
   inherited `threshold` parameter via overlay restructuring.
2. The accuracy of the new `best_threshold` retrieval guidance —
   specifically that it is a trial user attribute, not part of the
   standard best-params save path.
3. Whether the matrix should expand E3/E4 to also tune
   `merge_function` (categorical) or `aff_threshold` low/high. The
   current scope is `thresholds`-only because that's the highest-
   leverage knob and waterz batch makes it nearly free; the
   reviewer's open question on this is parked.

## Risks and Unknowns

- The merge-behaviour fix is structural; it stays correct as long as
  the Optuna tuner keeps the `_waterz_batch_enabled` shortcut at
  `optuna_tuner.py:562-583`. If a future refactor disables that
  shortcut and starts sampling `thresholds` per trial, the overlay's
  `n_trials: 1` becomes wrong (only the first sample would be tried).
  Worth re-checking on any tuner refactor.
- The matrix's expected-Δ ranges remain educated guesses, not
  measurements. Unchanged from `code_v1`.
- The `channel_order: xyz` decision for waterz remains an unverified
  assumption. With single-trial batch mode the user can read
  `are_thr_*` user attrs after one decode and immediately see whether
  the sweep is degenerate, which makes the verify-step cheaper than
  before.
- Two ideas (E8 Dice+BCE, E10 abiss tune) still defer overlay
  authoring. If GPU budget allows running them, expect a small
  follow-up. Note: an abiss tune overlay will hit the same OmegaConf
  merge issue and should also use `tune.n_trials` matching abiss
  batch-mode semantics
  (`optuna_tuner.py:539-560`).

## Changes Since Previous Code Version

Addressed the single High-severity finding in `review_v1.md`:

- waterz tune parameter-map merge: in
  `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml`,
  set `tune.n_trials: 1` (was 25) and added an explanatory comment
  block describing the OmegaConf merge behaviour, the
  `decode_waterz` `**kwargs` swallow, the waterz batch-mode shortcut
  in the tuner, and the `best_threshold` trial-user-attribute
  retrieval path.
- Rewrote E3 in `experiments.md` to drop the false
  "replaces the parameter map" claim and replaced it with an honest
  description of the merge + `n_trials: 1` rationale. Updated the
  E3/E4 table rows to reflect single-trial cost.
- Acknowledged the reviewer's open question on broader waterz tuning
  (`merge_function`, `aff_threshold`): scoped out of this iteration
  because it requires disabling the batch shortcut.
- No changes to the SDT overlay or any other matrix row — review_v1
  did not flag those.
