# Review v0

## Summary

`code_v0` correctly keeps the implementation scoped to a ranked experiment
matrix and does not touch source code. The new
`.agent/benchmark/nisb-base/experiments.md` is directionally useful, but two
launch-critical recipes are wrong enough that the matrix should be revised
before this CCC run completes.

## Diff Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9

Reviewed actual repository state against the run baseline. The tracked
`connectomics/evaluation/nerl.py` diff was present at run start and is not part
of this CCC code stage. The only new code-stage deliverable is the untracked
experiment matrix under `.agent/benchmark/nisb-base/experiments.md`.

## Findings

- High: The waterz tune command in the minimum set does not change the tuner
  decoder. `.agent/benchmark/nisb-base/experiments.md:115-122` overrides
  `decoding.steps[0]`, but the Optuna tuner resolves the decoder from
  `tune.parameter_space.decoding.function_name`, not from `decoding.steps[0]`
  (`connectomics/decoding/tuning/optuna_tuner.py:515-521`). The base config
  still sets `function_name: decode_affinity_cc`
  (`tutorials/neuron_nisb/base_banis.yaml:223-230`), so E3/E4 will not run
  `decode_waterz`. Worse, the proposed new `thresholds` parameter would be
  reconstructed into kwargs for `decode_affinity_cc`, whose signature only
  accepts `threshold`, `backend`, `edge_offset`, and `orphan_fill`
  (`connectomics/decoding/decoders/segmentation.py:455-461`). This makes the
  ranked minimum set non-runnable or all-trial-failing. Revise E3/E4 to set the
  tune parameter-space function, defaults, and parameter names explicitly, or
  replace the CLI with a small waterz tune overlay.

- High: The E9 SDT recipe adds an SDT channel but does not constrain the
  existing affinity BCE loss to affinity channels only. The document says to
  start from E2 and add `model.out_channels: 7` plus a SmoothL1 term on
  `6:7` (`.agent/benchmark/nisb-base/experiments.md:184-205`). E2 inherits
  `base_banis_v1_erosion2.yaml`, whose `PerChannelBCEWithLogitsLoss` has no
  `pred_slice` or `target_slice`, so it applies to all target channels. After
  adding SDT, BCE would train on the SDT regression target too. Revise E9 to
  either use named heads like `tutorials/neuron_nisb/9nm_base.yaml` or set the
  affinity BCE `pred_slice`/`target_slice` to `0:6` and the SDT SmoothL1 term to
  `6:7`.

- Medium: E5 is classified as a decode-only row that reuses E1 predictions, but
  TTA changes the raw prediction artifact. The row says "Reuses Pred? E1" at
  `.agent/benchmark/nisb-base/experiments.md:45-47`, while the command at
  `.agent/benchmark/nisb-base/experiments.md:141-148` correctly reruns inference
  from the checkpoint with TTA enabled. Update the table wording and sequencing
  so GPU/inference cost and artifact provenance are clear.

- Low: The E9 citation references a non-existent tutorial path:
  `tutorials/neuron_nisb/9nm_base_mednext_l_sdt_multitask.yaml`
  (`.agent/benchmark/nisb-base/experiments.md:184-188`). The matching tutorial
  is `tutorials/neuron_nisb/9nm_base.yaml`. Fix the reference so the matrix can
  be followed without guessing.

## Tests to Add

- Add a config/debug check for the final E3/E4 command or overlay that prints a
  resolved tune config with `tune.parameter_space.decoding.function_name:
  decode_waterz` and waterz defaults/parameters.

- Add a config/debug check for the final E9 overlay showing the affinity loss is
  sliced to `0:6` and the SDT loss is sliced to `6:7`, or that named heads route
  the two losses separately.

## Questions

- Should `code_v1` keep the "no YAML overlays" scope, or is a small
  `base_banis_waterz_tune.yaml` plus an SDT/Dice overlay acceptable to make the
  highest-risk rows launchable?

## Verdict

VERDICT: NEEDS_CHANGES
