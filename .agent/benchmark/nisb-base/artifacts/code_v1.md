# Code v1

## Overview

`code_v1` addresses the four findings in `review_v0.md` (two High, one
Medium, one Low) plus the reviewer's open question about overlay
scope. The two High findings made the matrix's "★ minimum set"
non-runnable (E3/E4 waterz tune was wired wrong, E9 SDT recipe would
have applied BCE to a regression target). Both are now fixed via
two small run-scoped overlay YAMLs under
`.agent/benchmark/nisb-base/configs/`. The Medium and Low findings are
plain text fixes inside `experiments.md`.

## What Changed

- **High — E3/E4 waterz tune.** The Optuna tuner reads the decoder
  from `tune.parameter_space.decoding.function_name`, not from
  `decoding.steps[0]`
  (`connectomics/decoding/tuning/optuna_tuner.py:515-521`). The prior
  CLI override would have left the tuner running `decode_affinity_cc`
  and tried to pass a `thresholds` kwarg into a function that only
  accepts `threshold/backend/edge_offset/orphan_fill`
  (`connectomics/decoding/decoders/segmentation.py:455-461`), so every
  trial would have failed.
  Fix: a new run-scoped overlay
  `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml`
  that sets `tune.parameter_space.decoding.function_name: decode_waterz`,
  pins waterz `defaults`, and replaces the parameter map with a single
  `thresholds` parameter swept over `[0.1, 0.6]` step 0.05, 25 trials.
  E3/E4 in `experiments.md` now point to that overlay; the broken CLI
  example was removed.

- **High — E9 SDT auxiliary head.** `base_banis_v1_erosion2.yaml`
  inherits a single `PerChannelBCEWithLogitsLoss` term with no
  `pred_slice`/`target_slice`, so adding a 7th SDT channel would have
  applied BCE to a SmoothL1 regression target.
  Fix: a new run-scoped overlay
  `.agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_sdt.yaml`
  that replaces the loss list with two sliced terms — affinity
  `PerChannelBCEWithLogitsLoss` on `0:6` and SDT
  `SmoothL1Loss(weight=10.0, tanh=true)` on `6:7` — sets
  `model.out_channels: 7`, and adds a second `skeleton_aware_edt`
  target. Top-level `pred_slice` slicing on a flat output is the
  pattern used in `tutorials/mito_mitolab.yaml`,
  `tutorials/fiber_linghu26.yaml`, and
  `tutorials/neuron_liconn_mit_x2.yaml` (verified).

- **Medium — E5 TTA classification.** The prior matrix marked E5 as
  decode-only that "Reuses Pred? E1", but TTA reruns inference (8×
  flip/rot symmetries) and produces a new prediction artifact.
  `experiments.md` now classifies E5 as type `infer`, expands the
  cost field to spell out the 8× inference cost, and changes the
  `Reuses Pred?` column to `new (not E1's saved prediction)`.

- **Low — broken citation.** `tutorials/neuron_nisb/9nm_base_mednext_l_sdt_multitask.yaml`
  was cited but does not exist (verified via `ls tutorials/neuron_nisb/`).
  E9 now references `tutorials/neuron_nisb/9nm_base.yaml`, which is the
  real named-head SDT pattern for the NISB family.

- **Reviewer's open question on overlay scope.** Answered in
  `experiments.md`: yes, run-scoped overlays under
  `.agent/benchmark/nisb-base/configs/` are the right form for
  highest-risk rows. Two were authored (E3 waterz, E9 SDT). The two
  remaining "may need an overlay" cases (E8 Dice+BCE, E10 abiss tune)
  are below the minimum set and remain as flagged TBD with the same
  pattern documented.

## Implementation Details

The overlays were chosen instead of CLI overrides because:

- The waterz tune fix needs to change three nested Hydra fields
  simultaneously (`function_name`, `defaults`, `parameters`). A
  multi-line CLI override list is fragile; a 30-line overlay is
  clearer and reviewable as a unit.
- The SDT fix replaces the loss list (not amends it). Hydra list
  overrides via CLI replace whole lists, so the CLI form would be a
  long inline YAML literal — strictly worse than a file.
- Putting overlays under `.agent/benchmark/nisb-base/configs/` rather
  than `tutorials/neuron_nisb/` keeps benchmark-specific YAMLs out of
  the canonical tutorial set. They are not picked up by
  `scripts/validate_tutorial_configs.py`, so they cannot regress the
  tutorial gate.

The overlays use relative `_base_:` paths
(`../../../../tutorials/neuron_nisb/base_banis*.yaml`) because Hydra
resolves `_base_:` against the location of the YAML, not against
`cwd`.

Top-level (non-named-head) `pred_slice`/`target_slice` was preferred
over named heads because:

- `base_banis_v1_erosion2.yaml` has no named heads in its inheritance
  chain (`base_banis.yaml → base_banis_v1.yaml → base_banis_v1_erosion2.yaml`,
  all flat outputs). Switching to named heads would touch five
  schema fields (`primary_head`, `heads.*`, `loss.losses[*].pred_head`,
  `inference.model.head`) instead of two slice fields.
- `pred_slice` is a registered loss-orchestrator field with multiple
  in-tree examples; this is not new functionality.

## Files Changed

| File | Purpose |
|---|---|
| `.agent/benchmark/nisb-base/experiments.md` | Updated — rewrote E3/E4 launch path, fixed E5 wording, rewrote E9, fixed broken citation, added paragraph answering reviewer's overlay-scope question. |
| `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml` | New — overlay for E3/E4 waterz Optuna sweep. |
| `.agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_sdt.yaml` | New — overlay for E9 SDT auxiliary head with sliced losses. |
| `.agent/benchmark/nisb-base/artifacts/code_v1.md` | New — this artifact. |

No source files under `connectomics/`, `tutorials/`, or `scripts/`
were edited. The two new overlays live entirely inside the CCC run
folder.

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

No commits were made. The CCC run folder
(`.agent/benchmark/nisb-base/`) remains untracked. The pre-existing
tracked-file diff (`M connectomics/evaluation/nerl.py`) was captured
in `state/run_start.diff` at run start and is not part of any code
stage.

## Verification

What was verified for `code_v1`:

- `connectomics/decoding/tuning/optuna_tuner.py:515-521` — confirmed
  the tuner reads decoder name from
  `param_space_cfg.decoding.function_name`. Defaults to
  `decode_instance_binary_contour_distance`. The new waterz overlay
  sets this explicitly.
- `connectomics/decoding/decoders/segmentation.py:455-461` — confirmed
  `decode_affinity_cc(threshold, backend, edge_offset, orphan_fill)`
  has no `thresholds` (plural) parameter. Confirms reviewer's High-1.
- `tutorials/neuron_nisb/9nm_base.yaml:10-95` — confirmed the
  named-head SDT pattern (`heads.{aff_r1,aff_r5,aff_r11,sdt}` with
  `pred_head: sdt` on the SmoothL1 term and a separate
  `skeleton_aware_edt` target). E9 now mirrors this pattern but uses
  flat-output slicing instead of named heads to minimize diff vs
  `base_banis_v1_erosion2.yaml`.
- `ls tutorials/neuron_nisb/` — confirmed the cited
  `9nm_base_mednext_l_sdt_multitask.yaml` does NOT exist. The Low
  finding was correct.
- Searched `tutorials/` and `connectomics/training/losses/` for
  `pred_slice` to confirm flat-output slicing is a supported,
  in-tree pattern. Hits in
  `mito_mitolab.yaml`, `fiber_linghu26.yaml`,
  `neuron_liconn_mit_x2.yaml`, `nuc_nucmm-z.yaml`, `minimal.yaml`.

What was **not** verified (still a launch-time concern):

- The new overlays were not loaded through
  `connectomics.config.load_config` or run with `--mode tune`.
  Static-only review — the fix targets the wiring identified by the
  reviewer, but a smoke test (e.g. `--fast-dev-run` for the SDT
  overlay; `--mode tune` with `n_trials: 1` for the waterz overlay)
  is the right next verification step before scheduling 25-trial
  sweeps. This is a recommended pre-launch step in the matrix, not a
  blocker for this CCC run.
- The `channel_order: xyz` decision for waterz (carried over from
  `code_v0`) is still untested; `experiments.md` continues to flag it
  as the matrix's primary verify-step on the first trial.

No tests were run. No source code changed; static guardrails
(`tests/unit/test_v3_guardrails.py`,
`tests/unit/test_public_api_snapshot.py`,
`tests/unit/test_v2_boundaries.py`) and
`scripts/validate_tutorial_configs.py` are unaffected by overlays
that live under `.agent/`.

## Review Focus

For `a2`'s next review, the most useful focus points are:

1. **Overlay correctness.** The `_base_:` path
   (`../../../../tutorials/neuron_nisb/base_banis*.yaml`), the
   `model.out_channels: 7` choice in the SDT overlay, and the
   `pred_slice: "0:6"` slicing on the existing PerChannelBCE term.
   Spot-check that flat-output slicing is the right form — i.e.
   that the loss orchestrator does not expect `pred_head` when
   `model.heads` is empty.
2. **Tune-overlay completeness.** The waterz overlay has to set
   `tune.parameter_space.decoding.function_name`, `defaults`, and
   `parameters` together. Confirm none of the other `tune.*` fields
   from `base_banis.yaml` (data, sampler, optimization, output,
   logging) are accidentally clobbered by the overlay.
3. **Run-scoped overlay scope decision.** Was it correct to keep the
   overlays under `.agent/benchmark/nisb-base/configs/` instead of
   promoting them to `tutorials/neuron_nisb/`?
4. **E8 / E10 deferral.** Both still defer overlay authoring until
   the row is selected. If you want them launchable upfront, that's
   a `code_v2` ask.

## Risks and Unknowns

- The overlays were not loaded; the relative `_base_:` paths could
  silently mis-resolve if Hydra's overlay search path changes. A
  one-trial smoke test before the full sweep mitigates this.
- The SDT overlay assumes `skeleton_aware_edt` produces a single
  channel with values in roughly `[-1, 1]` after the `tanh` wrap.
  If the on-the-fly kimimaro skeletonization is slow on
  `9nm`-resolution data, the dataloader may bottleneck training.
  Worth profiling on a small batch before scheduling 200k steps.
- The `channel_order: xyz` decision for waterz is still unverified at
  the data level. The matrix's pre-launch verify-step remains in
  place.
- Two ideas (E8 Dice+BCE, E10 abiss tune) still defer overlay
  authoring. If GPU budget allows running them, expect a small
  follow-up to author those overlays.

## Changes Since Previous Code Version

Addressed all four findings from `review_v0.md`:

- High-1 (E3/E4 waterz tune wiring): authored
  `.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml`;
  rewrote E3 / E4 sections of `experiments.md` to reference the
  overlay; removed the broken CLI override that targeted only
  `decoding.steps[0]`.
- High-2 (E9 SDT loss bleed onto regression target): authored
  `.agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_sdt.yaml`
  with `pred_slice: "0:6"` on the affinity BCE and `pred_slice: "6:7"`
  on the SDT SmoothL1 term; rewrote E9 in `experiments.md` to
  reference the overlay.
- Medium (E5 misclassified as decode-only): updated the List-A row
  type to `infer`, the cost column to spell out 8× inference, and the
  `Reuses Pred?` column to `new (not E1's saved prediction)`.
- Low (broken citation): replaced the missing
  `9nm_base_mednext_l_sdt_multitask.yaml` reference with
  `9nm_base.yaml` and explicitly noted the prior path does not exist.
- Reviewer's open question: answered with a paragraph in the E3
  section confirming run-scoped overlays under
  `.agent/benchmark/nisb-base/configs/` are the right form, and
  documented the same pattern for E8 / E10 if those rows are
  selected.
